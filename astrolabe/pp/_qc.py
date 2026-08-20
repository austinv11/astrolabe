"""
QC metric calculation and cell/gene filtering for spatial transcriptomics data.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scanpy as sc

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._shapes import _build_polygons
from astrolabe._utils import with_adata

__all__ = [
    "filter_cells",
    "filter_genes",
    "calculate_qc_metrics",
    "calculate_xenium_qc_metrics",
    "flag_controls",
    "filter_isolated",
    "calculate_shape_metrics",
    "calculate_transcript_assignment_qc",
]

log = logging.getLogger(__name__)


def _typed_value_set(values: set, col_type):
    """Build a pyarrow value_set matching *col_type*, or ``None`` if uncastable.

    obs_names/var_names are always strings, but a transcript table's
    ``cell_id`` column can be integer-typed (older/numeric Xenium exports)
    -- ``pc.is_in`` with a string value_set against an int column matches
    nothing (or raises), silently desyncing the table. Cast the value set to
    the column's own type instead; if the cast fails (e.g. non-numeric
    obs_names vs an int column), no value could ever have matched, so the
    caller should skip the sync.
    """
    import pyarrow as pa

    try:
        return pa.array(list(values)).cast(col_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return None


def _sync_transcripts_after_cell_filter(adata_uns: dict, removed_cell_ids: set) -> None:
    """Drop uns['transcripts'] rows belonging to cells a cell filter removed.

    Platform-agnostic by construction: only Xenium's reader ever populates
    uns['transcripts'] (MERSCOPE/CosMx have no per-transcript table), so
    this is a silent no-op for those modalities -- filter_cells/filter_genes
    are the single shared code path behind filter_xenium/filter_merscope/
    filter_cosmx, so all three platforms get identical treatment here.

    Rows whose cell_id doesn't match any *current* obs_names at all (e.g.
    Xenium's 'UNASSIGNED' sentinel for transcripts never assigned to a
    segmented cell) are never in removed_cell_ids, since that's built from
    obs_names -- so they always survive untouched, regardless of the
    sentinel's literal spelling.

    When rows are actually dropped, ``uns['transcripts_pruned']`` is set to
    ``True`` so downstream consumers whose denominators require the *full*
    table (e.g. :func:`calculate_transcript_assignment_qc`) can refuse to
    compute biased statistics from a pruned one.
    """
    transcripts = adata_uns.get("transcripts")
    if transcripts is None or not removed_cell_ids:
        return
    if "cell_id" not in transcripts.schema.names:
        return
    import pyarrow.compute as pc

    value_set = _typed_value_set(
        removed_cell_ids, transcripts.schema.field("cell_id").type
    )
    if value_set is None:
        log.debug(
            "  uns['transcripts'] sync skipped: obs_names are not castable to "
            "the cell_id column type (%s) so no row could match.",
            transcripts.schema.field("cell_id").type,
        )
        return
    keep = pc.invert(pc.is_in(transcripts.column("cell_id"), value_set=value_set))
    n_before = len(transcripts)
    adata_uns["transcripts"] = transcripts.filter(keep)
    n_after = len(adata_uns["transcripts"])
    if n_after != n_before:
        adata_uns["transcripts_pruned"] = True
        log.info(
            "  synced uns['transcripts']: %d -> %d rows (dropped filtered-out cells)",
            n_before, n_after,
        )


def _sync_transcripts_after_gene_filter(adata_uns: dict, removed_genes: set) -> None:
    """Drop uns['transcripts'] rows for genes/features a gene filter removed.

    See :func:`_sync_transcripts_after_cell_filter` for why this is
    platform-agnostic (no-op when uns['transcripts'] doesn't exist) and for
    the ``uns['transcripts_pruned']`` flag set when rows are dropped.
    """
    transcripts = adata_uns.get("transcripts")
    if transcripts is None or not removed_genes:
        return
    if "feature_name" not in transcripts.schema.names:
        return
    import pyarrow.compute as pc

    value_set = _typed_value_set(
        removed_genes, transcripts.schema.field("feature_name").type
    )
    if value_set is None:
        log.debug(
            "  uns['transcripts'] sync skipped: var_names are not castable to "
            "the feature_name column type (%s) so no row could match.",
            transcripts.schema.field("feature_name").type,
        )
        return
    keep = pc.invert(pc.is_in(transcripts.column("feature_name"), value_set=value_set))
    n_before = len(transcripts)
    adata_uns["transcripts"] = transcripts.filter(keep)
    n_after = len(adata_uns["transcripts"])
    if n_after != n_before:
        adata_uns["transcripts_pruned"] = True
        log.info(
            "  synced uns['transcripts']: %d -> %d rows (dropped filtered-out genes)",
            n_before, n_after,
        )


def _add_nucleus_metrics(adata: ad.AnnData) -> None:
    """Append nucleus_to_cell_area_ratio and has_nucleus columns to obs."""
    if "nucleus_area" in adata.obs.columns and "cell_area" in adata.obs.columns:
        nuc = adata.obs["nucleus_area"].to_numpy(dtype=np.float64)
        cell = adata.obs["cell_area"].to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(cell > 0, nuc / cell, np.nan)
        adata.obs["nucleus_to_cell_area_ratio"] = ratio
    if "nucleus_count" in adata.obs.columns:
        adata.obs["has_nucleus"] = adata.obs["nucleus_count"].to_numpy(dtype=np.int32) >= 1
    elif "nucleus_area" in adata.obs.columns:
        adata.obs["has_nucleus"] = adata.obs["nucleus_area"].to_numpy(dtype=np.float64) > 0.0


@with_adata
def calculate_qc_metrics(
    adata: ad.AnnData,
    *,
    qc_vars: list[str] | tuple[str, ...] = (),
    percent_top: list[int] | tuple[int, ...] | None = (50, 100, 200, 500),
    log1p: bool = True,
    exclude_controls: bool = True,
    noise_floor: bool = True,
    inplace: bool = True,
):
    """Compute per-cell and per-gene QC metrics.

    Wraps :func:`scanpy.pp.calculate_qc_metrics` and adds spatial QC columns
    when the relevant ``obs`` columns are present:
    ``nucleus_to_cell_area_ratio`` (ratio of nucleus area to cell area) and
    ``has_nucleus`` (bool flag derived from ``nucleus_count`` or ``nucleus_area``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    qc_vars
        ``var`` column names that flag feature subsets (e.g. control probes).
        A ``pct_counts_<name>`` column is added to ``obs`` for each
    percent_top
        Compute cumulative fractions for these top-N gene counts.
        ``None`` to skip
    log1p
        Whether to add ``log1p``-transformed counts alongside raw counts
    exclude_controls
        If ``True`` (default), compute metrics as if control probes
        (``var['is_control_probe'] == True`` — negative control probes,
        unassigned/deprecated codewords, blanks, etc.) were not part of the
        panel: per-cell metrics (``total_counts``, ``n_genes_by_counts``,
        ``pct_counts_in_top_*``, ...) are computed from real genes only, and
        per-gene metrics are only computed for real genes (control-probe rows
        in ``var`` are left with ``NaN`` for the new columns). Has no effect
        if ``var['is_control_probe']`` is absent (a warning is logged).

        **``total_counts`` convention**: when controls are excluded,
        ``obs['total_counts']`` deliberately counts *non-control* features
        only (this is the column every downstream astrolabe threshold —
        e.g. :func:`filter_cells` ``min_counts`` — operates on). The
        all-features sum, matching vendor per-cell totals such as Xenium's
        ``transcript_counts``, is additionally written to
        ``obs['total_counts_all_features']`` so both denominators stay
        available.
    noise_floor
        If ``True`` (default), also run :func:`calculate_noise_floor` with
        its defaults — the control-derived background threshold →
        ``uns['noise_floor']`` + ``var['below_noise_floor']``.  Skipped with
        a log message (never an error) when ``X`` is log1p-transformed or no
        control features are available, and when ``inplace=False`` (the flag
        is adata state, inconsistent with returning DataFrames).  Call
        :func:`calculate_noise_floor` directly to customize ``quantile=``/
        ``multiplier=``/``control_types=`` — re-running it overwrites the
        automatic result.
    inplace
        Modify *data* in place (default) or return ``(obs_df, var_df)``

    Returns
    -------
    None, or ``(obs_df, var_df)`` when ``inplace=False``.
    """
    qc_adata = adata
    ctrl_mask = None
    if exclude_controls:
        if "is_control_probe" in adata.var.columns:
            ctrl_mask = adata.var["is_control_probe"].to_numpy(dtype=bool)
            if ctrl_mask.any():
                qc_adata = adata[:, ~ctrl_mask].copy()
            else:
                ctrl_mask = None
        else:
            log.warning(
                "exclude_controls=True but var['is_control_probe'] not found; "
                "computing QC metrics over all features. If this data has control "
                "probes, run ab.pp.flag_controls(data, ...) first to mark them "
                "(pass var_col='is_control_probe' if the flag lives under a "
                "different column name)."
            )

    n_vars = qc_adata.n_vars
    qc_percent_top = percent_top
    if qc_percent_top is not None:
        pruned_top = [n for n in qc_percent_top if n > n_vars]
        qc_percent_top = [n for n in qc_percent_top if n <= n_vars] or None
        if pruned_top:
            log.info(
                "calculate_qc_metrics: percent_top values %s exceed the "
                "%d-feature panel and were skipped.",
                pruned_top, n_vars,
            )

    if ctrl_mask is not None:
        # qc_adata is a throwaway copy restricted to non-control genes.
        # Strip its obs (and all var columns except the qc_vars flags the QC
        # call itself reads) so every column present after the call is a
        # freshly computed metric. Merging that full set back UNCONDITIONALLY
        # is what makes re-running this function update stale metrics -- the
        # old merge dropped every column already present on adata.obs/var, so
        # a second call (e.g. after filtering) never refreshed total_counts.
        qc_adata.obs = pd.DataFrame(index=qc_adata.obs_names)
        keep_var = [c for c in qc_vars if c in qc_adata.var.columns]
        qc_adata.var = qc_adata.var[keep_var].copy()

    if gpu_enabled():
        rsc = _require_rapids()
        # rsc.pp.calculate_qc_metrics has no inplace= parameter (it always
        # writes in place) -- forwarding one raises. Emulate inplace=False by
        # working on a stripped copy and returning its columns instead.
        gpu_adata = qc_adata
        if ctrl_mask is None and not inplace:
            gpu_adata = adata.copy()
            gpu_adata.obs = pd.DataFrame(index=gpu_adata.obs_names)
            keep_var = [c for c in qc_vars if c in gpu_adata.var.columns]
            gpu_adata.var = gpu_adata.var[keep_var].copy()
        with _gpu_X(gpu_adata):
            rsc.pp.calculate_qc_metrics(
                gpu_adata,
                qc_vars=qc_vars,
                percent_top=qc_percent_top,
                log1p=log1p,
            )
        if ctrl_mask is None:
            if inplace:
                result = None
            else:
                result = (
                    gpu_adata.obs.copy(),
                    gpu_adata.var.drop(columns=keep_var, errors="ignore"),
                )
    else:
        result = sc.pp.calculate_qc_metrics(
            qc_adata,
            qc_vars=qc_vars,
            percent_top=qc_percent_top,
            log1p=log1p,
            inplace=inplace or ctrl_mask is not None,
        )

    if ctrl_mask is None:
        if inplace:
            _add_nucleus_metrics(adata)
            if noise_floor:
                _auto_noise_floor(adata)
        return result

    # qc_adata is a separate copy (subset to real genes); merge its computed
    # obs/var columns back onto the full adata rather than returning/assigning
    # metrics for a gene subset. obs/var were stripped before the call, so
    # everything present now (minus the qc_vars flags) is a fresh metric.
    obs_df = qc_adata.obs.copy()
    var_df = qc_adata.var.drop(columns=[c for c in qc_vars if c in qc_adata.var.columns])
    var_df = var_df.reindex(adata.var_names)

    # obs['total_counts'] above is the control-excluded total (the documented
    # convention every astrolabe filter threshold uses); also expose the
    # all-features sum, which matches vendor per-cell totals (e.g. Xenium's
    # transcript_counts) and flag_controls' denominator.
    obs_df["total_counts_all_features"] = np.asarray(
        adata.X.sum(axis=1), dtype=np.float64
    ).ravel()

    if inplace:
        for col in obs_df.columns:
            adata.obs[col] = obs_df[col]
        for col in var_df.columns:
            adata.var[col] = var_df[col]
        _add_nucleus_metrics(adata)
        if noise_floor:
            _auto_noise_floor(adata)
        return None
    return obs_df, var_df


def _auto_noise_floor(adata: ad.AnnData) -> None:
    """calculate_qc_metrics' automatic noise-floor step.

    Unlike a direct :func:`calculate_noise_floor` call, this never raises:
    QC metrics are legitimately (re)computed on log1p-transformed matrices
    and on panels without control probes, and the noise floor is simply not
    computable there — skip with a log message rather than failing the
    whole QC call.
    """
    if "log1p" in adata.uns:
        log.info(
            "calculate_qc_metrics: skipping the noise floor (X is "
            "log1p-transformed). Run pp.calculate_noise_floor(layer=...) "
            "against a raw-counts layer to compute it."
        )
        return
    try:
        calculate_noise_floor(adata)
    except ValueError:
        log.info(
            "calculate_qc_metrics: skipping the noise floor (no control "
            "features available to estimate it from)."
        )


# Xenium utilities
@with_adata(mirror_uns="control_probe_summary")
def calculate_xenium_qc_metrics(
        adata: ad.AnnData,
        transcripts=None,
        *,
        qv_col: str = "qv",
        gene_col: str = "feature_name",
        inplace: bool = True,
) -> ad.AnnData | None:
    """Compute per-cell and per-gene Xenium probe-specific QC metrics and store in obs/var.

    This function only computes metrics and flags control features in
    ``var['is_control']`` — it never removes features. Use
    :func:`~astrolabe.pp.filter_genes` (``exclude_controls=True``) or
    :func:`~astrolabe.pipeline.filter_xenium` (``remove_controls=True``) to
    remove them.

    When any control probes are present, a per-control-type summary is also
    written to ``uns['control_probe_summary']`` (skipped otherwise) and
    mirrored to ``mdata.uns['control_probe_summary'][<modality>]`` when
    called on a MuData.

    Per-cell metrics added to ``obs``
    ---------------------------------
    unassigned_codeword_total
        Total counts from unassigned codewords (i.e. probes that could not be
        confidently assigned to a gene).
    deprecated_codeword_total
        Total counts from deprecated codewords that are no longer used in the
        assay design.
    genomic_control_probe_total
        Total counts from probes that target the genome.
    negative_control_probe_total
        Total counts from negative control probes (e.g. NegControlProbe_*).
    negative_control_codeword_total
        Total counts from unassigned codewords that are also negative controls.
    any_negative_control_probe_total
        The sum of all non-Gene Expression totals.

    n_unassigned_codewords
        The number of unassigned codewords with at least one count.
    n_deprecated_codewords
        The number of deprecated codewords with at least one count.
    n_genomic_control_probes
        The number of genomic control probes with at least one count.
    n_negative_control_probes
        The number of negative control probes with at least one count.
    n_negative_control_codewords
        The number of unassigned codewords that are also negative controls with at least one count.
    n_any_negative_control_probes
        The number of any negative control probes with at least one count.

    pct_unassigned_codewords
        Percentage of the cell's total counts coming from unassigned codewords
        (``unassigned_codeword_total / total_counts * 100``). ``NaN`` for
        zero-count cells. The same count-sum convention applies to every
        ``pct_*`` column below.
    pct_deprecated_codewords
        Percentage of counts from deprecated codewords.
    pct_genomic_control_probes
        Percentage of counts from genomic control probes.
    pct_negative_control_codewords
        Percentage of counts from negative control codewords.
    pct_negative_control_probes
        Percentage of counts from negative control probes.
    pct_any_negative_control_probes
        Percentage of counts from any negative control probes (including
        unassigned codewords that are negative controls).

    Per-gene metrics added to ``var``
    ---------------------------------
    is_control
        Whether this feature is a control probe (e.g. NegControlProbe_*, etc.).
    mean_qv
        Mean calibrated Q-score per gene, computed from *transcripts* when provided.
        Added only when *transcripts* is not ``None``; skipped otherwise with a debug log.

    Parameters
    ----------
    data
        The AnnData/MuData object with Xenium counts that include control probe counts.
    transcripts
        :class:`pyarrow.Table` or :class:`pandas.DataFrame` returned by
        :func:`~astrolabe.io.read_xenium_transcripts`.  When provided, mean
        Q-score per gene is computed and stored in ``var['mean_qv']``.  When
        ``None``, the column is skipped and a debug message is logged.
    modality
        If a MuData is passed, the modality containing the Xenium data. Ignored if an AnnData is passed.
    qv_col
        Column in *transcripts* containing per-transcript quality values
    gene_col
        Column in *transcripts* containing gene/feature names
    inplace
        If ``True`` (default), modify the input AnnData in place and return ``None``. If ``False``, return a modified copy.

    Returns
    -------
    None or AnnData
        ``None`` if ``inplace=True``; modified copy if ``inplace=False``.
    """
    adata = adata.copy() if not inplace else adata

    if "feature_type" not in adata.var.columns:
        raise KeyError(
            "var['feature_type'] not found — calculate_xenium_qc_metrics requires "
            "Xenium data loaded via ab.io.read_xenium()"
        )

    unassigned_mask = adata.var['feature_type'] == 'Unassigned Codeword'
    deprecated_mask = adata.var['feature_type'] == 'Deprecated Codeword'
    genomic_control_mask = adata.var['feature_type'] == 'Genomic Control'
    negative_control_codeword_mask = adata.var['feature_type'] == 'Negative Control Codeword'
    negative_control_probe_mask = adata.var['feature_type'] == 'Negative Control Probe'
    any_negative_control_mask = unassigned_mask | deprecated_mask | genomic_control_mask | negative_control_codeword_mask | negative_control_probe_mask

    if any_negative_control_mask.sum() < 1:
        raise ValueError(
            "No control probes found in var['feature_type']; cannot compute Xenium QC metrics. "
            "Ensure the data includes control probe features from read_xenium()"
        )

    def _row_sum(x) -> np.ndarray:
        return np.asarray(x.sum(axis=1)).ravel()

    adata.obs['unassigned_codeword_total'] = _row_sum(adata[:, unassigned_mask].X)
    adata.obs['deprecated_codeword_total'] = _row_sum(adata[:, deprecated_mask].X)
    adata.obs['genomic_control_probe_total'] = _row_sum(adata[:, genomic_control_mask].X)
    adata.obs['negative_control_codeword_total'] = _row_sum(adata[:, negative_control_codeword_mask].X)
    adata.obs['negative_control_probe_total'] = _row_sum(adata[:, negative_control_probe_mask].X)
    adata.obs['any_negative_control_probe_total'] = _row_sum(adata[:, any_negative_control_mask].X)

    def _row_nnz(x) -> np.ndarray:
        return np.asarray((x > 0).sum(axis=1)).ravel()

    adata.obs['n_unassigned_codewords'] = _row_nnz(adata[:, unassigned_mask].X)
    adata.obs['n_deprecated_codewords'] = _row_nnz(adata[:, deprecated_mask].X)
    adata.obs['n_genomic_control_probes'] = _row_nnz(adata[:, genomic_control_mask].X)
    adata.obs['n_negative_control_codewords'] = _row_nnz(adata[:, negative_control_codeword_mask].X)
    adata.obs['n_negative_control_probes'] = _row_nnz(adata[:, negative_control_probe_mask].X)
    adata.obs['n_any_negative_control_probes'] = _row_nnz(adata[:, any_negative_control_mask].X)

    total_counts = _row_sum(adata.X).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Numerators are the *_total count sums, NOT the n_* nnz-feature
        # counts -- pct columns are fractions of the cell's transcript
        # counts, so both sides of the ratio must be counts. Zero-count
        # cells have no defined fraction -> NaN, not 0.
        def _pct(total_col: str) -> np.ndarray:
            return np.where(
                total_counts > 0,
                adata.obs[total_col].to_numpy(np.float64) / total_counts * 100.0,
                np.nan,
            )

        adata.obs['pct_unassigned_codewords'] = _pct('unassigned_codeword_total')
        adata.obs['pct_deprecated_codewords'] = _pct('deprecated_codeword_total')
        adata.obs['pct_genomic_control_probes'] = _pct('genomic_control_probe_total')
        adata.obs['pct_negative_control_codewords'] = _pct('negative_control_codeword_total')
        adata.obs['pct_negative_control_probes'] = _pct('negative_control_probe_total')
        adata.obs['pct_any_negative_control_probes'] = _pct('any_negative_control_probe_total')

    adata.var['is_control'] = adata.var['feature_type'].isin(['Unassigned Codeword', 'Deprecated Codeword', 'Genomic Control', 'Negative Control Codeword', 'Negative Control Probe'])

    if transcripts is None:
        transcripts = adata.uns.get("transcripts")

    if transcripts is not None:
        try:
            import pyarrow as pa
            if isinstance(transcripts, pa.Table):
                transcripts = transcripts.select([gene_col, qv_col]).to_pandas()
        except ImportError:
            pass
        if not isinstance(transcripts, pd.DataFrame):
            raise TypeError(
                f"transcripts must be a pyarrow.Table or pandas.DataFrame, "
                f"got {type(transcripts).__name__!r}."
            )
        mean_qv_series = transcripts.groupby(gene_col)[qv_col].mean()
        adata.var['mean_qv'] = adata.var_names.map(mean_qv_series).to_numpy(dtype=np.float64)
    else:
        log.debug(
            "calculate_xenium_qc_metrics: no transcripts found; skipping var['mean_qv']. "
            "Run ab.io.read_xenium_transcripts(mdata) or pass transcripts= explicitly."
        )

    # Stash per-gene control stats so gene_qv_scatter can still show individual
    # control probe points even if a caller later removes controls via
    # pp.filter_genes(exclude_controls=True) (e.g. pipeline.filter_xenium).
    ctrl_bool = adata.var['is_control'].values
    if ctrl_bool.any():
        col_sums = np.asarray(adata.X.sum(axis=0)).ravel()
        # n_cells_by_counts (cells with >=1 count per control feature) lets
        # gene_detection_scatter reconstruct a detection rate for stashed
        # controls after they've been removed from var, alongside the
        # existing total_counts backfill used by gene_qv_scatter.
        n_cells_pos = np.asarray((adata.X > 0).sum(axis=0)).ravel()
        ctrl_df = adata.var.loc[ctrl_bool, ['feature_type']].copy()
        ctrl_df['total_counts'] = col_sums[ctrl_bool]
        ctrl_df['n_cells_by_counts'] = n_cells_pos[ctrl_bool]
        if 'mean_qv' in adata.var.columns:
            ctrl_df['mean_qv'] = adata.var.loc[ctrl_bool, 'mean_qv'].values
        stash = ctrl_df.reset_index().to_dict(orient='list')
        stash['n_obs'] = int(adata.n_obs)  # denominator for the detection rate
        adata.uns['control_probe_summary'] = stash

    # Populate the generic obs['pct_counts_control'] so filter_cells works
    # uniformly regardless of whether data came from Xenium or another source.
    flag_controls(adata, var_col="is_control")

    return None if inplace else adata


@with_adata
def flag_controls(
    adata: ad.AnnData,
    *,
    names: list[str] | None = None,
    var_col: str | None = None,
    pattern: str | None = None,
    obs_key: str = "pct_counts_control",
    control_key: str = "is_control",
) -> None:
    """Flag features as negative controls and compute per-cell control percentage.

    Sets ``var[control_key]`` (bool) and ``obs[obs_key]`` (float, 0–100;
    ``NaN`` for zero-count cells, whose control fraction is undefined —
    note a ``NaN`` fails any :func:`filter_cells` ``max_pct_control``
    threshold, which is the conservative choice for empty segments).
    Exactly one of *names*, *var_col*, or *pattern* must be provided.

    Several functions already flag controls automatically and don't need
    this called first: :func:`~astrolabe.io.read_xenium`,
    :func:`~astrolabe.io.read_merscope`, and :func:`~astrolabe.io.read_cosmx`
    all set ``var['is_control_probe']`` at load time, and
    :func:`~astrolabe.pp.calculate_xenium_qc_metrics` derives ``var['is_control']``
    from ``var['feature_type']`` and calls this function internally
    (``var_col='is_control'``). Call ``flag_controls`` yourself when working
    with a platform/format that doesn't set one of those columns, or when the
    control set is defined by an explicit name list or naming pattern instead.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    names
        Explicit list of ``var_names`` to flag as controls.
    var_col
        Name of an existing boolean ``var`` column to use as the control
        flag (e.g. ``'is_control_probe'`` written by
        :func:`~astrolabe.io.read_xenium`).
    pattern
        Regular expression matched against ``var_names``; any match is
        flagged as a control.
    obs_key
        ``obs`` column name for the per-cell control percentage.
    control_key
        ``var`` column name for the boolean control flag.

    Examples
    --------
    >>> # From an existing var column (Xenium)
    >>> ab.pp.flag_controls(data, var_col='is_control_probe')

    >>> # By exact names
    >>> ab.pp.flag_controls(data, names=['NegCtrl1', 'NegCtrl2'])

    >>> # By regex pattern
    >>> ab.pp.flag_controls(data, pattern=r'^NegControl')
    """
    n_provided = sum(x is not None for x in [names, var_col, pattern])
    if n_provided != 1:
        raise ValueError(
            "Exactly one of names=, var_col=, or pattern= must be provided"
        )

    if names is not None:
        names_set = set(names)
        ctrl_mask = np.array([n in names_set for n in adata.var_names], dtype=bool)
    elif var_col is not None:
        if var_col not in adata.var.columns:
            raise KeyError(
                f"var['{var_col}'] not found. "
                f"Available columns: {list(adata.var.columns)}"
            )
        ctrl_mask = adata.var[var_col].to_numpy(dtype=bool)
    else:
        import re
        rx = re.compile(pattern)
        ctrl_mask = np.array([bool(rx.search(str(n))) for n in adata.var_names], dtype=bool)

    adata.var[control_key] = ctrl_mask

    if sp.issparse(adata.X):
        ctrl_counts = np.asarray(adata.X[:, ctrl_mask].sum(axis=1)).ravel()
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        ctrl_counts = adata.X[:, ctrl_mask].sum(axis=1)
        total_counts = adata.X.sum(axis=1)

    ctrl_counts = ctrl_counts.astype(np.float64)
    total_counts = total_counts.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Zero-count cells have no defined control fraction -> NaN, not 0
        # (a hard 0 reads as "clean cell" to any max_pct_control threshold).
        pct = np.where(total_counts > 0, ctrl_counts / total_counts * 100.0, np.nan)

    adata.obs[obs_key] = pct

    log.info(
        "flag_controls: %d / %d features flagged as controls; "
        "obs['%s'] computed (median %.2f%%, max %.2f%%)",
        int(ctrl_mask.sum()), adata.n_vars,
        obs_key, float(np.nanmedian(pct)), float(np.nanmax(pct)),
    )


@with_adata
def calculate_noise_floor(
    adata: ad.AnnData,
    *,
    control_key: str = "is_control_probe",
    control_types: list[str] | None = None,
    quantile: float = 0.95,
    multiplier: float = 1.0,
    layer: str | None = None,
    key_added: str = "noise_floor",
) -> None:
    """Estimate the panel's background noise floor from its control probes.

    Control probes measure signal that cannot be biological — nonspecific
    binding (negative probes / blanks) or decoding errors (system controls,
    blank/unassigned codewords).  Their per-feature mean counts therefore
    define an empirical *noise floor*: any real gene whose mean counts sit at
    or below it is statistically indistinguishable from background, no matter
    how plausible its detection pattern looks.

    :func:`calculate_qc_metrics` runs this automatically with its defaults
    (``noise_floor=True``) whenever controls are available and counts look
    raw — call this function directly to customize *quantile* / *multiplier*
    / *control_types* (re-running overwrites the automatic result).

    The floor is the *quantile*-th quantile of the control features' mean
    counts per cell (× *multiplier*).  Genes at or below it are flagged in
    ``var['below_noise_floor']``; the threshold and its provenance are stored
    in ``uns[key_added]``.  Remove the flagged genes with
    :func:`filter_genes` ``(exclude_below_noise_floor=True)`` — flagging and
    filtering are deliberately separate steps (the
    :func:`calculate_qc_metrics` / :func:`filter_cells` pattern), so the flag
    can be inspected and plotted before anything is removed.
    :func:`~astrolabe.pl.gene_detection_scatter` draws the threshold as a
    dashed line automatically once ``uns[key_added]`` exists.

    Control features themselves are never flagged (``False`` in the output
    column): control removal stays exclusively behind ``filter_genes``'s
    ``exclude_controls=True``, matching the library-wide convention that
    controls are kept in ``X`` and flagged, never silently dropped.

    .. note::
        Must run on **raw counts** — after :func:`normalize_total` every
        cell's total is a shared constant and per-gene means no longer
        measure absolute abundance against the controls.  This function
        raises if the input looks log1p-transformed (``uns['log1p']``
        present) unless an explicit raw-counts *layer* is passed.  In a
        standard pipeline, call it right after loading (or after
        ``pipeline.filter_<platform>``'s cell filtering), before
        normalization.

    If the flagged control columns have already been removed from ``var``
    (e.g. Xenium after ``pipeline.filter_xenium``), the control means fall
    back to the per-control stats stashed in ``uns['control_probe_summary']``
    by :func:`calculate_xenium_qc_metrics` — computed on the pre-filter
    matrix, which is logged.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    control_key
        Boolean ``var`` column flagging control features.  Falls back to
        ``'is_control'`` (written by :func:`flag_controls` /
        :func:`calculate_xenium_qc_metrics`) when absent.
    control_types
        Restrict the floor to specific control classes via
        ``var['feature_type']`` (e.g. ``['Negative Probe']`` to measure only
        nonspecific binding, excluding decoding-error controls).  ``None``
        uses every flagged control.
    quantile
        Quantile of the control mean-counts distribution used as the floor
        (default ``0.95`` — robust to a single outlier control, unlike the
        max).
    multiplier
        Scale factor applied to the quantile (default ``1.0``).  Raise it
        for a more conservative gene set.
    layer
        Layer holding raw counts.  ``None`` reads ``X`` directly.
    key_added
        ``uns`` key for the threshold record.

    Examples
    --------
    >>> ab.pp.calculate_noise_floor(mdata)
    >>> ab.pp.filter_genes(mdata, exclude_below_noise_floor=True, inplace=True)
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {multiplier}")
    if layer is None and "log1p" in adata.uns:
        raise ValueError(
            "adata.X appears to be log1p-transformed (uns['log1p'] is set). "
            "The noise floor compares absolute per-gene abundance against "
            "control probes and must run on raw counts — stash them first "
            "(adata.layers['counts'] = adata.X.copy() before normalize_total) "
            "and pass layer='counts'."
        )

    X = adata.layers[layer] if layer is not None else adata.X
    col_sums = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    gene_means = col_sums / max(adata.n_obs, 1)

    # --- control mask (var), with 'is_control' fallback ------------------
    ctrl_col = control_key if control_key in adata.var.columns else "is_control"
    if ctrl_col in adata.var.columns:
        ctrl_mask = adata.var[ctrl_col].to_numpy(dtype=bool)
    else:
        ctrl_mask = np.zeros(adata.n_vars, dtype=bool)
    if control_types is not None:
        if "feature_type" not in adata.var.columns:
            raise KeyError(
                "var['feature_type'] not found — control_types= needs "
                "per-type control labels (written by read_xenium, "
                "read_cosmx, and read_merscope)."
            )
        ctrl_mask &= adata.var["feature_type"].isin(control_types).to_numpy()

    # --- control means: live var rows, else the Xenium stash -------------
    if ctrl_mask.any():
        control_means = gene_means[ctrl_mask]
        control_source = "var"
    else:
        stash = adata.uns.get("control_probe_summary") or {}
        if "total_counts" in stash and stash.get("n_obs"):
            totals = np.asarray(stash["total_counts"], dtype=np.float64)
            if control_types is not None:
                types = np.asarray(stash["feature_type"])
                keep = np.isin(types, control_types)
                totals = totals[keep]
            control_means = totals / float(stash["n_obs"])
            control_source = "control_probe_summary"
            log.info(
                "calculate_noise_floor: no flagged controls in var; using %d "
                "control(s) stashed in uns['control_probe_summary'] "
                "(means computed on the pre-filter matrix).",
                len(control_means),
            )
        else:
            raise ValueError(
                "No control features found: nothing flagged by "
                f"var['{control_key}']/var['is_control'] and no "
                "uns['control_probe_summary'] stash. Run pp.flag_controls "
                "(or load via read_xenium/read_cosmx/read_merscope, which "
                "flag controls automatically), or compute the floor before "
                "removing controls."
            )
    if len(control_means) == 0:
        raise ValueError(
            "control_types= matched no control features; nothing to "
            "estimate a noise floor from."
        )

    threshold = float(np.quantile(control_means, quantile) * multiplier)
    below = (gene_means <= threshold) & ~ctrl_mask
    adata.var["below_noise_floor"] = below

    adata.uns[key_added] = {
        "threshold_mean_counts": threshold,
        "statistic": "mean_counts_per_cell",
        "quantile": float(quantile),
        "multiplier": float(multiplier),
        "n_controls": int(len(control_means)),
        "control_source": control_source,
        "control_types": list(control_types) if control_types is not None else None,
        "layer": layer,
        "n_below": int(below.sum()),
    }
    log.info(
        "calculate_noise_floor: threshold = %.4g mean counts/cell "
        "(quantile %.2f of %d controls%s); %d / %d genes at or below it "
        "-> var['below_noise_floor']",
        threshold, quantile, len(control_means),
        f" x {multiplier:g}" if multiplier != 1.0 else "",
        int(below.sum()), int((~ctrl_mask).sum()),
    )


@with_adata
def filter_cells(
    adata: ad.AnnData,
    *,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    max_pct_control: float | None = None,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    inplace: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Filter cells by QC thresholds.

    All thresholds are inclusive on the retained side (i.e.
    ``min_counts=100`` keeps cells with ``total_counts >= 100``).

    If ``uns['transcripts']`` is present (Xenium only — MERSCOPE/CosMx never
    populate it, so this is a no-op for them), rows belonging to removed
    cells are dropped from it too, keeping the transcript-level table
    consistent with the cell×gene matrix. Rows not tied to any real cell
    (e.g. Xenium's ``'UNASSIGNED'`` sentinel) are never touched by this.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    min_counts, max_counts
        Total UMI count bounds.  Requires ``obs['total_counts']`` — note
        that when :func:`calculate_qc_metrics` ran with its default
        ``exclude_controls=True``, this column counts **non-control features
        only** (the all-features total lives in
        ``obs['total_counts_all_features']`` and is *not* what these
        thresholds read).
    min_genes, max_genes
        Detected gene count bounds.  Requires ``obs['n_genes_by_counts']``.
    max_pct_control
        Maximum allowed percentage of counts from control features.
        Requires ``obs['pct_counts_control']`` — run
        :func:`flag_controls` first.
    min_cell_area, max_cell_area
        Cell area bounds in µm².  Requires ``obs['cell_area']``.
    inplace
        If ``False`` (default), return ``(filtered_adata, bool_mask)``
        where the mask is ``True`` for retained cells.
        If ``True``, subset in place and return ``None``.

    Returns
    -------
    None or tuple[AnnData, ndarray]
    """
    mask = np.ones(adata.n_obs, dtype=bool)

    def _apply(col: str, lo=None, hi=None):
        nonlocal mask
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs['{col}'] not found. "
                "Run pp.calculate_qc_metrics / pp.flag_controls first"
            )
        values = adata.obs[col].to_numpy(dtype=np.float64)
        if lo is not None:
            mask &= values >= lo
        if hi is not None:
            mask &= values <= hi

    if min_counts is not None or max_counts is not None:
        _apply("total_counts", lo=min_counts, hi=max_counts)
    if min_genes is not None or max_genes is not None:
        _apply("n_genes_by_counts", lo=min_genes, hi=max_genes)
    if max_pct_control is not None:
        _apply("pct_counts_control", hi=max_pct_control)
    if min_cell_area is not None or max_cell_area is not None:
        _apply("cell_area", lo=min_cell_area, hi=max_cell_area)

    n_removed = int((~mask).sum())
    log.info("filter_cells: removing %d / %d cells", n_removed, adata.n_obs)
    removed_cell_ids = set(adata.obs_names[~mask]) if "transcripts" in adata.uns else set()

    if inplace:
        _sync_transcripts_after_cell_filter(adata.uns, removed_cell_ids)
        adata._inplace_subset_obs(mask)
        return None
    else:
        filtered = adata[mask].copy()
        _sync_transcripts_after_cell_filter(filtered.uns, removed_cell_ids)
        return filtered, mask


@with_adata
def filter_genes(
    adata: ad.AnnData,
    *,
    min_counts: int | None = None,
    min_cells: int | None = None,
    exclude_controls: bool = False,
    exclude_below_noise_floor: bool = False,
    keep_feature_types: list[str] | None = None,
    inplace: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Filter genes/features by expression or type criteria.

    If ``uns['transcripts']`` is present (Xenium only — MERSCOPE/CosMx never
    populate it, so this is a no-op for them), rows for removed genes are
    dropped from it too, keeping the transcript-level table consistent with
    the cell×gene matrix — this is what keeps control-probe transcripts
    (removed via ``exclude_controls=True``) and any other filtered-out gene
    out of downstream transcript-level analyses (e.g.
    :func:`~astrolabe.io.read_xenium_transcripts`-based plots).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    min_counts
        Minimum total counts across all cells.  Requires
        ``var['total_counts']`` (run :func:`calculate_qc_metrics` first).
    min_cells
        Minimum number of cells expressing this gene.  Requires
        ``var['n_cells_by_counts']``.
    exclude_controls
        If ``True``, remove all features flagged as control probes
        (``var['is_control_probe'] == True``).
    exclude_below_noise_floor
        If ``True``, remove genes flagged at or below the control-derived
        noise floor (``var['below_noise_floor']``, written automatically by
        :func:`calculate_qc_metrics` when controls are available, or
        explicitly by :func:`calculate_noise_floor`).  Control features
        are never removed by this gate (their flag is always ``False``);
        use *exclude_controls* for that.
    keep_feature_types
        If given, retain only features whose ``var['feature_type']`` is in
        this list (e.g. ``['Gene Expression']``).
    inplace
        If ``False`` (default), return ``(filtered_adata, bool_mask)``.
        If ``True``, subset in place and return ``None``.

    Returns
    -------
    None or (AnnData, ndarray)
    """
    mask = np.ones(adata.n_vars, dtype=bool)

    # NaN in total_counts/n_cells_by_counts means "not evaluated" (per
    # calculate_qc_metrics's documented exclude_controls=True behavior:
    # per-gene metrics are left NaN on control-probe rows, not zeroed) --
    # not "failed the threshold." A plain `>= min_X` comparison already
    # evaluates to False for NaN in float64 (matching min_counts below), but
    # min_cells additionally *cast* the column to int32 first -- NaN cast to
    # int32 silently becomes INT32_MIN, which still reads as "failed" but
    # for the wrong reason and inconsistently with min_counts' float64 cast.
    # Net effect either way: every control-probe gene was silently dropped
    # by *any* min_counts/min_cells filtering, regardless of exclude_controls
    # -- a user filtering on cell/count thresholds alone (not asking to
    # remove controls at all) unexpectedly lost every control probe too.
    # NaN rows on *flagged control-probe genes* are therefore passed through
    # this filter (still removable via exclude_controls=True, the documented
    # and only intended gate for control-probe removal). The pass-through is
    # gated on var['is_control_probe'] so a NaN of any other origin (e.g. a
    # hand-edited column) still fails the threshold like any non-number.
    if "is_control_probe" in adata.var.columns:
        ctrl_rows = adata.var["is_control_probe"].to_numpy(dtype=bool)
    else:
        ctrl_rows = np.zeros(adata.n_vars, dtype=bool)

    if min_counts is not None:
        if "total_counts" not in adata.var.columns:
            raise KeyError("var['total_counts'] not found. Run pp.calculate_qc_metrics first")
        values = adata.var["total_counts"].to_numpy(dtype=np.float64)
        mask &= (values >= min_counts) | (np.isnan(values) & ctrl_rows)

    if min_cells is not None:
        if "n_cells_by_counts" not in adata.var.columns:
            raise KeyError("var['n_cells_by_counts'] not found. Run pp.calculate_qc_metrics first")
        values = adata.var["n_cells_by_counts"].to_numpy(dtype=np.float64)
        mask &= (values >= min_cells) | (np.isnan(values) & ctrl_rows)

    if exclude_controls:
        if "is_control_probe" not in adata.var.columns:
            log.warning("var['is_control_probe'] not found; cannot exclude controls.")
        else:
            mask &= ~adata.var["is_control_probe"].to_numpy(dtype=bool)

    if exclude_below_noise_floor:
        if "below_noise_floor" not in adata.var.columns:
            raise KeyError(
                "var['below_noise_floor'] not found. "
                "Run pp.calculate_noise_floor first."
            )
        mask &= ~adata.var["below_noise_floor"].to_numpy(dtype=bool)

    if keep_feature_types is not None:
        if "feature_type" not in adata.var.columns:
            raise KeyError(
            "var['feature_type'] not found. "
            "Populate it with feature-type labels (e.g. 'Gene Expression', control probe types) "
            "before calling filter_genes with keep_feature_types"
        )
        mask &= adata.var["feature_type"].isin(keep_feature_types).to_numpy()

    n_removed = (~mask).sum()
    log.info("filter_genes: removing %d / %d features", n_removed, adata.n_vars)
    removed_genes = set(adata.var_names[~mask]) if "transcripts" in adata.uns else set()

    if inplace:
        _sync_transcripts_after_gene_filter(adata.uns, removed_genes)
        adata._inplace_subset_var(mask)
        return None
    else:
        filtered = adata[:, mask].copy()
        _sync_transcripts_after_gene_filter(filtered.uns, removed_genes)
        return filtered, mask


@with_adata
def filter_isolated(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    neighbors_key: str = "spatial_neighbors",
    max_neighbor_rank: int | None = None,
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    inplace: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Remove spatially isolated cells based on their own kNN distances.

    Computes each cell's *outbound* k-nearest-neighbor distances directly
    from ``obsm[spatial_key]`` (:class:`scipy.spatial.cKDTree`) and removes
    cells that fail either criterion:

    - **nearest-neighbor distance** ``> min_neighbor_dist_um`` — the cell's
      closest neighbor is farther than this threshold, indicating a truly
      isolated cell.
    - **k-th-neighbor distance** ``> max_neighbor_dist_um`` — the cell's
      *k*-th nearest neighbor exceeds this threshold, indicating the cell
      sits in a very sparse region.

    Using outbound kNN distances (rather than the union-symmetrized graph
    stored by :func:`~astrolabe.tl.spatial_neighbors`) means a cell is only
    ever removed for **its own** isolation: a distant outlier that happens
    to list dense-region cells among its neighbors adds *inbound* edges to
    them on the symmetrized graph, but never changes their own outbound
    kNN distances — so it can no longer drag well-connected cells over the
    threshold with it.

    Both thresholds can be independently disabled by passing ``None``.
    Any ``obsp`` graphs present are automatically trimmed by AnnData to the
    retained cells on subsetting.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    spatial_key
        ``obsm`` key holding ``(n_cells, 2)`` µm centroid coordinates.
    neighbors_key
        ``uns`` key of a graph previously written by
        :func:`~astrolabe.tl.spatial_neighbors`; used **only** to default
        *k* to that graph's stored ``n_neighbors`` when *max_neighbor_rank*
        is not given. No ``obsp`` matrix is read.
    max_neighbor_rank
        The neighbor rank *k* whose distance the *max_neighbor_dist_um*
        threshold applies to.  ``None`` (default) uses the ``n_neighbors``
        recorded in ``uns[neighbors_key]`` if present, else ``5``.
    max_neighbor_dist_um
        Retain cells whose *k*-th nearest neighbor is within this distance
        (µm).  ``None`` disables this check.
    min_neighbor_dist_um
        Retain cells whose nearest neighbor is within this distance (µm).
        ``None`` disables this check.
    inplace
        If ``False`` (default), return ``(filtered_adata, bool_mask)``.
        If ``True``, subset in place and return ``None``.

    Returns
    -------
    None or (AnnData, ndarray)

    Raises
    ------
    KeyError
        If ``obsm[spatial_key]`` is absent.
    """
    from scipy.spatial import cKDTree

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found — filter_isolated needs cell "
            "centroid coordinates."
        )

    if max_neighbor_dist_um is None and min_neighbor_dist_um is None:
        log.warning("filter_isolated: both thresholds are None — no filtering applied.")
        if inplace:
            return None
        return adata.copy(), np.ones(adata.n_obs, dtype=bool)

    n_cells = adata.n_obs

    if max_neighbor_rank is not None:
        k = int(max_neighbor_rank)
    else:
        k = int(
            adata.uns.get(neighbors_key, {}).get("params", {}).get("n_neighbors", 5)
        )
    if k < 1:
        raise ValueError(f"max_neighbor_rank must be >= 1; got {k}")
    k_eff = min(k, n_cells - 1)

    if k_eff < 1:
        # A single cell has no neighbors at all: unconditionally isolated
        # under either active threshold.
        min_dists = np.full(n_cells, np.inf, dtype=np.float64)
        kth_dists = np.full(n_cells, np.inf, dtype=np.float64)
    else:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        tree = cKDTree(coords)
        # k+1 because each point's own 0-distance self-match is returned first.
        dists, _ = tree.query(coords, k=k_eff + 1)
        dists = dists[:, 1:]  # drop self column
        min_dists = dists[:, 0]
        kth_dists = dists[:, -1]

    mask = np.ones(n_cells, dtype=bool)
    if min_neighbor_dist_um is not None:
        mask &= min_dists <= min_neighbor_dist_um
    if max_neighbor_dist_um is not None:
        mask &= kth_dists <= max_neighbor_dist_um

    n_removed = int((~mask).sum())
    log.info(
        "filter_isolated: removing %d / %d cells "
        "(k=%d, max_neighbor=%.0f um, min_neighbor=%.0f um)",
        n_removed, n_cells, k_eff,
        max_neighbor_dist_um if max_neighbor_dist_um is not None else float("inf"),
        min_neighbor_dist_um if min_neighbor_dist_um is not None else float("inf"),
    )
    removed_cell_ids = set(adata.obs_names[~mask]) if "transcripts" in adata.uns else set()

    if inplace:
        _sync_transcripts_after_cell_filter(adata.uns, removed_cell_ids)
        adata._inplace_subset_obs(mask)
        return None
    else:
        filtered = adata[mask].copy()
        _sync_transcripts_after_cell_filter(filtered.uns, removed_cell_ids)
        return filtered, mask


# ---------------------------------------------------------------------------
# Shape metrics
# ---------------------------------------------------------------------------


def _min_rect_axes(gs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (major_axis, minor_axis, orientation) for each polygon's minimum
    rotated rectangle; orientation is the angle of the major-axis side in
    radians, mod π (an axis is undirected), NaN for degenerate geometries."""
    majors: list[float] = []
    minors: list[float] = []
    thetas: list[float] = []
    for geom in gs:
        # A degenerate-but-not-`is_empty` polygon (e.g. all vertices
        # collinear or coincident after `buffer(0)` self-intersection
        # cleanup) can have `minimum_rotated_rectangle` collapse to a
        # `Point` or `LineString` instead of a `Polygon` -- neither has an
        # `.exterior` attribute, so accessing it unconditionally raised
        # AttributeError partway through writing obs columns, leaving the
        # AnnData half-populated. Skip (NaN) rather than crash; `is_empty`
        # alone doesn't catch this case since the input geometry itself is
        # not empty, only its rotated-rectangle summary is degenerate.
        if geom.is_empty:
            majors.append(np.nan)
            minors.append(np.nan)
            thetas.append(np.nan)
            continue
        rect = geom.minimum_rotated_rectangle
        if rect.geom_type != "Polygon":
            majors.append(np.nan)
            minors.append(np.nan)
            thetas.append(np.nan)
            continue
        coords = np.array(rect.exterior.coords)[:4]
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[1]
        s1 = float(np.linalg.norm(v1))
        s2 = float(np.linalg.norm(v2))
        major_vec = v1 if s1 >= s2 else v2
        majors.append(max(s1, s2))
        minors.append(min(s1, s2))
        # arctan2 ∈ (−π, π]; numpy's % maps negatives into [0, π) too.
        thetas.append(float(np.arctan2(major_vec[1], major_vec[0]) % np.pi))
    return (
        np.array(majors, dtype=np.float32),
        np.array(minors, dtype=np.float32),
        np.array(thetas, dtype=np.float32),
    )




@with_adata
def calculate_shape_metrics(
    adata: ad.AnnData,
    *,
    boundaries_key: str = "cell_boundaries",
    prefix: str | None = None,
    counts_key: str | None = "total_counts",
    nucleus_key: str | None = "nucleus_boundaries",
) -> None:
    """Compute morphological shape metrics from cell or nucleus boundary polygons.

    Writes the following metrics to ``obs`` (all float32):

    - **area** — polygon area in µm².
    - **perimeter** — polygon perimeter in µm.
    - **span_x** / **span_y** — bounding-box width and height in µm.
    - **equivalent_radius** — radius of a circle with the same area (√(A/π)) in µm.
    - **bbox_major_axis_length** / **bbox_minor_axis_length** — side lengths
      of the *minimum rotated bounding rectangle* in µm.  Note these are
      **not** the image-moment ellipse axes ``skimage.measure.regionprops``
      calls ``major/minor_axis_length`` — the bbox sides of e.g. a thin
      crescent differ substantially from its moment axes, hence the
      ``bbox_`` prefix.
    - **bbox_orientation** — angle of the bbox major axis in radians, in
      ``[0, π)`` (mod π: an axis has no direction; 0 = x-aligned,
      π/2 = y-aligned).  Arbitrary side choice for squares (no unique major
      axis); ``NaN`` for degenerate geometries.
    - **polarity_major_angle** / **polarity_minor_angle** — the bbox axes
      *directed* by the nucleus position (radians, mod 2π): each axis's
      positive end is the half of the cell farther from the nucleus (the
      sign of its dot product with the cell-centroid − nucleus-centroid
      offset).  Written only when ``obsm[nucleus_key]`` exists and
      *boundaries_key* is a different key; ``NaN`` for cells without a
      usable nucleus polygon.  For a near-centered nucleus the signs are
      noise-driven — check **nucleus_offset_um** before leaning on them.
      This directed frame is what :func:`~astrolabe.pl.direction_histogram`
      reads by default (``relative_to='cell_axis'``).
    - **nucleus_offset_um** — length of the cell-centroid − nucleus-centroid
      offset in µm; how strongly the polarity frame is anchored.
    - **aspect_ratio** — ``bbox_major_axis_length / bbox_minor_axis_length``;
      1 = square/circle, higher values indicate elongation (a filter for
      elongated cells looks like ``cell_aspect_ratio > 3``). ``NaN`` when
      the minor axis is zero.
    - **circularity** (``esda.shape.isoperimetric_quotient``) — 4π·A / P²;
      1 = perfect circle, lower values indicate elongated or irregular shapes.
    - **solidity** (``esda.shape.convex_hull_ratio``) — polygon area /
      convex-hull area; values < 1 indicate concave or lobular boundaries.
    - **rna_density** — ``obs[counts_key] / area`` in counts/µm². Written only
      when *counts_key* is present in ``obs`` and the area is non-zero.
      With the default ``counts_key='total_counts'`` as written by
      :func:`calculate_qc_metrics` (``exclude_controls=True``), the
      numerator is the **control-excluded** count total — pass
      ``counts_key='total_counts_all_features'`` for the all-features
      density instead.

    Invalid boundary rings (self-intersections such as bow-ties) are
    repaired with :func:`shapely.validation.make_valid`, which keeps the
    **full** geometry (a bow-tie becomes a two-lobe MultiPolygon; the old
    ``buffer(0)`` repair silently discarded one lobe). Geometries that
    cannot be built at all (fewer than 3 vertices, or zero-area/collinear
    rings) get ``NaN`` for **every** metric — never a fake ``0.0`` area.
    Repaired and failed counts are logged.

    Parameters
    ----------
    boundaries_key
        ``obsm`` key of the awkward-array boundary polygons
        (``'cell_boundaries'`` or ``'nucleus_boundaries'``).
    prefix
        Column name prefix (e.g. ``'cell_'`` → ``'cell_circularity'``).
        Defaults to ``'cell_'`` for cell boundaries, ``'nucleus_'`` for
        nucleus boundaries; otherwise the key name with trailing
        ``'_boundaries'`` stripped.
    counts_key
        ``obs`` column containing raw transcript/count totals used to
        compute ``rna_density``.  Set to ``None`` to skip RNA density.
        The column is written by :func:`calculate_qc_metrics`
        (default ``'total_counts'``).
    nucleus_key
        ``obsm`` key of the nucleus boundary polygons used to direct the
        bbox axes into the polarity frame (skipped with a log message when
        absent, or when it equals *boundaries_key*).  Set to ``None`` to
        skip the polarity columns.

    Examples
    --------
    >>> ab.pp.calculate_qc_metrics(mdata)          # populates total_counts
    >>> ab.pp.calculate_shape_metrics(mdata)
    >>> ab.pp.calculate_shape_metrics(mdata, boundaries_key='nucleus_boundaries')
    """
    import geopandas as gpd
    from esda.shape import convex_hull_ratio, isoperimetric_quotient

    if boundaries_key not in adata.obsm:
        raise KeyError(f"obsm['{boundaries_key}'] not found")

    if prefix is None:
        if boundaries_key == "cell_boundaries":
            prefix = "cell_"
        elif boundaries_key == "nucleus_boundaries":
            prefix = "nucleus_"
        else:
            prefix = boundaries_key.replace("_boundaries", "") + "_"

    raw = adata.obsm[boundaries_key]
    polys, unbuildable, n_repaired = _build_polygons(raw)

    if n_repaired or unbuildable.any():
        log_fn = log.warning if unbuildable.any() else log.info
        log_fn(
            "calculate_shape_metrics: repaired %d invalid ring(s) via "
            "shapely.make_valid; %d unbuildable geometrie(s) get all-NaN metrics.",
            n_repaired, int(unbuildable.sum()),
        )

    gs = gpd.GeoSeries(polys)

    def _masked(values) -> np.ndarray:
        """float32 with unbuildable-geometry rows forced to NaN."""
        arr = np.asarray(values, dtype=np.float32).copy()
        arr[unbuildable] = np.nan
        return arr

    # ── Shapely vectorised metrics ────────────────────────────────────────────
    area_col = f"{prefix}area"
    if area_col in adata.obs.columns:
        # Vendor-supplied readers (read_xenium/read_merscope/read_cosmx) all
        # populate obs['cell_area']/['nucleus_area'] already -- this
        # polygon-based recomputation used to silently overwrite it under
        # the same column name, changing what the column *means* with no
        # warning. Worst on MERSCOPE, where CLAUDE.md documents
        # obs['cell_area'] as a bounding-box proxy (not a true polygon
        # area) -- the overwrite silently swaps in a different quantity
        # under an unchanged name. Still overwritten by default (this
        # function's entire purpose is to (re)compute polygon-derived shape
        # metrics, and renaming the output would break every existing
        # downstream reference to obs['cell_area']), but no longer silent.
        log.warning(
            "calculate_shape_metrics: overwriting existing obs['%s'] with a "
            "polygon-based recomputation from obsm['%s'] -- if this was "
            "vendor-supplied (e.g. MERSCOPE's bounding-box-derived "
            "cell_area), its meaning is now different, not just its "
            "precision.",
            area_col, boundaries_key,
        )
    area = _masked(gs.area)
    adata.obs[area_col] = area
    adata.obs[f"{prefix}perimeter"] = _masked(gs.length)

    bounds = gs.bounds  # DataFrame: minx, miny, maxx, maxy
    adata.obs[f"{prefix}span_x"] = _masked(bounds["maxx"] - bounds["minx"])
    adata.obs[f"{prefix}span_y"] = _masked(bounds["maxy"] - bounds["miny"])

    adata.obs[f"{prefix}equivalent_radius"] = np.sqrt(area / np.pi).astype(np.float32)

    # bbox_ prefix: these are the minimum-rotated-rectangle side lengths,
    # not the regionprops-style image-moment ellipse axes.
    major, minor, orientation = _min_rect_axes(gs)
    adata.obs[f"{prefix}bbox_major_axis_length"] = _masked(major)
    adata.obs[f"{prefix}bbox_minor_axis_length"] = _masked(minor)
    adata.obs[f"{prefix}bbox_orientation"] = _masked(orientation)
    with np.errstate(invalid="ignore", divide="ignore"):
        aspect = np.where(minor > 0, major / minor, np.nan).astype(np.float32)
    adata.obs[f"{prefix}aspect_ratio"] = _masked(aspect)

    # ── Nucleus-anchored polarity frame ──────────────────────────────────────
    # Directs the (undirected) bbox axes with the nucleus position: each
    # axis's positive end is the half of the cell farther from the nucleus,
    # i.e. the sign of its dot product with (cell centroid − nucleus
    # centroid). Turns the mod-π orientation into a full per-cell frame —
    # what pl.direction_histogram's default cell-axis rose reads.
    polarity_cols: list[str] = []
    if (
        nucleus_key is not None
        and nucleus_key != boundaries_key
        and nucleus_key in adata.obsm
    ):
        nuc_polys, nuc_unbuildable, nuc_repaired = _build_polygons(
            adata.obsm[nucleus_key]
        )
        ngs = gpd.GeoSeries(nuc_polys)
        if nuc_repaired or nuc_unbuildable.any():
            log.info(
                "calculate_shape_metrics: nucleus polarity — repaired %d nucleus "
                "ring(s); %d cell(s) without a usable nucleus get NaN polarity.",
                nuc_repaired, int(nuc_unbuildable.sum()),
            )

        cell_cent = np.column_stack([gs.centroid.x.to_numpy(), gs.centroid.y.to_numpy()])
        nuc_cent = np.column_stack([ngs.centroid.x.to_numpy(), ngs.centroid.y.to_numpy()])
        offset = cell_cent - nuc_cent  # µm vector pointing away from the nucleus
        bad = (
            unbuildable
            | nuc_unbuildable
            | ~np.isfinite(offset).all(axis=1)
            | ~np.isfinite(orientation)
        )
        offset = np.where(bad[:, None], 0.0, offset)

        theta = np.where(np.isfinite(orientation), orientation, 0.0).astype(np.float64)
        with np.errstate(invalid="ignore"):
            u = np.column_stack([np.cos(theta), np.sin(theta)])
            major_dir = np.where((u * offset).sum(axis=1) < 0, theta + np.pi, theta)
            theta_v = theta + np.pi / 2
            v = np.column_stack([np.cos(theta_v), np.sin(theta_v)])
            minor_dir = np.where((v * offset).sum(axis=1) < 0, theta_v + np.pi, theta_v)

        major_dir = np.mod(major_dir, 2 * np.pi).astype(np.float32)
        minor_dir = np.mod(minor_dir, 2 * np.pi).astype(np.float32)
        off_um = np.linalg.norm(offset, axis=1).astype(np.float32)
        major_dir[bad] = np.nan
        minor_dir[bad] = np.nan
        off_um[bad] = np.nan
        adata.obs[f"{prefix}polarity_major_angle"] = major_dir
        adata.obs[f"{prefix}polarity_minor_angle"] = minor_dir
        adata.obs[f"{prefix}nucleus_offset_um"] = off_um
        polarity_cols = [
            f"{prefix}polarity_major_angle",
            f"{prefix}polarity_minor_angle",
            f"{prefix}nucleus_offset_um",
        ]
    elif nucleus_key is not None and nucleus_key != boundaries_key:
        log.info(
            "calculate_shape_metrics: obsm['%s'] not found — skipping the "
            "nucleus-anchored polarity frame (polarity_major/minor_angle).",
            nucleus_key,
        )

    # ── esda shape metrics ────────────────────────────────────────────────────
    # circularity/solidity are safe to compute directly on the whole
    # GeoSeries: esda.shape.isoperimetric_quotient/convex_hull_ratio use
    # shapely.area/length/convex_hull, which give one 0 (-> NaN via 0/0,
    # with an expected RuntimeWarning) per input geometry, empty or not --
    # no length mismatch.
    with np.errstate(invalid="ignore", divide="ignore"):
        adata.obs[f"{prefix}circularity"] = _masked(isoperimetric_quotient(gs))
        adata.obs[f"{prefix}solidity"] = _masked(convex_hull_ratio(gs))

    # ── RNA density ───────────────────────────────────────────────────────────
    new_cols = [
        f"{prefix}area", f"{prefix}perimeter", f"{prefix}span_x", f"{prefix}span_y",
        f"{prefix}equivalent_radius", f"{prefix}bbox_major_axis_length",
        f"{prefix}bbox_minor_axis_length", f"{prefix}bbox_orientation",
        f"{prefix}aspect_ratio", f"{prefix}circularity", f"{prefix}solidity",
    ]
    new_cols.extend(polarity_cols)
    if counts_key is not None and counts_key in adata.obs.columns:
        counts = adata.obs[counts_key].to_numpy(dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            density = np.where(area > 0, counts / area, np.nan).astype(np.float32)
        adata.obs[f"{prefix}rna_density"] = density
        new_cols.append(f"{prefix}rna_density")
    elif counts_key is not None and counts_key not in adata.obs.columns:
        log.debug(
            "calculate_shape_metrics: obs['%s'] not found; skipping rna_density. "
            "Run ab.pp.calculate_qc_metrics() first.",
            counts_key,
        )

    log.info("calculate_shape_metrics: wrote %s", ", ".join(new_cols))


@with_adata
def calculate_transcript_assignment_qc(
    adata: ad.AnnData,
    *,
    transcripts=None,
    cell_id_col: str = "cell_id",
    gene_col: str = "feature_name",
    qv_col: str = "qv",
    qv_min: float | None = 20.0,
    unassigned_id: int | str | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Compute per-gene transcript assignment rates from the full transcript table.

    Reads the transcript-level table (which includes transcripts not mapped to
    any cell) and adds three columns to ``var``:

    - ``n_transcripts_total`` — total transcripts detected for each gene
    - ``n_transcripts_unassigned`` — transcripts with ``cell_id == unassigned_id``
    - ``pct_transcripts_unassigned`` — percentage unassigned (0–100)

    High ``pct_transcripts_unassigned`` indicates a gene whose transcripts
    frequently fall outside segmented cells, which may reflect probe
    performance issues, subcellular localisation, or over-segmentation.

    .. note::
        Requires the transcript table to be loaded first::

            ab.io.read_xenium_transcripts(mdata)
            # or pass transcripts= directly

        The stored table must be the **unfiltered** one: astrolabe's cell/gene
        filters (:func:`filter_cells`, :func:`filter_genes`,
        :func:`filter_isolated`, and the :mod:`~astrolabe.pipeline` recipes)
        prune ``uns['transcripts']`` rows in sync and set
        ``uns['transcripts_pruned'] = True``; computing assignment rates from
        a pruned table inflates ``pct_transcripts_unassigned`` by
        construction (assigned rows of removed cells vanish from the
        denominator while unassigned rows all survive), so this function
        raises in that case — run it *before* filtering, or pass a freshly
        loaded ``transcripts=`` table.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    transcripts
        PyArrow ``Table`` with at least ``cell_id_col`` and ``gene_col``
        columns.  If ``None``, reads from ``adata.uns['transcripts']``.
    cell_id_col
        Column in the transcript table identifying the assigned cell.
    gene_col
        Column in the transcript table with gene/feature names.
    qv_col
        Column with per-transcript calibrated Q-scores (ignored when absent).
    qv_min
        Minimum Q-score a transcript must reach to be counted, applied to
        both the total and unassigned tallies. The default ``20.0`` matches
        the convention ``X`` itself is built with
        (:func:`~astrolabe.io.read_xenium_transcripts`'s ``min_qv=20``
        default and Xenium Onboard Analysis's own cell-feature matrix), so
        the assignment rates describe the same transcript population as the
        count matrix. Idempotent when the table was already Q-filtered on
        load. ``None`` disables the filter (counts every detected
        transcript, including low-quality calls ``X`` excludes).
    unassigned_id
        Value in ``cell_id_col`` that marks an unassigned transcript.
        ``None`` (default) auto-detects from ``cell_id_col``'s actual
        column type: real Xenium Onboard Analysis (XOA) >= 2.0 uses the
        literal string ``'UNASSIGNED'`` for a string-typed ``cell_id``
        column; older/numeric exports use integer ``0``. Pass explicitly to
        override either convention.
    inplace
        If ``True`` (default), modify *data* in place and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    None or AnnData

    Raises
    ------
    ValueError
        If no transcript table is available, or the stored
        ``uns['transcripts']`` has been pruned by a filtering step
        (``uns['transcripts_pruned']`` set) and no fresh *transcripts* table
        was passed.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if not inplace:
        adata = adata.copy()

    if transcripts is None:
        if "transcripts" not in adata.uns:
            raise ValueError(
                "No transcript table found in adata.uns['transcripts']. "
                "Call ab.io.read_xenium_transcripts(mdata) first, or pass transcripts="
            )
        if adata.uns.get("transcripts_pruned", False):
            raise ValueError(
                "uns['transcripts'] has been pruned by a filtering step "
                "(uns['transcripts_pruned'] is set) — assignment rates computed "
                "from a pruned table are inflated by construction. Run "
                "calculate_transcript_assignment_qc BEFORE filtering, or pass a "
                "freshly loaded table via transcripts= "
                "(ab.io.read_xenium_transcripts(path))."
            )
        transcripts = adata.uns["transcripts"]

    if qv_min is not None and qv_col in transcripts.schema.names:
        # Match X's own transcript population (min_qv=20 on load / XOA's
        # matrix convention) so numerator and denominator describe the same
        # calls regardless of how the table itself was loaded.
        n_before_qv = len(transcripts)
        transcripts = transcripts.filter(
            pc.greater_equal(transcripts[qv_col], qv_min)
        )
        if len(transcripts) != n_before_qv:
            log.info(
                "calculate_transcript_assignment_qc: %d / %d transcripts below "
                "qv_min=%.1f excluded.",
                n_before_qv - len(transcripts), n_before_qv, qv_min,
            )

    # Total transcripts per gene
    total_tbl = (
        transcripts
        .group_by(gene_col)
        .aggregate([(gene_col, "count")])
    )
    total_df = total_tbl.to_pandas().rename(
        columns={gene_col: "gene", f"{gene_col}_count": "n_total"}
    ).set_index("gene")

    # Unassigned transcripts per gene — cast unassigned_id to match the column type
    # so pc.equal works regardless of whether cell_id is stored as int or string.
    col_type = transcripts.schema.field(cell_id_col).type
    is_string_col = pa.types.is_string(col_type) or pa.types.is_large_string(col_type)
    if unassigned_id is None:
        # Default sentinel silently reported 0% unassigned for every gene on
        # real Xenium Onboard Analysis (XOA) >= 2.0 output -- its
        # string-typed cell_id column uses the literal 'UNASSIGNED', never
        # the numeric 0 this used to always default to (reading as a clean
        # QC result while actually testing for a value that never occurs).
        unassigned_scalar = "UNASSIGNED" if is_string_col else 0
    elif is_string_col:
        unassigned_scalar = str(unassigned_id)
    else:
        unassigned_scalar = unassigned_id
    unassigned_tbl = (
        transcripts
        .filter(pc.equal(transcripts[cell_id_col], unassigned_scalar))
        .group_by(gene_col)
        .aggregate([(gene_col, "count")])
    )
    unassigned_df = unassigned_tbl.to_pandas().rename(
        columns={gene_col: "gene", f"{gene_col}_count": "n_unassigned"}
    ).set_index("gene")

    # Align to var_names (genes in X may differ from those in transcript table)
    n_total = total_df["n_total"].reindex(adata.var_names).fillna(0).astype(np.int64)
    n_unassigned = unassigned_df["n_unassigned"].reindex(adata.var_names).fillna(0).astype(np.int64)
    pct = np.where(n_total > 0, 100.0 * n_unassigned / n_total, np.nan)

    adata.var["n_transcripts_total"] = n_total.to_numpy()
    adata.var["n_transcripts_unassigned"] = n_unassigned.to_numpy()
    adata.var["pct_transcripts_unassigned"] = pct.astype(np.float32)

    log.info(
        "calculate_transcript_assignment_qc: %d genes; median pct_unassigned=%.1f%%",
        adata.n_vars,
        float(np.nanmedian(pct)),
    )

    if not inplace:
        return adata
    return None