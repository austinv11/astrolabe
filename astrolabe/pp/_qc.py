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
        qc_percent_top = [n for n in qc_percent_top if n <= n_vars] or None

    if gpu_enabled():
        rsc = _require_rapids()
        with _gpu_X(qc_adata):
            result = rsc.pp.calculate_qc_metrics(
                qc_adata,
                qc_vars=qc_vars,
                percent_top=qc_percent_top,
                log1p=log1p,
                inplace=inplace or ctrl_mask is not None,
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
        return result

    # qc_adata is a separate copy (subset to real genes); merge its computed
    # obs/var columns back onto the full adata rather than returning/assigning
    # metrics for a gene subset.
    obs_df = qc_adata.obs.drop(columns=adata.obs.columns, errors="ignore")
    var_df = qc_adata.var.drop(columns=adata.var.columns, errors="ignore")
    var_df = var_df.reindex(adata.var_names)

    if inplace:
        for col in obs_df.columns:
            adata.obs[col] = obs_df[col]
        for col in var_df.columns:
            adata.var[col] = var_df[col]
        _add_nucleus_metrics(adata)
        return None
    return obs_df, var_df


# Xenium utilities
@with_adata
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
        Percentage of counts from unassigned codewords out of all uniquely detected transcripts.
    pct_deprecated_codewords
        Percentage of counts from deprecated codewords out of all uniquely detected transcripts.
    pct_genomic_control_probes
        Percentage of counts from genomic control probes out of all uniquely detected transcripts.
    pct_negative_control_codewords
        Percentage of counts from negative control codewords that are also negative controls out of all uniquely detected transcripts.
    pct_negative_control_probes
        Percentage of counts from negative control probes out of all uniquely detected transcripts.
    pct_any_negative_control_probes
        Percentage of counts from any negative control probes (including unassigned codewords that are negative controls) out of all uniquely detected transcripts.

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
        def _pct(n_col: str) -> np.ndarray:
            return np.where(total_counts > 0, adata.obs[n_col].to_numpy(np.float64) / total_counts * 100.0, 0.0)

        adata.obs['pct_unassigned_codewords'] = _pct('n_unassigned_codewords')
        adata.obs['pct_deprecated_codewords'] = _pct('n_deprecated_codewords')
        adata.obs['pct_genomic_control_probes'] = _pct('n_genomic_control_probes')
        adata.obs['pct_negative_control_codewords'] = _pct('n_negative_control_codewords')
        adata.obs['pct_negative_control_probes'] = _pct('n_negative_control_probes')
        adata.obs['pct_any_negative_control_probes'] = _pct('n_any_negative_control_probes')

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
        ctrl_df = adata.var.loc[ctrl_bool, ['feature_type']].copy()
        ctrl_df['total_counts'] = col_sums[ctrl_bool]
        if 'mean_qv' in adata.var.columns:
            ctrl_df['mean_qv'] = adata.var.loc[ctrl_bool, 'mean_qv'].values
        adata.uns['control_probe_summary'] = ctrl_df.reset_index().to_dict(orient='list')

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

    Sets ``var[control_key]`` (bool) and ``obs[obs_key]`` (float, 0–100).
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
        pct = np.where(total_counts > 0, ctrl_counts / total_counts * 100.0, 0.0)

    adata.obs[obs_key] = pct

    log.info(
        "flag_controls: %d / %d features flagged as controls; "
        "obs['%s'] computed (median %.2f%%, max %.2f%%)",
        int(ctrl_mask.sum()), adata.n_vars,
        obs_key, float(np.median(pct)), float(pct.max()),
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

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    min_counts, max_counts
        Total UMI count bounds.  Requires ``obs['total_counts']``.
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

    if inplace:
        adata._inplace_subset_obs(mask)
        return None
    else:
        return adata[mask].copy(), mask


@with_adata
def filter_genes(
    adata: ad.AnnData,
    *,
    min_counts: int | None = None,
    min_cells: int | None = None,
    exclude_controls: bool = False,
    keep_feature_types: list[str] | None = None,
    inplace: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Filter genes/features by expression or type criteria.

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

    if min_counts is not None:
        if "total_counts" not in adata.var.columns:
            raise KeyError("var['total_counts'] not found. Run pp.calculate_qc_metrics first")
        mask &= adata.var["total_counts"].to_numpy(dtype=np.float64) >= min_counts

    if min_cells is not None:
        if "n_cells_by_counts" not in adata.var.columns:
            raise KeyError("var['n_cells_by_counts'] not found. Run pp.calculate_qc_metrics first")
        mask &= adata.var["n_cells_by_counts"].to_numpy(dtype=np.int32) >= min_cells

    if exclude_controls:
        if "is_control_probe" not in adata.var.columns:
            log.warning("var['is_control_probe'] not found; cannot exclude controls.")
        else:
            mask &= ~adata.var["is_control_probe"].to_numpy(dtype=bool)

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

    if inplace:
        adata._inplace_subset_var(mask)
        return None
    else:
        return adata[:, mask].copy(), mask


@with_adata
def filter_isolated(
    adata: ad.AnnData,
    *,
    neighbors_key: str = "spatial_neighbors",
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    inplace: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Remove spatially isolated cells using a pre-computed kNN distance graph.

    Reads per-cell neighbor distances from
    ``obsp[neighbors_key + '_distances']`` (written by
    :func:`~astrolabe.tl.spatial_neighbors`) and removes cells that fail
    either criterion:

    - **nearest-neighbor distance** ``> min_neighbor_dist_um`` — the cell's
      closest neighbor is farther than this threshold, indicating a truly
      isolated cell.
    - **maximum-neighbor distance** ``> max_neighbor_dist_um`` — the
      farthest stored neighbor exceeds this threshold, indicating the cell
      sits in a very sparse region.

    Both thresholds can be independently disabled by passing ``None``.
    After subsetting, the stored ``obsp`` graph is automatically trimmed by
    AnnData to the retained cells.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    neighbors_key
        Prefix of the ``obsp`` distance matrix written by
        :func:`~astrolabe.tl.spatial_neighbors`.  The full key used is
        ``neighbors_key + '_distances'``.
    max_neighbor_dist_um
        Retain cells whose farthest stored neighbor is within this distance
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
        If the distance matrix is absent — run
        :func:`~astrolabe.tl.spatial_neighbors` first.
    """
    dist_key = neighbors_key + "_distances"
    if dist_key not in adata.obsp:
        raise KeyError(
            f"obsp['{dist_key}'] not found. "
            f"Run ab.tl.spatial_neighbors(data) before ab.pp.filter_isolated()"
        )

    if max_neighbor_dist_um is None and min_neighbor_dist_um is None:
        log.warning("filter_isolated: both thresholds are None — no filtering applied.")
        if inplace:
            return None
        return adata.copy(), np.ones(adata.n_obs, dtype=bool)

    n_cells = adata.n_obs

    # Vectorised per-row min/max via COO representation (O(nnz), no Python loop).
    dist_coo = adata.obsp[dist_key].tocoo()
    min_dists = np.full(n_cells, np.inf, dtype=np.float64)
    max_dists = np.zeros(n_cells, dtype=np.float64)
    if dist_coo.nnz > 0:
        d = dist_coo.data.astype(np.float64)
        np.minimum.at(min_dists, dist_coo.row, d)
        np.maximum.at(max_dists, dist_coo.row, d)

    mask = np.ones(n_cells, dtype=bool)
    if min_neighbor_dist_um is not None:
        mask &= min_dists <= min_neighbor_dist_um
    if max_neighbor_dist_um is not None:
        mask &= max_dists <= max_neighbor_dist_um

    n_removed = int((~mask).sum())
    log.info(
        "filter_isolated: removing %d / %d cells "
        "(neighbors_key='%s', max_neighbor=%.0f um, min_neighbor=%.0f um)",
        n_removed, n_cells, neighbors_key,
        max_neighbor_dist_um if max_neighbor_dist_um is not None else float("inf"),
        min_neighbor_dist_um if min_neighbor_dist_um is not None else float("inf"),
    )

    if inplace:
        adata._inplace_subset_obs(mask)
        return None
    else:
        return adata[mask].copy(), mask


# ---------------------------------------------------------------------------
# Shape metrics
# ---------------------------------------------------------------------------


def _min_rect_axes(gs) -> tuple[np.ndarray, np.ndarray]:
    """Return (major_axis, minor_axis) lengths for each polygon's minimum rotated rectangle."""
    majors: list[float] = []
    minors: list[float] = []
    for geom in gs:
        if geom.is_empty:
            majors.append(np.nan)
            minors.append(np.nan)
            continue
        rect = geom.minimum_rotated_rectangle
        coords = np.array(rect.exterior.coords)[:4]
        s1 = float(np.linalg.norm(coords[1] - coords[0]))
        s2 = float(np.linalg.norm(coords[2] - coords[1]))
        if s1 >= s2:
            majors.append(s1); minors.append(s2)
        else:
            majors.append(s2); minors.append(s1)
    return np.array(majors, dtype=np.float32), np.array(minors, dtype=np.float32)


@with_adata
def calculate_shape_metrics(
    adata: ad.AnnData,
    *,
    boundaries_key: str = "cell_boundaries",
    prefix: str | None = None,
    counts_key: str | None = "total_counts",
) -> None:
    """Compute morphological shape metrics from cell or nucleus boundary polygons.

    Writes the following metrics to ``obs`` (all float32):

    - **area** — polygon area in µm².
    - **perimeter** — polygon perimeter in µm.
    - **span_x** / **span_y** — bounding-box width and height in µm.
    - **equivalent_radius** — radius of a circle with the same area (√(A/π)) in µm.
    - **major_axis_length** / **minor_axis_length** — side lengths of the minimum
      rotated bounding rectangle in µm.
    - **aspect_ratio** — ``major_axis_length / minor_axis_length``; 1 = square/circle,
      higher values indicate elongation. ``NaN`` when the minor axis is zero.
    - **circularity** (``esda.shape.isoperimetric_quotient``) — 4π·A / P²;
      1 = perfect circle, lower values indicate elongated or irregular shapes.
    - **solidity** (``esda.shape.convex_hull_ratio``) — polygon area /
      convex-hull area; values < 1 indicate concave or lobular boundaries.
    - **elongation** (``esda.shape.diameter_ratio``) — length / width;
      higher values indicate elongated cells (fibroblasts, neurons).
    - **rna_density** — ``obs[counts_key] / area`` in counts/µm². Written only
      when *counts_key* is present in ``obs`` and the area is non-zero.

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

    Examples
    --------
    >>> ab.pp.calculate_qc_metrics(mdata)          # populates total_counts
    >>> ab.pp.calculate_shape_metrics(mdata)
    >>> ab.pp.calculate_shape_metrics(mdata, boundaries_key='nucleus_boundaries')
    """
    import geopandas as gpd
    from esda.shape import convex_hull_ratio, diameter_ratio, isoperimetric_quotient
    from shapely.geometry import Polygon

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
    polys = []
    for i in range(len(raw)):
        verts = np.asarray(raw[i])
        try:
            p = Polygon(verts)
            if not p.is_valid:
                p = p.buffer(0)
        except Exception:
            p = Polygon()
        polys.append(p)

    gs = gpd.GeoSeries(polys)

    # ── Shapely vectorised metrics ────────────────────────────────────────────
    area = np.asarray(gs.area, dtype=np.float32)
    adata.obs[f"{prefix}area"] = area
    adata.obs[f"{prefix}perimeter"] = np.asarray(gs.length, dtype=np.float32)

    bounds = gs.bounds  # DataFrame: minx, miny, maxx, maxy
    adata.obs[f"{prefix}span_x"] = (bounds["maxx"] - bounds["minx"]).to_numpy(dtype=np.float32)
    adata.obs[f"{prefix}span_y"] = (bounds["maxy"] - bounds["miny"]).to_numpy(dtype=np.float32)

    adata.obs[f"{prefix}equivalent_radius"] = np.sqrt(area / np.pi).astype(np.float32)

    major, minor = _min_rect_axes(gs)
    adata.obs[f"{prefix}major_axis_length"] = major
    adata.obs[f"{prefix}minor_axis_length"] = minor
    with np.errstate(invalid="ignore", divide="ignore"):
        aspect = np.where(minor > 0, major / minor, np.nan).astype(np.float32)
    adata.obs[f"{prefix}aspect_ratio"] = aspect

    # ── esda shape metrics ────────────────────────────────────────────────────
    adata.obs[f"{prefix}circularity"] = np.asarray(isoperimetric_quotient(gs), dtype=np.float32)
    adata.obs[f"{prefix}solidity"] = np.asarray(convex_hull_ratio(gs), dtype=np.float32)
    adata.obs[f"{prefix}elongation"] = np.asarray(diameter_ratio(gs), dtype=np.float32)

    # ── RNA density ───────────────────────────────────────────────────────────
    new_cols = [
        f"{prefix}area", f"{prefix}perimeter", f"{prefix}span_x", f"{prefix}span_y",
        f"{prefix}equivalent_radius", f"{prefix}major_axis_length", f"{prefix}minor_axis_length",
        f"{prefix}aspect_ratio", f"{prefix}circularity", f"{prefix}solidity", f"{prefix}elongation",
    ]
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
    unassigned_id: int | str = 0,
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
    unassigned_id
        Value in ``cell_id_col`` that marks an unassigned transcript.
        ``0`` is the Xenium convention.
    inplace
        If ``True`` (default), modify *data* in place and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    None or AnnData
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
        transcripts = adata.uns["transcripts"]

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
    if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
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