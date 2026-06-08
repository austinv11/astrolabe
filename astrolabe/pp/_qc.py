"""
QC metric calculation and cell/gene filtering for spatial transcriptomics data.
"""

from __future__ import annotations

import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

import scanpy as sc

from astrolabe._utils import _resolve_adata, with_adata

__all__ = ["filter_cells", "filter_genes", "calculate_qc_metrics", "calculate_xenium_qc_metrics"]

log = logging.getLogger(__name__)


@with_adata
def calculate_qc_metrics(
    adata: ad.AnnData,
    *,
    qc_vars: list[str] | tuple[str, ...] = (),
    percent_top: list[int] | tuple[int, ...] | None = (50, 100, 200, 500),
    log1p: bool = True,
    inplace: bool = True,
):
    """Compute per-cell and per-gene QC metrics.

    Thin wrapper around :func:`scanpy.pp.calculate_qc_metrics` that accepts
    :class:`~mudata.MuData` via the ``modality`` kwarg.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    qc_vars
        ``var`` column names that flag feature subsets (e.g. control probes).
        A ``pct_counts_<name>`` column is added to ``obs`` for each.
    percent_top
        Compute cumulative fractions for these top-N gene counts.
        ``None`` to skip.
    log1p
        Whether to add ``log1p``-transformed counts alongside raw counts.
    inplace
        Modify *data* in place (default) or return ``(obs_df, var_df)``.

    Returns
    -------
    None, or ``(obs_df, var_df)`` when ``inplace=False``.
    """
    return sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=percent_top,
        log1p=log1p,
        inplace=inplace,
    )


# Xenium utilities
def calculate_xenium_qc_metrics(
        data: ad.AnnData | md.MuData,
        transcripts=None,
        *,
        modality: str | None = "xenium",
        qv_col: str = "qv",
        gene_col: str = "feature_name",
        remove_controls: bool = True,
        inplace: bool = True,
) -> ad.AnnData | None:
    """Compute per-cell and per-gene Xenium probe-specific QC metrics and store in obs/var.

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

    pct_unassigned_codeword
        Percentage of counts from unassigned codewords out of all uniquely detected transcripts.
    pct_deprecated_codeword
        Percentage of counts from deprecated codewords out of all uniquely detected transcripts.
    pct_genomic_control_probe
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
        Column in *transcripts* containing per-transcript quality values.
    gene_col
        Column in *transcripts* containing gene/feature names.
    remove_controls
        When ``True`` (the default), remove all features that are not gene expression after computing summary stats.
    inplace
        If ``True`` (default), modify the input AnnData in place and return ``None``. If ``False``, return a modified copy.

    Returns
    -------
    None or AnnData
        ``None`` if ``inplace=True``; modified copy if ``inplace=False``.
    """
    adata_ref = _resolve_adata(data, modality)
    adata = adata_ref.copy() if not inplace else adata_ref

    unassigned_mask = adata.var['feature_type'] == 'Unassigned Codeword'
    deprecated_mask = adata.var['feature_type'] == 'Deprecated Codeword'
    genomic_control_mask = adata.var['feature_type'] == 'Genomic Control'
    negative_control_codeword_mask = adata.var['feature_type'] == 'Negative Control Codeword'
    negative_control_probe_mask = adata.var['feature_type'] == 'Negative Control Probe'
    any_negative_control_mask = unassigned_mask | deprecated_mask | genomic_control_mask | negative_control_codeword_mask | negative_control_probe_mask

    if any_negative_control_mask.sum() < 1:
        log.debug(
            "No control probes found in var['feature_type']; skipping Xenium QC metrics.",
        )
        raise ValueError("No control probes found in var['feature_type']; cannot compute Xenium QC metrics.")

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

    total_counts = _row_sum(adata.X)
    adata.obs['pct_unassigned_codewords'] = adata.obs['n_unassigned_codewords'] / total_counts * 100
    adata.obs['pct_deprecated_codewords'] = adata.obs['n_deprecated_codewords'] / total_counts * 100
    adata.obs['pct_genomic_control_probes'] = adata.obs['n_genomic_control_probes'] / total_counts * 100
    adata.obs['pct_negative_control_codewords'] = adata.obs['n_negative_control_codewords'] / total_counts * 100
    adata.obs['pct_negative_control_probes'] = adata.obs['n_negative_control_probes'] / total_counts * 100
    adata.obs['pct_any_negative_control_probes'] = adata.obs['n_any_negative_control_probes'] / total_counts * 100

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

    # Stash per-gene control stats before optional removal so gene_qv_scatter
    # can still show individual control probe points even after controls are removed from var.
    ctrl_bool = adata.var['is_control'].values
    if ctrl_bool.any():
        col_sums = np.asarray(adata.X.sum(axis=0)).ravel()
        ctrl_df = adata.var.loc[ctrl_bool, ['feature_type']].copy()
        ctrl_df['total_counts'] = col_sums[ctrl_bool]
        if 'mean_qv' in adata.var.columns:
            ctrl_df['mean_qv'] = adata.var.loc[ctrl_bool, 'mean_qv'].values
        adata.uns['control_probe_summary'] = ctrl_df.reset_index().to_dict(orient='list')

    if remove_controls:
        adata._inplace_subset_var(~adata.var['is_control'])

    return None if inplace else adata


def filter_cells(
    data: ad.AnnData | md.MuData,
    *,
    modality: str | None = None,
    min_counts: int | None = None,
    max_counts: int | None = None,
    min_genes: int | None = None,
    max_genes: int | None = None,
    max_pct_control_probe: float | None = None,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    inplace: bool = True,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Filter cells by QC thresholds.

    All thresholds are inclusive on the retained side (i.e.
    ``min_counts=100`` keeps cells with total_counts >= 100).

    Parameters
    ----------
    adata
        AnnData.  ``calculate_qc_metrics`` should have been run first if
        using count-based thresholds.
    min_counts, max_counts
        Total UMI count bounds.  Requires ``obs['total_counts']``.
    min_genes, max_genes
        Detected gene count bounds.  Requires ``obs['n_genes_by_counts']``.
    max_pct_control_probe
        Maximum allowed percentage of control probe counts.
        Requires ``obs['pct_counts_control_probe']``.
    min_cell_area, max_cell_area
        Cell area bounds in µm².  Requires ``obs['cell_area']``.
    inplace
        If ``True`` (default), subset ``adata`` in place and return ``None``.
        If ``False``, return ``(filtered_adata, bool_mask)`` where the mask
        is ``True`` for retained cells.

    Returns
    -------
    None or (AnnData, ndarray)
    """
    adata = _resolve_adata(data, modality)
    mask = np.ones(adata.n_obs, dtype=bool)

    def _apply(col: str, lo=None, hi=None):
        nonlocal mask
        if col not in adata.obs.columns:
            raise KeyError(
                f"Column '{col}' not found in obs.  Run pp.calculate_qc_metrics first."
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
    if max_pct_control_probe is not None:
        _apply("pct_counts_control_probe", hi=max_pct_control_probe)
    if min_cell_area is not None or max_cell_area is not None:
        _apply("cell_area", lo=min_cell_area, hi=max_cell_area)

    n_removed = (~mask).sum()
    log.info("filter_cells: removing %d / %d cells", n_removed, adata.n_obs)

    if inplace:
        adata._inplace_subset_obs(mask)
        return None
    else:
        return adata[mask].copy(), mask


def filter_genes(
    data: ad.AnnData | md.MuData,
    *,
    modality: str | None = None,
    min_counts: int | None = None,
    min_cells: int | None = None,
    exclude_controls: bool = False,
    keep_feature_types: list[str] | None = None,
    inplace: bool = True,
) -> ad.AnnData | tuple[ad.AnnData, np.ndarray] | None:
    """Filter genes/features by expression or type criteria.

    Parameters
    ----------
    adata
        AnnData with ``var['is_control_probe']`` and ``var['feature_type']``
        (populated by :func:`astrolabe.io.read_xenium`).
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
        If ``True`` (default), subset in place.  If ``False``, return
        ``(filtered_adata, bool_mask)``.

    Returns
    -------
    None or (AnnData, ndarray)
    """
    adata = _resolve_adata(data, modality)
    mask = np.ones(adata.n_vars, dtype=bool)

    if min_counts is not None:
        if "total_counts" not in adata.var.columns:
            raise KeyError("var['total_counts'] not found. Run pp.calculate_qc_metrics first.")
        mask &= adata.var["total_counts"].to_numpy(dtype=np.float64) >= min_counts

    if min_cells is not None:
        if "n_cells_by_counts" not in adata.var.columns:
            raise KeyError("var['n_cells_by_counts'] not found. Run pp.calculate_qc_metrics first.")
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
            "before calling filter_genes with keep_feature_types."
        )
        mask &= adata.var["feature_type"].isin(keep_feature_types).to_numpy()

    n_removed = (~mask).sum()
    log.info("filter_genes: removing %d / %d features", n_removed, adata.n_vars)

    if inplace:
        adata._inplace_subset_var(mask)
        return None
    else:
        return adata[:, mask].copy(), mask