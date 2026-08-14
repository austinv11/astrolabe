"""
astrolabe.pipeline — composable, opinionated analysis recipes.

Each recipe bundles a set of related steps with sensible defaults so that
common workflows can be expressed concisely.  Recipes are designed to be
called in sequence, not as a single monolithic pipeline::

    ab.pipeline.filter_xenium(mdata)
    ab.pp.normalize_total(mdata)
    ab.pp.log1p(mdata)
    ab.tl.spatial_neighbors(mdata)
    ab.tl.cluster(mdata, neighbors_key='spatial_neighbors')

Functions
---------
filter_cells
    Platform-agnostic QC metrics, cell/gene filtering, and spatial
    isolation filtering.  Works with any segmented spatial dataset.
filter_xenium
    Xenium-specific extension of filter_cells: adds probe-type breakdown
    and control-probe metrics before deferring to filter_cells for the
    shared steps.
filter_merscope
    MERSCOPE-specific extension of filter_cells: flags Blank-* control
    probes via var['is_control_probe'] before deferring to filter_cells.
filter_cosmx
    CosMx-specific extension of filter_cells: flags NegPrb/SystemControl
    probes via var['is_control_probe'] before deferring to filter_cells.
"""

from __future__ import annotations

import logging

import anndata as ad
import mudata as md

from astrolabe._constants import UNS_DEFAULT_MODALITY_KEY

__all__ = ["filter_cells", "filter_xenium", "filter_merscope", "filter_cosmx"]

log = logging.getLogger(__name__)


def _log_step(step: str, n_before: int, n_after: int, axis: str = "cells") -> None:
    removed = n_before - n_after
    pct = removed / n_before * 100 if n_before else 0.0
    log.info(
        "  [%s] %d → %d %s  (removed %d, %.1f%%)",
        step,
        n_before,
        n_after,
        axis,
        removed,
        pct,
    )


def filter_cells(
    data: ad.AnnData | md.MuData,
    *,
    modality: str | None = None,
    # --- Cell filtering ---
    min_counts: int = 10,
    min_genes: int = 5,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    max_pct_control: float | None = None,
    # --- Spatial isolation filtering ---
    isolation_k: int = 5,
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    spatial_key: str = "spatial",
    # --- Gene filtering ---
    min_cells: int = 5,
    inplace: bool = True,
) -> ad.AnnData | md.MuData | None:
    """Platform-agnostic QC, cell filtering, and spatial isolation.

    Runs, in order:

    1. :func:`~astrolabe.pp.calculate_qc_metrics` — per-cell/gene counts
       and detected-gene metrics.
    2. :func:`~astrolabe.pp.filter_cells` — apply *min_counts*, *min_genes*,
       optional area bounds, and (if set) *max_pct_control*.
    3. :func:`~astrolabe.pp.filter_genes` — keep genes detected in at least
       *min_cells* cells.
    4. Spatial isolation filter — builds a k=*isolation_k* nearest-neighbor
       graph on ``obsm['spatial']`` and removes isolated or edge cells.
       Disabled by passing ``max_neighbor_dist_um=None`` and
       ``min_neighbor_dist_um=None``.

    Works with any segmented spatial dataset: Xenium, MERSCOPE, CosMx, or
    custom re-segmentations.  For Xenium data, use
    :func:`filter_xenium` which runs probe-type breakdown first and then
    defers to this function for the shared steps.

    Logs cell counts after each filter step at the INFO level.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        For :class:`~mudata.MuData`: modality key to filter.  ``None``
        resolves via ``uns['default_modality']``.
    min_counts
        Minimum total transcript counts per cell.
    min_genes
        Minimum number of genes detected per cell.
    min_cell_area
        Minimum cell area in µm².  ``None`` = no lower bound.
    max_cell_area
        Maximum cell area in µm².  ``None`` = no upper bound.
    max_pct_control
        Maximum percentage of counts from control probes per cell.
        ``None`` skips this filter (appropriate for non-Xenium data).
    isolation_k
        Number of nearest spatial neighbors for the isolation filter.
    max_neighbor_dist_um
        Remove cells whose *k*-th nearest neighbor exceeds this distance
        in µm.  ``None`` disables this threshold.
    min_neighbor_dist_um
        Remove cells whose nearest neighbor exceeds this distance in µm.
        ``None`` disables this threshold.
    spatial_key
        ``obsm`` key for cell centroid coordinates.
    min_cells
        Minimum number of cells a gene must be detected in.
    inplace
        Modify *data* in place and return ``None`` (default).

    Returns
    -------
    None, AnnData, or MuData

    Examples
    --------
    >>> # MERSCOPE / CosMx data after custom reading
    >>> ab.pipeline.filter_cells(mdata)
    >>> ab.pp.normalize_total(mdata)
    """
    import astrolabe.pp as pp
    import astrolabe.tl as tl
    from astrolabe._utils import _resolve_adata

    if not inplace:
        data = data.copy()

    adata = _resolve_adata(data, modality)
    mod_key = modality or (
        data.uns.get(UNS_DEFAULT_MODALITY_KEY) if isinstance(data, md.MuData) else None
    )

    log.info("filter_cells: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # 1. Generic QC metrics
    pp.calculate_qc_metrics(data, modality=modality)

    # 2. Cell filtering
    n_before = adata.n_obs
    pp.filter_cells(
        data,
        modality=modality,
        min_counts=min_counts,
        min_genes=min_genes,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        **({"max_pct_control": max_pct_control} if max_pct_control is not None else {}),
        inplace=True,
    )
    _log_step("filter_cells", n_before, adata.n_obs)

    # 3. Gene filtering
    n_before = adata.n_vars
    pp.filter_genes(data, modality=modality, min_cells=min_cells, inplace=True)
    _log_step("filter_genes", n_before, adata.n_vars, axis="genes")

    # 4. Spatial isolation filter
    if max_neighbor_dist_um is not None or min_neighbor_dist_um is not None:
        n_before = adata.n_obs
        log.info("  spatial_neighbors (k=%d)", isolation_k)
        tl.spatial_neighbors(
            data,
            modality=modality,
            n_neighbors=isolation_k,
            spatial_key=spatial_key,
        )
        pp.filter_isolated(
            data,
            modality=modality,
            neighbors_key="spatial_neighbors",
            max_neighbor_dist_um=max_neighbor_dist_um,
            min_neighbor_dist_um=min_neighbor_dist_um,
            inplace=True,
        )
        _log_step("filter_isolation", n_before, adata.n_obs)

    log.info("filter_cells: done — %d cells × %d genes", adata.n_obs, adata.n_vars)
    return None if inplace else data


def filter_xenium(
    mdata: md.MuData,
    *,
    xenium_key: str = "xenium",
    # --- Xenium-specific QC ---
    transcripts=None,
    remove_controls: bool = True,
    max_pct_control: float = 10.0,
    # --- Shared with filter_cells ---
    min_counts: int = 10,
    min_genes: int = 5,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    isolation_k: int = 5,
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    spatial_key: str = "spatial",
    min_cells: int = 5,
    inplace: bool = True,
) -> md.MuData | None:
    """Xenium-specific QC metrics and filtering.

    Extends :func:`filter_cells` with a Xenium-specific first step:

    1. :func:`~astrolabe.pp.calculate_xenium_qc_metrics` — probe-type
       breakdown, control flagging, optional per-gene Q-scores from
       *transcripts*.
    2–5. All shared steps from :func:`filter_cells` (``calculate_qc_metrics``,
       ``filter_cells``, ``filter_genes``, spatial isolation).
    6. Optionally removes control probe features from ``X`` when
       *remove_controls* is ``True``.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`~astrolabe.io.read_xenium`.
    xenium_key
        Key for the Xenium modality inside *mdata*.
    transcripts
        Optional :class:`pyarrow.Table` from
        :func:`~astrolabe.io.read_xenium_transcripts` for per-gene Q-score
        metrics.
    remove_controls
        Remove control probe features after computing QC metrics.
    max_pct_control
        Maximum percentage of counts from control probes per cell.
    min_counts
        Minimum total transcript counts per cell.
    min_genes
        Minimum number of genes detected per cell.
    min_cell_area
        Minimum cell area in µm².
    max_cell_area
        Maximum cell area in µm².
    isolation_k
        Number of nearest spatial neighbors for the isolation filter.
    max_neighbor_dist_um
        Maximum distance to the *k*-th nearest spatial neighbor in µm.
    min_neighbor_dist_um
        Remove cells whose nearest neighbor exceeds this distance (µm).
        Threshold applies to the 1st neighbor; cf. *max_neighbor_dist_um*
        which uses the k-th neighbor.
    spatial_key
        ``obsm`` key for cell centroids.
    min_cells
        Minimum number of cells a gene must be detected in.
    inplace
        Modify *mdata* in place and return ``None`` (default).

    Returns
    -------
    None or MuData

    Examples
    --------
    >>> ab.pipeline.filter_xenium(mdata)
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.cluster(mdata, neighbors_key='spatial_neighbors')
    """
    import astrolabe.pp as pp

    if not isinstance(mdata, md.MuData):
        raise TypeError(f"filter_xenium expects a MuData; got {type(mdata).__name__}")
    if xenium_key not in mdata.mod:
        raise KeyError(f"Modality '{xenium_key}' not found in mdata")

    if not inplace:
        mdata = mdata.copy()

    adata = mdata[xenium_key]
    log.info("filter_xenium: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # Xenium-specific: probe breakdown + control flagging
    log.info("  calculate_xenium_qc_metrics")
    pp.calculate_xenium_qc_metrics(
        mdata,
        transcripts=transcripts,
        modality=xenium_key,
    )

    # Shared steps
    filter_cells(
        mdata,
        modality=xenium_key,
        min_counts=min_counts,
        min_genes=min_genes,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        max_pct_control=max_pct_control,
        isolation_k=isolation_k,
        max_neighbor_dist_um=max_neighbor_dist_um,
        min_neighbor_dist_um=min_neighbor_dist_um,
        spatial_key=spatial_key,
        min_cells=min_cells,
        inplace=True,
    )

    # Remove control probe features from X
    if remove_controls and "is_control_probe" in adata.var.columns:
        n_before = adata.n_vars
        pp.filter_genes(mdata, modality=xenium_key, exclude_controls=True, inplace=True)
        _log_step("remove_controls", n_before, adata.n_vars, axis="genes")

    log.info(
        "filter_xenium: done — %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )
    return None if inplace else mdata


def filter_merscope(
    mdata: md.MuData,
    *,
    merscope_key: str = "merscope",
    # --- MERSCOPE-specific ---
    remove_controls: bool = True,
    max_pct_control: float = 5.0,
    # --- Shared with filter_cells ---
    min_counts: int = 10,
    min_genes: int = 5,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    isolation_k: int = 5,
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    spatial_key: str = "spatial",
    min_cells: int = 5,
    inplace: bool = True,
) -> md.MuData | None:
    """MERSCOPE-specific QC metrics and filtering.

    Extends :func:`filter_cells` with a MERSCOPE-specific first step:

    1. :func:`~astrolabe.pp.flag_controls` — maps ``var['is_control_probe']``
       (``Blank-*`` probes written by :func:`~astrolabe.io.read_merscope`) to
       ``obs['pct_counts_control']``.
    2–5. All shared steps from :func:`filter_cells` (``calculate_qc_metrics``,
       ``filter_cells``, ``filter_genes``, spatial isolation).
    6. Optionally removes blank probe features from ``X`` when *remove_controls*
       is ``True``.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`~astrolabe.io.read_merscope`.
    merscope_key
        Key for the MERSCOPE modality inside *mdata*.
    remove_controls
        Remove ``Blank-*`` probe features after computing QC metrics.
    max_pct_control
        Maximum percentage of counts from blank probes per cell.
    min_counts
        Minimum total transcript counts per cell.
    min_genes
        Minimum number of genes detected per cell.
    min_cell_area
        Minimum cell area in µm².
    max_cell_area
        Maximum cell area in µm².
    isolation_k
        Number of nearest spatial neighbors for the isolation filter.
    max_neighbor_dist_um
        Maximum distance to the *k*-th nearest spatial neighbor in µm.
    min_neighbor_dist_um
        Remove cells whose nearest neighbor exceeds this distance (µm).
        Threshold applies to the 1st neighbor; cf. *max_neighbor_dist_um*
        which uses the k-th neighbor.
    spatial_key
        ``obsm`` key for cell centroids.
    min_cells
        Minimum number of cells a gene must be detected in.
    inplace
        Modify *mdata* in place and return ``None`` (default).

    Returns
    -------
    None or MuData

    Examples
    --------
    >>> ab.pipeline.filter_merscope(mdata)
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.cluster(mdata, neighbors_key='spatial_neighbors')
    """
    import astrolabe.pp as pp

    if not isinstance(mdata, md.MuData):
        raise TypeError(f"filter_merscope expects a MuData; got {type(mdata).__name__}")
    if merscope_key not in mdata.mod:
        raise KeyError(f"Modality '{merscope_key}' not found in mdata")

    if not inplace:
        mdata = mdata.copy()

    adata = mdata[merscope_key]
    log.info("filter_merscope: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # MERSCOPE-specific: flag Blank-* probes as controls
    if "is_control_probe" in adata.var.columns:
        log.info("  flag_controls (var_col='is_control_probe')")
        pp.flag_controls(mdata, modality=merscope_key, var_col="is_control_probe")
    else:
        log.warning(
            "var['is_control_probe'] not found; skipping control-percentage filter. "
            "Load data via ab.io.read_merscope() to get blank probe annotations."
        )
        max_pct_control = None

    # Shared steps
    filter_cells(
        mdata,
        modality=merscope_key,
        min_counts=min_counts,
        min_genes=min_genes,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        max_pct_control=max_pct_control,
        isolation_k=isolation_k,
        max_neighbor_dist_um=max_neighbor_dist_um,
        min_neighbor_dist_um=min_neighbor_dist_um,
        spatial_key=spatial_key,
        min_cells=min_cells,
        inplace=True,
    )

    # Remove blank probe features from X
    if remove_controls and "is_control_probe" in adata.var.columns:
        n_before = adata.n_vars
        pp.filter_genes(
            mdata, modality=merscope_key, exclude_controls=True, inplace=True
        )
        _log_step("remove_controls", n_before, adata.n_vars, axis="genes")

    log.info(
        "filter_merscope: done — %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )
    return None if inplace else mdata


def filter_cosmx(
    mdata: md.MuData,
    *,
    cosmx_key: str = "cosmx",
    # --- CosMx-specific ---
    remove_controls: bool = True,
    max_pct_control: float = 5.0,
    # --- Shared with filter_cells ---
    min_counts: int = 10,
    min_genes: int = 5,
    min_cell_area: float | None = None,
    max_cell_area: float | None = None,
    isolation_k: int = 5,
    max_neighbor_dist_um: float | None = 60.0,
    min_neighbor_dist_um: float | None = 50.0,
    spatial_key: str = "spatial",
    min_cells: int = 5,
    inplace: bool = True,
) -> md.MuData | None:
    """CosMx-specific QC metrics and filtering.

    Extends :func:`filter_cells` with a CosMx-specific first step:

    1. :func:`~astrolabe.pp.flag_controls` — maps ``var['is_control_probe']``
       (``NegPrb*``, ``SystemControl*``, ``FalseCode*`` probes written by
       :func:`~astrolabe.io.read_cosmx`) to ``obs['pct_counts_control']``.
    2–5. All shared steps from :func:`filter_cells` (``calculate_qc_metrics``,
       ``filter_cells``, ``filter_genes``, spatial isolation).
    6. Optionally removes control probe features from ``X`` when
       *remove_controls* is ``True``.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`~astrolabe.io.read_cosmx`.
    cosmx_key
        Key for the CosMx modality inside *mdata*.
    remove_controls
        Remove negative-control probe features after computing QC metrics.
    max_pct_control
        Maximum percentage of counts from control probes per cell.
    min_counts
        Minimum total transcript counts per cell.
    min_genes
        Minimum number of genes detected per cell.
    min_cell_area
        Minimum cell area in µm².
    max_cell_area
        Maximum cell area in µm².
    isolation_k
        Number of nearest spatial neighbors for the isolation filter.
    max_neighbor_dist_um
        Maximum distance to the *k*-th nearest spatial neighbor in µm.
    min_neighbor_dist_um
        Remove cells whose nearest neighbor exceeds this distance (µm).
        Threshold applies to the 1st neighbor; cf. *max_neighbor_dist_um*
        which uses the k-th neighbor.
    spatial_key
        ``obsm`` key for cell centroids.
    min_cells
        Minimum number of cells a gene must be detected in.
    inplace
        Modify *mdata* in place and return ``None`` (default).

    Returns
    -------
    None or MuData

    Examples
    --------
    >>> ab.pipeline.filter_cosmx(mdata)
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.cluster(mdata, neighbors_key='spatial_neighbors')
    """
    import astrolabe.pp as pp

    if not isinstance(mdata, md.MuData):
        raise TypeError(f"filter_cosmx expects a MuData; got {type(mdata).__name__}")
    if cosmx_key not in mdata.mod:
        raise KeyError(f"Modality '{cosmx_key}' not found in mdata")

    if not inplace:
        mdata = mdata.copy()

    adata = mdata[cosmx_key]
    log.info("filter_cosmx: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # CosMx-specific: flag NegPrb/SystemControl/FalseCode probes as controls
    if "is_control_probe" in adata.var.columns:
        log.info("  flag_controls (var_col='is_control_probe')")
        pp.flag_controls(mdata, modality=cosmx_key, var_col="is_control_probe")
    else:
        log.warning(
            "var['is_control_probe'] not found; skipping control-percentage filter. "
            "Load data via ab.io.read_cosmx() to get negative-control probe annotations."
        )
        max_pct_control = None

    # Shared steps
    filter_cells(
        mdata,
        modality=cosmx_key,
        min_counts=min_counts,
        min_genes=min_genes,
        min_cell_area=min_cell_area,
        max_cell_area=max_cell_area,
        max_pct_control=max_pct_control,
        isolation_k=isolation_k,
        max_neighbor_dist_um=max_neighbor_dist_um,
        min_neighbor_dist_um=min_neighbor_dist_um,
        spatial_key=spatial_key,
        min_cells=min_cells,
        inplace=True,
    )

    # Remove control probe features from X
    if remove_controls and "is_control_probe" in adata.var.columns:
        n_before = adata.n_vars
        pp.filter_genes(mdata, modality=cosmx_key, exclude_controls=True, inplace=True)
        _log_step("remove_controls", n_before, adata.n_vars, axis="genes")

    log.info(
        "filter_cosmx: done — %d cells × %d genes",
        adata.n_obs,
        adata.n_vars,
    )
    return None if inplace else mdata
