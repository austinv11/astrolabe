"""Transcript-to-cell assignment via point-in-polygon spatial indexing."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._utils import with_adata

__all__ = ["assign_transcripts", "create_compartment_layers"]

log = logging.getLogger(__name__)

_CHUNK = 500_000  # transcripts per processing batch


def assign_transcripts(
    adata: ad.AnnData,
    transcripts: "object",  # pyarrow.Table | pd.DataFrame
    *,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    boundaries_key: str = "cell_boundaries",
    spatial_key: str = "spatial",
    max_assign_distance_um: float | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Assign transcripts to cells by point-in-polygon and build a count matrix.

    Takes an :class:`~anndata.AnnData` produced by
    :func:`~astrolabe.seg.segment_cellpose` or
    :func:`~astrolabe.seg.segment_stardist` (which has cell boundary polygons
    in µm but no expression data) and a transcript table in the same µm
    coordinate frame, and populates ``adata.X`` with per-cell per-gene counts.

    Uses a Shapely :class:`~shapely.STRtree` for polygon containment lookups.
    Transcripts that fall outside all cell boundaries can optionally be
    assigned to the nearest cell centroid within *max_assign_distance_um* µm.

    Parameters
    ----------
    adata
        AnnData with ``obsm[boundaries_key]`` containing cell boundary
        polygons as an awkward array of ``(n_verts, 2)`` µm coordinates
        (x, y).  Typically the output of a segmentation function.
    transcripts
        Transcript table as a :class:`pyarrow.Table` (e.g. from
        :func:`~astrolabe.io.read_xenium_transcripts`) or
        :class:`pandas.DataFrame`.  Must contain columns *x_col*, *y_col*,
        and *feature_col*.
    x_col, y_col
        Column names for transcript x/y coordinates in µm.
    feature_col
        Column name for the feature (gene) name.
    boundaries_key
        Key in ``obsm`` containing cell boundary polygons.
    spatial_key
        Key in ``obsm`` for cell centroids (µm).  Used only when
        *max_assign_distance_um* is set.
    max_assign_distance_um
        If given, transcripts that fall outside every cell boundary are
        assigned to the nearest cell centroid that is within this distance
        (in µm).  ``None`` (default) leaves unassigned transcripts
        unassigned.
    inplace
        If ``True`` (default), modify *adata* in place and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    None or AnnData

    Examples
    --------
    >>> new_cells = ab.seg.segment_cellpose(mdata, channel="DAPI", model_type="nuclei")
    >>> transcripts = ab.io.read_xenium_transcripts("Xenium_run/")
    >>> # strict containment only
    >>> ab.seg.assign_transcripts(new_cells, transcripts)
    >>> # also capture transcripts up to 5 µm outside the nearest cell boundary
    >>> ab.seg.assign_transcripts(new_cells, transcripts, max_assign_distance_um=5.0)
    """
    from shapely import points as _shapely_points
    from shapely.strtree import STRtree as _STRtree
    from shapely.geometry import Polygon as _Polygon
    import scipy.sparse as sp

    try:
        import pyarrow as _pa
    except ImportError:
        _pa = None  # type: ignore[assignment]
    df: pd.DataFrame = (
        transcripts.to_pandas()  # type: ignore[union-attr]
        if (_pa is not None and isinstance(transcripts, _pa.Table))
        else transcripts  # type: ignore[assignment]
    )

    if boundaries_key not in adata.obsm:
        raise KeyError(
            f"obsm['{boundaries_key}'] not found. "
            "Run a segmentation function first to populate cell boundaries."
        )
    if max_assign_distance_um is not None and spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found — needed for proximity assignment. "
            "Ensure the AnnData has centroid coordinates in obsm['spatial']."
        )

    for col in (x_col, y_col, feature_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in transcript table.")

    x_arr = df[x_col].to_numpy(dtype=np.float64)
    y_arr = df[y_col].to_numpy(dtype=np.float64)
    feat_arr = df[feature_col].to_numpy()
    n_transcripts = len(x_arr)

    log.info("assign_transcripts: %d transcripts, %d cells", n_transcripts, adata.n_obs)

    # Build shapely polygons from obsm boundaries
    cell_polys = []
    for cell_coords in adata.obsm[boundaries_key]:
        coords = np.asarray(cell_coords)
        cell_polys.append(_Polygon(coords) if len(coords) >= 3 else _Polygon())

    tree = _STRtree(cell_polys)

    # Unique genes
    gene_names = sorted(set(feat_arr.tolist()))
    gene_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)
    n_cells = adata.n_obs

    row_list: list[np.ndarray] = []
    col_list: list[np.ndarray] = []

    # Track which transcripts were assigned (needed for proximity fallback)
    assigned = np.zeros(n_transcripts, dtype=bool)

    for start in range(0, n_transcripts, _CHUNK):
        end = min(start + _CHUNK, n_transcripts)
        pts = _shapely_points(x_arr[start:end], y_arr[start:end])

        # result shape (2, N_hits):
        #   result[0] = input point indices (relative to pts)
        #   result[1] = tree polygon indices
        result = tree.query(pts, predicate="within")
        if result.shape[1] == 0:
            continue

        pt_idx_chunk, poly_idx = result[0], result[1]
        global_pt_idx = start + pt_idx_chunk
        assigned[global_pt_idx] = True

        gene_indices = np.array(
            [gene_idx[feat_arr[i]] for i in global_pt_idx], dtype=np.int32
        )
        row_list.append(poly_idx.astype(np.int32))
        col_list.append(gene_indices)

        if (start // _CHUNK) % 10 == 0:
            log.info(
                "assign_transcripts: processed %d / %d transcripts",
                end, n_transcripts,
            )

    # ── Proximity assignment for unassigned transcripts ───────────────────────
    if max_assign_distance_um is not None:
        from scipy.spatial import cKDTree

        unassigned_idx = np.where(~assigned)[0]
        log.info(
            "assign_transcripts: proximity pass — %d unassigned transcripts "
            "(radius=%.1f µm)",
            len(unassigned_idx), max_assign_distance_um,
        )

        if len(unassigned_idx) > 0:
            centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
            kdtree = cKDTree(centroids)

            unassigned_xy = np.column_stack(
                [x_arr[unassigned_idx], y_arr[unassigned_idx]]
            )
            dist, nearest_cell = kdtree.query(unassigned_xy, k=1, workers=-1)

            in_radius = dist <= max_assign_distance_um
            close_idx = unassigned_idx[in_radius]
            close_cells = nearest_cell[in_radius]

            if len(close_idx) > 0:
                gene_indices = np.array(
                    [gene_idx[feat_arr[i]] for i in close_idx], dtype=np.int32
                )
                row_list.append(close_cells.astype(np.int32))
                col_list.append(gene_indices)
                log.info(
                    "assign_transcripts: proximity-assigned %d transcripts",
                    len(close_idx),
                )

    # ── Build sparse matrix ───────────────────────────────────────────────────
    if row_list:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        data = np.ones(len(rows), dtype=np.float32)
        X = sp.coo_matrix((data, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    else:
        X = sp.csr_matrix((n_cells, n_genes), dtype=np.float32)

    n_assigned = int(X.sum())
    pct = n_assigned / n_transcripts * 100 if n_transcripts else 0.0
    log.info(
        "assign_transcripts: assigned %d / %d transcripts (%.1f%%)",
        n_assigned, n_transcripts, pct,
    )

    if not inplace:
        adata = adata.copy()

    # AnnData validates shape on X assignment; set _var first so n_vars matches.
    adata._var = pd.DataFrame(index=pd.Index(gene_names, name="gene_ids"))
    adata.X = X

    return None if inplace else adata


@with_adata
def create_compartment_layers(
    adata: ad.AnnData,
    transcripts: "object",  # pyarrow.Table | pd.DataFrame
    *,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    cell_boundaries_key: str = "cell_boundaries",
    nucleus_boundaries_key: str = "nucleus_boundaries",
    overlaps_nucleus_col: str | None = "overlaps_nucleus",
    spatial_key: str = "spatial",
    max_ambient_distance_um: float | None = None,
    layer_nucleus: str = "nucleus",
    layer_cytoplasm: str = "cytoplasm",
    layer_ambient: str = "ambient",
) -> None:
    """Partition transcripts into nucleus, cytoplasm, and ambient AnnData layers.

    Creates three sparse count layers aligned with ``adata.X`` (same shape):

    * **nucleus** — transcripts confirmed inside nuclear boundaries (uses the
      Xenium ``overlaps_nucleus`` column directly when present; falls back to
      STRtree containment against ``obsm[nucleus_boundaries_key]``).
    * **cytoplasm** — transcripts inside cell boundaries but outside the nucleus.
    * **ambient** — transcripts outside all cells, assigned to the nearest cell
      centroid (optionally capped by *max_ambient_distance_um*).

    Transcripts whose feature name is not in ``adata.var_names`` are silently
    skipped.  If no nucleus information is available, only the cytoplasm and
    ambient layers are written and a warning is emitted.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with cell boundary
        polygons in ``obsm[cell_boundaries_key]``.
    transcripts
        Transcript table (:class:`pyarrow.Table` or :class:`pandas.DataFrame`).
        Must contain *x_col*, *y_col*, and *feature_col* columns.
    modality
        For :class:`~mudata.MuData`: modality key.
    x_col, y_col
        Column names for transcript coordinates (µm).
    feature_col
        Column name for feature / gene name.
    cell_boundaries_key
        ``obsm`` key containing cell boundary polygons.
    nucleus_boundaries_key
        ``obsm`` key containing nucleus boundary polygons (used when
        *overlaps_nucleus_col* is absent).
    overlaps_nucleus_col
        Transcript table column name with a boolean flag indicating nuclear
        overlap (Xenium-native ``overlaps_nucleus``).  Set to ``None`` to force
        STRtree lookup even when the column exists.
    spatial_key
        ``obsm`` key for cell centroids (µm); used for ambient KDTree lookup.
    max_ambient_distance_um
        Maximum distance for ambient assignment.  ``None`` assigns every
        unassigned transcript to its nearest centroid regardless of distance.
    layer_nucleus, layer_cytoplasm, layer_ambient
        Names of the output ``adata.layers`` keys.

    Examples
    --------
    >>> transcripts = ab.io.read_xenium_transcripts(mdata)
    >>> ab.seg.create_compartment_layers(mdata, transcripts)
    >>> mdata['xenium'].layers['nucleus'].shape   # (n_cells, n_genes)
    """
    from shapely import points as _shapely_points
    from shapely.strtree import STRtree as _STRtree
    from shapely.geometry import Polygon as _Polygon
    from scipy.spatial import cKDTree
    import scipy.sparse as sp

    try:
        import pyarrow as _pa
    except ImportError:
        _pa = None  # type: ignore[assignment]
    df: pd.DataFrame = (
        transcripts.to_pandas()  # type: ignore[union-attr]
        if (_pa is not None and isinstance(transcripts, _pa.Table))
        else transcripts  # type: ignore[assignment]
    )

    if cell_boundaries_key not in adata.obsm:
        raise KeyError(f"obsm['{cell_boundaries_key}'] not found.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found.")
    for col in (x_col, y_col, feature_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in transcript table.")

    x_arr = df[x_col].to_numpy(dtype=np.float64)
    y_arr = df[y_col].to_numpy(dtype=np.float64)
    feat_arr = df[feature_col].to_numpy()
    n_transcripts = len(x_arr)

    # Map gene names → var column indices; unknowns stay at -1 (skipped)
    var_idx: dict[str, int] = {g: i for i, g in enumerate(adata.var_names)}
    n_obs, n_vars = adata.n_obs, adata.n_vars

    log.info(
        "create_compartment_layers: %d transcripts, %d cells, %d genes",
        n_transcripts, n_obs, n_vars,
    )

    # Build cell STRtree
    cell_polys = []
    for cell_coords in adata.obsm[cell_boundaries_key]:
        coords = np.asarray(cell_coords)
        cell_polys.append(_Polygon(coords) if len(coords) >= 3 else _Polygon())
    cell_tree = _STRtree(cell_polys)

    # Determine nuclear classification strategy
    nuc_strategy: str
    nuc_flag: np.ndarray | None = None
    nuc_tree: "_STRtree | None" = None

    if overlaps_nucleus_col is not None and overlaps_nucleus_col in df.columns:
        nuc_strategy = "fast"
        nuc_flag = df[overlaps_nucleus_col].to_numpy(dtype=bool)
        log.info(
            "create_compartment_layers: using column '%s' for nuclear classification",
            overlaps_nucleus_col,
        )
    elif nucleus_boundaries_key in adata.obsm:
        nuc_strategy = "strtree"
        nuc_polys = []
        for nuc_coords in adata.obsm[nucleus_boundaries_key]:
            coords = np.asarray(nuc_coords)
            nuc_polys.append(_Polygon(coords) if len(coords) >= 3 else _Polygon())
        nuc_tree = _STRtree(nuc_polys)
        log.info("create_compartment_layers: using nucleus STRtree for classification")
    else:
        nuc_strategy = "skip"
        log.warning(
            "create_compartment_layers: no nucleus information found — "
            "writing only '%s' and '%s' layers (skipping '%s')",
            layer_cytoplasm, layer_ambient, layer_nucleus,
        )

    assigned = np.zeros(n_transcripts, dtype=bool)
    nuc_rows_list: list[np.ndarray] = []
    nuc_cols_list: list[np.ndarray] = []
    cyt_rows_list: list[np.ndarray] = []
    cyt_cols_list: list[np.ndarray] = []
    n_skipped = 0

    # Chunked cell-containment + nucleus classification
    for chunk_start in range(0, n_transcripts, _CHUNK):
        chunk_end = min(chunk_start + _CHUNK, n_transcripts)
        pts = _shapely_points(x_arr[chunk_start:chunk_end], y_arr[chunk_start:chunk_end])

        cell_result = cell_tree.query(pts, predicate="within")
        if cell_result.shape[1] == 0:
            continue

        pt_local, cell_row = cell_result[0], cell_result[1]
        global_pt = chunk_start + pt_local
        assigned[global_pt] = True

        gene_cols_raw = np.array(
            [var_idx.get(feat_arr[i], -1) for i in global_pt], dtype=np.int32
        )

        # Nuclear determination
        if nuc_strategy == "fast":
            is_nuc = nuc_flag[global_pt]  # type: ignore[index]
        elif nuc_strategy == "strtree":
            nuc_pts = _shapely_points(x_arr[global_pt], y_arr[global_pt])
            nuc_res = nuc_tree.query(nuc_pts, predicate="within")  # type: ignore[union-attr]
            in_nuc_set = set(nuc_res[0].tolist()) if nuc_res.shape[1] > 0 else set()
            is_nuc = np.fromiter(
                (i in in_nuc_set for i in range(len(global_pt))),
                dtype=bool,
                count=len(global_pt),
            )
        else:
            is_nuc = np.zeros(len(global_pt), dtype=bool)

        known = gene_cols_raw >= 0
        n_skipped += int((~known).sum())

        nuc_rows_list.append(cell_row[known & is_nuc])
        nuc_cols_list.append(gene_cols_raw[known & is_nuc])
        cyt_rows_list.append(cell_row[known & ~is_nuc])
        cyt_cols_list.append(gene_cols_raw[known & ~is_nuc])

        if (chunk_start // _CHUNK) % 10 == 0:
            log.info(
                "create_compartment_layers: processed %d / %d transcripts",
                chunk_end, n_transcripts,
            )

    # Ambient: nearest-centroid assignment for unassigned transcripts
    unassigned_idx = np.where(~assigned)[0]
    log.info(
        "create_compartment_layers: %d unassigned → ambient", len(unassigned_idx)
    )

    amb_rows_arr = np.empty(0, dtype=np.int32)
    amb_cols_arr = np.empty(0, dtype=np.int32)
    if len(unassigned_idx) > 0:
        centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        kdtree = cKDTree(centroids)
        unassigned_xy = np.column_stack(
            [x_arr[unassigned_idx], y_arr[unassigned_idx]]
        )
        dist, nearest_cell = kdtree.query(unassigned_xy, k=1, workers=-1)

        if max_ambient_distance_um is not None:
            in_range = dist <= max_ambient_distance_um
            unassigned_idx = unassigned_idx[in_range]
            nearest_cell = nearest_cell[in_range]

        gene_cols_amb = np.array(
            [var_idx.get(feat_arr[i], -1) for i in unassigned_idx], dtype=np.int32
        )
        known_amb = gene_cols_amb >= 0
        amb_rows_arr = nearest_cell[known_amb].astype(np.int32)
        amb_cols_arr = gene_cols_amb[known_amb]

    # Build sparse CSR layers
    def _build_csr(rows_list: list[np.ndarray], cols_list: list[np.ndarray]):
        non_empty = [r for r in rows_list if len(r) > 0]
        if not non_empty:
            return sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
        r = np.concatenate(rows_list).astype(np.int32)
        c = np.concatenate(cols_list).astype(np.int32)
        d = np.ones(len(r), dtype=np.float32)
        return sp.coo_matrix((d, (r, c)), shape=(n_obs, n_vars)).tocsr()

    if nuc_strategy != "skip":
        adata.layers[layer_nucleus] = _build_csr(nuc_rows_list, nuc_cols_list)
        n_nuc = int(adata.layers[layer_nucleus].sum())
    else:
        n_nuc = 0

    adata.layers[layer_cytoplasm] = _build_csr(cyt_rows_list, cyt_cols_list)
    n_cyt = int(adata.layers[layer_cytoplasm].sum())

    adata.layers[layer_ambient] = _build_csr([amb_rows_arr], [amb_cols_arr])
    n_amb = int(adata.layers[layer_ambient].sum())

    log.info(
        "create_compartment_layers: nucleus=%d, cytoplasm=%d, ambient=%d, skipped=%d",
        n_nuc, n_cyt, n_amb, n_skipped,
    )