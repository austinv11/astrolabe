"""Transcript-to-cell assignment via point-in-polygon spatial indexing."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

__all__ = ["assign_transcripts"]

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