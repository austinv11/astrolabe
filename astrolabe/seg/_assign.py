"""Transcript-to-cell assignment via point-in-polygon spatial indexing."""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._constants import UNS_XENIUM_SEGMENTATION_KEY
from astrolabe._utils import _resolve_transcripts, with_adata

__all__ = ["assign_transcripts", "create_compartment_layers"]

log = logging.getLogger(__name__)

_CHUNK = 500_000  # transcripts per processing batch


def _normalize_feature_names(feat_arr: np.ndarray) -> np.ndarray:
    """Decode bytes feature names to str.

    Transcript tables read through some parquet/HDF5 paths carry feature
    names as ``bytes``; left as-is they never match str ``var_names`` and
    every transcript is silently skipped.
    """
    if feat_arr.dtype.kind == "S" or (
        feat_arr.dtype == object and len(feat_arr) > 0 and isinstance(feat_arr[0], bytes)
    ):
        return np.array(
            [f.decode() if isinstance(f, bytes) else str(f) for f in feat_arr],
            dtype=object,
        )
    return feat_arr


def _dedup_point_matches(
    pt_idx_raw: np.ndarray, poly_idx_raw: np.ndarray, poly_areas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve points matching >1 overlapping polygon to exactly one match.

    Ties are broken to the **smallest-area** polygon rather than STRtree
    insertion order — deterministic under any polygon ordering, and the
    smallest containing polygon is the most specific claim on the point
    (an enclosing over-segmented giant never steals a transcript from the
    small cell actually drawn around it).
    """
    order = np.lexsort((poly_areas[poly_idx_raw], pt_idx_raw))
    pt_sorted = pt_idx_raw[order]
    poly_sorted = poly_idx_raw[order]
    _, first_idx = np.unique(pt_sorted, return_index=True)
    return pt_sorted[first_idx], poly_sorted[first_idx]


def assign_transcripts(
    adata: ad.AnnData,
    transcripts: "object" = None,  # pyarrow.Table | pd.DataFrame | None
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
        and *feature_col*.  ``None`` (default) uses ``adata.uns['transcripts']``
        — present whenever *adata* descends from a ``read_xenium`` load (e.g. a
        :func:`~astrolabe.seg.voronoi_resegment` result), and already culled
        to the region of interest if the object was cropped with
        :func:`~astrolabe.pp.crop`.
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
    >>> # a voronoi_resegment result inherits uns['transcripts'] -- no table needed:
    >>> ab.seg.assign_transcripts(voronoi_cells, boundaries_key='voronoi_boundaries')
    >>> # a fresh segmentation AnnData has no uns['transcripts'] -- pass one:
    >>> new_cells = ab.seg.segment_cellpose(mdata, channel="DAPI")
    >>> ab.seg.assign_transcripts(new_cells, mdata['xenium'].uns['transcripts'])
    >>> # also capture transcripts up to 5 µm outside the nearest cell boundary
    >>> ab.seg.assign_transcripts(new_cells, transcripts, max_assign_distance_um=5.0)
    """
    from shapely import points as _shapely_points
    from shapely.strtree import STRtree as _STRtree
    from shapely.geometry import Polygon as _Polygon
    import scipy.sparse as sp

    transcripts = _resolve_transcripts(adata, transcripts, "assign_transcripts")
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
    feat_arr = _normalize_feature_names(df[feature_col].to_numpy())
    n_transcripts = len(x_arr)

    log.info("assign_transcripts: %d transcripts, %d cells", n_transcripts, adata.n_obs)

    # Build shapely polygons from obsm boundaries
    cell_polys = []
    for cell_coords in adata.obsm[boundaries_key]:
        coords = np.asarray(cell_coords)
        cell_polys.append(_Polygon(coords) if len(coords) >= 3 else _Polygon())

    tree = _STRtree(cell_polys)
    poly_areas = np.array([p.area for p in cell_polys], dtype=np.float64)

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
        # predicate="covered_by" (not "within"): "within" is
        # boundary-exclusive, so a transcript landing exactly on a polygon
        # edge/vertex matched no cell and silently fell through.
        result = tree.query(pts, predicate="covered_by")
        if result.shape[1] == 0:
            continue

        # A transcript within >1 overlapping cell polygon otherwise appears
        # once per overlapping cell here, double-counting it into every
        # overlapping cell's totals instead of exactly one. Deduplicate to
        # one match per point (smallest-area polygon on ties) -- see the
        # identical fix/comment in create_compartment_layers below.
        pt_idx_raw, poly_idx_raw = result[0], result[1]
        pt_idx_chunk, poly_idx = _dedup_point_matches(pt_idx_raw, poly_idx_raw, poly_areas)
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

    # Build the result through the public, fully-validated AnnData
    # constructor (never by assigning private attributes like `_var`, which
    # bypasses every dimension check). layers/varm are deliberately not
    # carried over: n_vars changes here, so any existing per-var structure
    # would be invalid.
    new = ad.AnnData(
        X=X,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene_ids")),
        uns=dict(adata.uns),
        obsm=dict(adata.obsm),
        obsp=dict(adata.obsp),
    )
    if inplace:
        # In-place var-dimension changes have no public AnnData API;
        # _init_as_actual is the exact code path AnnData.__init__ delegates
        # to, so the caller's object is rebuilt through the same validated
        # construction as `new` above.
        adata._init_as_actual(
            X=new.X, obs=new.obs, var=new.var, uns=new.uns,
            obsm=dict(new.obsm), obsp=dict(new.obsp),
        )
        return None
    return new


@with_adata
def create_compartment_layers(
    adata: ad.AnnData,
    transcripts: "object" = None,  # pyarrow.Table | pd.DataFrame | None
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

    * **nucleus** — transcripts inside *their assigned cell's own* nucleus
      boundary. A transcript inside a *neighboring* cell's nucleus but
      assigned to this cell counts as this cell's cytoplasm, never as
      nuclear signal. Uses STRtree containment against
      ``obsm[nucleus_boundaries_key]`` (row-aligned with the cell
      boundaries); the Xenium ``overlaps_nucleus`` column is used as a fast
      path **only** when ``uns['xenium_native_segmentation']`` is ``True``
      (set by :func:`~astrolabe.io.read_xenium`), i.e. when the boundaries
      are still the unmodified Xenium originals that column was computed
      against — for any re-segmentation the column describes the *wrong*
      nuclei and the STRtree path is used instead (logged).
    * **cytoplasm** — transcripts inside cell boundaries but outside their
      own cell's nucleus.
    * **ambient** — transcripts outside all cells, assigned to the nearest cell
      centroid (optionally capped by *max_ambient_distance_um*).

    Transcripts whose feature name is not in ``adata.var_names`` are skipped
    (with a summary count in the log); if *every* transcript is skipped this
    way, a :class:`ValueError` is raised instead of silently writing three
    all-zero layers. If no nucleus information is available, only the
    cytoplasm and ambient layers are written and a warning is emitted.

    .. note::
        The three layers partition the *transcript table passed in*, not
        ``adata.X``: ``nucleus + cytoplasm + ambient`` sums transcripts of
        this table (minus skipped features and out-of-range ambient), which
        only equals ``X`` when ``X`` was built from the same table with the
        same assignment rule. Xenium's shipped ``X`` uses its own on-device
        assignment, so small discrepancies against these layers are
        expected and are not an error.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with cell boundary
        polygons in ``obsm[cell_boundaries_key]``.
    transcripts
        Transcript table (:class:`pyarrow.Table` or :class:`pandas.DataFrame`).
        Must contain *x_col*, *y_col*, and *feature_col* columns.  ``None``
        (default) uses the modality's own ``uns['transcripts']`` (loaded by
        :func:`~astrolabe.io.read_xenium`, kept row-synced by filtering steps,
        and culled to the region of interest by :func:`~astrolabe.pp.crop`).
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
        STRtree lookup even when the column exists.  Even when present, the
        column is only trusted when ``uns['xenium_native_segmentation']`` is
        ``True`` — see the nucleus layer description above.
    spatial_key
        ``obsm`` key for cell centroids (µm); used for ambient KDTree lookup.
    max_ambient_distance_um
        Maximum distance for ambient assignment.  ``None`` assigns every
        unassigned transcript to its nearest centroid regardless of distance.
    layer_nucleus, layer_cytoplasm, layer_ambient
        Names of the output ``adata.layers`` keys.

    Examples
    --------
    >>> ab.seg.create_compartment_layers(mdata)   # uses uns['transcripts']
    >>> mdata['xenium'].layers['nucleus'].shape   # (n_cells, n_genes)
    """
    from shapely import points as _shapely_points
    from shapely.strtree import STRtree as _STRtree
    from shapely.geometry import Polygon as _Polygon
    from scipy.spatial import cKDTree
    import scipy.sparse as sp

    transcripts = _resolve_transcripts(adata, transcripts, "create_compartment_layers")
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
    feat_arr = _normalize_feature_names(df[feature_col].to_numpy())
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
    cell_areas = np.array([p.area for p in cell_polys], dtype=np.float64)

    # Determine nuclear classification strategy. The Xenium-native
    # overlaps_nucleus fast path is only valid for the unmodified Xenium
    # segmentation the column was computed against: applied to any other
    # segmentation it books transcripts overlapping a *neighboring* cell's
    # (original Xenium) nucleus as nuclear signal of whatever foreign cell
    # they were assigned to.
    nuc_strategy: str
    nuc_flag: np.ndarray | None = None
    nuc_tree: "_STRtree | None" = None
    has_flag_col = overlaps_nucleus_col is not None and overlaps_nucleus_col in df.columns
    native_seg = bool(adata.uns.get(UNS_XENIUM_SEGMENTATION_KEY, False))

    if has_flag_col and native_seg:
        nuc_strategy = "fast"
        nuc_flag = df[overlaps_nucleus_col].to_numpy(dtype=bool)
        log.info(
            "create_compartment_layers: using column '%s' for nuclear classification "
            "(uns['%s'] confirms unmodified Xenium segmentation)",
            overlaps_nucleus_col, UNS_XENIUM_SEGMENTATION_KEY,
        )
    elif nucleus_boundaries_key in adata.obsm:
        nuc_strategy = "strtree"
        nuc_polys = []
        for nuc_coords in adata.obsm[nucleus_boundaries_key]:
            coords = np.asarray(nuc_coords)
            nuc_polys.append(_Polygon(coords) if len(coords) >= 3 else _Polygon())
        nuc_tree = _STRtree(nuc_polys)
        if has_flag_col:
            log.info(
                "create_compartment_layers: column '%s' is present but "
                "uns['%s'] is not set — the segmentation is not (or not "
                "provably) the unmodified Xenium original the column refers "
                "to; using nucleus-boundary STRtree classification instead",
                overlaps_nucleus_col, UNS_XENIUM_SEGMENTATION_KEY,
            )
        else:
            log.info("create_compartment_layers: using nucleus STRtree for classification")
    else:
        nuc_strategy = "skip"
        if has_flag_col:
            log.warning(
                "create_compartment_layers: column '%s' is present but cannot "
                "be trusted (uns['%s'] not set) and obsm['%s'] is absent — "
                "writing only '%s' and '%s' layers (skipping '%s')",
                overlaps_nucleus_col, UNS_XENIUM_SEGMENTATION_KEY,
                nucleus_boundaries_key, layer_cytoplasm, layer_ambient, layer_nucleus,
            )
        else:
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

        # predicate="covered_by" (not "within"): "within" is
        # boundary-exclusive, so a transcript landing exactly on a polygon
        # edge/vertex matched no cell and silently fell through to ambient.
        cell_result = cell_tree.query(pts, predicate="covered_by")
        if cell_result.shape[1] == 0:
            continue

        # A transcript that lies within >1 overlapping cell polygon
        # otherwise appears once per overlapping cell in cell_result (one
        # row per (point, polygon) match) -- summed into *every* overlapping
        # cell's totals downstream instead of exactly one, double-counting
        # it and breaking the documented nucleus/cytoplasm/ambient partition
        # contract (measured excess on real data: 592 of 10.55M
        # transcripts). Deduplicate to one match per transcript point
        # (smallest-area cell on ties -- deterministic under any polygon
        # ordering, unlike STRtree insertion order).
        pt_local_raw, cell_row_raw = cell_result[0], cell_result[1]
        pt_local, cell_row = _dedup_point_matches(pt_local_raw, cell_row_raw, cell_areas)
        global_pt = chunk_start + pt_local
        assigned[global_pt] = True

        gene_cols_raw = np.array(
            [var_idx.get(feat_arr[i], -1) for i in global_pt], dtype=np.int32
        )

        # Nuclear determination
        if nuc_strategy == "fast":
            is_nuc = nuc_flag[global_pt]  # type: ignore[index]
        elif nuc_strategy == "strtree":
            # A transcript is nuclear only when inside its *own* assigned
            # cell's nucleus (nucleus polygons are row-aligned with cell
            # polygons, so "own" = matching row index). Testing "inside ANY
            # nucleus" (the previous behavior) booked a transcript sitting
            # in a *neighboring* cell's nucleus -- but assigned to this
            # cell -- as this cell's nuclear signal.
            nuc_pts = _shapely_points(x_arr[global_pt], y_arr[global_pt])
            nuc_res = nuc_tree.query(nuc_pts, predicate="covered_by")  # type: ignore[union-attr]
            is_nuc = np.zeros(len(global_pt), dtype=bool)
            if nuc_res.shape[1] > 0:
                own_nucleus = cell_row[nuc_res[0]] == nuc_res[1]
                is_nuc[nuc_res[0][own_nucleus]] = True
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
        n_skipped += int((~known_amb).sum())
        amb_rows_arr = nearest_cell[known_amb].astype(np.int32)
        amb_cols_arr = gene_cols_amb[known_amb]

    if n_transcripts > 0 and n_skipped >= n_transcripts:
        raise ValueError(
            "create_compartment_layers: every transcript's feature name failed "
            "to match adata.var_names — the layers would be all zeros. This "
            "usually means the transcript table and the AnnData use different "
            "feature naming (check the feature_col values against var_names)."
        )

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