"""
Neighborhood graph construction: spatial (µm-space) and transcriptomic
(expression-space).

``spatial_neighbors`` builds a KNN or radius-based graph on
``obsm['spatial']`` (micron coordinates) and stores it as a sparse
adjacency matrix in ``obsp``, following the scanpy/squidpy convention.
Unlike ``sc.pp.neighbors``, this does not write ``uns['neighbors']`` --
astrolabe's own graph-consuming functions (``tl.cluster``, ``tl.umap``,
etc.) default their ``neighbors_key`` to ``'spatial_neighbors'``
explicitly instead, so a transcriptomic graph written by
``sc.pp.neighbors`` / ``tl.transcriptional_neighbors`` is never silently
clobbered. Pass ``neighbors_key='neighbors'`` to those functions (or use
raw ``sc.tl.leiden``/``sc.tl.umap`` with an explicit ``neighbors_key``) to
use the spatial graph directly.

``transcriptional_neighbors`` is a thin ``@with_adata`` wrapper around
``sc.pp.neighbors`` -- the expression-space (PCA) counterpart -- so MuData
can be passed directly instead of unwrapping the modality by hand.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from astrolabe._utils import with_adata

__all__ = ["spatial_neighbors", "transcriptional_neighbors", "blend_neighbors"]

log = logging.getLogger(__name__)


@with_adata
def spatial_neighbors(
    adata: ad.AnnData,
    *,
    n_neighbors: int | None = None,
    radius: float | None = None,
    coord_type: Literal["generic", "grid", "delaunay"] = "generic",
    delaunay_max_dist: float | None = None,
    spatial_key: str = "spatial",
    key_added: str = "spatial_neighbors",
    set_diag: bool = False,
) -> None:
    """Build a spatial neighborhood graph from cell centroids.

    Results are stored in:

    - ``adata.obsp[key_added + '_connectivities']`` — binary sparse adjacency
    - ``adata.obsp[key_added + '_distances']`` — edge distances in µm, with
      the **same sparsity structure** as the connectivities (zero-distance
      edges between coincident centroids are kept as explicit stored zeros)
    - ``adata.uns[key_added]`` — graph construction parameters

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with
        ``obsm[spatial_key]`` containing ``(n_cells, 2)`` micron coordinates.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_neighbors
        Number of nearest neighbors per cell.  ``None`` (default) resolves
        to **6** for ``'generic'`` and ``'grid'`` KNN graphs — the spatial
        convention used by squidpy, deliberately smaller than scanpy's
        expression-space default of 15, which over-smooths spatial
        statistics.  Ignored for ``coord_type='delaunay'``.  When *radius*
        is also set, ``None`` means a pure radius graph, while an explicit
        integer truncates each cell's radius neighborhood to its
        *n_neighbors* nearest cells (top-k **and** within-radius
        intersection).
    radius
        Connect cells within this radius (µm) instead of a fixed-k graph.
        Ignored for ``coord_type='delaunay'``.  See *n_neighbors* for the
        interaction when both are given.
    coord_type
        Graph construction method:

        ``'generic'``
            Euclidean KNN (or radius) graph via ball-tree.  Good default
            for most single-cell resolution data.
        ``'delaunay'``
            Delaunay triangulation — connects cells that share a triangle
            edge.  Adapts to local cell density without a fixed k or
            radius; typically 5–7 neighbors per cell.  Long convex-hull
            edges can be pruned with *delaunay_max_dist*.
        ``'grid'``
            Chebyshev KNN for regular spot/bin arrays.  The default
            ``n_neighbors=6`` matches hexagonal Visium spot lattices
            (6 immediate neighbors); pass ``n_neighbors=8`` for square
            lattices such as Visium HD bins (8-connected).
    delaunay_max_dist
        Maximum edge length in µm for ``coord_type='delaunay'``.  Edges
        longer than this are removed, preventing spurious connections
        across empty tissue regions.  ``None`` keeps all Delaunay edges.
        A value of ~200 µm works well for Xenium data.  Ignored for other
        ``coord_type`` values.
    spatial_key
        Key in ``obsm`` for spatial coordinates.
    key_added
        Prefix for keys written to ``obsp`` and ``uns``.  Note squidpy
        stores its graph under fixed ``obsp`` keys
        (``'spatial_connectivities'``/``'spatial_distances'``); astrolabe
        namespaces both by *key_added* instead.
    set_diag
        Include self-loops: the diagonal is part of the stored sparsity
        structure of **both** the connectivity (value 1) and the distance
        matrix (explicit zero).

    Notes
    -----
    KNN/radius graphs use :func:`sklearn.neighbors.NearestNeighbors`
    (ball-tree), which scales to ~1 M cells.  Delaunay triangulation uses
    :func:`scipy.spatial.Delaunay` (2-D coordinates only).

    Directed KNN graphs are symmetrized by **union**: an edge is kept when
    either endpoint lists the other among its neighbors, so per-cell degree
    can exceed *n_neighbors*.  Stored distances are recomputed for the final
    edge set in the construction metric (Euclidean; Chebyshev for
    ``'grid'``).

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)                       # KNN k=6
    >>> ab.tl.spatial_neighbors(mdata, coord_type='delaunay',
    ...                         delaunay_max_dist=200)       # Delaunay
    """
    if coord_type not in ("generic", "grid", "delaunay"):
        raise ValueError(
            "coord_type must be one of 'generic', 'grid', or 'delaunay'; "
            f"got {coord_type!r}"
        )
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found.  "
            f"Ensure obsm['{spatial_key}'] has been populated with (n_cells, 2) coordinates."
        )

    coords = adata.obsm[spatial_key].astype(np.float64)  # (n_cells, 2)
    n_cells = coords.shape[0]
    metric = "chebyshev" if coord_type == "grid" else "euclidean"

    # Resolve the effective k. None with a radius means a pure radius graph;
    # otherwise default to 6 — squidpy's spatial convention (hex Visium spots
    # have 6 immediate neighbors), deliberately not scanpy's expression-space
    # default of 15.
    if coord_type == "delaunay":
        k: int | None = None
        astrolabe_method = "delaunay"
    elif radius is not None:
        k = n_neighbors
        astrolabe_method = "radius" if k is None else "knn_radius_intersection"
    else:
        k = 6 if n_neighbors is None else n_neighbors
        astrolabe_method = "knn"

    # ── Build directed edge arrays (rows, cols) ──────────────────────────────
    if coord_type == "delaunay":
        if coords.shape[1] != 2:
            raise ValueError(
                "coord_type='delaunay' requires 2-D coordinates; "
                f"obsm['{spatial_key}'] has {coords.shape[1]} columns"
            )
        from scipy.spatial import Delaunay as _Delaunay

        tri = _Delaunay(coords)
        # Extract unique undirected edges from triangle simplices — numpy path
        # avoids a Python set of 300K+ tuples for large datasets.
        simplices = tri.simplices  # (n_tri, 3)
        edges = np.concatenate([
            simplices[:, [0, 1]],
            simplices[:, [1, 2]],
            simplices[:, [0, 2]],
        ], axis=0)
        edges = np.sort(edges, axis=1)  # canonical order: i < j
        edge_arr = np.unique(edges, axis=0).astype(np.int32)  # (n_edges, 2)

        if delaunay_max_dist is not None:
            edge_d = np.linalg.norm(
                coords[edge_arr[:, 0]] - coords[edge_arr[:, 1]], axis=1
            )
            edge_arr = edge_arr[edge_d <= delaunay_max_dist]

        # Symmetrise: store both (i→j) and (j→i)
        rows = np.concatenate([edge_arr[:, 0], edge_arr[:, 1]])
        cols = np.concatenate([edge_arr[:, 1], edge_arr[:, 0]])

    elif radius is not None:
        nn = NearestNeighbors(metric=metric, algorithm="ball_tree")
        nn.fit(coords)
        _, indices = nn.radius_neighbors(coords, radius=radius, sort_results=True)

        # Radius returns variable-length index arrays per cell, sorted by
        # distance.  Remove each cell's own index (exact self only —
        # coincident *other* cells are kept), then optionally truncate to the
        # k nearest so an explicit n_neighbors acts as a top-k ∧ within-radius
        # intersection.
        row_parts: list[np.ndarray] = []
        col_parts: list[np.ndarray] = []
        for i, idx in enumerate(indices):
            keep = idx[idx != i]
            if k is not None:
                keep = keep[:k]
            row_parts.append(np.full(len(keep), i, dtype=np.int32))
            col_parts.append(keep.astype(np.int32))
        rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int32)
        cols = np.concatenate(col_parts) if col_parts else np.empty(0, np.int32)

    else:
        k_eff = min(k, n_cells - 1)
        if k_eff < 1:
            rows = np.empty(0, np.int32)
            cols = np.empty(0, np.int32)
        else:
            nn = NearestNeighbors(
                n_neighbors=k_eff, metric=metric, algorithm="ball_tree"
            )
            nn.fit(coords)
            # No-X kneighbors() queries the training set with each point
            # excluded from its own result *structurally* — robust to
            # duplicate coordinates, where filtering by index after a
            # kneighbors(coords) call can leave the wrong entry out.
            _, indices = nn.kneighbors()
            rows = np.repeat(np.arange(n_cells, dtype=np.int32), indices.shape[1])
            cols = indices.ravel().astype(np.int32)

    if set_diag:
        diag = np.arange(n_cells, dtype=np.int32)
        rows = np.concatenate([rows, diag])
        cols = np.concatenate([cols, diag])

    # ── Build sparse matrices ─────────────────────────────────────────────────
    # Connectivities come from the sparsity STRUCTURE of the edge list, not
    # from thresholding distances — a `dist > 0` mask silently deletes edges
    # between coincident centroids (e.g. Visium HD bin centers, merged FOVs),
    # whose true distance is exactly 0.
    conn_matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_cells, n_cells),
    )
    if coord_type != "delaunay":
        # Directed KNN is not symmetric; symmetrise by union — keep an edge
        # when either endpoint lists the other among its neighbors.
        conn_matrix = conn_matrix.maximum(conn_matrix.T)
    conn_matrix.data[:] = 1.0  # duplicates were summed on COO→CSR; re-binarize
    conn_matrix.sort_indices()

    # Distances share the connectivity matrix's exact sparsity structure
    # (indices/indptr), so zero-distance edges survive as explicit stored
    # zeros instead of vanishing into the implicit sparse background.
    row_idx = np.repeat(np.arange(n_cells), np.diff(conn_matrix.indptr))
    diffs = coords[row_idx] - coords[conn_matrix.indices]
    if coord_type == "grid":
        edge_d = np.abs(diffs).max(axis=1).astype(np.float32)
    else:
        edge_d = np.linalg.norm(diffs, axis=1).astype(np.float32)
    dist_matrix = sp.csr_matrix(
        (edge_d, conn_matrix.indices.copy(), conn_matrix.indptr.copy()),
        shape=(n_cells, n_cells),
    )

    mean_degree = conn_matrix.nnz / max(n_cells, 1)

    adata.obsp[key_added + "_connectivities"] = conn_matrix
    adata.obsp[key_added + "_distances"] = dist_matrix
    adata.uns[key_added] = {
        "connectivities_key": key_added + "_connectivities",
        "distances_key": key_added + "_distances",
        "params": {
            # scanpy's Neighbors machinery reads params['n_neighbors'] as an
            # int; for graphs without a fixed k (radius, delaunay) record the
            # rounded mean degree instead.
            "n_neighbors": int(k) if k is not None else int(round(mean_degree)),
            "radius": radius,
            "coord_type": coord_type,
            "delaunay_max_dist": delaunay_max_dist,
            "spatial_key": spatial_key,
            "metric": metric,
            # Honest provenance: how this graph was actually constructed.
            "astrolabe_method": astrolabe_method,
            # scanpy's own sc.tl.umap reads uns[neighbors_key]['params']['method']
            # unconditionally (KeyError, not a soft check, if absent) whenever
            # neighbors_key points here -- e.g. tl.umap/tl.cluster's default
            # neighbors_key='spatial_neighbors'. "umap" avoids both the crash
            # and a spurious "connectivities were not computed using umap"
            # warning; it is a label of convenience (this graph is built via
            # KNN/Delaunay + symmetrization, not literally UMAP's fuzzy
            # simplicial set), not a claim of exact algorithmic equivalence.
            # The honest construction record lives in 'astrolabe_method'.
            "method": "umap",
        },
    }

    log.info(
        "spatial_neighbors: %d cells, %d edges (mean degree %.1f) [%s]",
        n_cells,
        conn_matrix.nnz,
        conn_matrix.nnz / n_cells,
        coord_type,
    )


@with_adata
def transcriptional_neighbors(
    adata: ad.AnnData,
    *,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    use_rep: str | None = None,
    method: Literal["umap", "gauss"] = "umap",
    metric: str = "euclidean",
    random_state: int = 0,
    key_added: str | None = None,
    **kwargs: Any,
) -> None:
    """Build an expression-space (transcriptomic) neighborhood graph.

    Thin wrapper around :func:`scanpy.pp.neighbors` that accepts MuData
    directly (via ``@with_adata``), so ``ab.tl.transcriptional_neighbors(mdata)``
    replaces ``sc.pp.neighbors(mdata['xenium'])`` without unwrapping the
    modality by hand. This is the expression-space counterpart to
    :func:`spatial_neighbors`: it operates on ``obsm['X_pca']`` (or
    whatever *use_rep* points to), not µm coordinates.

    Results are stored in ``adata.obsp[...]`` and
    ``adata.uns[key_added or 'neighbors']``, matching scanpy's own
    convention -- the same key :func:`~astrolabe.tl.paga` and
    :func:`~astrolabe.tl.dpt` read by default, and the one raw
    ``sc.tl.leiden``/``sc.tl.umap`` calls fall back to when no
    ``neighbors_key`` is given.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`. Should have
        ``obsm['X_pca']`` populated first (see :func:`~astrolabe.tl.pca`).
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_neighbors
        Number of neighbors per cell.
    n_pcs
        Number of PCs to use from *use_rep*. ``None`` uses all available.
    use_rep
        Key in ``obsm`` to use as input. ``None`` uses ``obsm['X_pca']`` if
        present, otherwise scanpy computes PCA on ``X`` first.
    method
        ``'umap'`` (fuzzy simplicial set, default) or ``'gauss'`` (Gaussian
        kernel) connectivity model.
    metric
        Distance metric.  Not forwarded when a ``transformer=`` instance is
        passed via *kwargs* (scanpy computes neighbors inside the
        transformer in that case, and an explicit metric would conflict).
    random_state
        Random seed.
    key_added
        Key prefix for ``obsp``/``uns``. ``None`` (default) uses
        ``'neighbors'``, matching scanpy's convention.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pp.neighbors`.

    Examples
    --------
    >>> ab.tl.pca(mdata)
    >>> ab.tl.transcriptional_neighbors(mdata)
    >>> ab.tl.cluster(mdata, neighbors_key='neighbors')
    """
    if "transformer" not in kwargs:
        # only forward metric when scanpy computes neighbors itself — a
        # transformer instance owns its own metric and would conflict.
        kwargs["metric"] = metric
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        method=method,
        random_state=random_state,
        key_added=key_added,
        **kwargs,
    )
    log.info(
        "transcriptional_neighbors: %d cells -> uns['%s']",
        adata.n_obs,
        key_added or "neighbors",
    )


@with_adata
def blend_neighbors(
    adata: ad.AnnData,
    *,
    alpha: float = 0.5,
    expr_neighbors_key: str = "neighbors",
    spatial_neighbors_key: str = "spatial_neighbors",
    key_added: str = "joint_neighbors",
) -> None:
    """Blend an expression graph and a spatial graph into one joint graph.

    Combines the transcriptomic neighborhood graph (from ``sc.pp.neighbors``
    / :func:`transcriptional_neighbors`) with the spatial neighborhood graph
    (from :func:`spatial_neighbors`) into a single connectivity graph that
    captures both cell-type identity and tissue organization. The result is
    persisted in ``obsp``/``uns`` exactly like any other neighbors graph, so
    it is a drop-in ``neighbors_key`` for :func:`~astrolabe.tl.umap`
    (a cospatial embedding), :func:`~astrolabe.tl.cluster` (cospatial
    cluster labels), and the trajectory functions
    :func:`~astrolabe.tl.paga` / :func:`~astrolabe.tl.dpt`.

    Both source graphs must exist in ``obsp`` before calling this function.
    Run :func:`spatial_neighbors` and ``sc.pp.neighbors`` (after
    :func:`~astrolabe.tl.pca`) first.

    How the blend works
    -------------------
    The two connectivity matrices live on different scales (UMAP fuzzy
    weights vs. binary spatial adjacency), so each is first row-normalized
    to a row-stochastic transition matrix. *alpha* then mixes **transition
    mass**, not raw connectivity: each cell distributes a fraction *alpha*
    of its outgoing weight across its spatial neighbors and ``1 - alpha``
    across its expression neighbors. Because row normalization breaks
    symmetry, the blended matrix is finally symmetrized as
    ``(C + Cᵀ) / 2`` — Leiden implicitly symmetrizes asymmetric input
    anyway, and UMAP/PAGA/DPT expect a symmetric connectivity matrix, so
    doing it explicitly keeps every consumer on the same graph.

    The blended graph has **no metric distance matrix**: the two source
    graphs' distances are incommensurable (µm vs. PCA-space euclidean), so
    no meaningful blended distance exists.  Because scanpy's machinery
    requires a ``distances_key`` (and PAGA reads the distances matrix for
    its edge *structure*, binarizing the values), astrolabe stores a
    **unit-weight structural matrix** (every stored edge = 1.0) under
    ``obsp[key_added + '_distances']`` — do not interpret its values as
    distances.  Connectivity-consuming functions (leiden/umap/paga/diffmap/
    dpt) are unaffected.

    This is a lighter-weight, graph-level alternative to BANKSY's
    feature-level fusion (:func:`~astrolabe.pp.banksy_matrix`); the two are
    complementary, not redundant.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with both
        neighborhood graphs in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    alpha
        Spatial weight in [0, 1].  ``alpha=0`` reduces to the (row-normalized,
        symmetrized) expression graph; ``alpha=1`` to the (row-normalized,
        symmetrized) spatial graph; ``alpha=0.5`` (default) gives equal
        weight to both.
    expr_neighbors_key
        Key in ``uns`` for the transcriptomic neighborhood graph written
        by ``sc.pp.neighbors`` / :func:`transcriptional_neighbors`. Default
        ``'neighbors'``.
    spatial_neighbors_key
        Key in ``uns`` for the spatial neighborhood graph written by
        :func:`spatial_neighbors`. Default ``'spatial_neighbors'``.
    key_added
        Key prefix used to store the blended graph in ``obsp`` and ``uns``.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])                # expression graph
    >>> ab.tl.spatial_neighbors(mdata)                  # spatial graph
    >>> ab.tl.blend_neighbors(mdata, alpha=0.3)         # 30% spatial, 70% expression
    >>> ab.tl.cluster(mdata, neighbors_key="joint_neighbors")  # cospatial clusters
    >>> ab.tl.umap(mdata, neighbors_key="joint_neighbors",
    ...            key_added="X_spatial_umap")          # cospatial embedding
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha!r}")

    # ── Resolve connectivity matrices ─────────────────────────────────────────
    def _get_conn(key: str) -> sp.spmatrix:
        meta = adata.uns.get(key)
        if meta is None:
            raise KeyError(
                f"uns['{key}'] not found. "
                + (
                    "Run ab.tl.transcriptional_neighbors() first"
                    if key == expr_neighbors_key
                    else "Run ab.tl.spatial_neighbors() first"
                )
            )
        conn_key = meta.get("connectivities_key", f"{key}_connectivities")
        if conn_key not in adata.obsp:
            raise KeyError(
                f"obsp['{conn_key}'] not found (expected from uns['{key}'])"
            )
        return adata.obsp[conn_key].astype(np.float32)

    conn_expr = _get_conn(expr_neighbors_key)
    conn_spatial = _get_conn(spatial_neighbors_key)

    if conn_expr.shape != conn_spatial.shape:
        raise ValueError(
            f"Expression ({conn_expr.shape}) and spatial ({conn_spatial.shape}) "
            "connectivity matrices have different shapes"
        )

    # ── Row-normalize so both graphs are on the same scale ───────────────────
    def _row_norm(mat: sp.spmatrix) -> sp.spmatrix:
        row_sums = np.asarray(mat.sum(axis=1), dtype=np.float32).ravel()
        row_sums[row_sums == 0] = 1.0
        return sp.diags(1.0 / row_sums, format="csr") @ mat.tocsr()

    conn_joint = alpha * _row_norm(conn_spatial) + (1.0 - alpha) * _row_norm(conn_expr)

    # Row normalization breaks symmetry (cell i may weight j without j
    # weighting i equally). Symmetrize explicitly so every downstream
    # consumer (leiden/umap/paga/diffmap/dpt) sees the same graph instead of
    # each applying its own implicit fix-up.
    conn_joint = (conn_joint + conn_joint.T) * 0.5

    # ── Persist the blended graph like any other named neighbors graph ───────
    conn_joint_key = f"{key_added}_connectivities"
    if key_added in adata.uns:
        log.warning(
            "blend_neighbors: uns['%s'] already exists and will be overwritten. "
            "Pass a different key_added= to avoid this.",
            key_added,
        )
    conn_joint = conn_joint.astype(np.float32).tocsr()
    adata.obsp[conn_joint_key] = conn_joint

    # The blended graph has no METRIC distances: µm and PCA-space distances
    # are incommensurable, so no meaningful blended distance exists (the old
    # behavior of pointing at the SPATIAL µm distances silently fed the
    # wrong scale to anything reading distances).  But scanpy's
    # NeighborsView hard-requires the 'distances_key' dict entry, and
    # sc.tl.paga reads the distances matrix for its edge STRUCTURE (it
    # binarizes the values immediately) — so store a unit-weight structural
    # matrix: correct for every structural consumer, and obviously
    # non-metric (all 1.0) rather than subtly wrong.
    dist_joint_key = f"{key_added}_distances"
    dist_joint = conn_joint.copy()
    dist_joint.data = np.ones_like(dist_joint.data)
    adata.obsp[dist_joint_key] = dist_joint

    adata.uns[key_added] = {
        "connectivities_key": conn_joint_key,
        "distances_key": dist_joint_key,
        "params": {
            # see spatial_neighbors()'s params["method"] comment above: this
            # is a label of convenience so sc.tl.umap/sc.tl.leiden accept the
            # graph, not a claim of exact algorithmic equivalence.
            "method": "umap",
            # scanpy's Neighbors class requires an int n_neighbors; the
            # blended graph has no single k, so record the rounded mean
            # degree of the blended matrix.
            "n_neighbors": int(round(conn_joint.nnz / max(conn_joint.shape[0], 1))),
            "metric": "euclidean",
            "astrolabe_method": "blend_neighbors",
            "alpha": alpha,
            "expr_neighbors_key": expr_neighbors_key,
            "spatial_neighbors_key": spatial_neighbors_key,
        },
    }

    log.info(
        "blend_neighbors: blended graph (alpha=%.2f spatial / %.2f expression), "
        "%d non-zeros -> uns['%s']",
        alpha, 1.0 - alpha, conn_joint.nnz, key_added,
    )