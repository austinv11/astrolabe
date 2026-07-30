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
    n_neighbors: int = 15,
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
    - ``adata.obsp[key_added + '_distances']`` — Euclidean distances in µm
    - ``adata.uns[key_added]`` — graph construction parameters

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with
        ``obsm[spatial_key]`` containing ``(n_cells, 2)`` micron coordinates.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_neighbors
        Number of nearest neighbors per cell.  Ignored for
        ``coord_type='delaunay'``; only used with ``'generic'`` and
        ``'grid'``
    radius
        Connect cells within this radius (µm) instead of a fixed-k graph.
        Ignored for ``coord_type='delaunay'``.  When both *n_neighbors*
        and *radius* are set, cells are connected if within the top-k
        **and** within the radius
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
            Chebyshev 8-connected grid (for Visium-style spot arrays).
    delaunay_max_dist
        Maximum edge length in µm for ``coord_type='delaunay'``.  Edges
        longer than this are removed, preventing spurious connections
        across empty tissue regions.  ``None`` keeps all Delaunay edges.
        A value of ~200 µm works well for Xenium data.  Ignored for other
        ``coord_type`` values.
    spatial_key
        Key in ``obsm`` for spatial coordinates.
    key_added
        Prefix for keys written to ``obsp`` and ``uns``.
    set_diag
        Set diagonal of connectivity matrix to 1 (self-loops).

    Notes
    -----
    KNN/radius graphs use :func:`sklearn.neighbors.NearestNeighbors`
    (ball-tree), which scales to ~1 M cells.  Delaunay triangulation uses
    :func:`scipy.spatial.Delaunay`.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)                       # KNN k=15
    >>> ab.tl.spatial_neighbors(mdata, coord_type='delaunay',
    ...                         delaunay_max_dist=200)       # Delaunay
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found.  "
            f"Ensure obsm['{spatial_key}'] has been populated with (n_cells, 2) coordinates."
        )

    coords = adata.obsm[spatial_key].astype(np.float64)  # (n_cells, 2)
    n_cells = coords.shape[0]

    # ── Build edge arrays (rows, cols, dists) ────────────────────────────────
    if coord_type == "delaunay":
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
        dists = np.linalg.norm(
            coords[edge_arr[:, 0]] - coords[edge_arr[:, 1]], axis=1
        ).astype(np.float32)

        if delaunay_max_dist is not None:
            mask = dists <= delaunay_max_dist
            edge_arr = edge_arr[mask]
            dists = dists[mask]

        # Symmetrise: store both (i→j) and (j→i)
        rows = np.concatenate([edge_arr[:, 0], edge_arr[:, 1]])
        cols = np.concatenate([edge_arr[:, 1], edge_arr[:, 0]])
        dists = np.concatenate([dists, dists])

    else:
        if coord_type == "grid":
            nn = NearestNeighbors(
                n_neighbors=min(9, n_cells), metric="chebyshev", algorithm="ball_tree"
            )
        else:
            nn = NearestNeighbors(
                n_neighbors=min(n_neighbors + 1, n_cells),
                metric="euclidean",
                algorithm="ball_tree",
            )

        nn.fit(coords)

        if radius is not None:
            distances, indices = nn.radius_neighbors(coords, radius=radius, sort_results=True)
        else:
            distances, indices = nn.kneighbors(coords)

        if radius is not None:
            # Radius returns variable-length arrays per cell; filter self-edges after concat
            rows = np.concatenate([np.full(len(idx), i, dtype=np.int32)
                                    for i, idx in enumerate(indices)])
            cols = np.concatenate([idx.astype(np.int32) for idx in indices])
            dists = np.concatenate([d.astype(np.float32) for d in distances])
            not_self = cols != rows
            rows, cols, dists = rows[not_self], cols[not_self], dists[not_self]
        else:
            # kneighbors returns a regular (n_cells, k) matrix — fully vectorized
            flat_rows = np.repeat(np.arange(n_cells, dtype=np.int32), indices.shape[1])
            flat_cols = indices.ravel().astype(np.int32)
            flat_dists = distances.ravel().astype(np.float32)
            not_self = flat_cols != flat_rows
            rows = flat_rows[not_self]
            cols = flat_cols[not_self]
            dists = flat_dists[not_self]

    # ── Build sparse matrices ─────────────────────────────────────────────────
    dist_matrix = sp.csr_matrix(
        (dists, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32
    )
    if coord_type != "delaunay":
        # KNN is not always symmetric; symmetrise by taking the minimum distance
        dist_matrix = dist_matrix.maximum(dist_matrix.T)

    conn_matrix = (dist_matrix > 0).astype(np.float32)
    if set_diag:
        conn_matrix.setdiag(1)

    adata.obsp[key_added + "_connectivities"] = conn_matrix
    adata.obsp[key_added + "_distances"] = dist_matrix
    adata.uns[key_added] = {
        "connectivities_key": key_added + "_connectivities",
        "distances_key": key_added + "_distances",
        "params": {
            "n_neighbors": n_neighbors,
            "radius": radius,
            "coord_type": coord_type,
            "delaunay_max_dist": delaunay_max_dist,
            "spatial_key": spatial_key,
            "metric": "euclidean",
            # scanpy's own sc.tl.umap reads uns[neighbors_key]['params']['method']
            # unconditionally (KeyError, not a soft check, if absent) whenever
            # neighbors_key points here -- e.g. tl.umap/tl.cluster's default
            # neighbors_key='spatial_neighbors'. "umap" avoids both the crash
            # and a spurious "connectivities were not computed using umap"
            # warning; it is a label of convenience (this graph is built via
            # KNN/Delaunay + symmetrization, not literally UMAP's fuzzy
            # simplicial set), not a claim of exact algorithmic equivalence.
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
        Distance metric.
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
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        method=method,
        metric=metric,
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
    it is a drop-in ``neighbors_key`` for both :func:`~astrolabe.tl.umap`
    (a cospatial embedding) and :func:`~astrolabe.tl.cluster` (cospatial
    cluster labels — cells of the same type that are also spatially
    co-localised cluster together more readily than cells that share
    identity but are scattered across the tissue).

    Both source graphs must exist in ``obsp`` before calling this function.
    Run :func:`spatial_neighbors` and ``sc.pp.neighbors`` (after
    :func:`~astrolabe.tl.pca`) first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with both
        neighborhood graphs in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    alpha
        Spatial weight in [0, 1].  ``alpha=0`` reduces to the (row-normalized)
        expression graph; ``alpha=1`` reduces to the (row-normalized) spatial
        graph; ``alpha=0.5`` (default) gives equal weight to both.
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

    # ── Persist the blended graph like any other named neighbors graph ───────
    conn_joint_key = f"{key_added}_connectivities"
    if key_added in adata.uns:
        log.warning(
            "blend_neighbors: uns['%s'] already exists and will be overwritten. "
            "Pass a different key_added= to avoid this.",
            key_added,
        )
    adata.obsp[conn_joint_key] = conn_joint.astype(np.float32)
    adata.uns[key_added] = {
        "connectivities_key": conn_joint_key,
        "distances_key": adata.uns[spatial_neighbors_key].get(
            "distances_key", f"{spatial_neighbors_key}_distances"
        ),
        "params": {
            # see spatial_neighbors()'s params["method"] comment above: this
            # is a label of convenience so sc.tl.umap/sc.tl.leiden accept the
            # graph, not a claim of exact algorithmic equivalence.
            "method": "umap",
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