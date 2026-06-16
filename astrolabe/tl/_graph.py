"""
Spatial neighborhood graph construction.

We build a KNN or radius-based graph on ``obsm['spatial']`` (micron
coordinates) and store it as a sparse adjacency matrix in ``obsp``,
following the scanpy/squidpy convention so that downstream tools
(leiden, UMAP, etc.) work without modification.
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from astrolabe._utils import with_adata

__all__ = ["spatial_neighbors"]

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
        Number of nearest neighbours per cell.  Ignored for
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
            radius; typically 5–7 neighbours per cell.  Long convex-hull
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
        # Extract unique undirected edges from triangle simplices
        simplices = tri.simplices  # (n_tri, 3)
        edge_set: set[tuple[int, int]] = set()
        for a, b, c in simplices:
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(b, c), max(b, c)))
            edge_set.add((min(a, c), max(a, c)))

        edge_arr = np.array(list(edge_set), dtype=np.int32)  # (n_edges, 2)
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

        rows_list, cols_list, dist_list = [], [], []
        for i in range(n_cells):
            d, idx = distances[i], indices[i]
            not_self = idx != i
            d, idx = d[not_self], idx[not_self]
            rows_list.append(np.full(len(idx), i, dtype=np.int32))
            cols_list.append(idx.astype(np.int32))
            dist_list.append(d.astype(np.float32))

        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        dists = np.concatenate(dist_list)

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
        },
    }

    # Mirror into uns['neighbors'] so sc.tl.leiden / sc.tl.umap work directly.
    # This intentionally overwrites any prior transcriptomic neighbors graph.
    if "neighbors" in adata.uns:
        log.info(
            "spatial_neighbors: overwriting uns['neighbors'] (previously set by sc.pp.neighbors or a prior call). "
            "Re-run sc.pp.neighbors if you need transcriptomic neighbors downstream."
        )
    adata.uns["neighbors"] = {
        "connectivities_key": key_added + "_connectivities",
        "distances_key": key_added + "_distances",
        "params": {"n_neighbors": n_neighbors, "method": "spatial", "metric": "euclidean"},
    }

    log.info(
        "spatial_neighbors: %d cells, %d edges (mean degree %.1f) [%s]",
        n_cells,
        conn_matrix.nnz,
        conn_matrix.nnz / n_cells,
        coord_type,
    )