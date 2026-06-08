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
    coord_type: Literal["generic", "grid"] = "generic",
    spatial_key: str = "spatial",
    key_added: str = "spatial_neighbors",
    set_diag: bool = False,
) -> None:
    """Build a spatial KNN (or radius) neighborhood graph from cell centroids.

    Results are stored in:

    - ``adata.obsp[key_added + '_connectivities']`` — binary (or weighted)
      sparse adjacency matrix
    - ``adata.obsp[key_added + '_distances']`` — Euclidean distances
    - ``adata.uns[key_added]`` — dict with graph construction parameters

    Parameters
    ----------
    adata
        AnnData with ``obsm[spatial_key]`` containing ``(n_cells, 2)``
        micron coordinates.
    n_neighbors
        Number of nearest neighbors per cell.  Ignored when ``radius``
        is set and ``coord_type='generic'`` (radius-only graph).
    radius
        If given, connect cells within this radius (in microns) instead
        of using a fixed-k graph.  When both ``n_neighbors`` and ``radius``
        are set, cells are connected if they are within the top-k *and*
        within the radius.
    coord_type
        ``'generic'`` — use Euclidean distance on centroid coordinates.
        ``'grid'`` — assume a regular grid and use chebyshev neighbors
        (appropriate for Visium-style spots, not Xenium cells).
    spatial_key
        Key in ``obsm`` for spatial coordinates.  Default ``'spatial'``.
    key_added
        Prefix for the keys added to ``obsp`` and ``uns``.
    set_diag
        If ``True``, set the diagonal of the connectivity matrix to 1
        (self-loops).  Default ``False``.

    Notes
    -----
    Uses :func:`sklearn.neighbors.NearestNeighbors` which scales to
    ~1 M cells efficiently using a ball-tree.  For datasets > 500 k cells
    consider using approximate methods (e.g. pynndescent) and passing
    the result directly.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found.  "
            f"Ensure obsm['{spatial_key}'] has been populated with (n_cells, 2) coordinates."
        )

    coords = adata.obsm[spatial_key].astype(np.float64)  # (n_cells, 2)
    n_cells = coords.shape[0]

    if coord_type == "grid":
        # Chebyshev neighbors: 8-connected grid
        nn = NearestNeighbors(n_neighbors=min(9, n_cells), metric="chebyshev", algorithm="ball_tree")
    else:
        nn = NearestNeighbors(
            n_neighbors=min(n_neighbors + 1, n_cells),  # +1 because query point is included
            metric="euclidean",
            algorithm="ball_tree",
        )

    nn.fit(coords)

    if radius is not None:
        distances, indices = nn.radius_neighbors(coords, radius=radius, sort_results=True)
    else:
        distances, indices = nn.kneighbors(coords)

    # Build COO data for sparse matrices
    rows_list, cols_list, dist_list = [], [], []
    for i in range(n_cells):
        d = distances[i]
        idx = indices[i]
        # Exclude self (distance == 0)
        not_self = idx != i
        d = d[not_self]
        idx = idx[not_self]
        rows_list.append(np.full(len(idx), i, dtype=np.int32))
        cols_list.append(idx.astype(np.int32))
        dist_list.append(d.astype(np.float32))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    dists = np.concatenate(dist_list)

    dist_matrix = sp.csr_matrix(
        (dists, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32
    )
    # Symmetrize distances (take minimum of (i,j) and (j,i))
    dist_matrix = dist_matrix.maximum(dist_matrix.T)

    # Binary connectivity
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
            "spatial_key": spatial_key,
            "metric": "euclidean",
        },
    }

    # Also set adata.uns['neighbors'] so scanpy tools (leiden, UMAP) work
    # without requiring the user to call sc.pp.neighbors separately.
    adata.uns["neighbors"] = {
        "connectivities_key": key_added + "_connectivities",
        "distances_key": key_added + "_distances",
        "params": {"n_neighbors": n_neighbors, "method": "spatial", "metric": "euclidean"},
    }

    log.info(
        "spatial_neighbors: %d cells, %d edges (mean degree %.1f)",
        n_cells,
        conn_matrix.nnz,
        conn_matrix.nnz / n_cells,
    )