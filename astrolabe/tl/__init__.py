"""
astrolabe.tl — analysis tools for spatial transcriptomics.

Functions
---------
spatial_neighbors
    Build a spatial neighborhood graph from cell centroid coordinates.
cluster
    Leiden or Louvain community detection (thin wrapper around sc.tl.leiden/louvain).
umap
    UMAP embedding (thin wrapper around sc.tl.umap).
rank_genes_groups
    Differential expression between clusters (thin wrapper around sc.tl.rank_genes_groups).
spatial_autocorrelation
    Per-gene Moran's I spatial autocorrelation statistic.
"""

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._cluster import cluster
from astrolabe.tl._embedding import umap
from astrolabe.tl._de import rank_genes_groups
from astrolabe.tl._spatial_stats import spatial_autocorrelation

__all__ = [
    "spatial_neighbors",
    "cluster",
    "umap",
    "rank_genes_groups",
    "spatial_autocorrelation",
]