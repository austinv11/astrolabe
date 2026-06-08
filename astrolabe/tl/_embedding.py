"""
Dimensionality reduction embeddings.  Thin wrappers around scanpy.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from astrolabe._utils import with_adata

__all__ = ["umap"]


@with_adata
def umap(
    adata: ad.AnnData,
    *,
    neighbors_key: str = "spatial_neighbors",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    key_added: str = "X_umap",
    **kwargs,
) -> None:
    """Compute UMAP embedding of the neighborhood graph.

    Stores the embedding in ``obsm[key_added]``.  Requires that
    :func:`astrolabe.tl.spatial_neighbors` or ``sc.pp.neighbors`` has
    been run first.

    Parameters
    ----------
    adata
        AnnData with connectivity graph in ``obsp``.
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph.
    min_dist
        UMAP minimum distance parameter.  Smaller values produce tighter
        clusters; larger values preserve more global structure.
    spread
        Effective scale of embedded points.
    n_components
        Number of UMAP dimensions.
    random_state
        Random seed.
    key_added
        Key in ``obsm`` for the embedding.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.umap`.
    """
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )