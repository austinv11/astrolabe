"""
Community detection: Leiden (default) and Louvain clustering.
Thin wrappers around scanpy to keep the astrolabe namespace consistent.
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import scanpy as sc

from astrolabe._gpu import _require_rapids, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["cluster"]

log = logging.getLogger(__name__)


@with_adata
def cluster(
    adata: ad.AnnData,
    *,
    method: Literal["leiden", "louvain"] = "leiden",
    resolution: float = 1.0,
    neighbors_key: str = "spatial_neighbors",
    key_added: str | None = None,
    random_state: int = 0,
    **kwargs,
) -> None:
    """Cluster cells using the Leiden or Louvain algorithm.

    Wraps :func:`scanpy.tl.leiden` / :func:`scanpy.tl.louvain` and stores
    cluster labels in ``obs[key_added]``.  Requires that
    :func:`astrolabe.tl.spatial_neighbors` (or ``sc.pp.neighbors``) has
    been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with
        connectivity graph in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    method
        Clustering algorithm.  ``'leiden'`` is strongly preferred for
        spatial data due to its resolution-limit-free formulation.
    resolution
        Resolution parameter controlling cluster granularity.  Higher
        values yield more clusters.
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph.  Defaults to
        ``'spatial_neighbors'`` (set by :func:`astrolabe.tl.spatial_neighbors`).
        Pass ``'neighbors'`` if using a transcriptomic (PCA-based) graph.
    key_added
        Column name in ``obs`` for cluster labels.  Defaults to
        ``method`` (e.g. ``'leiden'``).
    random_state
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments forwarded to the underlying scanpy
        function.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata["xenium"])
    >>> ab.tl.cluster(mdata["xenium"], resolution=0.5)
    >>> mdata["xenium"].obs["leiden"]
    """
    # FIXME: TEMP: Use future default args for speed with leiden, should eventually be removed
    if method == 'leiden':
        # flavor='igraph'` and `n_iterations=2
        if 'flavor' not in kwargs:
            kwargs['flavor'] = 'igraph'
        if 'n_iterations' not in kwargs:
            kwargs['n_iterations'] = 2
    key_added = key_added or method

    if gpu_enabled() and method in ("leiden", "louvain"):
        rsc = _require_rapids()
        fn = rsc.tl.leiden if method == "leiden" else rsc.tl.louvain
        fn(
            adata,
            resolution=resolution,
            neighbors_key=neighbors_key,
            key_added=key_added,
            random_state=random_state,
            **kwargs,
        )
        n_clusters = adata.obs[key_added].nunique()
        log.info("cluster (%s GPU, res=%.2f): %d clusters", method, resolution, n_clusters)
        return

    if method == "leiden":
        sc.tl.leiden(
            adata,
            resolution=resolution,
            neighbors_key=neighbors_key,
            key_added=key_added,
            random_state=random_state,
            **kwargs,
        )
    elif method == "louvain":
        sc.tl.louvain(
            adata,
            resolution=resolution,
            neighbors_key=neighbors_key,
            key_added=key_added,
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"method must be 'leiden' or 'louvain', got {method!r}")

    n_clusters = adata.obs[key_added].nunique()
    log.info("cluster (%s, res=%.2f): %d clusters", method, resolution, n_clusters)