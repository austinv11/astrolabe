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
    neighbors_key: str = "neighbors",
    key_added: str | None = None,
    random_state: int = 0,
    **kwargs,
) -> None:
    """Cluster cells using the Leiden or Louvain algorithm.

    Wraps :func:`scanpy.tl.leiden` / :func:`scanpy.tl.louvain` and stores
    cluster labels in ``obs[key_added]``.  Requires that
    :func:`astrolabe.tl.transcriptional_neighbors` (or ``sc.pp.neighbors``)
    has been run first, matching this function's own default
    ``neighbors_key='neighbors'`` — cell identity is defined
    transcriptomically by default, consistent with :func:`astrolabe.tl.umap`
    (which has always defaulted to ``'neighbors'``). Pass
    ``neighbors_key='spatial_neighbors'`` explicitly for spatial-domain
    clustering on the graph built by :func:`astrolabe.tl.spatial_neighbors`.

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
        ``'neighbors'`` (the transcriptomic/PCA-based graph, set by
        :func:`astrolabe.tl.transcriptional_neighbors` or
        ``sc.pp.neighbors``), matching :func:`astrolabe.tl.umap`'s own
        default. Pass ``'spatial_neighbors'`` (set by
        :func:`astrolabe.tl.spatial_neighbors`) for spatial-domain
        clustering instead, or ``'joint_neighbors'`` (set by
        :func:`astrolabe.tl.blend_neighbors`) for cospatial clustering.
    key_added
        Column name in ``obs`` for cluster labels.  Defaults to
        ``method`` (e.g. ``'leiden'``).
    random_state
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments forwarded to the underlying scanpy
        function.  On the CPU Leiden path, ``flavor='igraph'`` and
        ``n_iterations=2`` are injected as defaults (scanpy's announced
        future defaults; markedly faster than legacy ``leidenalg``) unless
        explicitly overridden.

    Notes
    -----
    With ``flavor='igraph'`` (the default here), scanpy routes to igraph's
    ``community_leiden`` (``objective_function='modularity'``), whereas the
    legacy ``flavor='leidenalg'`` uses ``RBConfigurationVertexPartition``.
    Both optimize resolution-scaled modularity, but the implementations
    differ (directedness handling, tie-breaking, RNG), so the same numeric
    *resolution* can yield slightly different clusters between flavors.
    Pass ``flavor='leidenalg'`` to reproduce legacy scanpy behavior exactly.

    Examples
    --------
    >>> ab.tl.transcriptional_neighbors(mdata["xenium"])
    >>> ab.tl.cluster(mdata["xenium"], resolution=0.5)
    >>> mdata["xenium"].obs["leiden"]

    >>> # Spatial-domain clustering instead of expression-based:
    >>> ab.tl.spatial_neighbors(mdata["xenium"])
    >>> ab.tl.cluster(mdata["xenium"], neighbors_key="spatial_neighbors", key_added="spatial_leiden")
    """
    key_added = key_added or method

    if gpu_enabled() and method in ("leiden", "louvain"):
        # rsc.tl.leiden/louvain do not accept scanpy-only params like
        # flavor= / n_iterations= — forward only what the user explicitly
        # passed, on their own responsibility.
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
        # scanpy's announced future defaults (much faster than legacy
        # leidenalg); see the resolution-semantics note in the docstring.
        kwargs.setdefault("flavor", "igraph")
        kwargs.setdefault("n_iterations", 2)
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