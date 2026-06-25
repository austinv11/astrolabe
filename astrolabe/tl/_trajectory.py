"""
Trajectory inference — PAGA and diffusion pseudotime.
Thin wrappers around scanpy that follow astrolabe's @with_adata pattern.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import scanpy as sc

from astrolabe._utils import with_adata

__all__ = ["paga", "dpt"]

log = logging.getLogger(__name__)


@with_adata
def paga(
    adata: ad.AnnData,
    groupby: str,
    *,
    neighbors_key: str = "neighbors",
    threshold: float | None = None,
    key_added: str = "paga",
    **kwargs: Any,
) -> None:
    """Partition-based graph abstraction (PAGA) trajectory inference.

    Abstracts the cell neighborhood graph into a cluster-level connectivity
    graph, enabling trajectory and lineage inference at coarser resolution.

    Wraps :func:`scanpy.tl.paga`.  Results are stored in ``uns['paga']``
    and can be visualized with :func:`scanpy.pl.paga` or overlaid onto a
    UMAP with ``sc.pl.umap(adata, color=groupby, edges=True)``.

    .. note::
        Run :func:`astrolabe.tl.cluster` (Leiden) and ``sc.pp.neighbors``
        (or :func:`astrolabe.tl.spatial_neighbors`) before calling this
        function.  PAGA requires a transcriptomic neighborhood graph for
        meaningful trajectories; pass ``neighbors_key='spatial_neighbors'``
        to use the spatial graph instead.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with a
        precomputed neighborhood graph.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    groupby
        ``obs`` column with cluster labels (e.g. ``'leiden'``).
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph used to abstract
        trajectories.  Defaults to ``'neighbors'`` (the transcriptomic graph
        built by ``sc.pp.neighbors``).
    threshold
        Minimum PAGA connectivity weight to display edges.  ``None`` lets
        scanpy choose its default (0.01).
    key_added
        Key in ``uns`` where PAGA results are stored.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.paga`.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])        # transcriptomic graph
    >>> ab.tl.cluster(mdata, neighbors_key="neighbors")
    >>> ab.tl.paga(mdata, groupby="leiden")
    >>> sc.pl.paga(mdata["xenium"], threshold=0.05)
    """
    kw: dict[str, Any] = dict(
        groups=groupby,
        neighbors_key=neighbors_key,
        **kwargs,
    )
    if threshold is not None:
        kw["threshold"] = threshold

    sc.tl.paga(adata, **kw)

    n_groups = adata.obs[groupby].nunique()
    log.info("paga: %d groups, results in uns['%s']", n_groups, key_added)


@with_adata
def dpt(
    adata: ad.AnnData,
    *,
    root_cell: int | None = None,
    root_obs_key: str | None = None,
    root_obs_val: str | None = None,
    neighbors_key: str = "neighbors",
    n_dcs: int = 10,
    n_branchings: int = 0,
    min_group_size: float = 0.01,
    key_added: str = "dpt_pseudotime",
    random_state: int = 0,
    **kwargs: Any,
) -> None:
    """Diffusion pseudotime (DPT) trajectory ordering.

    Orders cells along a developmental or activation trajectory by computing
    diffusion pseudotime from a user-specified root cell.

    Runs :func:`scanpy.tl.diffmap` (if not already present) then
    :func:`scanpy.tl.dpt`.  Pseudotime values are stored in
    ``obs[key_added]``.

    .. note::
        Requires a transcriptomic neighborhood graph.  Run
        ``sc.pp.neighbors(adata)`` (after :func:`astrolabe.tl.pca`) before
        calling this function.

    Root cell selection
    -------------------
    Exactly one of the following must be provided to identify the root cell:

    * ``root_cell`` — integer index into ``adata.obs``.
    * ``root_obs_key`` + ``root_obs_val`` — selects the first cell where
      ``obs[root_obs_key] == root_obs_val`` (e.g. the first cell in a
      specific cluster or cell-type annotation).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with a
        precomputed neighborhood graph.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    root_cell
        Integer index of the root cell.
    root_obs_key
        ``obs`` column to search for the root cell (used with
        *root_obs_val*).
    root_obs_val
        Value in ``obs[root_obs_key]`` identifying the root cell group.
        The first matching cell is used as the root.
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph.
    n_dcs
        Number of diffusion components to compute.
    n_branchings
        Number of branchings to detect (0 = linear pseudotime only).
    min_group_size
        Minimum fraction of cells for a branching to be called.
    key_added
        ``obs`` column for the pseudotime values.
    random_state
        Random seed.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.dpt`.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])
    >>> ab.tl.paga(mdata, groupby="leiden")
    >>> # root = first cell annotated as 'Progenitor'
    >>> ab.tl.dpt(mdata, root_obs_key="leiden_annotation", root_obs_val="Progenitor")
    >>> ab.pl.points(mdata, color="dpt_pseudotime").show()
    """
    # ── Resolve root cell ─────────────────────────────────────────────────────
    if root_cell is not None:
        adata.uns["iroot"] = int(root_cell)
    elif root_obs_key is not None and root_obs_val is not None:
        if root_obs_key not in adata.obs.columns:
            raise KeyError(
                f"obs column '{root_obs_key}' not found. "
                "Run ab.tl.cluster() or ab.tl.annotate_clusters() first"
            )
        matches = (adata.obs[root_obs_key].astype(str) == str(root_obs_val)).values
        if not matches.any():
            raise ValueError(
                f"No cells found where obs['{root_obs_key}'] == {root_obs_val!r}. "
                f"Available values: {sorted(adata.obs[root_obs_key].unique().tolist())!r}"
            )
        n_matches = int(matches.sum())
        adata.uns["iroot"] = int(matches.argmax())
        if n_matches > 1:
            log.warning(
                "dpt: %d cells match root condition (obs['%s'] == %r); "
                "using first match (index %d). Pass root_cell= for reproducibility.",
                n_matches, root_obs_key, root_obs_val, adata.uns["iroot"],
            )
        else:
            log.info(
                "dpt: root cell set to index %d (first cell where obs['%s'] == %r)",
                adata.uns["iroot"], root_obs_key, root_obs_val,
            )
    elif "iroot" not in adata.uns:
        raise ValueError(
            "Root cell not specified. Pass root_cell=<int> or "
            "root_obs_key= + root_obs_val=, or set adata.uns['iroot'] manually"
        )

    # ── Diffusion map (prerequisite for DPT) ─────────────────────────────────
    existing_diffmap = adata.obsm.get("X_diffmap")
    if existing_diffmap is None:
        log.info("dpt: computing diffusion map (n_dcs=%d)", n_dcs)
        sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key=neighbors_key,
                      random_state=random_state)
    elif existing_diffmap.shape[1] != n_dcs:
        log.warning(
            "dpt: existing X_diffmap has %d components but n_dcs=%d was requested; "
            "recomputing diffusion map.",
            existing_diffmap.shape[1], n_dcs,
        )
        sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key=neighbors_key,
                      random_state=random_state)

    # ── DPT ──────────────────────────────────────────────────────────────────
    sc.tl.dpt(
        adata,
        n_dcs=n_dcs,
        n_branchings=n_branchings,
        min_group_size=min_group_size,
        neighbors_key=neighbors_key,
        **kwargs,
    )

    if key_added != "dpt_pseudotime" and "dpt_pseudotime" in adata.obs.columns:
        adata.obs[key_added] = adata.obs.pop("dpt_pseudotime")

    log.info("dpt: pseudotime written to obs['%s']", key_added)