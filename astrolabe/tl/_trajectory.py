"""
Trajectory inference — PAGA and diffusion pseudotime.
Thin wrappers around scanpy that follow astrolabe's @with_adata pattern.
"""

from __future__ import annotations

import logging
from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from astrolabe._utils import _check_single_library, with_adata

__all__ = ["paga", "dpt"]

log = logging.getLogger(__name__)


def _group_medoid(
    member_rep: np.ndarray, global_idx: np.ndarray
) -> tuple[int, float, float]:
    """Medoid of a cell group: the member nearest the group centroid.

    Returns ``(global_index, centroid_to_nearest_member_dist, group_span)``
    so the caller can apply the ring/multi-nest guard (a ring-shaped group's
    centroid lies far from every member, making the medoid an effectively
    arbitrary point on the ring).
    """
    centroid = member_rep.mean(axis=0)
    d = np.linalg.norm(member_rep - centroid, axis=1)
    span = float(np.ptp(member_rep, axis=0).max()) if member_rep.shape[0] else 0.0
    return int(global_idx[int(d.argmin())]), float(d.min()), span


@with_adata
def paga(
    adata: ad.AnnData,
    groupby: str,
    *,
    neighbors_key: str = "neighbors",
    key_added: str = "paga",
    **kwargs: Any,
) -> None:
    """Partition-based graph abstraction (PAGA) trajectory inference.

    Abstracts the cell neighborhood graph into a cluster-level connectivity
    graph, enabling trajectory and lineage inference at coarser resolution.

    Wraps :func:`scanpy.tl.paga`.  Results are stored in ``uns[key_added]``
    and can be visualized with :func:`~astrolabe.pl.paga_plot` /
    :func:`scanpy.pl.paga` or overlaid onto a UMAP with
    ``sc.pl.umap(adata, color=groupby, edges=True)``.  Edge display
    thresholding is a **plotting-time** choice, not a fitting parameter —
    pass ``threshold=`` to :func:`~astrolabe.pl.paga_plot` /
    :func:`scanpy.pl.paga` instead.

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
    key_added
        Key in ``uns`` where PAGA results are stored.  scanpy always writes
        ``uns['paga']``; when a non-default *key_added* is given the result
        is moved there afterwards.  Note that ``sc.pl.paga`` /
        :func:`~astrolabe.pl.paga_plot` read the fixed ``uns['paga']`` key,
        so results stored under a custom key must be copied back before
        plotting.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.paga`.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])        # transcriptomic graph
    >>> ab.tl.cluster(mdata, neighbors_key="neighbors")
    >>> ab.tl.paga(mdata, groupby="leiden")
    >>> sc.pl.paga(mdata["xenium"], threshold=0.05)
    """
    sc.tl.paga(adata, groups=groupby, neighbors_key=neighbors_key, **kwargs)

    if key_added != "paga":
        adata.uns[key_added] = adata.uns.pop("paga")

    n_groups = adata.obs[groupby].nunique()
    log.info("paga: %d groups, results in uns['%s']", n_groups, key_added)


@with_adata
def dpt(
    adata: ad.AnnData,
    *,
    root_cell: int | str | None = None,
    root_obs_key: str | None = None,
    root_obs_val: str | None = None,
    root_rep: str = "X_pca",
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
    Exactly one of the following must be provided to identify the root cell
    (passing both, or only half of the key/val pair, raises ``ValueError``):

    * ``root_cell`` — integer index into ``adata.obs``, or an ``obs_names``
      barcode string.
    * ``root_obs_key`` + ``root_obs_val`` — selects a root from the group of
      cells where ``obs[root_obs_key] == root_obs_val`` (e.g. a cluster or
      cell-type annotation). The root is the group's **medoid**: the member
      cell nearest the group centroid in ``root_rep`` space — a
      representative choice that is independent of row order, so it
      survives subsetting/re-sorting.

    ``root_rep`` names the ``obsm`` embedding the centroid is computed in:
    ``'X_pca'`` (default; falls back to ``adata.X`` with a log note when
    absent), ``'spatial'`` (µm coordinates — the cell nearest the group's
    tissue centroid; requires a single library, and warns when the group is
    ring-shaped/multi-nested so its spatial centroid is not meaningful), or
    any other ``obsm`` key (a missing non-default key raises ``KeyError``).

    The resolved root is written to ``adata.uns['iroot']`` (with a
    provenance record in ``adata.uns['dpt_root']``) and **persists on the
    object** after this call — a later ``dpt()`` call with no root arguments
    reuses it (logged).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with a
        precomputed neighborhood graph.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    root_cell
        Integer index of the root cell (``0 <= root_cell < n_obs``), or an
        ``obs_names`` barcode string.  Mutually exclusive with
        *root_obs_key*/*root_obs_val*.
    root_obs_key
        ``obs`` column to search for the root cell (used with
        *root_obs_val*).
    root_obs_val
        Value in ``obs[root_obs_key]`` identifying the root cell group.
        The group's medoid in *root_rep* space is used as the root.
    root_rep
        ``obsm`` key of the embedding used for the medoid centroid.
        ``'X_pca'`` (default) falls back to ``adata.X`` when absent;
        ``'spatial'`` uses µm coordinates (single library required).
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph.
    n_dcs
        Number of diffusion components to compute.
    n_branchings
        Number of branchings to detect (0 = linear pseudotime only).
    min_group_size
        Minimum fraction of cells for a branching to be called.
    key_added
        ``obs`` column for the pseudotime values.  When *n_branchings* > 0,
        the branch labels scanpy writes to ``obs['dpt_groups']`` are renamed
        to ``obs[key_added + '_groups']`` for a non-default *key_added*.
    random_state
        Random seed.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.dpt`.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])
    >>> ab.tl.paga(mdata, groupby="leiden")
    >>> # root = PCA-space medoid of the 'Progenitor' group (the default)
    >>> ab.tl.dpt(mdata, root_obs_key="leiden_annotation", root_obs_val="Progenitor")
    >>> # root = the Progenitor cell nearest the group's tissue centroid
    >>> ab.tl.dpt(mdata, root_obs_key="leiden_annotation",
    ...           root_obs_val="Progenitor", root_rep="spatial")
    >>> mdata["xenium"].uns["dpt_root"]        # which cell was used, and why
    >>> ab.pl.points(mdata, color="dpt_pseudotime").show()
    """
    # ── Resolve root cell ─────────────────────────────────────────────────────
    if root_cell is not None and (root_obs_key is not None or root_obs_val is not None):
        raise ValueError(
            "Pass either root_cell= or root_obs_key= + root_obs_val=, not both."
        )
    if (root_obs_key is None) != (root_obs_val is None):
        missing = "root_obs_val" if root_obs_val is None else "root_obs_key"
        raise ValueError(
            f"root_obs_key and root_obs_val must be passed together; {missing} is missing."
        )

    root_provenance: dict[str, Any] | None = None
    if root_cell is not None:
        if isinstance(root_cell, str):
            try:
                iroot = int(adata.obs_names.get_loc(root_cell))
            except KeyError:
                raise KeyError(
                    f"root_cell {root_cell!r} not found in obs_names."
                ) from None
        elif isinstance(root_cell, (int, np.integer)) and not isinstance(root_cell, bool):
            if not 0 <= int(root_cell) < adata.n_obs:
                raise IndexError(
                    f"root_cell={root_cell} out of range for {adata.n_obs} cells "
                    "(negative indices are not supported)."
                )
            iroot = int(root_cell)
        else:
            raise TypeError(
                f"root_cell must be an int index or obs_names string; "
                f"got {type(root_cell).__name__}."
            )
        adata.uns["iroot"] = iroot
        root_provenance = {"method": "explicit", "n_matches": None, "root_rep": None}
    elif root_obs_key is not None:
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
        if root_rep == "spatial":
            # µm frames from different libraries are incomparable.
            _check_single_library(adata)
        if root_rep in adata.obsm:
            rep_name = root_rep
            member_rep = np.asarray(adata.obsm[root_rep])[matches]
            if root_rep == "spatial":
                member_rep = member_rep[:, :2]
        elif root_rep == "X_pca":
            # Only the default falls back to X — an explicitly requested
            # rep must never silently degrade.
            log.info(
                "dpt: obsm['X_pca'] not found; computing the root medoid "
                "on adata.X instead."
            )
            rep_name = "X"
            member_rep = adata.X[matches]
            if sp.issparse(member_rep):
                member_rep = member_rep.toarray()
        else:
            raise KeyError(
                f"root_rep {root_rep!r} not found in obsm. "
                f"Available: {sorted(adata.obsm.keys())!r}"
            )
        member_rep = np.asarray(member_rep, dtype=np.float64)
        iroot, nearest_dist, span = _group_medoid(
            member_rep, np.where(matches)[0]
        )
        if root_rep == "spatial" and span > 0 and nearest_dist > 0.1 * span:
            log.warning(
                "dpt: the spatial centroid of obs['%s'] == %r lies %.0f µm "
                "from the nearest member cell — the group is likely "
                "ring-shaped or multi-nested, so its spatial medoid is an "
                "effectively arbitrary point on the ring. Pass an explicit "
                "root_cell= instead.",
                root_obs_key, root_obs_val, nearest_dist,
            )
        adata.uns["iroot"] = iroot
        log.info(
            "dpt: root = medoid of %d cells where obs['%s'] == %r in %r "
            "space (index %d, obs_name %r)",
            n_matches, root_obs_key, root_obs_val, rep_name,
            iroot, str(adata.obs_names[iroot]),
        )
        root_provenance = {
            "method": "medoid", "n_matches": n_matches, "root_rep": rep_name,
        }
    elif "iroot" not in adata.uns:
        raise ValueError(
            "Root cell not specified. Pass root_cell=<int> or "
            "root_obs_key= + root_obs_val=, or set adata.uns['iroot'] manually"
        )
    else:
        log.info(
            "dpt: reusing existing uns['iroot']=%d (no root arguments passed)",
            int(adata.uns["iroot"]),
        )
        # Don't fabricate provenance for an externally set iroot; only stamp
        # a 'reused' record when none exists yet.
        if "dpt_root" not in adata.uns:
            root_provenance = {"method": "reused", "n_matches": None, "root_rep": None}

    if root_provenance is not None:
        iroot = int(adata.uns["iroot"])
        adata.uns["dpt_root"] = {
            "iroot": iroot,
            "obs_name": str(adata.obs_names[iroot]),
            "root_obs_key": root_obs_key,
            "root_obs_val": root_obs_val,
            **root_provenance,
        }

    # ── Diffusion map (prerequisite for DPT) ─────────────────────────────────
    # The diffmap is graph-specific: reusing one computed on a different
    # neighbors_key would make this run's pseudotime silently ignore the
    # requested graph (e.g. a 'joint_neighbors' DPT riding on the expression
    # graph's diffmap). Track provenance and recompute on any mismatch.
    existing_diffmap = adata.obsm.get("X_diffmap")
    diffmap_source = adata.uns.get("diffmap_neighbors_key")
    # An externally computed diffmap (no provenance stamp) is trusted as
    # before; only a diffmap this function built on a *different* graph
    # triggers the recompute.
    if existing_diffmap is None or (
        diffmap_source is not None and diffmap_source != neighbors_key
    ):
        if existing_diffmap is not None:
            log.info(
                "dpt: existing X_diffmap was built on neighbors_key=%r; "
                "recomputing on %r",
                diffmap_source, neighbors_key,
            )
        else:
            log.info("dpt: computing diffusion map (n_dcs=%d)", n_dcs)
        sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key=neighbors_key,
                      random_state=random_state)
        adata.uns["diffmap_neighbors_key"] = neighbors_key
    elif existing_diffmap.shape[1] < n_dcs:
        # Only recompute when there are FEWER components than requested —
        # sc.tl.dpt happily uses the first n_dcs of a wider diffmap.
        log.warning(
            "dpt: existing X_diffmap has %d components but n_dcs=%d was requested; "
            "recomputing diffusion map.",
            existing_diffmap.shape[1], n_dcs,
        )
        sc.tl.diffmap(adata, n_comps=n_dcs, neighbors_key=neighbors_key,
                      random_state=random_state)
        adata.uns["diffmap_neighbors_key"] = neighbors_key

    # ── DPT ──────────────────────────────────────────────────────────────────
    # sc.tl.dpt always writes obs['dpt_pseudotime'] (and 'dpt_groups' when
    # branching). For a custom key_added, stash an earlier run's default-key
    # columns first — otherwise a second dpt run with key_added= destroys the
    # first run's pseudotime (scanpy overwrites it, and the rename below used
    # to pop the overwritten column away entirely).
    stash_pt = stash_groups = None
    if key_added != "dpt_pseudotime":
        if "dpt_pseudotime" in adata.obs.columns:
            stash_pt = adata.obs["dpt_pseudotime"].copy()
        if n_branchings > 0 and "dpt_groups" in adata.obs.columns:
            stash_groups = adata.obs["dpt_groups"].copy()

    sc.tl.dpt(
        adata,
        n_dcs=n_dcs,
        n_branchings=n_branchings,
        min_group_size=min_group_size,
        neighbors_key=neighbors_key,
        **kwargs,
    )

    if key_added != "dpt_pseudotime":
        if "dpt_pseudotime" in adata.obs.columns:
            adata.obs[key_added] = adata.obs.pop("dpt_pseudotime")
        if stash_pt is not None:
            adata.obs["dpt_pseudotime"] = stash_pt
        # Rename this run's branch labels too — but only when this run
        # produced them (n_branchings > 0), never a stale column from an
        # earlier default-key run.
        if n_branchings > 0 and "dpt_groups" in adata.obs.columns:
            adata.obs[key_added + "_groups"] = adata.obs.pop("dpt_groups")
        if stash_groups is not None:
            adata.obs["dpt_groups"] = stash_groups

    log.info("dpt: pseudotime written to obs['%s']", key_added)