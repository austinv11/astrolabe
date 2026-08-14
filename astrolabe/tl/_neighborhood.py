"""
Neighborhood-based spatial analyses.

nhood_enrichment
    Permutation test: are cell types spatially enriched as neighbors?
co_occurrence
    Co-occurrence probability ratio across spatial distance scales.
centrality_scores
    Graph centrality (degree, clustering) aggregated per cell type.
expression_by_distance
    Mean expression binned by distance to an anchor cell type.
ligrec
    Spatially-contextualized ligand-receptor permutation test.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import _check_single_library, _spawn_rngs, from_kwarg, with_adata
from astrolabe.tl._spatial_stats import _get_conn_matrix

__all__ = [
    "nhood_enrichment",
    "co_occurrence",
    "centrality_scores",
    "expression_by_distance",
    "ligrec",
    "composition",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _category_codes(adata: ad.AnnData, groupby: str) -> tuple[np.ndarray, list]:
    """Return (int codes, category list) for obs[groupby] cast to Categorical."""
    if groupby not in adata.obs.columns:
        raise KeyError(f"obs['{groupby}'] not found")
    labels = adata.obs[groupby].astype("category")
    return labels.cat.codes.to_numpy(), labels.cat.categories.tolist()


def _one_hot(codes: np.ndarray, n_types: int) -> sp.csr_matrix:
    """Sparse one-hot matrix (n_cells × n_types)."""
    n = len(codes)
    return sp.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), codes)),
        shape=(n, n_types),
    )


def _edge_arrays(adata: ad.AnnData, neighbors_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (src_indices, dst_indices) COO edge arrays from obsp."""
    coo = _get_conn_matrix(adata, neighbors_key).tocoo()
    return coo.row.astype(np.intp), coo.col.astype(np.intp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@with_adata(mirror_uns=from_kwarg("key_added"))
def nhood_enrichment(
    adata: ad.AnnData,
    groupby: str,
    *,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    seed: int = 0,
    n_jobs: int = 1,
    key_added: str = "nhood_enrichment",
) -> None:
    """Neighborhood enrichment permutation test.

    For each pair of cell types (A, B), asks whether cells of type A and type B
    are connected in the spatial graph more (or less) than expected by chance.
    Labels are permuted ``n_perms`` times to build a null distribution; the
    result is reported as a z-score.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column with cell-type / cluster labels.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    neighbors_key
        ``uns`` key written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Number of label permutations for the null distribution.
    seed
        Random seed for reproducibility.
    n_jobs
        Parallel workers for the permutation loop (joblib, thread-backed —
        no extra memory copies since arrays are shared, not pickled). ``1``
        (default) runs sequentially. ``-1`` uses all available cores.
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``zscore``
        (n_types × n_types) float array of enrichment z-scores.
    ``counts``
        (n_types × n_types) int array of observed edge counts.
    ``groups``
        List of cell-type labels corresponding to matrix rows/columns.

    Examples
    --------
    >>> ab.tl.nhood_enrichment(mdata, 'leiden')
    >>> ab.pl.nhood_enrichment_plot(mdata, 'leiden')
    """
    codes, categories = _category_codes(adata, groupby)
    n_types = len(categories)
    n_cells = adata.n_obs

    adj = _get_conn_matrix(adata, neighbors_key).astype(bool)
    C = _one_hot(codes, n_types)

    # Observed: (n_types × n_types) edge count matrix
    obs_counts = (C.T @ adj @ C).toarray().astype(np.float64)

    # Permutations — each perm draws from an independent RNG substream so the
    # loop is safe to parallelize; mean/std are computed two-pass afterward
    # (numpy's mean-then-deviations), which avoids the same catastrophic-
    # cancellation failure mode a single-pass sum-of-squares formula would hit.
    def _one_perm(child_rng: "np.random.Generator") -> np.ndarray:
        perm_codes = codes.copy()
        child_rng.shuffle(perm_codes)
        Cp = _one_hot(perm_codes, n_types)
        return (Cp.T @ adj @ Cp).toarray().astype(np.float64)

    child_rngs = _spawn_rngs(seed, n_perms)
    perm_mats = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_perm)(child_rng) for child_rng in child_rngs
    )
    perm_stack = np.stack(perm_mats)
    perm_mean = perm_stack.mean(axis=0)
    perm_std = perm_stack.std(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(perm_std > 0, (obs_counts - perm_mean) / perm_std, 0.0)

    adata.uns[key_added] = {
        "zscore": zscore,
        "counts": obs_counts.astype(np.int64),
        "groups": categories,
    }
    log.info("nhood_enrichment: %d cell types, %d permutations.", n_types, n_perms)


@with_adata(mirror_uns=from_kwarg("key_added"))
def co_occurrence(
    adata: ad.AnnData,
    groupby: str,
    *,
    spatial_key: str = "spatial",
    interval: tuple[float, float, int] | np.ndarray = (10.0, 500.0, 50),
    key_added: str = "co_occurrence",
) -> None:
    """Co-occurrence probability ratio across spatial scales.

    For each distance threshold *d* and each pair of cell types (A, B),
    computes the ratio:

    .. math::

        \\text{ratio}(A, B, d) =
        \\frac{P(\\text{type B in the annulus ending at }d\\text{ from a type A cell})}
             {P(\\text{type B among all neighbor-pairs found in that same annulus})}

    A ratio > 1 indicates enriched co-occurrence at that scale; < 1 indicates
    avoidance. Distance thresholds are **annular, not cumulative**: the ring
    for threshold ``interval[k]`` is ``(interval[k-1], interval[k]]``
    (``[0, interval[0]]`` for the first) — each bin reflects only pairs at
    that specific distance scale, not every closer pair re-included at every
    larger threshold too. The within-ring probability is over the neighbors
    actually present in that ring (of any type) — not a raw neighbor count —
    so under no real spatial structure the ratio hovers near 1 at every *d*.
    Both the numerator and the marginal denominator are **ring-local**
    (recomputed at each distance bin from that bin's own neighbor counts,
    not the whole-tissue type frequencies) — matching
    ``squidpy.gr.co_occurrence``'s convention, the standard one for this
    statistic. This also makes the ratio symmetric: ``ratio(A, B, d) ==
    ratio(B, A, d)``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column with cell-type / cluster labels.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    spatial_key
        ``obsm`` key for cell centroids in µm.
    interval
        Either a ``(start_µm, stop_µm, n_steps)`` tuple (evenly spaced) or a
        1-D array of explicit distance thresholds in µm.
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``occ``
        (n_types × n_types × n_intervals) float array of co-occurrence ratios.
    ``interval``
        1-D float array of distance thresholds used.
    ``groups``
        List of cell-type labels.

    Examples
    --------
    >>> ab.tl.co_occurrence(mdata, 'leiden', interval=(10, 300, 30))
    >>> ab.pl.co_occurrence_plot(mdata, 'leiden', source_groups=['0'])
    """
    _check_single_library(adata)

    if gpu_enabled():
        rsc = _require_rapids()
        # Convert astrolabe interval format to ndarray for rsc
        if isinstance(interval, tuple) and len(interval) == 3:
            start, stop, n = interval
            rsc_interval = np.linspace(float(start), float(stop), int(n))
        else:
            rsc_interval = np.asarray(interval, dtype=np.float64)
        rsc.gr.co_occurrence(
            adata,
            cluster_key=groupby,
            spatial_key=spatial_key,
            interval=rsc_interval,
        )
        rsc_key = f"{groupby}_co_occurrence"
        result = adata.uns.pop(rsc_key)
        adata.uns[key_added] = {
            "occ": result["occ"],
            "interval": result["interval"],
            "groups": list(adata.obs[groupby].cat.categories),
        }
        return

    from scipy.spatial import KDTree

    codes, categories = _category_codes(adata, groupby)
    n_types = len(categories)

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]

    if isinstance(interval, tuple) and len(interval) == 3:
        thresholds = np.linspace(float(interval[0]), float(interval[1]), int(interval[2]))
    else:
        thresholds = np.asarray(interval, dtype=np.float64)

    n_cells = len(coords)

    # Query all unique pairs within the maximum threshold
    tree = KDTree(coords)
    pairs = tree.query_pairs(float(thresholds[-1]), output_type="ndarray")  # (n_pairs, 2)

    occ = np.zeros((n_types, n_types, len(thresholds)), dtype=np.float64)

    if len(pairs) > 0:
        dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        src_codes = codes[pairs[:, 0]]
        dst_codes = codes[pairs[:, 1]]

        for k, d in enumerate(thresholds):
            # Annular, not cumulative: each threshold's ring is
            # (thresholds[k-1], thresholds[k]] (or [0, thresholds[0]] for the
            # first), not everything from 0 out to d. `dists <= d` at every k
            # re-included every closer pair at every larger threshold, so a
            # genuine, tightly-localized co-localization signal at the first
            # bin got diluted almost to invisibility by the second bin
            # onward -- defeating the point of a *distance-resolved* curve.
            lo = thresholds[k - 1] if k > 0 else 0.0
            mask = (dists > lo) & (dists <= d)
            s = src_codes[mask]
            t = dst_codes[mask]

            # Undirected: accumulate both (A→B) and (B→A)
            count_mat = np.zeros((n_types, n_types), dtype=np.float64)
            np.add.at(count_mat, (s, t), 1.0)
            np.add.at(count_mat, (t, s), 1.0)

            # Conditional prob P(B | A at d): divide by the total number of
            # neighbors of *any* type within d of a type-A cell (the row sum
            # of count_mat), not by the number of type-A cells. Dividing by
            # type_counts[A] instead gives the mean *count* of type-B
            # neighbors per A-cell -- an unbounded quantity that grows ~d^2
            # as the search radius sweeps in more cells of every type -- not
            # a probability, so the ratio against the (bounded) marginal
            # P(B) diverges with d instead of converging to ~1 far from any
            # real spatial structure.
            row_totals = count_mat.sum(axis=1)
            # Marginal P(B) is ring-local (computed from row_totals at this
            # distance bin), not the whole-tissue global type fraction --
            # matching squidpy.gr.co_occurrence's own convention (the
            # standard, expected one for this statistic in the spatial
            # ecosystem). A global marginal answers "is B enriched near A
            # relative to how common B is anywhere in the tissue"; the
            # ring-local marginal answers "is the A/B pairing enriched
            # relative to how common each type already is at this specific
            # distance scale" -- the latter is what users coming from
            # squidpy/the published co-occurrence literature expect, and
            # it's also what makes the ratio symmetric in (A, B) by
            # construction, since both count_mat and the row-total-derived
            # marginal are themselves symmetric quantities here.
            ring_total = row_totals.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                marginal_local = np.where(ring_total > 0, row_totals / ring_total, 0.0)
                cond = np.where(
                    row_totals[:, None] > 0,
                    count_mat / row_totals[:, None],
                    0.0,
                )
                occ[:, :, k] = np.where(
                    marginal_local[None, :] > 0, cond / marginal_local[None, :], 0.0
                )

    adata.uns[key_added] = {
        "occ": occ,
        "interval": thresholds,
        "groups": categories,
    }
    log.info(
        "co_occurrence: %d cell types, %d distance thresholds.", n_types, len(thresholds)
    )


@with_adata(mirror_uns=from_kwarg("key_added"))
def centrality_scores(
    adata: ad.AnnData,
    groupby: str,
    *,
    neighbors_key: str = "spatial_neighbors",
    max_closeness_n: int = 5_000,
    force_closeness: bool = False,
    key_added: str = "centrality_scores",
) -> None:
    """Spatial graph *group* centrality metrics per cell type, matching squidpy.

    Computes three **group-level** centrality metrics from the spatial
    graph — the same `networkx` group-centrality algorithms
    ``squidpy.gr.centrality_scores`` itself calls, treating each cell type
    as a single group rather than averaging individual cells' scores:

    - **degree_centrality** — :func:`networkx.group_degree_centrality`: the
      fraction of cells *outside* the group that are directly connected to
      *at least one* member of the group. A property of the group as a
      whole ("how much of the rest of the tissue does this cell type
      touch"), not a per-cell average.
    - **clustering** — :func:`networkx.average_clustering` restricted to
      the group's own cells: the mean individual clustering coefficient
      (fraction of a cell's neighbors that are also neighbors of each
      other) over the group's members. This one *is* equivalent to
      averaging a per-cell metric — group and per-cell clustering coincide
      by definition — so it's the one column here directly comparable to a
      naive "average this per-cell metric within groups" implementation.
    - **closeness_centrality** — :func:`networkx.group_closeness_centrality`:
      how close the group is, collectively, to every cell outside it
      (``|V-S| / sum_{v not in S} min_{u in S} d(u, v)``) — the group's
      *best* reach to each outside cell, not the average individual
      cell's reach. Skipped by default for datasets larger than
      ``max_closeness_n`` since the underlying shortest-path computation
      can still be expensive for many groups on a large graph; pass
      ``force_closeness=True`` to compute it anyway.

    An earlier version of this function computed the mean of each cell's
    *individual* `networkx.degree_centrality`/`closeness_centrality` within
    its group instead — a legitimate but different question ("how central
    is a typical cell of this type") from squidpy's own group-as-a-block
    definition. This version answers squidpy's question, so results are
    now directly comparable to ``squidpy.gr.centrality_scores``.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column with cell-type / cluster labels.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    neighbors_key
        ``uns`` key written by :func:`~astrolabe.tl.spatial_neighbors`.
    max_closeness_n
        Skip ``closeness_centrality`` (fill with NaN) when ``adata.n_obs``
        exceeds this, unless ``force_closeness=True``.
    force_closeness
        Compute ``closeness_centrality`` regardless of ``max_closeness_n`` —
        an explicit opt-in given the O(n²) cost.
    key_added
        ``uns`` key to write the result DataFrame into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    A :class:`pandas.DataFrame` indexed by cell type with columns
    ``degree_centrality``, ``clustering``, and ``closeness_centrality``
    (NaN if skipped for large datasets).

    Examples
    --------
    >>> ab.tl.centrality_scores(mdata, 'leiden')
    >>> ab.tl.centrality_scores(mdata, 'leiden', force_closeness=True)  # force closeness on a big dataset
    >>> ab.pl.centrality_scores_plot(mdata, 'leiden')
    """
    import networkx as nx

    codes, categories = _category_codes(adata, groupby)
    n = adata.n_obs

    adj = _get_conn_matrix(adata, neighbors_key).astype(bool)
    G = nx.from_scipy_sparse_array(adj)

    do_closeness = force_closeness or n <= max_closeness_n
    if not do_closeness:
        log.warning(
            "centrality_scores: skipping closeness_centrality (n=%d > %d). "
            "Pass force_closeness=True to compute it anyway.",
            n,
            max_closeness_n,
        )

    rows = []
    for i, _cat in enumerate(categories):
        group_nodes = np.where(codes == i)[0].tolist()
        if not group_nodes:
            rows.append(
                {"degree_centrality": np.nan, "clustering": np.nan, "closeness_centrality": np.nan}
            )
            continue
        rows.append(
            {
                "degree_centrality": nx.group_degree_centrality(G, group_nodes),
                "clustering": nx.average_clustering(G, nodes=group_nodes),
                "closeness_centrality": (
                    nx.group_closeness_centrality(G, group_nodes) if do_closeness else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows, index=pd.Index(categories, name=groupby))

    adata.uns[key_added] = result
    log.info("centrality_scores: %d cell types.", len(categories))


@with_adata(mirror_uns=from_kwarg("key_added"))
def expression_by_distance(
    adata: ad.AnnData,
    anchor_key: str,
    anchor_groups: str | list[str],
    *,
    genes: str | list[str] | None = None,
    spatial_key: str = "spatial",
    layer: str | None = None,
    n_bins: int = 50,
    max_dist_um: float | None = None,
    key_added: str = "expression_by_distance",
) -> None:
    """Expression gradient as a function of distance to an anchor cell type.

    Computes the distance of every cell to its nearest cell of any
    ``anchor_groups`` type, bins the distances, and returns per-bin mean
    expression ± standard error for each requested gene.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    anchor_key
        ``obs`` column defining cell types / annotations.
    anchor_groups
        Value(s) in ``obs[anchor_key]`` that define the anchor population.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    genes
        Gene(s) to compute gradients for.  ``None`` computes all genes
        (may be slow for large var).
    spatial_key
        ``obsm`` key for cell centroids in µm.
    layer
        Expression layer.  ``None`` uses ``X``.
    n_bins
        Number of distance bins.
    max_dist_um
        Furthest distance to include.  Defaults to the 95th-percentile
        distance across all cells.
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``mean``
        :class:`pandas.DataFrame` (n_bins × n_genes); index = bin centers in µm.
    ``se``
        :class:`pandas.DataFrame` — standard error (same shape as ``mean``).
    ``n_cells``
        1-D int array of cell counts per bin.
    ``distance_bins``
        1-D float array of bin edges.
    ``anchor_key``, ``anchor_groups``
        Stored for reference.

    Examples
    --------
    >>> ab.tl.expression_by_distance(mdata, 'leiden', '3', genes=['EPCAM', 'CD3D'])
    >>> ab.pl.expression_by_distance_plot(mdata, 'EPCAM')
    """
    from scipy.spatial import KDTree

    _check_single_library(adata)

    if isinstance(anchor_groups, str):
        anchor_groups = [anchor_groups]
    if anchor_key not in adata.obs.columns:
        raise KeyError(f"obs['{anchor_key}'] not found")
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    anchor_mask = adata.obs[anchor_key].isin(anchor_groups).to_numpy()
    if not anchor_mask.any():
        raise ValueError(
            f"No cells found matching anchor_groups={anchor_groups!r} "
            f"in obs['{anchor_key}']"
        )

    # Distance to nearest anchor cell
    anchor_coords = coords[anchor_mask]
    dists, _ = KDTree(anchor_coords).query(coords, k=1)
    dists = dists.astype(np.float64)

    # Resolve expression matrix
    if genes is not None:
        if isinstance(genes, str):
            genes = [genes]
        missing = [g for g in genes if g not in adata.var_names]
        if missing:
            raise KeyError(f"Genes not found in var_names: {missing}")
        gene_idx = [adata.var_names.get_loc(g) for g in genes]
        mat = adata.layers[layer] if layer is not None else adata.X
        mat = mat[:, gene_idx]
        gene_names = list(genes)
    else:
        mat = adata.layers[layer] if layer is not None else adata.X
        gene_names = adata.var_names.tolist()

    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=np.float32)

    # Bin edges
    if max_dist_um is None:
        max_dist_um = float(np.percentile(dists, 95))
    bin_edges = np.linspace(0.0, max_dist_um, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # in_range gates out cells beyond max_dist_um entirely -- np.digitize
    # maps them past the last edge (index n_bins after the -1 below), which
    # np.clip(..., 0, n_bins-1) used to silently fold back into the
    # *terminal* bin instead of excluding them. With the default
    # max_dist_um (95th percentile), that dumped the farthest 5% of cells
    # into the last bin alongside genuinely-in-range cells, systematically
    # distorting the tail of every expression-vs-distance curve.
    in_range = dists <= max_dist_um
    bin_idx = np.clip(np.digitize(dists, bin_edges) - 1, 0, n_bins - 1)

    n_genes = len(gene_names)
    mean_arr = np.full((n_bins, n_genes), np.nan, dtype=np.float32)
    se_arr = np.full((n_bins, n_genes), np.nan, dtype=np.float32)
    n_cells_arr = np.zeros(n_bins, dtype=np.int64)

    for k in range(n_bins):
        mask = in_range & (bin_idx == k)
        n_cells_arr[k] = int(mask.sum())
        if n_cells_arr[k] > 0:
            vals = mat[mask]
            mean_arr[k] = vals.mean(axis=0)
            if n_cells_arr[k] > 1:
                se_arr[k] = vals.std(axis=0) / np.sqrt(n_cells_arr[k])
            else:
                se_arr[k] = 0.0

    adata.uns[key_added] = {
        "mean": pd.DataFrame(mean_arr, index=bin_centers, columns=gene_names),
        "se": pd.DataFrame(se_arr, index=bin_centers, columns=gene_names),
        "n_cells": n_cells_arr,
        "distance_bins": bin_edges,
        "anchor_key": anchor_key,
        "anchor_groups": anchor_groups,
    }
    log.info(
        "expression_by_distance: %d genes, %d bins, max_dist=%.1f µm.",
        n_genes,
        n_bins,
        max_dist_um,
    )


@with_adata(mirror_uns=from_kwarg("key_added"))
def ligrec(
    adata: ad.AnnData,
    groupby: str,
    lr_pairs: pd.DataFrame,
    *,
    neighbors_key: str = "spatial_neighbors",
    layer: str | None = None,
    n_perms: int = 200,
    seed: int = 0,
    threshold: float = 0.05,
    n_jobs: int = 1,
    key_added: str = "ligrec",
) -> None:
    """Spatially-contextualized ligand-receptor permutation test.

    For each LR pair (L, R) and each directed cell-type pair (sender A →
    receiver B), computes ``(mean(expression_L) + mean(expression_R)) / 2``
    over all spatial graph edges where the source is of type A and the
    target is of type B — the CellPhoneDB (Efremova et al. 2020) scoring
    convention, restricted here to spatial-graph edges rather than the full
    cross product of every cell of each type. Cell-type labels are permuted
    ``n_perms`` times to produce a null distribution and an empirical
    p-value.

    By restricting to actual spatial neighbors the test identifies LR
    interactions that are plausible given the spatial architecture of the
    tissue — a more stringent criterion than bulk cell-type-level analysis.

    .. warning::

        The GPU path (``ab.use_gpu(True)``) delegates to ``rsc.gr.ligrec``,
        which has no spatial-graph-restriction parameter at all — it
        computes a CellPhoneDB-style cluster-mean statistic across *every*
        cell of each type, completely ignoring ``neighbors_key`` and
        ``layer``. This is a genuinely different, non-spatial test from the
        CPU path described above, not just a faster version of the same
        thing. Check ``uns[key_added]['metadata']['spatial_restricted']``
        (``True`` on CPU, ``False`` on GPU) if a result's provenance matters.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column with cell-type / cluster labels.
    lr_pairs
        :class:`pandas.DataFrame` with at least columns ``'ligand'`` and
        ``'receptor'``.  Pairs whose genes are absent from ``var_names``
        are silently dropped.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    neighbors_key
        ``uns`` key written by :func:`~astrolabe.tl.spatial_neighbors`.
    layer
        Expression layer.  ``None`` uses ``X``.
    n_perms
        Permutations for the null distribution.  Increase to 1 000 for
        final publication-quality p-values (slower).
    seed
        Random seed.
    threshold
        P-value threshold reported in the log summary (does not filter results).
    n_jobs
        Parallel workers for the CPU permutation loop (joblib, thread-backed
        — no extra memory copies since arrays are shared, not pickled).
        ``1`` (default) runs sequentially. ``-1`` uses all available cores.
        Ignored on the GPU path (``ab.use_gpu(True)``), which delegates to
        ``rsc.gr.ligrec`` instead.
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``means``
        :class:`pandas.DataFrame` — rows = (ligand, receptor) MultiIndex,
        columns = (sender, receiver) MultiIndex.  Values are the
        CellPhoneDB-style ``(mean(L) + mean(R)) / 2`` score over spatial
        edges.
    ``pvalues``
        Same shape as ``means``; empirical p-values.
    ``metadata``
        Dict storing ``groupby``, ``n_perms``, ``threshold``.

    Examples
    --------
    >>> import pandas as pd
    >>> lr = pd.DataFrame({'ligand': ['CXCL12', 'VEGFA'], 'receptor': ['CXCR4', 'KDR']})
    >>> ab.tl.ligrec(mdata, 'leiden', lr)
    >>> ab.pl.ligrec_plot(mdata)
    """
    if not {"ligand", "receptor"}.issubset(lr_pairs.columns):
        raise ValueError("lr_pairs must have 'ligand' and 'receptor' columns")

    if gpu_enabled():
        # rsc.gr.ligrec has no spatial-graph-restriction parameter at all --
        # it computes CellPhoneDB-style cluster-mean expression across every
        # cell of each type, completely ignoring `neighbors_key` and
        # `layer`. This is a genuinely different, non-spatial statistic from
        # the CPU path above (which restricts every product to actual
        # spatial-graph edges) -- toggling ab.use_gpu(True) silently swapped
        # one test for the other with no indication in the result. Since
        # there is no way to make rsc.gr.ligrec spatially-restricted, the
        # divergence is made loud instead of silent: a warning on every
        # call, and `spatial_restricted: False` recorded in the metadata so
        # a result can be checked programmatically, not just via a log line
        # a caller may not see.
        log.warning(
            "ligrec: GPU path (rsc.gr.ligrec) computes a non-spatial, "
            "cluster-mean statistic across ALL cells of each type -- "
            "neighbors_key=%r and layer=%r are ignored entirely. This is "
            "NOT equivalent to the CPU path's spatial-graph-edge-restricted "
            "test. Use ab.use_gpu(False) for a spatially-restricted result.",
            neighbors_key, layer,
        )
        rsc = _require_rapids()
        # rsc.gr.ligrec expects 'source'/'target' column names
        interactions_df = lr_pairs.rename(columns={"ligand": "source", "receptor": "target"})
        with _gpu_X(adata):
            rsc.gr.ligrec(
                adata,
                cluster_key=groupby,
                interactions=interactions_df,
                n_perms=n_perms,
                threshold=threshold,
                key_added=key_added,
                use_raw=False,
            )
        # Remap metadata from rsc DataFrame to astrolabe dict format
        result = adata.uns[key_added]
        adata.uns[key_added] = {
            "means": result["means"],
            "pvalues": result["pvalues"],
            "metadata": {
                "groupby": groupby, "n_perms": n_perms, "threshold": threshold,
                "spatial_restricted": False,
                "note": (
                    "GPU path: non-spatial cluster-mean statistic, not "
                    "restricted to spatial-graph edges. See ab.tl.ligrec docs."
                ),
            },
        }
        return

    # Filter to genes present in adata
    present = set(adata.var_names)
    valid = lr_pairs[
        lr_pairs["ligand"].isin(present) & lr_pairs["receptor"].isin(present)
    ].reset_index(drop=True)
    if len(valid) == 0:
        raise ValueError("No LR pairs have both genes in adata.var_names")

    n_dropped = len(lr_pairs) - len(valid)
    if n_dropped:
        log.info("ligrec: dropped %d LR pairs with genes not in var_names.", n_dropped)

    codes, categories = _category_codes(adata, groupby)
    n_types = len(categories)

    # Edge arrays
    edge_src, edge_dst = _edge_arrays(adata, neighbors_key)
    n_edges = len(edge_src)

    # Gene indices
    lig_idx = np.array([adata.var_names.get_loc(g) for g in valid["ligand"]])
    rec_idx = np.array([adata.var_names.get_loc(g) for g in valid["receptor"]])
    n_lr = len(valid)

    # Precompute per-edge ligand and receptor expression (n_edges × n_lr each)
    mat = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=np.float32)

    lig_expr = mat[np.ix_(edge_src, lig_idx)].astype(np.float64)   # (n_edges, n_lr)
    rec_expr = mat[np.ix_(edge_dst, rec_idx)].astype(np.float64)    # (n_edges, n_lr)

    # Cell-type pair flat index for each edge
    src_codes = codes[edge_src]
    dst_codes = codes[edge_dst]
    n_ct = n_types * n_types
    ct_idx = src_codes * n_types + dst_codes     # (n_edges,)

    def _mean_component_by_ct(values: np.ndarray, ct_indices: np.ndarray) -> np.ndarray:
        """Compute (n_ct, n_lr) mean of one component (L or R) by ct group."""
        scores = np.zeros((n_ct, n_lr), dtype=np.float64)
        counts = np.zeros(n_ct, dtype=np.float64)
        np.add.at(scores, ct_indices, values)
        np.add.at(counts, ct_indices, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(counts[:, None] > 0, scores / counts[:, None], 0.0)

    def _mean_by_ct(ct_indices: np.ndarray) -> np.ndarray:
        """CellPhoneDB-style score: (mean(L) + mean(R)) / 2 within each ct group."""
        mean_l = _mean_component_by_ct(lig_expr, ct_indices)
        mean_r = _mean_component_by_ct(rec_expr, ct_indices)
        return (mean_l + mean_r) / 2.0

    obs_scores = _mean_by_ct(ct_idx)  # (n_ct, n_lr)

    # Permutations — each draws from an independent RNG substream so the
    # loop is safe to parallelize.
    def _one_perm(child_rng: "np.random.Generator") -> np.ndarray:
        perm_codes = codes.copy()
        child_rng.shuffle(perm_codes)
        perm_ct_idx = perm_codes[edge_src] * n_types + perm_codes[edge_dst]
        perm_s = _mean_by_ct(perm_ct_idx)
        return (perm_s >= obs_scores).astype(np.float64)

    child_rngs = _spawn_rngs(seed, n_perms)
    perm_ge_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_perm)(child_rng) for child_rng in child_rngs
    )
    perm_ge = np.sum(perm_ge_list, axis=0) if perm_ge_list else np.zeros((n_ct, n_lr), dtype=np.float64)

    # (k+1)/(n+1) floor -- an exact 0 isn't a valid Monte Carlo p-value
    # estimate and breaks downstream -log10(p)/FDR steps.
    pvalues = (perm_ge + 1.0) / (n_perms + 1.0)  # (n_ct, n_lr)

    # Build DataFrames
    lr_index = pd.MultiIndex.from_arrays(
        [valid["ligand"], valid["receptor"]], names=["ligand", "receptor"]
    )
    ct_index = pd.MultiIndex.from_tuples(
        [(categories[a], categories[b]) for a in range(n_types) for b in range(n_types)],
        names=["sender", "receiver"],
    )

    means_df = pd.DataFrame(obs_scores.T, index=lr_index, columns=ct_index)
    pval_df = pd.DataFrame(pvalues.T, index=lr_index, columns=ct_index)

    n_sig = int((pval_df < threshold).values.sum())
    adata.uns[key_added] = {
        "means": means_df,
        "pvalues": pval_df,
        "metadata": {
            "groupby": groupby, "n_perms": n_perms, "threshold": threshold,
            "spatial_restricted": True,
            "neighbors_key": neighbors_key,
        },
    }
    log.info(
        "ligrec: %d LR pairs × %d cell-type pairs; %d significant at p < %.2f.",
        n_lr,
        n_ct,
        n_sig,
        threshold,
    )


@with_adata(mirror_uns=from_kwarg("key_added"))
def composition(
    adata: ad.AnnData,
    *,
    groupby: str,
    region_by: str,
    normalize: bool = True,
    key_added: str = "composition",
) -> pd.DataFrame:
    """Cell-type composition per spatial domain.

    Computes the fraction (or count) of each cell type in each spatial region
    and stores the result in ``adata.uns[key_added]``.  The most common use
    is to ask "what fraction of each spatial cluster is each annotated cell
    type?" after running :func:`~astrolabe.tl.cluster` and
    :func:`~astrolabe.tl.annotate_clusters`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column with cell-type / annotation labels (e.g.
        ``'leiden_annotation'``).
    region_by
        ``obs`` column with spatial domain / cluster labels (e.g.
        ``'leiden'``).
    normalize
        If ``True`` (default), values are fractions summing to 1 per region.
        If ``False``, values are raw cell counts.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        ``(n_regions × n_cell_types)`` DataFrame indexed by region.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> ab.tl.cluster(mdata, resolution=0.5, neighbors_key="spatial_neighbors")
    >>> ab.tl.annotate_clusters(mdata, {"0": "Epithelial", "1": "T cell"})
    >>> df = ab.tl.composition(mdata, groupby="leiden_annotation", region_by="leiden")
    >>> df  # rows = spatial clusters, columns = cell types, values = fraction
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"obs['{groupby}'] not found")
    if region_by not in adata.obs.columns:
        raise KeyError(f"obs['{region_by}'] not found")
    if groupby == region_by:
        raise ValueError(
            f"groupby and region_by must be different obs columns; both are '{groupby}'"
        )

    counts = (
        adata.obs
        .groupby([region_by, groupby], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    counts.columns.name = groupby
    counts.index.name = region_by

    if normalize:
        row_sums = counts.sum(axis=1)
        result = counts.div(row_sums.replace(0, 1), axis=0)
    else:
        result = counts.astype(float)

    adata.uns[key_added] = result
    log.info(
        "composition: %d regions × %d cell types",
        result.shape[0], result.shape[1],
    )
    return result
