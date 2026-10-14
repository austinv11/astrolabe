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
    if adata.obs[groupby].isna().any():
        raise ValueError(f"obs[{groupby!r}] contains missing category labels")
    labels = adata.obs[groupby].astype("category")
    # Pair indices multiply codes by the category count; int8 overflows at
    # twelve categories even though Pandas still uses int8 for those labels.
    return labels.cat.codes.to_numpy(dtype=np.intp), labels.cat.categories.tolist()


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
    """Neighborhood enrichment permutation test (Schapiro et al. 2017; Palla et al. 2022).

    For each pair of cell types (A, B), asks whether cells of type A and type B
    are connected in the spatial graph more (or less) than expected by chance.
    Labels are permuted ``n_perms`` times to build a null distribution; the
    result is reported as a z-score — the same statistic as histoCAT's
    neighborhood analysis (Schapiro et al. 2017) and
    ``squidpy.gr.nhood_enrichment`` (Palla et al. 2022).

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
    """Co-occurrence probability ratio across spatial scales (Palla et al. 2022).

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
        1-D array of explicit distance thresholds in µm (must be positive
        and strictly increasing).
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``occ``
        (n_types × n_types × n_intervals) float array of co-occurrence
        ratios. Rings that contain no neighbor pairs at all (or no
        neighbors of the conditioning type) are ``NaN`` — "no data at this
        scale", matching ``squidpy.gr.co_occurrence`` — never ``0.0``,
        which would falsely read as perfect avoidance.
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

    # Resolve + validate thresholds once, shared by the CPU and GPU paths.
    # Unsorted thresholds would silently produce negative-width "rings"
    # (lo > hi -> empty masks) rather than an error.
    if isinstance(interval, tuple) and len(interval) == 3:
        thresholds = np.linspace(float(interval[0]), float(interval[1]), int(interval[2]))
    else:
        thresholds = np.asarray(interval, dtype=np.float64).ravel()
    if len(thresholds) == 0:
        raise ValueError("interval must contain at least one distance threshold")
    if thresholds[0] <= 0 or (len(thresholds) > 1 and np.any(np.diff(thresholds) <= 0)):
        raise ValueError(
            f"interval thresholds must be positive and strictly increasing; got {thresholds!r}"
        )

    if gpu_enabled():
        rsc = _require_rapids()
        # rsc/squidpy require a Categorical cluster column; cast up front so
        # a plain string column doesn't fail deep inside rsc.
        if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
            adata.obs[groupby] = adata.obs[groupby].astype("category")
        _, categories = _category_codes(adata, groupby)
        # rsc/squidpy treat `interval` as n bin *edges* -> n-1 rings, whereas
        # this function's own convention (and the CPU path) treats each of
        # the n thresholds as the closing edge of a ring (the first ring
        # being [0, thresholds[0]]). Prepending 0.0 makes squidpy's n+1
        # edges produce exactly our n rings; storing interval[1:] (the ring
        # *closing* edges) keeps occ's last axis and the stored thresholds
        # the same length -- previously the stored interval had one more
        # entry than occ had bins, breaking pl.co_occurrence_plot on every
        # GPU result, and distances below thresholds[0] were silently
        # dropped instead of forming the first ring.
        rsc_interval = np.concatenate([[0.0], thresholds])
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
            "interval": np.asarray(result["interval"])[1:],
            "groups": list(categories),
        }
        return

    from scipy.spatial import KDTree

    codes, categories = _category_codes(adata, groupby)
    n_types = len(categories)

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]

    # Query all unique pairs within the maximum threshold
    tree = KDTree(coords)
    pairs = tree.query_pairs(float(thresholds[-1]), output_type="ndarray")  # (n_pairs, 2)

    # NaN, not 0.0: a ring with no observed neighbor pairs (or none of the
    # conditioning type) carries no information at that scale. Filling 0.0
    # instead read as "perfect avoidance" (ratio 0) -- squidpy yields NaN
    # for the same case, and pl.co_occurrence_plot gaps NaN correctly.
    occ = np.full((n_types, n_types, len(thresholds)), np.nan, dtype=np.float64)

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
                # NaN (not 0.0) wherever a ring holds no information: no
                # pairs at all in the ring, no neighbors of the conditioning
                # type A (row total 0), or a marginal of 0 for type B.
                marginal_local = np.where(ring_total > 0, row_totals / ring_total, np.nan)
                cond = np.where(
                    row_totals[:, None] > 0,
                    count_mat / row_totals[:, None],
                    np.nan,
                )
                occ[:, :, k] = np.where(
                    marginal_local[None, :] > 0, cond / marginal_local[None, :], np.nan
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
                # ddof=1: the standard error of a mean uses the *sample*
                # standard deviation (Bessel-corrected), not the population
                # one -- ddof=0 understates SE, most at small bin counts.
                se_arr[k] = vals.std(axis=0, ddof=1) / np.sqrt(n_cells_arr[k])
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
def ligrec(adata, groupby, lr_pairs, *, neighbors_key="spatial_neighbors", layer=None,
           n_perms=100, seed=None, expr_frac_threshold=.1, report_alpha=.05,
           corr_method="fdr_bh", n_jobs=1, min_valid_perms=20, library_key="library_id",
           max_scratch_bytes=256*1024**2, key_added="ligrec"):
    """Complex-aware spatial-edge ligand/receptor association.

    Uses minimum-subunit expression and within-library label permutations.
    Missing subunits, expression gates and too few valid null scores are
    explicitly untested. ``n_valid`` and ``status`` accompany p-values;
    zero-edge permutations are excluded from the denominator. Scratch space
    is bounded by chunked edge scoring; full expression is never densified.
    GPU inputs use the equivalent CPU statistic rather than changing the null.
    See ``tl.import_ligrec_resource`` for explicit complex/resource metadata.
    """
    from astrolabe.tl._communication import _ligrec
    return _ligrec(adata, groupby, lr_pairs, neighbors_key=neighbors_key, layer=layer,
                   n_perms=n_perms, seed=seed, expr_frac_threshold=expr_frac_threshold,
                   report_alpha=report_alpha, corr_method=corr_method, n_jobs=n_jobs,
                   min_valid_perms=min_valid_perms, library_key=library_key,
                   max_scratch_bytes=max_scratch_bytes, key_added=key_added)


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

    # observed=False: declared-but-empty categories keep their row/column
    # (as zeros) instead of being silently dropped -- matching
    # centrality_scores, which likewise keeps a NaN row for every declared
    # category. Routine after subsetting, since pandas/scanpy never prune
    # unused categories automatically.
    counts = (
        adata.obs
        .groupby([region_by, groupby], observed=False)
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
