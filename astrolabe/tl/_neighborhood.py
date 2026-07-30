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

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import from_kwarg, with_adata
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

    # Permutations — Welford's online algorithm avoids catastrophic
    # cancellation when computing variance via sum-of-squares minus mean².
    rng = np.random.default_rng(seed)
    perm_mean = np.zeros_like(obs_counts)
    perm_M2 = np.zeros_like(obs_counts)
    perm_codes = codes.copy()

    for i in range(1, n_perms + 1):
        rng.shuffle(perm_codes)
        Cp = _one_hot(perm_codes, n_types)
        m = (Cp.T @ adj @ Cp).toarray().astype(np.float64)
        delta = m - perm_mean
        perm_mean += delta / i
        perm_M2 += delta * (m - perm_mean)

    perm_std = np.sqrt(perm_M2 / n_perms)

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
        \\frac{P(\\text{type B within }d\\text{ of type A cell})}
             {P(\\text{type B globally})}

    A ratio > 1 indicates enriched co-occurrence at that scale; < 1 indicates
    avoidance.

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
    type_counts = np.bincount(codes, minlength=n_types).astype(np.float64)
    marginal = type_counts / n_cells  # P(type B globally)

    # Query all unique pairs within the maximum threshold
    tree = KDTree(coords)
    pairs = tree.query_pairs(float(thresholds[-1]), output_type="ndarray")  # (n_pairs, 2)

    occ = np.zeros((n_types, n_types, len(thresholds)), dtype=np.float64)

    if len(pairs) > 0:
        dists = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        src_codes = codes[pairs[:, 0]]
        dst_codes = codes[pairs[:, 1]]

        for k, d in enumerate(thresholds):
            mask = dists <= d
            s = src_codes[mask]
            t = dst_codes[mask]

            # Undirected: accumulate both (A→B) and (B→A)
            count_mat = np.zeros((n_types, n_types), dtype=np.float64)
            np.add.at(count_mat, (s, t), 1.0)
            np.add.at(count_mat, (t, s), 1.0)

            # Conditional prob P(B | A at d): divide by # type-A cells
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = np.where(
                    type_counts[:, None] > 0,
                    count_mat / type_counts[:, None],
                    0.0,
                )
                occ[:, :, k] = np.where(
                    marginal[None, :] > 0, cond / marginal[None, :], 0.0
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
    key_added: str = "centrality_scores",
) -> None:
    """Spatial graph centrality metrics aggregated per cell type.

    Computes three per-cell centrality metrics from the spatial graph and
    averages them within each cell type:

    - **degree_centrality** — fraction of other cells the cell is connected to;
      high = spatially well-connected hub.
    - **clustering** — fraction of a cell's neighbors that are also neighbors
      of each other; high = tightly clustered local neighborhood.
    - **closeness_centrality** — inverse mean shortest-path to all other cells;
      high = spatially central cell.  Skipped for datasets > 5 000 cells
      because the O(n²) computation is too slow.

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
    >>> ab.pl.centrality_scores_plot(mdata, 'leiden')
    """
    import networkx as nx

    codes, categories = _category_codes(adata, groupby)
    n = adata.n_obs

    adj = _get_conn_matrix(adata, neighbors_key).astype(bool)

    # Degree centrality: fast from sparse matrix
    degrees = np.asarray(adj.sum(axis=1)).ravel().astype(np.float64)
    degree_centrality = degrees / max(n - 1, 1)

    # Clustering coefficient via networkx (O(n * k²), fast for small k)
    G = nx.from_scipy_sparse_array(adj)
    clustering = np.fromiter(nx.clustering(G).values(), dtype=np.float64, count=n)

    # Closeness centrality: O(n²) — skip for large graphs
    _MAX_CLOSENESS = 5_000
    if n <= _MAX_CLOSENESS:
        closeness = np.fromiter(
            nx.closeness_centrality(G).values(), dtype=np.float64, count=n
        )
    else:
        log.warning(
            "centrality_scores: skipping closeness_centrality (n=%d > %d). "
            "Set a smaller subset or use degree/clustering only.",
            n,
            _MAX_CLOSENESS,
        )
        closeness = np.full(n, np.nan)

    cell_df = pd.DataFrame(
        {
            "degree_centrality": degree_centrality,
            "clustering": clustering,
            "closeness_centrality": closeness,
            groupby: adata.obs[groupby].to_numpy(),
        }
    )
    result = (
        cell_df.groupby(groupby)[
            ["degree_centrality", "clustering", "closeness_centrality"]
        ]
        .mean()
        .loc[categories]
    )

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
    bin_idx = np.clip(np.digitize(dists, bin_edges) - 1, 0, n_bins - 1)

    n_genes = len(gene_names)
    mean_arr = np.full((n_bins, n_genes), np.nan, dtype=np.float32)
    se_arr = np.full((n_bins, n_genes), np.nan, dtype=np.float32)
    n_cells_arr = np.zeros(n_bins, dtype=np.int64)

    for k in range(n_bins):
        mask = bin_idx == k
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
    key_added: str = "ligrec",
) -> None:
    """Spatially-contextualized ligand-receptor permutation test.

    For each LR pair (L, R) and each directed cell-type pair (sender A →
    receiver B), computes the mean of ``expression_L × expression_R`` over
    all spatial graph edges where the source is of type A and the target is
    of type B.  Cell-type labels are permuted ``n_perms`` times to produce
    a null distribution and an empirical p-value.

    By restricting to actual spatial neighbors the test identifies LR
    interactions that are plausible given the spatial architecture of the
    tissue — a more stringent criterion than bulk cell-type-level analysis.

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
    key_added
        ``uns`` key to write results into.

    Results stored in ``adata.uns[key_added]``
    ------------------------------------------
    Also mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    ``means``
        :class:`pandas.DataFrame` — rows = (ligand, receptor) MultiIndex,
        columns = (sender, receiver) MultiIndex.  Values are mean LR
        product scores over spatial edges.
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
            "metadata": {"groupby": groupby, "n_perms": n_perms, "threshold": threshold},
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

    # Precompute ligand × receptor products for every edge (n_edges × n_lr)
    mat = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(mat):
        mat = mat.toarray()
    mat = np.asarray(mat, dtype=np.float32)

    lig_expr = mat[np.ix_(edge_src, lig_idx)]   # (n_edges, n_lr)
    rec_expr = mat[np.ix_(edge_dst, rec_idx)]    # (n_edges, n_lr)
    products = lig_expr * rec_expr               # (n_edges, n_lr)

    # Cell-type pair flat index for each edge
    src_codes = codes[edge_src]
    dst_codes = codes[edge_dst]
    n_ct = n_types * n_types
    ct_idx = src_codes * n_types + dst_codes     # (n_edges,)

    def _mean_by_ct(ct_indices: np.ndarray) -> np.ndarray:
        """Compute (n_ct, n_lr) mean product matrix from edge ct assignments."""
        scores = np.zeros((n_ct, n_lr), dtype=np.float64)
        counts = np.zeros(n_ct, dtype=np.float64)
        np.add.at(scores, ct_indices, products.astype(np.float64))
        np.add.at(counts, ct_indices, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(counts[:, None] > 0, scores / counts[:, None], 0.0)

    obs_scores = _mean_by_ct(ct_idx)  # (n_ct, n_lr)

    # Permutations
    rng = np.random.default_rng(seed)
    perm_ge = np.zeros((n_ct, n_lr), dtype=np.float64)
    perm_codes = codes.copy()

    for _ in range(n_perms):
        rng.shuffle(perm_codes)
        perm_ct_idx = perm_codes[edge_src] * n_types + perm_codes[edge_dst]
        perm_s = _mean_by_ct(perm_ct_idx)
        perm_ge += (perm_s >= obs_scores).astype(np.float64)

    pvalues = perm_ge / n_perms  # (n_ct, n_lr)

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
        "metadata": {"groupby": groupby, "n_perms": n_perms, "threshold": threshold},
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
    >>> ab.tl.cluster(mdata, resolution=0.5)
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
