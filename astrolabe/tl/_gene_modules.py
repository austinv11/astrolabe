"""
Spatial co-expression gene modules.

Builds a gene × gene spatial correlation matrix (via spatial-lag Pearson
correlation or vectorised bivariate Moran's I), then clusters genes into
modules using Leiden community detection (default; the module count is
determined by the data), or hierarchical/spectral clustering into a fixed
``n_modules``, with a per-module permutation null for whether each module's
spatial co-expression exceeds chance.

This is a **novel astrolabe method** — a pragmatic combination of standard
ingredients (bivariate Moran's I / spatial-lag correlation, hierarchical
clustering, permutation testing) with no single published analog. Related
published approaches include hotspot (DeTomaso & Yosef 2021) and Giotto's
spatial co-expression modules (Dries et al. 2021).
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._utils import _spawn_rngs, from_kwarg, with_adata

__all__ = ["spatial_gene_modules"]

log = logging.getLogger(__name__)


def _row_normalize(W: sp.spmatrix) -> sp.csr_matrix:
    W = W.tocsr().astype(np.float64)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    # avoid division by zero for isolated cells
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    return D_inv @ W


def _spatial_corr_matrix(
    Z: np.ndarray,
    W_norm: sp.csr_matrix,
    method: str,
) -> np.ndarray:
    """Return (n_genes, n_genes) spatial co-expression matrix.

    The diagonal is left as computed (``spatial_lag``: Pearson self-corr = 1;
    ``moran_bv``: I(i,i) = gene i's own univariate Moran's I, *not* 1) —
    any diagonal convention needed for clustering is applied in
    :func:`_cluster_genes`, not here.
    """
    WZ = W_norm @ Z  # (n_cells, n_genes) — spatial lag of each gene

    if method == "spatial_lag":
        # Pearson correlation of the spatially-lagged expression vectors
        corr = np.corrcoef(WZ.T)  # (n_genes, n_genes)
        return np.nan_to_num(corr, nan=0.0)

    else:  # moran_bv
        # I_bi(i,j) = z_i^T W z_j / n  (for row-normalized W, S0 = n)
        n = Z.shape[0]
        corr = (Z.T @ WZ) / n  # (n_genes, n_genes)
        corr = (corr + corr.T) / 2  # symmetrise floating-point asymmetry
        # Deliberately NOT clipped to [-1, 1]: bivariate Moran's I is not a
        # correlation coefficient and is not bounded by ±1 — clipping would
        # silently truncate exactly the strongest co-expression pairs.
        return corr


def _cluster_genes(
    corr: np.ndarray,
    n_modules: int,
    method: str,
    *,
    resolution: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Return integer module labels (0-indexed) of length n_genes.

    The self-similarity convention lives here: distance diagonal is forced
    to 0 (hierarchical) / affinity diagonal to 1 (spectral) / zeroed with
    no self-loops (leiden) regardless of what the correlation matrix's own
    diagonal holds (for ``moran_bv`` it is each gene's univariate Moran's I,
    not 1). ``n_modules`` is ignored by ``'leiden'`` (the count is
    data-determined via ``resolution``); ``resolution``/``seed`` are ignored
    by the other two methods.
    """
    if method == "leiden":
        import igraph as ig
        import leidenalg

        n = corr.shape[0]
        # Modularity is only defined for nonnegative edge weights: negative
        # correlations are dropped (no edge), but positive moran_bv values
        # stay unclipped above 0 — same stance as _spatial_corr_matrix.
        A = np.where(corr > 0, corr, 0.0)
        np.fill_diagonal(A, 0.0)
        # kNN-sparsify (union-symmetrized): the permutation null rebuilds
        # this graph per perm, and modularity's resolution limit blurs block
        # structure on near-complete weighted graphs. Small panels
        # (k = n - 1) degenerate to the full positive graph.
        k = min(15, n - 1)
        edges: set[tuple[int, int]] = set()
        if k > 0:
            top_k = np.argpartition(A, -k, axis=1)[:, -k:]
            for i in range(n):
                for j in top_k[i]:
                    if A[i, j] > 0:
                        edges.add((min(i, int(j)), max(i, int(j))))
        edge_list = sorted(edges)
        weights = [float(A[i, j]) for i, j in edge_list]
        g = ig.Graph(n=n, edges=edge_list)
        # Direct leidenalg, NOT sc.tl.leiden: scanpy's igraph flavor seeds
        # igraph's process-global RNG, which is not deterministic inside the
        # thread-backed permutation pool. leidenalg takes the seed per call.
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights if edge_list else None,
            resolution_parameter=resolution,
            seed=seed,
            n_iterations=2,
        )
        raw = np.asarray(part.membership)
        # Relabel by descending module size (ties: first-gene order) so
        # module 0 is always the largest — stable, interpretable ids.
        uniq, counts = np.unique(raw, return_counts=True)
        order = uniq[np.lexsort((uniq, -counts))]
        remap = np.empty(len(uniq), dtype=np.int64)
        remap[order] = np.arange(len(uniq))
        labels = remap[raw]
    elif method == "hierarchical":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        # moran_bv values can exceed ±1 (see _spatial_corr_matrix); a
        # negative "distance" is meaningless, so the distance transform is
        # clipped to [0, 1] even though the correlation matrix itself is not.
        dist = np.clip((1.0 - corr) / 2.0, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        # 'average' (UPGMA), not 'ward': Ward's variance-minimization
        # objective is only defined for Euclidean distances, and correlation
        # distance is not Euclidean — Ward on a precomputed non-Euclidean
        # matrix silently optimizes an invalid objective. Average linkage is
        # well-defined for arbitrary dissimilarities.
        Z_link = linkage(condensed, method="average")
        labels = fcluster(Z_link, t=n_modules, criterion="maxclust") - 1
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        affinity = np.clip((corr + 1.0) / 2.0, 0.0, None)
        np.fill_diagonal(affinity, 1.0)
        sc = SpectralClustering(
            n_clusters=n_modules,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        labels = sc.fit_predict(affinity)
    else:
        raise ValueError(
            f"Unknown clustering method {method!r}; "
            "expected 'leiden', 'hierarchical', or 'spectral'."
        )

    return labels.astype(np.int32)


def _mean_intra_module_corr(
    corr: np.ndarray, labels: np.ndarray, modules: list[int]
) -> np.ndarray:
    """Mean off-diagonal correlation within each module (NaN for singletons)."""
    out = np.full(len(modules), np.nan, dtype=np.float64)
    for k, m in enumerate(modules):
        g = np.where(labels == m)[0]
        if len(g) < 2:
            continue
        block = corr[np.ix_(g, g)]
        out[k] = (block.sum() - np.trace(block)) / (len(g) * (len(g) - 1))
    return out


@with_adata(mirror_uns=from_kwarg("key_added"))
def spatial_gene_modules(
    adata: ad.AnnData,
    *,
    var_names: list[str] | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_modules: int = 10,
    resolution: float = 1.0,
    correlation: Literal["spatial_lag", "moran_bv"] = "spatial_lag",
    clustering: Literal["leiden", "hierarchical", "spectral"] = "leiden",
    score_cells: bool = True,
    n_perms: int = 100,
    seed: int = 0,
    n_jobs: int = 1,
    layer: str | None = None,
    max_working_bytes: int = 512*1024**2,
    key_added: str = "spatial_module",
) -> pd.DataFrame:
    """Cluster genes into spatial co-expression modules.

    Builds a gene × gene spatial correlation matrix and clusters it to find
    groups of genes that vary together across space.  Optionally scores each
    cell for each module, and tests each module's spatial co-expression
    against a permutation null.

    .. note::
        Novel astrolabe method (no single published analog — see the module
        docstring). ``clustering='leiden'`` (the default) determines the
        module count from the data via ``resolution`` — a genuine
        module-count selection. ``'hierarchical'``/``'spectral'`` instead
        always return exactly ``n_modules`` groups whether or not real
        structure exists for all of them, forcing genes with no real
        program into a catch-all. Either way, the per-module permutation
        p-value (``module_stats``, below) is what distinguishes genuinely
        spatially co-expressed modules from partitions of noise.

    **Permutation null** (``n_perms > 0``, default 100): cells' rows are
    jointly permuted — every gene's values move together, so gene–gene
    (non-spatial) correlation is preserved while the spatial arrangement is
    destroyed. Each permutation repeats the *entire* pipeline (correlation
    matrix **and** re-clustering), and the observed module with the r-th
    largest mean intra-module correlation is compared against the null
    distribution of the r-th largest statistic
    (``p = (1 + #{perm ≥ obs}) / (1 + n_valid_perms)``, one-sided).
    Re-clustering inside the null matters: the observed modules were
    *selected* to be the most-correlated gene groupings, so comparing them
    against un-selected null gene sets would be anti-conservative even on
    pure noise; the rank-matched comparison keeps the p-values calibrated.

    Results are stored in:

    - ``adata.uns[key_added]`` — dict with ``'gene_modules'`` DataFrame,
      ``'module_stats'`` DataFrame (per-module ``n_genes``,
      ``mean_intra_corr``, ``pval``; ``pval`` is NaN when ``n_perms=0`` or
      for singleton modules) and ``'params'``. Mirrored to
      ``mdata.uns[key_added][<modality>]`` when called on a MuData.
    - ``adata.obsm[key_added + '_scores']`` — ``(n_cells, n_modules_found)``
      float32 array of per-module scores, one column per module actually
      found (only when ``score_cells=True``; adata-resident only, not
      mirrored).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with spatial
        connectivity in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    var_names
        Genes to include.  ``None`` uses all genes in ``adata.var``.
    neighbors_key
        Prefix for the spatial graph in ``obsp``
        (``{neighbors_key}_connectivities``).
    n_modules
        Number of gene modules for ``clustering='hierarchical'`` or
        ``'spectral'``.  Ignored by ``'leiden'`` (the default), which
        determines the count from the data.
    resolution
        Leiden resolution parameter (``clustering='leiden'`` only): larger
        values yield more, smaller modules.  Ignored by
        ``'hierarchical'``/``'spectral'``.
    correlation
        Method for computing the gene × gene spatial correlation matrix:

        ``'spatial_lag'``
            Pearson correlation of each gene's spatially-lagged expression
            vector (W @ x) with every other gene's lag.  Fast and intuitive;
            genes that are spatially co-expressed with similar neighbors will
            cluster together.  Values lie in [−1, 1].
        ``'moran_bv'``
            Vectorised bivariate Moran's I for every gene pair:
            I(i,j) = zᵢᵀ W z_j / n (row-normalized W).  More formally
            grounded in spatial statistics; captures the degree to which
            high values of gene i coincide with spatially-lagged high values
            of gene j.  **Not bounded to [−1, 1]** (it is not a correlation
            coefficient); values are kept unclipped, and a log warning is
            emitted when |I| > 1 is observed.
    clustering
        Algorithm for clustering the gene × gene correlation matrix:

        ``'leiden'`` (default)
            Leiden community detection (RBConfiguration modularity, via
            ``leidenalg``) on a kNN-sparsified (k = 15) gene–gene graph
            whose edge weights are the **positive** correlations (negative
            values carry no edge — modularity requires nonnegative weights;
            positive ``moran_bv`` values stay unclipped).  The number of
            modules is determined by the data, controlled by
            ``resolution``.  Seeded per call, so the permutation null stays
            deterministic across ``n_jobs``.
        ``'hierarchical'``
            Average (UPGMA) linkage on distance = (1 − corr) / 2,
            clipped to [0, 1].  Deterministic.  (Not Ward: Ward's objective
            is only valid for Euclidean distances, which correlation
            distance is not.)
        ``'spectral'``
            Spectral clustering with affinity = (corr + 1) / 2.
            Requires ``scikit-learn``.
    score_cells
        Compute per-cell module scores as the mean of the module genes'
        **z-scored** expression (the same standardized quantity the modules
        were defined on — a raw-expression mean would be dominated by the
        module's highest-magnitude gene) and store the result in
        ``obsm[key_added + '_scores']``.
    n_perms
        Permutations for the per-module null (default 100 → minimum
        attainable p = 1/101).  ``0`` skips the null (``module_stats.pval``
        all-NaN).
    seed
        Seed for the permutation RNG.
    n_jobs
        Parallel workers for the permutation loop
        (:class:`joblib.Parallel`, thread-backed). ``1`` = sequential,
        ``-1`` = all cores.
    layer
        Expression layer to use.  ``None`` uses ``adata.X``.
    key_added
        Key prefix for ``uns`` and ``obsm``.

    Returns
    -------
    pd.DataFrame
        ``(n_genes_tested, 2)`` frame with columns ``gene`` and ``module``
        (integer 0-indexed module IDs), sorted by module then gene name.
        Undetected/zero-variance genes get ``module = -1`` ("excluded", not
        a real module) rather than being clustered at all: with no spatial
        signal to standardize, they'd otherwise all collapse to an
        identical all-zero profile and cluster together into a module
        indistinguishable from a genuine co-expression module.  (For
        ``'hierarchical'``/``'spectral'`` the remaining genes are always
        split into exactly ``n_modules`` groups; ``'leiden'`` returns
        however many modules the data support.)

    Examples
    --------
    >>> df = ab.tl.spatial_gene_modules(mdata)        # leiden, data-determined k
    >>> df[df.module == 0]["gene"].tolist()           # genes in module 0
    >>> mdata["xenium"].uns["spatial_module"]["module_stats"]  # per-module p
    >>> mdata["xenium"].obsm["spatial_module_scores"] # cell scores (n×k)
    >>> ab.tl.spatial_gene_modules(mdata, clustering="hierarchical",
    ...                            n_modules=8)       # fixed-k alternative
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )
    if n_perms < 0:
        raise ValueError(f"n_perms must be >= 0; got {n_perms}")

    # ── Resolve gene subset ────────────────────────────────────────────────────
    if var_names is not None:
        missing = [g for g in var_names if g not in adata.var_names]
        if missing:
            raise KeyError(f"var_names not found in adata: {missing[:5]}")
        gene_idx = np.array([adata.var_names.get_loc(g) for g in var_names])
        gene_names = list(var_names)
    else:
        gene_idx = np.arange(adata.n_vars)
        gene_names = list(adata.var_names)

    n_genes = len(gene_names)
    from astrolabe._coverage import _require_measured
    _require_measured(adata, gene_names)
    from astrolabe._memory import _budget
    from joblib import effective_n_jobs
    from astrolabe.tl._spatial_stats import _permutation_blocks
    blocks, library_key = _permutation_blocks(adata, neighbors_key)
    workers = effective_n_jobs(n_jobs) if n_perms else 1
    required = 8*adata.n_obs*n_genes*(4+3*workers) + 8*n_genes*n_genes*(4+4*workers)
    _budget(required, max_working_bytes, "Dense gene-module correlation and permutation workers")
    n_modules_eff = min(n_modules, n_genes)
    if clustering == "leiden":
        n_modules_eff = 0  # unused by leiden; count is data-determined
        log.info(
            "spatial_gene_modules: clustering='leiden' — n_modules is ignored; "
            "module count is determined by resolution=%.2f",
            resolution,
        )
    else:
        if resolution != 1.0:
            log.info(
                "spatial_gene_modules: resolution is ignored by "
                "clustering=%r (leiden-only parameter)", clustering,
            )
        if n_modules_eff < n_modules:
            log.warning(
                "n_modules=%d reduced to %d (only %d genes available)",
                n_modules, n_modules_eff, n_genes,
            )

    # ── Expression matrix ──────────────────────────────────────────────────────
    X = adata.layers[layer] if layer is not None else adata.X
    X = X[:, gene_idx]
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)  # (n_cells, n_genes)

    # Standardise: zero mean, unit variance per gene. Zero-variance
    # (undetected/constant) genes are excluded from clustering entirely,
    # below -- standardizing them here would give every one of them an
    # identical all-zero Z profile.
    mu = X.mean(axis=0)
    sigma_raw = X.std(axis=0)
    zero_var_mask = sigma_raw == 0
    sigma = np.where(zero_var_mask, 1.0, sigma_raw)
    Z = (X - mu) / sigma

    if zero_var_mask.any():
        log.warning(
            "spatial_gene_modules: %d of %d genes have zero variance "
            "(undetected or constant) -- excluded from clustering (module=-1) "
            "rather than pooled into a phantom module with each other.",
            int(zero_var_mask.sum()), n_genes,
        )

    valid_idx = np.where(~zero_var_mask)[0]
    n_valid = len(valid_idx)
    if clustering != "leiden":
        n_modules_eff = min(n_modules_eff, n_valid) if n_valid > 0 else 0

    # ── Spatial graph ──────────────────────────────────────────────────────────
    W_norm = _row_normalize(adata.obsp[conn_key])

    module_labels = np.full(n_genes, -1, dtype=np.int32)
    module_stats = pd.DataFrame(
        columns=["module", "n_genes", "mean_intra_corr", "pval"]
    )
    n_modules_found = 0
    if n_valid > 0 and (clustering == "leiden" or n_modules_eff > 0):
        Z_valid = Z[:, valid_idx]

        # ── Gene × gene spatial correlation (valid genes only) ─────────────────
        log.info(
            "spatial_gene_modules: building %s correlation matrix (%d × %d)",
            correlation, n_valid, n_valid,
        )
        corr = _spatial_corr_matrix(Z_valid, W_norm, correlation)
        if correlation == "moran_bv":
            off_diag = corr[~np.eye(n_valid, dtype=bool)] if n_valid > 1 else corr
            max_abs = float(np.abs(off_diag).max()) if off_diag.size else 0.0
            if max_abs > 1.0:
                log.warning(
                    "spatial_gene_modules: bivariate Moran's I values up to "
                    "|I|=%.3f observed (moran_bv is not bounded to [-1, 1]); "
                    "values are kept unclipped.",
                    max_abs,
                )

        # ── Cluster genes ────────────────────────────────────────────────────
        if clustering == "leiden":
            log.info(
                "spatial_gene_modules: clustering (leiden, resolution=%.2f)",
                resolution,
            )
        else:
            log.info(
                "spatial_gene_modules: clustering (%s, k=%d)",
                clustering, n_modules_eff,
            )
        valid_labels = _cluster_genes(
            corr, n_modules_eff, clustering, resolution=resolution, seed=seed
        )
        module_labels[valid_idx] = valid_labels
        n_modules_found = int(valid_labels.max()) + 1

        # ── Per-module permutation null ──────────────────────────────────────
        modules = sorted(set(valid_labels.tolist()))
        obs_stat = _mean_intra_module_corr(corr, valid_labels, modules)

        pvals = np.full(len(modules), np.nan, dtype=np.float64)
        if n_perms > 0:
            log.info(
                "spatial_gene_modules: permutation null (%d perms, %d modules)",
                n_perms, len(modules),
            )
            n_ranks = len(modules)

            def _one_perm(rng: np.random.Generator) -> np.ndarray:
                # Joint row permutation: gene-gene correlation preserved,
                # spatial arrangement destroyed. The permutation repeats the
                # *entire* pipeline -- including re-clustering -- so the null
                # distribution reflects the same "pick the most-correlated
                # gene groupings" selection step the observed statistic went
                # through. Scoring the observed (selection-optimized) modules
                # against un-selected null gene sets would be
                # anti-conservative even on pure noise.
                # Per-perm leiden seed drawn from the child Generator BEFORE
                # the permutation: each child stream from _spawn_rngs is
                # independent and consumed in a fixed order, so the seed is
                # identical regardless of n_jobs/thread scheduling.
                perm_seed = int(rng.integers(0, 2**31 - 1))
                perm = np.arange(Z_valid.shape[0])
                for block in blocks:
                    perm[block] = rng.permutation(block)
                corr_p = _spatial_corr_matrix(Z_valid[perm], W_norm, correlation)
                labels_p = _cluster_genes(
                    corr_p, n_modules_eff, clustering,
                    resolution=resolution, seed=perm_seed,
                )
                modules_p = sorted(set(labels_p.tolist()))
                stats_p = _mean_intra_module_corr(corr_p, labels_p, modules_p)
                finite = np.sort(stats_p[np.isfinite(stats_p)])[::-1]
                out = np.full(n_ranks, np.nan, dtype=np.float64)
                out[: min(n_ranks, len(finite))] = finite[:n_ranks]
                return out

            child_rngs = _spawn_rngs(seed, n_perms)
            perm_stats = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_perm)(child_rng) for child_rng in child_rngs
            )
            perm_arr = np.stack(perm_stats)  # (n_perms, n_ranks), desc-sorted

            # Rank-matched comparison: the observed module with the r-th
            # largest intra-module correlation is compared against the null
            # distribution of the r-th largest statistic across permutations
            # (Westfall-Young-style step-down against ordered statistics).
            finite_obs = np.where(np.isfinite(obs_stat))[0]
            order = finite_obs[np.argsort(obs_stat[finite_obs])[::-1]]
            for rank, mi in enumerate(order):
                null_r = perm_arr[:, rank]
                valid = np.isfinite(null_r)
                n_ge = int((null_r[valid] >= obs_stat[mi]).sum())
                # (k+1)/(n+1) -- an exact 0 isn't a valid Monte Carlo p
                # estimate.
                pvals[mi] = (n_ge + 1.0) / (int(valid.sum()) + 1.0)

        module_stats = pd.DataFrame(
            {
                "module": modules,
                "n_genes": [int((valid_labels == m).sum()) for m in modules],
                "mean_intra_corr": obs_stat,
                "pval": pvals,
            }
        )

    df = pd.DataFrame({"gene": gene_names, "module": module_labels})
    df = df.sort_values(["module", "gene"]).reset_index(drop=True)

    # ── Cell scores ────────────────────────────────────────────────────────────
    if score_cells:
        # Mean of the module genes' *z-scored* expression -- the same
        # standardized quantity the modules were defined on. A raw-expression
        # mean would be dominated by whichever module gene has the largest
        # magnitude rather than reflecting the shared spatial pattern.
        scores = np.zeros((adata.n_obs, n_modules_found), dtype=np.float32)
        for m in range(n_modules_found):
            m_genes = np.where(module_labels == m)[0]
            if len(m_genes) == 0:
                continue
            scores[:, m] = Z[:, m_genes].mean(axis=1).astype(np.float32)

        adata.obsm[key_added + "_scores"] = scores
        log.info("spatial_gene_modules: scores stored in obsm['%s_scores']", key_added)

    adata.uns[key_added] = {
        "gene_modules": df,
        "module_stats": module_stats,
        "params": {
            "n_modules": n_modules_found,
            "n_modules_requested": n_modules if clustering != "leiden" else None,
            "resolution": resolution if clustering == "leiden" else None,
            "correlation": correlation,
            "clustering": clustering,
            "neighbors_key": neighbors_key,
            "layer": layer,
            "n_perms": n_perms,
            "seed": seed,
        },
    }

    log.info(
        "spatial_gene_modules: %d genes → %d modules", n_genes, n_modules_found
    )
    return df
