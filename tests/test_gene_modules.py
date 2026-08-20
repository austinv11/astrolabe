"""Tests for tl._gene_modules: phantom-module exclusion, planted
co-expression block recovery, average-linkage/unclipped moran_bv semantics,
z-scored module scores, and the per-module permutation null (rank-matched,
so it stays calibrated on pure noise)."""
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pytest

from astrolabe.tl._gene_modules import spatial_gene_modules


def _adata_with_undetected_genes(n_cells=200, n_real_genes=6, n_undetected=4, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 500, (n_cells, 2)).astype(np.float32)

    # Two real spatial patterns (2 groups of 3 genes each, correlated with a
    # smooth spatial gradient) + a block of undetected (all-zero) genes.
    grad1 = coords[:, 0] / 500.0
    grad2 = coords[:, 1] / 500.0
    real = np.column_stack([
        grad1 + rng.normal(0, 0.05, n_cells) for _ in range(n_real_genes // 2)
    ] + [
        grad2 + rng.normal(0, 0.05, n_cells) for _ in range(n_real_genes // 2)
    ])
    undetected = np.zeros((n_cells, n_undetected))
    X = np.column_stack([real, undetected]).astype(np.float32)
    gene_names = [f"real_{i}" for i in range(n_real_genes)] + [f"undetected_{i}" for i in range(n_undetected)]

    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obsm["spatial"] = coords

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=7)
    rows = np.repeat(np.arange(n_cells), 6)
    cols = idxs[:, 1:].ravel()
    conn = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
    adata.obsp["spatial_neighbors_connectivities"] = conn
    return adata


class TestSpatialGeneModulesPhantomExclusion:
    def test_undetected_genes_get_module_negative_one(self):
        # Regression: undetected/zero-variance genes used to be silently
        # pooled into a real cluster (fcluster with criterion="maxclust"
        # always returns exactly n_modules groups regardless of whether real
        # structure exists), indistinguishable from a genuine co-expression
        # module. They should now be excluded entirely (module=-1).
        # Runs under the leiden default: exclusion semantics are
        # clustering-independent.
        adata = _adata_with_undetected_genes()
        df = spatial_gene_modules(adata, score_cells=False)

        undetected_rows = df[df["gene"].str.startswith("undetected_")]
        real_rows = df[df["gene"].str.startswith("real_")]
        assert (undetected_rows["module"] == -1).all()
        assert (real_rows["module"] != -1).all()

    def test_real_genes_still_cluster_into_requested_modules(self):
        # Fixed-k semantics — inherently a hierarchical/spectral claim.
        adata = _adata_with_undetected_genes()
        df = spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=2, score_cells=False
        )
        real_rows = df[df["gene"].str.startswith("real_")]
        assert set(real_rows["module"].unique()) <= {0, 1}
        assert real_rows["module"].nunique() == 2

    def test_score_cells_ignores_excluded_genes(self):
        adata = _adata_with_undetected_genes()
        spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=2,
            score_cells=True, key_added="sgm",
        )
        scores = adata.obsm["sgm_scores"]
        assert scores.shape == (adata.n_obs, 2)
        assert np.isfinite(scores).all()

    def test_all_genes_undetected_returns_all_excluded_without_crashing(self):
        n = 50
        adata = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
        adata.var_names = ["g0", "g1", "g2"]
        rng = np.random.default_rng(0)
        adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
        conn = sp.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
        adata.obsp["spatial_neighbors_connectivities"] = conn

        # Default (leiden) path — also exercises the n_valid == 0 gate there.
        df = spatial_gene_modules(adata, score_cells=False)
        assert (df["module"] == -1).all()


def _knn_conn(coords, k=7):
    from scipy.spatial import cKDTree
    n = len(coords)
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = idxs[:, 1:].ravel()
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _planted_modules_adata(n_cells=300, noise=0.1, seed=0):
    """Three planted co-expression blocks of 4 genes each, tied to three
    distinct smooth spatial patterns (x-gradient, y-gradient, radial)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 500, (n_cells, 2)).astype(np.float32)
    patterns = [
        coords[:, 0] / 500.0,
        coords[:, 1] / 500.0,
        np.exp(-np.hypot(coords[:, 0] - 250.0, coords[:, 1] - 250.0) / 120.0),
    ]
    cols, names, truth = [], [], []
    for mi, pat in enumerate(patterns):
        for gi in range(4):
            cols.append(pat + rng.normal(0, noise, n_cells))
            names.append(f"mod{mi}_g{gi}")
            truth.append(mi)
    X = np.column_stack(cols).astype(np.float32)

    adata = ad.AnnData(X=X)
    adata.var_names = names
    adata.obsm["spatial"] = coords
    adata.obsp["spatial_neighbors_connectivities"] = _knn_conn(coords)
    return adata, np.asarray(truth)


def _noise_adata(n_cells=300, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 500, (n_cells, 2)).astype(np.float32)
    X = rng.normal(0, 1, (n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = coords
    adata.obsp["spatial_neighbors_connectivities"] = _knn_conn(coords)
    return adata


class TestSpatialGeneModulesPlanted:
    @pytest.mark.parametrize("correlation", ["spatial_lag", "moran_bv"])
    def test_planted_blocks_recovered(self, correlation):
        from sklearn.metrics import adjusted_rand_score

        adata, truth = _planted_modules_adata()
        df = spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=3,
            correlation=correlation, score_cells=False, n_perms=0,
        )
        recovered = df.set_index("gene").loc[adata.var_names, "module"].to_numpy()
        ari = adjusted_rand_score(truth, recovered)
        assert ari > 0.8, f"planted-block ARI {ari:.2f} ({correlation})"

    def test_planted_modules_significant_under_null(self):
        adata, _ = _planted_modules_adata()
        spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=3,
            score_cells=False, n_perms=99, seed=0, key_added="sgm",
        )
        stats = adata.uns["sgm"]["module_stats"]
        assert set(stats.columns) >= {"module", "n_genes", "mean_intra_corr", "pval"}
        assert len(stats) == 3
        assert (stats["pval"] < 0.05).all(), f"planted modules not significant:\n{stats}"
        assert (stats["mean_intra_corr"] > 0.5).all()

    def test_null_pvalues_calibrated_on_noise(self):
        # On iid noise there is no spatial co-expression; the rank-matched
        # null (which repeats the clustering per permutation) must NOT call
        # the noise partitions significant. Without the rank-matching this
        # fails badly: the observed modules are *selected* to be the most
        # correlated groupings, so a fixed-gene-set null is anti-conservative
        # by construction.
        adata = _noise_adata()
        spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=3,
            score_cells=False, n_perms=99, seed=0, key_added="sgm",
        )
        pvals = adata.uns["sgm"]["module_stats"]["pval"].to_numpy()
        assert np.isfinite(pvals).all()
        assert (pvals > 0.05).all(), f"noise modules called significant: {pvals}"

    def test_n_perms_zero_gives_nan_pvals(self):
        adata, _ = _planted_modules_adata(seed=1)
        spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=3,
            score_cells=False, n_perms=0, key_added="sgm",
        )
        stats = adata.uns["sgm"]["module_stats"]
        assert stats["pval"].isna().all()
        assert np.isfinite(stats["mean_intra_corr"]).all()

    def test_null_reproducible_across_n_jobs(self):
        adata1, _ = _planted_modules_adata(seed=2)
        adata2, _ = _planted_modules_adata(seed=2)
        spatial_gene_modules(adata1, clustering="hierarchical", n_modules=3,
                             n_perms=25, seed=5, n_jobs=1,
                             score_cells=False, key_added="sgm")
        spatial_gene_modules(adata2, clustering="hierarchical", n_modules=3,
                             n_perms=25, seed=5, n_jobs=2,
                             score_cells=False, key_added="sgm")
        p1 = adata1.uns["sgm"]["module_stats"]["pval"].to_numpy()
        p2 = adata2.uns["sgm"]["module_stats"]["pval"].to_numpy()
        np.testing.assert_allclose(p1, p2)


class TestSpatialGeneModulesLeiden:
    """The leiden default: data-determined module count."""

    @pytest.mark.parametrize("correlation", ["spatial_lag", "moran_bv"])
    def test_planted_blocks_recovered_leiden(self, correlation):
        from sklearn.metrics import adjusted_rand_score

        adata, truth = _planted_modules_adata()
        # Default call: no clustering=, no n_modules= — leiden must both
        # recover the planted partition AND find the right module COUNT.
        df = spatial_gene_modules(
            adata, correlation=correlation, score_cells=False, n_perms=0,
        )
        recovered = df.set_index("gene").loc[adata.var_names, "module"].to_numpy()
        ari = adjusted_rand_score(truth, recovered)
        assert ari > 0.8, f"planted-block ARI {ari:.2f} ({correlation})"
        assert df["module"].nunique() == 3, (
            f"leiden found {df['module'].nunique()} modules, expected 3"
        )

    def test_leiden_finds_few_modules_on_noise(self):
        # The point of data-determined k (library_audit: module-count
        # selection): on iid noise, leiden must not manufacture anywhere
        # near a forced n_modules=10 partition of genuine-looking modules.
        adata = _noise_adata()
        df = spatial_gene_modules(adata, score_cells=False, n_perms=0)
        n_found = df.loc[df["module"] >= 0, "module"].nunique()
        assert n_found < adata.n_vars // 2, (
            f"leiden found {n_found} modules on pure noise"
        )

    def test_leiden_noise_pvals_not_significant(self):
        adata = _noise_adata()
        spatial_gene_modules(
            adata, score_cells=False, n_perms=99, seed=0, key_added="sgm",
        )
        pvals = adata.uns["sgm"]["module_stats"]["pval"].to_numpy()
        finite = pvals[np.isfinite(pvals)]  # singleton modules stay NaN
        assert (finite > 0.05).all(), f"noise modules called significant: {pvals}"

    def test_leiden_null_reproducible_across_n_jobs(self):
        # Acceptance test for the per-perm leiden seed threading: leidenalg
        # is seeded per call from each child RNG stream, so results must be
        # identical regardless of thread scheduling.
        adata1, _ = _planted_modules_adata(seed=2)
        adata2, _ = _planted_modules_adata(seed=2)
        spatial_gene_modules(adata1, n_perms=25, seed=5, n_jobs=1,
                             score_cells=False, key_added="sgm")
        spatial_gene_modules(adata2, n_perms=25, seed=5, n_jobs=2,
                             score_cells=False, key_added="sgm")
        p1 = adata1.uns["sgm"]["module_stats"]["pval"].to_numpy()
        p2 = adata2.uns["sgm"]["module_stats"]["pval"].to_numpy()
        np.testing.assert_allclose(p1, p2)

    def test_leiden_ignores_n_modules_and_records_resolution(self):
        adata1, _ = _planted_modules_adata(seed=4)
        adata2, _ = _planted_modules_adata(seed=4)
        # n_modules must be dead under leiden: wildly different requests
        # yield identical partitions.
        df1 = spatial_gene_modules(adata1, n_modules=3, n_perms=0,
                                   key_added="sgm")
        df2 = spatial_gene_modules(adata2, n_modules=7, n_perms=0,
                                   key_added="sgm")
        assert df1.equals(df2)
        params = adata1.uns["sgm"]["params"]
        assert params["resolution"] == 1.0
        assert params["n_modules_requested"] is None
        n_found = int(adata1.uns["sgm"]["module_stats"].shape[0])
        assert params["n_modules"] == n_found
        assert adata1.obsm["sgm_scores"].shape == (adata1.n_obs, n_found)

    def test_leiden_reproducible_same_seed(self):
        adata1, _ = _planted_modules_adata(seed=3)
        adata2, _ = _planted_modules_adata(seed=3)
        df1 = spatial_gene_modules(adata1, score_cells=False, n_perms=0, seed=7)
        df2 = spatial_gene_modules(adata2, score_cells=False, n_perms=0, seed=7)
        assert df1.equals(df2)

    def test_unknown_clustering_raises(self):
        adata, _ = _planted_modules_adata()
        with pytest.raises(ValueError, match="Unknown clustering"):
            spatial_gene_modules(adata, clustering="kmeans",
                                 score_cells=False, n_perms=0)


class TestSpatialGeneModulesScores:
    def test_scores_are_zscored_means_not_raw_means(self):
        # Scores must be means of the z-scored expression: a module gene
        # with a huge raw magnitude must not dominate the module score.
        adata, truth = _planted_modules_adata(seed=3)
        # inflate one module-0 gene's scale 100x (same spatial pattern)
        X = adata.X.copy()
        X[:, 0] = X[:, 0] * 100.0
        adata.X = X
        df = spatial_gene_modules(
            adata, clustering="hierarchical", n_modules=3,
            score_cells=True, n_perms=0, key_added="sgm",
        )
        scores = adata.obsm["sgm_scores"]
        # z-scored columns each have |mean|=0, std<=1 -> module-score std is
        # bounded ~1 regardless of the 100x gene; a raw mean would have
        # std ~25.
        assert scores.shape == (adata.n_obs, 3)
        assert np.abs(scores.mean(axis=0)).max() < 1e-5
        assert scores.std(axis=0).max() < 1.5
