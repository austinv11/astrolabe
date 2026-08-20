"""Tests for ICG log2FC NaN for zero-zero genes (audit fix 2.8), and for the
batch-4 bugfix pass (Mann-Whitney tie correction, log2fc pseudocount scale,
fdr_scope, and category-order consistency)."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from astrolabe.tl._icg import interaction_changed_genes, _wilcoxon_vectorized


def _adata_for_icg(n_per_type=20, n_genes=10, seed=42):
    """Build a controlled test fixture.

    Cells 0..(n-1) are type A; cells n..(2n-1) are type B.
    The first half of A cells connect to B; the second half have no B neighbors.
    This guarantees both contacted and non-contacted groups are non-empty.
    gene_0 is forced to zero so both groups always have zero mean.
    """
    n_cells = n_per_type * 2
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=0.6, format="csr", random_state=seed).astype(np.float32)
    X = X.tolil()
    X[:, 0] = 0.0  # gene_0 all-zero in both groups
    X = X.tocsr()

    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = ["A"] * n_per_type + ["B"] * n_per_type
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Controlled adjacency: first half of A (0..n/2-1) connect to B cells;
    # second half of A (n/2..n-1) have zero B-neighbors.
    adj = np.zeros((n_cells, n_cells), dtype=np.float32)
    half = n_per_type // 2
    b_start = n_per_type
    # A cells [0..half) each connect to B cell at (b_start + i)
    for i in range(half):
        adj[i, b_start + i] = 1.0
        adj[b_start + i, i] = 1.0
    # A cells [half..n_per_type) have no B connections

    adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
    return adata


class TestICGLog2FC:
    def test_zero_zero_gene_log2fc_is_nan(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        assert len(result) > 0, "expected at least one pair tested with min_cells=1"
        gene0_rows = result[result["gene"] == "gene_0"]
        assert len(gene0_rows) > 0, "gene_0 must appear in at least one pair result"
        assert gene0_rows["log2fc"].isna().all(), (
            "gene with zero expression in both contacted/non-contacted groups must have NaN log2fc"
        )

    def test_expressed_gene_log2fc_is_not_all_nan(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        expressed = result[result["gene"] != "gene_0"]["log2fc"]
        assert len(expressed) > 0, "expected results for expressed genes"
        assert not expressed.isna().all(), "some expressed genes must have numeric log2fc"

    def test_result_columns_present(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        assert len(result) > 0, "expected non-empty result with min_cells=1"
        for col in ("cell_type", "neighbor_type", "gene", "log2fc", "pval", "pval_adj"):
            assert col in result.columns, f"column '{col}' missing from ICG result"

    def test_empty_result_when_no_interactions(self):
        # Completely disconnected graph: no A-B contacts → empty result
        n_cells = 20
        X = sp.random(n_cells, 5, density=0.5, format="csr", random_state=0).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["cell_type"] = ["A"] * 10 + ["B"] * 10
        adata.var_names = [f"g{i}" for i in range(5)]
        adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix((n_cells, n_cells))
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
        result = interaction_changed_genes(adata, groupby="cell_type")
        assert len(result) == 0


class TestMannWhitneyTieCorrection:
    def test_matches_scipy_mannwhitneyu_on_zero_inflated_data(self):
        # Regression: the normal-approximation variance omitted the
        # standard tie-correction term (plain n1*n2*(n+1)/12 instead of the
        # tie-corrected formula). The naive formula *overstates* variance
        # whenever ties are common (the correction term is subtracted and
        # always >= 0), deflating |z| and giving p-values that are too
        # large -- measured 5-872x too conservative on realistic 75-78%-zero
        # count data, so real effects failed to clear BH significance.
        from scipy.stats import mannwhitneyu

        rng = np.random.default_rng(0)
        n1, n2, n_genes = 40, 35, 6
        # ~75% zeros, matching the audit's realistic zero-inflated scenario.
        X_a = rng.poisson(0.35, size=(n1, n_genes)).astype(np.float64)
        X_b = rng.poisson(0.35, size=(n2, n_genes)).astype(np.float64)
        assert (X_a == 0).mean() > 0.6  # confirm meaningfully zero-inflated

        _, _, pvals = _wilcoxon_vectorized(X_a, X_b)
        for j in range(n_genes):
            _, expected_p = mannwhitneyu(
                X_a[:, j], X_b[:, j], alternative="two-sided", method="asymptotic"
            )
            assert pvals[j] == pytest.approx(expected_p, rel=1e-6, abs=1e-9)

    def test_tie_correction_generally_shrinks_pvalues_on_tie_heavy_data(self):
        # The tie-correction term is always >= 0 and only ever subtracted
        # from the naive variance estimate, so tie-correcting alone can only
        # shrink sigma (never grow it), making genuinely tie-heavy genes
        # more significant, not less -- the old code's omission was too
        # conservative, not too liberal (matches the audit's finding: real
        # effects failed to clear BH significance). The *combined* fix also
        # adds a small, independent continuity correction (matching
        # scipy's use_continuity=True default) which pulls the other way by
        # a fixed ~0.5/sigma amount, so an exact per-gene inequality isn't
        # guaranteed for every gene (a gene with few ties and small sigma
        # can occasionally end up very slightly more conservative) -- this
        # checks the dominant, expected-common-case direction instead.
        rng = np.random.default_rng(1)
        n1, n2, n_genes = 30, 30, 8
        X_a = rng.poisson(0.4, size=(n1, n_genes)).astype(np.float64)
        X_b = rng.poisson(0.4, size=(n2, n_genes)).astype(np.float64)

        _, _, fixed_pvals = _wilcoxon_vectorized(X_a, X_b)

        # Reproduce the old (buggy: no tie correction, no continuity
        # correction) computation directly.
        from scipy.stats import norm, rankdata
        combined = np.vstack([X_a, X_b])
        ranked = np.apply_along_axis(rankdata, 0, combined)
        n = n1 + n2
        U1 = ranked[:n1].sum(axis=0) - n1 * (n1 + 1) / 2.0
        mu = n1 * n2 / 2.0
        sigma_old = np.sqrt(n1 * n2 * (n + 1) / 12.0)
        z_old = (U1 - mu) / sigma_old
        old_pvals = np.minimum(2.0 * norm.sf(np.abs(z_old)), 1.0)

        assert np.sum(fixed_pvals < old_pvals - 1e-9) > np.sum(fixed_pvals > old_pvals + 1e-9)
        assert np.any(fixed_pvals < old_pvals - 1e-6)  # genuinely differs somewhere


class TestICGLog2FCScale:
    def test_uses_expm1_scale_not_raw_log1p_mean_ratio(self):
        # Regression: log2fc was computed from a ratio of already-log1p-scale
        # means, not from expm1'd per-cell values averaged *then* ratio'd
        # (matching scanpy.tl.rank_genes_groups' convention) -- expm1 and
        # mean don't commute (Jensen's inequality), so these are genuinely
        # different quantities. Hand-picked exact values make this a precise
        # numeric check: "contacted" A cells (0-4) are log1p(1); "non-contacted"
        # A cells (5-9) are log1p(3) -- mean of expm1'd values is exactly 1.0
        # vs 3.0, so log2fc should be log2(1/3) ~ -1.585. The old
        # raw-log1p-mean-ratio approach instead compares log1p(1)=0.693
        # against log1p(3)=1.386 directly, giving log2fc ~ -1.0 instead.
        n_a, n_b = 10, 10
        n = n_a + n_b
        X = np.zeros((n, 1), dtype=np.float32)
        X[0:5, 0] = np.log1p(1.0)  # A cells that will contact B
        X[5:10, 0] = np.log1p(3.0)  # A cells that will not contact B
        adata = ad.AnnData(X=X)
        adata.var_names = ["gene_0"]
        adata.obs["cell_type"] = ["A"] * n_a + ["B"] * n_b

        # A cells 0-4 each connect to a distinct B cell (10-14); A cells 5-9
        # have no B connections at all.
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(5):
            adj[i, n_a + i] = 1.0
            adj[n_a + i, i] = 1.0
        adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}

        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        row = result[
            (result["cell_type"] == "A") & (result["neighbor_type"] == "B") & (result["gene"] == "gene_0")
        ].iloc[0]
        expected_expm1_log2fc = np.log2((1.0 + 1e-9) / (3.0 + 1e-9))
        naive_log_mean_log2fc = np.log2((np.log1p(1.0) + 1e-9) / (np.log1p(3.0) + 1e-9))
        assert row["log2fc"] == pytest.approx(expected_expm1_log2fc, abs=1e-6)
        assert row["log2fc"] != pytest.approx(naive_log_mean_log2fc, abs=1e-3)


class TestICGFDRScope:
    def test_global_and_pair_scope_give_different_results(self):
        # Note: BH-adjusted p-values under a larger pooled family are NOT
        # guaranteed to be pointwise >= the same test's per-pair-family
        # value (a test's rank, and therefore its adjustment, depends on the
        # *other* p-values sharing its family, which differ in both count
        # and composition between the two scopes) -- so this only checks
        # that the two scopes genuinely produce different results and both
        # remain valid probabilities, not a strict ordering between them.
        adata = _adata_for_icg(n_per_type=20, n_genes=10, seed=7)
        pair_result = interaction_changed_genes(
            adata.copy(), groupby="cell_type", min_cells=1, fdr_scope="pair"
        )
        global_result = interaction_changed_genes(
            adata.copy(), groupby="cell_type", min_cells=1, fdr_scope="global"
        )
        merged = pair_result.merge(
            global_result, on=["cell_type", "neighbor_type", "gene"], suffixes=("_pair", "_global")
        )
        assert not np.allclose(merged["pval_adj_global"], merged["pval_adj_pair"])
        assert merged["pval_adj_global"].between(0, 1).all()
        assert merged["pval_adj_pair"].between(0, 1).all()

    def test_default_scope_is_pair(self):
        adata = _adata_for_icg(n_per_type=20, n_genes=10, seed=7)
        default_result = interaction_changed_genes(adata.copy(), groupby="cell_type", min_cells=1)
        pair_result = interaction_changed_genes(
            adata.copy(), groupby="cell_type", min_cells=1, fdr_scope="pair"
        )
        pd.testing.assert_series_equal(
            default_result["pval_adj"].reset_index(drop=True),
            pair_result["pval_adj"].reset_index(drop=True),
        )


def _planted_icg_adata(n_a=60, n_b=30, n_genes=6, n_contacted=30, seed=0):
    """Planted contact-dependent signal.

    gene_0 in type-A cells is upregulated iff the cell has a B neighbor
    (the planted ICG); all other genes are iid noise in every cell.
    """
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    contacted = np.arange(n_contacted)          # A cells 0..n_contacted-1
    b_start = n_a

    X = np.log1p(rng.poisson(1.0, size=(n, n_genes)).astype(np.float64))
    X[:, 0] = np.log1p(rng.poisson(0.5, size=n).astype(np.float64))
    X[contacted, 0] = np.log1p(rng.poisson(6.0, size=n_contacted).astype(np.float64))

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["cell_type"] = ["A"] * n_a + ["B"] * n_b

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n_contacted):
        b = b_start + (i % n_b)
        adj[i, b] = 1.0
        adj[b, i] = 1.0
    adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
    return adata


class TestICGPermutationNull:
    def test_planted_gene_recovered_null_gene_not(self):
        # Planted ground truth: gene_0 is contact-dependent by construction;
        # every other gene is iid noise. Under the permutation null the
        # planted gene must clear BH < 0.05 in the (A, B) pair while the
        # null genes stay non-significant.
        adata = _planted_icg_adata()
        result = interaction_changed_genes(
            adata, groupby="cell_type", n_perms=199, seed=0
        )
        ab_rows = result[(result["cell_type"] == "A") & (result["neighbor_type"] == "B")]
        assert len(ab_rows) > 0

        planted = ab_rows[ab_rows["gene"] == "gene_0"].iloc[0]
        assert planted["pval_adj"] < 0.05, f"planted gene not recovered: {planted}"

        nulls = ab_rows[ab_rows["gene"] != "gene_0"]
        assert (nulls["pval_adj"] > 0.05).all(), (
            f"null genes spuriously significant:\n{nulls[nulls['pval_adj'] <= 0.05]}"
        )

    def test_p_asymptotic_column_present_and_pval_is_permutation(self):
        adata = _planted_icg_adata(seed=1)
        result = interaction_changed_genes(
            adata, groupby="cell_type", n_perms=49, seed=0
        )
        assert "p_asymptotic" in result.columns
        assert "pval" in result.columns
        # Permutation p-values are discrete with resolution 1/(n_valid+1) and
        # can never be exactly 0.
        assert (result["pval"] > 0).all()
        assert (result["pval"] >= 1.0 / 50.0 - 1e-12).all()

    def test_n_perms_zero_falls_back_to_asymptotic(self):
        adata = _planted_icg_adata(seed=2)
        result = interaction_changed_genes(
            adata, groupby="cell_type", n_perms=0
        )
        assert np.allclose(result["pval"], result["p_asymptotic"])

    def test_permutation_reproducible_across_n_jobs(self):
        adata = _planted_icg_adata(seed=3)
        r1 = interaction_changed_genes(
            adata.copy(), groupby="cell_type", n_perms=25, seed=7, n_jobs=1
        )
        r2 = interaction_changed_genes(
            adata.copy(), groupby="cell_type", n_perms=25, seed=7, n_jobs=2
        )
        pd.testing.assert_frame_equal(r1, r2)


class TestICGSkipSelfPairs:
    def _adata_with_self_contacts(self):
        # A cells 0..9 form 5 A-A contact pairs; A cells 10..29 have no A
        # neighbors. B exists so cross pairs are also testable.
        n_a, n_b = 30, 15
        n = n_a + n_b
        rng = np.random.default_rng(0)
        X = np.log1p(rng.poisson(1.0, size=(n, 4)).astype(np.float64))
        adata = ad.AnnData(X=X.astype(np.float32))
        adata.var_names = [f"g{i}" for i in range(4)]
        adata.obs["cell_type"] = ["A"] * n_a + ["B"] * n_b
        adj = np.zeros((n, n), dtype=np.float32)
        for i in range(0, 10, 2):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        for i in range(5):  # a few A-B contacts too
            adj[20 + i, n_a + i] = 1.0
            adj[n_a + i, 20 + i] = 1.0
        adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
        return adata

    def test_self_pairs_skipped_by_default(self):
        adata = self._adata_with_self_contacts()
        result = interaction_changed_genes(
            adata, groupby="cell_type", min_cells=2, n_perms=0
        )
        assert len(result) > 0
        assert (result["cell_type"] != result["neighbor_type"]).all()

    def test_self_pairs_included_when_disabled(self):
        adata = self._adata_with_self_contacts()
        result = interaction_changed_genes(
            adata, groupby="cell_type", min_cells=2, n_perms=0,
            skip_self_pairs=False,
        )
        self_rows = result[result["cell_type"] == result["neighbor_type"]]
        assert len(self_rows) > 0
        assert set(self_rows["cell_type"]) == {"A"}


class TestICGCategoryOrder:
    def test_unused_declared_category_does_not_break_or_appear_spuriously(self):
        # Consistency fix: interaction_changed_genes used to derive its cell
        # types via a standalone np.unique(labels) instead of the shared
        # _category_codes helper every other tl/_neighborhood.py function
        # uses. Final output rows are always sort_values()-ordered by the
        # (string) cell_type/neighbor_type columns regardless of iteration
        # order, so this doesn't change row *order* -- what it does change
        # is that a declared-but-unobserved category (zero cells) now flows
        # through the same `type_idx`/empty-group skip path as every other
        # category, instead of being invisible to the loop from the start.
        # Confirm this doesn't crash and doesn't fabricate a row for it.
        n = 30
        X = sp.random(n, 4, density=0.6, format="csr", random_state=0).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(4)]
        labels = pd.Categorical(
            ["Z"] * 10 + ["A"] * 10 + ["M"] * 10, categories=["Z", "A", "M", "W"]
        )
        adata.obs["cell_type"] = labels  # "W" is declared but has zero cells

        # Connect the first half (0-4) of each group's cells to the first
        # half of every *other* group, positionally -- leaves the second
        # half (5-9) of each group unconnected to that other group, so
        # every cross-type pair has both a contacted and non-contacted
        # subset (needed to pass min_cells and produce a result row).
        adj = np.zeros((n, n), dtype=np.float32)
        blocks = [range(0, 10), range(10, 20), range(20, 30)]
        for bi, block_a in enumerate(blocks):
            for bj, block_b in enumerate(blocks):
                if bi == bj:
                    continue
                for k in range(5):
                    a, b = block_a[k], block_b[k]
                    adj[a, b] = 1.0
                    adj[b, a] = 1.0
        adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}

        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        assert set(result["cell_type"]) == {"Z", "A", "M"}
        assert set(result["neighbor_type"]) == {"Z", "A", "M"}
        assert "W" not in set(result["cell_type"])
        assert "W" not in set(result["neighbor_type"])
