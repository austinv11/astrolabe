"""Tests for nhood_enrichment Welford variance algorithm (audit fix 2.1)
and the n_jobs parallelism added to nhood_enrichment / ligrec."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._neighborhood import (
    nhood_enrichment,
    ligrec,
    co_occurrence,
    centrality_scores,
)


def _adata_with_neighbors(n=30, seed=42):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(obsm={"spatial": rng.uniform(0, 100, (n, 2)).astype(np.float32)})
    adata.obs["cell_type"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    adj = sp.random(n, n, density=0.25, format="csr", random_state=seed)
    adj = (adj + adj.T).astype(bool).astype(np.float32)
    adata.obsp["spatial_neighbors_connectivities"] = adj
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
    return adata


class TestNhoodEnrichment:
    def test_no_nan_zscore(self):
        adata = _adata_with_neighbors()
        nhood_enrichment(adata, "cell_type", n_perms=50, seed=0)
        zscore = adata.uns["nhood_enrichment"]["zscore"]
        assert not np.any(np.isnan(zscore)), "z-scores must not be NaN"

    def test_result_shape(self):
        adata = _adata_with_neighbors()
        nhood_enrichment(adata, "cell_type", n_perms=20, seed=1)
        result = adata.uns["nhood_enrichment"]
        n_types = len(result["groups"])
        assert result["zscore"].shape == (n_types, n_types)
        assert result["counts"].shape == (n_types, n_types)

    def test_groups_match_categories(self):
        adata = _adata_with_neighbors()
        nhood_enrichment(adata, "cell_type", n_perms=10, seed=2)
        groups = adata.uns["nhood_enrichment"]["groups"]
        expected = sorted(adata.obs["cell_type"].unique())
        assert list(groups) == expected

    def test_counts_are_nonnegative_integers(self):
        adata = _adata_with_neighbors()
        nhood_enrichment(adata, "cell_type", n_perms=10, seed=3)
        counts = adata.uns["nhood_enrichment"]["counts"]
        assert np.all(counts >= 0)

    def test_reproducible_with_seed(self):
        adata1 = _adata_with_neighbors(seed=99)
        adata2 = _adata_with_neighbors(seed=99)
        nhood_enrichment(adata1, "cell_type", n_perms=20, seed=42)
        nhood_enrichment(adata2, "cell_type", n_perms=20, seed=42)
        np.testing.assert_allclose(
            adata1.uns["nhood_enrichment"]["zscore"],
            adata2.uns["nhood_enrichment"]["zscore"],
        )

    def test_welford_variance_not_negative(self):
        # With catastrophic cancellation the old sum-of-squares method could
        # produce negative variance. Welford's algorithm prevents this.
        # Use a large homogeneous graph to stress-test variance stability.
        n = 100
        adata = _adata_with_neighbors(n=n, seed=7)
        nhood_enrichment(adata, "cell_type", n_perms=200, seed=0)
        zscore = adata.uns["nhood_enrichment"]["zscore"]
        assert not np.any(np.isnan(zscore))
        assert not np.any(np.isinf(zscore))


class TestNhoodEnrichmentNJobs:
    def test_n_jobs_matches_sequential(self):
        # Each permutation draws from an independent RNG substream spawned
        # from `seed`, so results must be identical regardless of n_jobs.
        adata1 = _adata_with_neighbors(seed=11)
        adata2 = _adata_with_neighbors(seed=11)
        nhood_enrichment(adata1, "cell_type", n_perms=50, seed=5, n_jobs=1)
        nhood_enrichment(adata2, "cell_type", n_perms=50, seed=5, n_jobs=2)
        np.testing.assert_allclose(
            adata1.uns["nhood_enrichment"]["zscore"],
            adata2.uns["nhood_enrichment"]["zscore"],
        )

    def test_n_jobs_minus_one_runs(self):
        adata = _adata_with_neighbors()
        nhood_enrichment(adata, "cell_type", n_perms=20, seed=0, n_jobs=-1)
        assert not np.any(np.isnan(adata.uns["nhood_enrichment"]["zscore"]))


class TestCoOccurrence:
    def test_csr_ratio_stays_near_one(self):
        # Regression: co_occurrence used to divide the neighbor-pair count
        # by the number of type-A cells (`type_counts[:, None]`) instead of
        # by the total number of neighbors of *any* type within d
        # (`count_mat.sum(axis=1)`). The former is an unbounded mean-count
        # that grows ~d^2 with the search radius, not a probability, so the
        # ratio against the (bounded) marginal P(B) diverged with d even
        # under pure complete-spatial-randomness labels (no real structure
        # at all) -- confirmed on a 20k-cell synthetic CSR repro where the
        # buggy ratio climbed from ~0.6 at d=25 to ~1500 at d=500. A
        # correctly-normalized ratio should hover near 1.0 at every d.
        rng = np.random.default_rng(0)
        n = 4000
        coords = rng.uniform(0, 2000, (n, 2)).astype(np.float32)
        labels = rng.integers(0, 4, n).astype(str)

        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["ct"] = labels

        co_occurrence(adata, "ct", interval=(20.0, 400.0, 15))
        occ = adata.uns["co_occurrence"]["occ"]

        # Ignore the very first (smallest-radius) threshold, where sample
        # sizes are small enough for genuine finite-sample noise; every
        # later threshold should stay in a tight band around 1.0.
        assert np.all(occ[:, :, 2:] > 0.7)
        assert np.all(occ[:, :, 2:] < 1.3)

    def test_result_shape_and_groups(self):
        rng = np.random.default_rng(1)
        n = 500
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata.obs["ct"] = (["A"] * (n // 2)) + (["B"] * (n - n // 2))

        co_occurrence(adata, "ct", interval=(10.0, 100.0, 5))
        result = adata.uns["co_occurrence"]
        assert result["occ"].shape == (2, 2, 5)
        assert result["groups"] == ["A", "B"]
        assert not np.any(np.isnan(result["occ"]))


class TestCentralityScoresMaxCloseness:
    def _adata(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = rng.uniform(0, 200, (n, 2)).astype(np.float32)
        adata.obs["ct"] = (["A"] * (n // 2)) + (["B"] * (n - n // 2))
        spatial_neighbors(adata, n_neighbors=4)
        return adata

    def test_default_skips_above_cutoff(self):
        adata = self._adata(n=20)
        centrality_scores(adata, "ct", max_closeness_n=10)
        result = adata.uns["centrality_scores"]
        assert result["closeness_centrality"].isna().all()

    def test_force_closeness_computes_regardless_of_cutoff(self):
        # Regression: force_closeness lets a caller opt into the O(n^2)
        # computation past the hardcoded 5_000-cell cutoff, which previously
        # silently NaN-filled closeness_centrality with no way to opt back in.
        adata = self._adata(n=20)
        centrality_scores(adata, "ct", max_closeness_n=10, force_closeness=True)
        result = adata.uns["centrality_scores"]
        assert not result["closeness_centrality"].isna().any()


def _adata_for_ligrec(n=40, n_genes=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obsm={"spatial": rng.uniform(0, 500, (n, 2)).astype(np.float32)},
    )
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["cell_type"] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    spatial_neighbors(adata, n_neighbors=6)
    return adata


class TestLigrecNJobs:
    def test_n_jobs_matches_sequential(self):
        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        adata1 = _adata_for_ligrec()
        adata2 = _adata_for_ligrec()
        ligrec(adata1, "cell_type", lr_pairs, n_perms=50, seed=3, n_jobs=1)
        ligrec(adata2, "cell_type", lr_pairs, n_perms=50, seed=3, n_jobs=2)
        pd.testing.assert_frame_equal(
            adata1.uns["ligrec"]["pvalues"], adata2.uns["ligrec"]["pvalues"]
        )
        pd.testing.assert_frame_equal(
            adata1.uns["ligrec"]["means"], adata2.uns["ligrec"]["means"]
        )

    def test_n_jobs_minus_one_runs(self):
        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        adata = _adata_for_ligrec()
        ligrec(adata, "cell_type", lr_pairs, n_perms=20, seed=0, n_jobs=-1)
        assert "pvalues" in adata.uns["ligrec"]


class TestLigrecPlot:
    def _adata_with_ligrec_result(self, n_types=10):
        # Craft a ligrec-shaped uns result directly rather than running the
        # real permutation test, so significance is fully controlled: only
        # 3 of n_types**2 (sender, receiver) columns are significant, mimicking
        # the user report (1 LR pair x 100 cell-type-pair columns, only 21
        # actually significant) at a size small enough to assert on exactly.
        import astrolabe as ab

        adata = ad.AnnData(X=np.zeros((5, 1), dtype=np.float32))
        cats = [str(i) for i in range(n_types)]
        ct_index = pd.MultiIndex.from_tuples(
            [(a, b) for a in cats for b in cats], names=["sender", "receiver"]
        )
        lr_index = pd.MultiIndex.from_tuples(
            [("LIG1", "REC1")], names=["ligand", "receptor"]
        )
        pvalues = pd.DataFrame(
            np.full((1, len(ct_index)), 0.9), index=lr_index, columns=ct_index
        )
        # Make exactly 3 columns significant.
        pvalues.iloc[0, [0, 1, 2]] = 0.001
        means = pd.DataFrame(
            np.full((1, len(ct_index)), 1.0), index=lr_index, columns=ct_index
        )
        adata.uns["ligrec"] = {
            "means": means,
            "pvalues": pvalues,
            "metadata": {"groupby": "ct", "n_perms": 100, "threshold": 0.05},
        }
        return adata, ab

    def test_drops_insignificant_columns(self):
        # Regression: ligrec_plot filtered rows to significant LR pairs but
        # rendered *every* (sender, receiver) column regardless of whether it
        # had any signal -- for n_types cell types that's n_types**2 columns,
        # ballooning the figure width (figsize scales linearly with n_cols)
        # even when almost all of them were empty/uninformative.
        adata, ab = self._adata_with_ligrec_result(n_types=10)
        panel = ab.pl.ligrec_plot(adata, p_threshold=0.05)
        ax = panel.figure.axes[0]
        assert len(ax.get_xticklabels()) == 3

    def test_figsize_capped_for_many_significant_columns(self):
        # Even when many columns *are* genuinely significant, width should
        # not grow unboundedly -- it should saturate at a fixed cap instead.
        adata, ab = self._adata_with_ligrec_result(n_types=20)
        adata.uns["ligrec"]["pvalues"].iloc[0, :] = 0.001  # all 400 significant
        panel = ab.pl.ligrec_plot(adata, p_threshold=0.05, top_n=400)
        width, _ = panel.figure.get_size_inches()
        assert width <= 24.0
        assert "pvalues" in adata.uns["ligrec"]
