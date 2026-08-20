"""Tests for the neighborhood-analysis module: nhood_enrichment permutation
z-scores, co_occurrence ring semantics, ligrec gating/p-values, and the
n_jobs parallelism added to nhood_enrichment / ligrec."""
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
    expression_by_distance,
    composition,
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

    def test_permutation_variance_numerically_stable(self):
        # With catastrophic cancellation a single-pass sum-of-squares method
        # could produce negative variance; the two-pass mean-then-deviations
        # computation (which replaced an earlier Welford implementation when
        # the loop was parallelized) must not.
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

    def test_thresholds_are_annular_not_cumulative(self):
        # Regression: each threshold's ratio used to be computed from
        # `dists <= d` (cumulative from 0), so a strong, tightly-localized
        # signal at a close range spuriously persisted at every *larger*
        # threshold too, instead of correctly disappearing once the search
        # ring moves past it -- "distance-resolved" in name only.
        #
        # M isolated (A, B) pairs, each pair 3um apart, pairs spaced 1000um
        # from each other (no cross-pair contamination at these thresholds).
        # thresholds=[10, 30]: ring 1 = [0,10] contains exactly the paired
        # partner (real, strong co-localization); ring 2 = (10,30] contains
        # nothing at all for any cell (its only real neighbor was already
        # captured by ring 1). A correct annular ring 2 is *empty* -- NaN,
        # "no data at this scale" (squidpy's convention too), NOT 0.0
        # (which would falsely read as perfect avoidance) -- while the old
        # cumulative code would instead report ring 2 identical to ring 1
        # (~2.0), since `dists<=30` still includes the same distance-3 pair
        # with nothing new to dilute or change it.
        m = 15
        centers = np.column_stack([np.arange(m) * 1000.0, np.zeros(m)])
        a_coords = centers
        b_coords = centers + np.array([3.0, 0.0])
        coords = np.vstack([a_coords, b_coords]).astype(np.float32)
        labels = np.array(["A"] * m + ["B"] * m)

        adata = ad.AnnData(X=np.zeros((2 * m, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["ct"] = labels

        co_occurrence(adata, "ct", interval=(10.0, 30.0, 2))
        result = adata.uns["co_occurrence"]
        groups = result["groups"]
        a_idx, b_idx = groups.index("A"), groups.index("B")
        occ = result["occ"]

        assert occ[a_idx, b_idx, 0] == pytest.approx(2.0, rel=0.05)  # ring 1: strong
        # ring 2: empty -> NaN for every type pair, NOT ~2.0 again and NOT 0.0
        assert np.isnan(occ[:, :, 1]).all()

    def test_empty_ring_is_nan_not_zero(self):
        # Regression: an empty ring used to be filled with 0.0 -- the value
        # for "perfect avoidance" -- instead of NaN ("no data at this
        # scale"). Two cells 5um apart, thresholds [10, 20]: ring 2 (10,20]
        # has no pairs at all.
        adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
        adata.obsm["spatial"] = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        adata.obs["ct"] = ["A", "B"]

        co_occurrence(adata, "ct", interval=np.array([10.0, 20.0]))
        occ = adata.uns["co_occurrence"]["occ"]
        assert np.isfinite(occ[0, 1, 0])  # ring 1 has the real pair
        assert np.isnan(occ[:, :, 1]).all()  # ring 2 is empty -> NaN

    def test_unsorted_interval_raises(self):
        adata = ad.AnnData(X=np.zeros((5, 1), dtype=np.float32))
        adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 50, (5, 2)).astype(np.float32)
        adata.obs["ct"] = ["A", "B", "A", "B", "A"]
        with pytest.raises(ValueError, match="strictly increasing"):
            co_occurrence(adata, "ct", interval=np.array([50.0, 10.0, 30.0]))
        with pytest.raises(ValueError, match="strictly increasing"):
            co_occurrence(adata, "ct", interval=np.array([-5.0, 10.0]))


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

    def test_unused_category_does_not_raise_keyerror(self):
        # Regression: cell_df[groupby] was built via `.to_numpy()`, which
        # silently drops a Categorical Series' dtype (and with it, any
        # *unused* declared category) down to a plain object array --
        # `groupby` on a plain object column then only ever returns
        # categories actually observed in this call, with no "unused
        # category" concept. `.loc[categories]` (categories from
        # _category_codes, the *full* declared list) then hard-crashed with
        # KeyError the moment a declared category had zero cells -- routine
        # after subsetting, since scanpy/pandas never prune unused
        # categories automatically.
        adata = self._adata(n=20)
        # Declare a third category with zero actual cells.
        adata.obs["ct"] = pd.Categorical(adata.obs["ct"], categories=["A", "B", "C"])
        assert "C" not in adata.obs["ct"].to_numpy()

        centrality_scores(adata, "ct")
        result = adata.uns["centrality_scores"]
        assert list(result.index) == ["A", "B", "C"]
        assert result.loc["C"].isna().all()


class TestExpressionByDistance:
    def test_cells_beyond_max_dist_are_excluded_not_dumped_in_terminal_bin(self):
        # Regression: np.digitize maps a distance beyond the last bin edge to
        # index n_bins (n_bins+1 before the -1), which np.clip(...,0,n_bins-1)
        # silently folded back into the *terminal* bin instead of excluding
        # it -- with the default max_dist_um (95th percentile), the farthest
        # 5% of cells were always dumped into the last bin alongside
        # genuinely-in-range cells, systematically distorting the tail.
        rng = np.random.default_rng(6)
        n_near = 300
        n_far = 20
        # Anchor fixed at the origin; "near" cells spread over [0,100]^2 so
        # plenty legitimately land within the terminal bin's own range
        # (90,100] -- distinguishing "correctly excluded" from "the bin was
        # never going to have anything in it anyway."
        near = rng.uniform(0, 100, (n_near, 2)).astype(np.float32)
        # Far outliers, well beyond any reasonable max_dist_um, with an
        # extreme expression value that would visibly contaminate the
        # terminal bin's mean if wrongly included.
        far = rng.uniform(950, 1000, (n_far, 2)).astype(np.float32)
        coords = np.vstack([[[0.0, 0.0]], near, far]).astype(np.float32)
        expr = np.concatenate([
            [0.0],  # anchor cell itself
            rng.uniform(1, 2, n_near),
            np.full(n_far, 10_000.0),
        ]).astype(np.float32).reshape(-1, 1)

        adata = ad.AnnData(X=expr)
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        adata.obs["anchor"] = ["anchor"] + ["other"] * (n_near + n_far)

        # max_dist_um well short of the far cluster -- they must be
        # excluded entirely, not folded into the last bin.
        expression_by_distance(adata, "anchor", "anchor", genes=["gene_0"], n_bins=10, max_dist_um=100.0)
        result = adata.uns["expression_by_distance"]
        assert int(result["n_cells"][-1]) > 0  # terminal bin genuinely has near cells in it
        terminal_mean = result["mean"]["gene_0"].to_numpy()[-1]
        assert terminal_mean < 100.0  # nowhere near the 10_000 outlier contamination
        assert int(result["n_cells"][-1]) < n_near  # far cluster not counted here
        assert int(result["n_cells"].sum()) < n_near + n_far + 1  # far cluster excluded, not just relocated


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


class TestLigrecCPUPath:
    def test_pvalues_have_floor_never_exactly_zero(self):
        # Regression: pvalues = perm_ge / n_perms could land on an exact 0.0
        # (every permutation less extreme than observed) -- not a valid
        # Monte Carlo p-value estimate, and breaks downstream -log10(p) /
        # FDR steps. (k+1)/(n+1) floors it at 1/(n_perms+1) instead.
        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        adata = _adata_for_ligrec(n=60, seed=1)
        n_perms = 30
        ligrec(adata, "cell_type", lr_pairs, n_perms=n_perms, seed=0, n_jobs=1)
        pvalues = adata.uns["ligrec"]["pvalues"].to_numpy()
        assert np.all(pvalues >= 1.0 / (n_perms + 1.0) - 1e-12)
        assert not np.any(pvalues == 0.0)

    def test_metadata_records_spatial_restricted_true_on_cpu(self):
        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        adata = _adata_for_ligrec()
        ligrec(adata, "cell_type", lr_pairs, n_perms=10, seed=0)
        assert adata.uns["ligrec"]["metadata"]["spatial_restricted"] is True

    def test_expr_frac_gate_nans_untested_pairs(self):
        # CellPhoneDB expressing-fraction gate: type A never expresses the
        # ligand (gene_0 == 0 in every A cell), so every (sender=A, *) column
        # for that LR pair must be NaN -- untested, not scored -- in both
        # means and pvalues. B expresses everything, so (B, A) and (B, B)
        # stay tested (finite).
        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        adata = _adata_for_ligrec(n=40, seed=2)
        a_mask = (adata.obs["cell_type"] == "A").to_numpy()
        adata.X[a_mask, 0] = 0.0  # A never expresses the ligand
        adata.X[~a_mask, :] += 0.5  # B robustly expresses everything

        ligrec(adata, "cell_type", lr_pairs, n_perms=20, seed=0, expr_frac_threshold=0.1)
        means = adata.uns["ligrec"]["means"]
        pvalues = adata.uns["ligrec"]["pvalues"]
        for receiver in ("A", "B"):
            assert np.isnan(means.loc[("gene_0", "gene_1"), ("A", receiver)])
            assert np.isnan(pvalues.loc[("gene_0", "gene_1"), ("A", receiver)])
        for receiver in ("A", "B"):
            assert np.isfinite(means.loc[("gene_0", "gene_1"), ("B", receiver)])
            assert np.isfinite(pvalues.loc[("gene_0", "gene_1"), ("B", receiver)])

    def test_zero_edge_pair_is_nan_not_zero(self):
        # Two spatially separated blobs whose KNN graph never connects them:
        # cross-type (A, B) columns have zero spatial edges, which must be
        # NaN (no contact observed), not a fabricated score of 0 with p ~ 1.
        rng = np.random.default_rng(3)
        n_half = 20
        blob_a = rng.normal((0.0, 0.0), 2.0, (n_half, 2))
        blob_b = rng.normal((10_000.0, 10_000.0), 2.0, (n_half, 2))
        X = rng.random((2 * n_half, 4)).astype(np.float32) + 0.5
        adata = ad.AnnData(X=X, obsm={"spatial": np.vstack([blob_a, blob_b]).astype(np.float32)})
        adata.var_names = [f"gene_{i}" for i in range(4)]
        adata.obs["cell_type"] = ["A"] * n_half + ["B"] * n_half
        spatial_neighbors(adata, n_neighbors=5)

        lr_pairs = pd.DataFrame({"ligand": ["gene_0"], "receptor": ["gene_1"]})
        ligrec(adata, "cell_type", lr_pairs, n_perms=20, seed=0)
        means = adata.uns["ligrec"]["means"]
        pvalues = adata.uns["ligrec"]["pvalues"]
        assert np.isnan(means.loc[("gene_0", "gene_1"), ("A", "B")])
        assert np.isnan(pvalues.loc[("gene_0", "gene_1"), ("A", "B")])
        assert np.isfinite(means.loc[("gene_0", "gene_1"), ("A", "A")])
        assert np.isfinite(pvalues.loc[("gene_0", "gene_1"), ("A", "A")])

    def test_corr_method_adds_adjusted_pvalues(self):
        lr_pairs = pd.DataFrame({"ligand": ["gene_0", "gene_2"], "receptor": ["gene_1", "gene_3"]})
        adata = _adata_for_ligrec(n=40, seed=4)
        ligrec(adata, "cell_type", lr_pairs, n_perms=20, seed=0, corr_method="fdr_bh")
        result = adata.uns["ligrec"]
        assert "pvalues_adj" in result
        adj = result["pvalues_adj"]
        raw = result["pvalues"]
        assert adj.shape == raw.shape
        finite = np.isfinite(raw.to_numpy())
        # FDR-BH adjusted p is never below the raw p.
        assert (adj.to_numpy()[finite] >= raw.to_numpy()[finite] - 1e-12).all()
        assert result["metadata"]["corr_method"] == "fdr_bh"


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


class TestNeighborhoodFunctionsRejectMultiLibrary:
    # Regression (batch 7): co_occurrence/expression_by_distance had no
    # multi-library guard -- a concat_samples()-produced object would
    # silently compute spatially-meaningless statistics mixing unrelated
    # tissue sections instead of raising early (unlike pl.points()/etc,
    # which already guard via the same _check_single_library helper).
    def _multi_library_adata(self, n=30, seed=42):
        adata = _adata_with_neighbors(n=n, seed=seed)
        half = n // 2
        adata.obs["library_id"] = ["lib_a"] * half + ["lib_b"] * (n - half)
        return adata

    def test_co_occurrence_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            co_occurrence(adata, "cell_type")

    def test_expression_by_distance_raises(self):
        adata = self._multi_library_adata()
        adata = ad.AnnData(
            X=np.zeros((adata.n_obs, 1), dtype=np.float32),
            obs=adata.obs,
            obsm=adata.obsm,
        )
        adata.var_names = ["gene_0"]
        with pytest.raises(ValueError, match="librar"):
            expression_by_distance(adata, "cell_type", "A", genes=["gene_0"])

    def test_single_library_still_works(self):
        adata = _adata_with_neighbors()
        adata.obs["library_id"] = "only_lib"
        co_occurrence(adata, "cell_type", interval=(10.0, 50.0, 3))
        assert "co_occurrence" in adata.uns


class TestNhoodEnrichmentKnownAnswer:
    # Regression (batch 8, audit "moderate" item): nhood_enrichment had no
    # test validating its z-scores against a case with a known ground-truth
    # sign -- only shape/NaN/reproducibility checks. Two spatially separated,
    # internally-pure blobs give an unambiguous known answer: same-type edges
    # are wildly over-represented relative to the label-permutation null
    # (positive z), cross-type edges wildly under-represented (negative z),
    # since the real spatial graph (built by KNN) never connects the two
    # blobs at all.
    def _two_separated_blobs(self, n_per_blob=20, seed=0):
        rng = np.random.default_rng(seed)
        blob_a = rng.normal((0.0, 0.0), 2.0, (n_per_blob, 2))
        blob_b = rng.normal((10_000.0, 10_000.0), 2.0, (n_per_blob, 2))
        coords = np.vstack([blob_a, blob_b]).astype(np.float32)
        adata = ad.AnnData(obsm={"spatial": coords})
        adata.obs["cell_type"] = ["A"] * n_per_blob + ["B"] * n_per_blob
        spatial_neighbors(adata, n_neighbors=6)
        return adata

    def test_self_enrichment_strongly_positive_cross_strongly_negative(self):
        adata = self._two_separated_blobs()
        nhood_enrichment(adata, "cell_type", n_perms=200, seed=0)
        result = adata.uns["nhood_enrichment"]
        groups = list(result["groups"])
        zscore = result["zscore"]
        ia, ib = groups.index("A"), groups.index("B")

        assert zscore[ia, ia] > 5.0
        assert zscore[ib, ib] > 5.0
        assert zscore[ia, ib] < -5.0
        assert zscore[ib, ia] < -5.0


class TestCentralityScoresKnownAnswer:
    # Regression (batch 8, revised): centrality_scores now computes squidpy's
    # own *group*-level centrality (nx.group_degree_centrality /
    # nx.group_closeness_centrality / nx.average_clustering over each
    # category's node set) instead of averaging each cell's individual
    # centrality within its group -- those are genuinely different
    # statistics (e.g. on a star graph, "all leaves" as one group has
    # group_degree_centrality 1.0, not the 0.1 an individual leaf's own
    # degree centrality gives, since only one external node -- the hub --
    # needs to connect to *any* leaf for the group fraction to hit 1).
    # Known values are it's own thing to hand-derive for group metrics, so
    # this asserts against the same nx group-centrality functions called
    # directly, which is exactly what centrality_scores is documented to
    # delegate to -- a real regression test for the wiring (node-id /
    # category-code alignment, the do_closeness gate, empty categories),
    # not a tautology, since a wiring bug (wrong node ids, wrong category
    # membership) would still make this fail.
    def _star_graph_adata(self, n_leaves=10):
        n = n_leaves + 1
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obs["ct"] = ["hub"] + ["leaf"] * n_leaves
        adj = sp.lil_matrix((n, n), dtype=np.float32)
        for i in range(1, n):
            adj[0, i] = 1.0
            adj[i, 0] = 1.0
        adata.obsp["spatial_neighbors_connectivities"] = adj.tocsr()
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
        return adata

    def test_matches_networkx_group_centrality_directly(self):
        import networkx as nx

        adata = self._star_graph_adata(n_leaves=10)
        centrality_scores(adata, "ct", force_closeness=True)
        result = adata.uns["centrality_scores"]

        adj = adata.obsp["spatial_neighbors_connectivities"].astype(bool)
        G = nx.from_scipy_sparse_array(adj)
        hub_nodes, leaf_nodes = [0], list(range(1, 11))

        assert result.loc["hub", "degree_centrality"] == pytest.approx(
            nx.group_degree_centrality(G, hub_nodes)
        )
        assert result.loc["leaf", "degree_centrality"] == pytest.approx(
            nx.group_degree_centrality(G, leaf_nodes)
        )
        assert result.loc["hub", "closeness_centrality"] == pytest.approx(
            nx.group_closeness_centrality(G, hub_nodes)
        )
        assert result.loc["leaf", "closeness_centrality"] == pytest.approx(
            nx.group_closeness_centrality(G, leaf_nodes)
        )
        assert result.loc["hub", "clustering"] == pytest.approx(
            nx.average_clustering(G, nodes=hub_nodes)
        )
        assert result.loc["leaf", "clustering"] == pytest.approx(
            nx.average_clustering(G, nodes=leaf_nodes)
        )

    def test_hub_group_touches_more_of_the_tissue_than_a_single_leaf_pair(self):
        # A group-level distinction that *does* survive on a star graph:
        # a two-leaf group only ever touches the hub (1 external node out
        # of 9 remaining), while the hub-only group touches every leaf
        # (10 external nodes out of 10) -- a real, meaningful degree gap.
        adata = self._star_graph_adata(n_leaves=10)
        adata.obs["ct"] = ["hub"] + ["pair"] * 2 + ["other"] * 8
        centrality_scores(adata, "ct", force_closeness=True)
        result = adata.uns["centrality_scores"]

        assert result.loc["hub", "degree_centrality"] == pytest.approx(1.0)
        assert result.loc["pair", "degree_centrality"] == pytest.approx(1.0 / 9.0)


class TestCompositionKnownAnswer:
    # Regression (batch 8): tl.composition had zero test coverage at all.
    def _adata_with_regions(self):
        adata = ad.AnnData(X=np.zeros((8, 1), dtype=np.float32))
        adata.obs["region"] = ["R1"] * 4 + ["R2"] * 4
        adata.obs["cell_type"] = ["A", "A", "A", "B"] + ["A", "A", "B", "B"]
        return adata

    def test_normalized_fractions_match_hand_computed_values(self):
        adata = self._adata_with_regions()
        df = composition(adata, groupby="cell_type", region_by="region")

        assert df.loc["R1", "A"] == pytest.approx(0.75)
        assert df.loc["R1", "B"] == pytest.approx(0.25)
        assert df.loc["R2", "A"] == pytest.approx(0.5)
        assert df.loc["R2", "B"] == pytest.approx(0.5)
        assert "composition" in adata.uns

    def test_normalize_false_returns_raw_counts(self):
        adata = self._adata_with_regions()
        df = composition(adata, groupby="cell_type", region_by="region", normalize=False)

        assert df.loc["R1", "A"] == 3
        assert df.loc["R1", "B"] == 1
        assert df.loc["R2", "A"] == 2
        assert df.loc["R2", "B"] == 2

    def test_groupby_equals_region_by_raises(self):
        adata = self._adata_with_regions()
        with pytest.raises(ValueError, match="different"):
            composition(adata, groupby="region", region_by="region")
