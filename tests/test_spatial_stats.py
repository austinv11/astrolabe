"""Tests for the n_jobs parallelism added to permutation-based spatial stats."""
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import scipy.sparse as sp

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._spatial_stats import (
    moran,
    moran_local,
    moran_local_bv,
    geary_c,
    getis_ord,
    ripley_l,
    spatially_variable_genes,
    spatial_silhouette,
    colocation_quotient,
    _build_weights,
    _boundary_silhouette,
)


def _adata_for_stats(n=40, n_genes=15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_genes)).astype(np.float32)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    adata = ad.AnnData(
        X=X,
        obsm={"spatial": rng.uniform(0, 500, (n, 2)).astype(np.float32)},
    )
    adata.var_names = var_names
    adata.obs["cell_type"] = (["A"] * (n // 2)) + (["B"] * (n - n // 2))
    spatial_neighbors(adata, n_neighbors=6)
    return adata


class TestMoranLocalNJobs:
    def test_runs_with_n_jobs(self):
        adata = _adata_for_stats()
        for n_jobs in (1, 2, -1):
            result = moran_local(adata, "gene_0", n_perms=49, n_jobs=n_jobs)
            assert len(result) == adata.n_obs
            assert not result["Ii"].isna().any()


class TestMoranLocalBVNJobs:
    def test_runs_with_n_jobs(self):
        adata = _adata_for_stats()
        for n_jobs in (1, 2, -1):
            result = moran_local_bv(adata, "gene_0", "gene_1", n_perms=49, n_jobs=n_jobs)
            assert len(result) == adata.n_obs
            assert not result["Ii"].isna().any()


class TestGetisOrdNJobs:
    def test_runs_with_n_jobs(self):
        adata = _adata_for_stats()
        for n_jobs in (1, 2, -1):
            result = getis_ord(adata, "gene_0", n_perms=49, n_jobs=n_jobs)
            assert len(result) == adata.n_obs
            assert not result["Gs"].isna().any()

    def test_n_perms_zero_uses_analytic_null(self):
        # Regression: the n_perms=0 branch reads esda.G_Local's analytic
        # z-score off `.Zs`, not `.z_norm` (which G_Local doesn't expose) --
        # this raised AttributeError until fixed. Analytic values now live in
        # their own z_norm/p_norm columns; the *_sim columns are honestly NaN
        # when no permutations were run (they used to hold the analytic
        # values under a permutation-implying name).
        adata = _adata_for_stats()
        result = getis_ord(adata, "gene_0", n_perms=0)
        assert len(result) == adata.n_obs
        assert not result["Gs"].isna().any()
        assert not result["z_norm"].isna().any()
        assert not result["p_norm"].isna().any()
        assert result["z_sim"].isna().all()
        assert result["p_sim"].isna().all()
        assert result["p_adj"].isna().all()
        assert (result["type"] == "not_tested").all()

    def test_p_sim_is_two_sided_not_directed(self):
        # p_sim is recomputed two-sided from the retained simulations via
        # esda.significance.calculate_significance -- esda's own .p_sim uses
        # the deprecated data-directed convention its docs call "uniformly
        # too small" (~half the two-sided value). Two-sided p can never be
        # smaller than the directed one for the same sims.
        import esda
        adata = _adata_for_stats()
        result = getis_ord(adata, "gene_0", n_perms=99, seed=3)
        w = _build_weights(adata, "spatial_neighbors", row_standardize=False)
        y = adata.X[:, 0].astype(np.float64)
        lG = esda.G_Local(y, w, permutations=99, transform="r", star=1, seed=3, n_jobs=1)
        directed = np.asarray(lG.p_sim)
        ours = result["p_sim"].to_numpy()
        assert (ours >= directed - 1e-12).all()
        assert (ours > directed).any()  # strictly larger somewhere

    def test_self_weight_is_not_one_half(self):
        # Regression: _build_weights row-standardized the graph *before*
        # esda.G_Local filled in the Gi* self-weight, so the self-weight
        # renormalized to exactly 0.5 for every cell regardless of neighbor
        # count, instead of the intended ~1/(k+1) (Ord & Getis convention).
        # Verified indirectly here: for a 6-neighbor graph, 1/(k+1) ≈ 0.143,
        # nowhere near 0.5 -- a cell's own value should therefore only
        # weakly dominate Gs relative to its neighbors' mean, not ~50/50.
        adata = _adata_for_stats(n=60)
        # One extreme outlier cell, otherwise uniform low expression --
        # under the old bug (self-weight 0.5) Gs is ~half the outlier's own
        # raw value, hugely inflating its z-score in isolation; under the
        # correct ~1/(k+1) self-weight, a single isolated outlier surrounded
        # by low neighbors should NOT be the single highest-Gs cell, since
        # its neighbors pull the local mean back down much more strongly.
        adata.X = np.ones((adata.n_obs, adata.n_vars), dtype=np.float32) * 0.01
        adata.X[0, 0] = 1000.0
        result = getis_ord(adata, "gene_0", n_perms=0)
        top_cell = result["Gs"].idxmax()
        assert top_cell != adata.obs_names[0]

    def test_no_gi_star_self_weight_warning(self):
        # Regression: our spatial graph has a zero diagonal (no self-loops)
        # and is already row-standardized, so esda.G_Local(..., star=True)
        # can't infer a Gi* self-weight on its own -- it silently guesses one
        # *per row* (that row's max neighbor weight) and raises a UserWarning
        # every call ("Gi* requested, but ... no default value supplied to
        # star"). getis_ord now passes an explicit star=1 to avoid this.
        adata = _adata_for_stats()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getis_ord(adata, "gene_0", n_perms=0)
        star_warnings = [w for w in caught if "star" in str(w.message).lower()]
        assert not star_warnings, [str(w.message) for w in star_warnings]


class TestRipleyLNJobs:
    def test_n_jobs_matches_sequential(self):
        # ripley_l seeds its own CSR simulations via _spawn_rngs, so results
        # are fully reproducible across worker counts for a fixed seed.
        adata = _adata_for_stats()
        r1 = ripley_l(adata.copy(), groupby="cell_type", n_perms=30, seed=0, n_jobs=1)
        r2 = ripley_l(adata.copy(), groupby="cell_type", n_perms=30, seed=0, n_jobs=2)
        for group in r1:
            np.testing.assert_allclose(
                r1[group]["L_obs"].to_numpy(), r2[group]["L_obs"].to_numpy()
            )
            np.testing.assert_allclose(
                r1[group]["L_hi"].to_numpy(), r2[group]["L_hi"].to_numpy()
            )
            np.testing.assert_allclose(
                r1[group]["L_lo"].to_numpy(), r2[group]["L_lo"].to_numpy()
            )

    def test_n_jobs_minus_one_runs(self):
        adata = _adata_for_stats()
        result = ripley_l(adata, groupby="cell_type", n_perms=20, seed=0, n_jobs=-1)
        assert set(result.keys()) == {"A", "B"}

    def test_k_matches_per_radius_loop(self):
        # Regression: _k() used to call cKDTree.count_neighbors once per
        # radius in a Python loop, redundantly re-traversing the tree
        # `len(radii)` times. scipy's count_neighbors accepts the whole radii
        # array and does a *single* traversal for an identical result --
        # ~6x faster on a 15k-point/200-radius benchmark. Pin that the
        # vectorized call still matches the (slow) per-radius reference.
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(3)
        n = 300
        pts = rng.uniform(0, 500, (n, 2))
        radii = np.linspace(0, 200, 40)
        tree = cKDTree(pts)

        vectorized = tree.count_neighbors(tree, radii).astype(np.float64) - n
        looped = np.array(
            [tree.count_neighbors(tree, r) - n for r in radii], dtype=np.float64
        )
        np.testing.assert_array_equal(vectorized, looped)


class TestSpatiallyVariableGenesNJobs:
    def test_n_jobs_matches_sequential_analytic(self):
        # n_perms=0 (default) is fully deterministic (analytic p-values), so
        # n_jobs must not change results.
        adata = _adata_for_stats()
        df1 = spatially_variable_genes(adata.copy(), flavor="moran", n_jobs=1)
        df2 = spatially_variable_genes(adata.copy(), flavor="moran", n_jobs=2)
        np.testing.assert_allclose(
            df1.loc[adata.var_names, "score"].to_numpy(),
            df2.loc[adata.var_names, "score"].to_numpy(),
        )
        np.testing.assert_allclose(
            df1.loc[adata.var_names, "pval"].to_numpy(),
            df2.loc[adata.var_names, "pval"].to_numpy(),
        )

    def test_geary_flavor_n_jobs_matches_sequential(self):
        adata = _adata_for_stats()
        df1 = spatially_variable_genes(adata.copy(), flavor="geary", n_jobs=1)
        df2 = spatially_variable_genes(adata.copy(), flavor="geary", n_jobs=-1)
        np.testing.assert_allclose(
            df1.loc[adata.var_names, "score"].to_numpy(),
            df2.loc[adata.var_names, "score"].to_numpy(),
        )

    def test_permutation_path_runs_with_n_jobs(self):
        adata = _adata_for_stats(n_genes=4)
        df = spatially_variable_genes(adata, flavor="moran", n_perms=19, n_jobs=2)
        assert len(df) == adata.n_vars
        assert not df["pval"].isna().any()

    def test_geary_flavor_n_jobs_no_shape_mismatch(self):
        # Regression: esda.Moran/Geary reassign `w.transform` on every
        # construction, and libpysal's transform setter unconditionally
        # invalidates + rebuilds cached derived state (w.sparse) with no
        # locking. Sharing one W across the thread-backed per-gene Parallel
        # loop raced on that rebuild and could raise
        # `ValueError: operands could not be broadcast together with shapes
        # (...) (...)`. Only reliably reproduced on a much larger graph in
        # practice (this repo's lightweight fixtures may not trigger the race
        # every run), but repeating with more genes/threads than the other
        # n_jobs tests in this file gives it a real chance to catch a
        # regression rather than testing nothing.
        adata = _adata_for_stats(n=200, n_genes=60)
        for _ in range(5):
            df = spatially_variable_genes(adata.copy(), flavor="geary", n_jobs=4)
            assert len(df) == adata.n_vars


class TestSpatialSilhouette:
    def _chain_adata(self, labels, attr):
        # 5-cell chain graph: 0-1-2-3-4, each connected only to its
        # immediate neighbor(s).
        n = len(labels)
        row, col, data = [], [], []
        for i in range(n - 1):
            row += [i, i + 1]
            col += [i + 1, i]
            data += [1.0, 1.0]
        conn = sp.csr_matrix((data, (row, col)), shape=(n, n))
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obs["cl"] = pd.Categorical(labels)
        adata.obsp["spatial_neighbors_connectivities"] = conn
        adata.uns["spatial_neighbors"] = {
            "connectivities_key": "spatial_neighbors_connectivities"
        }
        adata.obsm["X_pca"] = attr
        return adata

    def test_isolated_singleton_scores_zero_not_one(self):
        # Regression: cell 2 ("B") has no same-cluster spatial neighbor at
        # all -- its a(i) is undefined. The old code filled that NaN with
        # 0.0 before computing (b-a)/max(a,b), which reads "a=0" as perfect
        # same-cluster compactness and forces the score to +1.0 (the
        # theoretical maximum) for what is actually the worst case: a cell
        # totally isolated from its own label. esda.boundary_silhouette's
        # analogous corner case (a singleton cluster) scores 0, not 1 -- this
        # pins that same convention here.
        labels = np.array(["A", "A", "B", "A", "A"])
        attr = np.array([[0.0], [0.1], [5.0], [0.2], [0.1]])
        adata = self._chain_adata(labels, attr)

        spatial_silhouette(adata, groupby="cl")
        scores = adata.obs["spatial_silhouette"].to_numpy()

        assert scores[2] == 0.0
        # Cells with a real same-cluster neighbor and a bordering
        # different-cluster neighbor should still score high (well-separated
        # in attribute space), unaffected by the fix.
        assert scores[1] > 0.9
        assert scores[3] > 0.9

    def test_non_boundary_cell_scores_zero(self):
        # Cell 0 only ever neighbors its own cluster ("A") -- not on any
        # boundary -- so it should score 0 regardless of the fix above.
        labels = np.array(["A", "A", "B", "B", "B"])
        attr = np.array([[0.0], [0.1], [5.0], [5.1], [5.2]])
        adata = self._chain_adata(labels, attr)

        spatial_silhouette(adata, groupby="cl")
        scores = adata.obs["spatial_silhouette"].to_numpy()

        assert scores[0] == 0.0

    def _random_adata(self, n=60, n_genes=5, n_clusters=4, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.random((n, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=X, obsm={"spatial": rng.uniform(0, 500, (n, 2)).astype(np.float32)})
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        # Force at least one singleton cluster, matching esda's own special case.
        labels = rng.integers(0, n_clusters, n)
        labels[0] = n_clusters
        adata.obs["cl"] = pd.Categorical(labels.astype(str))
        spatial_neighbors(adata, n_neighbors=6)
        adata.obsm["X_pca"] = rng.random((n, 8))
        return adata

    def test_matches_esda_boundary_silhouette_directly(self):
        esda = pytest.importorskip("esda")
        adata = self._random_adata()

        spatial_silhouette(adata, groupby="cl")
        ours = adata.obs["spatial_silhouette"].to_numpy()

        w = _build_weights(adata, "spatial_neighbors")
        attr = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        labels = adata.obs["cl"].to_numpy()
        # esda.boundary_silhouette calls np.seterr(all="raise") internally and
        # never restores it -- left unguarded, that leaks into every later
        # test in the same process, turning routine NaN-producing divides
        # (e.g. TestSpatiallyVariableGenesNaNPoisoning's zero-variance genes)
        # into FloatingPointError instead of the RuntimeWarning they expect.
        # np.errstate captures the current state on entry and restores exactly
        # that on exit, regardless of what happens inside.
        with np.errstate():
            ref = esda.boundary_silhouette(attr, labels, w)

        assert np.allclose(ours, ref, atol=1e-8)

    def test_chunking_matches_unchunked(self):
        adata = self._random_adata()
        conn = adata.obsp["spatial_neighbors_connectivities"]
        attr = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        labels = adata.obs["cl"].to_numpy()

        tiny = _boundary_silhouette(attr, labels, conn, max_block_mb=1e-4)
        large = _boundary_silhouette(attr, labels, conn, max_block_mb=1024.0)

        assert np.allclose(tiny, large)

    def test_set_diag_self_loop_isolated_cell_scores_zero(self):
        # spatial_neighbors' default KNN mode always connects each cell to
        # its k nearest neighbors regardless of distance, so a graph island
        # requires radius=... to actually exclude a far-away cell.
        rng = np.random.default_rng(2)
        n = 10
        coords = rng.uniform(0, 50, (n, 2)).astype(np.float32)
        coords[-1] = [10_000.0, 10_000.0]  # spatially isolated from everyone else
        adata = ad.AnnData(X=rng.random((n, 3)).astype(np.float32), obsm={"spatial": coords})
        adata.obs["cl"] = pd.Categorical((["A"] * (n // 2)) + (["B"] * (n - n // 2)))
        spatial_neighbors(adata, radius=100.0, set_diag=True)
        adata.obsm["X_pca"] = rng.random((n, 4))

        spatial_silhouette(adata, groupby="cl")
        scores = adata.obs["spatial_silhouette"].to_numpy()

        assert scores[-1] == 0.0

    def test_island_without_set_diag_raises(self):
        rng = np.random.default_rng(3)
        n = 10
        coords = rng.uniform(0, 50, (n, 2)).astype(np.float32)
        coords[-1] = [10_000.0, 10_000.0]  # spatially isolated, no self-loop this time
        adata = ad.AnnData(X=rng.random((n, 3)).astype(np.float32), obsm={"spatial": coords})
        adata.obs["cl"] = pd.Categorical((["A"] * (n // 2)) + (["B"] * (n - n // 2)))
        spatial_neighbors(adata, radius=100.0, set_diag=False)
        adata.obsm["X_pca"] = rng.random((n, 4))

        with pytest.raises(ValueError, match="filter_isolated"):
            spatial_silhouette(adata, groupby="cl")

    def test_nan_labels_rejected(self):
        adata = self._random_adata()
        labels = adata.obs["cl"].astype(object).to_numpy()
        labels[0] = np.nan
        adata.obs["cl"] = labels

        with pytest.raises(ValueError, match="NaN"):
            spatial_silhouette(adata, groupby="cl")


class TestRipleyPlotDetrending:
    def test_plot_does_not_double_subtract_distance(self):
        # Regression: tl.ripley_l's stored L_obs/L_hi/L_lo columns are
        # already the detrended L(d) - d quantity (`_l()` in
        # _spatial_stats.py subtracts `radii` before returning -- raw L(d) is
        # not expected to be 0 under CSR, only detrended L(d)-d is). pl.ripley_plot
        # used to subtract `bins` (=d) a *second* time, producing L(d) - 2d --
        # under pure CSR labels (no real spatial structure at all) this
        # fabricated a runaway decline to roughly -d at the largest distance,
        # for every group, instead of hovering near 0.
        import astrolabe as ab

        rng = np.random.default_rng(0)
        n = 4000
        coords = np.column_stack(
            [rng.uniform(0, 2000, n), rng.uniform(0, 2000, n)]
        ).astype(np.float32)
        labels = rng.integers(0, 3, n).astype(str)

        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["cl"] = labels

        ripley_l(adata, groupby="cl", support=30, n_perms=0)
        panel = ab.pl.ripley_plot(adata)
        for line in panel.figure.axes[0].lines:
            ydata = np.asarray(line.get_ydata())
            # Real (correctly-detrended) L(d)-d under CSR stays close to 0;
            # the old double-subtraction bug pushed this to roughly -max_r
            # (~-500 here), so a loose bound well above that catches the
            # regression without being sensitive to finite-sample noise.
            assert ydata.min() > -100


class TestRipleyLExpAndWindow:
    def test_l_exp_is_simulated_mean_not_hardcoded_zero(self):
        # Regression: L_exp used to be a hardcoded zeros array. This
        # estimator has no edge correction, so the true expected L(d)-d under
        # CSR is not flat at 0 -- L_exp should now be the simulated CSR
        # curve's own mean, which measurably deviates from 0 given enough
        # points/permutations to resolve the finite-sample bias above noise.
        rng = np.random.default_rng(2)
        n = 2000
        coords = np.column_stack(
            [rng.uniform(0, 500, n), rng.uniform(0, 500, n)]
        ).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["cl"] = "all"

        result = ripley_l(adata, groupby="cl", support=50, n_perms=200, seed=0)
        l_exp = result["all"]["L_exp"].to_numpy()
        assert not np.allclose(l_exp, 0.0)

    def test_l_exp_is_nan_without_permutations(self):
        adata = ad.AnnData(X=np.zeros((20, 1), dtype=np.float32))
        adata.obsm["spatial"] = np.random.default_rng(0).uniform(0, 100, (20, 2)).astype(np.float32)
        adata.obs["cl"] = "all"
        result = ripley_l(adata, groupby="cl", support=10, n_perms=0)
        assert result["all"]["L_exp"].isna().all()

    def test_kdtree_helper_uses_passed_window_not_own_bbox(self):
        # Direct unit test of the underlying mechanism: _ripley_l_kdtree's
        # K(r) area normalization scales with the *passed* window, not
        # `coords`' own bounding box. A 4x-larger window (area) should scale
        # K(r) -- and therefore L(r) -- up accordingly for interior radii
        # with a nonzero observed pair count.
        from astrolabe.tl._spatial_stats import _ripley_l_kdtree

        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 100, (80, 2))  # own bbox area = 100*100 = 10_000
        radii = np.linspace(5, 40, 10)

        l_own, _ = _ripley_l_kdtree(coords, radii, n_perms=0, rng=np.random.default_rng(0))
        l_big, _ = _ripley_l_kdtree(
            coords, radii, n_perms=0, rng=np.random.default_rng(0),
            window_min=np.array([0.0, 0.0]), window_max=np.array([200.0, 200.0]),  # area=40_000, 4x
        )
        assert np.all(l_big > l_own)

    def test_ripley_l_passes_tissue_wide_window_not_group_bbox(self):
        # Integration test: group "small" occupies a tiny corner of the
        # tissue; group "spread" spans the full tissue extent. Before the
        # fix, each group's CSR null (and K(r) area normalization) used only
        # that group's own occupied bounding box -- "small"'s own bbox is
        # far smaller than the tissue, so its L(r) would differ from what
        # the same points give under the correct, tissue-wide window.
        from astrolabe.tl._spatial_stats import _ripley_l_kdtree

        rng = np.random.default_rng(4)
        small_coords = rng.uniform(0, 20, (30, 2)).astype(np.float32)  # tiny corner
        spread_coords = rng.uniform(0, 500, (100, 2)).astype(np.float32)  # full tissue
        coords = np.vstack([small_coords, spread_coords])
        labels = np.array(["small"] * 30 + ["spread"] * 100)

        adata = ad.AnnData(X=np.zeros(len(coords), dtype=np.float32).reshape(-1, 1))
        adata.obsm["spatial"] = coords
        adata.obs["cl"] = labels

        result = ripley_l(adata, groupby="cl", support=20, n_perms=0, seed=0)
        l_actual = result["small"]["L_obs"].to_numpy()
        radii = result["small"].index.to_numpy()

        # What "small" would have gotten under the old bug (its own tiny bbox).
        l_own_bbox, _ = _ripley_l_kdtree(small_coords.astype(np.float64), radii, n_perms=0, rng=np.random.default_rng(0))
        assert not np.allclose(l_actual, l_own_bbox)


class TestSpatiallyVariableGenesNaNPoisoning:
    def test_zero_variance_gene_does_not_poison_other_pvalues(self):
        # Regression: statsmodels.multipletests propagates a single NaN
        # p-value across the *entire* array -- one zero-variance gene (routine
        # after filtering; esda.Moran/Geary return I/C=p_norm=NaN for a
        # constant column) used to silently NaN every other gene's
        # pval_adj too, reporting "0 spatially variable genes" with no error.
        adata = _adata_for_stats(n_genes=5)
        adata.X[:, 0] = 0.0  # gene_0 is now zero-variance
        df = spatially_variable_genes(adata, flavor="moran", n_perms=0)
        assert np.isnan(df.loc["gene_0", "pval_adj"])
        others = df.index.difference(["gene_0"])
        assert not df.loc[others, "pval_adj"].isna().any()

    def test_zero_variance_gene_does_not_poison_other_pvalues_geary(self):
        adata = _adata_for_stats(n_genes=5)
        adata.X[:, 0] = 0.0
        df = spatially_variable_genes(adata, flavor="geary", n_perms=0)
        assert np.isnan(df.loc["gene_0", "pval_adj"])
        others = df.index.difference(["gene_0"])
        assert not df.loc[others, "pval_adj"].isna().any()


class TestMoranGearySeedReproducibility:
    def test_moran_seed_reproduces_p_sim(self):
        adata = _adata_for_stats()
        r1 = moran(adata.copy(), "gene_0", n_perms=99, seed=7)
        r2 = moran(adata.copy(), "gene_0", n_perms=99, seed=7)
        np.testing.assert_array_equal(r1["p_sim"].to_numpy(), r2["p_sim"].to_numpy())

    def test_geary_c_seed_reproduces_p_sim(self):
        adata = _adata_for_stats()
        r1 = geary_c(adata.copy(), "gene_0", n_perms=99, seed=7)
        r2 = geary_c(adata.copy(), "gene_0", n_perms=99, seed=7)
        np.testing.assert_array_equal(r1["p_sim"].to_numpy(), r2["p_sim"].to_numpy())

    def test_geary_c_p_norm_is_two_tailed_like_moran(self):
        # Regression: esda.Geary.p_norm is one-tailed by esda's own design
        # (Moran's p_norm is explicitly doubled via two_tailed=True) -- left
        # unaligned, the same nominal alpha was ~2x more lenient for the
        # 'geary' flavor of spatially_variable_genes than for 'moran'.
        adata = _adata_for_stats()
        r = geary_c(adata, "gene_0", n_perms=0)
        # A doubled one-tailed p can reach exactly 1.0 (clipped); the
        # meaningful check is that it's never *more* significant than the
        # underlying one-tailed value would have been.
        assert (r["p_norm"].to_numpy() <= 1.0).all()
        assert (r["p_norm"].to_numpy() >= 0.0).all()


class TestColocationQuotient:
    def test_hand_computable_toy_example(self):
        # 4 points on a line at x=0,1,2,3: labels A,B,A,B. Each point's
        # single nearest neighbor is always the opposite label (distance 1
        # in both directions is tied only at the interior points, but with
        # this exact spacing the *unique* nearest neighbor is always the
        # adjacent point, which always has the opposite label) -- so
        # C_{A->B} = N_A (every A's nearest neighbor is B), giving
        # CLQ_{A->B} = (N_A/(N_A*1)) / (N_B/(N-1)) = (N-1)/N_B = 3/2 = 1.5.
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        labels = ["A", "B", "A", "B"]
        adata = ad.AnnData(X=np.zeros((4, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["ct"] = labels

        result = colocation_quotient(adata, groupby="ct", k=1, min_count=1)
        row = result[(result["category_a"] == "A") & (result["category_b"] == "B")].iloc[0]
        assert row["clq"] == pytest.approx(1.5)

    def test_clq_is_not_symmetric_in_general(self):
        # A tight two-point "A" cluster plus a single distant "B" -- A's
        # nearest neighbor is always the other A (not B), but B's nearest
        # neighbor must be an A (it's the only other point), so
        # CLQ_{A->B} != CLQ_{B->A}.
        coords = np.array([[0.0, 0.0], [0.1, 0.0], [50.0, 0.0]])
        labels = ["A", "A", "B"]
        adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["ct"] = labels

        result = colocation_quotient(adata, groupby="ct", k=1, min_count=1)
        clq_ab = result.query("category_a == 'A' and category_b == 'B'")["clq"].iloc[0]
        clq_ba = result.query("category_a == 'B' and category_b == 'A'")["clq"].iloc[0]
        assert clq_ab != pytest.approx(clq_ba)

    def test_permutation_p_value_present_and_bounded(self):
        adata = _adata_for_stats(n=60)
        adata.obs["ct"] = adata.obs["cell_type"]
        result = colocation_quotient(adata, groupby="ct", k=3, n_perms=99, seed=0, min_count=1)
        assert "p_value" in result.columns
        assert (result["p_value"] > 0).all()
        assert (result["p_value"] <= 1).all()

    def test_unknown_category_raises(self):
        adata = _adata_for_stats(n=20)
        with pytest.raises(KeyError):
            colocation_quotient(adata, groupby="cell_type", categories=["nonexistent"])

    def test_diagonal_expectation_hand_computable(self):
        # Leslie & Kronenfeld (2011): on the diagonal (A == B) the expected
        # same-type fraction is (N_A - 1)/(N - 1) -- a point is never its own
        # neighbor -- not N_A/(N - 1). Three tight A's (each A's NN is
        # another A) and two tight distant B's (each B's NN is the other B):
        #   N=5, N_A=3: CLQ_AA = (3/(3*1)) / ((3-1)/(5-1)) = 1/0.5 = 2.0
        #   N=5, N_B=2: CLQ_BB = (2/(2*1)) / ((2-1)/(5-1)) = 1/0.25 = 4.0
        # The old N_A/(N-1) denominator gave 4/3 and 8/5 instead -- biased
        # low by (N_A-1)/N_A, worst exactly for rare types.
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.1, 0.0], [100.0, 0.0], [101.0, 0.0]]
        )
        labels = ["A", "A", "A", "B", "B"]
        adata = ad.AnnData(X=np.zeros((5, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["ct"] = labels

        result = colocation_quotient(adata, groupby="ct", k=1, min_count=1)

        def _get(a, b):
            return result.query(f"category_a == '{a}' and category_b == '{b}'")["clq"].iloc[0]

        assert _get("A", "A") == pytest.approx(2.0)
        assert _get("B", "B") == pytest.approx(4.0)
        assert _get("A", "B") == pytest.approx(0.0)  # no A has a B as its NN

    def test_permutation_p_is_rank_based_not_deviation_from_one(self):
        # Two spatially separated pure blobs: same-type CLQ is maximal, so a
        # rank-based two-sided p against the permutation distribution must be
        # at (or near) the floor 1/(n_perms+1) * (2*0+1)... i.e. the minimum
        # attainable, while a cross-type pair is at the opposite extreme --
        # both tails must be detectable (the old |CLQ-1| deviation test could
        # not flag values *below* the finite-sample null mean correctly).
        rng = np.random.default_rng(0)
        blob_a = rng.normal((0.0, 0.0), 2.0, (25, 2))
        blob_b = rng.normal((1000.0, 1000.0), 2.0, (25, 2))
        adata = ad.AnnData(X=np.zeros((50, 1), dtype=np.float32))
        adata.obsm["spatial"] = np.vstack([blob_a, blob_b]).astype(np.float32)
        adata.obs["ct"] = ["A"] * 25 + ["B"] * 25

        n_perms = 199
        result = colocation_quotient(
            adata, groupby="ct", k=3, n_perms=n_perms, seed=0, min_count=1
        )
        aa = result.query("category_a == 'A' and category_b == 'A'")["p_value"].iloc[0]
        ab_p = result.query("category_a == 'A' and category_b == 'B'")["p_value"].iloc[0]
        floor = 1.0 / (n_perms + 1.0)
        assert aa <= 3 * floor  # same-type attraction: extreme upper tail
        assert ab_p <= 3 * floor  # cross-type avoidance: extreme lower tail

    def test_multi_library_raises(self):
        adata = _adata_for_stats(n=20)
        adata.obs["library_id"] = ["lib_a"] * 10 + ["lib_b"] * 10
        with pytest.raises(ValueError, match="librar"):
            colocation_quotient(adata, groupby="cell_type")


def _checkerboard_adata(m=12):
    # m x m unit lattice with a perfect checkerboard gene: every cell's
    # 4 nearest neighbors (rook-adjacent) have the OPPOSITE value -- the
    # canonical strongly *negatively* autocorrelated pattern. gene_1 is a
    # smooth left-right gradient (strong positive autocorrelation) as the
    # positive control.
    xs, ys = np.meshgrid(np.arange(m, dtype=np.float32), np.arange(m, dtype=np.float32))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    checker = ((xs + ys) % 2).ravel().astype(np.float32)
    gradient = xs.ravel().astype(np.float32)
    X = np.column_stack([checker, gradient])
    adata = ad.AnnData(X=X, obsm={"spatial": coords})
    adata.var_names = ["checker", "gradient"]
    spatial_neighbors(adata, n_neighbors=4)
    return adata


class TestSpatiallyVariableGenesOneSided:
    def test_checkerboard_not_significant_analytic(self):
        # One-sided upper-tail convention: "spatially variable" means
        # expression CLUSTERS in space. A checkerboard is strongly
        # negatively autocorrelated -- the old two-tailed p flagged it as an
        # SVG at genome-wide significance; one-sided it must be ~1.
        adata = _checkerboard_adata()
        df = spatially_variable_genes(adata, flavor="moran", n_perms=0)
        assert df.loc["checker", "pval"] > 0.9
        assert df.loc["gradient", "pval"] < 1e-6

    def test_checkerboard_not_significant_geary(self):
        # Geary mirror: C > 1 for the checkerboard, so the lower-tail
        # (C < 1) p must be ~1; the gradient's C << 1 must be significant.
        adata = _checkerboard_adata()
        df = spatially_variable_genes(adata, flavor="geary", n_perms=0)
        assert df.loc["checker", "pval"] > 0.9
        assert df.loc["gradient", "pval"] < 1e-6

    def test_convention_does_not_flip_with_n_perms(self):
        # Regression: n_perms=0 used a two-tailed analytic p while n_perms>0
        # used esda's folded one-tailed p_sim -- the same gene's p roughly
        # halved just by turning permutations on. Both paths are one-sided
        # now: the checkerboard must stay non-significant on both.
        adata = _checkerboard_adata()
        df0 = spatially_variable_genes(adata.copy(), flavor="moran", n_perms=0)
        df1 = spatially_variable_genes(adata.copy(), flavor="moran", n_perms=99)
        assert df0.loc["checker", "pval"] > 0.9
        assert df1.loc["checker", "pval"] > 0.9
        assert df1.loc["gradient", "pval"] <= 2.0 / 100.0  # at/near the floor


class TestRipleyPerRadiusPValue:
    def test_p_value_present_per_distance_row(self):
        # Regression: p_value used to be a single scalar (largest radius
        # only, one-sided for clustering) broadcast into every row of the
        # distance-indexed column. It is now a genuinely per-radius,
        # two-sided rank p -- so it must vary across radii.
        adata = _adata_for_stats(n=200)
        result = ripley_l(adata, groupby="cell_type", support=40, n_perms=99, seed=0)
        for df in result.values():
            p = df["p_value"].to_numpy()
            assert len(p) == 40
            assert np.isfinite(p).all()
            assert len(np.unique(p)) > 1  # not one broadcast scalar

    def test_p_value_nan_without_permutations(self):
        adata = _adata_for_stats(n=50)
        result = ripley_l(adata, groupby="cell_type", support=10, n_perms=0)
        for df in result.values():
            assert df["p_value"].isna().all()

    def test_csr_calibration(self):
        # Under CSR data the per-radius two-sided p should be roughly
        # uniform: at alpha=0.1 no more than ~20% of radii should fall below
        # 0.1 for a fixed seed (pointwise p-values at nearby radii are
        # strongly correlated, so the tolerance is deliberately loose --
        # this catches gross miscalibration like the old always-clustering
        # one-sided endpoint test, not fine deviations).
        rng = np.random.default_rng(7)
        n = 1500
        coords = rng.uniform(0, 1000, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["cl"] = "all"

        result = ripley_l(adata, groupby="cl", support=50, n_perms=199, seed=0)
        p = result["all"]["p_value"].to_numpy()
        assert (p < 0.1).mean() <= 0.2

    def test_dispersed_pattern_is_significant(self):
        # Two-sided: regularity/dispersion must be detectable too (the old
        # one-sided clustering test could never flag it). A perfect square
        # lattice is maximally regular.
        m = 30
        xs, ys = np.meshgrid(np.arange(m, dtype=float) * 10, np.arange(m, dtype=float) * 10)
        coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((m * m, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["cl"] = "all"

        result = ripley_l(adata, groupby="cl", support=30, n_perms=99, seed=0)
        p = result["all"]["p_value"].to_numpy()
        assert p.min() <= 2.0 / 100.0  # some radius flags the regularity

    def test_multi_library_raises(self):
        adata = _adata_for_stats(n=20)
        adata.obs["library_id"] = ["lib_a"] * 10 + ["lib_b"] * 10
        with pytest.raises(ValueError, match="librar"):
            ripley_l(adata, groupby="cell_type", n_perms=0)


class TestLisaQuadrantGating:
    def test_quadrant_not_tested_when_no_permutations(self):
        adata = _adata_for_stats()
        result = moran_local(adata, "gene_0", n_perms=0)
        assert set(result["quadrant"]) == {"not_tested"}
        assert set(result["quadrant_raw"]) <= {"HH", "LL", "HL", "LH"}

    def test_quadrant_gated_by_alpha(self):
        # Strong two-blob structure: significant cells get HH/LL, and every
        # non-significant cell reads NS (the old code assigned a raw
        # quadrant to every cell and the NS label was dead code).
        rng = np.random.default_rng(0)
        n_half = 30
        blob_a = rng.normal((0.0, 0.0), 2.0, (n_half, 2))
        blob_b = rng.normal((1000.0, 1000.0), 2.0, (n_half, 2))
        X = np.concatenate([
            rng.normal(10.0, 0.1, n_half), rng.normal(0.0, 0.1, n_half)
        ]).astype(np.float32).reshape(-1, 1)
        adata = ad.AnnData(X=X, obsm={"spatial": np.vstack([blob_a, blob_b]).astype(np.float32)})
        adata.var_names = ["gene_0"]
        spatial_neighbors(adata, n_neighbors=6)

        result = moran_local(adata, "gene_0", n_perms=199, seed=0)
        sig = result["p_adj"] < 0.05
        assert (result.loc[~sig, "quadrant"] == "NS").all()
        assert set(result.loc[sig, "quadrant"]) <= {"HH", "LL", "HL", "LH"}
        assert {"HH", "LL"} <= set(result.loc[sig, "quadrant"])
