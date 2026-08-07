"""Tests for the n_jobs parallelism added to permutation-based spatial stats."""
import warnings

import numpy as np
import pandas as pd
import anndata as ad
import pytest
import scipy.sparse as sp

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._spatial_stats import (
    moran_local,
    moran_local_bv,
    getis_ord,
    ripley_l,
    spatially_variable_genes,
    spatial_silhouette,
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
        # this raised AttributeError until fixed.
        adata = _adata_for_stats()
        result = getis_ord(adata, "gene_0", n_perms=0)
        assert len(result) == adata.n_obs
        assert not result["Gs"].isna().any()
        assert not result["z_sim"].isna().any()
        assert not result["p_sim"].isna().any()
        # n_perms=0 skips FDR correction entirely (p_adj falls back to the raw
        # p-value) -- also part of the same fix, worth pinning down here.
        np.testing.assert_allclose(result["p_adj"].to_numpy(), result["p_sim"].to_numpy())

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


class TestRipleyPlotDetrending:
    def test_plot_does_not_double_subtract_distance(self):
        # Regression: tl.ripley_l's stored L_obs/L_hi/L_lo columns are
        # already the detrended L(d) - d quantity (`_l()` in
        # _spatial_stats.py subtracts `radii` before returning; L_exp is
        # hardcoded to zeros for exactly this reason -- raw L(d) is not
        # expected to be 0 under CSR, only detrended L(d)-d is). pl.ripley_plot
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
