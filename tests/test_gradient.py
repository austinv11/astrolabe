"""Tests for tl._gradient: regression tests from earlier bugfix passes plus —
after the methodology overhaul — ground-truth recovery (planted linear
gradient, planted radial decay, planted interface step, planted peak →
negative divergence) and Type-I calibration of gradient_significance's
autocorrelation-matched surrogate null."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import scipy.sparse as sp

import astrolabe as ab
from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._gradient import (
    bearing_to_group,
    diffuse_expression,
    expression_gradient,
    gradient_field,
    gradient_significance,
    interface_flux,
    spatial_diffusion,
    decay_length,
    _mean_resultant_length,
    _edge_weights,
)


def _adata_for_gradient(n=40, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    # A field with a real spatial gradient so magnitude/coherence are nonzero.
    expr = (coords[:, 0] / 500.0 + rng.normal(0, 0.05, n)).astype(np.float32)
    adata = ad.AnnData(X=expr.reshape(-1, 1))
    adata.var_names = ["gene_0"]
    adata.obsm["spatial"] = coords
    spatial_neighbors(adata, n_neighbors=6)
    expression_gradient(adata, "gene_0")
    return adata


class TestGradientSignificanceNJobs:
    def test_n_jobs_matches_sequential(self):
        adata1 = _adata_for_gradient()
        adata2 = _adata_for_gradient()
        gradient_significance(adata1, "gene_0", n_perms=30, random_state=0, n_jobs=1)
        gradient_significance(adata2, "gene_0", n_perms=30, random_state=0, n_jobs=2)
        r1 = adata1.uns["gradient_significance"]
        r2 = adata2.uns["gradient_significance"]
        np.testing.assert_allclose(r1["p_magnitude"], r2["p_magnitude"])
        np.testing.assert_allclose(r1["p_coherence"], r2["p_coherence"])

    def test_n_jobs_minus_one_runs(self):
        adata = _adata_for_gradient()
        gradient_significance(adata, "gene_0", n_perms=20, n_jobs=-1)
        result = adata.uns["gradient_significance"]
        assert "gene_0" in result.index


class TestBearingToGroup:
    def _adata(self):
        # two landmark nests far apart + query cells near each nest: the
        # bearing must target the NEAREST nest, not the group centroid
        coords = np.array([
            [0.0, 0.0], [1.0, 0.0],          # nest cells (label T), region 1
            [1000.0, 1000.0],                # nest cell (label T), region 2
            [10.0, 0.0],                     # query near region 1
            [1000.0, 990.0],                 # query near region 2
        ], dtype=np.float32)
        adata = ad.AnnData(X=np.zeros((5, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        adata.obs["leiden"] = pd.Categorical(["T", "T", "T", "other", "other"])
        return adata

    def test_bearing_targets_nearest_member_not_centroid(self):
        adata = self._adata()
        bearing_to_group(adata, "leiden", "T")
        b = adata.obs["bearing_to_T"].to_numpy()

        # query at (10, 0): nearest member is (1, 0) -> bearing = pi (due West).
        # The group centroid is (~333, 333) -> bearing ~ +45 deg; must NOT match.
        assert b[3] == pytest.approx(np.pi, abs=0.01)
        # query at (1000, 990): nearest member is (1000, 1000) -> bearing = +pi/2
        assert b[4] == pytest.approx(np.pi / 2, abs=0.01)

    def test_member_cells_are_nan(self):
        adata = self._adata()
        bearing_to_group(adata, "leiden", "T")
        b = adata.obs["bearing_to_T"].to_numpy()
        assert np.isnan(b[:3]).all()
        assert np.isfinite(b[3:]).all()

    def test_missing_group_raises(self):
        adata = self._adata()
        with pytest.raises(ValueError, match="no cells"):
            bearing_to_group(adata, "leiden", "missing")


class TestDiffuseExpression:
    def _adata(self, n=60, seed=0):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        expr = (coords[:, 0] / 500.0 + rng.normal(0, 0.2, n)).astype(np.float32)
        adata = ad.AnnData(X=expr.reshape(-1, 1))
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)
        return adata

    def test_constant_field_is_invariant(self):
        adata = self._adata()
        adata.X = np.full_like(adata.X, 3.0)
        diffuse_expression(adata, "gene_0", n_steps=5)
        np.testing.assert_allclose(
            adata.obs["gene_0_diffused"].to_numpy(), 3.0, rtol=1e-10
        )

    def test_zero_steps_is_identity(self):
        adata = self._adata()
        diffuse_expression(adata, "gene_0", n_steps=0)
        np.testing.assert_allclose(
            adata.obs["gene_0_diffused"].to_numpy(),
            adata.X[:, 0],
            rtol=1e-6,
        )

    def test_matches_hand_rolled_coifman_lafon(self):
        # Pin the operator: the public smoother must reproduce the exact
        # alpha=1 construction the tutorial used to hand-roll
        # (K / (q_i q_j), then row-normalize, then M^n f).
        adata = self._adata()
        diffuse_expression(adata, "gene_0", n_steps=5)

        conn = adata.obsp["spatial_neighbors_connectivities"].astype(np.float64)
        q = np.asarray(conn.sum(axis=1)).ravel()
        q[q == 0] = 1.0
        K_hat = sp.diags(1.0 / q) @ conn @ sp.diags(1.0 / q)
        rs = np.asarray(K_hat.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        M = sp.diags(1.0 / rs) @ K_hat
        f = adata.X[:, 0].astype(np.float64)
        for _ in range(5):
            f = M @ f

        np.testing.assert_allclose(
            adata.obs["gene_0_diffused"].to_numpy(), f.astype(np.float32), rtol=1e-6
        )

    def test_coifman_lafon_differs_from_row_on_varying_density(self):
        # a dense clump + sparse remainder: density correction must matter
        rng = np.random.default_rng(1)
        clump = rng.uniform(0, 50, (80, 2))
        sparse_pts = rng.uniform(0, 500, (40, 2))
        coords = np.vstack([clump, sparse_pts]).astype(np.float32)
        expr = (coords[:, 0] / 500.0).astype(np.float32)
        adata = ad.AnnData(X=expr.reshape(-1, 1))
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)

        diffuse_expression(adata, "gene_0", n_steps=3, suffix="_cl")
        diffuse_expression(adata, "gene_0", n_steps=3, normalize="row", suffix="_row")
        assert not np.allclose(
            adata.obs["gene_0_cl"].to_numpy(), adata.obs["gene_0_row"].to_numpy()
        )

    def test_mudata_path_and_suffix(self):
        import mudata as md

        adata = self._adata()
        mdata = md.MuData({"xen": adata})
        mdata.uns["default_modality"] = "xen"
        diffuse_expression(mdata, "gene_0", n_steps=2, suffix="_smooth")
        assert "gene_0_smooth" in mdata["xen"].obs

    def test_unknown_normalize_raises(self):
        adata = self._adata()
        with pytest.raises(ValueError, match="normalize"):
            diffuse_expression(adata, "gene_0", normalize="sqrt")


class TestDirectionNaNForZeroGradients:
    def test_constant_gene_direction_all_nan(self):
        # A spatially constant gene has a zero gradient everywhere; a zero
        # vector has no direction, so the stored direction must be NaN --
        # arctan2(0, 0) = 0 used to pile every degenerate cell onto bearing 0
        # (visible as a direction median of exactly 0.0 in .describe()).
        rng = np.random.default_rng(4)
        n = 40
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=np.full((n, 1), 2.5, dtype=np.float32))
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)

        expression_gradient(adata, "gene_0")

        assert (adata.obs["gradient_gene_0_magnitude"].to_numpy() <= 1e-8).all()
        assert adata.obs["gradient_gene_0_direction"].isna().all()

    def test_real_gradient_directions_stay_finite(self):
        adata = _adata_for_gradient()
        mag = adata.obs["gradient_gene_0_magnitude"].to_numpy()
        direction = adata.obs["gradient_gene_0_direction"].to_numpy()
        nonzero = mag > 1e-8
        assert nonzero.any()
        assert np.isfinite(direction[nonzero]).all()
        assert np.isnan(direction[~nonzero]).all()


class TestGradientSignificanceLoggingQuiet:
    def test_no_per_surrogate_info_lines(self, caplog):
        import logging as _logging
        import re

        adata = _adata_for_gradient()
        with caplog.at_level(_logging.INFO, logger="astrolabe.tl._gradient"):
            gradient_significance(adata, "gene_0", n_perms=25, n_jobs=1)

        per_surrogate = [
            r.getMessage() for r in caplog.records
            if re.search(r"surrogate \d+/\d+", r.getMessage())
        ]
        assert per_surrogate == [], per_surrogate


class TestGradientSignificanceExcludeUnreliable:
    def test_flagged_cells_excluded_from_observed_stats(self):
        def _prepared():
            adata = _adata_for_gradient()
            flag = np.zeros(adata.n_obs, dtype=bool)
            flag[:5] = True
            adata.obs["gradient_unreliable"] = flag
            # plant absurd gradients on the flagged cells: excluded -> no
            # effect on observed magnitude; included -> dominates it
            adata.obsm["gradient_gene_0"] = adata.obsm["gradient_gene_0"].copy()
            adata.obsm["gradient_gene_0"][:5] = 1000.0
            return adata

        a_excl = _prepared()
        gradient_significance(a_excl, "gene_0", n_perms=10, random_state=0)
        a_incl = _prepared()
        gradient_significance(
            a_incl, "gene_0", n_perms=10, random_state=0, exclude_unreliable=False
        )

        m_excl = float(a_excl.uns["gradient_significance"].loc["gene_0", "magnitude"])
        m_incl = float(a_incl.uns["gradient_significance"].loc["gene_0", "magnitude"])
        assert m_incl > m_excl * 10


class TestExpressionGradientKeyAddedCollision:
    def test_multi_gene_call_with_key_added_does_not_collide(self):
        # Regression: `base = key_added or f"gradient_{key}"` reused the
        # exact same obsm key for every gene in a multi-key call whenever
        # key_added was explicitly given -- only the last gene's gradient
        # survived the loop, silently overwriting every earlier one.
        rng = np.random.default_rng(1)
        n = 40
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        expr = np.column_stack([
            coords[:, 0] / 500.0,
            coords[:, 1] / 500.0,
        ]).astype(np.float32)
        adata = ad.AnnData(X=expr)
        adata.var_names = ["gene_x", "gene_y"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)

        expression_gradient(adata, ["gene_x", "gene_y"], key_added="custom")

        assert "custom_gene_x" in adata.obsm
        assert "custom_gene_y" in adata.obsm
        # gene_x's gradient points mostly in x; gene_y's mostly in y --
        # distinct obsm entries should therefore have distinct values.
        assert not np.allclose(adata.obsm["custom_gene_x"], adata.obsm["custom_gene_y"])

    def test_gradient_field_reads_custom_gradient_key_added(self):
        adata = _adata_for_gradient()
        # Overwrite with a custom key_added, matching gradient_field's own
        # gradient_key_added parameter -- downstream consumers used to
        # always hardcode f"gradient_{key}" regardless of what
        # expression_gradient actually used to write obsm.
        expression_gradient(adata, "gene_0", key_added="custom")
        assert "custom_gene_0" in adata.obsm
        gradient_field(adata, "gene_0", n_grid=10, gradient_key_added="custom")
        assert "gradient_field_gene_0" in adata.uns

    def test_gradient_field_raises_helpful_error_without_gradient_key_added(self):
        rng = np.random.default_rng(9)
        n = 30
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=(coords[:, :1] / 500.0).astype(np.float32))
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)
        expression_gradient(adata, "gene_0", key_added="custom")
        assert "gradient_gene_0" not in adata.obsm  # only the custom-prefixed key exists
        with pytest.raises(KeyError):
            gradient_field(adata, "gene_0", n_grid=10)  # missing gradient_key_added="custom"


class TestMeanResultantLengthExcludesDegenerateGradients:
    def test_zero_magnitude_gradients_do_not_inflate_coherence(self):
        # Regression: arctan2(0, 0) = 0 is a real, defined value -- cells
        # with a degenerate zero gradient (routine in diffusion-smoothed
        # sparse regions) all "point" at angle exactly 0 and used to pile up
        # in the mean resultant length, inflating R̄ even when the real
        # (nonzero-magnitude) gradients have no coherent direction at all.
        rng = np.random.default_rng(0)
        n_real = 200
        angles = rng.uniform(-np.pi, np.pi, n_real)  # uniform -> true R̄ near 0
        real_grad = np.column_stack([np.cos(angles), np.sin(angles)])
        zero_grad = np.zeros((n_real, 2))
        mixed = np.vstack([real_grad, zero_grad])

        r_fixed = _mean_resultant_length(mixed)
        # Naive (unfiltered, reproducing the old bug): treat every zero
        # vector's arctan2(0,0)=0 as a real direction.
        naive_angles = np.arctan2(mixed[:, 1], mixed[:, 0])
        r_naive = float(np.abs(np.mean(np.exp(1j * naive_angles))))

        assert r_fixed < 0.25  # close to the true ~0 coherence of `angles`
        assert r_naive > r_fixed + 0.15  # the bug measurably inflates it

    def test_all_zero_gradients_return_nan(self):
        grad = np.zeros((10, 2))
        assert np.isnan(_mean_resultant_length(grad))


class TestEdgeWeightsRecomputedFromCoords:
    def test_no_1e8_weight_for_edges_missing_from_distance_matrix(self):
        # Regression: _edge_weights used to read distances from a stored
        # obsp distance matrix; a connectivity edge with no matching
        # distance-matrix entry (routine under tl.blend_neighbors, a joint
        # spatial+expression graph) read as a stored 0.0, giving
        # 1/(0+1e-8)=1e8 -- six orders of magnitude too large. Distances are
        # now always recomputed directly from coordinates, so this failure
        # mode can no longer occur structurally.
        rng = np.random.default_rng(2)
        n = 30
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["spatial"] = coords
        # A connectivity graph with no matching distances_key at all in uns
        # -- the scenario that used to silently fall through to reading a
        # (nonexistent -> all-zero) sparse distance matrix.
        conn = sp.random(n, n, density=0.2, format="csr", random_state=0)
        conn.setdiag(0)
        conn.eliminate_zeros()
        adata.obsp["spatial_neighbors_connectivities"] = conn

        src, dst, w = _edge_weights(coords.astype(np.float64), adata, "spatial_neighbors_connectivities", True)
        assert len(w) == conn.nnz
        assert np.all(w < 1e6)  # nowhere near the old 1e8 bug


class TestGradientFieldMaskThreshold:
    def test_default_threshold_gives_nonempty_mask_on_realistic_density(self):
        # Regression: mask_threshold=-5.0 (absolute log-density) was
        # unsatisfiable at realistic KDE log-density scales (-12 to -18 for
        # real Xenium data) -- U/V/divergence/curl came back entirely NaN,
        # and pl.gradient_streamlines/divergence_plot rendered silently
        # empty with no error. mask_threshold is now a *percentile* of the
        # KDE evaluated at actual cell locations, so it's satisfiable by
        # construction regardless of the absolute density scale.
        rng = np.random.default_rng(3)
        n = 300
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        expr = (coords[:, 0] / 500.0 + rng.normal(0, 0.05, n)).astype(np.float32)
        adata = ad.AnnData(X=expr.reshape(-1, 1))
        adata.var_names = ["gene_0"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)
        expression_gradient(adata, "gene_0")

        gradient_field(adata, "gene_0", n_grid=20)
        result = adata.uns["gradient_field_gene_0"]
        assert result["mask"].mean() > 0.5


def _interface_adata(seed=0, jitter=0.5, nan_cell=False):
    """Two cell-type blocks on a jittered grid: 'A' at x<50, 'B' at x>=50,
    with a linear expression ramp f = x/100 (true df/dn across the A|B
    interface = +0.01 per µm, oriented A→B)."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(10) * 10.0 + 5, np.arange(10) * 10.0 + 5)
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    coords = coords + rng.uniform(-jitter, jitter, coords.shape)
    labels = np.where(coords[:, 0] < 50, "A", "B").astype(object)
    if nan_cell:
        labels[0] = np.nan
    f = (coords[:, 0] / 100.0).astype(np.float32)
    adata = ad.AnnData(X=f.reshape(-1, 1))
    adata.var_names = ["g"]
    adata.obs["ct"] = pd.Categorical(labels)
    adata.obsm["spatial"] = coords.astype(np.float32)
    spatial_neighbors(adata, n_neighbors=6)
    expression_gradient(adata, "g", n_diffusion_steps=0)
    return adata


class TestInterfaceFlux:
    def test_planted_step_sign_magnitude_and_antisymmetry(self):
        adata = _interface_adata()
        interface_flux(adata, groupby="ct", keys="g")
        out = adata.uns["interface_flux"]
        flux = out["integrated_flux"]["g"]
        deriv = out["mean_normal_derivative"]["g"]
        length = out["interface_length"]

        # expression rises A→B, so integrated flux A→B is positive and the
        # matrix is exactly antisymmetric (an identity, not a finding)
        assert flux.loc["A", "B"] > 0
        assert flux.loc["B", "A"] == pytest.approx(-flux.loc["A", "B"])

        # length-weighted mean normal derivative recovers the planted slope
        assert deriv.loc["A", "B"] == pytest.approx(0.01, rel=0.3)

        # the A|B interface is a ~100 µm vertical line of shared facets
        assert length.loc["A", "B"] == pytest.approx(100.0, rel=0.3)

        # integrated flux ≈ derivative × interface length
        assert flux.loc["A", "B"] == pytest.approx(
            deriv.loc["A", "B"] * length.loc["A", "B"]
        )

    def test_nan_labeled_cells_are_excluded(self):
        adata = _interface_adata(nan_cell=True)
        interface_flux(adata, groupby="ct", keys="g")
        flux = adata.uns["interface_flux"]["integrated_flux"]["g"]
        # only categories A and B appear; the NaN cell contributes nothing
        assert list(flux.index) == ["A", "B"]
        assert np.isfinite(flux.loc["A", "B"])


class TestSpatialDiffusion:
    def _diffusion_adata(self, n=40, n_genes=5, seed=0):
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        X = rng.random((n, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=6)
        return adata

    def test_zero_variance_gene_scores_nan_not_maximum(self):
        # Regression: a zero-variance (undetected/constant) gene has no
        # signal to diffuse -- without the NaN guard it would score as a
        # real (and, under this metric, minimum-possible) 0 persistence for
        # a gene that was never actually tested. NaN correctly excludes it
        # instead of reporting a number that looks meaningful but isn't.
        adata = self._diffusion_adata()
        adata.X[:, 0] = 0.0  # gene_0 is now constant/undetected
        spatial_diffusion(adata, diffusion_score=True)
        scores = adata.var["diffusion_score"]
        assert np.isnan(scores.iloc[0])
        assert not scores.iloc[1:].isna().any()

    def test_censored_genes_score_at_max_diffusion_steps(self):
        # A gene that never converges within max_diffusion_steps is
        # right-censored at that value rather than left unbounded or NaN --
        # it did fail to converge, but that's a real (if imprecise) signal
        # ("at least this persistent"), not a missing measurement. An
        # unreachably tight tolerance forces every real gene to censor.
        adata = self._diffusion_adata()
        spatial_diffusion(
            adata, diffusion_score=True, max_diffusion_steps=3, diffusion_convergence_tol=1e-12
        )
        scores = adata.var["diffusion_score"].to_numpy()
        np.testing.assert_allclose(scores, 3.0, atol=1e-10)

    def test_higher_tolerance_converges_no_later_than_lower_tolerance(self):
        # A looser convergence tolerance should never take *more* steps to
        # satisfy than a tighter one on the same data -- the convergence
        # step count is monotonic in the tolerance.
        adata_loose = self._diffusion_adata()
        spatial_diffusion(
            adata_loose, diffusion_score=True, max_diffusion_steps=200, diffusion_convergence_tol=0.5,
        )
        loose_scores = adata_loose.var["diffusion_score"].to_numpy()

        adata_tight = self._diffusion_adata()
        spatial_diffusion(
            adata_tight, diffusion_score=True, max_diffusion_steps=200, diffusion_convergence_tol=0.05,
        )
        tight_scores = adata_tight.var["diffusion_score"].to_numpy()

        assert np.all(loose_scores <= tight_scores)
        assert np.any(loose_scores < tight_scores)

    def test_alpha_one_uses_full_density_normalization_not_sqrt(self):
        # Regression: A_hat used D^{-1/2} A D^{-1/2} (alpha=1/2) while the
        # docstring claims alpha=1 (D^{-1} A D^{-1}). End-to-end check on a
        # graph where node 1 has two neighbors of *different* degree (node 0:
        # degree 2; node 2: degree 4) -- alpha changes the row-normalized
        # diffusion operator M, and therefore how many steps it takes each
        # gene's pattern to reach uniformity. A triangle (0-1-2) plus two
        # pendants off node 2 (2-3, 2-4) is used rather than a simple path
        # or star: both of those turned out to be either alpha-invariant
        # after row-normalization or bipartite (a bipartite row-normalized
        # operator has eigenvalue -1, so f_t oscillates forever between two
        # states instead of converging to a fixed point at all) -- this
        # topology has neither degeneracy.
        n = 5
        row = [0, 1, 1, 2, 2, 0, 2, 3, 2, 4]
        col = [1, 0, 2, 1, 0, 2, 3, 2, 4, 2]
        conn = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        rng = np.random.default_rng(0)
        X = rng.random((n, 2)).astype(np.float32)
        adata = ad.AnnData(X=X.copy())
        adata.var_names = ["g0", "g1"]
        adata.obsm["spatial"] = np.arange(n, dtype=np.float32).reshape(-1, 1) * np.array([[1.0, 0.0]])
        adata.obsp["spatial_neighbors_connectivities"] = conn
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}

        max_steps, tol = 50, 0.3
        spatial_diffusion(
            adata, diffusion_score=True, max_diffusion_steps=max_steps, diffusion_convergence_tol=tol
        )
        actual = adata.var["diffusion_score"].to_numpy()

        def _manual_scores(alpha_sqrt: bool) -> np.ndarray:
            d = np.asarray(conn.sum(axis=1), dtype=np.float64).ravel()
            D_inv = sp.diags(1.0 / np.sqrt(d) if alpha_sqrt else 1.0 / d)
            A_hat = D_inv @ conn @ D_inv
            row_sums = np.asarray(A_hat.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1.0
            M = sp.diags(1.0 / row_sums) @ A_hat
            mat = X.astype(np.float64)
            target = mat.mean(axis=0)
            scale = np.abs(mat - target).mean(axis=0)
            scores = np.full(mat.shape[1], float(max_steps), dtype=np.float64)
            remaining = np.ones(mat.shape[1], dtype=bool)
            f_t = mat.copy()
            for t in range(1, max_steps + 1):
                if not remaining.any():
                    break
                f_t = M @ f_t
                rel_dev = np.abs(f_t - target).mean(axis=0) / scale
                newly_converged = remaining & (rel_dev < tol)
                scores[newly_converged] = float(t)
                remaining &= ~newly_converged
            return scores

        expected_alpha1 = _manual_scores(alpha_sqrt=False)
        expected_alpha_half = _manual_scores(alpha_sqrt=True)
        assert not np.allclose(expected_alpha1, expected_alpha_half)
        np.testing.assert_allclose(actual, expected_alpha1)
        assert not np.allclose(actual, expected_alpha_half)


class TestGradientFunctionsRejectMultiLibrary:
    # Regression (batch 7): tl/_gradient.py had no multi-library guard at all,
    # despite CLAUDE.md documenting one -- a concat_samples()-produced object
    # would silently compute spatially-meaningless gradients mixing unrelated
    # tissue sections instead of raising early.
    def _multi_library_adata(self, n=40, seed=0):
        adata = _adata_for_gradient(n=n, seed=seed)
        half = n // 2
        adata.obs["library_id"] = ["lib_a"] * half + ["lib_b"] * (n - half)
        return adata

    def test_expression_gradient_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            expression_gradient(adata, "gene_0")

    def test_gradient_field_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            gradient_field(adata, "gene_0")

    def test_gradient_significance_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            gradient_significance(adata, "gene_0", n_perms=5)

    def test_interface_flux_raises(self):
        adata = self._multi_library_adata()
        adata.obs["ct"] = "A"
        with pytest.raises(ValueError, match="librar"):
            interface_flux(adata, "ct", "gene_0")

    def test_spatial_diffusion_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            spatial_diffusion(adata)

    def test_decay_length_raises(self):
        adata = self._multi_library_adata()
        with pytest.raises(ValueError, match="librar"):
            decay_length(adata, "gene_0", source=0)

    def test_single_library_still_works(self):
        # library_id present but only one unique value -- must not raise.
        adata = _adata_for_gradient()
        adata.obs["library_id"] = "only_lib"
        gradient_significance(adata, "gene_0", n_perms=5)
        assert "gradient_significance" in adata.uns


class TestGroundTruthRecovery:
    def test_planted_linear_gradient_direction_and_magnitude(self):
        # expr = x/500 → true gradient (1/500, 0): direction 0 rad, magnitude
        # 0.002. The fixture has always planted this; now we assert against it.
        rng = np.random.default_rng(0)
        n = 400
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        expr = (coords[:, 0] / 500.0).astype(np.float32)
        adata = ad.AnnData(X=expr.reshape(-1, 1))
        adata.var_names = ["g"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=8)
        expression_gradient(adata, "g", n_diffusion_steps=0)

        reliable = ~adata.obs["gradient_unreliable"].to_numpy()
        grad = adata.obsm["gradient_g"][reliable]
        mean_dir = float(np.angle(np.mean(np.exp(1j * np.arctan2(grad[:, 1], grad[:, 0])))))
        assert abs(mean_dir) < 0.1  # ≈ 0 rad (pointing +x)
        assert np.median(np.linalg.norm(grad, axis=1)) == pytest.approx(0.002, rel=0.15)

    def test_planted_radial_decay_recovers_lambda_and_background(self):
        # f(r) = 2·e^(−r/80) + 0.5 — the background offset is exactly what an
        # offset-free model can only absorb by driving λ → ∞.
        rng = np.random.default_rng(1)
        n = 4000
        coords = rng.uniform(0, 600, (n, 2)).astype(np.float32)
        center = np.array([300.0, 300.0])
        r = np.linalg.norm(coords - center, axis=1)
        f = 2.0 * np.exp(-r / 80.0) + 0.5 + rng.normal(0, 0.05, n)
        adata = ad.AnnData(X=f.reshape(-1, 1).astype(np.float32))
        adata.var_names = ["g"]
        adata.obsm["spatial"] = coords

        decay_length(adata, "g", source=(300.0, 300.0), max_dist_um=300.0)
        out = adata.uns["decay_length_g"]
        assert out["lambda_um"] == pytest.approx(80.0, rel=0.25)
        assert out["background"] == pytest.approx(0.5, abs=0.15)
        assert out["lambda_se_um"] < 0.5 * out["lambda_um"]

    def test_planted_peak_gives_negative_divergence(self):
        # Gaussian expression bump: gradients converge into the peak, so the
        # divergence (= ∇²f) is negative there. The old sign wording taught
        # the opposite reading.
        rng = np.random.default_rng(2)
        n = 2000
        coords = rng.uniform(0, 400, (n, 2)).astype(np.float32)
        center = np.array([200.0, 200.0])
        r2 = ((coords - center) ** 2).sum(axis=1)
        f = np.exp(-r2 / (2 * 60.0**2)).astype(np.float32)
        adata = ad.AnnData(X=f.reshape(-1, 1))
        adata.var_names = ["g"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=8)
        expression_gradient(adata, "g", n_diffusion_steps=0)

        gradient_field(adata, "g", n_grid=25)
        gf = adata.uns["gradient_field_g"]
        assert "curl" not in gf  # curl-free by construction; no such output
        assert "F" in gf
        div = gf["divergence"]
        xg, yg = gf["X_grid"], gf["Y_grid"]
        ci = np.argmin(np.abs(xg - 200.0))
        cj = np.argmin(np.abs(yg - 200.0))
        # 3×3 patch around the peak (rows index y, cols index x)
        patch = div[max(cj - 1, 0):cj + 2, max(ci - 1, 0):ci + 2]
        assert np.nanmean(patch) < 0


def _jittered_lattice(nx, ny, spacing=10.0, jitter=0.5, seed=0):
    """Square lattice of cells with small positional jitter; returns
    (adata, ix, iy) where ix/iy are the integer lattice indices."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    ix, iy = gx.ravel(), gy.ravel()
    coords = np.stack([ix * spacing, iy * spacing], axis=1).astype(np.float64)
    coords += rng.normal(0, jitter, coords.shape)
    adata = ad.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    adata.var_names = ["dummy"]
    adata.obsm["spatial"] = coords.astype(np.float32)
    spatial_neighbors(adata, n_neighbors=6)
    return adata, ix, iy


class TestBoundaryHandling:
    """The one-sidedness criterion + free-intercept fit (tissue-edge fix)."""

    @staticmethod
    def _star_quality(angles_deg, radius=10.0):
        # _neighborhood_quality on a single center cell with equal-distance,
        # equal-weight neighbors at the given bearings.
        from astrolabe.tl._gradient import _neighborhood_quality

        angles = np.deg2rad(np.asarray(angles_deg, dtype=float))
        nbrs = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        coords = np.vstack([[0.0, 0.0], nbrs])
        src = np.zeros(len(angles), dtype=np.intp)
        dst = np.arange(1, len(angles) + 1, dtype=np.intp)
        w = np.ones(len(angles), dtype=np.float64)
        return _neighborhood_quality(coords, src, dst, w)

    def test_symmetric_star_is_not_onesided(self):
        _, aniso, onesided = self._star_quality(np.arange(0, 360, 45))
        assert onesided[0] == pytest.approx(0.0, abs=1e-9)
        assert aniso[0] == pytest.approx(1.0, rel=1e-6)

    def test_half_plane_star_matches_closed_form_and_is_flagged(self):
        # 8 bearings midpoint-sampled over the half-plane [-90°, 90°]: the
        # continuous limit of the onesidedness statistic is 2/π ≈ 0.637.
        # The UNcentered eigenvalue ratio is blind to this geometry (it reads
        # a half-disc as isotropic) — the whole reason the criterion exists.
        angles = -90.0 + (np.arange(8) + 0.5) * (180.0 / 8)
        _, _, onesided = self._star_quality(angles)
        assert onesided[0] == pytest.approx(2 / np.pi, rel=0.05)
        assert onesided[0] > 0.45  # crosses the flag threshold

    def test_lattice_edges_flagged_interior_not(self):
        adata, ix, iy = _jittered_lattice(15, 15, seed=0)
        adata.obs["f"] = np.random.default_rng(0).standard_normal(adata.n_obs)
        expression_gradient(adata, "f", n_diffusion_steps=0)
        flag = adata.obs["gradient_unreliable"].to_numpy()

        corner = ((ix == 0) | (ix == 14)) & ((iy == 0) | (iy == 14))
        rim = (ix == 0) | (ix == 14) | (iy == 0) | (iy == 14)
        interior = (ix >= 3) & (ix <= 11) & (iy >= 3) & (iy <= 11)

        assert flag[corner].all(), "corner cells must be flagged"
        assert flag[rim].mean() >= 0.5, "most rim cells must be flagged"
        # Threshold-validation checkpoint: interiors must not over-flag.
        assert flag[interior].mean() < 0.10, (
            f"interior false-positive rate {flag[interior].mean():.2f}"
        )

    def test_edge_magnitudes_do_not_dominate_on_noise(self):
        # Planted pure-noise field (true gradient 0): every magnitude is
        # estimator noise. Under the old constrained-through-center fit,
        # unflagged edge cells ran 2-4x interior (center-noise amplification
        # through the neighborhood moment m); the free-intercept fit keeps
        # both at the same noise scale.
        adata, ix, iy = _jittered_lattice(20, 20, seed=1)
        adata.obs["noise"] = np.random.default_rng(1).standard_normal(adata.n_obs)
        expression_gradient(adata, "noise", n_diffusion_steps=0)
        mag = adata.obs["gradient_noise_magnitude"].to_numpy()
        flag = adata.obs["gradient_unreliable"].to_numpy()

        edge = (ix <= 1) | (ix >= 18) | (iy <= 1) | (iy >= 18)
        interior = (ix >= 3) & (ix <= 16) & (iy >= 3) & (iy <= 16)
        edge_sel = edge & ~flag
        interior_sel = interior & ~flag
        assert interior_sel.sum() > 0
        if edge_sel.sum() > 0:  # all-flagged edge band is also a pass
            assert mag[edge_sel].mean() < 1.5 * mag[interior_sel].mean()

    def test_affine_field_recovered_exactly_everywhere(self):
        # An exact plane is recovered exactly by the free-intercept fit even
        # at flagged one-sided cells — the nonzero offset exercises the
        # intercept term specifically.
        adata, _, _ = _jittered_lattice(15, 15, seed=2)
        xy = adata.obsm["spatial"]
        adata.obs["aff"] = 3.7 + 0.004 * xy[:, 0] - 0.002 * xy[:, 1]
        expression_gradient(adata, "aff", n_diffusion_steps=0)
        np.testing.assert_allclose(
            adata.obsm["gradient_aff"],
            np.tile([0.004, -0.002], (adata.n_obs, 1)),
            atol=1e-6,
        )

    def test_onesidedness_column_written(self):
        adata = _adata_for_gradient()
        col = adata.obs["gradient_onesidedness"].to_numpy()
        assert col.dtype == np.float32
        assert np.all((col >= 0) & (col <= 1))


class TestSignificanceCalibration:
    def _smooth_isotropic_adata(self, seed):
        # Spatially autocorrelated but directionally isotropic field: smoothed
        # white noise. The old CSR null called ANY such field "directionally
        # coherent" because the shuffle destroyed the autocorrelation.
        rng = np.random.default_rng(seed)
        n = 500
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.var_names = ["dummy"]
        adata.obsm["spatial"] = coords
        spatial_neighbors(adata, n_neighbors=8)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        noise = rng.standard_normal(n)
        from astrolabe.tl._gradient import _diffuse_expression
        adata.obs["smooth_field"] = _diffuse_expression(noise, conn, 8)
        return adata

    def test_smooth_isotropic_field_is_not_called_coherent(self):
        # Type-I control for the coherence test under the matched null.
        p_vals = []
        for seed in (0, 1, 2):
            adata = self._smooth_isotropic_adata(seed)
            expression_gradient(adata, "smooth_field", n_diffusion_steps=0)
            gradient_significance(
                adata, "smooth_field", n_perms=59, random_state=seed, n_jobs=1
            )
            p_vals.append(
                float(adata.uns["gradient_significance"].loc["smooth_field", "p_coherence"])
            )
        # An isotropic-but-smooth field must not systematically reject:
        # at least 2 of 3 replicates above 0.05, and never at the floor.
        floor = 1.0 / 60.0
        assert sum(p > 0.05 for p in p_vals) >= 2, p_vals
        assert all(p > floor for p in p_vals), p_vals

    def test_true_directional_field_is_detected(self):
        # Positive control: a global linear ramp has one tissue-wide bearing.
        rng = np.random.default_rng(3)
        n = 500
        coords = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.var_names = ["dummy"]
        adata.obsm["spatial"] = coords
        adata.obs["ramp"] = coords[:, 0] / 500.0 + rng.normal(0, 0.02, n)
        spatial_neighbors(adata, n_neighbors=8)
        expression_gradient(adata, "ramp", n_diffusion_steps=0)
        gradient_significance(adata, "ramp", n_perms=59, random_state=0, n_jobs=1)
        res = adata.uns["gradient_significance"].loc["ramp"]
        assert res["p_coherence"] == pytest.approx(1.0 / 60.0)
        assert res["coherence"] > res["coherence_null_mean"] + 0.2


class TestGroupbyRestriction:
    """A gradient fitted across a cell-type boundary measures composition, not a
    within-type gradient. groupby= is the fix, and these are the tests that show
    it fixes the right thing and leaves real gradients alone."""

    @staticmethod
    def _two_slabs(seed=0, n=2000):
        """Type A on the left, type B on the right, sharp boundary at x=50.

        LINEAGE is high in A and low in B but FLAT within each type, so any
        gradient it shows is the boundary. GRADIENT climbs along y identically
        in both types, so it must survive the restriction untouched.
        """
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 100, n)
        y = rng.uniform(0, 100, n)
        ct = np.where(x < 50, "A", "B")
        lineage = np.where(ct == "A", 8.0, 1.0) + rng.normal(0, 0.25, n)
        grad_g = 0.06 * y + rng.normal(0, 0.25, n)
        adata = ad.AnnData(
            np.column_stack([lineage, grad_g]).astype(np.float32),
            var=pd.DataFrame(index=["LINEAGE", "GRADIENT"]),
        )
        adata.obsm["spatial"] = np.column_stack([x, y]).astype(np.float32)
        adata.obs["celltype"] = pd.Categorical(ct)
        adata.uns["spatial"] = {"lib": {"coordinate_unit": "um"}}
        ab.tl.spatial_neighbors(adata, n_neighbors=8)
        return adata

    def test_within_type_gradient_removes_the_composition_artifact(self):
        adata = self._two_slabs()
        mags = {}
        for label, kw in (("pool", {}), ("within", {"groupby": "celltype"})):
            for g in ("LINEAGE", "GRADIENT"):
                ab.tl.expression_gradient(adata, g, key_added=label, **kw)
            ok = ~adata.obs["gradient_unreliable"].to_numpy()
            mags[label] = {
                g: float(np.nanmean(adata.obs[f"{label}_{g}_magnitude"][ok]))
                for g in ("LINEAGE", "GRADIENT")
            }
        # the flat-within-type gene loses most of its apparent gradient
        assert mags["within"]["LINEAGE"] < 0.5 * mags["pool"]["LINEAGE"]
        # the genuine gradient is essentially untouched
        ratio = mags["within"]["GRADIENT"] / mags["pool"]["GRADIENT"]
        assert 0.8 < ratio < 1.25, ratio

    def test_restriction_is_recorded_in_provenance(self):
        adata = self._two_slabs(seed=1)
        ab.tl.expression_gradient(adata, "GRADIENT", groupby="celltype")
        p = adata.uns["gradient_params"]["gradient_GRADIENT"]
        assert p["groupby"] == "celltype"
        assert p["groups"] is None

    def test_ungrouped_path_is_unchanged(self):
        """groupby=None must be bit-for-bit the previous behaviour."""
        adata = self._two_slabs(seed=2)
        ab.tl.expression_gradient(adata, "GRADIENT", key_added="a")
        ab.tl.expression_gradient(adata, "GRADIENT", groupby=None, key_added="b")
        np.testing.assert_array_equal(
            adata.obsm["a_GRADIENT"], adata.obsm["b_GRADIENT"]
        )

    def test_significance_reports_per_group(self):
        adata = self._two_slabs(seed=3)
        ab.tl.expression_gradient(adata, "GRADIENT", groupby="celltype")
        ab.tl.gradient_significance(
            adata, "GRADIENT", groupby="celltype", n_perms=49, random_state=0
        )
        res = adata.uns["gradient_significance"]
        assert list(res.index.names) == ["key", "group"]
        assert set(res.index.get_level_values("group")) == {"A", "B"}
        assert (res["n_cells"] > 100).all()
        # a real y-gradient present in both types must be coherent in both
        assert (res["coherence"] > 0.5).all()

    def test_diffusion_respects_the_boundary(self):
        adata = self._two_slabs(seed=4)
        near = np.abs(adata.obsm["spatial"][:, 0] - 50) < 4
        ab.tl.diffuse_expression(adata, "LINEAGE", n_steps=6, suffix="_pool")
        ab.tl.diffuse_expression(
            adata, "LINEAGE", n_steps=6, groupby="celltype", suffix="_within"
        )
        raw = float(adata.X[near, 0].std())
        pooled = float(adata.obs["LINEAGE_pool"][near].std())
        within = float(adata.obs["LINEAGE_within"][near].std())
        assert pooled < 0.7 * raw, "pooled diffusion should blur the boundary"
        assert within > 0.9 * raw, "within-group diffusion must preserve it"

    def test_groups_without_groupby_raises(self):
        adata = self._two_slabs(seed=5)
        with pytest.raises(ValueError, match="groups= requires groupby="):
            ab.tl.expression_gradient(adata, "GRADIENT", groups=["A"])

    def test_unknown_group_raises(self):
        adata = self._two_slabs(seed=6)
        with pytest.raises(ValueError, match="not present"):
            ab.tl.expression_gradient(
                adata, "GRADIENT", groupby="celltype", groups=["Z"]
            )
