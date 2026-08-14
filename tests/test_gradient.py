"""Tests for the n_jobs parallelism added to gradient_significance, and for
the batch-3 bugfix pass (mask_threshold, _edge_weights, gradient_significance
arctan2 artifact, compartment_flux NaN-wraparound, spatial_diffusion, and the
expression_gradient key_added collision)."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import scipy.sparse as sp

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._gradient import (
    expression_gradient,
    gradient_field,
    gradient_significance,
    compartment_flux,
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


class TestCompartmentFluxNaNWraparound:
    def _flux_adata(self):
        # 5 cells: 0,1 = "A"; 2,3 = "B"; 4 = NaN label. Chain graph
        # 0-1-2-3-4 (each connected to its immediate neighbor(s)).
        n = 5
        row = [0, 1, 1, 2, 2, 3, 3, 4]
        col = [1, 0, 2, 1, 3, 2, 4, 3]
        conn = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obs["ct"] = pd.Categorical(["A", "A", "B", "B", np.nan])
        adata.obsm["spatial"] = np.array(
            [[0.0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=np.float32
        )
        adata.obsp["spatial_neighbors_connectivities"] = conn
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
        # Constant gradient pointing in +x for every cell.
        adata.obsm["gradient_g"] = np.tile(np.array([1.0, 0.0], dtype=np.float32), (n, 1))
        return adata

    def test_nan_labeled_cell_does_not_corrupt_last_category(self):
        adata = self._flux_adata()
        compartment_flux(adata, groupby="ct", keys="g")
        df = adata.uns["compartment_flux"]["flux"]["g"]
        # Categories are "A", "B" (alphabetical -- "B" sorts last). The only
        # real cross-boundary edge is 1<->2 (A<->B). The NaN cell (4) borders
        # only "B" (cell 3) -- under the old bug, that edge's code -1 wrapped
        # around to index the *last* category ("B"), fabricating a spurious
        # B->B "self flux" entry from a NaN-touching edge. Cell 3 (also "B")
        # has no other same-category neighbor, so B->B should have no data
        # (NaN) once NaN-touching edges are correctly excluded.
        assert np.isnan(df.loc["B", "B"])
        assert not np.isnan(df.loc["A", "B"])


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

    def test_compartment_flux_raises(self):
        adata = self._multi_library_adata()
        adata.obs["ct"] = "A"
        with pytest.raises(ValueError, match="librar"):
            compartment_flux(adata, "ct", "gene_0")

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
