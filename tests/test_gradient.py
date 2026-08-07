"""Tests for the n_jobs parallelism added to gradient_significance."""
import numpy as np
import anndata as ad

from astrolabe.tl._graph import spatial_neighbors
from astrolabe.tl._gradient import expression_gradient, gradient_significance


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
