"""Tests for pp.normalize_by_area (batch-5 bugfix: linear-scale correction)."""
import numpy as np
import anndata as ad
import pytest
import scipy.sparse as sp

from astrolabe.pp._normalize import normalize_by_area


def _adata_with_area(n=200, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    # Simulate log1p-normalized expression whose *linear*-scale mean
    # genuinely does grow with cell area (bigger cells really do capture
    # more transcripts) -- the scenario normalize_by_area is meant to
    # correct for.
    area = rng.uniform(50, 500, n).astype(np.float64)
    base_rate = area / 200.0  # bigger cells -> higher linear-scale counts
    counts = rng.poisson(base_rate[:, None] * rng.uniform(0.5, 2.0, n_genes)[None, :])
    X = np.log1p(counts).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cell_area"] = area
    return adata, area


def _residual_area_dependence(values: np.ndarray, area: np.ndarray) -> float:
    """abs(Pearson correlation) between per-cell total expression and area."""
    totals = values.sum(axis=1)
    return abs(np.corrcoef(totals, area)[0, 1])


class TestNormalizeByArea:
    def test_reduces_area_dependence_not_worsens_it(self):
        # Regression: dividing already-log1p-transformed values directly by
        # raw linear-scale area (the old behavior) is not a valid size
        # correction under any generative model -- it empirically produced
        # a *stronger* residual area-dependence (a spurious 1/area
        # hyperbola) than no correction at all. The fix does the division on
        # the linear (expm1'd) scale, where "counts per um^2" is a real
        # density, then re-applies log1p.
        adata, area = _adata_with_area()
        before_corr = _residual_area_dependence(adata.X, area)

        normalize_by_area(adata)
        after_corr = _residual_area_dependence(adata.X, area)

        assert after_corr < before_corr

    def test_output_stays_non_negative_and_log_scale(self):
        adata, area = _adata_with_area(n=50, n_genes=5)
        original_max = adata.X.max()
        normalize_by_area(adata)
        assert (adata.X >= 0).all()
        # Dividing linear-scale counts by area (>=50 here) then log1p'ing
        # again should not *inflate* values past the original log1p scale.
        assert adata.X.max() <= original_max + 1e-3

    def test_sparse_and_dense_agree(self):
        adata_dense, area = _adata_with_area(n=60, n_genes=8, seed=3)
        adata_sparse = adata_dense.copy()
        adata_sparse.X = sp.csr_matrix(adata_sparse.X)

        normalize_by_area(adata_dense)
        normalize_by_area(adata_sparse)

        np.testing.assert_allclose(
            adata_dense.X, np.asarray(adata_sparse.X.todense()), atol=1e-4
        )

    def test_raises_on_nonpositive_area(self):
        adata, _ = _adata_with_area(n=10, n_genes=3)
        adata.obs.loc[adata.obs_names[0], "cell_area"] = 0.0
        with pytest.raises(ValueError):
            normalize_by_area(adata)
