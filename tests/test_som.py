"""Tests for tl._som (batch-6 bugfix pass: BMU memory blowup, sigma scaling)."""
import numpy as np
import anndata as ad
import pytest

from astrolabe.tl._som import _batch_bmu, _default_sigma

minisom = pytest.importorskip("minisom")
from minisom import MiniSom  # noqa: E402
from astrolabe.tl._som import som  # noqa: E402


class TestBatchBMUMemoryEfficient:
    def test_matches_naive_full_broadcast_computation(self):
        # Regression: the squared-distance-expansion rewrite must give
        # identical BMU assignments to the old (memory-blowing)
        # d[:, None, :] - w_flat[None, :, :] broadcast form -- this is a
        # pure performance/memory fix, not a behavior change.
        rng = np.random.default_rng(0)
        n, n_feat = 500, 12
        data = rng.normal(size=(n, n_feat)).astype(np.float32)
        som_obj = MiniSom(6, 6, n_feat, random_seed=0)
        som_obj.random_weights_init(data)

        result = _batch_bmu(som_obj, data, chunk=64)

        w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float32)
        naive = np.linalg.norm(
            data[:, None, :] - w_flat[None, :, :], axis=2
        ).argmin(axis=1).astype(np.int32)

        np.testing.assert_array_equal(result, naive)

    def test_chunking_does_not_change_result(self):
        # Different chunk sizes can route through BLAS GEMM with a
        # different internal blocking strategy, occasionally reordering
        # floating-point summation enough to flip an argmin between two
        # near-exactly-tied candidate nodes -- not a bug in the chunking
        # logic itself (verified below: any disagreement is between
        # distances that differ by a tiny fraction). This is the same class
        # of float non-determinism the direct `d @ w_flat.T` matmul always
        # has across different call shapes, independent of this rewrite.
        rng = np.random.default_rng(1)
        n, n_feat = 300, 8
        data = rng.normal(size=(n, n_feat)).astype(np.float32)
        som_obj = MiniSom(5, 5, n_feat, random_seed=0)
        som_obj.random_weights_init(data)

        full = _batch_bmu(som_obj, data, chunk=10_000)
        chunked = _batch_bmu(som_obj, data, chunk=17)
        mismatches = np.where(full != chunked)[0]
        assert len(mismatches) <= 3  # near-tie float noise only, not systematic

        w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float64)
        for i in mismatches:
            d = data[i].astype(np.float64)
            dists = np.linalg.norm(d[None, :] - w_flat, axis=1)
            top2 = np.sort(dists)[:2]
            assert top2[1] - top2[0] < 1e-3  # genuinely near-tied, not a real disagreement


class TestDefaultSigma:
    def test_scales_with_grid_size(self):
        assert _default_sigma(5, 5) == pytest.approx(1.25)
        assert _default_sigma(45, 45) == pytest.approx(11.25)

    def test_floors_at_one(self):
        assert _default_sigma(2, 2) == pytest.approx(1.0)


class TestSomSigmaDefault:
    def test_auto_sigma_scales_with_auto_grid_size(self):
        # Regression: a fixed sigma=1.0 regardless of grid size collapsed
        # topology preservation at any grid size the library's own
        # auto-selection picks for a large dataset.
        rng = np.random.default_rng(2)
        n = 400
        adata = ad.AnnData(X=rng.normal(size=(n, 10)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)

        som(adata, x=20, y=20, n_iter=200, seed=0)
        assert adata.uns["som"]["params"]["sigma"] == pytest.approx(_default_sigma(20, 20))

    def test_explicit_sigma_still_respected(self):
        rng = np.random.default_rng(3)
        n = 100
        adata = ad.AnnData(X=rng.normal(size=(n, 5)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(n, 5)).astype(np.float32)
        som(adata, x=6, y=6, sigma=2.5, n_iter=100, seed=0)
        assert adata.uns["som"]["params"]["sigma"] == pytest.approx(2.5)
