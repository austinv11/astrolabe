"""Tests for banksy_sweep X_pca save/restore (audit fix 2.7)
and the n_jobs parallelism added to its lambda sweep."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

import astrolabe.pp as pp
import astrolabe.tl as tl
from astrolabe.tl._banksy_sweep import banksy_sweep


def _adata_for_sweep(n=40, n_genes=30, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n, n_genes, density=0.5, format="csr", random_state=seed).astype(np.float32)
    adata = ad.AnnData(X=X, obsm={"spatial": rng.uniform(0, 500, (n, 2)).astype(np.float32)})
    tl.spatial_neighbors(adata, n_neighbors=5)
    pp.normalize_total(adata)
    pp.log1p(adata)
    return adata


class TestBanksySweepPCARestore:
    def test_restores_original_pca(self):
        adata = _adata_for_sweep()
        tl.pca(adata, n_comps=10)
        original_pca = adata.obsm["X_pca"].copy()
        banksy_sweep(adata, lambda_params=[0.0, 0.5], resolutions=[0.3])
        np.testing.assert_allclose(
            adata.obsm["X_pca"], original_pca, rtol=1e-6,
            err_msg="X_pca must be restored to original after sweep",
        )

    def test_no_pca_not_created(self):
        adata = _adata_for_sweep()
        assert "X_pca" not in adata.obsm
        banksy_sweep(adata, lambda_params=[0.0], resolutions=[0.3])
        assert "X_pca" not in adata.obsm, "sweep must not leave X_pca when user had none"

    def test_sweep_returns_dataframe(self):
        adata = _adata_for_sweep()
        result = banksy_sweep(adata, lambda_params=[0.0, 0.5], resolutions=[0.3, 0.5])
        import pandas as pd
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"lambda", "resolution", "n_clusters", "obs_key"}

    def test_sweep_labels_written_to_obs(self):
        adata = _adata_for_sweep()
        result = banksy_sweep(adata, lambda_params=[0.0], resolutions=[0.3])
        for obs_key in result["obs_key"]:
            assert obs_key in adata.obs.columns, f"obs key '{obs_key}' missing"

    def test_uns_result_stored(self):
        adata = _adata_for_sweep()
        banksy_sweep(adata, lambda_params=[0.0], resolutions=[0.3])
        assert "banksy_sweep" in adata.uns


class TestBanksySweepNJobs:
    def test_n_jobs_matches_sequential(self):
        adata1 = _adata_for_sweep()
        adata2 = _adata_for_sweep()
        r1 = banksy_sweep(adata1, lambda_params=[0.0, 0.5], resolutions=[0.3, 0.5], n_jobs=1)
        r2 = banksy_sweep(adata2, lambda_params=[0.0, 0.5], resolutions=[0.3, 0.5], n_jobs=2)
        r1_sorted = r1.sort_values(["lambda", "resolution"]).reset_index(drop=True)
        r2_sorted = r2.sort_values(["lambda", "resolution"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(r1_sorted, r2_sorted)

    def test_n_jobs_does_not_leave_temp_keys(self):
        adata = _adata_for_sweep()
        banksy_sweep(adata, lambda_params=[0.0, 0.5], resolutions=[0.3], n_jobs=2)
        assert not any(k.startswith("_banksy_tmp_") for k in adata.obsm)
        assert not any(k.startswith("_banksy_tmp_") for k in adata.obsp)
        assert not any(k.startswith("_banksy_tmp_") for k in adata.uns)

    def test_default_neighbors_key_untouched(self):
        adata = _adata_for_sweep()
        assert "neighbors" not in adata.uns
        banksy_sweep(adata, lambda_params=[0.0], resolutions=[0.3])
        assert "neighbors" not in adata.uns
        assert "distances" not in adata.obsp
        assert "connectivities" not in adata.obsp
