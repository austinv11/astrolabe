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


class TestBanksySweepDefaultNPCs:
    def test_matches_tl_pca_default_of_fifty(self):
        # Regression: banksy_sweep hardcoded n_pcs=30 while tl.pca's own
        # documented default (astrolabe/tl/_embedding.py: n_comps=50) is 50,
        # silently diverging and producing different cluster counts than a
        # manual BANKSY run at "the same" settings. Both functions are
        # @with_adata decorated, which collapses their introspectable
        # signature to (data, *args, modality=None, **kwargs) -- inspecting
        # the actual default requires reading the source directly instead.
        import astrolabe.tl._banksy_sweep as mod
        with open(mod.__file__, encoding="utf-8") as f:
            src = f.read()
        assert "n_pcs: int = 50" in src


class TestBanksySweepTagCollision:
    def test_close_lambda_values_do_not_collide_under_old_two_decimal_rounding(self):
        # Regression: lam_tag used to be f"{lam:.2f}".rstrip("0").rstrip(".")
        # -- two distinct lambda values rounding to the same 2 decimals
        # (e.g. 0.201 and 0.204, both -> "0.2") collided onto the *same*
        # internal obsm/obsp/uns keys, silently overwriting one run's
        # intermediate state with another's (or raising KeyError under
        # n_jobs>1 when one thread's cleanup deleted a key another thread
        # was still using). Both runs must now produce genuinely distinct,
        # independently-correct results.
        adata = _adata_for_sweep(n=60, n_genes=25)
        result = banksy_sweep(
            adata, lambda_params=[0.201, 0.204], resolutions=[0.3], n_jobs=2
        )
        assert len(result) == 2
        assert result["obs_key"].nunique() == 2
        for obs_key in result["obs_key"]:
            assert obs_key in adata.obs.columns
            assert adata.obs[obs_key].nunique() >= 1
        # No leftover ephemeral keys despite the near-identical lambda tags.
        assert not any(k.startswith("_banksy_tmp_") for k in adata.obsm)
