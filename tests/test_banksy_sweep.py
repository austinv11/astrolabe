"""Tests for banksy_sweep X_pca save/restore (audit fix 2.7),
the n_jobs parallelism added to its lambda sweep, and the per-run quality
metric columns (neighbor_agreement / silhouette_expr / ari_to_prev_resolution)."""
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


def _planted_domain_adata(n=600, n_genes=10, mix=0.95, seed=0):
    """Two spatial domains, same cell types at mirrored proportions (see
    tests/test_banksy.py for the full rationale).

    Near-pure neighborhoods (mix=0.95) and a modest type signature: Leiden
    is community- (density-) based, so unlike KMeans it only prefers the
    domain partition when the neighborhood-composition signal is bimodal
    with a real density gap, not a smooth gradient.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, (n, 2)).astype(np.float32)
    domain = (coords[:, 0] > 500).astype(int)
    p_type1 = np.where(domain == 0, mix, 1.0 - mix)
    ctype = (rng.uniform(size=n) < p_type1).astype(int)
    X = rng.normal(0.0, 0.5, (n, n_genes))
    X[ctype == 1, : n_genes // 2] += 1.0
    X[ctype == 0, n_genes // 2:] += 1.0
    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = coords
    tl.spatial_neighbors(adata, n_neighbors=20)
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


class TestBanksySweepQualityMetrics:
    def test_metric_columns_present_and_valid(self):
        adata = _adata_for_sweep(n=60, n_genes=25)
        result = banksy_sweep(adata, lambda_params=[0.0, 0.5], resolutions=[0.3, 0.5])
        for col in ("neighbor_agreement", "silhouette_expr", "ari_to_prev_resolution"):
            assert col in result.columns, f"metric column '{col}' missing"
        assert result["neighbor_agreement"].between(0, 1).all()
        sil = result["silhouette_expr"].dropna()
        assert sil.between(-1, 1).all()
        # first resolution of each lambda has no previous run to compare to
        for lam, grp in result.groupby("lambda"):
            grp = grp.sort_values("resolution")
            assert np.isnan(grp["ari_to_prev_resolution"].iloc[0])
            assert np.isfinite(grp["ari_to_prev_resolution"].iloc[1:]).all()

    def test_high_lambda_has_higher_neighbor_agreement_on_planted_domains(self):
        # On the planted-domain fixture, high λ recovers spatially coherent
        # domains (most spatial-graph edges within a domain share a label)
        # while λ=0 recovers cell types (spatially interleaved by
        # construction, so many edges cross labels). Verified stable across
        # seeds 0-3 with a margin ≥ 0.08.
        adata = _planted_domain_adata()
        result = banksy_sweep(adata, lambda_params=[0.0, 0.9], resolutions=[0.1])
        agree = result.set_index("lambda")["neighbor_agreement"]
        assert agree[0.9] > agree[0.0] + 0.05, f"neighbor_agreement: {agree.to_dict()}"

    def test_no_leftover_leiden_uns_dicts(self):
        adata = _adata_for_sweep()
        result = banksy_sweep(adata, lambda_params=[0.0, 0.5], resolutions=[0.3])
        for obs_key in result["obs_key"]:
            assert obs_key in adata.obs.columns
            assert obs_key not in adata.uns, (
                f"leftover leiden params dict uns['{obs_key}'] not cleaned up"
            )


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
