"""Tests for tl.pca / tl.umap wrappers."""
import numpy as np
import anndata as ad
import mudata as md
import scanpy as sc
import pytest

from astrolabe.tl._embedding import pca, umap


def _adata(n=30, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    return ad.AnnData(rng.random((n, n_genes)).astype(np.float32))


class TestPCA:
    def test_smoke_writes_standard_outputs(self):
        adata = _adata()
        pca(adata)
        assert "X_pca" in adata.obsm
        assert "PCs" in adata.varm
        assert "variance_ratio" in adata.uns["pca"]

    def test_adaptive_n_comps_on_small_panel(self):
        # Regression: hardcoded n_comps=50 raised on <=50-gene targeted
        # panels; n_comps=None must adapt to min(50, min_dim - 1).
        adata = _adata(n=15, n_genes=8)
        pca(adata)
        assert adata.obsm["X_pca"].shape == (15, 7)

    def test_explicit_n_comps(self):
        adata = _adata()
        pca(adata, n_comps=5)
        assert adata.obsm["X_pca"].shape[1] == 5

    def test_auto_hvg_mask(self):
        adata = _adata()
        adata.var["highly_variable"] = [True] * 5 + [False] * 15
        pca(adata)
        # 5 masked-in genes -> at most 4 components under the adaptive default
        assert adata.obsm["X_pca"].shape[1] == 4

    def test_mask_var_none_disables_hvg(self):
        adata = _adata()
        adata.var["highly_variable"] = [True] * 5 + [False] * 15
        pca(adata, mask_var=None)
        assert adata.obsm["X_pca"].shape[1] > 4

    def test_use_rep_writes_embedding_and_variance_ratio(self):
        adata = _adata()
        rng = np.random.default_rng(1)
        adata.obsm["X_banksy"] = rng.random((30, 40)).astype(np.float32)
        pca(adata, use_rep="X_banksy", n_comps=6, key_added="banksy_pca")
        assert adata.obsm["banksy_pca"].shape == (30, 6)
        # the old hand-rolled sklearn branch dropped variance_ratio entirely
        assert "variance_ratio" in adata.uns["banksy_pca"]
        assert len(adata.uns["banksy_pca"]["variance_ratio"]) == 6

    def test_use_rep_ignores_hvg_column(self):
        # var['highly_variable'] must never be applied to obsm input
        # (dimension mismatch would raise inside scanpy otherwise).
        adata = _adata()
        adata.var["highly_variable"] = [True] * 5 + [False] * 15
        rng = np.random.default_rng(1)
        adata.obsm["R"] = rng.random((30, 12)).astype(np.float32)
        pca(adata, use_rep="R", n_comps=6)
        assert adata.obsm["X_pca"].shape == (30, 6)

    def test_use_rep_missing_raises(self):
        adata = _adata()
        with pytest.raises(KeyError, match="nope"):
            pca(adata, use_rep="nope")

    def test_use_rep_with_layer_raises(self):
        adata = _adata()
        adata.obsm["R"] = np.random.default_rng(0).random((30, 5)).astype(np.float32)
        adata.layers["l"] = adata.X.copy()
        with pytest.raises(ValueError, match="mutually exclusive"):
            pca(adata, use_rep="R", layer="l")

    def test_use_rep_with_explicit_mask_var_raises(self):
        adata = _adata()
        adata.obsm["R"] = np.random.default_rng(0).random((30, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="mask_var"):
            pca(adata, use_rep="R", mask_var="highly_variable")

    def test_accepts_mudata(self):
        adata = _adata()
        mdata = md.MuData({"xenium": adata})
        mdata.uns["default_modality"] = "xenium"
        pca(mdata)
        assert "X_pca" in mdata["xenium"].obsm


class TestUMAP:
    def test_smoke(self):
        adata = _adata()
        sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
        umap(adata)
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (30, 2)

    def test_custom_key_added(self):
        adata = _adata()
        sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
        umap(adata, key_added="X_custom_umap")
        assert "X_custom_umap" in adata.obsm
