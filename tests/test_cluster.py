"""Tests for tl.cluster (Leiden/Louvain wrapper)."""
import numpy as np
import anndata as ad
import mudata as md
import scanpy as sc
import pytest

from astrolabe.tl._cluster import cluster
from astrolabe.tl._graph import spatial_neighbors


def _adata_with_neighbors(n=40, seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(rng.random((n, 20)).astype(np.float32))
    sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
    return adata


class TestCluster:
    def test_leiden_smoke(self):
        adata = _adata_with_neighbors()
        cluster(adata)
        assert "leiden" in adata.obs
        assert adata.obs["leiden"].dtype.name == "category"
        assert adata.obs["leiden"].nunique() >= 1

    def test_louvain_smoke(self):
        pytest.importorskip("louvain")  # scanpy's louvain needs the louvain package
        adata = _adata_with_neighbors()
        cluster(adata, method="louvain")
        assert "louvain" in adata.obs

    def test_invalid_method_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(ValueError, match="method"):
            cluster(adata, method="kmeans")

    def test_custom_key_added(self):
        adata = _adata_with_neighbors()
        cluster(adata, key_added="my_clusters")
        assert "my_clusters" in adata.obs
        assert "leiden" not in adata.obs

    def test_spatial_neighbors_key(self):
        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.random((40, 10)).astype(np.float32))
        adata.obsm["spatial"] = rng.uniform(0, 500, (40, 2)).astype(np.float32)
        spatial_neighbors(adata)
        cluster(adata, neighbors_key="spatial_neighbors", key_added="spatial_leiden")
        assert "spatial_leiden" in adata.obs

    def test_flavor_override_respected(self):
        # explicit flavor='leidenalg' must not collide with the injected
        # igraph default (kwargs.setdefault, not unconditional injection)
        adata = _adata_with_neighbors()
        cluster(adata, flavor="leidenalg", key_added="legacy_leiden")
        assert "legacy_leiden" in adata.obs

    def test_resolution_monotone_smoke(self):
        adata = _adata_with_neighbors(n=60)
        cluster(adata, resolution=0.1, key_added="lo")
        cluster(adata, resolution=2.0, key_added="hi")
        assert adata.obs["hi"].nunique() >= adata.obs["lo"].nunique()

    def test_accepts_mudata(self):
        adata = _adata_with_neighbors()
        mdata = md.MuData({"xenium": adata})
        mdata.uns["default_modality"] = "xenium"
        cluster(mdata)
        assert "leiden" in mdata["xenium"].obs
