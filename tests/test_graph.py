"""Tests for spatial_neighbors KNN and Delaunay vectorization (audit fix 3.2/3.3)."""
import numpy as np
import anndata as ad
import mudata as md
import scanpy as sc
import scipy.sparse as sp
import pytest

from astrolabe.tl._graph import spatial_neighbors, transcriptional_neighbors, blend_neighbors
from astrolabe.tl._cluster import cluster
from astrolabe.tl._embedding import umap


def _adata_with_spatial(n=50, seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(obsm={"spatial": rng.uniform(0, 1000, (n, 2)).astype(np.float32)})
    return adata


def _adata_with_expression(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return ad.AnnData(rng.random((n, 20)).astype(np.float32))


class TestKNNGraph:
    def test_symmetric(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        diff = abs(conn - conn.T).max()
        assert diff < 1e-6, "KNN connectivity must be symmetric"

    def test_no_self_loops(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        assert conn.diagonal().sum() == 0, "no self-loops in connectivity"

    def test_uns_written(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=4)
        assert "spatial_neighbors" in adata.uns

    def test_does_not_touch_uns_neighbors(self):
        # spatial_neighbors must not clobber a transcriptomic graph written
        # to uns['neighbors'] by sc.pp.neighbors — astrolabe's own
        # neighbors-consuming functions default neighbors_key='spatial_neighbors'
        # explicitly instead of relying on this implicit scanpy convention.
        adata = _adata_with_spatial()
        adata.uns["neighbors"] = {"sentinel": True}
        spatial_neighbors(adata, n_neighbors=4)
        assert adata.uns["neighbors"] == {"sentinel": True}

    def test_distances_nonnegative(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        dist = adata.obsp["spatial_neighbors_distances"]
        assert dist.data.min() >= 0

    def test_obsp_keys_present(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        assert "spatial_neighbors_connectivities" in adata.obsp
        assert "spatial_neighbors_distances" in adata.obsp

    def test_params_include_method_for_scanpy_compat(self):
        # sc.tl.umap reads uns[neighbors_key]['params']['method'] unconditionally
        # (KeyError, not a soft check, if absent) -- tl.umap/tl.cluster default
        # neighbors_key='spatial_neighbors', so this graph must carry it.
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        assert adata.uns["spatial_neighbors"]["params"]["method"] == "umap"

    def test_radius_graph_symmetric(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, radius=200.0)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        diff = abs(conn - conn.T).max()
        assert diff < 1e-6, "radius graph must be symmetric"

    def test_radius_graph_no_self_loops(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, radius=200.0)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        assert conn.diagonal().sum() == 0


class TestDelaunayGraph:
    def test_symmetric(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, coord_type="delaunay")
        conn = adata.obsp["spatial_neighbors_connectivities"]
        diff = abs(conn - conn.T).max()
        assert diff < 1e-6, "Delaunay connectivity must be symmetric"

    def test_no_self_loops(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, coord_type="delaunay")
        conn = adata.obsp["spatial_neighbors_connectivities"]
        assert conn.diagonal().sum() == 0

    def test_uns_written(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, coord_type="delaunay")
        assert "spatial_neighbors" in adata.uns

    def test_max_dist_prunes_long_edges(self):
        adata = _adata_with_spatial(n=30)
        spatial_neighbors(adata, coord_type="delaunay", delaunay_max_dist=100.0)
        dist = adata.obsp["spatial_neighbors_distances"]
        if dist.nnz > 0:
            assert dist.data.max() <= 100.0 + 1e-6


class TestTranscriptionalNeighbors:
    def test_writes_default_uns_key(self):
        adata = _adata_with_expression()
        transcriptional_neighbors(adata, n_neighbors=5, use_rep="X")
        assert "neighbors" in adata.uns
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp

    def test_custom_key_added(self):
        adata = _adata_with_expression()
        transcriptional_neighbors(adata, n_neighbors=5, use_rep="X", key_added="expr_neighbors")
        assert "expr_neighbors" in adata.uns
        assert "expr_neighbors_connectivities" in adata.obsp
        assert "expr_neighbors_distances" in adata.obsp

    def test_does_not_touch_spatial_neighbors(self):
        rng = np.random.default_rng(0)
        adata = ad.AnnData(rng.random((30, 20)).astype(np.float32))
        adata.obsm["spatial"] = rng.uniform(0, 500, (30, 2)).astype(np.float32)
        spatial_neighbors(adata, n_neighbors=5)
        transcriptional_neighbors(adata, n_neighbors=5, use_rep="X")
        assert "spatial_neighbors" in adata.uns
        assert "spatial_neighbors_connectivities" in adata.obsp

    def test_accepts_mudata(self):
        adata = _adata_with_expression()
        mdata = md.MuData({"xenium": adata})
        mdata.uns["default_modality"] = "xenium"
        transcriptional_neighbors(mdata, n_neighbors=5, use_rep="X")
        assert "neighbors" in mdata["xenium"].uns

    def test_accepts_mudata_explicit_modality(self):
        adata = _adata_with_expression()
        mdata = md.MuData({"xenium": adata})
        transcriptional_neighbors(mdata, modality="xenium", n_neighbors=5, use_rep="X")
        assert "neighbors" in mdata["xenium"].uns


def _adata_with_both_graphs(n=30, seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(rng.random((n, 20)).astype(np.float32))
    adata.obsm["spatial"] = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
    spatial_neighbors(adata, n_neighbors=5)
    return adata


class TestBlendNeighbors:
    def test_persists_in_obsp_and_uns(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata)
        assert "joint_neighbors" in adata.uns, (
            "blend_neighbors must persist the joint graph in uns, unlike "
            "the removed spatial_umap() which deleted it after use"
        )
        assert "joint_neighbors_connectivities" in adata.obsp

    def test_shape_and_dtype(self):
        adata = _adata_with_both_graphs(n=25)
        blend_neighbors(adata)
        conn = adata.obsp["joint_neighbors_connectivities"]
        assert conn.shape == (25, 25)
        assert conn.dtype == np.float32

    def test_alpha_zero_is_row_normalized_expression(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, alpha=0.0)
        conn = adata.obsp["joint_neighbors_connectivities"]
        expr = adata.obsp["connectivities"].tocsr()
        row_sums = np.asarray(expr.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        expected = sp.diags(1.0 / row_sums) @ expr
        assert np.allclose(conn.toarray(), expected.toarray(), atol=1e-5)

    def test_alpha_one_is_row_normalized_spatial(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, alpha=1.0)
        conn = adata.obsp["joint_neighbors_connectivities"]
        spatial = adata.obsp["spatial_neighbors_connectivities"].tocsr()
        row_sums = np.asarray(spatial.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        expected = sp.diags(1.0 / row_sums) @ spatial
        assert np.allclose(conn.toarray(), expected.toarray(), atol=1e-5)

    def test_custom_key_added(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, key_added="my_joint")
        assert "my_joint" in adata.uns
        assert "my_joint_connectivities" in adata.obsp
        assert "joint_neighbors" not in adata.uns

    def test_missing_expr_graph_raises(self):
        adata = _adata_with_spatial()
        with pytest.raises(KeyError):
            blend_neighbors(adata)

    def test_downstream_umap(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata)
        umap(adata, neighbors_key="joint_neighbors", key_added="X_spatial_umap")
        assert "X_spatial_umap" in adata.obsm
        assert adata.obsm["X_spatial_umap"].shape == (len(adata), 2)

    def test_downstream_cluster(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata)
        cluster(adata, neighbors_key="joint_neighbors", key_added="cospatial_leiden")
        assert "cospatial_leiden" in adata.obs
