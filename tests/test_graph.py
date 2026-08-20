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
from astrolabe.tl._trajectory import dpt, paga


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

    def test_alpha_zero_is_symmetrized_row_normalized_expression(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, alpha=0.0)
        conn = adata.obsp["joint_neighbors_connectivities"]
        expr = adata.obsp["connectivities"].tocsr()
        row_sums = np.asarray(expr.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        norm = sp.diags(1.0 / row_sums) @ expr
        expected = (norm + norm.T) * 0.5
        assert np.allclose(conn.toarray(), expected.toarray(), atol=1e-5)

    def test_alpha_one_is_symmetrized_row_normalized_spatial(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, alpha=1.0)
        conn = adata.obsp["joint_neighbors_connectivities"]
        spatial = adata.obsp["spatial_neighbors_connectivities"].tocsr()
        row_sums = np.asarray(spatial.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        norm = sp.diags(1.0 / row_sums) @ spatial
        expected = (norm + norm.T) * 0.5
        assert np.allclose(conn.toarray(), expected.toarray(), atol=1e-5)

    def test_blended_matrix_symmetric(self):
        adata = _adata_with_both_graphs()
        blend_neighbors(adata, alpha=0.4)
        conn = adata.obsp["joint_neighbors_connectivities"]
        assert abs(conn - conn.T).max() < 1e-6

    def test_alpha_out_of_range_raises(self):
        adata = _adata_with_both_graphs()
        with pytest.raises(ValueError, match="alpha"):
            blend_neighbors(adata, alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            blend_neighbors(adata, alpha=-0.1)

    def test_params_have_int_n_neighbors_and_structural_distances(self):
        # scanpy's Neighbors machinery KeyErrors without params['n_neighbors'];
        # a distances_key resolving to the spatial µm distances silently fed
        # the wrong scale into distance consumers — the stored matrix must be
        # the unit-weight STRUCTURAL matrix (paga binarizes it anyway), never
        # a metric one.
        adata = _adata_with_both_graphs()
        blend_neighbors(adata)
        meta = adata.uns["joint_neighbors"]
        assert isinstance(meta["params"]["n_neighbors"], int)
        assert meta["params"]["n_neighbors"] > 0
        dist = adata.obsp[meta["distances_key"]]
        assert (dist.data == 1.0).all()
        conn = adata.obsp[meta["connectivities_key"]]
        assert dist.nnz == conn.nnz

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

    def test_full_downstream_stack(self):
        # Regression for the blended graph being invalid for scanpy's
        # Neighbors machinery (missing params['n_neighbors'] -> KeyError;
        # distances_key pointing at spatial µm distances): leiden, umap,
        # paga, and dpt must ALL run on neighbors_key='joint_neighbors'.
        adata = _adata_with_both_graphs(n=40)
        blend_neighbors(adata)
        cluster(adata, neighbors_key="joint_neighbors", key_added="joint_leiden")
        umap(adata, neighbors_key="joint_neighbors", key_added="X_joint_umap")
        paga(adata, groupby="joint_leiden", neighbors_key="joint_neighbors")
        dpt(adata, root_cell=0, neighbors_key="joint_neighbors", n_dcs=5)
        assert "joint_leiden" in adata.obs
        assert "X_joint_umap" in adata.obsm
        assert "paga" in adata.uns
        assert "dpt_pseudotime" in adata.obs
        assert np.isfinite(adata.obs["dpt_pseudotime"].values).all()


class TestKNNEdgeIdentity:
    def test_edges_match_hand_computed_knn_union(self):
        # Tiny deterministic fixture: brute-force the directed k=2 kNN edge
        # set, union-symmetrize it, and require the graph to match exactly.
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [10.0, 10.0], [11.0, 10.0]],
            dtype=np.float32,
        )
        adata = ad.AnnData(obsm={"spatial": coords})
        spatial_neighbors(adata, n_neighbors=2)
        conn = adata.obsp["spatial_neighbors_connectivities"].tocoo()
        got = {(int(r), int(c)) for r, c in zip(conn.row, conn.col)}

        n = len(coords)
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        directed = set()
        for i in range(n):
            for j in np.argsort(d[i])[:2]:
                directed.add((i, int(j)))
        expected = directed | {(j, i) for i, j in directed}
        assert got == expected

    def test_distances_match_euclidean(self):
        adata = _adata_with_spatial(n=25)
        spatial_neighbors(adata, n_neighbors=4)
        dist = adata.obsp["spatial_neighbors_distances"].tocoo()
        coords = adata.obsm["spatial"].astype(np.float64)
        expected = np.linalg.norm(coords[dist.row] - coords[dist.col], axis=1)
        assert np.allclose(dist.data, expected, atol=1e-4)


class TestZeroDistanceEdges:
    def test_coincident_centroids_keep_edges(self):
        # Three coincident points (e.g. Visium HD bin centers after merge)
        # plus distant others: the zero-distance edges among the coincident
        # trio must survive in the connectivity matrix.
        coords = np.array(
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [100.0, 100.0], [200.0, 200.0]],
            dtype=np.float32,
        )
        adata = ad.AnnData(obsm={"spatial": coords})
        spatial_neighbors(adata, n_neighbors=2)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert conn[i, j] == 1.0, f"zero-distance edge ({i},{j}) was dropped"

    def test_distance_matrix_shares_conn_structure(self):
        coords = np.array(
            [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [100.0, 100.0], [200.0, 200.0]],
            dtype=np.float32,
        )
        adata = ad.AnnData(obsm={"spatial": coords})
        spatial_neighbors(adata, n_neighbors=2)
        conn = adata.obsp["spatial_neighbors_connectivities"].tocsr()
        dist = adata.obsp["spatial_neighbors_distances"].tocsr()
        # Zero-distance edges must be explicit stored zeros, not implicit.
        assert np.array_equal(conn.indptr, dist.indptr)
        assert np.array_equal(conn.indices, dist.indices)

    def test_no_self_loops_with_duplicates(self):
        # Structural self-exclusion (no-X kneighbors) must hold even when
        # coordinates are duplicated.
        coords = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32
        )
        adata = ad.AnnData(obsm={"spatial": coords})
        spatial_neighbors(adata, n_neighbors=2)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        assert conn.diagonal().sum() == 0


class TestSpatialNeighborsDefaults:
    def test_default_k_is_6(self):
        adata = _adata_with_spatial(n=60)
        spatial_neighbors(adata)
        assert adata.uns["spatial_neighbors"]["params"]["n_neighbors"] == 6
        conn = adata.obsp["spatial_neighbors_connectivities"]
        degrees = np.asarray(conn.sum(axis=1)).ravel()
        # union symmetrization can only ADD edges beyond the directed k=6
        assert degrees.min() >= 6

    def test_coord_type_validated(self):
        adata = _adata_with_spatial()
        with pytest.raises(ValueError, match="coord_type"):
            spatial_neighbors(adata, coord_type="hexagonal")

    def test_params_record_honest_provenance(self):
        adata = _adata_with_spatial()
        spatial_neighbors(adata, n_neighbors=5)
        params = adata.uns["spatial_neighbors"]["params"]
        assert params["astrolabe_method"] == "knn"
        assert params["metric"] == "euclidean"
        assert params["method"] == "umap"  # scanpy-compat label

    def test_delaunay_requires_2d(self):
        rng = np.random.default_rng(0)
        adata = ad.AnnData(obsm={"spatial": rng.uniform(0, 100, (20, 3)).astype(np.float32)})
        with pytest.raises(ValueError, match="2-D"):
            spatial_neighbors(adata, coord_type="delaunay")

    def test_set_diag_consistent_structures(self):
        adata = _adata_with_spatial(n=20)
        spatial_neighbors(adata, n_neighbors=3, set_diag=True)
        conn = adata.obsp["spatial_neighbors_connectivities"].tocsr()
        dist = adata.obsp["spatial_neighbors_distances"].tocsr()
        assert (conn.diagonal() == 1.0).all()
        # diagonal must be part of BOTH stored structures (explicit 0 distance)
        assert np.array_equal(conn.indptr, dist.indptr)
        assert np.array_equal(conn.indices, dist.indices)
        assert dist.diagonal().sum() == 0.0


class TestGridGraph:
    def _lattice(self, side=5):
        xs, ys = np.meshgrid(
            np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32)
        )
        coords = np.column_stack([xs.ravel(), ys.ravel()])
        return ad.AnnData(obsm={"spatial": coords})

    def test_grid_honors_n_neighbors_8(self):
        # 7x7 integer lattice, k=8 chebyshev: the center cell's 8 nearest
        # are exactly its 8-connected box neighbors, and no border cell's
        # own k=8 pick reaches the center (chebyshev distance >= 2 rings are
        # fully absorbed by closer cells), so union symmetrization adds
        # nothing at the center.
        adata = self._lattice(7)
        spatial_neighbors(adata, coord_type="grid", n_neighbors=8)
        conn = adata.obsp["spatial_neighbors_connectivities"].tocsr()
        center = 3 * 7 + 3  # row-major index of (x=3, y=3)
        neighbors = set(conn[center].indices.tolist())
        expected = {
            (3 + dy) * 7 + (3 + dx)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        }
        assert neighbors == expected

    def test_grid_default_k_is_6(self):
        adata = self._lattice(5)
        spatial_neighbors(adata, coord_type="grid")
        assert adata.uns["spatial_neighbors"]["params"]["n_neighbors"] == 6

    def test_grid_metric_recorded(self):
        adata = self._lattice(4)
        spatial_neighbors(adata, coord_type="grid", n_neighbors=8)
        assert adata.uns["spatial_neighbors"]["params"]["metric"] == "chebyshev"


class TestRadiusIntersection:
    def _line(self, n=9):
        coords = np.column_stack(
            [np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32)]
        )
        return ad.AnnData(obsm={"spatial": coords})

    def test_pure_radius_ignores_k(self):
        adata = self._line()
        spatial_neighbors(adata, radius=3.5)  # n_neighbors=None -> pure radius
        conn = adata.obsp["spatial_neighbors_connectivities"]
        degrees = np.asarray(conn.sum(axis=1)).ravel()
        # interior cells reach +/-1, +/-2, +/-3 -> degree 6
        assert degrees.max() >= 6
        assert adata.uns["spatial_neighbors"]["params"]["astrolabe_method"] == "radius"

    def test_explicit_k_truncates_radius_rows(self):
        adata = self._line()
        spatial_neighbors(adata, radius=3.5, n_neighbors=2)
        conn = adata.obsp["spatial_neighbors_connectivities"]
        degrees = np.asarray(conn.sum(axis=1)).ravel()
        # directed degree <= 2; union symmetrization can add at most the
        # inbound edges from asymmetric picks.
        assert degrees.max() <= 4
        assert (
            adata.uns["spatial_neighbors"]["params"]["astrolabe_method"]
            == "knn_radius_intersection"
        )

    def test_radius_still_bounds_distances(self):
        adata = _adata_with_spatial(n=40)
        spatial_neighbors(adata, radius=150.0, n_neighbors=3)
        dist = adata.obsp["spatial_neighbors_distances"]
        if dist.nnz:
            assert dist.data.max() <= 150.0 + 1e-6
