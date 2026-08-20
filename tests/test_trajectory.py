"""Tests for paga and dpt wrappers."""
import logging

import numpy as np
import anndata as ad
import scanpy as sc
import pytest

from astrolabe.tl._trajectory import dpt, paga


def _adata_with_neighbors(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 30)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obs["cluster"] = ["A"] * (n // 4) + ["B"] * (n // 4) + ["A"] * (n // 4) + ["B"] * (n - 3 * (n // 4))
    sc.pp.pca(adata, n_comps=5)
    adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
    return adata


class TestRootCellSelection:
    def test_multiple_matches_no_warning(self, caplog):
        # Multiple matches are the intended input for a medoid — no warning.
        adata = _adata_with_neighbors()
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_obs_key="cluster", root_obs_val="A")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_single_root_match_works(self, caplog):
        # A one-cell group's medoid is that cell itself.
        adata = _adata_with_neighbors()
        adata.obs["unique_root"] = ["root"] + ["other"] * (len(adata) - 1)
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_obs_key="unique_root", root_obs_val="root")
        assert adata.uns["iroot"] == 0
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_root_cell_int_index(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_cell=0)
        assert "dpt_pseudotime" in adata.obs.columns

    def test_missing_root_obs_key_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(KeyError, match="nonexistent_col"):
            dpt(adata, root_obs_key="nonexistent_col", root_obs_val="A")

    def test_missing_root_obs_val_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(ValueError, match="No cells found"):
            dpt(adata, root_obs_key="cluster", root_obs_val="Z")

    def test_no_root_raises(self):
        adata = _adata_with_neighbors()
        adata.uns.pop("iroot", None)
        with pytest.raises(ValueError, match="Root cell not specified"):
            dpt(adata)

    def test_pseudotime_written(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_cell=0)
        assert "dpt_pseudotime" in adata.obs.columns
        vals = adata.obs["dpt_pseudotime"].values
        assert np.all(vals >= 0)

    def test_incomplete_root_pair_raises(self):
        adata = _adata_with_neighbors()
        # Must raise, never silently reuse a pre-seeded uns['iroot'].
        adata.uns["iroot"] = 0
        with pytest.raises(ValueError, match="passed together"):
            dpt(adata, root_obs_key="cluster")
        with pytest.raises(ValueError, match="passed together"):
            dpt(adata, root_obs_val="A")

    def test_root_cell_and_obs_key_mutually_exclusive(self):
        adata = _adata_with_neighbors()
        with pytest.raises(ValueError, match="not both"):
            dpt(adata, root_cell=0, root_obs_key="cluster", root_obs_val="A")

    def test_root_cell_obs_name(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_cell=str(adata.obs_names[3]))
        assert adata.uns["iroot"] == 3
        assert adata.uns["dpt_root"]["method"] == "explicit"

    def test_root_cell_out_of_range_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(IndexError):
            dpt(adata, root_cell=999)
        with pytest.raises(IndexError):
            dpt(adata, root_cell=-1)

    def test_root_cell_unknown_obs_name_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(KeyError, match="obs_names"):
            dpt(adata, root_cell="not_a_barcode")

    def test_provenance_recorded_explicit(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_cell=0)
        prov = adata.uns["dpt_root"]
        assert prov["method"] == "explicit"
        assert prov["iroot"] == 0
        assert prov["obs_name"] == str(adata.obs_names[0])


class TestMedoidRoot:
    def test_medoid_root_is_group_member(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_obs_key="cluster", root_obs_val="A")
        iroot = adata.uns["iroot"]
        assert adata.obs["cluster"].iloc[iroot] == "A"
        prov = adata.uns["dpt_root"]
        assert prov["method"] == "medoid"
        assert prov["root_rep"] == "X_pca"
        assert prov["n_matches"] == int((adata.obs["cluster"] == "A").sum())

    def test_medoid_matches_manual_computation(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_obs_key="cluster", root_obs_val="A")
        mask = (adata.obs["cluster"] == "A").to_numpy()
        rep = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        centroid = rep[mask].mean(axis=0)
        d = np.linalg.norm(rep[mask] - centroid, axis=1)
        expected = int(np.where(mask)[0][d.argmin()])
        assert adata.uns["iroot"] == expected

    def test_medoid_spatial_rep_picks_central_cell(self):
        adata = _adata_with_neighbors()
        # Plant a known geometry: group A cells in a tight blob with one
        # member exactly at the blob center; everything else far away.
        mask = (adata.obs["cluster"] == "A").to_numpy()
        coords = np.full((adata.n_obs, 2), 500.0, dtype=np.float32)
        a_idx = np.where(mask)[0]
        rng = np.random.default_rng(1)
        coords[a_idx] = 50.0 + rng.uniform(-5, 5, (len(a_idx), 2))
        coords[a_idx[2]] = coords[a_idx].mean(axis=0)  # planted medoid
        adata.obsm["spatial"] = coords
        dpt(adata, root_obs_key="cluster", root_obs_val="A", root_rep="spatial")
        assert adata.uns["iroot"] == a_idx[2]
        assert adata.uns["dpt_root"]["root_rep"] == "spatial"

    def test_root_rep_spatial_multi_library_raises(self):
        adata = _adata_with_neighbors()
        adata.obs["library_id"] = ["lib1"] * (adata.n_obs // 2) + \
            ["lib2"] * (adata.n_obs - adata.n_obs // 2)
        with pytest.raises(ValueError, match="libraries"):
            dpt(adata, root_obs_key="cluster", root_obs_val="A",
                root_rep="spatial")

    def test_root_rep_missing_raises(self):
        adata = _adata_with_neighbors()
        with pytest.raises(KeyError, match="root_rep"):
            dpt(adata, root_obs_key="cluster", root_obs_val="A",
                root_rep="X_nope")

    def test_default_rep_falls_back_to_X(self, caplog):
        adata = _adata_with_neighbors()
        del adata.obsm["X_pca"]
        with caplog.at_level(logging.INFO):
            dpt(adata, root_obs_key="cluster", root_obs_val="A")
        assert adata.uns["dpt_root"]["root_rep"] == "X"
        assert "dpt_pseudotime" in adata.obs

    def test_ring_group_warns_spatial_only(self, caplog):
        adata = _adata_with_neighbors()
        # Group A on a circle: centroid is at the center, far from members.
        mask = (adata.obs["cluster"] == "A").to_numpy()
        a_idx = np.where(mask)[0]
        theta = np.linspace(0, 2 * np.pi, len(a_idx), endpoint=False)
        coords = np.asarray(adata.obsm["spatial"]).copy()
        coords[a_idx, 0] = 50 + 40 * np.cos(theta)
        coords[a_idx, 1] = 50 + 40 * np.sin(theta)
        adata.obsm["spatial"] = coords
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_obs_key="cluster", root_obs_val="A",
                root_rep="spatial")
        assert "ring-shaped or multi-nested" in caplog.text

        caplog.clear()
        adata2 = _adata_with_neighbors()
        with caplog.at_level(logging.WARNING):
            dpt(adata2, root_obs_key="cluster", root_obs_val="A")
        assert "ring-shaped" not in caplog.text


class TestDiffmapRecompute:
    def test_recomputed_when_too_few_comps(self, caplog):
        adata = _adata_with_neighbors()
        sc.tl.diffmap(adata, n_comps=5)
        assert adata.obsm["X_diffmap"].shape[1] == 5
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_cell=0, n_dcs=10)
        assert "recomputing diffusion map" in caplog.text.lower()
        assert adata.obsm["X_diffmap"].shape[1] == 10

    def test_not_recomputed_when_matching(self, caplog):
        adata = _adata_with_neighbors()
        sc.tl.diffmap(adata, n_comps=10)
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_cell=0, n_dcs=10)
        assert "recomputing" not in caplog.text.lower()

    def test_not_recomputed_when_more_comps_exist(self, caplog):
        # sc.tl.dpt uses the first n_dcs of a wider diffmap — recomputing
        # would discard information for no reason (condition is <, not !=).
        adata = _adata_with_neighbors()
        sc.tl.diffmap(adata, n_comps=12)
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_cell=0, n_dcs=10)
        assert "recomputing" not in caplog.text.lower()
        assert adata.obsm["X_diffmap"].shape[1] == 12
        assert "dpt_pseudotime" in adata.obs

    def test_computed_when_absent(self):
        adata = _adata_with_neighbors()
        assert "X_diffmap" not in adata.obsm
        dpt(adata, root_cell=0, n_dcs=8)
        assert "X_diffmap" in adata.obsm
        assert adata.obsm["X_diffmap"].shape[1] == 8


class TestDptKeyAdded:
    def test_pseudotime_renamed(self):
        adata = _adata_with_neighbors()
        dpt(adata, root_cell=0, key_added="my_pt")
        assert "my_pt" in adata.obs
        assert "dpt_pseudotime" not in adata.obs

    def test_groups_renamed_with_branchings(self):
        adata = _adata_with_neighbors(n=40)
        dpt(adata, root_cell=0, n_branchings=1, key_added="my_pt")
        assert "my_pt" in adata.obs
        assert "dpt_groups" not in adata.obs
        assert "my_pt_groups" in adata.obs

    def test_second_run_with_custom_key_preserves_first_pseudotime(self):
        # Regression: sc.tl.dpt always writes obs['dpt_pseudotime']; a second
        # run with key_added= used to overwrite-and-pop the first run's
        # column away entirely.
        adata = _adata_with_neighbors(n=40)
        dpt(adata, root_cell=0)
        first = adata.obs["dpt_pseudotime"].copy()

        sc.pp.neighbors(adata, n_neighbors=8, use_rep="X", key_added="alt")
        dpt(adata, root_cell=0, neighbors_key="alt", key_added="dpt_alt")

        assert "dpt_pseudotime" in adata.obs
        assert "dpt_alt" in adata.obs
        np.testing.assert_array_equal(
            adata.obs["dpt_pseudotime"].to_numpy(), first.to_numpy()
        )

    def test_diffmap_recomputed_for_different_graph(self):
        # Regression: the second run used to silently reuse the first run's
        # diffmap, so its pseudotime ignored the requested graph entirely.
        adata = _adata_with_neighbors(n=40)
        dpt(adata, root_cell=0)
        assert adata.uns["diffmap_neighbors_key"] == "neighbors"
        dm_first = adata.obsm["X_diffmap"].copy()

        sc.pp.neighbors(adata, n_neighbors=8, use_rep="X", key_added="alt")
        dpt(adata, root_cell=0, neighbors_key="alt", key_added="dpt_alt")

        assert adata.uns["diffmap_neighbors_key"] == "alt"
        assert not np.allclose(adata.obsm["X_diffmap"], dm_first)
        # different graphs -> different orderings
        assert not np.allclose(
            adata.obs["dpt_alt"].to_numpy(), adata.obs["dpt_pseudotime"].to_numpy()
        )

    def test_stale_groups_not_renamed(self):
        # A dpt_groups column from an earlier default-key run must not be
        # hijacked by a later n_branchings=0 run with a custom key.
        adata = _adata_with_neighbors()
        adata.obs["dpt_groups"] = "stale"
        dpt(adata, root_cell=0, n_branchings=0, key_added="my_pt")
        assert "dpt_groups" in adata.obs
        assert "my_pt_groups" not in adata.obs


def _clustered_adata(n=40, seed=0):
    adata = _adata_with_neighbors(n=n, seed=seed)
    sc.tl.leiden(adata, resolution=1.0, flavor="igraph", n_iterations=2)
    return adata


class TestPaga:
    def test_smoke(self):
        adata = _clustered_adata()
        paga(adata, groupby="leiden")
        assert "paga" in adata.uns
        assert "connectivities" in adata.uns["paga"]

    def test_key_added_renames(self):
        adata = _clustered_adata()
        paga(adata, groupby="leiden", key_added="my_paga")
        assert "my_paga" in adata.uns
        assert "paga" not in adata.uns
        assert "connectivities" in adata.uns["my_paga"]

    def test_threshold_param_removed(self):
        # threshold is a sc.pl.paga plotting parameter; passing it here must
        # fail loudly instead of being silently forwarded/dropped.
        adata = _clustered_adata()
        with pytest.raises(TypeError):
            paga(adata, groupby="leiden", threshold=0.05)

    def test_obs_group_column_used(self):
        adata = _clustered_adata()
        paga(adata, groupby="leiden")
        n_groups = adata.obs["leiden"].nunique()
        assert adata.uns["paga"]["connectivities"].shape == (n_groups, n_groups)
