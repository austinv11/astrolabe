"""Tests for dpt root cell warning and diffmap recompute (audit fixes 2.2/2.3)."""
import logging

import numpy as np
import anndata as ad
import scanpy as sc
import pytest

from astrolabe.tl._trajectory import dpt


def _adata_with_neighbors(n=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 30)).astype(np.float32)
    adata = ad.AnnData(X)
    adata.obs["cluster"] = ["A"] * (n // 4) + ["B"] * (n // 4) + ["A"] * (n // 4) + ["B"] * (n - 3 * (n // 4))
    sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
    return adata


class TestRootCellSelection:
    def test_multiple_root_matches_warns(self, caplog):
        adata = _adata_with_neighbors()
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_obs_key="cluster", root_obs_val="A")
        assert "cells match root condition" in caplog.text

    def test_single_root_match_no_warning(self, caplog):
        adata = _adata_with_neighbors()
        adata.obs["unique_root"] = ["root"] + ["other"] * (len(adata) - 1)
        with caplog.at_level(logging.WARNING):
            dpt(adata, root_obs_key="unique_root", root_obs_val="root")
        assert "cells match root condition" not in caplog.text

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


class TestDiffmapRecompute:
    def test_recomputed_when_ndcs_mismatch(self, caplog):
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

    def test_computed_when_absent(self):
        adata = _adata_with_neighbors()
        assert "X_diffmap" not in adata.obsm
        dpt(adata, root_cell=0, n_dcs=8)
        assert "X_diffmap" in adata.obsm
        assert adata.obsm["X_diffmap"].shape[1] == 8
