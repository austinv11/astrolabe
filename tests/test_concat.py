"""Tests for concat_samples coordinate frame warning (audit fix 4.15)."""
import logging

import numpy as np
import anndata as ad
import mudata as md
import pytest

from astrolabe.io._concat import concat_samples


def _make_mdata(lib_id, n=5):
    rng = np.random.default_rng(abs(hash(lib_id)) % 2**31)
    adata = ad.AnnData(
        np.zeros((n, 3), dtype=np.float32),
        obsm={"spatial": rng.random((n, 2)).astype(np.float32)},
    )
    adata.var_names = ["g0", "g1", "g2"]
    adata.uns["spatial"] = {lib_id: {}}
    mdata = md.MuData({"xenium": adata})
    mdata.uns["default_modality"] = "xenium"
    mdata.uns["spatial"] = {lib_id: {}}
    return mdata


class TestConcatSamples:
    def test_warns_about_coordinate_frames(self, caplog):
        mdata_a = _make_mdata("run_A")
        mdata_b = _make_mdata("run_B")
        with caplog.at_level(logging.WARNING):
            concat_samples([mdata_a, mdata_b])
        assert "not comparable" in caplog.text.lower() or "coordinate" in caplog.text.lower()

    def test_library_id_column_present(self):
        mdata_a = _make_mdata("run_A")
        mdata_b = _make_mdata("run_B")
        result = concat_samples([mdata_a, mdata_b])
        assert "library_id" in result["xenium"].obs.columns

    def test_library_ids_correct(self):
        mdata_a = _make_mdata("run_A")
        mdata_b = _make_mdata("run_B")
        result = concat_samples([mdata_a, mdata_b])
        assert set(result["xenium"].obs["library_id"]) == {"run_A", "run_B"}

    def test_obs_names_unique(self):
        mdata_a = _make_mdata("alpha")
        mdata_b = _make_mdata("beta")
        result = concat_samples([mdata_a, mdata_b])
        obs_names = list(result["xenium"].obs_names)
        assert len(obs_names) == len(set(obs_names)), "obs_names must be globally unique"

    def test_cell_count_matches(self):
        mdata_a = _make_mdata("X", n=7)
        mdata_b = _make_mdata("Y", n=3)
        result = concat_samples([mdata_a, mdata_b])
        assert result["xenium"].n_obs == 10
