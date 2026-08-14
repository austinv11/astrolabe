"""Tests for concat_samples coordinate frame warning (audit fix 4.15)."""
import logging

import numpy as np
import anndata as ad
import mudata as md
import pandas as pd
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


def _make_mdata_with_transcripts(lib_id, n=5, cell_ids=None):
    import pyarrow as pa

    mdata = _make_mdata(lib_id, n=n)
    cell_ids = cell_ids or [f"c{i}" for i in range(n)]
    mdata["xenium"].obs_names = pd.Index(cell_ids, name="cell_id")
    mdata["xenium"].uns["transcripts"] = pa.table(
        {
            "cell_id": cell_ids,
            "feature_name": [f"gene{i}" for i in range(len(cell_ids))],
            "x_location": np.arange(len(cell_ids), dtype=np.float64),
            "y_location": np.arange(len(cell_ids), dtype=np.float64),
        }
    )
    return mdata


def _make_mdata_with_graph(lib_id, n=5):
    import scipy.sparse as sp

    mdata = _make_mdata(lib_id, n=n)
    adata = mdata["xenium"]
    adata.obsp["spatial_neighbors_connectivities"] = sp.eye(n, format="csr")
    adata.obsp["spatial_neighbors_distances"] = sp.eye(n, format="csr")
    adata.uns["spatial_neighbors"] = {
        "connectivities_key": "spatial_neighbors_connectivities",
        "distances_key": "spatial_neighbors_distances",
        "params": {"n_neighbors": 6},
    }
    return mdata


class TestConcatSamplesTranscripts:
    def test_merges_transcripts_present_on_every_sample(self):
        mdata_a = _make_mdata_with_transcripts("run_A", n=2, cell_ids=["a0", "a1"])
        mdata_b = _make_mdata_with_transcripts("run_B", n=2, cell_ids=["b0", "b1"])
        result = concat_samples([mdata_a, mdata_b])
        tx = result["xenium"].uns["transcripts"]
        assert tx.num_rows == 4
        assert set(tx["library_id"].to_pylist()) == {"run_A", "run_B"}

    def test_prefixes_cell_id_to_match_obs_names(self):
        mdata_a = _make_mdata_with_transcripts("run_A", n=2, cell_ids=["a0", "a1"])
        mdata_b = _make_mdata_with_transcripts("run_B", n=2, cell_ids=["b0", "b1"])
        result = concat_samples([mdata_a, mdata_b])
        tx = result["xenium"].uns["transcripts"]
        obs_names = set(result["xenium"].obs_names)
        assert set(tx["cell_id"].to_pylist()) <= obs_names

    def test_partial_presence_skips_merge_with_warning(self, caplog):
        mdata_a = _make_mdata_with_transcripts("run_A", n=2, cell_ids=["a0", "a1"])
        mdata_b = _make_mdata("run_B", n=2)
        with caplog.at_level(logging.WARNING):
            result = concat_samples([mdata_a, mdata_b])
        assert "transcripts" not in result["xenium"].uns
        assert "transcripts" in caplog.text.lower()

    def test_no_samples_have_transcripts_no_key_no_crash(self):
        mdata_a = _make_mdata("run_A", n=2)
        mdata_b = _make_mdata("run_B", n=2)
        result = concat_samples([mdata_a, mdata_b])
        assert "transcripts" not in result["xenium"].uns


class TestConcatSamplesDanglingGraphUns:
    def test_drops_stale_graph_metadata_whose_obsp_did_not_survive(self, caplog):
        mdata_a = _make_mdata_with_graph("run_A", n=3)
        mdata_b = _make_mdata_with_graph("run_B", n=3)
        with caplog.at_level(logging.WARNING):
            result = concat_samples([mdata_a, mdata_b])
        adata = result["xenium"]
        assert "spatial_neighbors" not in adata.uns
        assert "spatial_neighbors_connectivities" not in adata.obsp
        assert "dangling" in caplog.text.lower() or "spatial_neighbors" in caplog.text.lower()

    def test_no_graph_present_no_crash(self):
        mdata_a = _make_mdata("run_A", n=3)
        mdata_b = _make_mdata("run_B", n=3)
        result = concat_samples([mdata_a, mdata_b])
        assert "spatial_neighbors" not in result["xenium"].uns
