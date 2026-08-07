"""Tests for the mirror_uns mechanism added to with_adata (_utils.py).

Verifies that terminal-result `uns` writes are additively mirrored to
`mdata.uns[key][modality]` when a decorated function is called via MuData,
while leaving the bare-AnnData / bare-`@with_adata` call paths untouched.
"""
import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from astrolabe._utils import from_kwarg, with_adata
from astrolabe.tl._icg import interaction_changed_genes


@with_adata(mirror_uns=from_kwarg("key_added"))
def _stat_fn(adata, *, key_added="my_stat"):
    adata.uns[key_added] = {"val": 1}


@with_adata(mirror_uns="fixed_key")
def _conditional_fn(adata, *, write=True):
    if write:
        adata.uns["fixed_key"] = "written"


@with_adata(mirror_uns=lambda ba: ba.arguments["key_added"] or f"prefix_{ba.arguments['key']}")
def _dynamic_default_fn(adata, key, *, key_added=None):
    resolved = key_added or f"prefix_{key}"
    adata.uns[resolved] = f"result-for-{key}"


def test_mirrors_via_mudata(small_mdata):
    _stat_fn(small_mdata)
    assert small_mdata.uns["my_stat"]["xenium"] == {"val": 1}
    assert small_mdata["xenium"].uns["my_stat"] == {"val": 1}


def test_noop_via_bare_adata(small_adata):
    _stat_fn(small_adata)
    assert small_adata.uns["my_stat"] == {"val": 1}


def test_overwrite_warning_on_same_modality_rerun(small_mdata, caplog):
    _stat_fn(small_mdata)
    with caplog.at_level(logging.WARNING):
        _stat_fn(small_mdata)
    assert any("overwriting mdata.uns" in r.message for r in caplog.records)
    assert small_mdata.uns["my_stat"]["xenium"] == {"val": 1}


def test_different_keys_no_collision_no_warning(small_mdata, caplog):
    with caplog.at_level(logging.WARNING):
        _stat_fn(small_mdata, key_added="k1")
        _stat_fn(small_mdata, key_added="k2")
    assert small_mdata.uns["k1"]["xenium"] == {"val": 1}
    assert small_mdata.uns["k2"]["xenium"] == {"val": 1}
    assert not any("overwriting mdata.uns" in r.message for r in caplog.records)


def test_same_key_different_modalities_no_collision(small_mdata_two_mod, caplog):
    with caplog.at_level(logging.WARNING):
        _stat_fn(small_mdata_two_mod, key_added="shared", modality="xenium")
        _stat_fn(small_mdata_two_mod, key_added="shared", modality="merscope")
    assert small_mdata_two_mod.uns["shared"]["xenium"] == {"val": 1}
    assert small_mdata_two_mod.uns["shared"]["merscope"] == {"val": 1}
    assert not any("overwriting mdata.uns" in r.message for r in caplog.records)


def test_conditional_write_not_mirrored_when_skipped(small_mdata):
    _conditional_fn(small_mdata, write=False)
    assert "fixed_key" not in small_mdata.uns
    assert "fixed_key" not in small_mdata["xenium"].uns


def test_conditional_write_mirrored_when_present(small_mdata):
    _conditional_fn(small_mdata, write=True)
    assert small_mdata.uns["fixed_key"]["xenium"] == "written"


def test_from_kwarg_respects_caller_override(small_mdata):
    _stat_fn(small_mdata, key_added="custom")
    assert "custom" in small_mdata.uns
    assert "my_stat" not in small_mdata.uns


def test_dynamic_resolver_default_and_override(small_mdata):
    _dynamic_default_fn(small_mdata, "EPCAM")
    assert small_mdata.uns["prefix_EPCAM"]["xenium"] == "result-for-EPCAM"

    _dynamic_default_fn(small_mdata, "EPCAM", key_added="explicit")
    assert small_mdata.uns["explicit"]["xenium"] == "result-for-EPCAM"


def test_bare_with_adata_unaffected(small_mdata, small_adata):
    @with_adata
    def bare_fn(adata, x=1):
        return x * 2

    assert bare_fn(small_mdata) == 2
    assert bare_fn(small_adata) == 2


def _icg_mdata():
    n_per_type, n_genes = 20, 10
    n_cells = n_per_type * 2
    X = sp.random(n_cells, n_genes, density=0.6, format="csr", random_state=42).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = ["A"] * n_per_type + ["B"] * n_per_type
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    adj = np.zeros((n_cells, n_cells), dtype=np.float32)
    half = n_per_type // 2
    b_start = n_per_type
    for i in range(half):
        adj[i, b_start + i] = 1.0
        adj[b_start + i, i] = 1.0

    adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}

    mdata = md.MuData({"xenium": adata})
    mdata.uns["default_modality"] = "xenium"
    return mdata


def test_integration_interaction_changed_genes_mirrors_to_mudata():
    mdata = _icg_mdata()
    result = interaction_changed_genes(mdata, groupby="cell_type", min_cells=1)
    assert "icg" in mdata.uns
    mirrored = mdata.uns["icg"]["xenium"]
    pd.testing.assert_frame_equal(mirrored, mdata["xenium"].uns["icg"])
    pd.testing.assert_frame_equal(mirrored, result)
