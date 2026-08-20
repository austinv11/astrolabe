"""Tests for astrolabe.pipeline recipes (filter_cells / filter_xenium / filter_merscope).

Covers the audit fixes: top-level MuData staying in sync after in-place
filtering (data.update()), automatic flag_controls when max_pct_control is
requested, and MERSCOPE volume bounds.
"""
import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import pytest
import scipy.sparse as sp

import astrolabe.pipeline as pipeline


def _grid_coords(n, spacing=10.0):
    return np.array(
        [[(i % 4) * spacing, (i // 4) * spacing] for i in range(n)],
        dtype=np.float32,
    )


def _xenium_like_mdata():
    """10 cells x 8 features (6 Gene Expression + 2 Negative Control Probe).

    - c0..c7: 5 counts on every real gene (total 30, 6 genes detected), clean.
    - c8: 1 count on one gene -> fails min_counts=10.
    - c9: 4 counts on every real gene (total 24) but 96 control counts
      -> 80% control, fails max_pct_control=10.
    """
    n_cells, n_real, n_ctrl = 10, 6, 2
    X = np.zeros((n_cells, n_real + n_ctrl), dtype=np.float32)
    X[:8, :n_real] = 5.0
    X[8, 0] = 1.0
    X[9, :n_real] = 4.0
    X[9, n_real:] = 48.0  # 96 control counts total
    var = pd.DataFrame(
        {
            "feature_type": ["Gene Expression"] * n_real
            + ["Negative Control Probe"] * n_ctrl,
            "is_control_probe": [False] * n_real + [True] * n_ctrl,
        },
        index=[f"g{i}" for i in range(n_real)]
        + [f"NegControlProbe_{i}" for i in range(n_ctrl)],
    )
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_cells)])
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.obsm["spatial"] = _grid_coords(n_cells)
    mdata = md.MuData({"xenium": adata})
    mdata.uns["default_modality"] = "xenium"
    return mdata


def _merscope_like_mdata():
    """9 cells x 7 features (6 real + 1 Blank control); one low-volume cell."""
    n_cells = 9
    X = np.zeros((n_cells, 7), dtype=np.float32)
    X[:, :6] = 5.0
    var = pd.DataFrame(
        {"is_control_probe": [False] * 6 + [True]},
        index=[f"g{i}" for i in range(6)] + ["Blank-1"],
    )
    obs = pd.DataFrame(
        {"volume": [500.0] * 8 + [50.0]},
        index=[f"c{i}" for i in range(n_cells)],
    )
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.obsm["spatial"] = _grid_coords(n_cells)
    mdata = md.MuData({"merscope": adata})
    mdata.uns["default_modality"] = "merscope"
    return mdata


class TestFilterCellsMuDataSync:
    def test_top_level_mudata_updated_after_filtering(self):
        # Regression (SERIOUS): the recipes filtered the modality in place
        # but never called mdata.update(), leaving the top-level n_obs and
        # obs_names stale.
        mdata = _xenium_like_mdata()
        assert mdata.n_obs == 10
        pipeline.filter_cells(mdata)
        assert mdata["xenium"].n_obs < 10
        assert mdata.n_obs == mdata["xenium"].n_obs
        assert list(mdata.obs_names) == list(mdata["xenium"].obs_names)
        assert "c8" not in mdata.obs_names

    def test_auto_flag_controls_when_max_pct_control_set(self):
        # Regression (MINOR): max_pct_control without a prior flag_controls
        # call raised a bare KeyError; the recipe now derives
        # obs['pct_counts_control'] from var['is_control_probe'] itself.
        mdata = _xenium_like_mdata()
        pipeline.filter_cells(mdata, max_pct_control=10.0)
        assert "c9" not in mdata["xenium"].obs_names
        assert "pct_counts_control" in mdata["xenium"].obs.columns

    def test_max_pct_control_skipped_with_warning_when_unflaggable(self, caplog):
        import logging

        mdata = _xenium_like_mdata()
        del mdata["xenium"].var["is_control_probe"]
        with caplog.at_level(logging.WARNING, logger="astrolabe.pipeline"):
            pipeline.filter_cells(mdata, max_pct_control=10.0)
        assert any("skipping the control-percentage filter" in r.message
                   for r in caplog.records)
        # c9 passes min_counts (24 counts over all features when controls
        # are unflagged: 24 + 96 = 120) and is retained.
        assert "c9" in mdata["xenium"].obs_names


class TestFilterXenium:
    def test_removes_low_count_and_control_heavy_cells_and_controls(self):
        mdata = _xenium_like_mdata()
        pipeline.filter_xenium(mdata)
        adata = mdata["xenium"]
        assert "c8" not in adata.obs_names          # min_counts
        assert "c9" not in adata.obs_names          # max_pct_control (80% > 10%)
        assert adata.n_obs == 8
        assert adata.n_vars == 6                    # controls removed
        assert not adata.var["is_control_probe"].any()
        # Top-level MuData synced on both axes.
        assert mdata.n_obs == adata.n_obs
        assert list(mdata.obs_names) == list(adata.obs_names)

    def test_isolation_uses_own_outbound_distances(self):
        # An outlier cell far from the grid is removed by the isolation
        # step without dragging its nearest grid cells along.
        mdata = _xenium_like_mdata()
        coords = mdata["xenium"].obsm["spatial"].copy()
        coords[7] = [10000.0, 10000.0]  # c7 becomes a distant outlier
        mdata["xenium"].obsm["spatial"] = coords
        pipeline.filter_xenium(mdata)
        adata = mdata["xenium"]
        assert "c7" not in adata.obs_names
        # The remaining grid cells all survive.
        assert adata.n_obs == 7
        assert mdata.n_obs == 7


class TestFilterMerscope:
    def test_volume_bounds_route_to_vendor_volume(self):
        mdata = _merscope_like_mdata()
        pipeline.filter_merscope(mdata, min_volume=100.0)
        adata = mdata["merscope"]
        assert "c8" not in adata.obs_names          # volume 50 < 100
        assert adata.n_obs == 8
        assert "Blank-1" not in adata.var_names     # remove_controls default
        assert mdata.n_obs == adata.n_obs

    def test_volume_bounds_require_volume_column(self):
        mdata = _merscope_like_mdata()
        del mdata["merscope"].obs["volume"]
        with pytest.raises(KeyError, match="volume"):
            pipeline.filter_merscope(mdata, min_volume=100.0)

    def test_no_volume_filter_by_default(self):
        mdata = _merscope_like_mdata()
        pipeline.filter_merscope(mdata)
        # Low-volume cell passes every default threshold (30 counts, 6 genes).
        assert "c8" in mdata["merscope"].obs_names
