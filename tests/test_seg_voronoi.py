"""Tests for seg._voronoi's X-staleness warning and provenance clearing."""
import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe.seg._voronoi import voronoi_resegment


def _grid_adata(n_side=5, with_x=True):
    xs, ys = np.meshgrid(np.arange(n_side) * 10.0, np.arange(n_side) * 10.0)
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    n = len(coords)
    adata = ad.AnnData(
        X=np.ones((n, 2), dtype=np.float32) if with_x else None,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["g0", "g1"]) if with_x else None,
    )
    adata.obsm["spatial"] = coords
    return adata


class TestVoronoiResegment:
    def test_warns_that_x_reflects_original_segmentation(self, caplog):
        adata = _grid_adata()
        with caplog.at_level(logging.WARNING):
            voronoi_resegment(adata)
        assert "ORIGINAL segmentation" in caplog.text
        assert "assign_transcripts" in caplog.text
        assert "cell_boundaries" in adata.obsm
        assert len(adata.obsm["cell_boundaries"]) == adata.n_obs

    def test_clears_xenium_native_provenance_flag(self):
        adata = _grid_adata()
        adata.uns["xenium_native_segmentation"] = True
        voronoi_resegment(adata)
        assert adata.uns["xenium_native_segmentation"] is False

    def test_no_warning_when_x_is_none(self, caplog):
        adata = _grid_adata(with_x=False)
        with caplog.at_level(logging.WARNING):
            voronoi_resegment(adata)
        assert "ORIGINAL segmentation" not in caplog.text
