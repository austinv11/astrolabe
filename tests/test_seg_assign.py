"""Tests for seg._assign's transcript-in-overlapping-polygons double-count
fix (batch-5 bugfix pass)."""
import numpy as np
import pandas as pd
import anndata as ad
import awkward as ak

from astrolabe.seg._assign import assign_transcripts, create_compartment_layers


def _overlapping_cells():
    # Cell A: [0,0]-[6,6]; Cell B: [4,4]-[10,10] -- overlap region [4,6]x[4,6].
    poly_a = np.array([[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]], dtype=np.float32)
    poly_b = np.array([[4, 4], [10, 4], [10, 10], [4, 10], [4, 4]], dtype=np.float32)
    adata = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["cellA", "cellB"]),
        var=pd.DataFrame(index=["g0"]),
    )
    adata.obsm["cell_boundaries"] = ak.Array([poly_a, poly_b])
    adata.obsm["spatial"] = np.array([[3.0, 3.0], [7.0, 7.0]], dtype=np.float32)
    return adata


class TestAssignTranscriptsNoDoubleCount:
    def test_transcript_in_overlap_counted_once(self):
        # Regression: cell_tree.query(pts, predicate="within") returns one
        # row per (point, polygon) match -- a transcript inside >1
        # overlapping cell polygon appeared once per overlapping cell,
        # summed into *every* overlapping cell's total instead of exactly
        # one.
        adata = _overlapping_cells()
        transcripts = pd.DataFrame({
            "x_location": [5.0],  # inside both A and B's overlap region
            "y_location": [5.0],
            "feature_name": ["g0"],
        })
        result = assign_transcripts(adata, transcripts, inplace=False)
        assert result.X.sum() == 1.0

    def test_transcript_outside_overlap_still_assigned_correctly(self):
        adata = _overlapping_cells()
        transcripts = pd.DataFrame({
            "x_location": [1.0],  # inside A only
            "y_location": [1.0],
            "feature_name": ["g0"],
        })
        result = assign_transcripts(adata, transcripts, inplace=False)
        X = np.asarray(result.X.todense())
        assert X[0, 0] == 1.0
        assert X[1, 0] == 0.0


class TestCreateCompartmentLayersNoDoubleCount:
    def test_transcript_in_overlap_counted_once_across_layers(self):
        adata = _overlapping_cells()
        transcripts = pd.DataFrame({
            "x_location": [5.0],
            "y_location": [5.0],
            "feature_name": ["g0"],
        })
        create_compartment_layers(adata, transcripts, overlaps_nucleus_col=None)
        total = adata.layers["cytoplasm"].sum() + adata.layers["ambient"].sum()
        assert total == 1.0
