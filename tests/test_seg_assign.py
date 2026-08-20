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


def _two_cells_with_nuclei():
    """Two 10x10 cells side by side, each with a centered 4x4 nucleus.

    Cell A: [0,10]x[0,10], nucleus [3,7]x[3,7].
    Cell B: [10,20]x[0,10], nucleus [13,17]x[3,7].
    """
    cell_a = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], dtype=np.float32)
    cell_b = np.array([[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]], dtype=np.float32)
    nuc_a = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], dtype=np.float32)
    nuc_b = np.array([[13, 3], [17, 3], [17, 7], [13, 7], [13, 3]], dtype=np.float32)
    adata = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["cellA", "cellB"]),
        var=pd.DataFrame(index=["g0"]),
    )
    adata.obsm["cell_boundaries"] = ak.Array([cell_a, cell_b])
    adata.obsm["nucleus_boundaries"] = ak.Array([nuc_a, nuc_b])
    adata.obsm["spatial"] = np.array([[5.0, 5.0], [15.0, 5.0]], dtype=np.float32)
    return adata


class TestCreateCompartmentLayersPlantedPartition:
    """Ground-truth partition: known nucleus / cytoplasm / ambient counts."""

    def test_known_partition_recovered(self):
        adata = _two_cells_with_nuclei()
        transcripts = pd.DataFrame({
            #                       expected compartment (cell)
            "x_location": [5.0,     # A nucleus (inside nuc_a)
                           1.0,     # A cytoplasm (in cell A, outside nuc_a)
                           15.0,    # B nucleus
                           19.0,    # B cytoplasm
                           30.0],   # ambient (outside both cells, nearest B)
            "y_location": [5.0, 1.0, 5.0, 9.0, 5.0],
            "feature_name": ["g0"] * 5,
        })
        create_compartment_layers(adata, transcripts, overlaps_nucleus_col=None)
        nuc = np.asarray(adata.layers["nucleus"].todense())
        cyt = np.asarray(adata.layers["cytoplasm"].todense())
        amb = np.asarray(adata.layers["ambient"].todense())
        np.testing.assert_array_equal(nuc, [[1.0], [1.0]])
        np.testing.assert_array_equal(cyt, [[1.0], [1.0]])
        np.testing.assert_array_equal(amb, [[0.0], [1.0]])
        # Full partition: every transcript in exactly one layer.
        assert nuc.sum() + cyt.sum() + amb.sum() == 5.0

    def test_neighbor_nucleus_transcript_is_own_cell_cytoplasm(self):
        # Regression (BLOCKING): a transcript inside neighbor B's nucleus but
        # assigned to cell A (possible with overlapping/foreign
        # segmentations) was booked as A's *nuclear* signal because the
        # strtree path tested "inside ANY nucleus", not "inside the assigned
        # cell's own nucleus". Build the geometry directly: cell A overlaps
        # B's nucleus region.
        cell_a = np.array([[0, 0], [15.0, 0], [15.0, 10], [0, 10], [0, 0]], dtype=np.float32)
        cell_b = np.array([[10, 0], [20, 0], [20, 10], [10, 10], [10, 0]], dtype=np.float32)
        nuc_a = np.array([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]], dtype=np.float32)
        nuc_b = np.array([[13, 3], [17, 3], [17, 7], [13, 7], [13, 3]], dtype=np.float32)
        adata = ad.AnnData(
            X=np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["cellA", "cellB"]),
            var=pd.DataFrame(index=["g0"]),
        )
        # Make A the smaller cell so the overlap tie-break assigns the
        # contested point to A deterministically.
        adata.obsm["cell_boundaries"] = ak.Array([cell_a, cell_b])
        adata.obsm["nucleus_boundaries"] = ak.Array([nuc_a, nuc_b])
        adata.obsm["spatial"] = np.array([[7.5, 5.0], [15.0, 5.0]], dtype=np.float32)
        # Shrink A so it is smaller than B: [0,14]x[0,7] (area 98 < 100)
        cell_a2 = np.array([[0, 0], [14.0, 0], [14.0, 7], [0, 7], [0, 0]], dtype=np.float32)
        adata.obsm["cell_boundaries"] = ak.Array([cell_a2, cell_b])

        # Transcript at (13.5, 4): inside cell A (and B's overlap), inside
        # B's nucleus, NOT inside A's nucleus. Assigned to A (smaller area)
        # -> must be A's CYTOPLASM, not nuclear anywhere.
        transcripts = pd.DataFrame({
            "x_location": [13.5],
            "y_location": [4.0],
            "feature_name": ["g0"],
        })
        create_compartment_layers(adata, transcripts, overlaps_nucleus_col=None)
        nuc = np.asarray(adata.layers["nucleus"].todense())
        cyt = np.asarray(adata.layers["cytoplasm"].todense())
        np.testing.assert_array_equal(nuc, [[0.0], [0.0]])
        np.testing.assert_array_equal(cyt, [[1.0], [0.0]])

    def test_fast_path_requires_native_xenium_provenance(self):
        # overlaps_nucleus column present but no provenance flag -> STRtree
        # path must be used (the column describes foreign nuclei).
        adata = _two_cells_with_nuclei()
        transcripts = pd.DataFrame({
            "x_location": [1.0],   # A cytoplasm by geometry
            "y_location": [1.0],
            "feature_name": ["g0"],
            "overlaps_nucleus": [True],  # lies (would book as nuclear)
        })
        create_compartment_layers(adata, transcripts)
        assert adata.layers["nucleus"].sum() == 0.0
        assert adata.layers["cytoplasm"].sum() == 1.0

    def test_fast_path_used_with_native_flag(self):
        adata = _two_cells_with_nuclei()
        adata.uns["xenium_native_segmentation"] = True
        transcripts = pd.DataFrame({
            "x_location": [1.0],
            "y_location": [1.0],
            "feature_name": ["g0"],
            "overlaps_nucleus": [True],
        })
        create_compartment_layers(adata, transcripts)
        # With the flag set, the column is trusted verbatim.
        assert adata.layers["nucleus"].sum() == 1.0
        assert adata.layers["cytoplasm"].sum() == 0.0


class TestFeatureNameHandling:
    def test_bytes_feature_names_normalized(self):
        adata = _two_cells_with_nuclei()
        transcripts = pd.DataFrame({
            "x_location": [5.0],
            "y_location": [5.0],
            "feature_name": np.array([b"g0"], dtype=object),
        })
        create_compartment_layers(adata, transcripts, overlaps_nucleus_col=None)
        assert adata.layers["nucleus"].sum() == 1.0

    def test_all_skipped_raises(self):
        import pytest
        adata = _two_cells_with_nuclei()
        transcripts = pd.DataFrame({
            "x_location": [5.0, 15.0],
            "y_location": [5.0, 5.0],
            "feature_name": ["not_a_gene", "also_not"],
        })
        with pytest.raises(ValueError, match="feature name"):
            create_compartment_layers(adata, transcripts, overlaps_nucleus_col=None)

    def test_assign_transcripts_bytes_names_decoded(self):
        adata = _overlapping_cells()
        transcripts = pd.DataFrame({
            "x_location": [1.0],
            "y_location": [1.0],
            "feature_name": np.array([b"g0"], dtype=object),
        })
        result = assign_transcripts(adata, transcripts, inplace=False)
        assert list(result.var_names) == ["g0"]
        assert result.X.sum() == 1.0


class TestSmallestAreaTieBreak:
    def test_overlap_goes_to_smaller_cell(self):
        # Cell A: [0,6]x[0,6] area 36; Cell B: [4,10]x[4,10]... use
        # _overlapping_cells then enlarge B so areas differ decisively.
        poly_a = np.array([[0, 0], [6, 0], [6, 6], [0, 6], [0, 0]], dtype=np.float32)
        poly_b = np.array([[4, 4], [20, 4], [20, 20], [4, 20], [4, 4]], dtype=np.float32)
        adata = ad.AnnData(
            X=np.zeros((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["small", "big"]),
            var=pd.DataFrame(index=["g0"]),
        )
        adata.obsm["cell_boundaries"] = ak.Array([poly_a, poly_b])
        adata.obsm["spatial"] = np.array([[3.0, 3.0], [12.0, 12.0]], dtype=np.float32)
        transcripts = pd.DataFrame({
            "x_location": [5.0],  # in the overlap of both
            "y_location": [5.0],
            "feature_name": ["g0"],
        })
        result = assign_transcripts(adata, transcripts, inplace=False)
        X = np.asarray(result.X.todense())
        assert X[0, 0] == 1.0  # smaller cell wins the tie
        assert X[1, 0] == 0.0

    def test_assign_transcripts_inplace_constructs_valid_anndata(self):
        adata = _overlapping_cells()
        transcripts = pd.DataFrame({
            "x_location": [1.0, 7.0],
            "y_location": [1.0, 7.0],
            "feature_name": ["g0", "g1"],
        })
        assign_transcripts(adata, transcripts, inplace=True)
        # Rebuilt through validated construction: dims consistent, obsm kept.
        assert adata.n_vars == 2
        assert adata.X.shape == (2, 2)
        assert "cell_boundaries" in adata.obsm
        assert "spatial" in adata.obsm
        assert list(adata.obs_names) == ["cellA", "cellB"]


class TestTranscriptsDefaultFromUns:
    """transcripts=None resolves from adata.uns['transcripts'] (pa.Table or
    DataFrame) -- the pattern that makes a pp.crop-culled table usable with
    no manual read_xenium_transcripts re-read."""

    def _uns_table(self):
        import pyarrow as pa
        return pa.table({
            "x_location": [1.0],   # inside cell A only
            "y_location": [1.0],
            "feature_name": ["g0"],
        })

    def test_assign_transcripts_defaults_to_uns(self):
        adata = _overlapping_cells()
        adata.uns["transcripts"] = self._uns_table()
        explicit = assign_transcripts(adata, adata.uns["transcripts"], inplace=False)
        defaulted = assign_transcripts(adata, inplace=False)
        np.testing.assert_array_equal(
            np.asarray(defaulted.X.todense()), np.asarray(explicit.X.todense())
        )
        assert defaulted.X.sum() == 1.0

    def test_assign_transcripts_missing_uns_raises(self):
        import pytest
        adata = _overlapping_cells()
        with pytest.raises(ValueError, match="uns\['transcripts'\]"):
            assign_transcripts(adata, inplace=False)

    def test_create_compartment_layers_defaults_to_uns(self):
        adata = _two_cells_with_nuclei()
        adata.uns["transcripts"] = self._uns_table()
        create_compartment_layers(adata, overlaps_nucleus_col=None)
        assert adata.layers["cytoplasm"].sum() == 1.0
        assert adata.layers["nucleus"].sum() == 0.0

    def test_create_compartment_layers_missing_uns_raises(self):
        import pytest
        adata = _two_cells_with_nuclei()
        with pytest.raises(ValueError, match="uns\['transcripts'\]"):
            create_compartment_layers(adata, overlaps_nucleus_col=None)

    def test_create_compartment_layers_default_via_mudata(self):
        # @with_adata resolution: the uns lookup must happen on the resolved
        # modality, not the MuData wrapper.
        import mudata as md
        adata = _two_cells_with_nuclei()
        adata.uns["transcripts"] = self._uns_table()
        mdata = md.MuData({"xenium": adata})
        mdata.uns["default_modality"] = "xenium"
        create_compartment_layers(mdata, overlaps_nucleus_col=None)
        assert mdata["xenium"].layers["cytoplasm"].sum() == 1.0

    def test_pruned_flag_does_not_raise(self, caplog):
        # Deliberate divergence from pp.calculate_transcript_assignment_qc:
        # assignment/layering are not rate computations, so a pruned table is
        # valid input -- logged, never raised.
        import logging
        adata = _overlapping_cells()
        adata.uns["transcripts"] = self._uns_table()
        adata.uns["transcripts_pruned"] = True
        with caplog.at_level(logging.INFO):
            result = assign_transcripts(adata, inplace=False)
        assert result.X.sum() == 1.0
        assert "pruned" in caplog.text
