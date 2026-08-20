"""Tests for _add_nucleus_metrics helper and calculate_qc_metrics (audit fix 4.11)."""
import numpy as np
import pandas as pd
import anndata as ad
import pyarrow as pa
import pytest
import scipy.sparse as sp

from astrolabe.pp._qc import (
    _add_nucleus_metrics,
    calculate_qc_metrics,
    calculate_xenium_qc_metrics,
    filter_cells,
    filter_genes,
    filter_isolated,
    calculate_shape_metrics,
    calculate_transcript_assignment_qc,
)


def _make_adata(**obs_cols):
    n = len(next(iter(obs_cols.values())))
    obs = pd.DataFrame(obs_cols, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(obs=obs)


class TestNucleusAreaRatio:
    def test_normal_ratio(self):
        adata = _make_adata(nucleus_area=[5.0, 10.0], cell_area=[20.0, 25.0])
        _add_nucleus_metrics(adata)
        ratios = adata.obs["nucleus_to_cell_area_ratio"].values
        np.testing.assert_allclose(ratios, [5.0 / 20.0, 10.0 / 25.0], rtol=1e-6)

    def test_zero_cell_area_yields_nan(self):
        adata = _make_adata(nucleus_area=[5.0], cell_area=[0.0])
        _add_nucleus_metrics(adata)
        assert np.isnan(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"])

    def test_zero_nucleus_area_yields_zero_ratio(self):
        adata = _make_adata(nucleus_area=[0.0], cell_area=[10.0])
        _add_nucleus_metrics(adata)
        assert abs(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"] - 0.0) < 1e-6

    def test_both_zero_yields_nan(self):
        adata = _make_adata(nucleus_area=[0.0], cell_area=[0.0])
        _add_nucleus_metrics(adata)
        assert np.isnan(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"])


class TestHasNucleus:
    def test_from_nucleus_count(self):
        adata = _make_adata(nucleus_count=[0, 1, 2])
        _add_nucleus_metrics(adata)
        assert list(adata.obs["has_nucleus"]) == [False, True, True]

    def test_from_nucleus_area_fallback(self):
        adata = _make_adata(nucleus_area=[0.0, 0.1, 5.0])
        _add_nucleus_metrics(adata)
        assert list(adata.obs["has_nucleus"]) == [False, True, True]

    def test_count_takes_priority_over_area(self):
        # nucleus_count present → use count even if nucleus_area also present
        adata = _make_adata(nucleus_count=[0, 1], nucleus_area=[10.0, 0.0])
        _add_nucleus_metrics(adata)
        # count=0 → False even though area=10; count=1 → True even though area=0
        assert list(adata.obs["has_nucleus"]) == [False, True]

    def test_no_nucleus_columns_adds_nothing(self):
        adata = _make_adata(total_counts=[10, 20])
        _add_nucleus_metrics(adata)
        assert "nucleus_to_cell_area_ratio" not in adata.obs.columns
        assert "has_nucleus" not in adata.obs.columns

    def test_only_cell_area_no_nucleus_area(self):
        adata = _make_adata(cell_area=[10.0, 20.0])
        _add_nucleus_metrics(adata)
        # Without nucleus_area, ratio is not computed
        assert "nucleus_to_cell_area_ratio" not in adata.obs.columns


# ---------------------------------------------------------------------------
# uns['transcripts'] sync on cell/gene filtering
# ---------------------------------------------------------------------------


def _adata_with_transcripts(n=4):
    # c2 is the low-count cell that gets filtered out below; g1 is a control
    # gene that gets excluded. "UNASSIGNED" mimics Xenium's real sentinel for
    # transcripts never assigned to any segmented cell.
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(np.array([[5, 0], [4, 1], [1, 0], [3, 2]], dtype=np.float32))
    obs = pd.DataFrame(
        {
            "total_counts": [100.0, 100.0, 2.0, 100.0],
            "n_genes_by_counts": [2, 2, 1, 2],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(
        {"is_control_probe": [False, True]},
        index=["g0", "g1"],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    adata.uns["transcripts"] = pa.table({
        "cell_id": ["c0", "c1", "c2", "c3", "c3", "UNASSIGNED", "UNASSIGNED"],
        "feature_name": ["g0", "g0", "g0", "g0", "g1", "g0", "g1"],
    })
    return adata


class TestFilterCellsSyncsTranscripts:
    def test_inplace_drops_removed_cells_keeps_unassigned(self):
        adata = _adata_with_transcripts()
        filter_cells(adata, min_counts=10, inplace=True)
        assert "c2" not in adata.obs_names
        tx = adata.uns["transcripts"]
        assert "c2" not in tx.column("cell_id").to_pylist()
        # UNASSIGNED rows aren't tied to any cell, so a cell filter must not touch them.
        assert tx.column("cell_id").to_pylist().count("UNASSIGNED") == 2

    def test_non_inplace_syncs_returned_copy_not_original(self):
        adata = _adata_with_transcripts()
        filtered, mask = filter_cells(adata, min_counts=10, inplace=False)
        assert "c2" not in filtered.uns["transcripts"].column("cell_id").to_pylist()
        # Original must be untouched.
        assert "c2" in adata.uns["transcripts"].column("cell_id").to_pylist()

    def test_no_transcripts_key_is_a_noop(self):
        # MERSCOPE/CosMx never populate uns['transcripts'] -- must not crash.
        adata = _adata_with_transcripts()
        del adata.uns["transcripts"]
        filter_cells(adata, min_counts=10, inplace=True)
        assert "transcripts" not in adata.uns


class TestFilterGenesSyncsTranscripts:
    def test_inplace_drops_removed_genes(self):
        adata = _adata_with_transcripts()
        filter_genes(adata, exclude_controls=True, inplace=True)
        assert "g1" not in adata.var_names
        tx = adata.uns["transcripts"]
        assert "g1" not in tx.column("feature_name").to_pylist()
        assert set(tx.column("feature_name").to_pylist()) == {"g0"}

    def test_non_inplace_syncs_returned_copy_not_original(self):
        adata = _adata_with_transcripts()
        filtered, mask = filter_genes(adata, exclude_controls=True, inplace=False)
        assert "g1" not in filtered.uns["transcripts"].column("feature_name").to_pylist()
        assert "g1" in adata.uns["transcripts"].column("feature_name").to_pylist()


class TestFilterIsolatedSyncsTranscripts:
    def test_inplace_drops_removed_cells(self):
        # c0/c1/c3 form a tight trio (~10 um apart); c2 sits 1000+ um away
        # (isolated).
        adata = _adata_with_transcripts()
        adata.obsm["spatial"] = np.array(
            [[0.0, 0.0], [10.0, 0.0], [1000.0, 1000.0], [0.0, 10.0]],
            dtype=np.float32,
        )
        filter_isolated(
            adata,
            max_neighbor_rank=2,
            max_neighbor_dist_um=50.0,
            min_neighbor_dist_um=50.0,
            inplace=True,
        )
        assert "c2" not in adata.obs_names
        assert "c2" not in adata.uns["transcripts"].column("cell_id").to_pylist()
        assert adata.uns["transcripts"].column("cell_id").to_pylist().count("UNASSIGNED") == 2


class TestFilterGenesNaNHandling:
    def _adata_with_controls(self, n=20, n_genes=10, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.integers(0, 5, (n, n_genes)).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{i}" for i in range(n_genes)]
        adata.var["is_control_probe"] = [False] * (n_genes - 2) + [True, True]
        # calculate_qc_metrics(exclude_controls=True)'s documented behavior:
        # per-gene metrics are NaN, not zeroed, on control rows.
        total_counts = np.asarray(X.sum(axis=0), dtype=np.float64)
        n_cells_by_counts = np.asarray((X > 0).sum(axis=0), dtype=np.float64)
        total_counts[-2:] = np.nan
        n_cells_by_counts[-2:] = np.nan
        adata.var["total_counts"] = total_counts
        adata.var["n_cells_by_counts"] = n_cells_by_counts
        return adata

    def test_min_cells_alone_does_not_remove_controls(self):
        # Regression: NaN cast to int32 becomes INT32_MIN, always failing
        # `>= min_cells` -- every control probe was silently removed by
        # min_cells/min_counts filtering alone, even when exclude_controls
        # was never requested at all.
        adata = self._adata_with_controls()
        filtered, mask = filter_genes(adata, min_cells=1, exclude_controls=False, inplace=False)
        assert filtered.var["is_control_probe"].sum() == 2
        assert mask[-2:].all()

    def test_min_counts_alone_does_not_remove_controls(self):
        adata = self._adata_with_controls()
        filtered, mask = filter_genes(adata, min_counts=1, exclude_controls=False, inplace=False)
        assert filtered.var["is_control_probe"].sum() == 2

    def test_exclude_controls_still_removes_them(self):
        adata = self._adata_with_controls()
        filtered, mask = filter_genes(adata, min_cells=1, exclude_controls=True, inplace=False)
        assert filtered.var["is_control_probe"].sum() == 0

    def test_real_genes_still_filtered_by_min_cells(self):
        adata = self._adata_with_controls()
        adata.var["n_cells_by_counts"] = np.concatenate([
            np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float64),
            [np.nan, np.nan],
        ])
        filtered, mask = filter_genes(adata, min_cells=3, exclude_controls=False, inplace=False)
        # 2 real genes below threshold removed; controls untouched by this check.
        assert mask[:2].sum() == 0
        assert mask[2:8].all()
        assert mask[-2:].all()


class TestCalculateShapeMetrics:
    def _adata_with_boundaries(self, n=10, size=5.0, seed=0):
        import awkward as ak
        rng = np.random.default_rng(seed)
        xy = rng.uniform(0, 500, (n, 2)).astype(np.float32)
        polys = [
            [[cx - size, cy - size], [cx + size, cy - size], [cx + size, cy + size], [cx - size, cy + size]]
            for cx, cy in xy
        ]
        adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
        adata.obsm["cell_boundaries"] = ak.Array(polys)
        return adata

    def test_overwrites_vendor_area_with_warning(self, caplog):
        import logging
        adata = self._adata_with_boundaries()
        adata.obs["cell_area"] = np.full(adata.n_obs, 999.0)  # vendor-supplied stand-in
        with caplog.at_level(logging.WARNING, logger="astrolabe.pp._qc"):
            calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        assert any("overwriting" in r.message for r in caplog.records)
        # Real polygon area (10x10 square = 100), not the vendor placeholder.
        assert np.allclose(adata.obs["cell_area"].to_numpy(), 100.0, atol=1.0)

    def test_no_warning_when_column_absent(self, caplog):
        import logging
        adata = self._adata_with_boundaries()
        with caplog.at_level(logging.WARNING, logger="astrolabe.pp._qc"):
            calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        assert not any("overwriting" in r.message for r in caplog.records)

    def test_degenerate_polygon_does_not_crash(self):
        import awkward as ak
        adata = self._adata_with_boundaries(n=3)
        # Overwrite one cell's boundary with a degenerate (collinear) ring.
        polys = ak.to_list(adata.obsm["cell_boundaries"])
        polys[1] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]]
        adata.obsm["cell_boundaries"] = ak.Array(polys)
        # Must not raise -- previously AttributeError on `.exterior` for a
        # degenerate polygon whose minimum_rotated_rectangle collapses to a
        # non-Polygon (Point/LineString) geometry type.
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        assert np.isnan(adata.obs["cell_bbox_major_axis_length"].to_numpy()[1])

    def test_aspect_ratio_from_bbox_axes(self):
        adata = self._adata_with_boundaries(n=5)
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        major = adata.obs["cell_bbox_major_axis_length"].to_numpy()
        minor = adata.obs["cell_bbox_minor_axis_length"].to_numpy()
        aspect = adata.obs["cell_aspect_ratio"].to_numpy()
        assert np.all(major >= minor)
        np.testing.assert_allclose(aspect, major / minor, rtol=1e-5)
        # Axis-aligned 10x10 squares -> both bbox sides 10, aspect 1.
        np.testing.assert_allclose(major, 10.0, atol=1e-3)
        np.testing.assert_allclose(aspect, 1.0, atol=1e-4)

    def test_bowtie_keeps_both_lobes(self):
        # Self-intersecting "bow-tie" ring: two 0.25-area triangles meeting
        # at (0.5, 0.5). make_valid keeps BOTH lobes (area 0.5); the old
        # buffer(0) repair silently kept a single lobe (area 0.25).
        import awkward as ak
        adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
        adata.obsm["cell_boundaries"] = ak.Array(
            [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]
        )
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        np.testing.assert_allclose(adata.obs["cell_area"].to_numpy(), 0.5, atol=1e-6)

    def test_unbuildable_ring_gets_all_nan_not_zero(self):
        # < 3 vertices: no polygon can be built -- every metric must be NaN,
        # never a fake 0.0 area that pollutes downstream statistics.
        import awkward as ak
        adata = self._adata_with_boundaries(n=3)
        polys = ak.to_list(adata.obsm["cell_boundaries"])
        polys[0] = [[0.0, 0.0], [1.0, 1.0]]
        adata.obsm["cell_boundaries"] = ak.Array(polys)
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        for col in [
            "cell_area", "cell_perimeter", "cell_span_x", "cell_span_y",
            "cell_equivalent_radius", "cell_bbox_major_axis_length",
            "cell_bbox_minor_axis_length", "cell_bbox_orientation",
            "cell_aspect_ratio", "cell_circularity", "cell_solidity",
        ]:
            assert np.isnan(adata.obs[col].to_numpy()[0]), col
            # Other rows unaffected.
            assert np.isfinite(adata.obs[col].to_numpy()[1:]).all(), col

    def test_nucleus_anchored_polarity_frame(self):
        import awkward as ak
        # 10x4 rectangle centered at the origin (major axis along x).
        cell = [[-5.0, -2.0], [5.0, -2.0], [5.0, 2.0], [-5.0, 2.0]]
        # cell 0: nucleus centroid (-3, 0) -> offset (+3, 0): positive major
        #   axis must point +x (the far half), minor sign untouched (dot 0).
        nuc_left = [[-4.0, -1.0], [-2.0, -1.0], [-2.0, 1.0], [-4.0, 1.0]]
        # cell 1: nucleus centroid (0, 0.75) -> offset (0, -0.75): minor
        #   flips to point -y (away from the nucleus).
        nuc_up = [[-1.0, 0.0], [1.0, 0.0], [1.0, 1.5], [-1.0, 1.5]]
        # cell 2: degenerate nucleus ring -> NaN polarity, finite bbox metrics.
        nuc_bad = [[0.0, 0.0], [1.0, 1.0]]
        adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
        adata.obsm["cell_boundaries"] = ak.Array([cell, cell, cell])
        adata.obsm["nucleus_boundaries"] = ak.Array([nuc_left, nuc_up, nuc_bad])
        calculate_shape_metrics(adata, counts_key=None)
        maj = adata.obs["cell_polarity_major_angle"].to_numpy()
        mino = adata.obs["cell_polarity_minor_angle"].to_numpy()
        off = adata.obs["cell_nucleus_offset_um"].to_numpy()
        assert min(maj[0], 2 * np.pi - maj[0]) == pytest.approx(0.0, abs=1e-4)
        assert mino[0] == pytest.approx(np.pi / 2, abs=1e-4)
        assert off[0] == pytest.approx(3.0, abs=1e-3)
        assert mino[1] == pytest.approx(3 * np.pi / 2, abs=1e-4)
        assert off[1] == pytest.approx(0.75, abs=1e-3)
        assert np.isnan(maj[2]) and np.isnan(mino[2]) and np.isnan(off[2])
        assert np.isfinite(adata.obs["cell_bbox_orientation"].to_numpy()[2])

    def test_polarity_skipped_without_nucleus_boundaries(self):
        adata = self._adata_with_boundaries(n=3)
        calculate_shape_metrics(adata, counts_key=None)
        assert "cell_polarity_major_angle" not in adata.obs
        assert "cell_nucleus_offset_um" not in adata.obs

    def test_bbox_orientation_recovers_planted_angles(self):
        import awkward as ak
        # 4x1 rectangles: axis-aligned (major axis along x -> 0), the same
        # rotated by 30 deg (-> pi/6), and a tall 1x4 (-> pi/2). Orientation
        # is mod pi (an axis is undirected).
        base = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])
        theta = np.pi / 6
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated = base @ rot.T
        tall = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 4.0], [0.0, 4.0]])
        adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
        adata.obsm["cell_boundaries"] = ak.Array(
            [base.tolist(), rotated.tolist(), tall.tolist()]
        )
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        ori = adata.obs["cell_bbox_orientation"].to_numpy()
        assert np.all((ori >= 0) & (ori < np.pi))
        assert min(ori[0], np.pi - ori[0]) == pytest.approx(0.0, abs=1e-4)
        assert ori[1] == pytest.approx(np.pi / 6, abs=1e-4)
        assert ori[2] == pytest.approx(np.pi / 2, abs=1e-4)

    def test_circularity_and_solidity_hand_computed(self):
        import awkward as ak
        # Square, side 10: A=100, P=40 -> circularity = 4*pi*100/1600 = pi/4.
        square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        # L-shape: area 12; convex hull area 14 -> solidity = 6/7.
        l_shape = [
            [0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [2.0, 4.0], [2.0, 2.0], [0.0, 2.0],
        ]
        adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
        adata.obsm["cell_boundaries"] = ak.Array([square, l_shape])
        calculate_shape_metrics(adata, boundaries_key="cell_boundaries", counts_key=None)
        np.testing.assert_allclose(
            adata.obs["cell_circularity"].to_numpy()[0], np.pi / 4.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            adata.obs["cell_solidity"].to_numpy()[0], 1.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            adata.obs["cell_area"].to_numpy()[1], 12.0, rtol=1e-5
        )
        np.testing.assert_allclose(
            adata.obs["cell_solidity"].to_numpy()[1], 12.0 / 14.0, rtol=1e-5
        )


def _xenium_like_adata():
    """4 cells x 4 features: 3 Gene Expression + 1 Negative Control Probe."""
    X = sp.csr_matrix(np.array(
        [
            [10, 20, 30, 5],   # 60 gene counts + 5 control
            [1, 2, 3, 0],      # 6 gene counts, no control
            [0, 0, 0, 0],      # zero-count cell
            [4, 4, 4, 12],     # 12 gene counts + 12 control
        ],
        dtype=np.float32,
    ))
    var = pd.DataFrame(
        {
            "feature_type": [
                "Gene Expression", "Gene Expression", "Gene Expression",
                "Negative Control Probe",
            ],
            "is_control_probe": [False, False, False, True],
        },
        index=["g0", "g1", "g2", "NegControlProbe_1"],
    )
    obs = pd.DataFrame(index=["c0", "c1", "c2", "c3"])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestXeniumPctColumns:
    def test_pct_uses_count_sums_not_nnz_features(self):
        # Regression (BLOCKING): pct columns divided the *number of detected
        # control features* by total_counts instead of the control *count
        # sums* -- e.g. 12 control counts out of 24 totals reported as
        # 1/24*100 ~ 4.2% instead of 50%.
        adata = _xenium_like_adata()
        calculate_xenium_qc_metrics(adata)
        totals = np.array([65.0, 6.0, np.nan, 24.0])  # all-feature count sums
        expected = (
            adata.obs["negative_control_probe_total"].to_numpy(np.float64)
            / totals * 100.0
        )
        got = adata.obs["pct_negative_control_probes"].to_numpy(np.float64)
        np.testing.assert_allclose(got[[0, 1, 3]], expected[[0, 1, 3]], rtol=1e-6)
        np.testing.assert_allclose(got[3], 50.0, rtol=1e-6)

    def test_pct_nan_on_zero_count_cell(self):
        adata = _xenium_like_adata()
        calculate_xenium_qc_metrics(adata)
        assert np.isnan(adata.obs["pct_negative_control_probes"].to_numpy()[2])
        assert np.isnan(adata.obs["pct_any_negative_control_probes"].to_numpy()[2])


class TestQCMetricsRerun:
    def test_second_call_updates_total_counts(self):
        # Regression (BLOCKING): with controls present, the merge dropped
        # every pre-existing QC column, so a re-run never refreshed
        # total_counts etc.
        adata = _xenium_like_adata()
        calculate_qc_metrics(adata)
        first = adata.obs["total_counts"].to_numpy().copy()
        np.testing.assert_allclose(first, [60.0, 6.0, 0.0, 12.0])
        adata.X = adata.X * 2.0
        calculate_qc_metrics(adata)
        second = adata.obs["total_counts"].to_numpy()
        np.testing.assert_allclose(second, first * 2.0)

    def test_total_counts_all_features_emitted_with_controls(self):
        adata = _xenium_like_adata()
        calculate_qc_metrics(adata)
        # total_counts is control-excluded; the all-features sum lives in
        # total_counts_all_features (matches vendor per-cell totals).
        np.testing.assert_allclose(
            adata.obs["total_counts"].to_numpy(), [60.0, 6.0, 0.0, 12.0]
        )
        np.testing.assert_allclose(
            adata.obs["total_counts_all_features"].to_numpy(),
            [65.0, 6.0, 0.0, 24.0],
        )

    def test_inplace_false_returns_both_totals(self):
        adata = _xenium_like_adata()
        obs_df, var_df = calculate_qc_metrics(adata, inplace=False)
        assert "total_counts_all_features" in obs_df.columns
        np.testing.assert_allclose(
            obs_df["total_counts"].to_numpy(), [60.0, 6.0, 0.0, 12.0]
        )
        # Original untouched.
        assert "total_counts" not in adata.obs.columns
        # Control rows stay NaN in per-gene metrics.
        assert np.isnan(var_df.loc["NegControlProbe_1", "total_counts"])


class TestFilterIsolatedOutbound:
    def test_far_cell_removed_even_with_max_only(self):
        # Regression (BLOCKING): with min_neighbor_dist_um=None, the old
        # obsp-based code initialized max_dists to 0 for cells with no
        # stored neighbors, so a fully isolated cell PASSED the max check.
        line = np.column_stack([np.arange(6) * 10.0, np.zeros(6)])
        coords = np.vstack([line, [[10000.0, 0.0]]]).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((7, 1), dtype=np.float32))
        adata.obs_names = [f"c{i}" for i in range(7)]
        adata.obsm["spatial"] = coords
        filtered, mask = filter_isolated(
            adata,
            max_neighbor_rank=5,
            max_neighbor_dist_um=60.0,
            min_neighbor_dist_um=None,
            inplace=False,
        )
        assert not mask[6]

    def test_outlier_does_not_delete_dense_region_victims(self):
        # Regression (SERIOUS): the isolation criterion used the
        # union-symmetrized graph, so an outlier's inbound edges deleted the
        # well-connected cells it pointed at. Outbound kNN distances only
        # ever reflect a cell's own isolation.
        line = np.column_stack([np.arange(6) * 10.0, np.zeros(6)])
        coords = np.vstack([line, [[10000.0, 0.0]]]).astype(np.float32)
        adata = ad.AnnData(X=np.zeros((7, 1), dtype=np.float32))
        adata.obs_names = [f"c{i}" for i in range(7)]
        adata.obsm["spatial"] = coords
        filtered, mask = filter_isolated(
            adata,
            max_neighbor_rank=5,
            max_neighbor_dist_um=60.0,
            min_neighbor_dist_um=50.0,
            inplace=False,
        )
        # All six line cells retained (their own 5th neighbor is <= 50 um);
        # only the outlier is removed.
        assert mask[:6].all()
        assert not mask[6]
        assert filtered.n_obs == 6

    def test_k_defaults_to_stored_graph_n_neighbors(self):
        line = np.column_stack([np.arange(6) * 10.0, np.zeros(6)])
        adata = ad.AnnData(X=np.zeros((6, 1), dtype=np.float32))
        adata.obsm["spatial"] = line.astype(np.float32)
        # Stored spatial_neighbors metadata with k=2: every cell's 2nd
        # neighbor is within 20 um -> nothing removed at 25 um. With the
        # fallback k=5 the endpoint cells' 5th neighbor is 50 um away and
        # they would be removed.
        adata.uns["spatial_neighbors"] = {"params": {"n_neighbors": 2}}
        filtered, mask = filter_isolated(
            adata, max_neighbor_dist_um=25.0, min_neighbor_dist_um=None,
            inplace=False,
        )
        assert mask.all()


class TestTranscriptSyncIntCellIds:
    def test_int_cell_id_column_syncs(self):
        # Regression (SERIOUS): value_set was always string-typed, so an
        # int-typed cell_id column matched nothing and the table silently
        # desynced from obs.
        X = sp.csr_matrix(np.array([[5.0], [1.0]], dtype=np.float32))
        obs = pd.DataFrame(
            {"total_counts": [5.0, 1.0], "n_genes_by_counts": [1, 1]},
            index=["1", "2"],
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = ["g0"]
        adata.uns["transcripts"] = pa.table({
            "cell_id": pa.array([1, 2, 0], type=pa.int64()),
            "feature_name": ["g0", "g0", "g0"],
        })
        filter_cells(adata, min_counts=2, inplace=True)
        ids = adata.uns["transcripts"].column("cell_id").to_pylist()
        assert 2 not in ids
        assert ids == [1, 0]
        assert adata.uns["transcripts_pruned"] is True

    def test_uncastable_obs_names_skip_sync(self):
        X = sp.csr_matrix(np.array([[5.0], [1.0]], dtype=np.float32))
        obs = pd.DataFrame(
            {"total_counts": [5.0, 1.0], "n_genes_by_counts": [1, 1]},
            index=["cell_a", "cell_b"],
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = ["g0"]
        adata.uns["transcripts"] = pa.table({
            "cell_id": pa.array([1, 2, 0], type=pa.int64()),
            "feature_name": ["g0", "g0", "g0"],
        })
        # Must not raise; no row can match, so the table is untouched.
        filter_cells(adata, min_counts=2, inplace=True)
        assert len(adata.uns["transcripts"]) == 3


class TestFilterGenesNaNGatedOnControls:
    def test_nan_on_non_control_row_fails_threshold(self):
        adata = ad.AnnData(X=np.ones((4, 3), dtype=np.float32))
        adata.var_names = ["g0", "g1", "ctrl"]
        adata.var["is_control_probe"] = [False, False, True]
        adata.var["total_counts"] = [4.0, np.nan, np.nan]
        filtered, mask = filter_genes(adata, min_counts=1, inplace=False)
        # NaN on the control row passes through; NaN on a real gene fails.
        assert list(mask) == [True, False, True]


class TestCalculateTranscriptAssignmentQC:
    def _adata_with_transcripts(self, cell_id_dtype: str):
        adata = ad.AnnData(X=np.zeros((3, 2), dtype=np.float32))
        adata.var_names = ["g0", "g1"]
        if cell_id_dtype == "string":
            cell_ids = pa.array(["cell_1", "UNASSIGNED", "cell_1", "UNASSIGNED"])
        else:
            cell_ids = pa.array([1, 0, 1, 0], type=pa.int64())
        adata.uns["transcripts"] = pa.table({
            "cell_id": cell_ids,
            "feature_name": ["g0", "g0", "g1", "g1"],
        })
        return adata

    def test_auto_detects_unassigned_string_sentinel(self):
        # Regression: default unassigned_id=0 silently reported 0%
        # unassigned for every gene on real Xenium XOA >= 2.0 output, whose
        # string-typed cell_id column uses 'UNASSIGNED', never numeric 0.
        adata = self._adata_with_transcripts("string")
        calculate_transcript_assignment_qc(adata)
        pct = adata.var["pct_transcripts_unassigned"].to_numpy()
        np.testing.assert_allclose(pct, [50.0, 50.0])

    def test_auto_detects_unassigned_numeric_sentinel(self):
        adata = self._adata_with_transcripts("numeric")
        calculate_transcript_assignment_qc(adata)
        pct = adata.var["pct_transcripts_unassigned"].to_numpy()
        np.testing.assert_allclose(pct, [50.0, 50.0])

    def test_explicit_override_still_works(self):
        adata = self._adata_with_transcripts("string")
        calculate_transcript_assignment_qc(adata, unassigned_id="cell_1")
        pct = adata.var["pct_transcripts_unassigned"].to_numpy()
        np.testing.assert_allclose(pct, [50.0, 50.0])

    def test_qv_min_excludes_low_quality_transcripts(self):
        # qv_min=20 (default) matches X's own transcript population
        # (read_xenium_transcripts min_qv=20): low-QV rows must not count
        # toward either the total or unassigned tallies.
        adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
        adata.var_names = ["g0"]
        adata.uns["transcripts"] = pa.table({
            "cell_id": ["cell_1", "UNASSIGNED", "UNASSIGNED", "cell_1"],
            "feature_name": ["g0", "g0", "g0", "g0"],
            "qv": [40.0, 40.0, 5.0, 5.0],
        })
        calculate_transcript_assignment_qc(adata)
        assert adata.var["n_transcripts_total"].to_numpy()[0] == 2
        np.testing.assert_allclose(
            adata.var["pct_transcripts_unassigned"].to_numpy(), [50.0]
        )
        # qv_min=None counts everything.
        calculate_transcript_assignment_qc(adata, qv_min=None)
        assert adata.var["n_transcripts_total"].to_numpy()[0] == 4

    def test_refuses_pruned_stored_table(self):
        adata = self._adata_with_transcripts("string")
        adata.uns["transcripts_pruned"] = True
        with pytest.raises(ValueError, match="pruned"):
            calculate_transcript_assignment_qc(adata)
        # Passing a fresh table explicitly bypasses the guard.
        fresh = adata.uns["transcripts"]
        calculate_transcript_assignment_qc(adata, transcripts=fresh)
        assert "pct_transcripts_unassigned" in adata.var.columns

    def test_flag_set_by_cell_filter_triggers_guard(self):
        # End-to-end: filter_cells prunes the stored table and sets the
        # flag; a later assignment-QC call on the pruned table refuses.
        X = sp.csr_matrix(np.array([[5.0], [1.0]], dtype=np.float32))
        obs = pd.DataFrame(
            {"total_counts": [5.0, 1.0], "n_genes_by_counts": [1, 1]},
            index=["cell_1", "cell_2"],
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = ["g0"]
        adata.uns["transcripts"] = pa.table({
            "cell_id": ["cell_1", "cell_2", "UNASSIGNED"],
            "feature_name": ["g0", "g0", "g0"],
        })
        filter_cells(adata, min_counts=2, inplace=True)
        assert adata.uns.get("transcripts_pruned") is True
        with pytest.raises(ValueError, match="pruned"):
            calculate_transcript_assignment_qc(adata)


class TestCalculateNoiseFloor:
    """pp.calculate_noise_floor + pp.filter_genes(exclude_below_noise_floor=True)."""

    @staticmethod
    def _adata(n_cells=10):
        # Column sums -> mean counts/cell: g_hi=5.0, g_mid=1.0, g_low=0.1,
        # controls c1=0.2, c2=0.4.
        X = np.zeros((n_cells, 5), dtype=np.float32)
        X[:, 0] = 5.0   # g_hi
        X[:, 1] = 1.0   # g_mid
        X[0, 2] = 1.0   # g_low
        X[0, 3] = 2.0   # c1
        X[0, 4] = 4.0   # c2
        var = pd.DataFrame(
            {
                "feature_type": ["Gene Expression"] * 3
                + ["Negative Probe", "System Control"],
                "is_control_probe": [False] * 3 + [True] * 2,
            },
            index=["g_hi", "g_mid", "g_low", "NegPrb1", "SystemControl1"],
        )
        return ad.AnnData(X=X, var=var)

    def test_threshold_and_flags(self):
        from astrolabe.pp import calculate_noise_floor

        adata = self._adata()
        calculate_noise_floor(adata, quantile=1.0)
        nf = adata.uns["noise_floor"]
        assert nf["threshold_mean_counts"] == pytest.approx(0.4)
        assert nf["n_controls"] == 2
        assert nf["control_source"] == "var"
        flags = adata.var["below_noise_floor"]
        assert bool(flags["g_low"])          # 0.1 <= 0.4
        assert not bool(flags["g_mid"])      # 1.0 > 0.4
        assert not bool(flags["g_hi"])
        # Controls are never flagged, whatever their own counts.
        assert not bool(flags["NegPrb1"])
        assert not bool(flags["SystemControl1"])
        assert nf["n_below"] == 1

    def test_multiplier_and_control_types(self):
        from astrolabe.pp import calculate_noise_floor

        adata = self._adata()
        # Restrict to the Negative Probe class (mean 0.2); x3 -> 0.6.
        calculate_noise_floor(
            adata, quantile=1.0, multiplier=3.0,
            control_types=["Negative Probe"],
        )
        nf = adata.uns["noise_floor"]
        assert nf["threshold_mean_counts"] == pytest.approx(0.6)
        assert nf["n_controls"] == 1
        assert nf["control_types"] == ["Negative Probe"]
        assert bool(adata.var["below_noise_floor"]["g_low"])
        assert not bool(adata.var["below_noise_floor"]["g_mid"])

    def test_log1p_guard(self):
        from astrolabe.pp import calculate_noise_floor

        adata = self._adata()
        adata.layers["counts"] = adata.X.copy()
        adata.uns["log1p"] = {"base": None}
        with pytest.raises(ValueError, match="log1p"):
            calculate_noise_floor(adata)
        # Explicit raw-counts layer bypasses the guard.
        calculate_noise_floor(adata, layer="counts", quantile=1.0)
        assert adata.uns["noise_floor"]["layer"] == "counts"

    def test_no_controls_raises(self):
        from astrolabe.pp import calculate_noise_floor

        adata = self._adata()
        adata.var["is_control_probe"] = False
        with pytest.raises(ValueError, match="No control features"):
            calculate_noise_floor(adata)

    def test_stash_fallback_after_control_removal(self):
        from astrolabe.pp import calculate_noise_floor

        # Xenium-like: stash written by calculate_xenium_qc_metrics, controls
        # then removed from var (pipeline.filter_xenium's remove_controls).
        adata = self._adata()
        adata.var["feature_type"] = [
            "Gene Expression", "Gene Expression", "Gene Expression",
            "Negative Control Probe", "Negative Control Codeword",
        ]
        calculate_xenium_qc_metrics(adata)
        filter_genes(adata, exclude_controls=True, inplace=True)
        assert not adata.var["is_control_probe"].any()

        calculate_noise_floor(adata, quantile=1.0)
        nf = adata.uns["noise_floor"]
        assert nf["control_source"] == "control_probe_summary"
        assert nf["threshold_mean_counts"] == pytest.approx(0.4)
        assert bool(adata.var["below_noise_floor"]["g_low"])

    def test_filter_genes_gate(self):
        from astrolabe.pp import calculate_noise_floor

        adata = self._adata()
        calculate_noise_floor(adata, quantile=1.0)
        filtered, mask = filter_genes(adata, exclude_below_noise_floor=True)
        assert "g_low" not in filtered.var_names
        # Genes above the floor and ALL controls survive this gate.
        assert set(filtered.var_names) == {"g_hi", "g_mid", "NegPrb1", "SystemControl1"}

    def test_filter_genes_gate_requires_flag(self):
        adata = self._adata()
        with pytest.raises(KeyError, match="below_noise_floor"):
            filter_genes(adata, exclude_below_noise_floor=True)


class TestAutoNoiseFloor:
    """calculate_qc_metrics(noise_floor=True, the default) computes the noise
    floor automatically, and skips gracefully where it can't."""

    def test_computed_by_default(self):
        adata = TestCalculateNoiseFloor._adata()
        calculate_qc_metrics(adata)
        assert "noise_floor" in adata.uns
        assert "below_noise_floor" in adata.var.columns
        assert bool(adata.var["below_noise_floor"]["g_low"])
        assert not adata.var.loc[adata.var["is_control_probe"], "below_noise_floor"].any()

    def test_opt_out(self):
        adata = TestCalculateNoiseFloor._adata()
        calculate_qc_metrics(adata, noise_floor=False)
        assert "noise_floor" not in adata.uns
        assert "below_noise_floor" not in adata.var.columns

    def test_skips_on_log1p(self):
        adata = TestCalculateNoiseFloor._adata()
        adata.uns["log1p"] = {"base": None}
        calculate_qc_metrics(adata)  # must not raise
        assert "noise_floor" not in adata.uns

    def test_skips_without_controls(self):
        adata = TestCalculateNoiseFloor._adata()
        adata.var["is_control_probe"] = False
        calculate_qc_metrics(adata)  # must not raise
        assert "noise_floor" not in adata.uns

    def test_skips_when_not_inplace(self):
        adata = TestCalculateNoiseFloor._adata()
        calculate_qc_metrics(adata, inplace=False)
        assert "noise_floor" not in adata.uns
        assert "below_noise_floor" not in adata.var.columns
