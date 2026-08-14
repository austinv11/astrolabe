import json

import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import astrolabe as ab
from astrolabe.io._visium_hd import (
    _resolve_matrix_h5,
    _resolve_pixel_size,
    _resolve_cytassist_pixel_size,
    _find_he_image,
)


# ---------------------------------------------------------------------------
# Synthetic Space Ranger HD-like fixture
# ---------------------------------------------------------------------------


def _write_10x_h5(path, X, gene_ids, gene_names, barcodes, feature_types=None, genome="test_genome"):
    """Write a minimal 10x feature-barcode HDF5 matrix (CSC, genes x barcodes)."""
    import h5py

    X = np.asarray(X)
    n_genes = X.shape[1]
    Xt_csc = sp.csc_matrix(X.T)  # genes x bins CSC — bins are the CSC "columns"
    if feature_types is None:
        feature_types = ["Gene Expression"] * n_genes

    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=Xt_csc.data.astype(np.int32))
        g.create_dataset("indices", data=Xt_csc.indices.astype(np.int32))
        g.create_dataset("indptr", data=Xt_csc.indptr.astype(np.int32))
        g.create_dataset("shape", data=np.array([X.shape[1], X.shape[0]], dtype=np.int64))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        feat = g.create_group("features")
        feat.create_dataset("id", data=np.array(gene_ids, dtype="S"))
        feat.create_dataset("name", data=np.array(gene_names, dtype="S"))
        feat.create_dataset("feature_type", data=np.array(feature_types, dtype="S"))
        feat.create_dataset("genome", data=np.array([genome] * n_genes, dtype="S"))


@pytest.fixture
def small_visium_hd_dir(tmp_path):
    """A tiny on-disk Space Ranger HD-like directory: a 4x4 bin grid, 5 genes."""
    rng = np.random.default_rng(0)
    n_row, n_col = 4, 4
    bin_size_um = 2.0
    px_per_bin = 10.0
    pixel_size_um = bin_size_um / px_per_bin  # 0.2

    rows, cols = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing="ij")
    array_row = rows.ravel()
    array_col = cols.ravel()
    n_bins = n_row * n_col

    barcodes = [f"bin_{i:04d}-1" for i in range(n_bins)]
    gene_names = [f"Gene{i}" for i in range(5)]
    gene_ids = [f"ENSG{i:05d}" for i in range(5)]
    X = rng.poisson(2, size=(n_bins, 5)).astype(np.int32)

    outs_dir = tmp_path / "outs"
    square_dir = outs_dir / "binned_outputs" / "square_002um"
    spatial_dir = square_dir / "spatial"
    spatial_dir.mkdir(parents=True)

    _write_10x_h5(square_dir / "filtered_feature_bc_matrix.h5", X, gene_ids, gene_names, barcodes)

    pxl_row = array_row * px_per_bin + 100.0
    pxl_col = array_col * px_per_bin + 200.0
    tp = pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": np.ones(n_bins, dtype=np.int8),
            "array_row": array_row,
            "array_col": array_col,
            "pxl_row_in_fullres": pxl_row,
            "pxl_col_in_fullres": pxl_col,
        }
    )
    tp.to_parquet(spatial_dir / "tissue_positions.parquet")

    scalefactors = {
        "microns_per_pixel": pixel_size_um,
        "tissue_hires_scalef": 0.1,
        "tissue_lowres_scalef": 0.05,
        "fiducial_diameter_fullres": 1.0,
        "spot_diameter_fullres": 1.0,
    }
    (spatial_dir / "scalefactors_json.json").write_text(json.dumps(scalefactors))

    return outs_dir, X, barcodes, gene_names, pixel_size_um


# ---------------------------------------------------------------------------
# _resolve_matrix_h5
# ---------------------------------------------------------------------------


class TestResolveMatrixH5:
    def test_prefers_filtered_bc_over_everything(self, tmp_path):
        for name in [
            "raw_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5",
            "filtered_feature_cell_matrix.h5", "raw_feature_cell_matrix.h5",
        ]:
            (tmp_path / name).write_bytes(b"")
        assert _resolve_matrix_h5(tmp_path).name == "filtered_feature_bc_matrix.h5"

    def test_falls_back_to_filtered_cell_naming(self, tmp_path):
        (tmp_path / "raw_feature_cell_matrix.h5").write_bytes(b"")
        (tmp_path / "filtered_feature_cell_matrix.h5").write_bytes(b"")
        assert _resolve_matrix_h5(tmp_path).name == "filtered_feature_cell_matrix.h5"

    def test_falls_back_to_raw_cell_naming_when_only_option(self, tmp_path):
        (tmp_path / "raw_feature_cell_matrix.h5").write_bytes(b"")
        assert _resolve_matrix_h5(tmp_path).name == "raw_feature_cell_matrix.h5"

    def test_raises_when_nothing_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_matrix_h5(tmp_path)


# ---------------------------------------------------------------------------
# read_visium_hd
# ---------------------------------------------------------------------------


class TestReadVisiumHD:
    def test_basic_load(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)

        assert "visium_hd_bins" in mdata.mod
        assert mdata.uns["default_modality"] == "visium_hd_bins"
        bins = mdata["visium_hd_bins"]
        assert bins.n_obs == len(barcodes)
        assert bins.n_vars == len(gene_names)
        assert bins.X is None
        assert set(bins.obs.columns) == {"array_row", "array_col", "in_tissue"}
        assert bins.obsm["spatial"].shape == (len(barcodes), 2)
        assert set(bins.var_names) == set(gene_names)

    def test_bin_matrix_spec(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        bins = mdata["visium_hd_bins"]
        lib = next(iter(bins.uns["spatial"]))
        spec = bins.uns["spatial"][lib]["bin_matrix"]
        assert spec.bin_size_um == 2.0
        assert spec.n_obs == len(barcodes)
        assert spec.n_vars == len(gene_names)

    def test_pixel_size_from_scalefactors(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        bins = mdata["visium_hd_bins"]
        expected_x = (bins.obs["array_col"].to_numpy() * 10.0 + 200.0) * pixel_size_um
        expected_y = (bins.obs["array_row"].to_numpy() * 10.0 + 100.0) * pixel_size_um
        np.testing.assert_allclose(bins.obsm["spatial"][:, 0], expected_x, atol=1e-4)
        np.testing.assert_allclose(bins.obsm["spatial"][:, 1], expected_y, atol=1e-4)

    def test_missing_binned_outputs_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ab.io.read_visium_hd(str(tmp_path / "nonexistent"), load_he=False, load_segmented=False)


# ---------------------------------------------------------------------------
# CytAssist fallback pixel-scale bug (audit item #18 / batch 8)
#
# Regression: the CytAssist registration-target image is captured by a
# different scanner at a different pixel resolution than the fullres tissue
# image bin centroids are placed in. _attach_he_image used to silently
# register *any* fallback image (including CytAssist) with the bin-space
# pixel_size_um, placing it in a coordinate frame disjoint from the actual
# bin/cell data. _find_he_image now reports which branch matched, and
# _resolve_cytassist_pixel_size resolves the CytAssist image's own scale
# from a separate scalefactors_json.json key -- never falling back to the
# fullres pixel size.
# ---------------------------------------------------------------------------


class TestFindHEImage:
    def test_finds_fullres_pattern_not_cytassist(self, tmp_path):
        outs_dir = tmp_path / "outs"
        spatial_dir = outs_dir / "spatial"
        spatial_dir.mkdir(parents=True)
        (outs_dir / "sample_tissue_image.tif").write_bytes(b"")
        (spatial_dir / "cytassist_image.tiff").write_bytes(b"")

        found = _find_he_image(outs_dir)

        assert found is not None
        img_path, is_cytassist = found
        assert img_path.name == "sample_tissue_image.tif"
        assert is_cytassist is False

    def test_falls_back_to_cytassist_and_flags_it(self, tmp_path):
        outs_dir = tmp_path / "outs"
        spatial_dir = outs_dir / "spatial"
        spatial_dir.mkdir(parents=True)
        (spatial_dir / "cytassist_image.tiff").write_bytes(b"")

        found = _find_he_image(outs_dir)

        assert found is not None
        img_path, is_cytassist = found
        assert img_path.name == "cytassist_image.tiff"
        assert is_cytassist is True

    def test_returns_none_when_nothing_found(self, tmp_path):
        outs_dir = tmp_path / "outs"
        outs_dir.mkdir()
        assert _find_he_image(outs_dir) is None


class TestResolveCytAssistPixelSize:
    def test_resolves_regist_target_key(self, tmp_path):
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        (spatial_dir / "scalefactors_json.json").write_text(json.dumps({
            "microns_per_pixel": 0.2,
            "regist_target_img_scalef_microns_per_pixel": 1.375,
        }))

        assert _resolve_cytassist_pixel_size(spatial_dir) == pytest.approx(1.375)

    def test_returns_none_when_key_absent_does_not_fall_back_to_fullres_scale(self, tmp_path):
        # The critical regression check: no silent substitution of the
        # fullres microns_per_pixel when the CytAssist-specific key is missing.
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        (spatial_dir / "scalefactors_json.json").write_text(json.dumps({
            "microns_per_pixel": 0.2,
        }))

        assert _resolve_cytassist_pixel_size(spatial_dir) is None

    def test_returns_none_when_file_absent(self, tmp_path):
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        assert _resolve_cytassist_pixel_size(spatial_dir) is None


class TestResolvePixelSizeIgnoresCytAssistKey:
    def test_does_not_fall_back_to_regist_target_key(self, tmp_path):
        # Regression: _resolve_pixel_size used to try
        # regist_target_img_scalef_microns_per_pixel as a second choice when
        # microns_per_pixel was absent -- but that key describes a *different*
        # image's scale, not the fullres frame bin centroids are placed in.
        # Absent microns_per_pixel must fall through to the empirical
        # bin-grid-spacing derivation instead.
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        (spatial_dir / "scalefactors_json.json").write_text(json.dumps({
            "regist_target_img_scalef_microns_per_pixel": 1.375,
        }))
        tp = pd.DataFrame({
            "array_row": [0, 0],
            "array_col": [0, 1],
            "pxl_col_in_fullres": [100.0, 110.0],
        })

        resolved = _resolve_pixel_size(spatial_dir, tp, bin_size_um=2.0)

        assert resolved != pytest.approx(1.375)
        assert resolved == pytest.approx(2.0 / 10.0)


class TestReadVisiumHDCytAssistFallback:
    def test_load_he_skips_with_warning_when_only_cytassist_and_no_scale_key(
        self, small_visium_hd_dir, caplog
    ):
        import logging

        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        spatial_dir = outs_dir / "binned_outputs" / "square_002um" / "spatial"
        (spatial_dir / "cytassist_image.tiff").write_bytes(b"")

        with caplog.at_level(logging.WARNING):
            mdata = ab.io.read_visium_hd(str(outs_dir), load_he=True, load_segmented=False)

        assert "he" not in mdata.mod
        assert "cannot reliably register" in caplog.text.lower()


class TestNativeSegmentation:
    def _write_segmented_outputs(self, outs_dir, gene_names, pixel_size_um):
        seg_dir = outs_dir / "segmented_outputs"
        seg_dir.mkdir()
        n_cells = 2
        cell_barcodes = [f"cellid_{i + 1:09d}-1" for i in range(n_cells)]
        Xc = np.zeros((n_cells, len(gene_names)), dtype=np.int32)
        Xc[0, 0] = 5
        Xc[1, 1] = 7
        _write_10x_h5(
            seg_dir / "filtered_feature_bc_matrix.h5", Xc,
            [f"ENSG{i:05d}" for i in range(len(gene_names))], gene_names, cell_barcodes,
        )
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"cell_id": 1},
                 "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}},
                {"type": "Feature", "properties": {"cell_id": 2},
                 "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]]}},
            ],
        }
        (seg_dir / "cell_segmentations.geojson").write_text(json.dumps(geojson))
        return Xc

    def test_loads_and_becomes_primary(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        Xc = self._write_segmented_outputs(outs_dir, gene_names, pixel_size_um)

        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=True)

        assert "visium_hd_cells" in mdata.mod
        assert mdata.uns["default_modality"] == "visium_hd_cells"
        cells = mdata["visium_hd_cells"]
        assert cells.n_obs == 2
        assert "cell_boundaries" in cells.obsm
        np.testing.assert_array_equal(np.asarray(cells.X.todense()), Xc)

    def test_default_modality_stays_bins_when_absent(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=True)
        assert "visium_hd_cells" not in mdata.mod
        assert mdata.uns["default_modality"] == "visium_hd_bins"

    def test_loads_with_cell_naming_convention(self, small_visium_hd_dir):
        # Regression: a real Space Ranger v4.0.1 segmented_outputs/ directory
        # ships filtered_feature_cell_matrix.h5 (never a _bc_ variant) --
        # post-segmentation columns are cells, not raw bin barcodes, so 10x
        # renames the matrix accordingly. _resolve_matrix_h5 only knew the
        # binned_outputs-style "bc" naming and raised FileNotFoundError
        # against a real download until fixed.
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        seg_dir = outs_dir / "segmented_outputs"
        seg_dir.mkdir()
        n_cells = 2
        cell_barcodes = [f"cellid_{i + 1:09d}-1" for i in range(n_cells)]
        Xc = np.zeros((n_cells, len(gene_names)), dtype=np.int32)
        Xc[0, 0] = 5
        Xc[1, 1] = 7
        _write_10x_h5(
            seg_dir / "filtered_feature_cell_matrix.h5", Xc,
            [f"ENSG{i:05d}" for i in range(len(gene_names))], gene_names, cell_barcodes,
        )
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"cell_id": 1},
                 "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}},
                {"type": "Feature", "properties": {"cell_id": 2},
                 "geometry": {"type": "Polygon", "coordinates": [[[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]]}},
            ],
        }
        (seg_dir / "cell_segmentations.geojson").write_text(json.dumps(geojson))

        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=True)

        assert "visium_hd_cells" in mdata.mod
        cells = mdata["visium_hd_cells"]
        assert cells.n_obs == 2
        np.testing.assert_array_equal(np.asarray(cells.X.todense()), Xc)


# ---------------------------------------------------------------------------
# pp.load_bin_counts
# ---------------------------------------------------------------------------


class TestLoadBinCounts:
    def test_full_load(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        ab.pp.load_bin_counts(mdata)
        loaded = np.asarray(mdata["visium_hd_bins"].X.todense())
        np.testing.assert_array_equal(loaded, X)

    def test_full_load_with_duplicate_gene_names(self, tmp_path):
        # Regression: the full-load branch used to reindex by name
        # (full[target.obs_names, target.var_names]), which raises
        # InvalidIndexError as soon as var_names has a duplicate -- routine
        # for 10x gene panels (multiple gene_ids sharing one gene symbol).
        # astrolabe's own var_names are gene_name-indexed with no
        # var_names_make_unique() call (matches Xenium/MERSCOPE/CosMx
        # convention: reference genes by symbol), so this must be supported,
        # not just tolerated.
        rng = np.random.default_rng(1)
        n_row, n_col = 3, 3
        bin_size_um = 2.0
        px_per_bin = 10.0
        pixel_size_um = bin_size_um / px_per_bin
        rows, cols = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing="ij")
        n_bins = n_row * n_col
        barcodes = [f"bin_{i:04d}-1" for i in range(n_bins)]
        gene_names = ["Gene0", "Gene0", "Gene1"]  # duplicate symbol
        gene_ids = ["ENSG00000", "ENSG00001", "ENSG00002"]  # unique ids
        X = rng.poisson(2, size=(n_bins, 3)).astype(np.int32)

        outs_dir = tmp_path / "outs"
        square_dir = outs_dir / "binned_outputs" / "square_002um"
        spatial_dir = square_dir / "spatial"
        spatial_dir.mkdir(parents=True)
        _write_10x_h5(square_dir / "filtered_feature_bc_matrix.h5", X, gene_ids, gene_names, barcodes)
        tp = pd.DataFrame({
            "barcode": barcodes,
            "in_tissue": np.ones(n_bins, dtype=np.int8),
            "array_row": rows.ravel(),
            "array_col": cols.ravel(),
            "pxl_row_in_fullres": rows.ravel() * px_per_bin + 100.0,
            "pxl_col_in_fullres": cols.ravel() * px_per_bin + 200.0,
        })
        tp.to_parquet(spatial_dir / "tissue_positions.parquet")
        (spatial_dir / "scalefactors_json.json").write_text(json.dumps({
            "microns_per_pixel": pixel_size_um,
            "tissue_hires_scalef": 0.1, "tissue_lowres_scalef": 0.05,
            "fiducial_diameter_fullres": 1.0, "spot_diameter_fullres": 1.0,
        }))

        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        assert mdata["visium_hd_bins"].var_names.tolist() == gene_names  # duplicate preserved
        ab.pp.load_bin_counts(mdata)
        loaded = np.asarray(mdata["visium_hd_bins"].X.todense())
        np.testing.assert_array_equal(loaded, X)

    def test_region_um_partial_load(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        coords = mdata["visium_hd_bins"].obsm["spatial"]
        region = (
            float(coords[:, 0].min()) - 1, float(coords[:, 1].min()) - 1,
            float(coords[:, 0].min()) + 3, float(coords[:, 1].max()) + 1,
        )
        expected_mask = (
            (coords[:, 0] >= region[0]) & (coords[:, 0] <= region[2]) &
            (coords[:, 1] >= region[1]) & (coords[:, 1] <= region[3])
        )
        assert 0 < expected_mask.sum() < len(expected_mask)  # region actually cuts the grid

        ab.pp.load_bin_counts(mdata, region_um=region)
        result_bins = mdata["visium_hd_bins"]
        assert result_bins.n_obs == int(expected_mask.sum())
        np.testing.assert_array_equal(np.asarray(result_bins.X.todense()), X[expected_mask])

    def test_genes_subset(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        keep = [gene_names[0], gene_names[2]]
        ab.pp.load_bin_counts(mdata, genes=keep)
        result_bins = mdata["visium_hd_bins"]
        assert result_bins.n_vars == 2
        assert set(result_bins.var_names) == set(keep)
        np.testing.assert_array_equal(
            np.asarray(result_bins.X.todense()).sum(), X[:, [0, 2]].sum()
        )

    def test_bin_ids_partial_load(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        subset_ids = barcodes[:3]
        ab.pp.load_bin_counts(mdata, bin_ids=subset_ids)
        result_bins = mdata["visium_hd_bins"]
        assert result_bins.n_obs == 3
        assert list(result_bins.obs_names) == subset_ids
        np.testing.assert_array_equal(np.asarray(result_bins.X.todense()), X[:3])

    def test_inplace_false_leaves_original_untouched(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        result = ab.pp.load_bin_counts(mdata, inplace=False)
        assert mdata["visium_hd_bins"].X is None
        assert result.X is not None
        np.testing.assert_array_equal(np.asarray(result.X.todense()), X)


# ---------------------------------------------------------------------------
# tl.rebin_visium_hd
# ---------------------------------------------------------------------------


class TestRebinVisiumHD:
    def test_rebin_sums_correctly(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        ab.pp.load_bin_counts(mdata)

        result = ab.tl.rebin_visium_hd(mdata, target_bin_um=4.0)

        assert result.n_obs == 4  # 4x4 grid, factor=2 -> 2x2 groups
        assert result.n_vars == X.shape[1]
        assert result.X.sum() == X.sum()
        assert "visium_hd_bins_4um" in mdata.mod

    def test_rebin_without_materialized_x_loads_full(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        result = ab.tl.rebin_visium_hd(mdata, target_bin_um=4.0)
        assert result.X.sum() == X.sum()

    def test_non_integer_multiple_raises(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        with pytest.raises(ValueError):
            ab.tl.rebin_visium_hd(mdata, target_bin_um=3.0)


# ---------------------------------------------------------------------------
# seg.assign_bins — proves the PCA tie-break differs from nearest-centroid
# ---------------------------------------------------------------------------


def _make_bin2cell_fixture():
    """Two candidate cells with distinct gene profiles, plus one bin that is
    geometrically closer to cell B but transcriptionally identical to cell A's
    profile — a case where 'nearest' and 'pca' tie-breaks must disagree."""
    # Cell A: square [0,4]x[0,4], centroid (2,2). Cell B: square [10,14]x[0,4], centroid (12,2).
    poly_a = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=np.float32)
    poly_b = np.array([[10, 0], [14, 0], [14, 4], [10, 4], [10, 0]], dtype=np.float32)

    cell_seg = ad.AnnData(
        obs=pd.DataFrame({"cell_area": [16.0, 16.0]}, index=["cellA", "cellB"])
    )
    cell_seg.obsm["spatial"] = np.array([[2.0, 2.0], [12.0, 2.0]], dtype=np.float32)
    cell_seg.obsm["cell_boundaries"] = ak.Array([poly_a, poly_b])

    # 3 core bins per cell (inside the un-expanded polygons) + 1 contested bin
    # at x=8 (geometrically nearer to B's centroid, distance 4 vs A's distance
    # 6) whose own counts match cell A's profile, not cell B's.
    coords = np.array(
        [[1, 2], [2, 2], [3, 2], [11, 2], [12, 2], [13, 2], [8, 2]], dtype=np.float32
    )
    X = np.array(
        [
            [10, 0], [10, 0], [10, 0],   # core A: GeneA-dominant
            [0, 10], [0, 10], [0, 10],   # core B: GeneB-dominant
            [10, 0],                      # contested: GeneA-dominant, like A
        ],
        dtype=np.float32,
    )
    bin_data = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=[f"bin{i}" for i in range(7)]),
        var=pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene_name")),
    )
    bin_data.obsm["spatial"] = coords
    return bin_data, cell_seg


class TestAssignBins:
    def test_pca_tie_break_uses_transcriptional_similarity(self):
        bin_data, cell_seg = _make_bin2cell_fixture()

        result = ab.seg.assign_bins(
            bin_data, cell_seg, expand_um=6.0, tie_break="pca", min_bin_counts=1, inplace=False,
        )

        X = np.asarray(result.X.todense())
        # Contested bin's [10, 0] profile matches cell A -> A gets 4 bins / [40, 0],
        # B keeps its 3 core bins / [0, 30] -- NOT the geometrically-nearer choice.
        np.testing.assert_array_equal(X[0], [40.0, 0.0])
        np.testing.assert_array_equal(X[1], [0.0, 30.0])
        assert result.obs["n_bins"].tolist() == [4, 3]

    def test_nearest_tie_break_differs_from_pca(self):
        bin_data, cell_seg = _make_bin2cell_fixture()

        result = ab.seg.assign_bins(
            bin_data, cell_seg, expand_um=6.0, tie_break="nearest", min_bin_counts=1, inplace=False,
        )

        X = np.asarray(result.X.todense())
        # Nearest-centroid assigns the contested bin to B (geometrically closer),
        # dragging its GeneA-dominant counts into B's row.
        np.testing.assert_array_equal(X[0], [30.0, 0.0])
        np.testing.assert_array_equal(X[1], [10.0, 30.0])
        assert result.obs["n_bins"].tolist() == [3, 4]

    def test_inplace_true_mutates_cell_seg(self):
        bin_data, cell_seg = _make_bin2cell_fixture()
        result = ab.seg.assign_bins(bin_data, cell_seg, expand_um=6.0, inplace=True)
        assert result is None
        assert cell_seg.X is not None
        assert cell_seg.n_obs == 2

    def test_min_bin_counts_drops_unfed_cells(self):
        bin_data, _ = _make_bin2cell_fixture()
        # Build cell_seg with a third cell far from every bin -> 0 assigned bins.
        poly_a = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=np.float32)
        poly_b = np.array([[10, 0], [14, 0], [14, 4], [10, 4], [10, 0]], dtype=np.float32)
        poly_c = np.array([[100, 100], [104, 100], [104, 104], [100, 104], [100, 100]], dtype=np.float32)
        cell_seg = ad.AnnData(
            obs=pd.DataFrame({"cell_area": [16.0, 16.0, 16.0]}, index=["cellA", "cellB", "cellC"])
        )
        cell_seg.obsm["spatial"] = np.array([[2.0, 2.0], [12.0, 2.0], [102.0, 102.0]], dtype=np.float32)
        cell_seg.obsm["cell_boundaries"] = ak.Array([poly_a, poly_b, poly_c])

        result = ab.seg.assign_bins(bin_data, cell_seg, expand_um=6.0, min_bin_counts=1, inplace=False)

        assert result.n_obs == 2
        assert "cellC" not in result.obs_names

    def test_expanded_boundaries_key_present(self):
        bin_data, cell_seg = _make_bin2cell_fixture()
        result = ab.seg.assign_bins(bin_data, cell_seg, expand_um=6.0, inplace=False)
        assert "cell_boundaries_expanded" in result.obsm

    def test_no_core_bin_cell_can_still_win_a_contested_bin(self):
        # Regression: a cell with zero core/unambiguous-expansion bins of
        # its own was structurally excluded from `eligible_candidates` for
        # *any* contested bin it competed for, so it could never win one --
        # not just disadvantaged, excluded outright, regardless of true
        # geometric or transcriptional similarity. Demonstrated on synthetic
        # data where every bin (core and contested) has an *identical*
        # expression vector -- there is no transcriptional basis to prefer
        # either cell -- so the correct tie-break is purely geometric.
        #
        # Cell A: square [0,4]x[0,4], centroid (2,2), 3 core bins.
        # Cell B: square [10,14]x[0,4], centroid (12,2), *zero* core bins.
        # Contested bin at (9,2): distance to A=7, to B=3 -- B is the
        # correct (nearer) match once B is allowed to compete at all.
        poly_a = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=np.float32)
        poly_b = np.array([[10, 0], [14, 0], [14, 4], [10, 4], [10, 0]], dtype=np.float32)
        cell_seg = ad.AnnData(
            obs=pd.DataFrame({"cell_area": [16.0, 16.0]}, index=["cellA", "cellB"])
        )
        cell_seg.obsm["spatial"] = np.array([[2.0, 2.0], [12.0, 2.0]], dtype=np.float32)
        cell_seg.obsm["cell_boundaries"] = ak.Array([poly_a, poly_b])

        coords = np.array([[1, 2], [2, 2], [3, 2], [9, 2]], dtype=np.float32)
        X = np.array(
            [[10, 0], [10, 0], [10, 0], [10, 0]], dtype=np.float32
        )  # identical profile everywhere -- no transcriptional signal at all
        bin_data = ad.AnnData(
            X=sp.csr_matrix(X),
            obs=pd.DataFrame(index=[f"bin{i}" for i in range(4)]),
            var=pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene_name")),
        )
        bin_data.obsm["spatial"] = coords

        result = ab.seg.assign_bins(
            bin_data, cell_seg, expand_um=6.0, tie_break="pca", min_bin_counts=1, inplace=False,
        )
        # cellB (no core bins, but geometrically nearer) must win the
        # contested bin -- not automatically lose it to cellA.
        assert result.obs["n_bins"]["cellB"] == 1
        assert result.obs["n_bins"]["cellA"] == 3


# ---------------------------------------------------------------------------
# seg.segment_bin2cell — dispatch-only (no stardist/cellpose required)
# ---------------------------------------------------------------------------


class TestSegmentBin2Cell:
    def test_requires_exactly_one_source(self, small_mdata):
        with pytest.raises(ValueError):
            ab.seg.segment_bin2cell(small_mdata, use_he=True, use_expression=True)
        with pytest.raises(ValueError):
            ab.seg.segment_bin2cell(small_mdata, use_he=False, use_expression=False)

    def test_missing_he_modality_raises(self, small_visium_hd_dir):
        outs_dir, X, barcodes, gene_names, pixel_size_um = small_visium_hd_dir
        mdata = ab.io.read_visium_hd(str(outs_dir), load_he=False, load_segmented=False)
        with pytest.raises(KeyError):
            ab.seg.segment_bin2cell(mdata, use_he=True)
