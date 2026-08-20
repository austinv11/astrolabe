"""Tests for pp._if's Z-stack channel-axis handling (batch-5 bugfix pass),
H&E reduction/dtype-scale handling, and spatially-sorted batching."""
import anndata as ad
import awkward as ak
import mudata as md
import numpy as np
import pandas as pd
import pytest
import tifffile

from astrolabe._utils import ImageSpec, _dtype_scale
from astrolabe.pp._if import (
    _extract_intensities_from_spec,
    _resolve_channel_axis,
    _read_tile,
    extract_if_intensities,
)


class TestResolveChannelAxis:
    def test_explicit_c_axis_used(self):
        assert _resolve_channel_axis("CYX", 3) == 0
        assert _resolve_channel_axis("YXC", 3) == 2

    def test_z_stack_with_no_channel_returns_none(self):
        # Regression: a real Xenium morphology.ome.tif is 'ZYX' (multiple
        # DAPI z-planes, no antibody panel at all) -- this used to fall
        # through to the ndim==3 default (axis 0), silently treating the Z
        # axis as if it were a channel axis and reporting 11 spurious
        # "channels" for a single-stain z-stack.
        assert _resolve_channel_axis("ZYX", 3) is None

    def test_t_stack_with_no_channel_returns_none(self):
        assert _resolve_channel_axis("TYX", 3) is None

    def test_ambiguous_3d_with_no_labels_falls_back_to_axis_0(self):
        # No 'C', 'Z', or 'T' label at all -- preserve the prior best-guess
        # fallback for genuinely unlabeled arrays.
        assert _resolve_channel_axis("QYX", 3) == 0


class TestReadTileCollapsesZWhenNoChannel:
    def test_z_stack_max_projects_to_single_channel(self):
        # 5-plane 'ZYX' stack; channel_axis=None should collapse Z via
        # max-projection to a single implicit channel, not report 5.
        z, h, w = 5, 20, 20
        rng = np.random.default_rng(0)
        image = rng.uniform(0, 1, (z, h, w)).astype(np.float32)

        tile = _read_tile(image, None, row_min=2, row_max=10, col_min=3, col_max=12)

        assert tile.shape[0] == 1  # collapsed to a single channel
        assert tile.shape[1:] == (8, 9)
        expected = image[:, 2:10, 3:12].max(axis=0)
        np.testing.assert_allclose(tile[0], expected)

    def test_explicit_channel_axis_still_works_unchanged(self):
        c, h, w = 3, 20, 20
        rng = np.random.default_rng(1)
        image = rng.uniform(0, 1, (c, h, w)).astype(np.float32)
        tile = _read_tile(image, 0, row_min=0, row_max=5, col_min=0, col_max=5)
        assert tile.shape == (3, 5, 5)
        np.testing.assert_allclose(tile, image[:, 0:5, 0:5])


def _rgb_spec(path, img, stain_matrix=None):
    tifffile.imwrite(str(path), img, photometric="rgb")
    with tifffile.TiffFile(str(path)) as tf:
        shape = tf.series[0].shape
    return ImageSpec(
        path=str(path),
        pixel_size_um=1.0,  # µm == px, so polygons can be written directly
        dimension_order="YXC",
        full_shape=tuple(shape),
        level_shapes=[tuple(shape)],
        level_scales=[(1.0, 1.0)],
        stain_matrix=stain_matrix,
        is_brightfield=True,
    )


def _square(x0, y0, size):
    return np.array(
        [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
        dtype=np.float32,
    )


class TestDtypeScale:
    def test_integer_dtypes_map_to_type_max(self):
        assert _dtype_scale(np.uint8) == 255.0
        assert _dtype_scale(np.uint16) == 65535.0
        assert _dtype_scale(np.dtype("uint16")) == 65535.0

    def test_float_dtypes_map_to_one(self):
        assert _dtype_scale(np.float32) == 1.0
        assert _dtype_scale(np.float64) == 1.0


class TestHEDeconvolutionExtraction:
    def test_reduction_honored_for_he(self, tmp_path):
        # Regression: `reduction` was silently ignored on the H&E stain
        # path (always mean). Cell covers a gradient region -> max and mean
        # of the deconvolved concentrations must differ.
        h = w = 30
        img = np.full((h, w, 3), 220, dtype=np.uint8)
        # Darker (more stain) toward larger x inside the cell.
        for col in range(10, 20):
            img[10:20, col] = 220 - (col - 10) * 12
        spec = _rgb_spec(tmp_path / "he.tif", img, stain_matrix=np.eye(3))
        boundaries = ak.Array([_square(10, 10, 9)])

        mean_res = _extract_intensities_from_spec(boundaries, spec, reduction="mean")
        max_res = _extract_intensities_from_spec(boundaries, spec, reduction="max")
        assert np.all(max_res[0] >= mean_res[0])
        assert np.any(max_res[0] > mean_res[0] + 1e-4)

    def test_uint16_he_produces_nonzero_concentrations(self, tmp_path):
        # Regression: the data-dependent /255 guess normalized 16-bit
        # brightfield pixels to >1, which then clipped to 1.0 -> OD 0 ->
        # all-zero X. The dtype-derived scale (65535) must yield real OD.
        h = w = 20
        img = np.full((h, w, 3), 30000, dtype=np.uint16)  # mid-gray tissue
        spec = _rgb_spec(tmp_path / "he16.tif", img, stain_matrix=np.eye(3))
        boundaries = ak.Array([_square(5, 5, 10)])

        res = _extract_intensities_from_spec(boundaries, spec, reduction="mean")
        expected_od = -np.log10(30000.0 / 65535.0)
        np.testing.assert_allclose(res[0], expected_od, rtol=0.05)

    def test_spatial_sort_preserves_cell_output_order(self, tmp_path):
        # Cells scattered in an order deliberately not spatially sorted,
        # small batch_size: each cell's extracted value must still land at
        # its own row (identity check catches any index mixing from the
        # sorted batching).
        h = w = 64
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(h, w, 3)).astype(np.uint8)
        spec = _rgb_spec(tmp_path / "if.tif", img)  # no stain matrix: raw RGB
        origins = [(2, 2), (50, 50), (10, 40), (40, 6), (26, 26)]
        boundaries = ak.Array([_square(x, y, 4) for x, y in origins])

        res_small_batches = _extract_intensities_from_spec(
            boundaries, spec, reduction="mean", batch_size=2
        )
        res_one_batch = _extract_intensities_from_spec(
            boundaries, spec, reduction="mean", batch_size=100
        )
        np.testing.assert_allclose(res_small_batches, res_one_batch, rtol=1e-6)
        # Sanity: rows genuinely differ from each other (order is testable).
        assert not np.allclose(res_small_batches[0], res_small_batches[1])


class TestExtractIfIntensitiesCopiesSpatial:
    def test_filled_modality_carries_cell_centroids(self, tmp_path):
        # The filled 'if' modality is row-aligned 1:1 with the cell modality;
        # copying obsm['spatial'] over (like extract_morphology_intensities
        # does) is what lets pp.crop() subset it alongside the cells.
        h = w = 32
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(h, w, 3)).astype(np.uint8)
        spec = _rgb_spec(tmp_path / "if.tif", img)

        origins = [(2.0, 2.0), (20.0, 20.0)]
        xen = ad.AnnData(
            obs=pd.DataFrame(index=["c0", "c1"]),
            obsm={
                "spatial": np.array(origins, dtype=np.float32),
                "cell_boundaries": ak.Array([_square(x, y, 4) for x, y in origins]),
            },
        )
        if_adata = ad.AnnData(
            obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
            var=pd.DataFrame(index=pd.Index(["r", "g", "b"], name="channel_name")),
        )
        if_adata.uns["spatial"] = {"lib_if": {"images": {"full": spec}}}
        mdata = md.MuData({"xenium": xen, "if": if_adata})

        extract_if_intensities(mdata)

        filled = mdata["if"]
        assert filled.X.shape == (2, 3)
        assert list(filled.obs_names) == ["c0", "c1"]
        np.testing.assert_allclose(filled.obsm["spatial"], xen.obsm["spatial"])
