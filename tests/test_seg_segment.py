"""Tests for astrolabe.seg._segment — multi-channel image loading for StarDist."""
import numpy as np
import pytest
import tifffile

from astrolabe._utils import ImageSpec
from astrolabe.seg._segment import _to_hwc, _load_all_channels, _masks_to_adata, _um_to_px


def _plain_spec(shape_hw):
    return ImageSpec(
        path="unused.tif",
        pixel_size_um=1.0,
        dimension_order="YX",
        full_shape=tuple(shape_hw),
        level_shapes=[tuple(shape_hw)],
        level_scales=[(1.0, 1.0)],
    )


class TestMasksToAdata:
    def test_border_instance_contour_closed_and_flagged(self):
        # A label flush against the image border: the contour must follow
        # the border (padded sub-mask -> closed contour, full area), not be
        # cut by a straight chord; and on_image_edge must flag it.
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[0:6, 0:6] = 1     # touches top-left border
        masks[10:14, 10:14] = 2  # interior
        adata = _masks_to_adata(masks, _plain_spec(masks.shape))

        assert adata.n_obs == 2
        assert adata.obs["on_image_edge"].tolist() == [True, False]
        # Border cell's polygon must enclose (approximately) its full 6x6
        # pixel area -- an open contour closed by a chord loses ~half of it.
        border_area = float(adata.obs["cell_area"].iloc[0])
        interior_area = float(adata.obs["cell_area"].iloc[1])
        assert border_area > 0.8 * 36.0
        assert interior_area > 0.8 * 16.0

    def test_min_area_filter_drops_speckles(self):
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[5:13, 5:13] = 1  # 8x8 cell
        masks[16, 16] = 2      # single-pixel speckle
        adata = _masks_to_adata(masks, _plain_spec(masks.shape), min_area_um2=4.0)
        assert adata.n_obs == 1
        assert list(adata.obs_names) == ["1"]

    def test_nonconsecutive_labels_handled(self):
        masks = np.zeros((20, 20), dtype=np.int32)
        masks[2:6, 2:6] = 3   # labels 1 and 2 absent
        masks[10:14, 10:14] = 7
        adata = _masks_to_adata(masks, _plain_spec(masks.shape))
        assert adata.n_obs == 2
        assert list(adata.obs_names) == ["3", "7"]
        assert all(len(np.asarray(p)) >= 3 for p in adata.obsm["cell_boundaries"])


class TestUmToPxFourCorners:
    def test_rotated_affine_bbox_covers_full_region(self):
        # 90-degree rotation: px->um affine maps (col, row) -> (-row, col).
        # A 2-corner mapping of the um bbox [(0,0), (10,20)] through the
        # inverse would span only a diagonal pair and miss the true pixel
        # extent; all four corners must define the bbox.
        affine = np.array(
            [[0.0, -1.0, 0.0],
             [1.0,  0.0, 0.0],
             [0.0,  0.0, 1.0]],
            dtype=np.float64,
        )
        spec = ImageSpec(
            path="unused.tif",
            pixel_size_um=1.0,
            dimension_order="YX",
            full_shape=(100, 100),
            level_shapes=[(100, 100)],
            level_scales=[(1.0, 1.0)],
            registration_affine=affine,
        )
        # um bbox x in [-20, 0], y in [0, 10] maps back to cols 0..10, rows 0..20.
        row_min, row_max, col_min, col_max = _um_to_px(-20.0, 0.0, 0.0, 10.0, spec)
        assert row_min == 0 and row_max == 20
        assert col_min == 0 and col_max == 10


class TestToHwc:
    def test_yxc_passthrough(self):
        img = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)  # (Y=2,X=3,C=4)
        out = _to_hwc(img, "YXC")
        assert out.shape == (2, 3, 4)
        np.testing.assert_array_equal(out, img)

    def test_cyx_transposed_to_hwc(self):
        img = np.arange(4 * 2 * 3).reshape(4, 2, 3).astype(np.float32)  # (C=4,Y=2,X=3)
        out = _to_hwc(img, "CYX")
        assert out.shape == (2, 3, 4)
        np.testing.assert_array_equal(out, np.transpose(img, (1, 2, 0)))

    def test_no_channel_axis_adds_singleton(self):
        img = np.arange(2 * 3).reshape(2, 3).astype(np.float32)  # (Y=2,X=3)
        out = _to_hwc(img, "YX")
        assert out.shape == (2, 3, 1)
        np.testing.assert_array_equal(out[:, :, 0], img)

    def test_t_axis_collapsed(self):
        img = np.zeros((1, 2, 3, 4), dtype=np.float32)  # (T=1,Y=2,X=3,C=4)
        img[0] = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        out = _to_hwc(img, "TYXC")
        assert out.shape == (2, 3, 4)
        np.testing.assert_array_equal(out, img[0])


class TestLoadAllChannels:
    def _write_rgb_tiff(self, path, arr_yxc):
        # tifffile writes (Y,X,C) as photometric=rgb by default when the
        # last axis has size 3 -- axes come back as 'YXS' via tifffile,
        # matching real H&E TIFFs (the 'S'->'C' remap astrolabe's own H&E
        # reader applies).
        tifffile.imwrite(path, arr_yxc, photometric="rgb")

    def _spec(self, path, dim_order):
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        return ImageSpec(
            path=str(path),
            pixel_size_um=0.5,
            dimension_order=dim_order,
            full_shape=tuple(shape),
            level_shapes=[tuple(shape)],
            level_scales=[(1.0, 1.0)],
        )

    def test_full_load_rgb(self, tmp_path):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(20, 30, 3)).astype(np.uint8)
        path = tmp_path / "he.tif"
        self._write_rgb_tiff(path, arr)
        spec = self._spec(path, "YXC")  # 'S' already remapped to 'C' by our H&E reader convention

        out = _load_all_channels(spec)
        assert out.shape == (20, 30, 3)
        np.testing.assert_allclose(out, arr.astype(np.float32))

    def test_region_crop_rgb(self, tmp_path):
        rng = np.random.default_rng(1)
        arr = rng.integers(0, 255, size=(40, 50, 3)).astype(np.uint8)
        path = tmp_path / "he.tif"
        self._write_rgb_tiff(path, arr)
        spec = self._spec(path, "YXC")

        out = _load_all_channels(spec, region_px=(5, 15, 10, 25))
        assert out.shape == (10, 15, 3)
        np.testing.assert_allclose(out, arr[5:15, 10:25, :].astype(np.float32))


class TestSegmentStardistMultiChannel:
    """End-to-end regression for the 'axes (YXC) must be of length 2' crash.

    Reproduces the real bug: segment_stardist(..., model_name='2D_versatile_he')
    used to always load a single channel (2-D array) regardless of the
    model's channel requirement, which crashed inside StarDist as soon as a
    multi-channel model (n_channel_in > 1) was used. Builds a small
    *untrained* multi-channel StarDist2D model locally (no network / no
    pretrained-weight download) and monkeypatches `from_pretrained` to
    return it, so this stays fast and offline while still exercising the
    real model.predict_instances() call path.
    """

    def test_multichannel_model_runs_without_crashing(self, tmp_path, monkeypatch):
        stardist_models = pytest.importorskip("stardist.models")

        rng = np.random.default_rng(2)
        arr = rng.integers(0, 255, size=(32, 32, 3)).astype(np.uint8)
        path = tmp_path / "he.tif"
        tifffile.imwrite(path, arr, photometric="rgb")
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        spec = ImageSpec(
            path=str(path),
            pixel_size_um=0.5,
            dimension_order="YXC",
            full_shape=tuple(shape),
            level_shapes=[tuple(shape)],
            level_scales=[(1.0, 1.0)],
        )

        cfg = stardist_models.Config2D(
            n_channel_in=3, n_rays=8, grid=(2, 2),
            backbone="unet", unet_n_depth=1, unet_n_filter_base=8,
        )
        fake_model = stardist_models.StarDist2D(cfg, name=None, basedir=None)
        monkeypatch.setattr(
            stardist_models.StarDist2D, "from_pretrained",
            classmethod(lambda cls, name: fake_model),
        )

        import astrolabe as ab
        # Regression: a low prob_thresh forces real (>0) detections through
        # this untrained model -- the earlier version of this test used
        # default thresholds and happened to always land in the n_found==0
        # branch, silently never exercising the polygon-extraction code path
        # where the actual bug (details['coord'] shape mismatch, see below)
        # lived. n_found>0 here is what makes this test meaningful.
        result = ab.seg.segment_stardist(
            spec, model_name="2D_versatile_he", prob_thresh=0.0001, nms_thresh=0.9,
        )
        assert result.n_obs > 0
        assert "spatial" in result.obsm
        assert result.obsm["spatial"].shape == (result.n_obs, 2)
        assert "cell_boundaries" in result.obsm
        assert len(result.obsm["cell_boundaries"]) == result.n_obs
        # Regression: details['coord'] is (n_cells, 2, n_rays), not
        # (n_cells, n_rays, 2) -- transposing it wrong produced garbage
        # (n_rays, 2)-shaped "vertices" downstream and crashed inside
        # _transform_coords. Each boundary must be a real (n_verts, 2) array.
        for poly in result.obsm["cell_boundaries"]:
            poly = np.asarray(poly)
            assert poly.ndim == 2 and poly.shape[1] == 2

    def test_single_channel_model_unaffected(self, tmp_path, monkeypatch):
        # Regression guard: the pre-existing single-channel (fluorescence)
        # path -- load exactly the requested `channel` as a 2-D array --
        # must be untouched by the multi-channel branch added above.
        stardist_models = pytest.importorskip("stardist.models")

        rng = np.random.default_rng(3)
        arr = rng.integers(0, 4000, size=(32, 32)).astype(np.uint16)
        path = tmp_path / "dapi.tif"
        tifffile.imwrite(path, arr)
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        spec = ImageSpec(
            path=str(path),
            pixel_size_um=0.5,
            dimension_order="YX",
            full_shape=tuple(shape),
            level_shapes=[tuple(shape)],
            level_scales=[(1.0, 1.0)],
        )

        cfg = stardist_models.Config2D(
            n_channel_in=1, n_rays=8, grid=(2, 2),
            backbone="unet", unet_n_depth=1, unet_n_filter_base=8,
        )
        fake_model = stardist_models.StarDist2D(cfg, name=None, basedir=None)
        monkeypatch.setattr(
            stardist_models.StarDist2D, "from_pretrained",
            classmethod(lambda cls, name: fake_model),
        )

        import astrolabe as ab
        result = ab.seg.segment_stardist(
            spec, model_name="2D_versatile_fluo", prob_thresh=0.0001, nms_thresh=0.9,
        )
        assert result.n_obs > 0
        assert "spatial" in result.obsm
        assert result.obsm["spatial"].shape == (result.n_obs, 2)
        for poly in result.obsm["cell_boundaries"]:
            poly = np.asarray(poly)
            assert poly.ndim == 2 and poly.shape[1] == 2

    def test_zero_cells_found_does_not_crash(self, tmp_path, monkeypatch):
        # Regression: found independently of the multi-channel fix above --
        # when StarDist finds zero cells (e.g. an empty/background-only
        # crop), `np.stack([])` over an empty polygon list raised
        # "ValueError: need at least one array to stack". The cellpose path
        # already guards this (_masks_to_adata); the stardist path didn't.
        # prob_thresh=0.999999 deterministically yields zero detections
        # regardless of the (randomly initialized, untrained) model's output.
        stardist_models = pytest.importorskip("stardist.models")

        rng = np.random.default_rng(4)
        arr = rng.integers(0, 4000, size=(32, 32)).astype(np.uint16)
        path = tmp_path / "dapi.tif"
        tifffile.imwrite(path, arr)
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        spec = ImageSpec(
            path=str(path),
            pixel_size_um=0.5,
            dimension_order="YX",
            full_shape=tuple(shape),
            level_shapes=[tuple(shape)],
            level_scales=[(1.0, 1.0)],
        )

        cfg = stardist_models.Config2D(
            n_channel_in=1, n_rays=8, grid=(2, 2),
            backbone="unet", unet_n_depth=1, unet_n_filter_base=8,
        )
        fake_model = stardist_models.StarDist2D(cfg, name=None, basedir=None)
        monkeypatch.setattr(
            stardist_models.StarDist2D, "from_pretrained",
            classmethod(lambda cls, name: fake_model),
        )

        import astrolabe as ab
        result = ab.seg.segment_stardist(
            spec, model_name="2D_versatile_fluo", prob_thresh=0.999999,
        )
        assert result.n_obs == 0
        assert result.obsm["spatial"].shape == (0, 2)
        assert len(result.obsm["cell_boundaries"]) == 0
