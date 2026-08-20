"""Tests for rotate/flip multi-library warning (audit fix 1.6), and apply_affine."""
import logging

import numpy as np
import anndata as ad
import mudata as md
import pandas as pd
import pytest

from astrolabe._utils import ImageSpec
from astrolabe.pp._transform import rotate, flip, apply_affine, crop, auto_crop


def _adata_with_libraries(n_libs):
    rng = np.random.default_rng(0)
    adata = ad.AnnData(obsm={"spatial": rng.random((5, 2)).astype(np.float32)})
    adata.uns["spatial"] = {f"lib_{i}": {"images": {}} for i in range(n_libs)}
    return adata


class TestRotateMultiLibraryWarning:
    def test_single_library_no_warning(self, caplog):
        adata = _adata_with_libraries(1)
        with caplog.at_level(logging.WARNING):
            rotate(adata, 45.0, inplace=True)
        assert "multiple libraries" not in caplog.text.lower()

    def test_multiple_libraries_emits_warning(self, caplog):
        adata = _adata_with_libraries(2)
        with caplog.at_level(logging.WARNING):
            rotate(adata, 45.0, inplace=True)
        assert "multiple libraries" in caplog.text.lower()

    def test_three_libraries_emits_warning(self, caplog):
        adata = _adata_with_libraries(3)
        with caplog.at_level(logging.WARNING):
            rotate(adata, 90.0, inplace=True)
        assert "multiple libraries" in caplog.text.lower()

    def test_rotation_applied_to_coords(self):
        adata = _adata_with_libraries(1)
        original = adata.obsm["spatial"].copy()
        rotate(adata, 180.0, inplace=True)
        # 180-degree rotation: x → -x+cx*2, y → -y+cy*2 (about centroid)
        # Just verify the coords changed
        assert not np.allclose(adata.obsm["spatial"], original)


class TestFlipMultiLibraryWarning:
    def test_single_library_no_warning(self, caplog):
        adata = _adata_with_libraries(1)
        with caplog.at_level(logging.WARNING):
            flip(adata, "x", inplace=True)
        assert "multiple libraries" not in caplog.text.lower()

    def test_multiple_libraries_emits_warning(self, caplog):
        adata = _adata_with_libraries(2)
        with caplog.at_level(logging.WARNING):
            flip(adata, "y", inplace=True)
        assert "multiple libraries" in caplog.text.lower()


def _make_mdata_for_apply_affine():
    rng = np.random.default_rng(0)
    # Point-cloud modality with its own spatial obsm.
    adata_a = ad.AnnData(obsm={"spatial": rng.random((5, 2)).astype(np.float32)})
    adata_a.uns["spatial"] = {"lib_a": {"images": {}}}

    # Image-only modality (like 'he' before extraction) — no obsm['spatial'].
    spec = ImageSpec(
        path="fake.tif", pixel_size_um=1.0, dimension_order="CYX",
        full_shape=(3, 10, 10), level_shapes=[(3, 10, 10)], level_scales=[(1.0, 1.0)],
    )
    adata_b = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)))
    adata_b.uns["spatial"] = {"lib_b": {"images": {"full": spec}}}

    mdata = md.MuData({"a": adata_a, "b": adata_b})
    return mdata


class TestApplyAffine:
    def test_bad_matrix_shape_raises(self):
        mdata = _make_mdata_for_apply_affine()
        with pytest.raises(ValueError):
            apply_affine(mdata, np.eye(2))

    def test_transforms_spatial_coords_and_image_transform_chain(self):
        mdata = _make_mdata_for_apply_affine()
        original = mdata["a"].obsm["spatial"].copy()
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

        result = apply_affine(mdata, M, inplace=False)

        assert not np.allclose(result["a"].obsm["spatial"], original)
        spec = result["b"].uns["spatial"]["lib_b"]["images"]["full"]
        assert len(spec.transform_chain) == 1
        np.testing.assert_allclose(spec.transform_chain[0], M)
        # original untouched (inplace=False)
        np.testing.assert_allclose(mdata["a"].obsm["spatial"], original)

    def test_inplace_mutates_and_returns_none(self):
        mdata = _make_mdata_for_apply_affine()
        original = mdata["a"].obsm["spatial"].copy()
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

        result = apply_affine(mdata, M, inplace=True)

        assert result is None
        assert not np.allclose(mdata["a"].obsm["spatial"], original)
        spec = mdata["b"].uns["spatial"]["lib_b"]["images"]["full"]
        assert len(spec.transform_chain) == 1


def _adata_with_transcripts():
    import pyarrow as pa

    # 4 cells: two inside the crop region (unit square [0,10]x[0,10]), two outside.
    obsm = {"spatial": np.array([[2.0, 2.0], [5.0, 5.0], [50.0, 50.0], [80.0, 80.0]], dtype=np.float32)}
    adata = ad.AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]), obsm=obsm)
    # 6 transcripts: three inside the region, three outside.
    tx = pa.table(
        {
            "x_location": [1.0, 3.0, 8.0, 40.0, 60.0, 90.0],
            "y_location": [1.0, 4.0, 9.0, 40.0, 60.0, 90.0],
            "feature_name": ["A", "B", "C", "D", "E", "F"],
        }
    )
    adata.uns["transcripts"] = tx
    return adata


_REGION = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


class TestCropTranscriptsSync:
    def test_not_inplace_crops_transcripts_to_same_region_as_cells(self):
        adata = _adata_with_transcripts()

        result = crop(adata, _REGION, inplace=False)

        assert result.n_obs == 2
        tx = result.uns["transcripts"]
        assert tx.num_rows == 3
        assert tx["feature_name"].to_pylist() == ["A", "B", "C"]
        # Original untouched.
        assert adata.uns["transcripts"].num_rows == 6

    def test_inplace_crops_transcripts_to_same_region_as_cells(self):
        adata = _adata_with_transcripts()

        result = crop(adata, _REGION, inplace=True)

        assert result is None
        assert adata.n_obs == 2
        assert adata.uns["transcripts"].num_rows == 3

    def test_crop_transcripts_false_leaves_transcripts_untouched(self):
        adata = _adata_with_transcripts()

        result = crop(adata, _REGION, crop_transcripts=False, inplace=False)

        assert result.n_obs == 2
        assert result.uns["transcripts"].num_rows == 6

    def test_no_transcripts_key_still_works(self):
        obsm = {"spatial": np.array([[2.0, 2.0], [50.0, 50.0]], dtype=np.float32)}
        adata = ad.AnnData(obs=pd.DataFrame(index=["c0", "c1"]), obsm=obsm)

        result = crop(adata, _REGION, inplace=False)

        assert result.n_obs == 1
        assert "transcripts" not in result.uns

    def test_custom_transcript_column_names(self):
        import pyarrow as pa

        obsm = {"spatial": np.array([[2.0, 2.0], [50.0, 50.0]], dtype=np.float32)}
        adata = ad.AnnData(obs=pd.DataFrame(index=["c0", "c1"]), obsm=obsm)
        adata.uns["transcripts"] = pa.table({"px": [1.0, 40.0], "py": [1.0, 40.0]})

        result = crop(
            adata, _REGION, transcripts_x_col="px", transcripts_y_col="py", inplace=False
        )

        assert result.uns["transcripts"].num_rows == 1


_CROP_BOUNDS = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}


def _make_multimodal_mdata():
    # 'xenium': 4 cells (2 inside _REGION) + a transcripts table + uns['spatial'].
    xen = _adata_with_transcripts()
    xen.uns["spatial"] = {"lib_xen": {"images": {}}}
    # 'he': image-only modality -- 0 obs, no obsm['spatial'], only uns['spatial'].
    he = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)))
    he.uns["spatial"] = {"lib_he": {"images": {}}}
    # 'micro': micronuclei-like -- has obsm['spatial'] (1 of 3 inside _REGION)
    # but NO uns['spatial']. The old sibling gate ("spatial" in uns) skipped
    # this modality entirely.
    micro = ad.AnnData(
        obs=pd.DataFrame(index=[f"m{i}" for i in range(3)]),
        obsm={"spatial": np.array([[1.0, 1.0], [50.0, 50.0], [90.0, 90.0]], dtype=np.float32)},
    )
    mdata = md.MuData({"xenium": xen, "he": he, "micro": micro})
    mdata.uns["default_modality"] = "xenium"
    return mdata


class TestCropMuDataTypePreserving:
    def test_not_inplace_returns_cropped_mudata_and_leaves_original_untouched(self):
        mdata = _make_multimodal_mdata()

        result = crop(mdata, _REGION, inplace=False)

        assert isinstance(result, md.MuData)
        assert result["xenium"].n_obs == 2
        assert result["xenium"].uns["spatial_bounds"] == _CROP_BOUNDS
        assert result["he"].uns["spatial_bounds"] == _CROP_BOUNDS
        # Original completely untouched -- rows and viewport hints alike.
        assert mdata["xenium"].n_obs == 4
        assert mdata["micro"].n_obs == 3
        assert "spatial_bounds" not in mdata["xenium"].uns
        assert "spatial_bounds" not in mdata["he"].uns
        assert mdata["xenium"].uns["transcripts"].num_rows == 6

    def test_every_spatial_modality_subset_by_its_own_centroids(self):
        # Each modality is masked by its OWN obsm['spatial'], not by a shared
        # row mask -- and a modality with obsm['spatial'] but no uns['spatial']
        # (the micronuclei shape) is no longer skipped.
        mdata = _make_multimodal_mdata()

        result = crop(mdata, _REGION, inplace=False)

        assert result["micro"].n_obs == 1
        assert list(result["micro"].obs_names) == ["m0"]
        assert result["micro"].uns["spatial_bounds"] == _CROP_BOUNDS

    def test_inplace_subsets_all_modalities_and_updates_global_index(self):
        mdata = _make_multimodal_mdata()

        result = crop(mdata, _REGION, inplace=True)

        assert result is None
        assert mdata["xenium"].n_obs == 2
        assert mdata["micro"].n_obs == 1
        assert mdata["xenium"].uns["spatial_bounds"] == _CROP_BOUNDS
        assert mdata["he"].uns["spatial_bounds"] == _CROP_BOUNDS
        # mdata.update() must have re-synced the global obs index (same
        # contract as pipeline.filter_cells -- see tests/test_pipeline.py).
        assert mdata.n_obs == 3
        assert set(mdata.obs_names) == set(mdata["xenium"].obs_names) | set(
            mdata["micro"].obs_names
        )

    def test_transcripts_cropped_in_mudata_path(self):
        mdata = _make_multimodal_mdata()

        result = crop(mdata, _REGION, inplace=False)

        tx = result["xenium"].uns["transcripts"]
        assert tx.num_rows == 3
        assert tx["feature_name"].to_pylist() == ["A", "B", "C"]

    def test_image_only_modality_gets_bounds_hint_only(self):
        # A 0-obs image modality has nothing to subset; the spatial_bounds
        # stamp alone drives the renderer's pyramid-level and pixel-crop
        # choice, so it must still land.
        mdata = _make_multimodal_mdata()

        result = crop(mdata, _REGION, inplace=False)

        assert result["he"].n_obs == 0
        assert result["he"].uns["spatial_bounds"] == _CROP_BOUNDS

    def test_modality_without_spatial_info_left_unsubset_with_warning(self, caplog):
        mdata = _make_multimodal_mdata()
        tab = ad.AnnData(obs=pd.DataFrame(index=[f"t{i}" for i in range(3)]))
        mdata.mod["tab"] = tab
        mdata.update()

        with caplog.at_level(logging.WARNING):
            crop(mdata, _REGION, inplace=True)

        assert "left unsubset" in caplog.text
        assert mdata["tab"].n_obs == 3
        assert "spatial_bounds" not in mdata["tab"].uns

    def test_no_spatial_modality_anywhere_raises(self):
        he = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)))
        he.uns["spatial"] = {"lib_he": {"images": {}}}
        mdata = md.MuData({"he": he})
        with pytest.raises(KeyError):
            crop(mdata, _REGION)

    def test_modality_kwarg_removed(self):
        mdata = _make_multimodal_mdata()
        with pytest.raises(TypeError):
            crop(mdata, _REGION, modality="xenium")

    def test_plain_adata_input_returns_adata(self):
        # AnnData in -> AnnData out, exactly as before.
        adata = _adata_with_transcripts()
        result = crop(adata, _REGION, inplace=False)
        assert isinstance(result, ad.AnnData)
        assert result.n_obs == 2


def _adata_with_outlier(rng=None):
    # 99 cells tightly clustered in [0, 1] x [0, 1], plus one cell far away.
    # Mirrors the real bug this feature fixes: notebooks/04_if_and_he_images.ipynb
    # (xenium_tiny) had 357/358 cells in a ~245x135 um patch and one lone cell
    # ~580 um away, which blew a min/max-based crop region out to ~80% empty space.
    rng = rng or np.random.default_rng(0)
    cluster = rng.uniform(0.0, 1.0, size=(99, 2)).astype(np.float32)
    outlier = np.array([[1000.0, 1000.0]], dtype=np.float32)
    obsm = {"spatial": np.vstack([cluster, outlier])}
    adata = ad.AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(100)]), obsm=obsm)
    return adata, cluster


class TestAutoCrop:
    def test_default_bounds_match_percentile_formula(self):
        adata, _ = _adata_with_outlier()
        spatial = adata.obsm["spatial"].astype(np.float64)

        result = auto_crop(adata, inplace=False)

        x_lo, x_hi = np.percentile(spatial[:, 0], [1.0, 99.0])
        y_lo, y_hi = np.percentile(spatial[:, 1], [1.0, 99.0])
        w, h = (x_hi - x_lo) * 1.15, (y_hi - y_lo) * 1.15
        expected = {
            "xmin": (x_lo + x_hi) / 2.0 - w / 2.0,
            "xmax": (x_lo + x_hi) / 2.0 + w / 2.0,
            "ymin": (y_lo + y_hi) / 2.0 - h / 2.0,
            "ymax": (y_lo + y_hi) / 2.0 + h / 2.0,
        }
        bounds = result.uns["spatial_bounds"]
        for k in expected:
            assert bounds[k] == pytest.approx(expected[k])

    def test_outlier_excluded_from_bounds_by_default(self):
        # The whole point of the feature: the lone outlier at (1000, 1000)
        # must not set the crop frame the way it would with plain min/max.
        adata, cluster = _adata_with_outlier()

        result = auto_crop(adata, inplace=False)

        bounds = result.uns["spatial_bounds"]
        assert bounds["xmax"] < 20.0
        assert bounds["ymax"] < 20.0
        # The outlier cell itself must have been cropped out.
        assert result.n_obs == 99
        assert np.allclose(np.sort(result.obsm["spatial"], axis=0), np.sort(cluster, axis=0))

    def test_percentile_zero_uses_true_min_max_no_trimming(self):
        adata, _ = _adata_with_outlier()
        spatial = adata.obsm["spatial"].astype(np.float64)

        result = auto_crop(adata, percentile=0.0, inplace=False)

        x_lo, x_hi = spatial[:, 0].min(), spatial[:, 0].max()
        y_lo, y_hi = spatial[:, 1].min(), spatial[:, 1].max()
        bounds = result.uns["spatial_bounds"]
        assert bounds["xmax"] == pytest.approx(x_hi + (x_hi - x_lo) * 0.15 / 2.0)
        assert bounds["ymax"] == pytest.approx(y_hi + (y_hi - y_lo) * 0.15 / 2.0)
        # No trimming means every cell, including the outlier, is kept.
        assert result.n_obs == 100

    def test_padding_widens_bounds_proportionally(self):
        adata, _ = _adata_with_outlier()

        tight = auto_crop(adata, padding=0.0, inplace=False)
        padded = auto_crop(adata, padding=0.5, inplace=False)

        tb, pb = tight.uns["spatial_bounds"], padded.uns["spatial_bounds"]
        assert (pb["xmax"] - pb["xmin"]) == pytest.approx((tb["xmax"] - tb["xmin"]) * 1.5)
        assert (pb["ymax"] - pb["ymin"]) == pytest.approx((tb["ymax"] - tb["ymin"]) * 1.5)

    def test_inplace_returns_none_and_mutates(self):
        adata, _ = _adata_with_outlier()

        result = auto_crop(adata, inplace=True)

        assert result is None
        assert adata.n_obs == 99
        assert "spatial_bounds" in adata.uns

    def test_missing_spatial_key_raises(self):
        adata = ad.AnnData(obs=pd.DataFrame(index=["c0"]))
        with pytest.raises(KeyError):
            auto_crop(adata)

    def test_empty_adata_raises(self):
        obsm = {"spatial": np.empty((0, 2), dtype=np.float32)}
        adata = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], dtype=str)), obsm=obsm)
        with pytest.raises(ValueError):
            auto_crop(adata)

    def test_degenerate_span_falls_back_to_fixed_window(self):
        # Every cell at the exact same coordinate -- zero-width/height span
        # must not silently crop away every cell.
        obsm = {"spatial": np.full((10, 2), 5.0, dtype=np.float32)}
        adata = ad.AnnData(obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]), obsm=obsm)

        result = auto_crop(adata, inplace=False)

        assert result.n_obs == 10

    def _make_mdata_with_outlier(self):
        xen, _ = _adata_with_outlier()
        xen.uns["spatial"] = {"lib_xen": {"images": {}}}
        he = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)))
        he.uns["spatial"] = {"lib_he": {"images": {}}}
        mdata = md.MuData({"xenium": xen, "he": he})
        mdata.uns["default_modality"] = "xenium"
        return mdata

    def test_mudata_inplace_crops_all_modalities(self):
        # auto_crop must delegate to crop() faithfully: modality= picks whose
        # centroids define the box, then every spatial modality is cropped
        # and every modality gets the spatial_bounds stamp.
        mdata = self._make_mdata_with_outlier()

        result = auto_crop(mdata, modality="xenium", inplace=True)

        assert result is None
        assert mdata["xenium"].n_obs == 99
        assert "spatial_bounds" in mdata["he"].uns
        assert mdata["he"].uns["spatial_bounds"] == mdata["xenium"].uns["spatial_bounds"]
        assert mdata.n_obs == mdata["xenium"].n_obs

    def test_mudata_not_inplace_returns_cropped_mudata(self):
        mdata = self._make_mdata_with_outlier()

        result = auto_crop(mdata, inplace=False)

        assert isinstance(result, md.MuData)
        assert result["xenium"].n_obs == 99
        assert result["he"].uns["spatial_bounds"] == result["xenium"].uns["spatial_bounds"]
        # Original untouched.
        assert mdata["xenium"].n_obs == 100
        assert "spatial_bounds" not in mdata["he"].uns


class TestRotateFlipSyncSpatialBounds:
    # Regression: rotate()/flip() used to leave a pre-existing
    # uns['spatial_bounds'] (set by an earlier pp.crop()) describing the
    # *pre-transform* window, so a rotated/flipped, previously-cropped panel
    # rendered auto-zoomed to the wrong region.

    def _adata_with_bounds(self):
        rng = np.random.default_rng(0)
        adata = ad.AnnData(obsm={"spatial": rng.uniform(0, 10, (5, 2)).astype(np.float32)})
        adata.uns["spatial_bounds"] = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}
        return adata

    def test_rotate_transforms_existing_bounds(self):
        adata = self._adata_with_bounds()
        rotate(adata, 90.0, center=(0.0, 0.0), inplace=True)
        # 90 deg CCW about origin: (x, y) -> (-y, x). Corners of [0,10]x[0,10]
        # map to (0,0),(0,10),(-10,10),(-10,0) -> new box [-10,0] x [0,10].
        bounds = adata.uns["spatial_bounds"]
        assert bounds["xmin"] == pytest.approx(-10.0)
        assert bounds["xmax"] == pytest.approx(0.0)
        assert bounds["ymin"] == pytest.approx(0.0)
        assert bounds["ymax"] == pytest.approx(10.0)

    def test_rotate_not_inplace_transforms_copy_bounds_only(self):
        adata = self._adata_with_bounds()
        original = dict(adata.uns["spatial_bounds"])
        result = rotate(adata, 90.0, center=(0.0, 0.0), inplace=False)
        assert adata.uns["spatial_bounds"] == original
        assert result.uns["spatial_bounds"]["xmin"] == pytest.approx(-10.0)

    def test_rotate_no_existing_bounds_stays_absent(self):
        rng = np.random.default_rng(0)
        adata = ad.AnnData(obsm={"spatial": rng.uniform(0, 10, (5, 2)).astype(np.float32)})
        rotate(adata, 45.0, inplace=True)
        assert "spatial_bounds" not in adata.uns

    def test_flip_transforms_existing_bounds(self):
        adata = self._adata_with_bounds()
        flip(adata, "x", center=(0.0, 0.0), inplace=True)
        # axis='x' about origin: (x, y) -> (-x, y). [0,10]x[0,10] -> [-10,0]x[0,10].
        bounds = adata.uns["spatial_bounds"]
        assert bounds["xmin"] == pytest.approx(-10.0)
        assert bounds["xmax"] == pytest.approx(0.0)
        assert bounds["ymin"] == pytest.approx(0.0)
        assert bounds["ymax"] == pytest.approx(10.0)


class TestApplyAffineSyncsSpatialBounds:
    def test_translates_existing_bounds_per_modality(self):
        mdata = _make_mdata_for_apply_affine()
        mdata["a"].uns["spatial_bounds"] = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}
        mdata["b"].uns["spatial_bounds"] = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])

        apply_affine(mdata, M, inplace=True)

        expected = {"xmin": 5.0, "xmax": 15.0, "ymin": 3.0, "ymax": 13.0}
        assert mdata["a"].uns["spatial_bounds"] == pytest.approx(expected)
        assert mdata["b"].uns["spatial_bounds"] == pytest.approx(expected)

    def test_modality_without_bounds_stays_absent(self):
        mdata = _make_mdata_for_apply_affine()
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
        apply_affine(mdata, M, inplace=True)
        assert "spatial_bounds" not in mdata["a"].uns
        assert "spatial_bounds" not in mdata["b"].uns
