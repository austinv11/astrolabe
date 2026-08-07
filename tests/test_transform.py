"""Tests for rotate/flip multi-library warning (audit fix 1.6), and apply_affine."""
import logging

import numpy as np
import anndata as ad
import mudata as md
import pandas as pd
import pytest

from astrolabe._utils import ImageSpec
from astrolabe.pp._transform import rotate, flip, apply_affine, crop


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


class TestCropSyncsSiblingModalitySpatialBounds:
    def _make_mdata_with_image_sibling(self):
        xen = _adata_with_transcripts()
        xen.uns["spatial"] = {"lib_xen": {"images": {}}}
        he = ad.AnnData(obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)))
        he.uns["spatial"] = {"lib_he": {"images": {}}}
        mdata = md.MuData({"xenium": xen, "he": he})
        mdata.uns["default_modality"] = "xenium"
        return mdata

    def test_not_inplace_does_not_touch_mdata_at_all(self):
        # Regression test: inplace=False must leave *data* (mdata) completely
        # untouched, including sibling modalities -- previously the sibling
        # sync ran unconditionally, so an unrelated ab.pl.image(mdata,
        # modality='he') call after an inplace=False crop would silently
        # start rendering a stale cropped viewport instead of the full image.
        mdata = self._make_mdata_with_image_sibling()

        result = crop(mdata, _REGION, inplace=False)

        assert result.n_obs == 2
        assert "spatial_bounds" not in mdata["he"].uns
        assert "spatial_bounds" not in mdata["xenium"].uns

    def test_not_inplace_on_a_copy_syncs_siblings_without_touching_original(self):
        # The documented pattern for a fully-synced, non-destructive crop:
        # copy the MuData, then crop the copy in place.
        mdata = self._make_mdata_with_image_sibling()
        mdata_crop = mdata.copy()

        result = crop(mdata_crop, _REGION, inplace=True)

        assert result is None
        expected = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}
        assert mdata_crop["he"].uns["spatial_bounds"] == expected
        assert mdata_crop["xenium"].uns["spatial_bounds"] == expected
        # Original mdata (and its 'he' sibling) fully untouched.
        assert "spatial_bounds" not in mdata["he"].uns
        assert mdata["xenium"].n_obs == 4

    def test_inplace_syncs_sibling_bounds_too(self):
        mdata = self._make_mdata_with_image_sibling()

        result = crop(mdata, _REGION, inplace=True)

        assert result is None
        assert mdata["xenium"].n_obs == 2
        expected = {"xmin": 0.0, "xmax": 10.0, "ymin": 0.0, "ymax": 10.0}
        assert mdata["xenium"].uns["spatial_bounds"] == expected
        assert mdata["he"].uns["spatial_bounds"] == expected

    def test_sibling_without_spatial_uns_is_left_untouched(self):
        mdata = self._make_mdata_with_image_sibling()
        del mdata["he"].uns["spatial"]

        crop(mdata, _REGION, inplace=True)

        assert "spatial_bounds" not in mdata["he"].uns

    def test_plain_adata_input_has_no_sibling_propagation_error(self):
        # A bare AnnData (not part of any MuData) has no siblings to sync --
        # must not raise or behave differently from before this feature.
        adata = _adata_with_transcripts()
        result = crop(adata, _REGION, inplace=False)
        assert result.n_obs == 2
