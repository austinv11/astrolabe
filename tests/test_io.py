import json
import logging
from pathlib import Path

import anndata as ad
import h5py
import mudata as md
import numpy as np
import pytest
import tifffile

import astrolabe as ab
from astrolabe._utils import ImageSpec
from astrolabe.io._xenium import _attach_he_image


def _make_spec(image_path: Path) -> ImageSpec:
    return ImageSpec(
        path=str(image_path.resolve()),
        pixel_size_um=0.2125,
        dimension_order="CYX",
        full_shape=(1, 100, 100),
        level_shapes=[(1, 100, 100)],
        level_scales=[(1.0, 1.0)],
    )


@pytest.fixture
def mdata_with_image(small_mdata, tmp_path):
    image_path = tmp_path / "images" / "morph.tif"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake tiff data")
    small_mdata["xenium"].uns["spatial"] = {"sample": {"images": {"full": _make_spec(image_path)}}}
    return small_mdata, image_path


def test_save_stores_relative_path(mdata_with_image, tmp_path):
    mdata, image_path = mdata_with_image
    save_path = tmp_path / "out" / "analysis"
    ab.io.save(mdata, save_path)

    with h5py.File(str(save_path) + ".h5mu", "r") as f:
        stored = f["mod"]["xenium"]["uns"]["spatial"]["sample"]["images"]["full"].attrs["path"]
    assert not Path(stored).is_absolute()


def test_save_warns_about_relative_paths(mdata_with_image, tmp_path, caplog):
    mdata, image_path = mdata_with_image
    save_path = tmp_path / "out" / "analysis"
    with caplog.at_level(logging.WARNING, logger="astrolabe.io._io"):
        ab.io.save(mdata, save_path)
    assert any("relative" in rec.message for rec in caplog.records)


def test_round_trip_resolves_absolute_path(mdata_with_image, tmp_path):
    mdata, image_path = mdata_with_image
    save_path = tmp_path / "out" / "analysis"
    ab.io.save(mdata, save_path)

    loaded = ab.io.load(str(save_path) + ".h5mu")
    spec = loaded["xenium"].uns["spatial"]["sample"]["images"]["full"]
    assert Path(spec.path).resolve() == image_path.resolve()
    assert Path(spec.path).exists()


def test_load_warns_on_missing_image_and_path_remap_fixes_it(mdata_with_image, tmp_path, caplog):
    mdata, image_path = mdata_with_image
    save_path = tmp_path / "out" / "analysis"
    ab.io.save(mdata, save_path)

    new_image_path = tmp_path / "moved" / "morph.tif"
    new_image_path.parent.mkdir(parents=True)
    image_path.rename(new_image_path)

    with caplog.at_level(logging.WARNING, logger="astrolabe.io._io"):
        loaded = ab.io.load(str(save_path) + ".h5mu")
    spec = loaded["xenium"].uns["spatial"]["sample"]["images"]["full"]
    assert not Path(spec.path).exists()
    assert any("path_remap" in rec.message for rec in caplog.records)

    remapped = ab.io.load(str(save_path) + ".h5mu", path_remap={r"images": "moved"})
    remapped_spec = remapped["xenium"].uns["spatial"]["sample"]["images"]["full"]
    assert Path(remapped_spec.path).resolve() == new_image_path.resolve()


# ---------------------------------------------------------------------------
# attach_modality
# ---------------------------------------------------------------------------


class TestAttachModality:
    def test_preserves_existing_default(self, small_mdata, small_adata):
        ab.io.attach_modality(small_mdata, small_adata.copy(), "extra")
        assert small_mdata.uns["default_modality"] == "xenium"
        assert "extra" in small_mdata.mod

    def test_sets_default_when_unset(self, small_adata):
        mdata = md.MuData({"a": small_adata.copy()})
        mdata.uns.pop("default_modality", None)
        ab.io.attach_modality(mdata, small_adata.copy(), "b")
        assert mdata.uns["default_modality"] == "b"

    def test_set_default_true_overrides(self, small_mdata, small_adata):
        ab.io.attach_modality(small_mdata, small_adata.copy(), "extra", set_default=True)
        assert small_mdata.uns["default_modality"] == "extra"

    def test_does_not_leak_into_other_modalitys_own_uns(self, small_mdata, small_adata):
        # Regression test: read_xenium() does `mdata.uns['spatial'] = dict(adata.uns['spatial'])`
        # (a shallow copy) precisely so that later attaching a second modality's library
        # entry into mdata.uns['spatial'] doesn't also leak into the first modality's own
        # uns['spatial'] — which it would if the two were the same dict object.
        xen = small_mdata["xenium"]
        xen.uns["spatial"] = {"lib_xenium": {"images": {}}}
        small_mdata.uns["spatial"] = dict(xen.uns["spatial"])

        extra = small_adata.copy()
        extra.uns["spatial"] = {"lib_he": {"images": {}}}
        ab.io.attach_modality(small_mdata, extra, "he")

        assert "lib_he" in small_mdata.uns["spatial"]
        assert "lib_he" not in xen.uns["spatial"]

    def test_merges_spatial_uns(self, small_mdata, small_adata):
        extra = small_adata.copy()
        spec = ImageSpec(
            path="fake.tif", pixel_size_um=1.0, dimension_order="CYX",
            full_shape=(3, 10, 10), level_shapes=[(3, 10, 10)], level_scales=[(1.0, 1.0)],
        )
        extra.uns["spatial"] = {"extra_lib": {"images": {"full": spec}}}
        ab.io.attach_modality(small_mdata, extra, "extra")
        assert "extra_lib" in small_mdata.uns["spatial"]


# ---------------------------------------------------------------------------
# H&E loading (read_xenium he_image / he_alignment)
# ---------------------------------------------------------------------------


def _write_fake_he_tiff(path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, photometric="rgb")


def _he_spec(mdata):
    lib = next(iter(mdata["he"].uns["spatial"]))
    return mdata["he"].uns["spatial"][lib]["images"]["full"]


class TestAttachHeImage:
    def test_default_filenames_with_alignment(self, tmp_path, small_mdata):
        _write_fake_he_tiff(tmp_path / "he_image.ome.tif")
        # 10x's imagealignment.csv maps the H&E image's own raw pixel
        # coordinates to the morphology image's raw pixel grid (verified
        # against a real download: magnitude of the linear part matches
        # he_pixel_size_um / morph_pixel_size_um exactly). registration_affine
        # must map H&E pixels straight to microns, so _attach_he_image scales
        # the raw matrix by the run's pixel_size_um rather than inverting it.
        M = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
        np.savetxt(tmp_path / "he_imagealignment.csv", M, delimiter=",")

        _attach_he_image(small_mdata, tmp_path, True, True, pixel_size_um=0.2125)

        assert "he" in small_mdata.mod
        assert small_mdata.uns["default_modality"] == "xenium"
        spec = _he_spec(small_mdata)
        assert spec.registration_affine is not None
        scale = np.array([[0.2125, 0.0, 0.0], [0.0, 0.2125, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(spec.registration_affine, scale @ M)

    def test_override_filename_no_alignment(self, tmp_path, small_mdata):
        _write_fake_he_tiff(tmp_path / "custom_he.tif")

        _attach_he_image(small_mdata, tmp_path, "custom_he.tif", False, pixel_size_um=0.2125)

        assert "he" in small_mdata.mod
        assert _he_spec(small_mdata).registration_affine is None

    def test_missing_alignment_attaches_unregistered(self, tmp_path, small_mdata, caplog):
        _write_fake_he_tiff(tmp_path / "he_image.ome.tif")

        with caplog.at_level(logging.WARNING):
            _attach_he_image(small_mdata, tmp_path, True, True, pixel_size_um=0.2125)

        assert "he" in small_mdata.mod
        assert _he_spec(small_mdata).registration_affine is None
        assert any("not found" in rec.message for rec in caplog.records)

    def test_missing_image_skips_gracefully(self, tmp_path, small_mdata, caplog):
        with caplog.at_level(logging.WARNING):
            _attach_he_image(small_mdata, tmp_path, True, True, pixel_size_um=0.2125)

        assert "he" not in small_mdata.mod

    def test_explicit_override_missing_raises(self, tmp_path, small_mdata):
        with pytest.raises(FileNotFoundError):
            _attach_he_image(small_mdata, tmp_path, "nonexistent.tif", False, pixel_size_um=0.2125)

    def test_he_alignment_false_skips_lookup_even_if_present(self, tmp_path, small_mdata):
        _write_fake_he_tiff(tmp_path / "he_image.ome.tif")
        M = np.eye(3)
        np.savetxt(tmp_path / "he_imagealignment.csv", M, delimiter=",")

        _attach_he_image(small_mdata, tmp_path, True, False, pixel_size_um=0.2125)

        assert _he_spec(small_mdata).registration_affine is None


# ---------------------------------------------------------------------------
# read_annotations
# ---------------------------------------------------------------------------


def _write_fake_geojson(path: Path) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "region1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
            }
        ],
    }
    path.write_text(json.dumps(geojson))


class TestReadAnnotations:
    def test_returns_geodataframe(self, tmp_path):
        path = tmp_path / "annotation.geojson"
        _write_fake_geojson(path)

        gdf = ab.io.read_annotations(path)

        assert len(gdf) == 1
        assert gdf.iloc[0]["name"] == "region1"

    def test_attaches_to_mdata(self, tmp_path, small_mdata):
        path = tmp_path / "annotation.geojson"
        _write_fake_geojson(path)

        result = ab.io.read_annotations(path, mdata=small_mdata)

        assert small_mdata.uns["annotations"] is result

    def test_attaches_to_adata(self, tmp_path, small_adata):
        path = tmp_path / "annotation.geojson"
        _write_fake_geojson(path)

        result = ab.io.read_annotations(path, mdata=small_adata)

        assert small_adata.uns["annotations"] is result

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ab.io.read_annotations(tmp_path / "nope.geojson")


# ---------------------------------------------------------------------------
# align_xenium_annotations
# ---------------------------------------------------------------------------


class TestAlignXeniumAnnotations:
    def _attach_he_with_affine(self, mdata, small_adata, M):
        he = small_adata.copy()
        spec = ImageSpec(
            path="fake_he.tif", pixel_size_um=0.27, dimension_order="YXC",
            full_shape=(200, 150, 3), level_shapes=[(200, 150, 3)], level_scales=[(1.0, 1.0)],
            registration_affine=M,
        )
        he.uns["spatial"] = {"he_lib": {"images": {"full": spec}}}
        ab.io.attach_modality(mdata, he, "he")

    def test_transform_uses_he_registration_affine(self, small_mdata, small_adata):
        import geopandas as gpd
        from shapely.geometry import Point

        # Annotation coords are in H&E pixel space; align_xenium_annotations
        # must apply the H&E modality's own registration_affine directly
        # (verified against the real public dataset -- see module docstring).
        M = np.array([[0.2125, 0.0, 1.0], [0.0, 0.2125, 2.0], [0.0, 0.0, 1.0]])
        self._attach_he_with_affine(small_mdata, small_adata, M)

        gdf = gpd.GeoDataFrame(geometry=[Point(10, 20)])
        aligned = ab.io.align_xenium_annotations(gdf, small_mdata)

        pt = aligned.geometry.iloc[0]
        assert pt.x == pytest.approx(10 * 0.2125 + 1.0)
        assert pt.y == pytest.approx(20 * 0.2125 + 2.0)

    def test_missing_he_modality_raises(self, small_mdata):
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(geometry=[Point(1, 2)])

        with pytest.raises(KeyError):
            ab.io.align_xenium_annotations(gdf, small_mdata)

    def test_unregistered_he_image_raises(self, small_mdata, small_adata):
        import geopandas as gpd
        from shapely.geometry import Point

        self._attach_he_with_affine(small_mdata, small_adata, None)
        gdf = gpd.GeoDataFrame(geometry=[Point(1, 2)])

        with pytest.raises(ValueError, match="registration_affine"):
            ab.io.align_xenium_annotations(gdf, small_mdata)


class TestSaveLoadGeoDataFrame:
    def test_annotations_round_trip(self, small_mdata, tmp_path):
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {"name": ["r1", "r2"]},
            geometry=[
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            ],
        )
        small_mdata.uns["annotations"] = gdf

        save_path = tmp_path / "out" / "analysis"
        ab.io.save(small_mdata, save_path)
        loaded = ab.io.load(str(save_path) + ".h5mu")

        loaded_gdf = loaded.uns["annotations"]
        assert isinstance(loaded_gdf, gpd.GeoDataFrame)
        assert loaded_gdf["name"].tolist() == ["r1", "r2"]
        assert loaded_gdf.geometry.iloc[0].equals(gdf.geometry.iloc[0])
        assert loaded_gdf.geometry.iloc[1].equals(gdf.geometry.iloc[1])

    def test_nested_dict_property_round_trip(self, small_mdata, tmp_path):
        # Regression test: real Xenium annotation.geojson files carry a
        # 'classification' property that is a nested dict, which a plain
        # per-column anndata DataFrame writer can't serialize.
        import geopandas as gpd
        from shapely.geometry import Polygon

        gdf = gpd.GeoDataFrame(
            {
                "name": ["r1"],
                "classification": [{"name": "Tumor", "color": [255, 0, 0]}],
            },
            geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        )
        small_mdata.uns["annotations"] = gdf

        save_path = tmp_path / "out" / "analysis"
        ab.io.save(small_mdata, save_path)
        loaded = ab.io.load(str(save_path) + ".h5mu")

        loaded_gdf = loaded.uns["annotations"]
        assert loaded_gdf["classification"].iloc[0] == {"name": "Tumor", "color": [255, 0, 0]}


# ---------------------------------------------------------------------------
# read_xenium_transcripts
# ---------------------------------------------------------------------------


class TestReadXeniumTranscripts:
    def _mdata_pointing_at(self, tmp_path):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "feature_name": ["EPCAM", "CD3D", "NegControlProbe_00", "UnassignedCodeword_01"],
            "qv": [25.0, 25.0, 25.0, 25.0],
            "cell_id": [1, 1, 1, 1],
            "x_location": [0.0, 1.0, 2.0, 3.0],
            "y_location": [0.0, 1.0, 2.0, 3.0],
            "z_location": [0.0, 0.0, 0.0, 0.0],
        })
        pq.write_table(table, tmp_path / "transcripts.parquet")

        mdata = md.MuData({"xenium": ad.AnnData(X=np.zeros((1, 2), dtype=np.float32))})
        mdata.uns["spatial"] = {"lib": {"source_path": str(tmp_path)}}
        return mdata

    def test_default_excludes_control_transcripts(self, tmp_path):
        # Regression: include_controls's actual parameter default was True,
        # contradicting its own docstring ("If False (default), control-probe
        # transcripts are excluded") and every other place in the codebase
        # that assumes read_xenium's auto-loaded uns['transcripts'] already
        # excludes controls by default (pp.calculate_xenium_qc_metrics's
        # control_probe_summary stash exists specifically to compensate for
        # var['mean_qv'] being NaN on controls for exactly this reason; see
        # pl.gene_qv_scatter's docstring, which documents the same
        # assumption). With the buggy True default, a fresh
        # ab.datasets.xenium() + pipeline.filter_xenium() left the cell x
        # gene matrix control-free but uns['transcripts'] still full of
        # control-probe rows -- any downstream analysis consuming
        # uns['transcripts'] directly (e.g. tl.colocation_quotient in
        # pixel mode) silently reintroduced controls.
        mdata = self._mdata_pointing_at(tmp_path)
        ab.io.read_xenium_transcripts(mdata)
        names = set(mdata["xenium"].uns["transcripts"].column("feature_name").to_pylist())
        assert names == {"EPCAM", "CD3D"}

    def test_include_controls_true_keeps_them(self, tmp_path):
        mdata = self._mdata_pointing_at(tmp_path)
        ab.io.read_xenium_transcripts(mdata, include_controls=True)
        names = set(mdata["xenium"].uns["transcripts"].column("feature_name").to_pylist())
        assert names == {"EPCAM", "CD3D", "NegControlProbe_00", "UnassignedCodeword_01"}
