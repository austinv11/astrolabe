import subprocess
import sys

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import shapely

import astrolabe as ab
from astrolabe._utils import ImageSpec


def test_package_import_does_not_attempt_spatialdata():
    code = '''
import sys
class Forbidden:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "spatialdata" or fullname.startswith("spatialdata."):
            raise AssertionError("SpatialData import was attempted")
sys.meta_path.insert(0, Forbidden())
import astrolabe
import astrolabe.io
assert "spatialdata" not in sys.modules
assert callable(astrolabe.io.to_spatialdata)
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_spatialdata_roundtrip_retains_units_counts_transcripts_and_lazy_images(tmp_path):
    sd = pytest.importorskip("spatialdata")
    a = ad.AnnData(np.arange(8).reshape(4, 2))
    a.obs_names = ["c0", "c1", "c2", "c3"]
    a.var_names = ["g0", "g1"]
    a.obs["library_id"] = "slide"
    a.obs["core_id"] = ["a", "a", "b", "b"]
    a.obsm["spatial"] = np.array([[1, 1], [2, 2]]*2, float)
    a.obsm["cell_boundaries"] = np.array([np.asarray(shapely.box(x-.2, y-.2, x+.2, y+.2).exterior.coords) for x, y in a.obsm["spatial"]])
    ab.pp.set_tissue_support(a, gpd.GeoDataFrame({"library_id": ["slide", "slide"], "core_id": ["a", "b"]}, geometry=[shapely.box(0, 0, 4, 4)]*2))
    a.uns["transcripts"] = pa.table({"transcript_id": ["t0", "t1"], "cell_id": ["c0", "c2"],
        "library_id": ["slide", "slide"], "core_id": ["a", "b"], "feature_name": ["g0", "g1"],
        "x_location": [1., 1.], "y_location": [1., 1.], "z_location": [2., 3.]})
    a.uns["spatial"] = {"slide": {"images": {"morphology": ImageSpec(path=str(tmp_path / "unopened.tif"),
        pixel_size_um=.5, dimension_order="YX", full_shape=(4, 4), level_shapes=[(4, 4)], level_scales=[(1, 1)])}}}
    converted = ab.io.to_spatialdata(a)
    restored = ab.io.from_spatialdata(converted)
    np.testing.assert_array_equal(a.X, restored.X)
    np.testing.assert_array_equal(a.obsm["spatial"], restored.obsm["spatial"])
    assert list(restored.obs_names) == list(a.obs_names)
    assert list(restored.obs.core_id) == list(a.obs.core_id)
    pd.testing.assert_frame_equal(a.uns["transcripts"].to_pandas(), restored.uns["transcripts"].to_pandas(), check_dtype=False)
    assert isinstance(restored.uns["spatial"]["slide"]["images"]["morphology"], ImageSpec)
    path = tmp_path / "interchange.zarr"
    converted.write(path)
    reloaded = ab.io.from_spatialdata(sd.read_zarr(path))
    np.testing.assert_array_equal(a.X, reloaded.X)
    assert reloaded.uns["spatial_units"]["core_key"] == "core_id"


def test_custom_tissue_window_follows_spatialdata_transformation():
    pytest.importorskip("spatialdata")
    from spatialdata.transformations import Translation, set_transformation
    a = ad.AnnData(np.ones((2, 1)), obsm={"spatial": np.array([[1., 1.], [2., 2.]])})
    ab.pp.set_tissue_support(a, gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 3, 3)]), key_added="window")
    converted = ab.io.to_spatialdata(a, centroid_radius=.5)
    frame = next(iter(converted.tables["table"].uns["astrolabe_interchange"]["frames"].values()))
    for shape in converted.shapes.values():
        set_transformation(shape, Translation([10., 20.], axes=("x", "y")), to_coordinate_system=frame["coordinate_system"])
    restored = ab.io.from_spatialdata(converted)
    np.testing.assert_allclose(restored.obsm["spatial"], a.obsm["spatial"] + [10, 20])
    support = shapely.from_wkb(restored.uns["window"]["regions"].wkb.iloc[0])
    np.testing.assert_allclose(support.bounds, [10, 20, 13, 23])
    ab.tl.spatial_neighbors(restored, radius=2)


def test_real_cropped_xenium_spatialdata_roundtrip():
    pytest.importorskip("spatialdata")
    from pathlib import Path
    root = Path(__file__).parent / "data" / "xenium_4_tiny"
    source = ab.io.read_xenium(root)["xenium"]
    cropped = ab.pp.auto_crop(source, percentile=10, padding=0)
    assert 0 < cropped.n_obs < source.n_obs
    restored = ab.io.from_spatialdata(ab.io.to_spatialdata(cropped))
    assert (restored.X != cropped.X).nnz == 0
    np.testing.assert_array_equal(restored.obs_names, cropped.obs_names)
    np.testing.assert_allclose(restored.obsm["spatial"], cropped.obsm["spatial"])
    assert restored.uns["transcripts"].num_rows == cropped.uns["transcripts"].num_rows
