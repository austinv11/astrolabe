"""Small schema regressions derived from public SFEData vendor exports."""
import awkward as ak
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

import astrolabe as ab


def _merscope_counts(path):
    # IDs exceed float64's exact integer range: preserve integer/string identity.
    ids = [112824700230101249, 112824700230101252]
    pd.DataFrame({"cell": ids, "Gene": [4, 7]}).to_csv(path / "cell_by_gene.csv", index=False)
    pd.DataFrame({"EntityID": ids, "center_x": [1., 5.], "center_y": [2., 6.]}).to_csv(path / "cell_metadata.csv", index=False)
    return ids


@pytest.mark.parametrize("encoding", ["wkb", "wkt"])
def test_merscope_root_geometry_and_z_index(tmp_path, encoding):
    ids = _merscope_counts(tmp_path)
    polygons = [Polygon([(4, 5), (6, 5), (6, 7)]), Polygon([(0, 1), (2, 1), (2, 3)])]
    values = [getattr(p, encoding) for p in polygons]
    column = "Geometry" if encoding == "wkb" else "GeometryWKT"
    pd.DataFrame({"EntityID": [ids[1], ids[0], ids[0]], "ZIndex": [3, 3, 2],
                  "ZLevel": [6., 6., 4.], "Type": ["cell"] * 3,
                  column: [*values, values[0]]}).to_parquet(tmp_path / "cell_boundaries.parquet")
    data = ab.io.read_merscope(tmp_path, z_level=3)["merscope"]
    assert data.obs_names.tolist() == list(map(str, ids))
    np.testing.assert_array_equal(data.X.toarray().ravel(), [4, 7])
    np.testing.assert_allclose(ak.to_numpy(data.obsm["cell_boundaries"][0]), polygons[1].exterior.coords)
    np.testing.assert_allclose(ak.to_numpy(data.obsm["cell_boundaries"][1]), polygons[0].exterior.coords)


def test_merscope_disconnected_geometry_not_flattened(tmp_path):
    ids = _merscope_counts(tmp_path)
    geom = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(3, 0), (4, 0), (4, 1)])])
    pd.DataFrame({"EntityID": [ids[0]], "ZIndex": [0], "GeometryWKT": [geom.wkt]}).to_parquet(tmp_path / "cell_boundaries.parquet")
    with pytest.raises(ValueError, match="disconnected"):
        ab.io.read_merscope(tmp_path)
    assert ab.io.read_merscope(tmp_path, load_boundaries=False)["merscope"].n_obs == 2


def test_cosmx_hyphen_polygon_name_and_cellid_across_fovs(tmp_path):
    pd.DataFrame({"fov": [1, 2], "cell_ID": [367, 367], "Gene": [5, 9]}).to_csv(tmp_path / "sample_exprMat_file.csv", index=False)
    pd.DataFrame({"fov": [2, 1], "cell_ID": [367, 367],
                  "CenterX_global_px": [200., 100.], "CenterY_global_px": [400., 300.]}).to_csv(tmp_path / "sample_metadata_file.csv", index=False)
    pd.DataFrame({"fov": [2, 2, 2, 1, 1, 1], "cellID": [367] * 6,
                  "x_global_px": [200, 201, 200, 100, 101, 100],
                  "y_global_px": [400, 400, 401, 300, 300, 301],
                  "x_local_px": [1] * 6, "y_local_px": [2] * 6}).to_csv(tmp_path / "sample-polygons.csv", index=False)
    data = ab.io.read_cosmx(tmp_path)["cosmx"]
    assert data.obs_names.tolist() == ["1_367", "2_367"]
    np.testing.assert_allclose(data.obsm["spatial"], np.array([[100, 300], [200, 400]]) * .12028)
    np.testing.assert_allclose(ak.to_numpy(data.obsm["cell_boundaries"][0]), np.array([[100, 300], [101, 300], [100, 301]]) * .12028)
