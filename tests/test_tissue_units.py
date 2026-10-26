import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

import astrolabe as ab
from astrolabe.tl._tissue_stats import _sample_mask, _masked_l
from astrolabe.tl._spatial_stats import _permutation_blocks


def tma():
    a = ad.AnnData(np.ones((8, 1)))
    a.var_names = ["g"]
    a.obs["slide"] = "s"
    a.obs["core_id"] = np.repeat(["a", "b"], 4)
    a.obsm["spatial"] = np.array([[1, 1], [1, 2], [2, 1], [2, 2]] * 2, float)
    masks = gpd.GeoDataFrame({"slide": ["s", "s"], "core_id": ["a", "b"]},
                            geometry=[shapely.box(0, 0, 4, 4)] * 2)
    ab.pp.set_tissue_masks(a, masks, library_key="slide")
    return a, masks


def test_cores_are_separate_despite_identical_coordinates():
    a, _ = tma()
    ab.tl.spatial_neighbors(a, n_neighbors=3)
    graph = a.obsp["spatial_neighbors_connectivities"]
    assert graph[:4, 4:].nnz == 0
    blocks, key = _permutation_blocks(a, "spatial_neighbors")
    assert key == "slide" and list(map(len, blocks)) == [4, 4]
    a.obsm["spatial"][4:] += 100
    a.uns["tissue_support"]["regions"].loc[1, "wkb"] = shapely.to_wkb(shapely.box(100, 100, 104, 104), hex=True)
    ab.tl.spatial_neighbors(a, n_neighbors=3)
    np.testing.assert_array_equal(graph.toarray(), a.obsp["spatial_neighbors_connectivities"].toarray())


def test_mask_holes_block_graph_edges_and_csr_draws():
    hole = shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                           holes=[[(4, 2), (6, 2), (6, 8), (4, 8)]])
    a = ad.AnnData(np.ones((4, 1)))
    a.obsm["spatial"] = np.array([[3, 5], [7, 5], [3, 1], [7, 1]], float)
    ab.pp.set_tissue_masks(a, gpd.GeoDataFrame(geometry=[hole]))
    ab.tl.spatial_neighbors(a, radius=5)
    graph = a.obsp["spatial_neighbors_connectivities"]
    assert graph[0, 1] == 0 and graph[2, 3] == 1
    points = _sample_mask(hole, 5000, np.random.default_rng(2))
    assert shapely.covers(hole, shapely.points(points)).all()
    assert abs((points[:, 0] < 4).mean() - 40/88) < .03


def test_fragmented_core_stays_one_unit_and_region_distances_do_not_cross_cores():
    a, masks = tma()
    regions = masks.iloc[:1].copy()
    regions["region_id"] = "r"
    values = ab.tl.distance_to_region(a, regions, region="r")
    assert np.isfinite(values[:4]).all() and np.isnan(values[4:]).all()
    with pytest.raises(ValueError, match="unit labels"):
        ab.tl.distance_to_region(a, regions.drop(columns="core_id"), region="r")


def test_ripley_core_results_invariant_to_adding_unrelated_core():
    a, _ = tma()
    result = ab.tl.ripley_l(a, radii=[0, .5, 1], n_perms=19, seed=2)
    first = next(value for value in result.values() if value.core_id.iloc[0] == "a")
    subset = a[:4].copy()
    alone = next(iter(ab.tl.ripley_l(subset, radii=[0, .5, 1], n_perms=19, seed=2).values()))
    pd.testing.assert_frame_equal(first.drop(columns="p_value_adj"), alone.drop(columns="p_value_adj"))


def test_border_estimator_matches_manual_pair_counts():
    mask = shapely.box(0, 0, 10, 10)
    points = np.array([[5, 5], [5, 6], [.1, .1]], float)
    # Only the two central points are eligible centers; each has one neighbor.
    value = _masked_l(points, np.array([1.]), mask, "border")[0]
    assert value == pytest.approx(np.sqrt(100*2/(2*2)/np.pi)-1)


def test_polygon_contact_graph_and_tissue_transform():
    a = ad.AnnData(np.ones((3, 1)))
    a.obsm["spatial"] = np.array([[.5, .5], [1.5, .5], [4.5, .5]])
    a.obsm["cell_boundaries"] = np.array([np.asarray(shapely.box(x, 0, x+1, 1).exterior.coords) for x in (0, 1, 4)])
    ab.pp.set_tissue_masks(a, gpd.GeoDataFrame(geometry=[shapely.box(0, 0, 5, 2)]))
    ab.tl.spatial_neighbors(a, coord_type="contact")
    graph = a.obsp["spatial_neighbors_connectivities"]
    assert graph[0, 1] == 1 and graph[1, 2] == 0
    ab.pp.rotate(a, 90)
    ab.tl.spatial_neighbors(a, coord_type="contact")
    np.testing.assert_array_equal(graph.toarray(), a.obsp["spatial_neighbors_connectivities"].toarray())


def test_crop_updates_observation_window_without_changing_original():
    a, _ = tma()
    cropped = ab.pp.crop(a, np.asarray(shapely.box(.5, .5, 2.5, 3).exterior.coords))
    np.testing.assert_allclose(cropped.uns["tissue_support"]["regions"].area_um2, 5.)
    np.testing.assert_allclose(a.uns["tissue_support"]["regions"].area_um2, 16.)
    ab.tl.spatial_neighbors(cropped, n_neighbors=3)
    assert cropped.obsp["spatial_neighbors_connectivities"][:4, 4:].nnz == 0


def test_scale_sensitivity_retains_graph_and_adjusts_entire_family():
    from statsmodels.stats.multitest import multipletests
    a, _ = tma()
    a.X[:, 0] = [1, 2, 3, 7] * 2
    ab.tl.spatial_neighbors(a, radius=1.1)
    original = a.obsp["spatial_neighbors_connectivities"].copy()
    result = ab.tl.spatial_scale_sensitivity(a, "g", graph_specs={"small": {"radius": 1.1}, "large": {"radius": 3}}, n_perms=39)
    np.testing.assert_allclose(result.p_adjusted_family, multipletests(result.p_sim, method="holm")[1])
    assert list(result.scale) == ["small", "large"]
    assert (original != a.obsp["spatial_neighbors_connectivities"]).nnz == 0


def test_landmarks_do_not_cross_cores_and_pooled_gradient_is_rejected():
    a, _ = tma()
    a.obs["type"] = ["landmark"] + ["other"] * 7
    ab.tl.bearing_to_group(a, "type", "landmark")
    assert np.isfinite(a.obs.bearing_to_landmark.iloc[1:4]).all()
    assert a.obs.bearing_to_landmark.iloc[4:].isna().all()
    ab.tl.spatial_neighbors(a, n_neighbors=3)
    with pytest.raises(ValueError, match="single spatial unit"):
        ab.tl.expression_gradient(a, "g")
