"""Formula/reference checks for the new lightweight analysis APIs."""
from contextlib import contextmanager
import anndata as ad
import awkward as ak
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
from shapely.geometry import box, Polygon
import astrolabe as ab
from astrolabe._utils import ImageSpec
from tests.test_correctness_roadmap import cells


def test_niche_profiles_are_binary_nonself_fractions():
    a = cells(4)
    a.obs["type"] = pd.Categorical(["A","A","B","B"],categories=["B","A"])
    graph = sp.csr_matrix([[2,4,8,0],[4,2,0,0],[8,0,2,0],[0,0,0,2]])
    a.obsp["spatial_neighbors_connectivities"] = graph
    fractions = ab.tl.niche_composition(a,"type")
    assert sp.isspmatrix_csr(fractions)
    np.testing.assert_allclose(fractions.toarray(),[[.5,.5],[0,1],[0,1],[0,0]])
    assert a.uns["niche_composition"]["categories"].tolist() == ["B","A"]
    assert a.obs.niche_composition_valid.tolist() == [True,True,True,False]
    ab.tl.niches(a,"type",seed=1)
    assert pd.isna(a.obs.niches.iloc[3])


def test_regions_boundary_holes_overlap_and_signed_distance():
    a = cells(4); a.obsm["spatial"] = np.array([[0.,0.],[1.,1.],[2.,2.],[5.,5.]])
    geometry = Polygon([(0,0),(4,0),(4,4),(0,4)],holes=[[(1.5,1.5),(2.5,1.5),(2.5,2.5),(1.5,2.5)]])
    regions = gpd.GeoDataFrame({"region_id":["r"]},geometry=[geometry])
    membership = ab.tl.assign_regions(a,regions)
    assert membership.obs_name.tolist() == ["0","1"]
    dist = ab.tl.distance_to_region(a,regions,region="r")
    np.testing.assert_allclose(dist,[0,-np.sqrt(.5),.5,np.sqrt(2)])
    overlap = gpd.GeoDataFrame({"region_id":["r","s"]},geometry=[geometry,box(0,0,3,3)])
    with pytest.raises(ValueError,match="overlap"):
        ab.tl.assign_regions(a,overlap)
    assert len(ab.tl.assign_regions(a,overlap,overlaps="membership")) == 5


def test_region_summary_counts_composition_expression_and_density():
    a = cells(4); a.obs["region"] = ["r"]*4; a.obs["sample"] = ["donor"]*4
    a.X = np.arange(8.).reshape(4,2)
    result = ab.tl.region_summary(a,sample_key="sample",groupby="type",genes=["g0"],analysis_area={("donor","r"):100})
    np.testing.assert_allclose(result.fraction,[.5,.5]); np.testing.assert_allclose(result.density_per_um2,[.02,.02])
    np.testing.assert_allclose(result.mean_g0,[1,5])


def test_pseudobulk_exact_counts_metadata_and_missing_panel():
    a = cells(4); a.X = sp.csr_matrix([[1,0],[2,0],[3,4],[5,6]])
    a.layers["counts"] = a.X.copy()
    a.obs["sample"] = ["s1","s1","s2","s2"]
    a.obs["condition"] = ["ctrl","ctrl","case","case"]
    a.obs["library_id"] = ["l1","l1","l2","l2"]
    a.uns["measured_features"] = pd.DataFrame({"l1":[True,False],"l2":[True,True]},index=a.var_names)
    result = ab.pp.pseudobulk(a,sample_key="sample",metadata_keys=["condition"],min_cells=2)
    np.testing.assert_array_equal(result.X.toarray(),[[3,0],[8,10]])
    np.testing.assert_array_equal(result.layers["measured"].toarray(),[[True,False],[True,True]])
    assert result.obs.n_cells.tolist() == [2,2]
    a.layers["counts"].data = a.layers["counts"].data.astype(float)+.1
    with pytest.raises(ValueError,match="integer raw counts"):
        ab.pp.pseudobulk(a,sample_key="sample")


def test_lr_complex_minimum_subunit_and_explicit_untested_status():
    a = cells(4,3); a.var_names = ["L1","L2","R"]
    a.X = sp.csr_matrix([[6.,2.,3.]]*4)
    resource = pd.DataFrame({"ligand":["complex","missing"],"receptor":["R","R"],
                            "ligand_members":[["L1","L2"],["absent"]],"receptor_members":[["R"],["R"]]})
    resource = ab.tl.import_ligrec_resource(resource,species="human",version="test")
    ab.tl.ligrec(a,"type",resource,n_perms=39,seed=1)
    result = a.uns["ligrec"]
    np.testing.assert_allclose(result["means"].iloc[0].dropna(),2.5)
    assert (result["status"].iloc[1] == "missing_subunit").all()
    assert result["pvalues"].iloc[1].isna().all()
    assert result["metadata"]["resource"]["species"] == "human"


def test_cell_type_pair_indices_do_not_overflow_compact_category_codes():
    a = cells(32); a.X[:] = 1
    categories = [f"type{i}" for i in range(16)]
    a.obs["type"] = pd.Categorical(np.repeat(categories,2),categories=categories)
    pairs = pd.DataFrame({"ligand":["g0"],"receptor":["g1"]})
    ab.tl.ligrec(a,"type",pairs,n_perms=19,min_valid_perms=1,seed=0)
    assert a.uns["ligrec"]["means"].shape == (1,256)
    np.testing.assert_allclose(a.uns["ligrec"]["means"].to_numpy()[np.isfinite(a.uns["ligrec"]["means"])],1.)
    ab.tl.nhood_enrichment(a,"type",n_perms=9,seed=0)
    assert a.uns["nhood_enrichment"]["counts"].shape == (16,16)


def test_image_summary_and_glcm_match_independent_constant_reference(monkeypatch):
    a = cells(4); a.obsm["spatial"] = np.array([[2.,2.],[4.,4.],[20.,20.],[2.,2.]])
    a.obsm["cell_boundaries"] = ak.Array([[[1,1],[3,1],[3,3],[1,3]]]*4)
    spec = ImageSpec(path="fixture",pixel_size_um=1.,dimension_order="YX",full_shape=(10,10),
                     level_shapes=[(10,10)],level_scales=[(1.,1.)])
    @contextmanager
    def open_image(self):
        yield np.full((10,10),64,dtype=np.uint8)
    monkeypatch.setattr(ImageSpec,"open_zarr",open_image)
    result = ab.pp.extract_image_features(a,spec=spec,mode="context",radius_um=1,texture_range=(0,255))
    row = result.iloc[0]
    assert row.channel_0_mean == 64 and row.channel_0_median == 64 and row.channel_0_std == 0
    assert row.channel_0_contrast == 0 and row.channel_0_homogeneity == 1 and row.channel_0_energy == 1
    assert result.iloc[2].isna().all() and a.obs.image_features_coverage.iloc[2] == 0


def test_image_affine_chain_is_used_for_sampling():
    from astrolabe.pp._if import _resolve_affine
    spec = ImageSpec(path="fixture",pixel_size_um=2.,dimension_order="YX",full_shape=(10,10),
                     level_shapes=[(10,10)],level_scales=[(1.,1.)],
                     transform_chain=[np.array([[1,0,10],[0,1,20],[0,0,1]])])
    homogeneous = np.array([14.,26.,1.])
    np.testing.assert_allclose(_resolve_affine(None,spec)@homogeneous,[2,3,1])


def test_texture_matches_skimage_on_rectangular_pattern():
    from skimage.feature import graycomatrix, graycoprops
    from astrolabe.pp._image_features import _texture
    pattern = np.arange(80).reshape(8,10) % 4
    reference = graycomatrix(pattern.astype(np.uint8),[1],[0,np.pi/2,np.pi/4,3*np.pi/4],
                             levels=4,symmetric=True,normed=True)
    expected = [graycoprops(reference,name).mean() for name in ["contrast","homogeneity","energy"]]
    np.testing.assert_allclose(_texture(pattern,np.ones(pattern.shape,dtype=bool),levels=4),expected)


def cohort_cells():
    a = cells(24)
    a.obs["sample"] = np.repeat(["s1","s2","s3","s4"],6)
    a.obs["condition"] = np.repeat(["ctrl","ctrl","case","case"],6)
    a.obs["type"] = ["A","A","A","B","B","B"]*4
    a.obs["library_id"] = a.obs["sample"]
    a.obsm["spatial"] = np.tile(np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]],float),(4,1))
    ab.tl.spatial_neighbors(a,n_neighbors=2)
    return a


def test_cohort_contact_expectation_and_null_contrast():
    a = cohort_cells()
    result = ab.tl.diff_nhood_enrichment(a,"type",sample_key="sample",condition_key="condition",
                                         contrast=("case","ctrl"),n_perms=99,n_boot=99,seed=2)
    np.testing.assert_allclose(result.effect,0)
    np.testing.assert_allclose(result.pvalue,1)
    assert (result.n_tested == 2).all() and (result.n_reference == 2).all()
    graph = a.obsp["spatial_neighbors_connectivities"][:6,:6].toarray()
    fractions = (graph[:3,3:].sum(axis=1)/graph[:3].sum(axis=1)).mean()
    expected = fractions-3/5
    assert a.uns["diff_nhood_enrichment"]["sample_scores"].iloc[0].loc[("A","B")] == pytest.approx(expected)


def test_cohort_rejects_block_confounded_condition():
    a = cohort_cells(); a.obs["batch"] = a.obs.condition
    with pytest.raises(ValueError,match="confounded"):
        ab.tl.diff_nhood_enrichment(a,"type",sample_key="sample",condition_key="condition",
                                   contrast=("case","ctrl"),block_key="batch",n_perms=9,n_boot=9)
