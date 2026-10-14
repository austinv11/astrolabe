"""Independent regressions for the October correctness audit and new analyses."""
from contextlib import contextmanager
import importlib
from unittest.mock import patch
import anndata as ad
import awkward as ak
import geopandas as gpd
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp
from scipy.stats import false_discovery_control, ttest_ind
from shapely.geometry import Polygon, box
import astrolabe as ab
from astrolabe._statistics import permutation_p
from astrolabe._utils import ImageSpec
from astrolabe.tl._icg import _ttest_vectorized


def cells(n=40, genes=2, seed=0):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.poisson(.7, (n, genes)).astype(float))
    a.var_names = [f"g{i}" for i in range(genes)]
    a.obsm["spatial"] = rng.uniform(0, 100, (n,2))
    a.obs["type"] = ["A"]*(n//2)+["B"]*(n-n//2)
    ab.tl.spatial_neighbors(a, n_neighbors=4)
    return a


def test_monte_carlo_tails_ties_nan_and_infinity():
    observed = np.array([-1., 1., np.nan, np.inf])
    simulations = np.array([[0.]*199, [1.]*199, [0.]*199, [np.inf]*199])
    p, n = permutation_p(observed, simulations)
    np.testing.assert_allclose(p[[0,1,3]], [.01, 1, 1])
    assert np.isnan(p[2]); assert (n == 199).all()
    p, n = permutation_p(np.array([1.]), np.array([[1., np.nan, 0.]]))
    assert n[0] == 2 and p[0] == 1
    assert np.isnan(permutation_p(np.array(0.), np.array([]))[0])
    # Exhaustive exchangeable ranks: two-sided p-values are super-uniform.
    ranks = np.arange(200)
    p = [float(permutation_p(np.asarray(float(i)), np.delete(ranks,i))[0]) for i in ranks]
    for alpha in [.005,.01,.025,.05,.1]:
        assert np.mean(np.array(p) <= alpha) <= alpha + 1e-12


@pytest.mark.parametrize("method", ["moran_local", "getis_ord", "moran_local_bv"])
def test_local_fdr_uses_empirical_full_family(method):
    a = cells()
    kwargs = dict(n_perms=39, seed=1)
    result = getattr(ab.tl, method)(a, ["g0","g1"], **kwargs) if method != "moran_local_bv" else ab.tl.moran_local_bv(a, ["g0","g1"], ["g1","g0"], **kwargs)
    valid = result.p_sim.notna()
    expected = false_discovery_control(result.loc[valid,"p_sim"].to_numpy())
    np.testing.assert_allclose(result.loc[valid,"p_adj"], expected)
    label = "type" if method == "getis_ord" else "quadrant"
    assert (result.loc[result.p_adj.ge(.05),label] == "NS").all()


@pytest.mark.parametrize("method", ["moran_local", "getis_ord", "moran_local_bv"])
def test_constant_local_fields_are_untestable(method):
    a = cells(); a.X[:] = 1
    result = getattr(ab.tl, method)(a, "g0", n_perms=19, seed=0) if method != "moran_local_bv" else ab.tl.moran_local_bv(a,"g0","g1",n_perms=19,seed=0)
    assert result.p_sim.isna().all() and result.p_adj.isna().all()


def test_lr_empty_null_edges_never_create_significance():
    a = ad.AnnData(np.ones((30,2))); a.var_names = ["L","R"]
    a.obs["type"] = ["A"]*2+["B"]*28
    a.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(([1.,1.],([0,1],[1,0])),shape=(30,30))
    pairs = pd.DataFrame({"ligand":["L"], "receptor":["R"]})
    ab.tl.ligrec(a,"type",pairs,n_perms=999,seed=5)
    result = a.uns["ligrec"]
    p = result["pvalues"].loc[("L","R"),("A","A")]
    assert np.isnan(p) or p == 1
    assert result["n_valid"].loc[("L","R"),("A","A")] < 999


def test_graph_isolates_overlapping_coordinate_frames():
    a = ad.AnnData(np.ones((4,1)))
    a.obsm["spatial"] = np.array([[0.,0.],[100.,0.]]*2)
    a.obs["library_id"] = ["one","one","two","two"]
    ab.tl.spatial_neighbors(a,n_neighbors=1)
    graph = a.obsp["spatial_neighbors_connectivities"].toarray()
    np.testing.assert_array_equal(graph, [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])


def test_affine_moves_transcripts_annotations_and_invalidates_graph():
    a = cells(4)
    table = pa.table({"x_location":[1.,3.],"y_location":[0.,0.]})
    regions = gpd.GeoDataFrame({"region_id":["r"]},geometry=[box(0,0,5,5)])
    a.uns["transcripts"] = table; a.uns["annotations"] = regions
    m = md.MuData({"cells":a}); m.uns["transcripts"] = table; m.uns["annotations"] = regions
    out = ab.pp.apply_affine(m,np.array([[1,0,10],[0,1,20],[0,0,1]]))
    for obj in [out,out["cells"]]:
        np.testing.assert_allclose(obj.uns["transcripts"]["x_location"].to_numpy(),[11,13])
        np.testing.assert_allclose(obj.uns["annotations"].total_bounds,[10,20,15,25])
    assert "spatial_neighbors_connectivities" not in out["cells"].obsp


def test_affine_invalidates_custom_results_and_parent_mirror():
    a = cells(); m = md.MuData({"cells":a}); m.uns["default_modality"] = "cells"
    ab.tl.moran_local(m,"g0",n_perms=9,seed=0,key_added="custom_local")
    assert "custom_local" in m.uns
    ab.pp.apply_affine(m,np.array([[1,0,1],[0,1,0],[0,0,1]]),inplace=True)
    assert "custom_local" not in m.uns and "custom_local" not in m["cells"].uns


def test_transform_crop_analysis_image_and_persistence_workflow(tmp_path):
    import tifffile
    a = cells(60); original = a.obsm["spatial"].copy()
    tx = pa.table({"x_location":original[:,0],"y_location":original[:,1],"cell_id":a.obs_names})
    regions = gpd.GeoDataFrame({"region_id":["roi"]},geometry=[box(0,0,60,100)])
    image_path = tmp_path/"image.tif"
    tifffile.imwrite(image_path,np.arange(128*128,dtype=np.uint16).reshape(128,128))
    spec = ImageSpec(path=str(image_path),pixel_size_um=1.,dimension_order="YX",full_shape=(128,128),
                     level_shapes=[(128,128)],level_scales=[(1.,1.)])
    a.uns.update(transcripts=tx,annotations=regions,spatial={"library":{"images":{"full":spec}}})
    m = md.MuData({"cells":a}); m.uns.update(default_modality="cells",transcripts=tx,annotations=regions)
    ab.tl.moran_local(m,"g0",n_perms=9,key_added="pre_transform",seed=0)
    ab.pp.apply_affine(m,np.array([[1,0,100],[0,1,200],[0,0,1]]),inplace=True)
    region = np.array([[100,200],[160,200],[160,300],[100,300]])
    cropped = ab.pp.crop(m,region)
    assert cropped.obs_names.equals(cropped["cells"].obs_names)
    assert cropped.uns["transcripts"].num_rows == cropped["cells"].uns["transcripts"].num_rows
    assert cropped.uns["transcripts"].num_rows == cropped.n_obs
    ab.tl.spatial_neighbors(cropped)
    ab.tl.assign_regions(cropped,cropped.uns["annotations"])
    table = ab.tl.moran_local(cropped,["g0","g1"],n_perms=19,seed=0)
    features = ab.pp.extract_image_features(cropped,mode="context",radius_um=2)
    assert np.isfinite(features.to_numpy()).all()
    path = tmp_path/"processed.h5mu"; ab.io.save(cropped,path)
    restored = ab.io.load(path)
    pd.testing.assert_frame_equal(restored["cells"].uns["moran_local"],table)
    assert restored["cells"].uns["transcripts"].num_rows == cropped.n_obs
    np.testing.assert_allclose(restored["cells"].obsm["image_features"],features)


def test_result_tables_round_trip_and_failed_save_preserves_file(tmp_path):
    a = cells()
    frame = ab.tl.moran_local(a,"g0",n_perms=19,seed=0)
    m = md.MuData({"cells":a}); m.uns["table"] = frame
    columns = pd.MultiIndex.from_tuples([("A","B"),("B","A")],names=["sender","receiver"])
    index = pd.MultiIndex.from_tuples([("L","R")],names=["ligand","receptor"])
    m.uns["lr"] = pd.DataFrame([[1.,np.nan]],index=index,columns=columns)
    m.uns["typed"] = pd.DataFrame({"count":pd.array([1,None],dtype="Int64"),
        "label":pd.Categorical(["a",None],categories=["b","a"],ordered=True)})
    path = tmp_path/"results.h5mu"; ab.io.save(m,path)
    loaded = ab.io.load(path)
    for key in ["table","lr","typed"]:
        pd.testing.assert_frame_equal(loaded.uns[key], m.uns[key])
    original = path.read_bytes()
    with patch.object(md.MuData,"write_h5mu",side_effect=RuntimeError("injected write failure")):
        with pytest.raises(RuntimeError):
            ab.io.save(m,path,overwrite=True)
    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.h5mu")) == [path]
    pd.testing.assert_frame_equal(m.uns["table"],frame)


def test_missing_categories_raise_before_cooccurrence():
    a = cells(); a.obs.loc[a.obs_names[0],"type"] = None
    with pytest.raises(ValueError,match="missing"):
        ab.tl.co_occurrence(a,"type",interval=[1,10])


def test_welch_degenerate_cases_match_scipy():
    x = np.column_stack([np.full(10,2.), np.ones(10)])
    y = np.ones((10,2))
    stat, _, p = _ttest_vectorized(x,y)
    reference = ttest_ind(x,y,equal_var=False)
    np.testing.assert_allclose(stat,reference.statistic,equal_nan=True)
    np.testing.assert_allclose(p,reference.pvalue,equal_nan=True)
    _, _, insufficient = _ttest_vectorized(np.ones((1,1)),np.ones((5,1))*2)
    assert np.isnan(insufficient).all()


@pytest.mark.parametrize("scale,base", [("linear",None),("log1p",None),("log1p",2.)])
def test_icg_fold_changes_respect_expression_scale(scale,base):
    a = cells(30,1); a.X[:] = 1; a.X[:5] = 2
    a.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix((np.ones(10),
        (np.r_[np.arange(5),np.arange(15,20)],np.r_[np.arange(15,20),np.arange(5)])),shape=(30,30))
    if scale == "log1p":
        a.X = np.log1p(a.X)/(1 if base is None else np.log(base))
    result = ab.tl.interaction_changed_genes(a,groupby="type",pairs=[("A","B")],
        min_cells=2,n_perms=0,expression_scale=scale,log_base=base)
    assert result.log2fc.iloc[0] == pytest.approx(1.,abs=1e-8)
    assert result.pval.isna().all()


def test_gradient_uses_stored_layer_and_detects_stale_sources():
    a = cells(80,1); a.layers["signal"] = a.obsm["spatial"][:,0,None]/100
    ab.tl.expression_gradient(a,"g0",layer="signal")
    ab.tl.gradient_significance(a,"g0",n_perms=9)
    expected = a.uns["gradient_significance"].copy()
    ab.tl.gradient_significance(a,"g0",layer="signal",n_perms=9)
    pd.testing.assert_frame_equal(expected,a.uns["gradient_significance"])
    a.layers["signal"][0,0] += 1
    with pytest.raises(ValueError,match="source changed"):
        ab.tl.gradient_significance(a,"g0",n_perms=9)


def test_constant_gradient_has_no_significant_coherence():
    a = cells(80,1); a.X[:] = 1
    ab.tl.expression_gradient(a,"g0")
    ab.tl.gradient_significance(a,"g0",n_perms=19)
    result = a.uns["gradient_significance"]
    assert result.p_coherence.isna().all() and result.p_coherence_adj.isna().all()


def test_gpu_flag_restored_after_transfer_failure():
    module = importlib.import_module("astrolabe._gpu")
    ab.use_gpu(False)
    with patch.object(module,"to_gpu",side_effect=RuntimeError("entry")):
        with pytest.raises(RuntimeError):
            with ab.gpu(cells(4)):
                pass
    assert not module.gpu_enabled()
    with patch.object(module,"to_gpu"), patch.object(module,"to_cpu",side_effect=RuntimeError("exit")):
        with pytest.raises(RuntimeError):
            with ab.gpu(cells(4)):
                pass
    assert not module.gpu_enabled()


def test_modality_filter_updates_parent_index():
    a = cells(6,1); a.X = np.arange(6.)[:,None]
    a.obs["total_counts"] = a.X[:,0]
    m = md.MuData({"cells":a}); m.uns["default_modality"] = "cells"
    ab.pp.filter_cells(m,min_counts=3)
    assert m.obs_names.equals(m["cells"].obs_names)


def test_statistical_weights_strip_diagonal_without_mutation():
    from astrolabe.tl._spatial_stats import _build_weights
    a = cells(); conn = a.obsp["spatial_neighbors_connectivities"]
    conn.setdiag(1)
    assert np.all(conn.diagonal() == 1)
    weights = _build_weights(a,"spatial_neighbors")
    assert not weights.sparse.diagonal().any()
    assert np.all(conn.diagonal() == 1)


def test_visium_native_polygon_centroid_and_unmatched_cells(tmp_path):
    from astrolabe.io._visium_hd import _attach_native_boundaries
    a = ad.AnnData(np.ones((2,1)))
    a.obs_names = ["cellid_000000001-1","cellid_000000002-1"]
    geo = gpd.GeoDataFrame({"cell_id":[1]},geometry=[box(10,10,20,20)])
    path = tmp_path/"segmentation.geojson"; geo.to_file(path,driver="GeoJSON")
    _attach_native_boundaries(a,path,pixel_size_um=1.)
    np.testing.assert_allclose(a.obsm["spatial"][0],[15,15])
    assert np.isnan(a.obsm["spatial"][1]).all()
    assert a.obs.spatial_valid.tolist() == [True,False]
    with pytest.raises(ValueError,match="finite"):
        ab.tl.spatial_neighbors(a)


def test_public_ripley_tail_correction(monkeypatch):
    from astrolabe.tl import _spatial_stats
    a = cells()
    def simulate(coords,radii,n_perms,rng,**kwargs):
        return np.zeros(len(radii)),np.ones((n_perms,len(radii)))
    monkeypatch.setattr(_spatial_stats,"_ripley_l_kdtree",simulate)
    result = ab.tl.ripley_l(a,n_perms=199,support=5)
    np.testing.assert_allclose(result["all"].p_value,.01)


def test_svg_seed_reproducibility_preserves_global_rng():
    a = cells()
    np.random.seed(71); before = np.random.get_state()
    one = ab.tl.spatially_variable_genes(a, n_perms=19,seed=4,n_jobs=1)
    after = np.random.get_state()
    two = ab.tl.spatially_variable_genes(a,n_perms=19,seed=4,n_jobs=2)
    pd.testing.assert_frame_equal(one,two)
    np.testing.assert_array_equal(before[1],after[1]); assert before[2:] == after[2:]
