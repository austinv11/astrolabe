"""Regression contracts for the approved spatial-field gaps."""
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import astrolabe as ab

pytestmark = pytest.mark.filterwarnings("ignore::astrolabe._statistics.PermutationResolutionWarning")


def data():
    rng = np.random.default_rng(12)
    a = ad.AnnData(rng.uniform(1, 5, (32, 3)))
    a.var_names = ["L", "R", "common"]
    a.obsm["spatial"] = rng.uniform(0, 10, (32, 2))
    a.obs["slide"] = np.repeat(["one", "two"], 16)
    a.obs["type"] = pd.Categorical(["A", "B"] * 16)
    ab.tl.spatial_neighbors(a, n_neighbors=3, library_key="slide")
    return a


def lr(a, **kwargs):
    ab.tl.ligrec(a, "type", pd.DataFrame({"ligand": ["L"], "receptor": ["R"]}),
                 n_perms=29, min_valid_perms=1, seed=8, **kwargs)
    return a.uns["ligrec"]


def panel(a, overlapping=False):
    a.uns["measured_features_library_key"] = "slide"
    a.uns["measured_features"] = pd.DataFrame(
        {"one": [True, overlapping, True], "two": [False, True, True]}, index=a.var_names)
    a.X[16:, 0] = 0
    if not overlapping:
        a.X[:16, 1] = 0


def test_lr_resolves_graph_library_and_preserves_explicit_pooling():
    a = data()
    a.X[:16] = 1
    a.X[16:] = 100
    auto, explicit, pooled = lr(a.copy()), lr(a.copy(), library_key="slide"), lr(a.copy(), library_key=None)
    pd.testing.assert_frame_equal(auto["pvalues"], explicit["pvalues"])
    assert auto["metadata"]["library_key"] == "slide"
    assert pooled["metadata"]["library_key"] is None


@pytest.mark.parametrize("bad", ["missing", "cross_edge"])
def test_inference_rejects_invalid_exchangeability(bad):
    a = data()
    if bad == "missing":
        a.obs.loc[a.obs_names[0], "slide"] = None
    else:
        conn = a.obsp["spatial_neighbors_connectivities"].tolil()
        conn[0, 16] = 1
        a.obsp["spatial_neighbors_connectivities"] = conn.tocsr()
    with pytest.raises(ValueError):
        lr(a)
    with pytest.raises(ValueError):
        ab.tl.interaction_changed_genes(a, groupby="type", expression_scale="linear", n_perms=9)


def test_no_joint_measured_lr_is_untested():
    a = data()
    panel(a)
    result = lr(a)
    assert result["pvalues"].isna().all().all()
    assert (result["status"] == "unmeasured").all().all()
    assert (result["n_valid"] == 0).all().all()
    assert not result["eligible_libraries"].to_numpy().any()


def test_partial_lr_requires_opt_in_and_matches_selected_library():
    a = data()
    panel(a, overlapping=True)
    with pytest.raises(ValueError, match="Partial"):
        lr(a)
    eligible = lr(a, coverage_policy="eligible")
    subset = a[:16].copy()
    ab.tl.spatial_neighbors(subset, n_neighbors=3, library_key="slide")
    reference = lr(subset)
    for name in ("means", "pvalues", "n_valid"):
        pd.testing.assert_frame_equal(eligible[name], reference[name])


@pytest.mark.parametrize("method", ["moran", "spatially_variable_genes", "rank_genes_groups"])
def test_structural_zeros_rejected(method):
    a = data()
    panel(a)
    args = ("L",) if method == "moran" else ("type",) if method == "rank_genes_groups" else ()
    with pytest.raises(ValueError, match="not measured"):
        getattr(ab.tl, method)(a, *args)


def test_panel_selection_preserves_source_and_removes_derived_graphs():
    a = data()
    panel(a)
    common = ab.pp.select_analysis_panel(a)
    assert list(common.var_names) == ["common"]
    assert common.n_obs == 32 and not common.obsp
    selected = ab.pp.select_analysis_panel(a, genes=["L"])
    assert selected.n_obs == 16 and list(selected.var_names) == ["L"]
    assert a.shape == (32, 3) and a.obsp
    with pytest.raises(ValueError, match="No libraries"):
        ab.pp.select_analysis_panel(a, genes=["L", "R"])


def test_normalization_invalidates_only_changed_layer_and_parent_mirror():
    a = data()
    a.layers["counts"] = a.X.copy()
    m = md.MuData({"rna": a})
    ab.tl.moran(m, "L", n_perms=9, key_added="normalized", modality="rna")
    ab.tl.moran(m, "L", layer="counts", n_perms=9, key_added="counts_result", modality="rna")
    ab.pp.log1p(m, modality="rna")
    assert "normalized" not in a.uns and "normalized" not in m.uns
    assert "counts_result" in a.uns and "counts_result" in m.uns
    assert "spatial_neighbors" in a.uns
    assert ab.tl.validate_results(m, modality="rna").empty


def test_graph_rebuild_removes_dependent_results_and_retains_rebuilt_graph():
    a = data()
    ab.tl.moran(a, "L", n_perms=9)
    ab.tl.spatial_neighbors(a, n_neighbors=7, library_key="slide")
    assert "moran" not in a.uns
    assert "spatial_neighbors" in a.uns
    assert a.obsp["spatial_neighbors_connectivities"].nnz > 200
    assert ab.tl.validate_results(a).empty


@pytest.mark.parametrize("source", ["expression", "coordinates", "labels", "genes"])
def test_external_edits_have_explicit_validation(source):
    a = data()
    lr(a)
    if source == "expression":
        a.X[0, 0] += 1
    elif source == "coordinates":
        a.obsm["spatial"][0, 0] += 1
    elif source == "labels":
        a.obs.loc[a.obs_names[0], "type"] = "B"
    else:
        a = a[:, :2].copy()
    stale = ab.tl.validate_results(a)
    assert any(stale.result.str.startswith("ligrec:"))
    assert "ligrec" in a.uns
    ab.tl.validate_results(a, invalidate=True)
    assert "ligrec" not in a.uns
    assert ab.tl.validate_results(a).empty


def test_fingerprints_survive_h5ad_roundtrip(tmp_path):
    a = data()
    ab.tl.moran(a, "L", n_perms=9)
    path = tmp_path / "tracked.h5ad"
    a.write_h5ad(path)
    restored = ad.read_h5ad(path)
    assert ab.tl.validate_results(restored).empty


def test_consumers_reject_stale_graph_and_transitive_profile():
    a = data()
    ab.tl.niche_composition(a, "type")
    a.obs["niches"] = pd.Categorical(["n"] * a.n_obs)
    a.uns["niches"] = {"profile_key": "niche_composition"}
    a.obsm["spatial"][0] += 100
    with pytest.raises(ValueError, match="Stale result spatial_neighbors"):
        ab.tl.moran(a, "L", n_perms=9)
    with pytest.raises(ValueError, match="Stale result spatial_neighbors"):
        ab.tl.fit_niche_model(a)
    ab.tl.spatial_neighbors(a, library_key="slide", n_neighbors=3)
    assert "niche_composition" not in a.obsm
    ab.tl.moran(a, "L", n_perms=9)


def test_feature_subset_does_not_stale_geometric_graph():
    a = data()[:, ["L"]].copy()
    ab.tl.moran(a, "L", n_perms=9)
    assert ab.tl.validate_results(a).empty


def test_image_metadata_is_not_a_graph_dependency():
    a = data()
    a.uns["spatial"] = {"one": {"images": {}, "path": "original"}}
    ab.tl.spatial_neighbors(a, library_key="slide", n_neighbors=3)
    a.uns["spatial"]["one"]["path"] = "relocated"
    assert ab.tl.validate_results(a).empty


def test_plot_rejects_stale_statistical_result():
    a = data()
    ab.tl.nhood_enrichment(a, "type", n_perms=9)
    a.obsm["spatial"][0] += 100
    with pytest.raises(ValueError, match="Stale result"):
        ab.pl.nhood_enrichment_plot(a, "type")


def test_marker_coverage_follows_raw_axis_and_explicit_mask(monkeypatch):
    a = data()
    panel(a)
    a.raw = a.copy()
    subset = a[:, ["common"]].copy()
    with pytest.raises(ValueError, match="not measured"):
        ab.tl.rank_genes_groups(subset, "type")
    calls = []
    monkeypatch.setattr("astrolabe.tl._de.sc.tl.rank_genes_groups", lambda *args, **kwargs: calls.append(kwargs))
    ab.tl.rank_genes_groups(subset, "type", use_raw=False)
    ab.tl.rank_genes_groups(a, "type", use_raw=False, mask_var=np.array([False, False, True]))
    assert len(calls) == 2


def test_pseudobulk_honors_custom_coverage_axis():
    a = data()
    panel(a)
    a.layers["counts"] = np.ones(a.shape, dtype=int)
    result = ab.pp.pseudobulk(a, sample_key="slide", min_cells=1)
    np.testing.assert_array_equal(result.layers["measured"].toarray(), [[True, False, True], [False, True, True]])
