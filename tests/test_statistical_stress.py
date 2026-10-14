"""Regressions for stale outputs, exchangeability and Monte Carlo resolution."""
import warnings

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import false_discovery_control

import astrolabe as ab
from astrolabe._statistics import (PermutationResolutionWarning, adjust_p,
                                   permutation_resolution)
from tests.test_correctness_roadmap import cells


@pytest.mark.parametrize("container", ["adata", "mdata"])
@pytest.mark.parametrize("operation", ["crop", "affine"])
def test_spatial_changes_clear_svg_columns_and_preserve_other_gene_metadata(container, operation):
    a = cells(30)
    a.var["annotation"] = ["marker", "other"]
    a.var["sv_user_note"] = ["keep", "keep"]
    data = a if container == "adata" else md.MuData({"cells": a})
    if container == "mdata":
        data.uns["default_modality"] = "cells"
    ab.tl.spatially_variable_genes(data, key_added="custom_svg")
    if operation == "crop":
        out = ab.pp.crop(data, np.array([[0, 0], [50, 0], [50, 100], [0, 100]]))
    elif container == "adata":
        out = ab.pp.rotate(data, 30)
    else:
        out = ab.pp.apply_affine(data, np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]]))
    target = out if container == "adata" else out["cells"]
    assert "custom_svg" not in target.uns and "custom_svg" not in out.uns
    assert "sv_score" not in target.var and "sv_pval_adj" not in target.var
    assert "sv_p_min" not in target.var
    assert target.var.annotation.tolist() == ["marker", "other"]
    assert target.var.sv_user_note.tolist() == ["keep", "keep"]
    assert "sv_score" in a.var  # default copy leaves the original analysis intact


def test_loaded_svg_columns_are_cleared_without_registry():
    a = cells(30)
    ab.tl.spatially_variable_genes(a)
    a.uns.pop("_astrolabe_spatial_results")
    cropped = ab.pp.crop(a, np.array([[0, 0], [50, 0], [50, 100], [0, 100]]))
    assert not any(c.startswith("sv_") for c in cropped.var)


def libraries(seed=8):
    rng = np.random.default_rng(seed)
    a = ad.AnnData(rng.normal(size=(70, 3)))
    a.var_names = ["g0", "g1", "g2"]
    a.obs_names = [f"cell{i}" for i in range(70)]
    a.obs["library_id"] = ["small"]*20 + ["large"]*50
    a.obsm["spatial"] = rng.uniform(0, 100, (70, 2))
    ab.tl.spatial_neighbors(a, n_neighbors=4)
    return a


def local(a, method):
    if method == "moran_local_bv":
        return ab.tl.moran_local_bv(a, ["g0", "g1"], ["g1", "g2"], n_perms=99, seed=4)
    return getattr(ab.tl, method)(a, ["g0", "g1"], n_perms=99, seed=4)


@pytest.mark.parametrize("method", ["moran_local", "moran_local_bv", "getis_ord"])
def test_library_scoped_local_null_matches_independent_fits_and_pooled_bh(method):
    a = libraries()
    if method == "getis_ord":
        a.X = np.exp(a.X)
        a.X[20:] *= 100
    else:
        a.X[20:] += 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        combined = local(a, method)
        parts = [local(a[a.obs.library_id == lib].copy(), method) for lib in ["small", "large"]]
    expected = pd.concat(parts).reindex(combined.index)
    np.testing.assert_allclose(combined.p_sim, expected.p_sim, equal_nan=True)
    np.testing.assert_allclose(combined.p_adj, false_discovery_control(expected.p_sim), equal_nan=True)
    assert combined.attrs["n_libraries"] == 2
    assert combined.attrs["permutation_resolution"]["n_tests"] == 140
    assert combined.index.get_level_values("obs_name")[:70].tolist() == a.obs_names.tolist()


@pytest.mark.parametrize("method", ["moran_local", "moran_local_bv", "getis_ord"])
def test_library_baseline_changes_do_not_create_local_evidence(method):
    a = libraries()
    if method == "getis_ord":
        a.X = np.exp(a.X)
    shifted = a.copy()
    shifted.X[20:] = shifted.X[20:]*100 if method == "getis_ord" else shifted.X[20:]+100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        before, after = local(a, method), local(shifted, method)
    np.testing.assert_allclose(before.p_sim, after.p_sim)
    if method != "getis_ord":
        assert before.quadrant_raw.tolist() == after.quadrant_raw.tolist()


@pytest.mark.parametrize("flavor", ["moran", "geary"])
def test_library_shift_svg_uses_conditional_null_and_worker_reproducibility(flavor):
    a = libraries()
    # No within-library variation: shifts between slides cannot be evidence
    # against the conditional within-slide expression null.
    a.X[:20] = 1; a.X[20:] = 100
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        one = ab.tl.spatially_variable_genes(a, flavor=flavor, n_perms=99, seed=7, n_jobs=1)
        many = ab.tl.spatially_variable_genes(a, flavor=flavor, n_perms=99, seed=7, n_jobs=2)
    np.testing.assert_allclose(one.pval, 1)
    np.testing.assert_allclose(one.pval, many.pval)
    assert not one.spatially_variable.any()
    with pytest.raises(ValueError, match="multiple libraries require"):
        ab.tl.spatially_variable_genes(a, flavor=flavor, n_perms=0)


@pytest.mark.parametrize("method", ["moran_local", "moran_local_bv", "getis_ord"])
def test_missing_library_labels_are_rejected_before_local_inference(method):
    a = libraries(); a.obs.loc[a.obs_names[0], "library_id"] = None
    with pytest.raises(ValueError, match="library labels"):
        local(a, method)


def test_cross_library_graph_cannot_be_used_for_conditional_inference():
    a = libraries()
    graph = a.obsp["spatial_neighbors_connectivities"].tolil()
    graph[0, 25] = 1
    a.obsp["spatial_neighbors_connectivities"] = graph.tocsr()
    with pytest.raises(ValueError, match="edges across libraries"):
        local(a, "moran_local")
    with pytest.raises(ValueError, match="edges across libraries"):
        ab.tl.spatially_variable_genes(a, n_perms=99)


def test_too_small_library_is_explicitly_rejected():
    a = libraries()[[0, 1, *range(20, 70)]].copy()
    with pytest.raises(ValueError, match="three cells per library"):
        local(a, "moran_local")


@pytest.mark.parametrize("alternative,factor", [("two-sided", 2), ("greater", 1), ("less", 1)])
def test_resolution_uses_valid_draws_excludes_untested_and_distinguishes_tails(alternative, factor):
    floors, limited, info = permutation_resolution([.03, .1, np.nan], [99, 19, 999],
        alternative=alternative, warn=False)
    np.testing.assert_allclose(floors[:2], [factor/100, factor/20])
    assert np.isnan(floors[2]) and not limited[2]
    assert info["n_tests"] == 2
    assert limited.tolist() == [False, True, False]


def test_resolution_is_rank_one_diagnostic_and_not_a_claim_of_no_bh_discoveries():
    p = np.full(10, .02)
    with pytest.warns(PermutationResolutionWarning, match="BH/BY may still reject multiple"):
        floors, limited, info = permutation_resolution(p, np.full(10, 99))
    assert limited.all() and (adjust_p(p) < .05).all()
    assert info["required_valid_permutations_rank1"] == 400
    by = permutation_resolution(p, np.full(10, 99), method="fdr_by", warn=False)[2]
    assert by["rank1_threshold"] == pytest.approx(.05/(10*sum(1/np.arange(1, 11))))
    assert by["required_valid_permutations_rank1"] > 400


@pytest.mark.parametrize("method", ["fdr_bh", "fdr_by"])
def test_screening_does_not_shrink_the_requested_correction_family(method):
    p = np.array([.005, np.nan, .02])
    expected = false_discovery_control([.005, .02, 1., 1.], method=method[4:])[:2]
    actual = adjust_p(p, method, family_size=4)
    np.testing.assert_allclose(actual[[0, 2]], expected)
    assert np.isnan(actual[1])
    info = permutation_resolution(p, [99, 99, 99], method=method, family_size=4, warn=False)[2]
    assert info["n_tests"] == 4 and info["n_tested"] == 2


def test_resolution_analytic_and_untested_entries_do_not_warn():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        floors, flags, info = permutation_resolution([.01, np.nan], [0, 0])
    assert not captured and np.isnan(floors).all() and not flags.any()
    assert info["min_p_min"] is None


def test_local_resolution_and_custom_family_alpha_are_reported():
    a = cells(30)
    with pytest.warns(PermutationResolutionWarning):
        result = ab.tl.moran_local(a, "g0", n_perms=19, alpha=.01, seed=3)
    np.testing.assert_allclose(result.p_min, .1)
    assert result.resolution_limited.all()
    assert result.attrs["permutation_resolution"]["rank1_threshold"] == pytest.approx(.01/30)


@pytest.mark.parametrize("analysis", ["ripley", "clq"])
def test_pointwise_permutation_floors_do_not_claim_multiple_testing_correction(analysis):
    a = cells(40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        if analysis == "ripley":
            ab.tl.ripley_l(a, n_perms=39, support=5, seed=3)
            result = a.uns["ripley_l"]["all"]
        else:
            result = ab.tl.colocation_quotient(a, "type", n_perms=39, seed=3)
    np.testing.assert_allclose(result.p_min, .05)
    assert result.attrs["permutation_resolution"]["method"] is None
    assert result.attrs["permutation_resolution"]["rank1_threshold"] == .05


def test_sparse_panel_missing_subunits_remain_unavailable():
    a = cells(40)
    a.X[:] = 0
    resource = ab.tl.import_ligrec_resource(pd.DataFrame({
        "ligand": ["g0", "complex"], "receptor": ["g1", "g1"],
        "ligand_members": [["g0"], ["g0", "unmeasured"]]}), species="human")
    ab.tl.ligrec(a, "type", resource, n_perms=99, seed=0)
    result = a.uns["ligrec"]
    assert result["pvalues"].isna().all().all()
    assert result["p_min"].isna().all().all()
    assert not result["resolution_limited"].any().any()
    assert (result["status"].loc[("complex", "g1")] == "missing_subunit").all()


def test_lr_floor_uses_valid_scores_rather_than_attempted_draws():
    a = cells(40); a.X[:] = 1
    a.obs["type"] = ["common"]*38+["rare"]*2
    rows = np.arange(40); cols = (rows+1)%40
    a.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(
        (np.ones(80), (np.r_[rows, cols], np.r_[cols, rows])), shape=(40, 40))
    resource = ab.tl.import_ligrec_resource(pd.DataFrame({"ligand": ["g0"], "receptor": ["g1"]}), species="human")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        ab.tl.ligrec(a, "type", resource, n_perms=99, min_valid_perms=1, seed=12)
    result = a.uns["ligrec"]
    n = result["n_valid"].loc[("g0", "g1"), ("rare", "rare")]
    assert 0 < n < 99
    assert result["p_min"].loc[("g0", "g1"), ("rare", "rare")] == pytest.approx(1/(n+1))
    assert result["pvalues"].loc[("g0", "g1"), ("rare", "rare")] == 1


def test_lr_expression_gate_retains_original_family_and_strong_signal():
    a = cells(80); a.X[:] = 0
    a.X[:40, 0] = 10; a.X[40:, 1] = 10
    resource = ab.tl.import_ligrec_resource(pd.DataFrame({"ligand": ["g0"], "receptor": ["g1"]}), species="human")
    ab.tl.ligrec(a, "type", resource, n_perms=999, seed=5)
    result = a.uns["ligrec"]
    p = result["pvalues"].loc[("g0", "g1"), ("A", "B")]
    adjusted = result["pvalues_adj"].loc[("g0", "g1"), ("A", "B")]
    assert adjusted == pytest.approx(min(1, 4*p)) and adjusted < .05
    assert result["pvalues"].notna().sum().sum() == 1
    assert result["metadata"]["family_size"] == 4
    assert result["metadata"]["permutation_resolution"]["n_tests"] == 4


@pytest.mark.parametrize("scope", ["global", "pair"])
def test_icg_resolution_matches_the_selected_correction_family(scope):
    a = cells(120)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        result = ab.tl.interaction_changed_genes(a, groupby="type", expression_scale="linear",
            pairs=[("A", "B"), ("B", "A")], min_cells=1, n_perms=99, seed=3, fdr_scope=scope)
    valid = result.pval.notna()
    assert valid.any()
    np.testing.assert_allclose(result.loc[valid, "p_min"], 2/(result.loc[valid, "n_valid"]+1))
    info = result.attrs["permutation_resolution"]
    if scope == "global":
        assert info["n_tests"] == valid.sum()
    else:
        assert sum(f["n_tests"] for f in info["families"]) == valid.sum()


def test_gradient_reports_separate_one_and_two_sided_resolution():
    a = cells(60)
    ab.tl.expression_gradient(a, ["g0", "g1"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        ab.tl.gradient_significance(a, ["g0", "g1"], n_perms=39, random_state=4)
    result = a.uns["gradient_significance"]
    for kind, factor in [("magnitude", 2), ("coherence", 1)]:
        valid = result["p_"+kind].notna()
        assert valid.any()
        np.testing.assert_allclose(result.loc[valid, "p_min_"+kind], factor/(result.loc[valid, "n_valid_"+kind]+1))


def cohort():
    rng = np.random.default_rng(19)
    sizes = [12, 60, 24, 90, 18]
    a = ad.AnnData(np.ones((sum(sizes), 2)))
    a.obs["sample"] = np.repeat([f"s{i}" for i in range(5)], sizes)
    a.obs["library_id"] = a.obs["sample"]
    a.obs["condition"] = np.repeat(["control"]*2+["treated"]*3, sizes)
    types = []
    for i, n in enumerate(sizes):
        types.extend((["A", "B"]*(n//2)) if i % 2 else (["A"]*(n//2)+["B"]*(n//2)))
    a.obs["type"] = pd.Categorical(types, categories=["A", "B", "unobserved"])
    a.obsm["spatial"] = np.column_stack([np.concatenate([np.arange(n) for n in sizes]), rng.uniform(0, .01, sum(sizes))])
    ab.tl.spatial_neighbors(a, n_neighbors=2)
    return a


def test_unequal_cohort_sizes_and_missing_types_keep_sample_weighting_and_missing_inference():
    a = cohort()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        result = ab.tl.diff_nhood_enrichment(a, "type", sample_key="sample", condition_key="condition",
            contrast=("treated", "control"), n_perms=99, n_boot=99, seed=9)
    reference = []
    graph = a.obsp["spatial_neighbors_connectivities"]
    for sample in a.obs["sample"].unique():
        mask = np.flatnonzero(a.obs["sample"].to_numpy() == sample)
        local_graph = graph[mask][:, mask]
        labels = a.obs.iloc[mask]["type"].to_numpy()
        degree = np.asarray(local_graph.sum(axis=1)).ravel()
        contacts = np.asarray(local_graph[:, labels == "B"].sum(axis=1)).ravel()
        reference.append(np.mean(contacts[labels == "A"]/degree[labels == "A"])-(labels == "B").sum()/(len(mask)-1))
    pair = result[(result.sender == "A") & (result.receiver == "B")].iloc[0]
    assert pair.effect == pytest.approx(np.mean(reference[2:])-np.mean(reference[:2]))
    assert pair.n_tested == 3 and pair.n_reference == 2
    unavailable = result[result.sender == "unobserved"]
    assert unavailable.pvalue.isna().all() and unavailable.p_min.isna().all()
    assert not unavailable.resolution_limited.any()


def test_permutation_diagnostics_survive_save_and_load(tmp_path):
    a = cells(40); m = md.MuData({"cells": a}); m.uns["default_modality"] = "cells"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PermutationResolutionWarning)
        result = ab.tl.moran_local(m, "g0", n_perms=19, seed=1)
        svg = ab.tl.spatially_variable_genes(m, n_perms=np.int64(19), seed=1)
    path = tmp_path/"resolution.h5mu"
    ab.io.save(m, path)
    restored = ab.io.load(path)
    loaded = restored.uns["moran_local"]["cells"]
    pd.testing.assert_frame_equal(loaded, result)
    assert loaded.attrs["permutation_resolution"] == result.attrs["permutation_resolution"]
    pd.testing.assert_frame_equal(restored.uns["spatially_variable"]["cells"], svg)
