"""Independent reproductions for the eight approved package-review fixes."""
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp

import astrolabe as ab
from astrolabe._utils import ImageSpec

pytestmark = pytest.mark.filterwarnings("ignore::astrolabe._statistics.PermutationResolutionWarning")


def spatial_data(n=24, libraries=False):
    rng = np.random.default_rng(8)
    a = ad.AnnData(rng.uniform(1, 4, (n, 2)))
    a.var_names = ["g0", "g1"]
    a.obsm["spatial"] = rng.uniform(0, 10, (n, 2))
    a.obs["type"] = pd.Categorical(["A", "B"] * (n // 2))
    if libraries:
        a.obs["slide"] = np.repeat(["one", "two"], n // 2)
    ab.tl.spatial_neighbors(a, n_neighbors=3, library_key="slide" if libraries else None)
    return a


def global_test(a, method, **kwargs):
    args = ("g0", "g1") if method == "moran_bv" else ("g0",)
    return getattr(ab.tl, method)(a, *args, **kwargs)


@pytest.mark.parametrize("null", ["csr", "smooth"])
@pytest.mark.parametrize("constant", [True, False])
def test_gradient_null_uses_only_selected_group(null, constant):
    grid = np.array([(x, y) for x in range(7) for y in range(7)], float)
    values = np.ones(49) if constant else 2 + grid[:, 0] + .1 * grid[:, 1]**2
    a = ad.AnnData(np.r_[values, np.ones(49)][:, None])
    a.var_names = ["g"]
    a.obs["type"] = ["A"] * 49 + ["B"] * 49
    a.obsm["spatial"] = np.vstack([grid, grid + [20, 0]])
    ab.tl.spatial_neighbors(a, radius=1.5)
    variants = [a.copy(), a.copy(), a[:49].copy()]
    ab.tl.spatial_neighbors(variants[2], radius=1.5)
    variants[1].X[49:] = 1000
    results = []
    for i, variant in enumerate(variants):
        ab.tl.expression_gradient(variant, "g", groupby="type", groups=["A"], n_diffusion_steps=1)
        ab.tl.gradient_significance(variant, "g", null=null, n_perms=19, random_state=4,
                                    n_jobs=2 if i == 1 else 1)
        results.append(variant.uns["gradient_significance"])
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0], result)
    if constant:
        assert results[0].iloc[0].p_magnitude == 1


@pytest.mark.parametrize("method", ["moran", "geary_c", "moran_bv"])
@pytest.mark.parametrize("case", ["constant", "edgeless"])
def test_undefined_global_statistics_are_not_tested(method, case):
    a = spatial_data()
    if case == "constant":
        a.X[:, 0] = 0
    else:
        a.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix((a.n_obs, a.n_obs))
    result = global_test(a, method, n_perms=39, seed=2)
    assert result.p_sim.isna().all() and result.p_norm.isna().all()
    assert result.p_min.isna().all() and (result.n_valid == 0).all()
    assert (result.status == "not_tested").all()


@pytest.mark.parametrize("method", ["moran", "geary_c", "moran_bv"])
def test_global_rejects_nonfinite_input(method):
    a = spatial_data()
    a.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        global_test(a, method, n_perms=9)


@pytest.mark.parametrize("method", ["moran", "geary_c"])
def test_global_tied_null_is_not_significant(method):
    a = spatial_data()
    a.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(np.ones((24, 24)) - np.eye(24))
    result = global_test(a, method, n_perms=39, seed=4)
    assert result.iloc[0].p_sim == 1
    assert result.iloc[0].n_valid == 39


@pytest.mark.parametrize("method", ["moran", "geary_c", "moran_bv"])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_global_tail_counts_match_independent_reference(method, alternative):
    import esda
    from libpysal.weights import WSP

    a = spatial_data(libraries=True)
    graph = a.obsp["spatial_neighbors_connectivities"].copy()
    w = WSP(graph).to_W()
    def score(y):
        if method == "moran":
            return esda.Moran(y, w, permutations=0).I
        if method == "geary_c":
            return esda.Geary(y, w, permutations=0).C
        return esda.Moran_BV(a.X[:, 0], y, w, permutations=0).I
    y = a.X[:, 1 if method == "moran_bv" else 0]
    observed = score(y)
    rng = np.random.default_rng(5)
    sims = []
    for _ in range(39):
        shuffled = y.copy()
        for slide in ["one", "two"]:
            indices = np.flatnonzero(a.obs.slide.to_numpy() == slide)
            shuffled[indices] = rng.permutation(shuffled[indices])
        sims.append(score(shuffled))
    sims = np.asarray(sims)
    tol = 100 * np.finfo(float).eps * abs(observed)
    greater = (1 + np.sum(sims >= observed - tol)) / 40
    less = (1 + np.sum(sims <= observed + tol)) / 40
    expected = min(1, 2 * min(greater, less)) if alternative == "two-sided" else greater if alternative == "greater" else less
    state = np.random.get_state()
    row = global_test(a, method, n_perms=39, seed=5, alternative=alternative).iloc[0]
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert row.p_sim == expected
    assert row["C" if method == "geary_c" else "I"] == pytest.approx(observed, rel=1e-6)
    assert row.null_mean == pytest.approx(sims.mean(), rel=1e-6, abs=1e-12)
    assert row.null_std == pytest.approx(sims.std(), rel=1e-6)
    assert np.isnan(row.p_norm)


@pytest.mark.parametrize("method", ["moran", "geary_c", "moran_bv"])
def test_global_library_shifts_are_not_evidence(method):
    a = spatial_data(libraries=True)
    a.X[:] = np.repeat([1., 100.], 12)[:, None]
    row = global_test(a, method, n_perms=39, seed=3).iloc[0]
    assert row.p_sim == 1
    with pytest.raises(ValueError, match="n_perms"):
        global_test(a, method, n_perms=0)
    ab.tl.spatial_neighbors(a, library_key=None)
    pooled = global_test(a, method, n_perms=39, seed=3)
    assert pooled.attrs["n_libraries"] == 1 and pooled.attrs["library_key"] is None


@pytest.mark.parametrize("method", ["moran", "geary_c", "moran_bv"])
def test_global_detects_within_library_signal(method):
    a = spatial_data(n=80, libraries=True)
    grid = np.array([(x, y) for x in range(8) for y in range(5)], float)
    a.obsm["spatial"] = np.vstack([grid, grid])
    a.X[:] = np.tile(grid[:, 0], 2)[:, None]
    ab.tl.spatial_neighbors(a, radius=1.1, library_key="slide")
    row = global_test(a, method, n_perms=199, seed=4).iloc[0]
    assert row.p_sim <= .02


@pytest.mark.parametrize("bad", ["missing_column", "missing_value", "cross_edge"])
@pytest.mark.parametrize("method", ["moran", "nhood"])
def test_exchangeability_labels_and_graph_are_validated(bad, method):
    a = spatial_data(libraries=True)
    if bad == "missing_column":
        del a.obs["slide"]
    elif bad == "missing_value":
        a.obs.loc[a.obs_names[0], "slide"] = None
    else:
        graph = a.obsp["spatial_neighbors_connectivities"].tolil()
        graph[0, 12] = 1
        a.obsp["spatial_neighbors_connectivities"] = graph.tocsr()
    with pytest.raises(ValueError, match="librar|Stale result"):
        if method == "nhood":
            ab.tl.nhood_enrichment(a, "type", n_perms=9)
        else:
            ab.tl.moran(a, "g0", n_perms=9)


def test_neighborhood_pure_libraries_have_no_identifiable_enrichment():
    a = spatial_data(libraries=True)
    a.obs["type"] = np.repeat(["A", "B"], 12)
    ab.tl.nhood_enrichment(a, "type", n_perms=39, seed=0)
    result = a.uns["nhood_enrichment"]
    assert np.isnan(result["zscore"]).all()
    assert (result["status"] == "zero_variance").all()
    assert result["metadata"]["library_key"] == "slide"
    ab.tl.nhood_enrichment(a, "type", n_perms=39, seed=0, library_key=None)
    assert np.isfinite(a.uns["nhood_enrichment"]["zscore"]).all()


def test_neighborhood_matches_independent_restricted_label_counts():
    a = spatial_data(libraries=True)
    a.obs["type"] = ["A"] * 9 + ["B"] * 3 + ["A"] * 3 + ["B"] * 9
    codes = pd.Categorical(a.obs.type).codes.astype(int)
    coo = a.obsp["spatial_neighbors_connectivities"].tocoo()
    count = lambda labels: np.bincount(2 * labels[coo.row] + labels[coo.col], minlength=4).reshape(2, 2)
    sims = []
    for child in np.random.SeedSequence(6).spawn(39):
        rng = np.random.default_rng(child)
        shuffled = codes.copy()
        for indices in [np.arange(12), np.arange(12, 24)]:
            shuffled[indices] = rng.permutation(shuffled[indices])
        sims.append(count(shuffled))
    expected = (count(codes) - np.mean(sims, axis=0)) / np.std(sims, axis=0)
    for workers in [1, 2]:
        ab.tl.nhood_enrichment(a, "type", n_perms=39, seed=6, n_jobs=workers, library_key="slide")
        np.testing.assert_allclose(a.uns["nhood_enrichment"]["zscore"], expected)


def cohort_data():
    a = spatial_data(n=32)
    a.obs["sample"] = np.repeat(["s0", "s1", "s2", "s3"], 8)
    a.obs["library_id"] = a.obs["sample"]
    a.obs["condition"] = np.repeat(["ctrl", "ctrl", "case", "case"], 8)
    a.obs["donor"] = np.repeat(["d0", "d1", "d0", "d1"], 8)
    a.obs["block"] = "all"
    a.obs["type"] = pd.Categorical(a.obs.type, categories=["A", "B", "absent"])
    ab.tl.spatial_neighbors(a, n_neighbors=3)
    return a


def cohort_test(data, **kwargs):
    return ab.tl.diff_nhood_enrichment(data, "type", sample_key="sample", condition_key="condition",
        contrast=("case", "ctrl"), n_perms=19, n_boot=19, **kwargs)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_cohort_missing_type_is_unavailable_in_both_directions(paired, blocked):
    a = cohort_data()
    result = cohort_test(a, paired_key="donor" if paired else None, block_key="block" if blocked else None)
    missing = result[(result.sender == "absent") | (result.receiver == "absent")]
    assert missing[["effect", "ci_low", "ci_high", "pvalue", "p_min"]].isna().all().all()
    assert (missing.n_valid == 0).all()
    assert (missing.status == "insufficient_replicates").all()


@pytest.mark.parametrize("paired", [False, True])
def test_cohort_counts_only_usable_samples_or_complete_pairs(paired):
    a = cohort_data()
    a.obs.loc[a.obs["sample"] == "s0", "type"] = "A"
    result = cohort_test(a, paired_key="donor" if paired else None)
    row = result[(result.sender == "A") & (result.receiver == "B")].iloc[0]
    assert row.n_reference == 1 and row.n_tested == (1 if paired else 2)
    assert np.isnan(row.pvalue) and row.n_valid == 0
    assert row.status == "insufficient_replicates"


def test_cohort_singleton_same_type_is_not_observed_zero():
    a = cohort_data()
    a.obs["type"] = ["B"] + ["A"] * 7 + (["A", "B"] * 4) * 3
    cohort_test(a)
    assert np.isnan(a.uns["diff_nhood_enrichment"]["sample_scores"].loc["s0", ("B", "B")])


@pytest.mark.parametrize("container", ["adata", "mdata"])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("operation", ["filter", "crop", "isolation"])
def test_cell_selection_clears_owned_results_and_preserves_inputs(container, inplace, operation):
    a = cohort_data()
    a.obsm["spatial"][-1] = [1000, 1000]
    ab.tl.spatial_neighbors(a)
    a.obs["total_counts"] = np.r_[0, np.ones(31)]
    a.obs["gradient_user_note"] = "keep"
    a.var["sv_user_note"] = "keep"
    a.obsm["niche_input_embedding"] = np.ones((32, 2))
    a.uns["annotations"] = {"study": "keep"}
    data = md.MuData({"rna": a}) if container == "mdata" else a
    if container == "mdata":
        data.uns["default_modality"] = "rna"
    ab.tl.moran_local(data, "g0", n_perms=9, seed=0, key_added="custom_local")
    ab.tl.spatially_variable_genes(data, n_perms=9, seed=0, key_added="custom_svg")
    cohort_test(data, key_added="custom_cohort")
    ab.tl.ligrec(data, "type", pd.DataFrame({"ligand": ["g0"], "receptor": ["g1"]}),
                 n_perms=9, min_valid_perms=1, seed=0, key_added="custom_lr")
    if operation == "filter":
        returned = ab.pp.filter_cells(data, min_counts=1, inplace=inplace)
        out = data if inplace else returned[0]
    elif operation == "isolation":
        returned = ab.pp.filter_isolated(data, max_neighbor_dist_um=100, inplace=inplace)
        out = data if inplace else returned[0]
    else:
        returned = ab.pp.crop(data, np.array([[-1, -1], [8, -1], [8, 11], [-1, 11]]), inplace=inplace)
        out = data if inplace else returned
    target = out["rna"] if isinstance(out, md.MuData) else out
    assert target.n_obs < 32
    for key in ["custom_local", "custom_svg", "custom_cohort", "custom_lr"]:
        assert key not in target.uns
        if isinstance(out, md.MuData):
            assert key not in out.uns
    assert "spatial_neighbors" not in target.uns and "sv_pval" not in target.var
    assert (target.obs.gradient_user_note == "keep").all()
    assert (target.var.sv_user_note == "keep").all()
    assert "niche_input_embedding" in target.obsm and "annotations" in target.uns
    if not inplace:
        assert "custom_local" in a.uns and a.n_obs == 32


def test_noop_filter_preserves_results_and_other_modality_mirrors():
    a = spatial_data()
    a.obs["total_counts"] = 1
    m = md.MuData({"rna": a, "other": a.copy()})
    for modality in m.mod:
        ab.tl.moran(m, "g0", n_perms=9, modality=modality, key_added="custom")
    ab.pp.filter_cells(m, min_counts=1, modality="rna", inplace=True)
    assert set(m.uns["custom"]) == {"rna", "other"}
    m["rna"].obs.loc[m["rna"].obs_names[0], "total_counts"] = 0
    ab.pp.filter_cells(m, min_counts=1, modality="rna", inplace=True)
    assert set(m.uns["custom"]) == {"other"}


def aliased_sample(marker):
    a = spatial_data()
    a.uns["spatial"] = {"original": {"images": {"image": ImageSpec(
        path=f"{marker}.tif", pixel_size_um=1, dimension_order="YX", full_shape=(8, 8),
        level_shapes=[(8, 8)], level_scales=[(1., 1.)])}}}
    a.uns["transcripts"] = pa.table({"cell_id": a.obs_names, "library_id": ["original"] * a.n_obs})
    m = md.MuData({"rna": a})
    m.uns.update(default_modality="rna", spatial=a.uns["spatial"])
    return m


def test_concat_aliases_preserve_descriptors_transcripts_and_round_trip(tmp_path):
    first, second = aliased_sample("first"), aliased_sample("second")
    combined = ab.io.concat_samples({"first": first, "second": second})
    for obj in [combined, combined["rna"]]:
        assert set(obj.uns["spatial"]) == {"first", "second"}
        for alias in ["first", "second"]:
            assert obj.uns["spatial"][alias]["images"]["image"].path == f"{alias}.tif"
    tx = combined["rna"].uns["transcripts"]
    assert tx.column_names.count("library_id") == 1
    assert set(tx["library_id"].to_pylist()) == {"first", "second"}
    assert set(tx["cell_id"].to_pylist()) == set(combined["rna"].obs_names)
    assert set(first.uns["spatial"]) == {"original"}
    combined["rna"].uns["spatial"]["first"]["images"]["image"].transform_chain.append(np.eye(3))
    assert not first["rna"].uns["spatial"]["original"]["images"]["image"].transform_chain
    ab.io.save(combined, tmp_path / "aliased.h5mu")
    loaded = ab.io.load(tmp_path / "aliased.h5mu")
    assert set(loaded.uns["spatial"]) == {"first", "second"}
    for alias in ["first", "second"]:
        assert loaded["rna"].uns["spatial"][alias]["images"]["image"].path.endswith(f"{alias}.tif")


@pytest.mark.parametrize("bad", ["multiple_spatial", "conflicting_parent", "multiple_labels"])
def test_concat_rejects_ambiguous_aliased_input(bad):
    first, second = aliased_sample("first"), aliased_sample("second")
    if bad == "multiple_spatial":
        first["rna"].uns["spatial"]["extra"] = {}
    elif bad == "conflicting_parent":
        first.uns["spatial"] = {"different": {}}
    else:
        first["rna"].obs["library_id"] = ["one", "two"] * 12
    with pytest.raises(ValueError, match="single library"):
        ab.io.concat_samples({"first": first, "second": second})


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("k", [1, 3])
def test_clq_excludes_cell_identity(duplicate, k):
    a = ad.AnnData(np.ones((4, 1)))
    a.obsm["spatial"] = np.array([[0., 0.], [0. if duplicate else .2, 0.], [1., 0.], [3., 0.]])
    a.obs["type"] = ["A", "B", "A", "B"]
    result = ab.tl.colocation_quotient(a, "type", k=k, min_count=1, n_perms=0)
    # Hand-counted cross-type fractions for this four-cell geometry.
    expected = 1.5 if k == 1 else 1.
    row = result[(result.category_a == "A") & (result.category_b == "B")].iloc[0]
    assert row.clq == pytest.approx(expected)


def test_clq_multiple_coincident_cells_uses_other_cells():
    a = ad.AnnData(np.ones((4, 1)), obsm={"spatial": np.zeros((4, 2))})
    a.obs["type"] = ["A", "A", "B", "B"]
    result = ab.tl.colocation_quotient(a, "type", k=3, min_count=1, n_perms=9, seed=0)
    np.testing.assert_allclose(result.clq, 1)
    np.testing.assert_allclose(result.p_value, 1)
