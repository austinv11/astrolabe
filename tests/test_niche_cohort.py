import anndata as ad
import numpy as np
import pandas as pd
import pytest

import astrolabe as ab


def cohort(paired=False):
    rows = []
    rng = np.random.default_rng(3)
    for condition in ("control", "case"):
        for donor in range(8):
            n = 25+donor*5
            k = rng.binomial(n, .3)
            rows.extend({"sample": f"{condition}{donor}", "donor": str(donor) if paired else f"{condition}{donor}",
                         "condition": condition, "niches": "A" if i < k else "B", "age": 30+donor}
                        for i in range(n))
    return ad.AnnData(np.ones((len(rows), 1)), obs=pd.DataFrame(rows, index=list(map(str, range(len(rows))))))


@pytest.mark.parametrize("paired", [False, True])
def test_niche_contrast_counts_donors_and_is_invariant_to_cell_duplication(paired):
    a = cohort(paired)
    tables = []
    for obj in (a, ad.concat([a, a], index_unique="-")):
        tables.append(ab.tl.niche_abundance(obj, sample_key="sample", donor_key="donor", condition_key="condition", covariates=["age"]))
    first, repeated = [ab.tl.diff_niche_abundance(table, contrast=("case", "control"), paired=paired, covariates=["age"]) for table in tables]
    pd.testing.assert_frame_equal(first, repeated)
    assert (first.n_donors_tested == 8).all() and (first.n_donors_reference == 8).all()
    assert first.pvalue.between(0, 1).all()


def test_absent_niche_has_zero_count_and_valid_exposure():
    a = cohort()
    a.obs["niches"] = pd.Categorical(a.obs.niches, categories=["A", "B", "absent"])
    table = ab.tl.niche_abundance(a, sample_key="sample", donor_key="donor", condition_key="condition")
    assert (table.loc[table.niche == "absent", "count"] == 0).all()
    assert (table.exposure > 0).all()


def test_frozen_model_maps_without_refitting_training():
    a = ad.AnnData(np.ones((8, 1)))
    a.obsm["spatial"] = np.array([[i, 0] for i in range(8)], float)
    a.obs["type"] = pd.Categorical(["A"]*4+["B"]*4)
    a.obs["niches"] = pd.Categorical(["nA"]*4+["nB"]*4)
    ab.tl.spatial_neighbors(a, radius=1.1)
    ab.tl.niche_composition(a, "type")
    a.uns["niches"] = {"profile_key": "niche_composition"}
    model = ab.tl.fit_niche_model(a, quantile=1)
    centroids = model["centroids"].copy()
    heldout = a.copy()
    assigned = ab.tl.map_niches(heldout, model)
    assert assigned.iloc[[0, 1, 2, 5, 6, 7]].notna().all()
    assert assigned.iloc[[3, 4]].isna().all()  # Identical boundary profiles are ambiguous.
    assert list(assigned.iloc[:3].astype(str)) == ["nA"]*3
    assert list(assigned.iloc[-3:].astype(str)) == ["nB"]*3
    np.testing.assert_array_equal(model["centroids"], centroids)


def test_niche_state_pools_sections_without_extra_replicates():
    a = cohort(paired=True)
    a.layers["counts"] = np.ones(a.shape, dtype=int)
    a.obs["section"] = np.arange(a.n_obs) % 3
    result = ab.tl.niche_pseudobulk(a, donor_key="donor", condition_key="condition", min_cells=1)
    assert result.obs.groupby(["donor", "condition", "niches"], observed=True).size().max() == 1
    assert result.X.sum() == a.layers["counts"].sum()
    assert result.obs.donor.nunique() == 8


def test_abundance_matches_independent_statsmodels_design():
    from statsmodels.regression.linear_model import OLS
    a = cohort()
    table = ab.tl.niche_abundance(a, sample_key="sample", donor_key="donor", condition_key="condition", covariates=["age"])
    actual = ab.tl.diff_niche_abundance(table, contrast=("case", "control"), covariates=["age"]).set_index("niche")
    selected = table[table.niche == "A"]
    design = np.column_stack([np.ones(len(selected)), selected.condition == "case", selected.age])
    expected = OLS(selected.abundance.to_numpy(), design).fit(cov_type="HC3", use_t=True)
    np.testing.assert_allclose(actual.loc["A", ["effect", "se", "pvalue"]].to_numpy(dtype=float),
                               [expected.params[1], expected.bse[1], expected.pvalues[1]])
    broken = table.drop(table.index[0])
    with pytest.raises(ValueError, match="explicit count"):
        ab.tl.diff_niche_abundance(broken, contrast=("case", "control"))
