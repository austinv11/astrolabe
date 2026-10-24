import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import permutation_test

import astrolabe as ab


@pytest.mark.parametrize("paired", [False, True])
def test_localization_uses_donor_means_and_matches_exact_scipy(paired):
    first, second = np.array([.1, .4, .7, .6]), np.array([.2, .3, .4, .5])
    rows = []
    for condition, values in (("case", first), ("control", second)):
        for donor, value in enumerate(values):
            rows.extend({"donor": str(donor) if paired else f"{condition}{donor}",
                         "condition": condition, "rdi": value} for _ in range(2+donor))
    a = ad.AnnData(np.ones((len(rows), 1)), obs=pd.DataFrame(rows, index=list(map(str, range(len(rows))))))
    def run(data):
        return ab.tl.diff_subcellular(data, "rdi", donor_key="donor", condition_key="condition",
                                     contrast=("case", "control"), paired=paired, n_resamples=199, seed=3)
    result = run(a)
    reference = permutation_test((first, second), lambda x, y: x.mean()-y.mean(),
        permutation_type="samples" if paired else "independent", n_resamples=np.inf)
    assert result.pvalue.iloc[0] == pytest.approx(reference.pvalue)
    assert result.effect.iloc[0] == pytest.approx(reference.statistic)
    repeated = run(ad.concat([a, a], index_unique="-"))
    pd.testing.assert_frame_equal(result.drop(columns="n_cells_used"), repeated.drop(columns="n_cells_used"))
    assert repeated.n_cells_used.iloc[0] == 2 * result.n_cells_used.iloc[0]


def test_paired_counts_exclude_incomplete_and_all_missing_donors():
    rows = [
        {"donor": str(donor), "condition": condition, "rdi": value}
        for donor in range(3)
        for condition, value in (("case", .2 + donor * .1), ("control", .1))
    ]
    rows += [{"donor": "incomplete", "condition": "case", "rdi": .9}]
    rows += [{"donor": "missing", "condition": condition, "rdi": np.nan}
             for condition in ("case", "control")]
    a = ad.AnnData(np.ones((len(rows), 1)), obs=pd.DataFrame(rows, index=list(map(str, range(len(rows))))))
    result = ab.tl.diff_subcellular(a, "rdi", donor_key="donor", condition_key="condition",
        contrast=("case", "control"), paired=True, n_resamples=99).iloc[0]
    assert result.n_tested == result.n_reference == 3
    assert result.n_incomplete_pairs == 2
    assert result.n_cells_used == 6
    assert result.n_cells_excluded == 3
    assert result.effect == pytest.approx(.2)
