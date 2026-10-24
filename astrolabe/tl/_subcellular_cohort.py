"""Donor-level contrasts of existing subcellular summaries via SciPy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, permutation_test

from astrolabe._statistics import adjust_p
from astrolabe._utils import from_kwarg, with_adata


@with_adata(mirror_uns=from_kwarg("key_added"), spatial_result=True)
def diff_subcellular(adata, value_keys, *, donor_key, condition_key, contrast,
                      cell_type_key=None, paired=False, n_resamples=9999, seed=0,
                      key_added="diff_subcellular"):
    """Compare donor means of prespecified per-cell localization summaries.

    Uses SciPy's two-sided donor-label permutation test (within-pair label swaps
    when paired), and BCa donor bootstrap intervals. Independent labels must be
    exchangeable under the null; paired swaps require within-pair exchangeability.
    There is no covariate adjustment or cell-level spatial model here. Technical
    sections are pooled within donor/condition, and donors receive equal weight.
    Nonfinite cell summaries are omitted and counted. Incomplete paired donors
    are excluded separately per field/type and reported; their cells are included
    in ``n_cells_excluded``, not ``n_cells_used``. Supply raw localization
    indices, not cell-level p-values or cohort-fitted shrinkage scores.
    """
    keys = [value_keys] if isinstance(value_keys, str) else list(value_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Select distinct prespecified value_keys")
    if len(contrast) != 2 or contrast[0] == contrast[1]:
        raise ValueError("contrast requires distinct tested/reference levels")
    if not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    columns = [donor_key, condition_key] + ([] if cell_type_key is None else [cell_type_key])
    if adata.obs[columns].isna().any().any():
        raise ValueError("Donor, condition and cell-type labels must be complete")
    frame = adata.obs.loc[adata.obs[condition_key].isin(contrast), list(dict.fromkeys(columns + keys))].copy()
    if frame.empty:
        raise ValueError("No cells in the requested contrast")
    if not paired and (frame.groupby(donor_key, observed=True)[condition_key].nunique() > 1).any():
        raise ValueError("Unpaired inference requires independent donors in each condition")
    groups = [("all", frame)] if cell_type_key is None else frame.groupby(cell_type_key, observed=True, sort=False)
    rng = np.random.default_rng(seed)
    rows = []
    def mean_difference(first, second, axis=-1):
        return np.mean(first, axis=axis)-np.mean(second, axis=axis)
    for group, cells in groups:
        for key in keys:
            values = cells[key].to_numpy(dtype=float)
            valid = np.isfinite(values)
            means = cells.loc[valid].groupby([donor_key, condition_key], observed=True)[key].mean().unstack(condition_key)
            means = means.reindex(index=cells[donor_key].unique(), columns=list(contrast))
            excluded_pairs = 0
            used = valid.copy()
            if paired:
                complete = means.notna().all(axis=1)
                excluded_pairs = int((~complete).sum())
                means = means.loc[complete]
                used &= cells[donor_key].isin(means.index).to_numpy()
            first, second = (means[level].dropna().to_numpy() for level in contrast)
            row = {"cell_type": group, "field": key, "n_tested": len(first), "n_reference": len(second),
                   "n_cells_used": int(used.sum()), "n_cells_excluded": int((~used).sum()),
                   "n_incomplete_pairs": excluded_pairs, "effect": np.nan, "pvalue": np.nan,
                   "ci_low": np.nan, "ci_high": np.nan, "status": "insufficient_donors"}
            if min(len(first), len(second)) >= 3:
                test = permutation_test((first, second), mean_difference,
                    permutation_type="samples" if paired else "independent", alternative="two-sided",
                    n_resamples=n_resamples, batch=256, rng=rng)
                row.update(effect=float(test.statistic), pvalue=float(test.pvalue), status="tested")
                variation = np.ptp(first-second) if paired else max(np.ptp(first), np.ptp(second))
                if variation > 0:
                    interval = bootstrap((first, second), mean_difference, paired=paired,
                        method="BCa", n_resamples=n_resamples, batch=256, rng=rng).confidence_interval
                    row.update(ci_low=float(interval.low), ci_high=float(interval.high))
                    if not np.isfinite([interval.low, interval.high]).all():
                        row["status"] = "bootstrap_interval_unavailable"
                else:
                    row["status"] = "bootstrap_interval_unavailable"
            rows.append(row)
    result = pd.DataFrame(rows)
    result["padj"] = adjust_p(result.pvalue.to_numpy())
    result.attrs.update(inference_unit="donor", method="SciPy permutation_test and BCa bootstrap",
                        paired=paired, contrast=list(contrast), n_resamples=n_resamples, seed=seed,
                        family="all requested localization fields and cell types")
    adata.uns[key_added] = result
    return result
