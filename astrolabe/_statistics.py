"""Finite Monte Carlo inference shared by spatial analyses."""
from __future__ import annotations

import numpy as np
import warnings
from scipy.stats import false_discovery_control


class PermutationResolutionWarning(UserWarning):
    """A Monte Carlo floor cannot resolve an isolated discovery at rank one."""


def permutation_resolution(p, n_valid, *, alternative="two-sided", method="fdr_bh",
                           alpha=.05, label="permutation test", warn=True, family_size=None):
    """Return per-test floors, rank-one resolution flags and family metadata.

    BH can still reject a collection of small p-values when rank one is
    unreachable. This diagnoses sparse-discovery resolution, not universal
    lack of power. Untested/analytic entries have NaN floors and False flags.
    Floors depend on valid draws, not attempted draws. Equality is insufficient
    for the strict ``p_adj < alpha`` convention used by spatial map labels.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be greater, less, or two-sided")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    values, counts = np.broadcast_arrays(np.asarray(p, float), np.asarray(n_valid, float))
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("valid permutation counts must be finite nonnegative integers")
    tested = np.isfinite(values)
    empirical = tested & (counts > 0)
    factor = 2 if alternative == "two-sided" else 1
    floors = np.where(empirical, np.minimum(1., factor/(counts+1)), np.nan)
    m_tested = int(tested.sum())
    m = _family_size(family_size, m_tested)
    harmonic = float(np.sum(1./np.arange(1, m+1))) if method == "fdr_by" else 1.
    if method is None:
        threshold = alpha
    elif method in ("fdr_bh", "fdr_by", "bonferroni", "holm"):
        threshold = alpha/(max(1, m)*harmonic)
    else:
        threshold = None
    limited = np.zeros(values.shape, dtype=bool) if threshold is None else empirical & (floors >= threshold)
    summary = dict(n_tests=m, n_tested=m_tested, n_empirical_tests=int(empirical.sum()), alpha=float(alpha),
                   method=method, alternative=alternative, rank1_threshold=threshold,
                   resolution_assessed=threshold is not None,
                   min_p_min=float(floors[empirical].min()) if empirical.any() else None,
                   max_p_min=float(floors[empirical].max()) if empirical.any() else None,
                   n_resolution_limited=int(limited.sum()),
                   required_valid_permutations_rank1=int(np.floor(factor/threshold)) if threshold else None,
                   interpretation="rank-one Monte Carlo resolution; BH/BY may still reject multiple discoveries together; ties and few exchangeable units may impose stronger limits")
    if warn and limited.any():
        warnings.warn(
            f"{label}: {int(limited.sum())}/{m} tests have a permutation p-value floor "
            f"above or equal to the rank-one threshold {threshold:.3g} "
            f"(floors {summary['min_p_min']:.3g} to {summary['max_p_min']:.3g}). "
            f"At least {summary['required_valid_permutations_rank1']} valid permutations "
            "are needed for an isolated discovery. BH/BY may still reject multiple "
            "discoveries together; this is a resolution limit, not evidence of no effect.",
            PermutationResolutionWarning, stacklevel=3,
        )
    return floors, limited, summary


def annotate_permutation_resolution(frame, p_column, n_column, *, alternative="two-sided",
                                    method="fdr_bh", alpha=.05, label="permutation test",
                                    suffix="", warn=True):
    """Attach diagnostics to a result table without changing inference."""
    floors, limited, summary = permutation_resolution(
        frame[p_column].to_numpy(), frame[n_column].to_numpy(), alternative=alternative,
        method=method, alpha=alpha, label=label, warn=warn,
    )
    frame["p_min"+suffix] = floors
    frame["resolution_limited"+suffix] = limited
    frame.attrs["permutation_resolution"+suffix] = summary
    return frame


def permutation_p(observed, simulations, *, alternative="two-sided", min_valid=1):
    """Return (p, n_valid); the last simulation axis indexes permutations.

    Equal tails include ties with relative tolerance 100 machine eps. NaNs
    are untestable, whereas infinite test statistics are legitimate values.
    The plus-one correction is applied to each tail before doubling.
    """
    obs = np.asarray(observed, dtype=float)
    sims = np.asarray(simulations, dtype=float)
    if sims.shape[:-1] != obs.shape:
        raise ValueError("simulations must have observed.shape + (n_perms,)")
    valid = ~np.isnan(sims)
    n = valid.sum(axis=-1)
    tol = np.where(np.isfinite(obs), 100 * np.finfo(float).eps * np.abs(obs), 0)
    ge = (valid & (sims >= (obs - tol)[..., None])).sum(axis=-1)
    le = (valid & (sims <= (obs + tol)[..., None])).sum(axis=-1)
    if alternative == "greater":
        p = (ge + 1.) / (n + 1.)
    elif alternative == "less":
        p = (le + 1.) / (n + 1.)
    elif alternative == "two-sided":
        p = np.minimum(1., 2 * (np.minimum(ge, le) + 1.) / (n + 1.))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return np.where((n >= min_valid) & ~np.isnan(obs), p, np.nan), n


def _family_size(size, n_tested):
    if size is None:
        return n_tested
    if not isinstance(size, (int, np.integer)) or size < n_tested:
        raise ValueError("family_size must be an integer at least as large as the tested family")
    return int(size)


def adjust_p(p, method="fdr_bh", *, family_size=None):
    """Adjust all finite p-values as one family, retaining untested entries."""
    values = np.asarray(p, dtype=float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    size = _family_size(family_size, int(valid.sum()))
    if valid.any():
        # Retain screened hypotheses at p=1 for multiplicity; their public
        # entries remain NaN. Screening based on observed labels cannot
        # silently reduce the originally requested hypothesis family.
        family = np.r_[values[valid], np.ones(size-int(valid.sum()))]
        if method in ("fdr_bh", "fdr_by"):
            result[valid] = false_discovery_control(family, method=method[4:])[:valid.sum()]
        else:
            from statsmodels.stats.multitest import multipletests
            result[valid] = multipletests(family, method=method)[1][:valid.sum()]
    return result
