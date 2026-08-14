"""
Interaction Changed Genes (ICGs).

For every ordered cell-type pair (receiver A, neighbor B), split A-cells
into "contacted" (≥1 spatial neighbor of type B) vs "non-contacted" and
run a two-group DE test across all genes simultaneously.
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._utils import from_kwarg, with_adata
from astrolabe.tl._neighborhood import _category_codes

__all__ = ["interaction_changed_genes"]

log = logging.getLogger(__name__)


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    adjusted = pvals[order] * n / (np.arange(1, n + 1, dtype=np.float64))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _tie_correction(combined: np.ndarray) -> np.ndarray:
    """sum(t_i**3 - t_i) per gene column, t_i = tie-group sizes.

    Standard Mann-Whitney U tie-correction term (Hollander, Wolfe & Chicken,
    *Nonparametric Statistical Methods*; matches ``scipy.stats.mannwhitneyu``'s
    asymptotic-method variance). One column at a time -- same
    per-gene-column iteration cost `_wilcoxon_vectorized` already pays for
    `rankdata`, not a new bottleneck.
    """
    n_genes = combined.shape[1]
    corr = np.empty(n_genes, dtype=np.float64)
    for j in range(n_genes):
        _, counts = np.unique(combined[:, j], return_counts=True)
        corr[j] = np.sum(counts.astype(np.float64) ** 3 - counts)
    return corr


def _wilcoxon_vectorized(
    X_a: np.ndarray, X_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mann-Whitney U + normal approximation for every gene column."""
    from scipy.stats import norm, rankdata

    n1, n2 = X_a.shape[0], X_b.shape[0]
    n = n1 + n2
    combined = np.vstack([X_a, X_b])  # (n, n_genes)

    # rank each gene column independently (handles ties via average)
    ranked = np.apply_along_axis(rankdata, 0, combined)

    U1 = ranked[:n1].sum(axis=0) - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    # Tie-corrected variance -- omitting this (plain n1*n2*(n+1)/12) *over*states
    # the true variance whenever ties are common, which single-cell count data
    # always has (zero-inflation: most cells share the exact value 0 for most
    # genes) -- the correction term is always >= 0 and is only ever
    # subtracted. An overstated variance deflates |z| and produces p-values
    # that are too large -- measured 5-872x too conservative on realistic
    # 75-78%-zero count data relative to scipy.stats.mannwhitneyu's own
    # tie-corrected result, so real effects failed to clear BH significance.
    tie_corr = _tie_correction(combined)
    sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_corr / (n * (n - 1)))
    sigma = np.sqrt(np.maximum(sigma2, 0.0))
    # Continuity correction (matches scipy.stats.mannwhitneyu's
    # use_continuity=True default): shrink |U1-mu| by 0.5 toward 0 before
    # standardizing, the standard adjustment for approximating a discrete
    # statistic's distribution with a continuous normal one.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (np.abs(U1 - mu) - 0.5) / sigma, 0.0)
    pvals = np.where(sigma > 0, np.minimum(2.0 * norm.sf(z), 1.0), 1.0)
    return U1, pvals


def _ttest_vectorized(
    X_a: np.ndarray, X_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Welch's t-test for every gene column."""
    from scipy.stats import t as t_dist

    n1, n2 = X_a.shape[0], X_b.shape[0]
    mean1 = X_a.mean(axis=0)
    mean2 = X_b.mean(axis=0)
    var1 = X_a.var(axis=0, ddof=1)
    var2 = X_b.var(axis=0, ddof=1)

    se2 = var1 / n1 + var2 / n2
    se = np.sqrt(se2)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, (mean1 - mean2) / se, 0.0)
        dof = np.where(
            se > 0,
            se2 ** 2 / ((var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1)),
            1.0,
        )

    pvals = np.where(se > 0, 2.0 * t_dist.sf(np.abs(t_stats), dof), 1.0)
    return t_stats, pvals


@with_adata(mirror_uns=from_kwarg("key_added"))
def interaction_changed_genes(
    adata: ad.AnnData,
    *,
    groupby: str,
    neighbors_key: str = "spatial_neighbors",
    layer: str | None = None,
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    min_cells: int = 10,
    fdr_scope: Literal["pair", "global"] = "pair",
    key_added: str = "icg",
) -> pd.DataFrame:
    """Test for genes whose expression in cell type A changes when neighboring cell type B.

    For every ordered pair (receiver A, neighbor B), cells of type A are
    split into "contacted" (≥1 spatial neighbor of type B) and
    "non-contacted" groups; a two-group DE test is run across all genes at
    once.  Results are stored in ``adata.uns[key_added]`` and returned, and
    mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with spatial
        connectivity in ``obsp`` and cell-type labels in ``obs``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    groupby
        ``obs`` column containing discrete cell-type / cluster labels.
    neighbors_key
        Prefix for the spatial graph in ``obsp``
        (``{neighbors_key}_connectivities``).
    layer
        Expression layer to test.  ``None`` uses ``adata.X``.
    method
        Statistical test: ``'wilcoxon'`` (Mann-Whitney U + normal
        approximation, default) or ``'t-test'`` (Welch).
    min_cells
        Minimum cells required in *both* groups for a pair to be tested.
    fdr_scope
        ``'pair'`` (default) — Benjamini-Hochberg correction applied
        independently within each (cell_type, neighbor_type) pair's genes,
        matching this function's original behavior. ``'global'`` — BH
        applied once across every gene of every tested pair together. The
        two give different, both-defensible answers to "controlled at what
        family": per-pair FDR treats each pair as its own question ("which
        genes change when A meets B specifically"), and doesn't control the
        false-discovery rate of the *union* across all pairs, which
        multiplies with pair count and involves statistically dependent
        tests (the same cells are reused across many pairs' "non-contacted"
        groups). ``'global'`` controls that union-wide rate at the cost of
        real power loss on any individual pair.
    key_added
        Key written to ``adata.uns``.

    Returns
    -------
    pd.DataFrame
        Tidy frame sorted by ``(cell_type, neighbor_type, pval)`` with columns
        ``cell_type``, ``neighbor_type``, ``gene``, ``statistic``, ``pval``,
        ``pval_adj`` (scope set by *fdr_scope*), ``mean_contacted``,
        ``mean_non_contacted``, ``log2fc``, ``n_contacted``, ``n_non_contacted``.

    Examples
    --------
    >>> df = ab.tl.interaction_changed_genes(mdata, groupby="leiden")
    >>> df[df.pval_adj < 0.05].head(20)
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first"
        )
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")

    conn = adata.obsp[conn_key]
    # Shared with the rest of tl/_neighborhood.py's category-resolution
    # convention (declared .cat.categories order) instead of a separate
    # ad-hoc np.unique(labels) (alphabetical-by-value, and loses any
    # meaningful custom category order, e.g. Leiden's '0','1',...,'10').
    codes, categories = _category_codes(adata, groupby)
    gene_names = list(adata.var_names)

    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    type_idx = {ct: np.where(codes == i)[0] for i, ct in enumerate(categories)}
    test_fn = _wilcoxon_vectorized if method == "wilcoxon" else _ttest_vectorized

    parts: list[pd.DataFrame] = []

    for ct_a in categories:
        idx_a = type_idx[ct_a]
        conn_a = conn[idx_a, :]  # sparse (|A|, n_cells)

        for ct_b in categories:
            idx_b = type_idx[ct_b]
            if len(idx_b) == 0:
                continue

            neighbor_counts = np.asarray(conn_a[:, idx_b].sum(axis=1)).ravel()
            local_cont = np.where(neighbor_counts > 0)[0]
            local_non = np.where(neighbor_counts == 0)[0]

            if len(local_cont) < min_cells or len(local_non) < min_cells:
                continue

            X_cont = X[idx_a[local_cont], :]
            X_non = X[idx_a[local_non], :]

            stats, pvals = test_fn(X_cont, X_non)

            # log-scale means (still useful diagnostic columns as-is), but
            # log2fc itself is computed from expm1'd per-cell values *before*
            # averaging -- matching scanpy.tl.rank_genes_groups' convention
            # -- not from a ratio of already-log1p-transformed means. Taking
            # log2 of a ratio of log-scale means is a different, less
            # standard quantity, and the tiny 1e-9 pseudocount (calibrated
            # for linear-scale expm1'd values, which are genuinely 0 for
            # undetected genes) becomes wildly distorting when applied
            # directly to small log1p-scale means instead: a barely-nonzero
            # log1p mean of 0.001 against an exact-zero mean already gives
            # log2((0.001+1e-9)/1e-9) ~ 20, overstating a near-undetectable
            # difference as a ~million-fold change.
            mean_c = X_cont.mean(axis=0)
            mean_nc = X_non.mean(axis=0)
            expm1_c = np.expm1(X_cont).mean(axis=0)
            expm1_nc = np.expm1(X_non).mean(axis=0)
            log2fc = np.log2((expm1_c + 1e-9) / (expm1_nc + 1e-9))
            # Genes absent in both groups are not informative; mark them NaN
            # so they sort to the bottom rather than appearing as "no change".
            log2fc = np.where((mean_c == 0) & (mean_nc == 0), np.nan, log2fc)

            parts.append(
                pd.DataFrame(
                    {
                        "cell_type": ct_a,
                        "neighbor_type": ct_b,
                        "gene": gene_names,
                        "statistic": stats,
                        "pval": pvals,
                        "mean_contacted": mean_c,
                        "mean_non_contacted": mean_nc,
                        "log2fc": log2fc,
                        "n_contacted": len(local_cont),
                        "n_non_contacted": len(local_non),
                    }
                )
            )
            log.info(
                "ICG [%s | nb=%s]: %d vs %d cells",
                ct_a, ct_b, len(local_cont), len(local_non),
            )

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        if fdr_scope == "global":
            df["pval_adj"] = _bh_correct(df["pval"].to_numpy())
        else:
            df["pval_adj"] = df.groupby(
                ["cell_type", "neighbor_type"], observed=True
            )["pval"].transform(lambda p: _bh_correct(p.to_numpy()))
        df = df.sort_values(["cell_type", "neighbor_type", "pval"]).reset_index(drop=True)

    adata.uns[key_added] = df
    log.info(
        "interaction_changed_genes: %d rows across %d tested pairs",
        len(df), len(parts),
    )
    return df
