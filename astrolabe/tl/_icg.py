"""
Interaction Changed Genes (ICGs).

For every ordered cell-type pair (receiver A, neighbor B), split A-cells
into "contacted" (≥1 spatial neighbor of type B) vs "non-contacted" and
run a two-group DE test across all genes simultaneously, with a
label-permutation null as the default significance path.

Reference
---------
Dries, R., Zhu, Q., Dong, R. et al. *Giotto: a toolbox for integrative
analysis and visualization of spatial expression data.* Genome Biology 22,
78 (2021). doi:10.1186/s13059-021-02286-0 — the ``findInteractionChangedGenes``
method that originated this analysis, including its label-permutation null.
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._utils import _spawn_rngs, from_kwarg, with_adata
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
    per-gene-column iteration cost the ranking already pays, not a new
    bottleneck.
    """
    n_genes = combined.shape[1]
    corr = np.empty(n_genes, dtype=np.float64)
    for j in range(n_genes):
        _, counts = np.unique(combined[:, j], return_counts=True)
        corr[j] = np.sum(counts.astype(np.float64) ** 3 - counts)
    return corr


def _wilcoxon_vectorized(
    X_a: np.ndarray, X_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mann-Whitney U + normal approximation for every gene column.

    Returns ``(U1, z, pvals)`` — *z* is the signed, tie-corrected,
    continuity-corrected standardized statistic. It is what the permutation
    null compares across permutations: unlike raw U, whose null mean/scale
    depend on the two group sizes (which change under label permutation),
    z is comparable across permutations with different splits.
    """
    from scipy.stats import norm, rankdata

    n1, n2 = X_a.shape[0], X_b.shape[0]
    n = n1 + n2
    combined = np.vstack([X_a, X_b])  # (n, n_genes)

    # rank each gene column independently (handles ties via average)
    ranked = rankdata(combined, axis=0)

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
    diff = U1 - mu
    shrunk = np.maximum(np.abs(diff) - 0.5, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, np.sign(diff) * shrunk / sigma, 0.0)
    pvals = np.where(sigma > 0, np.minimum(2.0 * norm.sf(np.abs(z)), 1.0), 1.0)
    return U1, z, pvals


def _ttest_vectorized(
    X_a: np.ndarray, X_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch's t-test for every gene column.

    Returns ``(t, t, pvals)`` — the t statistic doubles as the standardized
    statistic compared across permutations (its dof varies mildly with the
    permuted split sizes, an accepted approximation).
    """
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
    return t_stats, t_stats, pvals


def _dense_rows(X, rows: np.ndarray) -> np.ndarray:
    """Densify only the requested row subset (X stays sparse otherwise)."""
    sub = X[rows]
    if sp.issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float64)


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
    skip_self_pairs: bool = True,
    pairs: "list[tuple[str, str]] | None" = None,
    n_perms: int = 200,
    seed: int = 0,
    n_jobs: int = 1,
    key_added: str = "icg",
) -> pd.DataFrame:
    """Test for genes whose expression in cell type A changes when neighboring cell type B.

    For every ordered pair (receiver A, neighbor B), cells of type A are
    split into "contacted" (≥1 spatial neighbor of type B) and
    "non-contacted" groups; a two-group DE test is run across all genes at
    once.  Results are stored in ``adata.uns[key_added]`` and returned, and
    mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    This is the analysis introduced by Giotto's
    ``findInteractionChangedGenes`` (Dries et al. 2021, Genome Biology).

    **Significance** (default: permutation null).  Cells in tissue are not
    exchangeable — expression is spatially autocorrelated, and the
    contacted/non-contacted split is itself spatially structured — so the
    asymptotic Mann-Whitney/Welch p-value's independence assumption is
    violated and that p-value is **anti-conservative**.  By default
    (``n_perms > 0``), significance therefore comes from a
    spatial-structure-preserving permutation null: the *groupby* labels are
    shuffled over cells while positions, the spatial graph, and each cell's
    expression stay fixed; the contact/non-contact split is recomputed per
    pair under each relabeling; and each gene's observed standardized
    statistic is compared against its own empirical null,
    ``p = (1 + #{|z_perm| ≥ |z_obs|}) / (1 + n_valid_perms)`` (permutations
    where a pair's split fails *min_cells* are excluded from that pair's
    null).  The asymptotic p-value is still reported as ``p_asymptotic``
    for reference/ranking, with the caveat above.

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
    skip_self_pairs
        Skip A-vs-A pairs (default ``True``). For a self pair, "contacted"
        means "has a same-type neighbor", so the split is a proxy for local
        density/isolation rather than a cell-cell interaction: any gene
        whose expression covaries with cell density (e.g. hypoxia, cell
        cycle, segmentation-quality artifacts) shows up as an "interaction"
        change. Pass ``False`` to test self pairs anyway — interpret them as
        density-associated, not interaction-specific, signals.
    pairs
        Restrict testing to these ``(receiver, neighbor)`` category pairs.
        ``None`` (default) tests every ordered pair. With the permutation
        null, computation scales with the number of pairs × ``n_perms`` —
        restrict to the pairs of interest when only a few matter.
    n_perms
        Number of label permutations for the empirical null (default 200 →
        minimum attainable p = 1/201). ``0`` skips the permutation null
        entirely and falls back to the asymptotic p-value as ``pval``
        (anti-conservative under spatial autocorrelation — see above).
    seed
        Seed for the permutation RNG (reproducible for a given seed
        regardless of ``n_jobs``).
    n_jobs
        Parallel workers for the permutation loop
        (:class:`joblib.Parallel`, thread-backed). ``1`` = sequential,
        ``-1`` = all cores.
    key_added
        Key written to ``adata.uns``.

    Returns
    -------
    pd.DataFrame
        Tidy frame sorted by ``(cell_type, neighbor_type, pval)`` with columns
        ``cell_type``, ``neighbor_type``, ``gene``, ``statistic``, ``pval``
        (permutation p when ``n_perms > 0``, else asymptotic),
        ``p_asymptotic``, ``pval_adj`` (BH on ``pval``; scope set by
        *fdr_scope*), ``mean_contacted``, ``mean_non_contacted``, ``log2fc``,
        ``n_contacted``, ``n_non_contacted``.

    Examples
    --------
    >>> df = ab.tl.interaction_changed_genes(mdata, groupby="leiden")
    >>> df[df.pval_adj < 0.05].head(20)

    References
    ----------
    Dries, R., Zhu, Q., Dong, R. et al. *Giotto: a toolbox for integrative
    analysis and visualization of spatial expression data.* Genome Biology
    22, 78 (2021). doi:10.1186/s13059-021-02286-0
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first"
        )
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")
    if n_perms < 0:
        raise ValueError(f"n_perms must be >= 0; got {n_perms}")

    conn = adata.obsp[conn_key].tocsr()
    # Shared with the rest of tl/_neighborhood.py's category-resolution
    # convention (declared .cat.categories order) instead of a separate
    # ad-hoc np.unique(labels) (alphabetical-by-value, and loses any
    # meaningful custom category order, e.g. Leiden's '0','1',...,'10').
    codes, categories = _category_codes(adata, groupby)
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    # Keep X sparse; only the per-pair row subsets are densified.
    X = adata.layers[layer] if layer is not None else adata.X
    if not sp.issparse(X):
        X = np.asarray(X, dtype=np.float64)

    type_idx = {ct: np.where(codes == i)[0] for i, ct in enumerate(categories)}
    test_fn = _wilcoxon_vectorized if method == "wilcoxon" else _ttest_vectorized

    def _split_pair(codes_arr: np.ndarray, code_a: int, code_b: int):
        """(contacted_rows, non_contacted_rows) global indices for pair (a, b)."""
        idx_a = np.where(codes_arr == code_a)[0]
        idx_b = np.where(codes_arr == code_b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            return None
        neighbor_counts = np.asarray(conn[idx_a][:, idx_b].sum(axis=1)).ravel()
        cont = idx_a[neighbor_counts > 0]
        non = idx_a[neighbor_counts == 0]
        if len(cont) < min_cells or len(non) < min_cells:
            return None
        return cont, non

    code_of = {ct: i for i, ct in enumerate(categories)}

    pair_filter = None
    if pairs is not None:
        pair_filter = {(str(a), str(b)) for a, b in pairs}
        unknown = {c for p in pair_filter for c in p} - {str(c) for c in categories}
        if unknown:
            raise ValueError(
                f"pairs references unknown {groupby} categories: {sorted(unknown)}"
            )

    parts: list[pd.DataFrame] = []
    tested_pairs: list[tuple] = []  # (ct_a, ct_b) in emission order
    obs_z: list[np.ndarray] = []  # per tested pair: (n_genes,) standardized stat

    for ct_a in categories:
        if len(type_idx[ct_a]) == 0:
            continue
        for ct_b in categories:
            if skip_self_pairs and ct_a == ct_b:
                continue
            if pair_filter is not None and (str(ct_a), str(ct_b)) not in pair_filter:
                continue
            if len(type_idx[ct_b]) == 0:
                continue

            split = _split_pair(codes, code_of[ct_a], code_of[ct_b])
            if split is None:
                continue
            cont, non = split

            X_cont = _dense_rows(X, cont)
            X_non = _dense_rows(X, non)

            stats, z, p_asym = test_fn(X_cont, X_non)

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

            tested_pairs.append((ct_a, ct_b))
            obs_z.append(np.abs(z))

            parts.append(
                pd.DataFrame(
                    {
                        "cell_type": ct_a,
                        "neighbor_type": ct_b,
                        "gene": gene_names,
                        "statistic": stats,
                        "p_asymptotic": p_asym,
                        "mean_contacted": mean_c,
                        "mean_non_contacted": mean_nc,
                        "log2fc": log2fc,
                        "n_contacted": len(cont),
                        "n_non_contacted": len(non),
                    }
                )
            )
            log.info(
                "ICG [%s | nb=%s]: %d vs %d cells",
                ct_a, ct_b, len(cont), len(non),
            )

    # ── Permutation null (default significance path) ──────────────────────────
    n_pairs = len(tested_pairs)
    if n_pairs > 0 and n_perms > 0:
        pair_codes = [(code_of[a], code_of[b]) for a, b in tested_pairs]

        def _one_perm(rng: np.random.Generator) -> np.ndarray:
            perm_codes = rng.permutation(codes)
            out = np.full((n_pairs, n_genes), np.nan, dtype=np.float64)
            for pi, (ca, cb) in enumerate(pair_codes):
                split_p = _split_pair(perm_codes, ca, cb)
                if split_p is None:
                    continue
                _, z_p, _ = test_fn(
                    _dense_rows(X, split_p[0]), _dense_rows(X, split_p[1])
                )
                out[pi] = np.abs(z_p)
            return out

        log.info(
            "interaction_changed_genes: permutation null (%d perms × %d pairs)",
            n_perms, n_pairs,
        )
        child_rngs = _spawn_rngs(seed, n_perms)
        perm_z_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_perm)(child_rng) for child_rng in child_rngs
        )
        perm_z = np.stack(perm_z_list)  # (n_perms, n_pairs, n_genes)
        obs_abs = np.stack(obs_z)  # (n_pairs, n_genes)

        valid = np.isfinite(perm_z)
        n_valid = valid.sum(axis=0)  # (n_pairs, n_genes)
        n_exceed = np.nansum(perm_z >= obs_abs[None, :, :], axis=0)
        # (k+1)/(n+1) -- an exact 0 isn't a valid Monte Carlo p estimate.
        pval_perm = (n_exceed + 1.0) / (n_valid + 1.0)

        for pi, part in enumerate(parts):
            part["pval"] = pval_perm[pi]
    else:
        if n_pairs > 0:
            log.warning(
                "interaction_changed_genes: n_perms=0 -- 'pval' falls back to "
                "the asymptotic p-value, which is anti-conservative under "
                "spatial autocorrelation (cells are not exchangeable)."
            )
        for part in parts:
            part["pval"] = part["p_asymptotic"]

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
        len(df), n_pairs,
    )
    return df
