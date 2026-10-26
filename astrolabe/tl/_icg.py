"""
Interaction Changed Genes (ICGs).

For every ordered cell-type pair (receiver A, neighbor B), split A-cells
into "contacted" (≥1 spatial neighbor of type B) vs "non-contacted" and
run a two-group DE test across all genes simultaneously, with a
within-receiver-type contact-flag permutation null.

Reference
---------
Dries, R., Zhu, Q., Dong, R. et al. *Giotto: a toolbox for integrative
analysis and visualization of spatial expression data.* Genome Biology 22,
78 (2021). doi:10.1186/s13059-021-02286-0 — the ``findInteractionChangedGenes``
method that motivated this analysis. Astrolabe's within-type contact-flag
association test is its own estimand, not an exact implementation of Giotto.
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._statistics import (permutation_p, adjust_p, permutation_resolution,
                                   annotate_permutation_resolution)
from astrolabe._utils import _spawn_rngs, from_kwarg, with_adata
from astrolabe.tl._neighborhood import _category_codes

__all__ = ["interaction_changed_genes"]

log = logging.getLogger(__name__)


def _bh_correct(pvals):
    return adjust_p(pvals)


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
        t_stats = np.where(se > 0, (mean1 - mean2) / se,
                           np.where((se == 0) & (mean1 != mean2), np.sign(mean1-mean2)*np.inf, np.nan))
        dof = np.where(
            se > 0,
            se2 ** 2 / ((var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1)),
            1.0,
        )

    pvals = np.where(se > 0, 2.0 * t_dist.sf(np.abs(t_stats), dof),
                      np.where((se == 0) & (mean1 != mean2), 0.0, np.nan))
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
    expression_scale: Literal["linear", "log1p"],
    log_base: float | None = None,
    block_key: str | None = None,
    library_key: str | None = "auto",
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    min_cells: int = 10,
    fdr_scope: Literal["pair", "global"] = "global",
    skip_self_pairs: bool = True,
    pairs: "list[tuple[str, str]] | None" = None,
    n_perms: int = 200,
    seed: int = 0,
    n_jobs: int = 1,
    key_added: str = "icg",
) -> pd.DataFrame:
    """Within-type contact association, with fixed receiver identities.

    Contact flags are permuted within receiver type, library and optional
    ``block_key``. This assumes exchangeability within those strata and
    does not establish signaling or causality. BH corrects genes and pairs
    together by default. ``n_perms=0`` is not tested (asymptotic p-values
    remain diagnostics). Specify the expression scale for fold changes:
    ``log1p`` uses natural logs unless ``log_base`` is supplied.
    """
    from astrolabe.tl._spatial_stats import _get_conn_matrix, _permutation_blocks
    conn_key = adata.uns.get(neighbors_key, {}).get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first"
        )
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")
    if n_perms < 0:
        raise ValueError(f"n_perms must be >= 0; got {n_perms}")

    if method not in ("wilcoxon", "t-test") or fdr_scope not in ("pair", "global"):
        raise ValueError("invalid method or fdr_scope")
    if min_cells < (2 if method == "t-test" else 1):
        raise ValueError("min_cells must be at least two for Welch, or one for Mann-Whitney")
    if expression_scale not in ("linear", "log1p"):
        raise ValueError("expression_scale must be linear or log1p")
    if log_base is not None and (not np.isfinite(log_base) or log_base <= 0 or log_base == 1):
        raise ValueError("log_base must be positive and different from one")
    blocks, library_key = _permutation_blocks(adata, neighbors_key, library_key=library_key)
    spatial_strata = np.zeros(adata.n_obs, dtype=int)
    for code, block in enumerate(blocks):
        spatial_strata[block] = code
    strata_keys = [block_key] if block_key is not None else []
    if block_key is not None and block_key not in adata.obs:
        raise KeyError(block_key)
    for k in strata_keys:
        if adata.obs[k].isna().any():
            raise ValueError(f"missing exchangeability labels in {k}")
    strata = (pd.MultiIndex.from_arrays([spatial_strata, adata.obs[block_key]]).factorize()[0]
              if strata_keys else spatial_strata)
    conn = _get_conn_matrix(adata, neighbors_key).tocsr(copy=True)
    conn.setdiag(0)
    conn.eliminate_zeros()
    # Shared with the rest of tl/_neighborhood.py's category-resolution
    # convention (declared .cat.categories order) instead of a separate
    # ad-hoc np.unique(labels) (alphabetical-by-value, and loses any
    # meaningful custom category order, e.g. Leiden's '0','1',...,'10').
    codes, categories = _category_codes(adata, groupby)
    gene_names = list(adata.var_names)
    from astrolabe._coverage import _require_measured
    _require_measured(adata, gene_names)
    n_genes = len(gene_names)

    # Keep X sparse; only the per-pair row subsets are densified.
    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "get"):
        X = X.get()
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

            mean_c = X_cont.mean(axis=0)
            mean_nc = X_non.mean(axis=0)
            try:
                with np.errstate(over="raise", invalid="raise"):
                    factor = 1 if log_base is None else np.log(log_base)
                    linear_c = np.expm1(X_cont * factor) if expression_scale == "log1p" else X_cont
                    linear_nc = np.expm1(X_non * factor) if expression_scale == "log1p" else X_non
            except FloatingPointError as exc:
                raise ValueError("expression scale inversion overflowed; check expression_scale/log_base") from exc
            if (not np.isfinite(linear_c).all() or not np.isfinite(linear_nc).all()
                    or (linear_c < 0).any() or (linear_nc < 0).any()):
                raise ValueError("fold changes require finite nonnegative linear expression")
            mean_linear_c, mean_linear_nc = linear_c.mean(axis=0), linear_nc.mean(axis=0)
            log2fc = np.log2((mean_linear_c + 1e-9) / (mean_linear_nc + 1e-9))
            log2fc = np.where((mean_linear_c == 0) & (mean_linear_nc == 0), np.nan, log2fc)

            tested_pairs.append((ct_a, ct_b))
            obs_z.append(z)

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

        fixed_splits = [_split_pair(codes, ca, cb) for ca, cb in pair_codes]

        def _one_perm(rng):
            out = np.full((n_pairs, n_genes), np.nan)
            for pi, split in enumerate(fixed_splits):
                cont, non = split
                receiver = np.concatenate([cont, non])
                flags = np.concatenate([np.ones(len(cont), dtype=bool), np.zeros(len(non), dtype=bool)])
                for stratum in np.unique(strata[receiver]):
                    idx = np.flatnonzero(strata[receiver] == stratum)
                    flags[idx] = rng.permutation(flags[idx])
                _, z_p, _ = test_fn(_dense_rows(X, receiver[flags]), _dense_rows(X, receiver[~flags]))
                out[pi] = z_p
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

        pval_perm, n_valid = permutation_p(np.stack(obs_z), np.moveaxis(perm_z, 0, -1))
        for pi, part in enumerate(parts):
            part["pval"] = pval_perm[pi]
            part["n_valid"] = n_valid[pi]
    else:
        for part in parts:
            part["pval"] = np.nan
            part["n_valid"] = 0

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not df.empty:
        if fdr_scope == "global":
            df["pval_adj"] = _bh_correct(df["pval"].to_numpy())
            annotate_permutation_resolution(df, "pval", "n_valid", label="contact-associated genes")
        else:
            df["pval_adj"] = df.groupby(
                ["cell_type", "neighbor_type"], observed=True
            )["pval"].transform(lambda p: _bh_correct(p.to_numpy()))
            df["p_min"] = np.nan
            df["resolution_limited"] = False
            families = []
            for pair, positions in df.groupby(["cell_type", "neighbor_type"], observed=True).indices.items():
                part = df.iloc[positions]
                floors, limited, summary = permutation_resolution(part.pval, part.n_valid,
                    label=f"contact-associated genes {pair}")
                df.loc[part.index, "p_min"] = floors
                df.loc[part.index, "resolution_limited"] = limited
                families.append(dict(cell_type=str(pair[0]), neighbor_type=str(pair[1]), **summary))
            df.attrs["permutation_resolution"] = {"scope": "pair", "families": families}
        df = df.sort_values(["cell_type", "neighbor_type", "pval"]).reset_index(drop=True)

    df.attrs.update(null="within-type contact flags", block_key=block_key, library_key=library_key,
                    expression_scale=expression_scale, log_base=log_base, fdr_scope=fdr_scope)
    if not df.empty:
        df["status"] = np.where(df.pval.notna(), "tested", "not_tested")
    adata.uns[key_added] = df
    log.info(
        "interaction_changed_genes: %d rows across %d tested pairs",
        len(df), n_pairs,
    )
    return df
