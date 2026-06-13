"""
Interaction Changed Genes (ICGs).

For every ordered cell-type pair (receiver A, neighbour B), split A-cells
into "contacted" (≥1 spatial neighbour of type B) vs "non-contacted" and
run a two-group DE test across all genes simultaneously.
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._utils import with_adata

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
    sigma = np.sqrt(n1 * n2 * (n + 1) / 12.0)
    z = (U1 - mu) / sigma
    pvals = np.minimum(2.0 * norm.sf(np.abs(z)), 1.0)
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


@with_adata
def interaction_changed_genes(
    adata: ad.AnnData,
    *,
    groupby: str,
    neighbors_key: str = "spatial_neighbors",
    layer: str | None = None,
    method: Literal["wilcoxon", "t-test"] = "wilcoxon",
    min_cells: int = 10,
    key_added: str = "icg",
) -> pd.DataFrame:
    """Test for genes whose expression in cell type A changes when neighbouring cell type B.

    For every ordered pair (receiver A, neighbour B), cells of type A are
    split into "contacted" (≥1 spatial neighbour of type B) and
    "non-contacted" groups; a two-group DE test is run across all genes at
    once.  Results are stored in ``adata.uns[key_added]`` and returned.

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
    key_added
        Key written to ``adata.uns``.

    Returns
    -------
    pd.DataFrame
        Tidy frame sorted by ``(cell_type, neighbor_type, pval)`` with columns
        ``cell_type``, ``neighbor_type``, ``gene``, ``statistic``, ``pval``,
        ``pval_adj`` (BH per pair), ``mean_contacted``, ``mean_non_contacted``,
        ``log2fc``, ``n_contacted``, ``n_non_contacted``.

    Examples
    --------
    >>> df = ab.tl.interaction_changed_genes(mdata, groupby="leiden")
    >>> df[df.pval_adj < 0.05].head(20)
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found.")

    conn = adata.obsp[conn_key]
    labels = np.asarray(adata.obs[groupby])
    cell_types = np.unique(labels)
    gene_names = list(adata.var_names)

    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    type_idx = {ct: np.where(labels == ct)[0] for ct in cell_types}
    test_fn = _wilcoxon_vectorized if method == "wilcoxon" else _ttest_vectorized

    parts: list[pd.DataFrame] = []

    for ct_a in cell_types:
        idx_a = type_idx[ct_a]
        conn_a = conn[idx_a, :]  # sparse (|A|, n_cells)

        for ct_b in cell_types:
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
            pvals_adj = _bh_correct(pvals)

            mean_c = X_cont.mean(axis=0)
            mean_nc = X_non.mean(axis=0)
            log2fc = np.log2((mean_c + 1e-9) / (mean_nc + 1e-9))

            parts.append(
                pd.DataFrame(
                    {
                        "cell_type": ct_a,
                        "neighbor_type": ct_b,
                        "gene": gene_names,
                        "statistic": stats,
                        "pval": pvals,
                        "pval_adj": pvals_adj,
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
        df = df.sort_values(["cell_type", "neighbor_type", "pval"]).reset_index(drop=True)

    adata.uns[key_added] = df
    log.info(
        "interaction_changed_genes: %d rows across %d tested pairs",
        len(df), len(parts),
    )
    return df
