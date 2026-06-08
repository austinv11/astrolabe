"""
Spatial statistical tests.

Moran's I is the canonical measure of spatial autocorrelation for gene
expression.  Our implementation follows the standard definition:

    I = (N / W) * (z^T W z) / (z^T z)

where N is the number of cells, W is the sum of all weights in the spatial
weight matrix, z = X_j - mean(X_j) is the mean-centered expression vector
for gene j, and W is the binary spatial adjacency matrix.

For N cells and G genes, the naive computation is O(N² × G).  We exploit
the sparse connectivity matrix to reduce this to O(nnz × G) where nnz is
the number of edges in the neighborhood graph (typically O(N × k) for a
k-NN graph).

Statistical significance is assessed via a z-score under the normality
assumption:

    E[I]   = -1 / (N - 1)
    Var[I] = ... (see Cliff & Ord 1981)

or, for small datasets, via permutation testing.
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from astrolabe._utils import with_adata

__all__ = ["spatial_autocorrelation"]

log = logging.getLogger(__name__)


@with_adata
def spatial_autocorrelation(
    adata: ad.AnnData,
    *,
    genes: list[str] | None = None,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 0,
    n_jobs: int = 1,
    key_added: str = "spatial_autocorrelation",
) -> None:
    """Compute Moran's I spatial autocorrelation for each gene.

    Results are stored in ``adata.uns[key_added]`` as a DataFrame with
    one row per gene containing:

    - ``I`` — Moran's I statistic (range approximately [-1, 1];
      positive values indicate spatial clustering, negative indicate
      dispersion, ~0 indicates random spatial distribution)
    - ``E_I`` — expected value under spatial randomness (= -1/(N-1))
    - ``z_score`` — (I - E_I) / sqrt(Var_I) under normality assumption
    - ``p_value`` — two-sided p-value under normality
    - ``p_value_perm`` — permutation p-value (only if ``n_perms > 0``)

    Parameters
    ----------
    adata
        AnnData with expression data and a spatial connectivity matrix.
    genes
        Subset of genes to test.  ``None`` tests all genes in ``var``.
        For large panels (> 500 genes) the full computation may take
        several minutes.
    layer
        Layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` for the spatial neighborhood graph.  Must have
        been set by :func:`astrolabe.tl.spatial_neighbors`.
    n_perms
        Number of permutations for empirical p-values.  ``0`` disables
        permutation testing.  Recommended: 999 or 9999 for publication.
    n_jobs
        Number of parallel workers for permutation testing.
    key_added
        Key in ``uns`` for the result DataFrame.

    Examples
    --------
    >>> ab.tl.spatial_autocorrelation(mdata["xenium"], n_perms=999)
    >>> mdata["xenium"].uns["spatial_autocorrelation"].sort_values("I", ascending=False).head()
    """
    if neighbors_key not in adata.uns:
        raise KeyError(
            f"uns['{neighbors_key}'] not found. Run ab.tl.spatial_neighbors first."
        )

    conn_key = adata.uns[neighbors_key].get(
        "connectivities_key", neighbors_key + "_connectivities"
    )
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )

    W = adata.obsp[conn_key]  # sparse (N, N) binary adjacency
    N = adata.n_obs
    W_sum = W.sum()

    if W_sum == 0:
        raise ValueError("Spatial weight matrix has all-zero weights. Check the neighborhood graph.")

    # Select expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if sp.issparse(X):
        X = np.asarray(X.todense(), dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    # Select genes
    if genes is not None:
        gene_idx = [adata.var_names.get_loc(g) for g in genes]
        X = X[:, gene_idx]
        gene_names = list(genes)
    else:
        gene_names = list(adata.var_names)

    n_genes = len(gene_names)
    log.info("Computing Moran's I for %d genes over %d cells", n_genes, N)

    # Mean-center each gene: z shape (N, G)
    z = X - X.mean(axis=0, keepdims=True)
    # Denominator: z^T z for each gene, shape (G,)
    z_dot_z = (z ** 2).sum(axis=0)  # (G,)

    # Numerator: z^T W z for each gene
    # W is (N, N), z is (N, G)
    # Wz = W @ z  →  (N, G), then sum element-wise with z → (G,)
    Wz = W @ z  # (N, G), sparse-dense product
    numerator = (z * np.asarray(Wz)).sum(axis=0)  # (G,)

    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.where(z_dot_z > 0, (N / W_sum) * numerator / z_dot_z, 0.0)

    # Expected value and variance under normality assumption
    # (Cliff & Ord 1981, eq. 2.3)
    E_I = -1.0 / (N - 1)

    # Variance of I under normality:
    S1 = 0.5 * ((W + W.T).power(2)).sum()
    S2 = ((np.asarray(W.sum(axis=1)) + np.asarray(W.sum(axis=0).T)) ** 2).sum()
    n = float(N)
    # Kurtosis correction term
    m2 = (z ** 2).mean(axis=0)
    m4 = (z ** 4).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b2 = np.where(m2 > 0, m4 / (m2 ** 2), 3.0)  # excess kurtosis approx

    # Var under normality (simplified for binary W, b2→3):
    # Var[I] = (n²S1 - nS2 + 3W²) / ((n²-1)W²)
    # We use the b2-corrected formula for correctness:
    numerator_var = (
        n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * W_sum**2)
        - b2 * ((n**2 - n) * S1 - 2 * n * S2 + 6 * W_sum**2)
    )
    denominator_var = (n - 1) * (n - 2) * (n - 3) * W_sum**2

    with np.errstate(divide="ignore", invalid="ignore"):
        Var_I = np.where(denominator_var > 0, numerator_var / denominator_var - E_I**2, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(Var_I > 0, (I - E_I) / np.sqrt(Var_I), np.nan)

    # Two-sided p-value under normality
    p_values = 2.0 * stats.norm.sf(np.abs(z_scores))

    result = pd.DataFrame(
        {
            "I": I,
            "E_I": np.full(n_genes, E_I),
            "z_score": z_scores,
            "p_value": p_values,
        },
        index=pd.Index(gene_names, name="gene"),
    )

    # ------------------------------------------------------------------ #
    # Permutation testing (optional)                                     #
    # ------------------------------------------------------------------ #
    if n_perms > 0:
        log.info("Running %d permutations for empirical p-values...", n_perms)
        perm_I = _permutation_morans_i(z, W, W_sum, N, n_perms, n_jobs)
        # Empirical two-sided p-value
        p_perm = (np.abs(perm_I) >= np.abs(I[np.newaxis, :])).mean(axis=0)
        result["p_value_perm"] = p_perm

    adata.uns[key_added] = result
    log.info(
        "spatial_autocorrelation: %d genes. Top I=%.3f (%s)",
        n_genes,
        result["I"].max(),
        result["I"].idxmax(),
    )


def _permutation_morans_i(
    z: np.ndarray,
    W,
    W_sum: float,
    N: int,
    n_perms: int,
    n_jobs: int,
) -> np.ndarray:
    """Return (n_perms, n_genes) array of permuted Moran's I values."""
    n_genes = z.shape[1]
    perm_I = np.zeros((n_perms, n_genes), dtype=np.float64)
    z_dot_z = (z ** 2).sum(axis=0)

    rng = np.random.default_rng(seed=42)

    for p in range(n_perms):
        perm_idx = rng.permutation(N)
        z_perm = z[perm_idx]
        Wz_perm = W @ z_perm
        num = (z_perm * np.asarray(Wz_perm)).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            perm_I[p] = np.where(z_dot_z > 0, (N / W_sum) * num / z_dot_z, 0.0)

    return perm_I