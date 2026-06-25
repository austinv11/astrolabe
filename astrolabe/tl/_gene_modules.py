"""
Spatial co-expression gene modules.

Builds a gene × gene spatial correlation matrix (via spatial-lag Pearson
correlation or vectorised bivariate Moran's I), then clusters genes into
modules using hierarchical or spectral clustering.
"""
from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._utils import with_adata

__all__ = ["spatial_gene_modules"]

log = logging.getLogger(__name__)


def _row_normalize(W: sp.spmatrix) -> sp.csr_matrix:
    W = W.tocsr().astype(np.float64)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    # avoid division by zero for isolated cells
    row_sums[row_sums == 0] = 1.0
    D_inv = sp.diags(1.0 / row_sums)
    return D_inv @ W


def _spatial_corr_matrix(
    Z: np.ndarray,
    W_norm: sp.csr_matrix,
    method: str,
) -> np.ndarray:
    """Return (n_genes, n_genes) spatial co-expression matrix."""
    WZ = W_norm @ Z  # (n_cells, n_genes) — spatial lag of each gene

    if method == "spatial_lag":
        # Pearson correlation of the spatially-lagged expression vectors
        corr = np.corrcoef(WZ.T)  # (n_genes, n_genes)
        np.fill_diagonal(corr, 1.0)
        return np.nan_to_num(corr, nan=0.0)

    else:  # moran_bv
        # I_bi(i,j) = z_i^T W z_j / n  (for row-normalized W, S0 = n)
        n = Z.shape[0]
        corr = (Z.T @ WZ) / n  # (n_genes, n_genes)
        corr = (corr + corr.T) / 2  # symmetrise floating-point asymmetry
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr


def _cluster_genes(
    corr: np.ndarray,
    n_modules: int,
    method: str,
) -> np.ndarray:
    """Return integer module labels (0-indexed) of length n_genes."""
    if method == "hierarchical":
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist = np.clip((1.0 - corr) / 2.0, 0.0, 1.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        Z_link = linkage(condensed, method="ward")
        labels = fcluster(Z_link, t=n_modules, criterion="maxclust") - 1
    else:  # spectral
        from sklearn.cluster import SpectralClustering

        affinity = (corr + 1.0) / 2.0
        np.fill_diagonal(affinity, 1.0)
        sc = SpectralClustering(
            n_clusters=n_modules,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=0,
        )
        labels = sc.fit_predict(affinity)

    return labels.astype(np.int32)


@with_adata
def spatial_gene_modules(
    adata: ad.AnnData,
    *,
    var_names: list[str] | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_modules: int = 10,
    correlation: Literal["spatial_lag", "moran_bv"] = "spatial_lag",
    clustering: Literal["hierarchical", "spectral"] = "hierarchical",
    score_cells: bool = True,
    layer: str | None = None,
    key_added: str = "spatial_module",
) -> pd.DataFrame:
    """Cluster genes into spatial co-expression modules.

    Builds a gene × gene spatial correlation matrix and clusters it to find
    groups of genes that vary together across space.  Optionally scores each
    cell for each module.

    Results are stored in:

    - ``adata.uns[key_added]`` — dict with ``'gene_modules'`` DataFrame and
      ``'params'``.
    - ``adata.obsm[key_added + '_scores']`` — ``(n_cells, n_modules)`` float32
      array of per-module mean expression scores (only when
      ``score_cells=True``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with spatial
        connectivity in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    var_names
        Genes to include.  ``None`` uses all genes in ``adata.var``.
    neighbors_key
        Prefix for the spatial graph in ``obsp``
        (``{neighbors_key}_connectivities``).
    n_modules
        Number of gene modules.
    correlation
        Method for computing the gene × gene spatial correlation matrix:

        ``'spatial_lag'``
            Pearson correlation of each gene's spatially-lagged expression
            vector (W @ x) with every other gene's lag.  Fast and intuitive;
            genes that are spatially co-expressed with similar neighbors will
            cluster together.
        ``'moran_bv'``
            Vectorised bivariate Moran's I for every gene pair:
            I(i,j) = zᵢᵀ W z_j / n (row-normalized W).  More formally
            grounded in spatial statistics; captures the degree to which
            high values of gene i coincide with spatially-lagged high values
            of gene j.  Values lie in [−1, 1].
    clustering
        Algorithm for clustering the gene × gene correlation matrix:

        ``'hierarchical'``
            Ward linkage on distance = (1 − corr) / 2.  Deterministic.
        ``'spectral'``
            Spectral clustering with affinity = (corr + 1) / 2.
            Requires ``scikit-learn``.
    score_cells
        Compute the mean (normalized) expression of each module's genes and
        store the result in ``obsm[key_added + '_scores']``.
    layer
        Expression layer for scoring.  ``None`` uses ``adata.X``.
    key_added
        Key prefix for ``uns`` and ``obsm``.

    Returns
    -------
    pd.DataFrame
        ``(n_genes_tested, 2)`` frame with columns ``gene`` and ``module``
        (integer 0-indexed module IDs), sorted by module then gene name.

    Examples
    --------
    >>> df = ab.tl.spatial_gene_modules(mdata, n_modules=8)
    >>> df[df.module == 0]["gene"].tolist()           # genes in module 0
    >>> mdata["xenium"].obsm["spatial_module_scores"] # cell scores (n×8)
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )

    # ── Resolve gene subset ────────────────────────────────────────────────────
    if var_names is not None:
        missing = [g for g in var_names if g not in adata.var_names]
        if missing:
            raise KeyError(f"var_names not found in adata: {missing[:5]}")
        gene_idx = np.array([adata.var_names.get_loc(g) for g in var_names])
        gene_names = list(var_names)
    else:
        gene_idx = np.arange(adata.n_vars)
        gene_names = list(adata.var_names)

    n_genes = len(gene_names)
    n_modules_eff = min(n_modules, n_genes)
    if n_modules_eff < n_modules:
        log.warning(
            "n_modules=%d reduced to %d (only %d genes available)",
            n_modules, n_modules_eff, n_genes,
        )

    # ── Expression matrix ──────────────────────────────────────────────────────
    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)[:, gene_idx]  # (n_cells, n_genes)

    # Standardise: zero mean, unit variance per gene
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Z = (X - mu) / sigma

    # ── Spatial graph ──────────────────────────────────────────────────────────
    W_norm = _row_normalize(adata.obsp[conn_key])

    # ── Gene × gene spatial correlation ───────────────────────────────────────
    log.info(
        "spatial_gene_modules: building %s correlation matrix (%d × %d)",
        correlation, n_genes, n_genes,
    )
    corr = _spatial_corr_matrix(Z, W_norm, correlation)

    # ── Cluster genes ──────────────────────────────────────────────────────────
    log.info("spatial_gene_modules: clustering (%s, k=%d)", clustering, n_modules_eff)
    module_labels = _cluster_genes(corr, n_modules_eff, clustering)

    df = pd.DataFrame({"gene": gene_names, "module": module_labels})
    df = df.sort_values(["module", "gene"]).reset_index(drop=True)

    # ── Cell scores ────────────────────────────────────────────────────────────
    if score_cells:
        scores = np.zeros((adata.n_obs, n_modules_eff), dtype=np.float32)
        X_score = adata.layers[layer] if layer is not None else adata.X
        if sp.issparse(X_score):
            X_score = X_score.toarray()
        X_score = np.asarray(X_score, dtype=np.float32)

        for m in range(n_modules_eff):
            m_genes = np.where(module_labels == m)[0]
            if len(m_genes) == 0:
                continue
            scores[:, m] = X_score[:, gene_idx[m_genes]].mean(axis=1)

        adata.obsm[key_added + "_scores"] = scores
        log.info("spatial_gene_modules: scores stored in obsm['%s_scores']", key_added)

    adata.uns[key_added] = {
        "gene_modules": df,
        "params": {
            "n_modules": n_modules_eff,
            "correlation": correlation,
            "clustering": clustering,
            "neighbors_key": neighbors_key,
            "layer": layer,
        },
    }

    log.info(
        "spatial_gene_modules: %d genes → %d modules", n_genes, n_modules_eff
    )
    return df
