"""Tests for pp.banksy_matrix (audit group 4): per-block z-scoring
(λ = variance split), AGF local-mean subtraction, the 2^-m agf_weight
default, and planted-spatial-domain recovery at high vs low λ."""
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pytest

import astrolabe.tl as tl
from astrolabe.pp._banksy import (
    _agf_magnitude,
    _build_weight_matrices,
    banksy_matrix,
)


def _planted_domain_adata(n=600, n_genes=10, mix=0.7, seed=0, n_neighbors=10):
    """Two spatial domains with the SAME two cell types at mirrored proportions.

    Domain 0 (x < 500) is *mix* type-1 / (1-mix) type-0; domain 1 is the
    reverse. Individual cells of a given type are identically distributed in
    both domains, so pure-expression clustering can only recover cell types
    — only the *neighborhood composition* distinguishes the domains.
    """
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 1000, (n, 2)).astype(np.float32)
    domain = (coords[:, 0] > 500).astype(int)
    p_type1 = np.where(domain == 0, mix, 1.0 - mix)
    ctype = (rng.uniform(size=n) < p_type1).astype(int)

    X = rng.normal(0.0, 0.3, (n, n_genes))
    X[ctype == 1, : n_genes // 2] += 2.0
    X[ctype == 0, n_genes // 2:] += 2.0

    adata = ad.AnnData(X=X.astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obsm["spatial"] = coords
    tl.spatial_neighbors(adata, n_neighbors=n_neighbors)
    return adata, domain, ctype


def _ring_graph_adata(n=40, n_genes=3, seed=0):
    """Small fixture with a simple KNN graph for weight-matrix unit tests."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 200, (n, 2)).astype(np.float64)
    X = rng.normal(0, 1, (n, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = coords.astype(np.float32)
    tl.spatial_neighbors(adata, n_neighbors=5)
    return adata


class TestBanksyVarianceSplit:
    def test_lambda_is_exact_variance_split_without_agf(self):
        adata = _ring_graph_adata(n=80, n_genes=6)
        lam = 0.8
        banksy_matrix(adata, lambda_param=lam)
        out = adata.obsm["X_banksy"].astype(np.float64)
        g = adata.n_vars
        var_self = out[:, :g].var(axis=0).sum()
        var_nbr = out[:, g:].var(axis=0).sum()
        # Each block is z-scored (unit variance per column) before the
        # sqrt(1-λ)/sqrt(λ) weights, so the split is exact by construction.
        assert var_self == pytest.approx((1.0 - lam) * g, rel=1e-4)
        assert var_nbr == pytest.approx(lam * g, rel=1e-4)

    def test_lambda_variance_split_with_agf_and_default_kappa(self):
        adata = _ring_graph_adata(n=80, n_genes=6)
        lam = 0.6
        banksy_matrix(adata, lambda_param=lam, agf=True)
        out = adata.obsm["X_banksy"].astype(np.float64)
        g = adata.n_vars
        var_self = out[:, :g].var(axis=0).sum()
        var_m0 = out[:, g:2 * g].var(axis=0).sum()
        var_m1 = out[:, 2 * g:].var(axis=0).sum()
        # Default agf_weight = 1/3 (banksy_py's 2^-m harmonic split):
        # M0 gets 2/3·λ, |M1| gets 1/3·λ.
        assert var_self == pytest.approx((1.0 - lam) * g, rel=1e-4)
        assert var_m0 == pytest.approx(lam * (2.0 / 3.0) * g, rel=1e-4)
        assert var_m1 == pytest.approx(lam * (1.0 / 3.0) * g, rel=1e-4)

    def test_lambda_zero_is_pure_zscored_expression(self):
        adata = _ring_graph_adata(n=50, n_genes=4)
        banksy_matrix(adata, lambda_param=0.0)
        out = adata.obsm["X_banksy"].astype(np.float64)
        g = adata.n_vars
        # Neighborhood block weighted by sqrt(0) = 0.
        assert np.allclose(out[:, g:], 0.0)
        # Own block is the z-scored expression: unit variance per column.
        assert np.allclose(out[:, :g].var(axis=0), 1.0, atol=1e-4)

    def test_zero_variance_gene_contributes_nothing(self):
        adata = _ring_graph_adata(n=50, n_genes=4)
        X = adata.X.copy()
        X[:, 0] = 7.0  # constant gene
        adata.X = X
        banksy_matrix(adata, lambda_param=0.5)
        out = adata.obsm["X_banksy"]
        g = adata.n_vars
        assert np.allclose(out[:, 0], 0.0)      # own block
        assert np.allclose(out[:, g], 0.0)      # M0 block (also constant)
        assert np.isfinite(out).all()


class TestAGFLocalMeanSubtraction:
    def test_constant_gene_has_zero_agf_magnitude(self):
        # With local-mean subtraction, x_j - M0_i = 0 for every neighbor of
        # every cell when the gene is spatially constant, so |M1| must be
        # exactly 0. Without the subtraction (the old DC-leakage bug), |M1|
        # would be |Σ w·exp(iθ)|·x — nonzero wherever a cell's neighbors are
        # not perfectly angularly balanced.
        adata = _ring_graph_adata(n=60, n_genes=2, seed=3)
        X = np.column_stack([
            np.full(adata.n_obs, 5.0, dtype=np.float32),
            np.asarray(adata.X)[:, 1],
        ])
        coords = adata.obsm["spatial"].astype(np.float64)
        W_norm, W_cos, W_sin = _build_weight_matrices(
            adata.obsp["spatial_neighbors_connectivities"], coords, agf=True
        )
        M0 = np.asarray(W_norm @ X, dtype=np.float32)
        M1 = _agf_magnitude(W_cos, W_sin, X, M0)
        assert np.allclose(M1[:, 0], 0.0, atol=1e-4)
        # The varying gene still produces genuine gradient signal.
        assert M1[:, 1].max() > 1e-3


class TestBanksyPlantedDomains:
    def test_high_lambda_separates_domains_better_than_lambda_zero(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        adata, domain, _ = _planted_domain_adata()

        aris = {}
        for lam in (0.0, 0.8):
            banksy_matrix(adata, lambda_param=lam, key_added=f"X_banksy_{lam}")
            labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
                adata.obsm[f"X_banksy_{lam}"]
            )
            aris[lam] = adjusted_rand_score(domain, labels)

        # λ=0 can only find cell types (which agree with the domain for
        # ~70% of cells → ARI ≈ 0.15); λ=0.8 sees the neighborhood
        # composition and recovers the planted domains far better (ARI ≈ 0.5
        # — cells within one neighbor-radius of the domain boundary have
        # genuinely mixed neighborhoods, which caps recovery below 1.0).
        assert aris[0.8] > aris[0.0] + 0.2, f"ARIs: {aris}"
        assert aris[0.8] > 0.4, f"high-λ ARI too low: {aris}"

    def test_lambda_zero_recovers_cell_types(self):
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        adata, _, ctype = _planted_domain_adata(seed=1)
        banksy_matrix(adata, lambda_param=0.0)
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(
            adata.obsm["X_banksy"]
        )
        assert adjusted_rand_score(ctype, labels) > 0.9


class TestBanksyValidation:
    def test_lambda_out_of_range_raises(self):
        adata = _ring_graph_adata()
        with pytest.raises(ValueError, match="lambda_param"):
            banksy_matrix(adata, lambda_param=1.5)

    def test_missing_graph_raises(self):
        adata = ad.AnnData(X=np.zeros((5, 3), dtype=np.float32))
        adata.obsm["spatial"] = np.zeros((5, 2), dtype=np.float32)
        with pytest.raises(KeyError, match="spatial_neighbors"):
            banksy_matrix(adata)

    def test_sparse_input_matches_dense(self):
        adata_d = _ring_graph_adata(n=40, n_genes=5, seed=7)
        adata_s = adata_d.copy()
        adata_s.X = sp.csr_matrix(adata_d.X)
        banksy_matrix(adata_d, lambda_param=0.3, agf=True)
        banksy_matrix(adata_s, lambda_param=0.3, agf=True)
        np.testing.assert_allclose(
            adata_d.obsm["X_banksy"], adata_s.obsm["X_banksy"], rtol=1e-5, atol=1e-6
        )
