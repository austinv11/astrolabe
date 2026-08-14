"""Tests for tl._gene_modules (batch-6 bugfix: phantom-module exclusion)."""
import numpy as np
import anndata as ad
import scipy.sparse as sp

from astrolabe.tl._gene_modules import spatial_gene_modules


def _adata_with_undetected_genes(n_cells=200, n_real_genes=6, n_undetected=4, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 500, (n_cells, 2)).astype(np.float32)

    # Two real spatial patterns (2 groups of 3 genes each, correlated with a
    # smooth spatial gradient) + a block of undetected (all-zero) genes.
    grad1 = coords[:, 0] / 500.0
    grad2 = coords[:, 1] / 500.0
    real = np.column_stack([
        grad1 + rng.normal(0, 0.05, n_cells) for _ in range(n_real_genes // 2)
    ] + [
        grad2 + rng.normal(0, 0.05, n_cells) for _ in range(n_real_genes // 2)
    ])
    undetected = np.zeros((n_cells, n_undetected))
    X = np.column_stack([real, undetected]).astype(np.float32)
    gene_names = [f"real_{i}" for i in range(n_real_genes)] + [f"undetected_{i}" for i in range(n_undetected)]

    adata = ad.AnnData(X=X)
    adata.var_names = gene_names
    adata.obsm["spatial"] = coords

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=7)
    rows = np.repeat(np.arange(n_cells), 6)
    cols = idxs[:, 1:].ravel()
    conn = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
    adata.obsp["spatial_neighbors_connectivities"] = conn
    return adata


class TestSpatialGeneModulesPhantomExclusion:
    def test_undetected_genes_get_module_negative_one(self):
        # Regression: undetected/zero-variance genes used to be silently
        # pooled into a real cluster (fcluster with criterion="maxclust"
        # always returns exactly n_modules groups regardless of whether real
        # structure exists), indistinguishable from a genuine co-expression
        # module. They should now be excluded entirely (module=-1).
        adata = _adata_with_undetected_genes()
        df = spatial_gene_modules(adata, n_modules=2, score_cells=False)

        undetected_rows = df[df["gene"].str.startswith("undetected_")]
        real_rows = df[df["gene"].str.startswith("real_")]
        assert (undetected_rows["module"] == -1).all()
        assert (real_rows["module"] != -1).all()

    def test_real_genes_still_cluster_into_requested_modules(self):
        adata = _adata_with_undetected_genes()
        df = spatial_gene_modules(adata, n_modules=2, score_cells=False)
        real_rows = df[df["gene"].str.startswith("real_")]
        assert set(real_rows["module"].unique()) <= {0, 1}
        assert real_rows["module"].nunique() == 2

    def test_score_cells_ignores_excluded_genes(self):
        adata = _adata_with_undetected_genes()
        spatial_gene_modules(adata, n_modules=2, score_cells=True, key_added="sgm")
        scores = adata.obsm["sgm_scores"]
        assert scores.shape == (adata.n_obs, 2)
        assert np.isfinite(scores).all()

    def test_all_genes_undetected_returns_all_excluded_without_crashing(self):
        n = 50
        adata = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32))
        adata.var_names = ["g0", "g1", "g2"]
        rng = np.random.default_rng(0)
        adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
        conn = sp.csr_matrix((np.ones(n), (np.arange(n), (np.arange(n) + 1) % n)), shape=(n, n))
        adata.obsp["spatial_neighbors_connectivities"] = conn

        df = spatial_gene_modules(adata, n_modules=2, score_cells=False)
        assert (df["module"] == -1).all()
