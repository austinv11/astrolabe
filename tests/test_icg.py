"""Tests for ICG log2FC NaN for zero-zero genes (audit fix 2.8)."""
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pytest

from astrolabe.tl._icg import interaction_changed_genes


def _adata_for_icg(n_per_type=20, n_genes=10, seed=42):
    """Build a controlled test fixture.

    Cells 0..(n-1) are type A; cells n..(2n-1) are type B.
    The first half of A cells connect to B; the second half have no B neighbors.
    This guarantees both contacted and non-contacted groups are non-empty.
    gene_0 is forced to zero so both groups always have zero mean.
    """
    n_cells = n_per_type * 2
    rng = np.random.default_rng(seed)
    X = sp.random(n_cells, n_genes, density=0.6, format="csr", random_state=seed).astype(np.float32)
    X = X.tolil()
    X[:, 0] = 0.0  # gene_0 all-zero in both groups
    X = X.tocsr()

    adata = ad.AnnData(X=X)
    adata.obs["cell_type"] = ["A"] * n_per_type + ["B"] * n_per_type
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    # Controlled adjacency: first half of A (0..n/2-1) connect to B cells;
    # second half of A (n/2..n-1) have zero B-neighbors.
    adj = np.zeros((n_cells, n_cells), dtype=np.float32)
    half = n_per_type // 2
    b_start = n_per_type
    # A cells [0..half) each connect to B cell at (b_start + i)
    for i in range(half):
        adj[i, b_start + i] = 1.0
        adj[b_start + i, i] = 1.0
    # A cells [half..n_per_type) have no B connections

    adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix(adj)
    adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
    return adata


class TestICGLog2FC:
    def test_zero_zero_gene_log2fc_is_nan(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        assert len(result) > 0, "expected at least one pair tested with min_cells=1"
        gene0_rows = result[result["gene"] == "gene_0"]
        assert len(gene0_rows) > 0, "gene_0 must appear in at least one pair result"
        assert gene0_rows["log2fc"].isna().all(), (
            "gene with zero expression in both contacted/non-contacted groups must have NaN log2fc"
        )

    def test_expressed_gene_log2fc_is_not_all_nan(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        expressed = result[result["gene"] != "gene_0"]["log2fc"]
        assert len(expressed) > 0, "expected results for expressed genes"
        assert not expressed.isna().all(), "some expressed genes must have numeric log2fc"

    def test_result_columns_present(self):
        adata = _adata_for_icg()
        result = interaction_changed_genes(adata, groupby="cell_type", min_cells=1)
        assert len(result) > 0, "expected non-empty result with min_cells=1"
        for col in ("cell_type", "neighbor_type", "gene", "log2fc", "pval", "pval_adj"):
            assert col in result.columns, f"column '{col}' missing from ICG result"

    def test_empty_result_when_no_interactions(self):
        # Completely disconnected graph: no A-B contacts → empty result
        n_cells = 20
        X = sp.random(n_cells, 5, density=0.5, format="csr", random_state=0).astype(np.float32)
        adata = ad.AnnData(X=X)
        adata.obs["cell_type"] = ["A"] * 10 + ["B"] * 10
        adata.var_names = [f"g{i}" for i in range(5)]
        adata.obsp["spatial_neighbors_connectivities"] = sp.csr_matrix((n_cells, n_cells))
        adata.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
        result = interaction_changed_genes(adata, groupby="cell_type")
        assert len(result) == 0
