"""Tests for pp.normalize_by_area — area-density normalization of RAW counts,
an alternative to normalize_total (never a step after it)."""
import numpy as np
import anndata as ad
import pytest
import scipy.sparse as sp

from astrolabe.pp._normalize import normalize_by_area


def _adata_with_area(n=400, n_genes=20, seed=0):
    rng = np.random.default_rng(seed)
    # Raw counts whose expected value grows linearly with cell area (bigger
    # cells capture proportionally more transcripts) — so counts/area is the
    # size-independent quantity the normalization should recover.
    area = rng.uniform(50, 500, n).astype(np.float64)
    gene_rates = rng.uniform(0.5, 2.0, n_genes)
    counts = rng.poisson(area[:, None] / 100.0 * gene_rates[None, :])
    adata = ad.AnnData(X=counts.astype(np.float32))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cell_area"] = area
    return adata, area


def _area_dependence(values: np.ndarray, area: np.ndarray) -> float:
    """abs(Pearson correlation) between per-cell total and area."""
    totals = np.asarray(values).sum(axis=1)
    return abs(np.corrcoef(totals, area)[0, 1])


class TestNormalizeByArea:
    def test_removes_area_dependence_of_proportional_counts(self):
        # Counts ∝ area: raw totals correlate strongly with area; after
        # counts/area normalization the dependence must essentially vanish
        # (not merely shrink, and above all not INVERT into a 1/area
        # artifact — which is exactly what stacking area normalization on
        # library-size-normalized data produces).
        adata, area = _adata_with_area()
        assert _area_dependence(adata.X, area) > 0.9

        normalize_by_area(adata, log1p=False)
        after = np.corrcoef(np.asarray(adata.X).sum(axis=1), area)[0, 1]
        assert abs(after) < 0.2, f"residual/inverted area dependence: r={after:.3f}"

    def test_raises_on_log_transformed_x(self):
        # normalize_by_area REPLACES library-size normalization; applying it
        # after normalize_total forces every gene's value to scale as 1/area
        # (per-cell linear totals are constant after normalize_total). The
        # log1p marker is the detectable half of that misuse.
        import scanpy as sc
        adata, _ = _adata_with_area(n=50, n_genes=5)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        with pytest.raises(ValueError, match="raw counts"):
            normalize_by_area(adata)

    def test_explicit_raw_layer_is_allowed_after_normalization(self):
        import scanpy as sc
        adata, _ = _adata_with_area(n=50, n_genes=5)
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        normalize_by_area(adata, layer="counts", layer_added="area_norm")
        assert "area_norm" in adata.layers
        assert (np.asarray(adata.layers["area_norm"]) >= 0).all()

    def test_median_area_cell_unchanged_when_linear(self):
        # scale = area / median(area): the median-area cell's counts pass
        # through unchanged (log1p=False), anchoring the output range.
        adata, area = _adata_with_area(n=51, n_genes=4, seed=2)
        raw = np.asarray(adata.X).copy()
        normalize_by_area(adata, log1p=False)
        med_idx = int(np.argsort(area)[len(area) // 2])
        np.testing.assert_allclose(np.asarray(adata.X)[med_idx], raw[med_idx], rtol=1e-5)

    def test_sparse_and_dense_agree(self):
        adata_dense, _ = _adata_with_area(n=60, n_genes=8, seed=3)
        adata_sparse = adata_dense.copy()
        adata_sparse.X = sp.csr_matrix(adata_sparse.X)

        normalize_by_area(adata_dense)
        normalize_by_area(adata_sparse)

        np.testing.assert_allclose(
            adata_dense.X, np.asarray(adata_sparse.X.todense()), atol=1e-4
        )

    def test_raises_on_nonpositive_area(self):
        adata, _ = _adata_with_area(n=10, n_genes=3)
        adata.obs.loc[adata.obs_names[0], "cell_area"] = 0.0
        with pytest.raises(ValueError):
            normalize_by_area(adata)
