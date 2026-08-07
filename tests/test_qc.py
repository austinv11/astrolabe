"""Tests for _add_nucleus_metrics helper and calculate_qc_metrics (audit fix 4.11)."""
import numpy as np
import pandas as pd
import anndata as ad
import pyarrow as pa
import pytest
import scipy.sparse as sp

from astrolabe.pp._qc import _add_nucleus_metrics, filter_cells, filter_genes, filter_isolated


def _make_adata(**obs_cols):
    n = len(next(iter(obs_cols.values())))
    obs = pd.DataFrame(obs_cols, index=[f"c{i}" for i in range(n)])
    return ad.AnnData(obs=obs)


class TestNucleusAreaRatio:
    def test_normal_ratio(self):
        adata = _make_adata(nucleus_area=[5.0, 10.0], cell_area=[20.0, 25.0])
        _add_nucleus_metrics(adata)
        ratios = adata.obs["nucleus_to_cell_area_ratio"].values
        np.testing.assert_allclose(ratios, [5.0 / 20.0, 10.0 / 25.0], rtol=1e-6)

    def test_zero_cell_area_yields_nan(self):
        adata = _make_adata(nucleus_area=[5.0], cell_area=[0.0])
        _add_nucleus_metrics(adata)
        assert np.isnan(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"])

    def test_zero_nucleus_area_yields_zero_ratio(self):
        adata = _make_adata(nucleus_area=[0.0], cell_area=[10.0])
        _add_nucleus_metrics(adata)
        assert abs(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"] - 0.0) < 1e-6

    def test_both_zero_yields_nan(self):
        adata = _make_adata(nucleus_area=[0.0], cell_area=[0.0])
        _add_nucleus_metrics(adata)
        assert np.isnan(adata.obs.loc["c0", "nucleus_to_cell_area_ratio"])


class TestHasNucleus:
    def test_from_nucleus_count(self):
        adata = _make_adata(nucleus_count=[0, 1, 2])
        _add_nucleus_metrics(adata)
        assert list(adata.obs["has_nucleus"]) == [False, True, True]

    def test_from_nucleus_area_fallback(self):
        adata = _make_adata(nucleus_area=[0.0, 0.1, 5.0])
        _add_nucleus_metrics(adata)
        assert list(adata.obs["has_nucleus"]) == [False, True, True]

    def test_count_takes_priority_over_area(self):
        # nucleus_count present → use count even if nucleus_area also present
        adata = _make_adata(nucleus_count=[0, 1], nucleus_area=[10.0, 0.0])
        _add_nucleus_metrics(adata)
        # count=0 → False even though area=10; count=1 → True even though area=0
        assert list(adata.obs["has_nucleus"]) == [False, True]

    def test_no_nucleus_columns_adds_nothing(self):
        adata = _make_adata(total_counts=[10, 20])
        _add_nucleus_metrics(adata)
        assert "nucleus_to_cell_area_ratio" not in adata.obs.columns
        assert "has_nucleus" not in adata.obs.columns

    def test_only_cell_area_no_nucleus_area(self):
        adata = _make_adata(cell_area=[10.0, 20.0])
        _add_nucleus_metrics(adata)
        # Without nucleus_area, ratio is not computed
        assert "nucleus_to_cell_area_ratio" not in adata.obs.columns


# ---------------------------------------------------------------------------
# uns['transcripts'] sync on cell/gene filtering
# ---------------------------------------------------------------------------


def _adata_with_transcripts(n=4):
    # c2 is the low-count cell that gets filtered out below; g1 is a control
    # gene that gets excluded. "UNASSIGNED" mimics Xenium's real sentinel for
    # transcripts never assigned to any segmented cell.
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(np.array([[5, 0], [4, 1], [1, 0], [3, 2]], dtype=np.float32))
    obs = pd.DataFrame(
        {
            "total_counts": [100.0, 100.0, 2.0, 100.0],
            "n_genes_by_counts": [2, 2, 1, 2],
        },
        index=["c0", "c1", "c2", "c3"],
    )
    var = pd.DataFrame(
        {"is_control_probe": [False, True]},
        index=["g0", "g1"],
    )
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    adata.uns["transcripts"] = pa.table({
        "cell_id": ["c0", "c1", "c2", "c3", "c3", "UNASSIGNED", "UNASSIGNED"],
        "feature_name": ["g0", "g0", "g0", "g0", "g1", "g0", "g1"],
    })
    return adata


class TestFilterCellsSyncsTranscripts:
    def test_inplace_drops_removed_cells_keeps_unassigned(self):
        adata = _adata_with_transcripts()
        filter_cells(adata, min_counts=10, inplace=True)
        assert "c2" not in adata.obs_names
        tx = adata.uns["transcripts"]
        assert "c2" not in tx.column("cell_id").to_pylist()
        # UNASSIGNED rows aren't tied to any cell, so a cell filter must not touch them.
        assert tx.column("cell_id").to_pylist().count("UNASSIGNED") == 2

    def test_non_inplace_syncs_returned_copy_not_original(self):
        adata = _adata_with_transcripts()
        filtered, mask = filter_cells(adata, min_counts=10, inplace=False)
        assert "c2" not in filtered.uns["transcripts"].column("cell_id").to_pylist()
        # Original must be untouched.
        assert "c2" in adata.uns["transcripts"].column("cell_id").to_pylist()

    def test_no_transcripts_key_is_a_noop(self):
        # MERSCOPE/CosMx never populate uns['transcripts'] -- must not crash.
        adata = _adata_with_transcripts()
        del adata.uns["transcripts"]
        filter_cells(adata, min_counts=10, inplace=True)
        assert "transcripts" not in adata.uns


class TestFilterGenesSyncsTranscripts:
    def test_inplace_drops_removed_genes(self):
        adata = _adata_with_transcripts()
        filter_genes(adata, exclude_controls=True, inplace=True)
        assert "g1" not in adata.var_names
        tx = adata.uns["transcripts"]
        assert "g1" not in tx.column("feature_name").to_pylist()
        assert set(tx.column("feature_name").to_pylist()) == {"g0"}

    def test_non_inplace_syncs_returned_copy_not_original(self):
        adata = _adata_with_transcripts()
        filtered, mask = filter_genes(adata, exclude_controls=True, inplace=False)
        assert "g1" not in filtered.uns["transcripts"].column("feature_name").to_pylist()
        assert "g1" in adata.uns["transcripts"].column("feature_name").to_pylist()


class TestFilterIsolatedSyncsTranscripts:
    def test_inplace_drops_removed_cells(self):
        # c0/c1/c3 form a tight mutually-connected trio (distance 10); c2 has
        # no stored neighbor at all (isolated).
        adata = _adata_with_transcripts()
        n = adata.n_obs
        row = np.array([0, 1, 1, 3, 3, 0])
        col = np.array([1, 0, 3, 1, 0, 3])
        data = np.full(6, 10.0)
        adata.obsp["spatial_neighbors_distances"] = sp.csr_matrix(
            (data, (row, col)), shape=(n, n)
        )
        filter_isolated(
            adata,
            neighbors_key="spatial_neighbors",
            max_neighbor_dist_um=50.0,
            min_neighbor_dist_um=50.0,
            inplace=True,
        )
        assert "c2" not in adata.obs_names
        assert "c2" not in adata.uns["transcripts"].column("cell_id").to_pylist()
        assert adata.uns["transcripts"].column("cell_id").to_pylist().count("UNASSIGNED") == 2
