"""Tests for tl.rank_genes_groups / tl.score_genes / tl.annotate_clusters."""
import logging

import numpy as np
import anndata as ad
import mudata as md
import pytest
from scipy import stats

from astrolabe.tl._de import annotate_clusters, rank_genes_groups, score_genes


def _adata_two_groups(n_per=25, n_genes=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(2 * n_per, n_genes)).astype(np.float32)
    # plant a marker for group B in gene 0
    X[n_per:, 0] += rng.poisson(3.0, size=n_per)
    adata = ad.AnnData(X)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs["group"] = ["A"] * n_per + ["B"] * n_per
    return adata


class TestRankGenesGroups:
    def test_smoke(self):
        adata = _adata_two_groups()
        rank_genes_groups(adata, groupby="group")
        assert "rank_genes_groups" in adata.uns
        assert "names" in adata.uns["rank_genes_groups"]

    def test_default_stores_all_genes(self):
        # n_genes=None default: the stored table must cover every gene so a
        # volcano plot has its null cloud and down-regulated arm.
        adata = _adata_two_groups(n_genes=12)
        rank_genes_groups(adata, groupby="group")
        names = adata.uns["rank_genes_groups"]["names"]
        assert len(names) == 12

    def test_explicit_n_genes_truncates(self):
        adata = _adata_two_groups(n_genes=12)
        rank_genes_groups(adata, groupby="group", n_genes=5)
        assert len(adata.uns["rank_genes_groups"]["names"]) == 5

    def test_planted_marker_ranks_first(self):
        adata = _adata_two_groups()
        rank_genes_groups(adata, groupby="group")
        assert adata.uns["rank_genes_groups"]["names"]["B"][0] == "gene_0"

    def test_wilcoxon_matches_scipy_mannwhitneyu(self):
        # tie_correct=True default: scanpy's tie-corrected normal
        # approximation should agree with scipy's asymptotic
        # (tie-corrected, no continuity correction) Mann-Whitney U.
        adata = _adata_two_groups(seed=3)
        rank_genes_groups(adata, groupby="group", groups=["B"])
        res = adata.uns["rank_genes_groups"]
        order = list(res["names"]["B"])
        pvals = dict(zip(order, res["pvals"]["B"]))

        X = adata.X
        grp = adata.obs["group"].values
        for gene in ("gene_0", "gene_3", "gene_7"):
            gi = list(adata.var_names).index(gene)
            x = np.asarray(X[grp == "B", gi]).ravel()
            y = np.asarray(X[grp == "A", gi]).ravel()
            _, p_ref = stats.mannwhitneyu(
                x, y, alternative="two-sided", method="asymptotic", use_continuity=False
            )
            assert pvals[gene] == pytest.approx(p_ref, rel=1e-4, abs=1e-12)

    def test_mirrors_to_mudata_uns(self):
        adata = _adata_two_groups()
        mdata = md.MuData({"xenium": adata})
        mdata.uns["default_modality"] = "xenium"
        rank_genes_groups(mdata, groupby="group")
        assert "rank_genes_groups" in mdata.uns
        assert "xenium" in mdata.uns["rank_genes_groups"]


class TestScoreGenes:
    def test_smoke(self):
        adata = _adata_two_groups()
        score_genes(
            adata, ["gene_0", "gene_1"], score_name="sig", ctrl_size=5, n_bins=3
        )
        assert "sig" in adata.obs
        assert np.isfinite(adata.obs["sig"].values).all()


class TestAnnotateClusters:
    def _adata(self):
        adata = _adata_two_groups()
        adata.obs["leiden"] = (["0"] * 20 + ["1"] * 20 + ["2"] * 10)
        return adata

    def test_basic_mapping(self):
        adata = self._adata()
        annotate_clusters(adata, "leiden", {"0": "T cell", "1": "B cell"})
        vals = adata.obs["leiden_annotation"]
        assert (vals[adata.obs["leiden"] == "0"] == "T cell").all()
        assert (vals[adata.obs["leiden"] == "2"] == "unknown").all()

    def test_unmatched_keys_warn(self, caplog):
        adata = self._adata()
        with caplog.at_level(logging.WARNING):
            annotate_clusters(adata, "leiden", {"0": "T cell", "99": "Ghost"})
        assert "match no cluster" in caplog.text
        assert "'99'" in caplog.text

    def test_all_matched_no_warning(self, caplog):
        adata = self._adata()
        with caplog.at_level(logging.WARNING):
            annotate_clusters(adata, "leiden", {"0": "T cell"})
        assert "match no cluster" not in caplog.text

    def test_unknown_label_collision_counts(self, caplog):
        # A real cell type deliberately named 'unknown' must not be counted
        # as unmapped in the log line.
        adata = self._adata()
        with caplog.at_level(logging.INFO):
            annotate_clusters(
                adata, "leiden", {"0": "unknown", "1": "B cell", "2": "T cell"}
            )
        assert "0 'unknown'" in caplog.text

    def test_missing_groupby_raises(self):
        adata = self._adata()
        with pytest.raises(KeyError, match="nope"):
            annotate_clusters(adata, "nope", {"0": "x"})

    def test_custom_key_added(self):
        adata = self._adata()
        annotate_clusters(adata, "leiden", {"0": "T"}, key_added="cell_type")
        assert "cell_type" in adata.obs
