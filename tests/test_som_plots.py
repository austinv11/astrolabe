"""Smoke + contract tests for the SOM plotting functions (pl._qc_plots)."""
import matplotlib

matplotlib.use("Agg")  # headless backend: avoid Tk backend flakiness in CI/sandboxes

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("minisom")

from astrolabe.pl._qc_plots import (  # noqa: E402
    som_component_plane,
    som_hits,
    som_quality_plot,
    som_stability_plot,
    som_umatrix,
)
from astrolabe.pl._spatial import FigurePanel  # noqa: E402
from astrolabe.tl._som import som, som_gene_modules, som_stability  # noqa: E402


def _blob_adata(seed=0, n_cells=400, n_genes=12):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=n_cells, centers=3, n_features=n_genes,
                      cluster_std=1.1, random_state=seed)
    adata = ad.AnnData(np.abs(X).astype(np.float32))
    adata.obsm["X_pca"] = X.astype(np.float32)
    return adata


@pytest.fixture(scope="module")
def som_adata():
    adata = _blob_adata()
    som(adata, use_rep="X_pca", x=8, y=8, n_metaclusters=3, n_epochs=10, seed=0)
    return adata


class TestUmatrixHits:
    def test_annotate_defaults_off(self, som_adata):
        for fn in (som_umatrix, som_hits):
            panel = fn(som_adata)
            assert isinstance(panel, FigurePanel)
            assert len(panel.figure.axes[0].texts) == 0

    def test_annotate_labels_every_node(self, som_adata):
        assert len(som_umatrix(som_adata, annotate=True).figure.axes[0].texts) == 64

    def test_umatrix_missing_key_raises(self):
        with pytest.raises(KeyError, match="not_a_key"):
            som_umatrix(_blob_adata(), key="not_a_key")


class TestQualityPlot:
    def test_reports_local_and_global_topology(self, som_adata):
        """The headline statistic must be the LOCAL one.

        A global Spearman over all node pairs is dominated by the overall trend
        and scores high even for a folded map, so the title has to surface both
        and the trustworthiness/continuity pair alongside them.
        """
        panel = som_quality_plot(som_adata)
        title = panel.figure.axes[0].get_title()
        assert "local rho" in title
        assert "global" in title
        assert "trustworthiness" in title and "continuity" in title

    def test_local_radius_changes_the_statistic(self, som_adata):
        t1 = som_quality_plot(som_adata, local_radius=2.0).figure.axes[0].get_title()
        t2 = som_quality_plot(som_adata, local_radius=6.0).figure.axes[0].get_title()
        assert t1 != t2

    def test_works_on_gene_modules_key(self):
        # score_cells=False: sc.tl.score_genes needs enough genes to build
        # matched-expression control bins, and this fixture is deliberately small
        adata = _blob_adata(n_cells=200, n_genes=30)
        som_gene_modules(adata, n_modules=3, n_pcs=10, n_epochs=10, seed=0,
                         score_cells=False, min_coherence=0.0)
        assert isinstance(
            som_quality_plot(adata, key="som_gene_modules"), FigurePanel
        )

    def test_stale_result_without_codebook_raises(self, som_adata):
        adata = som_adata.copy()
        adata.uns["stale"] = {"umatrix": np.zeros((4, 4), dtype=np.float32)}
        with pytest.raises(KeyError, match="weights"):
            som_quality_plot(adata, key="stale")


class TestComponentPlane:
    def test_refuses_an_embedding_trained_som(self):
        """Planes need the codebook in gene space; an X_pca SOM has no genes.

        The gene count must differ from the embedding width or the guard has
        nothing to detect -- which is exactly the case this asserts.
        """
        adata = _blob_adata(n_cells=200, n_genes=12)
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :5]
        som(adata, use_rep="X_pca", x=5, y=5, n_metaclusters=3, n_epochs=5, seed=0)
        with pytest.raises(ValueError, match="use_rep=None"):
            som_component_plane(adata, adata.var_names[0])

    def test_draws_for_a_gene_space_som(self):
        adata = _blob_adata(n_cells=200, n_genes=8)
        adata.var_names = [f"G{i}" for i in range(8)]
        som(adata, use_rep=None, x=5, y=5, n_metaclusters=3, n_epochs=10, seed=0)
        assert isinstance(som_component_plane(adata, "G0"), FigurePanel)


class TestStabilityPlot:
    def test_two_panels_and_plateau_in_title(self):
        adata = _blob_adata(n_cells=300)
        som(adata, use_rep="X_pca", x=6, y=6, n_metaclusters=3, n_epochs=10, seed=0)
        som_stability(adata, k_range=range(2, 6), n_seeds=2, seed=7)
        panel = som_stability_plot(adata)
        assert isinstance(panel, FigurePanel)
        # two data axes (ARI curve, merge heights)
        assert len(panel.figure.axes) == 2
        title = panel.figure._suptitle.get_text()
        assert "at the reported k" in title, title
        st = adata.uns["som_stability"]
        assert st["reference_k"] == 3          # the som() run above used 3
        assert 0.0 <= st["ari_at_reference_k"] <= 1.0
        assert st["flat_to"] >= st["flat_from"]

    def test_missing_result_raises(self, som_adata):
        with pytest.raises(KeyError, match="som_stability"):
            som_stability_plot(som_adata)


class TestStabilityRecoversPlantedK:
    def test_ari_plateau_ends_near_the_planted_cluster_count(self):
        """4 planted blobs: cuts at k <= 4 should reseed identically, and
        reproducibility should fall away once k forces the map to split noise."""
        from sklearn.datasets import make_blobs

        X, _ = make_blobs(n_samples=800, centers=4, n_features=10,
                          cluster_std=0.9, random_state=1)
        adata = ad.AnnData(np.abs(X).astype(np.float32))
        adata.obsm["X_pca"] = X.astype(np.float32)
        som(adata, use_rep="X_pca", x=8, y=8, n_metaclusters=4, n_epochs=20, seed=0)
        res = som_stability(adata, k_range=range(2, 11), n_seeds=3, seed=50)
        ari = dict(zip(res["k"], res["mean_ari"]))
        assert ari[4] > 0.9, f"k=4 (the planted count) unstable: {ari[4]:.3f}"
        assert ari[10] < ari[4], "reproducibility should decay past the planted k"
        st = adata.uns["som_stability"]
        # the reported k was 4, and the ARI stored for it must match the curve
        assert st["reference_k"] == 4
        assert st["ari_at_reference_k"] == pytest.approx(ari[4], abs=1e-6)
        # the flat region is a run of small successive changes, so it must lie
        # inside the tested range and not be a single point
        assert min(ari.values()) <= st["ari_at_reference_k"] <= max(ari.values())
        assert 2 <= st["flat_from"] <= st["flat_to"] <= 10
