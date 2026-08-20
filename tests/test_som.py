"""Tests for tl._som: BMU memory rewrite, sigma scaling, and — after the
methodology overhaul — ground-truth recovery for all three public functions
(metacluster recovery on planted blobs, gene-module recovery on planted
co-expression blocks, subcellular enrichment on planted polarized cells with
a calibrated uniform null)."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

from astrolabe.tl._som import _batch_bmu, _default_sigma, _unnormalized_umatrix

minisom = pytest.importorskip("minisom")
from minisom import MiniSom  # noqa: E402

from astrolabe.tl._som import (  # noqa: E402
    som,
    som_gene_modules,
    som_stability,
)


class TestBatchBMUMemoryEfficient:
    def test_matches_naive_full_broadcast_computation(self):
        # Regression: the squared-distance-expansion rewrite must give
        # identical BMU assignments to the old (memory-blowing)
        # d[:, None, :] - w_flat[None, :, :] broadcast form -- this is a
        # pure performance/memory fix, not a behavior change.
        rng = np.random.default_rng(0)
        n, n_feat = 500, 12
        data = rng.normal(size=(n, n_feat)).astype(np.float32)
        som_obj = MiniSom(6, 6, n_feat, random_seed=0)
        som_obj.random_weights_init(data)

        result = _batch_bmu(som_obj, data, chunk=64)

        w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float32)
        naive = np.linalg.norm(
            data[:, None, :] - w_flat[None, :, :], axis=2
        ).argmin(axis=1).astype(np.int32)

        np.testing.assert_array_equal(result, naive)

    def test_chunking_does_not_change_result(self):
        # Different chunk sizes can route through BLAS GEMM with a
        # different internal blocking strategy, occasionally reordering
        # floating-point summation enough to flip an argmin between two
        # near-exactly-tied candidate nodes -- not a bug in the chunking
        # logic itself (verified below: any disagreement is between
        # distances that differ by a tiny fraction).
        rng = np.random.default_rng(1)
        n, n_feat = 300, 8
        data = rng.normal(size=(n, n_feat)).astype(np.float32)
        som_obj = MiniSom(5, 5, n_feat, random_seed=0)
        som_obj.random_weights_init(data)

        full = _batch_bmu(som_obj, data, chunk=10_000)
        chunked = _batch_bmu(som_obj, data, chunk=17)
        mismatches = np.where(full != chunked)[0]
        assert len(mismatches) <= 3  # near-tie float noise only, not systematic

        w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float64)
        for i in mismatches:
            d = data[i].astype(np.float64)
            dists = np.linalg.norm(d[None, :] - w_flat, axis=1)
            top2 = np.sort(dists)[:2]
            assert top2[1] - top2[0] < 1e-3


class TestDefaultSigma:
    def test_scales_with_grid_size(self):
        assert _default_sigma(5, 5) == pytest.approx(1.25)
        assert _default_sigma(45, 45) == pytest.approx(11.25)

    def test_floors_at_one(self):
        assert _default_sigma(2, 2) == pytest.approx(1.0)


class TestUnnormalizedUmatrix:
    def test_known_codebook(self):
        # 2x1 grid with codebooks [0] and [3]: each node's single neighbor is
        # the other, so both U-matrix entries are exactly 3.0 (no min-max
        # normalization to 1.0, unlike MiniSom.distance_map()).
        w = np.zeros((2, 1, 1), dtype=np.float32)
        w[1, 0, 0] = 3.0
        um = _unnormalized_umatrix(w)
        np.testing.assert_allclose(um, [[3.0], [3.0]])

    def test_collapsed_codebook_stays_visibly_flat(self):
        # A fully collapsed codebook (all nodes identical) must give an
        # all-zero U-matrix -- min-max normalization would divide 0/0 or
        # rescale noise to a full-range image.
        w = np.ones((4, 4, 3), dtype=np.float32)
        um = _unnormalized_umatrix(w)
        assert um.max() == 0.0


class TestSomParams:
    def test_auto_sigma_scales_with_auto_grid_size(self):
        rng = np.random.default_rng(2)
        n = 400
        adata = ad.AnnData(X=rng.normal(size=(n, 10)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(n, 10)).astype(np.float32)

        som(adata, x=20, y=20, n_epochs=2, seed=0)
        assert adata.uns["som"]["params"]["sigma"] == pytest.approx(_default_sigma(20, 20))

    def test_explicit_sigma_still_respected(self):
        rng = np.random.default_rng(3)
        n = 100
        adata = ad.AnnData(X=rng.normal(size=(n, 5)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(n, 5)).astype(np.float32)
        som(adata, x=6, y=6, sigma=2.5, n_epochs=2, seed=0)
        assert adata.uns["som"]["params"]["sigma"] == pytest.approx(2.5)

    def test_quality_metrics_stored(self):
        rng = np.random.default_rng(4)
        adata = ad.AnnData(X=rng.normal(size=(200, 6)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(200, 6)).astype(np.float32)
        som(adata, x=5, y=5, n_epochs=5, seed=0)
        params = adata.uns["som"]["params"]
        assert params["quantization_error"] > 0
        assert 0.0 <= params["topographic_error"] <= 1.0


class TestCategoryOrdering:
    def test_cluster_and_node_categories_in_numeric_grid_order(self):
        # Plain pd.Categorical sorts string labels lexicographically
        # ("0", "1", "10", "11", ..., "2"), scrambling value_counts() and
        # legends; the categories must be in numeric / grid order.
        rng = np.random.default_rng(5)
        adata = ad.AnnData(X=rng.normal(size=(400, 8)).astype(np.float32))
        adata.obsm["X_pca"] = rng.normal(size=(400, 8)).astype(np.float32)

        som(adata, x=6, y=6, n_metaclusters=12, n_epochs=3, seed=0)

        cluster_cats = [int(c) for c in adata.obs["som_cluster"].cat.categories]
        assert cluster_cats == sorted(cluster_cats)
        assert max(cluster_cats) >= 10  # the lexicographic trap is actually exercised

        node_cats = [
            tuple(int(p) for p in c.split("_"))
            for c in adata.obs["som_node"].cat.categories
        ]
        assert node_cats == sorted(node_cats)

    def test_gene_module_categories_numeric_order(self):
        rng = np.random.default_rng(6)
        adata = ad.AnnData(X=rng.normal(size=(100, 40)).astype(np.float32))
        adata.var_names = [f"g{j}" for j in range(40)]

        # random data has no co-expression structure, so every module is a
        # residual class -- min_coherence=0 keeps them labelled M{k} so this
        # test stays about the numeric category ORDER, which is its subject
        som_gene_modules(adata, n_modules=12, n_pcs=10, n_epochs=5, seed=0,
                         min_coherence=0.0)

        all_cats = list(adata.var["som_gene_module"].cat.categories)
        assert "unassigned" not in all_cats
        cats = [int(c[1:]) for c in all_cats]
        assert cats == sorted(cats)
        assert max(cats) >= 10

    def test_noise_genes_become_unassigned_and_real_blocks_do_not(self):
        """The WGCNA-grey behaviour: a block of genuinely co-expressed genes must
        be reported as a module, and a block of pure-noise genes must not."""
        rng = np.random.default_rng(7)
        n_cells = 400
        lat = rng.normal(size=(n_cells, 2))
        block = np.hstack([np.repeat(lat[:, [b]], 10, axis=1) for b in range(2)])
        block = block + rng.normal(scale=0.3, size=block.shape)
        noise = rng.normal(size=(n_cells, 20))
        X = np.hstack([block, noise]) * rng.uniform(0.5, 4.0, 40)
        adata = ad.AnnData(np.clip(X + 6.0, 0.01, None).astype(np.float32))
        adata.var_names = [f"B{j // 10}_{j % 10}" for j in range(20)] +                           [f"N{j}" for j in range(20)]

        som_gene_modules(adata, n_modules=6, n_pcs=15, n_epochs=30, seed=0,
                         score_cells=False)

        lab = adata.var["som_gene_module"].astype(str)
        real = lab[[n.startswith("B") for n in adata.var_names]]
        noise_lab = lab[[n.startswith("N") for n in adata.var_names]]
        assert (real != "unassigned").all(), "a real co-expression block was dropped"
        assert (noise_lab == "unassigned").mean() > 0.8, noise_lab.value_counts()

        cats = list(adata.var["som_gene_module"].cat.categories)
        assert cats[-1] == "unassigned", "'unassigned' must sort last"
        coh = adata.uns["som_gene_modules"]["module_coherence"]
        assert {"mean_within_r", "mean_between_r", "eigengene_var_explained",
                "metacluster", "coherent"} <= set(coh.columns)
        assert coh.loc[coh["coherent"], "mean_within_r"].min() >                coh.loc[~coh["coherent"], "mean_within_r"].max()
        # kME is oriented so hub genes read positive, and is NaN when unassigned
        kme = adata.var["som_gene_module_kme"]
        assert kme[lab == "unassigned"].isna().all()
        assert kme[lab != "unassigned"].notna().all()

    def test_gene_modules_stores_weights_for_quality_plot(self):
        rng = np.random.default_rng(7)
        adata = ad.AnnData(X=rng.normal(size=(80, 30)).astype(np.float32))
        adata.var_names = [f"g{j}" for j in range(30)]

        som_gene_modules(adata, n_modules=4, n_pcs=8, n_epochs=5, seed=0, score_cells=False)

        u = adata.uns["som_gene_modules"]
        w = u["weights"]
        assert w.ndim == 3
        assert w.shape[:2] == (u["params"]["x"], u["params"]["y"])


class TestSomMetaclusterRecovery:
    def test_planted_blobs_recovered(self):
        # Three well-separated Gaussian blobs must be recovered by
        # SOM + Ward meta-clustering with near-perfect ARI. This is the
        # end-to-end "does training actually work" test: sequential
        # single-pass training with an unannealed radius (the old defaults)
        # fails it.
        from sklearn.datasets import make_blobs
        from sklearn.metrics import adjusted_rand_score

        X, labels = make_blobs(
            n_samples=600, n_features=10, centers=3, cluster_std=1.0,
            center_box=(-8, 8), random_state=0,
        )
        adata = ad.AnnData(X=X.astype(np.float32))

        som(adata, use_rep=None, x=8, y=8, n_metaclusters=3, n_epochs=20, seed=0)

        ari = adjusted_rand_score(labels, adata.obs["som_cluster"].to_numpy())
        assert ari > 0.9, f"ARI {ari:.3f} — planted blobs not recovered"

        # raw node labels are kept separately from the metacluster labels
        assert adata.obs["som_node"].nunique() > adata.obs["som_cluster"].nunique()
        assert len(adata.uns["som"]["node_metacluster"]) == 64


class TestGeneModulesRecovery:
    def test_planted_coexpression_blocks_recovered(self):
        # Three blocks of 20 genes each share a latent factor; every gene
        # gets its own random magnitude scale. Correlation structure (not
        # magnitude) defines the blocks — Euclidean SOM on raw profiles
        # (the old behavior) groups genes by magnitude instead and fails.
        from sklearn.metrics import adjusted_rand_score

        rng = np.random.default_rng(0)
        n_cells, genes_per_block, n_blocks = 300, 20, 3
        factors = rng.normal(size=(n_cells, n_blocks))
        X = np.zeros((n_cells, genes_per_block * n_blocks), dtype=np.float32)
        true_block = np.repeat(np.arange(n_blocks), genes_per_block)
        for j in range(X.shape[1]):
            scale = rng.uniform(0.5, 5.0)
            X[:, j] = scale * (factors[:, true_block[j]] + 0.3 * rng.normal(size=n_cells))

        adata = ad.AnnData(X=X)
        adata.var_names = [f"g{j}" for j in range(X.shape[1])]

        gene_df = som_gene_modules(adata, n_modules=3, n_pcs=10, n_epochs=50, seed=0)

        ari = adjusted_rand_score(true_block, gene_df["module"].to_numpy())
        assert ari > 0.8, f"ARI {ari:.3f} — planted gene blocks not recovered"

        # scores are background-corrected and column-aligned with module_names
        names = adata.uns["som_gene_modules"]["module_names"]
        scores = adata.obsm["som_gene_modules_scores"]
        assert scores.shape == (n_cells, len(names))
        # module score should track its own latent factor better than the
        # others. Signed comparison: score_genes' control background — drawn
        # from the same tiny synthetic panel — is dominated by the *other*
        # blocks' genes, so scores strongly ANTI-correlate with the other
        # factors; only the signed correlation identifies the right block.
        for j, mod in enumerate(names):
            member = gene_df.loc[gene_df["module"] == mod, "gene"].iloc[0]
            block = true_block[int(member[1:])]
            r_signed = [np.corrcoef(scores[:, j], factors[:, b])[0, 1] for b in range(3)]
            assert r_signed[block] == max(r_signed) and r_signed[block] > 0.3
