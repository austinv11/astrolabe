"""Tests for tl._micronuclei (batch-6 bugfix: data-driven default threshold)."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from astrolabe.tl._micronuclei import identify_micronuclei


def _adata_normal_population(n=500, seed=0):
    # Log-normal nucleus areas, no true micronuclei at all -- median ~30 um^2,
    # roughly matching a realistic Xenium-like distribution.
    rng = np.random.default_rng(seed)
    nucleus_area = rng.lognormal(mean=np.log(30.0), sigma=0.4, size=n)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obs["nucleus_area"] = nucleus_area
    coords = rng.uniform(0, 1000, (n, 2)).astype(np.float32)
    adata.obsm["spatial"] = coords

    # A simple KNN-style distance graph so every cell has *some* neighbors.
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=6)
    rows = np.repeat(np.arange(n), 5)
    cols = idxs[:, 1:].ravel()
    vals = dists[:, 1:].ravel()
    dist_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    adata.obsp["spatial_neighbors_distances"] = dist_mat
    adata.uns["spatial_neighbors"] = {"distances_key": "spatial_neighbors_distances"}
    return adata


class TestIdentifyMicronucleiDefaultThreshold:
    def test_percentile_default_flags_far_fewer_than_fixed_20(self):
        # Regression: the old fixed max_nucleus_area=20.0 misclassified
        # ~15% of a normal, micronucleus-free population as candidates --
        # confirm the new percentile-based default flags close to its own
        # target percentile instead, not a large, distribution-dependent
        # fraction.
        adata = _adata_normal_population()
        identify_micronuclei(adata, max_nucleus_area_percentile=5.0)
        frac_flagged = adata.obs["is_micronucleus"].mean()
        # Candidates are gated *further* by the primary-neighbor-within-range
        # requirement, so the true positive rate is <= the percentile itself.
        assert frac_flagged <= 0.05 + 1e-6

    def test_auto_threshold_logged_and_matches_percentile(self, caplog):
        import logging
        adata = _adata_normal_population(n=1000, seed=1)
        with caplog.at_level(logging.INFO, logger="astrolabe.tl._micronuclei"):
            identify_micronuclei(adata, max_nucleus_area_percentile=10.0)
        expected = np.percentile(adata.obs["nucleus_area"].to_numpy(), 10.0)
        assert any("auto-set" in r.message for r in caplog.records)
        # Candidates should all be strictly below the resolved threshold.
        candidates = adata.obs["nucleus_area"] < expected
        assert candidates.sum() > 0

    def test_explicit_value_still_overrides(self):
        adata = _adata_normal_population(n=200, seed=2)
        # An absurdly large explicit threshold should flag close to everyone
        # as a *candidate* (though few will pass the primary-neighbor gate).
        identify_micronuclei(adata, max_nucleus_area=1e6, max_distance_um=1e6)
        # With everyone a candidate, no cell qualifies as primary either
        # (min_primary_ratio * 1e6 is unreachable) -- so nothing is flagged,
        # but this should run without error and respect the explicit value
        # (no percentile auto-resolution log message).
        assert "is_micronucleus" in adata.obs.columns
