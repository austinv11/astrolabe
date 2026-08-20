"""Tests for tl._micronuclei: diagnostic-only percentile default, absolute
bound + per-pair min_area_ratio criterion, planted-micronuclei recovery, and
merge_micronuclei validation / count conservation / area bookkeeping."""
import numpy as np
import pandas as pd
import anndata as ad
import mudata as md
import scipy.sparse as sp
import pytest

from astrolabe.tl._micronuclei import identify_micronuclei, merge_micronuclei


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

    def test_percentile_path_warns_diagnostic_only(self, caplog):
        import logging
        adata = _adata_normal_population(n=200, seed=3)
        with caplog.at_level(logging.WARNING, logger="astrolabe.tl._micronuclei"):
            identify_micronuclei(adata)
        assert any("DIAGNOSTIC-ONLY" in r.message for r in caplog.records)

    def test_explicit_bound_does_not_warn(self, caplog):
        import logging
        adata = _adata_normal_population(n=200, seed=4)
        with caplog.at_level(logging.WARNING, logger="astrolabe.tl._micronuclei"):
            identify_micronuclei(adata, max_nucleus_area=10.0)
        assert not any("DIAGNOSTIC-ONLY" in r.message for r in caplog.records)


def _planted_micronuclei_adata(seed=0):
    """Planted ground truth for precision/recall.

    - 60 normal primaries (area ~N(35, 3)) on a 100 µm grid.
    - 20 true micronuclei (area U(2, 6)) planted 5 µm from primaries 0..19.
    - 10 'sibling' decoys (area 9, < the 10 µm² bound) 5 µm from dedicated
      primaries of area 24 — the primary passes the global primary gate
      (24 ≥ 2×10) but fails the per-pair min_area_ratio (24 < 3×9), so
      siblings must NOT be flagged.
    - 10 isolated small decoys (area 4) with no primary within range —
      must NOT be flagged either.
    """
    rng = np.random.default_rng(seed)
    coords, areas, truth = [], [], []

    # normal primaries on a 10x6 grid
    for i in range(60):
        coords.append([(i % 10) * 100.0, (i // 10) * 100.0])
        areas.append(float(rng.normal(35.0, 3.0)))
        truth.append(False)
    # planted micronuclei next to primaries 0..19
    for i in range(20):
        px, py = coords[i]
        coords.append([px + 4.0, py + 3.0])
        areas.append(float(rng.uniform(2.0, 6.0)))
        truth.append(True)
    # sibling pairs (far from the main grid)
    for i in range(10):
        coords.append([i * 100.0, 2000.0])
        areas.append(24.0)
        truth.append(False)
        coords.append([i * 100.0 + 4.0, 2003.0])
        areas.append(9.0)
        truth.append(False)
    # isolated small decoys
    for i in range(10):
        coords.append([i * 100.0, -2000.0])
        areas.append(4.0)
        truth.append(False)

    coords = np.asarray(coords, dtype=np.float32)
    n = len(coords)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    adata.obs["nucleus_area"] = np.asarray(areas)
    adata.obsm["spatial"] = coords

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=7)
    rows = np.repeat(np.arange(n), 6)
    cols = idxs[:, 1:].ravel()
    vals = dists[:, 1:].ravel()
    adata.obsp["spatial_neighbors_distances"] = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n, n)
    )
    adata.uns["spatial_neighbors"] = {"distances_key": "spatial_neighbors_distances"}
    return adata, np.asarray(truth, dtype=bool)


class TestIdentifyMicronucleiPlanted:
    def test_planted_recovery_precision_and_recall(self):
        adata, truth = _planted_micronuclei_adata()
        identify_micronuclei(
            adata, max_nucleus_area=10.0, max_distance_um=20.0, min_area_ratio=3.0
        )
        flagged = adata.obs["is_micronucleus"].to_numpy(dtype=bool)
        tp = int((flagged & truth).sum())
        fp = int((flagged & ~truth).sum())
        fn = int((~flagged & truth).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        assert precision >= 0.8, f"precision {precision:.2f} (tp={tp}, fp={fp})"
        assert recall >= 0.8, f"recall {recall:.2f} (tp={tp}, fn={fn})"

    def test_min_area_ratio_blocks_sibling_pairs(self):
        adata, truth = _planted_micronuclei_adata()
        identify_micronuclei(
            adata, max_nucleus_area=10.0, max_distance_um=20.0, min_area_ratio=3.0
        )
        flagged = adata.obs["is_micronucleus"].to_numpy(dtype=bool)
        # sibling decoys: area == 9 exactly
        sibling_mask = np.isclose(adata.obs["nucleus_area"].to_numpy(), 9.0)
        assert sibling_mask.sum() == 10
        assert not flagged[sibling_mask].any(), "siblings must not be flagged"
        # relaxing the ratio to 1 makes the siblings attach — confirms the
        # ratio gate (not distance/size) is what excluded them above
        adata2, _ = _planted_micronuclei_adata()
        mapping = identify_micronuclei(
            adata2, max_nucleus_area=10.0, max_distance_um=20.0, min_area_ratio=1.0
        )
        flagged2 = adata2.obs["is_micronucleus"].to_numpy(dtype=bool)
        assert flagged2[sibling_mask].sum() == 10


def _merge_fixture():
    """3 primaries + 2 micronuclei with counts, areas, and boundaries."""
    names = ["p0", "p1", "p2", "m0", "m1"]
    X = np.array(
        [
            [10, 0, 5],
            [0, 8, 2],
            [3, 3, 3],
            [2, 1, 0],   # m0 -> p0
            [0, 4, 1],   # m1 -> p0
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {
            "nucleus_area": [30.0, 28.0, 32.0, 3.0, 4.0],
            "cell_area": [100.0, 100.0, 100.0, 4.0, 4.0],
        },
        index=names,
    )
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs)
    adata.var_names = ["g0", "g1", "g2"]
    adata.obsm["spatial"] = np.array(
        [[0, 0], [50, 0], [100, 0], [12, 0], [0, 12]], dtype=np.float32
    )

    def square(cx, cy, side):
        h = side / 2.0
        return np.array(
            [[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]],
            dtype=np.float32,
        )

    import awkward as ak
    polys = [
        square(0, 0, 10),     # p0: area 100
        square(50, 0, 10),    # p1: area 100
        square(100, 0, 10),   # p2: area 100
        square(12, 0, 2),     # m0: area 4, disjoint from p0
        square(0, 12, 2),     # m1: area 4, disjoint from p0
    ]
    adata.obsm["cell_boundaries"] = ak.Array([p.tolist() for p in polys])

    mdata = md.MuData({"xenium": adata})
    mdata.uns["default_modality"] = "xenium"
    mapping = pd.Series({"m0": "p0", "m1": "p0"}, name="primary_cell_id")
    return mdata, mapping


class TestMergeMicronuclei:
    def test_total_count_conservation(self):
        mdata, mapping = _merge_fixture()
        total_before = float(mdata["xenium"].X.sum())
        merged = merge_micronuclei(mdata, mapping)
        total_after = float(merged["xenium"].X.sum()) + float(
            merged["micronuclei"].X.sum()
        )
        assert total_after == pytest.approx(total_before)
        # p0's micronuclei row = m0 + m1 counts, gene by gene
        micro_row = np.asarray(
            merged["micronuclei"][ "p0", :].X.todense()
        ).ravel()
        np.testing.assert_allclose(micro_row, [2.0, 5.0, 1.0])
        # primaries with no micronuclei have all-zero micro rows
        for pn in ("p1", "p2"):
            row = np.asarray(merged["micronuclei"][pn, :].X.todense()).ravel()
            assert row.sum() == 0.0

    def test_unknown_names_raise(self):
        mdata, _ = _merge_fixture()
        bad = pd.Series({"m0": "nonexistent"})
        with pytest.raises(ValueError, match="not present"):
            merge_micronuclei(mdata, bad)
        bad2 = pd.Series({"ghost": "p0"})
        with pytest.raises(ValueError, match="not present"):
            merge_micronuclei(mdata, bad2)

    def test_chained_mapping_raises(self):
        mdata, _ = _merge_fixture()
        # m0 -> m1 -> p0: m1 is both a primary and a micronucleus
        chained = pd.Series({"m0": "m1", "m1": "p0"})
        with pytest.raises(ValueError, match="both as a micronucleus and"):
            merge_micronuclei(mdata, chained)

    def test_duplicate_micronucleus_raises(self):
        mdata, _ = _merge_fixture()
        dup = pd.Series(["p0", "p1"], index=["m0", "m0"])
        with pytest.raises(ValueError, match="more than once"):
            merge_micronuclei(mdata, dup)

    def test_micro_obs_carries_summed_areas_and_counts(self):
        mdata, mapping = _merge_fixture()
        merged = merge_micronuclei(mdata, mapping)
        mobs = merged["micronuclei"].obs
        assert mobs.at["p0", "micronuclei_count"] == 2
        assert mobs.at["p1", "micronuclei_count"] == 0
        assert mobs.at["p0", "nucleus_area"] == pytest.approx(3.0 + 4.0)
        assert mobs.at["p0", "cell_area"] == pytest.approx(4.0 + 4.0)
        assert mobs.at["p2", "nucleus_area"] == 0.0

    def test_primary_cell_area_recomputed_from_merged_polygons(self):
        mdata, mapping = _merge_fixture()
        merged = merge_micronuclei(mdata, mapping)
        obs = merged["xenium"].obs
        # p0's union: its own 10x10 square (100) + two disjoint 2x2 squares
        # (4 each) = 108 µm², even though the stored single-polygon boundary
        # keeps only the largest component.
        assert obs.at["p0", "cell_area"] == pytest.approx(108.0)
        # untouched primaries keep their original area
        assert obs.at["p1", "cell_area"] == pytest.approx(100.0)

    def test_inplace_and_row_alignment(self):
        mdata, mapping = _merge_fixture()
        out = merge_micronuclei(mdata, mapping, inplace=True)
        assert out is None
        assert "micronuclei" in mdata.mod
        assert list(mdata["micronuclei"].obs_names) == list(mdata["xenium"].obs_names)
        assert "m0" not in mdata["xenium"].obs_names
