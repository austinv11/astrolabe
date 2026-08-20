"""Ground-truth and calibration tests for tl/_subcellular.py.

The module this replaced had a test suite that could not fail: its "uniform gene
stays null" check used 200 transcripts per cell, which put the frame artifact it
was meant to detect four times below its own threshold, and its peripheral-index
assertion was satisfied by an algebraic identity for *any* input whatsoever.

So the calibration tests here are written to be failable on purpose:

* the false-positive check is a **rate at nominal alpha**, run at several
  per-cell transcript counts, because the predecessor's bias scaled as
  ``1/sqrt(n_tx)`` and was worst where cells were sparsest;
* the shared-baseline check asserts a **variance decomposition**, which is the
  statistic that exposed the original defect (76% of the headline figure's
  variance was one baseline common to every gene);
* the probability integral transform is checked for uniformity directly, since
  every downstream null rests on it.
"""
import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import pytest

from astrolabe.tl._subcellular import (
    COMPARTMENTS,
    _cmh,
    subcellular_coords,
    subcellular_enrichment,
    subcellular_indices,
)

# ──────────────────────────────────────────────────────────────────────────────
# synthetic geometry: rotated ellipses with a concentric elliptical nucleus
# ──────────────────────────────────────────────────────────────────────────────
def _make(
    *, n_cells=250, seed=0, pattern=None, per_cell=(80, 160), nucleus_frac=0.45,
    r_cell=(5.0, 9.0), groups=False,
):
    """Build (adata, transcripts) with a known localization per gene.

    ``pattern`` maps gene -> 'uniform' | 'nuclear' | 'peripheral' | 'polar', or
    to a ``{group: kind}`` dict for group-dependent localization.
    """
    rng = np.random.default_rng(seed)
    pattern = pattern or {"U0": "uniform", "U1": "uniform"}
    genes = list(pattern)

    a = rng.uniform(*r_cell, n_cells)
    b = a * rng.uniform(0.6, 1.0, n_cells)
    rot = rng.uniform(0, 2 * np.pi, n_cells)
    cx = rng.uniform(0, 4000, n_cells)
    cy = rng.uniform(0, 4000, n_cells)
    ct, st = np.cos(rot), np.sin(rot)
    theta = np.linspace(0, 2 * np.pi, 33)[:-1]
    grp = np.where(np.arange(n_cells) % 2 == 0, "A", "B")

    def ring(sa, sb):
        out = []
        for i in range(n_cells):
            xx, yy = sa[i] * np.cos(theta), sb[i] * np.sin(theta)
            out.append(np.column_stack([cx[i] + xx * ct[i] - yy * st[i],
                                        cy[i] + xx * st[i] + yy * ct[i]]))
        return ak.Array(out)

    n_per = rng.integers(per_cell[0], per_cell[1], n_cells)
    rows = []
    for i in range(n_cells):
        for g in genes:
            k = max(1, int(n_per[i] / len(genes)))
            kind = pattern[g]
            if isinstance(kind, dict):
                kind = kind[grp[i]]
            if kind == "uniform":
                rr, ph = np.sqrt(rng.random(k)), rng.random(k) * 2 * np.pi
            elif kind == "nuclear":
                rr = np.sqrt(rng.random(k)) * nucleus_frac * 0.8
                ph = rng.random(k) * 2 * np.pi
            elif kind == "peripheral":
                rr = np.sqrt(rng.uniform(0.78, 1.0, k))
                ph = rng.random(k) * 2 * np.pi
            elif kind == "polar":
                # confined to the half-disc on the +major side of the CELL, so
                # detecting it requires the per-cell frame, not the image frame
                rr = np.sqrt(rng.random(k))
                ph = rng.random(k) * np.pi - np.pi / 2
            else:
                raise ValueError(kind)
            xx, yy = rr * a[i] * np.cos(ph), rr * b[i] * np.sin(ph)
            rows.append(pd.DataFrame({
                "cell_id": f"c{i}",
                "x_location": cx[i] + xx * ct[i] - yy * st[i],
                "y_location": cy[i] + xx * st[i] + yy * ct[i],
                "feature_name": g,
            }))
    tx = pd.concat(rows, ignore_index=True)

    names = [f"c{i}" for i in range(n_cells)]
    adata = ad.AnnData(
        np.zeros((n_cells, len(genes)), dtype=np.float32),
        obs=pd.DataFrame(index=names), var=pd.DataFrame(index=genes),
    )
    adata.obsm["spatial"] = np.column_stack([cx, cy]).astype(np.float32)
    adata.obsm["cell_boundaries"] = ring(a, b)
    adata.obsm["nucleus_boundaries"] = ring(a * nucleus_frac, b * nucleus_frac)
    if groups:
        adata.obs["celltype"] = pd.Categorical(grp)
    return adata, tx


def _variance_split(res):
    """(compartment main effect, gene main effect, interaction) as fractions."""
    L = res[[f"log2_vs_pooled_{c}" for c in COMPARTMENTS]].to_numpy()
    tot = L.var()
    dm, gm = L.mean(axis=0), L.mean(axis=1)
    resid = L - dm[None, :] - gm[:, None] + L.mean()
    return dm.var() / tot, gm.var() / tot, resid.var() / tot


# ──────────────────────────────────────────────────────────────────────────────
class TestCoordinate:
    def test_u_spans_the_cell_and_respects_the_nucleus(self):
        adata, tx = _make(n_cells=80, seed=1)
        co = subcellular_coords(adata, tx)
        u = co["u"].to_numpy()
        assert u.min() >= 0.0 and u.max() <= 2.0
        # u < 1 must mean inside the nucleus, by construction of the coordinate
        assert (co.loc[u < 1.0, "d_nucleus_um"] <= 0).all()
        assert (co.loc[u > 1.0, "d_nucleus_um"] >= 0).all()
        # nucleus_frac=0.45 semi-axes -> 0.45^2 of the area
        assert np.isclose((u < 1.0).mean(), 0.45 ** 2, atol=0.03)

    def test_compartments_are_ordered_cuts_on_u(self):
        adata, tx = _make(n_cells=80, seed=2)
        co = subcellular_coords(adata, tx)
        means = co.groupby("compartment", observed=True)["u"].mean()
        assert list(means.reindex(list(COMPARTMENTS)).diff().dropna() > 0) == [True] * 3

    def test_pit_is_uniform_on_uniform_in_polygon_data(self):
        """v ~ Uniform(0,1) is the assumption every null downstream rests on."""
        adata, tx = _make(n_cells=300, seed=3, per_cell=(120, 200))
        co = subcellular_coords(adata, tx)
        v = co["v"].to_numpy()
        assert np.isfinite(v).all()
        assert abs(v.mean() - 0.5) < 0.02, f"E[v] = {v.mean():.4f}"
        assert abs(v.var() - 1 / 12) < 0.006, f"Var[v] = {v.var():.5f}"
        counts, _ = np.histogram(v, bins=10, range=(0, 1))
        frac = counts / counts.sum()
        assert frac.min() > 0.075 and frac.max() < 0.125, frac

    def test_pit_is_uniform_per_cell_regardless_of_geometry(self):
        """The point of the PIT: cells of very different size and elongation
        must all yield Uniform(0,1), so pooling them needs no normalization."""
        adata, tx = _make(n_cells=200, seed=4, per_cell=(150, 250),
                          r_cell=(3.0, 14.0))
        co = subcellular_coords(adata, tx)
        per_cell = co.groupby("cell_id", observed=True)["v"].mean()
        assert abs(per_cell.mean() - 0.5) < 0.02
        # spread across cells should be sampling noise only: Var = 1/(12 n)
        n = co.groupby("cell_id", observed=True).size()
        assert per_cell.var() < 4.0 * float(np.mean(1.0 / (12 * n)))

    def test_schema_and_stored_outputs(self):
        adata, tx = _make(n_cells=60, seed=5, groups=True)
        co = subcellular_coords(adata, tx, groupby="celltype")
        assert set(co.columns) == {
            "cell_id", "gene", "x", "y", "u", "v", "d_nucleus_um",
            "d_membrane_um", "angle", "compartment", "group",
        }
        n_obs = adata.n_obs
        assert adata.obsm["subcellular_compartment_fractions"].shape == (n_obs, 4)
        assert adata.obsm["subcellular_null_fractions"].shape == (n_obs, 4)
        assert adata.obsm["subcellular_null_moments"].shape == (n_obs, 4)
        assert adata.obsm["subcellular_nucleus_centroid"].shape == (n_obs, 2)
        frac = adata.obsm["subcellular_compartment_fractions"]
        np.testing.assert_allclose(np.nansum(frac, axis=1), 1.0, atol=1e-5)
        nullf = adata.obsm["subcellular_null_fractions"]
        np.testing.assert_allclose(np.nansum(nullf, axis=1), 1.0, atol=1e-5)
        assert adata.uns["subcellular"]["compartment_names"] == list(COMPARTMENTS)

    def test_returned_frame_is_saveable(self, tmp_path):
        """Regression: an ndarray in DataFrame.attrs breaks to_parquet, because
        pandas serializes attrs to JSON."""
        adata, tx = _make(n_cells=40, seed=6)
        co = subcellular_coords(adata, tx)
        co.to_parquet(tmp_path / "coords.parquet")

    def test_missing_nucleus_boundaries_raises(self):
        adata, tx = _make(n_cells=30, seed=7)
        del adata.obsm["nucleus_boundaries"]
        with pytest.raises(KeyError, match="nucleus_boundaries"):
            subcellular_coords(adata, tx)

    def test_non_unique_obs_names_raises(self):
        adata, tx = _make(n_cells=30, seed=8)
        adata.obs_names = ["dup"] * adata.n_obs
        with pytest.raises(ValueError, match="unique"):
            subcellular_coords(adata, tx)

    def test_transcripts_default_to_uns(self):
        adata, tx = _make(n_cells=40, seed=9)
        adata.uns["transcripts"] = tx
        explicit = subcellular_coords(adata, tx)
        defaulted = subcellular_coords(adata)
        pd.testing.assert_frame_equal(defaulted, explicit)

    def test_angle_is_in_the_nucleus_anchored_frame(self):
        """A gene confined to one half of each cell must have a concentrated
        angle distribution once bearings are expressed relative to the cell's
        own polarity axis -- and the cells are randomly rotated, so an
        image-frame bearing could not achieve this."""
        adata, tx = _make(n_cells=200, seed=10,
                          pattern={"POLAR": "polar", "U": "uniform"})
        # an off-centre nucleus gives the polarity axis a defined direction
        from astrolabe.pp._qc import calculate_shape_metrics
        calculate_shape_metrics(adata)
        assert "cell_polarity_major_angle" in adata.obs
        co = subcellular_coords(adata, tx)
        ang = co.loc[co["gene"].astype(str) == "POLAR", "angle"].to_numpy()
        ang = ang[np.isfinite(ang)]
        if len(ang) < 100:
            pytest.skip("no cells with a usable polarity axis in this fixture")
        R_polar = float(np.abs(np.mean(np.exp(1j * ang))))
        ang_u = co.loc[co["gene"].astype(str) == "U", "angle"].to_numpy()
        ang_u = ang_u[np.isfinite(ang_u)]
        R_unif = float(np.abs(np.mean(np.exp(1j * ang_u))))
        assert R_polar > R_unif, (R_polar, R_unif)


# ──────────────────────────────────────────────────────────────────────────────
class TestGroundTruthRecovery:
    @pytest.fixture(scope="class")
    def planted(self):
        adata, tx = _make(
            n_cells=400, seed=11, per_cell=(150, 280),
            pattern={"NUC": "nuclear", "PER": "peripheral",
                     "U0": "uniform", "U1": "uniform", "U2": "uniform"},
        )
        co = subcellular_coords(adata, tx)
        res = subcellular_enrichment(adata, co).set_index("gene")
        return adata, co, res

    def test_nuclear_gene_is_called_nuclear(self, planted):
        _, _, res = planted
        assert res.loc["NUC", "p_adj_omnibus"] < 0.01
        assert res.loc["NUC", "cmh_log_or_nucleus"] > 1.0
        assert res.loc["NUC", "median_d"] < -0.5
        assert res.loc["NUC", "frac_nucleus"] > res.loc["U0", "frac_nucleus"]

    def test_peripheral_gene_is_called_peripheral(self, planted):
        _, _, res = planted
        assert res.loc["PER", "p_adj_omnibus"] < 0.01
        assert res.loc["PER", "median_d"] > 0.5
        assert res.loc["PER", "frac_cell_edge"] > res.loc["U0", "frac_cell_edge"]

    def test_directions_are_opposite(self, planted):
        _, _, res = planted
        assert res.loc["NUC", "median_d"] < 0 < res.loc["PER", "median_d"]

    def test_confinement_gives_infinite_not_missing_odds_ratio(self, planted):
        """A gene absent from a compartment has OR 0, not an unknown OR."""
        _, _, res = planted
        assert np.isneginf(res.loc["NUC", "cmh_log_or_cell_edge"])
        assert np.isfinite(res.loc["NUC", "p_cmh_cell_edge"])


# ──────────────────────────────────────────────────────────────────────────────
class TestCalibration:
    @pytest.mark.parametrize("per_cell", [(12, 24), (30, 55), (90, 160)])
    def test_all_uniform_panel_yields_no_calls(self, per_cell):
        """False-positive RATE at nominal alpha, at three per-cell depths.

        The predecessor's frame artifact reached log2 = +0.92 at ~10
        transcripts per cell, so the sparse settings are the ones that matter.
        """
        n_genes = 10
        adata, tx = _make(
            n_cells=600, seed=12, per_cell=per_cell,
            pattern={f"U{i}": "uniform" for i in range(n_genes)},
        )
        co = subcellular_coords(adata, tx)
        res = subcellular_enrichment(
            adata, co, min_gene_transcripts=20, min_cell_gene_transcripts=3,
            min_cells=10,
        )
        assert len(res) == n_genes
        n_sig = int((res["p_adj_omnibus"] < 0.05).sum())
        assert n_sig == 0, f"{n_sig}/{n_genes} false positives at {per_cell}"
        worst = res[[f"log2_vs_pooled_{c}" for c in COMPARTMENTS]].abs().to_numpy().max()
        assert worst < 0.35, f"shared baseline as large as {worst:.3f}"

    def test_no_shared_baseline_under_the_null(self):
        """The E3 regression: on null data the compartment main effect must not
        dominate. The position-binning predecessor scored 75.8% here."""
        adata, tx = _make(
            n_cells=500, seed=13, per_cell=(60, 120),
            pattern={f"U{i}": "uniform" for i in range(8)},
        )
        co = subcellular_coords(adata, tx)
        res = subcellular_enrichment(adata, co)
        comp_main, _, _ = _variance_split(res)
        assert comp_main < 0.25, f"compartment main effect {comp_main:.1%}"

    def test_signal_dominates_when_signal_exists(self):
        """The mirror image: with planted per-gene localization the
        gene x compartment interaction must carry the variance."""
        adata, tx = _make(
            n_cells=400, seed=14, per_cell=(120, 220),
            pattern={"NUC": "nuclear", "PER": "peripheral", "U0": "uniform",
                     "U1": "uniform", "U2": "uniform", "U3": "uniform"},
        )
        co = subcellular_coords(adata, tx)
        res = subcellular_enrichment(adata, co)
        comp_main, _, inter = _variance_split(res)
        assert inter > comp_main
        assert inter > 0.4, f"interaction only {inter:.1%}"

    def test_abundance_does_not_drive_the_odds_ratio(self):
        """Under the null with a wide abundance range the MH log OR must not
        track transcript count -- the real-data correlation is data structure
        (nuclear retention of low-abundance transcripts), not estimator bias,
        and this is the test that separates the two."""
        from scipy.stats import spearmanr

        rng = np.random.default_rng(15)
        n_genes = 24
        w = 1.0 / (np.arange(1, n_genes + 1) ** 1.6)
        w /= w.sum()
        adata, tx = _make(n_cells=900, seed=15, per_cell=(50, 110),
                          pattern={f"U{i:02d}": "uniform" for i in range(n_genes)})
        # re-draw gene labels from a power law, leaving every position untouched
        tx = tx.copy()
        tx["feature_name"] = np.asarray(
            [f"U{i:02d}" for i in range(n_genes)]
        )[rng.choice(n_genes, len(tx), p=w)]
        co = subcellular_coords(adata, tx)
        res = subcellular_enrichment(adata, co, min_gene_transcripts=30,
                                     min_cell_gene_transcripts=3, min_cells=10)
        assert res["n_transcripts"].max() / res["n_transcripts"].min() > 50
        for col in ("cmh_log_or_nucleus", "cmh_log_or_cell_edge"):
            m = np.isfinite(res[col])
            rho = spearmanr(res.loc[m, col], res.loc[m, "n_transcripts"]).statistic
            assert abs(rho) < 0.5, f"{col} tracks abundance: rho={rho:+.3f}"
        assert int((res["p_adj_omnibus"] < 0.05).sum()) == 0


# ──────────────────────────────────────────────────────────────────────────────
class TestGroupby:
    def test_opposite_localization_per_group_cancels_when_pooled(self):
        """The reason groupby= exists: a gene nuclear in one cell type and
        peripheral in another is invisible pooled and obvious per group."""
        adata, tx = _make(
            n_cells=600, seed=16, per_cell=(150, 260), groups=True,
            pattern={"FLIP": {"A": "nuclear", "B": "peripheral"},
                     "U0": "uniform", "U1": "uniform"},
        )
        pooled = subcellular_enrichment(
            adata, subcellular_coords(adata, tx)
        ).set_index("gene")
        co_g = subcellular_coords(adata, tx, groupby="celltype")
        per_g = subcellular_enrichment(adata, co_g, groupby="celltype")
        assert set(per_g["group"]) == {"A", "B"}
        by = per_g.set_index(["gene", "group"])

        dA = by.loc[("FLIP", "A"), "median_d"]
        dB = by.loc[("FLIP", "B"), "median_d"]
        assert dA < -0.4 < 0.4 < dB, (dA, dB)
        # pooled sits between the two, and much closer to zero than either
        assert abs(pooled.loc["FLIP", "median_d"]) < min(abs(dA), abs(dB))

    def test_groupby_without_group_column_raises(self):
        adata, tx = _make(n_cells=60, seed=17, groups=True)
        co = subcellular_coords(adata, tx)  # no groupby -> no 'group' column
        with pytest.raises(ValueError, match="group"):
            subcellular_enrichment(adata, co, groupby="celltype")


# ──────────────────────────────────────────────────────────────────────────────
class TestIndices:
    def test_uniform_fill_gives_unit_indices(self):
        """PDI/DI are ratios against a uniform fill of the same mask, so
        uniformly placed transcripts must give ~1 -- and rdi must give 0.5."""
        adata, tx = _make(n_cells=300, seed=18, per_cell=(120, 200))
        co = subcellular_coords(adata, tx)
        subcellular_indices(adata, co)
        assert abs(np.nanmedian(adata.obs["subcellular_rdi"]) - 0.5) < 0.02
        assert abs(np.nanmedian(adata.obs["subcellular_pdi"]) - 1.0) < 0.10
        assert abs(np.nanmedian(adata.obs["subcellular_di"]) - 1.0) < 0.12

    def test_pdi_orders_nuclear_below_uniform_below_peripheral(self):
        adata, tx = _make(
            n_cells=300, seed=19, per_cell=(150, 260),
            pattern={"NUC": "nuclear", "U": "uniform", "PER": "peripheral"},
        )
        co = subcellular_coords(adata, tx)
        med = {}
        for g in ("NUC", "U", "PER"):
            subcellular_indices(adata, co, genes=[g], min_cell_transcripts=20,
                                key_added=f"ix_{g}")
            med[g] = float(np.nanmedian(adata.obs[f"ix_{g}_pdi"]))
        assert med["NUC"] < 0.5 < med["U"] < med["PER"], med
        assert abs(med["U"] - 1.0) < 0.15

    def test_variance_decomposition_is_stored(self):
        adata, tx = _make(n_cells=250, seed=20, per_cell=(90, 160))
        co = subcellular_coords(adata, tx)
        subcellular_indices(adata, co)
        ix = adata.uns["subcellular_indices"]
        assert ix["tau2"] >= 0.0
        assert ix["mean_inv_n"] > 0.0
        assert ix["n_cells"] > 0
        # on homogeneous synthetic cells there is no real between-cell variation
        assert ix["tau2"] < 3.0 * ix["mean_inv_n"]
        # shrinkage never amplifies
        z = adata.obs["subcellular_rdi_z"].to_numpy()
        sh = adata.obs["subcellular_rdi_shrunk"].to_numpy()
        m = np.isfinite(z) & np.isfinite(sh)
        assert (np.abs(sh[m]) <= np.abs(z[m]) + 1e-6).all()

    def test_low_count_cells_are_excluded_not_guessed(self):
        adata, tx = _make(n_cells=150, seed=21, per_cell=(90, 160))
        co = subcellular_coords(adata, tx)
        subcellular_indices(adata, co, min_cell_transcripts=10_000)
        assert adata.obs["subcellular_pdi"].isna().all()
        assert adata.uns["subcellular_indices"]["n_excluded"] > 0


# ──────────────────────────────────────────────────────────────────────────────
class TestCMH:
    def test_matches_statsmodels_stratified_table(self):
        """The Mantel-Haenszel odds ratio, its Robins-Breslow-Greenland
        standard error and the continuity-corrected statistic, against an
        independent implementation."""
        sm = pytest.importorskip("statsmodels.stats.contingency_tables")
        rng = np.random.default_rng(0)
        n_strata = 40
        tables = []
        a_l, t_l, m1_l, n_l = [], [], [], []
        for _ in range(n_strata):
            a = int(rng.integers(1, 8)); b = int(rng.integers(1, 8))
            c = int(rng.integers(1, 8)); d = int(rng.integers(1, 8))
            tables.append([[a, b], [c, d]])
            a_l.append(a); t_l.append(a + b)
            m1_l.append(a + c); n_l.append(a + b + c + d)
        arr = np.array(tables).transpose(1, 2, 0)  # (2, 2, K) for statsmodels
        ref = sm.StratifiedTable(arr)

        gene_of_pair = np.zeros(n_strata, dtype=np.int64)
        log_or, se, chi2, n_st = _cmh(
            np.array(a_l, float), np.array(t_l, float), np.array(m1_l, float),
            np.array(n_l, float), gene_of_pair, 1,
        )
        assert n_st[0] == n_strata
        np.testing.assert_allclose(log_or[0], np.log(ref.oddsratio_pooled), rtol=1e-10)
        np.testing.assert_allclose(se[0], ref.logodds_pooled_se, rtol=1e-8)
        np.testing.assert_allclose(
            chi2[0], ref.test_null_odds(correction=True).statistic, rtol=1e-8
        )

    def test_absent_gene_strata_contribute_nothing(self):
        """Only occupied (gene, cell) pairs are passed to _cmh; a stratum where
        the gene has no transcripts must be a no-op, which is what makes the
        single-pass vectorization valid."""
        a = np.array([3.0, 2.0, 0.0])
        t = np.array([5.0, 4.0, 0.0])       # third stratum: gene absent
        m1 = np.array([6.0, 5.0, 7.0])
        n = np.array([20.0, 18.0, 25.0])
        g = np.zeros(3, dtype=np.int64)
        full = _cmh(a, t, m1, n, g, 1)
        trimmed = _cmh(a[:2], t[:2], m1[:2], n[:2], g[:2], 1)
        for x, y in zip(full[:3], trimmed[:3]):
            np.testing.assert_allclose(x[0], y[0], rtol=1e-12)


class TestNullProfile:
    """The stored uniform-in-polygon profile, and the figure that divides by it.

    ``u`` is a radial *fraction*, so its area element is not constant: inside the
    nucleus, area grows with ``u``, and uniformly placed transcripts already
    produce a steeply rising curve. Plotting raw density therefore shows a large
    geometric artifact that reads as perinuclear accumulation. These tests pin
    down that dividing by the stored null removes it.
    """

    @staticmethod
    def _ratio(adata, coords, mask=None, n_bins=60):
        from astrolabe.pl._subcellular_plots import _null_density

        u = coords["u"].to_numpy()
        if mask is not None:
            u = u[mask]
        bins = np.linspace(0.0, 2.0, n_bins + 1)
        h, _ = np.histogram(u, bins=bins, density=True)
        d = _null_density(adata, "subcellular", bins)
        r = h / d
        ok = np.isfinite(r)
        return r, ok, h

    def test_profile_is_stored_and_normalized(self):
        adata, tx = _make(n_cells=120, seed=30)
        subcellular_coords(adata, tx)
        prof = np.asarray(adata.uns["subcellular"]["null_profile"])
        assert len(prof) == adata.uns["subcellular"]["null_profile_bins"]
        assert prof.min() >= 0.0
        np.testing.assert_allclose(prof.sum(), 1.0, atol=1e-6)

    def test_uniform_panel_is_flat_when_null_referenced(self):
        adata, tx = _make(n_cells=600, seed=31, per_cell=(150, 260),
                          pattern={f"U{i}": "uniform" for i in range(4)})
        co = subcellular_coords(adata, tx)
        r, ok, h = self._ratio(adata, co)
        w = h[ok]
        mean = float(np.average(r[ok], weights=w))
        rms = float(np.sqrt(np.average((r[ok] - 1.0) ** 2, weights=w)))
        assert abs(mean - 1.0) < 0.06, f"not centred on 1: {mean:.3f}"
        assert rms < 0.25, f"not flat: rms {rms:.3f}"

    def test_raw_density_is_dominated_by_geometry(self):
        """The artifact the reference exists to remove: on uniformly placed
        transcripts the raw density spans two orders of magnitude."""
        adata, tx = _make(n_cells=400, seed=32, per_cell=(150, 260),
                          pattern={f"U{i}": "uniform" for i in range(3)})
        co = subcellular_coords(adata, tx)
        r, ok, h = self._ratio(adata, co)
        span_raw = h.max() / max(h[h > 0].min(), 1e-12)
        span_ref = r[ok].max() / max(r[ok].min(), 1e-12)
        assert span_raw > 20, span_raw
        assert span_ref < span_raw / 5, (span_raw, span_ref)

    def test_planted_genes_depart_from_the_reference(self):
        adata, tx = _make(n_cells=400, seed=33, per_cell=(150, 260),
                          pattern={"NUC": "nuclear", "PER": "peripheral",
                                   "UNI": "uniform"})
        co = subcellular_coords(adata, tx)
        g = co["gene"].astype(str).to_numpy()
        centers = np.linspace(0.0, 2.0, 61)
        centers = 0.5 * (centers[:-1] + centers[1:])
        out = {}
        for name in ("UNI", "NUC", "PER"):
            r, ok, h = self._ratio(adata, co, mask=g == name)
            inner = ok & (centers < 1.0)
            outer = ok & (centers > 1.0)
            out[name] = (
                float(np.average(r[inner], weights=h[inner] + 1e-9)),
                float(np.average(r[outer], weights=h[outer] + 1e-9)),
            )
        assert abs(out["UNI"][0] - 1.0) < 0.15, out["UNI"]
        assert abs(out["UNI"][1] - 1.0) < 0.15, out["UNI"]
        assert out["NUC"][0] > 2.0 and out["NUC"][1] < 0.5, out["NUC"]
        assert out["PER"][1] > 2.0 and out["PER"][0] < 0.5, out["PER"]
