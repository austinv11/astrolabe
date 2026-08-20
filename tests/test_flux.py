"""Ground-truth and calibration tests for the RNAflux / fluxmap port.

The calibration tests matter more than the recovery tests here. A compositional
decomposition of sparse count data will *always* return components and a SOM will
always return domains; the only thing that distinguishes signal from a decomposed
one-hot is whether the domains survive destroying the gene-position association
while changing nothing else. So:

* `TestPermutationControl` shuffles gene labels within each cell and requires the
  recovered partition to collapse to chance.
* `TestLowPlex` runs at ~60 transcripts per cell over ~400 genes -- the regime a
  targeted panel actually presents -- rather than the comfortable dense case.
"""
import anndata as ad
import awkward as ak
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("minisom")

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from astrolabe.tl._flux import flux, fluxmap  # noqa: E402


def _cells(
    *, n_cells=14, per_cell=1800, n_genes=30, seed=0, split=True, r=9.0,
    n_groups=0, bg_frac=0.4,
):
    """Circular cells whose left and right halves differ in composition.

    ``split=False`` makes the composition uniform across the cell, so no
    compositional domain exists to find. ``n_groups`` adds a ``var`` column
    assigning genes to that many groups, for the module-space path.
    """
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    third = max(2, n_genes // 3)
    left = np.arange(0, third)
    right = np.arange(third, 2 * third)
    bg = np.arange(2 * third, n_genes)

    cx = rng.uniform(0, 400, n_cells)
    cy = rng.uniform(0, 400, n_cells)
    theta = np.linspace(0, 2 * np.pi, 41)[:-1]
    polys = [
        np.column_stack([cx[i] + r * np.cos(theta), cy[i] + r * np.sin(theta)])
        for i in range(n_cells)
    ]
    nuc = [
        np.column_stack([cx[i] + 0.35 * r * np.cos(theta),
                         cy[i] + 0.35 * r * np.sin(theta)])
        for i in range(n_cells)
    ]

    rows = []
    for i in range(n_cells):
        rr = np.sqrt(rng.random(per_cell)) * r * 0.97
        ph = rng.random(per_cell) * 2 * np.pi
        dx, dy = rr * np.cos(ph), rr * np.sin(ph)
        g = np.empty(per_cell, dtype=int)
        is_bg = rng.random(per_cell) < bg_frac
        g[is_bg] = rng.choice(bg, is_bg.sum())
        if split:
            side = dx >= 0
            pick = ~is_bg & side
            g[pick] = rng.choice(right, pick.sum())
            pick = ~is_bg & ~side
            g[pick] = rng.choice(left, pick.sum())
        else:
            pick = ~is_bg
            g[pick] = rng.choice(np.r_[left, right], pick.sum())
        rows.append(pd.DataFrame({
            "cell_id": f"c{i}", "x_location": cx[i] + dx,
            "y_location": cy[i] + dy, "feature_name": genes[g],
        }))
    tx = pd.concat(rows, ignore_index=True)

    names = [f"c{i}" for i in range(n_cells)]
    var = pd.DataFrame(index=genes)
    if n_groups:
        # groups that RESPECT the planted structure: left markers, right markers,
        # then the background split across the remaining groups
        lab = np.array(["bg"] * n_genes, dtype=object)
        lab[left] = "L"
        lab[right] = "R"
        extra = max(1, n_groups - 2)
        for j, gi in enumerate(bg):
            lab[gi] = f"bg{j % extra}"
        var["module"] = lab
    adata = ad.AnnData(
        np.zeros((n_cells, n_genes), dtype=np.float32),
        obs=pd.DataFrame(index=names), var=var,
    )
    adata.obsm["spatial"] = np.column_stack([cx, cy]).astype(np.float32)
    adata.obsm["cell_boundaries"] = ak.Array(polys)
    adata.obsm["nucleus_boundaries"] = ak.Array(nuc)
    return adata, tx


def _recovery_ari(adata, out):
    """ARI between the recovered domains and the planted left/right split."""
    cent = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names,
                        columns=["cx", "cy"])
    j = out.join(cent, on="cell_id")
    truth = ((j["x"] - j["cx"]) >= 0).to_numpy()
    lab = out["domain"].astype(str).to_numpy()
    keep = lab != "background"
    return adjusted_rand_score(truth[keep], lab[keep])


def _permute_within_cells(tx, seed=0):
    """Shuffle gene labels within each cell; positions and counts untouched."""
    rng = np.random.default_rng(seed)
    cells = pd.Categorical(tx["cell_id"].astype(str)).codes.astype(np.int64)
    genes = tx["feature_name"].to_numpy()
    by_cell = np.argsort(cells, kind="stable")
    randomized = np.lexsort((rng.random(len(cells)), cells))
    out = genes.copy()
    out[by_cell] = genes[randomized]
    perm = tx.copy()
    perm["feature_name"] = out
    return perm


class TestDensityGate:
    def test_refuses_when_evidence_is_too_thin(self):
        adata, tx = _cells(n_cells=8, per_cell=40, seed=1)
        with pytest.raises(ValueError, match="too little evidence"):
            flux(adata, tx, sigma_um=0.6)

    def test_message_names_the_ways_out(self):
        adata, tx = _cells(n_cells=8, per_cell=40, seed=1)
        try:
            flux(adata, tx, sigma_um=0.6)
        except ValueError as e:
            msg = str(e)
        assert "gene_groups=" in msg
        assert "sigma_um" in msg
        assert "subcellular_enrichment" in msg

    def test_force_overrides(self):
        adata, tx = _cells(n_cells=8, per_cell=40, seed=1)
        out = flux(adata, tx, sigma_um=0.6, force=True)
        assert len(out) > 0
        assert adata.uns["flux"]["params"]["forced"] is True

    def test_gate_is_on_effective_size(self):
        adata, tx = _cells(n_cells=10, per_cell=1500, seed=2)
        flux(adata, tx, sigma_um=2.0)
        p = adata.uns["flux"]["params"]
        assert p["median_n_eff"] >= p["min_neighborhood_transcripts"]
        # n_eff of a Gaussian-weighted sample is strictly below the raw count
        assert (adata.uns["flux"]["params"]["median_n_eff"] > 0)


class TestGeneGroups:
    def test_reduces_the_composition_space(self):
        adata, tx = _cells(n_cells=10, per_cell=1200, n_genes=60, seed=3,
                           n_groups=6)
        flux(adata, tx, sigma_um=2.0, gene_groups="module", key_added="g")
        u = adata.uns["g"]
        assert u["params"]["n_genes"] == 60
        assert u["params"]["n_groups"] < 60
        assert set(u["group_names"]) <= set(adata.var["module"].astype(str))

    def test_accepts_an_explicit_mapping(self):
        adata, tx = _cells(n_cells=8, per_cell=1200, n_genes=30, seed=4)
        mapping = {g: ("even" if i % 2 == 0 else "odd")
                   for i, g in enumerate(adata.var_names)}
        flux(adata, tx, sigma_um=2.0, gene_groups=mapping, key_added="m")
        assert sorted(adata.uns["m"]["group_names"]) == ["even", "odd"]

    def test_unknown_genes_fall_into_unassigned_last(self):
        adata, tx = _cells(n_cells=8, per_cell=1200, n_genes=30, seed=5)
        mapping = {adata.var_names[0]: "A"}
        flux(adata, tx, sigma_um=2.0, gene_groups=mapping, key_added="m")
        names = adata.uns["m"]["group_names"]
        assert names[-1] == "unassigned"

    def test_missing_var_column_raises(self):
        adata, tx = _cells(n_cells=6, per_cell=600, seed=6)
        with pytest.raises(KeyError, match="som_gene_modules"):
            flux(adata, tx, sigma_um=2.0, gene_groups="not_a_column")


class TestNeighbourhoodIsPerCell:
    def test_touching_cells_do_not_bleed(self):
        """Two cells overlapping in space with disjoint transcriptomes: the
        deviation of every pixel must stay finite and the embedding must not be
        driven by the neighbour's transcripts."""
        rng = np.random.default_rng(7)
        theta = np.linspace(0, 2 * np.pi, 41)[:-1]
        r = 8.0
        centres = np.array([[0.0, 0.0], [9.0, 0.0]])  # closer than 2r: overlapping
        polys = [np.column_stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta)])
                 for c in centres]
        nuc = [np.column_stack([c[0] + 3 * np.cos(theta), c[1] + 3 * np.sin(theta)])
               for c in centres]
        rows = []
        for i, c in enumerate(centres):
            n = 900
            rr = np.sqrt(rng.random(n)) * r * 0.95
            ph = rng.random(n) * 2 * np.pi
            g = np.where(rng.random(n) < 0.5, f"A{i}", f"B{i}")
            rows.append(pd.DataFrame({
                "cell_id": f"c{i}",
                "x_location": c[0] + rr * np.cos(ph),
                "y_location": c[1] + rr * np.sin(ph),
                "feature_name": g,
            }))
        tx = pd.concat(rows, ignore_index=True)
        adata = ad.AnnData(
            np.zeros((2, 4), dtype=np.float32),
            obs=pd.DataFrame(index=["c0", "c1"]),
            var=pd.DataFrame(index=["A0", "B0", "A1", "B1"]),
        )
        adata.obsm["spatial"] = centres.astype(np.float32)
        adata.obsm["cell_boundaries"] = ak.Array(polys)
        adata.obsm["nucleus_boundaries"] = ak.Array(nuc)

        out = flux(adata, tx, sigma_um=2.0)
        fcols = [c for c in out.columns if c.startswith("flux_")]
        assert np.isfinite(out[fcols].to_numpy()).all()
        # a cell whose own composition is uniform has ~no internal deviation:
        # if the neighbour leaked in, pixels on the facing side would deviate
        j = out.join(pd.DataFrame(centres, index=adata.obs_names,
                                  columns=["cx", "cy"]), on="cell_id")
        facing = ((j.cell_id.astype(str) == "c0") & (j["x"] > j["cx"])) | \
                 ((j.cell_id.astype(str) == "c1") & (j["x"] < j["cx"]))
        away = ~facing
        mag = np.linalg.norm(out[fcols].to_numpy(), axis=1)
        assert mag[facing.to_numpy()].mean() < 3.0 * mag[away.to_numpy()].mean()


class TestShrinkage:
    @staticmethod
    def _evidence_ratio(emb):
        """mean |deviation| of the least-evidenced quartile over the most."""
        f = [c for c in emb.columns if c.startswith("flux_")]
        mag = np.linalg.norm(emb[f].to_numpy(), axis=1)
        lo = emb["n_eff"] <= np.percentile(emb["n_eff"], 25)
        hi = emb["n_eff"] >= np.percentile(emb["n_eff"], 75)
        return mag[lo].mean() / mag[hi].mean()

    def test_prior_damps_low_evidence_pixels_relative_to_well_evidenced_ones(self):
        """The claim is *differential*, not global.

        Without a prior, the pixels with least evidence carry the *largest*
        deviations -- they are the noisiest, so they shout loudest, which is
        exactly backwards. The scaler that follows re-normalizes each column, so
        a prior that damped everything equally would be invisible; what has to
        move is the ratio between low- and high-evidence pixels.
        """
        adata, tx = _cells(n_cells=10, per_cell=400, n_genes=20, seed=8)
        ratios = {}
        for prior in (0.0, 5.0, 20.0):
            emb = flux(adata, tx, sigma_um=1.2, prior_strength=prior,
                       min_neighborhood_transcripts=0,
                       min_transcripts_per_group=0, key_added=f"p{prior}")
            ratios[prior] = self._evidence_ratio(emb)
        assert ratios[0.0] > 1.15, (
            f"expected noisy pixels to dominate without a prior, got "
            f"{ratios[0.0]:.3f}"
        )
        assert ratios[5.0] < ratios[0.0], ratios
        assert ratios[20.0] < ratios[5.0], ratios

    def test_prior_strength_recorded(self):
        adata, tx = _cells(n_cells=6, per_cell=800, seed=9)
        flux(adata, tx, sigma_um=2.0, prior_strength=3.5)
        assert adata.uns["flux"]["params"]["prior_strength"] == 3.5


class TestDenseRecovery:
    @pytest.fixture(scope="class")
    def mapped(self):
        adata, tx = _cells(n_cells=14, per_cell=2000, seed=10)
        emb = flux(adata, tx, sigma_um=2.0)
        out = fluxmap(adata, emb, n_domains=2, x=6, y=6, n_epochs=10, seed=0)
        return adata, out

    def test_recovers_the_planted_partition(self, mapped):
        adata, out = mapped
        ari = _recovery_ari(adata, out)
        assert ari > 0.5, f"planted halves not recovered (ARI = {ari:.3f})"

    def test_stored_outputs(self, mapped):
        adata, out = mapped
        u = adata.uns["fluxmap"]
        assert len(u["domain_names"]) == u["params"]["n_domains_found"]
        assert u["domain_profiles"].shape[0] == len(u["domain_names"])
        frac = adata.obsm["fluxmap_fractions"]
        assert frac.shape == (adata.n_obs, len(u["domain_names"]))
        np.testing.assert_allclose(np.nansum(frac, axis=1), 1.0, atol=1e-5)
        assert u["params"]["quantization_error"] > 0
        assert 0.0 <= u["params"]["topographic_error"] <= 1.0

    def test_domain_polygons_per_cell(self, mapped):
        adata, _ = mapped
        for i in range(1, len(adata.uns["fluxmap"]["domain_names"]) + 1):
            key = f"fluxmap{i}_boundaries"
            assert key in adata.obsm
            arr = adata.obsm[key]
            assert len(arr) == adata.n_obs
            assert sum(len(v) > 0 for v in arr) > 0, f"{key} empty everywhere"

    def test_rejects_a_frame_without_flux_columns(self, mapped):
        adata, out = mapped
        with pytest.raises(ValueError, match="flux_"):
            fluxmap(adata, out[["cell_id", "x", "y", "n_eff"]])


def _subtle(*, n_cells=260, per_cell=60, n_genes=400, n_groups=8, delta=0.6,
            seed=0, r=7.0):
    """Low-plex cells whose halves differ by a *proportional shift*, not markers.

    The marker-set construction in :func:`_cells` makes the sparse regime look
    far easier than it is: when the left half draws only from genes L and the
    right only from genes R, a single transcript identifies the domain, so a
    one-hot neighbourhood is fully informative. Real subcellular domains differ
    by a modest shift spread over many genes with nothing exclusive to either,
    which is what this builds.
    """
    rng = np.random.default_rng(seed)
    genes = np.array([f"G{i:03d}" for i in range(n_genes)])
    base = 1.0 / (np.arange(1, n_genes + 1) ** 1.1)
    base /= base.sum()
    up = rng.random(n_genes) < 0.5
    pA = base * np.where(up, 1.0 + delta, 1.0); pA /= pA.sum()
    pB = base * np.where(up, 1.0, 1.0 + delta); pB /= pB.sum()

    cx = rng.uniform(0, 400, n_cells); cy = rng.uniform(0, 400, n_cells)
    th = np.linspace(0, 2 * np.pi, 41)[:-1]
    polys = [np.column_stack([cx[i] + r * np.cos(th), cy[i] + r * np.sin(th)])
             for i in range(n_cells)]
    nuc = [np.column_stack([cx[i] + .35 * r * np.cos(th),
                            cy[i] + .35 * r * np.sin(th)])
           for i in range(n_cells)]
    rows = []
    for i in range(n_cells):
        rr = np.sqrt(rng.random(per_cell)) * r * 0.97
        ph = rng.random(per_cell) * 2 * np.pi
        dx, dy = rr * np.cos(ph), rr * np.sin(ph)
        right = dx >= 0
        g = np.empty(per_cell, dtype=int)
        g[right] = rng.choice(n_genes, right.sum(), p=pA)
        g[~right] = rng.choice(n_genes, (~right).sum(), p=pB)
        rows.append(pd.DataFrame({
            "cell_id": f"c{i}", "x_location": cx[i] + dx,
            "y_location": cy[i] + dy, "feature_name": genes[g]}))
    tx = pd.concat(rows, ignore_index=True)

    lab = np.where(up, "up", "dn").astype(object)
    for j in range(n_genes):
        lab[j] = f"{lab[j]}{j % max(1, n_groups // 2)}"
    adata = ad.AnnData(
        np.zeros((n_cells, n_genes), dtype=np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame({"module": lab}, index=genes),
    )
    adata.obsm["spatial"] = np.column_stack([cx, cy]).astype(np.float32)
    adata.obsm["cell_boundaries"] = ak.Array(polys)
    adata.obsm["nucleus_boundaries"] = ak.Array(nuc)
    return adata, tx


class TestLowPlex:
    """~60 transcripts per cell over ~400 genes, differing by a subtle shift.

    Measured behaviour this pins down (mean over three SOM seeds): gene space
    recovers ~0.03 -- nothing -- while module space recovers ~0.14 with
    seed-to-seed agreement ~0.4. That is a real detection, not a clean
    segmentation, and the thresholds below are set to what is actually achieved
    rather than to what would be nice.
    """

    def test_gene_space_is_refused(self):
        adata, tx = _subtle()
        with pytest.raises(ValueError, match="per composition group"):
            flux(adata, tx, sigma_um=2.5)

    def test_module_space_is_allowed(self):
        adata, tx = _subtle()
        flux(adata, tx, sigma_um=2.5, gene_groups="module")
        p = adata.uns["flux"]["params"]
        assert p["median_n_eff_per_group"] >= p["min_transcripts_per_group"]

    def test_module_space_beats_gene_space(self):
        """The central claim: reducing the composition space is what makes a
        sparse panel workable."""
        adata, tx = _subtle(seed=3)
        rec = {}
        for label, kw in (("genes", {}), ("modules", {"gene_groups": "module"})):
            emb = flux(adata, tx, sigma_um=2.5, min_neighborhood_transcripts=0,
                       min_transcripts_per_group=0, key_added=f"f_{label}", **kw)
            out = fluxmap(adata, emb, n_domains=2, x=6, y=6, n_epochs=15, seed=0,
                          polygons=False, key_added=f"m_{label}")
            rec[label] = _recovery_ari(adata, out)
        assert rec["modules"] > 2 * max(rec["genes"], 0.01), rec
        assert rec["modules"] > 0.06, rec

    def test_seed_reproducibility_is_moderate(self):
        adata, tx = _subtle(seed=1)
        emb = flux(adata, tx, sigma_um=2.5, gene_groups="module")
        labs = []
        for s in (0, 1, 2):
            out = fluxmap(adata, emb, n_domains=2, x=6, y=6, n_epochs=15, seed=s,
                          polygons=False, key_added=f"fm{s}")
            labs.append(out["domain"].astype(str).to_numpy())
        aris = [adjusted_rand_score(labs[i], labs[j])
                for i, j in ((0, 1), (0, 2), (1, 2))]
        # honest bound: low-plex domains move between seeds, and the docstring
        # says so. This guards against a regression to chance, not to perfection.
        assert np.mean(aris) > 0.20, f"seed agreement collapsed: {aris}"


class TestPermutationControl:
    """The test that decides whether a domain map means anything.

    Shuffling gene labels within each cell preserves every cell's geometry, its
    transcript count, each gene's abundance and the whole spatial point pattern.
    It destroys only the association between gene identity and position -- which
    is the entire signal a compositional domain is supposed to represent.
    """

    def test_dense_case_collapses_under_permutation(self):
        adata, tx = _cells(n_cells=14, per_cell=2000, seed=11)
        real = fluxmap(adata, flux(adata, tx, sigma_um=2.0),
                       n_domains=2, x=6, y=6, n_epochs=10, seed=0,
                       polygons=False, key_added="real")
        perm_tx = _permute_within_cells(tx, seed=0)
        perm = fluxmap(adata, flux(adata, perm_tx, sigma_um=2.0, key_added="pf"),
                       n_domains=2, x=6, y=6, n_epochs=10, seed=0,
                       polygons=False, key_added="perm")
        ari_real = _recovery_ari(adata, real)
        ari_perm = _recovery_ari(adata, perm)
        assert ari_real > 0.5, f"real signal not recovered ({ari_real:.3f})"
        assert ari_perm < 0.05, (
            f"permuted data still yields the planted partition "
            f"(ARI = {ari_perm:.3f}) -- the decomposition is describing noise"
        )

    def test_low_plex_case_collapses_under_permutation(self):
        """The decisive test for the low-plex claim.

        Module-space flux recovers only ~0.14 here, which is weak enough that it
        could plausibly be an artifact of decomposing sparse counts. It is not:
        the same pipeline on permuted labels returns ~0.000.
        """
        adata, tx = _subtle(seed=2)
        real = fluxmap(
            adata, flux(adata, tx, sigma_um=2.5, gene_groups="module",
                        key_added="rf"),
            n_domains=2, x=6, y=6, n_epochs=15, seed=0, polygons=False,
            key_added="real",
        )
        perm = fluxmap(
            adata,
            flux(adata, _permute_within_cells(tx, seed=0), sigma_um=2.5,
                 gene_groups="module", key_added="pf"),
            n_domains=2, x=6, y=6, n_epochs=15, seed=0, polygons=False,
            key_added="perm",
        )
        ari_real, ari_perm = _recovery_ari(adata, real), _recovery_ari(adata, perm)
        assert abs(ari_perm) < 0.03, (
            f"permuted low-plex data yields the planted partition "
            f"(ARI = {ari_perm:.3f}) -- the decomposition is describing noise"
        )
        assert ari_real > 5 * abs(ari_perm), (ari_real, ari_perm)


class TestNoStructureNoDomains:
    def test_uniform_composition_gives_no_leading_split(self):
        adata, tx = _cells(n_cells=14, per_cell=2000, seed=12, split=False)
        out = flux(adata, tx, sigma_um=2.0)
        cent = pd.DataFrame(adata.obsm["spatial"], index=adata.obs_names,
                            columns=["cx", "cy"])
        j = out.join(cent, on="cell_id")
        side = (j["x"] - j["cx"]) >= 0
        f0 = out["flux_0"].to_numpy()
        d = abs(f0[side].mean() - f0[~side].mean()) / f0.std()
        assert d < 0.3, f"spurious split on uniform composition (d = {d:.2f})"


class TestDomainPolygonsCoverTheirCells:
    """A cell's pixels of one domain are normally several disconnected blobs.

    An earlier version dissolved them with ``_shapes.merge_polygons``, which is
    documented to return the largest component only -- so every other blob was
    dropped and a domain map drew a few isolated rectangles instead of the
    region the method actually found.
    """

    def test_domains_tile_the_cell(self):
        from shapely.geometry import Polygon

        adata, tx = _subtle(n_cells=40, per_cell=200, n_genes=40, n_groups=6)
        emb = flux(adata, tx, sigma_um=1.2, force=True)
        fluxmap(adata, emb, n_domains=3, seed=0)

        cell_area = sum(
            Polygon(np.asarray(
                ak.to_regular(adata.obsm["cell_boundaries"][i]))[:, :2]).area
            for i in range(adata.n_obs)
        )

        dom_area = 0.0
        for k in range(len(adata.uns["fluxmap"]["domain_names"])):
            arr = adata.obsm[f"fluxmap{k + 1}_boundaries"]
            assert arr.ndim >= 4, "expected a list of rings per cell"
            for i in range(len(arr)):
                for ring in arr[i]:
                    r = np.asarray(ring)
                    if r.ndim == 2 and len(r) >= 3:
                        dom_area += Polygon(r[:, :2]).area

        # the pixel-square covering slightly overhangs the polygon edge, so the
        # ratio sits a little above 1 rather than exactly at it
        assert 0.85 < dom_area / cell_area < 1.25, dom_area / cell_area

    def test_multipart_boundaries_render(self):
        """The renderer must draw every ring, not silently skip nested arrays."""
        from astrolabe.pl import boundaries

        adata, tx = _subtle(n_cells=30, per_cell=200, n_genes=40, n_groups=6)
        emb = flux(adata, tx, sigma_um=1.2, force=True)
        fluxmap(adata, emb, n_domains=2, seed=0)

        panel = boundaries(adata, boundaries_key="fluxmap1_boundaries",
                           color="#2a78d6", fill_alpha=0.7)
        fig = panel.to_figure()
        drawn = sum(len(c.get_paths()) for a in fig.axes for c in a.collections)
        assert drawn > 30, f"only {drawn} rings drawn"
