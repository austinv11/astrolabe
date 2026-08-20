"""Tests for pl._gradient_plots: flux_plot triangle/gating, the reworked
direction_histogram (magnitude weighting, unreliable exclusion, landmark-
relative mode), and decay_length_plot."""
import matplotlib

matplotlib.use("Agg")  # headless backend: avoid Tk backend flakiness in CI/sandboxes

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from astrolabe.pl._gradient_plots import (
    decay_length_plot,
    diffusion_comparison,
    direction_histogram,
    flux_plot,
)
from astrolabe.pl._spatial import FigurePanel, PlotGrid


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _flux_adata():
    groups = ["A", "B", "C"]
    flux = pd.DataFrame(
        [[np.nan, -2.0, 0.5], [2.0, np.nan, -1.0], [-0.5, 1.0, np.nan]],
        index=groups, columns=groups,
    )
    lengths = pd.DataFrame(
        [[np.nan, 5000.0, 10.0], [5000.0, np.nan, 8000.0], [10.0, 8000.0, np.nan]],
        index=groups, columns=groups,
    )
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.uns["interface_flux"] = {
        "integrated_flux": {"g": flux},
        "mean_normal_derivative": {"g": flux / 100.0},
        "interface_length": lengths,
        "params": {},
    }
    return adata


def _rose_adata(n=200, seed=0):
    rng = np.random.default_rng(seed)
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32))
    # half point East (0 rad) with large magnitude, half uniform with tiny
    directions = np.concatenate([
        rng.normal(0.0, 0.1, n // 2),
        rng.uniform(-np.pi, np.pi, n - n // 2),
    ])
    magnitudes = np.concatenate([
        np.full(n // 2, 1.0),
        np.full(n - n // 2, 0.05),
    ])
    adata.obs["gradient_g_direction"] = directions.astype(np.float32)
    adata.obs["gradient_g_magnitude"] = magnitudes.astype(np.float32)
    adata.obs["gradient_unreliable"] = np.zeros(n, dtype=bool)
    adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2)).astype(np.float32)
    return adata


def _bar_heights(panel):
    ax = panel.figure.axes[0]
    return np.array([p.get_height() for p in ax.patches])


# ---------------------------------------------------------------------------
# flux_plot
# ---------------------------------------------------------------------------


class TestFluxPlot:
    def test_lower_triangle_annotates_lower_only(self):
        adata = _flux_adata()
        panel = flux_plot(adata, "g")
        assert isinstance(panel, FigurePanel)
        ax = panel.figure.axes[0]
        # annotations: lower-triangle finite cells only (3 of them) + the
        # identity note
        cell_texts = [t for t in ax.texts if t.get_transform() is ax.transData]
        assert len(cell_texts) == 3
        note = [t for t in ax.texts if "identity" in t.get_text()]
        assert len(note) == 1

    def test_full_draws_both_triangles(self):
        adata = _flux_adata()
        panel = flux_plot(adata, "g", triangle="full")
        ax = panel.figure.axes[0]
        cell_texts = [t for t in ax.texts if t.get_transform() is ax.transData]
        assert len(cell_texts) == 6  # all finite off-diagonal entries

    def test_min_interface_um_masks_short_interfaces(self):
        adata = _flux_adata()
        # A|C interface is 10 µm -> masked at 100 µm gate; A|B and B|C stay
        panel = flux_plot(adata, "g", min_interface_um=100.0)
        ax = panel.figure.axes[0]
        cell_texts = [t for t in ax.texts if t.get_transform() is ax.transData]
        assert len(cell_texts) == 2

    def test_invalid_triangle_raises(self):
        adata = _flux_adata()
        with pytest.raises(ValueError, match="triangle"):
            flux_plot(adata, "g", triangle="upper")


# ---------------------------------------------------------------------------
# direction_histogram
# ---------------------------------------------------------------------------


class TestDirectionHistogram:
    def test_magnitude_vs_count_weighting_differ(self):
        adata = _rose_adata()
        h_mag = _bar_heights(
            direction_histogram(adata, "g", weights="magnitude", relative_to=None)
        )
        h_cnt = _bar_heights(
            direction_histogram(adata, "g", weights="count", relative_to=None)
        )
        assert not np.allclose(h_mag, h_cnt)
        # magnitude weighting concentrates mass in the East bins (the
        # strong-gradient population) more than counting does
        east_bin = np.argmax(h_mag)
        assert h_mag[east_bin] > h_cnt[east_bin]

    def test_exclude_unreliable_drops_flagged_cells(self):
        adata = _rose_adata()
        # flag the entire East-pointing strong population as unreliable
        flag = np.zeros(adata.n_obs, dtype=bool)
        flag[: adata.n_obs // 2] = True
        adata.obs["gradient_unreliable"] = flag

        h_excl = _bar_heights(direction_histogram(adata, "g", relative_to=None))
        h_incl = _bar_heights(
            direction_histogram(adata, "g", exclude_unreliable=False, relative_to=None)
        )
        assert not np.allclose(h_excl, h_incl)
        assert h_incl.max() > h_excl.max()  # East spike only when included

    def test_relative_to_obs_column_recenters(self):
        adata = _rose_adata()
        # reference bearing equal to each cell's own direction -> everything
        # lands at relative angle 0 ("toward")
        adata.obs["ref"] = adata.obs["gradient_g_direction"]
        panel = direction_histogram(adata, "g", relative_to="ref")
        h = _bar_heights(panel)
        # all mass in the bins adjacent to angle 0
        center_mass = h[np.abs(np.arange(len(h)) - len(h) // 2) <= 1].sum()
        assert h.sum() == pytest.approx(1.0, abs=1e-6)
        assert center_mass > 0.99
        # tick labels switch to toward/away
        labels = [t.get_text() for t in panel.figure.axes[0].get_xticklabels()]
        assert "toward" in labels and "away" in labels

    def test_relative_to_point_and_array(self):
        adata = _rose_adata()
        p1 = direction_histogram(adata, "g", relative_to=(50.0, 50.0))
        assert isinstance(p1, FigurePanel)
        ref = np.zeros(adata.n_obs)
        p2 = direction_histogram(adata, "g", relative_to=ref)
        assert isinstance(p2, FigurePanel)
        with pytest.raises(ValueError, match="shape"):
            direction_histogram(adata, "g", relative_to=np.zeros(3))

    def test_multi_key_legend_reports_rbar(self):
        adata = _rose_adata()
        adata.obs["gradient_h_direction"] = adata.obs["gradient_g_direction"]
        adata.obs["gradient_h_magnitude"] = adata.obs["gradient_g_magnitude"]
        panel = direction_histogram(adata, ["g", "h"], relative_to=None)
        legend = panel.figure.axes[0].get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert all("R̄=" in lab for lab in labels)

    def test_nan_directions_tolerated(self):
        adata = _rose_adata()
        d = adata.obs["gradient_g_direction"].to_numpy().copy()
        d[:50] = np.nan
        adata.obs["gradient_g_direction"] = d
        panel = direction_histogram(adata, "g", relative_to=None)
        assert isinstance(panel, FigurePanel)

    def test_invalid_weights_raises(self):
        adata = _rose_adata()
        with pytest.raises(ValueError, match="weights"):
            direction_histogram(adata, "g", weights="area")

    def test_cell_axis_default_is_nucleus_anchored_full_rose(self):
        adata = _rose_adata()
        # every cell's frame: distal (away-from-nucleus) major pole = +x,
        # away minor pole = +y — the rose is drawn in the proximal frame, so
        # the East-pointing strong population lands at "away (major)"
        # (relative angle ±180°, the rose's edge bins)
        adata.obs["cell_polarity_major_angle"] = np.zeros(adata.n_obs, dtype=np.float32)
        adata.obs["cell_polarity_minor_angle"] = np.full(
            adata.n_obs, np.pi / 2, dtype=np.float32
        )
        panel = direction_histogram(adata, "g")
        ax = panel.figure.axes[0]
        h = _bar_heights(panel)
        assert len(h) == 36  # full circle, each observation binned once
        assert h.sum() == pytest.approx(1.0, abs=1e-6)
        assert h[:2].sum() + h[-2:].sum() > 0.9
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "away\n(major)" in labels and "toward\n(major)" in labels
        assert "away\n(minor)" in labels and "toward\n(minor)" in labels
        assert "R̄" in ax.get_title()

    def test_cell_axis_flipped_frame_lands_at_proximal_pole(self):
        adata = _rose_adata()
        # distal major pole = -x: East-pointing gradients now read "toward
        # the nucleus", which the proximal display frame puts at relative
        # angle 0 (the rose's center bins)
        adata.obs["cell_polarity_major_angle"] = np.full(
            adata.n_obs, np.pi, dtype=np.float32
        )
        adata.obs["cell_polarity_minor_angle"] = np.full(
            adata.n_obs, np.pi / 2, dtype=np.float32
        )
        h = _bar_heights(direction_histogram(adata, "g"))
        center_mass = h[np.abs(np.arange(len(h)) - len(h) // 2) <= 1].sum()
        assert center_mass > 0.9

    def test_cell_axis_nan_polarity_cells_excluded(self):
        adata = _rose_adata()
        maj = np.zeros(adata.n_obs, dtype=np.float32)
        maj[: adata.n_obs // 2] = np.nan  # the strong East population
        adata.obs["cell_polarity_major_angle"] = maj
        adata.obs["cell_polarity_minor_angle"] = np.full(
            adata.n_obs, np.pi / 2, dtype=np.float32
        )
        h = _bar_heights(direction_histogram(adata, "g"))
        # only the weak uniform population remains -> no dominant spike
        assert h.sum() == pytest.approx(1.0, abs=1e-6)
        assert h.max() < 0.5

    def test_cell_axis_missing_polarity_raises_helpfully(self):
        adata = _rose_adata()
        with pytest.raises(KeyError, match="calculate_shape_metrics"):
            direction_histogram(adata, "g")

    def test_landmark_rose_has_toward_at_top(self):
        adata = _rose_adata()
        adata.obs["ref"] = adata.obs["gradient_g_direction"]
        panel = direction_histogram(adata, "g", relative_to="ref")
        ax = panel.figure.axes[0]
        # theta zero at North: "toward" (0°) renders at the top of the rose
        assert ax.get_theta_offset() == pytest.approx(np.pi / 2)

    def test_raw_frame_labels_are_degrees_not_compass(self):
        adata = _rose_adata()
        panel = direction_histogram(adata, "g", relative_to=None)
        ax = panel.figure.axes[0]
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "0°" in labels
        assert "N" not in labels and "E" not in labels
        assert ax.get_theta_offset() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# diffusion_comparison
# ---------------------------------------------------------------------------


def _diffusion_adata(n=80, seed=0):
    from astrolabe.tl._graph import spatial_neighbors

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 200, (n, 2)).astype(np.float32)
    expr = (coords[:, 0] / 200.0 + rng.normal(0, 0.2, n)).astype(np.float32)
    adata = ad.AnnData(X=expr.reshape(-1, 1))
    adata.var_names = ["g"]
    adata.obsm["spatial"] = coords
    spatial_neighbors(adata, n_neighbors=6)
    return adata


class TestDiffusionComparison:
    def test_returns_grid_with_raw_plus_step_panels(self):
        adata = _diffusion_adata()
        grid = diffusion_comparison(adata, "g", steps=(1, 3))
        assert isinstance(grid, PlotGrid)
        panels = [p for row in grid.rows for p in row if p is not None]
        assert len(panels) == 3  # raw + 2 step counts
        assert panels[0].title == "g (raw)"
        assert panels[1].title == "g (n_steps=1)"
        assert panels[2].title == "g (n_steps=3)"

    def test_shared_color_limits_across_panels(self):
        adata = _diffusion_adata()
        grid = diffusion_comparison(adata, "g", steps=(1, 5))
        panels = [p for row in grid.rows for p in row if p is not None]
        vmins = {p.layers[0].vmin for p in panels}
        vmaxs = {p.layers[0].vmax for p in panels}
        assert len(vmins) == 1 and len(vmaxs) == 1
        (vmin,), (vmax,) = vmins, vmaxs
        assert vmin is not None and vmax is not None and vmin < vmax

    def test_steps_order_insensitive_and_deduplicated(self):
        adata = _diffusion_adata()
        g1 = diffusion_comparison(adata, "g", steps=(5, 1, 1))
        panels = [p for row in g1.rows for p in row if p is not None]
        assert [p.title for p in panels] == ["g (raw)", "g (n_steps=1)", "g (n_steps=5)"]

    def test_does_not_mutate_adata(self):
        adata = _diffusion_adata()
        obs_before = list(adata.obs.columns)
        x_before = adata.X.copy()
        diffusion_comparison(adata, "g", steps=(2,))
        assert list(adata.obs.columns) == obs_before
        np.testing.assert_array_equal(adata.X, x_before)

    def test_invalid_steps_raise(self):
        adata = _diffusion_adata()
        with pytest.raises(ValueError, match="steps"):
            diffusion_comparison(adata, "g", steps=(0, 2))
        with pytest.raises(ValueError, match="steps"):
            diffusion_comparison(adata, "g", steps=())

    def test_unknown_normalize_raises(self):
        adata = _diffusion_adata()
        with pytest.raises(ValueError, match="normalize"):
            diffusion_comparison(adata, "g", normalize="sqrt")


# ---------------------------------------------------------------------------
# decay_length_plot
# ---------------------------------------------------------------------------


def _decay_adata(lam=80.0, se=8.0, f0=1.0, background=0.2, r2=0.9):
    r = np.linspace(10, 490, 25)
    y = f0 * np.exp(-r / lam) + background if np.isfinite(lam) else np.full_like(r, np.nan)
    adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
    adata.uns["decay_length_g"] = {
        "lambda_um": lam,
        "lambda_se_um": se,
        "f0": f0,
        "background": background,
        "r_squared": r2,
        "bin_centers": r,
        "bin_means": y,
        "bin_counts": np.linspace(200, 10, 25),
        "params": {"key": "g", "max_dist_um": 500.0, "n_bins": 25},
    }
    return adata


class TestDecayLengthPlot:
    def test_healthy_fit_smoke(self):
        panel = decay_length_plot(_decay_adata(), "g")
        assert isinstance(panel, FigurePanel)
        ax = panel.figure.axes[0]
        assert "lambda = 80" in ax.get_title()
        cautions = [t for t in ax.texts if "caution" in t.get_text()]
        assert cautions == []

    def test_poorly_identified_shows_caution(self):
        panel = decay_length_plot(_decay_adata(lam=42.0, se=30.0), "g")
        ax = panel.figure.axes[0]
        cautions = [t for t in ax.texts if "caution" in t.get_text()]
        assert len(cautions) == 1
        assert "SE > 50%" in cautions[0].get_text()

    def test_failed_fit_draws_bins_only(self):
        adata = _decay_adata(lam=np.nan, se=np.nan, f0=np.nan, background=np.nan, r2=np.nan)
        # give the bins real values even though the fit failed
        adata.uns["decay_length_g"]["bin_means"] = np.linspace(1, 0.2, 25)
        panel = decay_length_plot(adata, "g")
        ax = panel.figure.axes[0]
        assert "fit failed" in ax.get_title()
        assert len(ax.lines) == 0  # no fit curve, no background line

    def test_missing_uns_raises(self):
        adata = ad.AnnData(X=np.zeros((3, 1), dtype=np.float32))
        with pytest.raises(KeyError, match="decay_length_g"):
            decay_length_plot(adata, "g")
