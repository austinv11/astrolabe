"""Tests for the categorical-palette overflow bug family in astrolabe.pl._qc_plots.

Regression coverage for: ripley_plot, co_occurrence_plot, centrality_scores_plot,
and the shared _categorical_colors fallback (also used by gene_qv_scatter,
gene_detection_scatter, mean_variance_plot, expression_by_distance_plot). All of
these used to index a fixed 8-hue palette with `i % 8` (or a ListedColormap's
fractional lookup, which quantizes the same way), so any call with more than 8
categories/lines/bars silently reused colors -- two Leiden clusters (routine at
9-10 clusters) would be visually indistinguishable in the figure.
"""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import pytest

import astrolabe as ab
from astrolabe.pl._qc_plots import _categorical_colors


@pytest.fixture(autouse=True)
def _reset_theme():
    ab.pl.set_theme("light")
    yield
    ab.pl.set_theme("light")


def test_categorical_colors_default_cmap_stays_distinct_above_eight():
    # Root-cause fix: with the default theme palette (cmap=None), colors must
    # not repeat once n exceeds the 8-hue cat_core -- this is what every
    # affected _qc_plots function now delegates to.
    colors = _categorical_colors(None, 12)
    assert len(colors) == 12
    assert len(set(colors)) == 12


def test_categorical_colors_at_or_below_eight_uses_core_palette():
    colors = _categorical_colors(None, 8)
    assert set(colors) == set(ab.pl.get_theme().cat_core)


def test_categorical_colors_named_cmap_still_cycles():
    # An explicitly named cmap is the user's own choice, not astrolabe's
    # default palette -- modulo-cycling it is unchanged, expected behavior.
    colors = _categorical_colors("tab10", 15)
    assert len(colors) == 15
    assert colors[0] == colors[10]  # tab10 has 10 entries, so index 10 wraps to 0


def _ripley_adata(n_types: int) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    bins = np.arange(10, 210, 10, dtype=float)
    result = {}
    for i in range(n_types):
        result[f"type_{i}"] = pd.DataFrame(
            {
                "L_obs": np.full(len(bins), float(i)),
                "L_exp": np.zeros(len(bins)),
                "L_hi": np.full(len(bins), float(i) + 1),
                "L_lo": np.full(len(bins), float(i) - 1),
            },
            index=bins,
        )
    adata.uns["ripley_l"] = result
    return adata


def test_ripley_plot_distinct_colors_above_eight_cell_types():
    adata = _ripley_adata(9)
    panel = ab.pl.ripley_plot(adata)
    fig = panel.figure
    ax = fig.axes[0]
    line_colors = [line.get_color() for line in ax.lines if not line.get_label().startswith("_")]
    assert len(line_colors) == 9
    assert len(set(line_colors)) == 9
    plt.close(fig)


def _co_occurrence_adata(n_groups: int) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    interval = np.linspace(10, 100, 5)
    rng = np.random.default_rng(0)
    groups = [f"g{i}" for i in range(n_groups)]
    occ = rng.uniform(0.5, 1.5, size=(n_groups, n_groups, len(interval)))
    adata.uns["co_occurrence"] = {"occ": occ, "interval": interval, "groups": groups}
    return adata


def test_co_occurrence_plot_distinct_colors_above_eight_pairs():
    # 3 source x 3 target = 9 lines, one more than the 8-hue core palette.
    adata = _co_occurrence_adata(3)
    panel = ab.pl.co_occurrence_plot(adata, groupby="ignored")
    fig = panel.figure
    ax = fig.axes[0]
    line_colors = [line.get_color() for line in ax.lines if not line.get_label().startswith("_")]
    assert len(line_colors) == 9
    assert len(set(line_colors)) == 9
    plt.close(fig)


def _centrality_adata(n_groups: int) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    df = pd.DataFrame(
        {"degree_centrality": np.linspace(0.1, 0.9, n_groups)},
        index=[f"ct{i}" for i in range(n_groups)],
    )
    adata.uns["centrality_scores"] = df
    return adata


def test_centrality_scores_plot_distinct_colors_above_eight_groups():
    adata = _centrality_adata(9)
    panel = ab.pl.centrality_scores_plot(adata, groupby="ignored")
    fig = panel.figure
    ax = fig.axes[0]
    bar_colors = [patch.get_facecolor() for patch in ax.patches]
    assert len(bar_colors) == 9
    assert len(set(bar_colors)) == 9
    plt.close(fig)
