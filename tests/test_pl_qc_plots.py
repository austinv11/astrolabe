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


# ---------------------------------------------------------------------------
# gene_detection_scatter must render controls (post-calculate_qc_metrics NaN)
# ---------------------------------------------------------------------------


def _xenium_like_adata(n_cells=30, n_genes=6, n_controls=2):
    rng = np.random.default_rng(0)
    n_vars = n_genes + n_controls
    X = rng.poisson(3.0, size=(n_cells, n_vars)).astype(np.float32)
    # Guarantee every control feature has at least one count so it must draw.
    X[0, n_genes:] = 5.0
    var = pd.DataFrame(
        {
            "feature_type": ["Gene Expression"] * n_genes
            + ["Negative Control Probe"] * n_controls,
            "is_control_probe": [False] * n_genes + [True] * n_controls,
        },
        index=[f"g{i}" for i in range(n_genes)] + [f"NegControl_{i}" for i in range(n_controls)],
    )
    adata = ad.AnnData(X=X, var=var)
    return adata


def test_gene_detection_scatter_renders_controls_after_qc_metrics():
    """Regression: pp.calculate_qc_metrics(exclude_controls=True, the default)
    leaves control rows' var QC columns NaN, and the plot's finite filter then
    silently dropped every control point — the exact population the plot
    exists to show. Counts/detection must be recomputed from X for NaN rows."""
    adata = _xenium_like_adata()
    ab.pp.calculate_qc_metrics(adata)
    # Preconditions: the QC columns really are NaN on control rows.
    ctrl_mask = adata.var["is_control_probe"].to_numpy()
    assert np.isnan(adata.var.loc[ctrl_mask, "total_counts"]).all()

    panel = ab.pl.gene_detection_scatter(adata)
    ax = panel.figure.axes[0]
    ctrl_collections = [
        c for c in ax.collections if c.get_label() == "Negative Control Probe"
    ]
    assert ctrl_collections, "control category was not drawn at all"
    n_drawn = sum(len(c.get_offsets()) for c in ctrl_collections)
    assert n_drawn == int(ctrl_mask.sum())
    plt.close(panel.figure)


def test_gene_detection_scatter_renders_stashed_controls_after_removal():
    """Controls removed from var entirely (pipeline.filter_xenium) must still
    render from the uns['control_probe_summary'] stash, with a real detection
    rate reconstructed from the stashed n_cells_by_counts / n_obs."""
    adata = _xenium_like_adata()
    ab.pp.calculate_xenium_qc_metrics(adata)
    assert "n_cells_by_counts" in adata.uns["control_probe_summary"]
    assert adata.uns["control_probe_summary"]["n_obs"] == adata.n_obs
    ab.pp.calculate_qc_metrics(adata)
    ab.pp.filter_genes(adata, exclude_controls=True, inplace=True)
    assert not adata.var["is_control_probe"].any()

    panel = ab.pl.gene_detection_scatter(adata)
    ax = panel.figure.axes[0]
    ctrl_collections = [
        c for c in ax.collections if c.get_label() == "Negative Control Probe"
    ]
    assert ctrl_collections, "stashed controls were not drawn after removal"
    offsets = np.concatenate([c.get_offsets() for c in ctrl_collections])
    assert len(offsets) == 2
    # Detection rate came from the stash, not NaN.
    assert np.isfinite(offsets[:, 1]).all()
    plt.close(panel.figure)


# ---------------------------------------------------------------------------
# ripley_plot must use the stored L_exp as the CSR reference
# ---------------------------------------------------------------------------


def test_ripley_plot_draws_stored_L_exp_as_reference():
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    bins = np.arange(10, 110, 10, dtype=float)
    adata.uns["ripley_l"] = {
        "type_0": pd.DataFrame(
            {
                "L_obs": np.full(len(bins), 3.0),
                "L_exp": np.full(len(bins), -5.0),  # simulated null mean, NOT 0
                "L_hi": np.full(len(bins), -3.0),
                "L_lo": np.full(len(bins), -7.0),
            },
            index=bins,
        )
    }
    panel = ab.pl.ripley_plot(adata)
    ax = panel.figure.axes[0]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert dashed, "no dashed CSR reference drawn"
    assert any(np.allclose(ln.get_ydata(), -5.0) for ln in dashed), (
        "dashed reference must be the stored L_exp curve, not y=0"
    )
    # No spurious y=0 fallback line when L_exp is available.
    assert not any(
        len(np.unique(ln.get_ydata())) == 1 and np.allclose(ln.get_ydata(), 0.0)
        for ln in dashed
    )
    plt.close(panel.figure)


def test_ripley_plot_falls_back_to_zero_line_when_L_exp_all_nan():
    adata = ad.AnnData(X=np.zeros((1, 1), dtype=np.float32))
    bins = np.arange(10, 110, 10, dtype=float)
    adata.uns["ripley_l"] = {
        "type_0": pd.DataFrame(
            {
                "L_obs": np.full(len(bins), 3.0),
                "L_exp": np.full(len(bins), np.nan),  # n_perms=0
                "L_hi": np.full(len(bins), np.nan),
                "L_lo": np.full(len(bins), np.nan),
            },
            index=bins,
        )
    }
    panel = ab.pl.ripley_plot(adata)
    ax = panel.figure.axes[0]
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert any(
        len(np.unique(ln.get_ydata())) == 1 and np.allclose(ln.get_ydata(), 0.0)
        for ln in dashed
    ), "with all-NaN L_exp the y=0 fallback reference must be drawn"
    plt.close(panel.figure)


# ---------------------------------------------------------------------------
# volcano_plot must not silently substitute raw p for adjusted p
# ---------------------------------------------------------------------------


def test_volcano_plot_raises_on_missing_pvals_adj():
    n = 5
    adata = ad.AnnData(X=np.zeros((3, n), dtype=np.float32))
    names = np.rec.fromarrays([np.array([f"g{i}" for i in range(n)])], names=["grp"])
    lfc = np.rec.fromarrays([np.linspace(-2, 2, n)], names=["grp"])
    pvals = np.rec.fromarrays([np.linspace(0.01, 0.5, n)], names=["grp"])
    adata.uns["rank_genes_groups"] = {
        "names": names, "logfoldchanges": lfc, "pvals": pvals,
    }
    with pytest.raises(KeyError, match="pvals_adj"):
        ab.pl.volcano_plot(adata, group="grp")
    # Explicitly asking for raw p-values still works.
    panel = ab.pl.volcano_plot(adata, group="grp", pval_col="pvals")
    assert "pvals" in panel.figure.axes[0].get_ylabel()
    plt.close(panel.figure)


def test_gene_detection_scatter_labels_cosmx_control_types():
    """CosMx-style var['feature_type'] (written by read_cosmx) must color the
    two control classes separately rather than collapsing them into one
    boolean 'Control Probe' class."""
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(30, 8)).astype(np.float32)
    X[0, 4:] = 5.0  # every control feature draws
    var = pd.DataFrame(
        {
            "feature_type": ["Gene Expression"] * 4
            + ["Negative Probe"] * 2
            + ["System Control"] * 2,
            "is_control_probe": [False] * 4 + [True] * 4,
        },
        index=[f"g{i}" for i in range(4)]
        + ["NegPrb1", "NegPrb2", "SystemControl1", "FalseCode1"],
    )
    adata = ad.AnnData(X=X, var=var)
    ab.pp.calculate_qc_metrics(adata)

    panel = ab.pl.gene_detection_scatter(adata)
    ax = panel.figure.axes[0]
    by_label = {}
    for c in ax.collections:
        if len(c.get_offsets()):
            by_label.setdefault(c.get_label(), []).append(c)
    assert "Negative Probe" in by_label
    assert "System Control" in by_label
    assert sum(len(c.get_offsets()) for c in by_label["Negative Probe"]) == 2
    assert sum(len(c.get_offsets()) for c in by_label["System Control"]) == 2
    # The two control classes must be visually distinct.
    color_np = tuple(by_label["Negative Probe"][0].get_facecolor()[0])
    color_sc = tuple(by_label["System Control"][0].get_facecolor()[0])
    assert color_np != color_sc
    plt.close(panel.figure)


def test_gene_detection_scatter_draws_noise_floor_line():
    """When uns['noise_floor'] exists (pp.calculate_noise_floor), the scatter
    must draw the threshold as a vertical line at threshold*n_obs + 1."""
    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(30, 8)).astype(np.float32)
    X[0, 4:] = 5.0
    var = pd.DataFrame(
        {
            "feature_type": ["Gene Expression"] * 4
            + ["Negative Probe"] * 4,
            "is_control_probe": [False] * 4 + [True] * 4,
        },
        index=[f"g{i}" for i in range(4)] + [f"NegPrb{i}" for i in range(4)],
    )
    adata = ad.AnnData(X=X, var=var)
    ab.pp.calculate_qc_metrics(adata)
    ab.pp.calculate_noise_floor(adata)
    thr = adata.uns["noise_floor"]["threshold_mean_counts"]

    panel = ab.pl.gene_detection_scatter(adata)
    ax = panel.figure.axes[0]
    expected_x = thr * adata.n_obs + 1
    vlines = [
        ln for ln in ax.lines
        if len(ln.get_xdata()) and np.allclose(ln.get_xdata(), expected_x)
    ]
    assert vlines, "noise-floor vline not drawn"
    plt.close(panel.figure)

    # And it must be suppressible.
    panel2 = ab.pl.gene_detection_scatter(adata, show_noise_floor=False)
    ax2 = panel2.figure.axes[0]
    assert not [
        ln for ln in ax2.lines
        if len(ln.get_xdata()) and np.allclose(ln.get_xdata(), expected_x)
    ]
    plt.close(panel2.figure)
