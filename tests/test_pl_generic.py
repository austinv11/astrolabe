"""Tests for the generic field-resolved plots (astrolabe.pl.scatter/histogram/heatmap)
and the shared field resolver in astrolabe._utils.
"""
import matplotlib
matplotlib.use("Agg")  # headless backend: avoid Tk backend flakiness in CI/sandboxes

import numpy as np
import pytest

import astrolabe as ab
from astrolabe._utils import _coerce_field, _resolve_field
from astrolabe.pl._spatial import FigurePanel, PlotGrid


# ---------------------------------------------------------------------------
# _resolve_field / _coerce_field
# ---------------------------------------------------------------------------

def test_resolve_field_obs_axis_gene_expression(small_adata):
    gene = small_adata.var_names[0]
    values, is_cat = _resolve_field(small_adata, gene, axis="obs")
    assert not is_cat
    assert len(values) == small_adata.n_obs


def test_resolve_field_obs_axis_layer_override(small_adata):
    gene = small_adata.var_names[0]
    small_adata.layers["scaled"] = small_adata.X * 10
    default_values, _ = _resolve_field(small_adata, gene, axis="obs")
    layer_values, _ = _resolve_field(small_adata, gene, axis="obs", layer="scaled")
    assert np.allclose(layer_values, default_values * 10)


def test_resolve_field_obs_axis_categorical_obs_column(small_adata):
    values, is_cat = _resolve_field(small_adata, "cell_type", axis="obs")
    assert is_cat
    assert set(values) == {"A", "B"}


def test_resolve_field_obs_axis_numeric_obs_column(small_adata):
    values, is_cat = _resolve_field(small_adata, "cell_area", axis="obs")
    assert not is_cat
    assert np.allclose(values, small_adata.obs["cell_area"].to_numpy())


def test_resolve_field_obs_axis_unknown_key_raises(small_adata):
    with pytest.raises(KeyError):
        _resolve_field(small_adata, "not_a_field", axis="obs")


def test_resolve_field_var_axis(small_adata):
    small_adata.var["total_counts"] = np.arange(small_adata.n_vars, dtype=float)
    values, is_cat = _resolve_field(small_adata, "total_counts", axis="var")
    assert not is_cat
    assert len(values) == small_adata.n_vars


def test_resolve_field_var_axis_unknown_key_raises(small_adata):
    with pytest.raises(KeyError):
        _resolve_field(small_adata, "not_a_var_field", axis="var")


def test_resolve_field_var_axis_rejects_layer(small_adata):
    with pytest.raises(ValueError):
        _resolve_field(small_adata, "total_counts", axis="var", layer="x")


def test_resolve_field_invalid_axis_raises(small_adata):
    with pytest.raises(ValueError):
        _resolve_field(small_adata, "cell_area", axis="nonsense")


def test_coerce_field_none_passthrough(small_adata):
    assert _coerce_field(small_adata, None) == (None, False)


def test_coerce_field_raw_array_no_length_validation(small_adata):
    """Regression test: arrays longer than n_vars must still be accepted
    (mirrors gene_detection_scatter/gene_qv_scatter's stashed control-probe
    arrays, which are legitimately longer than adata.n_vars)."""
    long_array = np.arange(small_adata.n_vars + 50, dtype=float)
    values, is_cat = _coerce_field(small_adata, long_array, axis="var")
    assert not is_cat
    assert len(values) == small_adata.n_vars + 50


def test_coerce_field_raw_categorical_array(small_adata):
    values, is_cat = _coerce_field(small_adata, np.array(["x", "y", "x"]))
    assert is_cat
    assert list(values) == ["x", "y", "x"]


# ---------------------------------------------------------------------------
# scatter()
# ---------------------------------------------------------------------------

def test_scatter_obs_axis_returns_figure_panel(small_mdata):
    panel = ab.pl.scatter(small_mdata, x="cell_area", y="nucleus_area")
    assert isinstance(panel, FigurePanel)


def test_scatter_gene_expression_y(small_mdata):
    gene = small_mdata["xenium"].var_names[0]
    panel = ab.pl.scatter(small_mdata, x="cell_area", y=gene)
    assert isinstance(panel, FigurePanel)


def test_scatter_categorical_color_builds_legend(small_mdata):
    panel = ab.pl.scatter(small_mdata, x="cell_area", y="nucleus_area", color="cell_type")
    ax = panel.figure.axes[0]
    assert ax.get_legend() is not None


def test_scatter_layer_changes_values(small_mdata):
    adata = small_mdata["xenium"]
    gene = adata.var_names[0]
    adata.layers["scaled"] = adata.X * 100
    p1 = ab.pl.scatter(small_mdata, x="cell_area", y=gene)
    p2 = ab.pl.scatter(small_mdata, x="cell_area", y=gene, layer="scaled")
    y1 = sorted(p1.figure.axes[0].collections[0].get_offsets()[:, 1])
    y2 = sorted(p2.figure.axes[0].collections[0].get_offsets()[:, 1])
    assert not np.allclose(y1, y2)


def test_scatter_raw_array_xy(small_mdata):
    n = small_mdata.n_obs
    panel = ab.pl.scatter(small_mdata, x=np.arange(n), y=np.arange(n))
    assert isinstance(panel, FigurePanel)


def test_scatter_fixed_marker_color_via_c_kwarg(small_mdata):
    # Must not collide with the color= field-resolution kwarg.
    panel = ab.pl.scatter(small_mdata, x="cell_area", y="nucleus_area", c="darkorange")
    assert isinstance(panel, FigurePanel)


def test_scatter_unknown_field_raises(small_mdata):
    with pytest.raises(KeyError):
        ab.pl.scatter(small_mdata, x="not_a_field", y="cell_area")


def test_scatter_var_axis(small_mdata):
    adata = small_mdata["xenium"]
    adata.var["a"] = np.arange(adata.n_vars, dtype=float)
    adata.var["b"] = np.arange(adata.n_vars, dtype=float) * 2
    panel = ab.pl.scatter(small_mdata, x="a", y="b", axis="var")
    assert isinstance(panel, FigurePanel)


# ---------------------------------------------------------------------------
# histogram()
# ---------------------------------------------------------------------------

def test_histogram_obs_axis(small_mdata):
    panel = ab.pl.histogram(small_mdata, x="cell_area")
    assert isinstance(panel, FigurePanel)


def test_histogram_var_axis(small_mdata):
    adata = small_mdata["xenium"]
    adata.var["total_counts"] = np.arange(adata.n_vars, dtype=float) + 1
    panel = ab.pl.histogram(small_mdata, x="total_counts", axis="var")
    assert isinstance(panel, FigurePanel)


def test_histogram_unknown_field_raises(small_mdata):
    with pytest.raises(KeyError):
        ab.pl.histogram(small_mdata, x="not_a_field")


# ---------------------------------------------------------------------------
# heatmap()
# ---------------------------------------------------------------------------

def test_heatmap_continuous_binned(small_mdata):
    panel = ab.pl.heatmap(small_mdata, x="cell_area", y="nucleus_area", bins=5)
    ax = panel.figure.axes[0]
    assert len(ax.images) == 1


def test_heatmap_categorical_crosstab(small_mdata):
    panel = ab.pl.heatmap(small_mdata, x="cell_type", y="cell_type")
    ax = panel.figure.axes[0]
    assert len(ax.images) == 1
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]


def test_heatmap_values_and_agg(small_mdata):
    gene = small_mdata["xenium"].var_names[0]
    panel = ab.pl.heatmap(
        small_mdata, x="cell_area", y="nucleus_area", values=gene, agg="mean", bins=5
    )
    assert isinstance(panel, FigurePanel)


def test_heatmap_crosstab_axis_orientation(small_mdata):
    """x categories must land on the x-axis (regression: crosstab rows used
    to be the x field, silently swapping the plot relative to its labels)."""
    adata = small_mdata["xenium"]
    adata.obs["three_cat"] = (["p", "q", "r"] * adata.n_obs)[: adata.n_obs]
    panel = ab.pl.heatmap(small_mdata, x="three_cat", y="cell_type")
    ax = panel.figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["p", "q", "r"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]


def test_heatmap_field_list_rows(small_mdata):
    """y as a list of numeric fields: one row per field, aggregated per
    category of the categorical x — no manual tile/repeat long-form arrays."""
    panel = ab.pl.heatmap(
        small_mdata, x="cell_type", y=["cell_area", "nucleus_area"], agg="mean"
    )
    ax = panel.figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["cell_area", "nucleus_area"]
    # values must be the per-category means of each field
    adata = small_mdata["xenium"]
    expected = adata.obs.groupby("cell_type", observed=True)["cell_area"].mean()
    drawn = ax.images[0].get_array()
    np.testing.assert_allclose(drawn[0], expected.loc[["A", "B"]].to_numpy())


def test_heatmap_field_list_columns(small_mdata):
    panel = ab.pl.heatmap(
        small_mdata, x=["cell_area", "nucleus_area"], y="cell_type", agg="mean"
    )
    ax = panel.figure.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["cell_area", "nucleus_area"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]


def test_heatmap_field_list_gene_names(small_mdata):
    genes = list(small_mdata["xenium"].var_names[:3])
    panel = ab.pl.heatmap(small_mdata, x="cell_type", y=genes)
    ax = panel.figure.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == genes


def test_heatmap_field_list_requires_categorical_partner(small_mdata):
    with pytest.raises(ValueError, match="categorical"):
        ab.pl.heatmap(small_mdata, x="cell_area", y=["nucleus_area"])


def test_heatmap_field_list_rejects_values(small_mdata):
    with pytest.raises(ValueError, match="values"):
        ab.pl.heatmap(
            small_mdata, x="cell_type", y=["cell_area"], values="nucleus_area"
        )


def test_heatmap_both_axes_lists_raise(small_mdata):
    with pytest.raises(ValueError, match="Only one"):
        ab.pl.heatmap(small_mdata, x=["cell_area"], y=["nucleus_area"])


# ---------------------------------------------------------------------------
# Composability + rename safety
# ---------------------------------------------------------------------------

def test_scatter_and_histogram_compose_side_by_side(small_mdata):
    combo = ab.pl.scatter(small_mdata, x="cell_area", y="nucleus_area") | ab.pl.histogram(
        small_mdata, x="cell_area"
    )
    assert isinstance(combo, PlotGrid)


def test_old_heatmap_signature_raises_type_error(small_mdata):
    """Old call shape ab.pl.heatmap(mdata, genes, groupby=...) must fail loudly
    (TypeError) now that pl.heatmap is the generic two-field function, rather
    than silently producing the wrong plot."""
    with pytest.raises(TypeError):
        ab.pl.heatmap(small_mdata, ["a", "b"], groupby="cell_type")


def test_expression_heatmap_still_works(small_mdata):
    adata = small_mdata["xenium"]
    genes = list(adata.var_names[:3])
    panel = ab.pl.expression_heatmap(small_mdata, genes, groupby="cell_type")
    assert isinstance(panel, FigurePanel)
