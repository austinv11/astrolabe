"""
Generic field-resolved plots: scatter/histogram/heatmap over arbitrary
obs/var/gene-expression fields, passed as plain strings or raw arrays.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import anndata as ad
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astrolabe._utils import _coerce_field, with_adata
from astrolabe.pl._spatial import (
    _DEFAULT_CONTINUOUS_CMAP,
    FigurePanel,
    _build_cat_colors,
    _sorted_categories,
)
from astrolabe.pl._theme import (
    _legend_handles,
    _style_axes,
    _style_colorbar,
    _style_legend,
    get_theme,
)

__all__ = ["scatter", "histogram", "heatmap"]

_logger = logging.getLogger(__name__)


def _density_values(xv: np.ndarray, yv: np.ndarray, bins: int) -> np.ndarray:
    """Map each (x, y) point to the count in its 2-D histogram bin."""
    h, xe, ye = np.histogram2d(xv, yv, bins=bins)
    xi = np.clip(np.searchsorted(xe, xv, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, yv, side="right") - 1, 0, bins - 1)
    return h[xi, yi]


@with_adata
def scatter(
    adata: ad.AnnData,
    x: "str | np.ndarray",
    y: "str | np.ndarray",
    *,
    color: "str | np.ndarray | None" = None,
    axis: Literal["obs", "var"] = "obs",
    layer: str | None = None,
    palette: "dict[str, str] | list[str] | None" = None,
    category_order: "list[str] | None" = None,
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    vcenter: float | None = None,
    na_color: str | None = None,
    size: float = 4.0,
    alpha: float = 0.5,
    log_x: bool = False,
    log_y: bool = False,
    density: bool = False,
    density_bins: int = 50,
    annotate_corr: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Scatter plot of two arbitrary fields, optionally colored by a third.

    ``x``/``y``/``color`` may each be a field name (resolved against gene
    expression / ``obs`` columns when ``axis="obs"``, or ``var`` columns
    when ``axis="var"``) or a raw array already the right length.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    x, y
        Field name or array for each axis.
    color
        Field name or array to color points by. Categorical fields draw a
        legend; continuous fields draw a colorbar. ``None`` with
        ``density=True`` colors by local 2-D point density instead. For a
        literal fixed marker color (not a field), pass ``c=`` via
        ``**scatter_kwargs`` (matplotlib's own scatter kwarg).
    axis
        ``"obs"`` (default) resolves string fields per-cell; ``"var"``
        resolves them per-gene.
    layer
        Layer to read gene expression from (``axis="obs"`` only).
    palette
        Color mapping (dict) or cycle (list) for categorical *color*.
    category_order
        Explicit draw/legend order for categorical *color*. Defaults to
        numerically-aware natural sort of the unique values. Categories with
        no drawable points are dropped from the legend (with a warning).
    cmap
        Colormap for continuous *color* or density coloring.
    vcenter
        Midpoint for a diverging colormap when *color* is continuous
        (e.g. ``0`` for a signed statistic). ``None`` = plain linear scale.
    na_color
        Fallback color for categories missing from *palette*.
    size, alpha
        Marker size (points²) and opacity.
    log_x, log_y
        Log-scale the x/y axis.
    density
        Color points by local 2-D density when *color* is ``None``.
    density_bins
        Number of bins for the density estimate.
    annotate_corr
        Annotate the Spearman correlation coefficient in the corner.
    xlabel, ylabel
        Axis labels. Default to the field name when *x*/*y* are strings.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **scatter_kwargs
        Forwarded to ``ax.scatter`` (e.g. ``c=`` for a fixed marker color).

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pl.scatter(mdata, x="cell_area", y="EPCAM")
    >>> ab.pl.scatter(mdata, x="cell_area", y="EPCAM", layer="area_norm", c="darkorange")
    >>> ab.pl.scatter(mdata, x="cell_area", y="EPCAM", color="leiden")
    """
    xv, _ = _coerce_field(adata, x, axis=axis, layer=layer)
    yv, _ = _coerce_field(adata, y, axis=axis, layer=layer)
    cv, c_is_cat = _coerce_field(adata, color, axis=axis, layer=layer)

    finite = np.isfinite(xv) & np.isfinite(yv)
    if log_x:
        finite &= xv > 0
    if log_y:
        finite &= yv > 0
    xf, yf = xv[finite], yv[finite]
    cf = cv[finite] if cv is not None else None

    t = get_theme()
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    resolved_na = na_color if na_color is not None else t.muted

    if cf is not None and c_is_cat:
        # The legend must advertise only categories that actually have drawn
        # points: an explicit category_order built from unfiltered metadata
        # (e.g. every Xenium control type) can name categories whose values
        # were all dropped by the finite/log filter above.
        drawn_cats = set(cf)
        if category_order is not None:
            unique_cats = [c for c in category_order if c in drawn_cats]
            dropped = [c for c in category_order if c not in drawn_cats]
            if dropped:
                _logger.warning(
                    "scatter: %d categor%s in category_order have no drawable "
                    "points on the current axes and were dropped from the "
                    "legend: %s",
                    len(dropped),
                    "y" if len(dropped) == 1 else "ies",
                    ", ".join(str(c) for c in dropped),
                )
        else:
            unique_cats = _sorted_categories(drawn_cats)
        cat_colors = _build_cat_colors(unique_cats, palette, resolved_na)
        for cat in unique_cats:
            mask = cf == cat
            ax.scatter(
                xf[mask], yf[mask],
                c=cat_colors.get(str(cat), resolved_na),
                s=size, alpha=alpha, label=str(cat),
                rasterized=int(mask.sum()) > 5_000,
                **scatter_kwargs,
            )
        leg = ax.legend(
            handles=_legend_handles(
                unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats]
            ),
            bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8,
        )
        _style_legend(leg)
    elif cf is not None:
        norm = (
            mcolors.TwoSlopeNorm(
                vcenter, vmin=float(np.nanmin(cf)), vmax=float(np.nanmax(cf))
            )
            if vcenter is not None
            else mcolors.Normalize(vmin=float(np.nanmin(cf)), vmax=float(np.nanmax(cf)))
        )
        sc = ax.scatter(
            xf, yf, c=cf, cmap=cmap, norm=norm, s=size, alpha=alpha,
            linewidths=0, rasterized=len(xf) > 5_000, **scatter_kwargs,
        )
        _style_colorbar(fig.colorbar(sc, ax=ax))
    elif density and len(xf) > 1:
        dens = _density_values(xf, yf, density_bins)
        order = np.argsort(dens)  # draw dense clusters on top
        xf, yf, dens = xf[order], yf[order], dens[order]
        norm = mcolors.Normalize(vmin=float(dens.min()), vmax=float(dens.max()))
        sc = ax.scatter(
            xf, yf, c=dens, cmap=cmap, norm=norm, s=size, alpha=alpha,
            linewidths=0, rasterized=len(xf) > 5_000, **scatter_kwargs,
        )
        _style_colorbar(fig.colorbar(sc, ax=ax))
    else:
        point_color = scatter_kwargs.pop("c", t.cat_core[0])
        ax.scatter(
            xf, yf, c=point_color, s=size, alpha=alpha, linewidths=0,
            rasterized=len(xf) > 5_000, **scatter_kwargs,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if annotate_corr and len(xf) > 1:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(xf, yf)
        ax.text(
            0.02, 0.98, f"Spearman ρ = {rho:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=t.primary_ink,
            bbox=dict(boxstyle="round,pad=0.2", fc=t.surface, alpha=0.7, ec="none"),
        )

    ax.set_xlabel(xlabel or (x if isinstance(x, str) else "value"))
    ax.set_ylabel(ylabel or (y if isinstance(y, str) else "value"))
    if title:
        ax.set_title(title)
    ax.xaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax.yaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax.set_axisbelow(True)

    _style_axes(ax)
    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def histogram(
    adata: ad.AnnData,
    x: "str | np.ndarray",
    *,
    axis: Literal["obs", "var"] = "obs",
    layer: str | None = None,
    bins: "int | np.ndarray" = 50,
    log: bool = False,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    vlines: "list[float] | None" = None,
    vline_color: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of an arbitrary field.

    ``x`` may be a field name (resolved against gene expression / ``obs``
    columns when ``axis="obs"``, or ``var`` columns when ``axis="var"``) or
    a raw array.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    x
        Field name or array to histogram.
    axis
        ``"obs"`` (default) resolves a string field per-cell; ``"var"``
        resolves it per-gene.
    layer
        Layer to read gene expression from (``axis="obs"`` only).
    bins
        Number of bins or an explicit bin-edge array.
    log
        Use log-spaced bins and a log x-axis.
    color
        Bar fill color (a literal color, not a field).
    xlabel, ylabel
        Axis labels. *xlabel* defaults to the field name when *x* is a
        string; *ylabel* defaults to ``"Number of cells"``/``"Number of
        genes"`` depending on *axis*.
    title
        Axes title.
    vlines
        X-positions of vertical reference lines (e.g. filter thresholds).
    vline_color
        Color for *vlines*.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pl.histogram(mdata, x="cell_area")
    >>> ab.pl.histogram(mdata, x="total_counts", axis="var")
    """
    values, v_is_cat = _coerce_field(adata, x, axis=axis, layer=layer)
    if not v_is_cat:
        values = np.asarray(values, dtype=np.float64)
        finite_mask = np.isfinite(values)
        if not finite_mask.all():
            _logger.warning(
                "histogram: dropping %d non-finite value(s) of %d.",
                int((~finite_mask).sum()), len(values),
            )
            values = values[finite_mask]
    t = get_theme()
    color = color if color is not None else t.cat_core[0]
    vline_color = vline_color if vline_color is not None else t.muted

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if log and isinstance(bins, int):
        pos = values[values > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1) if len(pos) else bins

    ax.hist(values, bins=bins, color=color, edgecolor="none", **hist_kwargs)

    if log:
        ax.set_xscale("log")

    if vlines:
        for v in vlines:
            ax.axvline(v, color=vline_color, linestyle="--", linewidth=1.2)

    default_ylabel = "Number of cells" if axis == "obs" else "Number of genes"
    ax.set_xlabel(xlabel or (x if isinstance(x, str) else "value"))
    ax.set_ylabel(ylabel or default_ylabel)
    if title:
        ax.set_title(title)
    ax.yaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax.set_axisbelow(True)

    _style_axes(ax)
    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def heatmap(
    adata: ad.AnnData,
    x: "str | np.ndarray | list[str]",
    y: "str | np.ndarray | list[str]",
    *,
    values: "str | np.ndarray | None" = None,
    agg: str = "mean",
    axis: Literal["obs", "var"] = "obs",
    layer: str | None = None,
    bins: "int | np.ndarray" = 20,
    cmap: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 5),
    **kwargs: Any,
) -> FigurePanel:
    """Binned, crosstab, or fields-by-group heatmap of arbitrary fields.

    Three modes, chosen from what *x* and *y* resolve to:

    * both continuous — cells are binned with
      :func:`scipy.stats.binned_statistic_2d` (aggregating *values* by
      *agg*, or counting occurrences when *values* is ``None``);
    * either categorical — a :func:`pandas.crosstab` is drawn instead
      (one row/column per category);
    * **one of them a list of field names** — each listed field becomes its
      own row (*y* a list) or column (*x* a list), aggregated by *agg*
      within each category of the other, categorical, axis. This is the
      direct way to draw "mean of several scores per cluster" without
      building long-form arrays by hand.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    x, y
        Field name or array for each axis — or a list of numeric field
        names for one of them (the other must then resolve to a categorical
        field to group by; *values* must be ``None``).
    values
        Field name or array to aggregate per bin/cell. ``None`` counts
        occurrences instead.
    agg
        Aggregation statistic (``"mean"``, ``"sum"``, ``"median"``, ...)
        used when *values* is given, and always in the list-of-fields mode.
    axis
        ``"obs"`` (default) resolves string fields per-cell; ``"var"``
        resolves them per-gene.
    layer
        Layer to read gene expression from (``axis="obs"`` only).
    bins
        Number of bins (continuous-continuous path only).
    cmap
        Colormap. Defaults to the active theme's sequential colormap.
    xlabel, ylabel
        Axis labels. Default to the field name when *x*/*y* are strings.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pl.heatmap(mdata, x="cell_area", y="EPCAM", values="total_counts", agg="mean")
    >>> ab.pl.heatmap(mdata, x="leiden", y="cell_type", agg="count")
    >>> ab.pl.heatmap(mdata, x="leiden", y=["module_0_score", "module_1_score"])
    """
    x_multi = isinstance(x, (list, tuple))
    y_multi = isinstance(y, (list, tuple))
    if x_multi and y_multi:
        raise ValueError("Only one of x and y may be a list of field names.")

    t = get_theme()
    resolved_cmap = cmap or t.seq_cmap

    table = None  # (n_rows, n_cols) DataFrame: index → y-axis, columns → x-axis
    if x_multi or y_multi:
        fields = list(x) if x_multi else list(y)
        if not all(isinstance(f, str) for f in fields):
            raise TypeError(
                "A list passed as x/y must contain field-name strings; pass "
                "raw per-cell values as a numpy array instead."
            )
        if values is not None:
            raise ValueError(
                "values= cannot be combined with a list of fields — each "
                "listed field is aggregated as its own row/column."
            )
        group_arg = y if x_multi else x
        gv, g_cat = _coerce_field(adata, group_arg, axis=axis, layer=layer)
        if not g_cat:
            raise ValueError(
                "When x/y is a list of fields, the other axis must resolve "
                "to a categorical field to group by."
            )
        col_data = {}
        for f in fields:
            fv, f_cat = _coerce_field(adata, f, axis=axis, layer=layer)
            if f_cat:
                raise ValueError(
                    f"Listed field {f!r} is categorical; only numeric fields "
                    "can be aggregated."
                )
            col_data[f] = np.asarray(fv, dtype=np.float64)
        groups = pd.Series(np.asarray(gv).astype(str))
        cats = _sorted_categories(set(groups))
        agg_tbl = (
            pd.DataFrame(col_data)
            .groupby(groups, observed=True)
            .agg(agg)
            .reindex(cats)
        )  # (categories × fields)
        table = agg_tbl if x_multi else agg_tbl.T
        cbar_label = f"{agg}(value)"
    else:
        xv, x_cat = _coerce_field(adata, x, axis=axis, layer=layer)
        yv, y_cat = _coerce_field(adata, y, axis=axis, layer=layer)
        vv, _ = _coerce_field(adata, values, axis=axis, layer=layer)
        if values is not None:
            value_name = values if isinstance(values, str) else "value"
            cbar_label = f"{agg}({value_name})"
        else:
            cbar_label = "count"
        if x_cat or y_cat:
            xs = pd.Series(xv, name=str(x) if isinstance(x, str) else "x")
            ys = pd.Series(yv, name=str(y) if isinstance(y, str) else "y")
            # crosstab(ys, xs): index = y categories → imshow rows → y-axis,
            # columns = x categories → x-axis, matching the axis labels.
            table = (
                pd.crosstab(ys, xs, values=vv, aggfunc=agg)
                if vv is not None
                else pd.crosstab(ys, xs)
            )

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if table is None:
        from scipy.stats import binned_statistic_2d
        stat_values = vv if vv is not None else xv
        statistic = agg if vv is not None else "count"
        stat, xedges, yedges, _ = binned_statistic_2d(
            xv, yv, stat_values, statistic=statistic, bins=bins,
        )
        im = ax.imshow(
            stat.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=resolved_cmap, **kwargs,
        )
    else:
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap=resolved_cmap, **kwargs)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels(table.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels(table.index)

    _style_colorbar(fig.colorbar(im, ax=ax, label=cbar_label))

    ax.set_xlabel(xlabel or (x if isinstance(x, str) else "" if x_multi else "x"))
    ax.set_ylabel(ylabel or (y if isinstance(y, str) else "" if y_multi else "y"))
    if title:
        ax.set_title(title)

    _style_axes(ax)
    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)
