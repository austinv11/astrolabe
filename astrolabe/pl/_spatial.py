"""
Composable spatial plotting for astrolabe.

Public functions (:func:`points`, :func:`image`, :func:`boundaries`) return
a lazy :class:`SpatialPanel` that renders only when ``.show()``, ``.save()``,
``.to_figure()``, or a Jupyter display is triggered.

Compose before rendering:

- ``+``  — stack layers onto the same axes
- ``|``  — place panels side by side
- ``/``  — stack panels in rows

Plain ``matplotlib.figure.Figure`` and ``matplotlib.axes.Axes`` objects are
accepted on either side of ``|`` and ``/``.

Examples
--------
>>> ab.pl.points(mdata, "leiden").show()
>>> (ab.pl.image(mdata) + ab.pl.points(mdata, "leiden")).show()
>>> (ab.pl.points(mdata, "EPCAM") | ab.pl.points(mdata, "CD3D")).show()
"""

from __future__ import annotations

import functools
import io
import logging
import operator
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.figure
import tifffile
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend as _Legend
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._shapes import arrow, circle, rectangle
from astrolabe._utils import (
    ImageSpec,
    _call_image_filter,
    _check_single_library,
    _od_deconvolve,
    _resolve_adata,
    _resolve_field,
    _transform_coords,
    with_adata,
)
from astrolabe.pl._theme import (
    _extended_cat_colors,
    _legend_handles,
    _place_legend,
    _register_colorbar,
    _style_axes,
    get_theme,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "points",
    "image",
    "boundaries",
    "raster",
    "annotation",
    "gradient_quiver",
    "gradient_streamlines",
    "divergence_plot",
    "rectangle",
    "circle",
    "arrow",
    "Layer",
    "SpatialPanel",
    "PlotGrid",
    "FigurePanel",
    "layer",
    "hstack",
    "vstack",
    "as_panel",
]

log = logging.getLogger(__name__)

_DEFAULT_CONTINUOUS_CMAP = "astrolabe_seq"
_DEFAULT_DIVERGING_CMAP = "astrolabe_div"

# Floor on dynamic-sizing COLUMN WIDTH (PlotGrid.to_figure, nrows==1 branch,
# i.e. a '|' row of panels) below which chrome -- a title, one colorbar, a
# legend, y-axis tick labels -- no longer fits without overlapping. Pure
# data-aspect-ratio sizing has no such floor built in: a very tall/narrow
# crop (e.g. a small, elongated tissue region) can otherwise compute a
# column a fraction of an inch wide, which title text and colorbars then
# collide across regardless of how narrow the underlying data really is.
# Deliberately not applied to row HEIGHT (ncols==1, a '/' column of panels):
# a short row's chrome is small and roughly fixed (one title line, one
# x-axis label line), and vstacking a wide panel short next to a square
# panel is an existing, intentional, tested compression (see
# test_points_layer_markers_shrink_in_a_vstack_next_to_a_figure_panel).
_MIN_DYNAMIC_PANEL_IN = 4.0

# Core-palette slot 1 (orange) / slot 2 (aqua) -- replaces the old raw CSS
# color names ('lightsalmon'/'cyan') so default boundary tints stay part of
# the validated categorical family. See astrolabe.pl._theme for the palette.
_BOUNDARY_DEFAULT_COLORS: dict[str, str] = {
    "cell_boundaries": "#eb6834",
    "nucleus_boundaries": "#1baf7a",
}


# ---------------------------------------------------------------------------
# Layer — pure data, zero behavior
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Layer:
    """A single renderable element within a :class:`SpatialPanel`.

    Instances are created by the public plotting functions (:func:`points`,
    :func:`image`, :func:`boundaries`) and should not normally be constructed
    directly.

    Attributes
    ----------
    kind
        Dispatch key used by the renderer.  One of ``'points'``, ``'image'``,
        ``'boundaries'``.
    adata
        Already-resolved :class:`anndata.AnnData`.  Resolution happens
        eagerly in the public function, not at render time.
    """

    kind: Literal["points", "image", "boundaries", "raster", "annotation",
                  "quiver", "streamplot", "divergence"]
    adata: ad.AnnData | None = None

    # --- shared spatial / color ---
    spatial_key: str = "spatial"
    color: str | None = None
    size: float | None = None
    alpha: float = 0.7
    cmap: str | None = None
    palette: dict | list | None = None
    vmin: float | None = None
    vmax: float | None = None
    vcenter: float | None = None
    na_color: str | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    # --- image-specific ---
    image_key: str = "full"
    library_id: str | None = None
    channel: int | str | None = None
    gamma: float = 1.0
    image_cmap: str = "gray"
    normalize: bool = True
    dark_bg: bool = False

    # --- boundaries-specific ---
    boundaries_key: str = "cell_boundaries"
    linewidth: float = 0.5
    fill_alpha: float = 0.0

    # --- raster-specific ---
    bin_shape: Literal["square", "hex", "kde"] = "square"
    bins: int = 100
    agg: Literal["mean", "sum"] = "mean"
    fill_color: str | None = None
    bandwidth: float | str | None = None

    # --- annotation-specific ---
    shapes: list | None = None          # list of (n_verts, 2) ndarray in µm
    annotation_color: str = "white"     # stroke color (literal-color mode)
    annotation_fill: str | None = None  # fill color; None = transparent
    annotation_values: np.ndarray | None = None
    """Per-shape resolved color values (categorical or numeric), aligned 1:1
    with ``shapes``, when :func:`annotation` was given a GeoDataFrame and a
    ``color=`` matching one of its columns. ``None`` for literal-color mode."""

    # --- image-specific: scale bar ---
    scale_bar: bool | dict = False  # False | True | dict of kwargs

    # --- gradient/quiver/streamplot/divergence: pre-computed arrays ---
    # Stored here (not extra_kwargs) so they are not forwarded to matplotlib.
    precomputed: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.kind != "annotation" and self.adata is None:
            raise ValueError(f"adata is required for Layer kind={self.kind!r}")

    @property
    def resolved_na_color(self) -> str:
        """``na_color`` if explicitly set, else the active theme's muted gray.

        Resolved at draw time (not construction time) so a ``Layer`` built
        before a :func:`~astrolabe.pl.set_theme` call still tracks the
        theme that's active when it's actually rendered.
        """
        return self.na_color if self.na_color is not None else get_theme().muted


# ---------------------------------------------------------------------------
# _Renderable — shared base
# ---------------------------------------------------------------------------


# Above this many marks (scatter offsets, boundary polygons, quiver arrows,
# ...) a single Collection is drawn as an embedded raster rather than one SVG
# element per mark. A whole-section Xenium panel is ~160k cells, and
# matplotlib writes each one as a <g><use/></g> pair: measured on
# notebooks/05, four such panels came to 103 MB of SVG in one .ipynb, past
# GitHub's 100 MB per-file limit purely in point markers. Rasterizing only
# the dense Collections keeps every axis label, tick, title, legend and
# colorbar as real vector text -- the reason SVG is the display format at
# all -- while the marks, which are sub-pixel dots at any realistic figure
# size, cost a fixed amount instead of one element per cell.
_RASTERIZE_MIN_MARKS = 2000

# Raster resolution for those embedded marks. matplotlib's SVG backend takes
# the rasterized portion's dpi from the figure dpi at savefig time (see
# FigureCanvasSVG.print_svg -> MixedModeRenderer), so this is passed as
# savefig(dpi=...) even though SVG is otherwise resolution-independent.
_RASTERIZE_DPI = 200


def _n_marks(coll: matplotlib.collections.Collection) -> int:
    """Number of individually drawn marks in *coll*.

    A scatter is one path (the marker) repeated at N offsets; a boundary
    layer is N paths with no offsets -- so the mark count is the larger of
    the two, not either one alone.
    """
    try:
        n_offsets = len(coll.get_offsets())
    except Exception:  # pragma: no cover - defensive; not every Collection has offsets
        n_offsets = 0
    return max(n_offsets, len(coll.get_paths()))


def _rasterize_dense_marks(
    fig: matplotlib.figure.Figure,
) -> list[matplotlib.collections.Collection]:
    """Flag every Collection in *fig* with a lot of marks as rasterized.

    Returns the collections whose flag was actually flipped, so the caller
    can restore them afterwards -- a Figure handed back to the user (or
    re-rendered at another size) must not be left permanently altered by
    having been displayed once.
    """
    flipped = []
    for ax in fig.axes:
        for coll in ax.collections:
            if coll.get_rasterized():
                continue
            if _n_marks(coll) >= _RASTERIZE_MIN_MARKS:
                coll.set_rasterized(True)
                flipped.append(coll)
    return flipped


def _svg_mimebundle_for(fig: matplotlib.figure.Figure) -> dict[str, str]:
    """Render *fig* to an SVG Jupyter mimebundle.

    Shared by :meth:`_Renderable.show` and :meth:`_Renderable._repr_mimebundle_`
    so there is exactly one place that knows to pass ``bbox_extra_artists``
    -- a legend dodged past a colorbar is deliberately excluded from
    ``constrained_layout`` via ``set_in_layout(False)`` (see
    ``_place_legend`` in ``_theme.py``), which also excludes it from a bare
    ``bbox_inches="tight"`` crop unless explicitly re-included here; skipping
    that would silently drop the legend from every saved or displayed figure
    that has one.
    """
    buf = io.BytesIO()
    flipped = _rasterize_dense_marks(fig)
    try:
        fig.savefig(
            buf, format="svg", bbox_inches="tight",
            bbox_extra_artists=getattr(fig, "_ab_extra_artists", None),
            # Explicit, not the 'savefig.facecolor' rcParam default: that
            # rcParam tracks the *global* theme surface, which would silently
            # overwrite a black fig.patch set by _apply_dark_theme (image
            # dark_bg=True) back to the light theme's surface color, stranding
            # the dark theme's white title/tick text on a light background.
            facecolor=fig.get_facecolor(),
            # See _RASTERIZE_DPI: sets the resolution of the rasterized
            # dense-mark layers only; all other content stays vector.
            dpi=_RASTERIZE_DPI,
        )
    finally:
        for coll in flipped:
            coll.set_rasterized(False)
    buf.seek(0)
    return {"image/svg+xml": buf.getvalue().decode("utf-8")}


def _display_or_show(fig: matplotlib.figure.Figure) -> None:
    """Display *fig* for a ``.show()`` call -- shared by every ``_Renderable``
    subclass's ``show()`` override (the base one, and ``PlotGrid``'s, which
    forwards a *dynamic_sizing* kwarg ``to_figure`` doesn't take otherwise).

    In a notebook, publishes the same SVG mimebundle
    :meth:`_Renderable._repr_mimebundle_` uses (see :func:`_svg_mimebundle_for`
    for why this doesn't just delegate to ``plt.show()``). Outside one,
    falls back to ``plt.show()``, re-registering *fig* with pyplot first if
    it isn't (e.g. a :class:`FigurePanel` wrapping a figure some other
    function already called ``plt.close()`` on).
    """
    try:
        from IPython.display import publish_display_data
    except ImportError:
        publish_display_data = None

    if publish_display_data is not None:
        publish_display_data(data=_svg_mimebundle_for(fig))
    else:
        if fig.number not in plt.get_fignums():
            manager = plt._get_backend_mod().new_figure_manager_given_figure(fig.number, fig)
            gcf = plt._pylab_helpers.Gcf
            if hasattr(gcf, "_set_new_active_manager"):
                gcf._set_new_active_manager(manager)
            else:
                gcf.set_active(manager)
        plt.show()


class _Renderable(ABC):
    """Abstract base for anything that can be rendered or composed."""

    @abstractmethod
    def to_figure(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        """Render to a new matplotlib Figure and return it."""

    def show(self, *, figsize: tuple[float, float] | None = None) -> None:
        """Render and display.

        Renders the figure exactly once regardless of context -- see
        :func:`_display_or_show` for why this doesn't just delegate to
        ``display(self)`` or ``plt.show()`` directly.
        """
        fig = self.to_figure(figsize=figsize)
        _display_or_show(fig)
        plt.close(fig)

    def save(self, path: str, *, dpi: int = 150, **savefig_kwargs: Any) -> None:
        """Save the figure to *path* without displaying it.

        Parameters
        ----------
        path
            Output file path.  Format is inferred from the extension
            (e.g. ``'.pdf'``, ``'.png'``, ``'.svg'``).
        dpi
            Resolution in dots per inch.  For a vector format this sets the
            resolution of the rasterized dense-mark layers only (see
            :data:`_RASTERIZE_MIN_MARKS`); all other content stays vector.
        **savefig_kwargs
            Additional keyword arguments forwarded to ``figure.savefig()``.
        """
        fig = self.to_figure()
        # Same reasoning as _svg_mimebundle_for -- a 160k-cell scatter written
        # one <g><use/></g> per cell is what makes a saved SVG/PDF hundreds of
        # megabytes. No-op for raster formats (everything is rasterized there).
        _rasterize_dense_marks(fig)
        savefig_kwargs.setdefault("bbox_extra_artists", getattr(fig, "_ab_extra_artists", None))
        # See _svg_mimebundle_for: honor the figure's actual facecolor (which
        # _apply_dark_theme may have forced to black) rather than letting the
        # 'savefig.facecolor' rcParam silently revert it to the global theme.
        savefig_kwargs.setdefault("facecolor", fig.get_facecolor())
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
        plt.close(fig)

    def _repr_mimebundle_(
        self,
        include: Any = None,
        exclude: Any = None,
    ) -> dict[str, str]:
        """Jupyter display hook — renders to SVG bytes."""
        fig = self.to_figure()
        bundle = _svg_mimebundle_for(fig)
        plt.close(fig)
        return bundle

    # ------------------------------------------------------------------
    # Layout operators
    # ------------------------------------------------------------------

    def __or__(self, other: Any) -> PlotGrid:
        """Place this panel and *other* side by side (columns).

        *other* can be a :class:`SpatialPanel`, :class:`PlotGrid`,
        :class:`FigurePanel`, a ``matplotlib.figure.Figure``, or a
        ``matplotlib.axes.Axes``.

        Examples
        --------
        >>> ab.pl.points(mdata, "EPCAM") | ab.pl.points(mdata, "CD3D")
        """
        return _col_join(_to_grid(self), _to_grid(_coerce(other)))

    def __truediv__(self, other: Any) -> PlotGrid:
        """Stack this panel above *other* (rows).

        *other* can be a :class:`SpatialPanel`, :class:`PlotGrid`,
        :class:`FigurePanel`, a ``matplotlib.figure.Figure``, or a
        ``matplotlib.axes.Axes``.

        Examples
        --------
        >>> ab.pl.points(mdata, "leiden") / ab.pl.points(mdata, "EPCAM")
        """
        return _row_join(_to_grid(self), _to_grid(_coerce(other)))

    def __ror__(self, other: Any) -> PlotGrid:
        """Reflected ``|`` so that ``figure | panel`` works when the left
        operand is a plain ``matplotlib.figure.Figure`` or ``Axes``."""
        return _col_join(_to_grid(_coerce(other)), _to_grid(self))

    def __rtruediv__(self, other: Any) -> PlotGrid:
        """Reflected ``/`` so that ``figure / panel`` works when the left
        operand is a plain ``matplotlib.figure.Figure`` or ``Axes``."""
        return _row_join(_to_grid(_coerce(other)), _to_grid(self))


# ---------------------------------------------------------------------------
# SpatialPanel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialPanel(_Renderable):
    """A single plot panel containing one or more stacked spatial layers.

    Created by :func:`points`, :func:`image`, and :func:`boundaries`.  Use
    ``+`` to stack layers onto the same axes, ``|`` for side-by-side
    layout, and ``/`` for vertical stacking.

    Examples
    --------
    >>> panel = ab.pl.points(mdata, "leiden")
    >>> panel.show()

    >>> # Layer image behind scatter
    >>> (ab.pl.image(mdata) + ab.pl.points(mdata, "leiden")).show()

    >>> # Side-by-side
    >>> (ab.pl.points(mdata, "EPCAM") | ab.pl.points(mdata, "CD3D")).show()
    """

    layers: list[Layer]
    title: str | None = None
    frameon: bool = True
    figsize: tuple[float, float] = (5, 5)

    def __add__(self, other: SpatialPanel) -> SpatialPanel:
        """Stack *other*'s layers onto this panel (same axes).

        Returns a new :class:`SpatialPanel`; neither operand is mutated.

        Raises
        ------
        TypeError
            If *other* is not a :class:`SpatialPanel`.  Use ``|`` or ``/``
            to combine panels into a grid layout.
        """
        if not isinstance(other, SpatialPanel):
            raise TypeError(
                f"+ stacks layers onto the same axes and requires two SpatialPanel "
                f"objects; got {type(other).__name__!r}. "
                f"Use | for side-by-side layout or / for stacked layout"
            )
        return SpatialPanel(
            layers=list(self.layers) + list(other.layers),
            title=self.title,
            frameon=self.frameon,
            figsize=self.figsize,
        )

    def to_figure(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        # layout="constrained" (matching PlotGrid.to_figure), not a trailing
        # plt.tight_layout() call -- tight_layout() run AFTER _render_panel
        # can shift the colorbar's position one more time, invalidating the
        # legend-dodge measurement _place_legend already took mid-render
        # (see _theme.py). constrained_layout applies continuously and is
        # stable once converged, so nothing moves out from under it after.
        fig, ax = plt.subplots(1, 1, figsize=figsize or self.figsize, layout="constrained")
        # Close the just-created figure if any layer's guard raises mid-render
        # (e.g. _check_single_library on a multi-library object) -- otherwise
        # a blank figure leaks into pyplot's registry and gets auto-flushed
        # as an empty plot at the end of the notebook cell.
        try:
            _render_panel(ax, self)
            if any(lay.kind == "image" and lay.dark_bg for lay in self.layers):
                _add_dark_panel_backgrounds(fig, [ax])
        except Exception:
            plt.close(fig)
            raise
        return fig

    def __repr__(self) -> str:
        kinds = [lay.kind for lay in self.layers]
        return f"SpatialPanel(layers={kinds}, title={self.title!r})"


# ---------------------------------------------------------------------------
# FigurePanel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FigurePanel(_Renderable):
    """Wraps an existing ``matplotlib.figure.Figure`` for use in layout operators.

    Constructed automatically when a plain ``Figure`` or ``Axes`` is used
    with ``|`` or ``/``.  Use :func:`as_panel` to construct one explicitly
    when you need an introspectable reference.

    At render time the wrapped figure is rasterised to PNG and embedded in
    the layout axes.

    Examples
    --------
    >>> violin_fig = ab.pl.violin(mdata, "total_counts", return_fig=True)
    >>> violin_fig | ab.pl.points(mdata, "total_counts")   # auto-wrapped
    >>> ab.pl.as_panel(violin_fig) | ab.pl.points(mdata, "total_counts")
    """

    figure: matplotlib.figure.Figure
    figsize: tuple[float, float] | None = None

    @classmethod
    def from_obj(
        cls,
        obj: Any,
        figsize: tuple[float, float] | None = None,
    ) -> "FigurePanel":
        """Wrap *obj* as a :class:`FigurePanel`.

        Handles the common return types from scanpy plotting functions:
        ``matplotlib.figure.Figure``, ``matplotlib.axes.Axes``, list of Axes,
        objects with a ``.get_axes()`` method (DotPlot, Heatmap, etc.), objects
        with a ``.figure`` attribute, and falls back to ``plt.gcf()``.
        """
        if isinstance(obj, matplotlib.figure.Figure):
            return cls(figure=obj, figsize=figsize)
        if isinstance(obj, matplotlib.axes.Axes):
            return cls(figure=obj.get_figure(), figsize=figsize)
        if isinstance(obj, list) and obj and isinstance(obj[0], matplotlib.axes.Axes):
            return cls(figure=obj[0].get_figure(), figsize=figsize)
        if hasattr(obj, "get_axes"):
            axes_dict = obj.get_axes()
            if axes_dict:
                return cls(
                    figure=next(iter(axes_dict.values())).get_figure(),
                    figsize=figsize,
                )
        if hasattr(obj, "figure") and isinstance(obj.figure, matplotlib.figure.Figure):
            return cls(figure=obj.figure, figsize=figsize)
        return cls(figure=plt.gcf(), figsize=figsize)

    def to_figure(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        return self.figure

    def __repr__(self) -> str:
        size = tuple(self.figure.get_size_inches())
        return f"FigurePanel(size={size})"


# ---------------------------------------------------------------------------
# Panel extent / aspect ratio -- for aspect-aware PlotGrid sizing
# ---------------------------------------------------------------------------


def _panel_crop_bounds(panel: SpatialPanel) -> dict | None:
    """Return the ``pp.crop()``-set bounds for *panel*, if any.

    Factored out of ``_render_panel`` (which uses this same lookup to
    override the drawn extent) so aspect-ratio sizing and actual rendering
    can never drift out of sync -- both read the same crop-bounds-first
    priority.
    """
    return next(
        (
            lay.adata.uns["spatial_bounds"]
            for lay in panel.layers
            if lay.adata is not None and "spatial_bounds" in lay.adata.uns
        ),
        None,
    )


def _robust_extent(
    coords: np.ndarray, *, warn: bool = True
) -> tuple[float, float, float, float]:
    """Outlier-robust ``(xmin, xmax, ymin, ymax)`` for a coordinate array.

    A single far-away centroid (a segmentation artifact, a stray fiducial)
    can inflate a raw min/max extent enough to compress the whole tissue
    into a corner of the panel and misestimate the auto point size.  Per
    axis: when the raw range is under 1.5x the 0.1-99.9 percentile range,
    the exact raw min/max are kept (clean data renders unchanged);
    otherwise the percentile bounds padded by 2% are used, and a warning
    reports how many points fall outside the returned box.
    """
    xs = np.asarray(coords[:, 0], dtype=np.float64)
    ys = np.asarray(coords[:, 1], dtype=np.float64)
    lims: list[tuple[float, float]] = []
    clipped_any = False
    for v in (xs, ys):
        raw_lo, raw_hi = float(np.min(v)), float(np.max(v))
        lo, hi = np.percentile(v, [0.1, 99.9])
        robust_range = float(hi - lo)
        raw_range = raw_hi - raw_lo
        if robust_range <= 0 or raw_range < 1.5 * robust_range:
            lims.append((raw_lo, raw_hi))
        else:
            pad = 0.02 * robust_range
            lims.append((float(lo) - pad, float(hi) + pad))
            clipped_any = True
    (xmin, xmax), (ymin, ymax) = lims
    if clipped_any and warn:
        outside = int(((xs < xmin) | (xs > xmax) | (ys < ymin) | (ys > ymax)).sum())
        log.warning(
            "Auto extent: %d point(s) lie far outside the robust 0.1-99.9 "
            "percentile box and were excluded from the axis extent. Crop with "
            "pp.crop()/pp.auto_crop() or filter those cells to silence this.",
            outside,
        )
    return xmin, xmax, ymin, ymax


def _layer_extent(lay: Layer) -> tuple[float, float, float, float] | None:
    """Cheap, metadata-only ``(xmin, xmax, ymin, ymax)`` bbox for *lay*.

    Mirrors the exact coordinate source each ``_draw_*_layer`` function uses
    to set its own axis limits, so the aspect ratio computed here matches
    what actually gets drawn. Never loads pixel/zarr data. Returns ``None``
    for a layer kind this can't (yet) estimate cheaply -- callers treat that
    as "no opinion" rather than an error.
    """
    try:
        if lay.kind in ("points", "boundaries", "raster") and lay.adata is not None:
            coords = lay.adata.obsm[lay.spatial_key]
            # Robust, not raw min/max: one outlier centroid would otherwise
            # dictate the whole panel's aspect ratio (warn=False -- this is
            # a sizing estimate; the draw path emits the user-facing warning).
            return _robust_extent(coords, warn=False)
        if lay.kind == "image" and lay.adata is not None:
            spec: ImageSpec = lay.adata.uns["spatial"][lay.library_id]["images"][lay.image_key]
            h_full, w_full = _spec_yx_full_shape(spec)
            xmin, xmax, ymin, ymax = _compute_extent(spec, h_full, w_full)
            return xmin, xmax, ymin, ymax
        if lay.kind == "annotation" and lay.shapes:
            verts = np.concatenate(lay.shapes, axis=0)
            return (
                float(verts[:, 0].min()), float(verts[:, 0].max()),
                float(verts[:, 1].min()), float(verts[:, 1].max()),
            )
        if lay.kind in ("quiver", "streamplot", "divergence") and lay.precomputed:
            p = lay.precomputed
            xk = "X" if "X" in p else "X_grid"
            yk = "Y" if "Y" in p else "Y_grid"
            x, y = np.asarray(p[xk]), np.asarray(p[yk])
            return (
                float(np.nanmin(x)), float(np.nanmax(x)),
                float(np.nanmin(y)), float(np.nanmax(y)),
            )
    except Exception:
        log.debug("Could not estimate extent for layer kind=%r", lay.kind, exc_info=True)
    return None


def _panel_extent(panel: SpatialPanel) -> tuple[float, float, float, float] | None:
    """Unioned ``(xmin, xmax, ymin, ymax)`` bbox for every layer in *panel*.

    Crop bounds (``pp.crop()``) win if present, exactly as ``_render_panel``
    prioritizes them at draw time; otherwise the bbox is the union of every
    layer's :func:`_layer_extent`, so a ``+``-stacked multi-kind panel gets
    one bbox reflecting everything actually drawn.
    """
    bounds = _panel_crop_bounds(panel)
    if bounds is not None:
        return bounds["xmin"], bounds["xmax"], bounds["ymin"], bounds["ymax"]

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for lay in panel.layers:
        ext = _layer_extent(lay)
        if ext is None:
            continue
        xmin, xmax, ymin, ymax = ext
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
    if not xmins:
        return None
    return min(xmins), max(xmaxs), min(ymins), max(ymaxs)


def _cell_aspect(cell: _Renderable | None) -> float | None:
    """Width/height aspect ratio for one grid cell, or ``None`` if unknown.

    Only :class:`SpatialPanel` cells carry a real, trustworthy aspect ratio
    (µm-space data extent). A :class:`FigurePanel` wraps an arbitrary
    already-rendered figure -- its declared ``figsize`` says nothing
    reliable about the actual plotted content (e.g. a legend can widen a
    figure well past a square axes), so it is deliberately *not*
    introspected here: returning ``None`` lets the caller's default (1.0,
    "assume equal aspect ratio") apply instead. See
    :func:`_pad_figure_panel_to_aspect` for how that assumption is kept
    honest at render time -- the raster is padded with transparent margin
    to fit its assigned box rather than stretched to it.
    """
    if cell is None:
        return None
    if isinstance(cell, SpatialPanel):
        ext = _panel_extent(cell)
        if ext is not None:
            xmin, xmax, ymin, ymax = ext
            w, h = xmax - xmin, ymax - ymin
            if w > 0 and h > 0:
                return w / h
        if cell.figsize:
            fw, fh = cell.figsize
            if fw > 0 and fh > 0:
                return fw / fh
        return None
    return None


# ---------------------------------------------------------------------------
# PlotGrid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlotGrid(_Renderable):
    """A 2-D grid of panels arranged in rows and columns.

    Created by ``|`` (side-by-side) or ``/`` (stacked rows) operations, or
    by calling :func:`hstack` / :func:`vstack` explicitly.

    ``rows[i][j]`` is the panel at row *i*, column *j*.  ``None`` cells are
    rendered as blank/invisible axes.

    Examples
    --------
    >>> grid = ab.pl.points(mdata, "EPCAM") | ab.pl.points(mdata, "CD3D")
    >>> grid.shape
    (1, 2)
    >>> grid.show()
    """

    rows: list[list[_Renderable | None]]

    @property
    def shape(self) -> tuple[int, int]:
        """``(nrows, ncols)`` of the grid."""
        nrows = len(self.rows)
        ncols = max((len(r) for r in self.rows), default=0)
        return nrows, ncols

    def to_figure(
        self,
        figsize: tuple[float, float] | None = None,
        *,
        dynamic_sizing: bool = True,
    ) -> matplotlib.figure.Figure:
        """Render the grid to a new matplotlib Figure.

        Parameters
        ----------
        figsize
            Absolute total figure size in inches. Overrides both the
            per-panel scale and (when set) the aspect-ratio-driven split
            between columns/rows -- the grid still gets proportional
            ``width_ratios``/``height_ratios`` within this total unless
            *dynamic_sizing* is also ``False``.
        dynamic_sizing
            When ``True`` (default), each column/row is sized proportionally
            to the real µm-space aspect ratio of the panels it contains --
            e.g. a wide, short tissue section claims more figure width than
            a square UMAP panel next to it, instead of both getting an
            identical box. For a single row (``|``) or single column
            (``/``) -- the common case -- every panel keeps its own true
            aspect ratio exactly, with no distortion: the figure's total
            size grows or shrinks to fit rather than forcing panels into a
            fixed-size box. Grids where every cell already has the same
            aspect ratio render identically either way. Pass ``False`` to
            restore the old equal-box behavior.
        """
        nrows, ncols = self.shape
        ref = _first_non_none(self.rows)
        if ref is not None and hasattr(ref, "figsize") and ref.figsize is not None:
            per_w, per_h = ref.figsize
        elif ref is not None and isinstance(ref, FigurePanel):
            per_w, per_h = tuple(ref.figure.get_size_inches())
        else:
            per_w, per_h = 5.0, 5.0

        width_ratios = [1.0] * ncols
        height_ratios = [1.0] * nrows
        explicit_total: tuple[float, float] | None = None
        if dynamic_sizing:
            col_aspects: list[list[float]] = [[] for _ in range(ncols)]
            row_aspects: list[list[float]] = [[] for _ in range(nrows)]
            for i, row in enumerate(self.rows):
                for j in range(ncols):
                    a = _cell_aspect(row[j] if j < len(row) else None)
                    if a is not None and a > 0:
                        # Clamp: one extreme-aspect layer shouldn't warp the
                        # whole grid -- this is a greedy per-row/per-column
                        # heuristic, not an exact NxM layout solver.
                        a = min(max(a, 0.2), 5.0)
                        col_aspects[j].append(a)
                        row_aspects[i].append(a)
            col_aspect = [max(v) if v else 1.0 for v in col_aspects]
            row_aspect = [max(v) if v else 1.0 for v in row_aspects]

            if nrows == 1:
                # Exact: every column renders at its own true aspect ratio
                # simultaneously -- there's no other cell in the column to
                # compete with for row height, so the figure's total WIDTH
                # grows/shrinks to fit instead of squeezing/stretching any
                # panel to hit a pre-decided total.
                #
                # If the narrowest column would fall under the chrome floor,
                # scale the shared row height up (not just that column's
                # width) so every column's box aspect stays exactly equal to
                # its true µm-space data aspect. Flooring width alone (an
                # earlier version of this fix) left the box wider than its
                # data needs, which matplotlib's own equal-aspect enforcement
                # then "corrects" by padding the y-limits with a lot of
                # empty space -- trading a chrome collision for a squashed,
                # mostly-blank plot.
                min_aspect = min(col_aspect) if col_aspect else 1.0
                if min_aspect * per_h < _MIN_DYNAMIC_PANEL_IN:
                    per_h = _MIN_DYNAMIC_PANEL_IN / min_aspect
                widths = [a * per_h for a in col_aspect]
                width_ratios = widths
                explicit_total = (sum(widths), per_h)
            elif ncols == 1:
                # No analogous floor here: a short row's chrome (title, one
                # x-axis label line) is small and roughly fixed, unlike a
                # narrow column's (colorbar, legend, y-axis label, and
                # unbounded title-string width) -- and vstacking a wide
                # panel short next to a square panel is an existing,
                # intentional, tested compression (marker size shrinks to
                # match), not a bug to guard against.
                heights = [per_w / a for a in row_aspect]
                height_ratios = heights
                explicit_total = (per_w, sum(heights))
            else:
                # General NxM grid: a shared row height and column width
                # can't simultaneously satisfy every cell's own aspect when
                # aspects are mixed, but the two dimensions must still be
                # COUPLED. An earlier version normalized columns and rows
                # back to the per-panel default independently, which forced
                # e.g. a 2x2 grid of uniformly ~3.5:1-wide tissue panels
                # into near-square cell boxes -- equal-aspect enforcement
                # then filled every cell with vertical padding, rendered as
                # a large blank band across the middle of the figure.
                # Columns get width = aspect x per_h (the same rule as the
                # exact single-row path); each row then gets the height its
                # most demanding cell needs at its column's width. A
                # uniform-aspect grid therefore renders every box at exactly
                # its content aspect; mixed grids pad only what the aspect
                # mismatch itself requires.
                # Columns with no known-aspect cell (all FigurePanels) keep
                # the reference panel width; their content is aspect-padded
                # at render time regardless.
                col_widths = [
                    a * per_h if known else per_w
                    for a, known in zip(col_aspect, col_aspects)
                ]
                row_heights = []
                for i, row in enumerate(self.rows):
                    needs = []
                    for j in range(ncols):
                        a = _cell_aspect(row[j] if j < len(row) else None)
                        if a is not None and a > 0:
                            a = min(max(a, 0.2), 5.0)
                            needs.append(col_widths[j] / a)
                    row_heights.append(max(needs) if needs else per_h)
                width_ratios = col_widths
                height_ratios = row_heights
                explicit_total = (sum(col_widths), sum(row_heights))

        if explicit_total is not None:
            effective = figsize or explicit_total
        else:
            # Normalize so the mean column/row still equals per_w/per_h --
            # a grid where every cell has the same aspect ratio renders
            # with the exact same total figure size as before this
            # feature existed.
            w_scale = per_w / (sum(width_ratios) / ncols)
            h_scale = per_h / (sum(height_ratios) / nrows)
            col_widths = [w * w_scale for w in width_ratios]
            row_heights = [h * h_scale for h in height_ratios]
            effective = figsize or (sum(col_widths), sum(row_heights))

        fig, axes_grid = plt.subplots(
            nrows, ncols, figsize=effective, layout="constrained", squeeze=False,
            gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
        )

        # FigurePanel cells are deferred: the nominal gridspec ratios above
        # don't account for how much margin constrained_layout will actually
        # reserve for a neighboring SpatialPanel's axis labels/title/legend,
        # so a target_aspect computed from those ratios alone can be visibly
        # off (a few percent in a simple grid, worse once real chrome is
        # sharing a row/column). Turning each FigurePanel axes chrome-free
        # up front and resolving layout once before rasterizing means the
        # aspect used for padding is read from the actual final axes box,
        # not an estimate of it.
        # Same blank-figure-leak guard as SpatialPanel.to_figure: if any
        # cell's render raises (e.g. a multi-library guard), close the
        # just-created grid figure before re-raising.
        try:
            pending_figure_panels: list[tuple[matplotlib.axes.Axes, "FigurePanel"]] = []
            dark_axes: list[matplotlib.axes.Axes] = []
            for i, row in enumerate(self.rows):
                for j in range(ncols):
                    panel = row[j] if j < len(row) else None
                    ax = axes_grid[i][j]
                    if panel is None:
                        ax.set_visible(False)
                    elif isinstance(panel, SpatialPanel):
                        _render_panel(ax, panel)
                        if any(lay.kind == "image" and lay.dark_bg for lay in panel.layers):
                            dark_axes.append(ax)
                    else:
                        ax.axis("off")
                        pending_figure_panels.append((ax, panel))

            if pending_figure_panels:
                fig.canvas.draw()  # resolve constrained_layout with final chrome in place
                fig_w, fig_h = fig.get_size_inches()
                for ax, panel in pending_figure_panels:
                    pos = ax.get_position()
                    target_aspect = (pos.width * fig_w) / (pos.height * fig_h)
                    _render_figure_panel(ax, panel.to_figure(), target_aspect=target_aspect)

            # Computed last, after any FigurePanel content above is in its final
            # place, so each dark axes' tight bbox reflects the fully-settled
            # layout rather than an intermediate one.
            _add_dark_panel_backgrounds(fig, dark_axes)
        except Exception:
            plt.close(fig)
            raise

        return fig

    def show(
        self,
        *,
        figsize: tuple[float, float] | None = None,
        dynamic_sizing: bool = True,
    ) -> None:
        """Render and display. See :meth:`to_figure` for *dynamic_sizing*."""
        fig = self.to_figure(figsize=figsize, dynamic_sizing=dynamic_sizing)
        _display_or_show(fig)
        fig.clear()
        plt.close(fig)

    def __repr__(self) -> str:
        return f"PlotGrid(shape={self.shape})"


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _coerce(obj: Any) -> _Renderable:
    """Auto-wrap a matplotlib Figure or Axes into a FigurePanel."""
    if isinstance(obj, matplotlib.figure.Figure):
        return FigurePanel(obj)
    if isinstance(obj, matplotlib.axes.Axes):
        return FigurePanel(obj.figure)
    if isinstance(obj, _Renderable):
        return obj
    raise TypeError(
        f"Cannot use {type(obj).__name__!r} in a plot layout. "
        "Expected a SpatialPanel, PlotGrid, FigurePanel, "
        "matplotlib.figure.Figure, or matplotlib.axes.Axes"
    )


def _to_grid(obj: _Renderable) -> PlotGrid:
    if isinstance(obj, PlotGrid):
        return obj
    return PlotGrid(rows=[[obj]])


def _col_join(a: PlotGrid, b: PlotGrid) -> PlotGrid:
    """Side-by-side join: rows are zipped, shorter grid padded at bottom."""
    aw = max((len(r) for r in a.rows), default=0)
    bw = max((len(r) for r in b.rows), default=0)
    nrows = max(len(a.rows), len(b.rows))
    result = []
    for i in range(nrows):
        ra = a.rows[i] if i < len(a.rows) else []
        rb = b.rows[i] if i < len(b.rows) else []
        result.append(ra + [None] * (aw - len(ra)) + rb + [None] * (bw - len(rb)))
    return PlotGrid(rows=result)


def _row_join(a: PlotGrid, b: PlotGrid) -> PlotGrid:
    """Vertical join: rows from *a* first, shorter rows padded on the right."""
    ncols = max(
        max((len(r) for r in a.rows), default=0),
        max((len(r) for r in b.rows), default=0),
    )

    def _pad(r: list) -> list:
        return r + [None] * (ncols - len(r))

    return PlotGrid(rows=[_pad(r) for r in a.rows] + [_pad(r) for r in b.rows])


def _first_non_none(rows: list[list]) -> _Renderable | None:
    for row in rows:
        for cell in row:
            if cell is not None:
                return cell
    return None


# ---------------------------------------------------------------------------
# Public composition helpers
# ---------------------------------------------------------------------------


def layer(*panels: SpatialPanel) -> SpatialPanel:
    """Stack multiple :class:`SpatialPanel` objects onto the same axes.

    Equivalent to ``p1 + p2 + p3`` but accepts an arbitrary number of
    panels.

    Parameters
    ----------
    *panels
        One or more :class:`SpatialPanel` objects to merge.

    Returns
    -------
    SpatialPanel

    Examples
    --------
    >>> ab.pl.layer(ab.pl.image(mdata), ab.pl.points(mdata, "leiden"))
    """
    if len(panels) < 1:
        raise ValueError("layer() requires at least one panel")
    return functools.reduce(operator.add, panels)


def hstack(*items: Any) -> PlotGrid:
    """Place panels side by side (horizontal stack).

    Equivalent to ``p1 | p2 | p3``.  Each item can be a
    :class:`SpatialPanel`, :class:`PlotGrid`, :class:`FigurePanel`,
    ``matplotlib.figure.Figure``, or ``matplotlib.axes.Axes``.

    Parameters
    ----------
    *items
        One or more renderable objects.

    Returns
    -------
    PlotGrid

    Examples
    --------
    >>> ab.pl.hstack(ab.pl.points(mdata, "EPCAM"), violin_fig)
    """
    if len(items) < 1:
        raise ValueError("hstack() requires at least one item")
    return functools.reduce(operator.or_, [_coerce(it) for it in items])


def vstack(*items: Any) -> PlotGrid:
    """Stack panels in a column (vertical stack).

    Equivalent to ``p1 / p2 / p3``.

    Parameters
    ----------
    *items
        One or more renderable objects.

    Returns
    -------
    PlotGrid

    Examples
    --------
    >>> ab.pl.vstack(ab.pl.points(mdata, "leiden"), violin_fig)
    """
    if len(items) < 1:
        raise ValueError("vstack() requires at least one item")
    return functools.reduce(operator.truediv, [_coerce(it) for it in items])


def as_panel(
    fig: matplotlib.figure.Figure | matplotlib.axes.Axes,
) -> FigurePanel:
    """Wrap an existing matplotlib Figure or Axes as a :class:`FigurePanel`.

    Using ``|`` and ``/`` with a raw ``Figure`` or ``Axes`` auto-wraps it
    implicitly.  Use this only when you need an explicit
    :class:`FigurePanel` reference (e.g. to introspect or pass to a helper).

    Parameters
    ----------
    fig
        A ``matplotlib.figure.Figure`` or ``matplotlib.axes.Axes``.
        When an ``Axes`` is passed, its parent figure is used.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pl.as_panel(violin_fig) | ab.pl.points(mdata, "total_counts")
    >>> fig, ax = plt.subplots(); ax.hist(...)
    >>> ab.pl.as_panel(ax) | ab.pl.points(mdata, "leiden")
    """
    if not isinstance(fig, (matplotlib.figure.Figure, matplotlib.axes.Axes)):
        raise TypeError(
            f"as_panel() expects a matplotlib Figure or Axes, "
            f"got {type(fig).__name__!r}"
        )
    result = _coerce(fig)
    assert isinstance(result, FigurePanel)
    return result


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------


def points(
    data: ad.AnnData | md.MuData | Any,
    color: str | None = None,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    x_col: str = "x_location",
    y_col: str = "y_location",
    size: float | None = None,
    alpha: float = 0.7,
    cmap: str | None = None,
    palette: dict | list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    na_color: str | None = None,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
    **scatter_kwargs: Any,
) -> SpatialPanel:
    """Scatter plot of cells (or transcripts) in physical space.

    Returns a lazy :class:`SpatialPanel` that renders when shown.  Compose
    with ``+`` to overlay on an image or boundary layer, or ``|`` / ``/``
    for multi-panel layouts.

    Parameters
    ----------
    data
        One of:

        - :class:`anndata.AnnData` — uses ``obsm[spatial_key]`` for
          coordinates.
        - :class:`mudata.MuData` — resolved via ``uns['default_modality']``
          (or *modality*).
        - :class:`pyarrow.Table` / :class:`pandas.DataFrame` — coordinates
          are read from *x_col* and *y_col*; remaining columns become
          ``obs`` keys for *color*.

    color
        Gene name (looked up in ``var_names``) or ``obs`` column to color
        by.  For multi-gene layouts compose with ``|``::

            ab.pl.points(mdata, "EPCAM") | ab.pl.points(mdata, "CD3D")

    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    spatial_key
        Key in ``obsm`` for the ``(n, 2)`` coordinate array.
    x_col, y_col
        Column names for coordinates when *data* is a Table or DataFrame.
        Defaults match the Xenium transcript table schema.
    size
        Marker size in points².  Defaults to ``120 000 / n_cells``.
    alpha
        Marker transparency.
    cmap
        Colormap for continuous variables.  Defaults to ``'viridis'``.
    palette
        Color palette for categorical variables.
    vmin, vmax
        Color scale limits for continuous variables.
    vcenter
        Midpoint of a diverging colormap.
    na_color
        Color for points with ``NaN`` in the color column.  ``None``
        (default) uses the active theme's muted color.
    title
        Panel title.  Defaults to the *color* variable name.
    frameon
        Draw axis frame and tick labels.
    figsize
        ``(width, height)`` in inches when rendered standalone.
    **scatter_kwargs
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    SpatialPanel

    Examples
    --------
    >>> ab.pl.points(mdata, "leiden").show()
    >>> ab.pl.points(tx, color="feature_name")      # Xenium transcript table
    >>> ab.pl.points(tx, color="qv", cmap="plasma") # continuous Q-score
    """
    try:
        import pyarrow as pa
        _is_arrow = isinstance(data, pa.Table)
    except ImportError:
        _is_arrow = False

    if _is_arrow or isinstance(data, pd.DataFrame):
        adata = _table_to_adata(data, x_col=x_col, y_col=y_col)
    else:
        adata = _resolve_adata(data, modality)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found. "
            "Ensure read_xenium() was called or obsm['spatial'] exists"
        )

    lay = Layer(
        kind="points",
        adata=adata,
        spatial_key=spatial_key,
        color=color,
        size=size,
        alpha=alpha,
        cmap=cmap,
        palette=palette,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        na_color=na_color,
        extra_kwargs=scatter_kwargs,
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else (color or ""),
        frameon=frameon,
        figsize=figsize,
    )


@with_adata
def image(
    adata: ad.AnnData,
    *,
    image_key: str | None = None,
    library_id: str | None = None,
    channel: int | str | None = None,
    normalize: bool = True,
    gamma: float = 1.0,
    image_cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    dark_bg: bool | None = None,
    scale_bar: bool | dict = False,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (8, 8),
) -> SpatialPanel:
    """Display a single image channel as a spatial panel.

    Typical usage — display an image background, optionally overlaid with
    a scatter or boundary layer::

        ab.pl.image(mdata)
        ab.pl.image(mdata) + ab.pl.points(mdata, "leiden")
        ab.pl.image(mdata, channel="DAPI") + ab.pl.boundaries(mdata)

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`~mudata.MuData` carrying
        ``uns['spatial'][library_id]['images']``.
    image_key
        Which image to display (e.g. ``'full'``, ``'morphology'``, or a
        pyramid level such as ``'level_1'``).  ``None`` auto-selects
        ``'full'`` if present, otherwise the first available key.
    library_id
        Library identifier under ``uns['spatial']``.  Inferred
        automatically when there is exactly one library.
    channel
        Channel index (``int``) or channel name (matched against
        ``var_names`` on the ``'if'`` modality).
    gamma
        Gamma correction for display.  ``1.0`` = linear.
    image_cmap
        Matplotlib colormap for the grayscale image.
    title
        Panel title.
    frameon
        Draw axis frame.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    spatial_uns: dict = adata.uns.get("spatial", {})
    if library_id is None:
        ids = list(spatial_uns.keys())
        if len(ids) != 1:
            raise ValueError(
                f"Cannot infer library_id from {ids}. "
                "Pass library_id explicitly"
            )
        library_id = ids[0]

    lib_uns = spatial_uns.get(library_id, {})
    images = lib_uns.get("images", {})
    if not images:
        raise KeyError(
            f"No images loaded for library '{library_id}'. "
            "Load one first:\n"
            "  ab.io.read_ome_tiff('path/to/image.ome.tif', mdata=data)\n"
            "  # or: ab.io.read_xenium(..., morphology_image=True)"
        )
    if image_key is None:
        image_key = "full" if "full" in images else next(iter(images))
    elif image_key not in images:
        raise KeyError(
            f"Image key '{image_key}' not found. "
            f"Available keys: {list(images.keys())}"
        )

    # dark_bg=None (default) picks a light background for brightfield (H&E)
    # images and the existing dark background for everything else (IF panels,
    # DAPI/morphology, etc.); an explicit True/False always wins.
    if dark_bg is None:
        spec = images.get(image_key)
        dark_bg = not bool(getattr(spec, "is_brightfield", False))

    lay = Layer(
        kind="image",
        adata=adata,
        image_key=image_key,
        library_id=library_id,
        channel=channel,
        normalize=normalize,
        gamma=gamma,
        image_cmap=image_cmap,
        vmin=vmin,
        vmax=vmax,
        dark_bg=dark_bg,
        scale_bar=scale_bar,
    )
    if title is None:
        if channel is None:
            title = f"{library_id}/{image_key}"
        elif isinstance(channel, str):
            title = f"{library_id}/{image_key}/{channel}"
        else:
            title = f"{library_id}/{image_key}/ch{channel}"
    return SpatialPanel(
        layers=[lay],
        title=title,
        frameon=frameon,
        figsize=figsize,
    )


@with_adata
def boundaries(
    adata: ad.AnnData,
    *,
    boundaries_key: str = "cell_boundaries",
    spatial_key: str = "spatial",
    color: str | None = None,
    linewidth: float = 0.5,
    fill_alpha: float = 0.0,
    alpha: float = 0.8,
    palette: dict | list | None = None,
    cmap: str | None = None,
    na_color: str | None = None,
    max_boundaries: int | None = 50_000,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Draw cell or nucleus boundary polygons.

    Pass ``boundaries_key='nucleus_boundaries'`` for nucleus outlines.
    Layer two ``boundaries()`` panels to show both simultaneously::

        ab.pl.boundaries(mdata)
        ab.pl.boundaries(mdata, boundaries_key="nucleus_boundaries")
        ab.pl.boundaries(mdata) + ab.pl.boundaries(
            mdata, boundaries_key="nucleus_boundaries", linewidth=1.0,
        )

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`~mudata.MuData`.
    boundaries_key
        Key in ``obsm`` for the polygon array.  ``'cell_boundaries'``
        (default) or ``'nucleus_boundaries'``.
    spatial_key
        Key in ``obsm`` for centroid coordinates, used to set axis extent.
    color
        Gene name or ``obs`` column for polygon fill / edge color.
        ``None`` renders all polygons in a neutral color.
    linewidth
        Edge linewidth in points.
    fill_alpha
        Polygon fill opacity.  ``0.0`` (default) = outlines only.
    alpha
        Overall layer opacity.
    palette
        Categorical color palette (dict or list).
    cmap
        Colormap for continuous color variables.
    na_color
        Color for polygons with ``NaN`` in the color column.  ``None``
        (default) uses the active theme's muted color.
    max_boundaries
        Polygon count above which a warning is logged (the plot is still
        drawn).  At full-tissue scale (>~50k cells) an outline overlay
        renders as a solid wash that occludes everything under it — crop
        first with :func:`~astrolabe.pp.crop`/:func:`~astrolabe.pp.auto_crop`
        instead.  ``None`` disables the warning.
    title
        Panel title.
    frameon
        Draw axis frame.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    if boundaries_key not in adata.obsm:
        raise KeyError(
            f"obsm['{boundaries_key}'] not found. "
            "Re-run ab.io.read_xenium() with load_boundaries=True, "
            "or pass boundaries_key='nucleus_boundaries' for nucleus outlines"
        )

    n_polys = len(adata.obsm[boundaries_key])
    if max_boundaries is not None and n_polys > max_boundaries:
        log.warning(
            "boundaries: drawing %d polygons — at this scale the outlines "
            "overlap into a solid wash that occludes any layer underneath. "
            "Crop to a region of interest first (ab.pp.crop / ab.pp.auto_crop) "
            "or raise linewidth/alpha deliberately. Pass max_boundaries=None "
            "to silence this warning.",
            n_polys,
        )

    # Apply key-based default color when no explicit color/data mapping is given.
    effective_color = color
    if effective_color is None:
        effective_color = _BOUNDARY_DEFAULT_COLORS.get(boundaries_key)

    lay = Layer(
        kind="boundaries",
        adata=adata,
        spatial_key=spatial_key,
        boundaries_key=boundaries_key,
        color=effective_color,
        linewidth=linewidth,
        fill_alpha=fill_alpha,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        na_color=na_color,
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else (color or boundaries_key),
        frameon=frameon,
        figsize=figsize,
    )


def raster(
    data: ad.AnnData | md.MuData | Any,
    color: str | None = None,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    x_col: str = "x_location",
    y_col: str = "y_location",
    bin_shape: Literal["square", "hex", "kde"] = "square",
    bins: int = 100,
    agg: Literal["mean", "sum"] = "mean",
    bandwidth: float | str | None = None,
    fill_color: str | None = None,
    cmap: str | None = None,
    palette: dict | list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    na_color: str | None = None,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Bin cells into a 2-D grid and render as an image (no per-cell dots).

    Useful for large datasets where individual points overlap and scatter
    plots become unreadable.  Each bin shows an aggregated summary of the
    cells that fall within it.

    Color logic
    -----------
    - ``color=None, fill_color=None`` — density (count per bin) using *cmap*.
    - ``color=None, fill_color='steelblue'`` — density encoded as the
      *alpha* channel of the given solid color.
    - ``color='EPCAM'`` (continuous) — mean (or *agg='sum'*) expression
      per bin, colored by *cmap*.
    - ``color='leiden'`` (categorical) — dominant category per bin,
      colored by *palette*.

    If *fill_color* is set alongside a continuous *color*, the variable
    is still used to modulate alpha rather than hue.

    Parameters
    ----------
    data
        :class:`anndata.AnnData`, :class:`~mudata.MuData`,
        :class:`pyarrow.Table`, or :class:`pandas.DataFrame`.
    color
        Gene name or ``obs`` column to aggregate per bin.
        ``None`` renders density (cell count).  Categorical columns (e.g.
        ``color='leiden'``) are supported: each bin shows the dominant
        category, colored by *palette* the same way as :func:`points`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    spatial_key
        Key in ``obsm`` for the ``(n, 2)`` coordinate array.
    x_col, y_col
        Coordinate column names when *data* is a Table/DataFrame.
    bin_shape
        ``'square'`` (default) renders as a pixel image via ``imshow``;
        ``'hex'`` uses matplotlib hexagonal bins;
        ``'kde'`` computes a Gaussian KDE and renders the smooth density
        surface (only valid for density or continuous *color*).
    bins
        Number of bins along each axis for square bins, the ``gridsize``
        argument for hexagonal bins, or the evaluation grid resolution for
        KDE.
    agg
        Aggregation function for continuous *color* values:
        ``'mean'`` (default) or ``'sum'``.
    bandwidth
        KDE bandwidth passed to ``scipy.stats.gaussian_kde`` as
        ``bw_method``.  Accepts a scalar (Scott's-rule-scale factor),
        ``'scott'``, or ``'silverman'``.  ``None`` uses Scott's rule.
        Ignored when *bin_shape* is not ``'kde'``.
    fill_color
        Solid matplotlib color (e.g. ``'#2196F3'`` or ``'royalblue'``).
        When set, density / aggregated value modulates the *alpha* channel
        of this color instead of being mapped through *cmap*.
    cmap
        Colormap for continuous variables or density.  Defaults to
        ``'viridis'``.  Ignored when *fill_color* is set.
    palette
        color palette for categorical *color* variables.
    vmin, vmax
        color-scale limits.
    vcenter
        Midpoint of a diverging colormap (ignored when *fill_color*
        is set).
    na_color
        Color for empty bins in categorical mode.  ``None`` (default) uses
        the active theme's muted color.
    title
        Panel title.
    frameon
        Draw axis frame.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel

    Examples
    --------
    >>> ab.pl.raster(mdata)                              # density, viridis
    >>> ab.pl.raster(mdata, fill_color="steelblue")      # density, alpha
    >>> ab.pl.raster(mdata, "EPCAM")                     # mean expression
    >>> ab.pl.raster(mdata, "leiden")                    # dominant cluster
    >>> ab.pl.raster(mdata, bin_shape="hex", bins=60)    # hexagonal bins
    >>> ab.pl.raster(tx, color="feature_name", bin_shape="hex")
    >>> ab.pl.raster(mdata, bin_shape="kde")             # smooth KDE density
    >>> ab.pl.raster(mdata, "EPCAM", bin_shape="kde")    # weighted KDE by expression
    """
    try:
        import pyarrow as pa
        _is_arrow = isinstance(data, pa.Table)
    except ImportError:
        _is_arrow = False

    if _is_arrow or isinstance(data, pd.DataFrame):
        adata = _table_to_adata(data, x_col=x_col, y_col=y_col)
    else:
        adata = _resolve_adata(data, modality)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found"
        )

    lay = Layer(
        kind="raster",
        adata=adata,
        spatial_key=spatial_key,
        color=color,
        alpha=1.0,
        cmap=cmap,
        palette=palette,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        na_color=na_color,
        bin_shape=bin_shape,
        bins=bins,
        agg=agg,
        fill_color=fill_color,
        bandwidth=bandwidth,
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else (color or "density"),
        frameon=frameon,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Table → AnnData normalization
# ---------------------------------------------------------------------------


def _table_to_adata(data: Any, x_col: str, y_col: str) -> ad.AnnData:
    """Convert a coordinate table into a minimal AnnData for plotting.

    ``obsm['spatial']`` is extracted from (*x_col*, *y_col*).  All
    remaining columns are kept in ``obs`` so they are accessible as
    *color=* keys.
    """
    try:
        import pyarrow as pa
        if isinstance(data, pa.Table):
            data = data.to_pandas()
    except ImportError:
        pass

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"_table_to_adata expects a pyarrow.Table or pandas.DataFrame, "
            f"got {type(data).__name__!r}"
        )

    for col in (x_col, y_col):
        if col not in data.columns:
            raise KeyError(
                f"Column '{col}' not found in table. "
                f"Available columns: {list(data.columns)}. "
                f"Set x_col/y_col explicitly if your table uses different names"
            )

    coords = data[[x_col, y_col]].to_numpy(dtype=np.float32)
    obs = data.drop(columns=[x_col, y_col]).reset_index(drop=True)
    adata = ad.AnnData(obs=obs)
    adata.obsm["spatial"] = coords
    return adata


def _flatten_gdf_shapes(gdf: Any) -> tuple[list[np.ndarray], np.ndarray]:
    """Flatten a GeoDataFrame's (Multi)Polygon geometries to vertex arrays.

    Returns ``(shapes, row_index)`` where ``row_index[i]`` is the source
    GeoDataFrame row for ``shapes[i]`` — a ``MultiPolygon`` row expands to
    several shapes that all map back to the same row (and so get the same
    resolved color/label), matching how a single categorical value applies
    to the whole (possibly disjoint) region.
    """
    shapes: list[np.ndarray] = []
    row_index: list[int] = []
    for i, geom in enumerate(gdf.geometry):
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            shapes.append(np.asarray(poly.exterior.coords))
            row_index.append(i)
    return shapes, np.array(row_index, dtype=np.int64)


def annotation(
    shapes: "list[np.ndarray] | np.ndarray | gpd.GeoDataFrame",
    *,
    color: str | None = "yellow",
    fill_color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    fill_alpha: float = 0.2,
    palette: dict | list | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    na_color: str | None = None,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Overlay polygon annotation shapes on a spatial panel.

    Intended to be combined with ``+`` onto an existing image or points
    panel rather than used standalone::

        roi = ab.pl.rectangle(center=(500, 300), width=200, height=100)
        (ab.pl.image(mdata) + ab.pl.annotation([roi], color="yellow")).show()

        # Color regions by a GeoDataFrame column, with an automatic legend:
        gdf = ab.io.read_annotations("annotation.geojson")
        (ab.pl.points(mdata, color="total_counts")
         + ab.pl.annotation(gdf, color="name")).show()

    Parameters
    ----------
    shapes
        A single ``(n_verts, 2)`` array, a list of such arrays (µm), or a
        :class:`geopandas.GeoDataFrame` (e.g. from
        :func:`~astrolabe.io.read_annotations`) — its ``(Multi)Polygon``
        geometries are flattened automatically, one color/legend entry per
        row even when a row is a ``MultiPolygon``.
    color
        Either a literal matplotlib color for all shapes (default
        ``"yellow"``), or — when *shapes* is a GeoDataFrame and *color*
        matches one of its non-geometry columns — a field to color by,
        categorical or continuous, resolved the same way ``pl.points``'s
        ``color=`` is. Falls back to a literal color if *shapes* is a
        GeoDataFrame but *color* doesn't match a column.
    fill_color
        Fill color in literal-color mode. ``None`` = no fill (outline only).
        Ignored when *color* resolves to a data column (fill follows the
        resolved per-region color instead).
    linewidth
        Edge line width in points.
    alpha
        Overall opacity (applies to both edge and fill).
    fill_alpha
        Fill opacity — used when *fill_color* is set (literal-color mode),
        or always in data-driven coloring mode.
    palette
        Categorical color palette (dict or list), used only when *color*
        resolves to a categorical column.
    cmap
        Colormap for a continuous column.
    vmin, vmax, vcenter
        Continuous color scaling, as in ``pl.points``/``pl.boundaries``.
    na_color
        Color for regions with ``NaN``/missing values in the color column.
    title
        Optional panel title (rarely needed for annotation-only panels).
    frameon
        Draw axis frame.
    figsize
        Figure size when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    annotation_values: np.ndarray | None = None
    color_field: str | None = None
    literal_color = color

    is_gdf = hasattr(shapes, "geometry") and hasattr(shapes, "columns")
    if is_gdf:
        shapes_list, row_index = _flatten_gdf_shapes(shapes)
        if color is not None:
            try:
                values, _ = _resolve_gdf_field(shapes, color)
            except KeyError:
                values = None
            if values is not None:
                annotation_values = values[row_index]
                color_field = color
                literal_color = None
    else:
        if isinstance(shapes, np.ndarray):
            shapes = [shapes]
        shapes_list = list(shapes)

    lay = Layer(
        kind="annotation",
        adata=None,
        shapes=shapes_list,
        annotation_color=literal_color if literal_color is not None else "white",
        annotation_fill=fill_color,
        annotation_values=annotation_values,
        color=color_field,
        linewidth=linewidth,
        alpha=alpha,
        fill_alpha=fill_alpha,
        palette=palette,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        na_color=na_color,
    )
    return SpatialPanel(layers=[lay], title=title, frameon=frameon, figsize=figsize)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_panel(ax: matplotlib.axes.Axes, panel: SpatialPanel) -> None:
    """Draw all layers of *panel* onto *ax*, then apply title and frame."""
    # Shared legend entry list — image channel entries and default-color boundary
    # outline entries are both collected here so they appear in one combined legend
    # rather than two overlapping add_artist() calls.
    shared_legend: list[tuple[Any, str]] = []

    for lay in panel.layers:
        if lay.kind == "points":
            _draw_points_layer(ax, lay, shared_legend)
        elif lay.kind == "image":
            _draw_image_layer(ax, lay, shared_legend)
        elif lay.kind == "boundaries":
            _draw_boundary_layer(ax, lay, shared_legend)
        elif lay.kind == "raster":
            _draw_raster_layer(ax, lay, shared_legend)
        elif lay.kind == "annotation":
            _draw_annotation_layer(ax, lay, shared_legend)
        elif lay.kind == "quiver":
            _draw_quiver_layer(ax, lay)
        elif lay.kind == "streamplot":
            _draw_streamplot_layer(ax, lay)
        elif lay.kind == "divergence":
            _draw_divergence_layer(ax, lay)
        else:
            log.warning("Unknown layer kind %r; skipping.", lay.kind)

    if shared_legend:
        handles = [h for h, _ in shared_legend]
        labels = [lb for _, lb in shared_legend]
        _place_legend(ax, handles, labels)

    # Auto-zoom to crop region stored by pp.crop()
    bounds = _panel_crop_bounds(panel)
    if bounds is not None:
        ax.set_xlim(bounds["xmin"], bounds["xmax"])
        ax.set_ylim(bounds["ymin"], bounds["ymax"])

    if panel.title:
        ax.set_title(panel.title)
    ax.set_aspect("equal")
    if not panel.frameon:
        ax.axis("off")
    else:
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")

    # Spatial (µm-space) panels intentionally get no gridline -- they read
    # as a photographic/image-like view, not a statistical chart.
    _style_axes(ax, grid=False)

    if any(lay.kind == "image" and lay.dark_bg for lay in panel.layers):
        _apply_dark_theme(ax)


def _pad_to_aspect(
    img: np.ndarray,
    target_aspect: float,
    fill_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Pad *img* (H, W, C) with margin to reach *target_aspect*.

    Used instead of stretching so a wrapped figure's own content is never
    distorted to fit its assigned grid cell -- any gap between the raster's
    native aspect and the cell's is filled with margin, split evenly on
    both sides. ``bbox_inches="tight"`` can make a figure's exported raster
    wider than its nominal square ``figsize`` when it contains a legend
    that extends past the axes (the crop follows the legend, not the
    declared canvas size) -- padding back to the assigned cell's aspect
    keeps that legend from squeezing the actual plot.

    *fill_rgba* should match the theme surface color the wrapped figure was
    itself rendered against (opaque, not transparent): a transparent pad
    reveals whatever the destination axes' own facecolor is, and any
    sub-pixel difference from the raster's own background at the seam --
    plus anti-aliasing at that raster's tight-cropped edge -- shows as a
    visible border line once matplotlib stretches the low-res source into
    its (often much larger) final box.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0 or target_aspect <= 0:
        return img
    if img.shape[2] == 3:
        img = np.concatenate([img, np.ones((h, w, 1), dtype=img.dtype)], axis=2)
    fill = np.array(fill_rgba, dtype=img.dtype)

    current_aspect = w / h
    # Ignore near-equal aspects -- avoids a 1px pad from floating-point noise.
    if abs(current_aspect - target_aspect) / target_aspect < 1e-3:
        return img

    if current_aspect < target_aspect:
        target_w = max(w, int(round(h * target_aspect)))
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        left = np.tile(fill, (h, pad_left, 1))
        right = np.tile(fill, (h, pad_right, 1))
        return np.concatenate([left, img, right], axis=1)
    else:
        target_h = max(h, int(round(w / target_aspect)))
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        top = np.tile(fill, (pad_top, w, 1))
        bottom = np.tile(fill, (pad_bottom, w, 1))
        return np.concatenate([top, img, bottom], axis=0)


def _render_figure_panel(
    ax: matplotlib.axes.Axes,
    src: matplotlib.figure.Figure,
    *,
    target_aspect: float = 1.0,
) -> None:
    """Embed *src* into *ax* by rasterising to PNG and displaying via imshow.

    The raster is padded (not stretched) to *target_aspect* -- the aspect
    ratio of the grid cell *ax* actually occupies -- before display, so a
    wrapped figure's own content (e.g. a UMAP scatter with its legend) is
    never distorted to fill a cell shaped differently than its raster.

    Rasterizes at the theme's ``savefig.dpi`` (300 by default -- see
    ``_rcparams_for`` in ``_theme.py``), not the source figure's own dpi:
    this raster is the actual thing displayed (embedded via imshow, not
    re-rendered as vectors like the outer SVG-exported figure), and it is
    routinely stretched into a box larger than the source figure's own
    declared figsize once dynamic sizing or a wide sibling panel grows the
    grid, so it needs real margin above "just enough for its own canvas"
    to still look sharp once enlarged.
    """
    import matplotlib.image as mpimg
    rc_dpi = plt.rcParams.get("savefig.dpi", 300)
    dpi = max(300, rc_dpi) if isinstance(rc_dpi, (int, float)) else 300
    buf = io.BytesIO()
    # facecolor explicit for the same reason as _svg_mimebundle_for: honor
    # src's own facecolor (possibly forced black by _apply_dark_theme)
    # instead of the global-theme-tracking 'savefig.facecolor' rcParam.
    src.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=src.get_facecolor())
    buf.seek(0)
    img = mpimg.imread(buf)
    surface_rgba = mcolors.to_rgba(get_theme().surface)
    img = _pad_to_aspect(img, target_aspect, fill_rgba=surface_rgba)
    ax.imshow(img, aspect="auto")
    ax.axis("off")


def _auto_point_size(ax: matplotlib.axes.Axes, coords: np.ndarray, n_cells: int) -> float:
    """Default marker area (matplotlib scatter ``s``, in points²).

    Sized from the axes' *actual* physical dimensions, not just ``n_cells``
    -- a panel dynamically resized by :class:`PlotGrid` (e.g. compressed to
    a short strip to preserve a neighboring panel's true aspect) renders at
    a very different µm-per-point scale than a standalone default panel, so
    a purely count-based marker size can come out wildly too large (cells
    completely overlapping) or too small once the physical box changes.
    Instead this targets a marker diameter that's a fixed fraction of the
    average cell spacing implied by the visible extent and cell count, then
    converts that µm target to points using the axes' real scale -- so
    markers read as "cells at roughly their natural spacing" regardless of
    how the panel ends up sized.
    """
    # Robust ranges: a single outlier centroid inflating the raw range would
    # inflate the implied cell spacing (and so the marker size) with it.
    xmin, xmax, ymin, ymax = _robust_extent(coords, warn=False)
    x_range = xmax - xmin
    y_range = ymax - ymin
    if x_range <= 0 or y_range <= 0 or n_cells <= 0:
        return max(0.5, min(50, 120_000 / max(n_cells, 1)))

    fig = ax.get_figure()
    pos = ax.get_position()
    fig_w_in, fig_h_in = fig.get_size_inches()
    ax_w_pts = pos.width * fig_w_in * 72.0
    ax_h_pts = pos.height * fig_h_in * 72.0
    if ax_w_pts <= 0 or ax_h_pts <= 0:
        return max(0.5, min(50, 120_000 / n_cells))

    # points per µm -- equal in x/y since panels render at aspect="equal";
    # averaging both guards against using a stale pre-layout ax box where
    # they haven't quite converged to the same value yet.
    scale = (ax_w_pts / x_range + ax_h_pts / y_range) / 2.0

    avg_spacing_um = (x_range * y_range / n_cells) ** 0.5
    target_diameter_pts = avg_spacing_um * 0.7 * scale
    size = np.pi * (target_diameter_pts / 2.0) ** 2
    return max(0.3, min(200.0, size))


def _draw_points_layer(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    legend_entries: list | None = None,
) -> None:
    adata = lay.adata
    _check_single_library(adata)
    coords = adata.obsm[lay.spatial_key]
    n_cells = len(coords)
    size = lay.size if lay.size is not None else _auto_point_size(ax, coords, n_cells)

    # Impose robust axis limits only when outliers were actually clipped --
    # for clean data _robust_extent returns the exact raw min/max and the
    # limits are left to matplotlib's normal autoscale (so image/boundary
    # layers stacked with '+' keep contributing to the shared extent).
    robust_ext = _robust_extent(coords)
    raw_ext = (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
    )

    color_values, is_categorical = _resolve_color_values(adata, lay.color)

    if is_categorical and color_values is not None:
        _plot_categorical(
            ax,
            coords,
            color_values,
            palette=lay.palette,
            size=size,
            alpha=lay.alpha,
            na_color=lay.na_color,
            legend_entries=legend_entries,
            **lay.extra_kwargs,
        )
    else:
        norm = (
            mcolors.TwoSlopeNorm(lay.vcenter, vmin=lay.vmin, vmax=lay.vmax)
            if lay.vcenter is not None
            else None
        )
        vmin, vmax = lay.vmin, lay.vmax
        if norm is None and color_values is not None:
            # 1st/99.8th percentile stretch, matching pl.image()'s
            # _normalize_image_channel -- a plain continuous scatter left to
            # matplotlib's raw min/max washes out on a handful of outlier
            # cells (total_counts, cell_area, top SVGs all show this).
            # Skipped for the vcenter/diverging path: a signed statistic's
            # symmetric range is deliberate, not an outlier artifact.
            finite = color_values[np.isfinite(color_values)]
            if finite.size > 0:
                if vmin is None:
                    vmin = float(np.percentile(finite, 1.0))
                if vmax is None:
                    vmax = float(np.percentile(finite, 99.8))
                if vmax is not None and vmin is not None and vmax <= vmin:
                    vmax = vmin + 1.0
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_values,
            s=size,
            alpha=lay.alpha,
            cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
            rasterized=n_cells >= _RASTERIZE_MIN_MARKS,
            **lay.extra_kwargs,
        )
        if color_values is not None:
            _register_colorbar(ax, plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=lay.color))

    if robust_ext != raw_ext:
        ax.set_xlim(robust_ext[0], robust_ext[1])
        ax.set_ylim(robust_ext[2], robust_ext[3])


# Additive-blending colors for multi-channel fluorescence composites.
# Ch0 → blue (DAPI), Ch1 → green, Ch2 → red, Ch3 → magenta, Ch4 → cyan, Ch5 → yellow
_CHANNEL_COLORS_RGB: np.ndarray = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


_MAX_IMAGE_DIM = 4096  # max pixels on longest side loaded into memory for display


def _load_tiff_level(
    path: str,
    dimension_order: str,
    level_idx: int,
    *,
    crop: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, int]:
    """Open *path*, read pyramid level *level_idx*, collapse extra dims, stride if needed.

    Returns ``(raw, ch_axis, stride_step)`` where *raw* has axes ``(C, Y, X)``,
    *ch_axis* is the channel axis index (always 0 after collapsing), and
    *stride_step* is the integer stride used for downsampling (1 = none) —
    needed by the caller to compute the effective µm-per-pixel of *raw* for
    µm-specified image filters.  Z is reduced via
    max-intensity projection; T and other non-spatial dims are collapsed by
    taking index 0 (a view via basic indexing).  Stride downsampling is also
    a view.  The TiffFile is opened and closed within this call.

    Parameters
    ----------
    crop
        Optional ``(row_min, row_max, col_min, col_max)`` pixel crop applied
        **before** stride-downsampling so that the returned array covers only
        the visible region at full level resolution.
    """
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        level_idx = min(level_idx, len(levels) - 1)
        raw: np.ndarray = levels[level_idx].asarray()

    axes = dimension_order.upper()

    # Collapse non-spatial, non-channel dims.
    # Z → max-intensity projection (matches Xenium/standard fluorescence convention).
    # T and others → take index 0 (view via basic indexing).
    # Iterating in reverse keeps axis indices valid after each collapse.
    dims_to_drop = [(i, ax) for i, ax in enumerate(axes) if ax not in ("C", "Y", "X")]
    for i, dim_ax in sorted(dims_to_drop, reverse=True):
        if dim_ax == "Z":
            raw = raw.max(axis=i)
        else:
            raw = raw[(slice(None),) * i + (0,)]  # view (basic indexing)

    remaining = "".join(ax for ax in axes if ax in ("C", "Y", "X"))
    ch_ax = remaining.index("C") if "C" in remaining else 0
    y_ax = remaining.index("Y") if "Y" in remaining else (1 if raw.ndim >= 2 else 0)
    x_ax = remaining.index("X") if "X" in remaining else (raw.ndim - 1)

    # Spatial crop before stride — improves resolution for zoomed-in views.
    if crop is not None:
        row_min, row_max, col_min, col_max = crop
        sl: list[Any] = [slice(None)] * raw.ndim
        sl[y_ax] = slice(row_min, row_max)
        sl[x_ax] = slice(col_min, col_max)
        raw = raw[tuple(sl)]

    # Stride-downsample to fit within _MAX_IMAGE_DIM on the longest side.
    h, w = raw.shape[y_ax], raw.shape[x_ax]
    step = max(1, max(h, w) // _MAX_IMAGE_DIM)
    if step > 1:
        slices: list[Any] = [slice(None)] * raw.ndim
        slices[y_ax] = slice(None, None, step)
        slices[x_ax] = slice(None, None, step)
        raw = raw[tuple(slices)]   # view (no copy for C-contiguous stride)

    # Normalize to channel-first *after* crop/stride (which index by the
    # original y_ax/x_ax positions above). Without this, callers that assume
    # "(C, Y, X)" per this function's own contract — _apply_registration_rotation,
    # _apply_filter_chain, _apply_image_transform — silently misinterpret a
    # channels-last file (e.g. H&E's 'YXC') as having hundreds of "channels"
    # (one per image row). Only done when a real channel axis exists.
    if "C" in remaining and ch_ax != 0:
        raw = np.moveaxis(raw, ch_ax, 0)
        ch_ax = 0

    return raw, ch_ax, step


def _apply_filter_chain(
    raw: np.ndarray,
    filters: list,
    ch_ax: int,
    um_per_px: float | None = None,
) -> np.ndarray:
    """Apply each filter in *filters* independently to every channel of *raw*.

    *raw* is expected to be ``(C, Y, X)`` after ``_load_tiff_level`` (ch_ax==0).
    Each filter receives a 2-D float32 channel and must return float32.
    *um_per_px* is the effective physical pixel size of *raw* (level-0 pixel
    size × pyramid-level scale × stride step), forwarded so µm-specified
    filter bandwidths resolve to the same physical scale here as in
    ``pp.extract_if_intensities``'s full-resolution path.
    """
    raw = raw.astype(np.float32)
    if raw.ndim == 2:
        ch = raw
        for f in filters:
            ch = _call_image_filter(f, ch, um_per_px)
        return ch
    n_ch = raw.shape[ch_ax]  # ch_ax is always 0 here
    channels = []
    for i in range(n_ch):
        ch = raw[i].astype(np.float32)
        for f in filters:
            ch = _call_image_filter(f, ch, um_per_px)
        channels.append(ch)
    return np.stack(channels, axis=0)


def _select_level_and_crop(
    spec: ImageSpec,
    image_extent: list[float],
    bounds: dict,
) -> tuple[int, tuple[int, int, int, int]]:
    """Choose the finest useful pyramid level and pixel crop for a µm viewport.

    Selects the finest level where the *visible crop region* fits within
    ``_MAX_IMAGE_DIM`` pixels (capped at 4×_MAX_IMAGE_DIM full-image load to
    avoid reading excessively large levels).

    Returns
    -------
    level_idx
        Index into ``spec.level_shapes``.
    crop
        ``(row_min, row_max, col_min, col_max)`` pixel coordinates at *level_idx*.
    """
    xmin_img, xmax_img, ymin_img, ymax_img = image_extent
    img_w_um = max(xmax_img - xmin_img, 1e-9)
    img_h_um = max(ymax_img - ymin_img, 1e-9)

    x_frac = (bounds["xmax"] - bounds["xmin"]) / img_w_um
    y_frac = (bounds["ymax"] - bounds["ymin"]) / img_h_um
    crop_frac = max(x_frac, y_frac, 1e-9)

    # Full-image dimension at which the crop region would be exactly _MAX_IMAGE_DIM px.
    target_for_crop = int(_MAX_IMAGE_DIM / crop_frac)
    # Cap: don't load a level larger than 4×_MAX_IMAGE_DIM to bound memory.
    target_px = min(target_for_crop, _MAX_IMAGE_DIM * 4)
    level_idx = spec.select_pyramid_level(target_px)

    # Pixel coordinates of the crop at the chosen level.
    yscale, xscale = spec.level_scales[level_idx]
    h_full, w_full = _spec_yx_full_shape(spec)
    h_level = int(round(h_full / yscale))
    w_level = int(round(w_full / xscale))

    if spec.registration_affine is not None:
        # General case (handles rotation): map the µm bounds' four corners into
        # full-resolution pixel space via the inverse affine, then take the
        # axis-aligned bounding box — the same "bounding box of a rotated
        # region" approach used to physically rotate the loaded crop afterward.
        Minv = np.linalg.inv(spec.registration_affine)
        corners_um = np.array(
            [
                [bounds["xmin"], bounds["ymin"]], [bounds["xmax"], bounds["ymin"]],
                [bounds["xmin"], bounds["ymax"]], [bounds["xmax"], bounds["ymax"]],
            ],
            dtype=np.float64,
        )
        corners_px_full = _transform_coords(corners_um, Minv)
        col_min = max(0, int(corners_px_full[:, 0].min() / xscale))
        col_max = min(w_level, int(corners_px_full[:, 0].max() / xscale) + 1)
        row_min = max(0, int(corners_px_full[:, 1].min() / yscale))
        row_max = min(h_level, int(corners_px_full[:, 1].max() / yscale) + 1)
    else:
        col_min = max(0, int((bounds["xmin"] - xmin_img) / img_w_um * w_level))
        col_max = min(w_level, int((bounds["xmax"] - xmin_img) / img_w_um * w_level) + 1)
        row_min = max(0, int((bounds["ymin"] - ymin_img) / img_h_um * h_level))
        row_max = min(h_level, int((bounds["ymax"] - ymin_img) / img_h_um * h_level) + 1)

    return level_idx, (row_min, row_max, col_min, col_max)


def _spec_yx_full_shape(spec: ImageSpec) -> tuple[int, int]:
    """Return ``(h_full, w_full)`` by locating Y/X in ``spec.dimension_order``.

    Do not assume the last two axes of ``full_shape`` are ``(Y, X)`` — true
    for ``CYX``-style fluorescence/morphology images, but false for
    ``YXC``/``YXS``-style RGB images (e.g. H&E, where channels are stored
    last per :func:`astrolabe.io._he._extract_he_spec`).
    """
    axes = spec.dimension_order.upper()
    y_idx = axes.index("Y") if "Y" in axes else len(spec.full_shape) - 2
    x_idx = axes.index("X") if "X" in axes else len(spec.full_shape) - 1
    return spec.full_shape[y_idx], spec.full_shape[x_idx]


def _compute_extent(spec: ImageSpec, h_full: int, w_full: int) -> list[float]:
    """Map image corners to the reference µm frame.

    If ``spec.registration_affine`` is set, its full affine — including any
    rotation — is applied to the four image corners and the resulting
    axis-aligned bounding box is used as the extent (a pure scale+translation
    affine reduces to the same result as the old axis-aligned-only special
    case, so no separate branch is needed).  Otherwise falls back to
    ``pixel_size_um`` scaling from the origin.  Any transforms in
    ``spec.transform_chain`` are then applied on top.
    """
    if spec.registration_affine is not None:
        A = spec.registration_affine
        corners_px = np.array(
            [[0.0, 0.0], [w_full, 0.0], [0.0, h_full], [w_full, h_full]],
            dtype=np.float64,
        )
        corners_um = _transform_coords(corners_px, A)
        xmin, xmax = float(corners_um[:, 0].min()), float(corners_um[:, 0].max())
        ymin, ymax = float(corners_um[:, 1].min()), float(corners_um[:, 1].max())
    else:
        xmin, xmax = 0.0, w_full * spec.pixel_size_um
        ymin, ymax = 0.0, h_full * spec.pixel_size_um

    if spec.transform_chain:
        M = spec.combined_transform
        corners = np.array(
            [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]],
            dtype=np.float64,
        )
        corners = _transform_coords(corners, M)
        xmin, xmax = float(corners[:, 0].min()), float(corners[:, 0].max())
        ymin, ymax = float(corners[:, 1].min()), float(corners[:, 1].max())

    return [xmin, xmax, ymin, ymax]


def _normalize_image_channel(
    img_2d: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    gamma: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Percentile-stretch a 2-D channel to float32 [0, 1].

    *valid_mask* (from :func:`_apply_registration_rotation`) restricts the
    percentile computation to real image pixels: a rotated image's padding
    corners can be ~1% of the array or more, exactly enough to capture the
    99.8th-percentile anchor and crush the stretch of the actual tissue.
    """
    img_f = img_2d.astype(np.float32)
    if valid_mask is not None and valid_mask.shape != img_f.shape:
        # A later transform_chain rotation can reshape the array after the
        # mask was computed -- fall back to the unmasked stretch then.
        valid_mask = None
    ref = img_f if valid_mask is None else img_f[valid_mask]
    if ref.size == 0:
        ref = img_f
    lo = float(vmin) if vmin is not None else float(np.percentile(ref, 1.0))
    hi = float(vmax) if vmax is not None else float(np.percentile(ref, 99.8))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img_f - lo) / (hi - lo), 0.0, 1.0) ** gamma


def _apply_dark_theme(ax: matplotlib.axes.Axes) -> None:
    """Black axes background with white text — matches IF presentation style.

    Only styles *this* axes' own facecolor and chrome colors. The
    black background behind this axes' full chrome (title, tick labels,
    legend -- everything that renders *outside* the axes' own plot
    rectangle, in shared figure-level space) is a separate, later step:
    see :func:`_add_dark_panel_backgrounds`, called once per figure after
    layout is resolved. Composed panels (``|``/``hstack``, e.g. an
    IF/morphology image next to an H&E panel) share one figure across
    multiple axes, so this function must not touch ``fig.patch`` or assume
    it owns the whole figure -- a light-themed sibling panel's own chrome
    needs to stay legible regardless of what this axes does.
    """
    ax.set_facecolor("black")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    # Style all legends added to this axes (including add_artist channel legends).
    for artist in ax.get_children():
        if isinstance(artist, _Legend):
            artist.get_frame().set_facecolor("none")
            artist.get_frame().set_edgecolor("none")
            for text in artist.get_texts():
                text.set_color("white")


def _add_dark_panel_backgrounds(
    fig: matplotlib.figure.Figure, dark_axes: list[matplotlib.axes.Axes]
) -> None:
    """Draw one opaque black rectangle behind each dark-themed axes' full chrome.

    ``_apply_dark_theme`` colors an axes' own title/tick/label/legend text
    white and its plot area black, but title/tick-label/legend text all
    render *outside* that axes' own plot rectangle, in shared figure-level
    space. On a light ``fig.patch`` (deliberately left light so any
    light-themed sibling panel's own chrome stays legible -- see
    ``_apply_dark_theme``), that white text would otherwise strand itself
    the same way a light sibling's text used to. Rather than patching every
    chrome element with its own tiny bbox (fragile -- easy to miss one,
    e.g. a boundaries-layer legend), draw one black rectangle sized to each
    dark axes' own *tight* bounding box (title + ticks + labels + legend,
    everything matplotlib considers part of that axes) and place it behind
    that axes in the shared figure, computed once layout is fully resolved
    so the rectangle lands exactly where that chrome actually ends up.
    """
    if not dark_axes:
        return
    fig.canvas.draw()  # resolve constrained_layout with all final content in place
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax in dark_axes:
        bbox = ax.get_tightbbox(renderer).padded(3)
        fig_bbox = bbox.transformed(inv)
        rect = matplotlib.patches.Rectangle(
            (fig_bbox.x0, fig_bbox.y0), fig_bbox.width, fig_bbox.height,
            transform=fig.transFigure, facecolor="black", edgecolor="none",
            zorder=-10,
        )
        fig.add_artist(rect)


_SCALE_BAR_CANDIDATES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _round_scale_bar(raw: float) -> float:
    """Return the smallest "nice" number ≥ *raw* from the candidate list."""
    for v in _SCALE_BAR_CANDIDATES:
        if v >= raw:
            return float(v)
    return float(_SCALE_BAR_CANDIDATES[-1])


def _draw_scale_bar(
    ax: matplotlib.axes.Axes,
    extent: list[float],
    spec: bool | dict,
) -> None:
    """Draw a scale bar inside the image in µm data coordinates."""
    if spec is False:
        return
    kwargs: dict = {} if spec is True else dict(spec)

    xmin, xmax, ymin, ymax = extent
    x_range, y_range = xmax - xmin, ymax - ymin

    raw = min(x_range, y_range) * 0.20
    length_um = float(kwargs.get("length_um") or _round_scale_bar(raw))

    location = str(kwargs.get("location", "lower-right"))
    color    = kwargs.get("color", "white")
    fontsize = kwargs.get("fontsize", 8)
    lw       = kwargs.get("linewidth", 2.0)

    pad_x = x_range * 0.05
    pad_y = y_range * 0.05

    if "right" in location:
        x_end   = xmax - pad_x
        x_start = x_end - length_um
    else:
        x_start = xmin + pad_x
        x_end   = x_start + length_um

    y_bar = (ymin + pad_y) if "lower" in location else (ymax - pad_y)

    ax.plot(
        [x_start, x_end], [y_bar, y_bar],
        color=color, linewidth=lw, solid_capstyle="butt", zorder=10,
    )
    ax.text(
        (x_start + x_end) / 2.0, y_bar + y_range * 0.015,
        f"{length_um:g} µm",
        color=color, fontsize=fontsize, ha="center", va="bottom", zorder=10,
    )


def _apply_image_transform(
    img: np.ndarray,
    spec: ImageSpec,
    extent: list[float],
) -> tuple[np.ndarray, list[float]]:
    """Apply spec.transform_chain to pixel data and return updated (img, extent).

    - Pure flips (diagonal-only transform): reverse array axis — no copy, no resampling.
    - Rotations (off-diagonal entries present): scipy.ndimage.rotate at render time.
    """
    import scipy.ndimage

    M = spec.combined_transform
    if M is None:
        return img, extent

    # Decompose into rotation angle + flip.
    # M is in µm space; for axis-aligned content the linear part is:
    #   [[sx*cos θ, -sy*sin θ], [sx*sin θ, sy*cos θ]]
    # We detect flip by checking the sign of the determinant of the 2×2 block.
    L = M[:2, :2]
    det = float(np.linalg.det(L))
    angle_rad = float(np.arctan2(L[1, 0], L[0, 0]))
    angle_deg = float(np.degrees(angle_rad))

    # Identify spatial axes in the loaded (possibly C,Y,X or Y,X) array
    ndim = img.ndim
    if ndim == 3:
        y_ax, x_ax = 1, 2  # C, Y, X
    else:
        y_ax, x_ax = 0, 1  # Y, X

    # Apply flip if determinant < 0 (orientation-reversing transform)
    if det < 0:
        # Flip along x axis (mirror left-right)
        img = np.flip(img, axis=x_ax)

    # Apply rotation if non-trivial angle
    if abs(angle_deg) > 0.1:
        if ndim == 3:
            # Rotate each channel independently; reshape → rotate → reshape
            rotated_channels = [
                scipy.ndimage.rotate(
                    img[ci], angle_deg, reshape=True, order=1, prefilter=False
                )
                for ci in range(img.shape[0])
            ]
            img = np.stack(rotated_channels, axis=0)
        else:
            img = scipy.ndimage.rotate(img, angle_deg, reshape=True, order=1, prefilter=False)

    # extent is already the correct transformed bounding box from _compute_extent;
    # just return it unchanged.
    return img, extent


def _apply_registration_rotation(
    raw: np.ndarray,
    spec: ImageSpec,
    level_idx: int,
    crop_pixels: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, list[float] | None, np.ndarray | None]:
    """Physically rotate/flip *raw* per a rotated ``spec.registration_affine``.

    Unlike a pure scale+translation registration (already handled correctly
    by ``extent=`` alone), a rotated registration means the loaded pixel
    array's own row/col axes are not aligned with the µm x/y axes — it must
    be resampled, same as ``pp.rotate``/``pp.flip``'s ``transform_chain``
    handling in :func:`_apply_image_transform`.  Returns
    ``(raw, None, None)`` unchanged when the registration has no meaningful
    rotation.

    *crop_pixels* (if given) is the same ``(row_min, row_max, col_min, col_max)``
    region passed to :func:`_load_tiff_level` — used to recompute the true µm
    extent of exactly what was loaded, since a rotated crop's pixel-space
    bounding box loaded here is larger than the originally requested µm
    viewport (the classic "bounding box of a rotated region" inflation).

    The third return value is a 2-D boolean **validity mask** (``True`` on
    real pixels, ``False`` on the corner padding scipy's rotation adds),
    computed by rotating a ones companion image with ``cval=0``.  Padding
    corners can cover a nontrivial fraction of the array, so any percentile
    contrast stretch must ignore them (see :func:`_normalize_image_channel`).
    Rotation padding itself uses ``cval`` = white (dtype max) for
    brightfield/RGB images — black padding on an H&E slide reads as maximal
    optical density and blows up stain deconvolution — and the conventional
    0 (background) for fluorescence images.
    """
    A = spec.registration_affine
    if A is None:
        return raw, None, None

    L = A[:2, :2]
    off_diag = abs(L[0, 1]) + abs(L[1, 0])
    if off_diag <= 1e-6:
        return raw, None, None

    import scipy.ndimage

    yscale, xscale = spec.level_scales[level_idx]
    # row_min/row_max/col_min/col_max must be in *level*-space pixel units in
    # both branches, since they're converted to full-resolution units via the
    # `* yscale` / `* xscale` multiplication below. _spec_yx_full_shape already
    # returns full-resolution units, so the no-crop branch must first convert
    # down to level-space (previously it multiplied full-res h_full/w_full by
    # yscale/xscale directly, double-applying the level scale and producing a
    # wildly inflated extent whenever the whole image was rendered without a
    # prior pp.crop()).
    if crop_pixels is not None:
        row_min, row_max, col_min, col_max = crop_pixels
    else:
        h_full, w_full = _spec_yx_full_shape(spec)
        row_min, row_max = 0, int(round(h_full / yscale))
        col_min, col_max = 0, int(round(w_full / xscale))

    corners_px_full = np.array(
        [
            [col_min * xscale, row_min * yscale], [col_max * xscale, row_min * yscale],
            [col_min * xscale, row_max * yscale], [col_max * xscale, row_max * yscale],
        ],
        dtype=np.float64,
    )
    corners_um = _transform_coords(corners_px_full, A)
    extent = [
        float(corners_um[:, 0].min()), float(corners_um[:, 0].max()),
        float(corners_um[:, 1].min()), float(corners_um[:, 1].max()),
    ]

    det = float(np.linalg.det(L))
    angle_deg = float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))

    # raw is always (C, Y, X) per _load_tiff_level.
    if det < 0:
        raw = np.flip(raw, axis=2)

    valid_mask: np.ndarray | None = None
    if abs(angle_deg) > 0.1:
        # scipy.ndimage.rotate's positive-angle direction rotates axis 1 (col)
        # towards axis 0 (row) in the array's own row-down indexing sense.
        # This code displays the rotated array via imshow(..., origin='lower'),
        # which is row-UP (row 0 at the bottom) -- the opposite sense. That
        # sense mismatch flips the effective rotation direction, so the angle
        # must be negated here to match the affine's real-world rotation
        # (verified empirically: without the negation, a pixel that should
        # land at one corner of the viewport instead lands at the diagonally
        # opposite corner -- a full 180° point-reflection, since rotating by
        # -angle instead of +angle differs by exactly 180° for angles near
        # +-90°, as is the case for this dataset's ~90° H&E registration).
        #
        # Padding value: white (dtype max) for brightfield/RGB so the
        # rotation corners look like empty slide rather than pure black
        # (which OD-deconvolves to a maximal stain concentration); 0 for
        # fluorescence, where black *is* background.
        if spec.is_brightfield:
            cval = (
                float(np.iinfo(raw.dtype).max)
                if np.issubdtype(raw.dtype, np.integer)
                else 1.0
            )
        else:
            cval = 0.0
        pre_rotation_shape = raw.shape[1:]  # (Y, X) of what actually gets rotated
        rotated_channels = [
            scipy.ndimage.rotate(
                raw[ci], -angle_deg, reshape=True, order=1, prefilter=False, cval=cval
            )
            for ci in range(raw.shape[0])
        ]
        raw = np.stack(rotated_channels, axis=0)
        # Ones-companion rotation marks which output pixels contain real
        # image data (padding corners come out ~0). Slight interpolation
        # feathering at the boundary is fine -- 0.5 splits it cleanly.
        ones = np.ones(pre_rotation_shape, dtype=np.float32)
        rotated_ones = scipy.ndimage.rotate(
            ones, -angle_deg, reshape=True, order=1, prefilter=False, cval=0.0
        )
        valid_mask = rotated_ones > 0.5

    return raw, extent, valid_mask


def _draw_annotation_layer(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    legend_entries: list | None = None,
) -> None:
    """Render polygon annotation overlays onto *ax*.

    Two modes: a single literal color for every shape (default, and the
    fallback when *lay.annotation_values* is unset), or per-shape
    categorical/continuous coloring (mirrors ``_draw_boundary_layer``'s
    per-polygon coloring) when :func:`annotation` resolved *color* against a
    GeoDataFrame column.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    shapes = lay.shapes or []
    if not shapes:
        return

    values = lay.annotation_values
    if values is None:
        patches = [MplPolygon(s, closed=True) for s in shapes]
        fc = (
            mcolors.to_rgba(lay.annotation_fill, lay.fill_alpha)
            if lay.annotation_fill is not None
            else "none"
        )
        col = PatchCollection(
            patches,
            facecolor=fc,
            edgecolor=lay.annotation_color,
            linewidth=lay.linewidth,
            alpha=lay.alpha,
            zorder=4,
        )
        ax.add_collection(col)
        ax.autoscale_view()
        return

    patches = [MplPolygon(s, closed=True) for s in shapes]
    is_categorical = not np.issubdtype(values.dtype, np.number)

    if is_categorical:
        resolved_na = lay.resolved_na_color
        unique_cats = _sorted_categories(set(values))
        cat_colors = _build_cat_colors(unique_cats, lay.palette, resolved_na)
        base_c = [mcolors.to_rgb(cat_colors.get(str(v), resolved_na)) for v in values]
        edge_c = [(*rgb, lay.alpha) for rgb in base_c]
        face_c = [(*rgb, lay.fill_alpha) for rgb in base_c]
        col = PatchCollection(
            patches, facecolor=face_c, edgecolor=edge_c, linewidth=lay.linewidth, zorder=4,
        )
        ax.add_collection(col)
        if legend_entries is not None:
            handles = _legend_handles(unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats])
            legend_entries.extend(zip(handles, (str(c) for c in unique_cats)))
    else:
        vals = values.astype(np.float64)
        vmin = lay.vmin if lay.vmin is not None else float(np.nanmin(vals))
        vmax = lay.vmax if lay.vmax is not None else float(np.nanmax(vals))
        norm = (
            mcolors.TwoSlopeNorm(lay.vcenter, vmin=vmin, vmax=vmax)
            if lay.vcenter is not None
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        cmap_fn = plt.get_cmap(lay.cmap or _DEFAULT_CONTINUOUS_CMAP)
        mapped = [mcolors.to_rgba(cmap_fn(norm(v))) for v in vals]
        edge_c = [(*c[:3], lay.alpha) for c in mapped]
        face_c = [(*c[:3], lay.fill_alpha) for c in mapped]
        col = PatchCollection(
            patches, facecolor=face_c, edgecolor=edge_c, linewidth=lay.linewidth, zorder=4,
        )
        ax.add_collection(col)
        sm = plt.cm.ScalarMappable(cmap=cmap_fn, norm=norm)
        sm.set_array([])
        _register_colorbar(ax, plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=lay.color))

    ax.autoscale_view()


def _draw_image_layer(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    legend_entries: list | None = None,
) -> None:
    if lay.dark_bg:
        ax.set_facecolor("black")

    lib_uns = lay.adata.uns["spatial"][lay.library_id]
    spec: ImageSpec = lib_uns["images"][lay.image_key]

    # Full-resolution shape for physical extent (from metadata, not from loaded array).
    h_full, w_full = _spec_yx_full_shape(spec)
    extent = _compute_extent(spec, h_full, w_full)

    # Crop-aware level selection: if spatial_bounds are set (after ab.pp.crop), pick the
    # finest pyramid level where the *visible crop region* fits within _MAX_IMAGE_DIM px.
    bounds_uns = lay.adata.uns.get("spatial_bounds")
    crop_pixels: tuple[int, int, int, int] | None = None
    if bounds_uns is not None:
        level_idx, crop_pixels = _select_level_and_crop(spec, extent, bounds_uns)
        # Narrow the rendering extent to the cropped µm viewport.
        extent = [bounds_uns["xmin"], bounds_uns["xmax"], bounds_uns["ymin"], bounds_uns["ymax"]]
    else:
        level_idx = spec.select_pyramid_level(_MAX_IMAGE_DIM)

    log.debug(
        "Image '%s/%s': using pyramid level %d of %d (target ≤ %dpx, crop=%s)",
        lay.library_id, lay.image_key, level_idx, spec.n_levels, _MAX_IMAGE_DIM, crop_pixels,
    )

    raw, ch_ax, stride_step = _load_tiff_level(
        spec.path, spec.dimension_order, level_idx, crop=crop_pixels
    )

    # Physically rotate/flip pixels for a rotated registration_affine (e.g. an
    # H&E scan registered at an angle relative to the Xenium frame). No-op when
    # the registration is axis-aligned or absent. valid_mask marks real (non-
    # padding) pixels so percentile stretches ignore the rotation corners.
    raw, rotated_extent, valid_mask = _apply_registration_rotation(raw, spec, level_idx, crop_pixels)
    if rotated_extent is not None:
        extent = rotated_extent

    # Apply per-channel image filters (filter_chain set by ab.pp.add_image_filter).
    if spec.filter_chain:
        # Effective µm-per-pixel of `raw`: level-0 pixel size × the pyramid
        # level's downsampling factor × the display stride step. Registration
        # rotation (above) resamples onto the same grid spacing, so it does
        # not change the pixel size. Forwarding this keeps µm-specified
        # filter bandwidths physically identical across zoom levels and
        # consistent with pp.extract_if_intensities' full-resolution path.
        base_um_per_px = spec.um_per_px
        if base_um_per_px is not None:
            _, xscale = spec.level_scales[min(level_idx, spec.n_levels - 1)]
            eff_um_per_px: float | None = base_um_per_px * float(xscale) * stride_step
        else:
            eff_um_per_px = None
        raw = _apply_filter_chain(raw, spec.filter_chain, ch_ax, eff_um_per_px)

    # Apply lazy pixel transforms (rotate/flip stored in transform_chain).
    if spec.transform_chain:
        raw, extent = _apply_image_transform(raw, spec, extent)

    # Split channels from the numpy array (all zarr I/O is done above).
    if raw.ndim == 2:
        ch_slices: list[np.ndarray] = [raw]
        n_ch = 1
    else:
        n_ch = raw.shape[ch_ax]
        ch_slices = [raw[i] if ch_ax == 0 else np.take(raw, i, axis=ch_ax) for i in range(n_ch)]

    def _prep(arr2d: np.ndarray) -> np.ndarray:
        if lay.normalize:
            return _normalize_image_channel(arr2d, lay.vmin, lay.vmax, lay.gamma, valid_mask)
        dtype_max = (
            float(np.iinfo(arr2d.dtype).max)
            if np.issubdtype(arr2d.dtype, np.integer)
            else 1.0
        )
        return np.clip(arr2d.astype(np.float32) / dtype_max, 0.0, 1.0) ** lay.gamma

    is_brightfield_rgb = spec.is_brightfield and n_ch == 3

    if lay.channel is None:
        imgs = [_prep(ch) for ch in ch_slices]
        if n_ch == 1:
            img_display: np.ndarray = imgs[0]
            cmap: str | None = lay.image_cmap
        elif is_brightfield_rgb:
            # Brightfield RGB: channels ARE literal red/green/blue (or, when
            # separate_stains=True, the raw RGB planes underlying the
            # deconvolved stain labels) -- composite directly, no per-channel
            # pseudo-coloring or legend (that treatment is for IF panels).
            img_display = np.stack(imgs, axis=-1)
            cmap = None
        else:
            h_d, w_d = imgs[0].shape
            rgb = np.zeros((h_d, w_d, 3), dtype=np.float32)
            for i, ch_img in enumerate(imgs):
                rgb += ch_img[:, :, np.newaxis] * _CHANNEL_COLORS_RGB[i % len(_CHANNEL_COLORS_RGB)]
            img_display = np.clip(rgb, 0.0, 1.0)
            cmap = None
    else:
        ch_idx = _resolve_channel_index(lay.channel, lay.adata)
        if ch_idx >= n_ch:
            raise IndexError(f"channel={ch_idx} is out of range; image has {n_ch} channel(s)")
        # A *named* channel (not a raw int index) was requested on an H&E
        # image loaded with separate_stains=True -- its channel_names/var_names
        # ARE stain names (hematoxylin/eosin/residual) in that state, so apply
        # real per-pixel OD deconvolution and show that stain's concentration
        # map, rather than the raw (mislabeled) RGB plane at this index. An
        # explicit int index always means "raw channel N", bypassing this.
        if is_brightfield_rgb and spec.stain_matrix is not None and isinstance(lay.channel, str):
            rgb_chw = np.stack(ch_slices, axis=0).astype(np.float64)
            stains = _od_deconvolve(rgb_chw, spec.stain_matrix)  # (H, W, 3)
            img_display = _normalize_image_channel(
                stains[..., ch_idx], lay.vmin, lay.vmax, lay.gamma, valid_mask
            )
        else:
            img_display = _prep(ch_slices[ch_idx])
        cmap = lay.image_cmap

    ax.imshow(
        img_display,
        cmap=cmap,
        extent=extent,
        origin="lower",
        aspect="equal",
        rasterized=h_full * w_full > 500_000,
    )

    _draw_scale_bar(ax, extent, lay.scale_bar)

    # Channel legend for multi-channel composite (not shown for single-channel,
    # when a specific channel is selected, or for a brightfield RGB composite,
    # which is one photographic image rather than distinct pseudo-colored channels).
    if lay.channel is None and n_ch > 1 and not is_brightfield_rgb:
        if len(lay.adata.var_names) == n_ch:
            ch_names = list(lay.adata.var_names)
        elif spec.channel_names is not None and len(spec.channel_names) == n_ch:
            ch_names = list(spec.channel_names)
        else:
            ch_names = [f"Ch{i}" for i in range(n_ch)]
        if legend_entries is not None:
            for i in range(n_ch):
                color = tuple(float(v) for v in _CHANNEL_COLORS_RGB[i % len(_CHANNEL_COLORS_RGB)])
                handle = _legend_handles([ch_names[i]], [color])[0]
                legend_entries.append((handle, ch_names[i]))


def _draw_boundary_layer(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    legend_entries: list | None = None,
) -> None:
    adata = lay.adata
    _check_single_library(adata)
    raw = adata.obsm[lay.boundaries_key]

    # Vectorised extraction: avoid a Python loop over n_cells.
    # The awkward ListOffsetArray wraps a flat (total_verts, 2) NumpyArray;
    # ak.num gives per-cell vertex counts and ak.flatten recovers the flat array.
    # `raw` is either one ring per cell (n_cells, n_verts, 2) or, for a
    # multi-part region such as a flux domain, a list of rings per cell
    # (n_cells, n_rings, n_verts, 2). `poly_cell` maps each drawn ring back to
    # its cell so per-cell colouring stays correct in both layouts.
    try:
        import awkward as ak
        if isinstance(raw, ak.Array):
            if raw.ndim >= 4:
                rings = np.asarray(ak.num(raw))               # rings per cell
                owner = np.repeat(np.arange(len(raw)), rings)
                inner = ak.flatten(raw, axis=1)               # rings, verts, 2
                counts = np.asarray(ak.num(inner))
                flat = np.asarray(ak.flatten(inner))
            else:
                owner = np.arange(len(raw))
                counts = np.asarray(ak.num(raw))              # (n_cells,) int
                flat = np.asarray(ak.flatten(raw))            # (total_verts, 2)
            polys = np.split(flat, np.cumsum(counts[:-1]))
        else:
            raise TypeError
    except (ImportError, TypeError):
        polys = [np.asarray(raw[i]) for i in range(len(raw))]
        owner = np.arange(len(raw))

    if len(polys) != len(owner):     # empty trailing cells drop out of np.split
        owner = owner[:len(polys)]
    valid_mask = np.array([p.ndim == 2 and len(p) >= 3 for p in polys]) \
        if polys else np.zeros(0, dtype=bool)
    valid_polys = [polys[i] for i in np.where(valid_mask)[0]]
    valid_indices = np.asarray(owner)[np.where(valid_mask)[0]]

    if not valid_polys:
        return

    rasterized = len(valid_polys) >= _RASTERIZE_MIN_MARKS

    # Allow lay.color to be a literal matplotlib color string (e.g. 'steelblue', '#ff0000').
    # If it doesn't resolve to an obs column or gene, check whether it's a valid color first.
    literal_color: str | None = None
    try:
        color_values, is_categorical = _resolve_color_values(adata, lay.color)
    except KeyError:
        if lay.color is not None and mcolors.is_color_like(lay.color):
            color_values, is_categorical = None, False
            literal_color = lay.color
        else:
            raise

    if is_categorical and color_values is not None:
        unique_cats = _sorted_categories(set(color_values[valid_indices]))
        resolved_na = lay.resolved_na_color
        cat_colors = _build_cat_colors(unique_cats, lay.palette, resolved_na)
        base_c = [mcolors.to_rgb(cat_colors.get(str(color_values[i]), resolved_na)) for i in valid_indices]
        edge_c = [(*rgb, lay.alpha) for rgb in base_c]
        face_c = [(*rgb, lay.fill_alpha) for rgb in base_c]
        col = PolyCollection(
            valid_polys,
            facecolors=face_c,
            edgecolors=edge_c,
            linewidths=lay.linewidth,
            rasterized=rasterized,
        )
        ax.add_collection(col)
        if legend_entries is not None:
            handles = _legend_handles(unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats])
            legend_entries.extend(zip(handles, (str(c) for c in unique_cats)))
    elif color_values is not None:
        vals = color_values[valid_indices].astype(np.float64)
        vmin = lay.vmin if lay.vmin is not None else float(np.nanmin(vals))
        vmax = lay.vmax if lay.vmax is not None else float(np.nanmax(vals))
        norm = (
            mcolors.TwoSlopeNorm(lay.vcenter, vmin=vmin, vmax=vmax)
            if lay.vcenter is not None
            else mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        cmap_fn = plt.get_cmap(lay.cmap or _DEFAULT_CONTINUOUS_CMAP)
        mapped = [mcolors.to_rgba(cmap_fn(norm(v))) for v in vals]
        edge_c = [(*c[:3], lay.alpha) for c in mapped]
        face_c = [(*c[:3], lay.fill_alpha) for c in mapped]
        col = PolyCollection(
            valid_polys,
            facecolors=face_c,
            edgecolors=edge_c,
            linewidths=lay.linewidth,
            rasterized=rasterized,
        )
        ax.add_collection(col)
        sm = plt.cm.ScalarMappable(cmap=cmap_fn, norm=norm)
        sm.set_array([])
        _register_colorbar(ax, plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=lay.color))
    else:
        edge_color = literal_color if literal_color is not None else get_theme().muted
        face_rgb = mcolors.to_rgb(edge_color) if literal_color is not None else (0.0, 0.0, 0.0)
        edge_rgba = (*mcolors.to_rgb(edge_color), lay.alpha)
        col = PolyCollection(
            valid_polys,
            facecolors=[(*face_rgb, lay.fill_alpha)],
            edgecolors=[edge_rgba],
            linewidths=lay.linewidth,
            rasterized=rasterized,
        )
        ax.add_collection(col)
        if legend_entries is not None:
            label = lay.boundaries_key.replace("_", " ").title()
            # Unfilled ring when the layer draws outlines only, matching the
            # finalized dot-marker legend style; filled when the layer is
            # filled, since otherwise several filled layers of different
            # colours all read as the same empty ring in the key that is
            # supposed to tell them apart.
            handle = Line2D(
                [0], [0], marker="o", linestyle="none", markersize=6,
                markerfacecolor=(edge_color if lay.fill_alpha > 0 else "none"),
                markeredgecolor=edge_color, label=label,
            )
            legend_entries.append((handle, label))

    if lay.spatial_key in adata.obsm:
        coords = adata.obsm[lay.spatial_key]
        xmin, xmax, ymin, ymax = _robust_extent(coords)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.autoscale_view()


def _draw_raster_layer(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    legend_entries: list | None = None,
) -> None:
    adata = lay.adata
    _check_single_library(adata)
    coords = adata.obsm[lay.spatial_key]
    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    color_values, is_categorical = _resolve_color_values(adata, lay.color)
    if lay.bin_shape == "kde":
        _draw_raster_kde(ax, lay, x, y, color_values, is_categorical)
    elif lay.bin_shape == "hex":
        _draw_raster_hex(ax, lay, x, y, color_values, is_categorical, legend_entries)
    else:
        _draw_raster_square(ax, lay, x, y, color_values, is_categorical, legend_entries)


def _draw_raster_kde(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | None,
    is_categorical: bool,
) -> None:
    from scipy.stats import gaussian_kde

    if is_categorical:
        raise ValueError(
            "bin_shape='kde' is not supported for categorical color variables. "
            "Use bin_shape='square' or 'hex' for categorical coloring."
        )

    bins = lay.bins
    xgrid = np.linspace(x.min(), x.max(), bins)
    ygrid = np.linspace(y.min(), y.max(), bins)
    xx, yy = np.meshgrid(xgrid, ygrid)
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    if color_values is None:
        kde = gaussian_kde(np.vstack([x, y]), bw_method=lay.bandwidth)
        density = kde(grid_pts).reshape(bins, bins)
        label = "density"
    else:
        vals = color_values.astype(np.float64)
        # Weight points by their (absolute) value so the KDE reflects
        # where the signal is concentrated spatially.
        w = np.abs(vals)
        total = w.sum()
        weights = w / total if total > 0 else None
        kde = gaussian_kde(np.vstack([x, y]), bw_method=lay.bandwidth, weights=weights)
        density = kde(grid_pts).reshape(bins, bins)
        if lay.agg == "sum":
            # scale back to sum units
            density = density * total
        label = lay.color or "value"

    if lay.fill_color is not None:
        rgba = _density_to_rgba(density, lay.fill_color, lay.vmin, lay.vmax)
        ax.imshow(rgba, extent=extent, origin="lower", aspect="equal",
                  interpolation="bilinear")
    else:
        norm = (
            mcolors.TwoSlopeNorm(lay.vcenter, vmin=lay.vmin, vmax=lay.vmax)
            if lay.vcenter is not None else None
        )
        im = ax.imshow(
            density, extent=extent, origin="lower", aspect="equal",
            cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
            vmin=lay.vmin if norm is None else None,
            vmax=lay.vmax if norm is None else None,
            norm=norm, interpolation="bilinear",
        )
        _register_colorbar(ax, plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label))


def _draw_raster_square(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | None,
    is_categorical: bool,
    legend_entries: list | None = None,
) -> None:
    bins = lay.bins
    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)
    xi = np.clip(np.searchsorted(x_edges[1:-1], x), 0, bins - 1)
    yi = np.clip(np.searchsorted(y_edges[1:-1], y), 0, bins - 1)

    count_grid = np.zeros((bins, bins), dtype=np.float64)
    np.add.at(count_grid, (yi, xi), 1)
    extent = [float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])]

    if color_values is None:
        grid = np.where(count_grid > 0, count_grid, np.nan)
        if lay.fill_color is not None:
            rgba = _density_to_rgba(grid, lay.fill_color, lay.vmin, lay.vmax)
            ax.imshow(rgba, extent=extent, origin="lower", aspect="equal",
                      interpolation="nearest")
        else:
            im = ax.imshow(grid, extent=extent, origin="lower", aspect="equal",
                           cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
                           vmin=lay.vmin, vmax=lay.vmax, interpolation="nearest")
            _register_colorbar(ax, plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count"))

    elif is_categorical:
        unique_cats = _sorted_categories(set(color_values))
        n_cats = len(unique_cats)
        cat_to_int = {c: i for i, c in enumerate(unique_cats)}
        int_vals = np.array([cat_to_int[v] for v in color_values], dtype=np.intp)

        cat_counts = np.zeros((n_cats, bins, bins), dtype=np.int32)
        for cat_idx in range(n_cats):
            mask = int_vals == cat_idx
            np.add.at(cat_counts[cat_idx], (yi[mask], xi[mask]), 1)

        dominant = np.argmax(cat_counts, axis=0)
        resolved_na = lay.resolved_na_color
        cat_colors = _build_cat_colors(unique_cats, lay.palette, resolved_na)
        rgba = np.zeros((bins, bins, 4), dtype=np.float32)
        occupied = count_grid > 0
        for cat_idx, cat in enumerate(unique_cats):
            mask_2d = (dominant == cat_idx) & occupied
            rgba[mask_2d] = mcolors.to_rgba(cat_colors.get(str(cat), resolved_na))

        ax.imshow(rgba, extent=extent, origin="lower", aspect="equal",
                  interpolation="nearest")
        if legend_entries is not None:
            handles = _legend_handles(unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats])
            legend_entries.extend(zip(handles, (str(c) for c in unique_cats)))

    else:
        val_sum = np.zeros((bins, bins), dtype=np.float64)
        np.add.at(val_sum, (yi, xi), color_values.astype(np.float64))
        count_nz = np.where(count_grid > 0, count_grid, np.nan)
        grid = val_sum if lay.agg == "sum" else val_sum / count_nz
        grid = np.where(count_grid > 0, grid, np.nan)

        if lay.fill_color is not None:
            rgba = _density_to_rgba(grid, lay.fill_color, lay.vmin, lay.vmax)
            ax.imshow(rgba, extent=extent, origin="lower", aspect="equal",
                      interpolation="nearest")
        else:
            norm = (
                mcolors.TwoSlopeNorm(lay.vcenter, vmin=lay.vmin, vmax=lay.vmax)
                if lay.vcenter is not None else None
            )
            im = ax.imshow(
                grid, extent=extent, origin="lower", aspect="equal",
                cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
                vmin=lay.vmin if norm is None else None,
                vmax=lay.vmax if norm is None else None,
                norm=norm, interpolation="nearest",
            )
            _register_colorbar(ax, plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lay.color or "value"))


def _draw_raster_hex(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | None,
    is_categorical: bool,
    legend_entries: list | None = None,
) -> None:
    bins = lay.bins

    if color_values is None:
        if lay.fill_color is not None:
            ax.hexbin(x, y, gridsize=bins, cmap=_fill_alpha_cmap(lay.fill_color), linewidths=0.2,
                      extent=[x.min(), x.max(), y.min(), y.max()])
        else:
            hb = ax.hexbin(x, y, gridsize=bins,
                           cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
                           linewidths=0.2,
                           extent=[x.min(), x.max(), y.min(), y.max()])
            _register_colorbar(ax, plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="count"))

    elif is_categorical:
        unique_cats = _sorted_categories(set(color_values))
        cat_to_int = {c: i for i, c in enumerate(unique_cats)}
        int_vals = np.array([cat_to_int[v] for v in color_values], dtype=np.float64)

        def _mode(vals: np.ndarray) -> float:
            counts = np.bincount(vals.astype(np.intp), minlength=len(unique_cats))
            return float(counts.argmax())

        resolved_na = lay.resolved_na_color
        cat_colors = _build_cat_colors(unique_cats, lay.palette, resolved_na)
        cat_cmap = mcolors.ListedColormap(
            [cat_colors.get(str(c), resolved_na) for c in unique_cats]
        )
        ax.hexbin(
            x, y, C=int_vals, gridsize=bins,
            reduce_C_function=_mode, cmap=cat_cmap,
            vmin=-0.5, vmax=len(unique_cats) - 0.5,
            linewidths=0.2,
            extent=[x.min(), x.max(), y.min(), y.max()],
        )
        if legend_entries is not None:
            handles = _legend_handles(unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats])
            legend_entries.extend(zip(handles, (str(c) for c in unique_cats)))

    else:
        reduce_fn = np.sum if lay.agg == "sum" else np.mean
        if lay.fill_color is not None:
            ax.hexbin(x, y, C=color_values.astype(np.float64), gridsize=bins,
                      reduce_C_function=reduce_fn, cmap=_fill_alpha_cmap(lay.fill_color),
                      linewidths=0.2,
                      extent=[x.min(), x.max(), y.min(), y.max()])
        else:
            norm = (
                mcolors.TwoSlopeNorm(lay.vcenter, vmin=lay.vmin, vmax=lay.vmax)
                if lay.vcenter is not None else None
            )
            hb = ax.hexbin(
                x, y, C=color_values.astype(np.float64), gridsize=bins,
                reduce_C_function=reduce_fn,
                cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
                vmin=lay.vmin if norm is None else None,
                vmax=lay.vmax if norm is None else None,
                norm=norm, linewidths=0.2,
                extent=[x.min(), x.max(), y.min(), y.max()],
            )
            _register_colorbar(ax, plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label=lay.color or "value"))


def _fill_alpha_cmap(fill_color: str) -> mcolors.LinearSegmentedColormap:
    """Transparent-to-solid colormap for a single color; alpha encodes value."""
    r, g, b, _ = mcolors.to_rgba(fill_color)
    return mcolors.LinearSegmentedColormap.from_list(
        "_fill_alpha", [(r, g, b, 0.0), (r, g, b, 1.0)]
    )


def _density_to_rgba(
    grid: np.ndarray,
    fill_color: str,
    vmin: float | None,
    vmax: float | None,
) -> np.ndarray:
    """Map a 2D float grid to RGBA using a solid color with alpha = normalized value."""
    r, g, b, _ = mcolors.to_rgba(fill_color)
    finite = grid[np.isfinite(grid)]
    lo = vmin if vmin is not None else (float(finite.min()) if len(finite) > 0 else 0.0)
    hi = vmax if vmax is not None else (float(finite.max()) if len(finite) > 0 else 1.0)
    span = hi - lo if hi != lo else 1.0
    alpha_grid = np.where(np.isfinite(grid), np.clip((grid - lo) / span, 0.0, 1.0), 0.0)
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = alpha_grid
    return rgba


# ---------------------------------------------------------------------------
# color helpers
# ---------------------------------------------------------------------------


def _sorted_categories(cats) -> list:
    """Numeric-aware natural sort for category labels.

    Cluster labels are strings, so a plain ``sorted()`` orders them
    lexicographically ("0", "1", "10", "11", "2", ...) — wrong for the
    routine >9-cluster Leiden/BANKSY case, and it silently drives *color
    assignment* too (colors are handed out in sort order). Rules:

    - all labels parse as numbers → sort numerically ("9" before "10");
    - otherwise → natural key: digit runs compare as integers
      ("domain_9" before "domain_10"), remaining text case-insensitively;
    - no digits at all → equivalent to plain lexicographic order.

    Shared by every categorical draw site (points/boundaries/raster/
    annotation layers and ``pl.scatter``) so legend order and colors stay
    consistent across plot types.
    """
    cats = list(cats)
    try:
        return sorted(cats, key=lambda c: float(c))
    except (TypeError, ValueError):
        pass

    def _natural_key(c):
        parts = re.split(r"(\d+)", str(c))
        return [(0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts]

    return sorted(cats, key=_natural_key)


def _resolve_color_values(
    adata: ad.AnnData,
    col_name: str | None,
) -> tuple[np.ndarray | None, bool]:
    """Return ``(values_array, is_categorical)``.

    Searches ``var_names`` first (gene expression), then ``obs`` columns.
    Returns ``(None, False)`` when *col_name* is ``None``. Thin wrapper
    around the shared :func:`astrolabe._utils._resolve_field`.
    """
    if col_name is None:
        return None, False
    return _resolve_field(adata, col_name, axis="obs")


def _resolve_gdf_field(gdf: Any, col_name: str) -> tuple[np.ndarray, bool]:
    """Return ``(values_array, is_categorical)`` for a GeoDataFrame column.

    Mirrors :func:`astrolabe._utils._resolve_field`'s categorical-inference
    rule (``object``/``.cat`` dtype → categorical, else numeric), but scoped
    to plain ``gdf.columns`` lookup instead of AnnData's var_names/obs search
    — a GeoDataFrame has no gene-expression axis to check first.

    Raises
    ------
    KeyError
        If *col_name* is not a column of *gdf* (or is the geometry column).
    """
    if col_name not in gdf.columns or col_name == gdf.geometry.name:
        raise KeyError(
            f"'{col_name}' not found in GeoDataFrame columns. "
            f"Available columns: {[c for c in gdf.columns if c != gdf.geometry.name]}"
        )
    col = gdf[col_name]
    if hasattr(col, "cat") or col.dtype == object:
        return col.astype(str).to_numpy(), True
    return col.to_numpy(dtype=np.float64), False


def _plot_categorical(
    ax: matplotlib.axes.Axes,
    coords: np.ndarray,
    categories: np.ndarray,
    *,
    palette: dict | list | None,
    size: float,
    alpha: float,
    na_color: str | None = None,
    legend_entries: list | None = None,
    **scatter_kwargs: Any,
) -> None:
    # NOTE on cross-figure color persistence: an earlier version of this fix
    # tried deriving a stable draw order from the column's declared
    # `.cat.categories` so a label (e.g. Leiden cluster "7") would keep the
    # same color across two figures showing different subsets. That doesn't
    # work: AnnData recodes/prunes obs categoricals to only the observed
    # values on every `[]` subset (confirmed -- even a plain boolean-mask
    # view), so by the time a subset reaches this function the "full"
    # category list is already gone, and no caller here has an independent
    # copy of it to supply. Genuine persistence across subsets would need a
    # stateful global label->color registry, which is a larger design
    # decision (thread-safety under `n_jobs`, cache invalidation on theme
    # switch, etc.) deliberately left out of this fix -- see the astrolabe
    # bugfix-pass memory/plan for this note.
    resolved_na = na_color if na_color is not None else get_theme().muted
    unique_cats = _sorted_categories(set(categories))
    cat_colors = _build_cat_colors(unique_cats, palette, resolved_na)
    n_cells = len(coords)

    # Draw all categories in one interleaved (shuffled) scatter call rather
    # than one ax.scatter() per category. Separate per-category calls stack
    # in `unique_cats` (alphabetical) draw order, so at every point of visual
    # overlap the alphabetically-last category always wins regardless of its
    # actual abundance -- on a dense whole-tissue scatter this can make a
    # numerically substantial category (e.g. ~28% of cells) look almost
    # entirely absent just because it sorts earlier than a category drawn
    # over it. A fixed seed keeps the plot reproducible across re-renders.
    color_lookup = {str(cat): cat_colors.get(str(cat), resolved_na) for cat in unique_cats}
    point_colors = [color_lookup[str(cat)] for cat in categories]
    order = np.random.default_rng(0).permutation(n_cells)
    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=[point_colors[i] for i in order],
        s=size,
        alpha=alpha,
        rasterized=n_cells >= _RASTERIZE_MIN_MARKS,
        **scatter_kwargs,
    )

    if legend_entries is not None:
        handles = _legend_handles(unique_cats, [cat_colors.get(str(c), resolved_na) for c in unique_cats])
        legend_entries.extend(zip(handles, (str(c) for c in unique_cats)))


def _build_cat_colors(
    unique_cats: list,
    palette: dict | list | None,
    na_color: str,
) -> dict[str, Any]:
    if palette is None:
        t = get_theme()
        n_cats = len(unique_cats)
        colors = (
            t.cat_core if n_cats <= 8 else _extended_cat_colors(n_cats, t.name)
        )
        return {str(cat): colors[i] for i, cat in enumerate(unique_cats)}
    if isinstance(palette, dict):
        return {str(k): v for k, v in palette.items()}
    return {str(cat): palette[i % len(palette)] for i, cat in enumerate(unique_cats)}


def _resolve_channel_index(channel: int | str, adata: ad.AnnData) -> int:
    if isinstance(channel, int):
        return channel
    if hasattr(adata, "var_names") and channel in adata.var_names:
        return adata.var_names.get_loc(channel)
    raise KeyError(f"Channel '{channel}' not found in var_names.")


# ---------------------------------------------------------------------------
# Gradient / vector field public functions
# ---------------------------------------------------------------------------


def gradient_quiver(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    bin_size_um: float = 50.0,
    scale: float | None = None,
    cmap: str = "viridis",
    alpha: float = 0.85,
    color_percentiles: tuple[float, float] = (2.0, 98.0),
    gradient_key_added: str | None = None,
    exclude_unreliable: bool = True,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Quiver plot of expression gradient vectors binned to a coarse grid.

    Reads per-cell gradient vectors from ``obsm['gradient_{key}']`` (computed
    by :func:`~astrolabe.tl.expression_gradient`) and aggregates them into
    spatial bins of size *bin_size_um* µm.  Each bin's arrow is the mean
    (U, V) vector; the arrow **color is the mean of the per-cell magnitudes**,
    not the magnitude of the mean vector — where opposing directions cancel
    within a bin (e.g. between two nearby expression nests) the mean vector
    shrinks toward zero, and coloring by its magnitude would understate the
    gradient exactly where the structure is richest.  A short arrow with a
    hot color therefore reads as "steep but directionally mixed".

    Compose with other layers using ``+`` to overlay on a cell scatter or
    image::

        (ab.pl.points(mdata, color='leiden') +
         ab.pl.gradient_quiver(mdata, 'EPCAM')).show()

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key (e.g. ``'EPCAM'``); reads ``obsm['gradient_{key}']``.
    modality
        Modality key when *data* is a MuData.
    spatial_key
        ``obsm`` key for µm coordinates.
    bin_size_um
        Size of spatial bins in µm.  Smaller bins show finer spatial detail
        but may be noisy where cell density is low.
    scale
        Arrow scale passed to ``ax.quiver``.  Larger = shorter arrows.
        Auto-scaled to gradient magnitude range when ``None``.
    cmap
        Colormap for magnitude.
    alpha
        Arrow transparency.
    color_percentiles
        Robust ``(low, high)`` percentiles of the bin magnitudes used for the
        color scale (values outside are clipped). A plain min–max scale lets
        one extreme bin flatten the colormap for the whole tissue. Pass
        ``(0, 100)`` for the raw range.
    gradient_key_added
        ``key_added`` used in the :func:`~astrolabe.tl.expression_gradient`
        call this reads from, if it was given a custom one.
    exclude_unreliable
        Drop cells flagged ``obs['gradient_unreliable']`` (one-sided
        boundary neighborhoods) before binning.  Tissue-edge bins otherwise
        contain mostly unreliable cells and their inflated vectors dominate
        the arrow and color scales.  Set ``False`` to show every cell.
    title
        Panel title.
    frameon
        Draw axis frame and labels.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    adata = _resolve_adata(data, modality)
    grad_key = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
    if grad_key not in adata.obsm:
        raise KeyError(
            f"obsm['{grad_key}'] not found. "
            f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
        )
    _check_single_library(adata)

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)
    grad = adata.obsm[grad_key].astype(np.float64)
    if exclude_unreliable and "gradient_unreliable" in adata.obs:
        keep = ~adata.obs["gradient_unreliable"].to_numpy(dtype=bool)
        coords, grad = coords[keep], grad[keep]

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    n_x = max(2, int((x_max - x_min) / bin_size_um) + 1)
    n_y = max(2, int((y_max - y_min) / bin_size_um) + 1)

    # Bin cells into (n_y, n_x) grid and average U/V
    xi = np.clip(
        ((coords[:, 0] - x_min) / (x_max - x_min + 1e-8) * (n_x - 1)).astype(int),
        0, n_x - 1,
    )
    yi = np.clip(
        ((coords[:, 1] - y_min) / (y_max - y_min + 1e-8) * (n_y - 1)).astype(int),
        0, n_y - 1,
    )
    flat_idx = yi * n_x + xi
    n_bins = n_y * n_x
    U_sum = np.zeros(n_bins, dtype=np.float64)
    V_sum = np.zeros(n_bins, dtype=np.float64)
    M_sum = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(U_sum, flat_idx, grad[:, 0])
    np.add.at(V_sum, flat_idx, grad[:, 1])
    np.add.at(M_sum, flat_idx, np.linalg.norm(grad, axis=1))
    np.add.at(counts, flat_idx, 1)

    has_cells = counts > 0
    safe_counts = np.where(has_cells, counts, 1)
    U_bin = np.where(has_cells, U_sum / safe_counts, np.nan)
    V_bin = np.where(has_cells, V_sum / safe_counts, np.nan)
    # mean of magnitudes, not magnitude of the mean (see docstring)
    magnitude = np.where(has_cells, M_sum / safe_counts, np.nan)

    x_centers = x_min + (np.arange(n_x) + 0.5) * (x_max - x_min) / n_x
    y_centers = y_min + (np.arange(n_y) + 0.5) * (y_max - y_min) / n_y
    XC, YC = np.meshgrid(x_centers, y_centers)

    lay = Layer(
        kind="quiver",
        adata=adata,
        spatial_key=spatial_key,
        color=key,
        alpha=alpha,
        cmap=cmap,
        precomputed={
            "X": XC.ravel()[has_cells],
            "Y": YC.ravel()[has_cells],
            "U": U_bin[has_cells],
            "V": V_bin[has_cells],
            "magnitude": magnitude[has_cells],
            "scale": scale,
            "color_percentiles": color_percentiles,
            "key": key,
        },
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else f"{key} gradient",
        frameon=frameon,
        figsize=figsize,
    )


def gradient_streamlines(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    density: float = 1.5,
    linewidth: float | None = None,
    color_by: str = "magnitude",
    cmap: str = "viridis",
    alpha: float = 0.8,
    field_key: str | None = None,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Streamline plot of the interpolated expression gradient field.

    Reads the pre-interpolated grid from ``uns['gradient_field_{key}']``
    (computed by :func:`~astrolabe.tl.gradient_field`) and renders streamlines
    via ``matplotlib.axes.Axes.streamplot``.  Streamlines ascend the
    expression field and terminate on local maxima (peaks); they emanate
    from local minima.

    Streamlines are integral curves of the spatial ODE
    ``dx/dt = ∇f(x)`` — field lines of the static expression gradient.
    They show the *direction of steepest expression ascent*, not cell
    movement or RNA velocity.  NaN values outside the tissue mask are
    respected; streamlines do not enter masked voids.

    Compose with other spatial layers::

        (ab.pl.image(mdata) +
         ab.pl.gradient_streamlines(mdata, 'EPCAM')).show()

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key; reads ``uns['gradient_field_{key}']``.
    modality
        Modality key when *data* is a MuData.
    spatial_key
        ``obsm`` key for µm coordinates.
    density
        Streamline density passed to ``ax.streamplot``.  Higher = more lines.
    linewidth
        Line width.  When ``None``, scaled by normalized magnitude.
    color_by
        ``'magnitude'`` to color by gradient magnitude, or ``'uniform'``
        for a single color (black).
    cmap
        Colormap for ``color_by='magnitude'``.
    alpha
        Line transparency.
    field_key
        Full ``uns`` key to read, when :func:`~astrolabe.tl.gradient_field`
        was called with a custom ``key_added``. ``None`` (default) reads
        ``'gradient_field_{key}'``.
    title
        Panel title.
    frameon
        Draw axis frame and labels.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    adata = _resolve_adata(data, modality)
    field_key = field_key or f"gradient_field_{key}"
    if field_key not in adata.uns:
        raise KeyError(
            f"uns['{field_key}'] not found. "
            f"Run ab.tl.gradient_field(mdata, '{key}') first."
        )
    _check_single_library(adata)

    gf = adata.uns[field_key]
    U = np.array(gf["U"], dtype=np.float64)
    V = np.array(gf["V"], dtype=np.float64)
    x_grid = np.array(gf["X_grid"])
    y_grid = np.array(gf["Y_grid"])
    magnitude = np.sqrt(U ** 2 + V ** 2)

    lay = Layer(
        kind="streamplot",
        adata=adata,
        spatial_key=spatial_key,
        color=key,
        alpha=alpha,
        cmap=cmap,
        precomputed={
            "U": U,
            "V": V,
            "X_grid": x_grid,
            "Y_grid": y_grid,
            "magnitude": magnitude,
            "density": density,
            "linewidth": linewidth,
            "color_by": color_by,
            "key": key,
        },
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else f"{key} gradient streamlines",
        frameon=frameon,
        figsize=figsize,
    )


def divergence_plot(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    cmap: str = "RdBu_r",
    vcenter: float = 0.0,
    alpha: float = 0.8,
    field_key: str | None = None,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Heatmap of expression gradient divergence on a spatial panel.

    Reads ``uns['gradient_field_{key}']`` and renders the divergence field
    ``∂U/∂x + ∂V/∂y = ∇²f`` as a pseudocolor overlay.

    **Sign convention** — because the field is a *gradient*, its divergence
    is the Laplacian of expression: **negative divergence (blue with the
    default colormap) marks local expression maxima / peaks** (gradients
    converge into them; streamlines terminate there), and **positive
    divergence (red) marks local expression minima** (gradients point
    outward, ascending away).  This is the opposite of the fluid-flow
    "source/sink" intuition, where outward flow means a source of material —
    here outward *ascent* means a low point.

    No curl output exists: :func:`~astrolabe.tl.gradient_field` grids the
    scalar field and differentiates it, so the gridded field is curl-free by
    construction.

    Compose over a cell scatter or image::

        (ab.pl.points(mdata, 'leiden') +
         ab.pl.divergence_plot(mdata, 'EPCAM')).show()

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key; reads ``uns['gradient_field_{key}']``.
    modality
        Modality key when *data* is a MuData.
    spatial_key
        ``obsm`` key for µm coordinates.
    cmap
        Diverging colormap.  Defaults to ``'RdBu_r'`` (blue = expression
        peak, red = expression minimum; see the sign convention above).
    vcenter
        Center of the diverging colormap.
    alpha
        Heatmap transparency.
    field_key
        Full ``uns`` key to read, when :func:`~astrolabe.tl.gradient_field`
        was called with a custom ``key_added``. ``None`` (default) reads
        ``'gradient_field_{key}'``.
    title
        Panel title.
    frameon
        Draw axis frame and labels.
    figsize
        ``(width, height)`` in inches when rendered standalone.

    Returns
    -------
    SpatialPanel
    """
    adata = _resolve_adata(data, modality)
    field_key = field_key or f"gradient_field_{key}"
    if field_key not in adata.uns:
        raise KeyError(
            f"uns['{field_key}'] not found. "
            f"Run ab.tl.gradient_field(mdata, '{key}') first."
        )
    _check_single_library(adata)

    gf = adata.uns[field_key]
    field_data = np.array(gf["divergence"], dtype=np.float64)
    x_grid = np.array(gf["X_grid"])
    y_grid = np.array(gf["Y_grid"])

    field_name = "divergence"
    lay = Layer(
        kind="divergence",
        adata=adata,
        spatial_key=spatial_key,
        color=key,
        alpha=alpha,
        cmap=cmap,
        vcenter=vcenter,
        precomputed={
            "field": field_data,
            "X_grid": x_grid,
            "Y_grid": y_grid,
            "field_name": field_name,
            "key": key,
        },
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else f"{key} gradient {field_name}",
        frameon=frameon,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Gradient rendering functions (called from _render_panel dispatch)
# ---------------------------------------------------------------------------


def _draw_quiver_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    p = lay.precomputed
    X, Y = p["X"], p["Y"]
    U, V, magnitude = p["U"], p["V"], p["magnitude"]
    scale = p.get("scale")
    key = p.get("key", "")

    valid = np.isfinite(U) & np.isfinite(V)
    if not valid.any():
        return

    # Auto-scale: make arrows ~10% of the axis range by default
    if scale is None:
        x_range = float(X.max() - X.min()) if len(X) > 1 else 1.0
        mean_mag = float(magnitude[valid].mean()) if valid.any() else 1.0
        scale = mean_mag / (x_range * 0.05 + 1e-12)
        scale = max(scale, 1e-6)

    # Robust percentile color limits (clipped): a single extreme bin would
    # otherwise stretch a min-max Normalize and flatten the whole colormap.
    pct_lo, pct_hi = p.get("color_percentiles", (2.0, 98.0))
    finite_mag = magnitude[np.isfinite(magnitude)]
    if len(finite_mag):
        vmin = float(np.percentile(finite_mag, pct_lo))
        vmax = float(np.percentile(finite_mag, pct_hi))
        if vmax <= vmin:
            vmin, vmax = float(finite_mag.min()), float(finite_mag.max() + 1e-12)
    else:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    qv = ax.quiver(
        X[valid], Y[valid],
        U[valid], V[valid],
        magnitude[valid],
        cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
        norm=norm,
        scale=scale,
        scale_units="xy",
        alpha=lay.alpha,
        pivot="mid",
        width=0.004,
    )
    _register_colorbar(ax, plt.colorbar(qv, ax=ax, fraction=0.046, pad=0.04, label=f"{key} gradient magnitude"))
    ax.autoscale_view()


def _draw_streamplot_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    p = lay.precomputed
    U = p["U"]
    V = p["V"]
    x_grid = p["X_grid"]   # 1-D, shape (n_grid,)
    y_grid = p["Y_grid"]   # 1-D, shape (n_grid,)
    magnitude = p["magnitude"]
    density = float(p.get("density", 1.5))
    linewidth = p.get("linewidth")
    color_by = p.get("color_by", "magnitude")
    key = p.get("key", "")

    # Replace NaN with 0 for streamplot (it stops at low-speed regions)
    U_plot = np.where(np.isfinite(U), U, 0.0)
    V_plot = np.where(np.isfinite(V), V, 0.0)
    speed = np.sqrt(U_plot ** 2 + V_plot ** 2)

    if color_by == "magnitude":
        color = speed
        finite_mag = magnitude[np.isfinite(magnitude)]
        if len(finite_mag) == 0:
            finite_mag = np.array([0.0, 1.0])
        norm = mcolors.Normalize(
            vmin=float(np.nanmin(finite_mag)),
            vmax=float(np.nanmax(finite_mag)),
        )
        # Slim lines + small arrowheads: arrowheads are fixed display-size
        # (independent of the axes), so on a wide tissue strip rendered short
        # a fat ramp turns the panel into opaque arrowhead soup.
        lw = linewidth if linewidth is not None else (
            0.5 + 1.0 * np.clip(speed / (speed.max() + 1e-8), 0, 1)
        )
        sp_result = ax.streamplot(
            x_grid, y_grid, U_plot, V_plot,
            color=color,
            cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
            norm=norm,
            linewidth=lw,
            density=density,
            arrowsize=0.7,
        )
        sp_result.lines.set_alpha(lay.alpha)
        try:
            sp_result.arrows.set_alpha(lay.alpha)
        except AttributeError:
            pass
        _register_colorbar(ax, plt.colorbar(
            sp_result.lines, ax=ax, fraction=0.046, pad=0.04,
            label=f"{key} gradient magnitude",
        ))
    else:
        lw = linewidth if linewidth is not None else 1.2
        ax.streamplot(
            x_grid, y_grid, U_plot, V_plot,
            color=get_theme().primary_ink,
            linewidth=lw,
            density=density,
            arrowsize=1.2,
        )

    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(y_grid[0], y_grid[-1])


def _draw_divergence_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    p = lay.precomputed
    field = p["field"]
    x_grid = p["X_grid"]
    y_grid = p["Y_grid"]
    field_name = p.get("field_name", "divergence")
    key = p.get("key", "")

    valid_vals = field[np.isfinite(field)]
    if len(valid_vals) == 0:
        return

    abs_max = float(np.nanmax(np.abs(valid_vals)))
    vcenter = lay.vcenter if lay.vcenter is not None else 0.0
    norm = mcolors.TwoSlopeNorm(
        vcenter=vcenter,
        vmin=float(np.nanmin(valid_vals)),
        vmax=float(np.nanmax(valid_vals)),
    )

    XX, YY = np.meshgrid(x_grid, y_grid)
    pm = ax.pcolormesh(
        XX, YY, field,
        cmap=lay.cmap or _DEFAULT_DIVERGING_CMAP,
        norm=norm,
        alpha=lay.alpha,
        shading="auto",
    )
    label = f"{key} gradient {field_name}" if key else field_name
    _register_colorbar(ax, plt.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, label=label))
    ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_ylim(y_grid[0], y_grid[-1])
