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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patches as mpatches
import tifffile
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend as _Legend
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._shapes import arrow, circle, rectangle
from astrolabe._utils import ImageSpec, _resolve_adata, _transform_coords, with_adata

if TYPE_CHECKING:
    pass

__all__ = [
    "points",
    "image",
    "boundaries",
    "raster",
    "annotation",
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

_DEFAULT_CONTINUOUS_CMAP = "viridis"
_DEFAULT_CATEGORICAL_CMAP = "tab20"

_BOUNDARY_DEFAULT_COLORS: dict[str, str] = {
    "cell_boundaries": "lightsalmon",
    "nucleus_boundaries": "cyan",
}


# ---------------------------------------------------------------------------
# Layer — pure data, zero behaviour
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

    kind: Literal["points", "image", "boundaries", "raster", "annotation"]
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
    na_color: str = "#d3d3d3"
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
    bin_shape: Literal["square", "hex"] = "square"
    bins: int = 100
    agg: Literal["mean", "sum"] = "mean"
    fill_color: str | None = None

    # --- annotation-specific ---
    shapes: list | None = None          # list of (n_verts, 2) ndarray in µm
    annotation_color: str = "white"     # stroke colour
    annotation_fill: str | None = None  # fill colour; None = transparent

    # --- image-specific: scale bar ---
    scale_bar: bool | dict = False  # False | True | dict of kwargs

    def __post_init__(self) -> None:
        if self.kind != "annotation" and self.adata is None:
            raise ValueError(f"adata is required for Layer kind={self.kind!r}")


# ---------------------------------------------------------------------------
# _Renderable — shared base
# ---------------------------------------------------------------------------


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

        In a Jupyter notebook, displaying the object directly (leaving it as
        the last expression in a cell) is preferred and uses
        :meth:`_repr_mimebundle_`.  Call ``.show()`` when working in a
        script or when you need to display inside a loop.
        """
        fig = self.to_figure(figsize=figsize)
        plt.show()
        plt.close(fig)

    def save(self, path: str, *, dpi: int = 150, **savefig_kwargs: Any) -> None:
        """Save the figure to *path* without displaying it.

        Parameters
        ----------
        path
            Output file path.  Format is inferred from the extension
            (e.g. ``'.pdf'``, ``'.png'``, ``'.svg'``).
        dpi
            Resolution in dots per inch.  Ignored for vector formats.
        **savefig_kwargs
            Additional keyword arguments forwarded to ``figure.savefig()``.
        """
        fig = self.to_figure()
        fig.savefig(path, dpi=dpi, bbox_inches="tight", **savefig_kwargs)
        plt.close(fig)

    def _repr_mimebundle_(
        self,
        include: Any = None,
        exclude: Any = None,
    ) -> dict[str, str]:
        """Jupyter display hook — renders to SVG bytes."""
        fig = self.to_figure()
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return {"image/svg+xml": buf.getvalue().decode("utf-8")}

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
                f"objects; got {type(other).__name__!r}.  "
                f"Use | for side-by-side layout or / for stacked layout."
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
        fig, ax = plt.subplots(1, 1, figsize=figsize or self.figsize)
        _render_panel(ax, self)
        plt.tight_layout()
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

    def to_figure(
        self,
        figsize: tuple[float, float] | None = None,
    ) -> matplotlib.figure.Figure:
        return self.figure

    def __repr__(self) -> str:
        size = tuple(self.figure.get_size_inches())
        return f"FigurePanel(size={size})"


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
    ) -> matplotlib.figure.Figure:
        nrows, ncols = self.shape
        ref = _first_non_none(self.rows)
        if ref is not None and hasattr(ref, "figsize") and ref.figsize is not None:
            per_w, per_h = ref.figsize
        elif ref is not None and isinstance(ref, FigurePanel):
            per_w, per_h = tuple(ref.figure.get_size_inches())
        else:
            per_w, per_h = 5.0, 5.0

        effective = figsize or (ncols * per_w, nrows * per_h)
        fig, axes_grid = plt.subplots(
            nrows, ncols, figsize=effective, layout="constrained", squeeze=False
        )

        for i, row in enumerate(self.rows):
            for j in range(ncols):
                panel = row[j] if j < len(row) else None
                ax = axes_grid[i][j]
                if panel is None:
                    ax.set_visible(False)
                elif isinstance(panel, SpatialPanel):
                    _render_panel(ax, panel)
                else:
                    _render_figure_panel(ax, panel.to_figure())

        return fig

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
        f"Cannot use {type(obj).__name__!r} in a plot layout.  "
        "Expected a SpatialPanel, PlotGrid, FigurePanel, "
        "matplotlib.figure.Figure, or matplotlib.axes.Axes."
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
        raise ValueError("layer() requires at least one panel.")
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
        raise ValueError("hstack() requires at least one item.")
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
        raise ValueError("vstack() requires at least one item.")
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
            f"got {type(fig).__name__!r}."
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
    na_color: str = "#d3d3d3",
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
        Color for points with ``NaN`` in the color column.
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
            f"obsm['{spatial_key}'] not found.  "
            "Ensure read_xenium() was called or obsm['spatial'] exists."
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
    dark_bg: bool = True,
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
                f"Cannot infer library_id from {ids}.  "
                "Pass library_id explicitly."
            )
        library_id = ids[0]

    lib_uns = spatial_uns.get(library_id, {})
    images = lib_uns.get("images", {})
    if not images:
        raise KeyError(
            f"No images loaded for library '{library_id}'.  "
            "Load one first:\n"
            "  ab.io.read_ome_tiff('path/to/image.ome.tif', mdata=data)\n"
            "  # or: ab.io.read_xenium(..., morphology_image=True)"
        )
    if image_key is None:
        image_key = "full" if "full" in images else next(iter(images))
    elif image_key not in images:
        raise KeyError(
            f"Image key '{image_key}' not found.  "
            f"Available keys: {list(images.keys())}"
        )

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
        title = f"{library_id}/{image_key}" if channel is None else f"{library_id}/{image_key}/ch{channel}"
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
    na_color: str = "#d3d3d3",
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
        color for polygons with ``NaN`` in the color column.
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
            f"obsm['{boundaries_key}'] not found.  "
            "Re-run ab.io.read_xenium() with load_boundaries=True, "
            "or pass boundaries_key='nucleus_boundaries' for nucleus outlines."
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
    bin_shape: Literal["square", "hex"] = "square",
    bins: int = 100,
    agg: Literal["mean", "sum"] = "mean",
    fill_color: str | None = None,
    cmap: str | None = None,
    palette: dict | list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    na_color: str = "#d3d3d3",
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
        category, coloured by *palette* the same way as :func:`points`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    spatial_key
        Key in ``obsm`` for the ``(n, 2)`` coordinate array.
    x_col, y_col
        Coordinate column names when *data* is a Table/DataFrame.
    bin_shape
        ``'square'`` (default) renders as a pixel image via
        ``imshow``; ``'hex'`` uses matplotlib hexagonal bins.
    bins
        Number of bins along each axis for square bins, or the
        ``gridsize`` argument for hexagonal bins.
    agg
        Aggregation function for continuous *color* values:
        ``'mean'`` (default) or ``'sum'``.
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
        color for empty bins in categorical mode.
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
            f"obsm['{spatial_key}'] not found."
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
    )
    return SpatialPanel(
        layers=[lay],
        title=title if title is not None else (color or "density"),
        frameon=frameon,
        figsize=figsize,
    )


# ---------------------------------------------------------------------------
# Table → AnnData normalisation
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
            f"got {type(data).__name__!r}."
        )

    for col in (x_col, y_col):
        if col not in data.columns:
            raise KeyError(
                f"Column '{col}' not found in table.  "
                f"Available columns: {list(data.columns)}.  "
                f"Set x_col/y_col explicitly if your table uses different names."
            )

    coords = data[[x_col, y_col]].to_numpy(dtype=np.float32)
    obs = data.drop(columns=[x_col, y_col]).reset_index(drop=True)
    adata = ad.AnnData(obs=obs)
    adata.obsm["spatial"] = coords
    return adata


def annotation(
    shapes: list[np.ndarray] | np.ndarray,
    *,
    color: str = "yellow",
    fill_color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    fill_alpha: float = 0.2,
    title: str | None = None,
    frameon: bool = True,
    figsize: tuple[float, float] = (5, 5),
) -> SpatialPanel:
    """Overlay polygon annotation shapes on a spatial panel.

    Intended to be combined with ``+`` onto an existing image or points
    panel rather than used standalone::

        roi = ab.pl.rectangle(center=(500, 300), width=200, height=100)
        (ab.pl.image(mdata) + ab.pl.annotation([roi], color="yellow")).show()

    Parameters
    ----------
    shapes
        A single ``(n_verts, 2)`` array or a list of such arrays in µm.
        Use :func:`rectangle`, :func:`circle`, or :func:`arrow` to generate
        common shapes.
    color
        Edge colour of the annotation polygons.
    fill_color
        Fill colour.  ``None`` = no fill (outline only).
    linewidth
        Edge line width in points.
    alpha
        Overall opacity (applies to both edge and fill).
    fill_alpha
        Fill opacity, used when *fill_color* is set.
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
    if isinstance(shapes, np.ndarray):
        shapes = [shapes]
    lay = Layer(
        kind="annotation",
        adata=None,
        shapes=list(shapes),
        annotation_color=color,
        annotation_fill=fill_color,
        linewidth=linewidth,
        alpha=alpha,
        fill_alpha=fill_alpha,
    )
    return SpatialPanel(layers=[lay], title=title, frameon=frameon, figsize=figsize)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_LIBRARY_ID_KEY = "library_id"


def _check_single_library(adata: ad.AnnData) -> None:
    """Raise a clear error when cells from multiple libraries are present.

    Spatial coordinates are independent per Xenium run — each run starts its
    own µm coordinate frame from (0, 0).  Plotting cells from different runs
    on the same axes overlays unrelated tissue sections without any warning,
    which is almost never the desired behaviour.

    This check fires for ``'points'``, ``'boundaries'``, and ``'raster'``
    layers.  It is skipped for ``'image'`` (which already requires an explicit
    ``library_id=``) and ``'annotation'`` (no cell data).
    """
    if _LIBRARY_ID_KEY not in adata.obs.columns:
        return
    libs = adata.obs[_LIBRARY_ID_KEY].unique()
    if len(libs) <= 1:
        return
    lib_list = sorted(str(l) for l in libs)
    first = lib_list[0]
    panels_example = " | \\\n    ".join(
        f'ab.pl.points(adata[adata.obs["{_LIBRARY_ID_KEY}"] == "{lid}"], color)'
        for lid in lib_list
    )
    raise ValueError(
        f"This AnnData contains cells from {len(libs)} libraries: {lib_list}.\n"
        f"Each Xenium run has its own independent µm coordinate frame — plotting "
        f"them together overlays unrelated tissue sections on the same axes.\n\n"
        f"Filter to a single sample first:\n"
        f'    adata_sub = adata[adata.obs["{_LIBRARY_ID_KEY}"] == "{first}"]\n\n'
        f"or display each sample as a separate panel:\n"
        f"    {panels_example}"
    )


def _render_panel(ax: matplotlib.axes.Axes, panel: SpatialPanel) -> None:
    """Draw all layers of *panel* onto *ax*, then apply title and frame."""
    # Shared legend entry list — image channel entries and default-color boundary
    # outline entries are both collected here so they appear in one combined legend
    # rather than two overlapping add_artist() calls.
    shared_legend: list[tuple[Any, str]] = []

    for lay in panel.layers:
        if lay.kind == "points":
            _draw_points_layer(ax, lay)
        elif lay.kind == "image":
            _draw_image_layer(ax, lay, shared_legend)
        elif lay.kind == "boundaries":
            _draw_boundary_layer(ax, lay, shared_legend)
        elif lay.kind == "raster":
            _draw_raster_layer(ax, lay)
        elif lay.kind == "annotation":
            _draw_annotation_layer(ax, lay)
        else:
            log.warning("Unknown layer kind %r; skipping.", lay.kind)

    if shared_legend:
        handles = [h for h, _ in shared_legend]
        labels = [lb for _, lb in shared_legend]
        combined = _Legend(
            ax, handles, labels,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
        )
        combined.set_clip_on(False)
        ax.add_artist(combined)

    # Auto-zoom to crop region stored by pp.crop()
    bounds = next(
        (
            lay.adata.uns["spatial_bounds"]
            for lay in panel.layers
            if lay.adata is not None and "spatial_bounds" in lay.adata.uns
        ),
        None,
    )
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

    if any(lay.kind == "image" and lay.dark_bg for lay in panel.layers):
        _apply_dark_theme(ax)


def _render_figure_panel(
    ax: matplotlib.axes.Axes,
    src: matplotlib.figure.Figure,
) -> None:
    """Embed *src* into *ax* by rasterising to PNG and displaying via imshow."""
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    src.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = mpimg.imread(buf)
    ax.imshow(img, aspect="auto")
    ax.axis("off")


def _draw_points_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    adata = lay.adata
    _check_single_library(adata)
    coords = adata.obsm[lay.spatial_key]
    n_cells = len(coords)
    size = lay.size if lay.size is not None else max(0.5, 120_000 / n_cells)

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
            **lay.extra_kwargs,
        )
    else:
        norm = (
            mcolors.TwoSlopeNorm(lay.vcenter, vmin=lay.vmin, vmax=lay.vmax)
            if lay.vcenter is not None
            else None
        )
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color_values,
            s=size,
            alpha=lay.alpha,
            cmap=lay.cmap or _DEFAULT_CONTINUOUS_CMAP,
            vmin=lay.vmin if norm is None else None,
            vmax=lay.vmax if norm is None else None,
            norm=norm,
            rasterized=n_cells > 100_000,
            **lay.extra_kwargs,
        )
        if color_values is not None:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


# Additive-blending colours for multi-channel fluorescence composites.
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

    Returns ``(raw, ch_axis)`` where *raw* has axes ``(C, Y, X)`` and *ch_axis*
    is the channel axis index (always 0 after collapsing).  Z is reduced via
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

    return raw, ch_ax


def _apply_filter_chain(
    raw: np.ndarray,
    filters: list,
    ch_ax: int,
) -> np.ndarray:
    """Apply each filter in *filters* independently to every channel of *raw*.

    *raw* is expected to be ``(C, Y, X)`` after ``_load_tiff_level`` (ch_ax==0).
    Each filter receives a 2-D float32 channel and must return float32.
    """
    raw = raw.astype(np.float32)
    if raw.ndim == 2:
        ch = raw
        for f in filters:
            ch = f(ch)
        return ch
    n_ch = raw.shape[ch_ax]  # ch_ax is always 0 here
    channels = []
    for i in range(n_ch):
        ch = raw[i].astype(np.float32)
        for f in filters:
            ch = f(ch)
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
    h_full, w_full = spec.full_shape[-2], spec.full_shape[-1]
    h_level = int(round(h_full / yscale))
    w_level = int(round(w_full / xscale))

    col_min = max(0, int((bounds["xmin"] - xmin_img) / img_w_um * w_level))
    col_max = min(w_level, int((bounds["xmax"] - xmin_img) / img_w_um * w_level) + 1)
    row_min = max(0, int((bounds["ymin"] - ymin_img) / img_h_um * h_level))
    row_max = min(h_level, int((bounds["ymax"] - ymin_img) / img_h_um * h_level) + 1)

    return level_idx, (row_min, row_max, col_min, col_max)


def _compute_extent(spec: ImageSpec, h_full: int, w_full: int) -> list[float]:
    """Map image corners to the reference µm frame.

    If ``spec.registration_affine`` is set and axis-aligned (no rotation),
    the affine's scale and translation are used to place the image correctly
    in physical space.  Otherwise falls back to ``pixel_size_um`` scaling
    from the origin.  Any transforms in ``spec.transform_chain`` are then
    applied on top.
    """
    if spec.registration_affine is not None:
        A = spec.registration_affine
        off_diag = abs(A[0, 1]) + abs(A[1, 0])
        if off_diag <= 1e-6:
            sx, sy = float(A[0, 0]), float(A[1, 1])
            tx, ty = float(A[0, 2]), float(A[1, 2])
            xmin, xmax = tx, tx + w_full * sx
            ymin, ymax = ty, ty + h_full * sy
        else:
            log.warning(
                "registration_affine contains rotation (off-diagonal elements %.3g); "
                "falling back to pixel_size_um extent.",
                off_diag,
            )
            xmin, xmax = 0.0, w_full * spec.pixel_size_um
            ymin, ymax = 0.0, h_full * spec.pixel_size_um
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
) -> np.ndarray:
    """Percentile-stretch a 2-D channel to float32 [0, 1]."""
    img_f = img_2d.astype(np.float32)
    lo = float(vmin) if vmin is not None else float(np.percentile(img_f, 1.0))
    hi = float(vmax) if vmax is not None else float(np.percentile(img_f, 99.8))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img_f - lo) / (hi - lo), 0.0, 1.0) ** gamma


def _apply_dark_theme(ax: matplotlib.axes.Axes) -> None:
    """Black axes and figure background with white text — matches IF presentation style."""
    ax.set_facecolor("black")
    fig = ax.get_figure()
    if fig is not None:
        fig.patch.set_facecolor("black")
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


def _draw_annotation_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    """Render polygon annotation overlays onto *ax*."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    shapes = lay.shapes or []
    if not shapes:
        return

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
    h_full, w_full = spec.full_shape[-2], spec.full_shape[-1]
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

    raw, ch_ax = _load_tiff_level(spec.path, spec.dimension_order, level_idx, crop=crop_pixels)

    # Apply per-channel image filters (filter_chain set by ab.pp.add_image_filter).
    if spec.filter_chain:
        raw = _apply_filter_chain(raw, spec.filter_chain, ch_ax)

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
            return _normalize_image_channel(arr2d, lay.vmin, lay.vmax, lay.gamma)
        dtype_max = (
            float(np.iinfo(arr2d.dtype).max)
            if np.issubdtype(arr2d.dtype, np.integer)
            else 1.0
        )
        return np.clip(arr2d.astype(np.float32) / dtype_max, 0.0, 1.0) ** lay.gamma

    if lay.channel is None:
        imgs = [_prep(ch) for ch in ch_slices]
        if n_ch == 1:
            img_display: np.ndarray = imgs[0]
            cmap: str | None = lay.image_cmap
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
            raise IndexError(f"channel={ch_idx} is out of range; image has {n_ch} channel(s).")
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

    # Channel legend for multi-channel composite (not shown for single-channel or
    # when a specific channel is selected, since the colormap title is sufficient).
    if lay.channel is None and n_ch > 1:
        if len(lay.adata.var_names) == n_ch:
            ch_names = list(lay.adata.var_names)
        elif spec.channel_names is not None and len(spec.channel_names) == n_ch:
            ch_names = list(spec.channel_names)
        else:
            ch_names = [f"Ch{i}" for i in range(n_ch)]
        if legend_entries is not None:
            for i in range(n_ch):
                handle = mpatches.Patch(
                    color=tuple(float(v) for v in _CHANNEL_COLORS_RGB[i % len(_CHANNEL_COLORS_RGB)]),
                    label=ch_names[i],
                )
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
    try:
        import awkward as ak
        if isinstance(raw, ak.Array):
            counts = np.asarray(ak.num(raw))            # (n_cells,) int
            flat = np.asarray(ak.flatten(raw))          # (total_verts, 2) float32
            split_pts = np.cumsum(counts[:-1])
            polys = np.split(flat, split_pts)           # list of (n_verts_i, 2)
        else:
            raise TypeError
    except (ImportError, TypeError):
        polys = [np.asarray(raw[i]) for i in range(len(raw))]

    valid_mask = np.array([p.ndim == 2 and len(p) >= 3 for p in polys])
    valid_polys = [polys[i] for i in np.where(valid_mask)[0]]
    valid_indices = np.where(valid_mask)[0]

    if not valid_polys:
        return

    rasterized = len(valid_polys) > 10_000

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
        unique_cats = sorted(set(color_values[valid_indices]))
        cat_colors = _build_cat_colors(unique_cats, lay.palette, lay.na_color)
        base_c = [mcolors.to_rgb(cat_colors.get(str(color_values[i]), lay.na_color)) for i in valid_indices]
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
        ax.legend(
            handles=[
                mpatches.Patch(color=cat_colors.get(str(c), lay.na_color), label=str(c))
                for c in unique_cats
            ],
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
        )
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
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    else:
        edge_color = literal_color if literal_color is not None else "#555555"
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
            handle = mpatches.Patch(
                facecolor="none",
                edgecolor=edge_color,
                linewidth=1.5,
                label=label,
            )
            legend_entries.append((handle, label))

    if lay.spatial_key in adata.obsm:
        coords = adata.obsm[lay.spatial_key]
        ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
        ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    else:
        ax.autoscale_view()


def _draw_raster_layer(ax: matplotlib.axes.Axes, lay: Layer) -> None:
    adata = lay.adata
    _check_single_library(adata)
    coords = adata.obsm[lay.spatial_key]
    x = coords[:, 0].astype(np.float64)
    y = coords[:, 1].astype(np.float64)
    color_values, is_categorical = _resolve_color_values(adata, lay.color)
    if lay.bin_shape == "hex":
        _draw_raster_hex(ax, lay, x, y, color_values, is_categorical)
    else:
        _draw_raster_square(ax, lay, x, y, color_values, is_categorical)


def _draw_raster_square(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | None,
    is_categorical: bool,
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
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")

    elif is_categorical:
        unique_cats = sorted(set(color_values))
        n_cats = len(unique_cats)
        cat_to_int = {c: i for i, c in enumerate(unique_cats)}
        int_vals = np.array([cat_to_int[v] for v in color_values], dtype=np.intp)

        cat_counts = np.zeros((n_cats, bins, bins), dtype=np.int32)
        for cat_idx in range(n_cats):
            mask = int_vals == cat_idx
            np.add.at(cat_counts[cat_idx], (yi[mask], xi[mask]), 1)

        dominant = np.argmax(cat_counts, axis=0)
        cat_colors = _build_cat_colors(unique_cats, lay.palette, lay.na_color)
        rgba = np.zeros((bins, bins, 4), dtype=np.float32)
        occupied = count_grid > 0
        for cat_idx, cat in enumerate(unique_cats):
            mask_2d = (dominant == cat_idx) & occupied
            rgba[mask_2d] = mcolors.to_rgba(cat_colors.get(str(cat), lay.na_color))

        ax.imshow(rgba, extent=extent, origin="lower", aspect="equal",
                  interpolation="nearest")
        ax.legend(
            handles=[
                mpatches.Patch(color=cat_colors.get(str(c), lay.na_color), label=str(c))
                for c in unique_cats
            ],
            bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8,
        )

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
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_raster_hex(
    ax: matplotlib.axes.Axes,
    lay: Layer,
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray | None,
    is_categorical: bool,
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
            plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="count")

    elif is_categorical:
        unique_cats = sorted(set(color_values))
        cat_to_int = {c: i for i, c in enumerate(unique_cats)}
        int_vals = np.array([cat_to_int[v] for v in color_values], dtype=np.float64)

        def _mode(vals: np.ndarray) -> float:
            counts = np.bincount(vals.astype(np.intp), minlength=len(unique_cats))
            return float(counts.argmax())

        cat_colors = _build_cat_colors(unique_cats, lay.palette, lay.na_color)
        cat_cmap = mcolors.ListedColormap(
            [cat_colors.get(str(c), lay.na_color) for c in unique_cats]
        )
        ax.hexbin(
            x, y, C=int_vals, gridsize=bins,
            reduce_C_function=_mode, cmap=cat_cmap,
            vmin=-0.5, vmax=len(unique_cats) - 0.5,
            linewidths=0.2,
            extent=[x.min(), x.max(), y.min(), y.max()],
        )
        ax.legend(
            handles=[
                mpatches.Patch(color=cat_colors.get(str(c), lay.na_color), label=str(c))
                for c in unique_cats
            ],
            bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8,
        )

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
            plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)


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
    """Map a 2D float grid to RGBA using a solid color with alpha = normalised value."""
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


def _resolve_color_values(
    adata: ad.AnnData,
    col_name: str | None,
) -> tuple[np.ndarray | None, bool]:
    """Return ``(values_array, is_categorical)``.

    Searches ``var_names`` first (gene expression), then ``obs`` columns.
    Returns ``(None, False)`` when *col_name* is ``None``.
    """
    if col_name is None:
        return None, False

    if col_name in adata.var_names:
        idx = adata.var_names.get_loc(col_name)
        x_col = adata.X[:, idx]
        if sp.issparse(x_col):
            x_col = np.asarray(x_col.todense()).ravel()
        return x_col.astype(np.float64), False

    if col_name in adata.obs.columns:
        col = adata.obs[col_name]
        if hasattr(col, "cat") or col.dtype == object:
            return col.astype(str).to_numpy(), True
        return col.to_numpy(dtype=np.float64), False

    raise KeyError(
        f"'{col_name}' not found in var_names or obs columns.  "
        f"Available obs columns: {list(adata.obs.columns)}"
    )


def _plot_categorical(
    ax: matplotlib.axes.Axes,
    coords: np.ndarray,
    categories: np.ndarray,
    *,
    palette: dict | list | None,
    size: float,
    alpha: float,
    na_color: str = "#d3d3d3",
    **scatter_kwargs: Any,
) -> None:
    unique_cats = sorted(set(categories))
    cat_colors = _build_cat_colors(unique_cats, palette, na_color)
    n_cells = len(coords)

    for cat in unique_cats:
        mask = categories == cat
        color = cat_colors.get(str(cat), na_color)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color] * int(mask.sum()),
            s=size,
            alpha=alpha,
            label=str(cat),
            rasterized=n_cells > 100_000,
            **scatter_kwargs,
        )

    ax.legend(
        handles=[
            mpatches.Patch(color=cat_colors.get(str(c), na_color), label=str(c))
            for c in unique_cats
        ],
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )


def _build_cat_colors(
    unique_cats: list,
    palette: dict | list | None,
    na_color: str,
) -> dict[str, Any]:
    if palette is None:
        cmap = plt.get_cmap(_DEFAULT_CATEGORICAL_CMAP)
        n = max(len(unique_cats) - 1, 1)
        return {str(cat): cmap(i / n) for i, cat in enumerate(unique_cats)}
    if isinstance(palette, dict):
        return {str(k): v for k, v in palette.items()}
    return {str(cat): palette[i % len(palette)] for i, cat in enumerate(unique_cats)}


def _resolve_channel_index(channel: int | str, adata: ad.AnnData) -> int:
    if isinstance(channel, int):
        return channel
    if hasattr(adata, "var_names") and channel in adata.var_names:
        return adata.var_names.get_loc(channel)
    raise KeyError(f"Channel '{channel}' not found in var_names.")
