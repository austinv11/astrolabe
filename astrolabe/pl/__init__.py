"""
astrolabe.pl — composable spatial plotting.

All spatial plotting functions return a lazy :class:`SpatialPanel` that
renders only when shown, saved, or displayed in a Jupyter notebook.
Compose panels before rendering using:

- ``+`` — stack layers onto the same axes
- ``|`` — place panels side by side
- ``/`` — stack panels in a column

Plain ``matplotlib.figure.Figure`` and ``matplotlib.axes.Axes`` objects
work directly on either side of ``|`` and ``/`` without wrapping.

Functions
---------
points
    Scatter plot of cells or transcripts in physical (µm) space.
image
    Display a single IF / morphology image channel.
boundaries
    Draw cell or nucleus boundary polygons.
raster
    Bin cells into a 2-D grid (square pixels or hexagons) and render as
    an image; colour by density, expression, or category.
violin
    Per-cluster violin plots (wraps scanpy).

Classes
-------
SpatialPanel
    A single plot panel with one or more layers.
PlotGrid
    A 2-D grid of panels (created by ``|`` / ``/``).
FigurePanel
    Wraps a ``matplotlib.figure.Figure`` for layout operators.

Helpers
-------
layer
    Stack panels functionally (equivalent to ``p1 + p2``).
hstack
    Side-by-side layout (equivalent to ``p1 | p2``).
vstack
    Vertical layout (equivalent to ``p1 / p2``).
as_panel
    Explicitly wrap a Figure as a FigurePanel.
"""

from astrolabe.pl._spatial import (
    FigurePanel,
    PlotGrid,
    SpatialPanel,
    annotation,
    arrow,
    as_panel,
    boundaries,
    circle,
    hstack,
    image,
    layer,
    points,
    raster,
    rectangle,
    vstack,
)
from astrolabe.pl._qc_plots import (
    cell_area_histogram,
    nucleus_area_histogram,
    gene_qv_scatter,
    genes_per_cell_histogram,
    transcripts_per_cell_histogram,
    violin,
)

__all__ = [
    # Spatial layer functions
    "points",
    "image",
    "boundaries",
    "raster",
    "annotation",
    # Shape helpers
    "rectangle",
    "circle",
    "arrow",
    # Composable classes
    "SpatialPanel",
    "PlotGrid",
    "FigurePanel",
    # Explicit composition helpers
    "layer",
    "hstack",
    "vstack",
    "as_panel",
    # QC
    "violin",
    "gene_qv_scatter",
    "cell_area_histogram",
    "nucleus_area_histogram",
    "genes_per_cell_histogram",
    "transcripts_per_cell_histogram",
]
