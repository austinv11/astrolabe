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
    an image; color by density, expression, or category.
violin
    Per-cluster violin plots (wraps scanpy).
dotplot
    Dot plot of gene expression across cluster groups; returns FigurePanel for composition.
embedding_scatter
    Scatter plot of any 2-D embedding (UMAP, PCA, spatial_umap) coloured by obs or gene.
volcano_plot
    Volcano plot (log₂FC vs −log₁₀p) from rank_genes_groups results.
paga_plot
    PAGA trajectory graph (wraps scanpy); returns FigurePanel.
ripley_plot
    Ripley's L(d) − d curve plot from tl.ripley_l results.

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

from astrolabe._shapes import merge_polygons
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
from astrolabe.pl._filters import CLAHEFilter, FFTFilter, GaussianFilter
from astrolabe.pl._napari import view_napari
from astrolabe.pl._qc_plots import (
    cell_area_histogram,
    nucleus_area_histogram,
    gene_qv_scatter,
    gene_detection_scatter,
    mean_variance_plot,
    genes_per_cell_histogram,
    transcripts_per_cell_histogram,
    violin,
    dotplot,
    heatmap,
    embedding_scatter,
    volcano_plot,
    paga_plot,
    ripley_plot,
    intensity_histogram,
    intensity_histograms,
    scatter_gate,
    moran_plot,
    nhood_enrichment_plot,
    co_occurrence_plot,
    centrality_scores_plot,
    expression_by_distance_plot,
    ligrec_plot,
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
    "merge_polygons",
    # Composable classes
    "SpatialPanel",
    "PlotGrid",
    "FigurePanel",
    # Explicit composition helpers
    "layer",
    "hstack",
    "vstack",
    "as_panel",
    # QC / expression / embeddings
    "violin",
    "dotplot",
    "heatmap",
    "embedding_scatter",
    "volcano_plot",
    "paga_plot",
    "ripley_plot",
    "gene_qv_scatter",
    "gene_detection_scatter",
    "mean_variance_plot",
    "cell_area_histogram",
    "nucleus_area_histogram",
    "genes_per_cell_histogram",
    "transcripts_per_cell_histogram",
    "intensity_histogram",
    "intensity_histograms",
    "scatter_gate",
    "moran_plot",
    "nhood_enrichment_plot",
    "co_occurrence_plot",
    "centrality_scores_plot",
    "expression_by_distance_plot",
    "ligrec_plot",
    # Image filters
    "GaussianFilter",
    "CLAHEFilter",
    "FFTFilter",
    # Interactive viewer (requires napari: uv add napari)
    "view_napari",
]
