"""
astrolabe.seg — cell re-segmentation utilities.

Functions
---------
voronoi_resegment
    Replace cell boundary polygons with Voronoi tessellation from cell
    centroids, clipped to an auto-computed tissue boundary.
segment_cellpose
    Segment cells in a fluorescence image channel using Cellpose.
    Requires the optional ``cellpose`` package.
segment_stardist
    Segment nuclei in a fluorescence image channel using StarDist.
    Requires the optional ``stardist`` package.
assign_transcripts
    Assign transcripts to cell boundaries by point-in-polygon spatial
    indexing and build a per-cell expression count matrix.
create_compartment_layers
    Partition transcripts into nucleus, cytoplasm, and ambient AnnData layers.
evaluate_segmentation
    Evaluate predicted boundary polygons against a ground-truth label image;
    returns precision, recall, F1, and Dice at a configurable IoU threshold.
assign_bins
    Assign Visium HD 2µm bins to a pre-existing cell segmentation and build a
    per-cell count matrix (native bin2cell reimplementation, with a
    PCA-transcriptional-similarity tie-break for contested bins).
segment_bin2cell
    Convenience recipe: segment_stardist/segment_cellpose + assign_bins in
    one call, with a toggle for whether H&E or a bin-expression raster
    drives the segmentation.
"""

from astrolabe.seg._voronoi import voronoi_resegment
from astrolabe.seg._segment import segment_cellpose, segment_stardist
from astrolabe.seg._assign import assign_transcripts, create_compartment_layers
from astrolabe.seg._eval import evaluate_segmentation
from astrolabe.seg._audit import record_segmentation
from astrolabe.seg._bin2cell import assign_bins, segment_bin2cell

__all__ = [
    "record_segmentation",
    "voronoi_resegment",
    "segment_cellpose",
    "segment_stardist",
    "assign_transcripts",
    "create_compartment_layers",
    "evaluate_segmentation",
    "assign_bins",
    "segment_bin2cell",
]
