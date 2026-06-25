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
"""

from astrolabe.seg._voronoi import voronoi_resegment
from astrolabe.seg._segment import segment_cellpose, segment_stardist
from astrolabe.seg._assign import assign_transcripts, create_compartment_layers
from astrolabe.seg._eval import evaluate_segmentation

__all__ = [
    "voronoi_resegment",
    "segment_cellpose",
    "segment_stardist",
    "assign_transcripts",
    "create_compartment_layers",
    "evaluate_segmentation",
]
