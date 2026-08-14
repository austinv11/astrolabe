"""
astrolabe.pp — preprocessing functions for spatial transcriptomics.

Functions
---------
calculate_qc_metrics
    Add per-cell and per-gene QC columns to obs/var.
calculate_xenium_qc_metrics
    Xenium-specific probe-type breakdown and control metrics.
flag_controls
    Flag features as negative controls and compute per-cell control percentage.
filter_cells
    Filter cells by QC thresholds on obs columns.
filter_genes
    Filter genes/features by expression or type criteria.
filter_isolated
    Remove spatially isolated cells using a pre-computed kNN distance graph.
calculate_shape_metrics
    Compute morphological shape metrics (area, perimeter, spans, aspect ratio, equivalent radius,
    circularity, solidity, elongation, RNA density) from boundary polygons.
calculate_transcript_assignment_qc
    QC metrics for transcript-to-cell assignment quality.
normalize_total
    Normalize each cell to a target total count (thin wrapper around sc.pp.normalize_total).
log1p
    Log1p-transform X (thin wrapper around sc.pp.log1p).
normalize_by_area
    Normalize each cell's counts by its cell area.
highly_variable_genes
    Select highly variable genes for dimensionality reduction.
extract_if_intensities
    Extract per-cell mean IF or H&E stain intensities from cell boundary polygons.
load_bin_counts
    Materialize (all or part of) a lazily-referenced Visium HD bin x gene matrix.
crop
    Crop the spatial extent of an AnnData to a bounding box.
auto_crop
    Crop to a robust, percentile-trimmed bounding box around the cells themselves -- no manual region needed.
rotate
    Rotate spatial coordinates and image registration affines.
flip
    Flip spatial coordinates and image registration affines.
apply_affine
    Apply one affine transform across every modality in a MuData at once.
add_image_filter
    Append a pixel filter to an ImageSpec's filter_chain for lazy render-time application.
estimate_stain_matrix
    Re-estimate the H&E stain deconvolution matrix for an already-loaded image.
banksy_matrix
    Build the BANKSY spatially-augmented feature matrix (λ-weighted self + neighbor mean + optional AGF).
harmony
    Harmony batch correction on a PCA or BANKSY embedding (requires harmonypy).
"""

from astrolabe.pp._qc import (
    calculate_qc_metrics,
    calculate_xenium_qc_metrics,
    filter_cells,
    filter_genes,
    filter_isolated,
    flag_controls,
    calculate_shape_metrics,
    calculate_transcript_assignment_qc,
)
from astrolabe.pp._normalize import normalize_total, log1p, normalize_by_area, highly_variable_genes
from astrolabe.pp._if import extract_if_intensities
from astrolabe.pp._visium_hd import load_bin_counts
from astrolabe.pp._transform import crop, auto_crop, rotate, flip, apply_affine, add_image_filter, estimate_stain_matrix
from astrolabe.pp._banksy import banksy_matrix
from astrolabe.pp._harmony import harmony

__all__ = [
    "calculate_qc_metrics",
    "calculate_xenium_qc_metrics",
    "flag_controls",
    "filter_cells",
    "filter_genes",
    "filter_isolated",
    "calculate_shape_metrics",
    "normalize_total",
    "log1p",
    "normalize_by_area",
    "highly_variable_genes",
    "extract_if_intensities",
    "load_bin_counts",
    "crop",
    "auto_crop",
    "rotate",
    "flip",
    "apply_affine",
    "add_image_filter",
    "estimate_stain_matrix",
    "calculate_transcript_assignment_qc",
    "banksy_matrix",
    "harmony",
]