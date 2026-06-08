"""
astrolabe.pp — preprocessing functions for spatial transcriptomics.

Functions
---------
filter_cells
    Filter cells by QC thresholds on obs columns.
filter_genes
    Filter genes/features by expression or type criteria.
calculate_qc_metrics
    Add per-cell and per-gene QC columns to obs/var.
normalize_total
    Normalize each cell to a target total count (thin wrapper around sc.pp.normalize_total).
log1p
    Log1p-transform X (thin wrapper around sc.pp.log1p).
extract_if_intensities
    Extract per-cell mean IF intensities from cell boundary polygons.
"""

from astrolabe.pp._qc import calculate_qc_metrics, calculate_xenium_qc_metrics, filter_cells, filter_genes
from astrolabe.pp._normalize import normalize_total, log1p
from astrolabe.pp._if import extract_if_intensities
from astrolabe.pp._transform import crop, rotate, flip

__all__ = [
    "calculate_qc_metrics",
    "calculate_xenium_qc_metrics",
    "filter_cells",
    "filter_genes",
    "normalize_total",
    "log1p",
    "extract_if_intensities",
    "crop",
    "rotate",
    "flip",
]