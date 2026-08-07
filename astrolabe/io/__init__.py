"""
astrolabe.io — data readers for spatial transcriptomics and IF imaging.

Functions
---------
read_xenium
    Read a 10x Genomics Xenium output directory into MuData.
read_xenium_transcripts
    Lazily stream and filter Xenium transcript detections.
read_merscope
    Read a Vizgen MERSCOPE output directory into MuData.
read_cosmx
    Read a NanoString CosMx SMI output directory into MuData.
read_visium_hd
    Read a 10x Genomics Visium HD Space Ranger output directory into MuData
    (lazy 2µm bins; native segmentation auto-loaded as the primary modality
    when present).
read_ome_tiff
    Read an OME-TIFF multiplexed IF image (lazy, dask-backed).
read_he
    Read an H&E RGB image with optional stain channel separation (hematoxylin / eosin / residual).
extract_morphology_intensities
    Extract per-cell morphology image intensities and store as mdata['morphology'].
concat_samples
    Concatenate multiple single-sample MuData objects into one multi-sample object.
save
    Save a processed MuData to a single .h5mu file (handles awkward arrays and ImageSpec).
load
    Load a MuData previously saved with save().
attach_modality
    Generalized helper to insert an AnnData as a new modality into an existing MuData
    without disturbing an already-established default modality.
read_annotations
    Read a GeoJSON annotation file (ROIs, pathologist regions) into a GeoDataFrame,
    optionally attaching it to a MuData/AnnData's uns. Platform-agnostic.
align_xenium_annotations
    Register a Xenium Explorer annotation GeoJSON export (raw morphology-image
    pixel coordinates) into the Xenium micron frame.
"""

from astrolabe.io._xenium import (
    read_xenium,
    read_xenium_transcripts,
    extract_morphology_intensities,
    align_xenium_annotations,
)
from astrolabe.io._merscope import read_merscope
from astrolabe.io._cosmx import read_cosmx
from astrolabe.io._visium_hd import read_visium_hd
from astrolabe.io._ome_tiff import read_ome_tiff
from astrolabe.io._he import read_he
from astrolabe.io._concat import concat_samples
from astrolabe.io._io import save, load, attach_modality
from astrolabe.io._annotations import read_annotations

__all__ = [
    "read_xenium",
    "read_xenium_transcripts",
    "read_merscope",
    "read_cosmx",
    "read_visium_hd",
    "read_ome_tiff",
    "read_he",
    "extract_morphology_intensities",
    "concat_samples",
    "save",
    "load",
    "attach_modality",
    "read_annotations",
    "align_xenium_annotations",
]