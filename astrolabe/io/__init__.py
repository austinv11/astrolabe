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
"""

from astrolabe.io._xenium import read_xenium, read_xenium_transcripts, extract_morphology_intensities
from astrolabe.io._merscope import read_merscope
from astrolabe.io._cosmx import read_cosmx
from astrolabe.io._ome_tiff import read_ome_tiff
from astrolabe.io._he import read_he
from astrolabe.io._concat import concat_samples
from astrolabe.io._io import save, load

__all__ = [
    "read_xenium",
    "read_xenium_transcripts",
    "read_merscope",
    "read_cosmx",
    "read_ome_tiff",
    "read_he",
    "extract_morphology_intensities",
    "concat_samples",
    "save",
    "load",
]