"""
astrolabe.io — data readers for spatial transcriptomics and IF imaging.

Functions
---------
read_xenium
    Read a 10x Genomics Xenium output directory into MuData.
read_xenium_transcripts
    Lazily stream and filter Xenium transcript detections.
read_ome_tiff
    Read an OME-TIFF multiplexed IF image (lazy, dask-backed).
"""

from astrolabe.io._xenium import read_xenium, read_xenium_transcripts
from astrolabe.io._ome_tiff import read_ome_tiff

__all__ = [
    "read_xenium",
    "read_xenium_transcripts",
    "read_ome_tiff",
]