"""
Sample dataset downloads for astrolabe.

Each function downloads a small public dataset for the corresponding
platform, caches it under ``<cwd>/.astrolabe_cache/datasets/``, and
returns a ready-to-use :class:`mudata.MuData` or :class:`anndata.AnnData`.

Canonical usage
---------------
>>> import astrolabe as ab
>>> mdata = ab.datasets.xenium()        # 10x Genomics Human Lung Cancer FFPE (~254 MB)
>>> mdata = ab.datasets.xenium_tiny()   # 10x Genomics Human Kidney tiny (~20 MB, smoke tests)
>>> mdata = ab.datasets.merscope()      # Vizgen assembloid, Zenodo 13835782 (126 MB)
>>> mdata = ab.datasets.cosmx()         # NanoString Lung12 public S3 (155 MB)
>>> mdata = ab.datasets.visium_hd()      # 10x Genomics Visium HD Mouse Small Intestine
>>> mdata = ab.datasets.visium_hd_tiny() # 10x Genomics Visium HD tiny (297 MB, smoke tests)
>>> adata = ab.datasets.ome_tiff()      # OME Bio-Formats sample (2 x 264 KB)
>>> adata = ab.datasets.he()            # OpenSlide Huron H&E brightfield (36 MB)

Cache
-----
Files are written once to ``<cache_dir>/<platform>/`` and reused on
subsequent calls.  Override with the ``cache_dir`` parameter.
"""

from __future__ import annotations

import hashlib
import logging
import tarfile
import urllib.request
import zipfile
from pathlib import Path

__all__ = [
    "xenium", "xenium_tiny", "merscope", "cosmx",
    "visium_hd", "visium_hd_tiny", "ome_tiff", "he",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

# FFPE Human Lung Cancer dataset (CC BY 4.0) — individual files, no _outs.zip needed.
# 162k cells, 377-gene panel, ~5 GB total download (includes morphology OME-TIFF).
_XENIUM_BASE = (
    "https://cf.10xgenomics.com/samples/xenium/2.0.0/"
    "Xenium_V1_humanLung_Cancer_FFPE/"
    "Xenium_V1_humanLung_Cancer_FFPE"
)
# Maps remote file suffix → local canonical filename understood by read_xenium.
_XENIUM_FILES: dict[str, str] = {
    "_experiment.xenium": "experiment.xenium",
    "_cells.parquet": "cells.parquet",
    "_cell_feature_matrix.h5": "cell_feature_matrix.h5",
    "_cell_boundaries.parquet": "cell_boundaries.parquet",
    "_nucleus_boundaries.parquet": "nucleus_boundaries.parquet",
    "_transcripts.parquet": "transcripts.parquet",
    "_morphology.ome.tif": "morphology.ome.tif",
    "_he_image.ome.tif": "he_image.ome.tif",
    "_he_imagealignment.csv": "he_imagealignment.csv",
    "_annotation.geojson": "annotation.geojson",
}

# Tiny smoke-test dataset (~20 MB zip, ~350 cells).
_XENIUM_TINY_URL = (
    "https://cf.10xgenomics.com/samples/xenium/4.0.0/"
    "Xenium_V1_Protein_Human_Kidney_tiny/"
    "Xenium_V1_Protein_Human_Kidney_tiny_outs.zip"
)
# MD5 published by 10x for the archive above (XOA 4.0.0, CC BY 4.0).
_XENIUM_TINY_MD5 = "50c04dea5e751e1c7508ff24528242e8"

_MERSCOPE_CBG_URL = (
    "https://zenodo.org/api/records/13835782/files/"
    "hSO.hStrO_assembloid_region_0_cell_by_gene.csv/content"
)
_MERSCOPE_META_URL = (
    "https://zenodo.org/api/records/13835782/files/"
    "hSO.hStrO_assembloid_region_0_cell_metadata.csv/content"
)

_COSMX_EXPR_URL = (
    "https://nanostring-public-share.s3.amazonaws.com/SMI/Lung12/Lung12_exprMat_file.csv"
)
_COSMX_META_URL = (
    "https://nanostring-public-share.s3.amazonaws.com/SMI/Lung12/Lung12_metadata_file.csv"
)

# Visium HD Mouse Small Intestine (FFPE), Space Ranger 3.0.0 — no native
# segmentation (predates Space Ranger v4.0's on-board segmentation feature),
# good for demonstrating the seg.segment_bin2cell()/seg.assign_bins() path.
_VISIUM_HD_BASE = (
    "https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/"
    "Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine"
)

# Visium HD tiny smoke-test dataset (297 MB), Space Ranger v4.0.1 — includes
# native on-board segmentation (segmented_outputs/), good for demonstrating
# the native-segmentation path. 10x's own docs describe this dataset as
# "designed... for automated smoke testing".
_VISIUM_HD_TINY_URL = (
    "https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/"
    "Visium_HD_Tiny_3prime_Dataset/Visium_HD_Tiny_3prime_Dataset_outs.zip"
)

# "tubhiswt" (tubulin/histone wild-type) sample: a real two-channel
# acquisition of a 4-cell embryo (OME Bio-Formats sample data, CC BY 4.0),
# shipped by OME as two single-channel companion OME-TIFFs. Deliberately
# *not* the OME "bioformats-artificial/multi-channel.ome.tif" sample --
# that one is a synthetic test pattern with the filename/dimensions burned
# into the pixels as a watermark, which renders as illegible mirrored text
# under astrolabe's row-up image display convention (see
# _apply_registration_rotation's own regression tests for why that
# convention is deliberate) and looks broken in tutorials.
_OME_TIFF_BASE = (
    "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/tubhiswt-2D/"
)
_OME_TIFF_C0_URL = _OME_TIFF_BASE + "tubhiswt_C0.ome.tif"
_OME_TIFF_C1_URL = _OME_TIFF_BASE + "tubhiswt_C1.ome.tif"

_HE_URL = (
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Huron/Huron-1.tif"
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _cache_root(cache_dir) -> Path:
    return Path(cache_dir) if cache_dir else Path.cwd() / ".astrolabe_cache" / "datasets"


_UA = "Mozilla/5.0 (compatible; astrolabe-datasets/0.1; +https://github.com/austinv11/astrolabe)"

def _download(url: str, dest: Path, desc: str = "") -> None:
    """Download *url* to *dest*, skipping if *dest* already exists."""
    if dest.exists():
        log.debug("Cache hit: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", desc or url)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MB
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                fh.write(block)
                downloaded += len(block)
                if total:
                    log.info("  %.0f / %.0f MB", downloaded / 1e6, total / 1e6)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log.info("Saved to %s", dest)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir, filter="data")


def _find_visium_hd_outs(extract_dir: Path) -> Path:
    """Return the extracted directory that directly contains ``binned_outputs/``."""
    if (extract_dir / "binned_outputs").is_dir():
        return extract_dir
    for child in sorted(extract_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "binned_outputs").is_dir():
            return child
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / "binned_outputs").is_dir():
                return grandchild
    raise FileNotFoundError(
        f"Could not find binned_outputs/ in extracted Visium HD archive at {extract_dir}"
    )


def _find_xenium_outs(extract_dir: Path) -> Path:
    """Return the extracted Xenium output directory (contains experiment.xenium)."""
    # Handles both flat extraction and single-subdirectory extraction
    if (extract_dir / "experiment.xenium").exists():
        return extract_dir
    for child in sorted(extract_dir.iterdir()):
        if child.is_dir() and (child / "experiment.xenium").exists():
            return child
    raise FileNotFoundError(
        f"Could not find experiment.xenium in extracted Xenium archive at {extract_dir}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def xenium(cache_dir=None):
    """Download and return the 10x Genomics Xenium Human Lung Cancer FFPE dataset.

    Source: 10x Genomics CDN (CC BY 4.0), ~5 GB total (9 files: 6 expression
    files ~254 MB + morphology OME-TIFF ~4.8 GB + paired H&E image/alignment
    + pathologist annotation GeoJSON).
    FFPE Human Lung Adenocarcinoma, 162k cells, 377-gene panel.
    Returns a :class:`mudata.MuData` with a ``'xenium'`` modality (default
    modality) containing cell boundaries, spatial coordinates in microns,
    transcript detections, and the morphology image attached as an
    :class:`~astrolabe._utils.ImageSpec` under
    ``uns['spatial'][library_id]['images']['morphology']``; a separate
    ``'he'`` modality with the paired H&E image affine-registered into the
    same micron frame via 10x's alignment matrix; and pathologist ROI
    annotations in ``uns['annotations']``, registered into the same micron
    frame via :func:`~astrolabe.io.align_xenium_annotations`.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/xenium_ffpe/``.
    """
    d = _cache_root(cache_dir) / "xenium_ffpe"
    outs_dir = d / "outs"
    outs_dir.mkdir(parents=True, exist_ok=True)
    for suffix, local_name in _XENIUM_FILES.items():
        _download(
            _XENIUM_BASE + suffix,
            outs_dir / local_name,
            f"Xenium Human Lung Cancer FFPE ({local_name})",
        )
    from astrolabe import io
    mdata = io.read_xenium(
        str(outs_dir), morphology_image=True, he_image=True, he_alignment=True
    )
    annotations = io.read_annotations(outs_dir / "annotation.geojson")
    mdata.uns["annotations"] = io.align_xenium_annotations(annotations, mdata)
    return mdata


def _xenium_tiny_outs(cache_dir=None) -> Path:
    """Download (if needed), verify and extract the Xenium tiny archive.

    Returns the extracted ``outs/`` directory.  The archive's MD5 is checked
    against 10x's published value; a mismatch deletes the file and raises so a
    truncated or tampered download is never silently read.  Shared by
    :func:`xenium_tiny` and the test suite's ``xenium_tiny_dir`` fixture.
    """
    d = _cache_root(cache_dir) / "xenium_tiny"
    zip_path = d / "xenium_tiny.zip"
    _download(_XENIUM_TINY_URL, zip_path, "Xenium Human Kidney tiny (10x Genomics, 20 MB)")
    digest = hashlib.md5(zip_path.read_bytes()).hexdigest()
    if digest != _XENIUM_TINY_MD5:
        zip_path.unlink()
        raise RuntimeError(
            f"Xenium tiny archive checksum mismatch (got {digest}, expected "
            f"{_XENIUM_TINY_MD5}); the cached file was removed, re-run to re-download."
        )
    outs_dir = d / "outs"
    if not outs_dir.exists():
        log.info("Extracting Xenium archive ...")
        _extract_zip(zip_path, d / "_extracted")
        src = _find_xenium_outs(d / "_extracted")
        src.rename(outs_dir)
    return outs_dir


def xenium_tiny(cache_dir=None):
    """Download and return the 10x Genomics Xenium Human Kidney tiny dataset.

    Source: 10x Genomics CDN (CC BY 4.0), ~20 MB ZIP (~350 cells).
    Intended for smoke tests only.  For full-size analysis use
    :func:`xenium` which downloads the 254 MB Human Lung Cancer FFPE dataset.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/xenium_tiny/``.
    """
    from astrolabe import io
    return io.read_xenium(str(_xenium_tiny_outs(cache_dir)))


def merscope(cache_dir=None):
    """Download and return a MERSCOPE assembloid dataset (Zenodo 13835782, region 0).

    Source: Zenodo record 13835782 (Vizgen MERSCOPE platform, 2024), ~126 MB total.
    Returns a :class:`mudata.MuData` with ``'merscope'`` modality.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/merscope/``.
    """
    d = _cache_root(cache_dir) / "merscope"
    cbg = d / "cell_by_gene.csv"
    meta = d / "cell_metadata.csv"
    _download(_MERSCOPE_CBG_URL, cbg, "MERSCOPE cell_by_gene (Zenodo 13835782, 102 MB)")
    _download(_MERSCOPE_META_URL, meta, "MERSCOPE cell_metadata (Zenodo 13835782, 24 MB)")
    from astrolabe import io
    return io.read_merscope(str(d), load_boundaries=False)


def cosmx(cache_dir=None):
    """Download and return the NanoString CosMx Lung12 public dataset.

    Source: NanoString public S3 bucket (Lung12, NSCLC), ~155 MB total.
    Returns a :class:`mudata.MuData` with ``'cosmx'`` modality.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/cosmx/``.
    """
    d = _cache_root(cache_dir) / "cosmx"
    _download(_COSMX_EXPR_URL, d / "Lung12_exprMat_file.csv",
              "CosMx Lung12 exprMat (NanoString S3, 146 MB)")
    _download(_COSMX_META_URL, d / "Lung12_metadata_file.csv",
              "CosMx Lung12 metadata (NanoString S3, 9 MB)")
    from astrolabe import io
    return io.read_cosmx(str(d), load_boundaries=False)


def visium_hd(cache_dir=None):
    """Download and return the 10x Genomics Visium HD Mouse Small Intestine dataset.

    Source: 10x Genomics CDN (CC BY 4.0), Space Ranger 3.0.0 — 2µm binned
    outputs plus the full-resolution H&E microscopy image. This dataset
    predates Space Ranger v4.0's on-board segmentation feature, so
    ``uns['default_modality']`` stays ``'visium_hd_bins'`` after loading; use
    :func:`~astrolabe.seg.segment_bin2cell` (or
    :func:`~astrolabe.seg.segment_stardist` + :func:`~astrolabe.seg.assign_bins`)
    to derive a cell-level ``'visium_hd_cells'`` modality.

    Returns a :class:`mudata.MuData` with a ``'visium_hd_bins'`` modality
    (lazy — see :func:`~astrolabe.pp.load_bin_counts`) and a ``'he'`` modality.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/visium_hd/``.
    """
    d = _cache_root(cache_dir) / "visium_hd"
    outs_dir = d / "outs"
    if not (outs_dir / "binned_outputs").is_dir():
        tar_path = d / "Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz"
        _download(
            _VISIUM_HD_BASE + "_binned_outputs.tar.gz", tar_path,
            "Visium HD Mouse Small Intestine (binned_outputs)",
        )
        log.info("Extracting Visium HD archive ...")
        _extract_tar(tar_path, d / "_extracted")
        src = _find_visium_hd_outs(d / "_extracted")
        src.rename(outs_dir)
    _download(
        _VISIUM_HD_BASE + "_tissue_image.btf",
        outs_dir / "Visium_HD_Mouse_Small_Intestine_tissue_image.btf",
        "Visium HD Mouse Small Intestine (H&E tissue image)",
    )
    from astrolabe import io
    return io.read_visium_hd(str(outs_dir))


def visium_hd_tiny(cache_dir=None):
    """Download and return the 10x Genomics Visium HD tiny smoke-test dataset.

    Source: 10x Genomics CDN (CC BY 4.0), ~297 MB ZIP, Space Ranger v4.0.1.
    10x's own docs describe this dataset as downsampled specifically "to
    reduce file size and compute time" for automated testing. Includes
    native on-board segmentation (``segmented_outputs/``), so the returned
    MuData's ``uns['default_modality']`` is already ``'visium_hd_cells'``.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/visium_hd_tiny/``.
    """
    d = _cache_root(cache_dir) / "visium_hd_tiny"
    zip_path = d / "visium_hd_tiny.zip"
    _download(_VISIUM_HD_TINY_URL, zip_path, "Visium HD tiny (10x Genomics, 297 MB)")
    outs_dir = d / "outs"
    if not (outs_dir / "binned_outputs").is_dir():
        log.info("Extracting Visium HD archive ...")
        _extract_zip(zip_path, d / "_extracted")
        src = _find_visium_hd_outs(d / "_extracted")
        src.rename(outs_dir)
    from astrolabe import io
    return io.read_visium_hd(str(outs_dir))


def ome_tiff(cache_dir=None):
    """Download and return a 2-channel OME-TIFF sample image.

    Source: OME Bio-Formats "tubhiswt" sample (OME consortium, CC BY 4.0),
    a real two-color acquisition of a 4-cell embryo (~264 KB per channel):
    ``tubulin`` (transmitted-light channel) and ``histone`` (fluorescence
    channel marking chromosomes). OME ships this as two single-channel
    companion OME-TIFF files sharing one OME-XML document; astrolabe reads
    each channel's raw plane directly (bypassing tifffile's multi-file OME
    series stitching, which requires both files to be co-located) and
    combines them into one local multi-channel OME-TIFF on first download,
    cached alongside the originals so subsequent calls skip the combine step.
    Returns an :class:`anndata.AnnData` with 2 channel variables and an
    :class:`~astrolabe._utils.ImageSpec` in ``uns['spatial']``.
    Use ``ab.pp.extract_if_intensities`` to compute per-cell intensities
    after loading cell boundaries separately.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/ome_tiff/``.
    """
    d = _cache_root(cache_dir) / "ome_tiff"
    c0_path = d / "tubhiswt_C0.ome.tif"
    c1_path = d / "tubhiswt_C1.ome.tif"
    combined_path = d / "tubhiswt.ome.tif"
    _download(_OME_TIFF_C0_URL, c0_path, "OME-TIFF tubhiswt tubulin channel (OME, 264 KB)")
    _download(_OME_TIFF_C1_URL, c1_path, "OME-TIFF tubhiswt histone channel (OME, 264 KB)")
    if not combined_path.exists():
        import numpy as np
        import tifffile
        with tifffile.TiffFile(str(c0_path)) as tif:
            c0 = tif.pages[0].asarray()
        with tifffile.TiffFile(str(c1_path)) as tif:
            c1 = tif.pages[0].asarray()
        log.info("Combining tubhiswt channels into a single multi-channel OME-TIFF ...")
        tifffile.imwrite(
            str(combined_path),
            np.stack([c0, c1], axis=0),
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["tubulin", "histone"]}},
        )
    from astrolabe import io
    return io.read_ome_tiff(str(combined_path))


def he(cache_dir=None):
    """Download and return a brightfield H&E whole-slide image (Huron LE176 scanner).

    Source: OpenSlide test data (CC0-1.0), ~36 MB pyramidal TIFF.
    Returns an :class:`anndata.AnnData` with ``var_names``
    ``['hematoxylin', 'eosin', 'residual']`` and an
    :class:`~astrolabe._utils.ImageSpec` in ``uns['spatial']``.

    Parameters
    ----------
    cache_dir
        Directory to cache downloaded files.  Defaults to
        ``<cwd>/.astrolabe_cache/datasets/he/``.
    """
    d = _cache_root(cache_dir) / "he"
    fpath = d / "Huron-1.tif"
    _download(_HE_URL, fpath, "H&E brightfield slide (OpenSlide/Huron, CC0-1.0, 36 MB)")
    from astrolabe import io
    return io.read_he(str(fpath))
