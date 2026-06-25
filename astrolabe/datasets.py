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
>>> adata = ab.datasets.ome_tiff()      # OME Bio-Formats sample (226 KB)
>>> adata = ab.datasets.he()            # OpenSlide Huron H&E brightfield (36 MB)

Cache
-----
Files are written once to ``<cache_dir>/<platform>/`` and reused on
subsequent calls.  Override with the ``cache_dir`` parameter.
"""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["xenium", "xenium_tiny", "merscope", "cosmx", "ome_tiff", "he"]

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
}

# Tiny smoke-test dataset (~20 MB zip, ~350 cells).
_XENIUM_TINY_URL = (
    "https://cf.10xgenomics.com/samples/xenium/4.0.0/"
    "Xenium_V1_Protein_Human_Kidney_tiny/"
    "Xenium_V1_Protein_Human_Kidney_tiny_outs.zip"
)

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

_OME_TIFF_URL = (
    "https://downloads.openmicroscopy.org/images/OME-TIFF/2016-06/"
    "bioformats-artificial/multi-channel.ome.tif"
)

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

    Source: 10x Genomics CDN (CC BY 4.0), ~5 GB total (7 files: 6 expression
    files ~254 MB + morphology OME-TIFF ~4.8 GB).
    FFPE Human Lung Adenocarcinoma, 162k cells, 377-gene panel.
    Returns a :class:`mudata.MuData` with ``'xenium'`` modality containing
    cell boundaries, spatial coordinates in microns, transcript detections, and
    the morphology image attached as an :class:`~astrolabe._utils.ImageSpec`
    under ``uns['spatial'][library_id]['images']['morphology']``.

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
    return io.read_xenium(str(outs_dir), morphology_image=True)


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
    d = _cache_root(cache_dir) / "xenium_tiny"
    zip_path = d / "xenium_tiny.zip"
    _download(_XENIUM_TINY_URL, zip_path, "Xenium Human Kidney tiny (10x Genomics, 20 MB)")
    outs_dir = d / "outs"
    if not outs_dir.exists():
        log.info("Extracting Xenium archive ...")
        _extract_zip(zip_path, d / "_extracted")
        src = _find_xenium_outs(d / "_extracted")
        src.rename(outs_dir)
    from astrolabe import io
    return io.read_xenium(str(outs_dir))


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


def ome_tiff(cache_dir=None):
    """Download and return a 3-channel OME-TIFF sample image.

    Source: OME Bio-Formats artificial sample (OME consortium), 226 KB.
    Returns an :class:`anndata.AnnData` with 3 channel variables and an
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
    fpath = d / "multi-channel.ome.tif"
    _download(_OME_TIFF_URL, fpath, "OME-TIFF 3-channel sample (OME, 226 KB)")
    from astrolabe import io
    return io.read_ome_tiff(str(fpath))


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
