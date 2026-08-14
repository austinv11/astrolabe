"""
Reader for H&E (RGB) whole-slide or tile images with optional stain separation.

Canonical call
--------------
>>> he_adata = ab.io.read_he("slide.tif")
>>> # with stain separation already configured (default):
>>> ab.pl.image(mdata, modality="he", channel="hematoxylin").show()
>>> # or to register into an existing MuData:
>>> ab.io.read_he("slide.tif", mdata=mdata, mdata_key="he")

Schema (returned AnnData)
--------------------------
X           : None
obs         : empty DataFrame
var         : channel_name index: ['hematoxylin', 'eosin', 'residual']
              (or ['red', 'green', 'blue'] when separate_stains=False)
uns
    spatial
        <library_id>
            coordinate_unit : 'microns'
            images
                full        : ImageSpec  (stain_matrix set when separate_stains=True)
    astrolabe_schema_version : str

Notes
-----
- The image on disk is always stored as RGB.  ``ImageSpec.stain_matrix`` drives
  stain deconvolution in two places: lazily at render time (``pl.image``), and
  eagerly at analysis time via ``pp.extract_if_intensities(mdata, if_key='he')``
  which fills ``mdata['he'].X`` with per-cell ``[hematoxylin, eosin, residual]``
  concentrations suitable for downstream analysis.
- ``pixel_size_um`` is inferred from tifffile resolution tags when present and
  must be supplied manually via the ``pixel_size_um`` parameter for files that
  lack embedded metadata (e.g. plain TIFF exports from slide scanners).
- For SVS / NDPI files, pass ``pixel_size_um=None`` to attempt automatic
  extraction from the embedded metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import tifffile

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY
from astrolabe._utils import ImageSpec

__all__ = ["read_he"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_he(
    path: str | Path,
    *,
    library_id: str | None = None,
    pixel_size_um: float | None = None,
    affine: np.ndarray | None = None,
    separate_stains: bool = True,
    method: Literal["ruifrok", "macenko"] = "macenko",
    mdata: "md.MuData | None" = None,
    mdata_key: str = "he",
) -> ad.AnnData:
    """Read an H&E RGB image into an :class:`anndata.AnnData`.

    The image is stored as an :class:`~astrolabe._utils.ImageSpec` path
    descriptor; no pixel data is kept in memory.  When *separate_stains* is
    ``True`` (default), a stain deconvolution matrix is estimated and stored on
    the spec.  The matrix is used in two ways: (1) lazily at render time by
    ``pl.image`` to display hematoxylin / eosin / residual pseudo-channels, and
    (2) eagerly by ``pp.extract_if_intensities(mdata, if_key='he')`` to fill
    ``mdata['he'].X`` with per-cell stain concentrations for analysis.

    Parameters
    ----------
    path
        Path to an RGB TIFF, SVS, or similar pyramidal image file.
    library_id
        Key under ``uns['spatial']``.  Defaults to the file stem.
    pixel_size_um
        Physical size of one pixel in microns.  When ``None``, the reader
        attempts to extract it from TIFF resolution tags or XMP/EXIF metadata;
        falls back to 1.0 with a warning when unavailable.
    affine
        Optional ``(3, 3)`` homogeneous 2D matrix mapping image pixel
        coordinates ``(col, row)`` to the reference spatial frame in µm.
    separate_stains
        If ``True`` (default), estimate a stain deconvolution matrix and store
        it in ``ImageSpec.stain_matrix`` so that rendering produces
        ``[hematoxylin, eosin, residual]`` channels instead of raw RGB.
    method
        Stain estimation method.  ``'ruifrok'`` uses the published
        Ruifrok-Johnston matrix and requires no pixel sampling.  ``'macenko'`` (default)
        fits the matrix to the slide via SVD on optical-density space pixels
        sampled from the coarsest pyramid level.
    mdata
        If given, add the returned AnnData as ``mdata[mdata_key]`` in place.
    mdata_key
        Key to use when inserting into *mdata*.  Default ``'he'``.

    Returns
    -------
    anndata.AnnData
        AnnData with ``X=None``, channel names as ``var_names``, and an
        :class:`~astrolabe._utils.ImageSpec` in
        ``uns['spatial'][library_id]['images']['full']``.

    Examples
    --------
    Read a standalone H&E slide:

    >>> he_adata = ab.io.read_he("slide.tif")

    Register into an existing MuData and specify pixel size:

    >>> ab.io.read_he("slide.tif", mdata=mdata, pixel_size_um=0.2201)

    Use Macenko SVD for tissue-adaptive deconvolution:

    >>> ab.io.read_he("slide.tif", method="macenko")
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"H&E image not found: {path}")

    library_id = library_id or path.stem.split(".")[0]
    log.info("Reading H&E '%s' from %s", library_id, path)

    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (3, 3):
            raise ValueError(
                f"affine must be a (3, 3) homogeneous 2D matrix, got shape {affine.shape}"
            )

    spec = _extract_he_spec(path, pixel_size_um=pixel_size_um, registration_affine=affine)

    stain_matrix: np.ndarray | None = None
    if separate_stains:
        stain_matrix = _estimate_stain_matrix(spec, method=method)
        spec.stain_matrix = stain_matrix
        channel_names = ["hematoxylin", "eosin", "residual"]
    else:
        channel_names = ["red", "green", "blue"]

    spec.channel_names = channel_names

    images: dict[str, ImageSpec] = {"full": spec}
    if spec.n_levels > 1:
        for k in range(1, spec.n_levels):
            images[f"level_{k}"] = spec

    var_df = pd.DataFrame(
        {"channel_name": channel_names},
        index=pd.Index(channel_names, name="channel_name"),
    )
    obs = pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str))
    uns: dict = {
        "spatial": {
            library_id: {
                "coordinate_unit": "microns",
                "images": images,
                "source_path": str(path),
            }
        },
        UNS_SCHEMA_VERSION_KEY: SCHEMA_VERSION,
    }

    adata = ad.AnnData(obs=obs, var=var_df, uns=uns)

    log.info(
        "Loaded H&E: pixel_size=%.4f µm, %d pyramid level(s), shape=%s, stains=%s",
        spec.pixel_size_um,
        spec.n_levels,
        spec.full_shape,
        "separated" if separate_stains else "raw RGB",
    )

    if mdata is not None:
        from astrolabe.io._io import attach_modality
        attach_modality(mdata, adata, mdata_key)

    return adata


# ---------------------------------------------------------------------------
# ImageSpec factory
# ---------------------------------------------------------------------------


def _extract_he_spec(
    path: Path,
    pixel_size_um: float | None,
    registration_affine: np.ndarray | None,
) -> ImageSpec:
    """Open an H&E TIFF, extract metadata eagerly, close, return ImageSpec."""
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        raw_axes: str = series.axes  # e.g. 'YXS', 'CYX', 'YX'
        level_shapes: list[tuple[int, ...]] = [tuple(lv.shape) for lv in series.levels]

        if pixel_size_um is None:
            pixel_size_um = _infer_pixel_size(tif)

        # Map the 'S' (samples / RGB components) axis to 'C' so the renderer
        # treats the three color channels as independent image channels.
        dim_order = raw_axes.replace("S", "C")

    full_shape = level_shapes[0]
    # Do not assume the last two axes are (Y, X): H&E/RGB TIFFs are commonly
    # stored channels-last (YXC/YXS after the S->C mapping above), unlike the
    # CYX convention used for fluorescence/morphology images.
    axes_up = dim_order.upper()
    y_idx = axes_up.index("Y") if "Y" in axes_up else len(full_shape) - 2
    x_idx = axes_up.index("X") if "X" in axes_up else len(full_shape) - 1
    fh, fw = full_shape[y_idx], full_shape[x_idx]
    level_scales: list[tuple[float, float]] = [
        (fh / s[y_idx], fw / s[x_idx]) for s in level_shapes
    ]

    return ImageSpec(
        path=str(path),
        pixel_size_um=pixel_size_um,
        dimension_order=dim_order,
        full_shape=full_shape,
        level_shapes=level_shapes,
        level_scales=level_scales,
        registration_affine=registration_affine,
        is_brightfield=True,
    )


def _infer_pixel_size(tif: tifffile.TiffFile) -> float:
    """Try to read pixel size (µm) from TIFF tags / description; return 1.0 on failure."""
    try:
        page = tif.pages[0]
        # Standard TIFF XResolution tag (tag 282) — value is pixels per unit
        x_res = page.tags.get("XResolution")
        res_unit = page.tags.get("ResolutionUnit")  # 1=no unit, 2=inch, 3=cm
        if x_res is not None and res_unit is not None:
            num, den = x_res.value if isinstance(x_res.value, tuple) else (x_res.value, 1)
            ppu = float(num) / float(den) if den else float(num)
            unit_val = res_unit.value if hasattr(res_unit, "value") else int(res_unit)
            if ppu > 0 and unit_val in (2, 3):
                um_per_pixel = (25400.0 / ppu) if unit_val == 2 else (10000.0 / ppu)
                return float(um_per_pixel)
    except Exception:
        pass

    # SVS / Aperio: pixel size embedded in ImageDescription
    try:
        desc = tif.pages[0].description or ""
        import re
        m = re.search(r"MPP\s*=\s*([\d.]+)", desc)
        if m:
            return float(m.group(1))
    except Exception:
        pass

    log.warning(
        "Could not infer pixel size from H&E image metadata; defaulting to 1.0 µm. "
        "Pass pixel_size_um=<value> to set it explicitly."
    )
    return 1.0


# ---------------------------------------------------------------------------
# Stain matrix estimation
# ---------------------------------------------------------------------------


def _estimate_stain_matrix(
    spec: ImageSpec,
    method: Literal["ruifrok", "macenko"] = "ruifrok",
) -> np.ndarray:
    """Return a (3, 3) stain deconvolution matrix.

    Convention: ``concentrations_flat = od_flat @ M`` where *od_flat* has
    shape ``(N, 3)`` (optical density, base-10).  Rows of *M* correspond to
    the three output channels: [hematoxylin, eosin, residual].
    """
    if method == "ruifrok":
        return _ruifrok_matrix()
    if method == "macenko":
        return _macenko_matrix(spec)
    raise ValueError(f"Unknown stain estimation method {method!r}; expected 'ruifrok' or 'macenko'.")


def _ruifrok_matrix() -> np.ndarray:
    """Return the published Ruifrok-Johnston H&E deconvolution matrix."""
    try:
        from skimage.color import hed_from_rgb  # type: ignore[import]
        return np.array(hed_from_rgb, dtype=np.float64)
    except ImportError:
        # Hardcoded Ruifrok-Johnston values (Ruifrok & Johnston, Anal Quant Cytol Histol 2001)
        return np.array(
            [
                [ 1.88006,  -1.02180, -0.54944],
                [-1.07515,   1.82494, -0.57073],
                [-0.59761,  -0.59778,  1.43031],
            ],
            dtype=np.float64,
        )


def _macenko_matrix(spec: ImageSpec, sample_n: int = 50_000) -> np.ndarray:
    """Estimate a stain matrix for this slide using the Macenko SVD method.

    Reads a downsampled pyramid level (not the full image) for speed.
    """
    # The coarsest level (used here previously) can be a heavily-averaged
    # thumbnail for a large WSI (e.g. ~500x500), which pulls stain colors
    # toward gray and degrades separation -- exactly the failure mode this
    # function's own fallback warning below anticipates. A mid-resolution
    # level keeps enough real stain-color diversity while still avoiding a
    # full-resolution read.
    level_idx = spec.select_pyramid_level(target_max_dim=2048)

    with tifffile.TiffFile(spec.path) as tif:
        raw: np.ndarray = tif.series[0].levels[level_idx].asarray()

    # Normalize axes: ensure result is (H, W, 3) uint8/float
    axes = spec.dimension_order.upper()  # 'CYX' or 'YXC' etc. (S→C already done)
    if "C" in axes:
        c_ax = axes.index("C")
        if c_ax == 0:
            raw = np.moveaxis(raw[:3], 0, -1)   # (3,H,W) → (H,W,3)
        elif c_ax == raw.ndim - 1:
            raw = raw[..., :3]                   # already (H,W,3)
        else:
            raw = np.moveaxis(raw, c_ax, -1)[..., :3]
    elif raw.ndim == 2:
        raw = np.stack([raw, raw, raw], axis=-1)

    if raw.dtype == np.uint8:
        rgb = raw.astype(np.float64) / 255.0
    else:
        rgb = raw.astype(np.float64)
        rgb = np.clip(rgb / rgb.max() if rgb.max() > 0 else rgb, 0.0, 1.0)

    # Flatten and subsample
    pixels = rgb.reshape(-1, 3)
    eps = 1e-6
    od = -np.log10(np.clip(pixels, eps, 1.0))

    # Remove background (low OD norm → mostly white background)
    od_norm = np.linalg.norm(od, axis=1)
    # 0.15 OD is a standard minimum tissue threshold; below this is glass/background
    od_fg = od[od_norm > 0.15]
    if len(od_fg) < 10:
        log.warning("Too few foreground pixels for Macenko; falling back to Ruifrok matrix.")
        return _ruifrok_matrix()

    if len(od_fg) > sample_n:
        rng = np.random.default_rng(0)
        od_fg = od_fg[rng.choice(len(od_fg), sample_n, replace=False)]

    # SVD — take top 2 principal components
    _, _, Vt = np.linalg.svd(od_fg, full_matrices=False)
    plane = Vt[:2].copy()  # (2, 3)

    # SVD singular vectors have an arbitrary sign (np.linalg.svd gives no
    # sign guarantee, and it can differ run-to-run / slide-to-slide with no
    # visible cause). An unlucky sign can rotate the OD cloud's *actual*
    # angular cluster so it straddles the +-pi branch cut that arctan2
    # introduces below -- percentile(angles, 1)/percentile(angles, 99) on a
    # wrapped distribution reads as two points near the wrap, not the true
    # extremes of the real spread, collapsing v1~=v2 into a near-singular
    # stain matrix and silently falling back to Ruifrok on nearly every
    # real slide (condition number 400-900 observed). Fixed by orienting
    # each singular vector to point toward the OD cloud's own centroid
    # before computing angles, the standard sign-normalization for this
    # exact SVD-based stain-separation approach -- this doesn't change
    # *which* plane was found (a sign flip of a basis vector spans the same
    # plane), only where its zero-angle reference sits relative to the data.
    mean_dir = od_fg.mean(axis=0)
    for i in range(2):
        if np.dot(plane[i], mean_dir) < 0:
            plane[i] = -plane[i]

    # Project and find extreme angular percentiles
    proj = od_fg @ plane.T  # (N, 2)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    ang_lo = np.percentile(angles, 1)
    ang_hi = np.percentile(angles, 99)

    v1 = plane[0] * np.cos(ang_lo) + plane[1] * np.sin(ang_lo)
    v2 = plane[0] * np.cos(ang_hi) + plane[1] * np.sin(ang_hi)

    # Hematoxylin (blue/purple) absorbs red light strongly -> high OD in the
    # red channel; eosin (pink) absorbs comparatively little red -> low OD_R.
    # Canonical reference stain vectors (Ruifrok & Johnston 2001) separate on
    # this red component by a wide margin (~0.65 vs ~0.07); the blue
    # component this used to check separates them far more narrowly (~0.29
    # vs ~0.11, and the code's own prior comment describing "low B -> high
    # OD_B" for hematoxylin was physically backwards besides) -- a faintly
    # eosinophilic or over-blued slide could flip blue-channel ordering and
    # silently swap the hematoxylin/eosin channels. Assign: stain with the
    # higher OD_R (red) component = hematoxylin.
    if v1[0] > v2[0]:
        he = np.stack([v1, v2, np.cross(v1, v2)])
    else:
        he = np.stack([v2, v1, np.cross(v2, v1)])

    # Normalize rows to unit length
    norms = np.linalg.norm(he, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    he = he / norms

    # Invert to get deconvolution matrix (concentrations = od @ M). `he` is
    # stacked as rows = stain vectors (H, E, residual), each a unit vector
    # in OD-RGB space: od_pixel ~= concentrations @ he, so recovering
    # concentrations from an OD pixel needs M = inv(he), used as
    # `concentrations = od @ M = od @ inv(he)`. This previously inverted
    # `he.T` instead -- a real bug, but until the SVD branch-cut fix just
    # above, it was almost never reached: the branch-cut bug inflated `cond`
    # past the fallback threshold on nearly every slide, so execution always
    # took the Ruifrok fallback and never actually exercised this line.
    # Fixing the branch-cut alone, without also fixing this, would have
    # newly activated the wrong-convention inverse for the first time on
    # every slide that now passes the condition check -- both must ship
    # together.
    #
    # A poorly conditioned he (near-parallel stain vectors -- e.g. from a
    # coarse pyramid level with little stain-color diversity) inverts to
    # huge coefficients that blow up any real OD input; fall back rather
    # than return a deconvolution matrix that's numerically useless.
    cond = np.linalg.cond(he)
    if cond > 100:
        log.warning(
            "Macenko stain vectors are poorly separated (condition number %.1f); "
            "falling back to Ruifrok matrix.",
            cond,
        )
        return _ruifrok_matrix()
    try:
        M = np.linalg.inv(he)
    except np.linalg.LinAlgError:
        log.warning("Macenko matrix inversion failed; falling back to Ruifrok.")
        return _ruifrok_matrix()

    return M.astype(np.float64)
