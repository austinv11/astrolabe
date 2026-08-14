"""
Spatial selection and coordinate transformation.

Functions
---------
crop
    Filter cells whose centroids fall inside a polygon region; zoom plots
    automatically via ``uns['spatial_bounds']``.
auto_crop
    Crop to a robust bounding box computed from the cells themselves --
    percentile-trimmed to ignore stray outliers, no manual region needed.
rotate
    Rotate cell coordinates (and image transform chain) CCW by a given angle.
flip
    Mirror cell coordinates (and image transform chain) along x, y, or both axes.
apply_affine
    Apply one caller-supplied 3x3 affine transform across every modality in a
    MuData at once, keeping multi-image datasets (xenium + he + if + morphology)
    in a single coordinate frame.
estimate_stain_matrix
    Re-estimate the H&E stain deconvolution matrix for an already-loaded image.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import anndata as ad
import mudata as md
import numpy as np
import pyarrow as pa

from astrolabe._shapes import rectangle
from astrolabe._utils import (
    ImageFilter,
    ImageSpec,
    _resolve_adata_and_modality,
    _transform_coords,
    with_adata,
)

if TYPE_CHECKING:
    pass

__all__ = ["crop", "rotate", "flip", "apply_affine", "add_image_filter", "estimate_stain_matrix"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return boolean mask of points that fall inside *polygon*.

    Uses shapely's vectorised ``contains_xy`` (shapely ≥2.x).

    Parameters
    ----------
    points
        ``(N, 2)`` float array of ``[x, y]`` centroids in µm.
    polygon
        ``(M, 2)`` float array of polygon vertices in µm.  Need not be closed.
    """
    import shapely
    from shapely.geometry import Polygon

    geom = Polygon(polygon)
    return shapely.contains_xy(geom, points[:, 0], points[:, 1])


def _transform_boundary_obsm(
    raw,
    M: np.ndarray,
) -> object:
    """Apply affine *M* to an awkward-array boundary obsm entry.

    Handles both ``awkward.Array`` (variable-length vertex lists) and plain
    numpy arrays of shape ``(n_cells, n_verts, 2)``.
    """
    try:
        import awkward as ak  # type: ignore[import]

        if isinstance(raw, ak.Array):
            counts = np.asarray(ak.num(raw, axis=1))          # (n_cells,)
            flat = np.asarray(ak.flatten(raw, axis=1))         # (total_verts, 2)
            transformed = _transform_coords(flat.astype(np.float64), M).astype(flat.dtype)
            return ak.unflatten(ak.Array(transformed), counts, axis=0)
    except ImportError:
        pass

    # Fall back: treat as regular (n_cells, n_verts, 2) numpy array
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        n, v, _ = raw.shape
        flat = raw.reshape(-1, 2)
        transformed = _transform_coords(flat.astype(np.float64), M).astype(raw.dtype)
        return transformed.reshape(n, v, 2)

    log.warning(
        "Could not transform boundaries of type %s; skipping.", type(raw).__name__
    )
    return raw


def _crop_transcripts_uns(
    uns: dict, region: np.ndarray, x_col: str, y_col: str
) -> None:
    """Filter ``uns['transcripts']`` (a pyarrow.Table) to *region*, in place.

    No-op if ``'transcripts'`` is absent, so callers can gate cheaply and
    unconditionally on ``"transcripts" in adata.uns`` beforehand.
    """
    tx = uns.get("transcripts")
    if tx is None:
        return
    coords = np.column_stack(
        [np.asarray(tx[x_col]), np.asarray(tx[y_col])]
    ).astype(np.float64)
    mask = _points_in_polygon(coords, region)
    uns["transcripts"] = tx.filter(pa.array(mask))


def _append_image_transform(adata: ad.AnnData, M: np.ndarray) -> None:
    """Append µm-space affine *M* to all ImageSpec transform_chains in *adata.uns*."""
    libs = adata.uns.get("spatial", {})
    if len(libs) > 1:
        log.warning(
            "apply_affine: multiple libraries in uns['spatial'] (%s); "
            "the same affine will be applied to ALL library images. "
            "If libraries have different coordinate frames, pass a single-library "
            "AnnData (mdata['<modality>']) and handle each library separately.",
            list(libs.keys()),
        )
    for lib_data in libs.values():
        for spec in lib_data.get("images", {}).values():
            if isinstance(spec, ImageSpec):
                spec.transform_chain.append(M)


def _build_rotation_matrix(
    angle_deg: float,
    center: tuple[float, float],
) -> np.ndarray:
    """3×3 CCW rotation matrix in µm space around *center*."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = float(center[0]), float(center[1])

    T_to   = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R      = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    T_back = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]], dtype=np.float64)
    return T_back @ R @ T_to


def _build_flip_matrix(
    axis: str,
    center: tuple[float, float],
) -> np.ndarray:
    """3×3 reflection matrix in µm space around *center*.

    ``axis='x'`` mirrors across the vertical axis (negates x coordinates).
    ``axis='y'`` mirrors across the horizontal axis (negates y coordinates).
    ``axis='both'`` mirrors across both.
    """
    cx, cy = float(center[0]), float(center[1])
    sx = -1.0 if axis in ("x", "both") else 1.0
    sy = -1.0 if axis in ("y", "both") else 1.0
    return np.array(
        [
            [sx, 0,  cx * (1.0 - sx)],
            [0,  sy, cy * (1.0 - sy)],
            [0,  0,  1.0            ],
        ],
        dtype=np.float64,
    )


def _apply_coord_transform(
    adata: ad.AnnData,
    M: np.ndarray,
    spatial_key: str,
    boundaries_keys: list[str],
) -> None:
    """Transform cell centroids and boundary arrays in *adata* by affine *M*."""
    coords = adata.obsm[spatial_key].astype(np.float64)
    adata.obsm[spatial_key] = _transform_coords(coords, M).astype(
        adata.obsm[spatial_key].dtype
    )
    for bkey in boundaries_keys:
        if bkey in adata.obsm:
            adata.obsm[bkey] = _transform_boundary_obsm(adata.obsm[bkey], M)


def _transform_spatial_bounds(adata: ad.AnnData, M: np.ndarray) -> None:
    """Keep uns['spatial_bounds'] (the pp.crop() viewport hint) in sync with *M*.

    rotate()/flip() used to leave a pre-existing spatial_bounds (set by an
    earlier pp.crop()) untouched -- describing the *pre-transform* window --
    so a rotated/flipped, previously-cropped panel rendered auto-zoomed to
    the wrong region (severely visually clipped or offset), the same class
    of MuData-consistency gap pp.crop() itself already handles for sibling
    modalities. The 4 corners of the old bounding box are transformed by the
    same affine and re-enclosed in a new axis-aligned box, since a rotation
    can turn an axis-aligned rectangle into a non-axis-aligned one.
    """
    bounds = adata.uns.get("spatial_bounds")
    if bounds is None:
        return
    corners = np.array(
        [
            [bounds["xmin"], bounds["ymin"]],
            [bounds["xmax"], bounds["ymin"]],
            [bounds["xmax"], bounds["ymax"]],
            [bounds["xmin"], bounds["ymax"]],
        ],
        dtype=np.float64,
    )
    transformed = _transform_coords(corners, M)
    adata.uns["spatial_bounds"] = {
        "xmin": float(transformed[:, 0].min()),
        "xmax": float(transformed[:, 0].max()),
        "ymin": float(transformed[:, 1].min()),
        "ymax": float(transformed[:, 1].max()),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def crop(
    data: ad.AnnData | md.MuData,
    region: np.ndarray,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    crop_transcripts: bool = True,
    transcripts_x_col: str = "x_location",
    transcripts_y_col: str = "y_location",
    inplace: bool = False,
) -> ad.AnnData | None:
    """Filter cells to those whose centroids fall inside *region*.

    Cell coordinates are updated eagerly.  Images are **not** modified —
    :func:`~astrolabe.pl._spatial._render_panel` zooms the plot view by
    reading the bounding box stored in ``adata.uns['spatial_bounds']``.

    When ``uns['transcripts']`` (a :class:`pyarrow.Table`, e.g. from
    :func:`~astrolabe.io.read_xenium_transcripts`) is present, it is cropped
    to the same *region* by default, using the same point-in-polygon test as
    the cell centroids — so cells and transcripts can never disagree about
    which spatial window a cropped object represents.

    Unlike most ``pp``/``tl`` functions, this does **not** use ``@with_adata``:
    when *data* is a :class:`~mudata.MuData` **and** ``inplace=True``, every
    *other* modality sharing the same coordinate frame (e.g. a separate
    ``'he'``/``'if'`` image modality with no cells of its own to subset) also
    has its ``uns['spatial_bounds']`` viewport hint updated to match, in
    place on *data* — so a single ``ab.pp.crop(mdata, region, inplace=True)``
    keeps every modality's plot viewport in sync automatically, with no
    manual ``mdata['he'].uns['spatial_bounds'] = ...`` line needed.

    This sibling sync only happens for ``inplace=True``: with
    ``inplace=False`` *data* is never mutated at all (matching every other
    ``inplace=False`` function in this library), so cropping one modality out
    of a shared MuData without disturbing the rest of it (e.g. keeping the
    full-resolution ``mdata`` around for later cells) does not silently
    change what an unrelated ``ab.pl.image(mdata, modality=...)`` call
    renders elsewhere. To get a fully-synced crop *without* touching the
    original object, crop an explicit copy in place instead — see the second
    example below.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    region
        ``(n_verts, 2)`` polygon in µm.  Use :func:`~astrolabe.pl.rectangle`,
        :func:`~astrolabe.pl.circle`, or :func:`~astrolabe.pl.arrow` to
        generate common shapes.
    modality
        Which modality to crop cells/transcripts from, when *data* is a
        MuData. Falls back to ``uns['default_modality']`` if not given.
    spatial_key
        Key in ``obsm`` for cell centroids.
    crop_transcripts
        If ``True`` (default) and ``uns['transcripts']`` is present, filter it
        to the same *region* too. Set ``False`` to leave it untouched (e.g.
        when repeatedly cropping interactively and re-filtering a large table
        each time isn't wanted).
    transcripts_x_col, transcripts_y_col
        Column names for transcript x/y coordinates in ``uns['transcripts']``.
    inplace
        If ``False`` (default), return the filtered :class:`~anndata.AnnData`.
        If ``True``, modify in place and return ``None``.

    Returns
    -------
    None or AnnData

    Examples
    --------
    >>> rect = ab.pl.rectangle(center, 600.0, 600.0)
    >>> morph_crop = ab.pp.crop(mdata, rect, inplace=False)  # mdata itself untouched
    >>> ab.pl.image(mdata, modality='he')  # still renders the FULL H&E image

    To keep ``'he'`` cropped in sync too, without touching the original
    ``mdata`` used elsewhere in a notebook, crop an explicit copy in place:

    >>> mdata_crop = mdata.copy()
    >>> ab.pp.crop(mdata_crop, rect, inplace=True)  # mutates the copy only
    >>> # mdata_crop['he'].uns['spatial_bounds'] is now set to match too:
    >>> (ab.pl.image(mdata_crop, modality='he') + ab.pl.boundaries(mdata_crop)).show()
    """
    adata, resolved_modality = _resolve_adata_and_modality(data, modality)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found. "
            "Ensure spatial coordinates have been loaded"
        )

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    region = np.asarray(region, dtype=np.float64)
    mask = _points_in_polygon(centroids, region)

    n_kept = int(mask.sum())
    log.info("crop: keeping %d / %d cells inside region.", n_kept, len(mask))

    # Store bounding box for automatic plot zoom
    bounds = {
        "xmin": float(region[:, 0].min()),
        "xmax": float(region[:, 0].max()),
        "ymin": float(region[:, 1].min()),
        "ymax": float(region[:, 1].max()),
    }

    if inplace and isinstance(data, md.MuData):
        for key, sibling in data.mod.items():
            if key != resolved_modality and "spatial" in sibling.uns:
                sibling.uns["spatial_bounds"] = bounds

    if inplace:
        adata.uns["spatial_bounds"] = bounds
        if crop_transcripts and "transcripts" in adata.uns:
            _crop_transcripts_uns(adata.uns, region, transcripts_x_col, transcripts_y_col)
        adata._inplace_subset_obs(mask)
        return None
    else:
        subset = adata[mask].copy()
        subset.uns["spatial_bounds"] = bounds
        if crop_transcripts and "transcripts" in subset.uns:
            _crop_transcripts_uns(subset.uns, region, transcripts_x_col, transcripts_y_col)
        return subset


def auto_crop(
    data: ad.AnnData | md.MuData,
    *,
    modality: str | None = None,
    spatial_key: str = "spatial",
    percentile: float = 1.0,
    padding: float = 0.15,
    crop_transcripts: bool = True,
    transcripts_x_col: str = "x_location",
    transcripts_y_col: str = "y_location",
    inplace: bool = False,
) -> ad.AnnData | None:
    """Crop to a robust bounding box computed from the cells themselves.

    Building a crop region from the true min/max of cell centroids is
    fragile: a single stray or isolated cell far from the rest of the tissue
    silently drags the box out to include a large, mostly-empty margin (real
    example: a 358-cell sample where 357 cells sat in a ~245x135 um patch
    but one lone cell ~580 um away inflated a min/max box to ~1067 um tall,
    ~80% blank). ``auto_crop`` instead trims *percentile* off each end of
    each axis independently before computing bounds, so a handful of
    outliers on either side don't set the frame, then pads the trimmed box
    by *padding* (a fraction of its own width/height) and delegates to
    :func:`crop` for the actual filtering -- every other parameter here is
    forwarded to it unchanged, including its sibling-modality-sync and
    ``inplace`` behavior.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Which modality's cell centroids define the crop region, when *data*
        is a MuData. Falls back to ``uns['default_modality']`` if not given.
    spatial_key
        Key in ``obsm`` for cell centroids.
    percentile
        Trim this percentile of cells from each end of each axis before
        computing bounds (default ``1.0``, i.e. use the 1st-99th percentile
        range). ``0`` falls back to the true min/max (no outlier trimming).
    padding
        Fractional margin added around the trimmed bounding box, relative to
        its own width/height (default ``0.15`` = 15% on each side).
    crop_transcripts, transcripts_x_col, transcripts_y_col, inplace
        Forwarded to :func:`crop` -- see its docstring.

    Returns
    -------
    None or AnnData
        Same convention as :func:`crop`.

    Examples
    --------
    >>> ab.pp.auto_crop(mdata, inplace=True)  # tight crop around the cells, in place
    >>> tight = ab.pp.auto_crop(mdata, padding=0.05)  # returned copy, minimal margin
    """
    adata, _ = _resolve_adata_and_modality(data, modality)

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found. "
            "Ensure spatial coordinates have been loaded"
        )

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if len(centroids) == 0:
        raise ValueError("auto_crop: no cells to crop around (obsm is empty)")

    lo, hi = float(percentile), 100.0 - float(percentile)
    x_lo, x_hi = np.percentile(centroids[:, 0], [lo, hi])
    y_lo, y_hi = np.percentile(centroids[:, 1], [lo, hi])

    cx, cy = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    w, h = (x_hi - x_lo) * (1.0 + padding), (y_hi - y_lo) * (1.0 + padding)
    # A degenerate span (e.g. every kept cell sharing an x or y coordinate)
    # would produce a zero-area region and crop away everything; fall back
    # to a small fixed-size window around the centroid in that case.
    w = w if w > 0 else 1.0
    h = h if h > 0 else 1.0
    region = rectangle((cx, cy), w, h)

    return crop(
        data,
        region,
        modality=modality,
        spatial_key=spatial_key,
        crop_transcripts=crop_transcripts,
        transcripts_x_col=transcripts_x_col,
        transcripts_y_col=transcripts_y_col,
        inplace=inplace,
    )


@with_adata
def rotate(
    adata: ad.AnnData,
    angle_deg: float,
    *,
    center: tuple[float, float] | None = None,
    spatial_key: str = "spatial",
    inplace: bool = False,
) -> ad.AnnData | None:
    """Rotate spatial coordinates counter-clockwise by *angle_deg* degrees.

    Cell centroids and boundary arrays are transformed eagerly.
    Each ``ImageSpec`` in ``adata.uns['spatial']`` receives an entry in its
    ``transform_chain`` that is applied lazily at render time.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    angle_deg
        Rotation angle in degrees (positive = counter-clockwise).
    center
        ``(x, y)`` center of rotation in µm.  Defaults to the mean centroid
        of ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for cell centroids.
    inplace
        If ``False`` (default), return the modified copy.
        If ``True``, modify in place and return ``None``.

    Returns
    -------
    None or AnnData
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    if center is None:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        center = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    M = _build_rotation_matrix(angle_deg, center)
    boundaries_keys = ["cell_boundaries", "nucleus_boundaries"]

    if inplace:
        _apply_coord_transform(adata, M, spatial_key, boundaries_keys)
        _append_image_transform(adata, M)
        _transform_spatial_bounds(adata, M)
        return None
    else:
        out = adata.copy()
        _apply_coord_transform(out, M, spatial_key, boundaries_keys)
        _append_image_transform(out, M)
        _transform_spatial_bounds(out, M)
        return out


@with_adata
def flip(
    adata: ad.AnnData,
    axis: Literal["x", "y", "both"] = "x",
    *,
    center: tuple[float, float] | None = None,
    spatial_key: str = "spatial",
    inplace: bool = False,
) -> ad.AnnData | None:
    """Mirror spatial coordinates along one or both axes.

    ``axis='x'`` reflects across the vertical midline (negates x).
    ``axis='y'`` reflects across the horizontal midline (negates y).
    ``axis='both'`` applies both reflections (equivalent to a 180° rotation).

    Cell centroids and boundary arrays are transformed eagerly.
    Each ``ImageSpec`` in ``adata.uns['spatial']`` receives an entry in its
    ``transform_chain`` that is applied lazily at render time.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    axis
        Axis / axes to flip.
    center
        ``(x, y)`` origin of reflection in µm.  Defaults to the mean centroid
        of ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for cell centroids.
    inplace
        If ``False`` (default), return the modified copy.
        If ``True``, modify in place and return ``None``.

    Returns
    -------
    None or AnnData
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    if center is None:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        center = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    M = _build_flip_matrix(axis, center)
    boundaries_keys = ["cell_boundaries", "nucleus_boundaries"]

    if inplace:
        _apply_coord_transform(adata, M, spatial_key, boundaries_keys)
        _append_image_transform(adata, M)
        _transform_spatial_bounds(adata, M)
        return None
    else:
        out = adata.copy()
        _apply_coord_transform(out, M, spatial_key, boundaries_keys)
        _append_image_transform(out, M)
        _transform_spatial_bounds(out, M)
        return out


def apply_affine(
    mdata: md.MuData,
    matrix: np.ndarray,
    *,
    spatial_key: str = "spatial",
    boundaries_keys: tuple[str, ...] = ("cell_boundaries", "nucleus_boundaries"),
    inplace: bool = False,
) -> md.MuData | None:
    """Apply one 3×3 affine transform across every modality in *mdata*.

    Unlike :func:`rotate` / :func:`flip` (which resolve and transform a
    single :class:`~anndata.AnnData` via ``@with_adata``), this walks every
    entry in ``mdata.mod`` and transforms whichever of
    ``obsm[spatial_key]`` / boundary arrays it has, and appends *matrix* to
    every :class:`~astrolabe._utils.ImageSpec` transform chain found in that
    modality's ``uns['spatial']``. Use this to keep a multi-image MuData
    (e.g. Xenium point-cloud data plus paired H&E/IF/morphology images) in a
    single coordinate frame after a manual correction, rather than desyncing
    modalities by transforming only one of them via :func:`rotate` / :func:`flip`.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` to transform. Every modality is visited;
        modalities without ``obsm[spatial_key]`` simply have their image
        transform chains updated (if any), and modalities without any
        ``uns['spatial']`` images are left untouched.
    matrix
        ``(3, 3)`` homogeneous 2D affine matrix, in µm space.
    spatial_key
        Key in ``obsm`` for cell centroids, checked on each modality.
    boundaries_keys
        Additional ``obsm`` keys (boundary polygon arrays) to transform
        alongside *spatial_key* wherever present.
    inplace
        If ``False`` (default), return a transformed copy of *mdata*.
        If ``True``, modify in place and return ``None``.

    Returns
    -------
    None or MuData

    Examples
    --------
    >>> M = numpy.loadtxt("correction.csv", delimiter=",")
    >>> ab.pp.apply_affine(mdata, M, inplace=True)  # xenium + he + morphology all move together
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"matrix must be a (3, 3) homogeneous 2D matrix, got shape {matrix.shape}")

    target = mdata if inplace else mdata.copy()
    for adata in target.mod.values():
        if spatial_key in adata.obsm:
            _apply_coord_transform(adata, matrix, spatial_key, list(boundaries_keys))
        _append_image_transform(adata, matrix)
        _transform_spatial_bounds(adata, matrix)

    return None if inplace else target


@with_adata
def add_image_filter(
    adata: ad.AnnData,
    filter: ImageFilter,
    *,
    image_key: str | None = None,
    library_id: str | None = None,
) -> None:
    """Append *filter* to the pixel-processing pipeline of matching images.

    Filters are stored in :attr:`~astrolabe._utils.ImageSpec.filter_chain`
    and applied per-channel (2-D) in two places: at render time (``pl.image``,
    before normalization) and at analysis time (``pp.extract_if_intensities``,
    before per-cell reduction and stain deconvolution).  Multiple calls stack
    filters in order.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    filter
        Any :class:`~astrolabe._utils.ImageFilter` — callable that accepts
        and returns a 2-D ``float32`` NumPy array.
    image_key
        Key in ``uns['spatial'][lib]['images']`` to target.
        ``None`` targets all image keys in each matched library.
    library_id
        Library to target.  ``None`` targets all libraries.
    modality
        Modality to resolve when *data* is :class:`~mudata.MuData`.
        Falls back to ``uns['default_modality']`` if not given.

    Examples
    --------
    >>> import astrolabe as ab
    >>> ab.pp.add_image_filter(mdata, ab.pl.GaussianFilter(sigma=1.5))
    >>> ab.pp.add_image_filter(mdata, ab.pl.CLAHEFilter(clip_limit=0.02))
    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff=0.05, fov_size_px=220))
    >>> ab.pl.image(mdata).show()   # filters applied lazily at render time
    """
    spatial = adata.uns.get("spatial", {})
    for lib, lib_data in spatial.items():
        if library_id is not None and lib != library_id:
            continue
        for key, spec in lib_data.get("images", {}).items():
            if image_key is not None and key != image_key:
                continue
            if isinstance(spec, ImageSpec):
                spec.filter_chain.append(filter)


@with_adata
def estimate_stain_matrix(
    adata: ad.AnnData,
    *,
    image_key: str = "full",
    method: Literal["ruifrok", "macenko"] = "macenko",
    library_id: str | None = None,
) -> None:
    """Re-estimate the H&E stain deconvolution matrix for an already-loaded image.

    Updates :attr:`~astrolabe._utils.ImageSpec.stain_matrix` in place on every
    matching :class:`~astrolabe._utils.ImageSpec` found in
    ``uns['spatial'][*]['images'][image_key]``.  Use this after
    :func:`~astrolabe.io.read_he` when the default ``'ruifrok'`` matrix gives
    poor channel separation for a specific slide.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` holding H&E image
        metadata in ``uns['spatial']``.
    image_key
        Key in ``uns['spatial'][lib]['images']`` whose spec to update.
        Default ``'full'``.
    method
        ``'macenko'`` (default) — SVD-based estimation from slide pixels.
        ``'ruifrok'`` — restore the published Ruifrok-Johnston matrix.
    library_id
        Library to target.  ``None`` updates all libraries.
    modality
        Modality to resolve when *data* is :class:`~mudata.MuData`.

    Examples
    --------
    >>> ab.io.read_he("slide.tif", mdata=mdata)           # loaded with ruifrok (default)
    >>> ab.pp.estimate_stain_matrix(mdata, modality="he") # re-estimate with macenko
    >>> ab.pl.image(mdata, modality="he").show()
    """
    from astrolabe.io._he import _estimate_stain_matrix

    spatial = adata.uns.get("spatial", {})
    updated = 0
    for lib, lib_data in spatial.items():
        if library_id is not None and lib != library_id:
            continue
        spec = lib_data.get("images", {}).get(image_key)
        if not isinstance(spec, ImageSpec):
            continue
        spec.stain_matrix = _estimate_stain_matrix(spec, method=method)
        # Propagate to any alias specs (level_k entries share the same object)
        updated += 1
    if updated == 0:
        log.warning(
            "estimate_stain_matrix: no ImageSpec found at image_key=%r "
            "(library_id=%r). Check modality and image_key arguments.",
            image_key, library_id,
        )
