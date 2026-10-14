"""
IF intensity extraction from cell boundary polygons.

This is the bridge between the Xenium modality (which provides segmentation
boundaries in micron space) and the IF modality (which provides a dask-backed
multichannel image in pixel space).

The canonical workflow is:

    mdata = ab.io.read_xenium("run/")
    ab.io.read_ome_tiff("panel.ome.tif", mdata=mdata, mdata_key="if")
    ab.pp.extract_if_intensities(mdata)
    # mdata['if'].X is now (n_cells, n_channels) float32

Design decisions
----------------
- Extraction is done in image pixel coordinates.  Boundaries (stored in
  microns) are converted to pixels via the affine matrix or a fallback
  pixel_size_um scaling.
- We use shapely for polygon rasterization and skimage.draw.polygon for
  mask generation, operating channel-by-channel to keep peak memory
  proportional to (image_width × image_height) rather than
  (image_width × image_height × n_channels).
- For very large images the per-cell bounding box is extracted from the dask
  array before computing, so only the relevant tile is materialized.
- The result is stored as a dense float32 ndarray in mdata['if'].X.  For
  large panels this is at most (n_cells × n_channels × 4 bytes), which is
  small compared to the raw image.
"""

from __future__ import annotations

import logging

import anndata as ad
import awkward as ak
import mudata as md
import numpy as np
import pandas as pd
from skimage.draw import polygon as sk_polygon

from astrolabe._utils import (
    ImageSpec,
    _call_image_filter,
    _dtype_scale,
    _od_deconvolve,
    _transform_coords,
)

__all__ = ["extract_if_intensities"]

log = logging.getLogger(__name__)

_SORT_GRID_UM = 256.0  # coarse-grid cell size for spatial batch ordering


def extract_if_intensities(
    mdata: "md.MuData",
    *,
    xenium_key: str = "xenium",
    if_key: str = "if",
    library_id: str | None = None,
    image_key: str = "full",
    reduction: str = "mean",
    channel_axis: int | None = None,
    affine: np.ndarray | None = None,
    dtype: str = "float32",
    batch_size: int = 512,
) -> None:
    """Extract per-cell intensities from cell boundary polygons.

    Reads the image from ``mdata[if_key].uns``, iterates over cells in
    ``mdata[xenium_key]`` using boundary polygons stored in
    ``obsm['cell_boundaries']``, and fills ``mdata[if_key].X`` with a dense
    ``(n_cells, n_channels)`` matrix.

    Also handles H&E images loaded with :func:`~astrolabe.io.read_he`: when the
    image's :attr:`~astrolabe._utils.ImageSpec.stain_matrix` is set, pixel-level
    stain deconvolution is applied before averaging so that ``X`` contains
    per-cell hematoxylin / eosin / residual concentrations rather than raw RGB.
    Use ``if_key='he'`` (or whatever key was passed to ``read_he``).

    After this call, ``mdata[if_key]`` has the same ``obs_names`` as
    ``mdata[xenium_key]`` and the two modalities are aligned; the cell
    modality's ``obsm['spatial']`` centroids are copied over too (mirroring
    :func:`~astrolabe.io.extract_morphology_intensities`), so
    :func:`~astrolabe.pp.crop` can subset this modality alongside the cells.

    Parameters
    ----------
    mdata
        MuData containing both the Xenium and IF modalities.
    xenium_key
        Key for the Xenium modality in ``mdata``.
    if_key
        Key for the IF modality in ``mdata``.
    library_id
        Key under ``uns['spatial']``.  Inferred automatically if there is
        exactly one library in the IF modality.
    image_key
        Which image to use: ``'full'`` (default) or a pyramid level key
        such as ``'level_1'``.  Lower-resolution levels are faster but
        less accurate for small cells.
    reduction
        Aggregation function applied within each cell mask.  One of
        ``'mean'``, ``'median'``, ``'sum'``, ``'max'``, ``'min'``.  Also
        honored for H&E stain concentrations (applied per stain channel
        after pixel-level deconvolution).
    channel_axis
        Axis in the dask array corresponding to channels.  Inferred
        automatically from the OME dimension order when ``None``.
    affine
        ``(3, 3)`` affine matrix (homogeneous 2D) mapping Xenium micron
        coordinates to image pixel coordinates.  When ``None``, falls back
        to the affine stored in ``uns['spatial'][library_id]``, then to
        a simple scaling by ``1 / pixel_size_um``.
    dtype
        Output dtype for ``X``.  ``'float32'`` is sufficient for 16-bit
        image data after averaging.
    batch_size
        Number of cells to process per dask compute call.  Tune to
        balance memory and I/O overhead.

    Raises
    ------
    KeyError
        If the required modalities or image keys are not found.
    ValueError
        If boundary data (``obsm['cell_boundaries']``) is absent from
        the Xenium modality.
    """
    if xenium_key not in mdata.mod:
        raise KeyError(f"Modality '{xenium_key}' not found in mdata")
    if if_key not in mdata.mod:
        raise KeyError(
            f"Modality '{if_key}' not found in mdata. "
            f"Call ab.io.read_ome_tiff(..., mdata=mdata) first"
        )

    xen_adata: ad.AnnData = mdata[xenium_key]
    if_adata: ad.AnnData = mdata[if_key]

    if "cell_boundaries" not in xen_adata.obsm:
        raise ValueError(
            "obsm['cell_boundaries'] not found in the Xenium modality. "
            "Re-run ab.io.read_xenium with load_boundaries=True"
        )

    # ------------------------------------------------------------------ #
    # Resolve library_id and image                                        #
    # ------------------------------------------------------------------ #
    spatial_uns: dict = if_adata.uns.get("spatial", {})
    if library_id is None:
        ids = list(spatial_uns.keys())
        if len(ids) != 1:
            raise ValueError(
                f"Cannot infer library_id: found {ids}. Pass library_id explicitly."
            )
        library_id = ids[0]

    lib_uns = spatial_uns[library_id]
    images = lib_uns.get("images", {})
    if image_key not in images:
        raise KeyError(
            f"Image key '{image_key}' not found.  Available: {list(images.keys())}"
        )
    spec: ImageSpec = images[image_key]

    result = _extract_intensities_from_spec(
        xen_adata.obsm["cell_boundaries"],
        spec,
        reduction=reduction,
        channel_axis=channel_axis,
        affine=affine,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------ #
    # Store result and align obs                                          #
    # ------------------------------------------------------------------ #
    new_obs = pd.DataFrame(index=xen_adata.obs_names.copy())
    new_obs.index.name = "cell_id"

    if_adata_filled = ad.AnnData(
        X=result.astype(dtype),
        obs=new_obs,
        var=if_adata.var.copy(),
        uns=if_adata.uns.copy(),
    )
    for k, v in if_adata.obsm.items():
        if_adata_filled.obsm[k] = v
    # Rows are aligned 1:1 with the cell modality, so carry its centroids
    # over too (mirroring io.extract_morphology_intensities) — this is what
    # lets pp.crop() subset this modality alongside the cells.
    if "spatial" not in if_adata_filled.obsm and "spatial" in xen_adata.obsm:
        if_adata_filled.obsm["spatial"] = xen_adata.obsm["spatial"].copy()

    mdata.mod[if_key] = if_adata_filled
    mdata.update()

    log.info("extract_if_intensities: done. mdata['%s'].X shape = %s", if_key, result.shape)


# ---------------------------------------------------------------------------
# Shared extraction core
# ---------------------------------------------------------------------------


def _extract_intensities_from_spec(
    boundaries: "ak.Array",
    spec: ImageSpec,
    *,
    reduction: str = "mean",
    channel_axis: int | None = None,
    affine: np.ndarray | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """Extract per-cell mean intensities for all channels.

    Parameters
    ----------
    boundaries
        Awkward ragged array of shape ``(n_cells,)`` where each entry is an
        ``(n_verts, 2)`` float32 polygon in µm.
    spec
        :class:`~astrolabe._utils.ImageSpec` for the image to sample.
    reduction
        Pixel aggregation: ``'mean'``, ``'median'``, ``'sum'``, ``'max'``,
        or ``'min'``.  Applied after ``spec.filter_chain`` (if any).
    channel_axis
        Channel axis in the raw array.  Inferred from
        ``spec.dimension_order`` when ``None``.
    affine
        ``(3, 3)`` µm → pixel affine override.  Falls back to
        ``spec.registration_affine`` then ``1 / spec.pixel_size_um``.
    batch_size
        Cells per tile read.

    Returns
    -------
    np.ndarray
        ``(n_cells, n_channels)`` float32 intensity matrix.
    """
    n_cells = len(boundaries)
    if channel_axis is None:
        channel_axis = _resolve_channel_axis(spec.dimension_order, len(spec.full_shape))
    # channel_axis may still be None here: _resolve_channel_axis returns
    # None when there's no true channel axis at all (e.g. a real Xenium
    # morphology.ome.tif is 'ZYX' -- 11 DAPI z-planes, no antibody panel).
    # Every axis is then "spatial" for tile-cropping purposes; the
    # non-Y/X axes (Z here) get max-projected away inside _read_tile,
    # collapsing to a single implicit channel -- matching
    # pl/_spatial.py's renderer, which already max-projects Z the same way.
    ndim = len(spec.full_shape)
    if channel_axis is None:
        n_channels = 1
        spatial_axes = list(range(ndim))
    else:
        n_channels = spec.full_shape[channel_axis]
        spatial_axes = [i for i in range(ndim) if i != channel_axis]
    M = _resolve_affine(affine, spec)
    reduction_fn = _get_reduction(reduction)
    img_height = spec.full_shape[spatial_axes[-2]] if len(spatial_axes) >= 2 else spec.full_shape[spatial_axes[-1]]
    img_width = spec.full_shape[spatial_axes[-1]]

    result = np.zeros((n_cells, n_channels), dtype=np.float32)

    log.info(
        "Extracting intensities: %d cells × %d channels (image shape=%s)",
        n_cells, n_channels, spec.full_shape,
    )

    # Process cells in spatially sorted order (lexsort on a coarse centroid
    # grid), not obs order: each batch's tile read spans the union bbox of
    # its cells, and 512 cells in arbitrary obs order can span the whole
    # slide — forcing a near-full-resolution read per batch. Spatial
    # ordering keeps every batch's bbox local. Results are written back by
    # original cell index, so output order is unchanged.
    centroids = np.zeros((n_cells, 2), dtype=np.float64)
    for i in range(n_cells):
        verts = np.asarray(boundaries[i])
        if len(verts) > 0:
            centroids[i] = verts.mean(axis=0)
    grid = np.floor(centroids / _SORT_GRID_UM).astype(np.int64)
    sorted_order = np.lexsort((grid[:, 0], grid[:, 1]))

    with spec.open_zarr() as z:
        if hasattr(z, "keys"):
            level_keys = sorted([k for k in z.keys() if str(k).isdigit()], key=int)
            img_arr = z[level_keys[0]] if level_keys else z
        else:
            img_arr = z

        # White-point scale for stain deconvolution, derived from the
        # source dtype (not guessed per-cell from pixel values, which is
        # inconsistent across cells and wrong for 16-bit brightfield data).
        dtype_scale = _dtype_scale(getattr(img_arr, "dtype", np.uint8))

        for batch_start in range(0, n_cells, batch_size):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_idx = sorted_order[batch_start:batch_end]
            batch_boundaries = boundaries[batch_idx]

            all_verts = []
            for i in range(len(batch_boundaries)):
                verts = np.asarray(batch_boundaries[i])
                if len(verts) > 0:
                    all_verts.append(verts)

            if not all_verts:
                continue

            px_verts = _transform_coords(np.vstack(all_verts), M)
            row_min = max(0, int(np.floor(px_verts[:, 1].min())))
            row_max = min(img_height - 1, int(np.ceil(px_verts[:, 1].max())))
            col_min = max(0, int(np.floor(px_verts[:, 0].min())))
            col_max = min(img_width - 1, int(np.ceil(px_verts[:, 0].max())))

            if row_min >= row_max or col_min >= col_max:
                continue

            tile = _read_tile(img_arr, channel_axis, row_min, row_max + 1, col_min, col_max + 1)

            if spec.filter_chain:
                # Extraction always reads pyramid level 0 (full resolution),
                # so the effective µm-per-pixel is the spec's own level-0
                # pixel size. Passing it lets µm-specified filter bandwidths
                # (e.g. GaussianFilter(sigma_um=...)) resolve to the same
                # physical scale here as at render time, where the same
                # filter_chain runs on a downsampled level.
                um_per_px = spec.um_per_px
                for c in range(tile.shape[0]):
                    ch = tile[c].astype(np.float32)
                    for filt in spec.filter_chain:
                        ch = _call_image_filter(filt, ch, um_per_px)
                    tile[c] = ch.astype(np.float64)

            for i in range(len(batch_boundaries)):
                cell_idx = int(batch_idx[i])
                verts_um = np.asarray(batch_boundaries[i])
                if len(verts_um) < 3:
                    continue

                px = _transform_coords(verts_um, M)
                rows_px = np.clip(px[:, 1] - row_min, 0, row_max - row_min).astype(np.float64)
                cols_px = np.clip(px[:, 0] - col_min, 0, col_max - col_min).astype(np.float64)
                # Note: intensity sampling deliberately uses
                # skimage.draw.polygon (pixel-center-inclusive; dilates by
                # ~1 px per side), unlike seg._eval's polygon2mask
                # rasterization — for aggregating intensities over a cell,
                # slightly over-inclusive coverage of boundary pixels is
                # harmless, whereas IoU scoring needs exact containment.
                rr, cc = sk_polygon(rows_px, cols_px, shape=(row_max - row_min + 1, col_max - col_min + 1))
                if len(rr) == 0:
                    continue
                if getattr(spec, "stain_matrix", None) is not None:
                    result[cell_idx] = _deconvolve_cell_pixels(
                        tile[:, rr, cc], spec.stain_matrix, reduction_fn, dtype_scale
                    )
                else:
                    result[cell_idx] = reduction_fn(tile[:, rr, cc], axis=1).astype(np.float32)

            log.debug("Intensity extraction batch %d–%d", batch_start, batch_end)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_channel_axis(dimension_order: str, ndim: int) -> int | None:
    """Return the channel axis index given a numpy-order dimension order string.

    ``dimension_order`` is in numpy-axis order (e.g. ``'CYX'``, ``'TCYX'``),
    as stored in :attr:`~astrolabe._utils.ImageSpec.dimension_order`.

    Returns ``None`` when there is no true channel axis at all -- e.g. a
    real Xenium ``morphology.ome.tif`` is ``'ZYX'`` (multiple DAPI z-planes,
    no antibody panel). Silently treating that Z axis as if it were a
    channel axis (the previous fallback here) reported each z-plane as its
    own spurious "channel", disagreeing with ``pl/_spatial.py``'s renderer,
    which correctly max-projects Z. The caller collapses every non-spatial
    axis (Z included) down to a single implicit channel when this is
    ``None``, the same way the renderer already does.
    """
    order = dimension_order.upper()
    if ndim == 2:
        return None
    if "C" in order:
        return order.index("C")
    if "Z" in order or "T" in order:
        return None
    if ndim == 3:
        return 0
    if ndim == 4:
        return 1
    return 0


def _resolve_affine(
    affine: np.ndarray | None,
    spec: "ImageSpec",
) -> np.ndarray:
    """Return a (3, 3) affine mapping µm → image pixels (col, row).

    Priority:
    1. Explicit *affine* argument.
    2. Inverse of ``spec.registration_affine`` (which maps pixels → µm).
    3. Isotropic scaling by ``1 / spec.pixel_size_um``.
    """
    if affine is not None:
        return np.asarray(affine, dtype=np.float64)

    M = (np.asarray(spec.registration_affine, dtype=np.float64)
         if spec.registration_affine is not None
         else np.diag([spec.pixel_size_um, spec.pixel_size_um, 1.]))
    for transform in spec.transform_chain:
        M = np.asarray(transform) @ M
    if not np.isfinite(M).all():
        raise ValueError("image registration requires finite pixel size and affine")
    return np.linalg.inv(M)


def _read_tile(
    image,
    channel_axis: int | None,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> np.ndarray:
    """Extract a spatial tile from a zarr/numpy array and return (n_channels, H, W).

    ``channel_axis=None`` means there is no true channel axis (e.g. a
    'ZYX' z-stack) -- every axis is treated as spatial for cropping, and a
    size-1 channel axis is inserted afterward so the existing extra-axis
    collapse below (max-projection) folds Z away the same way it already
    folds away any other non-channel, non-spatial axis.
    """
    ndim = image.ndim
    slices = [slice(None)] * ndim
    spatial_axes = list(range(ndim)) if channel_axis is None else [i for i in range(ndim) if i != channel_axis]
    y_axis = spatial_axes[-2] if len(spatial_axes) >= 2 else spatial_axes[-1]
    x_axis = spatial_axes[-1]

    slices[y_axis] = slice(row_min, row_max)
    slices[x_axis] = slice(col_min, col_max)

    tile = np.asarray(image[tuple(slices)])

    if tile.ndim == 2:
        tile = tile[np.newaxis]
    elif channel_axis is None:
        tile = tile[np.newaxis]
    else:
        tile = np.moveaxis(tile, channel_axis if channel_axis < tile.ndim else 0, 0)

    while tile.ndim > 3:
        if tile.shape[1] == 1:
            tile = tile[:, 0]
        else:
            tile = tile.max(axis=1)

    return tile.astype(np.float64)


def _get_reduction(name: str):
    """Return a numpy reduction function for the given name."""
    options = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
        "max": np.max,
        "min": np.min,
    }
    if name not in options:
        raise ValueError(
            f"reduction must be one of {list(options.keys())}, got {name!r}"
        )
    return options[name]


def _deconvolve_cell_pixels(
    rgb_pixels: np.ndarray,
    stain_matrix: np.ndarray,
    reduction_fn,
    scale: float,
) -> np.ndarray:
    """Apply H&E stain deconvolution to a cell's pixels and reduce per stain.

    Parameters
    ----------
    rgb_pixels
        ``(3, n_pixels)`` float64 array of RGB values extracted from one cell.
    stain_matrix
        ``(3, 3)`` deconvolution matrix: ``conc = od @ M`` where *od* is
        optical-density-space RGB.  Convention matches
        :attr:`~astrolabe._utils.ImageSpec.stain_matrix`.
    reduction_fn
        Numpy reduction (mean/median/sum/max/min) applied over pixels for
        each stain channel — the caller's ``reduction`` choice, which is
        honored for H&E exactly like for raw IF channels.
    scale
        White-point divisor derived from the source image dtype (see
        :func:`astrolabe._utils._dtype_scale`), passed explicitly so
        normalization does not depend on each cell's own pixel values.

    Returns
    -------
    np.ndarray
        ``(3,)`` float32 reduced stain concentrations [H, E, residual].
    """
    stains = _od_deconvolve(rgb_pixels, stain_matrix, scale=scale)  # (n_pixels, 3)
    return reduction_fn(stains, axis=0).astype(np.float32)  # (3,)
