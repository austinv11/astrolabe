"""Cell segmentation wrappers — Cellpose and StarDist."""

from __future__ import annotations

import logging

import anndata as ad
import awkward as ak
import numpy as np

from astrolabe._utils import ImageSpec, _transform_coords

__all__ = ["segment_cellpose", "segment_stardist"]

log = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _resolve_spec(
    source: "ImageSpec | ad.AnnData | object",
    library_id: str | None,
    image_key: str,
) -> tuple[ImageSpec, dict | None]:
    """Return (ImageSpec, lib_meta) from a MuData, AnnData, or ImageSpec."""
    import mudata as md

    if isinstance(source, ImageSpec):
        return source, None

    if isinstance(source, (ad.AnnData, md.MuData)):
        uns = source.uns
        spatial = uns.get("spatial", {})
        if not spatial:
            raise KeyError(
                "No images found in uns['spatial']. "
                "Load an OME-TIFF with ab.io.read_ome_tiff() first."
            )
        if library_id is None:
            if len(spatial) == 1:
                library_id = next(iter(spatial))
            else:
                raise ValueError(
                    f"Multiple libraries in uns['spatial']: {list(spatial)}. "
                    "Pass library_id= to select one."
                )
        lib_meta = spatial[library_id]
        images = lib_meta.get("images", {})
        if image_key not in images:
            raise KeyError(
                f"Image '{image_key}' not found in uns['spatial']['{library_id}']['images']. "
                f"Available: {list(images)}"
            )
        return images[image_key], lib_meta

    raise TypeError(
        f"Expected ImageSpec, AnnData, or MuData; got {type(source).__name__}"
    )


def _um_to_px(
    xmin_um: float, ymin_um: float, xmax_um: float, ymax_um: float, spec: ImageSpec
) -> tuple[int, int, int, int]:
    """Convert µm bbox to (row_min, row_max, col_min, col_max) pixel bbox.

    All **four** corners of the µm rectangle are mapped through the affine
    (not just two diagonal ones): under a rotated registration, the pixel
    axis-aligned bounding box of the region is defined by different corners
    than the µm one, and a 2-corner mapping silently crops the region.
    """
    affine = spec.registration_affine
    pxum = spec.pixel_size_um or 1.0
    corners_um = np.array(
        [
            [xmin_um, ymin_um, 1.0],
            [xmin_um, ymax_um, 1.0],
            [xmax_um, ymin_um, 1.0],
            [xmax_um, ymax_um, 1.0],
        ],
        dtype=np.float64,
    )
    if affine is not None:
        inv = np.linalg.inv(affine)
        corners_px = (inv @ corners_um.T).T
        cs, rs = corners_px[:, 0], corners_px[:, 1]
    else:
        cs, rs = corners_um[:, 0] / pxum, corners_um[:, 1] / pxum

    row_min = max(0, int(np.floor(rs.min())))
    row_max = int(np.ceil(rs.max()))
    col_min = max(0, int(np.floor(cs.min())))
    col_max = int(np.ceil(cs.max()))
    return row_min, row_max, col_min, col_max


def _load_channel(
    spec: ImageSpec,
    channel: int | str,
    region_px: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Load one channel from a pyramidal TIFF. Returns (H, W) float32.

    Handles CYX, ZYX (where Z = z-planes / channels), TCYX, and YX layouts.
    When there is no C axis, Z is treated as the channel axis (common in Xenium
    morphology OME-TIFFs that encode multiple z-planes as SizeZ with SizeC=1).

    Parameters
    ----------
    region_px
        Optional ``(row_min, row_max, col_min, col_max)`` pixel crop to load.
        Uses zarr for the partial read, then slices to the exact region.
        Falls back to loading the full level and slicing when zarr is unavailable.
    """
    import tifffile

    ch_idx: int
    if isinstance(channel, str):
        if spec.channel_names and channel in spec.channel_names:
            ch_idx = spec.channel_names.index(channel)
        else:
            raise KeyError(
                f"Channel '{channel}' not found. Available: {spec.channel_names}"
            )
    else:
        ch_idx = int(channel)

    dim = spec.dimension_order  # numpy axis order, e.g. 'CYX', 'ZYX', 'YX'
    has_c = "C" in dim

    if region_px is not None:
        row_min, row_max, col_min, col_max = region_px
        # Use zarr for efficient region reads from large pyramidal TIFFs.
        try:
            import zarr
            with tifffile.TiffFile(spec.path) as tf:
                store = tf.series[0].levels[0].aszarr()
            z = zarr.open(store, mode="r")
            # Build one index/slice per axis in dim order.
            slices: list = []
            for ax in dim:
                if ax == "T":
                    slices.append(0)
                elif ax == "C":
                    slices.append(ch_idx)
                elif ax == "Z":
                    # Z is the channel axis when there is no C axis
                    slices.append(ch_idx if not has_c else 0)
                elif ax == "Y":
                    slices.append(slice(row_min, row_max))
                elif ax == "X":
                    slices.append(slice(col_min, col_max))
                else:
                    slices.append(0)
            img = np.asarray(z[tuple(slices)])
            if img.ndim == 3 and img.shape[0] == 1:
                img = img[0]
            return img.astype(np.float32)
        except Exception:
            # Fallback: load full level and slice.
            with tifffile.TiffFile(spec.path) as tf:
                img = tf.series[0].levels[0].asarray()
            img = _select_channel_from_ndarray(img, dim, ch_idx, has_c)
            return img[row_min:row_max, col_min:col_max].astype(np.float32)

    with tifffile.TiffFile(spec.path) as tf:
        img = tf.series[0].levels[0].asarray()

    return _select_channel_from_ndarray(img, dim, ch_idx, has_c).astype(np.float32)


def _select_channel_from_ndarray(
    img: np.ndarray, dim: str, ch_idx: int, has_c: bool
) -> np.ndarray:
    """Reduce a loaded TIFF array to a (H, W) image by selecting channel/z/t."""
    for ax in ("T",):
        pos = dim.find(ax)
        if pos >= 0:
            img = img.take(0, axis=pos)
            dim = dim[:pos] + dim[pos + 1:]
    if has_c:
        # Collapse Z=0 first, then select C
        pos = dim.find("Z")
        if pos >= 0:
            img = img.take(0, axis=pos)
            dim = dim[:pos] + dim[pos + 1:]
        pos = dim.find("C")
        if pos >= 0:
            img = img.take(ch_idx, axis=pos)
    else:
        # No C axis — treat Z as channel selector
        pos = dim.find("Z")
        if pos >= 0:
            img = img.take(ch_idx, axis=pos)
    return img


def _to_hwc(img: np.ndarray, dim: str) -> np.ndarray:
    """Reduce a loaded TIFF array to (H, W, C), channel axis last, collapsing T (and Z when a real C axis is present).

    Used by :func:`_load_all_channels` for multi-channel StarDist models
    (e.g. ``'2D_versatile_he'``, which requires ``axes='YXC'`` — a true RGB
    stack, not a single selected channel). ``dim`` must describe ``img``'s
    *current* axis order (already-collapsed axes removed from the string to
    match), mirroring how :func:`_select_channel_from_ndarray` is called.
    """
    pos = dim.find("T")
    if pos >= 0:
        img = img.take(0, axis=pos)
        dim = dim[:pos] + dim[pos + 1:]
    has_c = "C" in dim
    pos = dim.find("Z")
    if pos >= 0:
        if has_c:
            img = img.take(0, axis=pos)
            dim = dim[:pos] + dim[pos + 1:]
        else:
            # No dedicated C axis — Z plays the role of channel; keep it.
            dim = dim[:pos] + "C" + dim[pos + 1:]
    if "C" not in dim:
        img = img[..., np.newaxis]
        dim = dim + "C"
    y_pos, x_pos, c_pos = dim.index("Y"), dim.index("X"), dim.index("C")
    return np.transpose(img, (y_pos, x_pos, c_pos))


def _load_all_channels(
    spec: ImageSpec,
    region_px: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Load every channel from a pyramidal TIFF. Returns (H, W, C) float32.

    Counterpart to :func:`_load_channel` for multi-channel StarDist models
    (``model.config.n_channel_in > 1``, e.g. ``'2D_versatile_he'`` for H&E,
    which needs a true 3-channel RGB stack — selecting a single channel via
    :func:`_load_channel` and handing it a 2-D array raises
    ``ValueError: axes (YXC) must be of length 2`` inside StarDist).
    """
    import tifffile

    dim = spec.dimension_order
    has_c = "C" in dim

    if region_px is not None:
        row_min, row_max, col_min, col_max = region_px
        try:
            import zarr
            with tifffile.TiffFile(spec.path) as tf:
                store = tf.series[0].levels[0].aszarr()
            z = zarr.open(store, mode="r")
            slices: list = []
            kept_dim = ""
            for ax in dim:
                if ax == "T":
                    slices.append(0)
                elif ax == "C":
                    slices.append(slice(None))
                    kept_dim += ax
                elif ax == "Z":
                    if has_c:
                        slices.append(0)
                    else:
                        slices.append(slice(None))
                        kept_dim += ax
                elif ax == "Y":
                    slices.append(slice(row_min, row_max))
                    kept_dim += ax
                elif ax == "X":
                    slices.append(slice(col_min, col_max))
                    kept_dim += ax
                else:
                    slices.append(0)
            img = np.asarray(z[tuple(slices)])
            return _to_hwc(img, kept_dim).astype(np.float32)
        except Exception:
            # Fallback: load full level and slice.
            with tifffile.TiffFile(spec.path) as tf:
                img = tf.series[0].levels[0].asarray()
            img = _to_hwc(img, dim)
            return img[row_min:row_max, col_min:col_max].astype(np.float32)

    with tifffile.TiffFile(spec.path) as tf:
        img = tf.series[0].levels[0].asarray()

    return _to_hwc(img, dim).astype(np.float32)


def _masks_to_adata(
    masks: np.ndarray,
    spec: ImageSpec,
    offset_yx: tuple[int, int] = (0, 0),
    min_area_um2: float = 0.0,
) -> ad.AnnData:
    """Convert a 2-D integer label mask to AnnData with spatial obsm in µm.

    Per-instance work is restricted to each label's own bounding box (via
    ``scipy.ndimage.find_objects``), not a full-image ``masks == id`` scan
    per cell — O(H·W) total instead of O(n_cells · H·W).

    Each label's sub-mask is zero-padded by one pixel before contour
    extraction so instances touching the image border still produce a
    *closed* contour along the border (``find_contours`` otherwise returns
    an open contour that gets implicitly closed by a straight chord cutting
    across the cell).

    Parameters
    ----------
    offset_yx
        ``(row_offset, col_offset)`` in pixels to add to all centroids and
        boundary vertices before converting to µm.  Use when *masks* was
        extracted from a region of the full image rather than the full image.
    min_area_um2
        Drop instances whose polygon area (µm²) is below this. ``0.0``
        (default) keeps everything.

    Returns
    -------
    AnnData with ``obs['cell_area']`` (µm²) and ``obs['on_image_edge']``
    (``True`` for instances touching the segmented image/region border —
    typically cut off, so their area/boundary is partial).
    """
    from scipy import ndimage
    from skimage.measure import find_contours, regionprops_table

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = int(len(cell_ids))

    if n_cells == 0:
        log.warning("No cells found in segmentation mask.")
        adata = ad.AnnData()
        adata.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
        adata.obsm["cell_boundaries"] = ak.Array([])
        return adata

    log.info("Converting %d segmented cells to AnnData", n_cells)

    row_off, col_off = offset_yx
    H, W = masks.shape

    # Centroids in pixel (y, x), offset to full-image coordinates
    props = regionprops_table(masks, properties=["label", "centroid"])
    label_order = props["label"]  # same order as unique cell_ids
    centroids_yx_px = np.column_stack([
        props["centroid-0"] + row_off,
        props["centroid-1"] + col_off,
    ])

    # Per-label bounding-box slices, one linear pass over the image.
    objects = ndimage.find_objects(masks)

    # Per-cell contours in pixel (row, col) = (y, x), offset to full-image coordinates
    polys_yx_px: list[np.ndarray] = []
    on_edge = np.zeros(len(label_order), dtype=bool)
    for k, cell_id in enumerate(label_order):
        sl = objects[int(cell_id) - 1]  # find_objects indexes by label - 1
        if sl is None:
            polys_yx_px.append(np.zeros((0, 2), dtype=np.float32))
            continue
        rs, cs = sl
        on_edge[k] = rs.start == 0 or cs.start == 0 or rs.stop == H or cs.stop == W
        sub = np.pad((masks[sl] == cell_id).astype(np.uint8), 1)
        contours = find_contours(sub, level=0.5)
        if contours:
            # Select by area (shoelace) — avoids picking a large internal hole
            # as the "outer" boundary when len() would favor the wrong contour.
            poly = max(contours, key=lambda c: _shoelace_area(c[:, ::-1])).astype(np.float32)  # (N, 2) as (y, x)
            poly[:, 0] += rs.start - 1 + row_off  # -1 undoes the pad
            poly[:, 1] += cs.start - 1 + col_off
        else:
            poly = np.zeros((0, 2), dtype=np.float32)
        polys_yx_px.append(poly)

    # Pixel → µm conversion
    affine = spec.registration_affine
    pxum = spec.pixel_size_um

    def _px_to_um_xy(yx_px: np.ndarray) -> np.ndarray:
        """(N, 2) (y, x) pixels → (N, 2) (x, y) µm."""
        xy_px = yx_px[:, ::-1].astype(np.float64)
        if affine is not None:
            return _transform_coords(xy_px, affine).astype(np.float32)
        if pxum is not None:
            return (xy_px * pxum).astype(np.float32)
        return xy_px.astype(np.float32)

    centroids_um = _px_to_um_xy(centroids_yx_px)  # (n, 2) as (x, y)
    polys_um = [
        _px_to_um_xy(p) if len(p) >= 1 else np.zeros((0, 2), dtype=np.float32)
        for p in polys_yx_px
    ]

    # Compute cell areas via Shoelace formula
    areas = np.array([_shoelace_area(p) for p in polys_um], dtype=np.float32)

    import pandas as pd
    obs_names = np.array([str(int(cid)) for cid in label_order])

    if min_area_um2 > 0.0:
        keep = areas >= min_area_um2
        n_dropped = int((~keep).sum())
        if n_dropped:
            log.info(
                "Dropping %d segmented instances below min_area_um2=%.2f µm²",
                n_dropped, min_area_um2,
            )
        centroids_um = centroids_um[keep]
        polys_um = [p for p, k in zip(polys_um, keep) if k]
        areas = areas[keep]
        on_edge = on_edge[keep]
        obs_names = obs_names[keep]

    adata = ad.AnnData(
        obs=pd.DataFrame(
            {"cell_area": areas, "on_image_edge": on_edge}, index=obs_names.tolist()
        )
    )
    adata.obsm["spatial"] = centroids_um
    adata.obsm["cell_boundaries"] = ak.Array(polys_um)

    return adata


def _shoelace_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


# ── Public functions ──────────────────────────────────────────────────────────

def segment_cellpose(
    source: "ImageSpec | ad.AnnData | object",
    *,
    channel: int | str = 0,
    pretrained_model: str = "cpsam_v2",
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    normalize: bool = True,
    library_id: str | None = None,
    image_key: str = "full",
    gpu: bool = False,
    region_um: tuple[float, float, float, float] | None = None,
    min_area_um2: float = 0.0,
) -> ad.AnnData:
    """Segment cells in a fluorescence image channel using Cellpose.

    Requires the optional ``cellpose`` package (``pip install cellpose``).
    Runs on CPU by default; pass ``gpu=True`` if a CUDA device is available.

    Parameters
    ----------
    source
        Image source: an :class:`~astrolabe._utils.ImageSpec`, an
        :class:`~anndata.AnnData`, or a :class:`~mudata.MuData`.  When an
        AnnData / MuData is passed, the image is taken from
        ``uns['spatial'][library_id]['images'][image_key]``.
    channel
        Channel index (int) or channel name (str) to run Cellpose on.
        Use the nuclear/DAPI channel for nuclear segmentation.
    pretrained_model
        Cellpose model to use.  Default ``'cpsam_v2'`` (the Cellpose-SAM
        generalist model); classic choices like ``'cyto2'`` (whole-cell) or
        ``'nuclei'`` (nuclear) also work with older Cellpose versions.
    diameter
        Expected cell diameter in pixels.  ``None`` triggers Cellpose's
        automatic diameter estimation.
    flow_threshold
        Maximum allowed error of the flows for each mask.
    cellprob_threshold
        Threshold on the cell probability map.  Lower values include more
        dim or small cells.
    normalize
        Percentile-normalize the channel image before passing to Cellpose.
    library_id
        Library key in ``uns['spatial']``.  Required when *source* contains
        more than one library.
    image_key
        Image key within the library's ``images`` dict (default ``'full'``).
    gpu
        Use GPU acceleration.  Requires a CUDA-enabled environment.
    region_um
        Optional ``(xmin, ymin, xmax, ymax)`` bounding box in µm.  Only the
        portion of the morphology image that overlaps this region is loaded and
        segmented.  Useful for avoiding loading a multi-GB full-tissue image
        when you only need cells in a cropped sub-region.
    min_area_um2
        Drop segmented instances smaller than this polygon area (µm²) —
        debris / single-pixel speckles.  ``0.0`` (default) keeps everything.

    Returns
    -------
    AnnData
        New AnnData with:

        - ``obsm['spatial']`` — cell centroids in µm (x, y).
        - ``obsm['cell_boundaries']`` — awkward array of polygon vertices in µm.
        - ``obs['cell_area']`` — cell area in µm².
        - ``obs['on_image_edge']`` — ``True`` for cells touching the
          segmented image/region border (typically cut off).
        - no ``X`` / variables yet — run :func:`~astrolabe.seg.assign_transcripts`
          to populate expression.

    Examples
    --------
    >>> new_cells = ab.seg.segment_cellpose(mdata, channel='DAPI')
    >>> transcripts = ab.io.read_xenium_transcripts("Xenium_run/")
    >>> ab.seg.assign_transcripts(new_cells, transcripts)
    """
    try:
        from cellpose import models as cp_models
    except ImportError as e:
        raise ImportError(
            "cellpose is required for segment_cellpose. "
            "Install it with: pip install cellpose"
        ) from e

    spec, lib_meta = _resolve_spec(source, library_id, image_key)

    region_px: tuple[int, int, int, int] | None = None
    offset_yx: tuple[int, int] = (0, 0)
    if region_um is not None:
        region_px = _um_to_px(*region_um, spec)
        offset_yx = (region_px[0], region_px[2])  # (row_min, col_min)
        log.info(
            "segment_cellpose: restricting to region_um=%s → pixel bbox rows=%d:%d cols=%d:%d",
            region_um, region_px[0], region_px[1], region_px[2], region_px[3],
        )

    log.info("segment_cellpose: loading channel %r from %s", channel, spec.path)
    img = _load_channel(spec, channel, region_px=region_px)

    model = cp_models.CellposeModel(pretrained_model=pretrained_model, gpu=gpu)
    log.info(
        "segment_cellpose: running %s on image %s (diameter=%s)",
        pretrained_model, img.shape, diameter,
    )
    masks, _, _ = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize=normalize,
    )

    n_found = int(masks.max())
    log.info("segment_cellpose: found %d cells", n_found)
    return _masks_to_adata(masks, spec, offset_yx=offset_yx, min_area_um2=min_area_um2)


def segment_stardist(
    source: "ImageSpec | ad.AnnData | object",
    *,
    channel: int | str = 0,
    model_name: str = "2D_versatile_fluo",
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
    normalize: bool = True,
    norm_pct_low: float = 1.0,
    norm_pct_high: float = 99.8,
    library_id: str | None = None,
    image_key: str = "full",
    region_um: tuple[float, float, float, float] | None = None,
) -> ad.AnnData:
    """Segment nuclei/cells in a fluorescence image channel using StarDist.

    Requires the optional ``stardist`` package (``pip install stardist``).

    Parameters
    ----------
    source
        Image source: an :class:`~astrolabe._utils.ImageSpec`, an
        :class:`~anndata.AnnData`, or a :class:`~mudata.MuData`.
    channel
        Channel index (int) or channel name (str) to segment.
    model_name
        Pre-trained StarDist model to use.  Common choices:

        - ``'2D_versatile_fluo'`` — fluorescence nuclear staining (DAPI/Hoechst).
        - ``'2D_versatile_he'`` — H&E staining.
        - ``'2D_dsb2018_paper'`` — DSB 2018 challenge model.
    prob_thresh
        Instance probability threshold.  ``None`` uses the model's default.
    nms_thresh
        Non-maximum suppression overlap threshold.  ``None`` uses the model's
        default.
    normalize
        Percentile-normalize the channel image before prediction.
    norm_pct_low, norm_pct_high
        Percentile bounds for normalization when *normalize* is ``True``.
    library_id
        Library key in ``uns['spatial']``.  Required when *source* contains
        more than one library.
    image_key
        Image key within the library's ``images`` dict (default ``'full'``).
    region_um
        Optional ``(xmin, ymin, xmax, ymax)`` bounding box in µm.  Only the
        portion of the image overlapping this region is loaded and segmented.
        Useful for avoiding loading a multi-GB full-tissue image when only a
        cropped sub-region is needed.

    Returns
    -------
    AnnData
        New AnnData with:

        - ``obsm['spatial']`` — cell centroids in µm (x, y).
        - ``obsm['cell_boundaries']`` — awkward array of polygon vertices in µm.
        - ``obs['cell_area']`` — cell area in µm².
        - no ``X`` / variables yet — run :func:`~astrolabe.seg.assign_transcripts`
          to populate expression.

    Examples
    --------
    >>> new_cells = ab.seg.segment_stardist(mdata, channel='DAPI')
    >>> ab.seg.assign_transcripts(new_cells, transcripts)
    """
    try:
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize as csbdeep_normalize
    except (ImportError, RuntimeError) as e:
        raise ImportError(
            "stardist, csbdeep, and TensorFlow are required for segment_stardist. "
            "Install them with: pip install stardist tensorflow"
        ) from e

    spec, lib_meta = _resolve_spec(source, library_id, image_key)

    region_px: tuple[int, int, int, int] | None = None
    offset_yx: tuple[int, int] = (0, 0)
    if region_um is not None:
        region_px = _um_to_px(*region_um, spec)
        offset_yx = (region_px[0], region_px[2])  # (row_min, col_min)
        log.info(
            "segment_stardist: restricting to region_um=%s → pixel bbox rows=%d:%d cols=%d:%d",
            region_um, region_px[0], region_px[1], region_px[2], region_px[3],
        )

    model = StarDist2D.from_pretrained(model_name)

    # Multi-channel models (e.g. '2D_versatile_he', trained on 3-channel RGB
    # H&E) need the full channel stack -- axes='YXC' with n_channel_in==3 --
    # not a single selected channel. Loading just one channel and handing the
    # model a 2-D array raises "axes (YXC) must be of length 2" inside
    # StarDist. Single-channel models (e.g. '2D_versatile_fluo') keep the
    # existing behavior: load exactly the requested `channel`.
    n_channel_in = getattr(model.config, "n_channel_in", 1)
    if n_channel_in > 1:
        log.info(
            "segment_stardist: model expects %d channels; loading all channels from %s",
            n_channel_in, spec.path,
        )
        img = _load_all_channels(spec, region_px=region_px)
        norm_axis = (0, 1)  # per-channel percentiles over (Y, X), not pooled across channels
    else:
        log.info("segment_stardist: loading channel %r from %s", channel, spec.path)
        img = _load_channel(spec, channel, region_px=region_px)
        norm_axis = None

    if normalize:
        img = csbdeep_normalize(img, norm_pct_low, norm_pct_high, axis=norm_axis)

    predict_kwargs: dict = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    log.info("segment_stardist: running %s on image %s", model_name, img.shape)
    labels, details = model.predict_instances(img, **predict_kwargs)

    n_found = int(labels.max())
    log.info("segment_stardist: found %d cells", n_found)

    if n_found == 0:
        log.warning("No cells found in segmentation mask.")
        adata = ad.AnnData()
        adata.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
        adata.obsm["cell_boundaries"] = ak.Array([])
        return adata

    # StarDist returns polygon vertices directly in details['coord'], but as
    # (n_cells, 2, n_rays) -- axis 1 is the (y, x) pair, axis 2 is the ray
    # vertices (confirmed against details['points']: coord.mean(axis=2)
    # matches the reported cell centers). Transpose to (n_cells, n_verts, 2)
    # so downstream code can iterate per-cell (n_verts, 2) vertex arrays.
    # Use these directly instead of running find_contours on the label mask.
    coord_yx = np.transpose(details["coord"], (0, 2, 1))  # (n_cells, n_verts, 2)

    # StarDist's own reported instance centers, (n_cells, 2) as (y, x) —
    # preferred over averaging the ray vertices, which biases the centroid
    # toward whichever directions the rays extend furthest.
    points_yx = np.asarray(details.get("points"), dtype=np.float64) if "points" in details else None

    # Apply region offset so coordinates are in full-image pixel space.
    if offset_yx != (0, 0):
        row_off, col_off = offset_yx
        coord_yx = coord_yx.copy()
        coord_yx[:, :, 0] += row_off  # y (row)
        coord_yx[:, :, 1] += col_off  # x (col)
        if points_yx is not None:
            points_yx = points_yx + np.array([row_off, col_off], dtype=np.float64)

    affine = spec.registration_affine
    pxum = spec.pixel_size_um

    def _px_to_um_xy(yx_px: np.ndarray) -> np.ndarray:
        xy_px = yx_px[:, ::-1].astype(np.float64)
        if affine is not None:
            return _transform_coords(xy_px, affine).astype(np.float32)
        if pxum is not None:
            return (xy_px * pxum).astype(np.float32)
        return xy_px.astype(np.float32)

    polys_um = [_px_to_um_xy(np.asarray(verts)) for verts in coord_yx]

    if points_yx is not None and len(points_yx) == len(polys_um):
        centroids_um = _px_to_um_xy(points_yx).astype(np.float32)
    else:
        # Fallback: mean of ray vertices.
        centroids_um = np.stack(
            [p.mean(axis=0) if len(p) else np.zeros(2, np.float32) for p in polys_um]
        ).astype(np.float32)

    areas = np.array([_shoelace_area(p) for p in polys_um], dtype=np.float32)

    import pandas as pd
    obs_names = [str(i + 1) for i in range(len(polys_um))]
    adata = ad.AnnData(obs=pd.DataFrame({"cell_area": areas}, index=obs_names))
    adata.obsm["spatial"] = centroids_um
    adata.obsm["cell_boundaries"] = ak.Array(polys_um)

    return adata