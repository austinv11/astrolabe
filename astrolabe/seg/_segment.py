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


def _load_channel(spec: ImageSpec, channel: int | str) -> np.ndarray:
    """Load one channel from a pyramidal TIFF at full resolution. Returns (H, W) float32."""
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

    with tifffile.TiffFile(spec.path) as tf:
        img = tf.series[0].levels[0].asarray()

    dim = spec.dimension_order  # numpy axis order, e.g. 'CYX'

    for ax in ("T", "Z"):
        pos = dim.find(ax)
        if pos >= 0:
            img = img.take(0, axis=pos)
            dim = dim[:pos] + dim[pos + 1:]

    if "C" in dim:
        img = img.take(ch_idx, axis=dim.index("C"))

    return img.astype(np.float32)


def _masks_to_adata(
    masks: np.ndarray,
    spec: ImageSpec,
) -> ad.AnnData:
    """Convert a 2-D integer label mask to AnnData with spatial obsm in µm."""
    from skimage.measure import find_contours, regionprops_table

    cell_ids = np.unique(masks)
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = int(len(cell_ids))

    if n_cells == 0:
        log.warning("_masks_to_adata: no cells found in mask.")
        adata = ad.AnnData(n_obs=0)
        adata.obsm["spatial"] = np.zeros((0, 2), dtype=np.float32)
        adata.obsm["cell_boundaries"] = ak.Array([])
        return adata

    log.info("_masks_to_adata: converting %d cells", n_cells)

    # Centroids in pixel (y, x)
    props = regionprops_table(masks, properties=["label", "centroid"])
    label_order = props["label"]  # same order as unique cell_ids
    centroids_yx_px = np.column_stack([props["centroid-0"], props["centroid-1"]])

    # Per-cell contours in pixel (row, col) = (y, x)
    polys_yx_px: list[np.ndarray] = []
    for cell_id in label_order:
        cell_mask = (masks == cell_id).astype(np.uint8)
        contours = find_contours(cell_mask, level=0.5)
        if contours:
            poly = max(contours, key=len).astype(np.float32)  # (N, 2) as (y, x)
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

    adata = ad.AnnData(n_obs=n_cells)
    adata.obs_names = [str(int(cid)) for cid in label_order]
    adata.obs["cell_area"] = areas
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
    model_type: str = "cyto2",
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    normalize: bool = True,
    library_id: str | None = None,
    image_key: str = "full",
    gpu: bool = False,
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
        Use the nuclear/DAPI channel for ``model_type='nuclei'``.
    model_type
        Cellpose model to use.  Common choices: ``'cyto2'`` (whole-cell),
        ``'nuclei'`` (nuclear segmentation).
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
    >>> new_cells = ab.seg.segment_cellpose(mdata, channel='DAPI', model_type='nuclei')
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
    log.info("segment_cellpose: loading channel %r from %s", channel, spec.path)
    img = _load_channel(spec, channel)

    model = cp_models.CellposeModel(model_type=model_type, gpu=gpu)
    log.info(
        "segment_cellpose: running %s on image %s (diameter=%s)",
        model_type, img.shape, diameter,
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
    return _masks_to_adata(masks, spec)


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
    except ImportError as e:
        raise ImportError(
            "stardist and csbdeep are required for segment_stardist. "
            "Install them with: pip install stardist"
        ) from e

    spec, lib_meta = _resolve_spec(source, library_id, image_key)
    log.info("segment_stardist: loading channel %r from %s", channel, spec.path)
    img = _load_channel(spec, channel)

    if normalize:
        img = csbdeep_normalize(img, norm_pct_low, norm_pct_high)

    model = StarDist2D.from_pretrained(model_name)

    predict_kwargs: dict = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    log.info("segment_stardist: running %s on image %s", model_name, img.shape)
    labels, details = model.predict_instances(img, **predict_kwargs)

    n_found = int(labels.max())
    log.info("segment_stardist: found %d cells", n_found)

    # StarDist returns polygon vertices directly in details['coord']
    # as (n_cells, n_verts, 2) in (y, x) pixel order.
    # Use them instead of running find_contours on the label mask.
    coord_yx = details["coord"]  # (n_cells, n_verts, 2)

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

    # Centroids: mean of vertices
    centroids_um = np.stack(
        [p.mean(axis=0) if len(p) else np.zeros(2, np.float32) for p in polys_um]
    ).astype(np.float32)

    areas = np.array([_shoelace_area(p) for p in polys_um], dtype=np.float32)
    n_cells = len(polys_um)

    adata = ad.AnnData(n_obs=n_cells)
    adata.obs_names = [str(i + 1) for i in range(n_cells)]
    adata.obs["cell_area"] = areas
    adata.obsm["spatial"] = centroids_um
    adata.obsm["cell_boundaries"] = ak.Array(polys_um)

    return adata