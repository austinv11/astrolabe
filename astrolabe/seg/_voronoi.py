"""Voronoi-based cell re-segmentation."""

from __future__ import annotations

import logging

import anndata as ad
import awkward as ak
import numpy as np

from astrolabe._constants import UNS_XENIUM_SEGMENTATION_KEY
from astrolabe._utils import with_adata

__all__ = ["voronoi_resegment"]

log = logging.getLogger(__name__)


@with_adata
def voronoi_resegment(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    clip_to: "object | None" = None,
    clip_buffer_um: float = 50.0,
    key_added: str = "cell_boundaries",
    area_key: str | None = "cell_area",
    inplace: bool = True,
) -> "ad.AnnData | None":
    """Replace cell boundary polygons with Voronoi tessellation.

    Each cell is assigned the Voronoi region closest to its centroid.  The
    tessellation is clipped to the tissue boundary, computed automatically
    from the cell centroids via an alpha shape.

    This is useful when the original segmentation is unreliable (e.g. many
    cells without detected nuclei, irregular or overlapping boundaries), since
    Voronoi cells provide a complete, non-overlapping partition of the tissue.

    .. note::
        Voronoi boundaries are an approximation — they do not follow true cell
        membranes.  ``nucleus_boundaries`` are not modified.

    .. warning::
        Only the boundary polygons (and *area_key*) are replaced — ``X``
        still reflects the **original** segmentation's transcript-to-cell
        assignment. To rebuild expression counts against the new Voronoi
        boundaries, run :func:`~astrolabe.seg.assign_transcripts` afterward
        (it defaults to the object's own ``uns['transcripts']``, which this
        function's output inherits). A warning is logged as a reminder.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        For :class:`~mudata.MuData`: modality key.
    spatial_key
        ``obsm`` key for centroid coordinates (µm).
    clip_to
        A :class:`shapely.Geometry` defining the tissue boundary to clip
        Voronoi cells to.  ``None`` auto-computes an alpha shape from the
        centroids and buffers it by *clip_buffer_um* µm.
    clip_buffer_um
        µm buffer applied to the alpha shape when ``clip_to`` is ``None``.
        Prevents edge cells from being clipped too aggressively.
    key_added
        ``obsm`` key where new boundary polygons are written (overwrites
        existing ``'cell_boundaries'`` by default).
    area_key
        ``obs`` key updated with Voronoi polygon areas in µm².  ``None``
        skips the update.
    inplace
        Modify *data* in place and return ``None`` (default).  If ``False``,
        operate on a copy and return it.

    Returns
    -------
    None or AnnData

    Examples
    --------
    >>> ab.seg.voronoi_resegment(mdata)
    >>> # Verify new boundaries are present
    >>> mdata["xenium"].obsm["cell_boundaries"]
    """
    import geopandas as gpd
    from libpysal.cg import alpha_shape_auto, voronoi_frames
    from shapely.geometry import Point

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    if not inplace:
        adata = adata.copy()

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = centroids.shape[0]
    log.info("voronoi_resegment: %d cells", n_cells)

    # ------------------------------------------------------------------ #
    # Tissue boundary                                                     #
    # ------------------------------------------------------------------ #
    if clip_to is None:
        log.info("voronoi_resegment: computing alpha shape boundary")
        clip_geom = alpha_shape_auto(centroids)
        if clip_buffer_um > 0:
            clip_geom = clip_geom.buffer(clip_buffer_um)
    else:
        clip_geom = clip_to

    # ------------------------------------------------------------------ #
    # Voronoi tessellation                                                #
    # ------------------------------------------------------------------ #
    points_gs = gpd.GeoSeries([Point(xy) for xy in centroids])
    # as_gdf=False, return_input=False: newer libpysal defaults changed this
    # call to return a (geoms, ids) tuple instead of a single GeoDataFrame;
    # request the plain GeoSeries explicitly so the result is unambiguous
    # across libpysal versions.
    regions = voronoi_frames(points_gs, clip=clip_geom, as_gdf=False, return_input=False)

    # voronoi_frames returns polygons in the same order as input
    polys = list(regions.geometry) if hasattr(regions, "geometry") else list(regions)
    if len(polys) != n_cells:
        raise RuntimeError(
            f"voronoi_frames returned {len(polys)} polygons but expected {n_cells}. "
            "This is a libpysal version incompatibility; check your libpysal version."
        )

    # ------------------------------------------------------------------ #
    # Convert to awkward array (same format as original boundaries)      #
    # ------------------------------------------------------------------ #
    coords_list = []
    areas = np.empty(n_cells, dtype=np.float64)
    for i, poly in enumerate(polys):
        if poly is None or poly.is_empty:
            coords_list.append(np.zeros((0, 2), dtype=np.float32))
            areas[i] = 0.0
        else:
            ext = np.asarray(poly.exterior.coords, dtype=np.float32)
            coords_list.append(ext)
            areas[i] = poly.area

    adata.obsm[key_added] = ak.Array(coords_list)
    log.info("voronoi_resegment: wrote %s to obsm[%r]", key_added, key_added)

    if area_key is not None:
        adata.obs[area_key] = areas.astype(np.float32)
        log.info("voronoi_resegment: updated obs[%r] with Voronoi areas", area_key)

    # The boundaries no longer match whatever segmentation X (and any
    # Xenium-native transcript flags) were computed against.
    adata.uns[UNS_XENIUM_SEGMENTATION_KEY] = False
    if adata.X is not None:
        log.warning(
            "voronoi_resegment: boundary polygons were replaced but X still "
            "reflects the ORIGINAL segmentation's transcript-to-cell "
            "assignment. Run seg.assign_transcripts to rebuild counts "
            "against the new Voronoi boundaries."
        )

    log.info("voronoi_resegment: done")
    return None if inplace else adata
