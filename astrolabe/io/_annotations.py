"""
Reader for GeoJSON annotation files (ROIs, pathologist regions, etc.).

Platform-agnostic: unlike the other readers in this package, this is not tied
to any particular spatial platform. It simply parses a ``.geojson`` file into
a :class:`geopandas.GeoDataFrame` and, optionally, stores it on an existing
MuData/AnnData's ``uns``. Geometries are assumed to already be in the same
micron coordinate frame as the target dataset — no registration is performed
here (contrast with H&E image loading, which frequently needs an affine
alignment via ``read_he(affine=...)``).

Canonical call
--------------
>>> import astrolabe as ab
>>> mdata = ab.io.read_xenium("Xenium_run/")
>>> gdf = ab.io.read_annotations("Xenium_run/annotation.geojson", mdata=mdata)
>>> mdata.uns['annotations'] is gdf
True
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import geopandas as gpd
import mudata as md

__all__ = ["read_annotations"]

log = logging.getLogger(__name__)


def read_annotations(
    path: str | Path,
    *,
    mdata: "md.MuData | ad.AnnData | None" = None,
    key_added: str = "annotations",
) -> gpd.GeoDataFrame:
    """Read a GeoJSON annotation file (ROIs, pathologist regions, etc.).

    Works for any dataset (Xenium, MERSCOPE, CosMx, or a standalone
    annotation file) as long as the geometries are already expressed in the
    same micron coordinate frame as *mdata*/*adata* — this function performs
    no registration itself.

    Parameters
    ----------
    path
        Path directly to a ``.geojson`` (or ``.json``) annotation file.
    mdata
        If given (a :class:`~mudata.MuData` or a bare :class:`~anndata.AnnData`),
        the returned GeoDataFrame is also stored at ``uns[key_added]`` in place.
    key_added
        Key to store the GeoDataFrame under in ``uns`` when *mdata* is given.
        Default ``'annotations'``.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per annotated region/feature, with a ``geometry`` column
        (shapely polygons in µm) plus whatever properties were present in
        the source GeoJSON.

    Examples
    --------
    >>> gdf = ab.io.read_annotations("run/annotation.geojson")
    >>> ab.io.read_annotations("run/annotation.geojson", mdata=mdata)
    >>> mdata.uns['annotations']
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    log.info("Reading annotations from %s", path)
    gdf = gpd.read_file(path)
    log.info("Loaded %d annotation region(s)", len(gdf))

    if mdata is not None:
        mdata.uns[key_added] = gdf

    return gdf
