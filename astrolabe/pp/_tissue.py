"""Observed tissue masks and TMA spatial-unit contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shapely

from astrolabe._spatial_units import _units
from astrolabe._utils import with_adata


@with_adata
def set_tissue_masks(adata, regions, *, library_key="auto", core_key="auto",
                     spatial_key="spatial", key_added="tissue_support"):
    """Register observed Polygon/MultiPolygon masks in micron coordinates.

    Supply a GeoDataFrame with the same library/core columns as ``obs``.
    Multiple rows for one core are unioned; holes and fragments are retained.
    Every cell must lie in its own unit's mask. Core labels identify physical
    pieces, not donors; retain specimen/donor metadata separately in ``obs``.
    """
    from astrolabe._spatial_units import _unit_keys
    library_key, core_key = _unit_keys(adata, library_key, core_key)
    units, columns = _units(adata, library_key, core_key)
    if not hasattr(regions, "geometry") or any(key not in regions for key in columns):
        raise ValueError("regions requires geometry and matching library/core columns")
    if columns and regions[columns].isna().any().any():
        raise ValueError("Tissue support unit labels must be complete")
    geometry = regions.geometry
    if (geometry.isna().any() or geometry.is_empty.any() or not geometry.is_valid.all()
            or not geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
        raise ValueError("Observed tissue requires nonempty valid polygons")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Expected finite two-dimensional micron coordinates")
    rows = []
    for labels, indices in units:
        keep = np.ones(len(regions), dtype=bool)
        for key, label in zip(columns, labels):
            keep &= regions[key].to_numpy() == label
        union = shapely.union_all(geometry.to_numpy()[keep])
        if union.is_empty or not shapely.covers(union, shapely.points(coords[indices])).all():
            raise ValueError(f"Cells outside observed tissue support for {labels}")
        rows.append(dict(zip(columns, labels), wkb=shapely.to_wkb(union, hex=True), area_um2=float(union.area)))
    adata.uns[key_added] = {"regions": pd.DataFrame(rows), "units": "micron", "spatial_key": spatial_key}
    adata.uns["spatial_units"] = {"library_key": library_key, "core_key": core_key, "support_key": key_added}
