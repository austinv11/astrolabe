"""Region membership and distances in the dataset's micron frame."""
from __future__ import annotations
import anndata as ad
import numpy as np
import pandas as pd
import shapely
from astrolabe._utils import with_adata, from_kwarg


def _regions(adata, regions, region_key, library_key):
    if isinstance(regions, str):
        regions = adata.uns[regions]
    if region_key not in regions or not hasattr(regions, "geometry"):
        raise ValueError("regions must be a GeoDataFrame with a region ID column")
    if regions[region_key].isna().any() or regions.geometry.isna().any():
        raise ValueError("region IDs and geometry must be present")
    if not regions.geometry.is_valid.all() or regions.geometry.is_empty.any():
        raise ValueError("regions must contain nonempty valid polygons")
    if not regions.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("regions must be polygons")
    if library_key in adata.obs and adata.obs[library_key].nunique() > 1 and library_key not in regions:
        raise ValueError("multiple libraries require per-region library labels")
    dissolve_keys = [region_key] + ([library_key] if library_key in regions else [])
    return regions.dissolve(by=dissolve_keys, as_index=False)


@with_adata(mirror_uns=from_kwarg("key_added"))
def assign_regions(adata: ad.AnnData, regions="annotations", *, region_key="region_id",
                   library_key="library_id", spatial_key="spatial", overlaps="error", key_added="region"):
    """Assign centroids covered by polygons, including polygon boundaries.

    Holes exclude cells. ``overlaps='membership'`` returns every membership;
    default overlaps raise before writing labels. Outside cells are missing.
    """
    if overlaps not in ("error", "membership"):
        raise ValueError("overlaps must be error or membership")
    regions = _regions(adata, regions, region_key, library_key)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite")
    points = shapely.points(coords)
    rows = []
    labels = np.full(adata.n_obs, None, dtype=object)
    counts = np.zeros(adata.n_obs, dtype=int)
    for _, row in regions.iterrows():
        mask = np.asarray(shapely.covers(row.geometry, points))
        if library_key in regions:
            if library_key not in adata.obs:
                raise ValueError("regions with library labels require matching cell labels")
            mask &= adata.obs[library_key].to_numpy() == row[library_key]
        for idx in np.flatnonzero(mask):
            rows.append((adata.obs_names[idx], row[region_key]))
        counts[mask] += 1
        labels[mask] = str(row[region_key])
    if overlaps == "error" and (counts > 1).any():
        raise ValueError("overlapping regions cover the same cells; use overlaps='membership'")
    membership = pd.DataFrame(rows, columns=["obs_name", region_key])
    if overlaps == "error":
        adata.obs[key_added] = pd.Categorical(labels)
    adata.uns[key_added] = {"membership": membership, "region_key": region_key,
                            "spatial_key": spatial_key, "overlaps": overlaps}
    return membership


@with_adata
def distance_to_region(adata: ad.AnnData, regions="annotations", *, region,
                       region_key="region_id", library_key="library_id", spatial_key="spatial",
                       signed=True, key_added="distance_to_region"):
    """Distance in microns to one region's boundary; negative inside.

    Disconnected components with the same ID are unioned per library.
    Cells in libraries without the requested region have NaN distance.
    """
    regions = _regions(adata, regions, region_key, library_key)
    chosen = regions[regions[region_key] == region]
    if chosen.empty:
        raise KeyError(region)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("spatial coordinates must be finite")
    points = shapely.points(coords)
    values = np.full(adata.n_obs, np.nan)
    groups = chosen.groupby(library_key, observed=True) if library_key in chosen else [(None, chosen)]
    for lib, frame in groups:
        mask = np.ones(adata.n_obs, dtype=bool) if lib is None else adata.obs[library_key].to_numpy() == lib
        geometry = shapely.union_all(frame.geometry.to_numpy())
        dist = np.asarray(shapely.distance(points[mask], geometry.boundary if signed else geometry))
        if signed:
            dist = np.where(shapely.covers(geometry, points[mask]), -dist, dist)
        values[mask] = dist
    adata.obs[key_added] = values
    return values


@with_adata(mirror_uns=from_kwarg("key_added"))
def region_summary(adata: ad.AnnData, *, region_by="region", sample_key, groupby=None,
                   genes=None, layer=None, analysis_area=None, key_added="region_summary"):
    """Counts, composition and expression per biological sample and region.

    Density is reported only with an explicit mapping ``(sample, region) ->
    observed analysis area in square microns``; polygon area alone may include
    unobserved tissue and is never inferred as the sampling window.
    """
    columns = [sample_key, region_by] + ([] if groupby is None else [groupby])
    if adata.obs[sample_key].isna().any():
        raise ValueError("sample labels must be present")
    rows = []
    for labels, indices in adata.obs.groupby(columns, observed=True, dropna=True).indices.items():
        labels = labels if isinstance(labels, tuple) else (labels,)
        row = dict(zip(columns, labels)); row["n_cells"] = len(indices)
        if genes:
            mat = adata.layers[layer] if layer is not None else adata.X
            for gene in genes:
                row["mean_"+gene] = float(mat[indices, adata.var_names.get_loc(gene)].mean())
        if analysis_area is not None:
            area = analysis_area.get(tuple(labels[:2]), np.nan)
            if not np.isfinite(area) or area <= 0:
                raise ValueError("every sample-region requires a positive observed analysis area")
            row["density_per_um2"] = len(indices) / area
        rows.append(row)
    result = pd.DataFrame(rows)
    if groupby and len(result):
        totals = result.groupby([sample_key, region_by], observed=True).n_cells.transform("sum")
        result["fraction"] = result.n_cells / totals
    adata.uns[key_added] = result
    return result
