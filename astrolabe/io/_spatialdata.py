"""Explicit optional SpatialData interchange. No SpatialData import at module load."""
from __future__ import annotations


def _backend():
    try:
        import spatialdata
    except ImportError as exc:
        raise ImportError("Install astrolabe[spatialdata] to use SpatialData conversion") from exc
    return spatialdata


def to_spatialdata(data, *, modality=None, spatial_key="spatial", boundaries_key="cell_boundaries",
                   library_key="auto", core_key="auto", include_transcripts=True, centroid_radius=None):
    """Export counts, cell shapes, tissue masks and transcripts in micron frames.

    Each library/core has an explicit coordinate system. ImageSpec descriptors
    remain lazy file references in interchange metadata, including their affine
    transforms; pixels are not copied. Original IDs and biological metadata are
    retained. This function explicitly loads the optional SpatialData package.
    Without cell polygons, supply a positive ``centroid_radius`` in microns for
    SpatialData's circle representation. This display radius is not an inferred
    cell boundary or measured cell size.
    """
    sd = _backend()
    from spatialdata.models import ShapesModel, PointsModel, TableModel
    from spatialdata.transformations import Identity
    import copy
    import hashlib
    from pathlib import Path
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely
    from astrolabe._utils import _resolve_adata
    from astrolabe._spatial_units import _units, _support
    from astrolabe.io._io import _replace_specs
    from astrolabe._shapes import _build_polygons
    adata = _resolve_adata(data, modality)
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("SpatialData interchange requires unique cell and feature IDs")
    reserved = {"_astrolabe_region", "_astrolabe_instance"}
    if reserved & set(adata.obs):
        raise ValueError("Reserved interchange columns already exist")
    units, columns = _units(adata, library_key, core_key)
    coordinates = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("SpatialData conversion requires finite 2-D micron coordinates")
    if boundaries_key in adata.obsm:
        polygons, bad, _ = _build_polygons(adata.obsm[boundaries_key])
        if np.any(bad):
            raise ValueError("Cell shapes contain invalid/empty geometries")
        geometry = np.asarray(polygons, dtype=object)
        geometry_kind = "polygons"
    else:
        if centroid_radius is None or not np.isfinite(centroid_radius) or centroid_radius <= 0:
            raise ValueError("Without cell polygons, supply a positive centroid_radius in microns for display circles")
        geometry = shapely.points(coordinates)
        geometry_kind = "centroids"
    table = adata.copy()
    table.obsp.clear()
    support_key = adata.uns.get("spatial_units", {}).get("support_key")
    table.uns = {key: copy.deepcopy(adata.uns[key]) for key in
                 ("measured_features", "measured_features_library_key", support_key, "spatial_units") if key in adata.uns}
    table.obsm.clear()
    table.obsm[spatial_key] = coordinates.copy()
    region_labels = np.empty(adata.n_obs, dtype=object)
    shapes, points, frames = {}, {}, {}
    transcripts = adata.uns.get("transcripts") if include_transcripts else None
    if transcripts is not None:
        transcripts = transcripts.to_pandas() if hasattr(transcripts, "to_pandas") else transcripts.copy()
        if any(column not in transcripts for column in columns) and len(units) > 1:
            raise ValueError("Transcript tables require explicit library/core labels for multiple spatial units")
        membership = np.zeros(len(transcripts), dtype=np.int32)
        for labels, _ in units:
            keep = np.ones(len(transcripts), dtype=bool)
            for column, label in zip(columns, labels):
                if column in transcripts:
                    keep &= transcripts[column].eq(label).fillna(False).to_numpy(dtype=bool)
            membership += keep
        if not np.all(membership == 1):
            raise ValueError("Every transcript must belong to exactly one known spatial unit")
    for labels, indices in units:
        unit_id = hashlib.sha256(repr(tuple(map(str, labels))).encode()).hexdigest()[:16]
        region = "cells_" + unit_id
        coordinate_system = "microns_" + unit_id
        frames[region] = {"coordinate_system": coordinate_system, "unit_columns": columns, "unit_labels": list(map(str, labels))}
        frame = gpd.GeoDataFrame(index=adata.obs_names[indices].copy(), geometry=geometry[indices])
        frame["centroid_x_um"] = coordinates[indices, 0]
        frame["centroid_y_um"] = coordinates[indices, 1]
        if geometry_kind == "centroids":
            frame["radius"] = float(centroid_radius)
        shapes[region] = ShapesModel.parse(frame, transformations={coordinate_system: Identity()})
        region_labels[indices] = region
        support_key = adata.uns.get("spatial_units", {}).get("support_key")
        if support_key is not None:
            support = gpd.GeoDataFrame(geometry=[_support(adata, labels, columns, support_key)])
            shapes["tissue_" + unit_id] = ShapesModel.parse(support, transformations={coordinate_system: Identity()})
        if transcripts is not None:
            keep = np.ones(len(transcripts), dtype=bool)
            for column, label in zip(columns, labels):
                if column in transcripts:
                    keep &= transcripts[column].to_numpy() == label
            selected = transcripts.loc[keep].copy()
            if len(selected):
                axes = {"x": "x_location", "y": "y_location"}
                if "z_location" in selected:
                    axes["z"] = "z_location"
                points["transcripts_" + unit_id] = PointsModel.parse(selected, coordinates=axes,
                    feature_key="feature_name" if "feature_name" in selected else None,
                    transformations={coordinate_system: Identity()})
    table.obs["_astrolabe_region"] = pd.Categorical(region_labels)
    table.obs["_astrolabe_instance"] = adata.obs_names.to_numpy()
    table.uns["astrolabe_interchange"] = {"schema_version": 1, "units": "micron", "frames": frames,
        "spatial_key": spatial_key, "boundaries_key": boundaries_key, "geometry_kind": geometry_kind,
        "spatial_images": _replace_specs(copy.deepcopy(adata.uns.get("spatial", {})), Path.cwd()),
        "image_base_directory": str(Path.cwd()), "transcript_columns": [] if transcripts is None else list(transcripts.columns)}
    table = TableModel.parse(table, region=list(frames),
                             region_key="_astrolabe_region", instance_key="_astrolabe_instance")
    return sd.SpatialData(shapes=shapes, points=points, tables={"table": table})


def from_spatialdata(sdata, *, table_key="table", coordinate_system=None, units=None):
    """Import a linked SpatialData cell table and geometries into AnnData.

    Astrolabe exports retain micron units, core hierarchy and lazy image
    references. Foreign tables require explicit ``units='micron'`` and a target
    coordinate system; linked cells must be polygonal or point shapes. Native
    raster images require a separate image reader; they are not silently copied.
    """
    sd = _backend()
    from spatialdata.transformations import get_transformation
    import awkward as ak
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import shapely
    from pathlib import Path
    from astrolabe.io._io import _restore_specs, _iter_image_specs
    if not isinstance(sdata, sd.SpatialData):
        raise TypeError("Expected a SpatialData object")
    table = sdata.tables[table_key].copy()
    contract = table.uns.pop("astrolabe_interchange", {})
    if (units or contract.get("units")) != "micron":
        raise ValueError("Explicit micron units are required; convert pixel coordinates before import")
    if sdata.images and not contract.get("spatial_images"):
        raise ValueError("Native raster images need an explicit Astrolabe image-reader mapping")
    linkage = table.uns.get("spatialdata_attrs", {})
    region_key, instance_key = linkage.get("region_key"), linkage.get("instance_key")
    if region_key not in table.obs or instance_key not in table.obs:
        raise ValueError("Cell table must link to shapes through region and instance columns")
    coordinates = np.full((table.n_obs, 2), np.nan)
    boundaries = [None]*table.n_obs
    has_polygons = False
    support_key = table.uns.get("spatial_units", {}).get("support_key")
    support_rows = []
    for region in pd.unique(table.obs[region_key]):
        if region not in sdata.shapes:
            raise ValueError("Linked cells must be SpatialData shapes")
        target = coordinate_system or contract.get("frames", {}).get(region, {}).get("coordinate_system")
        if target is None:
            raise ValueError("Select an explicit target coordinate system")
        transformed = sd.transform(sdata.shapes[region], to_coordinate_system=target)
        affine = get_transformation(sdata.shapes[region], to_coordinate_system=target).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        if contract.get("spatial_images") and not np.allclose(affine, np.eye(3)):
            raise ValueError("Transformed SpatialData cells require an explicit image alignment; lazy external images cannot be realigned implicitly")
        selected = np.flatnonzero(table.obs[region_key].to_numpy() == region)
        ids = table.obs.iloc[selected][instance_key]
        if not transformed.index.is_unique or not ids.isin(transformed.index).all():
            raise ValueError("Cell instance IDs are missing or nonunique in linked shapes")
        geometry = transformed.loc[ids].geometry.to_numpy()
        if support_key is not None:
            tissue_name = "tissue_" + str(region).removeprefix("cells_")
            if tissue_name not in sdata.shapes:
                raise ValueError("The linked tissue observation window is missing from SpatialData shapes")
            support = shapely.union_all(sd.transform(sdata.shapes[tissue_name], to_coordinate_system=target).geometry.to_numpy())
            if support.is_empty or not support.is_valid or support.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("Invalid transformed tissue observation window")
            columns = contract.get("frames", {}).get(region, {}).get("unit_columns", [])
            support_rows.append({**{column: table.obs.iloc[selected[0]][column] for column in columns},
                                 "wkb": shapely.to_wkb(support, hex=True), "area_um2": support.area})
        if not shapely.is_valid(geometry).all() or shapely.is_empty(geometry).any():
            raise ValueError("Invalid linked cell geometry")
        # Transform the preserved measured centroids, rather than replacing
        # them by polygon centroids when a reconstruction changed shape.
        source = sdata.shapes[region].loc[ids]
        if {"centroid_x_um", "centroid_y_um"}.issubset(source):
            affine = get_transformation(sdata.shapes[region], to_coordinate_system=target).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
            xy = source[["centroid_x_um", "centroid_y_um"]].to_numpy()
            coordinates[selected] = np.column_stack([xy, np.ones(len(xy))]).dot(affine.T)[:, :2]
        else:
            coordinates[selected] = shapely.get_coordinates(shapely.centroid(geometry))
        for index, shape in zip(selected, geometry):
            if shape.geom_type == "Polygon" and not shape.interiors:
                boundaries[index] = np.asarray(shape.exterior.coords).tolist()
                has_polygons = True
            elif shape.geom_type != "Point":
                raise ValueError("Cell boundary arrays cannot represent holes/MultiPolygons; retain them as separate annotations")
    spatial_key = contract.get("spatial_key", "spatial")
    if support_key is not None:
        table.uns[support_key]["regions"] = pd.DataFrame(support_rows)
    table.obsm[spatial_key] = coordinates
    if has_polygons:
        if any(value is None for value in boundaries):
            raise ValueError("Mixed centroid and polygon cell geometry is unsupported")
        table.obsm[contract.get("boundaries_key", "cell_boundaries")] = ak.Array(boundaries)
    transcripts = []
    for name, point_frame in sdata.points.items():
        if not name.startswith("transcripts_"):
            continue
        region = "cells_" + name.removeprefix("transcripts_")
        target = coordinate_system or contract.get("frames", {}).get(region, {}).get("coordinate_system")
        if target is None:
            raise ValueError("Transcript coordinate system must be explicit")
        frame = sd.transform(point_frame, to_coordinate_system=target).compute()
        for axis in ("x", "y", "z"):
            if axis in frame:
                frame[axis + "_location"] = frame[axis]
        columns = contract.get("transcript_columns")
        transcripts.append(frame[list(columns)] if columns is not None and len(columns) else frame)
    if transcripts:
        table.uns["transcripts"] = pa.Table.from_pandas(pd.concat(transcripts, ignore_index=True), preserve_index=False)
    if contract.get("spatial_images"):
        table.uns["spatial"] = _restore_specs(contract["spatial_images"])
        for spec in _iter_image_specs(table.uns["spatial"]):
            if not Path(spec.path).is_absolute():
                spec.path = str(Path(contract["image_base_directory"]) / spec.path)
    table.uns.pop("spatialdata_attrs", None)
    for key in ("_astrolabe_region", "_astrolabe_instance"):
        if key in table.obs:
            del table.obs[key]
    return table
