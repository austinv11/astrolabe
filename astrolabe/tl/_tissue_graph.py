"""Graph construction within explicit tissue units and observed masks."""
from __future__ import annotations

import anndata as ad
import numpy as np
import scipy.sparse as sp
import shapely

from astrolabe._spatial_units import _support


def _partitioned_graph(adata, units, columns, *, library_key, core_key, tissue_support,
                       boundaries_key, contact_distance, **options):
    from astrolabe.tl._graph import spatial_neighbors
    spatial_key, key = options["spatial_key"], options["key_added"]
    coords = np.asarray(adata.obsm[spatial_key])
    row_parts, col_parts, distance_parts = [], [], []
    contact = options["coord_type"] == "contact"
    if contact:
        from astrolabe._shapes import _build_polygons
        polygons, bad, _ = _build_polygons(adata.obsm[boundaries_key])
        if np.any(bad):
            raise ValueError("Contact graphs require a valid polygon for every cell")
        polygons = np.asarray(polygons, dtype=object)
    metadata = {}
    for labels, idx in units:
        if contact:
            shapes = polygons[idx]
            tree = shapely.STRtree(shapes)
            rows, cols = tree.query(shapes, predicate="dwithin", distance=contact_distance)
            if not options["set_diag"]:
                keep = rows != cols
                rows, cols = rows[keep], cols[keep]
            distances = np.linalg.norm(coords[idx[rows]] - coords[idx[cols]], axis=1)
        else:
            # A graph needs only coordinates: avoid copying counts/images per core.
            temporary = ad.AnnData(obs=adata.obs.iloc[idx][[]].copy())
            temporary.obsm[spatial_key] = coords[idx]
            spatial_neighbors(temporary, **options, library_key=None, core_key=None, tissue_support=None)
            graph = temporary.obsp[key + "_connectivities"].tocoo()
            rows, cols = graph.row, graph.col
            distances = np.asarray(temporary.obsp[key + "_distances"][rows, cols]).ravel()
            metadata = temporary.uns[key]["params"]
        if tissue_support is not None:
            mask = _support(adata, labels, columns, tissue_support)
            if not shapely.covers(mask, shapely.points(coords[idx])).all():
                raise ValueError(f"Cells outside observed tissue mask for {labels}")
            keep = np.zeros(len(rows), dtype=bool)
            for start in range(0, len(rows), 16384):
                end = start + 16384
                segments = shapely.linestrings(np.stack([coords[idx[rows[start:end]]], coords[idx[cols[start:end]]]], axis=1))
                keep[start:end] = shapely.covers(mask, segments)
            # Degenerate self-segments are points, not invalid LineStrings.
            keep[rows == cols] = True
            rows, cols, distances = rows[keep], cols[keep], distances[keep]
        row_parts.append(idx[rows]); col_parts.append(idx[cols]); distance_parts.append(distances)
    rows, cols = np.concatenate(row_parts), np.concatenate(col_parts)
    adata.obsp[key + "_connectivities"] = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    adata.obsp[key + "_distances"] = sp.csr_matrix(
        (np.concatenate(distance_parts), (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    metadata.update(library_key=library_key, core_key=core_key, tissue_support=tissue_support,
                    spatial_key=spatial_key, coord_type=options["coord_type"], n_spatial_units=len(units),
                    boundaries_key=boundaries_key if contact else None,
                    contact_distance=contact_distance if contact else None)
    if contact:
        metadata.update(astrolabe_method="polygon_distance", metric="euclidean",
                        n_neighbors=int(round(len(rows) / max(adata.n_obs, 1))))
    adata.uns[key] = {"connectivities_key": key + "_connectivities", "distances_key": key + "_distances", "params": metadata}
