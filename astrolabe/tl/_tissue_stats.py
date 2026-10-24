"""Point-pattern inference inside observed tissue, including holes and fragments."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from astrolabe._spatial_units import _units, _support
from astrolabe._statistics import adjust_p, permutation_p


def _sample_mask(mask, n, rng):
    triangles = shapely.get_parts(shapely.constrained_delaunay_triangles(mask))
    areas = shapely.area(triangles)
    if not len(triangles) or not np.isclose(areas.sum(), mask.area):
        raise ValueError("Cannot triangulate the observed tissue mask")
    vertices = np.array([np.asarray(shape.exterior.coords)[:3] for shape in triangles])
    selected = vertices[rng.choice(len(triangles), n, p=areas/areas.sum())]
    uv = rng.random((n, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1-uv[flip]
    return selected[:, 0] + uv[:, :1]*(selected[:, 1]-selected[:, 0]) + uv[:, 1:]*(selected[:, 2]-selected[:, 0])


def _masked_l(points, radii, mask, edge_correction):
    n = len(points)
    values = np.full(len(radii), np.nan)
    if n < 2:
        return values
    tree = cKDTree(points)
    clearance = shapely.distance(shapely.points(points), mask.boundary)
    for i, radius in enumerate(radii):
        eligible = clearance >= radius if edge_correction == "border" else np.ones(n, dtype=bool)
        if not eligible.any():
            continue
        counts = tree.query_ball_point(points[eligible], radius, return_length=True) - 1
        k = mask.area * counts.sum() / (eligible.sum()*(n-1))
        values[i] = np.sqrt(k/np.pi)-radius
    return values


def _ripley_masked(adata, groupby, *, spatial_key, tissue_support, library_key, core_key,
                    radii, support, edge_correction, n_perms, seed, key_added):
    if edge_correction not in ("border", "none"):
        raise ValueError("edge_correction must be border or none")
    if not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")
    units, columns = _units(adata, library_key, core_key)
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Ripley analysis requires finite 2-D coordinates")
    groups = ["all"] if groupby is None else list(pd.unique(adata.obs[groupby].dropna()))
    results = {}
    entropy = int(np.random.SeedSequence(seed).entropy)
    for labels, indices in units:
        if tissue_support is None:
            if columns and any("core" in column.lower() for column in columns):
                raise ValueError("TMA Ripley analysis requires observed per-core tissue support")
            bounds = np.r_[coords[indices].min(axis=0), coords[indices].max(axis=0)]
            mask = shapely.box(*bounds)
        else:
            mask = _support(adata, labels, columns, tissue_support)
        if mask.area <= 0 or not shapely.covers(mask, shapely.points(coords[indices])).all():
            raise ValueError("All analyzed points must lie in a positive-area observed window")
        chosen_radii = (np.linspace(0, min(np.subtract(mask.bounds[2:], mask.bounds[:2]))/4, support)
                        if radii is None else np.asarray(radii, dtype=float))
        if chosen_radii.ndim != 1 or not len(chosen_radii) or not np.isfinite(chosen_radii).all() or (chosen_radii < 0).any() or (np.diff(chosen_radii) <= 0).any():
            raise ValueError("radii must be finite, nonnegative, and strictly increasing")
        for group in groups:
            selected = indices if groupby is None else indices[adata.obs[groupby].to_numpy()[indices] == group]
            identity = json.dumps([list(map(str, labels)), str(group)])
            digest = hashlib.sha256(identity.encode()).hexdigest()
            rng = np.random.default_rng(np.random.SeedSequence([entropy, int(digest[:16], 16)]))
            observed = _masked_l(coords[selected], chosen_radii, mask, edge_correction)
            simulations = np.array([_masked_l(_sample_mask(mask, len(selected), rng), chosen_radii, mask, edge_correction)
                                    for _ in range(n_perms)]) if len(selected) >= 2 else np.empty((0, len(chosen_radii)))
            if simulations.size:
                p, valid = permutation_p(observed, simulations.T)
                expected = np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in simulations.T])
                limits = np.array([np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else [np.nan, np.nan] for v in simulations.T]).T
            else:
                p = expected = np.full(len(chosen_radii), np.nan)
                valid = np.zeros(len(chosen_radii), dtype=int)
                limits = np.full((2, len(chosen_radii)), np.nan)
            frame = pd.DataFrame({"L_obs": observed, "L_exp": expected, "L_lo": limits[0], "L_hi": limits[1],
                                  "p_value": p, "n_valid": valid, "n_points": len(selected), "area_um2": mask.area,
                                  "group": str(group), "edge_correction": edge_correction}, index=pd.Index(chosen_radii, name="distance_um"))
            for key, value in zip(columns, labels):
                frame[key] = value
            results["unit_" + digest[:16]] = frame
    # One declared family includes every requested unit, group and radius.
    adjusted = adjust_p(np.concatenate([frame.p_value.to_numpy() for frame in results.values()]),
                        family_size=sum(len(frame) for frame in results.values())) if results else []
    offset = 0
    for frame in results.values():
        frame["p_value_adj"] = adjusted[offset:offset+len(frame)]
        offset += len(frame)
    adata.uns[key_added] = results
    return results
