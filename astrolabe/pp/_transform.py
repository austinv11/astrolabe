"""
Spatial selection and coordinate transformation.

Functions
---------
crop
    Filter cells whose centroids fall inside a polygon region; zoom plots
    automatically via ``uns['spatial_bounds']``.
rotate
    Rotate cell coordinates (and image transform chain) CCW by a given angle.
flip
    Mirror cell coordinates (and image transform chain) along x, y, or both axes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np

from astrolabe._utils import ImageSpec, _transform_coords, with_adata

if TYPE_CHECKING:
    pass

__all__ = ["crop", "rotate", "flip"]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Return boolean mask of points that fall inside *polygon*.

    Uses shapely's vectorised ``contains_xy`` (shapely ≥2.x).

    Parameters
    ----------
    points
        ``(N, 2)`` float array of ``[x, y]`` centroids in µm.
    polygon
        ``(M, 2)`` float array of polygon vertices in µm.  Need not be closed.
    """
    import shapely
    from shapely.geometry import Polygon

    geom = Polygon(polygon)
    return shapely.contains_xy(geom, points[:, 0], points[:, 1])


def _transform_boundary_obsm(
    raw,
    M: np.ndarray,
) -> object:
    """Apply affine *M* to an awkward-array boundary obsm entry.

    Handles both ``awkward.Array`` (variable-length vertex lists) and plain
    numpy arrays of shape ``(n_cells, n_verts, 2)``.
    """
    try:
        import awkward as ak  # type: ignore[import]

        if isinstance(raw, ak.Array):
            counts = np.asarray(ak.num(raw, axis=1))          # (n_cells,)
            flat = np.asarray(ak.flatten(raw, axis=1))         # (total_verts, 2)
            transformed = _transform_coords(flat.astype(np.float64), M).astype(flat.dtype)
            return ak.unflatten(ak.Array(transformed), counts, axis=0)
    except ImportError:
        pass

    # Fall back: treat as regular (n_cells, n_verts, 2) numpy array
    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        n, v, _ = raw.shape
        flat = raw.reshape(-1, 2)
        transformed = _transform_coords(flat.astype(np.float64), M).astype(raw.dtype)
        return transformed.reshape(n, v, 2)

    log.warning(
        "Could not transform boundaries of type %s; skipping.", type(raw).__name__
    )
    return raw


def _append_image_transform(adata: ad.AnnData, M: np.ndarray) -> None:
    """Append µm-space affine *M* to all ImageSpec transform_chains in *adata.uns*."""
    for lib_data in adata.uns.get("spatial", {}).values():
        for spec in lib_data.get("images", {}).values():
            if isinstance(spec, ImageSpec):
                spec.transform_chain.append(M)


def _build_rotation_matrix(
    angle_deg: float,
    center: tuple[float, float],
) -> np.ndarray:
    """3×3 CCW rotation matrix in µm space around *center*."""
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = float(center[0]), float(center[1])

    T_to   = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    R      = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    T_back = np.array([[1, 0,  cx], [0, 1,  cy], [0, 0, 1]], dtype=np.float64)
    return T_back @ R @ T_to


def _build_flip_matrix(
    axis: str,
    center: tuple[float, float],
) -> np.ndarray:
    """3×3 reflection matrix in µm space around *center*.

    ``axis='x'`` mirrors across the vertical axis (negates x coordinates).
    ``axis='y'`` mirrors across the horizontal axis (negates y coordinates).
    ``axis='both'`` mirrors across both.
    """
    cx, cy = float(center[0]), float(center[1])
    sx = -1.0 if axis in ("x", "both") else 1.0
    sy = -1.0 if axis in ("y", "both") else 1.0
    return np.array(
        [
            [sx, 0,  cx * (1.0 - sx)],
            [0,  sy, cy * (1.0 - sy)],
            [0,  0,  1.0            ],
        ],
        dtype=np.float64,
    )


def _apply_coord_transform(
    adata: ad.AnnData,
    M: np.ndarray,
    spatial_key: str,
    boundaries_keys: list[str],
) -> None:
    """Transform cell centroids and boundary arrays in *adata* by affine *M*."""
    coords = adata.obsm[spatial_key].astype(np.float64)
    adata.obsm[spatial_key] = _transform_coords(coords, M).astype(
        adata.obsm[spatial_key].dtype
    )
    for bkey in boundaries_keys:
        if bkey in adata.obsm:
            adata.obsm[bkey] = _transform_boundary_obsm(adata.obsm[bkey], M)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@with_adata
def crop(
    adata: ad.AnnData,
    region: np.ndarray,
    *,
    spatial_key: str = "spatial",
    inplace: bool = True,
) -> tuple[ad.AnnData, np.ndarray] | None:
    """Filter cells to those whose centroids fall inside *region*.

    Cell coordinates are updated eagerly.  Images are **not** modified —
    :func:`~astrolabe.pl._spatial._render_panel` zooms the plot view by
    reading the bounding box stored in ``adata.uns['spatial_bounds']``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    region
        ``(n_verts, 2)`` polygon in µm.  Use :func:`~astrolabe.pl.rectangle`,
        :func:`~astrolabe.pl.circle`, or :func:`~astrolabe.pl.arrow` to
        generate common shapes.
    spatial_key
        Key in ``obsm`` for cell centroids.
    inplace
        If ``True``, modify in place and return ``None``.
        If ``False``, return ``(filtered_adata, boolean_mask)``.

    Returns
    -------
    None or tuple[AnnData, ndarray]
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found.  "
            "Ensure spatial coordinates have been loaded."
        )

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    region = np.asarray(region, dtype=np.float64)
    mask = _points_in_polygon(centroids, region)

    n_kept = int(mask.sum())
    log.info("crop: keeping %d / %d cells inside region.", n_kept, len(mask))

    # Store bounding box for automatic plot zoom
    bounds = {
        "xmin": float(region[:, 0].min()),
        "xmax": float(region[:, 0].max()),
        "ymin": float(region[:, 1].min()),
        "ymax": float(region[:, 1].max()),
    }
    adata.uns["spatial_bounds"] = bounds

    if inplace:
        adata._inplace_subset_obs(mask)
        return None
    else:
        subset = adata[mask].copy()
        subset.uns["spatial_bounds"] = bounds
        return subset, mask


@with_adata
def rotate(
    adata: ad.AnnData,
    angle_deg: float,
    *,
    center: tuple[float, float] | None = None,
    spatial_key: str = "spatial",
    inplace: bool = True,
) -> ad.AnnData | None:
    """Rotate spatial coordinates counter-clockwise by *angle_deg* degrees.

    Cell centroids and boundary arrays are transformed eagerly.
    Each ``ImageSpec`` in ``adata.uns['spatial']`` receives an entry in its
    ``transform_chain`` that is applied lazily at render time.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    angle_deg
        Rotation angle in degrees (positive = counter-clockwise).
    center
        ``(x, y)`` centre of rotation in µm.  Defaults to the mean centroid
        of ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for cell centroids.
    inplace
        If ``True``, modify in place and return ``None``.
        If ``False``, return the modified copy.

    Returns
    -------
    None or AnnData
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found.")

    if center is None:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        center = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    M = _build_rotation_matrix(angle_deg, center)
    boundaries_keys = ["cell_boundaries", "nucleus_boundaries"]

    if inplace:
        _apply_coord_transform(adata, M, spatial_key, boundaries_keys)
        _append_image_transform(adata, M)
        return None
    else:
        out = adata.copy()
        _apply_coord_transform(out, M, spatial_key, boundaries_keys)
        _append_image_transform(out, M)
        return out


@with_adata
def flip(
    adata: ad.AnnData,
    axis: Literal["x", "y", "both"] = "x",
    *,
    center: tuple[float, float] | None = None,
    spatial_key: str = "spatial",
    inplace: bool = True,
) -> ad.AnnData | None:
    """Mirror spatial coordinates along one or both axes.

    ``axis='x'`` reflects across the vertical midline (negates x).
    ``axis='y'`` reflects across the horizontal midline (negates y).
    ``axis='both'`` applies both reflections (equivalent to a 180° rotation).

    Cell centroids and boundary arrays are transformed eagerly.
    Each ``ImageSpec`` in ``adata.uns['spatial']`` receives an entry in its
    ``transform_chain`` that is applied lazily at render time.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    axis
        Axis / axes to flip.
    center
        ``(x, y)`` origin of reflection in µm.  Defaults to the mean centroid
        of ``obsm[spatial_key]``.
    spatial_key
        Key in ``obsm`` for cell centroids.
    inplace
        If ``True``, modify in place and return ``None``.
        If ``False``, return the modified copy.

    Returns
    -------
    None or AnnData
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found.")

    if center is None:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        center = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    M = _build_flip_matrix(axis, center)
    boundaries_keys = ["cell_boundaries", "nucleus_boundaries"]

    if inplace:
        _apply_coord_transform(adata, M, spatial_key, boundaries_keys)
        _append_image_transform(adata, M)
        return None
    else:
        out = adata.copy()
        _apply_coord_transform(out, M, spatial_key, boundaries_keys)
        _append_image_transform(out, M)
        return out
