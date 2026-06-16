"""
Geometry helpers for spatial annotation and region selection.

All functions return closed ``(n_verts, 2)`` float64 arrays in µm coordinates.
They are used both by the plotting layer (:func:`astrolabe.pl.annotation`) and
the preprocessing layer (:func:`astrolabe.pp.crop`).
"""

from __future__ import annotations

import numpy as np

__all__ = ["rectangle", "circle", "arrow", "merge_polygons"]


def rectangle(
    center: tuple[float, float],
    width: float,
    height: float,
) -> np.ndarray:
    """Return a closed rectangular polygon in µm.

    Parameters
    ----------
    center
        ``(cx, cy)`` centroid in µm.
    width
        Full width along x in µm.
    height
        Full height along y in µm.

    Returns
    -------
    np.ndarray
        Shape ``(5, 2)`` — BL → BR → TR → TL → BL (closed).
    """
    cx, cy = float(center[0]), float(center[1])
    hw, hh = width / 2.0, height / 2.0
    return np.array(
        [
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
            [cx - hw, cy - hh],
        ],
        dtype=np.float64,
    )


def circle(
    center: tuple[float, float],
    radius: float,
    n_points: int = 64,
) -> np.ndarray:
    """Return a closed circular polygon in µm.

    Parameters
    ----------
    center
        ``(cx, cy)`` centroid in µm.
    radius
        Circle radius in µm.
    n_points
        Number of vertices (excluding the closing duplicate).

    Returns
    -------
    np.ndarray
        Shape ``(n_points + 1, 2)`` — closed polygon.
    """
    cx, cy = float(center[0]), float(center[1])
    angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    coords = np.stack([xs, ys], axis=1)
    return np.vstack([coords, coords[:1]])  # close


def arrow(
    start: tuple[float, float],
    end: tuple[float, float],
    head_width: float,
    tail_width: float | None = None,
) -> np.ndarray:
    """Return a closed arrow polygon in µm.

    Parameters
    ----------
    start
        Arrow tail ``(x, y)`` in µm.
    end
        Arrow tip ``(x, y)`` in µm.
    head_width
        Full width of the arrowhead in µm.
    tail_width
        Full width of the shaft in µm.  Defaults to ``head_width * 0.4``.

    Returns
    -------
    np.ndarray
        Shape ``(8, 2)`` — closed arrow polygon.

    Raises
    ------
    ValueError
        If *start* and *end* are the same point.
    """
    s = np.array([float(start[0]), float(start[1])], dtype=np.float64)
    e = np.array([float(end[0]),   float(end[1])],   dtype=np.float64)
    direction = e - s
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        raise ValueError("Start and end must be distinct points")

    unit = direction / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float64)  # 90° CCW

    if tail_width is None:
        tail_width = head_width * 0.4
    tw = tail_width / 2.0
    hw = head_width / 2.0
    head_len = head_width * 1.5
    shaft_len = max(0.0, length - head_len)

    shaft_end = s + unit * shaft_len
    pts = np.array(
        [
            s + perp * tw,            # tail left
            shaft_end + perp * tw,    # shaft left
            shaft_end + perp * hw,    # head shoulder left
            e,                         # tip
            shaft_end - perp * hw,    # head shoulder right
            shaft_end - perp * tw,    # shaft right
            s - perp * tw,            # tail right
            s + perp * tw,            # close (= first point)
        ],
        dtype=np.float64,
    )
    return pts


def merge_polygons(polys: list[np.ndarray]) -> np.ndarray:
    """Return the exterior boundary of the union of multiple polygons.

    Uses shapely for the polygon union.  If the result is a MultiPolygon
    (non-contiguous components), the largest component by area is returned.

    Parameters
    ----------
    polys
        List of ``(N, 2)`` float arrays in µm.  Open or closed polygons are
        both accepted; the closing vertex is ignored by shapely.

    Returns
    -------
    np.ndarray
        Closed ``(M, 2)`` float64 polygon (exterior of the union).

    Raises
    ------
    ValueError
        If no polygon in *polys* has at least 3 vertices.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    geoms = [Polygon(p[:, :2]) for p in polys if p is not None and len(p) >= 3]
    if not geoms:
        raise ValueError("No valid polygon (≥ 3 vertices) found in polys")

    merged = unary_union(geoms)

    if merged.geom_type == "MultiPolygon":
        merged = max(merged.geoms, key=lambda g: g.area)

    return np.array(merged.exterior.coords, dtype=np.float64)
