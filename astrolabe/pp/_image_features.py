"""Lazy tile-based image summaries and physically scaled texture features."""
from __future__ import annotations
import numpy as np
import pandas as pd
import shapely
from scipy.ndimage import map_coordinates
from astrolabe._utils import with_adata, from_kwarg, ImageSpec, _transform_coords, _call_image_filter
from astrolabe.pp._if import _resolve_affine, _resolve_channel_axis, _read_tile


def _texture(image, mask, *, levels=32, step=1):
    """Four-direction symmetric GLCMs with both endpoints inside the mask."""
    features = []
    for dy, dx in [(0, step), (step, 0), (step, step), (step, -step)]:
        height, width = image.shape
        ya = slice(max(0, -dy), min(height, height-dy))
        xa = slice(max(0, -dx), min(width, width-dx))
        yb = slice(max(0, dy), min(height, height+dy))
        xb = slice(max(0, dx), min(width, width+dx))
        if abs(dy) >= height or abs(dx) >= width:
            continue
        valid = mask[ya, xa] & mask[yb, xb]
        if not valid.any():
            continue
        a, b = image[ya, xa][valid], image[yb, xb][valid]
        hist = np.bincount(a*levels+b, minlength=levels**2).reshape(levels, levels).astype(float)
        hist += hist.T.copy(); hist /= hist.sum()
        difference = np.arange(levels)[:, None]-np.arange(levels)[None, :]
        features.append([np.sum(hist*difference**2), np.sum(hist/(1+difference**2)), np.sqrt(np.sum(hist**2))])
    return np.mean(features, axis=0) if features else np.full(3, np.nan)


@with_adata(mirror_uns=from_kwarg("key_added"))
def extract_image_features(adata, *, spec: ImageSpec | None = None, image_key="full",
                            library_id=None, library_key="library_id", mode="cell", radius_um=20.,
                            spatial_key="spatial", boundaries_key="cell_boundaries", grid_um=1.,
                            offset_um=1., levels=32, texture_range=None, channels=None,
                            max_grid_points=1_000_000, key_added="image_features"):
    """Cell or circular-context summaries and texture on a canonical micron grid.

    Stores a feature matrix in obsm and names/coverage/registration in uns.
    Tile reads avoid materializing a whole slide. Quantization uses the source
    integer dtype range, or an explicit ``texture_range=(low, high)`` for
    floating images. GLCM contrast, homogeneity and energy average four
    directions at the requested physical offset. Outside pixels are masked.
    Filters follow the image's existing filter chain. One image/library per call.
    """
    if mode not in ("cell", "context") or grid_um <= 0 or offset_um <= 0 or levels < 2 or radius_um <= 0:
        raise ValueError("invalid image feature parameters")
    if spec is None:
        libraries = adata.uns.get("spatial", {})
        if library_id is None:
            if len(libraries) != 1:
                raise ValueError("select library_id explicitly")
            library_id = next(iter(libraries))
        spec = libraries[library_id]["images"][image_key]
    if not isinstance(spec, ImageSpec):
        raise TypeError("spec must be an ImageSpec")
    if library_key in adata.obs and adata.obs[library_key].nunique() > 1 and library_id is None:
        raise ValueError("multiple libraries require an explicit library_id")
    eligible = (adata.obs[library_key].to_numpy() == library_id
                if library_id is not None and library_key in adata.obs else np.ones(adata.n_obs, dtype=bool))
    affine = _resolve_affine(None, spec)
    step = max(1, int(round(offset_um/grid_um)))
    if not np.isclose(step*grid_um, offset_um):
        raise ValueError("offset_um must be an integer multiple of grid_um")
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    channel_axis = _resolve_channel_axis(spec.dimension_order, len(spec.full_shape))
    n_channels = spec.full_shape[channel_axis] if channel_axis is not None else 1
    channels = list(range(n_channels)) if channels is None else list(channels)
    if any(c < 0 or c >= n_channels for c in channels):
        raise ValueError("channel index out of range")
    names = [f"channel_{c}_{feature}" for c in channels for feature in
             ["mean", "median", "std", "q10", "q90", "contrast", "homogeneity", "energy"]]
    result = np.full((adata.n_obs, len(names)), np.nan, dtype=np.float32)
    coverage = np.full(adata.n_obs, np.nan)
    with spec.open_zarr() as source:
        if hasattr(source, "keys"):
            keys = sorted([k for k in source.keys() if str(k).isdigit()], key=int)
            source = source[keys[0]] if keys else source
        dtype = np.dtype(source.dtype)
        value_range = texture_range
        if value_range is None:
            if not np.issubdtype(dtype, np.integer):
                raise ValueError("floating images require an explicit texture_range")
            info = np.iinfo(dtype); value_range = (info.min, info.max)
        lo, hi = value_range
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("texture_range must be finite and increasing")
        axes = spec.dimension_order.upper()
        height, width = spec.full_shape[axes.index("Y")], spec.full_shape[axes.index("X")]
        for idx in np.flatnonzero(eligible):
            if not np.isfinite(coords[idx]).all():
                continue
            if mode == "cell":
                vertices = np.asarray(adata.obsm[boundaries_key][idx])
                if len(vertices) < 3:
                    continue
                geometry = shapely.Polygon(vertices)
                if not geometry.is_valid:
                    raise ValueError(f"invalid cell polygon for {adata.obs_names[idx]}")
            else:
                geometry = shapely.Point(coords[idx]).buffer(radius_um)
            xmin, ymin, xmax, ymax = geometry.bounds
            xs = np.arange(np.ceil(xmin/grid_um), np.floor(xmax/grid_um)+1)*grid_um
            ys = np.arange(np.ceil(ymin/grid_um), np.floor(ymax/grid_um)+1)*grid_um
            if len(xs)*len(ys) > max_grid_points:
                raise ValueError("cell/context grid exceeds max_grid_points")
            if not len(xs) or not len(ys):
                continue
            xx, yy = np.meshgrid(xs, ys)
            xy = np.column_stack([xx.ravel(), yy.ravel()])
            mask = np.asarray(shapely.covers(geometry, shapely.points(xy))).reshape(xx.shape)
            pixels = _transform_coords(xy, affine)
            inside = ((pixels[:,0] >= 0) & (pixels[:,0] <= width-1) &
                      (pixels[:,1] >= 0) & (pixels[:,1] <= height-1)).reshape(xx.shape)
            valid = mask & inside
            coverage[idx] = valid.sum()/mask.sum() if mask.any() else np.nan
            if not valid.any():
                continue
            selected = pixels[valid.ravel()]
            c0, r0 = np.floor(selected.min(axis=0)).astype(int)
            c1, r1 = np.minimum(np.ceil(selected.max(axis=0)).astype(int)+2, [width, height])
            tile = _read_tile(source, channel_axis, r0, r1, c0, c1)
            for ci, channel in enumerate(channels):
                values = tile[channel]
                for filt in spec.filter_chain:
                    values = _call_image_filter(filt, values, spec.um_per_px)
                sampled = map_coordinates(values, [pixels[:,1]-r0, pixels[:,0]-c0],
                                           order=1, mode="constant", cval=np.nan).reshape(xx.shape)
                current = valid & np.isfinite(sampled)
                signal = sampled[current]
                if not signal.size:
                    continue
                quantized = np.minimum(levels-1, np.floor(np.clip((np.nan_to_num(sampled, nan=lo)-lo)/(hi-lo), 0, 1)*levels)).astype(int)
                result[idx, ci*8:(ci+1)*8] = [signal.mean(), np.median(signal), signal.std(),
                    *np.quantile(signal, [.1,.9]), *_texture(quantized, current, levels=levels, step=step)]
    adata.obsm[key_added] = result
    adata.obs[key_added + "_coverage"] = coverage
    adata.uns[key_added] = {"feature_names": names, "mode": mode, "radius_um": radius_um,
        "grid_um": grid_um, "offset_um": offset_um, "levels": levels, "texture_range": list(value_range),
        "channels": channels, "library_id": library_id, "image_path": spec.path,
        "micron_to_pixel": affine, "coverage_key": key_added + "_coverage"}
    return pd.DataFrame(result, index=adata.obs_names, columns=names)
