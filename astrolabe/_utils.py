"""
Shared utilities used across astrolabe submodules.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar, runtime_checkable

import anndata as ad
import mudata as md
import numpy as np

__all__: list[str] = ["with_adata", "ImageSpec", "ImageFilter"]

_F = TypeVar("_F", bound=Callable)


@runtime_checkable
class ImageFilter(Protocol):
    """Protocol for per-channel image filters stored in :attr:`ImageSpec.filter_chain`.

    A filter is any callable that accepts a 2-D ``float32`` NumPy array
    (one image channel) and returns a ``float32`` array of the same shape.
    """

    def __call__(self, channel: np.ndarray) -> np.ndarray:
        ...


@dataclass
class ImageSpec:
    """Metadata descriptor for a pyramidal image file.

    Stores the file path and eagerly-extracted structural metadata.
    Pixel data is never held in memory; use :meth:`open_zarr` for
    transient tile access or let the renderer open the file on demand.

    Attributes
    ----------
    path
        Absolute path to the image file (OME-TIFF or similar).
    pixel_size_um
        Physical size of one pixel in microns (isotropic X/Y assumed).
    dimension_order
        Axis-order string in **numpy** convention (e.g. ``'CYX'``, ``'TCYX'``),
        as returned by ``tifffile.TiffFile.series[0].axes``.
    full_shape
        Array shape at pyramid level 0 (finest resolution).
    level_shapes
        Shape at each pyramid level; ``level_shapes[0] == full_shape``.
        Level 0 is finest, level N-1 is coarsest.
    level_scales
        ``(y_scale, x_scale)`` relative to level 0 at each level.
        Level 0 = ``(1.0, 1.0)``; level 1 (2× downsampled) = ``(2.0, 2.0)``.
    registration_affine
        Optional 3×3 homogeneous matrix mapping image pixel coordinates
        ``(col, row)`` to the reference spatial frame in µm.
    channel_names
        Per-channel names in channel-axis order.
    """

    path: str
    pixel_size_um: float
    dimension_order: str
    full_shape: tuple[int, ...]
    level_shapes: list[tuple[int, ...]]
    level_scales: list[tuple[float, float]]
    registration_affine: np.ndarray | None = None
    channel_names: list[str] | None = None
    transform_chain: list = field(default_factory=list)
    """Sequence of 3×3 float64 µm-space affine matrices appended by
    :func:`~astrolabe.pp.rotate` / :func:`~astrolabe.pp.flip`.
    Applied in order (index 0 first) when computing image extent at render time."""

    filter_chain: list = field(default_factory=list)
    """Sequence of :class:`ImageFilter` callables appended by
    :func:`~astrolabe.pp.add_image_filter`.  Applied per-channel in order
    after pixel data is loaded from disk and before normalisation."""

    stain_matrix: np.ndarray | None = None
    """Optional ``(3, 3)`` float64 deconvolution matrix for H&E images.
    Maps optical-density-space RGB → stain concentrations ``[H, E, residual]``
    via ``concentrations_hw3 = od_hw3 @ stain_matrix``.
    Set by :func:`~astrolabe.io.read_he`; ``None`` for all other image types.
    Consumed by :func:`~astrolabe.pp.extract_if_intensities` which applies
    pixel-level deconvolution before per-cell averaging; **not** applied at render time."""

    @property
    def n_levels(self) -> int:
        """Number of pyramid levels (1 = single-resolution image)."""
        return len(self.level_shapes)

    @property
    def combined_transform(self) -> np.ndarray | None:
        """Composed 3×3 µm-space affine from :attr:`transform_chain`; ``None`` if empty."""
        if not self.transform_chain:
            return None
        result = np.eye(3, dtype=np.float64)
        for M in self.transform_chain:
            result = M @ result
        return result

    def select_pyramid_level(self, target_max_dim: int) -> int:
        """Return the finest pyramid level index that fits within *target_max_dim*.

        Iterates from finest (index 0) to coarsest and returns the first
        level whose longest spatial dimension is ≤ *target_max_dim*.
        Returns the coarsest level if none fits.
        """
        for i, shape in enumerate(self.level_shapes):
            if max(shape[-2], shape[-1]) <= target_max_dim:
                return i
        return self.n_levels - 1

    @contextmanager
    def open_zarr(self):
        """Transient zarr context for random-tile reads.

        Opens the image file as a zarr array or group, yields it, then
        closes the underlying store.  Intended for
        :func:`~astrolabe.pp.extract_if_intensities` where per-cell tile
        slicing requires sustained random access over many cells.

        Yields
        ------
        zarr.Array or zarr.Group
        """
        import tifffile as _tifffile
        import zarr as _zarr
        tif = _tifffile.TiffFile(self.path)
        store = tif.aszarr()
        z = _zarr.open(store, mode="r")
        try:
            yield z
        finally:
            store.close()
            tif.close()


def _transform_coords(coords: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homogeneous affine matrix to ``(N, 2)`` coordinates.

    Parameters
    ----------
    coords
        ``(N, 2)`` float array of ``[x, y]`` coordinates.
    M
        ``(3, 3)`` homogeneous affine matrix.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` transformed coordinates, same dtype as *coords*.
    """
    h = np.ones((len(coords), 3), dtype=np.float64)
    h[:, :2] = coords
    return (M @ h.T).T[:, :2].astype(coords.dtype)


def with_adata(fn: _F) -> _F:
    """Decorator: prepend ``data`` / ``modality`` args that resolve to AnnData before calling *fn*.

    The wrapped function must accept an :class:`~anndata.AnnData` as its first
    positional argument.  The decorator replaces that argument with:

    * ``data`` — :class:`~anndata.AnnData` **or** :class:`~mudata.MuData`
    * ``modality`` — keyword-only ``str | None = None``

    Resolution follows :func:`_resolve_adata`: MuData is unwrapped via the
    ``modality`` kwarg or ``uns['default_modality']``.

    Examples
    --------
    >>> @with_adata
    ... def my_fn(adata: ad.AnnData, *, inplace: bool = True):
    ...     ...
    >>> my_fn(mdata)                    # MuData resolved automatically
    >>> my_fn(mdata, modality="xenium") # explicit modality
    >>> my_fn(adata)                    # AnnData passed through unchanged
    """
    @functools.wraps(fn)
    def wrapper(data: ad.AnnData | md.MuData, *args, modality: str | None = None, **kwargs):
        adata = _resolve_adata(data, modality)
        return fn(adata, *args, **kwargs)
    # Remove __wrapped__ so inspect.signature() sees the wrapper's own signature
    # (with `data` / `modality`) rather than following through to the inner fn.
    del wrapper.__wrapped__  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def _resolve_adata(
    data: ad.AnnData | md.MuData,
    modality: str | None = None,
) -> ad.AnnData:
    """Return an AnnData from either an AnnData or a MuData.

    Parameters
    ----------
    data
        Either an AnnData (returned as-is) or a MuData.
    modality
        Modality key to select from a MuData.  When ``None``, falls back to
        ``data.uns['default_modality']`` which is set automatically by
        :func:`astrolabe.io.read_xenium`.

    Raises
    ------
    TypeError
        If ``data`` is neither AnnData nor MuData.
    KeyError
        If the resolved modality key is not present in the MuData.
    ValueError
        If ``data`` is a MuData and no modality can be inferred.
    """
    if isinstance(data, ad.AnnData):
        return data
    if not isinstance(data, md.MuData):
        raise TypeError(f"Expected AnnData or MuData, got {type(data).__name__!r}.")

    key = modality or data.uns.get("default_modality")
    if key is None:
        raise ValueError(
            "Cannot infer modality from MuData: uns['default_modality'] is not set "
            "and no modality= argument was given. Pass modality='xenium' (or the "
            "appropriate key), or ensure the MuData was created by ab.io.read_xenium()."
        )
    if key not in data.mod:
        raise KeyError(
            f"Modality '{key}' not found in MuData. Available: {list(data.mod.keys())}"
        )
    return data[key]
