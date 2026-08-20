"""
Shared utilities used across astrolabe submodules.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar, runtime_checkable

import anndata as ad
import mudata as md
import numpy as np
import scipy.sparse as sp

from astrolabe._constants import UNS_DEFAULT_MODALITY_KEY

__all__: list[str] = ["with_adata", "from_kwarg", "ImageSpec", "ImageFilter", "BinMatrixSpec"]

_F = TypeVar("_F", bound=Callable)
log = logging.getLogger(__name__)


@runtime_checkable
class ImageFilter(Protocol):
    """Protocol for per-channel image filters stored in :attr:`ImageSpec.filter_chain`.

    A filter is any callable that accepts a 2-D ``float32`` NumPy array
    (one image channel) and returns a ``float32`` array of the same shape.

    **Unit convention:** filter bandwidths should be specified in physical
    **microns**, not pixels — the same ``filter_chain`` is applied at very
    different pixel resolutions by its two consumers (full-resolution
    extraction in :func:`~astrolabe.pp.extract_if_intensities` vs a
    downsampled, zoom-dependent pyramid level in :func:`~astrolabe.pl.image`).
    To support this, a filter *may* accept an optional keyword-only
    ``um_per_px: float | None = None`` giving the effective µm-per-pixel of
    the array being filtered; astrolabe's built-in filters
    (:class:`~astrolabe.pl.GaussianFilter`, :class:`~astrolabe.pl.CLAHEFilter`,
    :class:`~astrolabe.pl.FFTFilter`) use it to convert their µm parameters
    to pixels per call.  Both consumers pass it via
    :func:`_call_image_filter`, which falls back to a plain
    ``filter(channel)`` call for filters that don't accept the keyword —
    such filters remain valid but are resolution-locked (pixel semantics).
    """

    def __call__(self, channel: np.ndarray, *, um_per_px: float | None = None) -> np.ndarray:
        ...


def _call_image_filter(
    filt: Callable,
    channel: np.ndarray,
    um_per_px: float | None = None,
) -> np.ndarray:
    """Invoke an :class:`ImageFilter`, forwarding *um_per_px* when accepted.

    Filters that don't declare a ``um_per_px`` keyword (e.g. bare
    ``lambda ch: ...`` callables) are called with the channel only and keep
    legacy pixel semantics.
    """
    accepts = False
    try:
        params = inspect.signature(filt).parameters
        accepts = "um_per_px" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        pass
    if accepts:
        return filt(channel, um_per_px=um_per_px)
    return filt(channel)


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
    after pixel data is loaded from disk and before normalization."""

    stain_matrix: np.ndarray | None = None
    """Optional ``(3, 3)`` float64 deconvolution matrix for H&E images.
    Maps optical-density-space RGB → stain concentrations ``[H, E, residual]``
    via ``concentrations_hw3 = od_hw3 @ stain_matrix``.
    Set by :func:`~astrolabe.io.read_he`; ``None`` for all other image types.
    Consumed by :func:`~astrolabe.pp.extract_if_intensities` (pixel-level
    deconvolution before per-cell averaging) and, when a stain name is passed
    as ``channel=`` to :func:`~astrolabe.pl.image`, at render time too."""

    is_brightfield: bool = False
    """``True`` for brightfield RGB images (set by :func:`~astrolabe.io.read_he`);
    ``False`` for fluorescence/IF images.  Drives :func:`~astrolabe.pl.image`'s
    rendering defaults: a brightfield image with ``channel=None`` composites
    its 3 channels as literal RGB (no per-channel pseudo-coloring, no legend)
    on a light background, instead of the multi-channel IF treatment."""

    @property
    def n_levels(self) -> int:
        """Number of pyramid levels (1 = single-resolution image)."""
        return len(self.level_shapes)

    @property
    def um_per_px(self) -> float | None:
        """Physical size of one level-0 pixel in microns, or ``None`` if unknown.

        Prefers :attr:`pixel_size_um`; falls back to the isotropic scale
        implied by :attr:`registration_affine` (``sqrt(|det|)`` of its 2×2
        linear part, which maps pixels → µm).  Used to resolve µm-specified
        image-filter bandwidths (see :class:`ImageFilter`) to the pixel grid
        actually being filtered.
        """
        px = self.pixel_size_um
        if px is not None and np.isfinite(px) and px > 0:
            return float(px)
        if self.registration_affine is not None:
            L = np.asarray(self.registration_affine, dtype=np.float64)[:2, :2]
            det = abs(float(np.linalg.det(L)))
            if det > 0:
                return float(np.sqrt(det))
        return None

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

        The spatial dimensions are located via :attr:`dimension_order` (the
        Y/X axis positions) rather than assuming the last two axes — true for
        ``CYX``-style fluorescence images, but false for channels-last
        ``YXC``/``YXS`` RGB images (e.g. H&E), where ``shape[-1]`` is the
        channel count, not a spatial extent.
        """
        axes = self.dimension_order.upper()
        ndim = len(self.full_shape)
        y_idx = axes.index("Y") if "Y" in axes else ndim - 2
        x_idx = axes.index("X") if "X" in axes else ndim - 1
        for i, shape in enumerate(self.level_shapes):
            if max(shape[y_idx], shape[x_idx]) <= target_max_dim:
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


@dataclass
class BinMatrixSpec:
    """Metadata descriptor for a lazy, disk-resident sparse bin × gene count matrix.

    Modeled directly on :class:`ImageSpec`: stores only the file path and
    eagerly-extracted structural metadata (shape, bin size). The actual count
    data is never held in memory by this descriptor — see
    :func:`~astrolabe.pp.load_bin_counts` for on-demand partial or full reads
    from :attr:`path`, and the module docstring of
    :mod:`astrolabe.io._visium_hd` for the overall lazy-bin-matrix design.

    Attributes
    ----------
    path
        Absolute path to the 10x feature-barcode HDF5 matrix
        (``filtered_feature_bc_matrix.h5`` or ``raw_feature_bc_matrix.h5``)
        that this descriptor refers to.
    bin_size_um
        Physical size of one square bin in microns (``2.0`` for the native
        VisiumHD resolution; see :func:`~astrolabe.tl.rebin_visium_hd` for
        computed coarser resolutions).
    n_obs
        Number of bins (rows of the logical bin × gene matrix).
    n_vars
        Number of genes/features (columns of the logical bin × gene matrix).
    """

    path: str
    bin_size_um: float
    n_obs: int
    n_vars: int
    transform_chain: list = field(default_factory=list)
    """Sequence of 3×3 float64 µm-space affine matrices appended by
    :func:`~astrolabe.pp.rotate` / :func:`~astrolabe.pp.flip` — mirrors
    :attr:`ImageSpec.transform_chain`. Bin centroids (``obsm['spatial']``) are
    transformed eagerly like any other coordinate array; this chain exists so
    a *lazily loaded* bin submatrix's coordinates (fetched later via
    :func:`~astrolabe.pp.load_bin_counts`) can still be reconciled with any
    transform applied after the bin modality was read."""


def _resolve_transcripts(adata: ad.AnnData, transcripts: object, func_name: str) -> object:
    """Default a ``transcripts=None`` argument to ``adata.uns['transcripts']``.

    The shared resolver behind the ``transcripts=None`` convention
    (``seg.assign_transcripts``, ``seg.create_compartment_layers``,
    ``tl.subcellular_coords``, ``tl.flux``). Unlike
    ``pp.calculate_transcript_assignment_qc``'s stricter resolver, a pruned
    table (``uns['transcripts_pruned']``) is fine here: these consumers are
    not rate computations, and a table whose QC-removed cells' rows were
    dropped is the right input — so it only logs, never raises, on the
    pruned flag.
    """
    if transcripts is not None:
        return transcripts
    if "transcripts" not in adata.uns:
        raise ValueError(
            f"{func_name}: no transcript table given and none found in "
            "adata.uns['transcripts']. Call ab.io.read_xenium_transcripts(mdata) "
            "first, or pass transcripts= explicitly."
        )
    if adata.uns.get("transcripts_pruned", False):
        log.info(
            "%s: using uns['transcripts'], which has been pruned by a "
            "filtering step (rows of removed cells/genes dropped).",
            func_name,
        )
    return adata.uns["transcripts"]


def _resolve_field(
    adata: ad.AnnData,
    key: str,
    *,
    axis: str = "obs",
    layer: str | None = None,
) -> tuple[np.ndarray, bool]:
    """Resolve a field name to a 1-D array of per-obs or per-var values.

    Parameters
    ----------
    adata
        Source AnnData.
    key
        Field name to resolve.
    axis
        ``"obs"`` (default): searches ``var_names`` first (gene expression,
        read from ``adata.layers[layer]`` if given, else ``adata.X``), then
        ``obs.columns``. ``"var"``: searches ``var.columns`` only.
    layer
        Layer to read gene expression from when ``axis="obs"``. Invalid
        (raises ``ValueError``) when ``axis="var"``, since layers are
        cell x gene, not gene-indexed.

    Returns
    -------
    tuple[np.ndarray, bool]
        ``(values, is_categorical)``.

    Raises
    ------
    KeyError
        If *key* is not found on the requested axis.
    ValueError
        If *axis* is invalid, or *layer* is given with ``axis="var"``.
    """
    if axis not in ("obs", "var"):
        raise ValueError(f"axis must be 'obs' or 'var', got {axis!r}")

    if axis == "var":
        if layer is not None:
            raise ValueError(
                "layer= is not valid with axis='var' (layers are cell x gene, not gene-indexed)"
            )
        if key in adata.var.columns:
            col = adata.var[key]
            if hasattr(col, "cat") or col.dtype == object:
                return col.astype(str).to_numpy(), True
            return col.to_numpy(dtype=np.float64), False
        raise KeyError(
            f"'{key}' not found in var columns. Available var columns: {list(adata.var.columns)}"
        )

    # axis == "obs"
    if key in adata.var_names:
        idx = adata.var_names.get_loc(key)
        mat = adata.layers[layer] if layer is not None else adata.X
        x_col = mat[:, idx]
        if sp.issparse(x_col):
            x_col = np.asarray(x_col.todense()).ravel()
        return np.asarray(x_col).astype(np.float64), False

    if key in adata.obs.columns:
        col = adata.obs[key]
        if hasattr(col, "cat") or col.dtype == object:
            return col.astype(str).to_numpy(), True
        return col.to_numpy(dtype=np.float64), False

    raise KeyError(
        f"'{key}' not found in var_names or obs columns. Available obs columns: {list(adata.obs.columns)}"
    )


def _coerce_field(
    adata: ad.AnnData,
    value,
    *,
    axis: str = "obs",
    layer: str | None = None,
) -> tuple[np.ndarray | None, bool]:
    """Dispatch a scatter/histogram/heatmap field argument.

    ``None`` passes through unchanged; a ``str`` is resolved via
    :func:`_resolve_field`; anything else (array-like) is used directly,
    with categorical-ness inferred from dtype. Array-like values are
    *not* length-validated against ``adata`` — callers may legitimately
    pass derived arrays (e.g. stashed control-probe stats) whose length
    differs from ``adata.n_obs``/``adata.n_vars``.
    """
    if value is None:
        return None, False
    if isinstance(value, str):
        return _resolve_field(adata, value, axis=axis, layer=layer)

    arr = np.asarray(value)
    is_categorical = arr.dtype == object or arr.dtype.kind in ("U", "S") or hasattr(value, "cat")
    if is_categorical:
        return arr.astype(str), True
    return arr.astype(np.float64), False


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
    # Always return float32 — affine-transformed coordinates are continuous even
    # when the input was integer pixel coordinates.
    out_dtype = coords.dtype if np.issubdtype(coords.dtype, np.floating) else np.float32
    return (M @ h.T).T[:, :2].astype(out_dtype)


def _aggregate_sparse_by_group(X, group_ids: np.ndarray, n_groups: int):
    """Sum sparse matrix rows by group via an indicator-matrix matmul.

    Shared by :func:`~astrolabe.tl.rebin_visium_hd` (grouping 2µm VisiumHD
    bins into coarser bins) and :func:`~astrolabe.seg.assign_bins` (grouping
    bins into segmented cells) — both need the identical "sum these rows,
    grouped by an integer label" operation on a potentially large sparse
    matrix, which a dense groupby would defeat the purpose of staying sparse.

    Parameters
    ----------
    X
        ``(n_rows, n_cols)`` sparse matrix (any scipy.sparse format).
    group_ids
        ``(n_rows,)`` integer array with values in ``[0, n_groups)``.
    n_groups
        Number of output groups/rows.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_groups, n_cols)`` — row *i* is the sum of all input rows with
        ``group_ids == i``.
    """
    n_rows = X.shape[0]
    group_ids = np.asarray(group_ids, dtype=np.int64)
    # indicator's dtype matches X's (not a hardcoded float64): scipy's
    # matmul promotes to the wider of the two operand dtypes, so a
    # float64 indicator against a float32 X (the common case for count
    # matrices) silently doubled peak memory on the largest Visium HD
    # bin matrices this is used on.
    indicator_dtype = X.dtype if np.issubdtype(X.dtype, np.floating) else np.float32
    indicator = sp.coo_matrix(
        (np.ones(n_rows, dtype=indicator_dtype), (group_ids, np.arange(n_rows))),
        shape=(n_groups, n_rows),
    ).tocsr()
    return (indicator @ X).tocsr()


def _dtype_scale(dtype) -> float:
    """White-point scale for normalizing image data to [0, 1] from its dtype.

    Integer dtypes map to their type maximum (255 for uint8, 65535 for
    uint16, ...); floating dtypes are assumed already in [0, 1] and map
    to 1.0. Deriving the scale from the *dtype* rather than from the data
    (``arr.max()``) keeps normalization consistent across calls — a
    data-dependent max differs per cell/tile and silently zeroes 16-bit
    brightfield data when divided by an assumed 255.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in ("u", "i"):
        return float(np.iinfo(dtype).max)
    return 1.0


def _od_deconvolve(
    rgb: np.ndarray, stain_matrix: np.ndarray, scale: float | None = None
) -> np.ndarray:
    """Apply H&E optical-density stain deconvolution: ``(3, ...)`` RGB → ``(..., 3)`` concentrations.

    Shared by :func:`astrolabe.pp._if._deconvolve_cell_pixels` (per-cell
    reduction, ``rgb`` shaped ``(3, n_pixels)``) and :mod:`astrolabe.pl._spatial`'s
    render-time single-stain view (``rgb`` shaped ``(3, H, W)``) — both just
    reduce/index the trailing stain axis of this function's output
    differently, so the OD-space math lives in exactly one place.

    Optical density is base-10 (``od = -log10(rgb)``). Note this differs
    from ``skimage.color.separate_stains``, which normalizes by
    ``log(1e-6)`` — astrolabe concentrations are a constant 6.0× skimage's
    for the same stain matrix (invariant under correlation/ranking).

    Parameters
    ----------
    rgb
        ``(3, ...)`` float array of RGB values, channel-first. Values may be
        in 0–255 (uint8 origin) or 0–1 range.
    stain_matrix
        ``(3, 3)`` deconvolution matrix: ``conc = od @ M`` where *od* is
        optical-density-space RGB. Convention matches
        :attr:`~astrolabe._utils.ImageSpec.stain_matrix`.
    scale
        Explicit white-point divisor mapping *rgb* to [0, 1] — derive it
        from the source image dtype via :func:`_dtype_scale` and pass it
        whenever the dtype is known. ``None`` falls back to a
        data-dependent guess (divide by 255 iff ``rgb.max() > 1``), which
        is per-call inconsistent and wrong for 16-bit brightfield data.

    Returns
    -------
    np.ndarray
        ``(..., 3)`` float64 stain concentrations ``[H, E, residual]``,
        clipped to non-negative (negative optical density has no physical
        meaning as a stain concentration).
    """
    # Floor at one 8-bit quantization step (1/255), not an arbitrary tiny
    # eps: a pure-black pixel (e.g. scipy.ndimage.rotate's default zero
    # padding) clipped to 1e-6 produced OD = 6.0 per channel -- far beyond
    # the ~2.4 OD ceiling an 8-bit brightfield sensor can actually encode --
    # and those synthetic spikes then dominated any percentile-based
    # contrast stretch of the deconvolved stain image.
    eps = 1.0 / 255.0
    rgb = rgb.astype(np.float64)
    if scale is not None:
        rgb = rgb / scale
    elif rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, eps, 1.0)

    od = -np.log10(np.moveaxis(rgb, 0, -1))  # (..., 3)
    stains = od @ stain_matrix  # (..., 3)
    return np.clip(stains, 0.0, None)


def _spawn_rngs(seed: int | None, n: int) -> list[np.random.Generator]:
    """Independent, reproducible RNG substreams for parallel permutation loops.

    A single shared ``Generator`` mutated sequentially (``rng.shuffle(...)``
    called once per iteration) is not safe to hand to concurrent workers.
    Spawning ``n`` independent child streams from one ``SeedSequence`` keeps
    the whole loop reproducible for a given *seed* regardless of worker count
    or execution order.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


class _KeyFromParam:
    """Marker built by :func:`from_kwarg` — see there."""

    __slots__ = ("param_name",)

    def __init__(self, param_name: str) -> None:
        self.param_name = param_name


def from_kwarg(param_name: str) -> _KeyFromParam:
    """Build a ``mirror_uns=`` spec for :func:`with_adata` that reads the
    mirrored ``uns`` key from parameter *param_name* of the wrapped function,
    evaluated against the actual call (respecting a caller-supplied override,
    falling back to the function's own default otherwise).

    Use this for the common ``key_added: str = "..."`` pattern::

        @with_adata(mirror_uns=from_kwarg("key_added"))
        def moran(adata, keys, *, key_added: str = "moran", ...): ...

    For functions whose default key is *computed* from other arguments
    (e.g. ``key_added or f"decay_length_{key}"``), pass a plain callable
    ``(bound_arguments: inspect.BoundArguments) -> str`` instead.
    """
    return _KeyFromParam(param_name)


def _mirror_to_mudata_uns(
    mdata: md.MuData, modality: str, key: str, value: object, fn: Callable
) -> None:
    """Additively mirror ``adata.uns[key]`` up to ``mdata.uns[key][modality]``.

    Never touches ``obs``/``var``/``obsm`` — those remain adata-only by
    construction, since this is only ever called with the single mirrored
    ``uns`` key. Stores the same object (no copy).
    """
    top = mdata.uns.get(key)
    if not isinstance(top, dict):
        top = {}
        mdata.uns[key] = top
    if modality in top:
        log.warning(
            "with_adata: overwriting mdata.uns[%r][%r] — %s was already run on "
            "modality %r before. The MuData-level mirror now reflects only the "
            "latest run (the per-modality adata.uns[%r] has typically also "
            "already been overwritten by the re-run).",
            key, modality, fn.__qualname__, modality, key,
        )
    top[modality] = value


def with_adata(
    fn: _F | None = None,
    *,
    mirror_uns: str | _KeyFromParam | Callable[[inspect.BoundArguments], str] | None = None,
) -> _F | Callable[[_F], _F]:
    """Decorator: prepend ``data`` / ``modality`` args that resolve to AnnData before calling *fn*.

    The wrapped function must accept an :class:`~anndata.AnnData` as its first
    positional argument.  The decorator replaces that argument with:

    * ``data`` — :class:`~anndata.AnnData` **or** :class:`~mudata.MuData`
    * ``modality`` — keyword-only ``str | None = None``

    Resolution follows :func:`_resolve_adata`: MuData is unwrapped via the
    ``modality`` kwarg or ``uns['default_modality']``.

    Parameters
    ----------
    mirror_uns
        Optional. When given, and *data* is a :class:`~mudata.MuData`, the
        ``uns`` entry *fn* wrote is additionally mirrored to
        ``mdata.uns[<key>][<modality>]`` after *fn* returns — purely
        additive; the original ``adata.uns[<key>]`` write is untouched, and
        no other attribute (``obs``/``var``/``obsm``) is ever touched by this
        mechanism. No-op when *data* is a bare AnnData, or when *fn* did not
        end up writing to that ``uns`` key this call (e.g. conditional
        writes). Accepts a literal ``str`` (fixed key), :func:`from_kwarg`
        (key read from a named parameter), or a callable
        ``(bound_arguments) -> str`` for keys computed from multiple
        call-time arguments.

    Examples
    --------
    >>> @with_adata
    ... def my_fn(adata: ad.AnnData, *, inplace: bool = True):
    ...     ...
    >>> my_fn(mdata)                    # MuData resolved automatically
    >>> my_fn(mdata, modality="xenium") # explicit modality
    >>> my_fn(adata)                    # AnnData passed through unchanged

    >>> @with_adata(mirror_uns=from_kwarg("key_added"))
    ... def my_stat(adata: ad.AnnData, *, key_added: str = "my_stat"):
    ...     adata.uns[key_added] = {...}
    >>> my_stat(mdata)  # also mirrored to mdata.uns["my_stat"]["xenium"]
    """
    def _decorate(fn: _F) -> _F:
        sig = inspect.signature(fn) if mirror_uns is not None else None

        def _resolve_mirror_key(adata: ad.AnnData, args: tuple, kwargs: dict) -> str | None:
            if isinstance(mirror_uns, str):
                return mirror_uns
            ba = sig.bind(adata, *args, **kwargs)  # type: ignore[union-attr]
            ba.apply_defaults()
            if isinstance(mirror_uns, _KeyFromParam):
                if mirror_uns.param_name not in ba.arguments:
                    raise TypeError(
                        f"with_adata(mirror_uns=from_kwarg({mirror_uns.param_name!r})) "
                        f"on {fn.__qualname__}: no such parameter."
                    )
                return ba.arguments[mirror_uns.param_name]
            return mirror_uns(ba)

        @functools.wraps(fn)
        def wrapper(data: ad.AnnData | md.MuData, *args, modality: str | None = None, **kwargs):
            adata, resolved_modality = _resolve_adata_and_modality(data, modality)
            result = fn(adata, *args, **kwargs)

            if mirror_uns is not None and resolved_modality is not None:
                uns_key = _resolve_mirror_key(adata, args, kwargs)
                if uns_key is not None and uns_key in adata.uns:
                    _mirror_to_mudata_uns(data, resolved_modality, uns_key, adata.uns[uns_key], fn)  # type: ignore[arg-type]

            return result

        # Remove __wrapped__ so inspect.signature() sees the wrapper's own signature
        # (with `data` / `modality`) rather than following through to the inner fn.
        del wrapper.__wrapped__  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if fn is not None:
        return _decorate(fn)
    return _decorate


def _resolve_adata_and_modality(
    data: ad.AnnData | md.MuData,
    modality: str | None = None,
) -> tuple[ad.AnnData, str | None]:
    """Like :func:`_resolve_adata`, but also returns the resolved modality key.

    Returns ``None`` for the modality when *data* is already an AnnData —
    there is no outer MuData to mirror results against.
    """
    if isinstance(data, ad.AnnData):
        return data, None
    if not isinstance(data, md.MuData):
        raise TypeError(f"Expected AnnData or MuData, got {type(data).__name__!r}")

    key = modality or data.uns.get(UNS_DEFAULT_MODALITY_KEY)
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
    return data[key], key


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
    adata, _ = _resolve_adata_and_modality(data, modality)
    return adata


_LIBRARY_ID_KEY = "library_id"


def _check_single_library(adata: ad.AnnData) -> None:
    """Raise a clear error when cells from multiple libraries are present.

    Spatial coordinates are independent per Xenium run — each run starts its
    own µm coordinate frame from (0, 0). Computing spatial statistics or
    plotting cells from different runs together silently mixes unrelated
    tissue sections, which is almost never the desired behavior. Shared by
    every spatially-aware ``pl``/``tl`` function that reads per-cell
    coordinates (``obsm['spatial']``) — a ``obs['library_id']`` column is
    written by :func:`astrolabe.io.concat_samples`.
    """
    if _LIBRARY_ID_KEY not in adata.obs.columns:
        return
    libs = adata.obs[_LIBRARY_ID_KEY].unique()
    if len(libs) <= 1:
        return
    lib_list = sorted(str(l) for l in libs)
    first = lib_list[0]
    raise ValueError(
        f"This AnnData contains cells from {len(libs)} libraries: {lib_list}.\n"
        f"Each Xenium run has its own independent µm coordinate frame — computing "
        f"spatial statistics or plotting them together mixes unrelated tissue "
        f"sections on the same axes.\n\n"
        f"Filter to a single sample first:\n"
        f'    adata_sub = adata[adata.obs["{_LIBRARY_ID_KEY}"] == "{first}"]'
    )
