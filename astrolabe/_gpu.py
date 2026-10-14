"""
Optional GPU acceleration via rapids-singlecell.

Install rapids-singlecell before enabling:
    uv add rapids-singlecell-cu12   # CUDA 12
    uv add rapids-singlecell-cu13   # CUDA 13

Usage
-----
>>> import astrolabe as ab
>>>
>>> # Context manager (recommended): auto-transfers X+layers on enter/exit
>>> with ab.gpu(mdata):
...     ab.pp.normalize_total(mdata)
...     ab.tl.pca(mdata)
...     ab.tl.transcriptional_neighbors(mdata)
...     ab.tl.cluster(mdata)
>>>
>>> # Persistent toggle with manual data management
>>> ab.use_gpu(True)
>>> ab.to_gpu(mdata)
>>> ab.pp.normalize_total(mdata)
>>> ab.to_cpu(mdata)
>>> ab.use_gpu(False)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import anndata as ad
    import mudata as md

log = logging.getLogger(__name__)

_GPU_ENABLED: bool = False


# ── Public API ────────────────────────────────────────────────────────────────

def use_gpu(enabled: bool) -> None:
    """Enable or disable GPU-accelerated routing via rapids-singlecell.

    When enabled, eligible astrolabe functions route through rapids-singlecell
    instead of scanpy.  Data is managed per-function: each function moves only
    the arrays it needs to GPU, then restores them to CPU afterward (unless the
    data was already on GPU when the function was called, in which case it is
    left on GPU).

    Parameters
    ----------
    enabled
        ``True`` to enable GPU routing; ``False`` to restore CPU-only behavior.

    See Also
    --------
    gpu : Context manager that also manages data transfer automatically.
    to_gpu : Explicitly transfer data to GPU.
    to_cpu : Explicitly transfer data back to CPU.
    """
    global _GPU_ENABLED
    _GPU_ENABLED = bool(enabled)
    log.info("GPU acceleration %s", "enabled" if _GPU_ENABLED else "disabled")


def gpu_enabled() -> bool:
    """Return ``True`` if GPU-accelerated routing is currently active."""
    return _GPU_ENABLED


class gpu:
    """Context manager that enables GPU routing and optionally manages data transfer.

    On entry, sets the GPU flag and (by default) transfers *data* to GPU.
    On exit, restores the prior flag state and (by default) transfers *data*
    back to CPU.

    The context manager restores the previous flag state correctly when nested,
    so ``with ab.gpu(): ...`` inside code that already called ``ab.use_gpu(True)``
    will leave GPU enabled after the inner context exits.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` to transfer.
        Pass ``None`` to skip automatic data transfer and manage it manually
    auto_transfer
        When ``True`` (default) and *data* is provided, calls :func:`to_gpu`
        on entry and :func:`to_cpu` on exit
    modality
        For :class:`~mudata.MuData`: restrict transfer to this one modality.
        ``None`` transfers all modalities

    Examples
    --------
    >>> with ab.gpu(mdata):
    ...     ab.pp.normalize_total(mdata)
    ...     ab.tl.pca(mdata)
    ...     ab.tl.transcriptional_neighbors(mdata)
    ...     ab.tl.cluster(mdata)
    """

    def __init__(self, data=None, *, auto_transfer: bool = True, modality=None):
        self._data = data
        self._auto_transfer = auto_transfer
        self._modality = modality
        self._prev_state: bool = False

    def __enter__(self):
        global _GPU_ENABLED
        self._prev_state = _GPU_ENABLED
        _GPU_ENABLED = True
        try:
            if self._auto_transfer and self._data is not None:
                to_gpu(self._data, modality=self._modality)
        except BaseException:
            _GPU_ENABLED = self._prev_state
            raise
        return self

    def __exit__(self, *_):
        global _GPU_ENABLED
        try:
            if self._auto_transfer and self._data is not None:
                to_cpu(self._data, modality=self._modality)
        finally:
            _GPU_ENABLED = self._prev_state


def to_gpu(data, modality=None, convert_all: bool = False) -> None:
    """Transfer AnnData / MuData expression matrix to GPU memory.

    For :class:`~mudata.MuData`, transfers every modality unless *modality* is
    specified.  Sparse matrices are converted to cuSPARSE format; dense arrays
    to CuPy.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        When *data* is a MuData, restrict transfer to this one modality.
    convert_all
        When ``True``, also transfer ``obsm``, ``varm``, and other arrays in
        addition to ``X`` and ``layers``.
    """
    rsc = _require_rapids()
    import mudata as mdt

    if isinstance(data, mdt.MuData):
        mods = [data.mod[modality]] if modality is not None else list(data.mod.values())
        for adata in mods:
            rsc.get.anndata_to_GPU(adata, convert_all=convert_all)
    else:
        rsc.get.anndata_to_GPU(data, convert_all=convert_all)


def to_cpu(data, modality=None, convert_all: bool = True) -> None:
    """Transfer AnnData / MuData expression matrix back to CPU memory.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        When *data* is a MuData, restrict transfer to this one modality.
    convert_all
        When ``True`` (default), also transfer ``obsm``, ``varm``, and other
        arrays in addition to ``X`` and ``layers``.
    """
    rsc = _require_rapids()
    import mudata as mdt

    if isinstance(data, mdt.MuData):
        mods = [data.mod[modality]] if modality is not None else list(data.mod.values())
        for adata in mods:
            rsc.get.anndata_to_CPU(adata, convert_all=convert_all)
    else:
        rsc.get.anndata_to_CPU(data, convert_all=convert_all)


# ── Internal helpers (used by individual functions, not part of public API) ───

def _require_rapids():
    """Import and return the rapids_singlecell module, or raise a helpful error."""
    try:
        import rapids_singlecell as rsc
        return rsc
    except ImportError as exc:
        raise ImportError(
            "rapids-singlecell is required for GPU acceleration. "
            "Install with: uv add rapids-singlecell-cu12  (CUDA 12) "
            "or: uv add rapids-singlecell-cu13  (CUDA 13)"
        ) from exc


def _is_cupy(arr) -> bool:
    """Return True if *arr* is a CuPy array or CuPy sparse matrix."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return True
        import cupyx.scipy.sparse as cpsp
        return isinstance(arr, cpsp.spmatrix)
    except ImportError:
        return False


def _to_cpu_arr(arr):
    """Convert a CuPy / CuPy-sparse array to numpy / scipy. No-op if already CPU."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        import cupyx.scipy.sparse as cpsp
        if isinstance(arr, cpsp.spmatrix):
            return arr.get()
    except ImportError:
        pass
    return arr


@contextmanager
def _gpu_X(adata: "ad.AnnData"):
    """Ensure ``adata.X`` is on GPU for the duration of the block.

    Yields ``moved: bool`` — ``True`` if we transferred X to GPU (meaning the
    caller should also bring any newly written outputs back to CPU on exit).
    If X was already on GPU, yields ``False`` and does nothing on exit.
    """
    rsc = _require_rapids()
    moved = not _is_cupy(adata.X)
    if moved:
        rsc.get.X_to_GPU(adata)
    try:
        yield moved
    finally:
        if moved:
            rsc.get.X_to_CPU(adata)
