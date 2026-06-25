"""
Persistence — save and load a processed MuData to/from a single .h5mu file.

ImageSpec objects in uns['spatial'][*]['images'] are registered with anndata's
IO spec registry (``anndata._io.specs._REGISTRY``) so that ``write_h5mu()`` /
``read_h5mu()`` handle them transparently without any pre/post-processing.

The HDF5 encoding type ``"astrolabe-image-spec" / "0.1.0"`` identifies the
group on load.  ``filter_chain`` entries are serialized as a JSON string —
each filter is a ``@dataclass`` with only scalar fields.

If registration fails (e.g. a future anndata API change), ``save()`` / ``load()``
fall back to the explicit pre/post-processing approach and log a warning.

Awkward arrays in obsm are handled natively by AnnData's h5 backend (≥0.10).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import mudata as md
import numpy as np

from astrolabe._utils import ImageSpec

__all__ = ["save", "load"]

log = logging.getLogger(__name__)


# ── Filter chain serialization ────────────────────────────────────────────────

def _filter_chain_to_json(filter_chain: list) -> str:
    records = []
    for f in filter_chain:
        d = dataclasses.asdict(f)
        d["__type"] = type(f).__name__
        records.append(d)
    return json.dumps(records)


def _filter_chain_from_json(s: str | bytes) -> list:
    if isinstance(s, (bytes, np.bytes_)):
        s = s.decode()
    if not s:
        return []
    import astrolabe.pl._filters as _filters_mod
    registry = {name: getattr(_filters_mod, name) for name in _filters_mod.__all__}
    result = []
    for rec in json.loads(s):
        rec = dict(rec)
        cls_name = rec.pop("__type")
        cls = registry.get(cls_name)
        if cls is None:
            log.warning("load: unknown filter type '%s'; skipping.", cls_name)
            continue
        result.append(cls(**rec))
    return result


# ── anndata IO spec registration (transparent serialization) ──────────────────

_SPECS_REGISTERED = False

try:
    import h5py as _h5py
    from anndata._io.specs import _REGISTRY
    from anndata.experimental import IOSpec

    _SPEC = IOSpec("astrolabe-image-spec", "0.1.0")

    @_REGISTRY.register_write(_h5py.Group, ImageSpec, _SPEC)
    def _write_image_spec(f, k, elem, *, _writer, dataset_kwargs={}):
        g = f.require_group(k)
        # Scalar string / float attrs
        g.attrs["path"] = elem.path
        g.attrs["pixel_size_um"] = float(elem.pixel_size_um)
        g.attrs["dimension_order"] = elem.dimension_order
        # JSON-encoded attrs for structured data (small, shape varies across specs)
        g.attrs["full_shape"] = json.dumps(list(elem.full_shape))
        g.attrs["level_shapes"] = json.dumps([list(s) for s in elem.level_shapes])
        g.attrs["level_scales"] = json.dumps([list(s) for s in elem.level_scales])
        g.attrs["channel_names"] = json.dumps(elem.channel_names)  # None → "null"
        g.attrs["filter_chain"] = _filter_chain_to_json(elem.filter_chain)
        # Optional numpy array fields — zero-length sentinel encodes None
        _writer.write_elem(
            g, "registration_affine",
            elem.registration_affine if elem.registration_affine is not None
            else np.zeros((0,), dtype=np.float64),
            dataset_kwargs=dataset_kwargs,
        )
        _writer.write_elem(
            g, "stain_matrix",
            elem.stain_matrix if elem.stain_matrix is not None
            else np.zeros((0,), dtype=np.float64),
            dataset_kwargs=dataset_kwargs,
        )
        _writer.write_elem(
            g, "transform_chain",
            np.stack(elem.transform_chain).astype(np.float64)
            if elem.transform_chain
            else np.zeros((0, 3, 3), dtype=np.float64),
            dataset_kwargs=dataset_kwargs,
        )

    @_REGISTRY.register_read(_h5py.Group, _SPEC)
    def _read_image_spec(elem, *, _reader):
        def _arr_or_none(key, shape):
            a = _reader.read_elem(elem[key])
            return a.reshape(shape) if a.size > 0 else None

        tc = _reader.read_elem(elem["transform_chain"])
        ch = json.loads(elem.attrs["channel_names"])
        return ImageSpec(
            path=str(elem.attrs["path"]),
            pixel_size_um=float(elem.attrs["pixel_size_um"]),
            dimension_order=str(elem.attrs["dimension_order"]),
            full_shape=tuple(json.loads(elem.attrs["full_shape"])),
            level_shapes=[tuple(s) for s in json.loads(elem.attrs["level_shapes"])],
            level_scales=[tuple(s) for s in json.loads(elem.attrs["level_scales"])],
            channel_names=ch if ch is not None else None,
            filter_chain=_filter_chain_from_json(elem.attrs.get("filter_chain", "[]")),
            registration_affine=_arr_or_none("registration_affine", (3, 3)),
            stain_matrix=_arr_or_none("stain_matrix", (3, 3)),
            transform_chain=[tc[i] for i in range(tc.shape[0])] if tc.ndim == 3 and tc.shape[0] > 0 else [],
        )

    import pyarrow as _pa

    _ARROW_SPEC = IOSpec("astrolabe-arrow-table", "0.1.0")

    @_REGISTRY.register_write(_h5py.Group, _pa.Table, _ARROW_SPEC)
    def _write_arrow_table(f, k, elem, *, _writer, dataset_kwargs={}):
        sink = _pa.BufferOutputStream()
        with _pa.ipc.new_stream(sink, elem.schema) as ipc_writer:
            ipc_writer.write_table(elem)
        _writer.write_elem(
            f.require_group(k), "data",
            np.frombuffer(sink.getvalue(), dtype=np.uint8),
            dataset_kwargs=dataset_kwargs,
        )

    @_REGISTRY.register_read(_h5py.Group, _ARROW_SPEC)
    def _read_arrow_table(elem, *, _reader):
        data = _reader.read_elem(elem["data"])
        return _pa.ipc.open_stream(_pa.BufferReader(data.tobytes())).read_all()

    _SPECS_REGISTERED = True
    log.debug("ImageSpec and pyarrow.Table registered with anndata IO spec registry.")

except Exception as _exc:
    log.warning(
        "Could not register ImageSpec/pyarrow.Table with anndata IO spec registry (%s); "
        "falling back to explicit pre/post-processing in save()/load().",
        _exc,
    )


# ── Fallback helpers (used only when _SPECS_REGISTERED is False) ──────────────

def _spec_to_dict(spec: ImageSpec) -> dict:
    return {
        "__astrolabe_type": "ImageSpec",
        "path": spec.path,
        "pixel_size_um": float(spec.pixel_size_um),
        "dimension_order": spec.dimension_order,
        "full_shape": np.array(spec.full_shape, dtype=np.int64),
        "level_shapes": np.array(spec.level_shapes, dtype=np.int64),
        "level_scales": np.array(spec.level_scales, dtype=np.float64),
        "registration_affine": (
            np.array(spec.registration_affine, dtype=np.float64)
            if spec.registration_affine is not None
            else np.zeros((0,), dtype=np.float64)
        ),
        "channel_names": (
            np.array(spec.channel_names, dtype=object)
            if spec.channel_names is not None
            else np.zeros((0,), dtype=object)
        ),
        "transform_chain": (
            np.stack(spec.transform_chain).astype(np.float64)
            if spec.transform_chain
            else np.zeros((0, 3, 3), dtype=np.float64)
        ),
        "filter_chain": _filter_chain_to_json(spec.filter_chain),
        "stain_matrix": (
            np.array(spec.stain_matrix, dtype=np.float64)
            if spec.stain_matrix is not None
            else np.zeros((0,), dtype=np.float64)
        ),
    }


def _dict_to_spec(d: dict) -> ImageSpec:
    ra_raw = np.asarray(d["registration_affine"], dtype=np.float64)
    registration_affine = ra_raw.reshape(3, 3) if ra_raw.size == 9 else None

    tc_raw = np.asarray(d.get("transform_chain", np.zeros((0, 3, 3))), dtype=np.float64)
    transform_chain = [tc_raw[i] for i in range(tc_raw.shape[0])] if tc_raw.ndim == 3 else []

    ch_raw = np.asarray(d.get("channel_names", np.zeros((0,), dtype=object)))
    channel_names = ch_raw.tolist() if ch_raw.size > 0 else None

    level_shapes = [tuple(int(x) for x in row)
                    for row in np.asarray(d["level_shapes"], dtype=np.int64)]
    level_scales = [tuple(float(x) for x in row)
                    for row in np.asarray(d["level_scales"], dtype=np.float64)]

    sm_raw = np.asarray(d.get("stain_matrix", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    stain_matrix = sm_raw.reshape(3, 3) if sm_raw.size == 9 else None

    return ImageSpec(
        path=str(d["path"]),
        pixel_size_um=float(d["pixel_size_um"]),
        dimension_order=str(d["dimension_order"]),
        full_shape=tuple(int(x) for x in np.asarray(d["full_shape"])),
        level_shapes=level_shapes,
        level_scales=level_scales,
        registration_affine=registration_affine,
        channel_names=channel_names,
        transform_chain=transform_chain,
        filter_chain=_filter_chain_from_json(d.get("filter_chain", "")),
        stain_matrix=stain_matrix,
    )


def _replace_specs(obj):
    if isinstance(obj, ImageSpec):
        return _spec_to_dict(obj)
    if isinstance(obj, dict):
        return {k: _replace_specs(v) for k, v in obj.items()}
    return obj


def _restore_specs(obj):
    if isinstance(obj, dict):
        if obj.get("__astrolabe_type") == "ImageSpec":
            return _dict_to_spec(obj)
        return {k: _restore_specs(v) for k, v in obj.items()}
    return obj


# ── Public API ────────────────────────────────────────────────────────────────

def save(
    mdata: md.MuData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a processed MuData to a single ``.h5mu`` file.

    ``ImageSpec`` objects in ``uns['spatial']`` are serialized transparently
    via anndata's IO spec registry.  ``filter_chain`` entries are round-tripped
    via JSON.  Awkward boundary arrays in ``obsm`` are written natively by
    AnnData's h5 backend.

    Parameters
    ----------
    mdata
        Processed :class:`~mudata.MuData` to save.
    path
        Destination file.  A ``.h5mu`` extension is appended if absent.
    overwrite
        Allow overwriting an existing file (default ``False``).

    Examples
    --------
    >>> ab.io.save(mdata, "results/my_analysis")
    >>> mdata2 = ab.io.load("results/my_analysis.h5mu")
    """
    path = Path(path)
    if path.suffix != ".h5mu":
        path = path.with_suffix(".h5mu")

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"'{path}' already exists. Pass overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    if _SPECS_REGISTERED:
        mdata.write_h5mu(str(path))
    else:
        mdata_copy = mdata.copy()
        mdata_copy.uns.update(_replace_specs(dict(mdata_copy.uns)))
        for adata in mdata_copy.mod.values():
            adata.uns.update(_replace_specs(dict(adata.uns)))
        mdata_copy.write_h5mu(str(path))

    log.info("save: wrote %s", path)


def load(path: str | Path) -> md.MuData:
    """Load a MuData previously saved with :func:`save`.

    Parameters
    ----------
    path
        Path to the ``.h5mu`` file written by :func:`save`.

    Returns
    -------
    MuData

    Examples
    --------
    >>> mdata = ab.io.load("results/my_analysis.h5mu")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found.")

    log.info("load: reading %s", path)
    mdata = md.read_h5mu(str(path))

    if not _SPECS_REGISTERED:
        mdata.uns.update(_restore_specs(dict(mdata.uns)))
        for adata in mdata.mod.values():
            adata.uns.update(_restore_specs(dict(adata.uns)))

    log.info("load: done — %d cells", mdata.n_obs)
    return mdata
