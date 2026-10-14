"""
Persistence — save and load a processed MuData to/from a single .h5mu file.

ImageSpec objects in uns['spatial'][*]['images'] (and BinMatrixSpec objects in
uns['spatial'][*]['bin_matrix'], the VisiumHD lazy bin-matrix descriptor) are
registered with anndata's IO spec registry (``anndata._io.specs._REGISTRY``)
so that ``write_h5mu()`` / ``read_h5mu()`` handle them transparently without
any pre/post-processing.

The HDF5 encoding type ``"astrolabe-image-spec" / "0.1.0"`` identifies the
group on load.  ``filter_chain`` entries are serialized as a JSON string —
each filter is a ``@dataclass`` with only scalar fields.

``ImageSpec.path`` is stored on disk **relative to the ``.h5mu`` file's own
directory** so that a saved analysis and its referenced image files can be
moved together as a unit; it is resolved back to an absolute path in memory
on both write (in-memory spec untouched) and read. If the image file cannot
be found at load time (e.g. the images were moved independently of the
``.h5mu`` file), ``load()`` logs a warning and, if given, applies
``path_remap`` — a ``{regex_pattern: replacement}`` mapping applied to the
stored path via ``re.sub``.

If registration fails (e.g. a future anndata API change), ``save()`` / ``load()``
fall back to the explicit pre/post-processing approach and log a warning.

Awkward arrays in obsm are handled natively by AnnData's h5 backend (≥0.10).

``attach_modality()`` is the generalized helper used by ``read_he``/``read_ome_tiff``
(and any other reader) to insert a new AnnData as a modality into an existing
MuData without disturbing an already-established ``uns['default_modality']``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np

from astrolabe._constants import UNS_DEFAULT_MODALITY_KEY
from astrolabe._utils import BinMatrixSpec, ImageSpec

__all__ = ["save", "load", "attach_modality"]

log = logging.getLogger(__name__)


# ── Relative/absolute path helpers (portable .h5mu + image bundles) ───────────

def _to_relative_path(path: str, base_dir: Path) -> str:
    """Best-effort: express *path* relative to *base_dir*.

    Falls back to the original (absolute) path when no relative path exists
    (e.g. different drives on Windows).
    """
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return path


def _to_absolute_path(path: str, base_dir: Path) -> str:
    """Resolve *path* against *base_dir* if it is relative; pass through absolute paths.

    Backward-compatible with ``.h5mu`` files written before paths were
    relativized (those stored absolute paths directly).
    """
    p = Path(path)
    return str(p) if p.is_absolute() else str((base_dir / p).resolve())


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
        dest_dir = Path(os.path.abspath(f.file.filename)).parent
        g.attrs["path"] = _to_relative_path(elem.path, dest_dir)
        g.attrs["pixel_size_um"] = float(elem.pixel_size_um)
        g.attrs["dimension_order"] = elem.dimension_order
        g.attrs["is_brightfield"] = bool(elem.is_brightfield)
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
        # NOTE: path is left as stored (relative or absolute) — it cannot be
        # resolved to absolute here, since mudata.read_h5mu() always opens the
        # file via a Python file object before wrapping it in h5py.File(), so
        # elem.file.filename is that object's repr, not a real path. Resolution
        # against the .h5mu file's own directory happens in io.load() instead,
        # where the real path is known.
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
            is_brightfield=bool(elem.attrs.get("is_brightfield", False)),
        )

    _BIN_MATRIX_SPEC = IOSpec("astrolabe-bin-matrix-spec", "0.1.0")

    @_REGISTRY.register_write(_h5py.Group, BinMatrixSpec, _BIN_MATRIX_SPEC)
    def _write_bin_matrix_spec(f, k, elem, *, _writer, dataset_kwargs={}):
        g = f.require_group(k)
        dest_dir = Path(os.path.abspath(f.file.filename)).parent
        g.attrs["path"] = _to_relative_path(elem.path, dest_dir)
        g.attrs["bin_size_um"] = float(elem.bin_size_um)
        g.attrs["n_obs"] = int(elem.n_obs)
        g.attrs["n_vars"] = int(elem.n_vars)
        _writer.write_elem(
            g, "transform_chain",
            np.stack(elem.transform_chain).astype(np.float64)
            if elem.transform_chain
            else np.zeros((0, 3, 3), dtype=np.float64),
            dataset_kwargs=dataset_kwargs,
        )

    @_REGISTRY.register_read(_h5py.Group, _BIN_MATRIX_SPEC)
    def _read_bin_matrix_spec(elem, *, _reader):
        tc = _reader.read_elem(elem["transform_chain"])
        # See the note in _read_image_spec: path is resolved to absolute
        # against the .h5mu file's own directory in io.load(), not here.
        return BinMatrixSpec(
            path=str(elem.attrs["path"]),
            bin_size_um=float(elem.attrs["bin_size_um"]),
            n_obs=int(elem.attrs["n_obs"]),
            n_vars=int(elem.attrs["n_vars"]),
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

    import geopandas as _gpd
    import pandas as _pd

    _GEODATAFRAME_SPEC = IOSpec("astrolabe-geodataframe", "0.1.0")

    @_REGISTRY.register_write(_h5py.Group, _gpd.GeoDataFrame, _GEODATAFRAME_SPEC)
    def _write_geodataframe(f, k, elem, *, _writer, dataset_kwargs={}):
        g = f.require_group(k)
        geom_col = elem.geometry.name
        g.attrs["geometry_column"] = geom_col
        g.attrs["crs"] = elem.crs.to_wkt() if elem.crs is not None else ""
        # Hex-encoded WKB (safe for fixed/variable-length string storage; raw
        # WKB bytes can legitimately contain trailing NUL bytes, which numpy's
        # byte-string dtypes silently strip).
        wkb_hex = np.array(
            [geom.wkb_hex if geom is not None else "" for geom in elem[geom_col]],
            dtype=object,
        )
        _writer.write_elem(g, "geometry_wkb_hex", wkb_hex, dataset_kwargs=dataset_kwargs)
        # Properties are serialized as JSON rather than written as a native
        # anndata DataFrame: GeoJSON property columns (e.g. a Xenium
        # annotation's 'classification') can hold nested dict/list values,
        # which anndata's per-column dtype writers can't handle but JSON can.
        props = _pd.DataFrame(elem.drop(columns=[geom_col])).reset_index(drop=True)
        props_json = props.to_json(orient="table").encode("utf-8")
        _writer.write_elem(
            g, "properties_json",
            np.frombuffer(props_json, dtype=np.uint8),
            dataset_kwargs=dataset_kwargs,
        )

    @_REGISTRY.register_read(_h5py.Group, _GEODATAFRAME_SPEC)
    def _read_geodataframe(elem, *, _reader):
        import io as _io_stdlib

        import shapely

        geom_col = str(elem.attrs.get("geometry_column", "geometry"))
        crs_wkt = str(elem.attrs.get("crs", ""))
        wkb_hex = _reader.read_elem(elem["geometry_wkb_hex"])
        geoms = [shapely.from_wkb(bytes.fromhex(h)) if h else None for h in wkb_hex]
        props_json = _reader.read_elem(elem["properties_json"]).tobytes().decode("utf-8")
        props = _pd.read_json(_io_stdlib.StringIO(props_json), orient="table")
        gdf = _gpd.GeoDataFrame(
            props, geometry=_gpd.GeoSeries(geoms, index=props.index, crs=crs_wkt or None)
        )
        if geom_col != "geometry":
            gdf = gdf.rename_geometry(geom_col)
        return gdf

    _SPECS_REGISTERED = True
    log.debug(
        "ImageSpec, BinMatrixSpec, pyarrow.Table, and geopandas.GeoDataFrame registered "
        "with anndata IO spec registry."
    )

except Exception as _exc:
    log.warning(
        "Could not register ImageSpec/BinMatrixSpec/pyarrow.Table with anndata IO spec "
        "registry (%s); falling back to explicit pre/post-processing in save()/load().",
        _exc,
    )


# ── Fallback helpers (used only when _SPECS_REGISTERED is False) ──────────────

def _spec_to_dict(spec: ImageSpec, base_dir: Path) -> dict:
    return {
        "__astrolabe_type": "ImageSpec",
        "path": _to_relative_path(spec.path, base_dir),
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
        "is_brightfield": bool(spec.is_brightfield),
    }


def _bin_matrix_spec_to_dict(spec: BinMatrixSpec, base_dir: Path) -> dict:
    return {
        "__astrolabe_type": "BinMatrixSpec",
        "path": _to_relative_path(spec.path, base_dir),
        "bin_size_um": float(spec.bin_size_um),
        "n_obs": int(spec.n_obs),
        "n_vars": int(spec.n_vars),
        "transform_chain": (
            np.stack(spec.transform_chain).astype(np.float64)
            if spec.transform_chain
            else np.zeros((0, 3, 3), dtype=np.float64)
        ),
    }


def _dict_to_bin_matrix_spec(d: dict) -> BinMatrixSpec:
    tc_raw = np.asarray(d.get("transform_chain", np.zeros((0, 3, 3))), dtype=np.float64)
    transform_chain = [tc_raw[i] for i in range(tc_raw.shape[0])] if tc_raw.ndim == 3 else []
    return BinMatrixSpec(
        path=str(d["path"]),
        bin_size_um=float(d["bin_size_um"]),
        n_obs=int(d["n_obs"]),
        n_vars=int(d["n_vars"]),
        transform_chain=transform_chain,
    )


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
        is_brightfield=bool(d.get("is_brightfield", False)),
    )


def _replace_specs(obj, base_dir: Path):
    if isinstance(obj, ImageSpec):
        return _spec_to_dict(obj, base_dir)
    if isinstance(obj, BinMatrixSpec):
        return _bin_matrix_spec_to_dict(obj, base_dir)
    if isinstance(obj, dict):
        return {k: _replace_specs(v, base_dir) for k, v in obj.items()}
    return obj


def _restore_specs(obj):
    if isinstance(obj, dict):
        if obj.get("__astrolabe_type") == "ImageSpec":
            return _dict_to_spec(obj)
        if obj.get("__astrolabe_type") == "BinMatrixSpec":
            return _dict_to_bin_matrix_spec(obj)
        return {k: _restore_specs(v) for k, v in obj.items()}
    return obj


# ── ImageSpec/BinMatrixSpec discovery (for save()/load() warnings and path_remap) ──

def _iter_image_specs(obj):
    """Recursively yield every :class:`ImageSpec`/:class:`BinMatrixSpec` nested
    in *obj* (dicts/lists/tuples) — both are path-referencing descriptors that
    need the same relative/absolute path handling at save/load time."""
    if isinstance(obj, (ImageSpec, BinMatrixSpec)):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_image_specs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_image_specs(v)


def _all_image_specs(mdata: md.MuData) -> list[ImageSpec | BinMatrixSpec]:
    specs = list(_iter_image_specs(dict(mdata.uns)))
    for adata in mdata.mod.values():
        specs.extend(_iter_image_specs(dict(adata.uns)))
    return specs


def _resolve_relative_paths(mdata: md.MuData, base_dir: Path) -> None:
    """Resolve every (still-relative) ``ImageSpec.path`` against *base_dir* in-place."""
    for spec in _all_image_specs(mdata):
        spec.path = _to_absolute_path(spec.path, base_dir)


def _remap_missing_paths(mdata: md.MuData, path_remap: dict[str, str] | None) -> None:
    """Warn about (and try to fix via *path_remap*) ``ImageSpec`` paths that don't exist."""
    for spec in _all_image_specs(mdata):
        if Path(spec.path).exists():
            continue

        remapped = False
        for pattern, replacement in (path_remap or {}).items():
            candidate = re.sub(pattern, replacement, spec.path)
            if candidate != spec.path and Path(candidate).exists():
                log.info("load: remapped image path '%s' -> '%s'", spec.path, candidate)
                spec.path = candidate
                remapped = True
                break

        if not remapped:
            log.warning(
                "load: image file '%s' does not exist. If the image was moved or renamed, "
                "pass path_remap={r'<old-path-regex>': '<new-path>'} to load() to map old "
                "paths (regex) to new paths that contain the image.",
                spec.path,
            )


# ── Public API ────────────────────────────────────────────────────────────────

def attach_modality(
    mdata: md.MuData,
    adata: ad.AnnData,
    key: str,
    *,
    set_default: bool = False,
) -> None:
    """Insert *adata* as a new modality ``mdata[key]`` in place.

    This is the generalized version of the attachment logic used internally
    by :func:`~astrolabe.io.read_he` and :func:`~astrolabe.io.read_ome_tiff`
    (``mdata=``/``mdata_key=``) — use it directly to attach any custom
    :class:`~anndata.AnnData` (e.g. a manually constructed modality, or a
    reader that doesn't expose its own ``mdata=`` parameter) into an existing
    :class:`~mudata.MuData`.

    Any ``adata.uns['spatial']`` entries are merged into ``mdata.uns['spatial']``.
    ``mdata.uns['default_modality']`` is left untouched unless it is unset
    (in which case *key* becomes the default), or *set_default=True* is
    passed explicitly to switch the default modality to *key*.

    ``mdata.update()`` is called afterward to refresh MuData's internal
    obs/var index bookkeeping for the newly attached modality (harmless with
    ``pull_on_update=False``, since it does not pull any columns) — without
    this, later operations that filter another modality's ``var``/``obs``
    (e.g. ``pipeline.filter_xenium``) can leave MuData's internal state
    inconsistent enough that :func:`save` fails with a
    ``NotImplementedError`` at write time.

    Parameters
    ----------
    mdata
        The :class:`~mudata.MuData` to modify in place.
    adata
        The :class:`~anndata.AnnData` to attach as a new modality.
    key
        Key to store *adata* under, i.e. ``mdata[key]``.
    set_default
        If ``True``, set ``mdata.uns['default_modality'] = key`` even if a
        default is already established.  Default ``False`` — never changes
        an already-established default modality.

    Examples
    --------
    >>> mdata = ab.io.read_xenium("run/")
    >>> custom_adata = ad.AnnData(...)
    >>> ab.io.attach_modality(mdata, custom_adata, "custom")
    >>> mdata.uns['default_modality']  # unchanged, still 'xenium'
    'xenium'
    """
    mdata.mod[key] = adata
    if "spatial" in adata.uns:
        mdata.uns.setdefault("spatial", {}).update(adata.uns["spatial"])
    if set_default:
        mdata.uns[UNS_DEFAULT_MODALITY_KEY] = key
    else:
        mdata.uns.setdefault(UNS_DEFAULT_MODALITY_KEY, key)
    mdata.update()


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

    Image paths are written **relative to the ``.h5mu`` file's own directory**
    (falling back to an absolute path only when no relative path exists, e.g.
    the image is on a different drive on Windows) — move the ``.h5mu`` file
    together with its referenced image files to keep them resolvable, or use
    ``load(..., path_remap=...)`` to point ``load()`` at their new location.

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
    dest_dir = Path(os.path.abspath(path)).parent

    n_specs = len(_all_image_specs(mdata))
    if n_specs:
        log.warning(
            "save: %d referenced file path(s) (images and/or VisiumHD bin matrices) "
            "will be saved relative to '%s' — move the referenced file(s) together "
            "with this .h5mu file to keep them resolvable, or pass path_remap=... "
            "to load() if you relocate them independently.",
            n_specs, dest_dir,
        )

    import tempfile
    from astrolabe.io._tables import encode_tables

    mdata_copy = mdata.copy()
    mdata_copy.uns = encode_tables(dict(mdata_copy.uns))
    for adata in mdata_copy.mod.values():
        adata.uns = encode_tables(dict(adata.uns))
    if not _SPECS_REGISTERED:
        mdata_copy.uns.update(_replace_specs(dict(mdata_copy.uns), dest_dir))
        for adata in mdata_copy.mod.values():
            adata.uns.update(_replace_specs(dict(adata.uns), dest_dir))
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".h5mu", dir=dest_dir)
    os.close(fd)
    try:
        mdata_copy.write_h5mu(temporary)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    log.info("save: wrote %s", path)


def load(
    path: str | Path,
    *,
    path_remap: dict[str, str] | None = None,
) -> md.MuData:
    """Load a MuData previously saved with :func:`save`.

    Image paths stored in ``ImageSpec`` objects are resolved relative to this
    file's own directory (see :func:`save`). If a resolved image path does not
    exist on disk — e.g. the images were moved independently of the ``.h5mu``
    file — a warning is logged for each missing path, and *path_remap* (if
    given) is applied to try to fix it.

    Parameters
    ----------
    path
        Path to the ``.h5mu`` file written by :func:`save`.
    path_remap
        Optional ``{regex_pattern: replacement}`` mapping applied via
        ``re.sub(pattern, replacement, old_path)`` to any ``ImageSpec.path``
        that does not exist on disk after loading. Only applied when the
        resulting path exists; the first matching pattern (in dict order)
        wins. Paths that still don't exist after remapping (or with no
        matching pattern) are left as-is with a logged warning.

    Returns
    -------
    MuData

    Examples
    --------
    >>> mdata = ab.io.load("results/my_analysis.h5mu")
    >>> # images moved from /old/data/... to /new/data/...
    >>> mdata = ab.io.load(
    ...     "results/my_analysis.h5mu",
    ...     path_remap={r"^/old/data/": "/new/data/"},
    ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'{path}' not found.")

    log.info("load: reading %s", path)
    mdata = md.read_h5mu(str(path))
    from astrolabe.io._tables import decode_tables
    mdata.uns = decode_tables(dict(mdata.uns))
    for adata in mdata.mod.values():
        adata.uns = decode_tables(dict(adata.uns))

    if not _SPECS_REGISTERED:
        mdata.uns.update(_restore_specs(dict(mdata.uns)))
        for adata in mdata.mod.values():
            adata.uns.update(_restore_specs(dict(adata.uns)))

    # ImageSpec.path is stored relative to this .h5mu file's own directory
    # (see save()); resolve it back to absolute now that we know the real
    # path this file was loaded from (elem.file.filename is not reliable here
    # — see the note in _read_image_spec).
    base_dir = Path(os.path.abspath(path)).parent
    _resolve_relative_paths(mdata, base_dir)

    _remap_missing_paths(mdata, path_remap)

    log.info("load: done — %d cells", mdata.n_obs)
    return mdata
