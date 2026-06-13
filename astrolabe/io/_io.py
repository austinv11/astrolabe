"""
Persistence — save and load a processed MuData to/from a single .h5mu file.

The only thing mudata's native writer cannot handle is the **ImageSpec**
dataclass stored in ``uns['spatial'][*]['images']``.  It contains numpy arrays,
strings, and image filter objects that h5py cannot serialise directly.

Strategy: convert ImageSpec ↔ plain dict (numpy arrays + strings) in-place on
a copy of the MuData before writing.  A ``'__astrolabe_type': 'ImageSpec'``
marker allows automatic reconstruction after ``md.read_h5mu``.

``filter_chain`` entries are serialised as a JSON string — each filter is a
``@dataclass`` with only scalar fields, so ``dataclasses.asdict`` + ``json``
gives a lossless round-trip.

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


# ── Filter chain serialisation ────────────────────────────────────────────────

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
    from astrolabe.pl._filters import CLAHEFilter, FFTFilter, GaussianFilter
    registry = {
        "GaussianFilter": GaussianFilter,
        "CLAHEFilter": CLAHEFilter,
        "FFTFilter": FFTFilter,
    }
    result = []
    for rec in json.loads(s):
        rec = dict(rec)
        cls_name = rec.pop("__type")
        cls = registry.get(cls_name)
        if cls is None:
            log.warning("load: unknown filter type '%s'; skipping.", cls_name)
            continue
        # JSON deserialises tuples as lists — coerce back where needed
        if cls_name == "CLAHEFilter" and isinstance(rec.get("kernel_size"), list):
            rec["kernel_size"] = tuple(rec["kernel_size"])
        result.append(cls(**rec))
    return result


# ── ImageSpec ↔ dict ──────────────────────────────────────────────────────────

def _spec_to_dict(spec: ImageSpec) -> dict:
    """Convert an ImageSpec to a plain dict that h5mu can serialise."""
    return {
        "__astrolabe_type": "ImageSpec",
        "path": spec.path,
        "pixel_size_um": float(spec.pixel_size_um),
        "dimension_order": spec.dimension_order,
        "full_shape": np.array(spec.full_shape, dtype=np.int64),
        "level_shapes": np.array(spec.level_shapes, dtype=np.int64),
        "level_scales": np.array(spec.level_scales, dtype=np.float64),
        # Use a (0,) sentinel array when the field is None
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
        # Stack list of (3,3) matrices → (N,3,3); empty → (0,3,3)
        "transform_chain": (
            np.stack(spec.transform_chain).astype(np.float64)
            if spec.transform_chain
            else np.zeros((0, 3, 3), dtype=np.float64)
        ),
        # Serialise filter callables as a JSON string
        "filter_chain": _filter_chain_to_json(spec.filter_chain),
        # H&E stain deconvolution matrix — (0,) sentinel when absent
        "stain_matrix": (
            np.array(spec.stain_matrix, dtype=np.float64)
            if spec.stain_matrix is not None
            else np.zeros((0,), dtype=np.float64)
        ),
    }


def _dict_to_spec(d: dict) -> ImageSpec:
    """Reconstruct an ImageSpec from the dict produced by _spec_to_dict."""
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

    filter_chain = _filter_chain_from_json(d.get("filter_chain", ""))

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
        filter_chain=filter_chain,
        stain_matrix=stain_matrix,
    )


# ── uns walking ───────────────────────────────────────────────────────────────

def _replace_specs(obj):
    """Recursively replace ImageSpec objects with serialisable dicts."""
    if isinstance(obj, ImageSpec):
        return _spec_to_dict(obj)
    if isinstance(obj, dict):
        return {k: _replace_specs(v) for k, v in obj.items()}
    return obj


def _restore_specs(obj):
    """Recursively convert dicts with __astrolabe_type marker back to ImageSpec."""
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

    Handles **ImageSpec** objects in ``uns['spatial']`` by converting them to
    plain dicts before writing and restoring them automatically on
    :func:`load`.  ``ImageSpec.filter_chain`` entries are round-tripped via
    JSON — all three built-in filter classes are fully serialisable.
    Awkward boundary arrays in ``obsm`` are written natively by AnnData's h5
    backend.

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

    # Work on a shallow copy so we don't mutate the caller's object
    mdata_copy = mdata.copy()

    # Replace ImageSpec → dict throughout uns (MuData level + each modality)
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

    # Reconstruct ImageSpec objects from their dict representations
    mdata.uns.update(_restore_specs(dict(mdata.uns)))
    for adata in mdata.mod.values():
        adata.uns.update(_restore_specs(dict(adata.uns)))

    log.info("load: done — %d cells", mdata.n_obs)
    return mdata
