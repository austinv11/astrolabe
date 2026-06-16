"""
Reader for OME-TIFF multiplexed immunofluorescence images.

Canonical call
--------------
>>> import astrolabe as ab
>>> if_adata = ab.io.read_ome_tiff("panel.ome.tif")
>>> # or, to register directly into an existing MuData:
>>> ab.io.read_ome_tiff("panel.ome.tif", mdata=mdata, key="if")

Schema (returned AnnData)
--------------------------
X           : None  (populated later by pp.extract_if_intensities)
obs         : empty DataFrame — row index will be set to cell_ids when
              pp.extract_if_intensities is called
var         : channel_name (index), round (int or NaN), channel (int),
              excitation_wavelength_nm (float or NaN),
              emission_wavelength_nm (float or NaN)
uns
    spatial
        <library_id>
            coordinate_unit     : 'microns'
            images
                full            : ImageSpec  (path + metadata; no pixel data in memory)
                <level_k>       : ImageSpec  (same path, same spec; named for explicit access)
    astrolabe_schema_version : str

Notes
-----
- No pixel data is read at load time.  The :class:`~astrolabe._utils.ImageSpec`
  stored under ``images`` holds file path and structural metadata only.
  Pixel data is loaded on demand by the renderer or intensity extractor.
- For multi-round panels (CODEX / PhenoCycler), channels are flattened
  across rounds.  ``var['round']`` and ``var['channel']`` record provenance.
- The AnnData ``obs`` index is intentionally empty at load time because
  cell segmentation has not yet occurred.  ``pp.extract_if_intensities``
  populates both ``X`` and ``obs`` from the segmentation boundaries in
  ``obsm['cell_boundaries']``.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import tifffile

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY, UNS_DEFAULT_MODALITY_KEY
from astrolabe._utils import ImageSpec

__all__ = ["read_ome_tiff"]

log = logging.getLogger(__name__)

# Units that OME-XML reports as microns (case-insensitive match)
_MICRON_UNITS = {"µm", "um", "micron", "microns", "micrometer", "micrometers"}


def read_ome_tiff(
    path: str | Path,
    *,
    library_id: str | None = None,
    channel_names: list[str] | None = None,
    load_pyramid: bool = True,
    round_axis: int | None = None,
    affine: np.ndarray | None = None,
    mdata: "md.MuData | None" = None,
    mdata_key: str = "if",
) -> ad.AnnData:
    """Read an OME-TIFF multiplexed IF image into an :class:`anndata.AnnData`.

    The returned object has no cell-level observations (``X`` is ``None``).
    Use :func:`astrolabe.pp.extract_if_intensities` to extract per-cell mean
    intensities from cell boundaries and populate ``X``.

    Parameters
    ----------
    path
        Path to the ``.ome.tif`` or ``.ome.tiff`` file.
    library_id
        Key under ``uns['spatial']``. Defaults to the file stem
        (e.g. ``"panel"`` for ``panel.ome.tif``).
    channel_names
        Override the channel names parsed from OME-XML metadata.  Must have
        length equal to the number of channels.  Use when the embedded names
        are uninformative (e.g. ``"Channel:0:0"``).
    load_pyramid
        If ``True`` (default), expose all pyramid levels as additional
        ``ImageSpec`` entries under ``uns['spatial'][library_id]['images']``,
        keyed as ``'level_1'``, ``'level_2'``, etc.  These share the same
        file path and differ only in which level is selected at read time.
    round_axis
        For multi-round acquisitions (CODEX/PhenoCycler), the axis index in
        the image array that represents the acquisition round.  When set,
        channels are flattened across rounds and ``var['round']`` is
        populated.  Inferred automatically from the OME-XML dimension order
        when ``None``.
    affine
        Optional ``(3, 3)`` affine matrix (homogeneous 2D) mapping image
        pixel coordinates ``(col, row)`` to the reference spatial frame in µm.
        Stored as ``ImageSpec.registration_affine``.
    mdata
        If given, add this AnnData as ``mdata[mdata_key]`` in place.
    mdata_key
        Key to use when inserting into ``mdata``.  Default ``'if'``.

    Returns
    -------
    anndata.AnnData
        AnnData with ``X=None``, channels as ``var_names``, and an
        :class:`~astrolabe._utils.ImageSpec` in
        ``uns['spatial'][library_id]['images']['full']``.

    Examples
    --------
    Read a standalone IF panel:

    >>> if_adata = ab.io.read_ome_tiff("panel.ome.tif")

    Read and register into an existing MuData (adds as ``mdata['if']``):

    >>> ab.io.read_ome_tiff("panel.ome.tif", mdata=mdata)

    Supply an affine alignment matrix from 10x's imagealignment.csv:

    >>> M = numpy.loadtxt("imagealignment.csv", delimiter=",")
    >>> ab.io.read_ome_tiff("panel.ome.tif", affine=M)
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"OME-TIFF not found: {path}")
    if not re.search(r"\.ome\.tiff?$", path.name, re.IGNORECASE):
        log.warning(
            "%s does not have an .ome.tif/.ome.tiff extension; "
            "proceeding anyway but OME-XML metadata may be absent.",
            path.name,
        )

    library_id = library_id or path.stem.split(".")[0]
    log.info("Reading OME-TIFF '%s' from %s", library_id, path)

    # ------------------------------------------------------------------ #
    # Validate affine before opening the file                             #
    # ------------------------------------------------------------------ #
    if affine is not None:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (3, 3):
            raise ValueError(
                f"affine must be a (3, 3) homogeneous 2D matrix, got shape {affine.shape}"
            )

    # ------------------------------------------------------------------ #
    # Build ImageSpec: open file eagerly for metadata, then close        #
    # ------------------------------------------------------------------ #
    spec = _extract_image_spec(path, registration_affine=affine, channel_names_override=channel_names)

    images: dict[str, ImageSpec] = {"full": spec}
    if load_pyramid and spec.n_levels > 1:
        for k in range(1, spec.n_levels):
            images[f"level_{k}"] = spec   # same file/metadata; level selected at render time

    # ------------------------------------------------------------------ #
    # Build var DataFrame (channel metadata)                              #
    # ------------------------------------------------------------------ #
    n_channels, var_df = _build_var(
        path=path,
        spec=spec,
        channel_names_override=channel_names,
        round_axis=round_axis,
    )

    # ------------------------------------------------------------------ #
    # Assemble AnnData                                                    #
    # ------------------------------------------------------------------ #
    obs = pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str))
    uns: dict = {
        "spatial": {
            library_id: {
                "coordinate_unit": "microns",
                "images": images,
                "source_path": str(path),
            }
        },
        UNS_SCHEMA_VERSION_KEY: SCHEMA_VERSION,
    }

    adata = ad.AnnData(obs=obs, var=var_df, uns=uns)
    # X intentionally left as None — populated by pp.extract_if_intensities

    log.info(
        "Loaded OME-TIFF: %d channels, pixel_size=%.4f µm, %d pyramid level(s), shape=%s",
        n_channels,
        spec.pixel_size_um,
        spec.n_levels,
        spec.full_shape,
    )

    # ------------------------------------------------------------------ #
    # Optionally insert into MuData                                       #
    # ------------------------------------------------------------------ #
    if mdata is not None:
        mdata.mod[mdata_key] = adata
        mdata.uns.setdefault("spatial", {}).update(uns["spatial"])
        # Only set default_modality if one wasn't already established (e.g. by read_xenium).
        mdata.uns.setdefault(UNS_DEFAULT_MODALITY_KEY, mdata_key)

    return adata


# ---------------------------------------------------------------------------
# ImageSpec factory
# ---------------------------------------------------------------------------


def _extract_image_spec(
    path: str | Path,
    registration_affine: np.ndarray | None = None,
    channel_names_override: list[str] | None = None,
) -> ImageSpec:
    """Open a pyramidal OME-TIFF, extract all metadata eagerly, close, return ImageSpec.

    No pixel data is read.  The returned :class:`~astrolabe._utils.ImageSpec`
    is picklable and safe to store in ``uns``.
    """
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        # series.axes is numpy-axis-order (e.g. 'CYX', 'TCYX')
        dim_order: str = series.axes
        level_shapes: list[tuple[int, ...]] = [tuple(lv.shape) for lv in series.levels]

        fh, fw = level_shapes[0][-2], level_shapes[0][-1]
        level_scales: list[tuple[float, float]] = [
            (fh / s[-2], fw / s[-1]) for s in level_shapes
        ]

        ome_meta = _parse_ome_xml(tif)
        pixel_size_um, _ = _extract_pixel_size(ome_meta)
        if np.isnan(pixel_size_um):
            pixel_size_um = 1.0   # neutral fallback; warning already emitted

        # Channel count from shape
        c_ax = dim_order.upper().index("C") if "C" in dim_order.upper() else None
        n_ch = level_shapes[0][c_ax] if c_ax is not None else 1

        # Channel names
        if channel_names_override is not None and len(channel_names_override) == n_ch:
            ch_names: list[str] = list(channel_names_override)
        else:
            channels_meta = ome_meta.get("channels", [])
            if channels_meta:
                ch_names = [
                    ch.get("name") or ch.get("fluor") or f"Channel_{i}"
                    for i, ch in enumerate(channels_meta[:n_ch])
                ]
                ch_names += [f"Channel_{i}" for i in range(len(ch_names), n_ch)]
            else:
                ch_names = [f"Channel_{i}" for i in range(n_ch)]

    return ImageSpec(
        path=str(path),
        pixel_size_um=pixel_size_um,
        dimension_order=dim_order,
        full_shape=level_shapes[0],
        level_shapes=level_shapes,
        level_scales=level_scales,
        registration_affine=registration_affine,
        channel_names=ch_names,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_ome_xml(tif) -> dict:
    """Extract a flat dict of key OME metadata from a TiffFile."""
    meta: dict = {}

    if tif.ome_metadata is None:
        log.warning("No OME-XML metadata found in TIFF; channel names will be auto-generated.")
        return meta

    try:
        ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}

        root = ET.fromstring(tif.ome_metadata)

        image_el = root.find("ome:Image", ns) or root.find(".//Image")
        if image_el is None:
            return meta

        pixels_el = image_el.find("ome:Pixels", ns) or image_el.find(".//Pixels")
        if pixels_el is None:
            return meta

        attrs = pixels_el.attrib
        meta["dimension_order"] = attrs.get("DimensionOrder", "")
        meta["size_x"] = int(attrs.get("SizeX", 0))
        meta["size_y"] = int(attrs.get("SizeY", 0))
        meta["size_c"] = int(attrs.get("SizeC", 1))
        meta["size_z"] = int(attrs.get("SizeZ", 1))
        meta["size_t"] = int(attrs.get("SizeT", 1))
        meta["physical_size_x"] = float(attrs.get("PhysicalSizeX", 0.0))
        meta["physical_size_y"] = float(attrs.get("PhysicalSizeY", 0.0))
        meta["physical_size_x_unit"] = attrs.get("PhysicalSizeXUnit", "µm")
        meta["physical_size_y_unit"] = attrs.get("PhysicalSizeYUnit", "µm")
        meta["type"] = attrs.get("Type", "")

        channels = []
        for ch_el in pixels_el.findall("ome:Channel", ns) or pixels_el.findall(".//Channel"):
            ch = {
                "name": ch_el.get("Name", ""),
                "id": ch_el.get("ID", ""),
                "fluor": ch_el.get("Fluor", ""),
                "excitation_wavelength": _safe_float(ch_el.get("ExcitationWavelength")),
                "emission_wavelength": _safe_float(ch_el.get("EmissionWavelength")),
                "samples_per_pixel": int(ch_el.get("SamplesPerPixel", 1)),
            }
            channels.append(ch)
        meta["channels"] = channels

    except Exception as exc:
        log.warning("Failed to parse OME-XML: %s", exc)

    return meta


def _build_var(
    *,
    path: str | Path,
    spec: ImageSpec,
    channel_names_override: list[str] | None,
    round_axis: int | None,
) -> tuple[int, pd.DataFrame]:
    """Build the var DataFrame for channel metadata."""
    with tifffile.TiffFile(str(path)) as tif:
        ome_meta = _parse_ome_xml(tif)

    channels_meta: list[dict] = ome_meta.get("channels", [])
    size_c: int = ome_meta.get("size_c", 0)

    # Infer n_channels from ImageSpec shape
    dim_order = spec.dimension_order.upper()
    c_ax = dim_order.index("C") if "C" in dim_order else None
    if c_ax is not None:
        n_channels = spec.full_shape[c_ax]
    elif size_c > 0:
        n_channels = size_c
    else:
        n_channels = spec.full_shape[0] if len(spec.full_shape) >= 3 else 1

    if channel_names_override is not None:
        if len(channel_names_override) != n_channels:
            raise ValueError(
                f"channel_names has {len(channel_names_override)} entries "
                f"but image has {n_channels} channels."
            )
        names = list(channel_names_override)
    elif channels_meta:
        names = [
            ch.get("name") or ch.get("fluor") or f"Channel_{i}"
            for i, ch in enumerate(channels_meta[:n_channels])
        ]
        names += [f"Channel_{i}" for i in range(len(names), n_channels)]
    else:
        names = [f"Channel_{i}" for i in range(n_channels)]

    names = _deduplicate_names(names)

    excitation = [
        _safe_float(channels_meta[i].get("excitation_wavelength")) if i < len(channels_meta) else np.nan
        for i in range(n_channels)
    ]
    emission = [
        _safe_float(channels_meta[i].get("emission_wavelength")) if i < len(channels_meta) else np.nan
        for i in range(n_channels)
    ]

    var_df = pd.DataFrame(
        {
            "channel_name": names,
            "channel": np.arange(n_channels, dtype=np.int32),
            "round": np.full(n_channels, np.nan),
            "excitation_wavelength_nm": excitation,
            "emission_wavelength_nm": emission,
        },
        index=pd.Index(names, name="channel_name"),
    )

    if round_axis is not None and round_axis < len(spec.full_shape):
        n_rounds = spec.full_shape[round_axis]
        n_channels_per_round = n_channels // n_rounds
        rounds = np.repeat(np.arange(n_rounds, dtype=np.int32), n_channels_per_round)
        channels_within_round = np.tile(
            np.arange(n_channels_per_round, dtype=np.int32), n_rounds
        )
        if len(rounds) != n_channels:
            log.warning(
                "round_axis=%d gives %d rounds × %d ch/round = %d, "
                "but image has %d channels; skipping var['round'] / var['channel'].",
                round_axis, n_rounds, n_channels_per_round,
                n_rounds * n_channels_per_round, n_channels,
            )
        if len(rounds) == n_channels:
            var_df["round"] = rounds
            var_df["channel"] = channels_within_round

    return n_channels, var_df


def _extract_pixel_size(ome_meta: dict) -> tuple[float, str]:
    """Return (pixel_size_um, original_unit_string)."""
    px = ome_meta.get("physical_size_x", 0.0)
    unit = ome_meta.get("physical_size_x_unit", "µm")

    if px == 0.0:
        log.warning(
            "PhysicalSizeX is 0 or absent in OME metadata. "
            "Pixel size will be set to NaN. Set manually via "
            "spec.pixel_size_um = <value> after loading."
        )
        return float("nan"), unit

    unit_lc = unit.lower().strip()
    if unit_lc in _MICRON_UNITS:
        return float(px), unit
    elif unit_lc in ("nm", "nanometer", "nanometers", "nanometre", "nanometres"):
        return float(px) / 1000.0, unit
    elif unit_lc in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return float(px) * 1000.0, unit
    else:
        log.warning("Unknown pixel size unit %r; assuming microns.", unit)
        return float(px), unit


def _infer_dimension_order(ndim: int) -> str:
    """Return a plausible numpy-axis dimension order for images without OME-XML."""
    defaults = {2: "YX", 3: "CYX", 4: "CZYX", 5: "TCZYX"}
    return defaults.get(ndim, "YX")


def _deduplicate_names(names: list[str]) -> list[str]:
    """Append _2, _3, etc. to duplicate channel names."""
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 1
            result.append(name)
    return result


def _safe_float(val) -> float:
    """Parse a value to float; return NaN on failure."""
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")
