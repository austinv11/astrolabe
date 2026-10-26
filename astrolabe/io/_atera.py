"""
Reader for 10x Genomics Atera output bundles.

Supports Atera Onboard Analysis (v1.0+) and Atera WTA Preview output
directories where primary matrix, cell segmentation, analysis, and transcript
outputs are stored in Zarr v2 archives (``*.zarr.zip`` or ``*.zarr``) and/or
Parquet files alongside ``experiment.spatial`` / ``experiment.xenium``,
``panel_config.json``, and ``morphology_2d/`` / ``morphology_focus/`` /
``morphology_3d/`` OME-TIFF images.  Shares 10x feature annotation,
control-probe flagging, optional file resolution, H&E alignment, and Parquet
fallback helpers with :mod:`astrolabe.io._xenium`.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import logging
from pathlib import Path
import tarfile
from typing import Iterator

import anndata as ad
import awkward as ak
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp

from astrolabe._constants import (
    SCHEMA_VERSION,
    UNS_DEFAULT_MODALITY_KEY,
    UNS_SCHEMA_VERSION_KEY,
    UNS_XENIUM_SEGMENTATION_KEY,
)
from astrolabe.io._ome_tiff import _extract_image_spec
from astrolabe.io._xenium import (
    _BOUNDARIES_CANDIDATES,
    _CELLS_CANDIDATES,
    _NUCLEUS_BOUNDARIES_CANDIDATES,
    _OBS_ALWAYS,
    _annotate_xenium_var,
    _attach_boundaries,
    _attach_he_image,
    _attach_obs,
    _build_control_mask_arrow,
    _read_cells_table,
    _resolve_optional_file,
    _source_path_from_mdata,
)

__all__ = ["read_atera", "read_atera_transcripts"]

log = logging.getLogger(__name__)

# Indicators of a valid Atera output directory
_BUNDLE_INDICATORS = (
    "experiment.spatial",
    "experiment.xenium",
    "cell_feature_matrix.zarr.zip",
    "cell_feature_matrix.zarr",
    "csc_cell_feature_matrix.zarr.zip",
    "csc_cell_feature_matrix.zarr",
    "cells.zarr.zip",
    "cells.zarr",
    "transcripts.zarr.zip",
    "transcripts.zarr",
)

# Normalize snake_case feature_type strings in 10x Zarr stores to the canonical
# 10x title-case names expected by _flag_control_probes and
# astrolabe.pp.calculate_xenium_qc_metrics.
_FEATURE_TYPE_MAP = {
    "gene": "Gene Expression",
    "gene_expression": "Gene Expression",
    "negative_control_probe": "Negative Control Probe",
    "negative_control_codeword": "Negative Control Codeword",
    "unassigned_codeword": "Unassigned Codeword",
    "deprecated_codeword": "Deprecated Codeword",
    "genomic_control": "Genomic Control",
}

# Primary cell-level columns in cell_feature_matrix.zarr.zip /obs kept by default
# when load_onboard_analysis=False (extends Xenium's _OBS_ALWAYS with total_counts,
# cell_id, z_level, nucleus_count).
_PRIMARY_OBS_COLUMNS = frozenset(
    [*_OBS_ALWAYS, "cell_id", "total_counts", "z_level", "nucleus_count"]
)

# Primary gene-level columns in cell_feature_matrix.zarr.zip /var kept by default
# when load_onboard_analysis=False.
_PRIMARY_VAR_COLUMNS = frozenset({
    "feature_id",
    "feature_name",
    "feature_type",
    "genome",
})


def read_atera(
    path: str | Path,
    *,
    library_id: str | None = None,
    var_names: str = "gene_name",
    dtype: str = "float32",
    load_boundaries: bool = True,
    load_transcripts: bool = True,
    load_onboard_analysis: bool = False,
    morphology_image: bool | str | Path = True,
    morphology_3d: bool | str | Path = False,
    he_image: bool | str | Path = False,
    he_alignment: bool | str | Path | None = True,
) -> md.MuData:
    """Read a 10x Genomics Atera output bundle into a :class:`~mudata.MuData`.

    Cells and genes are **not** filtered; control probes (``NegControlProbe_*``,
    ``NegControlCodeword_*``, ``UnassignedCodeword_*``, etc.) are retained in
    ``X`` and flagged in ``var['is_control_probe']``.  All spatial coordinates
    (centroids and polygon vertices) are stored in physical **microns (µm)**.

    Parameters
    ----------
    path
        Path to the Atera output directory (or its parent if an ``outs/``
        subdirectory is present).
    library_id
        Identifier key under ``uns['spatial']``.  Defaults to the directory
        basename (or parent basename when reading from ``outs/``).
    var_names
        Column to use as ``adata.var_names``: ``'gene_name'`` (default, human-
        readable gene symbols from ``feature_name``) or ``'gene_id'`` (Ensembl
        IDs from ``feature_id``).
    dtype
        NumPy dtype for ``X`` (default ``'float32'``).
    load_boundaries
        Load cell and nucleus boundary polygons from ``cells.zarr.zip`` (or
        Parquet fallback) into ``obsm['cell_boundaries']`` and
        ``obsm['nucleus_boundaries']`` as Awkward ragged arrays.
    load_transcripts
        If ``True`` (default), automatically load transcripts (from
        ``transcripts.parquet`` or ``transcripts.zarr.zip``) into
        ``uns['transcripts']`` via :func:`read_atera_transcripts` when present
        in the bundle (skipped gracefully if absent).  Pass ``False`` to skip
        transcript loading at read time; call :func:`read_atera_transcripts`
        later when needed.
    load_onboard_analysis
        If ``False`` (default), keep only primary cell/gene metadata in ``obs``
        and ``var`` (matching :func:`~astrolabe.io.read_xenium`).  If ``True``,
        also load onboard secondary analysis from
        ``cell_feature_matrix.zarr.zip`` (or ``analysis.zarr.zip`` /
        ``analysis.tar.gz``): ``obs`` clustering/annotation columns
        (``leiden_*``, ``azimuth_*``, ``gene_expression_*``, ``filtered``,
        ``is_sketch``), ``obsm`` embeddings (``X_pca``, ``X_umap``), and
        ``var`` / ``varm`` differential expression arrays (``de_*``,
        ``highly_variable``, ``filtered``).
    morphology_image
        ``True`` (default) attaches all 2-D multi-focus channel OME-TIFFs in
        ``morphology_2d/*.ome.tif`` (or ``morphology_focus/*.ome.tif``) as
        ``'morphology_2d_<stem>'`` and aliases the primary DAPI channel
        (``ch0000_dapi.ome.tif``, or the first file) as ``'full'`` and
        ``'morphology'``.  Pass a filename string/Path to attach a specific
        file as ``'full'``, or ``False`` to skip.
    morphology_3d
        ``False`` (default) skips 3-D morphology stacks.  ``True`` attaches
        ``morphology_3d/*.ome.tif`` (or ``morphology.ome.tif`` fallback) under
        ``'morphology_3d_<stem>'`` and ``'morphology_3d'``.  Pass a filename
        string/Path to attach a specific 3-D OME-TIFF under ``'morphology_3d'``.
    he_image
        ``False`` (default) skips post-run H&E loading.  ``True`` auto-detects
        ``he_image.ome.tif`` in the bundle and attaches it as a separate
        ``mdata['he']`` modality if found.  Pass a filename string/Path to load
        a specific file.
    he_alignment
        ``True`` (default) auto-detects ``he_imagealignment.csv`` for
        *he_image*.  Pass a filename string/Path to use a specific CSV, or
        ``False`` to attach the H&E image unregistered.

    Returns
    -------
    mudata.MuData
        Container with the ``'atera'`` modality (and optionally ``'he'``) and
        ``mdata.uns['default_modality'] = 'atera'``.
    """
    bundle = _resolve_bundle_dir(Path(path))

    if library_id is None:
        library_id = bundle.parent.name if bundle.name == "outs" else bundle.name

    log.info("Reading Atera bundle from %s (library_id=%r)", bundle, library_id)

    # 1. Experiment metadata (experiment.spatial / experiment.xenium + panel_config.json)
    experiment_meta, pixel_size_um = _read_experiment_spatial(bundle)
    panel_config = _read_panel_config(bundle)

    # 2. Count matrix, obs, var, obsm from cell_feature_matrix.zarr.zip
    adata = _read_counts(
        bundle,
        var_names=var_names,
        dtype=dtype,
        load_onboard_analysis=load_onboard_analysis,
    )

    # 3. Cell metadata & polygon boundaries from cells.zarr.zip (and/or cells.parquet)
    _attach_cells_and_boundaries(
        adata,
        bundle,
        load_boundaries=load_boundaries,
    )

    # 3b. External onboard analysis (analysis.zarr.zip / analysis.tar.gz) when requested
    if load_onboard_analysis:
        _attach_external_onboard_analysis(adata, bundle)

    # 4. Spatial metadata & lazy morphology image specs
    spatial_entry: dict = {
        "coordinate_unit": "microns",
        "scalefactors": {
            "spot_diameter_fullres": 1.0,
            "tissue_hires_scalef": 1.0,
            "fiducial_diameter_fullres": 1.0,
        },
        "metadata": {
            "chemistry_description": experiment_meta.get("chemistry_version", "Atera"),
            "software_version": (
                experiment_meta.get("analysis_sw_version")
                or experiment_meta.get("instrument_sw_version", "")
            ),
            "pixel_size_um": pixel_size_um,
        },
        "experiment": experiment_meta,
        "source_path": str(bundle.resolve()),
        "images": {},
    }
    if panel_config is not None:
        spatial_entry["panel_config"] = panel_config

    if morphology_image is not False:
        _attach_morphology_2d(
            spatial_entry["images"],
            bundle,
            morphology_image,
            pixel_size_um=pixel_size_um,
        )

    if morphology_3d is not False:
        _attach_morphology_3d(
            spatial_entry["images"],
            bundle,
            morphology_3d,
            pixel_size_um=pixel_size_um,
        )

    adata.uns["spatial"] = {library_id: spatial_entry}
    adata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    mdata = md.MuData({"atera": adata})
    mdata.uns[UNS_DEFAULT_MODALITY_KEY] = "atera"
    mdata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    mdata.uns["spatial"] = dict(adata.uns["spatial"])

    if he_image is not False:
        _attach_he_image(
            mdata, bundle, he_image, he_alignment, pixel_size_um=pixel_size_um
        )

    if load_transcripts:
        tx_path = _find_zarr_or_parquet(
            bundle,
            ("transcripts.parquet", "transcripts.zarr.zip", "transcripts.zarr"),
        )
        if tx_path is not None:
            read_atera_transcripts(mdata, bundle)
        else:
            log.debug("No transcripts file in %s; skipping uns['transcripts'].", bundle)

    log.info("Loaded Atera data: %d cells × %d genes", adata.n_obs, adata.n_vars)
    return mdata


# ---------------------------------------------------------------------------
# Bundle & Zarr helpers
# ---------------------------------------------------------------------------


def _resolve_bundle_dir(path: Path) -> Path:
    """Resolve the directory that directly contains the Atera output files."""
    if not path.exists():
        raise FileNotFoundError(f"Atera path does not exist: {path}")

    for candidate in (path, path / "outs"):
        if candidate.is_dir() and any((candidate / ind).exists() for ind in _BUNDLE_INDICATORS):
            return candidate

    raise FileNotFoundError(
        f"Could not find Atera output files in '{path}' or '{path / 'outs'}'. "
        f"Expected at least one of: {_BUNDLE_INDICATORS}"
    )


def _find_zarr_or_parquet(bundle: Path, names: tuple[str, ...]) -> Path | None:
    """Return the first existing path among *names* inside *bundle*, or ``None``."""
    for name in names:
        candidate = bundle / name
        if candidate.exists():
            return candidate
    return None


@contextmanager
def _open_zarr_group(path: Path) -> Iterator:
    """Open a Zarr group from either a ``.zarr.zip`` file or a ``.zarr`` directory."""
    import zarr

    if path.is_file() and path.name.endswith(".zip"):
        store = zarr.storage.ZipStore(str(path), mode="r")
        try:
            yield zarr.open_group(store=store, mode="r")
        finally:
            store.close()
    else:
        yield zarr.open_group(store=str(path), mode="r")


def _decode_string_array(arr: np.ndarray) -> np.ndarray:
    """Convert a Zarr array of bytes, object, StringDType ('T'), or unicode ('U') into a 1-D object str ndarray."""
    data = np.asarray(arr)
    if data.dtype.kind == "S":
        return np.char.decode(data, "utf-8").astype(object)
    if data.dtype.kind == "T":
        return data.astype(object)
    if data.dtype.kind == "O":
        return np.array(
            [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in data],
            dtype=object,
        )
    if data.dtype.kind == "U":
        return data.astype(object)
    return data.astype(object)


def _decode_10x_cell_id(cell_id_arr: np.ndarray) -> np.ndarray:
    """Convert 10x's ``(N, 2)`` ``uint32`` ``[prefix, suffix]`` array to string barcodes.

    10x encodes each cell barcode as an 8-digit hex representation of
    ``prefix`` with digits ``0-9a-f`` shifted to ``a-p``, followed by
    ``-{suffix}`` (e.g. ``(4022, 1) -> '00000fb6' -> 'aaaaaplg-1'``).
    Also passes 1-D string or integer arrays through cleanly.
    """
    arr = np.asarray(cell_id_arr)
    if arr.ndim == 1:
        return _decode_string_array(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2) cell_id array or 1-D array, got shape {arr.shape}")

    n = arr.shape[0]
    if n == 0:
        return np.empty(0, dtype=object)

    prefix = arr[:, 0].astype(np.uint32)
    suffix = arr[:, 1].astype(np.uint32)

    shifts = np.array([28, 24, 20, 16, 12, 8, 4, 0], dtype=np.uint32)
    nibbles = ((prefix[:, None] >> shifts) & np.uint32(0xF)).astype(np.uint8) + ord("a")
    prefix_strs = np.char.decode(nibbles.view(f"|S8").ravel(), "ascii")
    suffix_strs = suffix.astype(str)
    return (prefix_strs.astype(object) + "-" + suffix_strs.astype(object))


def _normalize_feature_types(ft_series: pd.Series) -> pd.Series:
    """Normalize snake_case 10x Zarr feature_type strings to canonical 10x names."""
    return ft_series.astype(str).map(lambda x: _FEATURE_TYPE_MAP.get(x.lower(), x))


# ---------------------------------------------------------------------------
# Metadata JSONs (experiment.spatial & panel_config.json)
# ---------------------------------------------------------------------------


def _read_experiment_spatial(bundle: Path) -> tuple[dict, float]:
    """Parse ``experiment.spatial`` (or fallback ``experiment.xenium``)."""
    exp_path = bundle / "experiment.spatial"
    if not exp_path.exists():
        exp_path = bundle / "experiment.xenium"
    if not exp_path.exists():
        log.warning("experiment.spatial not found in %s; using default pixel_size=0.2125 µm", bundle)
        return {}, 0.2125

    with open(exp_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    pixel_size_um = float(
        meta.get("pixel_size", meta.get("pixel_size_um", 0.2125))
    )
    return meta, pixel_size_um


def _read_panel_config(bundle: Path) -> dict | list | None:
    """Parse ``panel_config.json`` (or ``gene_panel.json``) if present in *bundle*."""
    for name in ("panel_config.json", "gene_panel.json"):
        panel_path = bundle / name
        if panel_path.exists():
            try:
                with open(panel_path, encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:
                log.warning("Failed to parse %s in %s: %s", name, bundle, exc)
                return None
    return None


# ---------------------------------------------------------------------------
# Count matrix, obs, var, obsm (cell_feature_matrix.zarr.zip)
# ---------------------------------------------------------------------------


def _read_zarr_table(
    grp,
    *,
    allowed_columns: frozenset[str] | None = None,
    extra_keep: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, np.ndarray]]:
    """Read an ``/obs`` or ``/var`` Zarr group into a DataFrame.

    Supports both standard ``anndata`` Zarr groups and 10x Atera Zarr groups
    (where categorical columns are subgroups with ``/categories``, ``/codes``,
    and optional ``/color`` hex palettes, and multi-column DE arrays like
    ``de_leiden_*`` may be 2-D arrays routed to ``varm``/``obsm``).
    """
    import zarr

    cols: dict[str, pd.Series | np.ndarray | pd.Categorical] = {}
    colors_uns: dict[str, list[str]] = {}
    multi_dim: dict[str, np.ndarray] = {}

    index_key = grp.attrs.get("_index")
    keys = list(grp.keys())
    col_order = grp.attrs.get("column-order")
    if col_order is not None:
        ordered = [k for k in col_order if k in keys]
        remaining = [k for k in keys if k not in ordered]
        keys = ordered + remaining

    for key in keys:
        if (
            allowed_columns is not None
            and key not in allowed_columns
            and key not in extra_keep
            and key != index_key
        ):
            continue

        item = grp[key]
        if isinstance(item, zarr.Group):
            if "categories" in item and "codes" in item:
                cats = _decode_string_array(item["categories"][:])
                codes = np.asarray(item["codes"][:]).ravel()
                cols[key] = pd.Categorical.from_codes(codes, categories=pd.Index(cats))
                if "color" in item:
                    colors_uns[f"{key}_colors"] = [
                        str(c) for c in _decode_string_array(item["color"][:])
                    ]
        else:
            raw = np.asarray(item[:])
            if raw.ndim == 2 and raw.shape[1] == 1:
                raw = raw.ravel()
            if raw.ndim > 1:
                multi_dim[key] = (
                    _decode_string_array(raw)
                    if raw.dtype.kind in ("S", "O", "U", "T")
                    else raw
                )
            elif raw.dtype.kind in ("S", "O", "U", "T"):
                cols[key] = _decode_string_array(raw)
            else:
                cols[key] = raw

    df = pd.DataFrame(cols)
    if index_key and index_key in df.columns:
        df.index = pd.Index(df.pop(index_key).astype(object), name=index_key)
    return df, colors_uns, multi_dim


def _read_cell_features_v4_group(
    cf_grp,
    *,
    var_names: str,
    dtype: str,
) -> ad.AnnData:
    """Read 10x's ``/cell_features`` Zarr group (v4.x schema used in WTA Preview)."""
    n_obs = int(cf_grp.attrs["number_cells"])
    n_vars_raw = int(cf_grp.attrs["number_features"])

    if "cell_id" in cf_grp:
        barcodes = _decode_10x_cell_id(cf_grp["cell_id"][:])
    else:
        barcodes = np.arange(n_obs).astype(str)
    obs = pd.DataFrame(index=pd.Index(barcodes.astype(object), name="cell_id"))

    feature_ids = [str(x) for x in cf_grp.attrs.get("feature_ids", [])]
    feature_keys = [str(x) for x in cf_grp.attrs.get("feature_keys", feature_ids)]
    feature_types_raw = [str(x) for x in cf_grp.attrs.get("feature_types", ["gene"] * n_vars_raw)]

    # Read CSR count matrix from cf_grp['csc'] (where indptr has length n_obs + 1)
    # or CSC count matrix from cf_grp['data', 'indices', 'indptr'] (length n_vars_raw + 1)
    if "csc" in cf_grp and cf_grp["csc"]["indptr"].shape[0] == n_obs + 1:
        csc_sub = cf_grp["csc"]
        data = csc_sub["data"][:].astype(np.float32)
        indices = csc_sub["indices"][:]
        indptr = csc_sub["indptr"][:]
        X = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars_raw))
    else:
        data = cf_grp["data"][:].astype(np.float32)
        indices = cf_grp["indices"][:]
        indptr = cf_grp["indptr"][:]
        if len(indptr) == n_vars_raw + 1:
            X = sp.csc_matrix((data, indices, indptr), shape=(n_obs, n_vars_raw)).tocsr()
        else:
            X = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars_raw))

    # Drop synthetic 'aggregate_gene' summary feature (e.g. 'Total transcripts')
    keep_feat = np.array([ft.lower() != "aggregate_gene" for ft in feature_types_raw], dtype=bool)
    if not np.all(keep_feat):
        X = X[:, keep_feat]
        feature_ids = [fid for fid, k in zip(feature_ids, keep_feat) if k]
        feature_keys = [fk for fk, k in zip(feature_keys, keep_feat) if k]
        feature_types_raw = [ft for ft, k in zip(feature_types_raw, keep_feat) if k]

    norm_ft = _normalize_feature_types(pd.Series(feature_types_raw)).to_numpy()
    var = pd.DataFrame(
        {
            "gene_id": feature_ids,
            "feature_type": norm_ft,
        },
        index=pd.Index(feature_keys, name="gene_name"),
    )

    # Populate standard per-cell count columns from X (can be refined by cells.parquet if present)
    gene_mask = norm_ft == "Gene Expression"
    probe_mask = norm_ft == "Negative Control Probe"
    cw_mask = norm_ft == "Negative Control Codeword"
    obs["transcript_counts"] = np.asarray(X[:, gene_mask].sum(axis=1)).ravel().astype(np.int32)
    obs["control_probe_counts"] = np.asarray(X[:, probe_mask].sum(axis=1)).ravel().astype(np.int32)
    obs["control_codeword_counts"] = np.asarray(X[:, cw_mask].sum(axis=1)).ravel().astype(np.int32)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    if var_names == "gene_name":
        adata.var_names_make_unique()
    return _annotate_xenium_var(adata, var_names=var_names, dtype=dtype)


def _read_counts(
    bundle: Path,
    *,
    var_names: str = "gene_name",
    dtype: str = "float32",
    load_onboard_analysis: bool = False,
) -> ad.AnnData:
    """Load ``cell_feature_matrix.zarr.zip`` (or CSC fallback) into an AnnData."""
    if var_names not in ("gene_name", "gene_id"):
        raise ValueError(f"var_names must be 'gene_name' or 'gene_id', got {var_names!r}")

    zarr_path = _find_zarr_or_parquet(
        bundle,
        (
            "cell_feature_matrix.zarr.zip",
            "cell_feature_matrix.zarr",
            "csc_cell_feature_matrix.zarr.zip",
            "csc_cell_feature_matrix.zarr",
        ),
    )
    if zarr_path is None:
        raise FileNotFoundError(
            f"No cell_feature_matrix.zarr.zip found in {bundle}"
        )

    is_csc_file = zarr_path.name.startswith("csc_")

    with _open_zarr_group(zarr_path) as root:
        if "cell_features" in root and "X" not in root:
            return _read_cell_features_v4_group(
                root["cell_features"],
                var_names=var_names,
                dtype=dtype,
            )

        # --- 1. Read obs ---
        obs_allowed = None if load_onboard_analysis else _PRIMARY_OBS_COLUMNS
        obs_extra = frozenset({"barcode", "_index", "centroid_column", "centroid_row", "x_centroid", "y_centroid"})
        if "obs" in root:
            obs, obs_colors, obs_multi = _read_zarr_table(
                root["obs"],
                allowed_columns=obs_allowed,
                extra_keep=obs_extra,
            )
        else:
            obs, obs_colors, obs_multi = pd.DataFrame(), {}, {}

        # Resolve obs index (cell_id barcode string)
        if "barcode" in obs.columns:
            barcodes = obs.pop("barcode").astype(object)
            obs.index = pd.Index(barcodes, name="cell_id")
            if "cell_id" in obs.columns and not load_onboard_analysis:
                obs.drop(columns=["cell_id"], inplace=True)
        elif "_index" in obs.columns:
            barcodes = obs.pop("_index").astype(object)
            obs.index = pd.Index(barcodes, name="cell_id")
            if "cell_id" in obs.columns and not load_onboard_analysis:
                obs.drop(columns=["cell_id"], inplace=True)
        elif obs.index.name is not None:
            obs.index = obs.index.astype(object)
            obs.index.name = "cell_id"
            if "cell_id" in obs.columns and not load_onboard_analysis:
                obs.drop(columns=["cell_id"], inplace=True)
        elif "cell_id" in obs.columns:
            cell_ids = obs.pop("cell_id")
            obs.index = pd.Index(cell_ids.astype(object), name="cell_id")
        elif "cell_id" in obs_multi:
            decoded_ids = _decode_10x_cell_id(obs_multi.pop("cell_id"))
            obs.index = pd.Index(decoded_ids.astype(object), name="cell_id")

        # Extract spatial coordinates from obs if present (centroid_column=X, centroid_row=Y)
        spatial_coords: np.ndarray | None = None
        if "centroid_column" in obs.columns and "centroid_row" in obs.columns:
            x = obs.pop("centroid_column").to_numpy(dtype=np.float32)
            y = obs.pop("centroid_row").to_numpy(dtype=np.float32)
            spatial_coords = np.column_stack([x, y])
        elif "x_centroid" in obs.columns and "y_centroid" in obs.columns:
            x = obs.pop("x_centroid").to_numpy(dtype=np.float32)
            y = obs.pop("y_centroid").to_numpy(dtype=np.float32)
            spatial_coords = np.column_stack([x, y])

        # --- 2. Read var ---
        var_allowed = None if load_onboard_analysis else _PRIMARY_VAR_COLUMNS
        var_extra = frozenset({"_index", "gene_id", "gene_name", "gene_ids", "feature_types"})
        if "var" in root:
            var, _, var_multi = _read_zarr_table(
                root["var"],
                allowed_columns=var_allowed,
                extra_keep=var_extra,
            )
        else:
            var, var_multi = pd.DataFrame(), {}

        # When _read_zarr_table pops _index='feature_id', restore it as 'gene_id' column
        if var.index.name in ("feature_id", "gene_id") and "gene_id" not in var.columns:
            var["gene_id"] = var.index.astype(object).to_numpy()
        elif "feature_id" in var.columns and "gene_id" not in var.columns:
            var = var.rename(columns={"feature_id": "gene_id"})

        if "feature_types" in var.columns and "feature_type" not in var.columns:
            var = var.rename(columns={"feature_types": "feature_type"})
        if "feature_type" in var.columns:
            var["feature_type"] = _normalize_feature_types(var["feature_type"]).to_numpy()

        if "feature_name" in var.columns:
            gene_names = var.pop("feature_name").astype(object).to_numpy()
        elif "gene_name" in var.columns:
            gene_names = var.pop("gene_name").astype(object).to_numpy()
        elif var.index.name is not None:
            gene_names = var.index.astype(object).to_numpy()
        elif "gene_id" in var.columns:
            gene_names = var["gene_id"].astype(object).to_numpy()
        else:
            gene_names = np.arange(len(var)).astype(str).astype(object)

        var.index = pd.Index(gene_names, name="gene_name")
        if "gene_id" not in var.columns:
            var["gene_id"] = gene_names

        # --- 3. Read X ---
        x_node = root["X"]
        n_obs = len(obs)
        n_vars = len(var)
        import zarr

        if isinstance(x_node, zarr.Group):
            data = x_node["data"][:].astype(np.float32)
            indices = x_node["indices"][:]
            indptr = x_node["indptr"][:]
            enc_type = x_node.attrs.get("encoding-type", "")

            if enc_type == "csc_matrix" or (is_csc_file and enc_type != "csr_matrix"):
                if len(indptr) == n_vars + 1:
                    X = sp.csc_matrix((data, indices, indptr), shape=(n_obs, n_vars)).tocsr()
                else:
                    X = sp.csc_matrix((data, indices, indptr), shape=(n_vars, n_obs)).T.tocsr()
            else:
                if len(indptr) == n_obs + 1:
                    X = sp.csr_matrix((data, indices, indptr), shape=(n_obs, n_vars))
                else:
                    X = sp.csc_matrix((data, indices, indptr), shape=(n_obs, n_vars)).tocsr()
        else:
            X = sp.csr_matrix(np.asarray(x_node[:], dtype=np.float32))

        # --- 4. Read obsm & varm (when load_onboard_analysis=True) ---
        obsm: dict[str, np.ndarray] = {}
        varm: dict[str, np.ndarray] = {}
        if spatial_coords is not None:
            obsm["spatial"] = spatial_coords

        if load_onboard_analysis:
            for k, arr in obs_multi.items():
                if k != "cell_id":
                    obsm[k] = arr
            for k, arr in var_multi.items():
                varm[k] = arr
            if "obsm" in root:
                obsm_grp = root["obsm"]
                for key in obsm_grp.keys():
                    item = obsm_grp[key]
                    if not isinstance(item, zarr.Group):
                        arr = np.asarray(item[:])
                        if arr.dtype.kind == "f":
                            arr = arr.astype(np.float32, copy=False)
                        obsm[key] = arr

    adata = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, varm=varm)
    if var_names == "gene_name":
        adata.var_names_make_unique()

    # Reuse _xenium._annotate_xenium_var for var_names swap, _flag_control_probes, and dtype
    adata = _annotate_xenium_var(adata, var_names=var_names, dtype=dtype)

    if load_onboard_analysis and obs_colors:
        for color_key, palette in obs_colors.items():
            adata.uns[color_key] = palette

    return adata


def _attach_external_onboard_analysis(adata: ad.AnnData, bundle: Path) -> None:
    """Attach clustering and embeddings from ``analysis.zarr.zip`` / ``analysis.tar.gz`` if present."""
    analysis_zarr = _find_zarr_or_parquet(bundle, ("analysis.zarr.zip", "analysis.zarr"))
    if analysis_zarr is not None:
        try:
            with _open_zarr_group(analysis_zarr) as root:
                if "cell_groups" in root:
                    cg = root["cell_groups"]
                    grouping_names = cg.attrs.get("grouping_names", [])
                    group_names_list = cg.attrs.get("group_names", [])
                    n_obs = adata.n_obs
                    for idx, g_name in enumerate(grouping_names):
                        key = str(idx)
                        if g_name in adata.obs.columns or key not in cg:
                            continue
                        grp = cg[key]
                        indices = np.asarray(grp["indices"][:], dtype=np.int64).ravel()
                        indptr = np.asarray(grp["indptr"][:], dtype=np.int64).ravel()
                        labels = (
                            [str(x) for x in group_names_list[idx]]
                            if idx < len(group_names_list)
                            else [str(i + 1) for i in range(len(indptr) - 1)]
                        )
                        codes = np.full(n_obs, -1, dtype=np.int32)
                        for c_i in range(len(indptr) - 1):
                            start, end = int(indptr[c_i]), int(indptr[c_i + 1])
                            cell_pos = indices[start:end]
                            valid = (cell_pos >= 0) & (cell_pos < n_obs)
                            codes[cell_pos[valid]] = c_i
                        adata.obs[str(g_name)] = pd.Categorical.from_codes(
                            codes, categories=pd.Index(labels)
                        )
        except Exception as exc:
            log.warning("Failed to read onboard analysis from %s: %s", analysis_zarr.name, exc)

    # Check for PCA / UMAP CSV projections in analysis/ directory or analysis.tar.gz
    umap_rel = "analysis/umap/gene_expression_2_components/projection.csv"
    pca_rel = "analysis/pca/gene_expression_10_components/projection.csv"

    for obsm_key, rel_path in (("X_umap", umap_rel), ("X_pca", pca_rel)):
        if obsm_key in adata.obsm:
            continue
        csv_file = bundle / rel_path
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file, index_col=0)
                df.index = df.index.astype(str)
                adata.obsm[obsm_key] = df.reindex(adata.obs_names).to_numpy(dtype=np.float32)
            except Exception as exc:
                log.warning("Failed to load %s from %s: %s", obsm_key, csv_file, exc)
        elif (bundle / "analysis.tar.gz").exists():
            try:
                with tarfile.open(bundle / "analysis.tar.gz", "r:gz") as tf:
                    if rel_path in tf.getnames():
                        extracted = tf.extractfile(rel_path)
                        if extracted is not None:
                            df = pd.read_csv(extracted, index_col=0)
                            df.index = df.index.astype(str)
                            adata.obsm[obsm_key] = df.reindex(adata.obs_names).to_numpy(
                                dtype=np.float32
                            )
            except Exception as exc:
                log.warning("Failed to load %s from analysis.tar.gz: %s", obsm_key, exc)


# ---------------------------------------------------------------------------
# Cells & polygon boundaries (cells.zarr.zip + Parquet fallback)
# ---------------------------------------------------------------------------


def _attach_cells_and_boundaries(
    adata: ad.AnnData,
    bundle: Path,
    *,
    load_boundaries: bool = True,
) -> None:
    """Populate missing spatial/summary columns and polygon sets from ``cells.zarr.zip``.

    Uses Xenium's Parquet/CSV readers (:func:`_read_cells_table`,
    :func:`_attach_obs`, :func:`_attach_boundaries`) to populate any missing
    ``_OBS_ALWAYS`` columns when ``cells.parquet`` is present alongside or
    instead of ``cells.zarr.zip``.
    """
    # If cells.parquet / cells.csv.gz is present and obs lacks standard Xenium/Atera
    # per-cell count breakdown columns (e.g. cell_area, nucleus_area, transcript_counts),
    # merge them via _xenium._read_cells_table.
    if any((bundle / c).exists() for c in _CELLS_CANDIDATES) and not set(
        _OBS_ALWAYS
    ).issubset(adata.obs.columns):
        cells_df = _read_cells_table(bundle)
        existing_obs = adata.obs.copy()
        _attach_obs(adata, cells_df)
        for col in existing_obs.columns:
            if col not in adata.obs.columns:
                adata.obs[col] = existing_obs[col]

    cells_path = _find_zarr_or_parquet(bundle, ("cells.zarr.zip", "cells.zarr"))
    if cells_path is None:
        if "spatial" not in adata.obsm:
            log.warning("cells.zarr.zip not found in %s and spatial coordinates absent from obs", bundle)

        if load_boundaries:
            if any((bundle / c).exists() for c in _BOUNDARIES_CANDIDATES):
                _attach_boundaries(adata, bundle, _BOUNDARIES_CANDIDATES, "cell_boundaries")
            if any((bundle / c).exists() for c in _NUCLEUS_BOUNDARIES_CANDIDATES):
                _attach_boundaries(adata, bundle, _NUCLEUS_BOUNDARIES_CANDIDATES, "nucleus_boundaries")
            if "cell_boundaries" in adata.obsm or "nucleus_boundaries" in adata.obsm:
                adata.uns[UNS_XENIUM_SEGMENTATION_KEY] = True
        return

    with _open_zarr_group(cells_path) as root:
        # 1. Decode cell barcodes from /cell_id if present
        if "cell_id" in root:
            cell_barcodes = _decode_10x_cell_id(root["cell_id"][:])
        else:
            cell_barcodes = adata.obs_names.to_numpy()

        # 2. Extract /cell_summary columns if present
        if "cell_summary" in root:
            summary_node = root["cell_summary"]
            summary_arr = np.asarray(summary_node[:])
            col_names = summary_node.attrs.get(
                "column_names",
                [
                    "cell_centroid_x",
                    "cell_centroid_y",
                    "cell_area",
                    "nucleus_centroid_x",
                    "nucleus_centroid_y",
                    "nucleus_area",
                    "z_level",
                    "nucleus_count",
                ],
            )
            summary_df = pd.DataFrame(
                summary_arr[:, : len(col_names)],
                index=pd.Index(cell_barcodes.astype(object), name="cell_id"),
                columns=list(col_names),
            )
            if not summary_df.index.equals(adata.obs_names):
                summary_df = summary_df.reindex(adata.obs_names)

            if "spatial" not in adata.obsm and {
                "cell_centroid_x",
                "cell_centroid_y",
            }.issubset(summary_df.columns):
                adata.obsm["spatial"] = summary_df[
                    ["cell_centroid_x", "cell_centroid_y"]
                ].to_numpy(dtype=np.float32)

            for col in ("cell_area", "nucleus_area", "z_level", "nucleus_count"):
                if col in summary_df.columns and col not in adata.obs.columns:
                    if col == "nucleus_count":
                        adata.obs[col] = summary_df[col].fillna(0).to_numpy(dtype=np.int32)
                    else:
                        adata.obs[col] = summary_df[col].to_numpy(dtype=np.float32)

        # 3. Extract polygon boundaries from /polygon_sets
        if load_boundaries and "polygon_sets" in root:
            poly_sets = root["polygon_sets"]
            # Per 10x Atera / Xenium Zarr spec:
            #   polygon_sets/0 -> nucleus boundaries
            #   polygon_sets/1 -> cell boundaries
            mapping = (
                ("0", "nucleus_boundaries"),
                ("1", "cell_boundaries"),
                ("nucleus", "nucleus_boundaries"),
                ("cell", "cell_boundaries"),
            )
            for poly_key, obsm_key in mapping:
                if poly_key in poly_sets and obsm_key not in adata.obsm:
                    try:
                        adata.obsm[obsm_key] = _read_polygon_set(
                            poly_sets[poly_key],
                            cell_barcodes=cell_barcodes,
                            obs_names=adata.obs_names,
                        )
                    except Exception as exc:
                        log.warning(
                            "Failed to load polygon_sets/%s from %s: %s",
                            poly_key,
                            cells_path.name,
                            exc,
                        )
            if "cell_boundaries" in adata.obsm or "nucleus_boundaries" in adata.obsm:
                adata.uns[UNS_XENIUM_SEGMENTATION_KEY] = True


def _read_polygon_set(
    poly_grp,
    *,
    cell_barcodes: np.ndarray,
    obs_names: pd.Index,
) -> ak.Array:
    """Convert a 10x Zarr ``/polygon_sets/<idx>`` group into an Awkward ragged array.

    Supports all three vertex storage layouts encountered in 10x Zarr files:
    - ``(n_polys, max_verts, 2)`` padded 3-D vertex array
    - ``(n_polys, 2 * max_verts)`` padded 2-D vertex array
    - ``(total_verts, 2)`` flat ragged 2-D vertex array
    """
    num_vertices = np.asarray(poly_grp["num_vertices"][:], dtype=np.int64).ravel()
    raw_verts = np.asarray(poly_grp["vertices"][:], dtype=np.float32)
    n_polys = len(num_vertices)

    if "cell_index" in poly_grp:
        cell_idx = np.asarray(poly_grp["cell_index"][:], dtype=np.int64).ravel()
    else:
        cell_idx = np.arange(n_polys, dtype=np.int64)

    # Unpack vertices per polygon into a list/ragged structure of (n_verts_i, 2)
    total_verts = int(num_vertices.sum())
    if raw_verts.ndim == 3 and raw_verts.shape[0] == n_polys and raw_verts.shape[2] == 2:
        max_v = raw_verts.shape[1]
        mask = np.arange(max_v)[None, :] < num_vertices[:, None]
        flat_coords = raw_verts[mask].astype(np.float32, copy=False)
    elif (
        raw_verts.ndim == 2
        and raw_verts.shape[0] == n_polys
        and raw_verts.shape[1] % 2 == 0
        and (raw_verts.shape[1] > 2 or total_verts != n_polys)
    ):
        max_v = raw_verts.shape[1] // 2
        reshaped = raw_verts.reshape(n_polys, max_v, 2)
        mask = np.arange(max_v)[None, :] < num_vertices[:, None]
        flat_coords = reshaped[mask].astype(np.float32, copy=False)
    elif raw_verts.ndim == 2 and raw_verts.shape[0] == total_verts and raw_verts.shape[1] == 2:
        flat_coords = raw_verts.astype(np.float32, copy=False)
    elif raw_verts.ndim == 1 and raw_verts.size == total_verts * 2:
        flat_coords = raw_verts.reshape(total_verts, 2).astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"Unrecognized polygon vertices shape {raw_verts.shape} for "
            f"num_vertices length {n_polys} (sum={total_verts})"
        )

    n_obs = len(obs_names)
    n_barcodes = len(cell_barcodes)
    if n_polys == 0:
        return ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.zeros(n_obs + 1, dtype=np.int64)),
                ak.contents.NumpyArray(np.empty((0, 2), dtype=np.float32)),
            )
        )

    if (
        cell_idx.min() >= 1
        and cell_idx.max() == n_barcodes
        and not np.any(cell_idx == 0)
    ):
        idx_0 = cell_idx - 1
    else:
        idx_0 = cell_idx

    # Fast vectorized path when cell_barcodes matches obs_names in order
    # (covers both n_polys == n_obs and n_polys < n_obs when some cells lack a nucleus)
    barcodes_aligned = (
        n_barcodes == n_obs
        and np.array_equal(cell_barcodes.astype(object), obs_names.to_numpy().astype(object))
    )
    if (
        barcodes_aligned
        and idx_0[0] >= 0
        and idx_0[-1] < n_obs
        and (n_polys == 1 or bool(np.all(idx_0[:-1] < idx_0[1:])))
    ):
        ordered_counts = np.zeros(n_obs, dtype=np.int64)
        ordered_counts[idx_0] = num_vertices
        out_offsets = np.empty(n_obs + 1, dtype=np.int64)
        out_offsets[0] = 0
        np.cumsum(ordered_counts, out=out_offsets[1:])
        return ak.Array(
            ak.contents.ListOffsetArray(
                ak.index.Index64(out_offsets),
                ak.contents.NumpyArray(np.ascontiguousarray(flat_coords, dtype=np.float32)),
            )
        )

    # General path: map each polygon's cell_index -> cell barcode -> obs_names position
    poly_offsets = np.empty(n_polys + 1, dtype=np.int64)
    poly_offsets[0] = 0
    np.cumsum(num_vertices, out=poly_offsets[1:])

    barcode_to_poly: dict[str, int] = {}
    for p_i, c_i in enumerate(idx_0):
        if 0 <= c_i < n_barcodes:
            bc = str(cell_barcodes[c_i])
            if bc not in barcode_to_poly:
                barcode_to_poly[bc] = p_i

    ordered_chunks: list[np.ndarray] = []
    ordered_counts = np.zeros(n_obs, dtype=np.int64)
    for obs_i, bc in enumerate(obs_names):
        p_i = barcode_to_poly.get(str(bc))
        if p_i is not None:
            start, end = int(poly_offsets[p_i]), int(poly_offsets[p_i + 1])
            ordered_counts[obs_i] = end - start
            if end > start:
                ordered_chunks.append(flat_coords[start:end])

    out_offsets = np.empty(n_obs + 1, dtype=np.int64)
    out_offsets[0] = 0
    np.cumsum(ordered_counts, out=out_offsets[1:])
    out_coords = (
        np.ascontiguousarray(np.vstack(ordered_chunks), dtype=np.float32)
        if ordered_chunks
        else np.empty((0, 2), dtype=np.float32)
    )
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(out_offsets),
            ak.contents.NumpyArray(out_coords),
        )
    )


# ---------------------------------------------------------------------------
# Morphology OME-TIFF attachment (morphology_2d/, morphology_focus/, morphology_3d/)
# ---------------------------------------------------------------------------


def _load_spec_with_fallback_pixel_size(path: Path, pixel_size_um: float):
    """Extract an ImageSpec from *path*, using *pixel_size_um* if OME-XML has none."""
    spec = _extract_image_spec(path)
    if spec.pixel_size_um == 1.0 and pixel_size_um != 1.0:
        spec.pixel_size_um = pixel_size_um
    return spec


def _attach_morphology_2d(
    images_dict: dict,
    bundle: Path,
    morphology_image: bool | str | Path,
    *,
    pixel_size_um: float,
) -> None:
    """Attach 2-D morphology OME-TIFFs into ``uns['spatial'][library_id]['images']``."""
    if not isinstance(morphology_image, bool):
        img_path = _resolve_optional_file(
            bundle, morphology_image, "morphology.ome.tif", "morphology_image"
        )
        if img_path is not None:
            spec = _load_spec_with_fallback_pixel_size(img_path, pixel_size_um)
            images_dict["full"] = spec
            images_dict["morphology"] = spec
        return

    # Check both morphology_2d/ (Atera v1.0) and morphology_focus/ (Atera WTA Preview)
    for dir_name in ("morphology_2d", "morphology_focus"):
        morph_2d_dir = bundle / dir_name
        if morph_2d_dir.is_dir():
            focus_files = sorted(
                f for f in morph_2d_dir.glob("*.ome.tif") if not f.name.startswith(".")
            )
            for f in focus_files:
                stem = f.name[: -len(".ome.tif")]
                try:
                    spec = _load_spec_with_fallback_pixel_size(f, pixel_size_um)
                    images_dict[f"morphology_2d_{stem}"] = spec
                    # Prefer ch0000_dapi as the default 'full' / 'morphology' image
                    if "full" not in images_dict or stem.startswith("ch0000"):
                        images_dict["full"] = spec
                        images_dict["morphology"] = spec
                except Exception as exc:
                    log.warning("Failed to read %s/%s: %s", dir_name, f.name, exc)
            if focus_files:
                return

    # Fallback for single-file morphology_focus.ome.tif or morphology.ome.tif
    for default_name in ("morphology_focus.ome.tif", "morphology.ome.tif"):
        candidate = bundle / default_name
        if candidate.exists():
            try:
                spec = _load_spec_with_fallback_pixel_size(candidate, pixel_size_um)
                images_dict["full"] = spec
                images_dict["morphology"] = spec
                return
            except Exception as exc:
                log.warning("Failed to read %s: %s", default_name, exc)


def _attach_morphology_3d(
    images_dict: dict,
    bundle: Path,
    morphology_3d: bool | str | Path,
    *,
    pixel_size_um: float,
) -> None:
    """Attach 3-D morphology OME-TIFFs into ``uns['spatial'][library_id]['images']``."""
    if not isinstance(morphology_3d, bool):
        img_path = _resolve_optional_file(
            bundle, morphology_3d, "morphology_3d/ch0000_dapi_3d.ome.tif", "morphology_3d"
        )
        if img_path is not None:
            images_dict["morphology_3d"] = _load_spec_with_fallback_pixel_size(
                img_path, pixel_size_um
            )
        return

    morph_3d_dir = bundle / "morphology_3d"
    if morph_3d_dir.is_dir():
        files_3d = sorted(
            f for f in morph_3d_dir.glob("*.ome.tif") if not f.name.startswith(".")
        )
        for f in files_3d:
            stem = f.name[: -len(".ome.tif")]
            try:
                spec = _load_spec_with_fallback_pixel_size(f, pixel_size_um)
                images_dict[f"morphology_3d_{stem}"] = spec
                if "morphology_3d" not in images_dict or stem.startswith("ch0000"):
                    images_dict["morphology_3d"] = spec
            except Exception as exc:
                log.warning("Failed to read morphology_3d/%s: %s", f.name, exc)
        if files_3d:
            return

    # Fallback to root morphology.ome.tif 3-D z-stack (e.g. Atera WTA Preview)
    root_3d = bundle / "morphology.ome.tif"
    if root_3d.exists():
        try:
            images_dict["morphology_3d"] = _load_spec_with_fallback_pixel_size(
                root_3d, pixel_size_um
            )
        except Exception as exc:
            log.warning("Failed to read morphology.ome.tif: %s", exc)


# ---------------------------------------------------------------------------
# Transcripts (transcripts.parquet & transcripts.zarr.zip)
# ---------------------------------------------------------------------------


def read_atera_transcripts(
    mdata: md.MuData,
    path: str | Path | None = None,
    *,
    atera_key: str = "atera",
    inplace: bool = True,
    cell_ids: list[str] | None = None,
    genes: list[str] | None = None,
    min_qv: float | None = 20.0,
    z_range: tuple[float, float] | None = None,
    include_controls: bool = True,
    columns: list[str] | None = None,
) -> md.MuData | None:
    """Load per-transcript coordinates and metadata into ``mdata[atera_key].uns['transcripts']``.

    Reads ``transcripts.parquet`` (when present) or ``transcripts.zarr.zip`` /
    ``transcripts.zarr`` (at full resolution ``/grid`` or ``/grids/0`` across
    all spatial tiles) into a :class:`pyarrow.Table` with standardized columns
    matching :func:`~astrolabe.io.read_xenium_transcripts`:

    - ``transcript_id`` (``int64`` / ``uint64``)
    - ``feature_name`` (``string``)
    - ``x_location``, ``y_location``, ``z_location`` (``float32``, in µm)
    - ``qv`` (``float32``, Phred quality score)
    - ``fov_name`` (``string``, when present)
    - ``cell_id`` / ``overlaps_nucleus`` (when present)

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`read_atera`.  The table is
        stored in ``mdata[atera_key].uns['transcripts']``.
    path
        Path to the Atera output directory (or a direct path to
        ``transcripts.zarr.zip`` / ``transcripts.parquet``).  When ``None``
        (default), resolved automatically via
        :func:`~astrolabe.io._xenium._source_path_from_mdata`.
    atera_key
        Modality key inside *mdata* (default ``'atera'``).
    inplace
        If ``True`` (default), store the table in *mdata* in place and return
        ``None``.  If ``False``, return a modified copy.
    cell_ids
        Restrict to transcripts assigned to these cell IDs.
    genes
        Restrict to transcripts for these gene/feature names.
    min_qv
        Minimum calibrated quality value (Q-score).  Transcripts below this
        threshold are dropped during loading.  Pass ``None`` to disable.
    z_range
        ``(z_min, z_max)`` in microns; filter by ``z_location``.
    include_controls
        If ``True`` (default), keep negative control probes and unassigned
        codewords in ``uns['transcripts']`` (matching ``X``, where controls
        are flagged in ``var['is_control_probe']`` rather than removed at load
        time).  If ``False``, drop control features and no-calls on load.
    columns
        Optional subset of columns to retain in the resulting table.
    """
    if path is None:
        path = _source_path_from_mdata(mdata)

    p = Path(path)
    if p.is_file() or (p.is_dir() and p.name.endswith(".zarr")):
        tx_path = p
    else:
        bundle = _resolve_bundle_dir(p)
        tx_path = _find_zarr_or_parquet(
            bundle,
            ("transcripts.parquet", "transcripts.zarr.zip", "transcripts.zarr"),
        )
        if tx_path is None:
            raise FileNotFoundError(
                f"Neither transcripts.zarr.zip nor transcripts.parquet found in {bundle}"
            )

    log.info("Reading Atera transcripts from %s (min_qv=%s)", tx_path, min_qv)

    if tx_path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pq_filters: list[tuple] = []
        if min_qv is not None and min_qv > 0:
            pq_filters.append(("qv", ">=", min_qv))
        if cell_ids is not None:
            pq_filters.append(("cell_id", "in", list(cell_ids)))
        if genes is not None:
            pq_filters.append(("feature_name", "in", list(genes)))
        if z_range is not None:
            pq_filters.append(("z_location", ">=", z_range[0]))
            pq_filters.append(("z_location", "<=", z_range[1]))

        schema_names = set(pq.read_schema(tx_path).names)
        read_cols = columns
        if (
            read_cols is not None
            and not include_controls
            and "feature_name" not in read_cols
            and "feature_name" in schema_names
        ):
            read_cols = [*read_cols, "feature_name"]

        table = pq.read_table(
            tx_path,
            columns=read_cols,
            filters=pq_filters if pq_filters else None,
        )
    else:
        with _open_zarr_group(tx_path) as root:
            table = _read_transcripts_zarr_group(
                root,
                min_qv=min_qv,
                z_range=z_range,
                genes=genes,
                cell_ids=cell_ids,
                include_controls=include_controls,
            )

    if not include_controls and "feature_name" in table.column_names and len(table) > 0:
        ctrl_mask = _build_control_mask_arrow(table.column("feature_name"))
        table = table.filter(pc.invert(ctrl_mask))

    if columns is not None:
        keep_cols = [c for c in columns if c in table.column_names]
        table = table.select(keep_cols)

    log.info("Loaded %d Atera transcripts", len(table))

    target = mdata if inplace else copy.deepcopy(mdata)
    target[atera_key].uns["transcripts"] = table
    return None if inplace else target


def _read_transcripts_zarr_group(
    root,
    *,
    min_qv: float | None,
    z_range: tuple[float, float] | None,
    genes: list[str] | None,
    cell_ids: list[str] | None,
    include_controls: bool,
) -> pa.Table:
    """Extract and filter full-resolution transcripts from ``transcripts.zarr.zip``.

    Supports both:
    - **Atera v1.0 ``/grid`` schema (major_version 6)**: tiles under ``/grid/<tile_key>``
      sorted by gene with ``gene_offset`` ``(n_genes, 2)`` ``[start, end]`` slices,
      allowing O(1) slice reads when ``genes`` or ``include_controls=False`` is specified.
    - **Quadtree ``/grids/0`` schema**: tiles under ``/grids/0/<tile_key>`` with
      ``gene_identity`` or ``feature_name``.
    """
    import zarr

    gene_names_attr = root.attrs.get("gene_names") or root.attrs.get("feature_names")
    if gene_names_attr is None and "gene_names" in root:
        gene_names_attr = _decode_string_array(root["gene_names"][:]).tolist()
    gene_lookup = (
        np.asarray(gene_names_attr, dtype=object) if gene_names_attr is not None else None
    )

    cw_gene_names_attr = root.attrs.get("codeword_gene_names")
    cw_gene_lookup = (
        np.asarray(cw_gene_names_attr, dtype=object)
        if cw_gene_names_attr is not None
        else None
    )

    fov_names_attr = root.attrs.get("fov_names")
    if fov_names_attr is None and "fov_names" in root:
        fov_names_attr = _decode_string_array(root["fov_names"][:]).tolist()
    fov_lookup = (
        np.asarray(fov_names_attr, dtype=object) if fov_names_attr is not None else None
    )

    # Pre-compute allowed gene integer codes when gene_lookup is available
    allowed_gene_codes: set[int] | None = None
    if gene_lookup is not None:
        if genes is not None:
            gene_set = set(genes)
            allowed_gene_codes = {
                idx for idx, name in enumerate(gene_lookup) if str(name) in gene_set
            }
        elif not include_controls and "gene_category" in root:
            # In v6 transcripts.zarr.zip, /gene_category[:, 0] is 'is_gene'
            gene_cat = np.asarray(root["gene_category"][:], dtype=bool)
            if gene_cat.ndim == 2 and gene_cat.shape[0] == len(gene_lookup):
                allowed_gene_codes = set(np.where(gene_cat[:, 0])[0].tolist())

    cell_id_set = set(cell_ids) if cell_ids is not None else None

    # Locate full-resolution tile groups under /grid (Atera v1.0) or /grids/0 (quadtree)
    tile_groups: list = []
    if "grid" in root and isinstance(root["grid"], zarr.Group):
        grid_grp = root["grid"]
        for tile_key in grid_grp.keys():
            node = grid_grp[tile_key]
            if isinstance(node, zarr.Group) and ("location" in node or "x_location" in node):
                tile_groups.append(node)
    elif "grids" in root and "0" in root["grids"]:
        grid0 = root["grids"]["0"]
        for tile_key in grid0.keys():
            node = grid0[tile_key]
            if isinstance(node, zarr.Group) and ("location" in node or "x_location" in node):
                tile_groups.append(node)
    elif "location" in root or "x_location" in root:
        tile_groups.append(root)

    batch_tables: list[pa.Table] = []
    running_tx_id = 0

    for tgrp in tile_groups:
        # Fast-path for Atera v1.0 /grid/<tile> with gene_offset when filtering by gene
        if (
            "gene_offset" in tgrp
            and gene_lookup is not None
            and allowed_gene_codes is not None
        ):
            goff = np.asarray(tgrp["gene_offset"][:], dtype=np.int64)
            target_gis = [
                gi
                for gi in sorted(allowed_gene_codes)
                if 0 <= gi < len(goff) and goff[gi, 1] > goff[gi, 0]
            ]
            if not target_gis:
                continue

            loc_chunks = []
            qv_chunks = []
            feat_chunks = []
            cid_chunks = []
            nuc_chunks = []
            has_cid = "cell_id" in tgrp
            has_nuc = "overlaps_nucleus" in tgrp
            has_qv = "quality_score" in tgrp or "qv" in tgrp
            qv_key = "quality_score" if "quality_score" in tgrp else "qv"

            for gi in target_gis:
                s, e = int(goff[gi, 0]), int(goff[gi, 1])
                loc_chunks.append(np.asarray(tgrp["location"][s:e], dtype=np.float32))
                if has_qv:
                    qv_chunks.append(np.asarray(tgrp[qv_key][s:e], dtype=np.float32).ravel())
                else:
                    qv_chunks.append(np.full(e - s, 40.0, dtype=np.float32))
                feat_chunks.append(np.full(e - s, gene_lookup[gi], dtype=object))
                if has_cid:
                    cid_chunks.append(np.asarray(tgrp["cell_id"][s:e]))
                if has_nuc:
                    nuc_chunks.append(
                        np.asarray(tgrp["overlaps_nucleus"][s:e], dtype=np.uint8).ravel()
                    )

            loc = np.vstack(loc_chunks)
            x_loc = loc[:, 0]
            y_loc = loc[:, 1]
            z_loc = loc[:, 2] if loc.shape[1] >= 3 else np.zeros(len(x_loc), dtype=np.float32)
            qv = np.concatenate(qv_chunks)
            feat_col = np.concatenate(feat_chunks)

            keep = np.ones(len(x_loc), dtype=bool)
            if min_qv is not None and min_qv > 0:
                keep &= qv >= min_qv
            if z_range is not None:
                keep &= (z_loc >= z_range[0]) & (z_loc <= z_range[1])
            if not np.any(keep):
                continue

            x_loc = x_loc[keep]
            y_loc = y_loc[keep]
            z_loc = z_loc[keep]
            qv = qv[keep]
            feat_col = feat_col[keep]

            cell_col: np.ndarray | None = None
            if has_cid:
                raw_cid = np.vstack(cid_chunks)[keep]
                if raw_cid.ndim == 2 and raw_cid.shape[1] >= 2:
                    unassigned = (
                        (raw_cid[:, 0] == 0)
                        | (raw_cid[:, 1] == 0)
                        | (raw_cid[:, 0] == 0xFFFFFFFF)
                    )
                    decoded_cid = _decode_10x_cell_id(raw_cid)
                    decoded_cid[unassigned] = "UNASSIGNED"
                    cell_col = decoded_cid
                else:
                    cell_col = _decode_string_array(raw_cid.ravel())

            nuc_col: np.ndarray | None = (
                np.concatenate(nuc_chunks)[keep] if has_nuc else None
            )

            if cell_id_set is not None and cell_col is not None:
                cid_mask = np.isin(cell_col, list(cell_id_set))
                if not np.any(cid_mask):
                    continue
                x_loc = x_loc[cid_mask]
                y_loc = y_loc[cid_mask]
                z_loc = z_loc[cid_mask]
                qv = qv[cid_mask]
                feat_col = feat_col[cid_mask]
                cell_col = cell_col[cid_mask]
                if nuc_col is not None:
                    nuc_col = nuc_col[cid_mask]

            tx_id = np.arange(running_tx_id, running_tx_id + len(x_loc), dtype=np.int64)
            running_tx_id += len(x_loc)

            cols_dict: dict[str, pa.Array] = {
                "transcript_id": pa.array(tx_id, type=pa.int64()),
                "feature_name": pa.array(feat_col.astype(object), type=pa.string()),
                "x_location": pa.array(x_loc, type=pa.float32()),
                "y_location": pa.array(y_loc, type=pa.float32()),
                "z_location": pa.array(z_loc, type=pa.float32()),
                "qv": pa.array(qv, type=pa.float32()),
            }
            if cell_col is not None:
                cols_dict["cell_id"] = pa.array(cell_col.astype(object), type=pa.string())
            if nuc_col is not None:
                cols_dict["overlaps_nucleus"] = pa.array(nuc_col, type=pa.uint8())
            batch_tables.append(pa.table(cols_dict))
            continue

        # Standard full-tile read path
        # 1. Locations (N, 3) or separate x/y/z
        if "location" in tgrp:
            loc = np.asarray(tgrp["location"][:], dtype=np.float32)
            if loc.shape[0] == 0:
                continue
            x_loc = loc[:, 0]
            y_loc = loc[:, 1]
            z_loc = loc[:, 2] if loc.shape[1] >= 3 else np.zeros(len(x_loc), dtype=np.float32)
        else:
            x_loc = np.asarray(tgrp["x_location"][:], dtype=np.float32).ravel()
            if len(x_loc) == 0:
                continue
            y_loc = np.asarray(tgrp["y_location"][:], dtype=np.float32).ravel()
            z_loc = (
                np.asarray(tgrp["z_location"][:], dtype=np.float32).ravel()
                if "z_location" in tgrp
                else np.zeros(len(x_loc), dtype=np.float32)
            )

        n_tile = len(x_loc)
        keep = np.ones(n_tile, dtype=bool)

        # 2. Quality score (QV)
        if "quality_score" in tgrp:
            qv = np.asarray(tgrp["quality_score"][:], dtype=np.float32).ravel()
        elif "qv" in tgrp:
            qv = np.asarray(tgrp["qv"][:], dtype=np.float32).ravel()
        else:
            qv = np.full(n_tile, 40.0, dtype=np.float32)

        if min_qv is not None and min_qv > 0:
            keep &= qv >= min_qv

        # 3. Z-range filter
        if z_range is not None:
            keep &= (z_loc >= z_range[0]) & (z_loc <= z_range[1])

        # 4. Gene identity (via gene_offset, gene_identity, codeword_identity, or feature_name)
        gene_codes: np.ndarray | None = None
        feat_names: np.ndarray | None = None
        if "gene_offset" in tgrp and gene_lookup is not None:
            goff = np.asarray(tgrp["gene_offset"][:], dtype=np.int64)
            counts_per_gene = np.maximum(0, goff[:, 1] - goff[:, 0])
            if int(counts_per_gene.sum()) == n_tile:
                gene_codes = np.repeat(np.arange(len(goff), dtype=np.int64), counts_per_gene)
                if allowed_gene_codes is not None:
                    keep &= np.isin(gene_codes, list(allowed_gene_codes))
        elif "gene_identity" in tgrp:
            gene_codes = np.asarray(tgrp["gene_identity"][:], dtype=np.int64).ravel()
            if gene_lookup is not None:
                # 65535 (or >= len(gene_lookup)) is 10x's sentinel for no-call
                keep &= (gene_codes >= 0) & (gene_codes < len(gene_lookup))
            if allowed_gene_codes is not None:
                keep &= np.isin(gene_codes, list(allowed_gene_codes))
        elif "codeword_identity" in tgrp and cw_gene_lookup is not None:
            cw_codes = np.asarray(tgrp["codeword_identity"][:], dtype=np.int64).ravel()
            valid_cw = (cw_codes >= 0) & (cw_codes < len(cw_gene_lookup))
            keep &= valid_cw
            feat_names = cw_gene_lookup[np.clip(cw_codes, 0, len(cw_gene_lookup) - 1)]
            if genes is not None:
                keep &= np.isin(feat_names, list(genes))
        elif "feature_name" in tgrp:
            feat_names = _decode_string_array(tgrp["feature_name"][:]).ravel()
            if genes is not None:
                keep &= np.isin(feat_names, list(genes))

        if not np.any(keep):
            continue

        # Slice arrays by keep mask
        x_loc = x_loc[keep]
        y_loc = y_loc[keep]
        z_loc = z_loc[keep]
        qv = qv[keep]

        if feat_names is not None:
            feat_col = feat_names[keep]
        elif gene_codes is not None and gene_lookup is not None:
            feat_col = gene_lookup[gene_codes[keep]]
        elif gene_codes is not None:
            feat_col = gene_codes[keep].astype(str)
        else:
            feat_col = np.full(len(x_loc), "", dtype=object)

        # 5. Transcript ID & FOV name (from /id (N, 2) = [transcript_id, fov_index] or /transcript_id)
        tx_id: np.ndarray
        fov_col: np.ndarray | None = None
        if "id" in tgrp:
            id_arr = np.asarray(tgrp["id"][:])[keep]
            if id_arr.ndim == 2 and id_arr.shape[1] >= 1:
                tx_id = id_arr[:, 0].astype(np.int64)
                if id_arr.shape[1] >= 2 and fov_lookup is not None and len(fov_lookup) > 0:
                    fov_idx = id_arr[:, 1].astype(np.int64)
                    valid_fov = (fov_idx >= 0) & (fov_idx < len(fov_lookup))
                    fov_col = np.where(
                        valid_fov,
                        fov_lookup[np.clip(fov_idx, 0, len(fov_lookup) - 1)],
                        "",
                    )
            else:
                tx_id = id_arr.ravel().astype(np.int64)
        elif "transcript_id" in tgrp:
            tx_id = np.asarray(tgrp["transcript_id"][:])[keep].ravel().astype(np.int64)
        else:
            tx_id = np.arange(running_tx_id, running_tx_id + len(x_loc), dtype=np.int64)
            running_tx_id += len(x_loc)

        if fov_col is None and "fov_name" in tgrp:
            fov_col = _decode_string_array(tgrp["fov_name"][:]).ravel()[keep]

        # 6. Optional cell_id & overlaps_nucleus in tile
        cell_col = None
        if "cell_id" in tgrp:
            raw_cid = np.asarray(tgrp["cell_id"][:])[keep]
            if raw_cid.ndim == 2 and raw_cid.shape[1] >= 2:
                unassigned = (raw_cid[:, 0] == 0) | (raw_cid[:, 1] == 0) | (raw_cid[:, 0] == 0xFFFFFFFF)
                decoded_cid = _decode_10x_cell_id(raw_cid)
                decoded_cid[unassigned] = "UNASSIGNED"
                cell_col = decoded_cid
            else:
                cell_col = _decode_string_array(raw_cid.ravel())

        nuc_col = None
        if "overlaps_nucleus" in tgrp:
            nuc_col = np.asarray(tgrp["overlaps_nucleus"][:])[keep].ravel().astype(np.uint8)

        # Filter by cell_ids if requested and cell_col is present
        if cell_id_set is not None and cell_col is not None:
            cid_mask = np.isin(cell_col, list(cell_id_set))
            if not np.any(cid_mask):
                continue
            x_loc = x_loc[cid_mask]
            y_loc = y_loc[cid_mask]
            z_loc = z_loc[cid_mask]
            qv = qv[cid_mask]
            feat_col = feat_col[cid_mask]
            tx_id = tx_id[cid_mask]
            cell_col = cell_col[cid_mask]
            if fov_col is not None:
                fov_col = fov_col[cid_mask]
            if nuc_col is not None:
                nuc_col = nuc_col[cid_mask]

        cols_dict = {
            "transcript_id": pa.array(tx_id, type=pa.int64()),
            "feature_name": pa.array(feat_col.astype(object), type=pa.string()),
            "x_location": pa.array(x_loc, type=pa.float32()),
            "y_location": pa.array(y_loc, type=pa.float32()),
            "z_location": pa.array(z_loc, type=pa.float32()),
            "qv": pa.array(qv, type=pa.float32()),
        }
        if fov_col is not None:
            cols_dict["fov_name"] = pa.array(fov_col.astype(object), type=pa.string())
        if cell_col is not None:
            cols_dict["cell_id"] = pa.array(cell_col.astype(object), type=pa.string())
        if nuc_col is not None:
            cols_dict["overlaps_nucleus"] = pa.array(nuc_col, type=pa.uint8())

        batch_tables.append(pa.table(cols_dict))

    if not batch_tables:
        return pa.table({
            "transcript_id": pa.array([], type=pa.int64()),
            "feature_name": pa.array([], type=pa.string()),
            "x_location": pa.array([], type=pa.float32()),
            "y_location": pa.array([], type=pa.float32()),
            "z_location": pa.array([], type=pa.float32()),
            "qv": pa.array([], type=pa.float32()),
        })

    return pa.concat_tables(batch_tables, promote_options="default")
