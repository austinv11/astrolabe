"""
Reader for 10x Genomics Visium HD Space Ranger output directories.

Canonical call
--------------
>>> import astrolabe as ab
>>> mdata = ab.io.read_visium_hd("Visium_HD_run/outs/")

Unlike Xenium/MERSCOPE/CosMx, Visium HD does not ship pre-computed
single-cell segmentation by default — it produces a regular grid of 2µm
square "bins". This reader intentionally focuses on the native 2µm
resolution only (see :func:`~astrolabe.tl.rebin_visium_hd` for a bespoke,
computed conversion to 8µm/16µm or other bin sizes) and keeps the bin x gene
count matrix **lazy**: ``mdata['visium_hd_bins'].X`` is left ``None`` and a
:class:`~astrolabe._utils.BinMatrixSpec` path descriptor is stored instead —
mirroring how IF/H&E images stay off-heap until
:func:`~astrolabe.pp.extract_if_intensities` is called. Use
:func:`~astrolabe.pp.load_bin_counts` to materialize all or part of the bin
matrix into memory.

If the Space Ranger run includes on-board segmentation (v4.0+,
``segmented_outputs/``), that cell x gene result is loaded eagerly as
``mdata['visium_hd_cells']`` and becomes the **primary** (default) modality —
matching the "cell x gene where possible" convention shared with
Xenium/MERSCOPE/CosMx. Otherwise the default modality stays
``'visium_hd_bins'`` until the user produces a segmentation (their own, or
via :func:`~astrolabe.seg.segment_stardist`/:func:`~astrolabe.seg.segment_cellpose`)
and calls :func:`~astrolabe.seg.assign_bins` (or the
:func:`~astrolabe.seg.segment_bin2cell` recipe) to build one.

Schema
------
mdata['visium_hd_bins']
    X           : None  (lazy — see pp.load_bin_counts)
    obs         : barcode (index), array_row, array_col, in_tissue
    obsm
        spatial : float32 (n_bins, 2)  [x, y] in µm (from pxl_col/row_in_fullres)
    var         : gene_id, gene_name, feature_type
    uns
        spatial
            <library_id>
                coordinate_unit : 'microns'
                source_path     : str
                images          : {}
                bin_matrix      : BinMatrixSpec

mdata['he'] (optional, when load_he=True and an image is found)
    Same schema as :func:`~astrolabe.io.read_he`. When the fullres tissue
    image is found, it's registered into the same pixel frame as the bins
    (a pure ``pixel_size_um`` scale, no separate alignment matrix needed:
    Space Ranger already places both in one frame). When only the CytAssist
    registration-target image is present (no fullres tissue image shipped
    with the run), it's registered with its own, different pixel scale from
    ``scalefactors_json.json``'s ``regist_target_img_scalef_microns_per_pixel``
    instead — never with the bin-space ``pixel_size_um``, which describes a
    different scanner's pixel resolution. If that scale factor isn't present,
    the image is skipped (logged) rather than silently mis-registered.

mdata['visium_hd_cells'] (optional, only when segmented_outputs/ is found)
    X                    : (n_cells, n_genes)  sparse float32
    obsm
        spatial          : float32 (n_cells, 2)  [x, y] in µm
        cell_boundaries  : awkward (n_cells,) of (n_verts, 2) float32 polygons
    uns['default_modality'] : 'visium_hd_cells'
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import awkward as ak
import geopandas as gpd
import h5py
import mudata as md
import numpy as np
import pandas as pd
import scanpy as sc

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY, UNS_DEFAULT_MODALITY_KEY
from astrolabe._utils import BinMatrixSpec
from astrolabe.io._he import read_he
from astrolabe.io._io import attach_modality

__all__ = ["read_visium_hd"]

log = logging.getLogger(__name__)

_NATIVE_BIN_SIZE_UM = 2.0
# binned_outputs/square_XXXum/ uses the "bc" (barcode) naming convention;
# segmented_outputs/ (post cell-segmentation, so columns are cells rather
# than raw bin barcodes) uses "cell" instead -- e.g. a real Space Ranger
# v4.0.1 run ships segmented_outputs/filtered_feature_cell_matrix.h5, never
# a _bc_ variant. Both naming families are tried here so _resolve_matrix_h5
# works for either directory regardless of which convention it contains.
_MATRIX_CANDIDATES = [
    "filtered_feature_bc_matrix.h5",
    "raw_feature_bc_matrix.h5",
    "filtered_feature_cell_matrix.h5",
    "raw_feature_cell_matrix.h5",
]
_HE_PATTERNS = ["*_tissue_image.btf", "*_tissue_image.tif", "*_tissue_image.tiff"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_visium_hd(
    path: str | Path,
    *,
    library_id: str | None = None,
    load_he: bool = True,
    he_path: str | Path | None = None,
    load_segmented: bool = True,
    segmented_outputs_dir: str | Path | None = None,
    dtype: str = "float32",
) -> md.MuData:
    """Read a 10x Genomics Visium HD Space Ranger output directory into a :class:`mudata.MuData`.

    Only the native 2µm bin resolution (``binned_outputs/square_002um/``) is
    read — see :func:`~astrolabe.tl.rebin_visium_hd` for computed 8µm/16µm
    (or any other) conversion. The bin x gene matrix is left **lazy**
    (``X=None``); call :func:`~astrolabe.pp.load_bin_counts` to materialize it.

    Parameters
    ----------
    path
        Path to the Space Ranger output directory that directly contains
        ``binned_outputs/`` (i.e. Space Ranger's own ``outs/``).
    library_id
        Identifier stored under ``uns['spatial']``. Defaults to *path*'s own
        name, or its parent's name when *path* is literally named ``outs``.
    load_he
        If ``True`` (default), auto-discover and attach the paired H&E image
        as a new ``'he'`` modality (via :func:`~astrolabe.io.read_he`).
        Visium HD ships an H&E image with every run, so this defaults to
        ``True`` unlike Xenium's opt-in ``he_image=``. Missing file → warn
        and skip, same graceful behavior as Xenium's optional-file handling.
    he_path
        Explicit override for the H&E image path (bare filename resolved
        relative to *path*, or absolute). Required to exist if given.
    load_segmented
        If ``True`` (default) and a ``segmented_outputs/`` directory is found
        (Space Ranger v4.0+ on-board segmentation), load it eagerly as
        ``mdata['visium_hd_cells']`` and make it the primary modality.
    segmented_outputs_dir
        Explicit override for the ``segmented_outputs/`` directory (bare name
        resolved relative to *path*, or absolute).
    dtype
        NumPy dtype for materialized count matrices (the native segmentation
        matrix, if loaded). ``'float32'`` is the default.

    Returns
    -------
    mudata.MuData
        With a ``'visium_hd_bins'`` modality (lazy 2µm bins), an optional
        ``'he'`` modality, and an optional ``'visium_hd_cells'`` modality
        (native segmentation) that becomes ``uns['default_modality']`` when
        present — otherwise ``default_modality`` stays ``'visium_hd_bins'``.

    Examples
    --------
    >>> mdata = ab.io.read_visium_hd("Visium_HD_run/outs/")
    >>> mdata.uns['default_modality']
    'visium_hd_bins'
    >>> ab.pp.load_bin_counts(mdata, region_um=(0, 0, 500, 500))  # materialize a crop
    """
    path = Path(path).resolve()
    binned_dir = path / "binned_outputs" / "square_002um"
    if not binned_dir.is_dir():
        raise FileNotFoundError(
            f"binned_outputs/square_002um not found under {path}. "
            "Pass the Space Ranger output directory (Space Ranger's own 'outs/')."
        )

    library_id = library_id or (path.parent.name if path.name == "outs" else path.name)
    log.info("Reading Visium HD run '%s' from %s", library_id, path)

    # ------------------------------------------------------------------ #
    # 1. 2µm bin matrix (var + shape; X stays lazy)                       #
    # ------------------------------------------------------------------ #
    h5_path = _resolve_matrix_h5(binned_dir)
    n_features, n_barcodes, var_df, barcode_order = _read_h5_metadata(h5_path)

    spatial_dir = binned_dir / "spatial"
    tissue_positions_df = _read_tissue_positions(spatial_dir)
    pixel_size_um = _resolve_pixel_size(spatial_dir, tissue_positions_df, _NATIVE_BIN_SIZE_UM)

    obs_df = tissue_positions_df.reindex(barcode_order)
    missing = obs_df["array_row"].isna()
    if missing.any():
        n_missing = int(missing.sum())
        log.warning(
            "%d / %d bins in the count matrix have no matching row in "
            "tissue_positions.parquet; their obs columns will be NaN.",
            n_missing, len(obs_df),
        )

    bin_adata = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "array_row": obs_df["array_row"].to_numpy(),
                "array_col": obs_df["array_col"].to_numpy(),
                "in_tissue": obs_df["in_tissue"].to_numpy(),
            },
            index=pd.Index(barcode_order, name="barcode"),
        ),
        var=var_df,
    )
    bin_adata.obsm["spatial"] = np.column_stack(
        [
            obs_df["pxl_col_in_fullres"].to_numpy(dtype=np.float64) * pixel_size_um,
            obs_df["pxl_row_in_fullres"].to_numpy(dtype=np.float64) * pixel_size_um,
        ]
    ).astype(np.float32)

    bin_matrix_spec = BinMatrixSpec(
        path=str(h5_path),
        bin_size_um=_NATIVE_BIN_SIZE_UM,
        n_obs=n_barcodes,
        n_vars=n_features,
    )
    bin_adata.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "source_path": str(path),
            "images": {},
            "bin_matrix": bin_matrix_spec,
        }
    }
    bin_adata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    mdata = md.MuData({"visium_hd_bins": bin_adata})
    mdata.uns["spatial"] = dict(bin_adata.uns["spatial"])
    mdata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    mdata.uns[UNS_DEFAULT_MODALITY_KEY] = "visium_hd_bins"

    log.info(
        "Loaded %d bins × %d features (lazy — X not materialized; pixel_size=%.4f µm)",
        bin_adata.n_obs, bin_adata.n_vars, pixel_size_um,
    )

    # ------------------------------------------------------------------ #
    # 2. Paired H&E image (loaded by default — Visium HD ships one)       #
    # ------------------------------------------------------------------ #
    if load_he:
        _attach_he_image(mdata, path, he_path, pixel_size_um=pixel_size_um, spatial_dir=spatial_dir)

    # ------------------------------------------------------------------ #
    # 3. Native on-board segmentation (Space Ranger v4.0+)                 #
    # ------------------------------------------------------------------ #
    if load_segmented:
        seg_dir = _resolve_segmented_outputs_dir(path, segmented_outputs_dir)
        if seg_dir is not None:
            _attach_native_segmentation(mdata, seg_dir, pixel_size_um=pixel_size_um, dtype=dtype)

    return mdata


# ---------------------------------------------------------------------------
# Private helpers — bin matrix metadata (light h5py-only pass)
# ---------------------------------------------------------------------------


def _resolve_matrix_h5(binned_dir: Path) -> Path:
    for name in _MATRIX_CANDIDATES:
        candidate = binned_dir / name
        if candidate.exists():
            if name != _MATRIX_CANDIDATES[0]:
                log.info("%s not found; using %s", _MATRIX_CANDIDATES[0], name)
            return candidate
    raise FileNotFoundError(
        f"None of {_MATRIX_CANDIDATES} found in {binned_dir}"
    )


def _read_h5_metadata(h5_path: Path) -> tuple[int, int, pd.DataFrame, list[str]]:
    """Read only /matrix/shape, /matrix/features/*, and /matrix/barcodes.

    Never touches /matrix/data, /matrix/indices, or /matrix/indptr — the
    arrays that scale with the number of non-zero entries rather than the
    (comparatively tiny) number of features/barcodes.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["matrix"]
        n_features, n_barcodes = (int(x) for x in np.asarray(grp["shape"]))

        feat = grp["features"]
        gene_id = np.asarray(feat["id"]).astype(str)
        gene_name = np.asarray(feat["name"]).astype(str) if "name" in feat else gene_id
        var_dict = {"gene_id": gene_id, "gene_name": gene_name}
        if "feature_type" in feat:
            var_dict["feature_type"] = np.asarray(feat["feature_type"]).astype(str)

        barcode_order = np.asarray(grp["barcodes"]).astype(str).tolist()

    # gene_name is the index (matches Xenium/MERSCOPE/CosMx: users reference
    # genes by symbol, e.g. pl.points(mdata, "EPCAM")); gene_id kept as a column.
    var_df = pd.DataFrame(var_dict, index=pd.Index(gene_name, name="gene_name"))
    return n_features, n_barcodes, var_df, barcode_order


# ---------------------------------------------------------------------------
# Private helpers — tissue_positions.parquet / scalefactors_json.json
# ---------------------------------------------------------------------------


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    tp_path = spatial_dir / "tissue_positions.parquet"
    if not tp_path.exists():
        # Older Space Ranger versions may still export a csv.
        tp_path_csv = spatial_dir / "tissue_positions.csv"
        if tp_path_csv.exists():
            df = pd.read_csv(tp_path_csv)
        else:
            raise FileNotFoundError(f"tissue_positions.parquet not found in {spatial_dir}")
    else:
        df = pd.read_parquet(tp_path)
    df = df.set_index("barcode")
    df.index = df.index.astype(str)
    return df


def _read_scalefactors(spatial_dir: Path) -> dict:
    """Read scalefactors_json.json, or {} if absent."""
    sf_path = spatial_dir / "scalefactors_json.json"
    if not sf_path.exists():
        return {}
    with sf_path.open() as fh:
        return json.load(fh)


def _resolve_pixel_size(
    spatial_dir: Path, tissue_positions_df: pd.DataFrame, bin_size_um: float
) -> float:
    """Resolve microns-per-pixel for the **fullres tissue image** frame that
    bin centroids (``pxl_row/col_in_fullres``) are already expressed in.

    Prefers scalefactors_json.json's own ``microns_per_pixel``, else derives
    it empirically from the bin grid spacing. Deliberately does **not** fall
    back to ``regist_target_img_scalef_microns_per_pixel`` — that key
    describes the CytAssist registration-target image's *own*, different
    pixel scale (see :func:`_resolve_cytassist_pixel_size`), not the fullres
    image bins are already placed in; conflating the two silently
    mis-registers whichever image ends up using this value.
    """
    scalefactors = _read_scalefactors(spatial_dir)
    if scalefactors.get("microns_per_pixel"):
        return float(scalefactors["microns_per_pixel"])

    log.info(
        "scalefactors_json.json has no microns_per_pixel field; deriving pixel_size_um "
        "empirically from the bin grid spacing."
    )
    return _empirical_pixel_size(tissue_positions_df, bin_size_um)


def _resolve_cytassist_pixel_size(spatial_dir: Path) -> float | None:
    """Resolve microns-per-pixel for the CytAssist registration-target image.

    ``None`` if ``scalefactors_json.json`` has no
    ``regist_target_img_scalef_microns_per_pixel`` entry — the caller must
    not silently substitute the fullres image's pixel size instead (that is
    exactly the bug this helper exists to avoid: the CytAssist image is
    captured by a different scanner at a different pixel resolution, so
    registering it with the fullres scale places it in a coordinate frame
    disjoint from the actual bin/cell data).
    """
    scalefactors = _read_scalefactors(spatial_dir)
    value = scalefactors.get("regist_target_img_scalef_microns_per_pixel")
    return float(value) if value else None


def _empirical_pixel_size(df: pd.DataFrame, bin_size_um: float) -> float:
    """Median pixel spacing between array-adjacent bins, converted to µm/px.

    Visium HD bins sit on a regular square lattice indexed by
    (array_row, array_col); the pixel distance between two bins one
    array_col apart (same array_row) is exactly one bin_size_um step.
    """
    d = df.sort_values(["array_row", "array_col"])
    same_row = d["array_row"].to_numpy()[1:] == d["array_row"].to_numpy()[:-1]
    col_step = d["array_col"].to_numpy()[1:] - d["array_col"].to_numpy()[:-1]
    adjacent = same_row & (col_step == 1)
    if not adjacent.any():
        log.warning(
            "Could not find array-adjacent bin pairs to derive pixel_size_um empirically; "
            "falling back to 1.0 µm/px. Pass a corrected pixel_size_um manually if needed."
        )
        return 1.0
    px_col = d["pxl_col_in_fullres"].to_numpy(dtype=np.float64)
    pixel_steps = np.abs(px_col[1:][adjacent] - px_col[:-1][adjacent])
    median_step_px = float(np.median(pixel_steps))
    if median_step_px <= 0:
        return 1.0
    return bin_size_um / median_step_px


# ---------------------------------------------------------------------------
# Private helpers — H&E image
# ---------------------------------------------------------------------------


def _find_he_image(path: Path) -> tuple[Path, bool] | None:
    """Return ``(image_path, is_cytassist)``, or ``None`` if nothing is found."""
    for pattern in _HE_PATTERNS:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0], False
    # Fall back to the CytAssist stain image (present in a full `outs/` run).
    # This image is captured by a different scanner at a different pixel
    # resolution than the fullres tissue image -- callers must register it
    # with its own scale (_resolve_cytassist_pixel_size), not pixel_size_um.
    for candidate in (path / "spatial" / "cytassist_image.tiff",
                      path / "binned_outputs" / "square_002um" / "spatial" / "cytassist_image.tiff"):
        if candidate.exists():
            return candidate, True
    return None


def _attach_he_image(
    mdata: md.MuData,
    path: Path,
    he_path: str | Path | None,
    *,
    pixel_size_um: float,
    spatial_dir: Path,
) -> None:
    is_cytassist = False
    if he_path is not None:
        img_path = Path(he_path)
        if not img_path.is_absolute():
            img_path = path / img_path
        if not img_path.exists():
            raise FileNotFoundError(f"he_path: specified file not found: {img_path}")
    else:
        found = _find_he_image(path)
        if found is None:
            log.warning(
                "No H&E image found under %s (checked %s and spatial/cytassist_image.tiff); "
                "skipping. Pass he_path= to point at it explicitly.",
                path, _HE_PATTERNS,
            )
            return
        img_path, is_cytassist = found

    if is_cytassist:
        # The CytAssist registration-target image is captured by a different
        # scanner at a different pixel resolution than the fullres tissue
        # image bin centroids are already placed in -- registering it with
        # the bin-space pixel_size_um silently places it in a coordinate
        # frame disjoint from the actual bin/cell data. Its own scale
        # (regist_target_img_scalef_microns_per_pixel) must be used instead;
        # if that field isn't present there is no reliable way to register
        # it, so skip attaching rather than silently mis-registering.
        cytassist_pixel_size_um = _resolve_cytassist_pixel_size(spatial_dir)
        if cytassist_pixel_size_um is None:
            log.warning(
                "Found only a CytAssist image (%s) and scalefactors_json.json has "
                "no regist_target_img_scalef_microns_per_pixel entry -- cannot "
                "reliably register it against the bin coordinate frame. Skipping. "
                "Pass he_path= to an already-registered image, or supply the scale "
                "factor manually via ab.io.read_he(..., affine=...).",
                img_path,
            )
            return
        image_pixel_size_um = cytassist_pixel_size_um
        log.info(
            "Registering CytAssist fallback image with its own scale (%.4f µm/px) "
            "from regist_target_img_scalef_microns_per_pixel -- best-effort, no "
            "fullres tissue image was found.",
            image_pixel_size_um,
        )
    else:
        # Visium HD's fullres image is already pixel-aligned to pxl_row/col_in_fullres
        # by Space Ranger — the same frame bin centroids were placed in above — so
        # registration is a pure pixel_size_um scale, no separate alignment matrix.
        image_pixel_size_um = pixel_size_um

    affine = np.array(
        [[image_pixel_size_um, 0.0, 0.0], [0.0, image_pixel_size_um, 0.0], [0.0, 0.0, 1.0]]
    )

    read_he(img_path, affine=affine, pixel_size_um=image_pixel_size_um, mdata=mdata, mdata_key="he")
    log.info("Attached H&E image from %s (pixel-aligned to bins)", img_path)


# ---------------------------------------------------------------------------
# Private helpers — native on-board segmentation (Space Ranger v4.0+)
# ---------------------------------------------------------------------------


def _resolve_segmented_outputs_dir(
    path: Path, segmented_outputs_dir: str | Path | None
) -> Path | None:
    if segmented_outputs_dir is not None:
        candidate = Path(segmented_outputs_dir)
        if not candidate.is_absolute():
            candidate = path / candidate
        if not candidate.is_dir():
            raise FileNotFoundError(f"segmented_outputs_dir not found: {candidate}")
        return candidate

    candidate = path / "segmented_outputs"
    return candidate if candidate.is_dir() else None


def _attach_native_segmentation(
    mdata: md.MuData, seg_dir: Path, *, pixel_size_um: float, dtype: str
) -> None:
    h5_path = _resolve_matrix_h5(seg_dir)
    log.info("Loading native Visium HD segmentation from %s", seg_dir)
    cells_adata = sc.read_10x_h5(h5_path, gex_only=False)
    if dtype != "float32":
        cells_adata.X = cells_adata.X.astype(dtype)

    geojson_path = seg_dir / "cell_segmentations.geojson"
    if geojson_path.exists():
        _attach_native_boundaries(cells_adata, geojson_path, pixel_size_um=pixel_size_um)
    else:
        log.warning(
            "%s not found; %s has no obsm['spatial']/obsm['cell_boundaries'] "
            "and its bins cannot be re-derived without it.",
            geojson_path, "visium_hd_cells",
        )

    attach_modality(mdata, cells_adata, "visium_hd_cells", set_default=True)
    log.info(
        "Attached native segmentation: %d cells × %d genes (primary modality)",
        cells_adata.n_obs, cells_adata.n_vars,
    )


def _attach_native_boundaries(cells_adata: ad.AnnData, geojson_path: Path, *, pixel_size_um: float) -> None:
    """Attach obsm['spatial']/obsm['cell_boundaries'] from Space Ranger's cell_segmentations.geojson.

    The GeoJSON's integer ``cell_id`` property maps to the matrix's barcode
    string as ``cellid_{cell_id:09d}-1`` (per 10x's segmented-outputs
    documentation) — this exact format has not been verified against a real
    downloaded dataset and should be treated as best-effort; a low match rate
    is logged loudly rather than silently producing an empty result.
    """
    gdf = gpd.read_file(geojson_path)
    if "cell_id" not in gdf.columns:
        log.warning(
            "cell_segmentations.geojson has no 'cell_id' property; cannot match "
            "polygons to matrix barcodes. Skipping obsm['cell_boundaries']."
        )
        return

    barcode_from_cell_id = gdf["cell_id"].astype(int).map(lambda i: f"cellid_{i:09d}-1")
    pos = pd.Series(np.arange(cells_adata.n_obs, dtype=np.int64), index=cells_adata.obs_names)
    matrix_pos = barcode_from_cell_id.map(pos)

    n_matched = int(matrix_pos.notna().sum())
    match_rate = n_matched / len(gdf) if len(gdf) else 0.0
    if match_rate < 0.5:
        log.warning(
            "Only %.0f%% of cell_segmentations.geojson polygons matched a matrix "
            "barcode via the 'cellid_{cell_id:09d}-1' convention (%d / %d). This "
            "id-format assumption may not hold for this Space Ranger version — "
            "obsm['cell_boundaries'] will be incomplete.",
            match_rate * 100, n_matched, len(gdf),
        )

    n_cells = cells_adata.n_obs
    spatial = np.zeros((n_cells, 2), dtype=np.float32)
    boundaries: list[np.ndarray] = [np.zeros((0, 2), dtype=np.float32) for _ in range(n_cells)]

    valid = matrix_pos.notna()
    for gi, mi in zip(gdf.index[valid], matrix_pos[valid].astype(int)):
        geom = gdf.geometry.loc[gi]
        coords = np.asarray(geom.exterior.coords, dtype=np.float64) * pixel_size_um
        boundaries[mi] = coords.astype(np.float32)
        spatial[mi] = coords.mean(axis=0)

    cells_adata.obsm["spatial"] = spatial
    cells_adata.obsm["cell_boundaries"] = ak.Array(boundaries)
