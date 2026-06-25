"""
Reader for NanoString CosMx SMI output directories.

Canonical call
--------------
>>> import astrolabe as ab
>>> mdata = ab.io.read_cosmx("CosMx_run/")

Schema
------
mdata['cosmx']
    X           : (n_cells, n_genes)  sparse float32, raw transcript counts
    obs         : cell_id (index), fov, cell_area (µm², converted from px² via pixel_size_um²),
                  width, height  (all optional)
    obsm
        spatial            : float32 (n_cells, 2)  [x_centroid, y_centroid] in µm
        cell_boundaries    : awkward (n_cells,) of (n_verts, 2) float32 polygons
    var         : gene_name (index)
    uns
        spatial
            <library_id>
                coordinate_unit : 'microns'
                pixel_size_um   : float
                source_path     : str
                images          : {}
        default_modality : 'cosmx'
        astrolabe_schema_version : str
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import awkward as ak
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY, UNS_DEFAULT_MODALITY_KEY

__all__ = ["read_cosmx"]

log = logging.getLogger(__name__)

# CosMx pixel edge length is 120 nm = 0.12028 µm (NanoString SMI ReadMe).
_DEFAULT_PIXEL_SIZE_UM = 0.12028

# Gene names with these prefixes are negative control probes.
_CONTROL_PREFIXES = ("NegPrb", "Negative", "SystemControl", "FalseCode")


def read_cosmx(
    path: str | Path,
    *,
    library_id: str | None = None,
    load_boundaries: bool = True,
    pixel_size_um: float = _DEFAULT_PIXEL_SIZE_UM,
    dtype: str = "float32",
) -> md.MuData:
    """Read a NanoString CosMx SMI output directory into a :class:`mudata.MuData`.

    Parameters
    ----------
    path
        Path to the CosMx output directory.  Must contain the expression
        matrix file (``*_exprMat_file.csv``) and optionally the metadata
        file (``*_metadata_file.csv``) and polygons file (``*_polygons.csv``).
    library_id
        Identifier stored under ``uns['spatial']``. Defaults to the
        directory name.
    load_boundaries
        If ``True`` (default), load cell boundary polygons from
        ``*_polygons.csv`` and store them in ``obsm['cell_boundaries']``.
        Set ``False`` to skip for faster loading.
    pixel_size_um
        Microns per pixel. Default is ``0.12028`` (120 nm pixel edge length
        per NanoString SMI ReadMe). Used to convert global pixel coordinates
        to microns.
    dtype
        NumPy dtype for ``X``.  ``'float32'`` is the default.

    Returns
    -------
    mudata.MuData
        MuData with a single ``'cosmx'`` modality and
        ``uns['default_modality'] = 'cosmx'``.

    Notes
    -----
    CosMx cell IDs are compound keys: ``"{fov}_{cell_ID}"``.  This matches
    the key used in the polygon file so that spatial joins are unambiguous.
    Global pixel coordinates are converted to microns via ``pixel_size_um``.
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"CosMx directory not found: {path}")

    library_id = library_id or path.name
    log.info("Reading CosMx run '%s' from %s", library_id, path)

    # ------------------------------------------------------------------ #
    # 1. Expression matrix  →  X + var                                   #
    # ------------------------------------------------------------------ #
    expr_path = _find_glob(path, "*_exprMat_file.csv", "expression matrix")
    expr_df = pd.read_csv(expr_path)

    # cell_ID == 0 is CosMx's per-FOV "background"/unassigned transcript
    # pseudo-cell (NanoString SMI ReadMe) — it is never a real segmented
    # cell and never appears in the metadata file, so it has no centroid
    # or boundary. Drop it here rather than carry NaN spatial coordinates
    # downstream into neighbor-graph / filtering code that can't handle them.
    if "cell_ID" in expr_df.columns:
        n_bg = int((expr_df["cell_ID"] == 0).sum())
        if n_bg:
            log.info(
                "Dropping %d background pseudo-cell row(s) (cell_ID == 0)", n_bg
            )
            expr_df = expr_df[expr_df["cell_ID"] != 0]

    # Build a compound cell_id key: "{fov}_{cell_ID}"
    expr_df["_cell_id"] = (
        expr_df["fov"].astype(str) + "_" + expr_df["cell_ID"].astype(str)
    )
    expr_df = expr_df.set_index("_cell_id")
    expr_df.index.name = "cell_id"

    # Drop identifier and metadata columns; keep only gene counts
    drop_cols = [c for c in ("fov", "cell_ID") if c in expr_df.columns]
    gene_df = expr_df.drop(columns=drop_cols, errors="ignore")

    X = sp.csr_matrix(gene_df.to_numpy(dtype=dtype))
    var = pd.DataFrame(index=pd.Index(gene_df.columns, name="gene_name"))
    obs = pd.DataFrame(index=gene_df.index)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # ------------------------------------------------------------------ #
    # 2. Metadata  →  obs + obsm['spatial']                              #
    # ------------------------------------------------------------------ #
    meta_path = _find_glob(path, "*_metadata_file.csv", "metadata", required=False)
    if meta_path is not None:
        meta_df = pd.read_csv(meta_path)
        adata = _attach_obs(adata, meta_df, pixel_size_um)
    else:
        log.warning("No metadata file found in %s; obs and obsm['spatial'] not populated.", path)

    # ------------------------------------------------------------------ #
    # 3. Cell boundaries                                                   #
    # ------------------------------------------------------------------ #
    if load_boundaries:
        poly_path = _find_glob(path, "*_polygons.csv", "polygons", required=False)
        if poly_path is not None:
            _attach_boundaries(adata, poly_path, pixel_size_um)
        else:
            log.warning("No polygons file found in %s; skipping obsm['cell_boundaries'].", path)

    # ------------------------------------------------------------------ #
    # 4. Control probe flag                                               #
    # ------------------------------------------------------------------ #
    adata.var["is_control_probe"] = np.array(
        [any(n.startswith(p) for p in _CONTROL_PREFIXES) for n in adata.var_names],
        dtype=bool,
    )
    n_ctrl = int(adata.var["is_control_probe"].sum())
    if n_ctrl:
        log.info(
            "Flagged %d / %d features as control probes (NegPrb/Negative/SystemControl/FalseCode)",
            n_ctrl, adata.n_vars,
        )

    # ------------------------------------------------------------------ #
    # 5. uns['spatial']                                                   #
    # ------------------------------------------------------------------ #
    adata.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "pixel_size_um": pixel_size_um,
            "images": {},
            "source_path": str(path),
        }
    }
    adata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # 6. Wrap in MuData                                                   #
    # ------------------------------------------------------------------ #
    mdata = md.MuData({"cosmx": adata})
    mdata.uns["spatial"] = adata.uns["spatial"]
    mdata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    mdata.uns[UNS_DEFAULT_MODALITY_KEY] = "cosmx"

    log.info("Loaded %d cells × %d genes (%d control probes)", adata.n_obs, adata.n_vars, n_ctrl)
    return mdata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_glob(
    path: Path,
    pattern: str,
    label: str,
    required: bool = True,
) -> Path | None:
    """Find the first file matching *pattern* in *path*; raise if required and missing."""
    matches = sorted(path.glob(pattern))
    if not matches:
        # Also try in immediate subdirectories (some CosMx exports have a flat vs nested layout)
        matches = sorted(path.glob(f"*/{pattern}"))
    if not matches:
        if required:
            raise FileNotFoundError(
                f"No {label} file matching '{pattern}' found in {path}."
            )
        return None
    if len(matches) > 1:
        log.debug("Multiple %s files found; using %s", label, matches[0])
    return matches[0]


def _attach_obs(
    adata: ad.AnnData,
    meta_df: pd.DataFrame,
    pixel_size_um: float,
) -> ad.AnnData:
    """Merge CosMx metadata into obs and extract obsm['spatial'] in microns."""
    # Build compound key to match adata.obs_names
    if "fov" in meta_df.columns and "cell_ID" in meta_df.columns:
        meta_df["_cell_id"] = (
            meta_df["fov"].astype(str) + "_" + meta_df["cell_ID"].astype(str)
        )
        meta_df = meta_df.set_index("_cell_id")
    elif "cell_id" in meta_df.columns:
        meta_df = meta_df.set_index("cell_id")
    meta_df.index.name = "cell_id"

    meta_aligned = meta_df.reindex(adata.obs_names)

    # Global centroid columns (in pixels) — normalize to various naming conventions
    x_col = next(
        (c for c in ("CenterX_global_px", "x_global_px", "x_centroid_global_px", "CenterX_local_px")
         if c in meta_aligned.columns),
        None,
    )
    y_col = next(
        (c for c in ("CenterY_global_px", "y_global_px", "y_centroid_global_px", "CenterY_local_px")
         if c in meta_aligned.columns),
        None,
    )

    if x_col and y_col:
        xy_um = meta_aligned[[x_col, y_col]].to_numpy(dtype=np.float32) * pixel_size_um
        adata.obsm["spatial"] = xy_um
    else:
        log.warning("Could not find global centroid columns in metadata; obsm['spatial'] not set.")

    # Keep informative obs columns
    keep_cols = [
        c for c in ("fov", "Area", "Width", "Height", "cell_ID")
        if c in meta_aligned.columns
    ]
    if "Area" in meta_aligned.columns:
        # Convert Area from px² to µm²
        meta_aligned = meta_aligned.copy()
        meta_aligned["cell_area"] = meta_aligned["Area"] * (pixel_size_um ** 2)
        keep_cols = [c for c in keep_cols if c != "Area"] + ["cell_area"]

    if keep_cols:
        adata.obs = meta_aligned[keep_cols].copy()
    adata.obs.index.name = "cell_id"

    return adata


def _attach_boundaries(
    adata: ad.AnnData,
    poly_path: Path,
    pixel_size_um: float,
) -> None:
    """Load CosMx polygon file and store cell boundaries as an awkward array.

    CosMx polygon CSV columns:
        cell_ID, fov, x_local_px, y_local_px  (or globalX / globalY variants)
    Each row is one vertex; rows for the same cell follow consecutively.
    """
    log.debug("Reading boundaries from %s", poly_path)
    import os
    file_mb = os.path.getsize(poly_path) / 1e6
    if file_mb > 500:
        log.warning(
            "CosMx boundary file is %.0f MB; loading into RAM may be slow. "
            "Consider filtering to a subregion before calling read_cosmx.",
            file_mb,
        )
    poly_df = pd.read_csv(poly_path)

    # Build compound cell_id key (same scheme as obs_names)
    if "fov" in poly_df.columns and "cell_ID" in poly_df.columns:
        poly_df["_cell_id"] = (
            poly_df["fov"].astype(str) + "_" + poly_df["cell_ID"].astype(str)
        )
    elif "cell_id" in poly_df.columns:
        poly_df["_cell_id"] = poly_df["cell_id"].astype(str)
    else:
        log.warning("Cannot identify cell ID column in polygon file; skipping boundaries.")
        return

    # Coordinate columns — prefer global, fall back to local
    x_col = next(
        (c for c in ("x_global_px", "globalX", "x", "x_local_px", "vertexX")
         if c in poly_df.columns),
        None,
    )
    y_col = next(
        (c for c in ("y_global_px", "globalY", "y", "y_local_px", "vertexY")
         if c in poly_df.columns),
        None,
    )

    if x_col is None or y_col is None:
        log.warning("Cannot find coordinate columns in polygon file; skipping boundaries.")
        return

    # Convert pixels → microns
    poly_df["_x_um"] = poly_df[x_col].astype(np.float32) * pixel_size_um
    poly_df["_y_um"] = poly_df[y_col].astype(np.float32) * pixel_size_um

    n_cells = adata.n_obs
    cell_pos = pd.Series(np.arange(n_cells, dtype=np.int32), index=adata.obs_names)

    cell_idx = poly_df["_cell_id"].map(cell_pos)
    valid = cell_idx.notna()
    if not valid.all():
        poly_df = poly_df[valid]
        cell_idx = cell_idx[valid]

    cell_idx_arr = cell_idx.to_numpy(dtype=np.int32)
    order = np.argsort(cell_idx_arr, kind="stable")
    cell_idx_sorted = cell_idx_arr[order]
    coords = poly_df[["_x_um", "_y_um"]].to_numpy(dtype=np.float32)[order]

    counts = np.bincount(cell_idx_sorted, minlength=n_cells).astype(np.int64)
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    adata.obsm["cell_boundaries"] = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.contents.NumpyArray(coords),
        )
    )

    log.debug("Attached %d boundary polygons", int((counts > 0).sum()))
