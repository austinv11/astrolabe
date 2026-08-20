"""
Reader for Vizgen MERSCOPE output directories.

Canonical call
--------------
>>> import astrolabe as ab
>>> mdata = ab.io.read_merscope("MERSCOPE_run/")

Schema
------
mdata['merscope']
    X           : (n_cells, n_genes)  sparse float32, raw transcript counts
    obs         : cell_id (index), volume, min_x, min_y, max_x, max_y,
                  cell_area (µm², derived from bounding box; optional)
    obsm
        spatial            : float32 (n_cells, 2)  [x_centroid, y_centroid] in µm
        cell_boundaries    : awkward (n_cells,) of (n_verts, 2) float32 polygons
        nucleus_boundaries : awkward (n_cells,) of (n_verts, 2) float32 polygons
    var         : gene_name (index),
                  feature_type ('Gene Expression' | 'Blank Barcode'),
                  is_control_probe (bool, True for Blank-*/Blank_*)
    uns
        spatial
            <library_id>
                coordinate_unit : 'microns'
                source_path     : str
                images          : {}
        default_modality : 'merscope'
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

__all__ = ["read_merscope"]

log = logging.getLogger(__name__)

# Gene names with these prefixes are blank barcodes (negative controls).
_CONTROL_PREFIXES = ("Blank-", "Blank_")

_CELLS_CANDIDATES = [
    "cell_by_gene.csv",
    "cell_by_gene.csv.gz",
]
_META_CANDIDATES = [
    "cell_metadata.csv",
    "cell_metadata.csv.gz",
]
_BOUNDARIES_DIR = "cell_boundaries"


def read_merscope(
    path: str | Path,
    *,
    library_id: str | None = None,
    load_boundaries: bool = True,
    z_level: int = 0,
    dtype: str = "float32",
) -> md.MuData:
    """Read a Vizgen MERSCOPE output directory into a :class:`mudata.MuData`.

    Parameters
    ----------
    path
        Path to the MERSCOPE output directory. Must contain
        ``cell_by_gene.csv`` and ``cell_metadata.csv``.
    library_id
        Identifier stored under ``uns['spatial']``. Defaults to the
        directory name.
    load_boundaries
        If ``True`` (default), load cell and nucleus boundaries from the
        ``cell_boundaries/`` subdirectory and store as ragged arrays in
        ``obsm['cell_boundaries']`` / ``obsm['nucleus_boundaries']``.
        Set ``False`` to skip for faster loading.
    z_level
        Which z-slice boundary file to load (default 0).  MERSCOPE stores
        boundaries per z-level as ``cell_boundaries/feature_data_{z}.parquet``.
    dtype
        NumPy dtype for ``X``.  ``'float32'`` is the default.

    Returns
    -------
    mudata.MuData
        MuData with a single ``'merscope'`` modality and
        ``uns['default_modality'] = 'merscope'``.
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"MERSCOPE directory not found: {path}")

    library_id = library_id or path.name
    log.info("Reading MERSCOPE run '%s' from %s", library_id, path)

    # ------------------------------------------------------------------ #
    # 1. cell_by_gene.csv  →  X                                          #
    # ------------------------------------------------------------------ #
    cbg_df = _read_csv_candidate(path, _CELLS_CANDIDATES, "cell_by_gene")

    # First column is cell_id (named 'Unnamed: 0', 'EntityID', or 'cell' depending on MERSCOPE version)
    first_col = cbg_df.columns[0]
    if first_col == "Unnamed: 0" or first_col.lower() in ("cell_id", "entityid", "cell"):
        cbg_df = cbg_df.set_index(first_col)
    cbg_df.index = cbg_df.index.astype(str)
    cbg_df.index.name = "cell_id"

    # Drop any remaining non-gene metadata columns
    drop_cols = [c for c in cbg_df.columns if c.lower() in ("fov", "volume", "min_x", "min_y", "max_x", "max_y")]
    if drop_cols:
        cbg_df = cbg_df.drop(columns=drop_cols)

    X = sp.csr_matrix(cbg_df.to_numpy(dtype=dtype))
    var = pd.DataFrame(index=pd.Index(cbg_df.columns, name="gene_name"))
    obs = pd.DataFrame(index=cbg_df.index)

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # ------------------------------------------------------------------ #
    # 2. cell_metadata.csv  →  obs + obsm['spatial']                     #
    # ------------------------------------------------------------------ #
    meta_df = _read_csv_candidate(path, _META_CANDIDATES, "cell_metadata")
    adata = _attach_obs(adata, meta_df)

    # ------------------------------------------------------------------ #
    # 3. Cell boundaries                                                   #
    # ------------------------------------------------------------------ #
    if load_boundaries:
        bnd_dir = path / _BOUNDARIES_DIR
        if bnd_dir.is_dir():
            _attach_boundaries(adata, bnd_dir, z_level)
        else:
            log.warning(
                "cell_boundaries/ directory not found in %s; skipping boundaries.", path
            )

    # ------------------------------------------------------------------ #
    # 4. Control probe flag + feature_type                                 #
    # ------------------------------------------------------------------ #
    is_ctrl = np.array(
        [any(n.startswith(p) for p in _CONTROL_PREFIXES) for n in adata.var_names],
        dtype=bool,
    )
    adata.var["is_control_probe"] = is_ctrl
    # MERFISH has a single control class (blank barcodes: codewords with no
    # probe assigned), mirrored into var['feature_type'] (the Xenium
    # convention) so QC plots label controls by type on every platform.
    adata.var["feature_type"] = np.where(is_ctrl, "Blank Barcode", "Gene Expression")
    n_ctrl = int(is_ctrl.sum())
    if n_ctrl:
        log.info("Flagged %d / %d features as blank barcodes (Blank-*)", n_ctrl, adata.n_vars)

    # ------------------------------------------------------------------ #
    # 5. uns['spatial']                                                   #
    # ------------------------------------------------------------------ #
    adata.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "images": {},
            "source_path": str(path),
        }
    }
    adata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # 6. Wrap in MuData                                                   #
    # ------------------------------------------------------------------ #
    mdata = md.MuData({"merscope": adata})
    mdata.uns["spatial"] = adata.uns["spatial"]
    mdata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    mdata.uns[UNS_DEFAULT_MODALITY_KEY] = "merscope"

    log.info("Loaded %d cells × %d genes (%d control probes)", adata.n_obs, adata.n_vars, n_ctrl)
    return mdata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_csv_candidate(path: Path, candidates: list[str], label: str) -> pd.DataFrame:
    for name in candidates:
        p = path / name
        if p.exists():
            log.debug("Reading %s from %s", label, p)
            compression = "gzip" if p.suffix == ".gz" else "infer"
            return pd.read_csv(p, compression=compression)
    raise FileNotFoundError(
        f"No {label} file found in {path}. Expected one of: {candidates}"
    )


def _attach_obs(adata: ad.AnnData, meta_df: pd.DataFrame) -> ad.AnnData:
    """Merge cell metadata into obs and extract obsm['spatial']."""
    # Normalize the cell-ID column; MERSCOPE uses 'EntityID' in some versions
    # and an unnamed integer index in others.
    id_col_candidates = ["EntityID", "cell_id", "Unnamed: 0"]
    id_col = next((c for c in id_col_candidates if c in meta_df.columns), None)
    if id_col is not None:
        meta_df = meta_df.set_index(id_col)
    meta_df.index = meta_df.index.astype(str)
    meta_df.index.name = "cell_id"

    # Align to the expression matrix cell order (best-effort reindex)
    common = adata.obs_names.intersection(meta_df.index)
    if len(common) == 0:
        log.warning(
            "Cell IDs in cell_by_gene do not overlap with cell_metadata; "
            "skipping metadata merge."
        )
        return adata

    if len(common) < adata.n_obs:
        log.warning(
            "%d / %d cells have no metadata entry; spatial coordinates will be NaN.",
            adata.n_obs - len(common),
            adata.n_obs,
        )

    meta_aligned = meta_df.reindex(adata.obs_names)

    # Spatial coordinates — normalize column names across MERSCOPE versions
    x_col = next((c for c in ("center_x", "x_centroid", "x", "center_x_micron") if c in meta_aligned.columns), None)
    y_col = next((c for c in ("center_y", "y_centroid", "y", "center_y_micron") if c in meta_aligned.columns), None)

    if x_col and y_col:
        adata.obsm["spatial"] = meta_aligned[[x_col, y_col]].to_numpy(dtype=np.float32)
    else:
        log.warning("Could not find centroid columns in cell_metadata; obsm['spatial'] not set.")

    # Keep informative obs columns
    keep_cols = [
        c for c in ("volume", "min_x", "min_y", "max_x", "max_y", "fov")
        if c in meta_aligned.columns
    ]
    # Add bounding-box area as a proxy for cell_area when volume is absent
    if "min_x" in meta_aligned.columns and "max_x" in meta_aligned.columns:
        meta_aligned["cell_area"] = (
            (meta_aligned["max_x"] - meta_aligned["min_x"]) *
            (meta_aligned["max_y"] - meta_aligned["min_y"])
        )
        keep_cols.append("cell_area")

    if keep_cols:
        adata.obs = meta_aligned[keep_cols].copy()
    adata.obs.index.name = "cell_id"

    return adata


def _attach_boundaries(adata: ad.AnnData, bnd_dir: Path, z_level: int) -> None:
    """Load cell and nucleus boundaries from MERSCOPE feature_data parquet files.

    MERSCOPE stores boundaries in
    ``cell_boundaries/feature_data_{z}.parquet`` with columns:
    EntityID, ZIndex, Type ('Cell' | 'Nucleus'), Coordinates (list of [x, y]).
    """
    parquet_path = bnd_dir / f"feature_data_{z_level}.parquet"
    if not parquet_path.exists():
        # Try without z suffix (some versions)
        alt = bnd_dir / "feature_data.parquet"
        if alt.exists():
            parquet_path = alt
        else:
            log.warning(
                "Boundary file not found at %s; skipping boundaries.", parquet_path
            )
            return

    log.debug("Reading boundaries from %s", parquet_path)
    bnd_df = pd.read_parquet(parquet_path)

    # Normalize entity ID column
    id_col = next((c for c in ("EntityID", "cell_id", "entity_id") if c in bnd_df.columns), None)
    if id_col is None:
        log.warning("No entity ID column found in boundary file; skipping boundaries.")
        return

    bnd_df[id_col] = bnd_df[id_col].astype(str)

    # Type column distinguishes Cell vs Nucleus boundaries
    type_col = next((c for c in ("Type", "type") if c in bnd_df.columns), None)

    for obsm_key, type_label in (("cell_boundaries", "Cell"), ("nucleus_boundaries", "Nucleus")):
        if type_col is not None:
            subset = bnd_df[bnd_df[type_col] == type_label]
        else:
            # If no Type column, treat all as cell boundaries
            subset = bnd_df if obsm_key == "cell_boundaries" else pd.DataFrame()

        if subset.empty:
            continue

        coords_col = next(
            (c for c in ("Coordinates", "coordinates", "geometry") if c in subset.columns),
            None,
        )
        if coords_col is None:
            log.warning("No Coordinates column in boundary file for %s; skipping.", obsm_key)
            continue

        _build_awkward_boundaries(adata, subset, id_col, coords_col, obsm_key)


def _build_awkward_boundaries(
    adata: ad.AnnData,
    bnd_df: pd.DataFrame,
    id_col: str,
    coords_col: str,
    obsm_key: str,
) -> None:
    """Convert a MERSCOPE boundary DataFrame into an awkward array and store in obsm."""
    n_cells = adata.n_obs
    cell_pos = {cid: i for i, cid in enumerate(adata.obs_names)}

    all_coords: list[np.ndarray] = []
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    vertex_count = np.zeros(n_cells, dtype=np.int64)

    # First pass: determine per-cell vertex counts
    for cid_raw, coords_raw in zip(bnd_df[id_col], bnd_df[coords_col]):
        cid = str(cid_raw)
        idx = cell_pos.get(cid)
        if idx is None:
            continue
        coords = _parse_coords(coords_raw)
        if coords is None or len(coords) == 0:
            continue
        vertex_count[idx] += len(coords)
        all_coords.append((idx, coords))

    np.cumsum(vertex_count, out=offsets[1:])
    total_verts = int(offsets[-1])
    flat_coords = np.zeros((total_verts, 2), dtype=np.float32)

    # Second pass: fill the flat coordinate buffer
    written = np.zeros(n_cells, dtype=np.int64)
    for idx, coords in all_coords:
        start = offsets[idx] + written[idx]
        n = len(coords)
        flat_coords[start : start + n] = coords
        written[idx] += n

    adata.obsm[obsm_key] = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.contents.NumpyArray(flat_coords),
        )
    )


def _parse_coords(raw) -> np.ndarray | None:
    """Parse MERSCOPE Coordinates cell into a (n_verts, 2) float32 array."""
    if raw is None:
        return None

    # Already a numpy array or list of [x, y] pairs
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32)
        if arr.ndim == 1 and len(arr) % 2 == 0:
            arr = arr.reshape(-1, 2)
        return arr if arr.ndim == 2 and arr.shape[1] == 2 else None

    if isinstance(raw, (list, tuple)):
        arr = np.array(raw, dtype=np.float32)
        if arr.ndim == 1 and len(arr) % 2 == 0:
            arr = arr.reshape(-1, 2)
        return arr if arr.ndim == 2 and arr.shape[1] == 2 else None

    # WKT string fallback (rare but possible)
    if isinstance(raw, str):
        try:
            from shapely import wkt
            geom = wkt.loads(raw)
            coords = np.array(list(geom.exterior.coords), dtype=np.float32)
            return coords[:, :2]
        except Exception:
            return None

    return None
