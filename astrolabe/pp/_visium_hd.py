"""
Lazy materialization of Visium HD bin x gene counts.

:func:`~astrolabe.io.read_visium_hd` leaves the bin x gene matrix lazy
(``X=None``, a :class:`~astrolabe._utils.BinMatrixSpec` path descriptor
stored instead) since it can be very large (millions of 2µm bins). This is
the direct analog of :func:`~astrolabe.pp.extract_if_intensities` — the
explicit "materialize it now" step.
"""

from __future__ import annotations

import logging

import anndata as ad
import h5py
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from astrolabe._utils import with_adata

__all__ = ["load_bin_counts"]

log = logging.getLogger(__name__)

_LARGE_PARTIAL_LOAD_WARN = 200_000


def _resolve_bin_matrix_spec(adata: ad.AnnData, library_id: str | None):
    spatial: dict = adata.uns.get("spatial", {})
    if not spatial:
        raise KeyError(
            "No uns['spatial'] found. load_bin_counts() applies to a modality "
            "created by ab.io.read_visium_hd() (typically mdata['visium_hd_bins'])."
        )
    if library_id is None:
        if len(spatial) == 1:
            library_id = next(iter(spatial))
        else:
            raise ValueError(
                f"Multiple libraries in uns['spatial']: {list(spatial)}. Pass library_id=."
            )
    lib_meta = spatial[library_id]
    spec = lib_meta.get("bin_matrix")
    if spec is None:
        raise KeyError(
            f"uns['spatial'][{library_id!r}] has no 'bin_matrix' descriptor. "
            "load_bin_counts() only applies to a modality created by ab.io.read_visium_hd()."
        )
    return spec, library_id


def _resolve_bin_mask(
    adata: ad.AnnData,
    region_um: tuple[float, float, float, float] | None,
    bin_ids: list[str] | None,
) -> np.ndarray | None:
    mask = None
    if bin_ids is not None:
        mask = adata.obs_names.isin(bin_ids)
        n_found = int(mask.sum())
        n_requested = len(set(bin_ids))
        if n_found < n_requested:
            log.warning(
                "load_bin_counts: %d / %d requested bin_ids not found in obs_names; skipped.",
                n_requested - n_found, n_requested,
            )
    if region_um is not None:
        xmin, ymin, xmax, ymax = region_um
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        region_mask = (
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
            (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        )
        mask = region_mask if mask is None else (mask & region_mask)
    return mask


def _resolve_gene_mask(adata: ad.AnnData, genes: list[str] | None) -> np.ndarray | None:
    if genes is None:
        return None
    mask = np.isin(np.asarray(adata.var_names), np.asarray(genes))
    n_found = int(mask.sum())
    n_requested = len(set(genes))
    if n_found < n_requested:
        log.warning(
            "load_bin_counts: %d / %d requested gene(s) not found in var_names; skipped.",
            n_requested - n_found, n_requested,
        )
    return mask


def _read_partial_csc(
    h5_path: str,
    col_indices: np.ndarray,
    row_indices: np.ndarray | None,
    n_vars_out: int,
) -> sp.csr_matrix:
    """Read a ``(len(col_indices), n_vars_out)`` submatrix from a 10x HDF5 matrix.

    The 10x feature-barcode HDF5 format stores the matrix genes × bins in CSC
    layout — bins are the CSC "columns", so selecting a subset of bins means
    reading only their ``indptr``-bounded ``data``/``indices`` slices, never
    the full arrays (which scale with the number of non-zero entries across
    the *entire* matrix, easily 10s–100s of millions for a full VisiumHD run).
    """
    col_indices = np.sort(np.asarray(col_indices, dtype=np.int64))

    with h5py.File(h5_path, "r") as f:
        grp = f["matrix"]
        indptr = np.asarray(grp["indptr"])
        data_ds = grp["data"]
        idx_ds = grp["indices"]
        n_features_total = int(np.asarray(grp["shape"])[0])

        row_remap = None
        if row_indices is not None:
            row_remap = -np.ones(n_features_total, dtype=np.int64)
            row_remap[row_indices] = np.arange(len(row_indices))

        data_chunks: list[np.ndarray] = []
        row_out_chunks: list[np.ndarray] = []
        col_out_chunks: list[np.ndarray] = []

        for out_col, col in enumerate(col_indices):
            start, end = int(indptr[col]), int(indptr[col + 1])
            if end == start:
                continue
            d = data_ds[start:end]
            r = idx_ds[start:end]
            if row_remap is not None:
                mapped = row_remap[r]
                keep = mapped >= 0
                d = d[keep]
                r = mapped[keep]
            if len(d) == 0:
                continue
            data_chunks.append(d)
            row_out_chunks.append(np.full(len(d), out_col, dtype=np.int64))
            col_out_chunks.append(r)

    if not data_chunks:
        return sp.csr_matrix((len(col_indices), n_vars_out), dtype=np.float32)

    data = np.concatenate(data_chunks).astype(np.float32)
    rows = np.concatenate(row_out_chunks)
    cols = np.concatenate(col_out_chunks)
    return sp.coo_matrix((data, (rows, cols)), shape=(len(col_indices), n_vars_out)).tocsr()


@with_adata
def load_bin_counts(
    adata: ad.AnnData,
    *,
    genes: list[str] | None = None,
    region_um: tuple[float, float, float, float] | None = None,
    bin_ids: list[str] | None = None,
    library_id: str | None = None,
    layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Materialize (all or part of) a lazily-referenced Visium HD bin x gene matrix.

    Passing *region_um* or *bin_ids* subsets *adata* to just the matching
    bins (like :func:`~astrolabe.pp.crop`) and reads only those bins'
    columns directly from the HDF5 matrix via ``h5py`` — memory scales with
    the requested subset, not the full matrix. With neither given, the full
    matrix is loaded via :func:`scanpy.read_10x_h5` (fast, but loads
    everything — a warning is logged naming the resulting shape).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` — typically
        ``mdata['visium_hd_bins']`` (or ``mdata`` itself, resolved via
        ``uns['default_modality']``/``modality=``).
    genes
        Restrict to these gene names (matched against ``var_names``).
        ``None`` (default) loads all genes.
    region_um
        ``(xmin, ymin, xmax, ymax)`` bounding box in µm. Only bins whose
        ``obsm['spatial']`` centroid falls inside are kept/loaded.
    bin_ids
        Explicit bin barcodes (``obs_names``) to keep/load. Combined with
        *region_um* via intersection when both are given.
    library_id
        Library key in ``uns['spatial']``. Inferred automatically when there
        is exactly one.
    layer
        If given, write into ``adata.layers[layer]`` instead of ``adata.X``.
    inplace
        If ``True`` (default), modify *adata* in place (subsetting obs/var
        as needed) and return ``None``. If ``False``, operate on a copy and
        return it.

    Returns
    -------
    None or AnnData

    Examples
    --------
    >>> mdata = ab.io.read_visium_hd("Visium_HD_run/outs/")
    >>> ab.pp.load_bin_counts(mdata, region_um=(0, 0, 500, 500))  # crop-sized subset
    >>> mdata['visium_hd_bins'].X.shape
    (1234, 18085)
    """
    spec, library_id = _resolve_bin_matrix_spec(adata, library_id)

    bin_mask = _resolve_bin_mask(adata, region_um, bin_ids)
    gene_mask = _resolve_gene_mask(adata, genes)

    target = adata if inplace else adata.copy()

    if bin_mask is not None:
        from astrolabe._state import _subset_cells
        _subset_cells(target, bin_mask, inplace=True)
    if gene_mask is not None:
        target._inplace_subset_var(gene_mask)

    if bin_mask is not None:
        col_indices = np.where(bin_mask)[0]
        if len(col_indices) > _LARGE_PARTIAL_LOAD_WARN:
            log.warning(
                "load_bin_counts: partial load of %d bins requested — this reads one "
                "HDF5 column per bin and may be slow. Omit region_um/bin_ids to use "
                "the faster full-matrix load instead.",
                len(col_indices),
            )
        row_indices = np.where(gene_mask)[0] if gene_mask is not None else None
        n_vars_out = len(row_indices) if row_indices is not None else spec.n_vars
        log.info(
            "load_bin_counts: reading %d bins × %d genes from %s",
            len(col_indices), n_vars_out, spec.path,
        )
        X = _read_partial_csc(spec.path, col_indices, row_indices, n_vars_out)
    else:
        log.warning(
            "load_bin_counts: materializing the full %d × %d bin matrix (no region_um/"
            "bin_ids given) — this may use substantial memory for large tissue sections.",
            target.n_obs, target.n_vars,
        )
        full = sc.read_10x_h5(spec.path, gex_only=False)
        # Positional indexing, not name-based: `full` is read from the exact
        # same underlying HDF5 file (spec.path) that produced adata's obs/var
        # in the first place, in the same order — bin_mask is None in this
        # branch by construction (that's the condition for reaching it), so
        # every row already lines up 1:1 with no subsetting needed. gene_mask
        # (only set when genes= was passed) was computed against that same
        # pre-subset var order, so it applies positionally too. Name-based
        # reindexing (the previous `full[target.obs_names, target.var_names]`)
        # broke here: 10x gene panels routinely have duplicate gene symbols
        # (multiple gene_ids sharing one name), and pandas refuses to reindex
        # by a non-unique Index (`InvalidIndexError`).
        if full.n_obs != target.n_obs or not np.array_equal(
            np.asarray(full.obs_names), np.asarray(target.obs_names)
        ):
            raise RuntimeError(
                "load_bin_counts: full HDF5 read's bin order doesn't match "
                f"adata's obs_names ({full.n_obs} vs {target.n_obs} bins) — "
                "the BinMatrixSpec path may point at a different file than "
                "the one obs/var were built from."
            )
        if gene_mask is not None:
            full = full[:, gene_mask].copy()
        X = full.X.astype(np.float32)

    if layer is not None:
        target.layers[layer] = X
    else:
        target.X = X

    return None if inplace else target
