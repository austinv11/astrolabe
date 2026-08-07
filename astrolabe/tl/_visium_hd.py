"""
Bespoke Visium HD bin-resolution conversion.

Visium HD ships pre-computed 8µm/16µm binned matrices alongside the native
2µm resolution, but :func:`~astrolabe.io.read_visium_hd` intentionally reads
only the 2µm bins. :func:`rebin_visium_hd` is the deliberate, computed
alternative: it re-derives coarser (or arbitrary, non-standard) bin sizes
directly from the 2µm data using the same integer-grouping rule Space Ranger
itself uses, rather than requiring a second pre-computed file.
"""

from __future__ import annotations

import logging

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY
from astrolabe._utils import _aggregate_sparse_by_group
from astrolabe.io._io import attach_modality
from astrolabe.pp._visium_hd import load_bin_counts

__all__ = ["rebin_visium_hd"]

log = logging.getLogger(__name__)


def _fmt_um(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def rebin_visium_hd(
    data: ad.AnnData | md.MuData,
    *,
    target_bin_um: float,
    modality: str | None = None,
    key_added: str | None = None,
) -> ad.AnnData:
    """Aggregate native 2µm Visium HD bins into coarser (or custom) square bins.

    Groups bins by ``array_row // factor``, ``array_col // factor`` (the same
    integer-grouping rule Space Ranger itself uses to derive its own 8µm/16µm
    binned outputs from the 2µm data), sums each group's counts, and
    recomputes ``obsm['spatial']`` as the mean centroid per group. Supports
    any integer multiple of the native bin size — not just the standard
    8µm/16µm — e.g. ``target_bin_um=4`` or ``target_bin_um=32``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` (typically ``mdata['visium_hd_bins']``) or
        :class:`~mudata.MuData`. When a MuData is given, the result is also
        attached as a new sibling modality (``key_added``) without changing
        ``uns['default_modality']``.
    target_bin_um
        Target bin size in µm. Must be an integer multiple of the source
        modality's native bin size (``2.0`` for a fresh
        :func:`~astrolabe.io.read_visium_hd` read).
    modality
        Source modality key when *data* is a MuData. Defaults to
        ``'visium_hd_bins'``.
    key_added
        Modality key to attach the result under, when *data* is a MuData.
        Defaults to ``f"visium_hd_bins_{target_bin_um}um"``.

    Returns
    -------
    AnnData
        The rebinned ``AnnData`` (also attached into *data* when it is a
        MuData).

    Examples
    --------
    >>> mdata = ab.io.read_visium_hd("Visium_HD_run/outs/")
    >>> ab.tl.rebin_visium_hd(mdata, target_bin_um=8)
    >>> mdata['visium_hd_bins_8um'].shape
    """
    modality = modality or "visium_hd_bins"
    is_mdata = isinstance(data, md.MuData)
    adata = data[modality] if is_mdata else data

    spatial: dict = adata.uns.get("spatial", {})
    if len(spatial) != 1:
        raise ValueError(
            f"rebin_visium_hd: expected exactly one library in uns['spatial'], "
            f"found {list(spatial)}."
        )
    library_id, lib_meta = next(iter(spatial.items()))
    bin_spec = lib_meta.get("bin_matrix")
    if bin_spec is None:
        raise KeyError(
            f"uns['spatial'][{library_id!r}] has no 'bin_matrix' descriptor. "
            "rebin_visium_hd() expects a modality created by ab.io.read_visium_hd()."
        )
    native_bin_um = bin_spec.bin_size_um

    if target_bin_um <= native_bin_um:
        raise ValueError(
            f"target_bin_um ({target_bin_um}) must be larger than the native "
            f"bin size ({native_bin_um} µm)."
        )
    factor_f = target_bin_um / native_bin_um
    factor = int(round(factor_f))
    if not np.isclose(factor_f, factor):
        raise ValueError(
            f"target_bin_um ({target_bin_um}) must be an integer multiple of "
            f"the native bin size ({native_bin_um} µm)."
        )

    for col in ("array_row", "array_col"):
        if col not in adata.obs.columns:
            raise KeyError(
                f"obs['{col}'] not found; rebin_visium_hd() expects a modality "
                "created by ab.io.read_visium_hd()."
            )

    group_row = adata.obs["array_row"].to_numpy(dtype=np.int64) // factor
    group_col = adata.obs["array_col"].to_numpy(dtype=np.int64) // factor
    col_span = int(group_col.max()) + 1 if len(group_col) else 1
    combined = group_row * col_span + group_col

    unique_combined, group_ids, counts = np.unique(
        combined, return_inverse=True, return_counts=True
    )
    n_groups = len(unique_combined)

    if adata.X is not None:
        X = adata.X
    else:
        log.info(
            "rebin_visium_hd: bin matrix not yet materialized; loading the full "
            "matrix via pp.load_bin_counts."
        )
        X = load_bin_counts(adata, inplace=False).X

    X_new = _aggregate_sparse_by_group(X, group_ids, n_groups)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    sums = np.zeros((n_groups, 2), dtype=np.float64)
    np.add.at(sums, group_ids, coords)
    centroids = (sums / counts[:, None]).astype(np.float32)

    out_row = unique_combined // col_span
    out_col = unique_combined % col_span

    obs = pd.DataFrame(
        {"array_row": out_row, "array_col": out_col, "n_bins": counts},
        index=pd.Index([f"{r}_{c}" for r, c in zip(out_row, out_col)], name="bin_id"),
    )
    result = ad.AnnData(X=X_new, obs=obs, var=adata.var.copy())
    result.obsm["spatial"] = centroids
    result.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "source_path": lib_meta.get("source_path", ""),
            "images": {},
        }
    }
    result.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    log.info(
        "rebin_visium_hd: %d native %sµm bins → %d bins at %sµm",
        adata.n_obs, native_bin_um, n_groups, target_bin_um,
    )

    if is_mdata:
        key_added = key_added or f"visium_hd_bins_{_fmt_um(target_bin_um)}um"
        attach_modality(data, result, key_added, set_default=False)

    return result
