"""Ownership and invalidation of results derived from spatial observations."""
from __future__ import annotations

import numpy as np


_DEFAULT_RESULTS = (
    "gradient_params", "gradient_significance", "moran", "moran_local", "moran_bv",
    "moran_local_bv", "getis_ord", "geary_c", "ripley_l", "ligrec", "nhood_enrichment",
    "co_occurrence", "icg", "interaction_changed_genes", "spatially_variable",
    "spatially_variable_genes", "diff_nhood_enrichment",
)


def _spatial_result_keys(adata):
    keys = {key for entry in adata.uns.get("_astrolabe_spatial_results", {}).values()
            for key in entry.get("uns", [])}
    return keys | (set(_DEFAULT_RESULTS) & set(adata.uns))


def _clear_spatial_mirrors(parent, modality, keys):
    """Remove only this modality's entries from registered parent mirrors."""
    if parent is None:
        return
    for key in keys:
        value = parent.uns.get(key)
        if isinstance(value, dict) and modality in value:
            del value[modality]
            if not value:
                del parent.uns[key]


def _invalidate_spatial(adata, *, parent=None, modality=None):
    """Discard owned outputs and graphs after changing cells or geometry."""
    _clear_spatial_mirrors(parent, modality, _spatial_result_keys(adata))
    adata.uns.pop("_astrolabe_dependencies", None)
    gradient_keys = list(adata.uns.get("gradient_params", {}))
    for entry in adata.uns.pop("_astrolabe_spatial_results", {}).values():
        for attr, keys in entry.items():
            for key in keys:
                if key in getattr(adata, attr):
                    del getattr(adata, attr)[key]
    graph_keys = {key for key, value in adata.uns.items() if isinstance(value, dict)
                  and value.get("params", {}).get("spatial_key") is not None}
    while True:
        dependent = {key for key, value in adata.uns.items() if isinstance(value, dict)
                     and value.get("params", {}).get("spatial_neighbors_key") in graph_keys}
        if dependent <= graph_keys:
            break
        graph_keys |= dependent
    for key in graph_keys:
        value = adata.uns.pop(key)
        for name in ("connectivities_key", "distances_key"):
            adata.obsp.pop(value.get(name, ""), None)
    for base in gradient_keys:
        adata.obsm.pop(base, None)
        for key in (base + "_magnitude", base + "_direction"):
            if key in adata.obs:
                del adata.obs[key]
    if gradient_keys:
        for key in ("gradient_unreliable", "gradient_onesidedness"):
            if key in adata.obs:
                del adata.obs[key]
    for key in ("sv_score", "sv_pval", "sv_pval_adj", "sv_spatially_variable",
                "sv_flavor", "sv_n_valid", "sv_p_min", "sv_resolution_limited"):
        if key in adata.var:
            del adata.var[key]
    for key in _DEFAULT_RESULTS:
        adata.uns.pop(key, None)


def _subset_cells(adata, mask, *, inplace, parent=None, modality=None):
    """Subset observations and invalidate results only when cells are removed."""
    mask = np.asarray(mask, dtype=bool)
    if inplace:
        if not mask.all():
            _invalidate_spatial(adata, parent=parent, modality=modality)
        adata._inplace_subset_obs(mask)
        return adata
    result = adata[mask].copy()
    if not mask.all():
        _invalidate_spatial(result)
    return result
