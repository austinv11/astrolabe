"""Explicit common-panel and eligible-library analysis selection."""
from __future__ import annotations

import numpy as np

from astrolabe._coverage import _coverage, _require_measured
from astrolabe._state import _invalidate_spatial
from astrolabe._utils import with_adata


@with_adata
def select_analysis_panel(adata, *, genes=None, libraries=None):
    """Return an AnnData copy with explicitly selected libraries and measured genes.

    ``genes=None`` selects the common measured panel. With explicit genes and
    no libraries, retain only libraries that measured every requested gene.
    Explicit libraries must all have measured the requested genes. Graphs and
    derived spatial results are cleared if observations or features change.
    The source object is retained; rebuild the graph on the returned object.
    """
    panel = adata.uns.get("measured_features")
    key = adata.uns.get("measured_features_library_key", "library_id")
    mask = np.ones(adata.n_obs, dtype=bool)
    if libraries is not None:
        if key not in adata.obs:
            raise ValueError(f"Library selection requires obs[{key!r}]")
        libraries = [libraries] if isinstance(libraries, str) else list(libraries)
        if not set(libraries).issubset(set(adata.obs[key])):
            raise ValueError("Unknown requested library")
        mask &= adata.obs[key].isin(libraries).to_numpy()
    if genes is not None:
        genes = [genes] if isinstance(genes, str) else list(genes)
        if libraries is None:
            eligible, labels = _coverage(adata, genes)
            mask &= np.isin(labels, eligible.index[eligible])
    if not mask.any():
        raise ValueError("No libraries have the requested measured panel")
    selected = adata[mask].copy()
    if genes is None:
        if panel is None:
            genes = list(adata.var_names)
        else:
            _coverage(selected, adata.var_names)
            columns = selected.obs[key].unique()
            genes = list(adata.var_names[panel.loc[adata.var_names, columns].all(axis=1)])
    if not genes:
        raise ValueError("The selected libraries have no common measured features")
    _require_measured(selected, genes)
    selected = selected[:, genes].copy()
    if not mask.all() or not selected.var_names.equals(adata.var_names):
        _invalidate_spatial(selected)
    selected.uns["analysis_panel"] = {"genes": list(genes),
        "libraries": list(selected.obs[key].unique()) if key in selected.obs else []}
    return selected
