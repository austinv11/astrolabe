"""Measured-panel eligibility, distinct from zero expression."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _coverage(adata, genes, *, feature_names=None):
    """Return per-library eligibility and observation labels for requested genes."""
    genes = list(dict.fromkeys(genes))
    missing = set(genes) - set(adata.var_names if feature_names is None else feature_names)
    if missing:
        raise KeyError(f"Unknown features: {sorted(missing)}")
    panel = adata.uns.get("measured_features")
    if panel is None:
        return pd.Series([True], index=["all"]), np.repeat("all", adata.n_obs)
    if not isinstance(panel, pd.DataFrame) or not panel.index.is_unique or not panel.columns.is_unique:
        raise ValueError("measured_features must have unique feature and library labels")
    key = adata.uns.get("measured_features_library_key", "library_id")
    if key not in adata.obs or adata.obs[key].isna().any():
        raise ValueError(f"Measured-feature coverage requires complete obs[{key!r}]")
    labels = adata.obs[key].to_numpy()
    libraries = pd.Index(pd.unique(labels))
    if not libraries.isin(panel.columns).all() or not pd.Index(genes).isin(panel.index).all():
        raise ValueError("Measured-feature metadata is incomplete for the selected cells/features")
    selected = panel.loc[genes, libraries]
    if selected.isna().any().any() or not selected.isin([True, False]).all().all():
        raise ValueError("Measured-feature coverage must contain boolean values")
    return selected.astype(bool).all(axis=0), labels


def _require_measured(adata, genes, *, feature_names=None):
    """Reject implicit analysis of structural zeros from unmeasured panels."""
    if "measured_features" not in adata.uns:
        return
    eligible, _ = _coverage(adata, genes, feature_names=feature_names)
    if not eligible.all():
        raise ValueError(
            "Requested features were not measured in every selected library: "
            f"{list(eligible.index[~eligible])}. Select an explicit common panel or "
            "eligible libraries with pp.select_analysis_panel."
        )


def _eligible_cells(adata, genes):
    eligible, labels = _coverage(adata, genes)
    return np.isin(labels, eligible.index[eligible]), eligible
