"""Fingerprinted segmentation provenance."""
from __future__ import annotations

from astrolabe._dependencies import _digest
from astrolabe._utils import with_adata


@with_adata
def record_segmentation(adata, *, version, source, parameters=None,
                         boundaries_key="cell_boundaries", transcripts_key="transcripts", layer="counts"):
    """Record an immutable fingerprinted segmentation version and provenance.

    Record after importing or computing a segmentation. Version names cannot
    be overwritten. This stores metadata and fingerprints, not additional copies
    of large arrays; retain alternative reconstructions as separate modalities.
    """
    if not isinstance(version, str) or not version or not source:
        raise ValueError("Supply a nonempty version name and source")
    history = adata.uns.setdefault("segmentation_versions", {})
    if version in history:
        raise ValueError("Segmentation version already exists; use a new version name")
    counts = adata.layers[layer] if layer is not None else adata.X
    history[version] = {"source": str(source), "parameters": dict(parameters or {}),
        "cell_ids": _digest(adata.obs_names), "feature_ids": _digest(adata.var_names),
        "counts": _digest(counts), "boundaries": _digest(adata.obsm.get(boundaries_key)),
        "transcripts": _digest(adata.uns.get(transcripts_key)),
        "layer": "X" if layer is None else layer, "n_cells": adata.n_obs}
    return history[version]

