"""Descriptive reconstruction checks; no inferred ground truth or new test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._dependencies import _digest
from astrolabe._utils import from_kwarg, with_adata


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


@with_adata(mirror_uns=from_kwarg("key_added"), spatial_result=True)
def reconstruction_qc(adata, *, marker_pairs=(), min_counts=1, layer="counts",
                        key_added="reconstruction_qc"):
    """Summarize Proseg reassignment, displacement and user-defined marker mixing.

    Reassignment uses the reconstructed cell's source identity, so renamed cells
    are not counted as biological changes. Marker pairs must be supplied from
    independent knowledge; joint detection is descriptive and may be real biology.
    Neither displacement nor marker mixing certifies a reconstruction as correct.
    Proseg 3.2 does not provide assignment probabilities in its transcript export.
    """
    if not isinstance(min_counts, int) or min_counts < 1:
        raise ValueError("min_counts must be a positive integer")
    if "reconstruction" not in adata.uns:
        raise ValueError("Import a reconstruction before computing reconstruction QC")
    molecules = adata.uns["transcripts_inferred"].to_pandas().copy()
    source_ids = adata.obs.original_cell_id.astype(str)
    reconstructed_source = molecules.cell_id.map(source_ids)
    original_source = molecules.source_cell_id.astype("string")
    foreground = molecules.reconstruction_status == "assigned"
    comparable = foreground & original_source.notna() & (original_source != "UNASSIGNED")
    molecules["reassigned"] = comparable & (reconstructed_source.astype("string") != original_source)
    molecules["foreground"] = foreground
    molecules["comparable"] = comparable
    rows = []
    for gene, frame in molecules.groupby("feature_name", sort=False, observed=True):
        retained = frame[frame.reconstruction_status != "filtered"]
        delta = retained.displacement_um.to_numpy(dtype=float)
        rows.append({"gene": gene, "n_input": len(frame), "n_foreground": int(frame.foreground.sum()),
            "n_comparable": int(frame.comparable.sum()), "n_reassigned": int(frame.reassigned.sum()),
            "n_background": int((frame.reconstruction_status == "background").sum()),
            "n_unassigned": int((frame.reconstruction_status == "unassigned").sum()),
            "n_filtered": int((frame.reconstruction_status == "filtered").sum()),
            "median_displacement_um": float(np.median(delta)) if len(delta) else np.nan,
            "p95_displacement_um": float(np.quantile(delta, .95)) if len(delta) else np.nan})
    genes = pd.DataFrame(rows).set_index("gene")
    genes["reassignment_fraction"] = genes.n_reassigned / genes.n_comparable.replace(0, np.nan)
    markers = []
    counts = adata.layers[layer] if layer is not None else adata.X
    from astrolabe._coverage import _require_measured
    for first, second in marker_pairs:
        if first not in adata.var_names or second not in adata.var_names:
            raise ValueError("Marker pairs must name measured features in var_names")
        _require_measured(adata, [first, second])
        values = counts[:, adata.var_names.get_indexer([first, second])]
        values = values.toarray() if sp.issparse(values) else np.asarray(values)
        markers.append({"marker_a": first, "marker_b": second, "n_cells": adata.n_obs,
                        "n_joint_detected": int((values >= min_counts).all(axis=1).sum())})
    result = {"genes": genes, "marker_pairs": pd.DataFrame(markers, columns=["marker_a", "marker_b", "n_cells", "n_joint_detected"]),
              "parameters": {"min_counts": min_counts, "displacement_dimensions": "xy",
                             "probability_semantics": adata.uns["reconstruction"]["probability_semantics"],
                             "interpretation": "descriptive; no ground truth or correctness probability"}}
    adata.uns[key_added] = result
    return result
