"""Optional Proseg execution and auditable alternative reconstructions."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp

from astrolabe._dependencies import _digest
from astrolabe._spatial_units import _units
from astrolabe._utils import with_adata


def _frame(table):
    if isinstance(table, (str, Path)):
        return pd.read_csv(table, dtype={"original_cell_id": str})
    return (table.to_pandas() if hasattr(table, "to_pandas") else table.copy()).reset_index(drop=True)


def _read_count_export(path, genes, result, *, integer):
    """Align Proseg's cell-by-gene Matrix Market export by explicit identities."""
    from scipy.io import mmread

    names = pd.Index(genes["gene"])
    if names.hasnans or not names.is_unique or not names.isin(result.var_names).all():
        raise ValueError("Backend count export requires unique measured gene identities")
    matrix = sp.coo_matrix(mmread(path))
    if matrix.shape != (result.n_obs, len(names)):
        raise ValueError("Backend count matrix shape disagrees with cell/gene metadata")
    values = matrix.data
    if not np.isfinite(values).all() or (values < 0).any() or (integer and (values != np.floor(values)).any()):
        raise ValueError("Backend counts must be finite and nonnegative; hard counts must be integers")
    cells = result.obs["cell"].to_numpy()
    if not np.array_equal(np.sort(cells), np.arange(result.n_obs)):
        raise ValueError("Backend matrix rows require contiguous cell IDs")
    columns = result.var_names.get_indexer(names)[matrix.col]
    aligned = sp.csr_matrix((values, (matrix.row, columns)), shape=result.shape)
    return aligned[cells.astype(int)]


@with_adata
def import_proseg_reconstruction(adata, transcript_metadata, cell_metadata, *,
                                 transcripts_key="transcripts", id_map=None,
                                 polygons=None, expected_counts=None, provenance=None):
    """Return a separate reconstruction; retain observed and inferred coordinates.

    Proseg 3.2 columns are validated explicitly. Hard foreground assignments
    produce integer counts; optional expected counts remain a separate layer.
    Background, unassigned and filtered molecules remain in the audit table.
    No assignment probability is invented when the backend does not supply one.
    Canonical transcript coordinates are observed by default; select inferred
    positions explicitly with ``select_transcript_coordinates``.
    """
    original = _frame(adata.uns[transcripts_key])
    output, cells = _frame(transcript_metadata), _frame(cell_metadata)
    required = {"transcript_id", "gene", "assignment", "background", "x", "y", "z", "observed_x", "observed_y", "observed_z"}
    if not required.issubset(output) or not {"cell", "original_cell_id", "centroid_x", "centroid_y", "centroid_z"}.issubset(cells):
        raise ValueError("Unsupported Proseg output schema; expected version 3.2 transcript/cell metadata")
    if "transcript_id" not in original or original.transcript_id.isna().any() or original.transcript_id.duplicated().any():
        raise ValueError("Original transcripts require unique, complete IDs")
    if output.transcript_id.isna().any() or output.transcript_id.duplicated().any() or cells.cell.isna().any() or cells.cell.duplicated().any():
        raise ValueError("Proseg transcript and cell IDs must be unique and present")
    if not output.background.isin([True, False]).all():
        raise ValueError("Proseg background flags must be boolean")
    backend_ids = original.transcript_id if id_map is None else pd.Series(id_map)
    if len(backend_ids) != len(original) or backend_ids.duplicated().any():
        raise ValueError("id_map must give one unique backend ID per original molecule")
    lookup = pd.Series(np.arange(len(original)), index=backend_ids.to_numpy())
    if not output.transcript_id.isin(lookup.index).all():
        raise ValueError("Backend returned transcript IDs absent from the original input")
    source_rows = lookup.loc[output.transcript_id].to_numpy()
    if not np.array_equal(output.gene.astype(str).to_numpy(), original.iloc[source_rows].feature_name.astype(str).to_numpy()):
        raise ValueError("Backend gene identities disagree with original molecules")
    if not output.assignment.dropna().isin(cells.cell).all():
        raise ValueError("Transcript assignments refer to missing reconstructed cells")
    inferred = original.copy()
    inferred["source_cell_id"] = original.cell_id if "cell_id" in original else None
    inferred["cell_id"] = pd.Series([None]*len(original), dtype=object)
    inferred["reconstruction_status"] = "filtered"
    cell_ids = pd.Index(["proseg_" + str(value) for value in cells.cell], name="cell_id")
    cell_lookup = dict(zip(cells.cell, cell_ids))
    assignments = output.assignment.map(cell_lookup)
    background = output.background.to_numpy(dtype=bool)
    assigned = assignments.notna().to_numpy() & ~background
    inferred.loc[source_rows, "cell_id"] = assignments.where(assigned, None).to_numpy()
    inferred.loc[source_rows, "reconstruction_status"] = np.where(background, "background", np.where(assigned, "assigned", "unassigned"))
    for axis in ("x", "y"):
        source_col = axis + "_location"
        original_values = original.iloc[source_rows][source_col].to_numpy(dtype=float) if source_col in original else np.zeros(len(output))
        observed = output["observed_" + axis].to_numpy(dtype=float)
        estimated = output[axis].to_numpy(dtype=float)
        if not np.isfinite(np.r_[original_values, observed, estimated]).all():
            raise ValueError("Transcript coordinates must be finite")
        if not np.allclose(original_values, observed, atol=1e-3, rtol=1e-5):
            raise ValueError("Backend observed coordinates do not match the source frame")
        if source_col not in inferred:
            inferred[source_col] = 0.
        inferred[source_col] = inferred[source_col].astype(float)
        inferred.loc[source_rows, source_col] = estimated
    # Proseg 3.2 detrends, clips and rescales z to [0, 1]. Its z output is
    # not a micron displacement and cannot be inverted at clipped extremes.
    # Preserve physical input z and retain both backend coordinates separately.
    for column in ("z", "observed_z"):
        if not np.isfinite(output[column].to_numpy(dtype=float)).all():
            raise ValueError("Backend normalized z coordinates must be finite")
        inferred["backend_" + column + "_normalized"] = np.nan
        inferred.loc[source_rows, "backend_" + column + "_normalized"] = output[column].to_numpy()
    if "z_location" not in inferred:
        inferred["z_location"] = 0.
    observed = inferred.copy()
    for axis in ("x", "y", "z"):
        observed[axis + "_location"] = original[axis + "_location"] if axis + "_location" in original else 0.
    delta = inferred[[a + "_location" for a in "xyz"]].to_numpy()-observed[[a + "_location" for a in "xyz"]].to_numpy()
    inferred["displacement_um"] = observed["displacement_um"] = np.linalg.norm(delta, axis=1)
    if "assignment_probability" in output:
        values = output.assignment_probability.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Assignment probabilities must lie in [0, 1]")
        for frame in (inferred, observed):
            frame["assignment_probability"] = np.nan
            frame.loc[source_rows, "assignment_probability"] = values
    features = adata.var_names
    if not output.gene.isin(features).all():
        raise ValueError("Reconstruction contains features outside the measured source panel")
    rows = cell_ids.get_indexer(assignments[assigned])
    columns = features.get_indexer(output.loc[assigned, "gene"])
    counts = sp.csr_matrix((np.ones(assigned.sum(), dtype=np.int64), (rows, columns)), shape=(len(cells), len(features)))
    cell_obs = cells.copy(); cell_obs.index = cell_ids
    cell_obs["cell_id"] = cell_ids
    # Carry source metadata only through explicit original-cell identity.
    original_ids = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs else pd.Series(adata.obs_names, index=adata.obs_names)
    if not original_ids.is_unique:
        raise ValueError("Import one spatial unit at a time when original cell IDs repeat")
    source_obs = adata.obs.copy(); source_obs.index = original_ids
    if not cells.original_cell_id.astype(str).isin(source_obs.index).all():
        raise ValueError("Reconstructed original_cell_id does not map to source cells")
    inherited = source_obs.reindex(cells.original_cell_id.astype(str))
    for key in inherited:
        if key not in cell_obs:
            cell_obs[key] = inherited[key].to_numpy()
    result = ad.AnnData(counts, obs=cell_obs, var=adata.var.copy())
    result.layers["counts"] = counts.copy()
    result.obsm["spatial"] = cells[["centroid_x", "centroid_y"]].to_numpy(dtype=float)
    result.obs.rename(columns={"centroid_z": "backend_centroid_z_normalized"}, inplace=True)
    if expected_counts is not None:
        values = sp.csr_matrix(expected_counts, dtype=float)
        if values.shape != counts.shape or not np.isfinite(values.data).all() or (values.data < 0).any():
            raise ValueError("Expected counts must be finite, nonnegative and aligned to cell_metadata/source genes")
        result.layers["expected_counts"] = values
    if polygons is not None:
        import geopandas as gpd
        import shapely
        if isinstance(polygons, (str, Path)):
            import gzip
            content = Path(polygons).read_bytes()
            if content[:2] == b"\x1f\x8b":
                content = gzip.decompress(content)
            geometry = gpd.GeoDataFrame.from_features(json.loads(content)["features"])
        else:
            geometry = polygons.copy()
        result.uns["reconstructed_cell_polygons"] = geometry
        # Keep full polygon topology in the annotation table; project only
        # simple rings into the boundary-array representation.
        id_column = "cell" if "cell" in geometry else "cell_id"
        if id_column not in geometry or not geometry[id_column].is_unique:
            raise ValueError("Polygon cell IDs must be unique and explicit")
        ordered = geometry.set_index(id_column).reindex(cells.cell)
        if ordered.geometry.isna().any():
            raise ValueError("Reconstruction is missing cell polygons")
        rings = [shape.geoms[0] if shape.geom_type == "MultiPolygon" and len(shape.geoms) == 1 else shape for shape in ordered.geometry]
        if all(shape.geom_type == "Polygon" and not shape.interiors for shape in rings):
            import awkward as ak
            result.obsm["cell_boundaries"] = ak.Array([np.asarray(shape.exterior.coords).tolist() for shape in rings])
    support_key = adata.uns.get("spatial_units", {}).get("support_key", "tissue_support")
    for key in ("spatial_units", support_key, "measured_features", "measured_features_library_key", "spatial"):
        if key in adata.uns:
            result.uns[key] = copy.deepcopy(adata.uns[key])
    result.uns["transcripts_original"] = pa.Table.from_pandas(original, preserve_index=False)
    result.uns["transcripts_observed"] = pa.Table.from_pandas(observed, preserve_index=False)
    result.uns["transcripts_inferred"] = pa.Table.from_pandas(inferred, preserve_index=False)
    result.uns["transcripts"] = result.uns["transcripts_observed"]
    result.uns["reconstruction"] = {"backend": "Proseg", "coordinate_source": "observed",
        "count_semantics": "integer foreground hard assignments", "input_fingerprint": _digest(original),
        "probability_semantics": "model-conditional" if "assignment_probability" in output else "not supplied",
        "coordinate_semantics": "inferred xy in microns; physical input z retained; backend normalized z stored separately",
        "displacement_dimensions": "xy",
        "source_cells": list(adata.obs_names), "provenance": dict(provenance or {}),
        "input_transcripts_pruned": bool(adata.uns.get("transcripts_pruned", False)),
        "accounting": {str(key): int(value) for key, value in inferred.reconstruction_status.value_counts().items()}}
    return result


@with_adata
def select_transcript_coordinates(adata, *, source):
    """Explicitly select observed or model-inferred molecule positions."""
    if source not in ("observed", "inferred"):
        raise ValueError("source must be observed or inferred")
    key = "transcripts_" + source
    if key not in adata.uns:
        raise ValueError("No reconstruction coordinate table is available")
    from astrolabe._state import _invalidate_spatial
    _invalidate_spatial(adata)
    adata.uns["transcripts"] = adata.uns[key]
    adata.uns["reconstruction"]["coordinate_source"] = source


@with_adata
def run_proseg(adata, *, output_dir, executable="proseg", library_key="auto", core_key="auto",
                transcripts_key="transcripts", diffusion=True, threads=1, extra_args=(),
                include_expected_counts=False):
    """Run Proseg 3.2 per physical unit; return a mapping of units to AnnData.

    Requires an installed executable; never installs software or imports
    SpatialData. Writes auditable inputs, outputs, logs and a manifest into a
    new directory. Proseg 3.2 exposes no seed option: repeated fits need not be
    identical. Supply complete micron transcript tables with initial nuclear
    assignments. Original data are retained; attach the returned reconstruction
    as a separate MuData modality if desired. Integer assignments are checked
    against Proseg's own count export. Set ``include_expected_counts=True`` to
    retain its posterior expected counts in a separate layer; Proseg's text
    export rounds these to two decimal places.
    """
    binary = shutil.which(str(executable))
    if binary is None:
        raise FileNotFoundError("Install Proseg 3.2 separately and provide its executable path")
    binary = str(Path(binary).resolve())
    version = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if not version.startswith("proseg 3.2."):
        raise ValueError(f"Unsupported Proseg version: {version}; this adapter targets 3.2")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not isinstance(include_expected_counts, bool):
        raise ValueError("include_expected_counts must be boolean")
    numeric_options = {"--burnin-samples", "--samples", "--recorded-samples", "--hillclimb",
        "--ncomponents", "--morphology-steps-per-iter", "--cell-compactness",
        "--nuclear-reassignment-prob", "--prior-seg-reassignment-prob", "--burnin-voxel-size",
        "--voxel-size", "--diffusion-probability", "--diffusion-sigma-near", "--diffusion-sigma-far",
        "--diffusion-sigma-z", "--diffusion-proposal-sigma", "--diffusion-proposal-sigma-z",
        "--burnin-dispersion", "--dispersion"}
    switches = {"--no-factorization", "--variable-burnin-dispersion"}
    extra_args = tuple(map(str, extra_args))
    tokens = iter(extra_args)
    for token in tokens:
        if token in switches:
            continue
        if token not in numeric_options:
            raise ValueError(f"Unsupported extra_args option {token!r}; adapter data/output arguments cannot be overridden")
        try:
            value = float(next(tokens))
        except (StopIteration, ValueError) as exc:
            raise ValueError(f"{token} requires a numeric value") from exc
        if not np.isfinite(value):
            raise ValueError(f"{token} requires a finite numeric value")
    original = _frame(adata.uns[transcripts_key])
    required = {"transcript_id", "feature_name", "cell_id", "x_location", "y_location", "z_location", "overlaps_nucleus"}
    if not required.issubset(original) or original.transcript_id.isna().any() or original.transcript_id.duplicated().any():
        raise ValueError("Complete unique transcript IDs, micron xyz, genes, cell IDs and nuclear assignments are required")
    if not np.isfinite(original[["x_location", "y_location", "z_location"]].to_numpy(dtype=float)).all():
        raise ValueError("Transcript coordinates must be finite microns")
    if original.feature_name.isna().any() or not original.feature_name.isin(adata.var_names).all():
        raise ValueError("Every transcript gene must belong to the input feature panel")
    if original.overlaps_nucleus.isna().any() or not original.overlaps_nucleus.isin([True, False]).all():
        raise ValueError("Nuclear assignment flags must be boolean")
    units, columns = _units(adata, library_key, core_key)
    if len(units) > 1 and any(column not in original for column in columns):
        raise ValueError("Multi-unit transcript input requires explicit library/core labels, including unassigned molecules")
    partitions = []
    membership = np.zeros(len(original), dtype=np.int32)
    for labels, indices in units:
        keep = np.ones(len(original), dtype=bool)
        for column, value in zip(columns, labels):
            if column in original:
                keep &= original[column].eq(value).fillna(False).to_numpy(dtype=bool)
        assigned = original.loc[keep, "cell_id"].dropna().astype(str)
        assigned = assigned[assigned != "UNASSIGNED"]
        if not assigned.isin(adata.obs_names[indices]).all():
            raise ValueError(f"Initial assignments must belong to the same spatial unit: {labels}")
        if not keep.any():
            raise ValueError(f"No transcripts for unit {labels}")
        membership += keep
        partitions.append(keep)
    if not np.all(membership == 1):
        raise ValueError("Every transcript must belong to exactly one known spatial unit; check missing or unknown library/core labels")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    results, records = {}, []
    for (labels, indices), keep in zip(units, partitions):
        raw = original.loc[keep].reset_index(drop=True)
        if raw.empty:
            raise ValueError(f"No transcripts for unit {labels}")
        unit_id = hashlib.sha256(repr(tuple(map(str, labels))).encode()).hexdigest()[:16]
        directory = root / ("unit_" + unit_id)
        directory.mkdir()
        prepared = raw.copy()
        prepared["transcript_id"] = np.arange(len(raw), dtype=np.uint64)
        prepared["cell_id"] = prepared.cell_id.fillna("UNASSIGNED").astype(str)
        prepared["overlaps_nucleus"] = prepared.overlaps_nucleus.astype(int)
        prepared.to_csv(directory / "input.csv", index=False)
        raw[["transcript_id"]].assign(backend_id=np.arange(len(raw))).to_csv(directory / "transcript_id_map.csv", index=False)
        command = [binary, "input.csv", "--gene-column", "feature_name", "--transcript-id-column", "transcript_id",
            "--x-column", "x_location", "--y-column", "y_location", "--z-column", "z_location",
            "--cell-id-column", "cell_id", "--cell-id-unassigned", "UNASSIGNED", "--compartment-column", "overlaps_nucleus",
            "--compartment-nuclear", "1", "--nthreads", str(threads), "--output-transcript-metadata", "transcripts.csv.gz",
            "--output-cell-metadata", "cells.csv.gz", "--output-cell-polygons", "polygons.geojson",
            "--output-counts", "counts.mtx.gz", "--output-gene-metadata", "genes.csv.gz", *extra_args]
        if include_expected_counts:
            command.extend(["--output-expected-counts", "expected-counts.mtx.gz"])
        if not diffusion:
            command.append("--no-diffusion")
        record = {"unit": dict(zip(columns, map(str, labels))), "version": version, "command": command,
                  "input_fingerprint": _digest(raw), "diffusion": diffusion, "deterministic_seed": "not supported by Proseg 3.2",
                  "status": "running", "include_expected_counts": include_expected_counts,
                  "input_transcripts_pruned": bool(adata.uns.get("transcripts_pruned", False))}
        records.append(record)
        (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps({"version": version, "runs": records}, indent=2), encoding="utf-8")
        try:
            with (directory / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=directory, check=True, stdout=log, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            record.update(status="failed", error=str(exc))
            (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"version": version, "runs": records}, indent=2), encoding="utf-8")
            raise
        record["status"] = "backend_completed"
        (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        (root / "manifest.json").write_text(json.dumps({"version": version, "runs": records}, indent=2), encoding="utf-8")
        try:
            subset = adata[indices].copy()
            subset.uns[transcripts_key] = pa.Table.from_pandas(raw, preserve_index=False)
            result = import_proseg_reconstruction(subset, directory / "transcripts.csv.gz", directory / "cells.csv.gz",
                transcripts_key=transcripts_key, id_map=np.arange(len(raw)), polygons=directory / "polygons.geojson", provenance=record)
            genes = pd.read_csv(directory / "genes.csv.gz", keep_default_na=False)
            native_counts = _read_count_export(directory / "counts.mtx.gz", genes, result, integer=True)
            if (native_counts != result.layers["counts"]).nnz:
                raise ValueError("Transcript hard assignments disagree with Proseg's count export")
            record["count_validation"] = "matched_backend_export"
            if include_expected_counts:
                result.layers["expected_counts"] = _read_count_export(
                    directory / "expected-counts.mtx.gz", genes, result, integer=False)
                result.uns["reconstruction"]["expected_count_semantics"] = "posterior mean; backend export rounded to two decimal places"
        except Exception as exc:
            record.update(status="import_failed", error=str(exc))
            (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({"version": version, "runs": records}, indent=2), encoding="utf-8")
            raise
        results[unit_id] = result
        record["status"] = "import_completed"
        result.uns["reconstruction"]["provenance"] = dict(record)
        (directory / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"version": version, "runs": records}, indent=2), encoding="utf-8")
    # Return separate units: a caller can attach each one as a distinct modality
    # or concatenate explicitly with io.concat_samples and its coverage policy.
    return results
