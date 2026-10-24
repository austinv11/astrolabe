"""Held-out Proseg controls with physical truth and a richer marker panel.

Predeclared cases: clean, 0.8-micron measurement jitter, 20% cytoplasmic
leakage with 4-micron Gaussian displacement, and a 10-micron stress case.
Use seeds 901 and 902, two independent cores and Proseg's default model and
sampling schedule. No fitting parameters are tuned against these outcomes.
These synthetic controls cannot establish efficacy on real tissues.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as sp
from scipy.spatial import cKDTree

import astrolabe as ab


def generate(seed, scenario):
    rng = np.random.default_rng(seed)
    centers = np.array([(12*x, 12*y) for y in range(4) for x in range(4)], float)
    genes = pd.Index([f"marker_a_{i}" for i in range(20)] + [f"marker_b_{i}" for i in range(20)]
                     + [f"rare_{i}" for i in range(4)] + [f"shared_{i}" for i in range(4)]
                     + [f"boundary_{i}" for i in range(4)])
    rows, obs = [], []
    for core in ("a", "b"):
        for cell, center in enumerate(centers):
            cell_type = 2 if cell == 15 else cell % 2
            obs.append({"cell_id": f"{core}{cell}", "core_id": core, "library_id": "slide", "true_type": cell_type})
            weight = np.full(len(genes), .002)
            weight[40:44] = .2 if cell_type == 2 else .002
            if cell_type < 2:
                weight[cell_type*20:(cell_type+1)*20] = .1
            weight[44:48] = .15  # Genuine shared expression.
            weight[48:52] = .25 if cell % 4 in (1, 2) else .005
            feature = rng.choice(genes, 400, p=weight/weight.sum())
            angle = rng.uniform(0, 2*np.pi, 400)
            radius = 5*np.sqrt(rng.uniform(0, 1, 400))
            true_xy = center + np.column_stack([radius*np.cos(angle), radius*np.sin(angle)])
            # Draw every error term in all scenarios to keep biological truth
            # fixed when the observation mechanism changes.
            jitter = rng.normal(0, .8, true_xy.shape)
            leak = (rng.uniform(0, 1, 400) < .2) & (radius > 2)
            shifts = rng.normal(size=true_xy.shape)
            observed = true_xy.copy()
            if scenario == "jitter":
                observed += jitter
            elif scenario in ("leak", "far"):
                observed[leak] += shifts[leak] * (4 if scenario == "leak" else 10)
            distance, nearest = cKDTree(centers).query(observed)
            z = rng.uniform(0, 4, 400)
            for i in range(400):
                rows.append({"transcript_id": f"{core}-{cell}-{i}", "feature_name": feature[i],
                    "cell_id": f"{core}{nearest[i]}" if distance[i] <= 5.2 else "UNASSIGNED",
                    "core_id": core, "library_id": "slide", "overlaps_nucleus": bool(distance[i] <= 2),
                    "x_location": observed[i, 0], "y_location": observed[i, 1], "z_location": z[i],
                    "true_x": true_xy[i, 0], "true_y": true_xy[i, 1],
                    "true_cell": f"{core}{cell}", "true_type": cell_type})
    frame = pd.DataFrame(rows)
    obs = pd.DataFrame(obs).set_index("cell_id", drop=False)
    assigned = frame.cell_id != "UNASSIGNED"
    counts = sp.csr_matrix((np.ones(assigned.sum(), dtype=int),
        (obs.index.get_indexer(frame.loc[assigned, "cell_id"]), genes.get_indexer(frame.loc[assigned, "feature_name"]))),
        shape=(len(obs), len(genes)))
    data = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes))
    data.uns["transcripts"] = pa.Table.from_pandas(frame, preserve_index=False)
    return data


def assess(data, reconstructed):
    original = data.uns["transcripts"].to_pandas().set_index("transcript_id")
    frames = []
    for result in reconstructed.values():
        frame = result.uns["transcripts_inferred"].to_pandas()
        frame["predicted_source"] = frame.cell_id.map(result.obs.original_cell_id)
        assert result.uns["reconstruction"]["provenance"]["count_validation"] == "matched_backend_export"
        assert "expected_counts" in result.layers
        frames.append(frame)
    frame = pd.concat(frames).set_index("transcript_id").loc[original.index]
    assert len(frame) == len(original) and frame.index.is_unique
    rows = []
    for core in ("a", "b"):
        mask = original.core_id == core
        source, fitted = original.loc[mask], frame.loc[mask]
        initial = source.cell_id == source.true_cell
        correct = fitted.predicted_source == source.true_cell
        true_xy = source[["true_x", "true_y"]].to_numpy()
        raw_error = source[["x_location", "y_location"]].to_numpy()-true_xy
        fit_error = fitted[["x_location", "y_location"]].to_numpy()-true_xy
        boundary = source.feature_name.str.startswith("boundary_")
        shared = source.feature_name.str.startswith("shared_")
        rare = source.true_type == 2
        rows.append({"core": core, "n_input": len(source), "initial_accuracy": float(initial.mean()),
            "reconstructed_accuracy": float(correct.mean()),
            "originally_correct_changed": float((~correct[initial]).mean()),
            "originally_wrong_recovered": float(correct[~initial].mean()) if (~initial).any() else None,
            "foreground_fraction": float((fitted.reconstruction_status == "assigned").mean()),
            "raw_position_rmse_um": float(np.sqrt(np.square(raw_error).sum(axis=1).mean())),
            "inferred_position_rmse_um": float(np.sqrt(np.square(fit_error).sum(axis=1).mean())),
            "boundary_gene_accuracy": float(correct[boundary].mean()),
            "shared_gene_accuracy": float(correct[shared].mean()),
            "rare_cell_accuracy": float(correct[rare].mean())})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"seeds": [901, 902], "schedule": "Proseg defaults", "model": "Proseg defaults",
              "truth": "two cores, 16 cells/core, 400 molecules/cell, 52 genes, genuine shared/boundary/rare expression",
              "runs": []}
    for seed in report["seeds"]:
        for scenario in ("clean", "jitter", "leak", "far"):
            data = generate(seed, scenario)
            for diffusion in (False, True):
                started = perf_counter()
                reconstructed = ab.seg.run_proseg(data, executable=args.executable,
                    output_dir=args.output/f"{seed}_{scenario}_{diffusion}", diffusion=diffusion,
                    include_expected_counts=True)
                result = {"seed": seed, "scenario": scenario, "diffusion": diffusion,
                          "elapsed_seconds": perf_counter()-started, "cores": assess(data, reconstructed)}
                report["runs"].append(result)
                (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps(result), flush=True)
