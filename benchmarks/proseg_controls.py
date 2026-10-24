"""Opt-in Proseg clean/displaced controls with independent molecule truth.

Requires an explicitly supplied Proseg 3.2 executable. No downloads. The seed
controls simulation only: Proseg 3.2 exposes no seed. Results measure assignment
accuracy, marker leakage, and retention of a real boundary-associated gene.
Do not interpret this small synthetic benchmark as general correction efficacy.
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

import astrolabe as ab


def generate(noisy):
    rng = np.random.default_rng(284)
    rows = []
    for core in ("a", "b"):
        for cell in range(9):
            center = np.array([12*(cell % 3), 12*(cell//3), 0.])
            for molecule in range(160):
                offset = rng.normal(0, [2., 2., .7])
                nuclear = np.linalg.norm(offset[:2]) < 1.5
                xyz = center + offset
                gene = "boundary" if cell % 3 == 1 and molecule % 5 == 0 else f"marker{cell%2}" if molecule % 4 else "shared"
                assigned = cell
                displaced = noisy and not nuclear and cell % 3 < 2 and molecule % 3 == 0
                if displaced:
                    xyz[0] += 10
                    assigned = cell+1
                rows.append({"transcript_id": f"{core}-{cell}-{molecule}", "cell_id": f"{core}{assigned}",
                    "true_cell": f"{core}{cell}", "core_id": core, "library_id": "slide", "feature_name": gene,
                    "x_location": xyz[0], "y_location": xyz[1], "z_location": xyz[2], "overlaps_nucleus": nuclear})
    table = pd.DataFrame(rows)
    obs = pd.DataFrame(index=[f"{core}{cell}" for core in ("a", "b") for cell in range(9)])
    obs["core_id"] = np.repeat(["a", "b"], 9); obs["library_id"] = "slide"
    genes = pd.Index(["marker0", "marker1", "shared", "boundary"])
    counts = sp.csr_matrix((np.ones(len(table), dtype=int), (obs.index.get_indexer(table.cell_id), genes.get_indexer(table.feature_name))), shape=(18, 4))
    a = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=genes))
    a.uns["transcripts"] = pa.Table.from_pandas(table, preserve_index=False)
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", choices=("full", "reduced"), default="full",
                        help="Use Proseg's default sampling schedule, or an explicit reduced smoke run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for noisy in (False, True):
        for diffusion in (False, True):
            case = f"{'displaced' if noisy else 'clean'}_diffusion_{diffusion}"
            original = generate(noisy)
            source = original.uns["transcripts"].to_pandas()
            # Four synthetic genes support two expression components. Preserve
            # the model settings between schedules; change only sampling effort.
            options = ["--ncomponents", "2", "--no-factorization"]
            if args.schedule == "reduced":
                options += ["--burnin-samples", "100", "--samples", "100", "--recorded-samples", "50",
                            "--morphology-steps-per-iter", "10"]
            started = perf_counter()
            results = ab.seg.run_proseg(original, output_dir=args.output / case, executable=args.executable,
                diffusion=diffusion, extra_args=options)
            frames = []
            for reconstructed in results.values():
                frame = reconstructed.uns["transcripts_inferred"].to_pandas()
                frame["predicted_source"] = frame.cell_id.map(reconstructed.obs.original_cell_id)
                frames.append(frame)
            frame = pd.concat(frames, ignore_index=True)
            correct = frame.predicted_source == frame.true_cell
            foreground = frame.reconstruction_status == "assigned"
            boundary = frame.feature_name == "boundary"
            markers = frame.feature_name.isin(["marker0", "marker1"])
            marker_type = frame.feature_name.str[-1]
            assigned_type = frame.predicted_source.str[-1].map(lambda x: str(int(x) % 2) if pd.notna(x) else None)
            report.append({"case": case, "schedule": args.schedule, "elapsed_seconds": perf_counter()-started,
                "n_input": len(frame),
                "initial_correct_fraction": float((source.cell_id == source.true_cell).mean()),
                "correct_fraction_all_input": float(correct.mean()),
                "foreground_fraction": float(foreground.mean()),
                "marker_leakage_fraction_all_markers": float((foreground & markers & (marker_type != assigned_type)).sum()/markers.sum()),
                "true_boundary_gene_correct_fraction": float(correct[boundary].mean())})
            (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(report[-1]), flush=True)
