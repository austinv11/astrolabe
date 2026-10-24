"""Real Xenium-to-Proseg integration; vendor assignments are not ground truth."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import astrolabe as ab
import mudata as md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = ab.io.read_xenium(args.fixture)["xenium"]
    observed = original.uns["transcripts"]
    counts = original.X.copy()
    started = perf_counter()
    results = ab.seg.run_proseg(original, executable=args.executable, output_dir=args.output,
                                include_expected_counts=True)
    assert (counts != original.X).nnz == 0
    assert observed.equals(original.uns["transcripts"])
    report = {"elapsed_seconds": perf_counter()-started, "original_cells": original.n_obs,
              "original_transcripts": len(observed), "original_matrix_total": int(counts.sum()),
              "meaning": "integration and accounting; no independent assignment truth", "units": {}}
    for unit, result in results.items():
        qc = ab.seg.reconstruction_qc(result)
        assert int(qc["genes"].n_input.sum()) == len(observed)
        assert result.uns["reconstruction"]["provenance"]["count_validation"] == "matched_backend_export"
        ab.io.save(md.MuData({"reconstruction": result}), args.output/f"{unit}.h5mu")
        restored = ab.io.load(args.output/f"{unit}.h5mu")["reconstruction"]
        assert (result.X != restored.X).nnz == 0
        assert (result.layers["expected_counts"] != restored.layers["expected_counts"]).nnz == 0
        report["units"][unit] = {"reconstructed_cells": result.n_obs, "hard_count_total": int(result.X.sum()),
            "expected_count_total": float(result.layers["expected_counts"].sum()),
            "accounting": result.uns["reconstruction"]["accounting"], "roundtrip": "passed"}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
