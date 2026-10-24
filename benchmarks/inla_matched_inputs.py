"""Run three saved Windows calibration inputs unchanged in another environment."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("worker", Path(__file__).resolve().parents[1] / "astrolabe/tl/_inla_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    reference = json.loads((args.source / "results.json").read_text())
    records = []
    for index in (0, 100, 200):
        case = reference["manifest"]["cases"][index]
        destination = args.output / case["directory"]
        destination.mkdir()
        for filename in ("graph.npz", "inputs.npz", "config.json"):
            shutil.copyfile(args.source / case["directory"] / filename, destination / filename)
        worker.run(destination)
        result = json.loads((destination / "result.json").read_text())
        fields = ("posterior_mean", "posterior_sd", "credible_low", "credible_high")
        records.append({"scenario": case["scenario"], "seed": case["seed"],
                        "difference_from_windows": {key: result["results"][0][key] - reference["runs"][index][key] for key in fields},
                        "result": result})
        (args.output / "summary.json").write_text(json.dumps(records, indent=2))
        print(case["scenario"], records[-1]["difference_from_windows"], flush=True)


if __name__ == "__main__":
    main()
