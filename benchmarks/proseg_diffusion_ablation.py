"""Separate Proseg's small positional adjustment from its long-range diffusion.

The upstream documentation distinguishes --diffusion-probability 0 (small
adjustments remain) from --no-diffusion (all adjustments disabled). Fresh seeds
903/904 and all clean/jitter/leak cases are retained; no settings are optimized.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrolabe as ab
from proseg_validation import generate, assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"seeds": [903, 904], "scenarios": ["clean", "jitter", "leak"],
              "arms": ["disabled", "near_only", "default"], "runs": [],
              "reference": "https://github.com/dcjones/proseg#general-advice",
              "selection": "Documented model ablation, no parameter search; all outcomes retained"}
    path = args.output/"summary.json"
    path.write_text(json.dumps(report, indent=2))
    for seed in report["seeds"]:
        for scenario in report["scenarios"]:
            data = generate(seed, scenario)
            for arm in report["arms"]:
                started = time.perf_counter()
                row = {"seed": seed, "scenario": scenario, "arm": arm}
                try:
                    result = ab.seg.run_proseg(data, executable=args.executable,
                        output_dir=args.output/f"{seed}_{scenario}_{arm}", diffusion=arm != "disabled",
                        extra_args=("--diffusion-probability", "0") if arm == "near_only" else (),
                        include_expected_counts=True)
                    row.update(status="fitted", cores=assess(data, result))
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                row["seconds"] = time.perf_counter()-started
                report["runs"].append(row)
                path.write_text(json.dumps(report, indent=2)+"\n")
                print(seed, scenario, arm, row["status"], flush=True)


if __name__ == "__main__":
    main()
