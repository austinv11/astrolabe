"""Summarize every prespecified extended INLA run, including failures."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def summary(rows):
    fitted = [row for row in rows if row["status"] == "fitted"]
    result = {"attempted": len(rows), "fitted": len(fitted), "failed": len(rows) - len(fitted)}
    if fitted:
        for key in ("covered", "excludes_zero"):
            count = sum(row[key] for row in fitted)
            ci = binomtest(count, len(fitted)).proportion_ci(method="exact")
            result[key] = {"count": count, "rate": count / len(fitted), "exact_95_interval": [ci.low, ci.high]}
        result["mean_bias"] = float(np.mean([row["posterior_mean"] - row["truth"] for row in fitted]))
        result["mean_posterior_sd"] = float(np.mean([row["posterior_sd"] for row in fitted]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["runs"]
    if len(rows) != len(data["manifest"]["cases"]):
        raise ValueError("Validation has not completed all prespecified fits")
    groups = sorted({(row["scenario"], row["prior"]) for row in rows})
    data["summary"] = {}
    for scenario, prior in groups:
        subset = [row for row in rows if (row["scenario"], row["prior"]) == (scenario, prior)]
        result = summary(subset)
        result["by_donor_count"] = {str(n): summary([row for row in subset if row["n_donors"] == n])
                                    for n in sorted({row["n_donors"] for row in subset})}
        if prior != "default":
            baseline = {row["seed"]: row for row in rows if row["scenario"] == scenario and row["prior"] == "default" and row["status"] == "fitted"}
            paired = [row for row in subset if row["status"] == "fitted" and row["seed"] in baseline]
            result["paired_to_default"] = {
                "n": len(paired),
                "max_absolute_mean_change": max(abs(row["posterior_mean"] - baseline[row["seed"]]["posterior_mean"]) for row in paired),
                "max_absolute_sd_change": max(abs(row["posterior_sd"] - baseline[row["seed"]]["posterior_sd"]) for row in paired),
                "zero_exclusion_changes": sum(row["excludes_zero"] != baseline[row["seed"]]["excludes_zero"] for row in paired)}
        data["summary"][scenario + "/" + prior] = result
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
