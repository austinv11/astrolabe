"""Larger conditional null check using the public adapter's shared-design batch.

Each gene is an independent simulated response/field/donor-effect realization;
all share a prespecified unequal-core graph, predictor and exposure. This checks
conditional coverage for ONE fixed design, not all varying tissue designs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest
import astrolabe as ab
from inla_calibration import simulate


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--python", required=True)
parser.add_argument("--replicates", type=int, default=200)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
template, _, _ = simulate(820, "unequal_core_sizes")
exposure = template.obs.exposure.to_numpy()
seeds = range(4201, 4201+args.replicates)
columns = [simulate(seed, "unequal_core_sizes", exposure_override=exposure)[0].X for seed in seeds]
data = ad.AnnData(sparse.hstack(columns, format="csr"), obs=template.obs.copy(),
                 var=pd.DataFrame(index=[f"seed_{seed}" for seed in seeds]))
data.layers["counts"] = data.X.copy()
data.obsp["graph"] = template.obsp["graph"].copy()
data.uns["spatial_neighbors"] = {"connectivities_key": "graph"}
start = time.perf_counter()
results = ab.tl.conditional_expression(data, list(data.var_names), predictor="x",
    exposure_key="exposure", donor_key="donor", python_executable=args.python,
    timeout=3600)
elapsed = time.perf_counter()-start
excluded = int(((results.credible_low > 0) | (results.credible_high < 0)).sum())
interval = binomtest(excluded, len(results)).proportion_ci(method="exact")
report = {"scenario": "unequal cores, fixed exposure and design; null effect",
          "exposure_seed": 820, "response_seeds": [4201, 4200+args.replicates],
          "n_cells": data.n_obs, "n_cores": 16, "n_donors": 8, "nominal_credible_level": .95,
          "n_fitted": int((results.status == "fitted").sum()), "n_replicates": len(results),
          "false_exclusions": excluded, "false_exclusion_rate": excluded/len(results),
          "exact_95_interval": [interval.low, interval.high], "elapsed_seconds": elapsed,
          "mean_bias": float(results.posterior_mean.mean()),
          "interpretation": "Conditional fixed-design check; does not replace varying-design coverage evidence",
          "backend": data.uns["conditional_expression"]["backend"],
          "parameters": data.uns["conditional_expression"]["parameters"],
          "results": results.to_dict("records")}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({k: v for k, v in report.items() if k not in {"backend", "parameters", "results"}}), flush=True)
