"""Predeclare and prepare broader varying-design and prior-sensitivity checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from inla_calibration import simulate


PRIORS = {
    "default": {},
    "spatial_sd_half": {"spatial_sd_u": .5},
    "spatial_sd_double": {"spatial_sd_u": 2.},
    "phi_more_iid": {"phi_alpha": 2/3},
    "phi_more_spatial": {"phi_alpha": 1/3},
    "donor_rate_tenth": {"donor_rate": .000005},
    "donor_rate_tenfold": {"donor_rate": .0005},
}


def prepare(root, mode, replicates):
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    scenarios = ["unequal_core_sizes", "donor_condition_null", "regional_confounding"] if mode == "calibration" else ["donor_condition_null", "donor_condition"]
    priors = {"default": {}} if mode == "calibration" else PRIORS
    for scenario in scenarios:
        for seed in range(6201, 6201+replicates):
            # Four predeclared donor/core templates; observations, covariates,
            # exposure, spatial field and donor realizations vary across seeds.
            donors = [4, 6, 8, 12][(seed-6201) % 4] if mode == "calibration" else 8
            data, truth, covariates = simulate(seed, scenario, n_donors=donors,
                core_width_shift=(seed-6201) % 3, donor_imbalance=mode == "calibration")
            for prior, settings in priors.items():
                directory = f"case-{len(cases):05d}"
                destination = root/directory
                destination.mkdir()
                sparse.save_npz(destination/"graph.npz", data.obsp["graph"])
                design = data.obs[["x", *covariates]].to_numpy(dtype=float)
                np.savez(destination/"inputs.npz", counts=data.X.toarray(), design=design,
                         offset=np.log(data.obs.exposure.to_numpy()), donor=data.obs.donor.to_numpy()+1)
                config = {"genes": ["gene"], "fixed": ["target", *[f"covariate_{i}" for i in range(len(covariates))]],
                          "donor": True, "num_threads": 1, "keep_artifacts": False}
                (destination/"config.json").write_text(json.dumps(config))
                cases.append({"directory": directory, "scenario": scenario, "seed": seed,
                              "truth": truth, "prior": prior, "prior_settings": settings,
                              "n_cells": data.n_obs, "n_donors": donors})
    manifest = {"mode": mode, "replicates_per_scenario": replicates, "cases": cases,
                "nominal_credible_level": .95, "selection": "All predefined settings retained; no default selection or tuning",
                "method": "Shipped INLA worker; exact prior-table identity per fit; identical graph/prior computations cached only within this benchmark"}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Prepared {len(cases)} fits in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["calibration", "sensitivity"], required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.mode, args.replicates)
