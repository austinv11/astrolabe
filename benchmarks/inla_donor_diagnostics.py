"""Prespecified prior/approximation ablations using existing INLA model terms.

Oracle arms use known simulation parameters solely to diagnose the integration;
they are not deployable inference procedures. No production settings are changed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from inla_calibration import simulate


SETTINGS = {
    "default": {},
    "donor_pc": {"donor_pc_u": 1.},
    "donor_oracle": {"donor_fixed_precision": 16.},
    "all_oracle": {"donor_fixed_precision": 16., "all_oracle": True},
    "full_laplace": {"strategy": "laplace"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=7201)
    parser.add_argument("--scenarios", nargs="+", default=["unequal_core_sizes", "donor_condition_null", "donor_condition"])
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--holdout", action="store_true", help="Vary true donor/spatial SD, phi and NB size independently of fitting settings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for scenario in args.scenarios:
        for seed in range(args.seed_start, args.seed_start + args.replicates):
            donors = [4, 6, 8, 12][(seed-args.seed_start) % 4]
            simulation = {}
            if args.holdout:
                if set(args.settings) != {"default", "donor_pc"}:
                    raise ValueError("Held-out comparison retains exactly baseline and donor PC; no oracle arms")
                rng = np.random.default_rng(seed+100000)
                simulation = {"donor_sd": float(rng.choice([0., .1, .25, .5])),
                              "spatial_sd": float(rng.choice([.15, .3, .6])),
                              "phi": float(rng.choice([.2, .5, .8])),
                              "nb_size": int(rng.choice([5, 10, 30]))}
            data, truth, covariates = simulate(seed, scenario, n_donors=donors,
                core_width_shift=(seed-args.seed_start) % 3, donor_imbalance=True, **simulation)
            for name in args.settings:
                folder = args.output/f"case-{len(cases):05d}"
                folder.mkdir()
                sparse.save_npz(folder/"graph.npz", data.obsp["graph"])
                np.savez(folder/"inputs.npz", counts=data.X.toarray(), design=data.obs[["x", *covariates]].to_numpy(float),
                    offset=np.log(data.obs.exposure.to_numpy()), donor=data.obs.donor.to_numpy()+1)
                config = {"genes": ["gene"], "fixed": ["target", *[f"covariate_{i}" for i in range(len(covariates))]],
                          "donor": True, "num_threads": 1, "keep_artifacts": False}
                (folder/"config.json").write_text(json.dumps(config))
                cases.append({"directory": folder.name, "scenario": scenario, "seed": seed, "truth": truth,
                    "prior": name, "prior_settings": SETTINGS[name], "n_cells": data.n_obs, "n_donors": donors,
                    "simulation_parameters": simulation or {"donor_sd": .25, "spatial_sd": .3, "phi": .5, "nb_size": 10}})
    manifest = {"mode": "donor-prior-diagnostics", "replicates_per_scenario": args.replicates,
        "nominal_credible_level": .95, "cases": cases, "seed_start": args.seed_start, "holdout": args.holdout,
        "selection": "All prespecified arms retained; oracle parameters are diagnostic only",
        "pc_prior_rationale": "Same pc.prec(1, .01) as the existing spatial SD, on the same log-expression scale; chosen before new fits",
        "reference": "https://www.inla.r-inla-download.org/r-inla.org/doc/prior/pc.prec.pdf"}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(len(cases), "prepared fits")


if __name__ == "__main__":
    main()
