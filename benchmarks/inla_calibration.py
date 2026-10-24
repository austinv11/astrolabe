"""Predeclared optional BYM2 checks; Bayesian coverage, not p-value calibration.

Run from the repository root with --python pointing at a prepared pyINLA 0.1.11
environment. No packages or engines are installed. All scenarios/seeds are
retained, including failures and misspecified-model stress cases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.stats import binomtest
import astrolabe as ab


SCENARIOS = {
    "smooth_null": (0., False, False, False),
    "planted_distance": (.5, False, False, False),
    "unequal_core_sizes": (0., True, False, False),
    "donor_condition": (.5, True, True, False),
    "donor_condition_null": (0., True, True, False),
    "regional_confounding": (0., False, False, True),
    "transcript_misassignment": (.5, False, False, False),
}


def simulate(seed, scenario, exposure_override=None, *, n_donors=8, core_width_shift=0, donor_imbalance=False,
             donor_sd=.25, spatial_sd=.3, phi=.5, nb_size=10):
    rng = np.random.default_rng(seed)
    beta, unequal, condition, confounding = SCENARIOS[scenario]
    blocks, coordinates, donors, cores = [], [], [], []
    for core in range(n_donors*2):
        width = 3 if not unequal else (2+(core+core_width_shift) % 3)
        if donor_imbalance and core//2 < n_donors//2:
            width += 1
        xy = np.array([(i, j) for i in range(3) for j in range(width)])
        adjacency = (np.abs(xy[:, None]-xy[None, :]).sum(axis=2) == 1).astype(float)
        blocks.append(adjacency)
        coordinates.append(xy)
        donors.extend([core//2]*len(xy))
        cores.extend([core]*len(xy))
    donor, core = np.array(donors), np.array(cores)
    xy = np.concatenate(coordinates)
    n = len(donor)
    x = (donor >= n_donors//2).astype(float) if condition else xy[:, 1].astype(float)
    if not condition:
        x = (x-x.mean())/x.std()
    spatial = []
    for adjacency in blocks:
        covariance = linalg.pinvh(np.diag(adjacency.sum(axis=1))-adjacency)
        covariance /= np.exp(np.log(np.diag(covariance)).mean())
        spatial.extend(rng.multivariate_normal(np.zeros(len(adjacency)), covariance))
    spatial = np.asarray(spatial)
    nuisance = .7*x + rng.normal(0, .7, n) if confounding else rng.normal(size=n)
    eta = 2 + beta*x + spatial_sd*(np.sqrt(phi)*spatial+np.sqrt(1-phi)*rng.normal(size=n))
    eta += rng.normal(0, donor_sd, n_donors)[donor]
    if confounding:
        eta += .5*nuisance
    exposure = rng.uniform(.5, 2., n)
    if exposure_override is not None:
        exposure = np.asarray(exposure_override)
    mu = exposure*np.exp(eta)
    counts = rng.negative_binomial(nb_size, nb_size/(nb_size+mu))
    graph = sparse.block_diag(blocks, format="csr")
    if scenario == "transcript_misassignment":
        # Twenty percent of each cell's molecules are reassigned to adjacent
        # cells, preserving the total. This violates the fitted observation model.
        contaminated = counts.copy()
        for i, count in enumerate(counts):
            moved = rng.binomial(count, .2)
            neighbors = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
            contaminated[i] -= moved
            np.add.at(contaminated, rng.choice(neighbors, moved), 1)
        counts = contaminated
    data = ad.AnnData(sparse.csr_matrix(counts[:, None]),
        obs=pd.DataFrame({"x": x, "exposure": exposure, "donor": donor, "core_id": core,
                          "nuisance": nuisance}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["gene"]))
    data.layers["counts"] = data.X.copy()
    data.obsp["graph"] = graph
    data.uns["spatial_neighbors"] = {"connectivities_key": "graph"}
    return data, beta, ["nuisance"] if confounding else []


def summarize(rows):
    summary = {}
    for scenario in SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        fitted = [row for row in selected if row["status"] == "fitted"]
        item = {"attempted": len(selected), "fitted": len(fitted), "failures": len(selected)-len(fitted)}
        if fitted:
            for name in ("covered", "excludes_zero"):
                count = sum(row[name] for row in fitted)
                interval = binomtest(count, len(fitted)).proportion_ci(method="exact")
                item[name] = {"rate": count/len(fitted), "exact_95_interval": [interval.low, interval.high]}
            item["mean_bias"] = float(np.mean([row["posterior_mean"]-row["truth"] for row in fitted]))
            item["mean_seconds"] = float(np.mean([row["seconds"] for row in fitted]))
        summary[scenario] = item
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--measure-memory", action="store_true", help="Sample summed parent/child RSS using optional psutil")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, backends = [], {}
    for scenario in args.scenarios:
        for seed in range(2201, 2201+args.replicates):
            data, truth, covariates = simulate(seed, scenario)
            start = time.perf_counter()
            row = {"scenario": scenario, "seed": seed, "truth": truth}
            stopped = threading.Event()
            peak = {"parent_rss_bytes": 0, "backend_summed_rss_bytes": 0}
            if args.measure_memory:
                import psutil
                parent = psutil.Process()
                def sample():
                    while not stopped.wait(.02):
                        try:
                            peak["parent_rss_bytes"] = max(peak["parent_rss_bytes"], parent.memory_info().rss)
                            total = 0
                            for child in parent.children(recursive=True):
                                try:
                                    total += child.memory_info().rss
                                except psutil.Error:
                                    pass
                            peak["backend_summed_rss_bytes"] = max(peak["backend_summed_rss_bytes"], total)
                        except psutil.Error:
                            pass
                monitor = threading.Thread(target=sample, daemon=True)
                monitor.start()
            try:
                result = ab.tl.conditional_expression(data, ["gene"], predictor="x", exposure_key="exposure",
                    donor_key="donor", covariates=covariates, python_executable=args.python).iloc[0]
                row.update(result.to_dict())
                row["covered"] = bool(result.credible_low <= truth <= result.credible_high)
                row["excludes_zero"] = bool(result.credible_low > 0 or result.credible_high < 0)
                backend = data.uns["conditional_expression"]["backend"]
                backends[backend["engine_sha256"]] = backend
            except Exception as exc:
                row.update(status="failed", error=str(exc))
            finally:
                stopped.set()
                if args.measure_memory:
                    monitor.join()
                    row.update(peak)
            row["seconds"] = time.perf_counter()-start
            rows.append(row)
            report = {"seeds": [2201, 2200+args.replicates], "nominal_credible_level": .95,
                      "interpretation": "Limited calibration screen; 20 replicates cannot establish 95% coverage. Misassignment is an explicit model-violation stress test.",
                      "summary": summarize(rows), "runs": rows, "backends": backends}
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(scenario, seed, row["status"], round(row["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
