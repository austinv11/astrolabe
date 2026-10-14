"""Independent-dataset stress calibration for heterogeneous spatial analyses.

Run ``python benchmarks/statistical_stress.py --seeds 200 --perms 999``.
Reports both family discoveries and a preselected raw-p hypothesis per dataset,
so coarse BH resolution cannot alone make the null calibration appear sound.
Nulls assume exchangeability within libraries or across biological samples.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrolabe as ab
from benchmarks.statistical_calibration import interval


def library_data(seed, *, positive=False, signal=False):
    rng = np.random.default_rng(71000+seed)
    xy = rng.uniform(0, 100, (100, 2))
    values = rng.gamma(2, 1, (100, 4)) if positive else rng.normal(size=(100, 4))
    if signal:
        values[:, 0] = np.where(xy[:, 0] < 50, 12., -12.)+rng.normal(0, .1, 100)
    values[25:] = values[25:]*100 if positive else values[25:]*4+100
    a = ad.AnnData(values); a.var_names = [f"g{i}" for i in range(4)]
    a.obs["library_id"] = ["small"]*25+["large"]*75
    a.obsm["spatial"] = xy
    ab.tl.spatial_neighbors(a, n_neighbors=4)
    return a


def sample_data(seed, *, missing=False, signal=False):
    rng = np.random.default_rng(82000+seed)
    sizes = rng.integers(20, 100, 10)
    conditions = np.array(["control"]*6+["treated"]*4)
    xy, types = [], []
    for i, n in enumerate(sizes):
        labels = rng.choice(["A", "B"], n, p=[.5, .5])
        if signal:
            labels = np.resize(["A", "B"], n) if conditions[i] == "treated" else np.array(["A"]*(n//2)+["B"]*(n-n//2))
        if missing and i == 0:
            labels[:3] = "C"
        types.extend(labels)
        xy.append(np.column_stack([np.arange(n), rng.uniform(0, .01, n)]))
    a = ad.AnnData(np.ones((sum(sizes), 2)))
    a.obs["sample"] = np.repeat([f"s{i}" for i in range(10)], sizes)
    a.obs["library_id"] = a.obs["sample"]
    a.obs["condition"] = np.repeat(conditions, sizes)
    a.obs["type"] = pd.Categorical(types, categories=["A", "B", "C", "unobserved"])
    a.obsm["spatial"] = np.vstack(xy)
    ab.tl.spatial_neighbors(a, n_neighbors=2)
    return a


def evaluate(scenario, seed, perms):
    if scenario.startswith("library_"):
        a = library_data(seed, positive="getis" in scenario, signal="signal" in scenario)
        if "svg" in scenario:
            result = ab.tl.spatially_variable_genes(a, flavor="geary" if "geary" in scenario else "moran",
                                                   n_perms=perms, seed=seed)
            raw, adjusted = result.pval, result.pval_adj
        elif "bivariate" in scenario:
            result = ab.tl.moran_local_bv(a, "g0", "g1", n_perms=perms, seed=seed)
            raw, adjusted = result.p_sim, result.p_adj
        else:
            method = ab.tl.getis_ord if "getis" in scenario else ab.tl.moran_local
            result = method(a, "g0", n_perms=perms, seed=seed)
            raw, adjusted = result.p_sim, result.p_adj
        p = float(raw.iloc[0])
        return p, int((adjusted < .05).sum()), int(adjusted.notna().sum()), int(adjusted.isna().sum())
    if scenario.startswith("cohort_"):
        a = sample_data(seed, missing="missing" in scenario, signal="signal" in scenario)
        result = ab.tl.diff_nhood_enrichment(a, "type", sample_key="sample", condition_key="condition",
            contrast=("treated", "control"), n_perms=perms, n_boot=99, seed=seed)
        selected = result[(result.sender == "A") & (result.receiver == "B")].iloc[0]
        return float(selected.pvalue), int((result.padj < .05).sum()), int(result.padj.notna().sum()), int(result.padj.isna().sum())
    rng = np.random.default_rng(93000+seed)
    screened = scenario == "sparse_screened_lr"
    n, genes = (80, 20) if screened else (120, 2)
    a = ad.AnnData(rng.poisson(.05, (n, genes)).astype(float)); a.var_names = [f"g{i}" for i in range(genes)]
    a.obs["type"] = rng.permutation(["A"]*(n//2)+["B"]*(n//2))
    a.obsm["spatial"] = rng.uniform(0, 100, (n, 2))
    ab.tl.spatial_neighbors(a, n_neighbors=4)
    pairs = pd.DataFrame({"ligand": [f"g{i}" for i in range(genes if screened else 1)],
                          "receptor": [f"g{(i+1)%genes}" for i in range(genes if screened else 1)]})
    pairs["ligand_members"] = pairs.ligand.map(lambda gene: [gene])
    resource = ab.tl.import_ligrec_resource(pd.concat([pairs, pd.DataFrame({"ligand": ["complex"],
        "receptor": ["g1"], "ligand_members": [["g0", "unmeasured"]]})], ignore_index=True), species="human")
    ab.tl.ligrec(a, "type", resource, n_perms=perms, seed=seed)
    result = a.uns["ligrec"]
    assert (result["status"].loc[("complex", "g1")] == "missing_subunit").all()
    # The preselected LR hypothesis can be gated out; it remains untested.
    p = float(result["pvalues"].loc[("g0", "g1"), ("A", "B")])
    adj = result["pvalues_adj"].to_numpy()
    return p, int((adj < .05).sum()), int(np.isfinite(adj).sum()), int(np.isnan(adj).sum())


def run(seeds, perms, scenarios=None):
    if seeds < 1 or perms < 1:
        raise ValueError("seeds and permutations must be positive")
    available = ["library_moran", "library_bivariate", "library_getis", "library_svg_moran",
                 "library_svg_geary", "cohort_unequal", "cohort_missing_types", "sparse_panel_lr",
                 "sparse_screened_lr", "library_signal", "cohort_signal"]
    scenarios = available if scenarios is None else scenarios
    if not scenarios or set(scenarios)-set(available):
        raise ValueError("unknown or empty stress scenario selection")
    outcomes, records = {}, {}; started = time.perf_counter()
    for scenario in scenarios:
        rows = []
        for seed in range(seeds):
            p, discoveries, tested, untested = evaluate(scenario, seed, perms)
            rows.append(dict(seed=seed, raw_p=float(p) if np.isfinite(p) else None,
                             discoveries=discoveries, tested=tested, untested=untested))
        power = "signal" in scenario
        family = interval([r["discoveries"] > 0 for r in rows], power=power)
        if power:
            # Strong planted controls must retain useful detection, rather
            # than passing every null merely because no test can reject.
            family["passed"] = family["rate"] >= .5
        valid = [r["raw_p"] < .05 for r in rows if r["raw_p"] is not None]
        raw = interval(valid, power=power) if valid else {"datasets": 0, "passed": True, "status": "untested"}
        raw["informative"] = len(valid) >= 20
        # Screening selects tests using the observed labels. Its conditional
        # raw-p rejection rate is descriptive, not an exchangeable null guard.
        raw_guard = raw["passed"] if not scenario.startswith("sparse_") else True
        outcomes[scenario] = dict(family=family, raw_preselected=raw, passed=family["passed"] and raw_guard,
                                 kind="strong planted signal" if power else "null")
        records[scenario] = rows
        print(scenario, outcomes[scenario], flush=True)
    return dict(seeds=seeds, permutations=perms, elapsed_seconds=time.perf_counter()-started,
                outcomes=outcomes, records=records,
                interpretation="generator-specific calibration; gated raw LR results are conditional on being tested")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--perms", type=int, default=999)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("analysis/implementation_validation/statistical_stress.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR); warnings.filterwarnings("ignore")
    with threadpool_limits(limits=1):
        result = run(args.seeds, args.perms, args.scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if not all(value["passed"] for value in result["outcomes"].values()):
        raise SystemExit("Stress calibration detected excess null rejection or failed planted controls")
