"""Maintained statistical calibration with independent datasets and power controls.

Run ``python benchmarks/statistical_calibration.py --seeds 200 --perms 199``.
Reports Wilson intervals and flags clear excess rejection with a one-sided
exact binomial test (p < .01). All-null local FDR is P(any BH discovery).
Null generators define the exchangeability assumptions being validated;
results do not certify arbitrary tissues or causal signaling claims.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time
import warnings
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrolabe as ab
from astrolabe.tl._gradient import _diffuse_expression


def dataset(values, coords):
    result = ad.AnnData(np.asarray(values,dtype=float))
    result.var_names = [f"g{i}" for i in range(result.n_vars)]
    result.obsm["spatial"] = coords
    ab.tl.spatial_neighbors(result,n_neighbors=6)
    return result


def interval(records, power=False):
    n = len(records); k = sum(records)
    ci = binomtest(k,n).proportion_ci(confidence_level=.95,method="wilson")
    excess = None if power else binomtest(k,n,.05,alternative="greater").pvalue
    return dict(rejections=int(k),datasets=n,rate=float(k/n),ci95=[float(ci.low),float(ci.high)],
                excess_p=None if excess is None else float(excess),passed=True if power else bool(excess >= .01))


def run(seeds, perms):
    outcomes = {}; records = {}; started = time.perf_counter()
    coords = np.asarray([(x,y) for x in range(15) for y in range(15)],float)
    scenarios = ["sparse_moran", "continuous_moran", "sparse_bivariate", "sparse_getis", "local_signal"]
    for scenario in scenarios:
        discovered = []; detail = []
        for seed in range(seeds):
            rng = np.random.default_rng(24000+seed)
            values = rng.normal(size=(225,2)) if scenario == "continuous_moran" else rng.poisson(.15,(225,2))
            if scenario == "local_signal":
                values[:,0] = rng.poisson(np.where(coords[:,0] < 7,15.,.15))
            a = dataset(values,coords)
            if scenario == "sparse_bivariate":
                result = ab.tl.moran_local_bv(a,"g0","g1",n_perms=perms,seed=seed)
            elif scenario == "sparse_getis":
                result = ab.tl.getis_ord(a,"g0",n_perms=perms,seed=seed)
            else:
                result = ab.tl.moran_local(a,"g0",n_perms=perms,seed=seed)
            n = int((result.p_adj < .05).sum()); discovered.append(n > 0)
            detail.append(dict(seed=seed,n_discoveries=n))
        outcomes[scenario] = interval(discovered,power=scenario == "local_signal")
        records[scenario] = detail
        print(scenario,outcomes[scenario],flush=True)
    for scenario in ["icg_null", "icg_type_marker", "icg_signal", "icg_block_confounding"]:
        rejected = []; detail = []
        for seed in range(seeds):
            rng = np.random.default_rng(32000+seed)
            n = 120
            types = rng.permutation(np.array(["A"]*60+["B"]*60))
            rows = np.arange(n); cols = (rows+1)%n
            graph = sp.csr_matrix((np.ones(2*n),(np.r_[rows,cols],np.r_[cols,rows])),shape=(n,n))
            contacted = np.asarray(graph[:,types == "B"].sum(axis=1)).ravel() > 0
            mean = np.where(types == "B",30.,2.) if scenario == "icg_type_marker" else np.full(n,2.)
            if scenario in ("icg_signal", "icg_block_confounding"):
                mean[(types == "A") & contacted] = 8.
            a = ad.AnnData(rng.poisson(mean)[:,None].astype(float)); a.var_names = ["g0"]
            a.obs["type"] = types; a.obsp["spatial_neighbors_connectivities"] = graph
            a.obs["block"] = contacted.astype(str)
            result = ab.tl.interaction_changed_genes(a,groupby="type",expression_scale="linear",
                pairs=[("A","B")],min_cells=5,n_perms=perms,seed=seed,
                block_key="block" if scenario == "icg_block_confounding" else None)
            p = float(result.pval.iloc[0]); rejected.append(p < .05)
            detail.append(dict(seed=seed,pvalue=p))
        outcomes[scenario] = interval(rejected,power=scenario == "icg_signal")
        records[scenario] = detail
        print(scenario,outcomes[scenario],flush=True)
    for scenario in ["gradient_null", "gradient_signal"]:
        rejected = []; detail = []
        for seed in range(seeds):
            rng = np.random.default_rng(41000+seed)
            xy = rng.uniform(0,500,(150,2))
            a = dataset(rng.normal(size=(150,1)),xy)
            if scenario == "gradient_signal":
                a.X[:,0] = xy[:,0]/500+rng.normal(0,.02,150)
            else:
                a.X[:,0] = _diffuse_expression(a.X[:,0],a.obsp["spatial_neighbors_connectivities"],n_steps=8)
            ab.tl.expression_gradient(a,"g0",n_diffusion_steps=0)
            ab.tl.gradient_significance(a,"g0",n_perms=perms,random_state=seed)
            result = a.uns["gradient_significance"].iloc[0]
            p = float(result.p_coherence); rejected.append(p < .05)
            detail.append(dict(seed=seed,p_coherence=p,p_magnitude=float(result.p_magnitude)))
        outcomes[scenario] = interval(rejected,power=scenario == "gradient_signal")
        if scenario == "gradient_null":
            outcomes["gradient_magnitude_null"] = interval([row["p_magnitude"] < .05 for row in detail])
        records[scenario] = detail
        print(scenario,outcomes[scenario],flush=True)
    return {"seeds":seeds,"permutations":perms,"elapsed_seconds":time.perf_counter()-started,
            "versions":{name:importlib.metadata.version(name) for name in ["numpy","scipy","esda","anndata"]},
            "outcomes":outcomes,"records":records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",type=int,default=200)
    parser.add_argument("--perms",type=int,default=199)
    parser.add_argument("--output",type=Path,default=Path("analysis/implementation_validation/calibration.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR); warnings.filterwarnings("ignore")
    with threadpool_limits(limits=1):
        result = run(args.seeds,args.perms)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    if not all(value["passed"] for value in result["outcomes"].values()):
        raise SystemExit("Calibration detected excess null rejection")
