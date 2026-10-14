"""Measure LR scratch allocation on a sparse synthetic panel.

This uses tracemalloc (including NumPy allocations), not process RSS. Input,
import and graph construction are excluded. Scale cells/pairs explicitly when
profiling a production workload; no E-by-interaction tensor is allocated.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import tracemalloc
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import astrolabe as ab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells",type=int,default=30000)
    parser.add_argument("--pairs",type=int,default=64)
    parser.add_argument("--perms",type=int,default=39)
    parser.add_argument("--workers",type=int,default=2)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    data = ad.AnnData(sp.random(args.cells,2000,density=.005,format="csr",random_state=0))
    data.var_names = [f"g{i}" for i in range(2000)]
    data.obs["cell_type"] = rng.choice(["A","B","C","D"],args.cells)
    data.obsm["spatial"] = rng.uniform(0,1000,(args.cells,2))
    ab.tl.spatial_neighbors(data,n_neighbors=4)
    pairs = pd.DataFrame({"ligand":[f"g{i}" for i in range(args.pairs)],
                          "receptor":[f"g{i+args.pairs}" for i in range(args.pairs)]})
    started = time.perf_counter(); tracemalloc.start()
    ab.tl.ligrec(data,"cell_type",pairs,n_perms=args.perms,n_jobs=args.workers,
                 expr_frac_threshold=0,seed=0,max_scratch_bytes=256*1024**2)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    output = dict(cells=args.cells,genes=2000,pairs=args.pairs,permutations=args.perms,
                   workers=args.workers,edges=data.obsp["spatial_neighbors_connectivities"].nnz,
                   peak_traced_bytes=peak,elapsed_seconds=time.perf_counter()-started,
                   previous_edge_tensor_bytes=16*args.pairs*data.obsp["spatial_neighbors_connectivities"].nnz)
    dest = Path("analysis/implementation_validation/ligrec_memory.json")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
    if peak > 256*1024**2*max(1,args.workers):
        raise SystemExit("measured LR allocation exceeded the worker scratch target")


if __name__ == "__main__":
    main()
