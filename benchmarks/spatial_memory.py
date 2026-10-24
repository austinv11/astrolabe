"""Opt-in process-isolated local-null memory benchmark; no data downloads.

Run: python benchmarks/spatial_memory.py --output docs/spatial-memory-results.json
The 39-permutation runs exercise allocation and equivalence, not discovery power.
Peak RSS includes Python, ESDA weights, compilation and outputs; the API's working
budget governs simulation buffers and is not a cap on total process memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def worker(n, disk):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import psutil
    import scipy.sparse as sp
    import astrolabe as ab

    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = threading.Event()
    def sample():
        while not done.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    start = time.perf_counter()
    rng = np.random.default_rng(2026)
    a = ad.AnnData(rng.poisson(3, (n, 1)).astype(float))
    a.var_names = ["g"]
    index = np.arange(n)
    columns = (index[:, None] + np.array([-2, -1, 1, 2])) % n
    graph = sp.csr_matrix((np.ones(n*4), columns.ravel(), np.arange(0, n*4+1, 4)), shape=(n, n))
    a.obsp["spatial_neighbors_connectivities"] = graph
    a.uns["spatial_neighbors"] = {"connectivities_key": "spatial_neighbors_connectivities"}
    result = ab.tl.moran_local(a, "g", n_perms=39, seed=9, disk_backed=disk,
                               max_working_bytes=256*1024**2)
    elapsed = time.perf_counter()-start
    peak[0] = max(peak[0], process.memory_info().rss)
    done.set(); monitor.join()
    import importlib.metadata as metadata
    return {"cells": n, "genes": 1, "neighbors": 4, "permutations": 39,
            "disk_backed": disk, "seconds": elapsed, "peak_rss_bytes": peak[0],
            "working_budget_bytes": 256*1024**2,
            "result_sha256": hashlib.sha256(pd.util.hash_pandas_object(result, index=True).to_numpy().tobytes()).hexdigest(),
            "python": sys.version.split()[0], "esda": metadata.version("esda")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", type=int, default=[100_000, 1_000_000])
    parser.add_argument("--output", type=Path, default=Path("spatial-memory-results.json"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--disk-backed", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.cells[0], args.disk_backed)))
    else:
        results = []
        for n in args.cells:
            for disk in (False, True):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cells", str(n)]
                if disk:
                    command.append("--disk-backed")
                run = subprocess.run(command, check=True, capture_output=True, text=True)
                record = json.loads(run.stdout.strip().splitlines()[-1])
                results.append(record)
                args.output.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
                print(json.dumps(record), flush=True)
            assert results[-1]["result_sha256"] == results[-2]["result_sha256"], "Disk and memory results differ"
