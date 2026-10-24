"""Compare uncached public fits with saved PC-donor held-out inputs/results."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import astrolabe as ab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    source = json.loads((args.source / "results.json").read_text())
    selected = [(case, row) for case, row in zip(source["manifest"]["cases"], source["runs"])
                if case["prior"] == "donor_pc" and case["seed"] in (8201, 8202, 8203)]
    if len(selected) != 3:
        raise ValueError("Require the three prespecified PC donor reference cases")
    records = []
    fields = ("posterior_mean", "posterior_sd", "credible_low", "credible_high")
    for case, reference in selected:
        folder = args.source / case["directory"]
        config = json.loads((folder / "config.json").read_text())
        graph = sparse.load_npz(folder / "graph.npz")
        with np.load(folder / "inputs.npz", allow_pickle=False) as inputs:
            obs = pd.DataFrame(inputs["design"], columns=config["fixed"],
                               index=[f"c{i}" for i in range(graph.shape[0])])
            obs["exposure"] = np.exp(inputs["offset"])
            obs["donor"] = inputs["donor"]
            obs["core"] = connected_components(graph, directed=False)[1]
            data = ad.AnnData(sparse.csr_matrix(inputs["counts"]), obs=obs,
                             var=pd.DataFrame(index=config["genes"]))
        data.layers["counts"] = data.X.copy()
        data.obsp["graph"] = graph
        data.uns["spatial_neighbors"] = {"connectivities_key": "graph"}
        result = ab.tl.conditional_expression(data, "gene", predictor="target",
            covariates=config["fixed"][1:], exposure_key="exposure", donor_key="donor",
            core_key="core", python_executable=args.python).iloc[0]
        difference = {key: float(result[key] - reference[key]) for key in fields}
        np.testing.assert_allclose([result[key] for key in fields],
                                   [reference[key] for key in fields], atol=1e-4, rtol=1e-4)
        records.append({"scenario": case["scenario"], "seed": case["seed"],
                        "difference": difference, "agreement": "passed",
                        "backend": data.uns["conditional_expression"]["backend"]})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(case["seed"], difference, flush=True)


if __name__ == "__main__":
    main()
