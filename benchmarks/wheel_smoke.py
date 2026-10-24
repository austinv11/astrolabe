"""Run with a clean wheel environment from outside the repository root.

python /path/to/benchmarks/wheel_smoke.py --fixture /path/to/tests/data/xenium_4_tiny
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile


class NoOptionalImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"spatialdata", "pyinla"}:
            raise AssertionError(f"An ordinary operation attempted to import {fullname}")


sys.meta_path.insert(0, NoOptionalImports())
import astrolabe as ab
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--fixture", type=Path, required=True)
args = parser.parse_args()
package = Path(ab.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), f"Not an isolated wheel import: {package}"
data = ab.io.read_xenium(args.fixture)
a = data["xenium"]
ab.pp.calculate_qc_metrics(data)
ab.tl.spatial_neighbors(data, radius=15)
gene = a.var_names[np.asarray(a.X.sum(axis=0)).ravel().argmax()]
ab.tl.moran(data, gene, n_perms=99, seed=9)
with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
    path = Path(temporary) / "roundtrip.h5mu"
    ab.io.save(data, path)
    restored = ab.io.load(path)
    assert (restored["xenium"].X != a.X).nnz == 0
    stale = ab.tl.validate_results(restored)
    assert stale.empty, stale.to_dict("records")
assert "spatialdata" not in sys.modules
assert "pyinla" not in sys.modules
assert callable(ab.tl.conditional_expression)
print(json.dumps({"package": str(package), "shape": list(a.shape), "total_counts": float(a.X.sum()),
                  "spatialdata_imported": False, "pyinla_imported": False,
                  "reader_graph_inference_roundtrip": "passed"}))
