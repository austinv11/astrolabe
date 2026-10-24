"""Compare real MERSCOPE/CosMx reader results directly with local vendor files.

Public cropped files: SFEData/vizgen.tar.gz and SFEData/cosmx.tar.gz under
https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/ .
The source archives are not bundled; download and unpack them explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import astrolabe as ab
import awkward as ak
import numpy as np
import pandas as pd
import shapely


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merscope", type=Path, required=True)
    parser.add_argument("--cosmx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"source": "https://github.com/pachterlab/SFEData", "vendors": {}}
    for name, root in (("merscope", args.merscope), ("cosmx", args.cosmx)):
        if name == "merscope":
            data = ab.io.read_merscope(root, z_level=3)
            count_path, meta_path = root / "cell_by_gene.csv", root / "cell_metadata.csv"
            counts = pd.read_csv(count_path, index_col=0)
            metadata = pd.read_csv(meta_path, index_col=0)
            counts.index, metadata.index = counts.index.astype(str), metadata.index.astype(str)
            xy = metadata.loc[counts.index, ["center_x", "center_y"]].to_numpy()
            boundary_path = root / "cell_boundaries.parquet"
            boundaries = pd.read_parquet(boundary_path)
            boundaries = boundaries[boundaries.ZIndex == 3].set_index("EntityID")
            boundaries.index = boundaries.index.astype(str)
            polygons = [np.asarray(shapely.from_wkb(boundaries.loc[cell, "Geometry"]).exterior.coords) for cell in counts.index]
        else:
            data = ab.io.read_cosmx(root)
            count_path, meta_path = next(root.glob("*_exprMat_file.csv")), next(root.glob("*_metadata_file.csv"))
            counts = pd.read_csv(count_path)
            counts = counts[counts.cell_ID != 0]
            counts.index = counts.fov.astype(str) + "_" + counts.cell_ID.astype(str)
            counts = counts.drop(columns=["fov", "cell_ID"])
            metadata = pd.read_csv(meta_path)
            metadata.index = metadata.fov.astype(str) + "_" + metadata.cell_ID.astype(str)
            xy = metadata.loc[counts.index, ["CenterX_global_px", "CenterY_global_px"]].to_numpy() * .12028
            boundary_path = next(root.glob("*polygons.csv"))
            boundaries = pd.read_csv(boundary_path)
            boundaries["id"] = boundaries.fov.astype(str) + "_" + boundaries.cellID.astype(str)
            polygons = [boundaries.loc[boundaries.id == cell, ["x_global_px", "y_global_px"]].to_numpy() * .12028 for cell in counts.index]
        a = data[name]
        assert a.obs_names.tolist() == counts.index.tolist()
        assert a.var_names.tolist() == counts.columns.tolist()
        np.testing.assert_array_equal(a.X.toarray(), counts.to_numpy())
        np.testing.assert_allclose(a.obsm["spatial"], xy, rtol=1e-6)
        for actual, expected in zip(a.obsm["cell_boundaries"], polygons):
            np.testing.assert_allclose(ak.to_numpy(actual), expected, rtol=1e-6)
        ab.pp.calculate_qc_metrics(data)
        ab.tl.spatial_neighbors(data, radius=30)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "roundtrip.h5mu"
            ab.io.save(data, path)
            restored = ab.io.load(path)[name]
            assert (a.X != restored.X).nnz == 0
            np.testing.assert_array_equal(a.obsm["spatial"], restored.obsm["spatial"])
            assert ak.to_list(a.obsm["cell_boundaries"]) == ak.to_list(restored.obsm["cell_boundaries"])
        report["vendors"][name] = {"cells": a.n_obs, "genes": a.n_vars, "counts": int(a.X.sum()),
            "boundaries": len(polygons), "raw_alignment_qc_graph_roundtrip": "passed",
            "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (count_path, meta_path, boundary_path)}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
