"""Licensed vendor files: independent schema, count and round-trip invariants."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

import astrolabe as ab


FIXTURE = Path(__file__).parent / "data" / "xenium_4_tiny"


def test_xenium4_vendor_counts_coordinates_and_roundtrip(tmp_path):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest
    with h5py.File(FIXTURE / "cell_feature_matrix.h5") as f:
        matrix = f["matrix"]
        expected = sp.csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]),
                                 shape=tuple(matrix["shape"][:])).T.tocsr()
        barcodes = matrix["barcodes"].asstr()[:]
    data = ab.io.read_xenium(FIXTURE, library_id="vendor_fixture")
    a = data["xenium"]
    assert a.shape == (358, 520)
    assert (a.X != expected).nnz == 0
    assert a.X.sum() == 31714
    np.testing.assert_array_equal(a.obs_names, barcodes)
    cells = pd.read_parquet(FIXTURE / "cells.parquet").set_index("cell_id").loc[a.obs_names]
    np.testing.assert_allclose(a.obsm["spatial"], cells[["x_centroid", "y_centroid"]])
    assert "cell_boundaries" in a.obsm and "nucleus_boundaries" in a.obsm
    assert len(a.uns["transcripts"]) > 0
    ab.pp.calculate_qc_metrics(data)
    ab.tl.spatial_neighbors(data, radius=15)
    assert a.obsp["spatial_neighbors_connectivities"].nnz > 0
    path = tmp_path / "vendor.h5mu"
    ab.io.save(data, path)
    restored = ab.io.load(path)["xenium"]
    assert (restored.X != expected).nnz == 0
    np.testing.assert_array_equal(restored.obsm["spatial"], a.obsm["spatial"])
    assert restored.uns["transcripts"].equals(a.uns["transcripts"])
