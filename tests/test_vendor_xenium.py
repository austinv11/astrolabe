"""Real vendor files (downloaded, never vendored): schema, count and round-trip invariants.

The data come from the ``xenium_tiny_dir`` session fixture in ``conftest.py``
(10x Xenium V1 Protein Human Kidney tiny, XOA 4.0.0, CC BY 4.0).  Only the
checksums of the six files the reader consumes live in the repository; they pin
the exact vendor bytes these expectations were derived from.
"""
import hashlib

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

import astrolabe as ab


# SHA-256 of the unmodified archive members, recorded 2026-10-23.
XENIUM4_TINY_SHA256 = {
    "cell_boundaries.parquet": "8d0309239bea32672733d045dd31f237cc13e8cd4b3459fdbe35a7c77833057b",
    "cell_feature_matrix.h5": "1e20d87cffd9784d678e4ca90511c5ac2fe61becea363e81a8ecab87e6abfa73",
    "cells.parquet": "8f2f5a00782c356c521671e3267c45cea567e7365428726228443ae704256246",
    "experiment.xenium": "c44938ee3c0895c8d270fa1541739145d49143e95c0efe96132a0a179d1cd233",
    "nucleus_boundaries.parquet": "14b6c61380605841490328dd38d08ece166a7a4e074510fdcf3a8b5561ac5ae9",
    "transcripts.parquet": "065a258da4d06a2274cefc249f11d504c917919b35be2093b6923a8f75b4e3b5",
}


def test_xenium4_vendor_counts_coordinates_and_roundtrip(tmp_path, xenium_tiny_dir):
    FIXTURE = xenium_tiny_dir
    for name, digest in XENIUM4_TINY_SHA256.items():
        assert hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest() == digest, name
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
