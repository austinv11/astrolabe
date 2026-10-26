"""Tests for astrolabe.io._atera (read_atera, read_atera_transcripts, filter_atera)."""

from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
import tifffile
import zarr

import astrolabe as ab
from astrolabe.io._atera import _decode_10x_cell_id


def _encode_10x_barcode(barcode: str) -> tuple[int, int]:
    """Inverse of _decode_10x_cell_id: 'aaaaaplg-1' -> (4022, 1)."""
    prefix_str, suffix_str = barcode.split("-")
    hex_chars = "".join(f"{ord(c) - ord('a'):x}" for c in prefix_str)
    return int(hex_chars, 16), int(suffix_str)


def _create_zarr_array(grp, name: str, data: np.ndarray, *, attrs: dict | None = None):
    arr = np.asarray(data)
    if arr.dtype.kind in ("U", "O"):
        arr = np.asarray(arr, dtype="S")
    z = grp.create_array(name, shape=arr.shape, dtype=arr.dtype, attributes=attrs)
    z[:] = arr
    return z


def _build_synthetic_atera_bundle(root_dir: Path, *, n_cells: int = 8) -> Path:
    """Create a realistic synthetic 10x Atera output bundle in *root_dir*."""
    root_dir.mkdir(parents=True, exist_ok=True)

    # 1. experiment.spatial & panel_config.json
    exp_meta = {
        "major_version": 1,
        "minor_version": 0,
        "pixel_size": 0.2125,
        "z_step_size": 0.75,
        "chemistry_version": "Atera v1",
        "segmentation_stain": "DAPI + 18S",
        "instrument_sw_version": "1.0.0",
        "analysis_sw_version": "atera-1.0.0",
    }
    (root_dir / "experiment.spatial").write_text(json.dumps(exp_meta), encoding="utf-8")
    (root_dir / "panel_config.json").write_text(
        json.dumps({"panel_name": "Human Immuno-Oncology", "num_targets": 4}),
        encoding="utf-8",
    )

    # Generate 10x-style barcodes and (prefix, suffix) pairs
    barcodes = [f"aaaaab{chr(ord('a') + i)}b-1" for i in range(n_cells)]
    cell_id_pairs = np.array([_encode_10x_barcode(bc) for bc in barcodes], dtype=np.uint32)

    # Features: 4 real genes + 2 control features (6 total)
    feature_ids = [
        "ENSG00000119888",
        "ENSG00000167286",
        "ENSG00000153563",
        "ENSG00000010610",
        "NegControlProbe_0001",
        "NegControlCodeword_0500",
    ]
    feature_names = [
        "EPCAM",
        "CD3D",
        "CD8A",
        "CD4",
        "NegControlProbe_0001",
        "NegControlCodeword_0500",
    ]
    feature_types = [
        "Gene Expression",
        "Gene Expression",
        "Gene Expression",
        "Gene Expression",
        "Negative Control Probe",
        "Negative Control Codeword",
    ]
    n_vars = len(feature_names)

    # Build a CSR count matrix where every cell has >= 5 genes and >= 20 counts
    rng = np.random.default_rng(42)
    dense_real = rng.integers(3, 12, size=(n_cells, 4), dtype=np.int32)
    dense_ctrl = np.zeros((n_cells, 2), dtype=np.int32)
    dense_ctrl[0, 0] = 1
    dense_ctrl[1, 1] = 1
    dense_X = np.hstack([dense_real, dense_ctrl])
    csr_X = sp.csr_matrix(dense_X)

    # 2. cell_feature_matrix.zarr.zip
    cfm_path = root_dir / "cell_feature_matrix.zarr.zip"
    store = zarr.storage.ZipStore(str(cfm_path), mode="w")
    try:
        zroot = zarr.open_group(store=store, mode="w", zarr_format=2)

        # /X
        x_grp = zroot.create_group(
            "X",
            attributes={"encoding-type": "csr_matrix", "shape": [n_cells, n_vars]},
        )
        _create_zarr_array(x_grp, "data", csr_X.data.astype(np.int32))
        _create_zarr_array(x_grp, "indices", csr_X.indices.astype(np.int64))
        _create_zarr_array(x_grp, "indptr", csr_X.indptr.astype(np.int64))

        # /obs
        obs_grp = zroot.create_group("obs")
        centroids_x = np.linspace(10.0, 10.0 * n_cells, n_cells, dtype=np.float32)
        centroids_y = np.linspace(20.0, 20.0 + 5.0 * (n_cells - 1), n_cells, dtype=np.float32)
        _create_zarr_array(obs_grp, "barcode", np.array(barcodes, dtype="S"))
        _create_zarr_array(obs_grp, "cell_area", np.full(n_cells, 120.0, dtype=np.float32))
        _create_zarr_array(obs_grp, "nucleus_area", np.full(n_cells, 45.0, dtype=np.float32))
        _create_zarr_array(obs_grp, "centroid_column", centroids_x)
        _create_zarr_array(obs_grp, "centroid_row", centroids_y)
        _create_zarr_array(obs_grp, "total_counts", dense_X.sum(axis=1).astype(np.int32))
        _create_zarr_array(obs_grp, "transcript_counts", dense_real.sum(axis=1).astype(np.int32))
        _create_zarr_array(obs_grp, "control_probe_counts", dense_ctrl[:, 0].astype(np.int32))
        _create_zarr_array(obs_grp, "control_codeword_counts", dense_ctrl[:, 1].astype(np.int32))
        # Onboard secondary analysis in obs
        _create_zarr_array(obs_grp, "filtered", np.ones(n_cells, dtype=bool))
        _create_zarr_array(obs_grp, "is_sketch", np.ones(n_cells, dtype=bool))
        az_grp = obs_grp.create_group("azimuth_broad")
        _create_zarr_array(az_grp, "categories", np.array(["Epithelial", "T cell"], dtype="S"))
        _create_zarr_array(az_grp, "codes", np.array([i % 2 for i in range(n_cells)], dtype=np.int8))
        _create_zarr_array(az_grp, "color", np.array(["#1f77b4", "#ff7f0e"], dtype="S"))

        leiden_grp = obs_grp.create_group("leiden_res_1.0")
        _create_zarr_array(leiden_grp, "categories", np.array(["0", "1"], dtype="S"))
        _create_zarr_array(leiden_grp, "codes", np.array([i % 2 for i in range(n_cells)], dtype=np.int8))

        # /obsm (onboard PCA & UMAP)
        obsm_grp = zroot.create_group("obsm")
        _create_zarr_array(obsm_grp, "X_pca", rng.normal(size=(n_cells, 4)).astype(np.float32))
        _create_zarr_array(obsm_grp, "X_umap", rng.normal(size=(n_cells, 2)).astype(np.float32))

        # /var
        var_grp = zroot.create_group("var")
        _create_zarr_array(var_grp, "feature_id", np.array(feature_ids, dtype="S"))
        _create_zarr_array(var_grp, "feature_name", np.array(feature_names, dtype="S"))
        ft_grp = var_grp.create_group("feature_type")
        ft_cats = ["Gene Expression", "Negative Control Probe", "Negative Control Codeword"]
        ft_codes = [ft_cats.index(ft) for ft in feature_types]
        _create_zarr_array(ft_grp, "categories", np.array(ft_cats, dtype="S"))
        _create_zarr_array(ft_grp, "codes", np.array(ft_codes, dtype=np.int8))
        # Onboard secondary analysis in var (1D boolean + 2D per-cluster DE matrix)
        _create_zarr_array(var_grp, "filtered", np.array([True, True, True, True, False, False]))
        _create_zarr_array(var_grp, "highly_variable", np.array([True, True, False, False, False, False]))
        _create_zarr_array(var_grp, "de_leiden_1.0_logfc", np.zeros((n_vars, 2), dtype=np.float32))
    finally:
        store.close()

    # 3. cells.zarr.zip
    cells_path = root_dir / "cells.zarr.zip"
    cstore = zarr.storage.ZipStore(str(cells_path), mode="w")
    try:
        croot = zarr.open_group(store=cstore, mode="w", zarr_format=2)
        _create_zarr_array(croot, "cell_id", cell_id_pairs)

        summary = np.column_stack([
            centroids_x,
            centroids_y,
            np.full(n_cells, 120.0),
            centroids_x + 0.5,
            centroids_y + 0.5,
            np.full(n_cells, 45.0),
            np.full(n_cells, 3.0),
            np.ones(n_cells),
        ]).astype(np.float64)
        _create_zarr_array(
            croot,
            "cell_summary",
            summary,
            attrs={
                "column_names": [
                    "cell_centroid_x",
                    "cell_centroid_y",
                    "cell_area",
                    "nucleus_centroid_x",
                    "nucleus_centroid_y",
                    "nucleus_area",
                    "z_level",
                    "nucleus_count",
                ]
            },
        )

        psets = croot.create_group("polygon_sets")
        # polygon_sets/0: nucleus boundaries (4 vertices per cell, padded to 5)
        p0 = psets.create_group("0")
        _create_zarr_array(p0, "cell_index", np.arange(n_cells, dtype=np.uint32))
        _create_zarr_array(p0, "num_vertices", np.full(n_cells, 4, dtype=np.int32))
        nuc_verts = np.zeros((n_cells, 5, 2), dtype=np.float32)
        for i in range(n_cells):
            cx, cy = centroids_x[i], centroids_y[i]
            nuc_verts[i, :4] = [[cx - 2, cy - 2], [cx + 2, cy - 2], [cx + 2, cy + 2], [cx - 2, cy - 2]]
        _create_zarr_array(p0, "vertices", nuc_verts)

        # polygon_sets/1: cell boundaries (2D flat-padded shape (n_cells, 2*max_verts))
        p1 = psets.create_group("1")
        _create_zarr_array(p1, "cell_index", np.arange(n_cells, dtype=np.uint32))
        _create_zarr_array(p1, "num_vertices", np.full(n_cells, 4, dtype=np.int32))
        cell_verts = np.zeros((n_cells, 5, 2), dtype=np.float32)
        for i in range(n_cells):
            cx, cy = centroids_x[i], centroids_y[i]
            cell_verts[i, :4] = [[cx - 5, cy - 5], [cx + 5, cy - 5], [cx + 5, cy + 5], [cx - 5, cy - 5]]
        _create_zarr_array(p1, "vertices", cell_verts.reshape(n_cells, 10))
    finally:
        cstore.close()

    # 4. transcripts.zarr.zip
    tx_path = root_dir / "transcripts.zarr.zip"
    tstore = zarr.storage.ZipStore(str(tx_path), mode="w")
    try:
        troot = zarr.open_group(
            store=tstore,
            mode="w",
            zarr_format=2,
            attributes={
                "gene_names": feature_names,
                "fov_names": ["FOV_0", "FOV_1"],
                "spatial_units": "micron",
            },
        )

        grids = troot.create_group("grids")
        g0 = grids.create_group("0")

        # Tile "0,0": 4 valid transcripts + 1 low-QV transcript + 1 no-call (65535)
        t00 = g0.create_group("0,0")
        _create_zarr_array(
            t00,
            "location",
            np.array(
                [
                    [10.0, 20.0, 1.5],
                    [20.0, 25.0, 2.0],
                    [30.0, 30.0, 1.0],
                    [40.0, 35.0, 2.5],
                    [50.0, 40.0, 1.0],
                    [60.0, 45.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        _create_zarr_array(
            t00,
            "gene_identity",
            np.array([0, 1, 4, 5, 0, 65535], dtype=np.uint16),
        )
        _create_zarr_array(
            t00,
            "quality_score",
            np.array([35.0, 28.0, 22.0, 25.0, 10.0, 30.0], dtype=np.float32),
        )
        _create_zarr_array(
            t00,
            "id",
            np.array(
                [[101, 0], [102, 0], [103, 0], [104, 1], [105, 1], [106, 1]],
                dtype=np.int64,
            ),
        )
        # Assign first 3 to cells 0, 1, 0; 4th unassigned (0, 0)
        tile_cids = np.array(
            [
                cell_id_pairs[0],
                cell_id_pairs[1],
                cell_id_pairs[0],
                [0, 0],
                cell_id_pairs[2],
                [0, 0],
            ],
            dtype=np.uint32,
        )
        _create_zarr_array(t00, "cell_id", tile_cids)
        _create_zarr_array(t00, "overlaps_nucleus", np.array([1, 1, 0, 0, 1, 0], dtype=np.uint8))

        # Tile "0,1": 2 additional valid transcripts for CD8A and CD4
        t01 = g0.create_group("0,1")
        _create_zarr_array(
            t01,
            "location",
            np.array([[30.0, 30.0, 1.2], [40.0, 35.0, 1.8]], dtype=np.float32),
        )
        _create_zarr_array(t01, "gene_identity", np.array([2, 3], dtype=np.uint16))
        _create_zarr_array(t01, "quality_score", np.array([32.0, 31.0], dtype=np.float32))
        _create_zarr_array(t01, "id", np.array([[201, 1], [202, 1]], dtype=np.int64))
        _create_zarr_array(
            t01,
            "cell_id",
            np.array([cell_id_pairs[2], cell_id_pairs[3]], dtype=np.uint32),
        )
        _create_zarr_array(t01, "overlaps_nucleus", np.array([1, 0], dtype=np.uint8))
    finally:
        tstore.close()

    # 5. morphology_2d and morphology_3d OME-TIFFs
    morph_2d = root_dir / "morphology_2d"
    morph_2d.mkdir()
    tifffile.imwrite(
        str(morph_2d / "ch0000_dapi.ome.tif"),
        np.zeros((32, 32), dtype=np.uint16),
    )
    tifffile.imwrite(
        str(morph_2d / "ch0001_18s.ome.tif"),
        np.zeros((32, 32), dtype=np.uint16),
    )

    morph_3d = root_dir / "morphology_3d"
    morph_3d.mkdir()
    tifffile.imwrite(
        str(morph_3d / "ch0000_dapi_3d.ome.tif"),
        np.zeros((4, 32, 32), dtype=np.uint16),
    )

    return root_dir


class TestReadAtera:
    def test_10x_cell_id_roundtrip(self):
        barcodes = np.array(["aaaaaplg-1", "aaaabjej-1", "ffffffff-2"], dtype=object)
        pairs = np.array([_encode_10x_barcode(b) for b in barcodes], dtype=np.uint32)
        decoded = _decode_10x_cell_id(pairs)
        np.testing.assert_array_equal(decoded, barcodes)

    def test_read_atera_defaults(self, tmp_path):
        bundle = _build_synthetic_atera_bundle(tmp_path / "atera_run")
        mdata = ab.io.read_atera(bundle)

        assert mdata.uns["default_modality"] == "atera"
        assert mdata.uns["astrolabe_schema_version"] == "0.1"
        assert "atera" in mdata.mod

        adata = mdata["atera"]
        assert adata.shape == (8, 6)
        assert adata.obs.index.name == "cell_id"
        assert adata.var.index.name == "gene_name"
        assert list(adata.var_names[:4]) == ["EPCAM", "CD3D", "CD8A", "CD4"]

        # Primary obs columns present; onboard secondary analysis excluded by default
        assert "cell_area" in adata.obs.columns
        assert "nucleus_area" in adata.obs.columns
        assert "z_level" in adata.obs.columns
        assert "nucleus_count" in adata.obs.columns
        assert "azimuth_broad" not in adata.obs.columns
        assert "leiden_res_1.0" not in adata.obs.columns
        assert "filtered" not in adata.obs.columns
        assert "X_umap" not in adata.obsm
        assert "X_pca" not in adata.obsm
        assert "de_leiden_1.0_logfc" not in adata.varm

        # Control flags
        assert adata.var["is_control_probe"].tolist() == [False, False, False, False, True, True]
        assert adata.var.loc["NegControlProbe_0001", "feature_type"] == "Negative Control Probe"
        assert adata.var.loc["NegControlCodeword_0500", "feature_type"] == "Negative Control Codeword"

        # Spatial coordinates & boundaries
        assert adata.obsm["spatial"].shape == (8, 2)
        assert adata.obsm["spatial"].dtype == np.float32
        assert "cell_boundaries" in adata.obsm
        assert "nucleus_boundaries" in adata.obsm
        assert len(adata.obsm["cell_boundaries"]) == 8
        assert len(adata.obsm["nucleus_boundaries"]) == 8
        first_cell_poly = ak.to_numpy(adata.obsm["cell_boundaries"][0])
        assert first_cell_poly.shape == (4, 2)

        # Morphology 2D attached by default, 3D off by default
        images = adata.uns["spatial"]["atera_run"]["images"]
        assert "full" in images
        assert "morphology_2d_ch0000_dapi" in images
        assert "morphology_2d_ch0001_18s" in images
        assert images["full"] is images["morphology_2d_ch0000_dapi"]
        assert "morphology_3d" not in images

        # Transcripts auto-loaded into uns['transcripts'] (6 rows across 2 tiles with QV>=20 and valid gene)
        tx = adata.uns["transcripts"]
        assert len(tx) == 6
        assert set(tx.column("feature_name").to_pylist()) == {
            "EPCAM",
            "CD3D",
            "CD8A",
            "CD4",
            "NegControlProbe_0001",
            "NegControlCodeword_0500",
        }
        assert "UNASSIGNED" in tx.column("cell_id").to_pylist()

    def test_load_onboard_analysis_true_and_morphology_3d(self, tmp_path):
        bundle = _build_synthetic_atera_bundle(tmp_path / "outs")
        mdata = ab.io.read_atera(
            tmp_path,
            load_onboard_analysis=True,
            morphology_3d=True,
            var_names="gene_id",
        )
        adata = mdata["atera"]

        # Library ID resolved from parent when reading via outs/
        assert tmp_path.name in adata.uns["spatial"]
        images = adata.uns["spatial"][tmp_path.name]["images"]
        assert "morphology_3d" in images
        assert "morphology_3d_ch0000_dapi_3d" in images

        # Onboard analysis loaded
        assert "azimuth_broad" in adata.obs.columns
        assert isinstance(adata.obs["azimuth_broad"].dtype, pd.CategoricalDtype)
        assert adata.uns["azimuth_broad_colors"] == ["#1f77b4", "#ff7f0e"]
        assert "leiden_res_1.0" in adata.obs.columns
        assert "X_pca" in adata.obsm
        assert "X_umap" in adata.obsm
        assert "highly_variable" in adata.var.columns
        assert "de_leiden_1.0_logfc" in adata.varm
        assert adata.varm["de_leiden_1.0_logfc"].shape == (6, 2)

        # var_names="gene_id"
        assert adata.var.index.name == "gene_id"
        assert "ENSG00000119888" in adata.var_names
        assert "gene_name" in adata.var.columns

    def test_read_atera_transcripts_filtering(self, tmp_path):
        bundle = _build_synthetic_atera_bundle(tmp_path / "atera_run")
        mdata = ab.io.read_atera(bundle, load_transcripts=False)
        assert "transcripts" not in mdata["atera"].uns

        ab.io.read_atera_transcripts(mdata, include_controls=False, min_qv=20.0)
        tx = mdata["atera"].uns["transcripts"]
        names = set(tx.column("feature_name").to_pylist())
        assert names == {"EPCAM", "CD3D", "CD8A", "CD4"}

    def test_filter_atera_pipeline_and_save_load_roundtrip(self, tmp_path):
        bundle = _build_synthetic_atera_bundle(tmp_path / "atera_run")
        mdata = ab.io.read_atera(bundle)

        ab.pipeline.filter_atera(
            mdata,
            min_counts=5,
            min_genes=2,
            min_cells=1,
            max_neighbor_dist_um=None,
            min_neighbor_dist_um=None,
        )
        adata = mdata["atera"]
        # Control probes removed by filter_atera(remove_controls=True)
        assert adata.n_vars == 4
        assert not adata.var["is_control_probe"].any()
        assert "control_probe_summary" in adata.uns
        assert "mean_qv" in adata.var.columns

        # Transcripts table synced after gene filtering (controls dropped)
        tx_names = set(adata.uns["transcripts"].column("feature_name").to_pylist())
        assert tx_names == {"EPCAM", "CD3D", "CD8A", "CD4"}

        # Save & load round-trip
        save_path = tmp_path / "saved_atera"
        ab.io.save(mdata, save_path)
        restored = ab.io.load(str(save_path) + ".h5mu")
        assert restored.uns["default_modality"] == "atera"
        np.testing.assert_allclose(restored["atera"].obsm["spatial"], adata.obsm["spatial"])
        assert restored["atera"].uns["transcripts"].equals(adata.uns["transcripts"])

    def test_wta_preview_cell_features_v4_and_external_analysis(self, tmp_path):
        """Validate WTA Preview schema: /cell_features v4 Zarr, analysis.zarr.zip, morphology_focus/, transcripts.parquet."""
        root_dir = tmp_path / "wta_preview_outs"
        root_dir.mkdir(parents=True)
        n_cells = 6
        barcodes = [f"aaaaab{chr(ord('a') + i)}b-1" for i in range(n_cells)]
        cell_id_pairs = np.array([_encode_10x_barcode(bc) for bc in barcodes], dtype=np.uint32)

        # Include 'aggregate_gene' at the end (must be dropped automatically)
        feature_ids = ["ENSG00000119888", "ENSG00000167286", "NegControlProbe_0001", "Total transcripts"]
        feature_keys = ["EPCAM", "CD3D", "NegControlProbe_0001", "Total transcripts"]
        feature_types = ["gene", "gene", "negative_control_probe", "aggregate_gene"]

        dense_with_agg = np.array(
            [
                [5, 2, 1, 7],
                [3, 4, 0, 7],
                [6, 1, 0, 7],
                [2, 5, 1, 7],
                [4, 3, 0, 7],
                [1, 6, 0, 7],
            ],
            dtype=np.uint32,
        )
        csr_with_agg = sp.csr_matrix(dense_with_agg)

        # 1. cell_feature_matrix.zarr.zip with /cell_features v4 schema
        store = zarr.storage.ZipStore(str(root_dir / "cell_feature_matrix.zarr.zip"), mode="w")
        try:
            zroot = zarr.open_group(store=store, mode="w", zarr_format=2)
            cf = zroot.create_group(
                "cell_features",
                attributes={
                    "major_version": 4,
                    "minor_version": 1,
                    "number_cells": n_cells,
                    "number_features": len(feature_keys),
                    "feature_ids": feature_ids,
                    "feature_keys": feature_keys,
                    "feature_types": feature_types,
                },
            )
            _create_zarr_array(cf, "cell_id", cell_id_pairs)
            csc_grp = cf.create_group("csc")
            _create_zarr_array(csc_grp, "data", csr_with_agg.data.astype(np.uint32))
            _create_zarr_array(csc_grp, "indices", csr_with_agg.indices.astype(np.uint32))
            _create_zarr_array(csc_grp, "indptr", csr_with_agg.indptr.astype(np.uint64))
        finally:
            store.close()

        # 2. cells.zarr.zip where nucleus polygon count < n_cells (cell 2 has no nucleus)
        cstore = zarr.storage.ZipStore(str(root_dir / "cells.zarr.zip"), mode="w")
        try:
            croot = zarr.open_group(store=cstore, mode="w", zarr_format=2)
            _create_zarr_array(croot, "cell_id", cell_id_pairs)
            summary = np.column_stack([
                np.arange(n_cells, dtype=np.float64) * 10.0,
                np.arange(n_cells, dtype=np.float64) * 15.0,
                np.full(n_cells, 100.0),
                np.arange(n_cells, dtype=np.float64) * 10.0,
                np.arange(n_cells, dtype=np.float64) * 15.0,
                np.array([40.0, 40.0, np.nan, 40.0, 40.0, 40.0]),
                np.zeros(n_cells),
                np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            ])
            _create_zarr_array(
                croot,
                "cell_summary",
                summary,
                attrs={
                    "column_names": [
                        "cell_centroid_x",
                        "cell_centroid_y",
                        "cell_area",
                        "nucleus_centroid_x",
                        "nucleus_centroid_y",
                        "nucleus_area",
                        "z_level",
                        "nucleus_count",
                    ]
                },
            )
            psets = croot.create_group("polygon_sets")
            p0 = psets.create_group("0")
            nuc_cell_idx = np.array([0, 1, 3, 4, 5], dtype=np.uint32)
            _create_zarr_array(p0, "cell_index", nuc_cell_idx)
            _create_zarr_array(p0, "num_vertices", np.full(len(nuc_cell_idx), 4, dtype=np.int32))
            _create_zarr_array(p0, "vertices", np.ones((len(nuc_cell_idx), 4, 2), dtype=np.float32))

            p1 = psets.create_group("1")
            _create_zarr_array(p1, "cell_index", np.arange(n_cells, dtype=np.uint32))
            _create_zarr_array(p1, "num_vertices", np.full(n_cells, 4, dtype=np.int32))
            _create_zarr_array(p1, "vertices", np.ones((n_cells, 4, 2), dtype=np.float32))
        finally:
            cstore.close()

        # 3. analysis.zarr.zip with /cell_groups
        astore = zarr.storage.ZipStore(str(root_dir / "analysis.zarr.zip"), mode="w")
        try:
            aroot = zarr.open_group(store=astore, mode="w", zarr_format=2)
            cg = aroot.create_group(
                "cell_groups",
                attributes={
                    "grouping_names": ["gene_expression_graphclust"],
                    "group_names": [["1", "2"]],
                },
            )
            g0 = cg.create_group("0")
            _create_zarr_array(g0, "indices", np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32))
            _create_zarr_array(g0, "indptr", np.array([0, 3, 6], dtype=np.uint32))
        finally:
            astore.close()

        # 4. morphology_focus/ and morphology.ome.tif
        mf = root_dir / "morphology_focus"
        mf.mkdir()
        tifffile.imwrite(str(mf / "ch0000_dapi.ome.tif"), np.zeros((16, 16), dtype=np.uint16))
        tifffile.imwrite(str(root_dir / "morphology.ome.tif"), np.zeros((3, 16, 16), dtype=np.uint16))

        # 5. transcripts.parquet
        tx_table = pa.table({
            "transcript_id": pa.array([1, 2, 3], type=pa.uint64()),
            "cell_id": pa.array([barcodes[0], barcodes[1], "UNASSIGNED"]),
            "overlaps_nucleus": pa.array([1, 0, 0], type=pa.uint8()),
            "feature_name": pa.array(["EPCAM", "CD3D", "NegControlProbe_0001"]),
            "x_location": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            "y_location": pa.array([4.0, 5.0, 6.0], type=pa.float32()),
            "z_location": pa.array([0.5, 0.5, 0.5], type=pa.float32()),
            "qv": pa.array([35.0, 30.0, 25.0], type=pa.float32()),
            "fov_name": pa.array(["A1", "A1", "A1"]),
        })
        pq.write_table(tx_table, root_dir / "transcripts.parquet")

        mdata = ab.io.read_atera(root_dir, load_onboard_analysis=True, morphology_3d=True)
        adata = mdata["atera"]

        # Aggregate gene dropped -> 3 features (2 real + 1 negative control probe)
        assert adata.shape == (6, 3)
        np.testing.assert_array_equal(adata.X.toarray(), dense_with_agg[:, :3])
        assert adata.var["feature_type"].tolist() == [
            "Gene Expression",
            "Gene Expression",
            "Negative Control Probe",
        ]
        assert adata.var["is_control_probe"].tolist() == [False, False, True]
        assert adata.obs["transcript_counts"].tolist() == [7, 7, 7, 7, 7, 7]
        assert adata.obs["control_probe_counts"].tolist() == [1, 0, 0, 1, 0, 0]

        # Cell 2 has an empty nucleus polygon, others have 4 vertices
        assert len(adata.obsm["nucleus_boundaries"][2]) == 0
        assert len(adata.obsm["nucleus_boundaries"][0]) == 4

        # External analysis.zarr.zip loaded into obs
        assert "gene_expression_graphclust" in adata.obs.columns
        assert list(adata.obs["gene_expression_graphclust"]) == ["1", "1", "1", "2", "2", "2"]

        # morphology_focus and morphology.ome.tif attached
        images = adata.uns["spatial"]["wta_preview_outs"]["images"]
        assert "morphology_2d_ch0000_dapi" in images
        assert "morphology_3d" in images

        # transcripts.parquet loaded
        assert len(adata.uns["transcripts"]) == 3

    def test_atera_v1_grid_gene_offset_transcripts(self, tmp_path):
        """Validate Atera v1.0 /grid/<tile> transcripts.zarr.zip with gene_offset & codeword_identity."""
        bundle = _build_synthetic_atera_bundle(tmp_path / "atera_v1")
        barcodes = [f"aaaaab{chr(ord('a') + i)}b-1" for i in range(8)]
        cell_id_pairs = np.array([_encode_10x_barcode(bc) for bc in barcodes], dtype=np.uint32)
        feature_names = [
            "EPCAM",
            "CD3D",
            "CD8A",
            "CD4",
            "NegControlProbe_0001",
            "NegControlCodeword_0500",
        ]

        # Overwrite transcripts.zarr.zip with v6 /grid/<tile> schema
        tx_path = bundle / "transcripts.zarr.zip"
        tx_path.unlink()
        tstore = zarr.storage.ZipStore(str(tx_path), mode="w")
        try:
            troot = zarr.open_group(
                store=tstore,
                mode="w",
                zarr_format=2,
                attributes={
                    "major_version": 6,
                    "minor_version": 0,
                    "gene_names": feature_names,
                    "codeword_gene_names": feature_names,
                    "spatial_units": "micron",
                },
            )
            gene_cat = np.array(
                [
                    [True, False],
                    [True, False],
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                ],
                dtype=bool,
            )
            _create_zarr_array(troot, "gene_category", gene_cat)

            grid = troot.create_group("grid")
            t00 = grid.create_group("0,0")
            # 5 transcripts sorted by gene: 2 EPCAM (idx 0), 2 CD3D (idx 1), 1 NegControlProbe_0001 (idx 4)
            goff = np.array(
                [[0, 2], [2, 4], [4, 4], [4, 4], [4, 5], [5, 5]], dtype=np.uint32
            )
            _create_zarr_array(t00, "gene_offset", goff)
            _create_zarr_array(
                t00,
                "location",
                np.array(
                    [
                        [10.0, 20.0, 1.0],
                        [11.0, 21.0, 1.5],
                        [20.0, 25.0, 2.0],
                        [21.0, 26.0, 2.5],
                        [30.0, 30.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            _create_zarr_array(
                t00,
                "quality_score",
                np.array([[35.0], [12.0], [28.0], [32.0], [25.0]], dtype=np.float16),
            )
            _create_zarr_array(
                t00,
                "codeword_identity",
                np.array([[0], [0], [1], [1], [4]], dtype=np.uint32),
            )
            _create_zarr_array(
                t00,
                "cell_id",
                np.array(
                    [
                        cell_id_pairs[0],
                        cell_id_pairs[0],
                        cell_id_pairs[1],
                        [0xFFFFFFFF, 0xFFFFFFFF],
                        cell_id_pairs[2],
                    ],
                    dtype=np.uint32,
                ),
            )
            _create_zarr_array(
                t00,
                "overlaps_nucleus",
                np.array([[1], [1], [1], [0], [0]], dtype=np.uint8),
            )
        finally:
            tstore.close()

        # 1. Full read (all genes + controls with min_qv=20 -> 4 transcripts)
        mdata = ab.io.read_atera(bundle)
        tx = mdata["atera"].uns["transcripts"]
        assert len(tx) == 4
        assert tx.column("feature_name").to_pylist() == [
            "EPCAM",
            "CD3D",
            "CD3D",
            "NegControlProbe_0001",
        ]
        assert tx.column("cell_id").to_pylist()[2] == "UNASSIGNED"

        # 2. Targeted gene_offset slice read for genes=["CD3D"]
        ab.io.read_atera_transcripts(mdata, genes=["CD3D"], include_controls=False, min_qv=20.0)
        tx_cd3d = mdata["atera"].uns["transcripts"]
        assert len(tx_cd3d) == 2
        assert tx_cd3d.column("feature_name").to_pylist() == ["CD3D", "CD3D"]
