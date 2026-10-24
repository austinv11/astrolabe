import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import scipy.sparse as sp

import astrolabe as ab


def original():
    a = ad.AnnData(sp.csr_matrix([[2, 0], [0, 2]]), obs=pd.DataFrame(index=["c0", "c1"]))
    a.var_names = ["g0", "g1"]
    a.obs["donor"] = "d0"
    a.uns["transcripts"] = pa.table({"transcript_id": ["t0", "t1", "t2", "t3"], "cell_id": ["c0", "c0", "c1", "c1"],
        "feature_name": ["g0", "g0", "g1", "g1"], "x_location": [0., 1., 2., 3.], "y_location": [0.]*4, "z_location": [5.]*4})
    return a


def output():
    return pd.DataFrame({"transcript_id": [0, 1, 2], "gene": ["g0", "g0", "g1"], "assignment": [0, 1, 1],
                         "background": [False, False, True], "x": [0., 1.5, 2.], "y": [0.]*3, "z": [0.]*3,
                         "observed_x": [0., 1., 2.], "observed_y": [0.]*3, "observed_z": [0.]*3})


def cells():
    return pd.DataFrame({"cell": [0, 1], "original_cell_id": ["c0", "c1"],
                         "centroid_x": [0., 2.], "centroid_y": [0., 0.], "centroid_z": [0., 0.]})


def test_reconstruction_preserves_inputs_and_accounts_for_every_molecule():
    a = original()
    result = ab.seg.import_proseg_reconstruction(a, output(), cells(), id_map=np.arange(4), expected_counts=[[.8, .2], [1.2, .3]])
    np.testing.assert_array_equal(a.X.toarray(), [[2, 0], [0, 2]])
    np.testing.assert_array_equal(result.X.toarray(), [[1, 0], [1, 0]])
    assert result.uns["reconstruction"]["accounting"] == {"assigned": 2, "background": 1, "filtered": 1}
    assert result.uns["reconstruction"]["probability_semantics"] == "not supplied"
    assert result.uns["transcripts"]["x_location"].to_pylist() == [0., 1., 2., 3.]
    assert result.uns["transcripts_inferred"]["x_location"].to_pylist() == [0., 1.5, 2., 3.]
    assert result.uns["transcripts_inferred"]["z_location"].to_pylist() == [5.]*4
    assert (result.obs.donor == "d0").all()
    assert not np.equal(result.layers["expected_counts"].data, np.floor(result.layers["expected_counts"].data)).all()
    ab.seg.select_transcript_coordinates(result, source="inferred")
    assert result.uns["reconstruction"]["coordinate_source"] == "inferred"
    assert result.uns["transcripts"]["x_location"].to_pylist()[1] == 1.5


@pytest.mark.parametrize("bad", ["gene", "cell", "id", "frame", "probability"])
def test_reconstruction_rejects_misaligned_or_invalid_outputs(bad):
    frame = output()
    if bad == "gene":
        frame.loc[0, "gene"] = "g1"
    elif bad == "cell":
        frame.loc[0, "assignment"] = 999
    elif bad == "id":
        frame.loc[0, "transcript_id"] = 2
    elif bad == "frame":
        frame.loc[0, "observed_x"] = 100
    else:
        frame["assignment_probability"] = 1.2
    with pytest.raises(ValueError):
        ab.seg.import_proseg_reconstruction(original(), frame, cells(), id_map=np.arange(4))


def test_qc_compares_source_cell_ids_and_preserves_versions():
    reconstructed = ab.seg.import_proseg_reconstruction(original(), output(), cells(), id_map=np.arange(4))
    result = ab.seg.reconstruction_qc(reconstructed, marker_pairs=[("g0", "g1")])
    assert result["genes"].loc["g0", "n_reassigned"] == 1
    assert result["genes"].loc["g0", "reassignment_fraction"] == .5
    assert result["genes"].loc["g1", "n_filtered"] == 1
    assert result["marker_pairs"].n_joint_detected.iloc[0] == 0
    record = ab.seg.record_segmentation(reconstructed, version="fit1", source="Proseg 3.2")
    assert record["n_cells"] == 2
    with pytest.raises(ValueError, match="already exists"):
        ab.seg.record_segmentation(reconstructed, version="fit1", source="other")


@pytest.mark.parametrize("bad", ["unknown_unit", "cross_unit", "missing_id", "nuclear_flag", "coordinate"])
def test_runner_rejects_invalid_input_before_creating_output(tmp_path, monkeypatch, bad):
    from types import SimpleNamespace
    monkeypatch.setattr("astrolabe.seg._proseg.shutil.which", lambda path: str(tmp_path / "proseg.exe"))
    monkeypatch.setattr("astrolabe.seg._proseg.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout="proseg 3.2.0"))
    a = original()
    a.obs["core_id"] = ["a", "b"]
    frame = a.uns["transcripts"].to_pandas()
    frame["core_id"] = ["a", "a", "b", "b"]
    frame["overlaps_nucleus"] = True
    if bad == "unknown_unit":
        frame.loc[0, "core_id"] = "missing"
    elif bad == "cross_unit":
        frame.loc[0, "cell_id"] = "c1"
    elif bad == "missing_id":
        frame.loc[0, "transcript_id"] = None
    elif bad == "nuclear_flag":
        frame["overlaps_nucleus"] = "yes"
    else:
        frame.loc[0, "x_location"] = np.nan
    a.uns["transcripts"] = pa.Table.from_pandas(frame, preserve_index=False)
    with pytest.raises(ValueError):
        ab.seg.run_proseg(a, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fail_import", [False, True])
def test_runner_records_import_outcome_and_retains_iterable_options(tmp_path, monkeypatch, fail_import):
    import json
    from types import SimpleNamespace

    commands = []
    monkeypatch.setattr("astrolabe.seg._proseg.shutil.which", lambda path: str(tmp_path / "proseg.exe"))
    def run(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(stdout="proseg 3.2.0")
        commands.append(command)
        import gzip
        from scipy.io import mmwrite
        directory = kwargs["cwd"]
        pd.DataFrame({"gene": ["g1", "g0"]}).to_csv(directory / "genes.csv.gz", index=False)
        with gzip.open(directory / "counts.mtx.gz", "wb") as stream:
            mmwrite(stream, sp.coo_matrix([[0, 2], [2, 0]]))
        with gzip.open(directory / "expected-counts.mtx.gz", "wb") as stream:
            mmwrite(stream, sp.coo_matrix([[.25, 1.75], [1.75, .25]]))
    monkeypatch.setattr("astrolabe.seg._proseg.subprocess.run", run)
    def import_result(*args, **kwargs):
        directory = args[1].parent
        assert json.loads((directory / "manifest.json").read_text())["status"] == "backend_completed"
        if fail_import:
            raise ValueError("invalid backend output")
        result = original()
        result.obs["cell"] = [0, 1]
        result.layers["counts"] = result.X.copy()
        result.uns["reconstruction"] = {"provenance": dict(kwargs["provenance"])}
        return result
    monkeypatch.setattr("astrolabe.seg._proseg.import_proseg_reconstruction", import_result)
    a = original()
    frame = a.uns["transcripts"].to_pandas()
    frame["overlaps_nucleus"] = True
    a.uns["transcripts"] = pa.Table.from_pandas(frame, preserve_index=False)
    root = tmp_path / "run"
    def invoke():
        return ab.seg.run_proseg(a, output_dir=root, extra_args=iter(["--samples", "10"]), include_expected_counts=True)
    if fail_import:
        with pytest.raises(ValueError, match="invalid backend output"):
            invoke()
    else:
        results = invoke()
        assert next(iter(results.values())).uns["reconstruction"]["provenance"]["status"] == "import_completed"
        result = next(iter(results.values()))
        np.testing.assert_allclose(result.layers["expected_counts"].toarray(), [[1.75, .25], [.25, 1.75]])
        assert result.uns["reconstruction"]["provenance"]["count_validation"] == "matched_backend_export"
    record = json.loads((root / "manifest.json").read_text())["runs"][0]
    assert record["status"] == ("import_failed" if fail_import else "import_completed")
    assert json.loads(next(root.glob("unit_*/manifest.json")).read_text()) == record
    index = commands[0].index("--samples")
    assert commands[0][index+1] == "10"


@pytest.mark.parametrize("bad", ["shape", "gene", "fractional", "negative"])
def test_backend_count_export_rejects_invalid_alignment(tmp_path, bad):
    from scipy.io import mmwrite
    from astrolabe.seg._proseg import _read_count_export
    result = original()
    result.obs["cell"] = [0, 1]
    genes = pd.DataFrame({"gene": ["g0", "g1"]})
    matrix = np.eye(2)
    if bad == "shape":
        matrix = np.eye(3)
    elif bad == "gene":
        genes.loc[0, "gene"] = "unknown"
    elif bad == "fractional":
        matrix[0, 0] = .5
    else:
        matrix[0, 0] = -1
    path = tmp_path / "counts.mtx"
    mmwrite(path, sp.coo_matrix(matrix))
    with pytest.raises(ValueError):
        _read_count_export(path, genes, result, integer=True)
