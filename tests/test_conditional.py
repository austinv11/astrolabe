"""Input contracts and opt-in numerical checks for the INLA integration."""
import json
import os
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import astrolabe as ab
from astrolabe.tl import _conditional


@pytest.fixture
def counts():
    n = 48
    x = np.tile([-1., 1.], n//2)
    rng = np.random.default_rng(12)
    data = ad.AnnData(sparse.csr_matrix(rng.poisson(np.exp(2+.5*x))[:, None]),
        obs=pd.DataFrame({"x": x, "exposure": np.ones(n), "core_id": np.repeat(np.arange(8), 6),
                          "donor": np.repeat(np.arange(4), 12), "type": "T"},
                         index=[f"c{i}" for i in range(n)]), var=pd.DataFrame(index=["g"]))
    data.layers["counts"] = data.X.copy()
    block = sparse.diags([np.ones(5), np.ones(5)], [-1, 1], shape=(6, 6), format="csr")
    data.obsp["graph"] = sparse.block_diag([block]*8, format="csr")
    data.uns["spatial_neighbors"] = {"connectivities_key": "graph"}
    return data


def fit(data, **kwargs):
    return ab.tl.conditional_expression(data, ["g"], predictor="x", exposure_key="exposure", **kwargs)


@pytest.mark.parametrize("problem,match", [
    ("fractional", "integer counts"), ("negative", "integer counts"),
    ("nan_counts", "integer counts"), ("exposure", "Exposure"),
    ("constant", "rank deficient"), ("missing_predictor", "complete"),
    ("weighted", "symmetric binary"), ("asymmetric", "symmetric binary"),
    ("cross_core", "cross-library/core"), ("panel", "not measured"),
])
def test_invalid_inputs_before_backend(counts, monkeypatch, problem, match):
    def unexpected(*args, **kwargs):
        pytest.fail("Backend must not start for invalid inputs")
    monkeypatch.setattr(_conditional, "_run_inla", unexpected)
    if problem in {"fractional", "negative", "nan_counts"}:
        counts.layers["counts"] = counts.layers["counts"].astype(float)
        counts.layers["counts"][0, 0] = {"fractional": .5, "negative": -1, "nan_counts": np.nan}[problem]
    elif problem == "exposure":
        counts.obs.loc["c0", "exposure"] = 0
    elif problem == "constant":
        counts.obs["x"] = 1
    elif problem == "missing_predictor":
        counts.obs.loc["c0", "x"] = np.nan
    elif problem == "panel":
        counts.obs["library_id"] = "a"
        counts.uns["measured_features"] = pd.DataFrame({"a": [False]}, index=["g"])
    else:
        graph = counts.obsp["graph"].tolil()
        if problem == "weighted":
            graph[0, 1] = graph[1, 0] = .5
        elif problem == "asymmetric":
            graph[0, 1] = 0
        else:
            graph[0, 12] = graph[12, 0] = 1
        counts.obsp["graph"] = graph.tocsr()
    with pytest.raises(ValueError, match=match):
        fit(counts)
    assert "conditional_expression" not in counts.uns


def test_population_requirements(counts):
    with pytest.raises(ValueError, match="at least three cells"):
        fit(counts[:2].copy())
    with pytest.raises(ValueError, match="both cell_type"):
        fit(counts, cell_type="T")
    counts.obs["donor"] = 0
    with pytest.raises(ValueError, match="three biological donors"):
        fit(counts, donor_key="donor")


@pytest.mark.parametrize("prior", [None, 1., (0., .01), (1., 0.), (1., 1.), (np.nan, .01), (1., .01, .5)])
def test_invalid_donor_prior_rejected_before_backend(counts, monkeypatch, prior):
    monkeypatch.setattr(_conditional, "_run_inla", lambda *args: pytest.fail("Invalid prior reached backend"))
    with pytest.raises(ValueError, match="donor_sd_prior"):
        fit(counts, donor_key="donor", donor_sd_prior=prior)


def test_native_donor_prior_mismatch_rejected():
    from astrolabe.tl._inla_worker import _check_donor_prior
    model = {"random": [{"id": "donor", "model": "iid", "hyper": [{"prior": "pc.prec", "param": [1., .01]}]}]}
    lines = ["!donor!", "prior = pcprec", "parameters = 1 0.01", "fixed = 0", "!INLA!"]
    _check_donor_prior(model, lines)
    for changed in ("parameters = 1 0.1", "prior = loggamma", "fixed = 1"):
        key = changed.split(" = ")[0]
        invalid = [changed if line.startswith(key + " = ") else line for line in lines]
        with pytest.raises(RuntimeError, match="donor precision prior"):
            _check_donor_prior(model, invalid)


def test_categorical_design():
    frame = pd.DataFrame({"region": ["B", "A", "B", "A", "B", "A"],
                          "batch": ["y", "x", "x", "y", "y", "x"]})
    design, encoding = _conditional._design(frame, "region", ["batch"], ("B", "A"))
    np.testing.assert_array_equal(design.target, [1, 0, 1, 0, 1, 0])
    assert encoding["covariate_1"] == "batch=y"
    with pytest.raises(ValueError, match="two requested"):
        _conditional._design(frame, "region", [], ("B", "C"))


def test_backend_failure_is_atomic(counts, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("engine failed")
    monkeypatch.setattr(_conditional, "_run_inla", fail)
    original = counts.layers["counts"].copy()
    with pytest.raises(RuntimeError, match="engine failed"):
        fit(counts)
    assert "conditional_expression" not in counts.uns
    assert (original != counts.layers["counts"]).nnz == 0


def test_selected_population_and_log_exposure_are_passed_to_backend(counts, monkeypatch):
    counts.obs["exposure"] = np.geomspace(.5, 2, len(counts))
    counts.obs["type"] = ["T"]*12+["B"]*36
    def capture(directory, *args):
        with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
            np.testing.assert_allclose(inputs["offset"], np.log(counts.obs.exposure.iloc[:12]))
            np.testing.assert_array_equal(inputs["counts"], counts.layers["counts"][:12].toarray())
            np.testing.assert_array_equal(inputs["design"][:, 0], counts.obs.x.iloc[:12])
        graph = sparse.load_npz(directory / "graph.npz")
        assert graph.shape == (12, 12)
        return {"backend": {"name": "test double"}, "results": [{"gene": "g", "status": "fitted",
            "posterior_mean": .5, "posterior_sd": .1, "credible_low": .3, "credible_high": .7,
            "probability_positive": .99, "probability_negative": .01, "posterior_sign_error": .01}]}
    monkeypatch.setattr(_conditional, "_run_inla", capture)
    result = fit(counts, cell_type_key="type", cell_type="T")
    assert result.n_cells.tolist() == [12]
    assert counts.uns["conditional_expression"]["parameters"]["n_units"] == 2
    assert counts.n_obs == 48


def test_large_population_reaches_backend_without_allocation_cutoff(counts, monkeypatch):
    # 1,440 cells exceed both former cell and estimated-workspace cutoffs.
    large = ad.concat([counts] * 30, index_unique="-", pairwise=True, uns_merge="same")

    def capture(directory, *args):
        graph = sparse.load_npz(directory / "graph.npz")
        assert graph.shape == (1440, 1440)
        with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
            assert inputs["counts"].shape == (1440, 1)
        raise RuntimeError("reached backend")

    monkeypatch.setattr(_conditional, "_run_inla", capture)
    with pytest.raises(RuntimeError, match="reached backend"):
        fit(large)


def test_donor_graph_independence_without_core_labels(counts):
    del counts.obs["core_id"]
    with pytest.raises(ValueError, match="library_key or core_key"):
        fit(counts, donor_key="donor")
    graph = counts.obsp["graph"].tolil()
    graph[0, 12] = graph[12, 0] = 1
    counts.obsp["graph"] = graph.tocsr()
    with pytest.raises(ValueError, match="different biological donors"):
        fit(counts, donor_key="donor")


def test_worker_rejects_unsupported_version_before_import(monkeypatch, tmp_path):
    from astrolabe.tl import _inla_worker
    monkeypatch.setattr(_inla_worker, "version", lambda name: "0.1.12")
    with pytest.raises(RuntimeError, match="requires pyINLA 0.1.11"):
        _inla_worker.run(tmp_path)


def test_worker_rejects_missing_package(monkeypatch, tmp_path):
    from importlib.metadata import PackageNotFoundError
    from astrolabe.tl import _inla_worker
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(_inla_worker, "version", missing)
    with pytest.raises(RuntimeError, match="optional astrolabe"):
        _inla_worker.run(tmp_path)


def test_linux_engine_identity_hashes_native_code(tmp_path):
    import hashlib
    from astrolabe.tl._inla_worker import _engine_identity
    launcher = tmp_path / "inla.mkl.run"
    launcher.write_bytes(b"shell launcher")
    (tmp_path / "bin").mkdir()
    engine = tmp_path / "bin" / "inla"
    engine.write_bytes(b"native executable")
    identity = _engine_identity(launcher)
    assert identity["engine_sha256"] == hashlib.sha256(engine.read_bytes()).hexdigest()
    assert identity["launcher_sha256"] == hashlib.sha256(launcher.read_bytes()).hexdigest()
    engine.unlink()
    with pytest.raises(RuntimeError, match="Cannot identify"):
        _engine_identity(launcher)


def test_worker_rejects_missing_engine_without_download(monkeypatch, tmp_path):
    from types import ModuleType
    from astrolabe.tl import _inla_worker
    monkeypatch.setattr(_inla_worker, "version", lambda name: "0.1.11")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, os.environ.get(name, "1"))
    (tmp_path / "config.json").write_text(json.dumps({"num_threads": 1}))
    def forbidden(*args, **kwargs):
        pytest.fail("No backend work or downloads are allowed without an installed engine")
    modules = {
        "pyinla": {"is_binary_installed": lambda: False},
        "pyinla.binary": {"BinaryManager": forbidden, "get_binary_version": forbidden},
        "pyinla.pc_bym": {"pc_bym_phi": forbidden},
        "pyinla.fmesher_io": {"read_fmesher_file": forbidden},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    with pytest.raises(RuntimeError, match="never downloads"):
        _inla_worker.run(tmp_path)


def test_native_prior_mismatch_rejected(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from astrolabe.tl import _inla_worker
    def native(**kwargs):
        artifact = Path(kwargs["working_directory"]) / "inla.model"
        artifact.mkdir()
        (artifact / "Model.ini").write_text("prior1 = table: $inladatadir/prior.dat\n")
        return object()
    table = np.column_stack([np.arange(4), np.zeros(4)])
    with pytest.raises(RuntimeError, match="different BYM2 prior"):
        _inla_worker._fit_native(SimpleNamespace(pyinla=native), {}, None, None,
            {"num_threads": 1, "native_root": str(tmp_path)}, tmp_path, 0, table,
            lambda path: table+1)
    assert not list(tmp_path.iterdir())


def test_timeout_terminates_worker_tree_and_cleans_native_scratch(monkeypatch, tmp_path):
    seen = {}
    class Process:
        pid = 123456
        returncode = 0
        def __init__(self, command, **kwargs):
            seen["native_root"] = Path(command[-1])
            assert seen["native_root"].exists()
            self.calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", .01)
        def poll(self):
            return 0
    monkeypatch.setattr(_conditional.subprocess, "Popen", Process)
    def terminated(*args, **kwargs):
        seen["terminated"] = True
    monkeypatch.setattr(_conditional.subprocess, "run", terminated)
    if hasattr(_conditional.os, "killpg"):
        monkeypatch.setattr(_conditional.os, "killpg", terminated)
    with pytest.raises(RuntimeError, match="exceeded timeout"):
        _conditional._run_inla(tmp_path, sys.executable, .01)
    assert seen["terminated"]
    assert not seen["native_root"].exists()


def test_optional_import_isolation():
    script = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'pyinla', 'spatialdata'}:
            raise AssertionError('unexpected optional import: '+name)
sys.meta_path.insert(0, Block())
import astrolabe
assert callable(astrolabe.tl.conditional_expression)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=90)


@pytest.mark.skipif(not os.environ.get("ASTROLABE_INLA_PYTHON"), reason="explicit optional INLA environment required")
@pytest.mark.parametrize("donor_sd_prior", [(1., .01), (.7, .05)])
def test_optional_native_backend_and_direct_reference(counts, tmp_path, donor_sd_prior):
    python = os.environ["ASTROLABE_INLA_PYTHON"]
    graph = counts.obsp["graph"].tolil()
    graph[0, 1] = graph[1, 0] = 0
    counts.obsp["graph"] = graph.tocsr()
    result = fit(counts, donor_key="donor", donor_sd_prior=donor_sd_prior, python_executable=python, output_dir=tmp_path)
    record = counts.uns["conditional_expression"]
    assert result.status.tolist() == ["fitted"]
    assert record["backend"]["prior_identity_verified"]
    assert record["backend"]["donor_prior_identity_verified"]
    assert record["parameters"]["donor_sd_prior"] == list(donor_sd_prior)
    assert record["parameters"]["n_components"] == 9  # eight cores and one isolated cell
    assert record["parameters"]["n_units"] == 8
    assert record["parameters"]["n_donors"] == 4
    assert not {"pvalue", "padj"} & set(result.columns)
    assert result.probability_tail_truncated.dtype == bool
    # A separate direct backend call reads the same numeric inputs. It does not
    # call the worker or adapter, and uses the backend's own prior construction.
    directory = Path(record["parameters"]["artifacts"])
    script = tmp_path / "direct.py"
    script.write_text('''
import json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
import pyinla
from pyinla.pc_bym import pc_bym_phi
p=Path(sys.argv[1]); a=np.load(p/'inputs.npz'); g=sparse.load_npz(p/'graph.npz')
donor_prior=json.loads((p/'config.json').read_text())['donor_sd_prior']
prior=pc_bym_phi(graph=g.toarray(),alpha=.5,u=.5,scale_model=True,adjust_for_con_comp=True,return_as_table=True)
t=np.column_stack([prior['theta'],prior['logprior']])
data=pd.DataFrame({'target':a['design'][:,0],'y':a['counts'][:,0],'cell':np.arange(1,g.shape[0]+1),'donor':a['donor']})
model={'response':'y','fixed':['1','target'],'random':[
 {'id':'cell','model':'bym2','graph':g,'scale.model':True,'adjust.for.con.comp':True,
  'hyper':[{}, {'prior':'table: '+' '.join(format(v,'.17g') for v in t.ravel(order='F'))}]},
 {'id':'donor','model':'iid','hyper':[{'prior':'pc.prec','param':donor_prior}]}]}
r=pyinla.pyinla(model=model,family='nbinomial',data=data,offset=a['offset'],num_threads=1,
 control={'inla':{'strategy':'simplified.laplace','int.strategy':'ccd'}})
(p/'direct.json').write_text(json.dumps(r.summary_fixed.loc['target'].to_dict()))
''', encoding="utf-8")
    subprocess.run([python, "-I", str(script), str(directory)], check=True, timeout=300)
    direct = json.loads((directory / "direct.json").read_text())
    np.testing.assert_allclose(result.loc[0, ["posterior_mean", "posterior_sd", "credible_low", "credible_high"]].to_numpy(dtype=float),
                               [direct[k] for k in ("mean", "sd", "0.025quant", "0.975quant")], rtol=1e-5, atol=1e-6)
    counts.write_h5ad(tmp_path / "result.h5ad")
    restored = ad.read_h5ad(tmp_path / "result.h5ad")
    pd.testing.assert_frame_equal(restored.uns["conditional_expression"]["results"], result)
    counts.obs.loc["c0", "exposure"] *= 2
    assert not ab.tl.validate_results(counts).empty
    zero = counts.copy()
    zero.layers["counts"] = sparse.csr_matrix(zero.shape)
    result = fit(zero, python_executable=python)
    assert result.status.tolist() == ["zero_counts"]
    assert result.posterior_mean.isna().all()
    assert not zero.uns["conditional_expression"]["backend"]["prior_identity_verified"]
    zero.write_h5ad(tmp_path / "zero.h5ad")
