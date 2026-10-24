"""Standalone optional backend worker; never imported by Astrolabe.

The subprocess may use a separate Python environment. Interchange is JSON/NPZ,
not pickle, and the backend receives only explicit numeric design columns.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def _engine_identity(launcher):
    """Fingerprint native code as well as any shell launcher."""
    engine = launcher
    if launcher.suffix == ".run":
        candidates = [launcher.parent / "bin" / "inla", launcher.parent / "inla.mkl", launcher.parent / "inla"]
        engine = next((path for path in candidates if path.is_file()), None)
        if engine is None:
            raise RuntimeError("Cannot identify the native INLA engine behind its launcher")
    return {"engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "launcher_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest()}


def _fit_native(pyinla, model, data, offset, config, directory, index, table, read_fmesher_file):
    import numpy as np
    # INLA's Windows executable can fail while writing deep output paths even
    # when Python can create them. Run in a short, fresh native work directory;
    # copy artifacts to the requested destination only after native execution.
    with tempfile.TemporaryDirectory(prefix="g-", dir=config.get("native_root")) as temporary:
        work = Path(temporary)
        try:
            result = pyinla.pyinla(
                model=model, family="nbinomial", data=data, offset=offset,
                control={"inla": {"strategy": "simplified.laplace", "int.strategy": "ccd"}},
                keep=True, working_directory=str(work), num_threads=config["num_threads"],
            )
            artifact = work / "inla.model"
            lines = (artifact / "Model.ini").read_text().splitlines()
            line = next(v for v in lines if v.startswith("prior1 = table:"))
            filename = line.replace(chr(92), "/").rsplit("/", 1)[-1]
            actual = read_fmesher_file(str(artifact / "data.files" / filename))
            if not np.array_equal(actual, table):
                raise RuntimeError("INLA received a different BYM2 prior table; refusing inference")
            _check_donor_prior(model, lines)
            return result
        finally:
            if config.get("keep_artifacts", False):
                shutil.copytree(work, directory / f"gene-{index}")


def _check_donor_prior(model, lines):
    """Check the donor prior actually serialized by the optional wrapper."""
    import numpy as np
    donor = next((term for term in model.get("random", []) if term["id"] == "donor"), None)
    if donor is None:
        return
    prior = donor.get("hyper", [{}])[0]
    if prior.get("prior") not in ("pc.prec", "loggamma") or prior.get("fixed", False):
        return  # Diagnostic oracle models have no inferred donor precision.
    values, active = {}, False
    for line in lines:
        if line.startswith("!") and line.endswith("!"):
            active = line == "!donor!"
        elif active and " = " in line:
            key, value = line.split(" = ", 1)
            values[key] = value
    expected_name = {"pc.prec": "pcprec", "loggamma": "loggamma"}[prior["prior"]]
    actual = np.fromstring(values.get("parameters", ""), sep=" ")
    expected = np.asarray(prior["param"], dtype=float)
    if (values.get("prior") != expected_name or values.get("fixed") != "0"
            or actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-12, atol=0)):
        raise RuntimeError("INLA received a different donor precision prior; refusing inference")


def run(directory, native_root=None):
    # Fail before invoking any backend function that could download an engine.
    try:
        wrapper_version = version("pyinla")
    except ModuleNotFoundError:
        raise RuntimeError("Install optional astrolabe[spatial-inference] on Python 3.12, or select a prepared Python 3.12 environment") from None
    if wrapper_version != "0.1.11":
        raise RuntimeError("This adapter requires pyINLA 0.1.11; other versions need prior-serialization revalidation")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if native_root is not None:
        config["native_root"] = str(native_root)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(config["num_threads"])
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import pyinla
    from pyinla.binary import BinaryManager, get_binary_version
    from pyinla.pc_bym import pc_bym_phi
    from pyinla.fmesher_io import read_fmesher_file

    if not pyinla.is_binary_installed():
        raise RuntimeError("INLA engine is not installed. Install it explicitly using pyINLA before fitting; Astrolabe never downloads it")
    executable = Path(BinaryManager().get_binary_path())
    identity = _engine_identity(executable)
    graph = sparse.load_npz(directory / "graph.npz")
    with np.load(directory / "inputs.npz", allow_pickle=False) as inputs:
        counts, design, offset, donor = (inputs[k] for k in ("counts", "design", "offset", "donor"))
    # The wrapper's default path emits an empty second table. Use its own PC
    # prior unchanged and verify the actual engine input exactly for every fit.
    prior = pc_bym_phi(graph=graph.toarray(), alpha=.5, u=.5, scale_model=True,
                       adjust_for_con_comp=True, return_as_table=True)
    table = np.column_stack([prior["theta"], prior["logprior"]])
    if table.shape[1] != 2 or len(table) < 4 or not np.isfinite(table).all() or not (np.diff(table[:, 0]) > 0).all():
        raise RuntimeError("Backend produced an invalid BYM2 PC prior table")
    encoded = "table: " + " ".join(format(v, ".17g") for v in table.ravel(order="F"))
    records, hyperparameters = [], []
    for index, gene in enumerate(config["genes"]):
        record = {"gene": gene, "status": "zero_counts", "posterior_mean": None,
                  "posterior_sd": None, "credible_low": None, "credible_high": None,
                  "probability_positive": None, "probability_negative": None,
                  "posterior_sign_error": None, "probability_tail_truncated": False}
        if not np.any(counts[:, index]):
            records.append(record)
            continue
        data = pd.DataFrame(design, columns=config["fixed"])
        data["y"] = counts[:, index]
        data["cell"] = np.arange(1, len(data)+1)
        random = [{"id": "cell", "model": "bym2", "graph": graph.copy(),
                   "scale.model": True, "adjust.for.con.comp": True,
                   "hyper": [{}, {"prior": encoded}]}]
        if config["donor"]:
            data["donor"] = donor
            random.append({"id": "donor", "model": "iid",
                           "hyper": [{"prior": "pc.prec", "param": config.get("donor_sd_prior", [1., .01])}]})
        result = _fit_native(pyinla,
            {"response": "y", "fixed": ["1", *config["fixed"]], "random": random},
            data, offset, config, directory, index, table, read_fmesher_file)
        row = result.summary_fixed.loc["target"]
        marginal = result.marginals_fixed["target"]
        grid = marginal["x"].to_numpy(dtype=float)
        density = marginal["y"].to_numpy(dtype=float)
        if not np.isfinite(grid).all() or not np.isfinite(density).all() or (density < 0).any() or not (np.diff(grid) > 0).all():
            raise RuntimeError(f"Invalid posterior marginal for gene {gene!r}")
        probability_negative = float(np.asarray(pyinla.pmarginal(0, marginal)).item())
        probability_positive = 1-probability_negative
        values = [float(row[k]) for k in ("mean", "sd", "0.025quant", "0.975quant")]
        if not np.isfinite(values).all() or values[1] <= 0 or values[2] >= values[3] or not 0 <= probability_negative <= 1:
            raise RuntimeError(f"Invalid posterior summary for gene {gene!r}")
        record.update(status="fitted", posterior_mean=values[0], posterior_sd=values[1],
                      credible_low=values[2], credible_high=values[3],
                      probability_positive=probability_positive, probability_negative=probability_negative,
                      posterior_sign_error=min(probability_negative, probability_positive),
                      probability_tail_truncated=bool(0 <= grid[0] or 0 >= grid[-1]))
        records.append(record)
        hyperparameters.append({"gene": gene, "summary": result.summary_hyperpar.to_dict()})
    build_info = executable.parent / "BUILDINFO"
    backend = {
        "method": "negative-binomial BYM2/INLA", "pyinla_version": wrapper_version,
        **identity, "engine_metadata": get_binary_version(),
        "build_info": build_info.read_text(errors="replace") if build_info.exists() else "unavailable",
        "python_version": sys.version, "numpy_version": np.__version__,
        "scipy_version": version("scipy"), "pandas_version": pd.__version__,
        "strategy": "simplified.laplace", "integration": "ccd", "num_threads": config["num_threads"],
        "priors": {"bym2_precision": "pc.prec(1, 0.01)", "bym2_phi": "pc(u=0.5, alpha=0.5)",
                   "donor_precision": ("pc.prec(" + ", ".join(map(str, config.get("donor_sd_prior", [1., .01]))) + ")") if config["donor"] else "not used",
                   "nb_size": "pc.mgamma(7)",
                   "intercept": "flat", "fixed_effects": "normal(mean=0, precision=0.001)"},
        "prior_table_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
        "prior_identity_verified": bool(hyperparameters), "n_fitted": len(hyperparameters),
        "donor_prior_identity_verified": bool(hyperparameters) and config["donor"],
        "hyperparameter_summaries_json": json.dumps(hyperparameters, allow_nan=False),
        "inference": "Bayesian; no p-values or BH adjustment",
    }
    (directory / "result.json").write_text(json.dumps({"results": records, "backend": backend}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None)
