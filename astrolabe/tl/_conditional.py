"""Explicit optional INLA integration for conditional count associations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from astrolabe._coverage import _require_measured
from astrolabe._dependencies import _digest
from astrolabe._memory import _budget
from astrolabe._spatial_units import _units
from astrolabe._utils import from_kwarg, with_adata


def _design(frame, predictor, covariates, contrast):
    columns = [predictor, *covariates]
    if len(set(columns)) != len(columns):
        raise ValueError("predictor and covariates must be distinct")
    if frame[columns].isna().any().any():
        raise ValueError("Predictor and covariates must be complete")
    if contrast is None:
        if not pd.api.types.is_numeric_dtype(frame[predictor]):
            raise ValueError("A categorical predictor requires contrast=(tested, reference)")
        target = frame[predictor].to_numpy(dtype=float)
    else:
        if len(contrast) != 2 or contrast[0] == contrast[1]:
            raise ValueError("contrast requires distinct tested/reference levels")
        if not frame[predictor].isin(contrast).all():
            raise ValueError("Select only the two requested contrast levels before fitting")
        target = (frame[predictor] == contrast[0]).to_numpy(dtype=float)
    design = pd.DataFrame({"target": target})
    encoding = {"target": predictor}
    for key in covariates:
        if pd.api.types.is_numeric_dtype(frame[key]):
            values = frame[[key]].reset_index(drop=True).astype(float)
        else:
            values = pd.get_dummies(frame[key].reset_index(drop=True), drop_first=True, dtype=float)
        for label in values:
            name = f"covariate_{len(encoding)}"
            design[name] = values[label].to_numpy()
            encoding[name] = key if pd.api.types.is_numeric_dtype(frame[key]) else f"{key}={label}"
    array = np.column_stack([np.ones(len(frame)), design.to_numpy()])
    if not np.isfinite(array).all():
        raise ValueError("Predictor and covariates must be finite")
    if len(frame) <= array.shape[1] or np.linalg.matrix_rank(array) != array.shape[1]:
        raise ValueError("Design is rank deficient or has no residual information")
    return design, encoding


def _run_inla(directory, python_executable, timeout):
    worker = Path(__file__).with_name("_inla_worker.py")
    log = directory / "backend.log"
    try:
        with tempfile.TemporaryDirectory(prefix="inla-") as native_root, log.open("w", encoding="utf-8") as stream:
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
            with subprocess.Popen(
                [str(python_executable), "-I", str(worker), str(directory), native_root],
                stdout=stream, stderr=subprocess.STDOUT, **options,
            ) as process:
                try:
                    process.wait(timeout=timeout)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    # Terminate our worker's native engine too, before temporary
                    # interchange files are cleaned or an interrupted call exits.
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    raise
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"INLA exceeded timeout={timeout}; no results were stored") from exc
    if process.returncode or not (directory / "result.json").exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Optional INLA backend failed; no results were stored.\n{tail}")
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


@with_adata(mirror_uns=from_kwarg("key_added"), spatial_result=True)
def conditional_expression(
    adata, genes, *, predictor, exposure_key, covariates=(), contrast=None,
    donor_key=None, donor_sd_prior=(1.0, 0.01), cell_type_key=None, cell_type=None, layer="counts",
    neighbors_key="spatial_neighbors", library_key="auto", core_key="auto",
    python_executable=None, output_dir=None, max_cells=1000,
    max_working_bytes=512 * 1024**2, timeout=1800, num_threads=1,
    key_added="conditional_expression",
):
    """Fit negative-binomial BYM2 associations using optional pyINLA 0.1.11.

    Parameters
    ----------
    genes : str or sequence of str
        Prespecified measured genes; counts must be nonnegative integers.
    predictor : str
        Numeric obs column (effect per original unit), or a categorical column
        with ``contrast=(tested, reference)``. Covariates are supplied explicitly.
    exposure_key : str
        Positive, finite obs exposure. Its logarithm is an offset; no exposure
        or normalization is inferred from the requested genes.
    donor_key : str, optional
        Biological donor IDs for an IID intercept shared across that donor's
        cores. At least three donors are required, not a guarantee of precision.
    donor_sd_prior : tuple of float
        Published PC precision prior specified as ``(u, alpha)``, meaning
        ``P(donor SD > u) = alpha`` on the natural-log expression scale.
        Default ``(1, 0.01)`` uses the same scale as the spatial SD prior.
        Choose this scale for the biological question; small cohorts can be
        sensitive to it. Used only when ``donor_key`` is supplied.
    cell_type_key, cell_type : str, optional
        Explicit single-cell-type restriction; supply both or neither.
    neighbors_key : str
        Metadata for a symmetric binary graph. Cross-library/core edges are
        rejected. BYM2 scaling/constraints apply to each connected component.
    python_executable : path, optional
        Python with pyINLA and its engine already installed. Defaults to this
        interpreter. A separate Python 3.12 environment supports Python 3.13 hosts.
        Nothing is installed or downloaded by this function.
    output_dir : path, optional
        Retain native backend files in a new unique subdirectory. Otherwise
        temporary engine interchange files are removed. This is not disk-backed
        array computation: inputs and the backend's dense prior algebra use RAM.
    max_cells, max_working_bytes : int
        Bounds on dense prior work. The byte estimate is not a process RSS cap.
    timeout : float
        Maximum worker runtime in seconds for the entire gene batch.

    Returns
    -------
    pandas.DataFrame
        Natural-log effects, posterior SDs, 95% credible intervals and posterior
        sign probabilities. These are not p-values; no BH correction is applied.
        Results and design/backend provenance are stored in ``uns[key_added]``.

    Notes
    -----
    Uses published BYM2/INLA with simplified Laplace and CCD integration. This
    optional integration is experimental; see docs/g5-inla-proposal.md for priors,
    validation and limitations. Varying-design simulations show undercoverage,
    especially for small donor cohorts; nominal 95% coverage is not established.
    The model does not propagate segmentation error
    or correct selection of predictors/niches from the same expression data.
    """
    genes = [genes] if isinstance(genes, str) else list(genes)
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    try:
        donor_prior = np.asarray(donor_sd_prior, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("donor_sd_prior requires (u > 0, 0 < alpha < 1)") from None
    if (donor_prior.shape != (2,) or not np.isfinite(donor_prior).all()
            or donor_prior[0] <= 0 or not 0 < donor_prior[1] < 1):
        raise ValueError("donor_sd_prior requires finite (u > 0, 0 < alpha < 1)")
    if not genes or len(set(genes)) != len(genes) or not adata.var_names.is_unique:
        raise ValueError("Select distinct genes on a unique feature axis")
    if not adata.obs_names.is_unique:
        raise ValueError("Observation names must be unique")
    if not pd.Index(genes).isin(adata.var_names).all():
        raise KeyError("Requested genes are absent from var_names")
    if (cell_type_key is None) != (cell_type is None):
        raise ValueError("Supply both cell_type_key and cell_type, or neither")
    mask = np.ones(adata.n_obs, dtype=bool)
    if cell_type_key is not None:
        if adata.obs[cell_type_key].isna().any():
            raise ValueError("Cell-type labels must be complete")
        mask = (adata.obs[cell_type_key] == cell_type).to_numpy()
    indices = np.flatnonzero(mask)
    n = len(indices)
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    if n < 3 or n > max_cells:
        raise ValueError(f"Selected {n} cells; supported range is 3..max_cells={max_cells}")
    if not isinstance(num_threads, int) or isinstance(num_threads, bool) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    _budget(96*n*n + 32*n*10000 + 8*n*len(genes), max_working_bytes, "BYM2 dense prior workspace")
    selected = adata[indices]
    _require_measured(selected, genes)
    design, encoding = _design(selected.obs, predictor, list(covariates), contrast)
    exposure = selected.obs[exposure_key].to_numpy(dtype=float)
    if not np.isfinite(exposure).all() or (exposure <= 0).any():
        raise ValueError("Exposure must be finite and strictly positive")
    matrix = selected.X if layer is None else selected.layers[layer]
    matrix = matrix[:, selected.var_names.get_indexer(genes)]
    counts = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    counts = np.asarray(counts, dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("Expression must contain finite nonnegative integer counts")
    metadata = adata.uns.get(neighbors_key, {})
    graph_key = metadata.get("connectivities_key")
    if graph_key not in adata.obsp:
        raise ValueError("neighbors_key must identify an existing spatial connectivity graph")
    graph = sparse.csr_matrix(adata.obsp[graph_key][indices][:, indices], dtype=float)
    graph.sum_duplicates()
    graph.setdiag(0)
    graph.eliminate_zeros()
    if graph.nnz == 0 or not np.isfinite(graph.data).all() or (graph.data != 1).any() or (graph != graph.T).nnz:
        raise ValueError("BYM2 requires a symmetric binary graph with at least one edge")
    units, unit_columns = _units(adata, library_key, core_key, neighbors_key)
    codes = np.empty(adata.n_obs, dtype=int)
    for code, (_, rows) in enumerate(units):
        codes[rows] = code
    codes = codes[indices]
    rows, cols = graph.nonzero()
    if (codes[rows] != codes[cols]).any():
        raise ValueError("Spatial graph contains cross-library/core edges")
    donor_codes = np.zeros(n, dtype=int)
    n_donors = 0
    if donor_key is not None:
        if selected.obs[donor_key].isna().any():
            raise ValueError("Donor labels must be complete")
        donor_codes, donors = pd.factorize(selected.obs[donor_key], sort=False)
        n_donors = len(donors)
        if n_donors < 3:
            raise ValueError("A donor random intercept requires at least three biological donors")
        if (donor_codes[rows] != donor_codes[cols]).any():
            raise ValueError("Spatial graph contains edges between different biological donors")
        if not unit_columns:
            raise ValueError("Supply library_key or core_key for a multi-donor spatial model")
        if any(len(np.unique(donor_codes[codes == code])) != 1 for code in np.unique(codes)) and unit_columns:
            raise ValueError("Each physical library/core must belong to one donor")
    n_components = connected_components(graph, directed=False, return_labels=False)
    parameters = {
        "predictor": predictor, "exposure_key": exposure_key, "covariates": list(covariates),
        "contrast": [] if contrast is None else [str(v) for v in contrast],
        "donor_key": donor_key or "", "cell_type_key": cell_type_key or "",
        "donor_sd_prior": donor_prior.tolist() if donor_key is not None else [],
        "cell_type": "all" if cell_type is None else str(cell_type), "layer": layer or "X",
        "neighbors_key": neighbors_key, "unit_columns": unit_columns, "design_encoding": encoding,
        "n_cells": n, "n_donors": n_donors, "n_units": len(np.unique(codes)),
        "n_components": int(n_components), "graph_sha256": _digest(graph),
        "input_sha256": _digest((counts, design, exposure, donor_codes)),
        "effect_scale": "natural log; per predictor unit or tested versus reference",
        "max_cells": max_cells, "max_working_bytes": max_working_bytes,
    }
    config = {"genes": genes, "fixed": list(design), "donor": donor_key is not None,
              "donor_sd_prior": donor_prior.tolist(),
              "num_threads": num_threads, "keep_artifacts": output_dir is not None}
    python_executable = sys.executable if python_executable is None else python_executable
    def fit(directory):
        sparse.save_npz(directory / "graph.npz", graph)
        np.savez(directory / "inputs.npz", counts=counts, design=design.to_numpy(),
                 offset=np.log(exposure), donor=donor_codes+1)
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return _run_inla(directory, python_executable, timeout)
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="astrolabe-inla-") as directory:
            report = fit(Path(directory))
    else:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="astrolabe-inla-", dir=root))
        report = fit(directory)
        parameters["artifacts"] = str(directory)
    result = pd.DataFrame(report["results"])
    for column in ("posterior_mean", "posterior_sd", "credible_low", "credible_high",
                   "probability_positive", "probability_negative", "posterior_sign_error"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype(float)
    result["n_cells"] = n
    result["n_donors"] = n_donors
    adata.uns[key_added] = {"results": result, "parameters": parameters, "backend": report["backend"]}
    return result
