"""Process-isolated validation driver; reuses only identical graph/prior tables.

Runs the shipped worker for every distinct design/response. Sensitivity overrides
are confined to this benchmark and never change the public adapter's defaults.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    root = Path(sys.argv[1]).resolve()
    source = Path(__file__).resolve().parents[1]/"astrolabe/tl/_inla_worker.py"
    spec = importlib.util.spec_from_file_location("validated_worker", source)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    # Limit BLAS before importing the optional backend.
    import os
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np
    import pyinla.pc_bym as pc
    original_prior, original_fit = pc.pc_bym_phi, worker._fit_native
    cache, settings = {}, {}
    def cached_prior(**kwargs):
        kwargs.update(u=settings.get("phi_u", .5), alpha=settings.get("phi_alpha", .5))
        key = (hashlib.sha256(kwargs["graph"].tobytes()).hexdigest(), kwargs["u"], kwargs["alpha"])
        if key not in cache:
            cache[key] = original_prior(**kwargs)
        return copy.deepcopy(cache[key])
    pc.pc_bym_phi = cached_prior
    def sensitivity_fit(pyinla, model, data, offset, config, directory, index, table, reader):
        model = copy.deepcopy(model)
        if "spatial_sd_u" in settings:
            model["random"][0]["hyper"][0] = {"prior": "pc.prec", "param": [settings["spatial_sd_u"], .01]}
        if "donor_rate" in settings:
            model["random"][1]["hyper"] = [{"prior": "loggamma", "param": [1., settings["donor_rate"]]}]
        return original_fit(pyinla, model, data, offset, config, directory, index, table, reader)
    worker._fit_native = sensitivity_fit
    manifest = json.loads((root/"manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        settings.clear()
        settings.update(case["prior_settings"])
        directory = root/case["directory"]
        started = time.perf_counter()
        row = {key: value for key, value in case.items() if key != "directory"}
        try:
            worker.run(directory)
            result = json.loads((directory/"result.json").read_text())
            row.update(result["results"][0])
            row["covered"] = row["credible_low"] <= row["truth"] <= row["credible_high"]
            row["excludes_zero"] = row["credible_low"] > 0 or row["credible_high"] < 0
            row["prior_identity_verified"] = result["backend"]["prior_identity_verified"]
            row["engine_sha256"] = result["backend"]["engine_sha256"]
            # The shipped worker records its defaults. Override labels here
            # so benchmark provenance describes the sensitivity fit accurately.
            result["backend"]["benchmark_prior_settings"] = dict(settings)
            result["backend"]["priors"]["bym2_precision"] = f"pc.prec({settings.get('spatial_sd_u', 1)}, 0.01)"
            result["backend"]["priors"]["bym2_phi"] = f"pc(u={settings.get('phi_u', .5)}, alpha={settings.get('phi_alpha', .5)})"
            result["backend"]["priors"]["donor_precision"] = f"loggamma(1, {settings.get('donor_rate', .00005)})"
            (directory/"result.json").write_text(json.dumps(result, allow_nan=False))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        row["seconds"] = time.perf_counter()-started
        rows.append(row)
        (root/"results.json").write_text(json.dumps({"manifest": manifest, "runs": rows}, indent=2, allow_nan=False))
        print(case["scenario"], case["seed"], case["prior"], row["status"], round(row["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
