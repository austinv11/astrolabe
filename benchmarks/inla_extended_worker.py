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
    import pyinla
    import pyinla.pc_bym as pc
    original_prior, original_fit = pc.pc_bym_phi, worker._fit_native
    cache, settings = {}, {}
    native_checks = {}
    native_call = pyinla.pyinla
    def checked_native(**kwargs):
        kwargs["control"]["inla"]["strategy"] = settings.get("strategy", "simplified.laplace")
        if settings.get("all_oracle"):
            kwargs["control"]["family"] = {"hyper": [{"id": "size", "initial": float(np.log(10)), "fixed": True}]}
        result = native_call(**kwargs)
        if settings.get("all_oracle") and result.summary_hyperpar is None:
            # No hyperparameter posterior exists when every one is fixed.
            import pandas as pd
            result.summary_hyperpar = pd.DataFrame()
        sections, current = {}, None
        model_path = Path(kwargs["working_directory"])/"inla.model/Model.ini"
        for line in model_path.read_text().splitlines():
            if line.startswith("!") and line.endswith("!"):
                current = sections.setdefault(line.strip("!"), {})
            elif current is not None and " = " in line:
                key, value = line.split(" = ", 1)
                current[key] = value
        donor = sections.get("donor", {})
        if "donor_pc_u" in settings:
            assert donor.get("prior", donor.get("prior0")) in ("pcprec", "pc.prec"), donor
            actual = donor.get("parameters", donor.get("parameters0", ""))
            np.testing.assert_allclose(np.fromstring(actual, sep=" "), [settings["donor_pc_u"], .01])
        if "donor_fixed_precision" in settings:
            assert donor.get("fixed", donor.get("fixed0")) == "1", donor
            # pyINLA writes initial values with six decimal places.
            np.testing.assert_allclose(float(donor.get("initial", donor.get("initial0"))), np.log(settings["donor_fixed_precision"]), atol=5e-7)
        if settings.get("all_oracle"):
            cell = sections["cell"]
            assert cell["fixed0"] == cell["fixed1"] == "1", cell
            np.testing.assert_allclose([float(cell["initial0"]), float(cell["initial1"])], [np.log(1/.3**2), 0.], atol=5e-7)
            likelihood = next(values for values in sections.values() if values.get("likelihood") == "nbinomial")
            assert likelihood.get("fixed", likelihood.get("fixed0")) == "1", likelihood
            np.testing.assert_allclose(float(likelihood.get("initial", likelihood.get("initial0"))), np.log(10), atol=5e-7)
        native_checks.clear()
        native_checks.update({key: value for key, value in sections.items() if key in ("donor", "INLA")})
        return result
    pyinla.pyinla = checked_native
    def cached_prior(**kwargs):
        kwargs.update(u=settings.get("phi_u", .5), alpha=settings.get("phi_alpha", .5))
        key = (hashlib.sha256(kwargs["graph"].tobytes()).hexdigest(), kwargs["u"], kwargs["alpha"])
        if key not in cache:
            cache[key] = original_prior(**kwargs)
        return copy.deepcopy(cache[key])
    pc.pc_bym_phi = cached_prior
    def sensitivity_fit(pyinla, model, data, offset, config, directory, index, table, reader):
        model = copy.deepcopy(model)
        # Preserve the original comparison arm when production defaults change.
        if len(model["random"]) > 1:
            model["random"][1]["hyper"] = [{"prior": "loggamma", "param": [1., .00005]}]
        if "spatial_sd_u" in settings:
            model["random"][0]["hyper"][0] = {"prior": "pc.prec", "param": [settings["spatial_sd_u"], .01]}
        if "donor_rate" in settings:
            model["random"][1]["hyper"] = [{"prior": "loggamma", "param": [1., settings["donor_rate"]]}]
        if "donor_pc_u" in settings:
            model["random"][1]["hyper"] = [{"prior": "pc.prec", "param": [settings["donor_pc_u"], .01]}]
        if "donor_fixed_precision" in settings:
            model["random"][1]["hyper"] = [{"initial": float(np.log(settings["donor_fixed_precision"])), "fixed": True}]
        if settings.get("all_oracle"):
            model["random"][0]["hyper"][0].update(initial=float(np.log(1/.3**2)), fixed=True)
            model["random"][0]["hyper"][1].update(initial=0., fixed=True)
        return original_fit(pyinla, model, data, offset, config, directory, index, table, reader)
    worker._fit_native = sensitivity_fit
    manifest = json.loads((root/"manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        settings.clear()
        settings.update(case["prior_settings"])
        native_checks.clear()
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
            row["native_settings"] = dict(native_checks)
            row["hyperparameters"] = json.loads(result["backend"]["hyperparameter_summaries_json"])
            # The shipped worker records its defaults. Override labels here
            # so benchmark provenance describes the sensitivity fit accurately.
            result["backend"]["benchmark_prior_settings"] = dict(settings)
            result["backend"]["priors"]["bym2_precision"] = f"pc.prec({settings.get('spatial_sd_u', 1)}, 0.01)"
            result["backend"]["priors"]["bym2_phi"] = f"pc(u={settings.get('phi_u', .5)}, alpha={settings.get('phi_alpha', .5)})"
            result["backend"]["priors"]["donor_precision"] = f"loggamma(1, {settings.get('donor_rate', .00005)})"
            if "donor_pc_u" in settings:
                result["backend"]["priors"]["donor_precision"] = f"pc.prec({settings['donor_pc_u']}, 0.01)"
            if "donor_fixed_precision" in settings:
                result["backend"]["priors"]["donor_precision"] = f"fixed({settings['donor_fixed_precision']})"
            if settings.get("all_oracle"):
                result["backend"]["priors"].update(bym2_precision="fixed(1/0.3**2)", bym2_phi="fixed(0.5)", nb_size="fixed(10)")
            result["backend"]["strategy"] = settings.get("strategy", "simplified.laplace")
            (directory/"result.json").write_text(json.dumps(result, allow_nan=False))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        row["seconds"] = time.perf_counter()-started
        rows.append(row)
        (root/"results.json").write_text(json.dumps({"manifest": manifest, "runs": rows}, indent=2, allow_nan=False))
        print(case["scenario"], case["seed"], case["prior"], row["status"], round(row["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
