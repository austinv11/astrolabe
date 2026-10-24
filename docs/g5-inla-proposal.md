# Optional conditional expression: published BYM2 with INLA

## Method and supported question

Use the established negative-binomial spatial generalized linear mixed-model
framework with the published BYM2 graph effect, fitted by the INLA engine through
pyINLA. This replaces the withdrawn custom PyMC proposal. BYM2 is documented in
[Riebler et al. (2016)](https://doi.org/10.1177/0962280216660421);
[smiDE](https://doi.org/10.1186/s13059-025-03867-1) also discusses BYM2 for spatial
expression analysis. The Python interface is described by the
[INLA authors](https://arxiv.org/abs/2603.27276).

For prespecified genes within a cell type, estimate one prespecified predictor's
log-expression effect conditional on supplied covariates, count exposure, a
spatial graph and an optional shared donor intercept. A spatial effect is not
evidence of causality and cannot resolve a confounded design.

## Tissue and donor structure

Use a symmetric binary graph with no cross-library/core edges. Apply the published
BYM2 scaling and sum-to-zero constraints separately to connected components;
singleton components follow the backend's documented handling. Disconnected fields
are conditionally independent given their shared hyperparameters. Donor IID effects
remain shared across the donor's cores. This uses supported backend terms directly.
Core identity is not a substitute for donor replication.

## Dependency and output decision

Approved: expose this established backend as an optional dependency. The public
entry point is `tl.conditional_expression`; ordinary package imports do not load
pyINLA, inspect its engine, install software or download binaries.

pyINLA 0.1.11 runs on native Windows without R. Its wheel metadata declares a
proprietary license and NumPy `<2.1`; available wheels cover Python 3.10–3.12.
Install `astrolabe[spatial-inference]` in Python 3.12, or use a separately installed
Python 3.12 environment selected with `python_executable`. The extra is marked
for Python 3.12; on Python 3.13+ it does not install a backend. Use the separate
environment in that case. Ordinary installations do not inherit the NumPy cap.
The wrapper offers a license-activation API; no activation was required for the
reference fits here. Dependency terms are not covered by Astrolabe's MIT license.

Import/fitting is explicit; no automatic package or engine downloads by Astrolabe.
Record wrapper and engine versions, executable fingerprints, graph, priors,
covariates and inference settings. The public adapter rejects missing engines,
unsupported versions, failed fits, invalid counts/offsets and rank-deficient designs.

Results are posterior means, standard deviations, credible intervals and posterior
sign probabilities. They are **not frequentist p-values**; no BH correction of
posterior probabilities. Use the documented PC priors, with their numerical values
recorded, and simplified Laplace inference with CCD hyperparameter integration.
The donor intercept uses an explicit PC precision prior with
`donor_sd_prior=(1, .01)`: `P(donor SD > 1) = .01` on the log-expression scale.
This matches the spatial SD prior's scale; callers can supply another positive
scale/probability pair justified by their experiment. Native prior parameters
are verified after serialization, alongside the BYM2 table. The inherited
log-gamma donor baseline and the explicit PC specification are compared in the
[undercoverage investigation](inla-undercoverage-investigation.md).
Sign probabilities are numerical marginal integrals. `probability_tail_truncated`
flags when zero lies outside the backend's finite marginal grid; reported zero
or one then reflects truncated numerical support, not absolute certainty.
Calibration and prior-sensitivity checks remain required before inferential claims.
The integration's current validation does not establish general nominal coverage.
In a 200-response fixed-design null check, 7 intervals exclude zero (3.5%; exact
95% interval 1.42–7.08%). Smaller varying-design checks have less favorable results;
see [all validation results and limitations](spatial-validation.md).

Initial scope is bounded reference-sized datasets: the wrapper's BYM2 prior
calculation uses dense graph algebra. Enforce an explicit cell/memory limit rather
than silently replacing the method with an approximation.

## Reference evidence and backend defect

A 96-cell, eight-core, four-donor Gaussian check with fixed covariance parameters
agrees with an independent analytic posterior: maximum mean error `6.6e-7` and
standard-deviation error `7.4e-6`. This checks core constraints and shared donor
covariance. A negative-binomial check recovers a planted coefficient of `0.5` as
`0.42336` with 95% credible interval `[0.3048, 0.5433]`. Its exported 10,000-row
prior table exactly matches the backend-generated table. A separate public
adapter/direct-call regression agrees within `rtol=1e-5, atol=1e-6`. These are
implementation checks, not calibration evidence.

The wrapper's default BYM2-prior path writes a second, empty table and crashes the
engine. Supplying the same PC prior through its documented explicit-table input,
using `pyinla.pc_bym.pc_bym_phi` unchanged, permits fitting. No alternative prior
was substituted. The adapter pins the wrapper version, serializes in column-major
order and checks the actual engine input for exact numerical identity on every
fit. It uses a fresh short native work directory to avoid wrapper reuse and
Windows long-path failures. Retained artifacts are copied after fitting.

Reference artifacts are under `analysis/spatial_gaps_2026_10_23/inla_*`.
The optional integration remains experimental. The seeded calibration screen is
specified in `benchmarks/inla_calibration.py`; it includes nulls, planted effects,
unequal cores, donor conditions, measured confounding and transcript misassignment.
All 140 fits complete, but nominal 95% interval coverage ranges from 85–100%
across modeled scenarios and falls to 20% under transcript misassignment.
The last scenario explicitly violates the observation model. Report coverage and
false exclusion with their binomial uncertainty; do not label posterior sign
probabilities as calibrated frequentist tests.

The subsequent 300-fit screen using the inherited log-gamma donor prior confirms undercoverage: 88% for unequal cores,
85% for donor-condition nulls and 92% for adjusted regional confounding. Four-donor
condition fits cover only 18/25 null coefficients. A separate 140-fit sensitivity
screen varies seven published prior settings; no setting resolves the issue on
those seeds. Those seven settings did not change the donor prior family.
These findings keep the integration experimental. See
[extended results](inla-extended-results.json) and
[prior sensitivity](inla-sensitivity-results.json). The subsequent PC-donor
investigation is separate and retains the original results.

## Installation and use

For a compatible Python 3.12 host, install the extra with
`uv pip install ".[spatial-inference]"` from the repository root. Alternatively, prepare an isolated
environment (replace the executable path appropriately on Windows):

```shell
uv venv --python 3.12 .inla-env
uv pip install --python .inla-env/bin/python "pyinla==0.1.11"
.inla-env/bin/python benchmarks/prepare_inla.py --download
```

The last command explicitly downloads the native engine using pyINLA and adds
the compatibility link needed by its Linux installation check for newer portable
archives. It runs only when invoked. Review its
dependency terms and installation documentation before running it. On Windows,
use `.inla-env/Scripts/python.exe`; on Linux/macOS use `.inla-env/bin/python`.
When installing Astrolabe in the same environment, use the `spatial-inference`
extra: its Zarr constraint preserves lazy TIFF reading with pyINLA's older NumPy
requirement. Fits record hashes of both the native executable and its launcher.

```python
result = ab.tl.conditional_expression(
    data, ["CXCL12"], predictor="distance_um", exposure_key="count_exposure",
    covariates=["cell_area"], donor_key="donor_id",
    cell_type_key="cell_type", cell_type="Fibroblast",
    neighbors_key="spatial_neighbors", layer="counts",
    python_executable=".inla-env/Scripts/python.exe",
)
```

Choose the exposure and covariates for the scientific question. For categorical
predictors supply `contrast=(tested, reference)` and select those levels explicitly.
The reported effect is on the natural-log scale, per original predictor unit.
Multi-donor fits require physical library/core labels as well as donor IDs;
graph edges between donors are rejected even when physical labels are absent.
Do not interpret a single-specimen association as a population-level effect.
This interface supports one target coefficient, not random slopes, automatic
interaction tests, spline omnibus tests or expression-selected niche correction.
Zero-count genes have unavailable estimates and `status="zero_counts"`.

The default limit is 1,000 selected cells and 512 MiB estimated prior workspace.
The estimate is not a total process-memory cap. Inputs and dense prior calculations
remain in memory; native INLA requires temporary interchange files, automatically
cleaned. `output_dir` explicitly retains artifacts in a unique subdirectory.
