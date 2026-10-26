# Testing and validation

Run `uv run pytest -q` from the repository root. Tests use fixed seeds and small
fixtures; optional backend checks skip when their dependencies are unavailable.

## Statistical checks

`tests/test_spatial_stats.py` checks spatial statistics and permutation tails.
`tests/test_statistical_stress.py` covers unequal libraries, sparse panels and
screened hypotheses. `tests/test_niche_cohort.py` checks donor-level contrasts;
`tests/test_subcellular_cohort.py` compares results against SciPy references.
These checks validate specified implementations and designs, not arbitrary
tissues or causal interpretations.

## Optional backends

Install the relevant extras to exercise cohort, communication and SpatialData
integrations. SpatialData tests verify lazy imports and coordinate/count
round-trips. Native INLA checks in `tests/test_conditional.py` require
`ASTROLABE_INLA_PYTHON` pointing to a prepared interpreter; see
[backend setup](analysis.md#conditional-expression-with-bym2inla).

The BYM2/INLA integration remains experimental. Prior calibration found
undercoverage, particularly for condition comparisons with few donors and high
donor variability. Agreement with direct backend fits verifies the adapter,
not general 95% interval coverage. A large-input handoff test checks removal of
allocation cutoffs without claiming native-backend scalability.

## Scope

Vendor fixtures check particular exported formats, not every vendor version.
Synthetic tests do not certify arbitrary tissue geometries, GPU drivers or
optional dependency versions. The benchmark notebooks compare implementations
with their reference packages; consult each notebook for comparison limits.
