# G5: conditional spatial inference implementation plan

Status: scope and optional BYM2/INLA backend approved. `tl.conditional_expression`
implements the published graph model through pyINLA without R. The custom PyMC
proposal is withdrawn. The approved outputs are Bayesian credible intervals and
posterior sign probabilities, without p-values or BH correction. See
[the backend specification](g5-inla-proposal.md) for supported designs and limits.
Other approved gap work remains separate. This plan concerns G5 only.

## Questions and units of inference

Support two explicit questions:

1. Within a cell type, is expression associated with a prespecified distance,
   region, or niche after accounting for specified covariates and spatial
   dependence? A single specimen supports a specimen-specific association.
2. Does that association or subcellular localization differ across conditions
   in a biological cohort? Donors provide population-level replication;
   sections and cells remain nested observations.

Record the target predictor, contrast, cell type, analysis population, nuisance
covariates, library, specimen, donor, expression layer, and inferential unit.
Condition effects and distance effects are separate contrasts. Pairing and
multiple sections must be represented explicitly.

## Statistical design

For cell-level counts, the candidate model is a negative-binomial regression
with an explicit exposure/normalization offset, target and nuisance effects,
and a spatial dependence component within each physical library. Repeated
specimens require an appropriate donor structure. This describes the model
requirements, implemented using the approved BYM2 graph model and INLA engine.

Distance is continuous and measured in microns. Begin with a prespecified linear
effect; offer fixed spline bases only after validating their omnibus contrasts.
For categorical regions/niches, define an explicit reference and contrast.
Do not select a distance transform or neighborhood scale by the smallest p-value.

Use the existing PyDESeq2 pathway for compatible sample-level categorical
pseudobulk contrasts. Do not treat binned distance rows as independent biological
replicates or describe pseudobulk as a cell-level spatial covariance model.
G4 provides the niche/cohort tables; G5 adds the conditional spatial question.

Check rank, residual information, predictor variation, and covariate overlap.
Reject an unidentifiable requested contrast: adding a spatial random effect
cannot separate perfectly confounded region, condition, and location effects.
The user chooses covariates for the scientific question; do not automatically
adjust for every spatial annotation, which could remove the effect of interest.

## Backend evaluation and decision point

Compare established implementations before choosing a backend. Assess supported
responses and offsets, spatial covariance, repeated samples, contrasts, convergence
diagnostics, Windows/Linux installation, licensing, and memory requirements.
Verify adapter output against a direct call to the same backend.

Use a documented established statistical procedure, with a primary method
reference and matching reference implementation. A generic modeling framework
does not supply that justification. Do not assemble a custom model or testing
procedure to satisfy unsupported requirements. If an established method supports
only a narrower question, present that scope explicitly before implementation.

smiDE is a relevant methodological comparator: its paper describes R-based spatial
models and reports a non-commercial package license. These properties require an
explicit dependency/distribution decision before adopting it in this MIT-licensed
Python library. It is not selected as a bundled/default dependency by this plan.
[smiDE paper and code availability](https://link.springer.com/article/10.1186/s13059-025-03867-1).

Present the selected backend, installation implications, supported designs, and
small-reference results for review before implementing its public adapter. A
generic donor random intercept is not a substitute for within-tissue spatial
covariance. No claim of spatially calibrated inference follows from residualizing
expression and applying unrestricted permutations.

## Data and API integration

Proposed public entry points are separate conditional expression and subcellular
cohort functions; final names follow backend selection. Parameters must expose
the target predictor/contrast, covariates, cell-type restriction, layer, library,
sample, donor, and backend rather than infer biological replication from filenames.

Use G1's library resolution, G2's measured-feature eligibility, and G3's source
dependency tracking. Preserve AnnData/MuData handling and parent result mirrors.
Keep raw counts intact and make any model-derived expression representations
explicit. Distances reuse the existing region geometry and micron conventions.

Return natural-log effects, posterior SDs and credible intervals, posterior sign
probabilities, cell and independent-donor counts, eligibility status, and the
actual design/backend version. The approved Bayesian backend does not supply
frequentist p-values or BH-adjusted values. Report zero-count genes separately.

Niches or regions defined from the same expression being tested require an
exploratory interpretation or an independent/held-out definition. A fitted spatial
covariance does not by itself correct this selection problem.

## Subcellular cohort extension

Implemented separately as `tl.diff_subcellular`: SciPy donor-label permutation
tests and BCa donor bootstrap intervals on prespecified localization summaries.
Direct-reference and donor-duplication tests pass. This does not implement the
cell-level conditional-expression model.

Preserve the existing per-cell compartment information and compute interpretable
localization effect summaries per donor and cell type. Technical sections must
not multiply the donor count. Begin with explicit paired/unpaired donor-level
contrasts; use donor-level bootstrap or valid donor-label randomization under
its stated assumptions. Do not pool transcript-level p-values to claim cohort
replication. Covariate-adjusted or more complex repeated-measure designs require
separate validated model support.

## Validation gates

- Independent backend agreement on small problems, including effects, standard
  errors, contrasts, missing features, and failed fits.
- Seeded null experiments: smooth spatial fields unrelated to the target,
  regional confounding, unequal densities/cell counts, repeated sections,
  missing panels, donor imbalance, and transcript misassignment.
- Planted distance, region, and condition-interaction effects spanning plausible
  magnitudes. Evaluate power together with false-positive rates and interval
  coverage; a procedure that never rejects is not sufficient.
- Separate single-specimen conditional claims from cohort population claims.
  Extra cells or duplicated sections cannot count as extra independent donors.
- Predeclare simulation scenarios and nominal levels. Report uncertainty on
  rejection/coverage estimates; do not infer success merely from a nonsignificant
  excess-rejection test. Include held-out scenarios after tuning.
- Record runtime and peak process memory. Provide explicit failure/unsupported
  statuses instead of silently falling back to an independence model.

Segmentation contamination may bias estimates even when covariance is modeled.
Sensitivity tests quantify this limitation; passing tests does not establish
causality or certify arbitrary tissue and assignment errors.

## Implementation order

1. Review this plan with the user — completed and approved.
2. Specify designs and benchmark candidate backends using independent examples.
3. Review the concrete backend/API choice and its dependency implications.
4. Implement the validated conditional-expression adapter and input checks.
5. Add donor-level subcellular contrasts.
6. Run statistical calibration, regression tests, integration checks, and document
   supported questions and assumptions directly.

The plan and backend/API reviews are complete. Calibration evidence and remaining
scientific limitations are recorded separately from implementation status.
