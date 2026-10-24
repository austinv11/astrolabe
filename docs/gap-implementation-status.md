# Approved gap implementation record

All G1–G10 scopes and G7a are approved, subject to subsequent user constraints.
G5 uses the approved optional BYM2/INLA backend through Python; the custom
PyMC proposal is withdrawn. Its conditional-expression adapter is implemented
with bounded dense-prior work and Bayesian outputs. This record
separates completed implementation from validation still required.

| Item | Status | Validation |
| --- | --- | --- |
| G1: shared inference grouping | Implemented | Custom-library and cross-edge regressions pass |
| G2: measured-feature eligibility | Common/eligible panels, raw/masked markers, scoring and custom-library pseudobulk implemented | Partial/disjoint panels and raw-axis regressions pass |
| G3: dependency freshness | Tracking, invalidation, transitive analysis/plot consumer checks implemented | External edits, sparse inputs, Awkward buffers and clean-wheel round-trip checks pass |
| G4: cohort niche workflow | Standard statsmodels OLS/HC3 contrasts, frozen mapping and donor/condition state pseudobulk implemented | Five seeded scenarios x 1,000 replicates pass stated gates; numerical reference and donor-duplication checks pass |
| G5: conditional inference | Optional published BYM2/INLA adapter implemented; custom PyMC withdrawn | Windows/Linux analytic and direct checks pass. Original 140, fixed-design 200, extended 300 and prior-sensitivity 140 fits complete. Extended coverage is 85–92%; defaults remain unchanged and inference remains experimental. Separate subcellular contrasts pass SciPy references |
| G6: tissue support and TMAs | Masks, core/contact graphs, crop windows, distances/bearings, mask Ripley and scale sensitivity implemented | Geometry/unit and Holm-family checks pass. Single-domain analyses reject pooled cores and require separate per-core calls |
| G7/G7a: segmentation assessment and Proseg | Per-core runner/import, native hard/expected count exports, provenance, failure audits and descriptive QC implemented | Native count alignment and real Xenium round-trip pass. Original/held-out controls fail correction-efficacy gate. Six independent MERFISH boundary runs show no consistent diffusion benefit; retain experimental opt-in status |
| G8: lazy optional SpatialData interchange | Explicit conversion implemented | Four isolated-environment tests pass, including real cropped Xenium, transformed masks and import isolation; image pixels remain external references |
| G9: memory budgets and optional disk backing | Bounded local nulls and module allocation guards implemented | 100k/1m-cell process-RSS runs complete with exact disk/in-memory equivalence; scratch cleanup tested |
| G10: real-data and distribution validation | Xenium fixture plus real MERSCOPE/CosMx validation, checksum manifests, workflow guide and explicit archive contents | Full suites including native INLA: Windows 1,080 passed/4 skipped; Linux 1,022 passed/14 skipped. MERSCOPE/CosMx raw alignment and round-trips pass. Hosted CI status is recorded in spatial-validation.md |

Constraints: no migration/deprecation notes; preserve existing user changes;
no automatic SpatialData imports; in-memory computation by default; separate
unrelated TMA cores; distinguish cores/specimens/donors; preserve observed
coordinates and counts when importing probabilistic reconstructions.
Use established statistical methods throughout the remaining work. Document
primary references, supported designs, and agreement with reference implementations;
do not treat earlier scope approvals as authorization to invent new procedures.

See [spatial workflows](spatial-workflows.md) for APIs and
[spatial validation](spatial-validation.md) for numerical results and exclusions.
Outstanding scientific evidence is explicit: broad G5 calibration is not established,
no demonstrated Proseg correction efficacy, and vendor crop checks do not validate
every export version or dataset. General annotation-uncertainty
propagation is not implemented without validated probability semantics.
