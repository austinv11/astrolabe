# G7a: transcript displacement and spillover correction

Research reviewed 2026-10-23. G7's segmentation/annotation assessment is approved.
The user requested consideration of transcript-coordinate errors and spillover
into neighboring cells. The user approved this additional correction feature:
optional Proseg integration, per-core TMA isolation, provenance-preserving outputs,
correction diagnostics, and validation. The Proseg 3.2 adapter, input accounting,
per-core execution, provenance and descriptive QC are implemented. Reduced and
full-schedule clean/displaced control runs did not meet the correction-efficacy gate; this
remains an explicit experimental integration. See [validation results](spatial-validation.md)
and [usage contracts](spatial-workflows.md).

## Recommendation

Add an optional Proseg-backed workflow for transcript displacement and cell
reassignment, preserving observed data and exposing inferred outputs separately.
Start with preparing inputs, running a supported backend version, importing its
results, and comparing corrected versus original analyses. Use the published
backend instead of introducing an unvalidated native correction algorithm.

This is a probabilistic reconstruction. The true original position of an
individual molecule is generally not identifiable from an observed coordinate,
gene identity, and neighboring expression alone. Distinguish measured coordinates,
model-inferred positions, cell assignments, and corrected expression estimates.

## Different mechanisms need different treatments

| Mechanism | Appropriate treatment |
| --- | --- |
| Systematic channel/image registration error | Correct a transform using independent image/landmark evidence before molecular assignment. |
| Boundary errors or projection overlap | Refine assignment/segmentation, retaining z where available. |
| Local displacement or RNA leakage | Model local displacement and assignment uncertainty. |
| Background or incorrect gene detection | Model an unassigned/background component and apply platform-appropriate detection QC. |

An unexpected marker in a neighboring cell is not by itself evidence of an error.
True transitional states, shared programs, and subcellular localization are
explicit preservation targets. Do not interpret a transcript detection quality
score as a positional standard deviation without platform-specific calibration.

## Research basis and alternatives

**Proseg — primary candidate.** The peer-reviewed method jointly infers cell
geometry and molecular assignment. Its probabilistic treatment includes transcript
repositioning, making it directly relevant to displaced detections rather than
only mixed count profiles.
[Cell simulation as cell segmentation, Nature Methods, 2025](https://www.nature.com/articles/s41592-025-02697-0).

The current repository documents diffusion controls and transcript outputs with
revised positions and assignment probabilities. It also supports count matrices,
2D polygons and z-layer polygons; run unrelated tissue pieces separately. These
are backend capabilities to verify against the pinned version, not promises that
every output is available identically across versions.

Verified for the implemented 3.2 adapter: the transcript export has no assignment
probability field, and z is detrended, clipped and normalized. Preserve physical
input z and store normalized backend z separately. Only xy displacement is
reported in microns; no missing probability or inverse z transform is invented.
[Proseg usage and output documentation](https://github.com/dcjones/proseg#usage).

**MisTIC — assignment-focused comparator.** This December 2025 preprint estimates
transcript reassignment probabilities using spatial and expression features, with
cell-group supervision. It is relevant when retaining an existing segmentation
is essential. Annotation dependence and preservation of novel states need testing;
this assessment does not treat the preprint as an established universal default.
[MisTIC preprint](https://www.biorxiv.org/content/10.64898/2025.12.11.693759v1).

**FastReseg — segmentation-refinement comparator.** Its workflow identifies
poor-fitting transcript groups and can trim, split, or merge assignments using
local spatial and expression information. It addresses related segmentation
errors and is useful as a published comparison.
[FastReseg, Scientific Reports, 2025](https://www.nature.com/articles/s41598-025-08733-5).

**ResolVI — expression correction comparator.** The September 2026 paper models
misassignment and background downstream of segmentation to infer expression and
representations. It does not provide independent measurements of recovered
molecule positions. Treat its outputs as a distinct count/model analysis path.
[ResolVI, Nature Methods, 2026](https://www.nature.com/articles/s41592-026-03212-9).

**SPLIT — reference-guided expression purification.** The April 2026 study uses
single-nucleus references and RCTD decompositions to separate mixed expression.
It is a relevant comparison when a suitable reference exists, but does not locate
the original source coordinate of each molecule. Reference mismatch must be
evaluated, especially for new or disease-specific states.
[SPLIT, Nature Methods, 2026](https://www.nature.com/articles/s41592-026-03089-8).

Method-specific benchmark results are not a universal ranking. Validate the
selected workflow on Astrolabe's intended platforms, panels, and biological
questions; do not automatically stack multiple correction methods.

## Proposed workflow

1. **Validate and partition inputs.** Require transcript IDs, genes, micron
   coordinates, and suitable initial segmentation/nuclear assignments. Preserve
   z, detection QC and unassigned molecules when available. Check frame alignment,
   panel coverage, ID uniqueness, and coordinate units. A filtered transcript table
   is explicitly recorded and cannot masquerade as complete input.
2. **Isolate TMA cores.** Fit each unrelated core independently by default, within
   its slide/library identity. No cross-core candidates or borrowed background
   pools. A fragmented core can remain one analysis unit; a disconnected component
   is not automatically a donor. Keep specimen and donor mappings.
3. **Run the optional backend.** Record executable version, input fingerprints,
   parameters, and reproducibility controls supported by that version. Benchmark
   diffusion-enabled and disabled fits. Check installation/execution on supported
   environments before advertising platform support. No runtime auto-installation.
4. **Import an alternative reconstruction.** Keep input coordinates, assignments,
   counts, and polygons. Store inferred positions, displacement distances,
   model assignment probabilities, and background/unassigned status where the
   backend supplies them. Preserve transcript identity for auditing.
5. **Compare outcomes.** Report reassignment fractions, count changes, displacement
   distributions, and affected cell/gene/core summaries, together with downstream
   changes. Default analysis remains tied to the user's explicit selected input.

Proseg may revise segmentation, so this is not a promise to correct coordinates
while holding every boundary fixed. A fixed-segmentation requirement would favor
an assignment-specific backend and needs a separate validated implementation.

## Storage and downstream contracts

- Store corrected cells in a separate AnnData/MuData modality when cell identities
  or geometry change. Use alternative layers only when their observation/feature
  axes genuinely match; record mappings rather than assuming row correspondence.
- Keep integer assignment-derived counts distinct from posterior expected counts
  or normalized expression. Fractional expectations must not be rounded silently
  or supplied as raw observed counts to existing integer-count DE APIs.
- A backend assignment probability is conditional on its model. Do not relabel a
  single probability as a complete posterior over all candidate cells. Preserve
  uncertainty summaries/samples if supplied and validate their calibration.
- Model-inferred coordinates require explicit selection for subcellular analysis.
  The reconstruction's spatial assumptions could alter the localization being
  tested. Raw-versus-reconstructed sensitivity analysis is required for such claims.
- Preserve 3D information even when a 2D visualization or projected polygon is
  requested. A 3D backend does not make Astrolabe's current 2D statistical functions
  volumetric methods.
- Use G2's feature-coverage contract and G3's dependency tracking. Recompute graph,
  niche, LR, gradient and subcellular outputs for a changed reconstruction.

## Validation before supporting correction

Use controlled simulations with known origin cells and true positions. Vary
coordinate jitter, local leakage, z-overlap, segmentation errors, background,
target-panel sparsity, rare cells, and close same-type neighbors. Include zero-error
controls, genuine shared expression, boundary-associated activation, and real
subcellular patterns. Tune on one set and evaluate held-out scenarios.

Measure assignment accuracy/precision, the fraction of originally correct
molecules incorrectly changed, calibration of supplied probabilities, and positional
error where truth exists. Account for all input molecules as assigned, background,
unassigned, or explicitly filtered; distinguish hard assignments from expected
count output. Report per-core performance and peak process memory.

On real data, use independent annotations or stains and held-out marker/reference
information where feasible. Better agreement with the same labels used for fitting
is not independent validation. Compare original assignment, Proseg with and without
diffusion, and an appropriate alternative; test whether true DE/localization signals
survive and spurious neighborhood/LR signals decrease.

G5 must evaluate downstream uncertainty introduced by correction. Corrected
expression and coordinates remain inferred quantities; using them in ordinary
tests does not automatically propagate assignment uncertainty.

## Approved initial scope

Implement an optional Proseg adapter, per-core execution, provenance-preserving
import, correction diagnostics, and independent validation. Treat MisTIC,
FastReseg, ResolVI, and SPLIT as research comparators; implementing all of them
is not part of this initial proposal. Automatic registration correction and a
new native probabilistic segmentation algorithm are also outside this scope.
