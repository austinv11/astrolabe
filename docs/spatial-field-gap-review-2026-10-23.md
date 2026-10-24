# Remaining gaps in Astrolabe and the spatial omics field

Assessment date: 2026-10-23. Reviewed the working tree after the approved R1–R8
fixes, public APIs, implementation paths, tests, validation records, and current
primary literature/documentation. The previous full suite passed 988 tests with
one skip. This assessment ran three additional targeted reproductions; it did
not rerun that suite or change package implementations. G1–G10 are approved;
G5's implementation plan is also approved, with specialized backends optional.
G6 explicitly includes tissue microarrays (TMAs). G7a's optional Proseg-backed
transcript displacement/spillover correction is also approved.
G8 requires SpatialData imports only upon explicit use of its conversion feature.
G9 requires disk-backed computation to be explicitly opt-in.
Subsequent G5 constraints require an established Python method; the custom PyMC
proposal is withdrawn. The user approved optional BYM2/INLA through pyINLA, now
implemented with explicit limits and Bayesian outputs. See
[the backend decision](g5-backend-decision.md) and validation record.
Approval collection is complete. Implementations and local validation are recorded;
see [the implementation record](gap-implementation-status.md).

## Assessment

Astrolabe already has substantial breadth: platform readers, lazy images,
segmentation utilities, BANKSY, spatial statistics, niches, regions, pseudobulk,
sample-level neighborhood contrasts, ligand–receptor associations, gradients,
and subcellular analysis. Its most valuable next step is to make analyses
reliable across heterogeneous panels, biological samples, tissue geometries,
and segmentation choices.

The field provides several useful reference points:

- Cohort questions increasingly concern outcome-associated niches and
  within-niche states. QUICHE explicitly models differential spatial enrichment
  across samples, including rare niches; it is a relevant comparison for
  extending Astrolabe's existing cohort tools.
  [QUICHE, Nature Cancer, February 2026](https://www.nature.com/articles/s43018-026-01122-5).
- Spatial dependence and segmentation error can distort differential expression.
  The smiDE study demonstrates that treating neighboring cells as independent
  can inflate significance. Library-restricted permutations alone do not establish
  exchangeability within anatomical compartments.
  [smiDE, Genome Biology, January 2026](https://link.springer.com/article/10.1186/s13059-025-03867-1).
- Segmentation quality needs biological evaluation. A benchmark published on
  23 October 2026 compares 11 methods across 15 datasets and reports no universal
  winner. This supports sensitivity analysis across plausible segmentations.
  [Segmentation benchmark, Nature Methods](https://www.nature.com/articles/s41592-026-03272-x).
- Spatial communication methods perform differently across platforms and tissue
  contexts. A 2026 benchmark compares nine methods using simulations and real
  datasets; an association score should therefore be validated for its intended
  question rather than treated as evidence of signaling by itself.
  [Spatial CCI benchmark, Genome Biology, April 2026](https://link.springer.com/article/10.1186/s13059-026-04063-5).
- SpatialData provides exchange of images, labels, points, shapes, tables, and
  coordinate transforms through Zarr. Astrolabe can offer an optional bridge
  while retaining its deliberately simpler MuData architecture.
  [SpatialData API](https://spatialdata.scverse.org/en/stable/api/SpatialData.html).

These findings motivate the priorities below. They do not establish that any
particular external method should become Astrolabe's default.

## Approval queue

| ID | Priority | Gap | Evidence |
| --- | --- | --- | --- |
| G1 | High | Inconsistent library grouping across inference APIs | Reproduced |
| G2 | High | Incomplete propagation of measured-feature coverage | Reproduced |
| G3 | High | Results remain available after their inputs change | Reproduced |
| G4 | High | Cohort-level niche abundance and state contrasts | Capability gap |
| G5 | High | Conditional spatial inference and confounding controls | Methodological gap |
| G6 | High | Tissue-aware geometry and neighborhood-scale sensitivity | Capability gap |
| G7 | Medium | Segmentation/annotation uncertainty and biological QC | Capability gap |
| G8 | Medium | Optional standards-based interchange | Integration gap |
| G9 | High for large datasets | Bounded-memory statistical execution | Source-identified limitation |
| G10 | High before release | Real-data validation and reproducible delivery | Validation gap |

### G1 — Use one library-grouping contract

**Status:** Approved by the user; implementation is recorded in [the current status](gap-implementation-status.md).

**Evidence:** `_communication.py:71` defaults to `library_id` independently of
the graph. If that column is absent, LR permutations pool all cells, even when
the graph records another grouping. `_icg.py:195` similarly omits a missing
library column from its strata. Global/local autocorrelation and neighborhood
enrichment already use `_permutation_blocks` to resolve graph provenance.

**Reproduction:** Two libraries have distinct cell types and constant expression
levels of 1 and 100. The graph uses `library_key='slide'`. Native LR returns
raw p=0.01 for B→B with default arguments, but p=1.0 when explicitly supplied
`library_key='slide'`, using the same 99 permutations and seed.

**Proposal:** Share graph-derived grouping resolution across native LR and ICG.
Use `library_key='auto'`, preserve explicit `None` for deliberate pooling, reject
missing requested labels, and validate cross-library edges. ICG must additionally
retain receiver-type and optional block restrictions.

**Acceptance:** Default and explicitly resolved grouping agree for custom names;
missing labels fail clearly; intentional pooling remains explicit; worker counts
do not alter seeded results. Initial scope is native LR and ICG, with documentation
and regression checks.

### G2 — Treat unmeasured features as unavailable

**Status:** Approved by the user; implementation is recorded in [the current status](gap-implementation-status.md).

**Evidence:** `io/_concat.py:289` records per-library `measured_features`, and
pseudobulk DE consumes coverage. Native LR checks only global `var_names`
(`_communication.py:82`). Most other expression analyses do not inspect this
metadata. Outer-join structural zeros therefore enter computations as observations.

**Reproduction:** One library measures only L and the other only R. No library
measures the complete interaction, but native LR reports all four cell-type
pairs as tested, with positive scores and finite p-values. These p-values were
not significant; the defect is reporting an estimable interaction at all.

**Proposal:** Introduce a shared coverage resolver. Define eligibility per library,
field, and complex. Reject unsupported partial-coverage analysis by default;
provide explicit common-panel or eligible-library scopes. Report unavailable
interactions as untested with zero valid draws and explain excluded libraries.
Apply the policy to LR, SVGs, gradients, modules, and marker-analysis entry points.

**Acceptance:** Disjoint panels cannot produce tested interactions; missing
subunits remain distinct from measured zeros; fully measured inputs retain their
results; selection and save/load preserve coverage metadata.

### G3 — Track dependencies beyond cell selection

**Status:** Approved by the user; implementation is recorded in [the current status](gap-implementation-status.md).

**Evidence:** R5 handles package-managed cell selection and coordinate transforms.
`pp/_normalize.py` changes expression and `_graph.py` replaces graphs without
invalidating the analyses that used them. Gradient significance has fingerprint
checks, but most stored results have no corresponding freshness check.

**Reproduction:** Moran's I remains stored as -0.058646 after `pp.log1p`, although
recomputation gives 0.008404. Rebuilding the graph with eight instead of three
neighbors retains 0.008404, while recomputation gives -0.030036.

**Proposal:** Record source layer, fields, graph, labels, and relevant geometry
dependencies. Invalidate or mark affected results stale after package-managed
mutations. Check freshness in consumers and provide explicit validation for
external AnnData edits; ordinary dictionary/array mutation cannot be intercepted
reliably without a different object model.

**Acceptance:** Normalization, graph replacement, feature filtering, and label
changes cannot silently reuse affected results. Unrelated layers and input
annotations survive; MuData mirrors follow the same policy.

### G4 — Complete the cohort niche workflow

**Status:** Approved by the user; implementation is recorded in [the current status](gap-implementation-status.md).

**Present:** `niche_composition` and `niches` discover shared composition clusters.
`rank_genes_pseudobulk` supports covariates and paired designs;
`diff_nhood_enrichment` compares sample contact fractions. Users can already
aggregate counts by cell type and niche for some state comparisons.

**Missing:** A supported route from niche discovery to sample-level differential
niche abundance, exposure/denominator choices, within-niche state contrasts,
and reliable mapping onto held-out samples. Pairwise contact enrichment answers
a different question from niche abundance. Current niche clustering refits on
all supplied cells and does not retain a reusable assignment model.

**Proposal:** Start with per-sample niche count/exposure tables and a validated
contrast API supporting donor pairing and covariates. Build state contrasts on
existing pseudobulk machinery. Keep discovery separate from confirmatory testing;
assess held-out-sample assignment and rare-niche stability before interpreting
cohort differences. QUICHE is a useful external comparator, not an equivalence claim.

**Acceptance:** Unequal cell sampling, absent niches, repeated sections, and
imbalanced cohorts have calibrated nulls and interpretable effect sizes. Duplicating
cells within a specimen must not create additional biological replication.

### G5 — Add conditional spatial questions and stronger null validation

**Status:** Scope and implementation plan approved by the user, including optional
specialized backends, subsequently restricted to Python and established methods.
The custom PyMC proposal is withdrawn. Conditional-expression implementation is
deferred until a suitable method is verified and its supported design reviewed.
Approved plan: [G5 design](g5-conditional-spatial-inference-plan.md).

**Present:** SVGs test spatial pattern, LR scores spatial association, ICG tests
contact association with optional blocks, and pseudobulk tests replicate-level
expression differences. These are useful but distinct estimands.

**Missing:** A general supported analysis for expression versus distance or niche
after accounting for cell identity, tissue region, technical covariates, donor,
and residual spatial dependence. Global SVGs can reflect cell-type composition;
within-library shuffling can disrupt genuine anatomical structure unrelated to
the target hypothesis. Subcellular inference also needs a donor-level extension
for population claims across specimens.

**Proposal:** Specify each inferential question and null explicitly. Add
cell-type/region-restricted workflows and evaluate an established conditional
or spatial mixed-model backend. Expand simulation checks for smooth independent
fields, shared region effects, and segmentation-induced contamination. Do not
assume arbitrary residual shuffling or spatial blocks automatically calibrate a test.

**Acceptance:** Null rejection and power are evaluated separately for association
within a tissue and contrasts across biological samples. Report effect sizes and
uncertainty; avoid causal interpretations of LR or expression derivatives.

### G6 — Represent tissue support and test neighborhood sensitivity

**Status:** Approved by the user with explicit TMA support; implementation is recorded in [the current status](gap-implementation-status.md).

**Evidence:** `_graph.py` provides centroid KNN, radius, grid, and Delaunay graphs.
They lack a tissue barrier/mask policy or a polygon-contact graph definition.
`ripley_l` uses a rectangular observation window and pointwise envelopes.
Regions support polygons, but that geometry is not a shared observation-window
contract across statistical methods.

**Proposal:** Represent observed tissue support, including holes and disconnected
pieces. Add explicit contact/distance graph definitions and barrier-aware edge
filtering. Provide mask-aware point-pattern nulls and appropriate edge handling.
Add a neighborhood sensitivity runner across prespecified physical radii or k,
reporting effect and discovery stability; selecting the best radius requires its
own multiple-testing or held-out validation policy.

**TMA requirements:** Treat each core as an explicit spatial analysis unit,
identified within its slide/library, with its own observed tissue mask. Cores on
the same slide may represent unrelated specimens. Do not infer biological
relationships from proximity or assume that each disconnected component is a
different donor; one core may itself contain fragmented tissue.

Spatial neighborhoods, distances to relevant tissue regions, smoothing, and
point-pattern simulations must respect core identity by default. Use per-core
observation windows, excluding holes and inter-core gaps; a slide-wide bounding
rectangle is not observed tissue. Permutations must preserve core-specific
distributions. Report core-level results and require an explicit, justified
design for any pooling or cohort contrast.

Retain separate core, specimen, donor, and condition metadata so repeated cores
from one donor are not counted as independent donors. Resolve core grouping
consistently with G1, preserve masks/labels through selection and persistence,
and carry the hierarchy into G4/G5 cohort designs.

**Acceptance:** Synthetic holes, tissue tears, density gradients, and separate
fragments do not become unexplained spatial effects. Pure rigid transforms and
cell-order changes preserve results. Pointwise and simultaneous envelopes are
identified correctly. Add TMA cases with adjacent unrelated cores, distinct
expression baselines, overlapping local coordinate ranges, fragmented cores,
and repeated cores from one donor. Verify no cross-core graph edges or null
mixing, correct per-core area, and unchanged core-specific results when unrelated
cores are moved or added. Verify that technical cores do not increase the
independent-donor count.

### G7 — Assess segmentation and annotation uncertainty biologically

**Status:** G7 and its G7a extension are approved by the user. G7a includes optional
Proseg integration, per-core TMA isolation, provenance-preserving correction
outputs, diagnostics, and validation. See the approved
[spillover correction design](g7-transcript-spillover-plan.md).
See [the current implementation and validation status](gap-implementation-status.md).

**Present:** Polygon assignment, Cellpose/StarDist, Voronoi and bin2cell utilities,
shape/control QC, unassigned-transcript rates, and IoU/Dice evaluation exist.

**Missing:** Comparative segmentation versions, misassignment/contamination
diagnostics, cell-type purity checks, uncertain labels, and downstream sensitivity
to plausible boundary changes. A good image overlap score does not establish
valid expression or communication results.

**Proposal:** First support importing alternative segmentations/assignments,
recording their provenance, and comparing biological QC plus downstream effects.
Consider Proseg/Baysor outputs through optional adapters; Sopa already integrates
these methods. Preserve assignment and annotation confidence where supplied.
Develop uncertainty propagation only after defining what those probabilities mean.
[Sopa segmentation workflows](https://prism-oncology.github.io/sopa/tutorials/api_usage/).

**Acceptance:** Controlled transcript leakage and split/merge perturbations are
detectable; reports show whether cell-type, DE, niche, and LR conclusions change.

### G8 — Add an optional SpatialData interchange layer

**Status:** Approved by the user with strict lazy-import requirements;
see [the current implementation and validation status](gap-implementation-status.md).

**Evidence:** The public I/O API targets MuData/H5MU with ImageSpec and Arrow
transcripts. No SpatialData conversion API is exposed. The README intentionally
chooses MuData and assumes a shared coordinate frame.

**Proposal:** An optional import/export adapter should preserve table-to-instance
links, IDs, units, transforms, images, polygons, and transcript coordinates. Require
an explicit target frame when conversion is ambiguous. Retain lazy image access
and document which elements can be converted exactly.

**Import requirement:** Import SpatialData and its adapter-only heavy dependencies
inside the explicitly invoked conversion operation. Neither `import astrolabe`,
subpackage imports, public API exports, runtime annotations, nor ordinary I/O or
analysis paths may import SpatialData. Avoid import-based availability probes;
report a missing optional dependency only when the conversion feature is called.
Keep the core installation independent of SpatialData.

**Acceptance:** Round-trip real cropped data through both ecosystems, compare
coordinates and instance linkage numerically, and verify conversion does not load
full-resolution imagery unexpectedly. Include TMA core identity and masks in
round-trip checks. In fresh subprocesses, verify that package/subpackage imports
and representative unrelated operations do not attempt to import SpatialData,
including when it is installed. Verify that explicit conversion triggers its
lazy import and gives an actionable error when it is unavailable.

### G9 — Establish memory budgets for large spatial analyses

**Status:** Approved by the user with disk-backed computation optional;
see [the current implementation and validation status](gap-implementation-status.md).

**Evidence:** Local Moran/BV/Getis retain simulations (`_spatial_stats.py:655`,
`:800`, `:895`). BANKSY densifies expression (`pp/_banksy.py:278`); spatial gene
modules densify and standardize it (`tl/_gene_modules.py:392`). Existing sparse
graphs, lazy image access, and streamed native LR are useful foundations, but
do not bound memory across the analysis stack.

**Scale calculation:** One float64 matrix for one million cells and 999 draws
occupies approximately 7.4 GiB, before temporary arrays. This is an allocation
estimate, not a measured Astrolabe production benchmark.

**Proposal:** Add memory estimates/limits, chunked local-null tail accumulation,
bounded-memory module computation, and supported backed execution where feasible.
Do not claim arbitrary chunking is equivalent: preserve the original conditional
null and deterministic random-stream policy.

**Execution requirement:** In-memory computation remains the default. Streaming
and chunking should bound working memory without requiring disk-backed arrays.
Enable disk-backed computation only through an explicit user option, with a
configurable working directory and documented scratch-file lifecycle. Do not
silently spill computation to disk when a memory limit is exceeded; report the
limit and available options. This requirement concerns computation backing,
not existing explicit file loading or saving.

**Acceptance:** Compare numerical results against small in-memory references and
measure process peak RSS/runtime at 100k and one million cells with specified
genes, edges, workers, and permutation counts. Verify equivalent results across
in-memory and explicitly enabled disk-backed modes, confirm default execution
does not create disk-backed computation scratch files, and test cleanup after
both successful and failed disk-backed runs.

### G10 — Validate complete real-data workflows before release

**Status:** Approved by the user. Implementation and validation are in progress.

**Present:** Extensive regressions, independent numerical checks, synthetic
calibration, benchmark notebooks, and a configured Windows/Linux CI matrix.
`docs/validation.md` explicitly limits claims about full vendor datasets and GPUs.

**Missing:** Maintained end-to-end validation across real vendor/schema versions,
multi-donor cohorts and different panels; clean wheel installation checks;
representative process-memory measurements; and concise API documentation that
consistently identifies assumptions. For example, `tl/__init__.py` still describes
CLQ as a gene-pair cell/pixel method, although the public implementation uses
categorical nearest-neighbor colocation.

**Proposal:** Add small licensed real-data fixtures plus opt-in full-dataset runs,
version/checksum manifests, independent reference outputs, clean wheel smoke
tests, and a data-contract/analysis-assumptions guide. Record calibration results
against a source revision. Review core notebook dependencies and optional backend
compatibility as part of install testing.

**Acceptance:** A reproducible run goes from vendor output through cohort analysis
and save/load, with explicit dataset sizes, dependency versions, resource usage,
expected invariants, and known exclusions.

## Deliberate scope choices

Native 3D analysis, nonlinear registration, spot deconvolution, reference-based
cell annotation, and foundation-model adapters are additional opportunities.
They should follow the intended user workflows. Spatial graph construction is
currently explicitly 2D; image z-stacks do not imply volumetric statistical support.
The README's pre-aligned, segmented-cell focus makes a full registration or
deconvolution system a larger product decision.

Transferable spatial representations deserve an adapter boundary and held-out
validation. Novae demonstrates this direction; it does not make training a new
foundation model a prerequisite for Astrolabe.
[Novae, Nature Methods, December 2025](https://pubmed.ncbi.nlm.nih.gov/41372623/).

My recommended order is G1–G3, then G10 and G9 alongside G4–G6. G7 and G8 can
build on those data contracts. The most promising positioning is a dependable,
Scanpy-like spatial analysis library with transparent assumptions and optional
connections to specialized tools.

## Reproduction record

The additional checks use small seeded synthetic objects and the current local
environment. They write only review artifacts:

```powershell
.\.venv\Scripts\python.exe analysis/spatial_gaps_2026_10_23/reproduce.py
```

The script and `evidence.json` are retained in that ignored analysis directory.
G1–G3 are demonstrated behaviors. G4–G10 are proposals based on source inspection,
documented validation limits, and the literature; they are not newly demonstrated
type-I-error failures. No implementation proposal is approved by this report.
