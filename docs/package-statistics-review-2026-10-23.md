# Package and statistical implementation review

Review date: 2026-10-23. Source baseline: `ef2a130`.
**R1–R8 approved, implemented, and validated.**

## Scope and evidence

Reviewed package configuration, public API wiring, AnnData/MuData handling,
concatenation, persistence, spatial-result invalidation, shared inference helpers,
global/local spatial statistics, neighborhood and cohort tests, ICG, ligand/receptor
inference, gradients, and subcellular inference. This is a targeted source and
synthetic-data review, not certification of every algorithm or supported dataset.

Reproductions and their output are retained locally under
`analysis/package_review_2026_10_23/` (an ignored analysis directory):

```powershell
.\.venv\Scripts\python.exe analysis/package_review_2026_10_23/reproduce.py
```

The script constructs synthetic objects and writes `evidence.json`. It does not
alter package source. Baseline validation covered all 922 collected tests:
**921 passed, one skipped**, across the full run and a retry of environment-blocked
tests. The initial run passed 844 tests; 77 failed during fixture setup solely
because the sandbox denied access to pytest's temporary directory. All 77 passed
when rerun with the required access. There were no assertion failures in either
run. Commands and full output are retained in `pytest.log`, `failed-tests.txt`,
and `pytest-retry.log`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --tb=short
.\.venv\Scripts\python.exe -m pytest -q --tb=short '@analysis/package_review_2026_10_23/failed-tests.txt'
```

The environment used Python 3.12.12, NumPy 2.4.6, SciPy 1.17.1, pandas 2.3.3,
ESDA 2.9.0, AnnData 0.12.16, and MuData 0.3.8. A passing baseline does not resolve
the independently reproduced defects below.

The newer inference paths have useful safeguards: explicit permutation tails and
valid-draw counts, pooled correction for local tests, library-specific local nulls,
retained screened hypotheses for native ligand/receptor correction, and biological
sample-level cohort inference. The findings below identify gaps around those
mechanisms. Existing tests do not cover the reproduced cases adequately.

## Approved changes

| ID | Priority | Proposed change | Status |
| --- | --- | --- | --- |
| R1 | High | Keep grouped gradient nulls inside the selected group | Implemented |
| R2 | High | Make global permutation p-values valid for undefined statistics and ties | Implemented |
| R3 | High | Respect library boundaries in global autocorrelation inference | Implemented |
| R4 | High | Respect library boundaries in neighborhood enrichment | Implemented |
| R5 | High | Invalidate derived results consistently after cell selection | Implemented |
| R6 | Medium | Treat absent receiver types as unavailable in cohort contrasts | Implemented |
| R7 | High | Preserve spatial metadata when concatenating aliased libraries | Implemented |
| R8 | Medium | Exclude self by identity in colocation nearest-neighbor queries | Implemented |

Each item was approved separately before implementation. All eight include
source changes, regression checks, and current-behavior documentation. The
findings and line references below describe the reviewed source baseline.

The package has not been released. Per the user's instruction, do not add
migration notes, deprecation notices, or backward-compatibility guidance. Document
the resulting behavior directly.

## R1 — Grouped gradient nulls include excluded cells

**Location:** `astrolabe/tl/_gradient.py:1213`, `:1262`, `:1270`–`:1279`.

The graph and observed statistic are restricted to a group, but surrogate
calibration, sorted expression values, noise generation, and CSR shuffling still
use all cells. This changes the reference distribution when an excluded cell
type changes expression.

**Reproduction:** A selected group contains 49 cells with expression fixed at 1;
25 pass the geometry reliability mask. Changing only the other group's expression
from 1 to 100 leaves observed magnitude at zero but changes its CSR-null p-value
from **1.0 to 0.02**. The default smooth-null p-value also changes, from **1.0 to
0.64**, despite identical selected-group data.

**Proposed implementation:** Extract the selected group's expression, coordinates,
and induced graph before calibrating or generating either null. Preserve the
fitted smoothing and weighting; apply the reliability mask within that group.
Use only the selected group when computing the surrogate value distribution.

**Validation:** Changing excluded cells must leave the selected group's statistic,
null calibration, and p-values unchanged for both nulls. Compare with an independent
fit on the isolated group; check seeds, worker counts, and constant expression.

## R2 — Undefined global statistics receive small p-values

**Location:** `astrolabe/tl/_spatial_stats.py:522`–`:531`, `:696`, `:1686`–`:1702`.

Global `moran`, `moran_bv`, and `geary_c` expose ESDA's legacy `p_sim` directly.
For a constant gene, their statistic is NaN, but all three return **p_sim=0.005**
with 199 permutations. A consumer filtering p-values alone can report a discovery
for an undefined statistic. ESDA's legacy folded-tail arithmetic also differs
from this package's newer inclusive, equal-tail calculation.

**Proposed implementation:** Validate finite inputs and estimable statistics;
report unavailable inference as NaN with an explicit status. Compute empirical
p-values through the shared helper with an explicit alternative, defaulting to
two-sided inference; retain a clearly named smaller-tail diagnostic if useful.
Expose valid-draw counts and resolution metadata consistently.

The shared helper's inclusive tails and plus-one correction follow the conventions
documented by [SciPy's permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

**Validation:** Constant/all-zero fields, NaN inputs, edgeless graphs, tied null
distributions, zero permutations where supported, and direct tail-count references.
Defined statistics with an entirely tied null must not produce a small p-value.

## R3 — Global autocorrelation ignores library-specific exchangeability

**Location:** `astrolabe/tl/_spatial_stats.py:522`, `:696`, `:1686`.

These global wrappers pass the combined expression vector to ESDA, whose
permutations exchange values across libraries. Isolating the graph's edges does
not isolate the permutation null. This differs from the existing local and SVG
implementations, which respect the graph's library grouping.

**Reproduction:** Two disconnected libraries each have a constant expression
baseline, 1 and 100. Global Moran and Geary return **p_sim=0.005**; their analytic
p-values are below 1e-26. The library-aware Moran SVG test on exactly the same
input returns **p=1.0**. The global wrappers attribute differences between slides
to spatial evidence under an inappropriate pooled null for independent slides.

**Proposed implementation:** Reuse the graph's library grouping and validate
missing labels and cross-library edges. Shuffle expression only within libraries
(the response vector for bivariate Moran). Use the conditional simulation mean
and variance for diagnostics. Reject pooled analytic inference for multiple
independent libraries; retain an explicit pooled mode for deliberately exchangeable
data. Preserve the current global statistic while changing its reference null.

**Validation:** Library-baseline shifts alone must not generate discoveries;
within-library planted structure must remain detectable. Check single-library
compatibility, custom library keys, and the explicit pooled mode.

## R4 — Neighborhood enrichment shuffles labels across libraries

**Location:** `astrolabe/tl/_neighborhood.py:151`–`:155`.

`nhood_enrichment` shuffles every cell label together even when the graph was
constructed separately per library. It therefore changes each slide's type
composition in the null.

**Reproduction:** One library contains only type A, another only type B; all edges
stay within libraries. The function reports same-type z-scores **9.52 and 8.50**
and cross-type z-scores **−11.79**. Within-library label permutations cannot change
this dataset, so the enrichment is driven by composition differences between
libraries rather than an identifiable within-library arrangement.

**Proposed implementation:** Honor the graph's library grouping, with an explicit
override, and shuffle labels within each library. Validate exchangeability labels
and cross-library edges. Record the grouping and handle zero-variance nulls as
unavailable inference rather than silently presenting them as ordinary z-scores.
[Squidpy's implementation](https://squidpy.readthedocs.io/en/stable/_modules/squidpy/gr/_nhood.html)
also provides library-restricted label shuffling.

**Validation:** Heterogeneous library compositions, pure-type libraries, missing
labels, custom grouping keys, and identical results across worker counts.

## R5 — Cell selection leaves stale statistical results

**Location:** `astrolabe/pp/_qc.py:939`, `:1204`;
`astrolabe/_utils.py:661`–`:663`; `astrolabe/pp/_transform.py:217`.

QC filtering subsets cells without invalidating previously computed statistics.
Cropping invokes invalidation, but the module-based result registry omits
`_cohort`, leaving differential neighborhood results behind.

**Reproductions:** Filtering a 20-cell object to 19 cells retains a 20-row local
Moran table, including the removed cell. Cropping a 32-cell cohort to 16 cells
removes its spatial graph but retains the old `diff_nhood_enrichment` results.
Those values describe the original inputs and can be mistaken for current results.

**Proposed implementation:** Centralize invalidation for package-managed cell
selection, including both in-place and returned-copy QC paths. Register cohort
spatial outputs explicitly and clear affected MuData parent mirrors. Track output
ownership/dependencies so custom result names are covered and original input
annotations are retained. Keep this initial fix scoped to cell selection and
existing geometry invalidation; broader automatic expression-change tracking is
separate work.

**Validation:** Filter/crop chains after local tests, SVG, native communication,
and cohort contrasts, including custom keys and MuData. Assert removed results
cannot be consumed as current and input annotations survive.

## R6 — An absent receiver type is scored as observed zero

**Location:** `astrolabe/tl/_cohort.py:140`–`:143`.

Sample scoring checks that the sender has connected cells, but does not check
that the receiver exists. An absent receiver produces zero contacts and zero
expectation, yielding an artificial valid score of zero. This contradicts the
documented rule that missing types produce untestable comparisons.

**Reproduction:** A declared but completely absent receiver type returns effect
**0**, confidence interval **[0, 0]**, **p=1**, and two replicates per condition.
The existing missing-type test only checks the sender direction. Partial absence
can also distort contrasts and inflate their usable-replicate counts.

**Proposed implementation:** Require both types to be represented in each sample
before scoring, with at least two cells for a same-type comparison. Retain NaN
for unavailable pairs, count only eligible biological units, and label insufficient
replication explicitly in paired and unpaired outputs.

**Validation:** Absent sender, absent receiver, singleton self-pairs, partial
absence by condition, incomplete usable pairs, and unavailable p-value diagnostics.

## R7 — Explicit library aliases lose spatial metadata

**Location:** `astrolabe/io/_concat.py:222`–`:230`.

Dictionary keys rename observation library IDs and transcript prefixes, but
`uns['spatial']` is merged under the original IDs. If both inputs have the same
original ID, the second metadata entry overwrites the first despite distinct
explicit aliases.

**Reproduction:** Concatenating `{'first': sample1, 'second': sample2}` when both
inputs use the original ID `original` yields observation libraries `first` and
`second`, but spatial metadata contains only `original` with sample2's marker.
This can lose image registration and break image lookup by the advertised IDs.

**Proposed implementation:** Validate the single-library contract on dictionary
inputs too, and consistently remap spatial metadata to each explicit alias in the
modality and parent stores. Preserve each input descriptor and align transcript
library labels with the renamed observations. Reject ambiguous input metadata.

**Validation:** Duplicate original IDs with unique aliases, distinct image
descriptors, transcript alignment, unchanged input objects, and save/load round trips.

## R8 — Coincident coordinates can become self-neighbors in CLQ

**Location:** `astrolabe/tl/_spatial_stats.py:1866`–`:1867`.

`colocation_quotient` discards the first KDTree result assuming it is the query
cell. For coincident centroids, another cell can be returned first, leaving the
query itself in its neighborhood. The calculation's finite-population denominator
assumes self-exclusion, so this changes the estimand.

**Reproduction:** Four cells, including coincident A/B centroids, with `k=1`
produce **CLQ(A→B)=0.75**. Excluding each query's own index instead gives **1.5**.
Both the observed statistic and its permutation distribution use the faulty graph.

**Proposed implementation:** Remove each query index by identity, then take its
first k remaining neighbors. Keep distinct cells at zero distance and document
tie handling consistently with `spatial_neighbors`.

**Validation:** Coincident cells with different labels, multiple duplicate
coordinates, `k=1`, `k=n−1`, and ordinary coordinates without distance ties.

## Implementation validation

`tests/test_review_regressions.py` adds 67 regression cases covering all eight
items. Global permutation scores and tails are checked against independently
constructed ESDA references. Other checks cover grouped-gradient invariance,
library composition preservation, custom result invalidation in AnnData/MuData,
missing cohort types and complete donor counts, aliased image/transcript metadata
with save/load round trips, and hand-counted CLQ examples.

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_review_regressions.py --tb=short
.\.venv\Scripts\python.exe -m pytest -q --tb=short
```

Focused run: **67 passed**. Full suite: **988 passed, one skipped** in 350.97
seconds. Logs are retained locally as `implementation-focused.log` and
`implementation-full.log` under the analysis directory above. `git diff --check`
reported no whitespace errors. See `docs/analysis.md` and public API docstrings
for the resulting behavior.

## Limits and deferred questions

The reproductions establish implementation defects; they do not estimate their
prevalence in real tissue. GPU hardware, complete vendor datasets, and wheel
installation in a clean environment were not exercised. Statistical interpretation
still depends on justified exchangeability and biological replication. Ripley's
rectangular observation window and pointwise envelopes, and subcellular transcript
dependence, remain documented modeling limitations requiring study-specific judgment.
