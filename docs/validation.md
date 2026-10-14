# Implementation validation

Validated locally on Windows with Python 3.12.12, NumPy 2.4.6, SciPy 1.17.1, AnnData 0.12.16, MuData 0.3.8, Scanpy 1.12.1 and ESDA 2.9.0. Real optional backends: PyDESeq2 0.5.4 and LIANA 1.10.0.

The full suite passed with **921 passed, 1 skipped** (optional Louvain), with exit code 0 in 353.73 seconds after the statistical follow-up. The optional Napari test plugin is disabled because its unused GUI/session fixtures held an earlier runner open after all assertions passed. Tests include the transform/crop/analysis/image/save workflow, resource complexes, category-index overflow, library-specific nulls, screened LR families and permutation resolution. The runnable example covering the five retained feature areas completed successfully. An additional persistence check verified SVG diagnostics with a NumPy integer permutation budget. The dependency lock and whitespace checks passed.

## Statistical calibration

Each scenario uses 200 independent datasets and 199 permutations. Intervals are 95% Wilson intervals. All-null local rejection is the probability of any BH discovery. The one-sided exact binomial guard flags excess over 5% at p < .01; every tested null passed that guard.

| Scenario | Rejections / 200 | Rate | 95% interval |
|---|---:|---:|---:|
| sparse moran | 0 | 0.0% | 0.0%–1.9% |
| continuous moran | 0 | 0.0% | 0.0%–1.9% |
| sparse bivariate | 0 | 0.0% | 0.0%–1.9% |
| sparse getis | 0 | 0.0% | 0.0%–1.9% |
| local signal | 189 | 94.5% | 90.4%–96.9% |
| icg null | 7 | 3.5% | 1.7%–7.0% |
| icg type marker | 12 | 6.0% | 3.5%–10.2% |
| icg signal | 200 | 100.0% | 98.1%–100.0% |
| icg block confounding | 0 | 0.0% | 0.0%–1.9% |
| gradient null | 4 | 2.0% | 0.8%–5.0% |
| gradient magnitude null | 10 | 5.0% | 2.7%–9.0% |
| gradient signal | 200 | 100.0% | 98.1%–100.0% |

Zero local null discoveries indicate conservatism for these generators and this permutation resolution, not universal FDR control or absence of biological signal. The planted Moran signal was detected in 94.5% of datasets; ICG and gradient positive controls were detected in 100%. The gradient null is generated with the same graph diffusion family as the surrogate model. Blocking renders the ICG contact/anatomy confounding control uninformative (p=1). These controls do not establish causal interpretation or arbitrary-tissue validity.

Reproduce with `python benchmarks/statistical_calibration.py --seeds 200 --perms 199`. Per-dataset records are written to `analysis/implementation_validation/calibration.json`.

## Heterogeneous statistical stress tests

`benchmarks/statistical_stress.py` ran 200 independent datasets per scenario
with 999 permutations. It reports family-level discoveries and one preselected
raw-p hypothesis separately. The latter prevents coarse BH resolution alone
from making a null calibration appear successful. Every assessed null passed
the one-sided exact binomial excess-rejection guard against 5% at p < .01.
Strong planted family controls must have at least 50% detection.

| Scenario | Family discoveries / 200 | Rate | Preselected raw rejections / 200 |
|---|---:|---:|---:|
| unequal libraries, shifted Moran baselines | 0 | 0% | 15 |
| unequal libraries, shifted bivariate baselines | 0 | 0% | 12 |
| unequal libraries, scaled Getis expression | 0 | 0% | 12 |
| shifted-library Moran SVG | 5 | 2.5% | 11 |
| shifted-library Geary SVG | 10 | 5% | 11 |
| unequal cohorts and cell counts | 5 | 2.5% | 10 |
| missing cohort cell types | 4 | 2% | 10 |
| sparse LR panel with missing subunits | 2 | 1% | screening leaves only 1 tested dataset |
| sparse LR panel with label-dependent screening | 0 | 0% | screening leaves only 3 tested datasets |
| strong within-library spatial signal | 171 | 85.5% | 97 |
| strong cohort neighborhood contrast | 200 | 100% | 200 |

The library signal's family detection interval is 80.0%–89.7%; the cohort
signal's is 98.1%–100%. Raw rejection rates of 6%–7.5% in some local nulls
have intervals including 5% and did not trigger the excess guard; this does
not prove exact nominal calibration. Sparse LR raw rates after screening
are conditional descriptions with insufficient tested datasets, not null
calibration evidence. Its family-level rate is measured across all 200 runs.
These generators do not certify arbitrary tissue geometry, spatial dependence,
confounding or biological replication designs.

Stress testing uncovered two source issues. Local tests could mix expression
distributions from different libraries despite separate graphs; they now fit
library-specific nulls before one pooled correction. SVG permutations also
remain within libraries, and multi-library analytic SVG inference is rejected.
LR expression gates could shrink the requested correction family. With 200
datasets and 199 permutations, that screened-panel case produced **36/200
(18%)** false-discovery runs before the fix and **0/200** after retaining
screened hypotheses in the correction family (same seeds and budget).
The corrected 999-permutation run also produced 0/200. Positive LR regression
controls still detect a strong interaction after the full-family correction.

Reproduce with `python benchmarks/statistical_stress.py --seeds 200 --perms 999`.
The run completed in 419.38 seconds; strict JSON records and Wilson intervals
are in `analysis/implementation_validation/statistical_stress.json`.
The before/after screening records are `sparse_screening_before.json` and
`sparse_screening_after.json` in the same directory. Regenerate the corrected
comparison with `--perms 199 --scenarios sparse_screened_lr`.

## Allocation profile

On 30,000 cells × 2,000 genes, 64 LR pairs, 145,528 directed edges, 39 permutations and 2 workers, peak traced allocation during LR scoring was 16.3 MiB after adding resolution diagnostics (14.63 seconds). The former two edge-expression tensors alone would require 142.1 MiB. Input storage, imports and graph construction are excluded; this is tracemalloc, not process RSS. Edge scoring streams CSR blocks and null counts; no edge-by-pair tensor is retained.

Reproduce with `python benchmarks/ligrec_memory.py`; increase cells/pairs explicitly to profile a production workload.

## Scope

All sixteen numbered audit findings have source fixes and maintained regressions. Numerical/reference checks cover empirical tail ranks, SciPy Welch and tied Mann–Whitney arithmetic, independent neighborhood fractions, skimage GLCMs, PyDESeq2 direct-backend results and LIANA local maps. Persistence checks cover MultiIndex tables, categorical/nullable columns and failed writes.

The Windows/Linux × Python 3.12/3.13 × base/extras CI matrix is configured but was not executed remotely in this workspace. CUDA hardware/driver execution and full real-vendor datasets remain unvalidated. No deployment or publication was performed.
