# Spatial implementation validation

## Scope

These checks validate specific implementations and designs. They do not establish
causality, universally calibrated spatial inference, or reliable recovery of
unknown molecule positions. G5's custom PyMC proposal is withdrawn; see
[the backend decision](g5-backend-decision.md).

## Donor-level abundance

`benchmarks/niche_calibration.py` tests the public statsmodels OLS/HC3 wrapper
with unequal cell yields and repeated sections. Five fixed scenarios each have
1,000 replicates: balanced null, unequal donor counts, paired null, rare-niche
null, and a planted 0.15 fraction difference. The initial 400-replicate imbalanced
scenario had 5.5% rejection but an upper confidence bound of 8.21%, narrowly
missing the prespecified 8% bound. The run was expanded to 1,000 replicates without
changing the model, simulation, or acceptance threshold.

Final null rejection rates were 4.7%, 5.6%, 3.1%, and 3.4%; all upper 95% binomial
bounds were below 8%. Power for the specified large effect was 100%. Interval
coverage ranged from 94.4% to 96.9%. Values and seeds are in
[niche-calibration-results.json](niche-calibration-results.json). This does not
cover all compositions, informative sampling or covariate confounding; the gates
are not proof of exact 5% error control.

## Process memory

`benchmarks/spatial_memory.py` isolates cases in fresh processes and samples RSS
every 20 ms. A four-neighbor ring graph, one gene and 39 permutations exercise
local Moran allocation and disk/in-memory equivalence. The working budget is
256 MiB. Python 3.12.12 and ESDA 2.9.0 were used on Windows.

| Cells | In-memory time / peak RSS | Disk-backed time / peak RSS |
| --- | --- | --- |
| 100,000 | 3.33 s / 584 MB | 3.18 s / 570 MB |
| 1,000,000 | 13.41 s / 2,033 MB | 10.49 s / 2,008 MB |

Result hashes match exactly between modes. RSS includes imports, compilation,
graph weights and outputs, so exceeds the simulation budget. Low-permutation
runs check resources, not discovery power. See
[spatial-memory-results.json](spatial-memory-results.json).

## Proseg controls: efficacy gate not met

Proseg 3.2.0 was built and exercised on Windows. Per-core import, integer molecule
accounting, source preservation and H5MU persistence work.
`benchmarks/proseg_controls.py` supplies independent molecule truth for clean and
displaced detections in two cores, with a real boundary-associated gene. Reduced
runs use 100 burn-in/100 retained iterations and 10 morphology steps per iteration.
They do not establish convergence and differ from the default morphology schedule.

| Input / diffusion | Initially correct | Correct among all input molecules | Boundary-gene correctly retained |
| --- | --- | --- | --- |
| Clean / off | 100% | 76.49% | 81.25% |
| Clean / on | 100% | 94.31% | 93.23% |
| Displaced / off | 83.33% | 60.94% | 55.73% |
| Displaced / on | 83.33% | 77.95% | 66.15% |

The follow-up uses the executable's default schedule: 200 burn-in iterations,
200 samples, 100 recorded samples and 4,000 morphology steps per iteration.
The same inputs and two-component/no-factorization settings are retained.

| Input / diffusion | Initially correct | Correct among all input molecules | Boundary-gene correctly retained |
| --- | --- | --- | --- |
| Clean / off | 100% | 87.99% | 89.06% |
| Clean / on | 100% | 93.75% | 89.58% |
| Displaced / off | 83.33% | 70.03% | 67.71% |
| Displaced / on | 83.33% | 78.26% | 68.75% |

Background/discarded molecules count as unrecovered, so foreground filtering
cannot inflate recovery. Neither schedule improved the displaced case over
initial assignments. These results **do not pass the correction-efficacy gate**.
The adapter remains an explicit experimental integration; automatic correction
is not recommended from these results. Proseg 3.2 provides no seed control.
Independent real-data validation remains necessary before efficacy claims.
Both runs are retained in [proseg-control-results.json](proseg-control-results.json).

Proseg is a published method:
[Cell simulation as cell segmentation](https://www.nature.com/articles/s41592-025-02697-0).
This limited experiment does not establish its performance across intended datasets.

## Held-out reconstruction and native count agreement

`benchmarks/proseg_validation.py` fixes seeds 901/902 and four inputs before
fitting: clean coordinates, 0.8-micron jitter, 20% cytoplasmic leakage with
4-micron perturbations, and a 10-micron displacement stress. Each input has two
independent cores, 16 cells/core, 400 molecules/cell and 52 genes, including
genuine shared, boundary and rare-cell expression. All 16 default-schedule fits
(32 core results) are retained in [the held-out results](proseg-heldout-results.json).

Mean assignment accuracy across cores falls from 100% to 84–89% for clean data,
from approximately 90–91% to 80–82% for jitter/leakage, and from approximately
85% to 77–78% for the displacement stress. Background and filtered transcripts
remain in the scoring denominator. Diffusion does not restore initial accuracy;
these held-out cases also fail the correction-efficacy gate. Do not tune these
cases until they pass or interpret foreground filtering as recovery.

Every run now compares hard reconstructed counts with the native backend's
MatrixMarket export after aligning cell IDs and genes. Optional expected counts
come from a separate native export, rounded to two decimals by Proseg. They are
posterior mean counts, not hard assignments or transcript assignment probabilities.

`benchmarks/proseg_vendor.py` runs the real Xenium fixture through default Proseg,
count validation, reconstruction QC and H5MU save/load. Of 20,999 input molecules,
15,014 are assigned, 5,950 background and 35 filtered; 354 reconstructed cells
have 15,014 hard counts and 14,837.99 expected counts. Round-trip checks pass.
The fixture's original matrix total (31,714) differs from its transcript subset;
neither vendor assignments nor that matrix are independent reconstruction truth.
See [the vendor integration results](proseg-vendor-results.json).

The additional diffusion ablation uses fresh seeds 903/904 for clean, jittered
and leaking coordinates, retaining all 18 runs (36 core results). It distinguishes
`--no-diffusion` from `--diffusion-probability 0`, which retains near-only
adjustments. Mean assignment accuracy is:

| Input | Original | Disabled | Near only | Default diffusion |
| --- | ---: | ---: | ---: | ---: |
| Clean | 100% | 88.4% | 86.9% | 84.3% |
| Jitter | 90.7% | 81.1% | 84.0% | 81.7% |
| Leakage | 90.2% | 79.2% | 81.8% | 80.4% |

Near-only adjustments improve on the other Proseg arms for these noisy inputs,
but remain worse than the original assignments and harm clean inputs. Positional
RMSE also increases: for leakage, 2.36 microns initially versus 2.76 near-only
and 3.30 with default diffusion. All native count checks pass. These are
descriptive averages across two simulated datasets, not independent biological
replicates. No Proseg default is selected from these outcomes. See
[all ablation results](proseg-diffusion-ablation-results.json).

## Optional conditional expression

The approved implementation uses published BYM2/INLA through optional pyINLA
0.1.11, without R. An independent analytic Gaussian posterior with eight cores
and shared effects for four donors agrees to maximum absolute errors `6.6e-7`
(means) and `7.4e-6` (SDs). A count example yields effect `0.42336` for a planted
`0.5`, with credible interval `[0.3048, 0.5433]`. The prior table used by the
engine exactly matches the backend-generated table. These results are retained
in [the baseline reference results](inla-reference-results.json). Those count
results used the inherited log-gamma donor prior; current references explicitly
specify the PC donor prior.

The public adapter has a separate direct-call agreement test, input rejection,
optional-import isolation, disconnected-core/shared-donor, serialization and
dependency-freshness tests. It uses no p-values or BH adjustment. A bounded seeded
screen is implemented in `benchmarks/inla_calibration.py`; it reports interval
coverage, zero exclusion and exact binomial uncertainty. This integration remains
experimental: numerical agreement and a small simulation screen do not establish
calibration for arbitrary tissue, covariate selection or segmentation errors.

All **140 fits** in the initial seven-scenario screen complete (20 seeds/scenario).
This screen and the fixed-design, extended and sensitivity records below use
the inherited log-gamma donor prior, not the current explicit PC prior.
The table reports coverage of the true coefficient by a 95% credible interval,
and exclusion of zero. Zero exclusion is a false exclusion only in null rows;
in planted-effect rows it measures detection. No priors were tuned to these runs.

| Scenario | True coefficient | Coverage | Zero exclusion | Mean bias |
| --- | ---: | ---: | ---: | ---: |
| Smooth spatial null | 0 | 95% | 5% | -0.017 |
| Planted distance effect | 0.5 | 100% | 100% | -0.014 |
| Unequal core sizes, null | 0 | 85% | 15% | 0.021 |
| Donor condition effect | 0.5 | 95% | 80% | -0.013 |
| Donor condition null | 0 | 85% | 15% | -0.028 |
| Measured regional confounding, adjusted | 0 | 90% | 10% | -0.014 |
| Neighbor transcript misassignment, model violation | 0.5 | 20% | 100% | -0.118 |

These results **do not establish general nominal calibration**. For example,
3/20 false exclusions have an exact 95% interval of 3.2–37.9%, so the unfavorable
point estimate is imprecise, not evidence to suppress. The misassignment stress
reallocates 20% of molecules to graph neighbors and explicitly violates the fitted
observation model: spatial covariance does not correct that measurement error.
Retain the raw inputs and evaluate reconstruction separately. Full per-fit results,
binomial intervals and runtime are in [the calibration record](inla-calibration-results.json).

The separate fixed-design check uses 200 independent simulated response/field/
donor-effect realizations on one unequal-core graph with fixed exposure and
predictor: 141 cells, 16 cores and eight donors, seeds 4201–4400. All fits complete.
Seven 95% credible intervals exclude the true zero effect: **3.5%**, with exact
95% binomial interval **1.42–7.08%**; mean bias is `0.00397`. Runtime is 489 seconds
with the public API's shared-design gene batch. This is evidence for that one
conditional design, not a replacement for varying-design experiments. Full
results are in [the fixed-design record](inla-fixed-design-results.json).

A separate 144-cell/eight-donor run samples peak parent RSS at 291 MB and peak
summed backend-process RSS at 225 MB (decimal units), completing in 17 seconds
while other validation ran. Summed RSS can count shared pages more than once;
it is not unique physical memory or an API memory guarantee. See
[the memory run](inla-memory-results.json).

### Larger varying-design and prior checks

The additional screen fits **300 datasets**, 100 per null scenario, with seeds
6201–6300. Donor counts cycle through 4/6/8/12; core widths and donor cell counts
vary, as do exposure, counts and latent effects. All fits complete. Exact 95%
binomial intervals accompany the false-exclusion rates:

| Scenario | 95% interval coverage | False exclusion (95% interval) |
| --- | ---: | ---: |
| Unequal core sizes | 88% | 12% (6.36–20.02%) |
| Donor condition | 85% | 15% (8.65–23.53%) |
| Adjusted regional confounding | 92% | 8% (3.52–15.16%) |

For donor-condition nulls, coverage is 18/25 with four donors, 21/25 with six,
and 23/25 with eight or twelve. These subsets remain small. The first two pooled
scenarios have false-exclusion intervals above 5%; this strengthens the evidence
of undercoverage. No nominal-coverage or calibrated-sign-probability claim is made.
The benchmark caches identical graph/prior calculations only; every native fit
still checks the serialized prior exactly. An uncached public API fit agrees
within `6e-6` in all four mean/SD/interval summaries. See
[all extended runs and donor strata](inla-extended-results.json).

The separate **140-fit sensitivity screen** uses ten seeds for null and planted
donor effects, each with seven prespecified settings: the default, spatial PC
scale halved/doubled, phi PC probability 1/3 or 2/3, and donor log-gamma rate
multiplied/divided by ten. All fits complete. Null coverage is 8/10 for every
setting; planted-effect coverage is 7–8/10. Zero-exclusion decisions never change
relative to default. Maximum paired mean and SD changes are 0.0162 and 0.0269.
This bounded sensitivity check does not resolve undercoverage; it varied the
donor prior's rate without changing its family. [All sensitivity settings and runs](inla-sensitivity-results.json)
are retained, including unfavorable outcomes.

The separate [donor-prior investigation](inla-undercoverage-investigation.md)
compares the established PC prior with the inherited prior, oracle parameters
and full Laplace. The public API exposes `donor_sd_prior=(1, .01)`, meaning
`P(donor SD > 1) = .01` on the log-expression scale, and checks the native prior
serialization. This is a published INLA prior, not a new interval correction.

## Real files and optional environments

`tests/data/xenium_4_tiny` retains six unmodified files from 10x's CC BY 4.0
XOA 4.0 test dataset. The archive MD5 and individual SHA-256 values are verified.
The reader is checked directly against HDF5 counts and parquet cell coordinates:
358 cells, 520 features, 31,714 counts. It also exercises QC, graphs and save/load.
Images are absent; attribution/source links are in the fixture README. This adds
real Xenium coverage, not equivalent real-file coverage for every vendor.

Two additional public SFEData crops now pass direct vendor-file comparisons,
QC, spatial graphs and H5MU round-trips: **MERSCOPE 1,058 cells / 88 genes /
74,469 counts**, and **CosMx 27 cells / 960 genes / 58,633 counts**. All loaded
cell boundaries are compared with their source geometry. These checks exposed
and fixed root-level MERSCOPE WKB Parquet loading, case-insensitive boundary
types, ZIndex selection, and CosMx hyphenated polygon filenames / `cellID` keys.
Multipart or holed MERSCOPE boundaries raise explicitly because the ragged
boundary representation cannot preserve their topology. The source files are
downloaded separately, not redistributed. Checksums and outcomes are in
[the vendor record](vendor-validation-results.json).

### Independent Proseg boundary comparison

The [ComSeg research benchmark](https://doi.org/10.5281/zenodo.11237477) supplies
independently imaged accepted cell masks and matching nuclear priors. Its
[published preparation code](https://github.com/tdefa/script_Benchmark/blob/master/ComSeg/main_merfish_breast.py)
specifies 0.108 microns per pixel. The central 2,000-pixel window contains 47,498
transcripts and 142 nuclear seeds. Proseg receives only coordinates, genes and
the published nuclear assignments; reference cell boundaries are used only for
evaluation. The source is CC BY 4.0; extracted member CRCs and recorded SHA-256
hashes are checked, without claiming a whole-archive checksum from partial reads.

Eight uniquely mapped reference cells lie fully inside the scoring window;
344 extranuclear transcripts are scored. Across three unseeded repetitions,
assignment agreement is **48.3–50.6% without diffusion** and **46.8–51.2% with
diffusion**. The published watershed agrees on 86.6%; the published Baysor output
agrees on 19.5% under this conservative nuclear-label matching rule. Those
comparators use the full crop, whereas Proseg uses a smaller context window, so
these numbers are not a general ranking of methods. All 47,498 input molecules
are accounted for in every Proseg run.

This small 2D geometric comparison **does not establish spillage correction**:
the masks do not reveal molecules' pre-spillage origins, z stacks were collapsed
by the original benchmark, and diffusion shows no consistent improvement.
Retain experimental opt-in status. See [all repetitions and limitations](proseg-independent-results.json).

An isolated SpatialData 0.8 environment verifies import isolation, linked
IDs/counts/transcripts/image metadata, Zarr round-trip, and transformed custom
masks. A fresh subprocess rejects SpatialData imports during ordinary Astrolabe
imports, even with SpatialData installed.

## Distribution and continuous integration

The latest full Windows run passes **1,080 tests**, with four optional-feature
skips and 291 warnings, in 500 seconds. The Linux run passes **1,022 tests**, with
14 skips and 271 warnings, in 386 seconds. Both enable the native INLA regression;
the Linux environment omits the separate cohort/communication extras. Tests cover
direct backend agreement, isolated cells, shared donors, failed-import audits,
native count alignment, worker cleanup, real-format readers and cross-drive paths.
The installed wheel's packaged worker also completes a native
fit while keeping pyINLA out of the parent process.
`uv lock --check`, `uv build` and `git diff --check` succeed.

Linux setup exposed pyINLA's outdated check for a sibling executable, fixed by an
explicit installer compatibility link. Its NumPy constraint also selected an
older TIFF reader incompatible with Zarr >=3.2; the inference extra now constrains
Zarr accordingly. No package import or fit downloads software. The Linux engine
is build `26.10.24-1`; its native binary and launcher have separate hashes.
The analytic Gaussian comparison has maximum mean/SD errors `1.12e-6`/`7.31e-6`.
Three saved inputs run unchanged on Windows and Linux agree within `2.45e-5`
over posterior means, SDs and interval endpoints, despite different native builds.
See [Linux references](inla-linux-reference-results.json) and
[matched-input comparisons](inla-matched-platform-results.json).

The wheel was installed into a clean Python 3.12 environment without editable
source access. `benchmarks/wheel_smoke.py` verifies the import location, rejects
any attempted SpatialData import, and runs real Xenium loading, QC, graph
construction, Moran inference and H5MU save/load. Counts, coordinates and result
freshness survive the round-trip: 358 cells, 520 features and 31,714 counts.

The source archive uses an explicit inclusion list. Source, tests (including
the licensed fixture), docs, examples, validation scripts and font assets are
retained; executed notebooks and local editor files are excluded. The validated
archive shrank from 276 MB to approximately 1.94 MB without editing notebooks.

CI includes separate SpatialData 0.8-compatible optional jobs on Windows/Linux,
INLA-extra installation/native-reference jobs on Windows/Linux, and a clean-wheel
job on Linux. The inference jobs explicitly install the engine during CI setup
and upload numerical reference records; ordinary package code never installs it.
Hosted validation runs on [the separate snapshot branch](https://github.com/austinv11/astrolabe/actions/runs/40111547286),
commit `4d36a3cfbeaf8ed4eec0d34ab04feba551f22eac`: **all 13 jobs pass**. This covers
Python 3.12/3.13 on Windows/Linux, base and cohort/communication installations,
both optional SpatialData and native INLA jobs, and the clean-wheel workflow.
The [CI record](ci-validation-results.json) retains job URLs and conclusions.
The initial hosted run exposed
Windows Git line-ending conversion of a checksummed vendor JSON file; `.gitattributes`
now preserves fixture bytes. The original six vendor checksums remain unchanged,
including after an isolated `core.autocrlf=true` checkout. Local optional-environment
validation passes all four SpatialData tests.

The rebuilt wheel also passes the Xenium smoke workflow after installing
`astrolabe[spatial-inference]` into its isolated environment. NumPy 2.0.2 and
pandas 2.3.3 satisfy the optional wrapper's constraints; the ordinary environment
remains independent. Import guards reject both pyINLA and SpatialData during
ordinary operations. Explicit fitting with an installed wrapper but no engine
fails clearly without a download.
