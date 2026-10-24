# Spatial analysis contracts and workflows

## Libraries, TMA cores, and donors

Keep physical and biological identifiers separate. A spatial unit is a
`(library, core)` pair. Different cores can have identical coordinates without
being neighbors. Disconnected fragments of one core can share a mask. Neither
cores nor fragments automatically supply independent biological replication.

```python
ab.pp.set_tissue_support(data, masks, library_key="slide", core_key="core_id")
ab.tl.spatial_neighbors(data, radius=20)
```

`masks` is a GeoDataFrame of Polygon/MultiPolygon geometries in microns with
matching unit columns. Holes remain unobserved. Graph edges stay within their
unit and observed tissue. `coord_type="contact"` uses cell polygons;
`contact_distance` is their permitted separation in microns. Cropping intersects
the observation mask; affine transforms update it.

Region distances and landmark bearings never search unrelated cores. Missing
landmarks produce missing values. Local spatial nulls and native LR/ICG
permutations respect resolved physical units. Explicit `library_key=None` in APIs
supporting pooled permutations is an intentional pooling choice.

Directional gradient inference and other analyses requiring one coordinate domain
reject multi-core pooling. Subset a core and rebuild its graph before those
analyses. Units resolve from registered metadata, graph parameters, or conventional
`library_id`/`core_id` columns. Cores remain distinct from donors and specimens.

## Measured panels and source freshness

Outer concatenation stores `uns['measured_features']`, a feature-by-library
coverage table. Unmeasured entries are structural missingness, not observed zeros.

```python
common = ab.pp.select_analysis_panel(data)
eligible = ab.pp.select_analysis_panel(data, genes=["CXCL12", "CXCR4"])
```

Native LR additionally supports `coverage_policy="eligible"`; each interaction
uses libraries measuring every subunit. No eligible library means untested.
Pseudobulk retains a separate `layers['measured']` matrix.

Registered results track expression, graphs, labels and geometry. Astrolabe
mutations invalidate affected outputs; consumers reject stale upstream results.
After external edits, inspect `ab.tl.validate_results(data)` or discard stale
outputs with `invalidate=True`. Rebuild graphs after cell subsetting. Unregistered
external results cannot be certified by these fingerprints.

## Donor-level niche abundance and state

```python
table = ab.tl.niche_abundance(
    data, sample_key="section", donor_key="donor",
    condition_key="condition", exposure="all_cells"
)
contrasts = ab.tl.diff_niche_abundance(
    table, contrast=("treated", "control"), paired=True
)
pb = ab.tl.niche_pseudobulk(
    data, donor_key="donor", condition_key="condition",
    niche_key="niches", cell_type_key="cell_type"
)
ab.tl.rank_genes_pseudobulk(
    pb, condition_key="condition", contrast=("treated", "control"),
    paired_key="donor"
)
```

Abundance denominators are all cells, assigned cells, or supplied tissue area.
Absent niches have explicit zeros. Technical samples pool within donor/condition;
contrasts use equally weighted donor fractions/densities and standard
[statsmodels OLS/HC3](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html)
with residual-t inference. This approximate linear model is not a spatial count
model or joint compositional model. Covariates must be constant within
donor/condition. Three donors is an input floor, not an adequate-study-size claim.
If storage loses DataFrame attributes, pass `donor_key`, `condition_key`, and
`exposure` explicitly to `diff_niche_abundance`.

Niche-state counts use the existing PyDESeq2 pathway without imputing absent
strata. `fit_niche_model` freezes composition scaling, centroids and empirical
distance cutoffs; `map_niches` applies them without refitting. Ties, isolates and
distant profiles stay unassigned. Cutoffs are not calibrated probabilities.
Inspect held-out assignment rates and rare-niche stability. Discovery and testing
on the same samples remain exploratory.

## Subcellular contrasts and scale sensitivity

`tl.diff_subcellular` aggregates prespecified per-cell localization summaries to
donor means, then calls SciPy's
[permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
and [BCa bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
Donor labels must be exchangeable under the null; paired inference swaps labels
within complete pairs. It does not adjust covariates. Use raw localization
summaries, not cell-level p-values or cohort-fitted shrinkage scores. Missing
values and incomplete pairs are reported; degenerate intervals stay unavailable.

```python
ab.tl.spatial_scale_sensitivity(
    data, ["CXCL12"],
    graph_specs={"r20": {"radius": 20}, "r40": {"radius": 40}},
    n_perms=999, seed=7
)
```

This repeats existing Moran/Geary tests over prespecified graphs. Holm adjustment
covers all requested fields/scales; no best scale is selected. Mask-aware
`ripley_l(..., radii=[...])` defaults to border correction and simulates inside
each core's polygons. Radii with no eligible interior centers stay untested.
The single-unit call without a mask or explicit radii retains the rectangular
window estimator; use an observed mask for irregular tissue.

## Optional integrations

`seg.run_proseg` requires a separately installed Proseg 3.2 executable, runs units
independently and returns separate AnnData reconstructions. Inputs, commands,
versions, logs and transcript-ID maps are retained. Input data remain intact;
hard assignments are integer counts, expected counts occupy a separate layer.
Use `include_expected_counts=True` to request that layer; native Proseg exports
round posterior means to two decimal places. Every run verifies hard counts
against the backend's own matrix export after aligning genes and reconstructed IDs.
Select inferred coordinates explicitly with `seg.select_transcript_coordinates`.

Proseg 3.2 normalizes/clips z: backend values remain normalized audit columns,
while physical input z is preserved. Inferred xy/displacement are in microns.
Its transcript export supplies no assignment probabilities; none are invented.
`seg.reconstruction_qc` describes reassignment, displacement and user-specified
marker co-detection, without certifying correctness. `seg.record_segmentation`
records immutable named fingerprints. Read the [validation record](spatial-validation.md)
before interpreting correction quality.

Install `astrolabe[spatialdata]` only when needed. Only explicit
`io.to_spatialdata`/`io.from_spatialdata` calls import SpatialData. Counts, IDs,
micron frames, masks and transcripts round-trip. ImageSpec objects remain lazy
external file descriptors; pixels are not exported. Without polygons, supply a
positive `centroid_radius` for display circles. Foreign imports require explicit
micron units and a coordinate system. Modified cell transforms with external
images require explicit image realignment. Unsupported cell topology is rejected.

## Memory and unsupported inference

Local Moran/Getis nulls are chunked. `max_working_bytes` limits simulation
workspace, not total RSS, graph storage or results. `disk_backed=False` is the
default; `True` explicitly selects temporary memmaps, optionally under
`scratch_dir`, cleaned after success or exceptions. No automatic spill occurs.
The bounded conditional kernel is serial; `n_jobs` does not parallelize it.
Gene-module guards reject oversized dense workloads before allocation.

`tl.conditional_expression` provides optional negative-binomial BYM2/INLA fits.
Install `astrolabe[spatial-inference]` on Python 3.12 or explicitly select a
prepared external Python 3.12 interpreter. Ordinary imports never load pyINLA;
the adapter never installs packages or downloads an engine. Supply counts,
positive exposures, a prespecified target/covariates and a symmetric binary
graph without cross-core edges. Donor IDs can connect the same donor's independent
cores through a shared intercept. The donor SD prior is explicit:
`donor_sd_prior=(1, .01)` specifies a 1% prior
probability of SD exceeding 1 on the log-expression scale. Choose this scale
using the study context and assess sensitivity, especially with few donors.
Results include Bayesian credible intervals and posterior sign probabilities,
not p-values. The integration is experimental and
bounded to small dense-prior problems. See [installation, examples and limitations](g5-inla-proposal.md).
