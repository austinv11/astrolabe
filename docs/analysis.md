# Statistical inference and analysis extensions

Astrolabe uses AnnData/MuData and micron coordinates. Cells are already segmented
and images registered. Independent libraries have independent coordinate frames:
`tl.spatial_neighbors` isolates `obs['library_id']` automatically.

## Statistical defaults

`moran_local`, `moran_local_bv`, and `getis_ord` calculate empirical equal-tail
permutation p-values. The plus-one correction applies to each tail before
doubling. `p_adj` corrects one family across every requested cell and field in
the call. Labels use `p_adj < alpha`. BH is the default; its independence or
positive-dependence assumptions matter for overlapping spatial tests. Pass
`p_adjust_method='fdr_by'` for arbitrary dependence. Normal z-scores and the
smaller-tail `p_dir` are diagnostics. `quadrant_raw` reports the sign pattern
without significance. Zero permutations or undefined fields are `not_tested`.

Local tests fit each library separately when the spatial graph records a library
grouping (normally `library_id`), then adjust one pooled family across the
requested cells and fields. This preserves each library's expression baseline
and variance in the conditional null. Missing library labels, cross-library
graph edges, and libraries with fewer than three cells are rejected. For SVG
tests, expression permutations stay within libraries. Multi-library SVG
analysis requires `n_perms > 0`; the pooled analytic approximation does not
account for library-specific expression distributions. The graph's explicit
`library_key=None` setting requests a pooled, exchangeable coordinate frame.

Tail arithmetic follows the finite-permutation correction described by
[SciPy's permutation-test reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Finite permutations limit resolution: the smallest equal-tail p-value is
`2/(n_perms+1)`. Increasing the number of cells or tested fields may require more
permutations for useful FDR power. A lack of discoveries at coarse resolution
does not establish the absence of spatial structure. Ripley p-values remain
pointwise across radii; their envelope is not a simultaneous confidence band.

Local Moran/Getis, SVG, ICG, native LR, gradient, replicate-neighborhood and
pointwise Ripley/CLQ results include `n_valid`, `p_min` and
`resolution_limited` (with metric suffixes for gradients). `p_min` is the
theoretical Monte Carlo floor: one-tail
tests use `1/(n_valid+1)`, equal-tail tests use `min(1, 2/(n_valid+1))`.
Untested or analytic entries have a missing floor. Counts refer to valid null
scores rather than attempted draws. Gradient tables use separate `_magnitude`
and `_coherence` suffixes; LR stores matching tables in its result dictionary.

`result.attrs['permutation_resolution']` records family size, alpha, the
rank-one threshold and the required valid permutation count for an isolated
discovery. LR stores this metadata in `result['metadata']`; ICG pair-wise
correction reports each family separately. BH's rank-one threshold is
`alpha/m`; BY also divides by the harmonic sum through `m`. A
`PermutationResolutionWarning` flags floors that cannot cross that strict
threshold. BH/BY can still reject several discoveries together at higher
ranks, so the flag is not a claim that every discovery is impossible.
Rank-one diagnostics support BH, BY, Bonferroni, Holm and uncorrected tests;
other correction methods expose floors but mark `resolution_assessed=False`.
Ties and few biological replicates can impose stronger limits than this
Monte Carlo floor. Plan families before examining results; simply increasing
permutations cannot supply additional biological replication.

```python
result = ab.tl.moran_local(data, 'GENE', n_perms=999, seed=42)
result[['p_sim', 'p_adj', 'n_valid', 'p_min', 'resolution_limited']]
result.attrs['permutation_resolution']
```

`interaction_changed_genes` is a within-type contact association. Receiver
identities stay fixed; contact flags are shuffled within receiver type, library,
and optional `block_key`. Exchangeability within those strata is required.
This does not remove anatomical confounding automatically, establish causality,
or reproduce Giotto's method exactly. Gene/pair BH is global by default.
`expression_scale='linear'` or `'log1p'` is required. For non-natural logs, provide
`log_base`. Asymptotic p-values are diagnostics; zero permutations are untested.

Gradient significance reuses the fitted layer, graph, coordinates, smoothing and
group definitions. It checks source fingerprints and rejects stale gradients or
conflicting overrides. Undefined directional coherence has NaN significance.
The smooth surrogate null models graph-matched noise, not every biological null.

`ligrec` scores actual spatial edges, including when GPU routing is enabled.
Missing/null zero-edge groups are excluded from valid permutation counts.
`min_valid_perms=20` is the default minimum; `status` and `n_valid` explain
untested entries. Interactions and cell-type pairs share one correction family.
That family contains all requested interactions with measured subunits and
present cell-type pairs, including tests excluded by expression screening,
missing edges or too few valid draws. Screened tests count as p=1 only during
correction and remain NaN in public results. Label-dependent expression gates
must not silently shrink the correction family. `metadata['family_size']`
records the original family size.

## Compositional niches

```python
ab.tl.niche_composition(data, 'cell_type')
ab.tl.niches(data, 'cell_type', resolution=1, seed=0)
ab.pl.points(data, 'niches')
ab.tl.composition(data, groupby='cell_type', region_by='niches')
```

Profiles are sparse, non-self one-hop fractions over binary graph edges, with
library isolation and declared cell-type ordering. Niches use joint Leiden
clustering of variance-scaled profiles across libraries. Isolated cells are
unassigned. Niche names describe compositions rather than cell-intrinsic states.

## Regions

```python
membership = ab.tl.assign_regions(data, regions, region_key='region_id')
distance = ab.tl.distance_to_region(data, regions, region='tumor')
summary = ab.tl.region_summary(data, sample_key='sample', groupby='cell_type')
```

Regions are polygon GeoDataFrames in the same micron frame. Membership includes
boundaries and excludes holes. Overlapping assignments raise unless
`overlaps='membership'` explicitly requests all memberships. Multi-library
regions require matching library labels. Signed distance is negative inside,
positive outside, and zero on the boundary. Repeated region IDs form a union per
library. Counts, composition and optional expression summaries use biological
sample IDs. Density requires an explicit observed analysis area in square
microns; a polygon that extends outside imaged tissue is not an observed area.

## Biological replication

Install `astrolabe[cohort]` for PyDESeq2. `sample_key` identifies a biological
sample, such as a donor-condition specimen; technical sections share that ID.
FOV IDs are not biological replicates. `paired_key` identifies the donor shared
by repeated conditions. Metadata must be constant within each sample.
The adapter follows [PyDESeq2's counts, metadata and design workflow](https://pydeseq2.readthedocs.io/en/latest/auto_examples/plot_step_by_step.html).

```python
bulk = ab.pp.pseudobulk(data, sample_key='sample', groupby='cell_type',
                       layer='counts', metadata_keys=['condition', 'donor'])
de = ab.tl.rank_genes_pseudobulk(bulk, condition_key='condition',
                               contrast=('treated', 'control'), paired_key='donor')
contacts = ab.tl.diff_nhood_enrichment(data, 'cell_type', sample_key='sample',
                                     condition_key='condition',
                                     contrast=('treated', 'control'), paired_key='donor')
```

Pseudobulk requires integer raw counts and stores eligibility for small groups.
Outer concatenation records measured-feature coverage separately from zeros;
DE uses the common measured panel within each stratum. DE rejects insufficient
replication, incomplete pairs and confounded/saturated designs. `padj` is the
backend's per-stratum correction; `padj_global` adjusts all returned tests.

Neighborhood contrasts use each sample's mean contact fraction minus the
finite-population target-type expectation, with equal sample weights. Conditions
are permuted at sample level within optional blocks, or paired differences are
sign-flipped at donor level. Confidence intervals resample biological units.
Missing cell types yield untestable comparisons rather than zero scores.

## Complex-aware communication

```python
resource = pd.DataFrame({
    'ligand': ['ligand_complex'], 'receptor': ['receptor_complex'],
    'ligand_members': [['L1', 'L2']], 'receptor_members': [['R1', 'R2']],
})
resource = ab.tl.import_ligrec_resource(resource, species='human', version='study-v1')
ab.tl.ligrec(data, 'cell_type', resource, layer='normalized', n_perms=999, seed=0)
```

Complex expression is the minimum subunit expression. Every subunit must be
measured. Monomer ligand/receptor tables are supported. Supplied resources record
species and version. `astrolabe[communication]` adds offline LIANA bundled human
resources and `tl.ligrec_local` local cosine maps on the selected layer and
stored graph. Maps represent expression association, not transport or causal
signaling. Interaction names/metadata and backend layers accompany the obsm
score matrix. Input resources and tissue context constrain interpretation.
The optional maps use [LIANA's local bivariate API](https://liana.readthedocs.io/en/stable/generated/liana.mt.bivariate.__call__.html).

## Image features

```python
features = ab.pp.extract_image_features(data, spec=image_spec,
                                        mode='context', radius_um=20,
                                        grid_um=1, offset_um=1)
ab.tl.pca(data, use_rep='image_features', key_added='image_pca')
```

Cell polygons or circular centroid contexts provide mean, median, standard
deviation, 10th/90th quantiles, GLCM contrast, homogeneity and energy. Sampling
uses lazy level-zero tiles and a canonical micron grid through the full image
registration/transform chain. The four texture directions use an explicit
physical offset. Integer images use the dtype's range; floating images require
`texture_range=(low, high)`. Use the same range and grid for comparisons. Both
GLCM endpoints must lie in the mask. Outside-image samples stay missing and
coverage is reported. Select a library explicitly for multi-library data.

## Coordinate changes and persistence

Affine transforms move centroids, boundaries, Arrow transcripts and GeoDataFrame
annotations together, including MuData parent stores. Image transforms compose
once per descriptor. Derived spatial results/graphs are invalidated. Filtering
refreshes parent MuData observation/variable indexes. Expression embeddings stay
available.
Derived spatial gene columns, including SVG scores, p-values and significance
flags, are also cleared after a crop removes cells or coordinates are transformed.
Unrelated gene annotations remain available.

`io.save` round-trips typed tables, including MultiIndex rows/columns, using
Arrow IPC and typed axis metadata without pickle. It writes a sibling temporary
file and replaces the destination only after a complete write. Failed writes
retain the prior file and original in-memory object. Referenced images remain
relative to the output directory.

The maintained calibration script is `benchmarks/statistical_calibration.py`.
It reports independent-dataset null rejection, Wilson intervals and planted
signal power. Synthetic calibration does not certify arbitrary tissues, GPU
drivers, real vendor datasets, or every optional dependency version.

`benchmarks/statistical_stress.py` adds unequal library sizes and expression
baselines, uneven biological cohorts, missing cell types, sparse panels and
label-dependent screening. It reports family discoveries and a preselected
raw-p hypothesis separately, plus strong within-library and cohort controls.
Sparse LR raw results are descriptive after screening; their small tested
sample count cannot certify calibration. See `docs/validation.md` for results.
