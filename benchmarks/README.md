# astrolabe Benchmark Notebooks

Astrolabe reimplements several published algorithms from scratch instead of
depending on the package that originated them, to keep the core dependency
footprint lean and avoid pulling in heavy or conflicting dependency trees
(numba, dask, tensorflow, ...). This directory validates each reimplementation
against a real reference package (or, where none exists, the literal
published formula) by running both on the same input and checking they
correlate.

Every notebook prints a numeric agreement metric (Pearson r, Spearman rho, or
Cohen's kappa, whichever fits) plus a short note on any genuine algorithmic
delta between astrolabe's version and the reference. Unlike `notebooks/`,
these are **not** tutorials - they don't guard optional imports with
`try/except`, since each isolated environment (see below) guarantees the
reference package is present.

Each notebook also times both the astrolabe call and the reference call and
reports the ratio. Most of these runtime numbers are not apples-to-apples,
and each notebook says so inline where it applies (e.g. astrolabe running a
permutation null the reference skips, or one side reusing precomputed input
the other doesn't). Treat them as a rough sense of scale, not a rigorous
benchmark - neither side is warmed up, JIT-compiled code pays a one-time
compilation cost on its first call, and every number reflects one run on one
small dataset in one environment.

## Isolated environments, grouped by reference package

Each subdirectory is a standalone `uv` project (own `pyproject.toml`, own
lockfile, own venv) with astrolabe installed as an editable path dependency
plus whatever reference package(s) its notebooks need. Environments are
grouped by **shared reference package**, not one per notebook, so a package
like `squidpy` is only resolved and installed once even though four
notebooks use it.

| Group | Reference package(s) | Notebooks | What's validated |
|---|---|---|---|
| [`banksy_py/`](banksy_py/) | `banksy-py` (PyPI) | `banksy_matrix.ipynb` | `pp.banksy_matrix`'s **final λ-weighted, per-block-z-scored matrix** vs `banksy_py`'s, block by block, plus a graph-injection diagnosis of the residual gaps |
| [`bin2cell/`](bin2cell/) | `bin2cell`, `cellpose` | `bin2cell_assign_bins.ipynb`, `seg_eval.ipynb` | `seg.assign_bins` vs `bin2cell`; `seg.evaluate_segmentation` vs `cellpose.metrics` (see note below on why not `stardist`) |
| [`squidpy/`](squidpy/) | `squidpy` | `nhood_enrichment.ipynb`, `co_occurrence.ipynb`, `ligrec.ipynb`, `centrality_scores.ipynb`, `spatial_neighbors.ipynb` | `tl.nhood_enrichment`/`co_occurrence`/`ligrec`/`centrality_scores` vs `squidpy.gr.*`; `tl.spatial_neighbors` edge sets (KNN/Delaunay/radius, Jaccard = 1.0) vs `squidpy.gr.spatial_neighbors`, plus lattice formula checks for grid mode and the k∧radius intersection |
| [`astropy_stats/`](astropy_stats/) | `astropy` | `ripley_l.ipynb`, `gradient_significance.ipynb` | `tl.ripley_l` vs `astropy.stats.RipleysKEstimator`; `tl.gradient_significance`'s R̄ vs `astropy.stats.circvar` |
| [`esda/`](esda/) | `esda` (already a core astrolabe dependency) | `colocation_quotient.ipynb`, `spatial_silhouette.ipynb`, `gene_modules_moranbv.ipynb` | `tl.colocation_quotient` vs the published formula; `tl.spatial_silhouette` vs `esda.boundary_silhouette`; `tl.spatial_gene_modules`'s `moran_bv` mode vs `esda.Moran_BV` (off-diagonal, unclipped) and `esda.Moran` (the new univariate diagonal) |
| [`skimage/`](skimage/) | none - `scikit-image` is already a core astrolabe dependency | `stain_deconv.ipynb` | `method='ruifrok'` vs `skimage.color.rgb2hed`; `method='macenko'` vs an independent re-derivation (skimage has no Macenko estimator) |
| [`sepal/`](sepal/) | none (see notebook) | `spatial_diffusion.ipynb` | `tl.spatial_diffusion`'s diffusion score vs a direct re-derivation of the Sepal (Andersson & Lundeberg 2021) formula — a self-referential regression test, not external validation (see note below) |
| [`scipy_stats/`](scipy_stats/) | `statsmodels` | `icg.ipynb` | `tl.interaction_changed_genes`'s vectorized Mann-Whitney (`p_asymptotic`) vs a per-gene `scipy.stats` + `statsmodels` loop — a **vectorization-equivalence check**, not external validation (the reference re-types astrolabe's own algorithm; the real external reference, Giotto's `findInteractionChangedGenes`, is an R package and out of scope) |
| [`minisom/`](minisom/) | `minisom` (already an astrolabe optional extra) | `som_training_quality.ipynb` | `tl.som` end-to-end training quality: planted-blob metacluster recovery + presentation-order invariance vs the pre-overhaul sequential-minisom configuration, QE/TE, and xenium_tiny SOM-vs-Leiden agreement |
| [`synthetic_gradient/`](synthetic_gradient/) | none — analytic ground truth **is** the reference (see note below) | `gradient_ground_truth.ipynb` | `tl.expression_gradient`/`decay_length`/`gradient_field`/`interface_flux`/`gradient_significance` vs closed-form planted fields |

## Running a notebook

```bash
cd benchmarks/<group>
uv sync
uv run --project . jupyter lab           # interactive
# or headlessly:
uv run --project . jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

`astrolabe` is pulled into each environment as an editable path dependency
(`../..`), so local source changes are picked up without reinstalling - no
need to re-run `uv sync` after editing astrolabe itself, only after changing
a group's `pyproject.toml`.

## A note on `squidpy/centrality_scores.ipynb`

`tl.centrality_scores` now computes squidpy's own group-level `networkx`
centrality algorithms (`group_degree_centrality`, `group_closeness_centrality`,
`average_clustering` over the group), by request, instead of averaging each
cell's individual centrality within its group - a different but equally
valid definition astrolabe's docstring always documented accurately. All
three metrics now match squidpy exactly (r=1.0).

## A note on `squidpy/co_occurrence.ipynb`

`tl.co_occurrence` used to normalize each type's ring-local co-occurrence
rate against the *global* tissue-wide frequency instead of the ring-local
one squidpy uses, making its output matrix asymmetric by construction - a
real bug, now fixed. Agreement improved from Pearson r~0.73/Spearman
rho~0.55 to r~0.92/rho~0.92 (exact values drift slightly run-to-run with
the Leiden clustering).

## A note on `squidpy/ligrec.ipynb`

`tl.ligrec`'s CPU path used to score each spatial-graph edge as the product
`L_i * R_j`, averaged over edges, instead of the published CellPhoneDB
(Efremova et al. 2020) convention `(mean(L) + mean(R)) / 2` - a real bug,
now fixed. On the injected complete graph the match with `squidpy.gr.ligrec`
is now exact (Pearson r = 1.0, max |diff| = 0). One gate difference matters
for the comparison itself: astrolabe's CellPhoneDB expressing-fraction gate
(`expr_frac_threshold`, default 0.1) NaNs *both* `means` and `pvalues` for
untested entries, while squidpy's `threshold=` masks only `pvalues` and
always reports every mean - so the notebook disables astrolabe's gate
(`expr_frac_threshold=0.0`) to compare the scoring convention rather than
the gating policy, and asserts the comparison set is non-empty.

## A note on `banksy_py/banksy_matrix.ipynb`

Three rounds of fixes are reflected here. First, `pp.banksy_matrix` used to
weight every neighbor uniformly (`1/k`) instead of published BANKSY's
default `scaled_gaussian` scheme (Gaussian decay from each cell's own median
neighbor distance, row-normalized) - fixed. Second, it neither z-scored the
X/M0/|M1| blocks before applying the `sqrt`-λ weights (so λ's variance-split
meaning silently depended on the data scale) nor subtracted the local
neighborhood mean inside the AGF term (so M0's DC component leaked into
|M1|), and its `agf_weight` default didn't match banksy_py's `2^-m` harmonic
split - all fixed. The notebook now compares the **final λ-weighted
matrices** (the ones PCA actually consumes) block by block: the X block
matches to float32 exactly, M0 r = 0.9896, |M1| r = 0.9204 (up from 0.8329
before the AGF mean subtraction). Third, the residual gaps are now
**diagnosed, not just suspected**: feeding banksy_py's own directed
k-neighbor graph through astrolabe makes M0 exact (r = 1.0000), and its
directed 2k-neighbor AGF graph makes |M1| exact (r = 1.0000) - so the entire
remaining divergence is graph construction (astrolabe's symmetrized shared
kNN graph vs banksy_py's directed per-term graphs, including its 2× larger
AGF neighbor set), a documented, deliberate API difference, not feature math.
A final planted-domain section then asks whether that divergence *matters*:
with weak signal and one domain 4× denser than the rest, astrolabe's
symmetrized shared k-graph recovers domains at least as well as banksy_py's
full directed k + 2k construction in every scenario tested, while the naive
third option - a directed k-graph shared across both blocks - collapses
under density heterogeneity (the AGF term needs either the 2k set or the
symmetrized graph's variable-degree neighborhoods to stay stable). The
symmetrized graph is therefore validated as a genuine improvement, not a
convenience tradeoff.

## A note on `esda/gene_modules_moranbv.ipynb`

Two conventions changed in `tl.spatial_gene_modules`'s `moran_bv` matrix and
are validated by this notebook: values are **no longer clipped to [-1, 1]**
(bivariate Moran's I is not a correlation coefficient; clipping silently
truncated exactly the strongest co-expression pairs), and the **diagonal is
now each gene's univariate Moran's I** instead of a hardcoded 1.0. Both the
symmetrized off-diagonal (vs `esda.Moran_BV`) and the new diagonal (vs
`esda.Moran`) match esda to ~1e-8.

## A note on `esda/spatial_silhouette.ipynb`

`tl.spatial_silhouette` used to reimplement its own graph-restricted
approximation of boundary silhouette despite claiming equivalence to
`esda.boundary_silhouette` (which averages over every cell in a cluster, not
just graph-connected ones) - a real bug, now fixed by delegating directly to
`esda.boundary_silhouette`. Agreement went from r~0.53 to an exact match.

## A note on `sepal/`

No installable Sepal (Andersson & Lundeberg 2021) package exists on PyPI -
the `sepal` package that does exist there is an unrelated MRI library - so
`spatial_diffusion.ipynb` re-derives Sepal's published "steps to uniformity"
formula directly as its own reference implementation. Two real bugs were
found and fixed: a sign inversion in `diffusion_score`, and a redesign from
a fixed-horizon variance-ratio proxy to Sepal's actual time-to-uniformity
metric. Agreement is now an exact match (Pearson r = 1.0, 100% identical
step counts). Because the reference cell hand-rebuilds the *identical*
diffusion operator and convergence rule astrolabe uses, this group is
explicitly a **self-referential regression test** (it guards against future
refactor drift), not external validation.

## A note on `minisom/`

The original notebook here, `som_bmu.ipynb`, compared astrolabe's vectorized
BMU assignment against `minisom.MiniSom.winner()` **on the same trained
codebook** — a tautology (both are the identical squared-Euclidean argmin,
so the 100% match it reported was guaranteed by construction and validated
nothing about training: initialization, presentation order, iteration
budget, decay, scaling, or meta-clustering). It has been deleted and
replaced by `som_training_quality.ipynb`, which validates training quality
end-to-end: planted-blob metacluster recovery (ARI ≈ 0.96–0.98 under both
random and label-ordered sample presentation) vs the pre-overhaul
raw-minisom configuration (sequential single-pass, `n_iter = 0.6·n`,
asymptotic decay — ARI collapses to ≈ 0.47 on label-ordered data, with
worse QE/TE in every condition), plus xenium_tiny SOM-vs-Leiden agreement.
The real-data section also validates `tl.som`'s scaling rule: `scale=None`
(the default) z-scores raw feature matrices (the FlowSOM convention) but
leaves `obsm` embeddings like the default `use_rep='X_pca'` unwhitened — an
earlier default z-scored (whitened) the embedding and measurably collapsed
SOM-vs-Leiden agreement. A k-means control in the notebook shows the same
collapse from whitening alone (no SOM in the loop), and the notebook asserts
the default training space for `X_pca` is identical to an explicit
`scale=False`.

## A note on `synthetic_gradient/`

This group has **no external reference package** — deliberately. The
gradient/flux statistics it covers are either novel (`tl.interface_flux`)
or assembled from published pieces in a way no package reproduces, so
**synthetic fields with closed-form answers are the reference**: a planted
linear ramp (exact MLS recovery), a radial exponential (λ = 80 µm recovered
as 79.8 ± 0.9 with the background offset), a Gaussian bump (negative
divergence at the peak, within a few % of the analytic Laplacian; no `curl`
output), a two-block interface (integrated flux within ~4% of the a·H
closed form, interface length within ~1% of the 4/π Poisson–Voronoi excess,
bit-exact antisymmetry), and Type-I calibration of
`tl.gradient_significance`'s smoothed-surrogate null (0/20 false positives
on autocorrelated-but-isotropic fields; directional positive control at the
permutation floor). Building it surfaced one small real bug, now fixed:
`interface_flux` accumulated its two matrix triangles in separate
`np.add.at` passes whose float additions interleave differently, so the
documented identity Φ(B→A) = −Φ(A→B) held only to ~1e-16 rather than
bit-exactly; it now antisymmetrizes by transpose-subtract.

## A note on `bin2cell/`

`bin2cell_assign_bins.ipynb`'s reference now resolves contested bins with
the same PCA-based tie-break `seg.assign_bins` uses by default (nearest
candidate in provisional-profile PCA space, approximating bin2cell's
"transcriptional similarity" rule) instead of simple nearest-centroid
distance. Agreement improved modestly (Pearson r on per-cell bin count:
0.646 -> 0.664, exact match rate: 8.8% -> 12.0%) - a larger, structural gap
remains because `assign_bins` resolves bins in three ordered tiers while
this reference collapses the first two into one pass, so the two can
disagree about which bins are even unambiguous before any tie-break runs.
Closing that gap would mean copying `assign_bins`' own tiering into the
reference, which defeats the point of an independent check.

`stardist`, this group's originally-planned segmentation backend, can't
install on Python 3.12 (its latest PyPI release hard-pins
`numba==0.53.1`). `cellpose` is used instead; its `average_precision` metric
is built on the same algorithm family as `stardist.matching`, so this
substitution changes how candidates are produced, not what's being
validated.
