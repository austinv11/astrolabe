# Astrolabe

This is meant as a lightweight alternative to spatialdata-based packages.

Spatialdata, while powerful, can be extremely difficult to manage with other 
dependencies and introduces extreme complexity (especially in comparison to scanpy idioms).

Notably the data structure that we leverage is the `MuData` object, a multi-modal extension to AnnData.
It has a much more familiar API that should feel more intuitive.

A critical distinction from the spatialdata-based approach is that everything is
with respect to a single coordinate system.

The reference coordinate system is defined by cellular-level observations and coordinates as we assume
that cells are correctly segmented and images are pre-aligned. As a result, this module intends to support
downstream analysis, rather than the pre-processing steps of segmentation and alignment.

Analysis now includes compositional niches, polygon regions and distances,
biological-sample pseudobulk and neighborhood contrasts, complex-aware
ligand/receptor associations, and image texture/context features.
See [the analysis guide](docs/analysis.md) and the runnable
[synthetic example](examples/analysis_extensions.py).

Local statistical defaults use empirical permutation p-values with one BH
family across requested cells and fields (BY is available). Contact-associated
gene tests require an explicit expression scale and use within-type contact
permutations. Coordinate transforms invalidate spatial results and keep
transcripts, annotations and image registration together.
Local nulls respect library-specific expression distributions. Native inference
reports valid permutation counts, p-value floors and warnings about limited
resolution. Spatial gene results are cleared after cropping or transforming.

Optional backends can be installed with
`pip install 'astrolabe[cohort,communication]'`. The core additions
reuse the existing dependencies. Run `pytest -q` for regression checks.

Conditional count associations with disconnected tissue fields are available as
the experimental `tl.conditional_expression` integration. Install
`astrolabe[spatial-inference]` on Python 3.12, or select a separate prepared
Python 3.12 backend environment. It uses published BYM2/INLA and reports Bayesian
credible intervals and posterior sign probabilities. The backend is optional,
has separate proprietary licensing, and is never imported or downloaded during
ordinary use. See [installation and supported scope](docs/analysis.md#conditional-expression-with-bym2inla).
