# astrolabe Tutorial Notebooks

A library of self-contained Jupyter notebooks demonstrating `astrolabe`'s
features end to end. Every notebook can be opened in a fresh kernel and run
top to bottom on its own — none of them require having run another notebook
first, and none hands off state via a saved checkpoint. Most start from
`ab.datasets.xenium()`, the 10x Genomics Human Lung Cancer FFPE dataset
(~5 GB total: 6 expression files ~254 MB + morphology OME-TIFF ~4.8 GB;
~162k cells, 377-gene panel; downloaded once to `.astrolabe_cache/datasets/`).

## Where to start

**[`01_core_workflow.ipynb`](01_core_workflow.ipynb)** is the entry point.
It walks through the everyday pipeline almost every analysis needs: load and
inspect a Xenium run, QC and filter, normalize, build composable spatial
plots, build a spatial graph and cluster, run differential expression and
annotate clusters, then save/reload the result. Read this one first — every
other notebook assumes you've seen its conventions (and several reference it
directly as "the baseline").

The remaining notebooks (`02`–`09`) are independent showcases of more
advanced or specialized features. Read them in any order, based on whatever
question you actually have.

## Showcase notebooks

| Notebook | Question it answers |
|---|---|
| [`02_spatial_statistics.ipynb`](02_spatial_statistics.ipynb) | Is "spatially organized" something I can actually quantify, gene by gene? Global/local Moran's I, Geary's C, Getis-Ord, bivariate co-localization, spatial lag smoothing, Ripley's L for cell-type point patterns, and genome-wide spatially-variable-gene scoring. |
| [`03_neighborhood_analysis.ipynb`](03_neighborhood_analysis.ipynb) | Which cell types sit next to which, does that look meaningful, and does contact actually change either one's biology? Neighborhood enrichment, co-occurrence, centrality, expression-by-distance, ligand-receptor signaling, regional composition, and interaction-changed genes. |
| [`04_if_and_he_images.ipynb`](04_if_and_he_images.ipynb) | My data isn't just transcripts — how do I bring in protein (IF) or histology (H&E) imagery and get per-cell values out of it? Reading OME-TIFF/H&E images, per-cell intensity extraction, H&E stain deconvolution, and image filters. |
| [`05_segmentation_and_crop.ipynb`](05_segmentation_and_crop.ipynb) | What if I only care about one region, or the shipped segmentation is wrong? Drawing and cropping to an ROI, rotate/flip, Voronoi resegmentation, optional Cellpose/StarDist segmentation, transcript reassignment, and segmentation evaluation. |
| [`06_trajectory_and_gene_modules.ipynb`](06_trajectory_and_gene_modules.ipynb) | Some clusters might be a continuum, not discrete bins — and are there gene programs that vary together in space? PAGA connectivity, diffusion pseudotime, and spatial gene modules. |
| [`07_banksy_spatial_domains.ipynb`](07_banksy_spatial_domains.ipynb) | What if I want clusters that bake in spatial context directly, instead of checking for spatial structure after the fact? BANKSY feature augmentation, built by hand once and then via the `banksy_sweep` convenience. |
| [`08_multi_sample_and_persistence.ipynb`](08_multi_sample_and_persistence.ipynb) | I have more than one Xenium run — how do I analyze them together without mixing up incompatible coordinate systems? `concat_samples`, the library's built-in single-library plotting guard, and a deeper save/load round-trip. |
| [`09_merscope_and_cosmx.ipynb`](09_merscope_and_cosmx.ipynb) | Does the same playbook work if my data didn't come from Xenium? The same load → QC/filter → normalize shape repeated on Vizgen MERSCOPE and NanoString CosMx data, with what's identical and what's platform-specific called out explicitly. |

## Conventions shared by every notebook

- Standard preamble: `import logging; logging.basicConfig(level=logging.INFO)`,
  `import numpy as np`, `import astrolabe as ab` (plus `import scanpy as sc`
  where scanpy is called directly), so the library's own `log.info()`
  progress messages are visible.
- Any cell that needs an optional dependency (`cellpose`, `stardist`,
  `harmonypy`, `napari`) is wrapped in a `try`/`except ImportError` that
  prints a one-line install hint and skips gracefully — every notebook runs
  top-to-bottom on a base `uv sync`, with no extras installed.
- The first time a `pl.points()`-style call appears in a notebook, it shows
  that nothing renders until `.show()`, plus one `+` (overlay) and one `|`
  (side-by-side) composition.
- Markdown precedes every code cell that does something new, explaining
  *why* that step answers the question the notebook is currently asking —
  not narrating what the code obviously does.

## Running the notebooks yourself

```bash
uv sync
uv run --with jupyter --with ipykernel jupyter lab notebooks/
```

Or headlessly, to reproduce the verification used while building these:

```bash
uv run --with jupyter --with nbconvert --with ipykernel \
    jupyter nbconvert --to notebook --execute --inplace notebooks/01_core_workflow.ipynb
```

`09_merscope_and_cosmx.ipynb` downloads larger public datasets (~126 MB and
~155 MB respectively) on first run; everything else downloads in seconds.
