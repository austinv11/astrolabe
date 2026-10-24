"""
astrolabe — lightweight spatial transcriptomics analysis in Python.

A scanpy-idiomatic library built on MuData (multi-modal AnnData) as its
core data structure.  Designed as a lightweight alternative to SpatialData
for downstream analysis of pre-segmented, pre-aligned spatial data.

Submodules
----------
io       — data readers (Xenium, MERSCOPE, CosMx, OME-TIFF, H&E)
pp       — preprocessing (QC, normalization, IF intensity extraction)
tl       — analysis tools (spatial graph, clustering, spatial statistics)
pl       — plotting (spatial scatter, image overlay, violin)
seg      — cell re-segmentation (Voronoi tessellation)
pipeline — composable analysis recipes (filter_xenium, …)

Typical workflow
----------------
>>> import astrolabe as ab
>>>
>>> # 1. Load Xenium run
>>> mdata = ab.io.read_xenium("Xenium_run/")
>>>
>>> # 2. (Optional) attach IF image and extract per-cell intensities
>>> ab.io.read_ome_tiff("panel.ome.tif", mdata=mdata)
>>> ab.pp.extract_if_intensities(mdata)
>>>
>>> # 3. QC + normalization  (pass mdata directly — default modality is 'xenium')
>>> ab.pp.calculate_qc_metrics(mdata)
>>> ab.pp.filter_cells(mdata, min_counts=10, max_pct_control=5.0)
>>> ab.pp.normalize_total(mdata)
>>> ab.pp.log1p(mdata)
>>>
>>> # 4. Expression-based clustering (cluster() defaults to the transcriptomic graph)
>>> ab.pp.highly_variable_genes(mdata)
>>> ab.tl.pca(mdata)
>>> ab.tl.transcriptional_neighbors(mdata)
>>> ab.tl.cluster(mdata, resolution=0.5)
>>>
>>> # 5. Spatial graph, for tl.moran/nhood_enrichment/etc.
>>> ab.tl.spatial_neighbors(mdata)
>>>
>>> # 6. Visualize
>>> ab.pl.points(mdata, color="leiden")
"""

import mudata

# Keep observation/feature metadata pulls explicit.
if hasattr(mudata, "set_options"):
    mudata.set_options(pull_on_update=False)

from astrolabe import io, pp, tl, pl, seg, pipeline, datasets
from astrolabe._summary import summary, Summary
from astrolabe._gpu import gpu, gpu_enabled, to_gpu, to_cpu, use_gpu

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    __version__ = _pkg_version("astrolabe")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "io", "pp", "tl", "pl", "seg", "pipeline", "datasets",
    "summary", "Summary",
    "gpu", "gpu_enabled", "to_gpu", "to_cpu", "use_gpu",
    "__version__",
]
