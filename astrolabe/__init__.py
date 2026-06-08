"""
astrolabe — lightweight spatial transcriptomics analysis in Python.

A scanpy-idiomatic library built on MuData (multi-modal AnnData) as its
core data structure.  Designed as a lightweight alternative to SpatialData
for downstream analysis of pre-segmented, pre-aligned spatial data.

Submodules
----------
io  — data readers (Xenium, OME-TIFF)
pp  — preprocessing (QC, normalization, IF intensity extraction)
tl  — analysis tools (spatial graph, clustering, spatial statistics)
pl  — plotting (spatial scatter, image overlay, violin)

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
>>> ab.pp.filter_cells(mdata, min_counts=10, max_pct_control_probe=20.0)
>>> ab.pp.normalize_total(mdata)
>>> ab.pp.log1p(mdata)
>>>
>>> # 4. Build spatial graph + cluster
>>> ab.tl.spatial_neighbors(mdata)
>>> ab.tl.cluster(mdata, resolution=0.5)
>>>
>>> # 5. Visualise
>>> ab.pl.spatial_scatter(mdata, color="leiden")
"""

import mudata

# Opt into mudata 0.4 behaviour now: update() does not pull obs/var columns
# from individual modalities. Use mdata.pull_obs()/pull_var() explicitly.
mudata.set_options(pull_on_update=False)

from astrolabe import io, pp, tl, pl

__version__ = "0.1.0"

__all__ = ["io", "pp", "tl", "pl", "__version__"]