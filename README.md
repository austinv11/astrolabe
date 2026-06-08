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