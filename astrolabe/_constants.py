"""
Shared schema constants used across astrolabe submodules.

These cover identity strings that appear in multiple independent modules
and would otherwise require a multi-file grep to rename.  They are
intentionally kept to a small, stable set.
"""

SCHEMA_VERSION = "0.1"
UNS_SCHEMA_VERSION_KEY = "astrolabe_schema_version"
UNS_DEFAULT_MODALITY_KEY = "default_modality"
UNS_XENIUM_SEGMENTATION_KEY = "xenium_native_segmentation"
"""``adata.uns`` flag set ``True`` by :func:`~astrolabe.io.read_xenium` when
the loaded cell/nucleus boundaries are the unmodified Xenium originals —
i.e. the same segmentation the transcript table's ``overlaps_nucleus`` /
``cell_id`` columns were computed against. Cleared (set ``False``) by any
function that replaces the segmentation (e.g.
:func:`~astrolabe.seg.voronoi_resegment`). Consumed by
:func:`~astrolabe.seg.create_compartment_layers` to decide whether the
Xenium-native ``overlaps_nucleus`` fast path is trustworthy."""
