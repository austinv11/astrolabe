"""
Reader for 10x Genomics Xenium output directories.

Canonical call
--------------
>>> import astrolabe as ab
>>> mdata = ab.io.read_xenium("Xenium_run/")

The returned MuData object has a single ``'xenium'`` modality unless
``morphology_image=True`` is passed, in which case an
:class:`~astrolabe._utils.ImageSpec` for ``morphology.ome.tif`` is attached
under ``uns['spatial'][library_id]['images']['morphology']``. Passing
``he_image=True`` additionally attaches a paired H&E image as a separate
``'he'`` modality (affine-registered into the Xenium micron frame when an
alignment matrix is found or supplied via ``he_alignment``); see
:func:`read_xenium`'s docstring for the full set of override options. Neither
of these change ``uns['default_modality']``, which stays ``'xenium'``.

Schema
------
mdata['xenium']
    X           : (n_cells, n_genes+n_controls)  sparse float32, raw UMI counts
    obs         : cell_id (index), cell_area, nucleus_area,
                  transcript_counts, control_probe_counts,
                  control_codeword_counts, nucleus_count, z_level (if present)
    obsm
        spatial            : float32 (n_cells, 2)  [x_centroid, y_centroid] in µm
        cell_boundaries    : awkward (n_cells,) of (n_verts, 2) float32 polygons
        nucleus_boundaries : awkward (n_cells,) of (n_verts, 2) float32 polygons
    obsp        : (populated later by tl.spatial_neighbors)
    var         : gene_id (index), feature_type, is_control_probe
    uns
        spatial
            <library_id>
                coordinate_unit : 'microns'
                source_path     : str    (absolute path to the Xenium directory)
                experiment      : dict   (full experiment.xenium parsed JSON)
                images          : {}     (empty unless morphology_image=True)
        astrolabe_schema_version : str
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anndata as ad
import awkward as ak
import geopandas as gpd
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scanpy as sc
import scipy.sparse as sp
import copy

from astrolabe._constants import SCHEMA_VERSION, UNS_SCHEMA_VERSION_KEY, UNS_DEFAULT_MODALITY_KEY
from astrolabe.io._ome_tiff import _extract_image_spec

__all__ = [
    "read_xenium",
    "read_xenium_transcripts",
    "extract_morphology_intensities",
    "align_xenium_annotations",
]

log = logging.getLogger(__name__)

# Columns from cells.parquet we promote to obs; everything else is dropped.
_OBS_COLUMNS = [
    "cell_area",
    "nucleus_area",
    "transcript_counts",
    "control_probe_counts",
    "control_codeword_counts",
    "nucleus_count",
    "z_level",                    # present in Xenium 2.x multi-z outputs
    "x_centroid",                  # pulled into obsm, then dropped from obs
    "y_centroid",
]

# Columns always present in the cells table regardless of Xenium version.
_OBS_ALWAYS = [
    "cell_area",
    "nucleus_area",
    "transcript_counts",
    "control_probe_counts",
    "control_codeword_counts",
]

# Prefixes that flag a feature as a control in the feature matrix.
_CONTROL_PREFIXES = (
    "NegControlProbe_",
    "NegControlCodeword_",
    "UnassignedCodeword_",
    "DeprecatedCodeword_",
    "GenomicControl_",
    "BLANK_",
    "antisense_",
)

# Xenium cells table filenames in preference order (newer runs use parquet).
_CELLS_CANDIDATES = ["cells.parquet", "cells.csv.gz"]
_BOUNDARIES_CANDIDATES = ["cell_boundaries.parquet", "cell_boundaries.csv.gz"]
_NUCLEUS_BOUNDARIES_CANDIDATES = ["nucleus_boundaries.parquet", "nucleus_boundaries.csv.gz"]

# Common (but not universal — Xenium export conventions vary) filenames for the
# paired H&E image and its alignment matrix. Both are overridable per-call.
_DEFAULT_HE_IMAGE_FILENAME = "he_image.ome.tif"
_DEFAULT_HE_ALIGNMENT_FILENAME = "he_imagealignment.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_xenium(
    path: str | Path,
    *,
    library_id: str | None = None,
    load_boundaries: bool = True,
    load_transcripts: bool = True,
    morphology_image: bool = False,
    he_image: bool | str | Path = False,
    he_alignment: bool | str | Path | None = True,
    var_names: str = "gene_name",
    dtype: str = "float32",
) -> md.MuData:
    """Read a 10x Genomics Xenium output directory into a :class:`mudata.MuData`.

    Parameters
    ----------
    path
        Path to the Xenium output directory. Must contain
        ``cell_feature_matrix.h5`` and either ``cells.parquet`` or
        ``cells.csv.gz``.
    library_id
        Identifier stored under ``uns['spatial']``. Defaults to the
        directory name (``Path(path).name``).
    load_boundaries
        If ``True`` (default), load ``cell_boundaries.parquet`` and
        ``nucleus_boundaries.parquet`` and store polygon vertices as ragged
        arrays in ``obsm['cell_boundaries']`` and ``obsm['nucleus_boundaries']``
        (awkward arrays, shape ``(n_cells,)`` of variable-length (n_verts, 2)
        float32 arrays). Set ``False`` to skip for faster loading when
        boundary data are not needed.
    load_transcripts
        If ``True`` (default), stream ``transcripts.parquet`` via
        :func:`read_xenium_transcripts` (with its default filters: ``min_qv=20``,
        controls included) and store the result as a :class:`pyarrow.Table` in
        ``mdata['xenium'].uns['transcripts']``.  Set ``False`` to skip; the
        file can be large (100 M–500 M rows) and loading may take tens of seconds.
    morphology_image
        If ``True``, attach an :class:`~astrolabe._utils.ImageSpec` for
        ``morphology.ome.tif`` under
        ``uns['spatial'][library_id]['images']['morphology']``.  Metadata is
        extracted eagerly (file opened and closed immediately); no pixel data
        is read into memory.
    he_image
        Controls attaching a paired H&E image as a new ``'he'`` modality
        (via :func:`~astrolabe.io.read_he`). Export conventions for the
        auxiliary H&E file are not standardized across Xenium runs, so this
        does not assume a single fixed filename:

        - ``False`` (default): skip H&E entirely.
        - ``True``: look for ``he_image.ome.tif`` in *path*; if not found,
          log a warning and skip (same graceful behavior as *morphology_image*
          when no morphology file is present).
        - ``str`` / ``Path``: an explicit override — a bare filename
          (resolved relative to *path*) or an absolute path. Required to
          exist; raises ``FileNotFoundError`` if not, since this is an
          explicit request.
    he_alignment
        Controls the affine matrix used to register the H&E image into the
        Xenium micron frame (stored as ``ImageSpec.registration_affine``).
        Only consulted when *he_image* resolves to a real file.

        - ``True`` (default): look for ``he_imagealignment.csv`` in *path*
          (a plain ``(3, 3)`` comma-delimited matrix); if not found, log a
          warning and attach the H&E image **unregistered**
          (``registration_affine=None``) rather than skipping it.
        - ``False`` / ``None``: skip alignment lookup entirely and always
          attach unregistered.
        - ``str`` / ``Path``: an explicit override, resolved the same way as
          *he_image*. Required to exist; raises ``FileNotFoundError`` if not.
    var_names
        Which column of the feature matrix to use as ``var_names``.  One of
        ``'gene_name'`` (default, human-readable) or ``'gene_id'``
        (Ensembl/feature ID, guaranteed unique).
    dtype
        NumPy dtype for ``X``.  ``'float32'`` is the default and is
        sufficient for raw UMI counts up to ~16 M per cell.

    Returns
    -------
    mudata.MuData
        MuData with a single ``'xenium'`` modality (plus an optional
        ``'morphology'`` image attached to it if ``morphology_image=True``,
        and/or a separate ``'he'`` modality if ``he_image`` resolves to a
        file — ``uns['default_modality']`` remains ``'xenium'`` in both cases).
        When ``load_transcripts=True``, ``mdata['xenium'].uns['transcripts']``
        holds the filtered transcript detections as a :class:`pyarrow.Table`.

    Notes
    -----
    Control probes (``NegControlProbe_*``, ``NegControlCodeword_*``,
    ``BLANK_*``, etc.) are retained in ``X`` and flagged in
    ``var['is_control_probe']``.  Use :func:`astrolabe.pp.filter_genes` or
    boolean indexing on ``var`` to remove them before analysis.

    The spatial coordinate unit is always microns. ``obsm['spatial']`` has
    shape ``(n_cells, 2)`` with columns ``[x_centroid, y_centroid]``.
    """
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Xenium directory not found: {path}")

    library_id = library_id or path.name
    log.info("Reading Xenium run '%s' from %s", library_id, path)

    # ------------------------------------------------------------------ #
    # 1. experiment.xenium — parse metadata first so pixel_size is known  #
    # ------------------------------------------------------------------ #
    experiment = _read_experiment_json(path)
    pixel_size_um: float = float(
        experiment.get("pixel_size", experiment.get("pixel_size_um", 0.2125))
    )

    # ------------------------------------------------------------------ #
    # 2. cell_feature_matrix.h5  →  X + var                              #
    # ------------------------------------------------------------------ #
    h5_path = path / "cell_feature_matrix.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"cell_feature_matrix.h5 not found in {path}.  "
            "Ensure this is a complete Xenium output directory."
        )

    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata = _annotate_xenium_var(adata, var_names=var_names, dtype=dtype)

    # ------------------------------------------------------------------ #
    # 3. cells table  →  obs + obsm['spatial']                           #
    # ------------------------------------------------------------------ #
    cells_df = _read_cells_table(path)
    adata = _attach_obs(adata, cells_df)

    # ------------------------------------------------------------------ #
    # 4. Cell boundaries  →  obsm['cell_boundaries']                     #
    # ------------------------------------------------------------------ #
    if load_boundaries:
        _attach_boundaries(adata, path, _BOUNDARIES_CANDIDATES, "cell_boundaries")
        _attach_boundaries(adata, path, _NUCLEUS_BOUNDARIES_CANDIDATES, "nucleus_boundaries")

    # ------------------------------------------------------------------ #
    # 5. uns['spatial']                                                   #
    # ------------------------------------------------------------------ #
    adata.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "experiment": experiment,
            "images": {},
            "source_path": str(path),
        }
    }
    adata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION

    # ------------------------------------------------------------------ #
    # 6. Morphology image (optional, lazy)                                #
    # ------------------------------------------------------------------ #
    if morphology_image:
        _attach_morphology_image(adata, path, library_id)

    # ------------------------------------------------------------------ #
    # 7. Wrap in MuData                                                   #
    # ------------------------------------------------------------------ #
    mdata = md.MuData({"xenium": adata})
    # Shallow-copy so later top-level mutations of mdata.uns['spatial'] (e.g.
    # attach_modality() adding an 'he' library entry) don't alias into and
    # leak onto adata.uns['spatial'] (mdata['xenium']'s own uns) — they are
    # the same dict object if assigned directly.
    mdata.uns["spatial"] = dict(adata.uns["spatial"])
    mdata.uns[UNS_SCHEMA_VERSION_KEY] = SCHEMA_VERSION
    mdata.uns[UNS_DEFAULT_MODALITY_KEY] = "xenium"

    log.info(
        "Loaded %d cells × %d features (%d control probes)",
        adata.n_obs,
        adata.n_vars,
        adata.var["is_control_probe"].sum(),
    )

    # ------------------------------------------------------------------ #
    # 7b. Paired H&E image (optional, affine-aligned)                     #
    # ------------------------------------------------------------------ #
    if he_image:
        _attach_he_image(mdata, path, he_image, he_alignment, pixel_size_um=pixel_size_um)

    # ------------------------------------------------------------------ #
    # 8. Transcripts (optional)                                           #
    # ------------------------------------------------------------------ #
    if load_transcripts:
        tx_path = path / "transcripts.parquet"
        if not tx_path.exists():
            log.warning(
                "transcripts.parquet not found in %s; skipping uns['transcripts'].", path
            )
        else:
            read_xenium_transcripts(mdata)

    return mdata


def read_xenium_transcripts(
    mdata: "md.MuData",
    *,
    inplace: bool = True,
    cell_ids: list[str] | None = None,
    genes: list[str] | None = None,
    min_qv: float | None = 20.0,
    z_range: tuple[float, float] | None = None,
    include_controls: bool = True,
    columns: list[str] | None = None,
    chunksize: int = 500_000,
) -> "md.MuData | None":
    """Read Xenium transcript detections and store them in ``mdata``.

    Streams ``transcripts.parquet`` in chunks via PyArrow with predicate
    pushdown, so peak memory scales with rows that *pass* the filters rather
    than the total file size (commonly 100 M–500 M rows).

    The filtered :class:`pyarrow.Table` is stored as
    ``mdata['xenium'].uns['transcripts']``.  Functions that consume transcript
    data (:func:`~astrolabe.pp.calculate_xenium_qc_metrics`,
    :func:`~astrolabe.pl.gene_qv_scatter`) will use this key automatically
    when no explicit ``transcripts=`` argument is provided.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`read_xenium`.  The source
        directory is resolved from ``uns['spatial'][library_id]['source_path']``.
    inplace
        If ``True`` (default), store the table in *mdata* in place and return
        ``None``.  If ``False``, return a copy of *mdata* with the table
        added, leaving the original unmodified.
    cell_ids
        Restrict to transcripts assigned to these cell IDs.
    genes
        Restrict to transcripts for these gene/feature names.
    min_qv
        Minimum per-transcript Q-score (``qv`` column).  Default ``20``.
        Pass ``None`` to disable.
    z_range
        ``(z_min, z_max)`` in microns; filter by ``z_location``.
    include_controls
        If ``True`` (default), control-probe transcripts are included —
        matching ``X``, which always retains control-probe *genes*
        (flagged via ``var['is_control_probe']``, never removed on load).
        Pass ``False`` to exclude them, e.g. for analyses that only care
        about real genes and want a smaller table.
    columns
        Subset of parquet columns to load.  ``None`` loads all columns.
        Standard columns: ``transcript_id``, ``cell_id``, ``overlaps_nucleus``,
        ``feature_name``, ``x_location``, ``y_location``, ``z_location``, ``qv``.
    chunksize
        Rows per PyArrow batch.

    Returns
    -------
    None or mudata.MuData
        ``None`` when ``inplace=True``; a modified copy when ``inplace=False``.

    Examples
    --------
    >>> import astrolabe as ab
    >>> mdata = ab.io.read_xenium("Xenium_run/", load_transcripts=False)
    >>> ab.io.read_xenium_transcripts(mdata, genes=["EPCAM", "CD3D"])
    >>> mdata["xenium"].uns["transcripts"]  # pyarrow.Table
    """
    path = _source_path_from_mdata(mdata)
    tx_path = path / "transcripts.parquet"
    if not tx_path.exists():
        raise FileNotFoundError(f"transcripts.parquet not found at {tx_path}")

    log.info("Streaming transcripts from %s", tx_path)

    pf = pq.ParquetFile(tx_path)
    batches: list[pa.RecordBatch] = []

    for batch in pf.iter_batches(batch_size=chunksize, columns=columns):
        mask = pa.array([True] * len(batch), type=pa.bool_())

        if min_qv is not None and "qv" in batch.schema.names:
            mask = pc.and_(mask, pc.greater_equal(batch.column("qv"), min_qv))

        if cell_ids is not None and "cell_id" in batch.schema.names:
            cell_id_set = pa.array(cell_ids)
            mask = pc.and_(mask, pc.is_in(batch.column("cell_id"), value_set=cell_id_set))

        if genes is not None and "feature_name" in batch.schema.names:
            gene_set = pa.array(genes)
            mask = pc.and_(mask, pc.is_in(batch.column("feature_name"), value_set=gene_set))

        if z_range is not None and "z_location" in batch.schema.names:
            z_min, z_max = z_range
            mask = pc.and_(
                mask,
                pc.and_(
                    pc.greater_equal(batch.column("z_location"), z_min),
                    pc.less_equal(batch.column("z_location"), z_max),
                ),
            )

        if not include_controls and "feature_name" in batch.schema.names:
            is_control = _build_control_mask_arrow(batch.column("feature_name"))
            mask = pc.and_(mask, pc.invert(is_control))

        filtered = pa.RecordBatch.from_pydict(
            {name: pc.filter(batch.column(name), mask) for name in batch.schema.names},
            schema=batch.schema,
        )
        if len(filtered) > 0:
            batches.append(filtered)

    if not batches:
        schema = pq.read_schema(tx_path)
        if columns is not None:
            schema = pa.schema([schema.field(c) for c in columns if c in schema.names])
        table = pa.table({name: pa.array([], type=schema.field(name).type) for name in schema.names})
    else:
        table = pa.Table.from_batches(batches)

    log.info("Loaded %d transcript detections", len(table))

    if inplace:
        mdata["xenium"].uns["transcripts"] = table
        return None

    mdata_copy = copy.deepcopy(mdata)
    mdata_copy["xenium"].uns["transcripts"] = table
    return mdata_copy


def align_xenium_annotations(
    gdf: "gpd.GeoDataFrame",
    mdata: md.MuData,
    *,
    he_key: str = "he",
    image_key: str = "full",
    library_id: str | None = None,
) -> "gpd.GeoDataFrame":
    """Register a Xenium Explorer GeoJSON annotation export into the Xenium micron frame.

    Xenium Explorer's ROI/annotation GeoJSON export (the ``annotation.geojson``
    bundled with the public Human Lung Cancer FFPE sample, and any file
    exported the same way) stores coordinates as raw pixel indices into the
    paired *auxiliary H&E image* — not the morphology/DAPI image. This reuses
    the H&E modality's own ``registration_affine`` (the matrix parsed from
    ``he_imagealignment.csv`` by :func:`read_xenium`/:func:`read_he`) to map
    annotation vertices, which share the H&E image's raw pixel coordinate
    system, into the shared Xenium micron frame.

    Empirically verified against the public Human Lung Cancer FFPE sample: with
    this transform, points sampled inside all 11 real annotation regions land
    on real tissue signal in the morphology image (>99% average, checked
    against a max-intensity projection across the morphology image's z-stack
    since any single z-plane is sparse/out-of-focus in places) and every
    region's footprint sits within the H&E image's own registered footprint.
    A previous formula that treated annotation coordinates as morphology-image
    pixels (rather than H&E-image pixels) was substantially wrong.

    Requires ``read_xenium(..., he_image=True, he_alignment=True)`` (or an
    equivalent manual :func:`read_he` call with a real ``affine=``) so a valid
    ``registration_affine`` is available — annotation coordinates are only
    interpretable relative to the H&E image's own pixel grid, so alignment is
    impossible without it.

    Parameters
    ----------
    gdf
        GeoDataFrame as returned by :func:`read_annotations`, with raw
        (unregistered) H&E pixel coordinates.
    mdata
        :class:`~mudata.MuData` with an aligned H&E modality attached (e.g.
        via ``read_xenium(..., he_image=True)``).
    he_key
        Key of the H&E modality in *mdata*.
    image_key
        Key of the H&E image inside ``uns['spatial'][library_id]['images']``.
    library_id
        Library to read the H&E image from. ``None`` uses whichever library
        is first in ``uns['spatial']``.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of *gdf* with geometry replaced by registered micron coordinates.

    Examples
    --------
    >>> gdf = ab.io.read_annotations("run/annotation.geojson")
    >>> gdf = ab.io.align_xenium_annotations(gdf, mdata)
    >>> (ab.pl.points(mdata, color='total_counts') + ab.pl.annotation(gdf, color='name')).show()
    """
    import shapely
    from astrolabe._utils import _transform_coords

    he_adata = mdata[he_key]
    spatial = he_adata.uns["spatial"]
    lib = library_id or next(iter(spatial))
    images = spatial[lib]["images"]
    if image_key not in images:
        raise KeyError(
            f"align_xenium_annotations: no {image_key!r} image found in "
            f"mdata[{he_key!r}].uns['spatial'][{lib!r}]['images']. "
            "Call read_xenium(..., he_image=True) first."
        )
    spec = images[image_key]
    if spec.registration_affine is None:
        raise ValueError(
            f"align_xenium_annotations: the H&E image (mdata[{he_key!r}]) has no "
            "registration_affine (no he_imagealignment.csv was found/supplied). "
            "Annotation coordinates are recorded in the H&E image's own pixel "
            "space and cannot be placed in the Xenium micron frame without it."
        )
    M = spec.registration_affine

    def _transform(coords: np.ndarray) -> np.ndarray:
        return _transform_coords(coords.astype(np.float64), M)

    out = gdf.copy()
    geom_col = out.geometry.name
    out[geom_col] = out.geometry.apply(lambda geom: shapely.transform(geom, _transform))
    out = out.set_geometry(geom_col)
    out.crs = None
    return out


def extract_morphology_intensities(
    mdata: md.MuData,
    *,
    xenium_key: str = "xenium",
    morphology_key: str = "morphology",
    image_key: str = "morphology",
    boundaries_key: str = "cell_boundaries",
    library_id: str | None = None,
    reduction: str = "mean",
    batch_size: int = 512,
) -> None:
    """Extract per-cell morphology image intensities and store as a new modality.

    Requires ``morphology_image=True`` when calling :func:`read_xenium` so that
    the :class:`~astrolabe._utils.ImageSpec` is available.

    Creates ``mdata[morphology_key]`` with:

    * ``X`` — ``(n_cells, n_channels)`` float32 per-cell mean intensities.
    * ``obs_names`` — aligned with ``mdata[xenium_key].obs_names``.
    * ``var_names`` — channel names from the ImageSpec, or ``['ch0', 'ch1', …]``
      when names are absent.
    * ``obsm['spatial']`` — copied from the Xenium modality.

    Parameters
    ----------
    mdata
        :class:`~mudata.MuData` returned by :func:`read_xenium`.
    xenium_key
        Key for the Xenium modality.
    morphology_key
        Key under which the new morphology modality will be stored.
    image_key
        Key of the image inside ``uns['spatial'][library_id]['images']``.
        Defaults to ``'morphology'``, matching :func:`read_xenium`.
    boundaries_key
        Which boundary polygon set to use: ``'cell_boundaries'`` (default)
        or ``'nucleus_boundaries'``.
    library_id
        Spatial library identifier.  Inferred automatically when the Xenium
        modality has exactly one library.
    reduction
        Per-cell pixel aggregation: ``'mean'``, ``'median'``, ``'sum'``,
        ``'max'``, or ``'min'``.
    batch_size
        Cells per tile-read batch.

    Raises
    ------
    KeyError
        If the morphology image or boundary key are absent.
    """
    from astrolabe.pp._if import _extract_intensities_from_spec

    if xenium_key not in mdata.mod:
        raise KeyError(
            f"Modality '{xenium_key}' not found in MuData. "
            f"Available modalities: {list(mdata.mod.keys())}"
        )
    xen_adata = mdata[xenium_key]

    if boundaries_key not in xen_adata.obsm:
        raise KeyError(
            f"obsm['{boundaries_key}'] not found in the '{xenium_key}' modality.  "
            "Re-run read_xenium with load_boundaries=True."
        )

    # Resolve library_id
    spatial_uns: dict = xen_adata.uns.get("spatial", {})
    if library_id is None:
        ids = list(spatial_uns.keys())
        if len(ids) != 1:
            raise ValueError(
                f"Cannot infer library_id; found {ids}. Pass library_id explicitly."
            )
        library_id = ids[0]

    images = spatial_uns[library_id].get("images", {})
    if image_key not in images:
        raise KeyError(
            f"Image key '{image_key}' not found in uns['spatial']['{library_id}']['images'].  "
            f"Available: {list(images.keys())}.  "
            "Ensure read_xenium was called with morphology_image=True."
        )

    spec = images[image_key]

    result = _extract_intensities_from_spec(
        xen_adata.obsm[boundaries_key],
        spec,
        reduction=reduction,
        batch_size=batch_size,
    )

    # Build var: channel names from ImageSpec or generic fallback
    n_channels = result.shape[1]
    if spec.channel_names and len(spec.channel_names) == n_channels:
        var_names = list(spec.channel_names)
    else:
        var_names = [f"ch{i}" for i in range(n_channels)]

    morph_obs = pd.DataFrame(index=xen_adata.obs_names.copy())
    morph_obs.index.name = xen_adata.obs_names.name or "cell_id"

    morph_var = pd.DataFrame(index=pd.Index(var_names, name="channel"))

    morph_adata = ad.AnnData(X=result, obs=morph_obs, var=morph_var)

    if "spatial" in xen_adata.obsm:
        morph_adata.obsm["spatial"] = xen_adata.obsm["spatial"].copy()

    morph_adata.uns["spatial"] = {
        library_id: {
            "coordinate_unit": "microns",
            "images": {image_key: spec},
            "source_path": spatial_uns[library_id].get("source_path", ""),
        }
    }
    morph_adata.uns["boundaries_key"] = boundaries_key

    mdata.mod[morphology_key] = morph_adata
    mdata.update()

    log.info(
        "extract_morphology_intensities: created '%s' modality — %d cells × %d channels",
        morphology_key, morph_adata.n_obs, morph_adata.n_vars,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _source_path_from_mdata(mdata: md.MuData) -> Path:
    """Extract the Xenium output directory path stored in a MuData's uns."""
    spatial = mdata.uns.get("spatial", {})
    for lib_meta in spatial.values():
        if "source_path" in lib_meta:
            return Path(lib_meta["source_path"])
    raise ValueError(
        "Cannot resolve the Xenium source directory from this MuData object. "
        "Ensure it was created by ab.io.read_xenium()."
    )


def _read_experiment_json(path: Path) -> dict:
    """Parse experiment.xenium; return empty dict if absent (older runs)."""
    exp_path = path / "experiment.xenium"
    if not exp_path.exists():
        log.warning("experiment.xenium not found in %s; metadata will be incomplete", path)
        return {}
    with exp_path.open() as fh:
        return json.load(fh)


def _annotate_xenium_var(
    adata: ad.AnnData,
    *,
    var_names: str,
    dtype: str,
) -> ad.AnnData:
    """Post-hoc annotation of var from sc.read_10x_h5 output.

    Normalizes column names, optionally reindexes to gene_id, adds the
    is_control_probe flag, and casts X to the requested dtype.
    """
    if var_names not in ("gene_name", "gene_id"):
        raise ValueError(f"var_names must be 'gene_name' or 'gene_id', got {var_names!r}")

    # scanpy names the column 'feature_types'; normalize to 'feature_type'
    if "feature_types" in adata.var.columns:
        adata.var = adata.var.rename(columns={"feature_types": "feature_type"})

    # Preserve both ID and name columns (scanpy may have set index to names)
    if "gene_ids" in adata.var.columns:
        adata.var = adata.var.rename(columns={"gene_ids": "gene_id"})
    if "gene_names" in adata.var.columns:
        adata.var = adata.var.rename(columns={"gene_names": "gene_name"})

    # If user wants gene_id as index, swap (gene_id column must exist)
    if var_names == "gene_id" and "gene_id" in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.to_numpy()
        adata.var_names = pd.Index(adata.var["gene_id"], name="gene_id")
    else:
        adata.var.index.name = "gene_name"

    feature_type = adata.var.get("feature_type", pd.Series(dtype=str))
    adata.var["is_control_probe"] = _flag_control_probes(
        adata.var_names.to_numpy(dtype=str),
        feature_type.to_numpy(dtype=str) if len(feature_type) > 0
        else np.full(adata.n_vars, "Gene Expression"),
    )

    if dtype != "float32":
        adata.X = adata.X.astype(dtype) if not sp.issparse(adata.X) else adata.X.astype(dtype)

    return adata


def _flag_control_probes(names: np.ndarray, feature_types: np.ndarray) -> np.ndarray:
    """Return a bool array: True where a feature is a control probe/codeword."""
    by_name = np.array(
        [any(n.startswith(p) for p in _CONTROL_PREFIXES) for n in names],
        dtype=bool,
    )
    # Also flag anything whose feature_type is not 'Gene Expression'
    by_type = feature_types != "Gene Expression"
    return by_name | by_type


def _read_cells_table(path: Path) -> pd.DataFrame:
    """Read cells.parquet (or cells.csv.gz fallback) into a DataFrame."""
    for candidate in _CELLS_CANDIDATES:
        cells_path = path / candidate
        if cells_path.exists():
            log.debug("Reading cells table from %s", cells_path)
            if cells_path.suffix == ".parquet":
                return pd.read_parquet(cells_path)
            else:
                return pd.read_csv(cells_path, compression="gzip")
    raise FileNotFoundError(
        f"No cells table found in {path}. "
        f"Expected one of: {_CELLS_CANDIDATES}"
    )


def _attach_obs(adata: ad.AnnData, cells_df: pd.DataFrame) -> ad.AnnData:
    """Merge cells table into adata.obs and extract obsm['spatial'].

    The cells table is indexed by cell_id (barcode) and must align with
    adata.obs_names.  We do an explicit reindex so any mismatch is caught
    immediately rather than silently producing NaN rows.
    """
    # Normalize index: cells table may use 'cell_id' as a column or as index
    if "cell_id" in cells_df.columns:
        cells_df = cells_df.set_index("cell_id")
    cells_df.index = cells_df.index.astype(str)

    # Verify alignment
    missing = set(adata.obs_names) - set(cells_df.index)
    if missing:
        n = len(missing)
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{n} cell IDs in the feature matrix are absent from the cells "
            f"table.  First few: {sample}.  Check that the files come from "
            f"the same Xenium run."
        )

    cells_aligned = cells_df.reindex(adata.obs_names)

    # spatial coordinates → obsm
    spatial = cells_aligned[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    adata.obsm["spatial"] = spatial

    # Retain only the declared obs columns (drop centroid columns + anything unknown)
    obs_keep = [
        c for c in _OBS_ALWAYS + ["nucleus_count", "z_level"]
        if c in cells_aligned.columns
    ]
    adata.obs = cells_aligned[obs_keep].copy()
    adata.obs.index.name = "cell_id"

    return adata


def _attach_boundaries(
    adata: ad.AnnData,
    path: Path,
    candidates: list[str],
    obsm_key: str,
) -> None:
    """Load a boundary parquet and store as an awkward ragged array in obsm.

    The boundary table must have columns (cell_id, vertex_x, vertex_y).
    Falls back gracefully if the file is unavailable.
    """
    for candidate in candidates:
        bnd_path = path / candidate
        if bnd_path.exists():
            break
    else:
        log.warning(
            "No boundary file found in %s (%s). Skipping obsm['%s'].",
            path,
            candidates,
            obsm_key,
        )
        return

    log.debug("Reading %s from %s", obsm_key, bnd_path)
    cols = ["cell_id", "vertex_x", "vertex_y"]
    bnd_df = (
        pd.read_parquet(bnd_path, columns=cols)
        if bnd_path.suffix == ".parquet"
        else pd.read_csv(bnd_path, compression="gzip", usecols=cols)
    )

    n_cells = adata.n_obs

    # Map each cell_id string → its integer position in adata.obs_names.
    # Cells absent from the boundary file get NaN and are treated as empty.
    cell_pos = pd.Series(np.arange(n_cells, dtype=np.int32), index=adata.obs_names)
    cell_idx = bnd_df["cell_id"].astype(str).map(cell_pos)

    valid = cell_idx.notna()
    if not valid.all():
        bnd_df = bnd_df[valid]
        cell_idx = cell_idx[valid]

    cell_idx_arr = cell_idx.to_numpy(dtype=np.int32)

    # Stable sort so vertices within each cell stay in their original file order.
    order = np.argsort(cell_idx_arr, kind="stable")
    cell_idx_sorted = cell_idx_arr[order]
    coords = bnd_df[["vertex_x", "vertex_y"]].to_numpy(dtype=np.float32)[order]

    # Build offsets from per-cell vertex counts.
    counts = np.bincount(cell_idx_sorted, minlength=n_cells).astype(np.int64)
    offsets = np.empty(n_cells + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    # Construct the awkward array directly — no Python loop over cells.
    adata.obsm[obsm_key] = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.contents.NumpyArray(coords),
        )
    )


def _attach_morphology_image(
    adata: ad.AnnData, path: Path, library_id: str
) -> None:
    """Build an ImageSpec for the morphology image and store it in uns.

    Prefers ``morphology_focus/`` (2D per-channel autofocus images stored as
    a multi-file OME-TIFF) when the directory exists.  Falls back to
    ``morphology.ome.tif``.  Metadata is extracted eagerly (file opened and
    closed immediately); no pixel data is loaded.

    The ImageSpec is stored under ``images['morphology']`` regardless of
    which source was used.
    """
    morph_path: Path | None = None
    source_label: str = ""

    # Prefer morphology_focus/: 2D per-channel autofocus images (multi-file OME-TIFF).
    # The first file alphabetically is the OME-TIFF master; tifffile follows
    # UUID links to assemble all channels transparently.
    focus_dir = path / "morphology_focus"
    if focus_dir.is_dir():
        candidates = sorted(
            f for f in focus_dir.glob("morphology_focus_*.ome.tif")
            if not f.name.startswith(".")
        )
        if not candidates:
            candidates = sorted(
                f for f in (
                    list(focus_dir.glob("*.ome.tif")) + list(focus_dir.glob("*.ome.tiff"))
                )
                if not f.name.startswith(".")
            )
        if candidates:
            morph_path = candidates[0]
            source_label = f"morphology_focus/{morph_path.name}"

    # Fall back to morphology.ome.tif
    if morph_path is None:
        fallback = path / "morphology.ome.tif"
        if fallback.exists():
            morph_path = fallback
            source_label = "morphology.ome.tif"

    if morph_path is None:
        log.warning(
            "No morphology image found in %s (checked morphology_focus/ and morphology.ome.tif).",
            path,
        )
        return

    spec = _extract_image_spec(morph_path)
    adata.uns["spatial"][library_id]["images"]["morphology"] = spec

    log.info(
        "Attached morphology image from %s: %d level(s), shape=%s, pixel_size=%.4f µm",
        source_label,
        spec.n_levels,
        spec.full_shape,
        spec.pixel_size_um,
    )


def _resolve_optional_file(
    path: Path,
    value: bool | str | Path | None,
    default_filename: str,
    label: str,
) -> Path | None:
    """Resolve a ``he_image``/``he_alignment``-style tri-state parameter to a path.

    ``True`` tries *default_filename* under *path*, logging a warning and
    returning ``None`` if it doesn't exist (soft, auto-discovery semantics).
    ``False``/``None`` returns ``None`` (explicitly skip). A ``str``/``Path``
    is an explicit override (bare filename resolved relative to *path*, or an
    absolute path) that must exist — raises ``FileNotFoundError`` if not,
    since the caller named it explicitly.
    """
    if value is False or value is None:
        return None
    if value is True:
        candidate = path / default_filename
        if not candidate.exists():
            log.warning(
                "%s: default file '%s' not found; skipping.", label, candidate
            )
            return None
        return candidate

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = path / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"{label}: specified file not found: {candidate}")
    return candidate


def _attach_he_image(
    mdata: md.MuData,
    path: Path,
    he_image: bool | str | Path,
    he_alignment: bool | str | Path | None,
    *,
    pixel_size_um: float,
) -> None:
    """Locate, optionally align, and attach a paired H&E image as ``mdata['he']``.

    File locations are not assumed to follow a single fixed convention across
    Xenium exports — *he_image*/*he_alignment* accept ``True`` (try the
    common default filename, skipping gracefully if absent), an explicit
    filename/path override (required to exist), or ``False``/``None`` to skip.
    An alignment matrix is optional: when none is found or supplied, the H&E
    image is attached unregistered (``registration_affine=None``) rather than
    being skipped entirely.
    """
    img_path = _resolve_optional_file(path, he_image, _DEFAULT_HE_IMAGE_FILENAME, "he_image")
    if img_path is None:
        return

    align_path = _resolve_optional_file(
        path, he_alignment, _DEFAULT_HE_ALIGNMENT_FILENAME, "he_alignment"
    )
    affine: np.ndarray | None = None
    if align_path is not None:
        # 10x's imagealignment.csv is a (3, 3) matrix mapping the auxiliary
        # (H&E) image's own raw pixel coordinates to the Xenium morphology
        # image's raw pixel grid -- a pixel-index-to-pixel-index mapping, NOT
        # a pixel->micron one. Empirically verified against the public Human
        # Lung Cancer FFPE sample: the matrix's linear part has magnitude
        # ~1.29, matching he_pixel_size_um / morph_pixel_size_um exactly (a
        # pure pixel-grid unit conversion), and applying it directly (no
        # inversion) to the H&E image's own corners plus all 11 real
        # pathologist-annotation regions lands them on real tissue in the
        # morphology image and within the H&E image's own footprint --
        # whereas inverting it (the previous approach) does not. Converting
        # the mapped morphology-pixel coordinates to microns needs the run's
        # own pixel_size_um (morphology and H&E can have different physical
        # pixel sizes; only the morphology/Xenium one defines the micron
        # frame).
        raw_affine = np.loadtxt(align_path, delimiter=",")
        scale = np.array(
            [[pixel_size_um, 0.0, 0.0], [0.0, pixel_size_um, 0.0], [0.0, 0.0, 1.0]]
        )
        affine = scale @ raw_affine
    else:
        log.info(
            "No H&E alignment matrix found/specified for %s; "
            "attaching H&E image unregistered.",
            img_path,
        )

    from astrolabe.io._he import read_he
    # library_id intentionally left at read_he's own default (file stem) so
    # the H&E image gets its own uns['spatial'] entry rather than colliding
    # with the Xenium modality's library_id.
    read_he(img_path, affine=affine, mdata=mdata, mdata_key="he")

    log.info(
        "Attached H&E image from %s (%s)",
        img_path,
        "aligned" if affine is not None else "unregistered",
    )


def _build_control_mask_arrow(feature_name_col) -> pa.Array:
    """Return a PyArrow boolean array: True where feature_name is a control."""
    mask = None
    for prefix in _CONTROL_PREFIXES:
        m = pc.starts_with(feature_name_col, pattern=prefix)
        mask = m if mask is None else pc.or_(mask, m)
    return mask