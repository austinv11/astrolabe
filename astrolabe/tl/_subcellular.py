"""Geometry-anchored subcellular transcript localization.

Where is a gene's mRNA inside the cell -- in the nucleus, at the nuclear
envelope, out in the cytoplasm, or against the plasma membrane? This module
answers that with a coordinate built from the **actual segmentation polygons**,
and with an inference structure whose replication unit is the cell.

The design rule the whole module is built on:

    Whatever transformation the data goes through, the null must go through
    the identical transformation.

The predecessor of this module (a SOM binning transcript offsets in a frame
derived from the transcript cloud's *own* second moments, compared against a
fixed uniform disc) violated that rule three ways at once, and the result was
that ~76% of the variance in its headline gene x domain figure was a single
baseline shared by every gene. See ``.claude/findings/som_audit.md``.

The coordinate
--------------
Per transcript, from the cell and nucleus polygons:

.. code-block:: text

    u = d_ncent / (d_ncent + d_nuc)          inside the nucleus   -> [0, 1]
    u = 1 + d_nuc / (d_nuc + d_cell)         in the cytoplasm     -> [1, 2]

``0`` = nucleus centre, ``1`` = nuclear envelope, ``2`` = plasma membrane.
``d_nuc``/``d_cell`` are distances to the nucleus/cell *boundary* and
``d_ncent`` to the nucleus centroid. Nothing in it is estimated from the
transcripts, so it cannot studentize away the quantity being measured.

Each cell's polygon is then rasterized once, ``u`` evaluated at every interior
pixel, and each transcript's ``u`` mapped through that cell's own empirical CDF:

.. code-block:: text

    v = F_c(u)

Under uniform-in-polygon placement ``v ~ Uniform(0, 1)`` for **every cell
regardless of its geometry**, so cell size, shape, concavity and
nucleus-to-cell area ratio are absorbed exactly rather than approximately.
This is the probability integral transform, the construction behind
``spatstat``'s ``cdf.test`` (Baddeley, Rubak & Turner 2015,
*Spatial Point Patterns*, CRC).

Two layers of inference
-----------------------
**Layer 1 (headline, relative).** Cochran--Mantel--Haenszel / conditional
logistic regression **stratified by cell**: is gene *g* in this compartment
more often than the *other transcripts of the same cell*? Every per-cell
nuisance -- the 3-D-to-2-D projection kernel, segmentation error, z-extent,
optical-crowding censoring, cell size -- acts identically on all genes within a
cell and therefore cancels exactly in a stratum-conditional comparison. That
matters because the projection bias alone is first-order: a *perfectly
uniformly filled* spherical cell projects to a 2-D disc with
``E[r] = 0.589`` against ``2/3``, and a 40% excess of mass in the inner
quarter-area. An absolute uniform-in-polygon test would report that geometry as
nuclear enrichment for every gene in the panel.

**Layer 2 (absolute).** Berman's Z1 (Berman 1986, *Appl Stat* 35:54-62;
``spatstat::berman.test``) on ``v``, per cell, aggregated across cells by a
Wilcoxon signed-rank test. Because ``v`` is the PIT coordinate its null mean is
exactly ``1/2`` and variance exactly ``1/12`` for every cell, so the per-cell
statistic is standardized without any weighting hack.

Caveats worth stating in any writeup
------------------------------------
* Layer 1 measures localization **relative to the panel**. With a targeted
  panel that is the only well-posed question -- a panel-wide effect is not
  identifiable -- but it does mean "not significant" never means "uniform".
* Expect the MH log odds ratio to **correlate with gene abundance** on real
  data (measured on the Xenium FFPE lung reference: Spearman -0.61 against
  transcript count for the nuclear contrast, +0.65 for cell-edge). This is not
  an estimator artifact -- under a simulated null with a 375x abundance range
  and a single gene holding 47% of all transcripts, the correlation is -0.09
  and the false-positive rate is 0. It is structure in the data, and nuclear
  retention of low-abundance transcripts is a documented mechanism that would
  produce exactly it (Battich et al. 2013; Bahar Halpern et al. 2015): a gene
  with few mature cytoplasmic copies has a larger *fraction* of its molecules
  still nuclear. The ``abundance_rank`` column is reported so any specific
  finding can be checked against it rather than assumed independent of it.
* Layer 2 is absolute and therefore carries the projection bias above.
* On FFPE tissue the cytoplasmic shell can be thin (~1.2 um median on the
  Xenium FFPE lung reference), comparable to segmentation error. The
  nucleus-versus-not contrast then carries most of the power and the finer
  ``cytoplasm``/``cell_edge`` split is noisy. ``obs[... '_shell_um']`` reports
  it per cell so this is checkable rather than assumed.

References
----------
Battich et al. (2013) *Nat Methods* 10:1127-1133.
Samacoits et al. (2018) *Nat Commun* 9:4127.
Stueland et al. (2019) *Sci Rep* 9:8267 (RNA Distribution Index).
Mah et al. (2024) *Genome Biol* 25:82 (Bento; the compartment vocabulary).
Mantel & Haenszel (1959) *JNCI* 22:719-748; Robins, Breslow & Greenland
(1986) *Biometrics* 42:311-323 (the variance estimator).
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._shapes import _build_polygons
from astrolabe._utils import _resolve_transcripts, from_kwarg, with_adata

__all__ = ["subcellular_coords", "subcellular_enrichment", "subcellular_indices"]

log = logging.getLogger(__name__)

#: Compartment names, ordered nucleus -> periphery. Deliberately Bento's
#: vocabulary (Mah et al. 2024) minus its ``none`` class: Bento assigns
#: patterns with a supervised random forest that can decline to label, while
#: these are a partition of position, so every transcript gets exactly one.
COMPARTMENTS: tuple[str, ...] = ("nucleus", "nuclear_edge", "cytoplasm", "cell_edge")

#: Bin count for the stored uniform-in-polygon ``u`` histogram. The pooled
#: null density is what turns a raw profile into an enrichment: ``u`` is a
#: *radial fraction*, so its area element is not constant and uniformly
#: placed transcripts already produce a rising curve inside the nucleus.
#: Dividing by this histogram cancels that geometry exactly.
NULL_PROFILE_BINS: int = 200

_CELL_CHUNK = 4000  # cells per rasterization batch
_TX_CHUNK = 1_000_000  # transcripts per geometry batch


# ──────────────────────────────────────────────────────────────────────────────
# geometry
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_geometry(adata, cell_boundaries_key, nucleus_boundaries_key):
    """Build per-cell polygons, boundaries and nucleus centroids.

    Returns ``(cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent, usable)`` --
    object arrays of length ``n_obs`` plus a bool mask of cells with *both* a
    buildable cell polygon and a buildable nucleus polygon. Precomputing the
    boundary/centroid geometries once is what makes the per-transcript pass a
    handful of vectorized shapely calls rather than a Python loop.
    """
    import shapely

    for key in (cell_boundaries_key, nucleus_boundaries_key):
        if key not in adata.obsm:
            raise KeyError(
                f"obsm['{key}'] not found. Geometry-anchored subcellular "
                "analysis needs both cell and nucleus boundary polygons."
            )

    cell_poly, cell_bad, cell_rep = _build_polygons(adata.obsm[cell_boundaries_key])
    nuc_poly, nuc_bad, nuc_rep = _build_polygons(adata.obsm[nucleus_boundaries_key])
    if cell_rep or nuc_rep:
        log.info(
            "subcellular: repaired %d cell and %d nucleus ring(s) via make_valid",
            cell_rep, nuc_rep,
        )

    cell_poly = np.asarray(cell_poly, dtype=object)
    nuc_poly = np.asarray(nuc_poly, dtype=object)
    usable = ~(cell_bad | nuc_bad)

    cell_bnd = np.empty(len(cell_poly), dtype=object)
    nuc_bnd = np.empty(len(nuc_poly), dtype=object)
    nuc_cent = np.full((len(nuc_poly), 2), np.nan)
    area = np.zeros(len(cell_poly))
    if usable.any():
        idx = np.flatnonzero(usable)
        cell_bnd[idx] = shapely.boundary(cell_poly[idx])
        nuc_bnd[idx] = shapely.boundary(nuc_poly[idx])
        cent = shapely.centroid(nuc_poly[idx])
        nuc_cent[idx, 0] = shapely.get_x(cent)
        nuc_cent[idx, 1] = shapely.get_y(cent)
        area[idx] = shapely.area(cell_poly[idx])
        # Prepared geometries cache a GEOS spatial index per polygon, which
        # makes the millions of repeated containment queries below ~8x cheaper.
        shapely.prepare(cell_poly[idx])
        shapely.prepare(nuc_poly[idx])
    return cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent, area, usable


def _radial_u(px, py, ci, cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent):
    """Evaluate the ``[0, 2]`` radial coordinate at arbitrary points.

    *ci* gives each point's cell index. Returns
    ``(u, d_nuc_signed, d_cell, in_nucleus, in_cell)`` with ``d_nuc_signed``
    negative inside the nucleus (the usual signed-distance convention, so
    "distance from the envelope" is ``abs``).
    """
    import shapely

    pts = shapely.points(px, py)
    d_cell = shapely.distance(pts, cell_bnd[ci])
    d_nuc = shapely.distance(pts, nuc_bnd[ci])
    d_ncent = np.hypot(px - nuc_cent[ci, 0], py - nuc_cent[ci, 1])
    in_nuc = shapely.contains_xy(nuc_poly[ci], px, py)
    in_cell = shapely.contains_xy(cell_poly[ci], px, py)

    tiny = 1e-12
    u_in = d_ncent / np.maximum(d_ncent + d_nuc, tiny)
    u_out = 1.0 + d_nuc / np.maximum(d_nuc + d_cell, tiny)
    u = np.where(in_nuc, u_in, u_out)
    return (
        np.clip(u, 0.0, 2.0),
        np.where(in_nuc, -d_nuc, d_nuc),
        d_cell,
        in_nuc,
        in_cell,
    )


def _assign_compartments(u, d_nuc_signed, d_cell, mode, bins, nuclear_edge_um,
                         cell_edge_um):
    """Integer compartment code per point, indexing :data:`COMPARTMENTS`."""
    if mode == "relative":
        b0, b1, b2 = bins
        if not 0.0 < b0 < b1 < b2 < 2.0:
            raise ValueError(
                f"compartment_bins must satisfy 0 < b0 < b1 < b2 < 2, got {bins}"
            )
        return np.digitize(u, [b0, b1, b2]).astype(np.int8)
    if mode == "metric":
        code = np.full(len(u), 2, dtype=np.int8)  # cytoplasm
        code[u < 1.0] = 0                          # nucleus
        code[np.abs(d_nuc_signed) <= nuclear_edge_um] = 1
        edge = (d_cell <= cell_edge_um) & (u >= 1.0)
        code[edge & (code != 1)] = 3
        return code
    raise ValueError(
        f"compartment_mode must be 'relative' or 'metric', got {mode!r}"
    )


def _raster_points(poly, step):
    """Pixel-centre lattice covering *poly*'s bounding box."""
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + step / 2.0, maxx, step)
    ys = np.arange(miny + step / 2.0, maxy, step)
    if len(xs) == 0:
        xs = np.array([(minx + maxx) / 2.0])
    if len(ys) == 0:
        ys = np.array([(miny + maxy) / 2.0])
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


# ──────────────────────────────────────────────────────────────────────────────
# tl.subcellular_coords
# ──────────────────────────────────────────────────────────────────────────────
@with_adata
def subcellular_coords(
    adata: ad.AnnData,
    transcripts=None,
    *,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    spatial_key: str = "spatial",
    cell_boundaries_key: str = "cell_boundaries",
    nucleus_boundaries_key: str = "nucleus_boundaries",
    groupby: str | None = None,
    groups: list[str] | None = None,
    compartment_mode: str = "relative",
    compartment_bins: tuple[float, float, float] = (0.7, 1.3, 1.7),
    nuclear_edge_um: float = 1.0,
    cell_edge_um: float = 1.0,
    null_points_per_cell: int = 200,
    null_pixel_um: float | None = None,
    min_shell_um: float = 0.0,
    min_cell_transcripts: int = 1,
    key_added: str = "subcellular",
) -> pd.DataFrame:
    """Place every transcript in a geometry-anchored intracellular coordinate.

    Computes the ``u in [0, 2]`` radial coordinate described in the module
    docstring (``0`` nucleus centre, ``1`` nuclear envelope, ``2`` plasma
    membrane), its per-cell probability-integral transform ``v``, a named
    compartment, and a bearing in the cell's nucleus-anchored polarity frame.
    Nothing here is estimated from the transcripts themselves.

    Every cell polygon is rasterized once at *null_pixel_um* to obtain that
    cell's exact uniform-in-polygon null: the empirical CDF that defines ``v``,
    the null occupancy of each compartment, and three moments
    (``mean x``, ``mean y``, ``mean |p|^2``) from which
    :func:`subcellular_indices` reconstructs the null second moment about *any*
    centre analytically, so no cell is ever rasterized twice.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    transcripts
        Transcript table (:class:`pyarrow.Table` or :class:`pandas.DataFrame`).
        ``None`` (default) uses the modality's own ``uns['transcripts']``.
    x_col, y_col
        Transcript coordinate columns (um).
    feature_col
        Gene / feature name column.
    cell_id_col
        Column identifying each transcript's cell; values must match ``obs_names``.
    spatial_key
        ``obsm`` key holding cell centroids (um), used for the bearing.
    cell_boundaries_key, nucleus_boundaries_key
        ``obsm`` keys holding the polygons. **Both are required** -- the
        coordinate is nucleus-referenced by construction.
    groupby
        ``obs`` column (e.g. a cluster or cell-type label) added to the output
        as a ``group`` column, so downstream analysis can be run per cell type.
        Localization programs are cell-type-specific; pooling lineages can
        cancel opposite effects.
    groups
        Restrict to these *groupby* values.
    compartment_mode
        ``'relative'`` (default) cuts ``u`` at *compartment_bins* -- scale-free,
        and the only workable choice when the cytoplasmic shell is thin.
        ``'metric'`` defines the two edge compartments by physical distance to
        their membrane (*nuclear_edge_um*, *cell_edge_um*), which is closer to
        what "edge" means biologically but degenerates when the whole shell is
        thinner than the band.
    compartment_bins
        ``(b0, b1, b2)`` cuts on ``u`` for ``'relative'`` mode. The default
        ``(0.7, 1.3, 1.7)`` puts ``nuclear_edge`` symmetrically about the
        envelope at ``u = 1``.
    nuclear_edge_um, cell_edge_um
        Band half-widths for ``'metric'`` mode.
    null_points_per_cell
        Target interior pixel count per cell. The lattice step is derived per
        cell as ``sqrt(area / n)``, so every cell's null CDF gets the same
        resolution (``1 / n``) and a large cell costs no more to evaluate than
        a small one -- which is what keeps the runtime linear in cells rather
        than in tissue area. Ignored when *null_pixel_um* is given.
    null_pixel_um
        Fixed rasterization step in um, overriding *null_points_per_cell*. Use
        when you want a stated physical resolution rather than a stated number
        of quantiles. Either way the lattice is deterministic -- not a Monte
        Carlo sample -- so results are reproducible without a seed.
    min_shell_um
        Exclude cells whose mean cytoplasmic shell thickness is below this.
        ``0.0`` (default) excludes none but still reports the value.
    min_cell_transcripts
        Exclude cells with fewer assigned transcripts.
    key_added
        Prefix for stored outputs.

    Stores
    ------
    ``obs[key_added + '_n_transcripts']``
        Assigned, in-polygon transcript count per cell.
    ``obs[key_added + '_shell_um']``
        Mean cytoplasmic shell thickness (um); ``NaN`` for excluded cells.
    ``obsm[key_added + '_compartment_fractions']``
        Float32 ``(n_obs, 4)`` observed occupancy, columns :data:`COMPARTMENTS`.
    ``obsm[key_added + '_null_fractions']``
        Float32 ``(n_obs, 4)`` uniform-in-polygon null occupancy for that cell's
        own geometry.
    ``obsm[key_added + '_null_moments']``
        Float32 ``(n_obs, 4)``: ``mean x``, ``mean y``, ``mean |p|^2`` of the
        cell's mask pixels, and the pixel count.
    ``obsm[key_added + '_nucleus_centroid']``
        Float32 ``(n_obs, 2)`` nucleus polygon centroid (um).
    ``uns[key_added]``
        ``compartment_names``, ``pooled_fractions`` (the panel-wide occupancy
        that is Layer 1's reference), ``null_profile`` (the density of ``u``
        under uniform-in-polygon placement, over ``null_profile_bins`` equal
        bins of ``[0, 2]`` -- the reference
        :func:`~astrolabe.pl.subcellular_profile` divides by, since ``u`` is a
        radial fraction whose area element makes even uniform placement rise
        inside the nucleus), ``params``, and exclusion counts.

    Returns
    -------
    pd.DataFrame
        One row per retained transcript: ``cell_id``, ``gene``, ``x``, ``y``
        (um, for plotting), ``u``, ``v``,
        ``d_nucleus_um`` (signed, negative inside), ``d_membrane_um``,
        ``angle`` (radians in the cell's polarity frame, ``NaN`` without one),
        ``compartment``, and ``group`` when *groupby* is given.

    Examples
    --------
    >>> coords = ab.tl.subcellular_coords(mdata)              # uses uns['transcripts']
    >>> res = ab.tl.subcellular_enrichment(mdata, coords)
    >>> ab.pl.subcellular_profile(mdata, coords, genes=['ACTA2', 'CAPN8']).show()
    """
    if not adata.obs_names.is_unique:
        raise ValueError(
            "adata.obs_names must be unique for subcellular_coords "
            "(per-cell transcript grouping is index-based)."
        )

    transcripts = _resolve_transcripts(adata, transcripts, "subcellular_coords")
    try:
        import pyarrow as pa
        if isinstance(transcripts, pa.Table):
            transcripts = transcripts.to_pandas()
    except ImportError:
        pass

    for col in (cell_id_col, x_col, y_col, feature_col):
        if col not in transcripts.columns:
            raise KeyError(f"Column '{col}' not found in the transcript table.")

    tx = transcripts[[cell_id_col, x_col, y_col, feature_col]].copy()
    tx[cell_id_col] = tx[cell_id_col].astype(str)

    (cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent, cell_area,
     usable) = _prepare_geometry(adata, cell_boundaries_key, nucleus_boundaries_key)

    # ---- cell eligibility -------------------------------------------------
    eligible = usable.copy()
    group_vals = None
    if groupby is not None:
        if groupby not in adata.obs:
            raise KeyError(f"obs['{groupby}'] not found")
        group_vals = adata.obs[groupby].astype(str).to_numpy()
        if groups is not None:
            eligible &= np.isin(group_vals, np.asarray(groups, dtype=str))

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    ci_all = tx[cell_id_col].map(obs_index).to_numpy(dtype=np.float64)
    known = np.isfinite(ci_all)
    ci_all = np.where(known, ci_all, 0).astype(np.int64)
    keep = known & eligible[ci_all]
    n_unmatched = int((~known).sum())
    if n_unmatched:
        log.info(
            "subcellular_coords: %d transcript(s) reference a cell id absent from "
            "obs_names and were dropped", n_unmatched,
        )
    tx = tx[keep].reset_index(drop=True)
    ci = ci_all[keep]
    if len(tx) == 0:
        raise ValueError(
            f"No transcripts matched obs_names via '{cell_id_col}' for cells with "
            "both cell and nucleus polygons. Check the id column and the boundaries."
        )

    n_obs = adata.n_obs
    px = tx[x_col].to_numpy(dtype=np.float64)
    py = tx[y_col].to_numpy(dtype=np.float64)

    # ---- transcript geometry, chunked ------------------------------------
    n_tx = len(px)
    u = np.empty(n_tx); d_nuc_s = np.empty(n_tx)
    d_cell = np.empty(n_tx); in_cell = np.empty(n_tx, dtype=bool)
    for s in range(0, n_tx, _TX_CHUNK):
        e = min(s + _TX_CHUNK, n_tx)
        sl = slice(s, e)
        u[sl], d_nuc_s[sl], d_cell[sl], _, in_cell[sl] = _radial_u(
            px[sl], py[sl], ci[sl], cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent
        )
    n_outside = int((~in_cell).sum())
    if n_outside:
        log.info(
            "subcellular_coords: %d transcript(s) (%.1f%%) fall outside their own "
            "cell polygon and were dropped -- assignment/segmentation disagreement",
            n_outside, 100.0 * n_outside / n_tx,
        )
    tx = tx[in_cell].reset_index(drop=True)
    ci, px, py = ci[in_cell], px[in_cell], py[in_cell]
    u, d_nuc_s, d_cell = u[in_cell], d_nuc_s[in_cell], d_cell[in_cell]
    n_tx = len(u)

    # ---- per-cell null from a rasterized polygon -------------------------
    # Everything below is accumulated with bincount/add.at rather than a loop
    # over cells: at 150k cells a per-cell Python body costs minutes, and this
    # is the step the whole module's runtime lives in.
    n_comp = len(COMPARTMENTS)
    npx_all = np.zeros(n_obs)
    cnt_all = np.zeros((n_obs, n_comp))
    sx_all = np.zeros(n_obs); sy_all = np.zeros(n_obs); sq_all = np.zeros(n_obs)
    shell_num = np.zeros(n_obs); shell_den = np.zeros(n_obs)
    null_hist = np.zeros(NULL_PROFILE_BINS)
    v = np.full(n_tx, np.nan)

    order = np.argsort(ci, kind="stable")
    ci_sorted = ci[order]
    starts = np.searchsorted(ci_sorted, np.arange(n_obs), side="left")
    stops = np.searchsorted(ci_sorted, np.arange(n_obs), side="right")
    cells_with_tx = np.flatnonzero(stops > starts)

    if null_pixel_um is not None:
        step_um = np.full(n_obs, float(null_pixel_um))
    else:
        if null_points_per_cell < 16:
            raise ValueError(
                f"null_points_per_cell must be >= 16, got {null_points_per_cell}"
            )
        with np.errstate(invalid="ignore"):
            step_um = np.sqrt(np.maximum(cell_area, 0.0) / null_points_per_cell)
        step_um[~np.isfinite(step_um) | (step_um <= 0)] = 0.5

    for cs in range(0, len(cells_with_tx), _CELL_CHUNK):
        batch = cells_with_tx[cs:cs + _CELL_CHUNK]
        gx_l, gy_l, gc_l = [], [], []
        for c in batch:
            gxc, gyc = _raster_points(cell_poly[c], step_um[c])
            gx_l.append(gxc); gy_l.append(gyc)
            gc_l.append(np.full(len(gxc), c, dtype=np.int64))
        gx = np.concatenate(gx_l); gy = np.concatenate(gy_l)
        gc = np.concatenate(gc_l)
        gu, gdn, gdc, _, ginside = _radial_u(
            gx, gy, gc, cell_poly, cell_bnd, nuc_poly, nuc_bnd, nuc_cent
        )
        gx, gy, gc = gx[ginside], gy[ginside], gc[ginside]
        gu, gdn, gdc = gu[ginside], gdn[ginside], gdc[ginside]
        if len(gu) == 0:
            continue
        gcode = _assign_compartments(
            gu, gdn, gdc, compartment_mode, compartment_bins,
            nuclear_edge_um, cell_edge_um,
        )

        np.add.at(npx_all, gc, 1.0)
        np.add.at(cnt_all, (gc, gcode), 1.0)
        np.add.at(sx_all, gc, gx)
        np.add.at(sy_all, gc, gy)
        np.add.at(sq_all, gc, gx ** 2 + gy ** 2)
        cyto = gu >= 1.0
        np.add.at(shell_num, gc[cyto], (gdn + gdc)[cyto])
        np.add.at(shell_den, gc[cyto], 1.0)
        # pooled null density of u: every interior pixel of every cell,
        # i.e. exactly uniform-in-polygon placement put through the same
        # coordinate the transcripts go through
        null_hist += np.bincount(
            np.clip((gu * (NULL_PROFILE_BINS / 2.0)).astype(np.int64),
                    0, NULL_PROFILE_BINS - 1),
            minlength=NULL_PROFILE_BINS,
        )

        # ---- probability integral transform, vectorized -------------------
        # Merge this batch's pixels and transcripts into one array sorted by
        # (cell, u, pixel-before-transcript). A running count of pixels, rebased
        # at each cell's first row, then gives every transcript the number of
        # its own cell's pixels with u <= its u -- i.e. F_c(u) -- without ever
        # materializing a per-cell CDF.
        tsel = order[np.concatenate([
            np.arange(starts[c], stops[c]) for c in batch
        ])] if len(batch) else np.empty(0, dtype=np.int64)
        if len(tsel) == 0:
            continue
        tu, tc = u[tsel], ci[tsel]
        allu = np.concatenate([gu, tu])
        allc = np.concatenate([gc, tc])
        is_tx = np.concatenate([
            np.zeros(len(gu), dtype=np.int8), np.ones(len(tu), dtype=np.int8)
        ])
        srt = np.lexsort((is_tx, allu, allc))
        c_s, tx_s = allc[srt], is_tx[srt]
        pix_cum = np.cumsum(1 - tx_s)                      # pixels seen so far
        first = np.searchsorted(c_s, c_s, side="left")     # row 0 of each cell
        base = np.concatenate([[0], pix_cum])[first]       # pixels before the cell
        rank = pix_cum - base                              # pixels of this cell <= u
        npx_cell = npx_all[c_s]
        is_t = tx_s == 1
        # srt indexes the concatenation, so transcript rows map back by offset
        tx_pos = srt[is_t] - len(gu)
        with np.errstate(divide="ignore", invalid="ignore"):
            # rank / n_px, not the mid-rank: the transcript is not a member of
            # the pixel sample, so P(u_pixel <= u_tx) is already an unbiased
            # estimate of F_c(u_tx) and a mid-rank would subtract a spurious
            # 1/(2 n_px). Deterministic; the randomized PIT of Dunn & Smyth
            # (1996) would make v exactly continuous at the cost of a seed, and
            # nothing downstream needs that (Berman uses only the mean of v,
            # and the shape test is two-sample so the discretization cancels).
            v[tsel[tx_pos]] = np.where(
                npx_cell[is_t] > 0, rank[is_t] / npx_cell[is_t], np.nan
            )

    n_degenerate = int((npx_all[cells_with_tx] == 0).sum())
    if n_degenerate:
        log.info(
            "subcellular_coords: %d cell(s) produced no interior raster pixel and "
            "therefore have no null", n_degenerate,
        )

    has_px = npx_all > 0
    null_frac = np.full((n_obs, n_comp), np.nan)
    null_frac[has_px] = cnt_all[has_px] / npx_all[has_px, None]
    null_mom = np.full((n_obs, 4), np.nan)
    null_mom[has_px, 0] = sx_all[has_px] / npx_all[has_px]
    null_mom[has_px, 1] = sy_all[has_px] / npx_all[has_px]
    null_mom[has_px, 2] = sq_all[has_px] / npx_all[has_px]
    null_mom[has_px, 3] = npx_all[has_px]
    shell = np.full(n_obs, np.nan)
    shell[has_px] = 0.0
    ok_shell = shell_den > 0
    shell[ok_shell] = shell_num[ok_shell] / shell_den[ok_shell]

    # ---- compartment codes for the transcripts ---------------------------
    code = _assign_compartments(
        u, d_nuc_s, d_cell, compartment_mode, compartment_bins,
        nuclear_edge_um, cell_edge_um,
    )

    # ---- cell-level filters ----------------------------------------------
    n_per_cell = np.bincount(ci, minlength=n_obs)
    ok_cell = (n_per_cell >= min_cell_transcripts) & np.isfinite(shell)
    if min_shell_um > 0:
        ok_cell &= shell >= min_shell_um
    keep_tx = ok_cell[ci] & np.isfinite(v)
    n_dropped_cells = int((n_per_cell > 0).sum() - ok_cell[n_per_cell > 0].sum())
    if n_dropped_cells:
        log.info(
            "subcellular_coords: excluding %d cell(s) below min_cell_transcripts=%d "
            "or min_shell_um=%.2f", n_dropped_cells, min_cell_transcripts, min_shell_um,
        )
    tx = tx[keep_tx].reset_index(drop=True)
    ci, u, v = ci[keep_tx], u[keep_tx], v[keep_tx]
    d_nuc_s, d_cell, code = d_nuc_s[keep_tx], d_cell[keep_tx], code[keep_tx]
    px, py = px[keep_tx], py[keep_tx]

    # ---- bearing in the nucleus-anchored polarity frame ------------------
    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    raw_bearing = np.arctan2(py - centroids[ci, 1], px - centroids[ci, 0])
    pol_col = "cell_polarity_major_angle"
    if pol_col in adata.obs:
        pol = adata.obs[pol_col].to_numpy(dtype=np.float64)
        angle = np.mod(raw_bearing - pol[ci] + np.pi, 2 * np.pi) - np.pi
    else:
        log.info(
            "subcellular_coords: obs['%s'] not found -- 'angle' is left NaN. Run "
            "ab.pp.calculate_shape_metrics() for the nucleus-anchored frame; raw "
            "image-frame bearings are mounting-rotation artifacts, so they are not "
            "substituted.", pol_col,
        )
        angle = np.full(len(u), np.nan)

    # ---- observed per-cell occupancy and the pooled reference ------------
    obs_counts = np.zeros((n_obs, n_comp), dtype=np.float64)
    np.add.at(obs_counts, (ci, code), 1.0)
    tot = obs_counts.sum(axis=1)
    obs_frac = np.full_like(obs_counts, np.nan)
    nz = tot > 0
    obs_frac[nz] = obs_counts[nz] / tot[nz, None]
    pooled = np.bincount(code, minlength=n_comp).astype(np.float64)
    pooled /= pooled.sum()

    adata.obs[key_added + "_n_transcripts"] = tot.astype(np.int32)
    adata.obs[key_added + "_shell_um"] = np.where(nz, shell, np.nan).astype(np.float32)
    adata.obsm[key_added + "_compartment_fractions"] = obs_frac.astype(np.float32)
    adata.obsm[key_added + "_null_fractions"] = null_frac.astype(np.float32)
    adata.obsm[key_added + "_null_moments"] = null_mom.astype(np.float32)
    adata.obsm[key_added + "_nucleus_centroid"] = nuc_cent.astype(np.float32)
    adata.uns[key_added] = {
        "compartment_names": list(COMPARTMENTS),
        "pooled_fractions": pooled.astype(np.float32),
        "null_profile": (null_hist / max(null_hist.sum(), 1.0)).astype(np.float32),
        "null_profile_bins": int(NULL_PROFILE_BINS),
        "params": {
            "compartment_mode": compartment_mode,
            "compartment_bins": tuple(float(b) for b in compartment_bins),
            "nuclear_edge_um": float(nuclear_edge_um),
            "cell_edge_um": float(cell_edge_um),
            "null_points_per_cell": int(null_points_per_cell),
            "null_pixel_um": (None if null_pixel_um is None else float(null_pixel_um)),
            "min_shell_um": float(min_shell_um),
            "min_cell_transcripts": int(min_cell_transcripts),
            "groupby": groupby,
            "n_transcripts_out_of_polygon": n_outside,
            "n_transcripts_unmatched": n_unmatched,
            "n_cells_excluded": n_dropped_cells,
            "n_cells_no_raster": n_degenerate,
        },
    }

    out = pd.DataFrame({
        "cell_id": pd.Categorical(tx[cell_id_col].to_numpy()),
        "gene": pd.Categorical(tx[feature_col].to_numpy()),
        "x": px.astype(np.float32),
        "y": py.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "d_nucleus_um": d_nuc_s.astype(np.float32),
        "d_membrane_um": d_cell.astype(np.float32),
        "angle": angle.astype(np.float32),
        "compartment": pd.Categorical(
            np.asarray(COMPARTMENTS)[code], categories=list(COMPARTMENTS)
        ),
    })
    # Deliberately no ndarray in out.attrs: pandas serializes attrs to JSON in
    # to_parquet(), so stashing the cell-index array there would make the
    # returned frame unsaveable. Consumers re-map cell_id -> obs index instead.
    if group_vals is not None:
        out["group"] = pd.Categorical(group_vals[ci])

    log.info(
        "subcellular_coords: %d transcripts in %d cells; pooled occupancy %s -> "
        "obs['%s_n_transcripts'], obsm['%s_compartment_fractions'], uns['%s']",
        len(out), int(nz.sum()),
        ", ".join(f"{n}={f:.3f}" for n, f in zip(COMPARTMENTS, pooled)),
        key_added, key_added, key_added,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# inference helpers
# ──────────────────────────────────────────────────────────────────────────────
def _pair_table(gene_codes, cell_idx, n_genes, n_cells):
    """Collapse transcripts to unique ``(gene, cell)`` pairs.

    Returns ``(pair_gene, pair_cell, pair_total, inverse)`` where *inverse*
    maps each transcript to its pair row. A linear key keeps this a single
    ``np.unique`` rather than a groupby, which matters at 10^7 transcripts.
    """
    key = gene_codes.astype(np.int64) * np.int64(n_cells) + cell_idx.astype(np.int64)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return (uniq // n_cells).astype(np.int64), (uniq % n_cells).astype(np.int64), cnt, inv


def _cmh(a, t, m1, n, gene_of_pair, n_genes):
    """Cochran--Mantel--Haenszel test of a 2x2xK table, vectorized over genes.

    Each row of the inputs is one stratum (one cell) of one gene's table:
    *a* = gene transcripts in the compartment, *t* = gene transcripts in the
    cell, *m1* = all transcripts in the compartment in that cell, *n* = all
    transcripts in that cell. Strata where the gene is absent contribute
    exactly zero to every sum, so only occupied ``(gene, cell)`` pairs need to
    be passed -- which is what makes this a single vectorized pass rather than
    a loop over 377 genes x 150k cells.

    Returns ``(log_or, se, chi2, n_strata)`` per gene: the Mantel--Haenszel
    log odds ratio, its Robins--Breslow--Greenland standard error, and the
    continuity-corrected MH statistic (1 df).
    """
    a = a.astype(np.float64); t = t.astype(np.float64)
    m1 = m1.astype(np.float64); n = n.astype(np.float64)
    b = t - a
    c = m1 - a
    d = (n - m1) - b

    safe_n = np.maximum(n, 1.0)
    E = t * m1 / safe_n
    denom = safe_n * safe_n * np.maximum(n - 1.0, 1.0)
    V = np.where(n > 1.0, t * (n - t) * m1 * (n - m1) / denom, 0.0)

    R = a * d / safe_n
    S = b * c / safe_n
    P = (a + d) / safe_n
    Q = (b + c) / safe_n

    def g(w):
        return np.bincount(gene_of_pair, weights=w, minlength=n_genes)

    sa, sE, sV = g(a), g(E), g(V)
    sR, sS = g(R), g(S)
    sPR, sQS = g(P * R), g(Q * S)
    sPSQR = g(P * S + Q * R)
    n_strata = np.bincount(gene_of_pair, minlength=n_genes)

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(
            sV > 0, np.maximum(np.abs(sa - sE) - 0.5, 0.0) ** 2 / sV, np.nan
        )
        ok = (sR > 0) & (sS > 0)
        # A gene entirely absent from (or confined to) a compartment has a true
        # odds ratio of 0 / infinity -- reporting +-inf is informative where NaN
        # would read as "could not be computed". The MH chi2 above is unaffected
        # and remains the p-value in those cases; only the SE is undefined.
        log_or = np.where(
            ok,
            np.log(np.where(ok, sR, 1.0) / np.where(ok, sS, 1.0)),
            np.where(sR > 0, np.inf, np.where(sS > 0, -np.inf, np.nan)),
        )
        var = np.where(
            ok,
            sPR / (2.0 * np.where(ok, sR, 1.0) ** 2)
            + sPSQR / (2.0 * np.where(ok, sR * sS, 1.0))
            + sQS / (2.0 * np.where(ok, sS, 1.0) ** 2),
            np.nan,
        )
        se = np.sqrt(var)
    return log_or, se, chi2, n_strata


def _bh(p):
    """Benjamini--Hochberg over the finite entries only, NaN preserved."""
    from scipy.stats import false_discovery_control

    out = np.full(len(p), np.nan)
    fin = np.isfinite(p)
    if fin.any():
        out[fin] = false_discovery_control(np.asarray(p, dtype=np.float64)[fin])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# tl.subcellular_enrichment
# ──────────────────────────────────────────────────────────────────────────────
@with_adata(mirror_uns=from_kwarg("key_added"))
def subcellular_enrichment(
    adata: ad.AnnData,
    coords: pd.DataFrame | None = None,
    *,
    transcripts=None,
    groupby: str | None = None,
    groups: list[str] | None = None,
    min_gene_transcripts: int = 50,
    min_cell_gene_transcripts: int = 3,
    min_cells: int = 20,
    coords_key: str = "subcellular",
    key_added: str = "subcellular_enrichment",
    **coords_kwargs,
) -> pd.DataFrame:
    """Test which genes are subcellularly localized, with the cell as the replicate.

    Returns **one row per gene** (or per gene x group). That is deliberate: the
    scientific family is the panel, so BH is applied across genes, and the
    per-compartment numbers are post-hoc detail on a gene that already passed.
    Adjusting across ``genes x compartments`` instead -- as the predecessor of
    this module did, over 9425 gene x domain rows -- makes the FDR
    uninterpretable and reports a yield inflated by the number of bins chosen.

    **Layer 1, the headline** (``cmh_log_or_*``, ``p_cmh_*``, ``p_adj_omnibus``):
    a Cochran--Mantel--Haenszel test **stratified by cell**, one-vs-rest per
    compartment. It asks whether gene *g* occupies this compartment more than
    *the other transcripts of the same cell*, so every per-cell nuisance --
    the 3-D-to-2-D projection kernel, segmentation error, z-extent,
    optical-crowding censoring, cell size, nucleus-to-cell ratio -- cancels
    exactly rather than approximately. This is the number to filter on.
    The per-gene omnibus takes the smallest of the four compartment p-values
    with a within-gene Bonferroni factor of 4 (valid under any dependence
    between them, which matters because the four compartments are a
    rank-deficient partition), then BH across genes.

    **Layer 2, absolute** (``median_d``, ``p_berman``): Berman's Z1 on the PIT
    coordinate ``v`` per cell, aggregated across cells by a Wilcoxon
    signed-rank test. ``v`` has null mean ``1/2`` and variance ``1/12`` in
    *every* cell, so ``Z = sqrt(n) (mean(v) - 1/2) / sqrt(1/12)`` needs no
    weighting. Signed-rank rather than a t-test because the per-cell statistic
    is discrete and right-skewed at small counts. This layer is genuinely
    absolute and therefore **carries the projection bias** -- read it as "is
    this gene non-uniform inside the cell", never as the primary evidence.

    ``shape_ks`` / ``p_shape`` are a two-sample Kolmogorov--Smirnov comparison
    of the gene's ``v`` distribution against the rest of the panel's. Two-sample
    rather than against a continuous ``Uniform(0, 1)`` for two reasons: ``v`` is
    discretized to *null_points_per_cell* levels, so a continuous-null test
    rejects on the discretization alone, and the relative form matches Layer 1's
    framing. Its replication unit is nonetheless the *transcript*, so at Xenium
    scale it rejects for almost everything; it is here only because it sees
    distribution *shape* -- a mid-cytoplasmic ring, say -- that the mean-based
    layers cannot. Treat it as a descriptor, not as inference.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    coords
        Output of :func:`subcellular_coords`. ``None`` computes it, forwarding
        *transcripts* and any extra keyword arguments to that function.
    transcripts
        Passed to :func:`subcellular_coords` when *coords* is ``None``.
    groupby
        ``obs`` column defining groups (e.g. a cell type). The whole analysis
        runs **within each group** -- the pooled reference is the group's own
        transcripts and the strata are its cells -- and the output gains a
        ``group`` column. Localization programs are cell-type-specific, so a
        gene that is peripheral in one lineage and nuclear in another cancels
        when pooled; this is how to avoid that. Requires *coords* to carry a
        ``group`` column (i.e. the same *groupby* passed to
        :func:`subcellular_coords`), which happens automatically when *coords*
        is computed here.
    groups
        Restrict to these *groupby* values.
    min_gene_transcripts
        Minimum transcripts (within the group) for a gene to be tested.
    min_cell_gene_transcripts
        A cell contributes to Layer 2 only with at least this many transcripts
        of the gene. The floor can be low because Berman's ``Z`` is exactly
        standardized at every ``n`` -- unlike a t-test on raw domain fractions,
        which needs 20-30 per cell before it is even approximately calibrated.
        Layer 1 needs no floor at all: CMH weights single-transcript strata
        correctly by construction.

        **Layer 2 is nonetheless out of reach for most genes on a targeted
        panel.** At ~56 transcripts per cell across 377 genes the average gene
        has well under one copy per cell, so only the most abundant genes ever
        clear this floor in enough cells. That is a property of the data, not a
        tuning problem, and it is the main reason Layer 1 is the headline. Dense
        high-plex data (thousands of transcripts per cell) is where Layer 2
        earns its place.
    min_cells
        Minimum eligible cells for Layer 2.
    coords_key
        ``uns``/``obsm`` prefix written by :func:`subcellular_coords`, read here
        for the per-cell geometric null.
    key_added
        ``uns`` key for the result table.

    Stores
    ------
    ``uns[key_added]``
        The returned DataFrame, additionally mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Returns
    -------
    pd.DataFrame
        One row per gene (x group): ``n_transcripts``, ``n_cells``, the four
        ``frac_*`` observed occupancies with their ``expected_*`` per-cell
        geometric nulls and ``log2_vs_pooled_*`` effect sizes, Layer 1's
        ``cmh_log_or_*`` / ``cmh_se_*`` / ``p_cmh_*`` plus ``p_omnibus`` /
        ``p_adj_omnibus``, Layer 2's ``median_d`` / ``cells_positive`` /
        ``p_berman`` / ``p_adj_berman``, ``shape_ks`` / ``p_shape``, and
        ``abundance_rank``.

    Examples
    --------
    >>> res = ab.tl.subcellular_enrichment(mdata)
    >>> res.query('p_adj_omnibus < 0.05').nlargest(10, 'cmh_log_or_cell_edge')
    >>> per_type = ab.tl.subcellular_enrichment(mdata, groupby='leiden')
    """
    from scipy import special as sp_special
    from scipy import stats as sp_stats

    if coords is None:
        coords = subcellular_coords(
            adata, transcripts, groupby=groupby, groups=groups,
            key_added=coords_key, **coords_kwargs,
        )
    if groupby is not None and "group" not in coords.columns:
        raise ValueError(
            "groupby was given but `coords` has no 'group' column -- pass the "
            "same groupby= to subcellular_coords, or let this function compute "
            "coords itself."
        )

    null_key = coords_key + "_null_fractions"
    if null_key not in adata.obsm:
        raise KeyError(
            f"obsm['{null_key}'] not found. Run ab.tl.subcellular_coords() first "
            "(or let this function compute it)."
        )
    null_frac = np.asarray(adata.obsm[null_key], dtype=np.float64)
    null_frac = np.nan_to_num(null_frac, nan=0.0)

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    n_cells_total = adata.n_obs
    n_comp = len(COMPARTMENTS)

    if groupby is None:
        chunks = [(None, coords)]
    else:
        levels = (
            [str(g) for g in groups] if groups is not None
            else sorted(pd.unique(coords["group"].astype(str)))
        )
        chunks = [(g, coords[coords["group"].astype(str) == g]) for g in levels]

    frames = []
    for gname, cd in chunks:
        if len(cd) == 0:
            log.info("subcellular_enrichment: group %r has no transcripts", gname)
            continue
        cell_idx = cd["cell_id"].astype(str).map(obs_index).to_numpy(dtype=np.int64)
        gene_names, gene_codes = np.unique(
            cd["gene"].astype(str).to_numpy(), return_inverse=True
        )
        n_genes = len(gene_names)
        comp_code = np.asarray(
            pd.Categorical(cd["compartment"], categories=list(COMPARTMENTS)).codes
        )
        v = cd["v"].to_numpy(dtype=np.float64)

        pg, pc, pt, inv = _pair_table(gene_codes, cell_idx, n_genes, n_cells_total)
        cell_tot = np.bincount(cell_idx, minlength=n_cells_total).astype(np.float64)
        gene_tot = np.bincount(gene_codes, minlength=n_genes).astype(np.float64)

        row: dict[str, np.ndarray] = {
            "gene": gene_names,
            "n_transcripts": gene_tot.astype(np.int64),
            "n_cells": np.bincount(pg, minlength=n_genes).astype(np.int64),
        }
        if gname is not None:
            row["group"] = np.asarray([str(gname)] * n_genes, dtype=object)

        pooled = np.bincount(comp_code, minlength=n_comp).astype(np.float64)
        pooled /= max(pooled.sum(), 1.0)

        p_per_comp = np.full((n_genes, n_comp), np.nan)
        for k, cname in enumerate(COMPARTMENTS):
            in_k = (comp_code == k).astype(np.float64)
            a_pair = np.bincount(inv, weights=in_k, minlength=len(pt))
            cell_m1 = np.bincount(cell_idx, weights=in_k, minlength=n_cells_total)
            log_or, se, chi2, _ = _cmh(
                a_pair, pt, cell_m1[pc], cell_tot[pc], pg, n_genes
            )
            p = sp_stats.chi2.sf(chi2, 1)
            p_per_comp[:, k] = p
            obs_k = np.bincount(gene_codes, weights=in_k, minlength=n_genes)
            frac_k = obs_k / np.maximum(gene_tot, 1.0)
            # transcript-weighted per-cell geometric null for this gene
            exp_w = np.bincount(
                pg, weights=pt * null_frac[pc, k], minlength=n_genes
            )
            row[f"frac_{cname}"] = frac_k
            row[f"expected_{cname}"] = exp_w / np.maximum(gene_tot, 1.0)
            row[f"log2_vs_pooled_{cname}"] = np.log2(
                (frac_k + 1e-9) / (pooled[k] + 1e-9)
            )
            row[f"cmh_log_or_{cname}"] = log_or
            row[f"cmh_se_{cname}"] = se
            row[f"p_cmh_{cname}"] = p

        # ---- per-gene omnibus: Bonferroni over the four, then BH over genes
        with np.errstate(invalid="ignore"):
            p_omni = np.nanmin(p_per_comp, axis=1) * n_comp
        p_omni = np.clip(p_omni, 0.0, 1.0)

        # ---- Layer 2: Berman Z1 per (gene, cell) -> signed-rank across cells
        sum_v = np.bincount(inv, weights=v, minlength=len(pt))
        mean_v = sum_v / pt
        d_pair = (mean_v - 0.5) / np.sqrt(1.0 / 12.0)
        z_pair = np.sqrt(pt) * d_pair
        elig = pt >= min_cell_gene_transcripts

        median_d = np.full(n_genes, np.nan)
        frac_pos = np.full(n_genes, np.nan)
        p_berman = np.full(n_genes, np.nan)
        p_shape = np.full(n_genes, np.nan)
        shape_ks = np.full(n_genes, np.nan)

        # per-gene histogram of v on a shared grid -> both ECDFs in one pass
        n_bins = 256
        vbin = np.clip((v * n_bins).astype(np.int64), 0, n_bins - 1)
        v_hist = np.bincount(
            gene_codes * n_bins + vbin, minlength=n_genes * n_bins
        ).reshape(n_genes, n_bins).astype(np.float64)
        v_total = v_hist.sum(axis=0)

        order_g = np.argsort(pg[elig], kind="stable")
        pg_e = pg[elig][order_g]
        d_e, z_e = d_pair[elig][order_g], z_pair[elig][order_g]
        gstart = np.searchsorted(pg_e, np.arange(n_genes), side="left")
        gstop = np.searchsorted(pg_e, np.arange(n_genes), side="right")
        tested = gene_tot >= min_gene_transcripts
        for gi in np.flatnonzero(tested):
            sl = slice(gstart[gi], gstop[gi])
            if gstop[gi] - gstart[gi] >= min_cells:
                median_d[gi] = float(np.median(d_e[sl]))
                frac_pos[gi] = float((d_e[sl] > 0).mean())
                zz = z_e[sl]
                if np.any(zz != 0):
                    p_berman[gi] = float(
                        sp_stats.wilcoxon(zz, zero_method="zsplit").pvalue
                    )
            n_g = int(gene_tot[gi])
            cdf_g = np.cumsum(v_hist[gi]) / max(n_g, 1)
            n_rest = int(v_total.sum()) - n_g
            if n_rest > 0:
                cdf_rest = np.cumsum(v_total - v_hist[gi]) / n_rest
                dstat = float(np.abs(cdf_g - cdf_rest).max())
                n_eff = n_g * n_rest / (n_g + n_rest)
                shape_ks[gi] = dstat
                p_shape[gi] = float(sp_special.kolmogorov(np.sqrt(n_eff) * dstat))

        p_omni[~tested] = np.nan
        row["p_omnibus"] = p_omni
        row["p_adj_omnibus"] = _bh(p_omni)
        row["median_d"] = median_d
        row["cells_positive"] = frac_pos
        row["p_berman"] = p_berman
        row["p_adj_berman"] = _bh(p_berman)
        row["shape_ks"] = shape_ks
        row["p_shape"] = p_shape
        row["abundance_rank"] = (
            pd.Series(-gene_tot).rank(method="min").to_numpy().astype(np.int64)
        )
        frames.append(pd.DataFrame(row)[tested].reset_index(drop=True))

    result = (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=["gene"])
    )
    adata.uns[key_added] = result
    n_sig = int((result["p_adj_omnibus"] < 0.05).sum()) if len(result) else 0
    n_l2 = int(np.isfinite(result["p_berman"]).sum()) if len(result) else 0
    log.info(
        "subcellular_enrichment: %d row%s tested%s; %d localized at "
        "p_adj_omnibus<0.05 (cell-stratified CMH, the headline test). Layer 2 "
        "(absolute, per-cell) reached %d row(s) -- the rest lack %d+ copies of "
        "the gene in %d+ cells -> uns['%s']",
        len(result), "" if len(result) == 1 else "s",
        f" across {len(frames)} group(s)" if groupby else "", n_sig, n_l2,
        min_cell_gene_transcripts, min_cells, key_added,
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# tl.subcellular_indices
# ──────────────────────────────────────────────────────────────────────────────
@with_adata
def subcellular_indices(
    adata: ad.AnnData,
    coords: pd.DataFrame | None = None,
    *,
    transcripts=None,
    genes: list[str] | None = None,
    min_cell_transcripts: int = 30,
    spatial_key: str = "spatial",
    coords_key: str = "subcellular",
    key_added: str = "subcellular",
    **coords_kwargs,
) -> None:
    """Per-cell RNA distribution indices, normalized against the cell's own mask.

    Implements the three indices of Stueland et al. (2019) *Sci Rep* 9:8267 --
    the **RNA** Distribution Index calculator (not "radial"; the name is worth
    getting right) -- each a ratio against the same moment of a *uniform fill
    of the segmented cell*, which is what makes them dimensionless and
    comparable between cells of different size and shape:

    - **PDI**, peripheral distribution index: second moment of the transcripts
      about the **nucleus centroid**, over the same moment of the cell mask.
      ``1`` diffuse, ``< 1`` perinuclear, ``> 1`` peripheral.
    - **DI**, dispersion index: the same about the **RNA centroid**. ``1``
      diffuse, ``< 1`` concentrated anywhere, ``> 1`` spread to the periphery.
    - **PI**, polarization index: distance from the RNA centroid to the cell
      centroid, over the mask's radius of gyration. Grows with polarization.

    Also stores the PIT-based radial index ``rdi = mean(v)`` (``0.5`` under
    uniform placement) and its standardized form
    ``rdi_z = (rdi - 1/2) / sqrt(1/12)``.

    All mask moments are reconstructed **analytically** from the three scalars
    :func:`subcellular_coords` stored in
    ``obsm[coords_key + '_null_moments']`` -- for any centre ``q``,
    ``mean|p - q|^2 = mean|p|^2 - 2 q . mean(p) + |q|^2`` -- so no cell is ever
    rasterized twice.

    .. note::
       ``rdi_z`` has null variance ``1/n_c``, so a raw map of it renders
       low-count cells as extreme values: the volcano-shaped map that looks
       like regional biology and is sampling noise. ``rdi_shrunk`` applies
       random-effects (James--Stein) shrinkage ``d * tau2 / (tau2 + 1/n)`` with
       ``tau2`` estimated by moments as ``max(0, Var(d) - mean(1/n))``, and both
       terms are stored in ``uns``. **Map the shrunk column and quote the
       variance decomposition**: if ``tau2`` is ~0 the honest reading is that
       there is no between-cell variation to see -- and that is then a
       measurement, not an assertion about a flat figure.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    coords
        Output of :func:`subcellular_coords`; computed if ``None``.
    transcripts
        Passed to :func:`subcellular_coords` when *coords* is ``None``.
    genes
        Restrict the indices to these genes -- the per-cell distribution of one
        transcript species, the way the original RDI calculator is used on a
        single smFISH probe. ``None`` uses every transcript in the cell.
    min_cell_transcripts
        Cells below this are left ``NaN``. The default 30 is a real floor, not
        a formality: these are second moments of a point set.
    spatial_key
        ``obsm`` key for cell centroids (um).
    coords_key
        Prefix of the :func:`subcellular_coords` outputs to read.
    key_added
        Prefix for the ``obs`` columns written.

    Stores
    ------
    ``obs[key_added + '_rdi']`` / ``_rdi_z`` / ``_rdi_shrunk``
        The PIT radial index, standardized, and shrunk (map the shrunk one).
    ``obs[key_added + '_pdi']`` / ``_di`` / ``_pi``
        Stueland's three indices.
    ``uns[key_added + '_indices']``
        ``tau2``, ``mean_inv_n``, ``n_cells``, ``n_excluded``, ``genes``.

    Examples
    --------
    >>> ab.tl.subcellular_indices(mdata, coords)
    >>> ab.pl.points(mdata, color='subcellular_rdi_shrunk').show()
    """
    if coords is None:
        coords = subcellular_coords(
            adata, transcripts, key_added=coords_key, **coords_kwargs
        )
    if genes is not None:
        coords = coords[coords["gene"].astype(str).isin(set(genes))]

    mom_key = coords_key + "_null_moments"
    nuc_key = coords_key + "_nucleus_centroid"
    for k in (mom_key, nuc_key):
        if k not in adata.obsm:
            raise KeyError(
                f"obsm['{k}'] not found. Run ab.tl.subcellular_coords() first."
            )
    mom = np.asarray(adata.obsm[mom_key], dtype=np.float64)
    nuc_c = np.asarray(adata.obsm[nuc_key], dtype=np.float64)
    cell_c = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    n_obs = adata.n_obs
    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    ci = coords["cell_id"].astype(str).map(obs_index).to_numpy(dtype=np.int64)
    txx = coords["x"].to_numpy(dtype=np.float64)
    txy = coords["y"].to_numpy(dtype=np.float64)
    v = coords["v"].to_numpy(dtype=np.float64)

    n = np.bincount(ci, minlength=n_obs).astype(np.float64)
    sx = np.bincount(ci, weights=txx, minlength=n_obs)
    sy = np.bincount(ci, weights=txy, minlength=n_obs)
    sq = np.bincount(ci, weights=txx ** 2 + txy ** 2, minlength=n_obs)
    sv = np.bincount(ci, weights=v, minlength=n_obs)

    ok = n >= min_cell_transcripts
    safe = np.maximum(n, 1.0)
    mx, my, mq = sx / safe, sy / safe, sq / safe          # transcript moments
    px, py, pq = mom[:, 0], mom[:, 1], mom[:, 2]          # mask moments

    def m2(qx, qy, ex, ey, eq):
        """mean |p - q|^2 for a point set, from its first and second moments."""
        return eq - 2.0 * (qx * ex + qy * ey) + qx ** 2 + qy ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        num_pdi = m2(nuc_c[:, 0], nuc_c[:, 1], mx, my, mq)
        den_pdi = m2(nuc_c[:, 0], nuc_c[:, 1], px, py, pq)
        pdi = np.where(den_pdi > 0, num_pdi / den_pdi, np.nan)

        num_di = mq - (mx ** 2 + my ** 2)
        den_di = m2(mx, my, px, py, pq)
        di = np.where(den_di > 0, num_di / den_di, np.nan)

        rg = np.sqrt(np.maximum(pq - (px ** 2 + py ** 2), 0.0))
        pi = np.where(
            rg > 0, np.hypot(mx - cell_c[:, 0], my - cell_c[:, 1]) / rg, np.nan
        )
        rdi = sv / safe
        rdi_z = (rdi - 0.5) / np.sqrt(1.0 / 12.0)

    for arr in (pdi, di, pi, rdi, rdi_z):
        arr[~ok] = np.nan

    # ---- empirical-Bayes shrinkage of rdi_z (null variance is exactly 1/n)
    fin = ok & np.isfinite(rdi_z)
    if fin.sum() >= 2:
        mean_inv_n = float(np.mean(1.0 / n[fin]))
        tau2 = float(max(0.0, np.var(rdi_z[fin], ddof=1) - mean_inv_n))
    else:
        mean_inv_n, tau2 = float("nan"), 0.0
    rdi_shrunk = np.full(n_obs, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rdi_shrunk[fin] = rdi_z[fin] * tau2 / (tau2 + 1.0 / n[fin])

    adata.obs[key_added + "_rdi"] = rdi.astype(np.float32)
    adata.obs[key_added + "_rdi_z"] = rdi_z.astype(np.float32)
    adata.obs[key_added + "_rdi_shrunk"] = rdi_shrunk.astype(np.float32)
    adata.obs[key_added + "_pdi"] = pdi.astype(np.float32)
    adata.obs[key_added + "_di"] = di.astype(np.float32)
    adata.obs[key_added + "_pi"] = pi.astype(np.float32)
    adata.uns[key_added + "_indices"] = {
        "tau2": tau2,
        "mean_inv_n": mean_inv_n,
        "n_cells": int(fin.sum()),
        "n_excluded": int(((n > 0) & ~ok).sum()),
        "min_cell_transcripts": int(min_cell_transcripts),
        "genes": list(genes) if genes is not None else None,
    }
    log.info(
        "subcellular_indices: %d cells scored (%d excluded below %d transcripts); "
        "between-cell variance tau2=%.4f vs sampling noise mean(1/n)=%.4f -> "
        "obs['%s_rdi_shrunk'], obs['%s_pdi'], obs['%s_di'], obs['%s_pi']",
        int(fin.sum()), int(((n > 0) & ~ok).sum()), min_cell_transcripts,
        tau2, mean_inv_n, key_added, key_added, key_added, key_added,
    )
