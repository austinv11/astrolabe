"""
bin2cell-style Visium HD bin-to-cell assignment.

Reimplements the core algorithm of bin2cell natively:

    Polański, K. et al. (2024). "Bin2cell reconstructs cells from high
    resolution Visium HD data." Bioinformatics, 40(9), btae546.
    https://doi.org/10.1093/bioinformatics/btae546

— nucleus/cell segmentation (StarDist/Cellpose, reused unmodified from
:mod:`astrolabe.seg`), polygon expansion (the polygon-space analog of
bin2cell's raster ``expand_labels`` step), and a PCA-transcriptional-
similarity tie-break for bins ambiguously placed within more than one
candidate cell's expanded footprint — matching the published method's
approach rather than a plain nearest-centroid fallback.

Segmentation is decoupled from bin assignment, mirroring
:func:`~astrolabe.seg.assign_transcripts`: :func:`assign_bins` takes ANY
pre-existing segmentation (:func:`~astrolabe.seg.segment_stardist`,
:func:`~astrolabe.seg.segment_cellpose`, :func:`~astrolabe.seg.voronoi_resegment`,
native Space Ranger output, or a fully custom one) as input.
:func:`segment_bin2cell` is a convenience recipe wiring
:func:`~astrolabe.seg.segment_stardist`/:func:`~astrolabe.seg.segment_cellpose`
and :func:`assign_bins` together for the common case, with an explicit
toggle for whether H&E (vs. a synthetic bin-expression raster) drives the
segmentation step.
"""

from __future__ import annotations

import logging
import os
import tempfile

import anndata as ad
import awkward as ak
import mudata as md
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tifffile
from shapely import points as shapely_points
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from astrolabe._utils import ImageSpec, _aggregate_sparse_by_group
from astrolabe.io._io import attach_modality
from astrolabe.pp._visium_hd import load_bin_counts
from astrolabe.seg._segment import segment_cellpose, segment_stardist

__all__ = ["assign_bins", "segment_bin2cell"]

log = logging.getLogger(__name__)

_CHUNK = 500_000  # bins per STRtree query batch, matching seg._assign's convention


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_polygons(boundaries) -> list:
    polys = []
    for coords in boundaries:
        c = np.asarray(coords)
        polys.append(Polygon(c[:, :2]) if len(c) >= 3 else Polygon())
    return polys


def _classify_matches(tree, xs: np.ndarray, ys: np.ndarray):
    """Query *tree* for every point in (xs, ys); split into unambiguous vs. contested.

    Returns ``(single_local_idx, single_poly_idx, multi_candidates)`` where
    *single_local_idx*/*single_poly_idx* are fully-vectorized 1:1 arrays (no
    Python-level loop, since these are expected to be the overwhelming
    majority of points), and *multi_candidates* is a ``dict[local_idx, list[poly_idx]]``
    built with a small Python loop restricted to genuinely contested points.
    """
    pts = shapely_points(xs, ys)
    result = tree.query(pts, predicate="within")
    if result.shape[1] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), {}

    pt_idx, poly_idx = result[0], result[1]
    order = np.argsort(pt_idx, kind="stable")
    pt_sorted = pt_idx[order]
    poly_sorted = poly_idx[order]
    uniq_pts, first_pos, counts = np.unique(pt_sorted, return_index=True, return_counts=True)

    single_mask = counts == 1
    single_local_idx = uniq_pts[single_mask].astype(np.int64)
    single_poly_idx = poly_sorted[first_pos[single_mask]].astype(np.int64)

    multi_u = uniq_pts[~single_mask]
    multi_fp = first_pos[~single_mask]
    multi_cnt = counts[~single_mask]
    multi_candidates: dict[int, list[int]] = {
        int(u): poly_sorted[fp: fp + c].tolist()
        for u, fp, c in zip(multi_u.tolist(), multi_fp.tolist(), multi_cnt.tolist())
    }
    return single_local_idx, single_poly_idx, multi_candidates


def _normalize_rows(X, target_sum: float) -> np.ndarray:
    """Total-count normalize (to a fixed, externally-supplied *target_sum*) + log1p.

    *target_sum* must be the same value across every call whose outputs will
    be compared in the same space (e.g. a PCA fit on provisional cell
    profiles and a later ``.transform()`` of a single contested bin's row) —
    recomputing a "median total" independently per call (as a naive
    per-matrix normalize_total would) makes a single-row call's target trivially
    equal to that row's own total, silently breaking comparability with
    whatever the PCA was fit on.
    """
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    totals = X.sum(axis=1, keepdims=True)
    totals_safe = np.where(totals > 0, totals, 1.0)
    normed = X / totals_safe * target_sum
    return np.log1p(normed)


# ---------------------------------------------------------------------------
# assign_bins
# ---------------------------------------------------------------------------


def assign_bins(
    bin_data: ad.AnnData,
    cell_seg: ad.AnnData,
    *,
    boundaries_key: str = "cell_boundaries",
    spatial_key: str = "spatial",
    expand_um: float = 4.0,
    tie_break: str = "pca",
    n_pcs: int = 15,
    min_bin_counts: int = 1,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Assign Visium HD bins to cells and build a per-cell count matrix (bin2cell).

    Takes an :class:`~anndata.AnnData` produced by
    :func:`~astrolabe.seg.segment_stardist`, :func:`~astrolabe.seg.segment_cellpose`,
    :func:`~astrolabe.seg.voronoi_resegment`, native Space Ranger segmentation
    (``mdata['visium_hd_cells']``), or any custom source — anything with
    ``obsm[boundaries_key]`` cell boundary polygons in µm — and populates its
    ``X`` by summing the 2µm bins assigned to each cell, following the
    published bin2cell algorithm (see module docstring for citation):

    1. **Core bins** — bins whose centroid falls inside a cell's own
       (un-expanded) boundary polygon are assigned directly; real segmented
       cells don't overlap, so this is unambiguous by construction.
    2. **Expansion** — every cell polygon is buffered by *expand_um* (the
       polygon-space analog of bin2cell's raster ``expand_labels`` step).
       Remaining bins falling in exactly one cell's expansion ring are
       assigned directly ("unambiguous expansion").
    3. **Contested bins** — bins falling in more than one cell's expansion
       ring are resolved by ``tie_break``: ``'pca'`` (default) fits a PCA on
       each candidate cell's *provisional* transcriptional profile (built
       only from its core + unambiguous-expansion bins) and assigns the bin
       to whichever of its own candidates is nearest in PCA space —
       matching bin2cell's "transcriptional similarity" tie-break.
       ``'nearest'`` assigns to the geometrically nearest candidate
       centroid instead. Bins whose candidates all lack a usable profile
       (or when PCA cannot be fit) silently fall back to ``'nearest'``.
    4. Cells with fewer than *min_bin_counts* total assigned bins are dropped.

    Parameters
    ----------
    bin_data
        The lazy (or already-materialized) Visium HD bins :class:`~anndata.AnnData`
        — typically ``mdata['visium_hd_bins']``. If ``bin_data.X`` is ``None``,
        the full bin matrix is loaded via :func:`~astrolabe.pp.load_bin_counts`.
    cell_seg
        AnnData with ``obsm[boundaries_key]`` cell boundary polygons (µm) and
        ``obsm[spatial_key]`` centroids — the segmentation to assign bins into.
        No expression data (``X``) is expected on input; it is populated (or
        overwritten) by this function.
    boundaries_key
        ``obsm`` key on *cell_seg* holding cell boundary polygons.
    spatial_key
        ``obsm`` key on both *bin_data* and *cell_seg* holding centroids (µm).
    expand_um
        Distance (µm) to buffer each cell polygon before assigning
        expansion-tier bins. Default ``4.0`` matches bin2cell's own default
        (``max_bin_distance=2`` bins × 2µm).
    tie_break
        ``'pca'`` (default) or ``'nearest'``.
    n_pcs
        Number of PCA components for the ``'pca'`` tie-break.
    min_bin_counts
        Minimum total assigned bins for a cell to be retained.
    inplace
        If ``True`` (default), modify *cell_seg* in place and return ``None``.
        If ``False``, operate on a copy and return it.

    Returns
    -------
    None or AnnData

    Examples
    --------
    >>> cell_seg = ab.seg.segment_stardist(mdata['he'], model_name='2D_versatile_he')
    >>> ab.seg.assign_bins(mdata['visium_hd_bins'], cell_seg)
    >>> ab.io.attach_modality(mdata, cell_seg, 'visium_hd_cells', set_default=True)
    """
    if tie_break not in ("pca", "nearest"):
        raise ValueError(f"tie_break must be 'pca' or 'nearest', got {tie_break!r}")
    if boundaries_key not in cell_seg.obsm:
        raise KeyError(f"obsm['{boundaries_key}'] not found on cell_seg.")
    if spatial_key not in cell_seg.obsm or spatial_key not in bin_data.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found on bin_data and/or cell_seg.")

    n_cells = cell_seg.n_obs
    n_bins = bin_data.n_obs

    if bin_data.X is not None:
        X_bins = bin_data.X
    else:
        log.info("assign_bins: bin matrix not yet materialized; loading via pp.load_bin_counts.")
        X_bins = load_bin_counts(bin_data, inplace=False).X

    X_bins = sp.csr_matrix(X_bins)

    log.info("assign_bins: %d bins, %d candidate cells", n_bins, n_cells)

    orig_polys = _build_polygons(cell_seg.obsm[boundaries_key])
    orig_tree = STRtree(orig_polys)
    expanded_polys = [p.buffer(expand_um) if not p.is_empty else p for p in orig_polys]
    exp_tree = STRtree(expanded_polys)

    bin_coords = np.asarray(bin_data.obsm[spatial_key], dtype=np.float64)
    cell_centroids = np.asarray(cell_seg.obsm[spatial_key], dtype=np.float64)

    cell_of_bin = np.full(n_bins, -1, dtype=np.int64)

    # ── Tier 0: core (unbuffered) containment ──────────────────────────────
    for start in range(0, n_bins, _CHUNK):
        end = min(start + _CHUNK, n_bins)
        single_idx, single_poly, multi = _classify_matches(
            orig_tree, bin_coords[start:end, 0], bin_coords[start:end, 1]
        )
        cell_of_bin[start + single_idx] = single_poly
        for local_i, cands in multi.items():
            cell_of_bin[start + local_i] = cands[0]  # arbitrary but rare (overlapping input segmentation)
    is_core = cell_of_bin >= 0
    log.info("assign_bins: %d core bins assigned", int(is_core.sum()))

    # ── Tier 1/2: expansion ring (unambiguous vs. contested) ───────────────
    remaining_idx = np.where(~is_core)[0]
    contested_candidates: dict[int, list[int]] = {}
    n_unambig_expansion = 0
    for start in range(0, len(remaining_idx), _CHUNK):
        chunk_idx = remaining_idx[start:start + _CHUNK]
        single_local, single_poly, multi = _classify_matches(
            exp_tree, bin_coords[chunk_idx, 0], bin_coords[chunk_idx, 1]
        )
        cell_of_bin[chunk_idx[single_local]] = single_poly
        n_unambig_expansion += len(single_local)
        for local_i, cands in multi.items():
            contested_candidates[int(chunk_idx[local_i])] = cands
    log.info(
        "assign_bins: %d unambiguous-expansion bins, %d contested bins, %d unassigned",
        n_unambig_expansion, len(contested_candidates),
        n_bins - int((cell_of_bin >= 0).sum()) - len(contested_candidates),
    )

    # ── Provisional per-cell profile (core + unambiguous-expansion only) ───
    provisional_mask = cell_of_bin >= 0
    if provisional_mask.any():
        provisional_X = _aggregate_sparse_by_group(
            X_bins[provisional_mask], cell_of_bin[provisional_mask], n_cells
        )
    else:
        provisional_X = sp.csr_matrix((n_cells, X_bins.shape[1]))
    profile_totals = np.asarray(provisional_X.sum(axis=1)).ravel()
    eligible_mask = profile_totals > 0

    # ── Resolve contested bins ──────────────────────────────────────────────
    pca = None
    cell_pca_row = None
    eligible_pca = None
    target_sum = 1.0
    if tie_break == "pca" and contested_candidates and eligible_mask.sum() >= 2:
        eligible_cell_idx = np.where(eligible_mask)[0]
        # Fixed normalization target shared by the PCA fit (provisional cell
        # profiles) and every later per-bin transform() call — see _normalize_rows.
        target_sum = float(np.median(profile_totals[eligible_mask]))
        normed_profiles = _normalize_rows(provisional_X[eligible_mask], target_sum)
        n_components = min(n_pcs, normed_profiles.shape[0] - 1, normed_profiles.shape[1])
        if n_components >= 1:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            eligible_pca = pca.fit_transform(normed_profiles)
            cell_pca_row = -np.ones(n_cells, dtype=np.int64)
            cell_pca_row[eligible_cell_idx] = np.arange(len(eligible_cell_idx))

    used_fallback = False
    for bin_idx, candidates in contested_candidates.items():
        candidates = sorted(set(candidates))
        # PCA-based comparison is only fair when *every* candidate has a
        # provisional profile to embed -- restricting to just the eligible
        # subset (the previous behavior) doesn't just disadvantage a
        # candidate with no core/unambiguous-expansion bins of its own, it
        # excludes it from the contest entirely: with exactly one eligible
        # candidate left, `argmin` over a single-element list "wins" by
        # construction, with no actual distance comparison ever happening.
        # A small cell whose entire footprint is contested bins then loses
        # every single one to any eligible neighbor regardless of true
        # similarity (confirmed on synthetic data with *identical*
        # expression vectors across all bins -- no transcriptional signal
        # to prefer either cell -- where the small cell still lost half its
        # contested bins purely from this structural exclusion). Falling
        # back to geometric nearest-centroid among *all* candidates whenever
        # any candidate lacks a PCA profile keeps the comparison fair.
        eligible_candidates = (
            candidates
            if pca is not None and all(cell_pca_row[c] >= 0 for c in candidates)
            else []
        )
        if eligible_candidates:
            bin_normed = _normalize_rows(X_bins[bin_idx], target_sum)
            bin_pca = pca.transform(bin_normed)
            cand_rows = [cell_pca_row[c] for c in eligible_candidates]
            dists = np.linalg.norm(eligible_pca[cand_rows] - bin_pca, axis=1)
            winner = eligible_candidates[int(np.argmin(dists))]
        else:
            used_fallback = True
            cand_centroids = cell_centroids[candidates]
            dists = np.linalg.norm(cand_centroids - bin_coords[bin_idx], axis=1)
            winner = candidates[int(np.argmin(dists))]
        cell_of_bin[bin_idx] = winner

    if used_fallback:
        log.warning(
            "assign_bins: some contested bins fell back to nearest-centroid tie-break "
            "(PCA unavailable — too few candidates with a usable transcriptional "
            "profile, or tie_break='nearest' was requested)."
        )

    # ── Final aggregation ───────────────────────────────────────────────────
    final_mask = cell_of_bin >= 0
    final_X = _aggregate_sparse_by_group(X_bins[final_mask], cell_of_bin[final_mask], n_cells)
    n_bins_per_cell = np.bincount(cell_of_bin[final_mask], minlength=n_cells)

    keep_mask = n_bins_per_cell >= min_bin_counts
    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise ValueError(
            "assign_bins: no cells retained after min_bin_counts filtering. "
            "Lower min_bin_counts or increase expand_um."
        )

    cell_seg = cell_seg if inplace else cell_seg.copy()
    cell_seg._var = bin_data.var.copy()
    cell_seg.obs["n_bins"] = n_bins_per_cell
    cell_seg.obsm[f"{boundaries_key}_expanded"] = ak.Array(
        [
            np.asarray(p.exterior.coords, dtype=np.float32) if not p.is_empty else np.zeros((0, 2), dtype=np.float32)
            for p in expanded_polys
        ]
    )
    cell_seg.X = final_X
    cell_seg._inplace_subset_obs(keep_mask)

    log.info(
        "assign_bins: %d / %d cells retained (min_bin_counts=%d); %d total bins assigned (%.1f%%)",
        n_kept, n_cells, min_bin_counts, int(final_mask.sum()), 100.0 * final_mask.sum() / n_bins,
    )

    return None if inplace else cell_seg


# ---------------------------------------------------------------------------
# segment_bin2cell (convenience recipe)
# ---------------------------------------------------------------------------


def _rasterize_bin_expression(bins_adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize per-bin total counts onto the array_row/array_col grid.

    Returns ``(image, registration_affine)`` where *image* is a ``(H, W)``
    float32 array (one pixel per bin — exact, no interpolation, since the
    array grid is itself a uniform square lattice by platform construction)
    and *registration_affine* maps this raster's own pixel coordinates back
    into the same µm frame as ``bins_adata.obsm['spatial']``.
    """
    array_row = bins_adata.obs["array_row"].to_numpy(dtype=np.int64)
    array_col = bins_adata.obs["array_col"].to_numpy(dtype=np.int64)

    if bins_adata.X is not None:
        totals = np.asarray(bins_adata.X.sum(axis=1)).ravel()
    else:
        totals = np.asarray(load_bin_counts(bins_adata, inplace=False).X.sum(axis=1)).ravel()

    library_id = next(iter(bins_adata.uns["spatial"]))
    bin_size_um = bins_adata.uns["spatial"][library_id]["bin_matrix"].bin_size_um

    min_row, min_col = int(array_row.min()), int(array_col.min())
    h, w = int(array_row.max()) - min_row + 1, int(array_col.max()) - min_col + 1
    img = np.zeros((h, w), dtype=np.float32)
    img[array_row - min_row, array_col - min_col] = totals.astype(np.float32)

    coords = np.asarray(bins_adata.obsm["spatial"], dtype=np.float64)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(array_row), min(1000, len(array_row)), replace=False)
    origin_x = float(np.median(coords[sample, 0] - (array_col[sample] - min_col) * bin_size_um))
    origin_y = float(np.median(coords[sample, 1] - (array_row[sample] - min_row) * bin_size_um))
    affine = np.array(
        [[bin_size_um, 0.0, origin_x], [0.0, bin_size_um, origin_y], [0.0, 0.0, 1.0]]
    )
    return img, affine


def segment_bin2cell(
    data: md.MuData,
    *,
    use_he: bool = True,
    use_expression: bool = False,
    backend: str = "stardist",
    he_modality: str = "he",
    bins_modality: str = "visium_hd_bins",
    image_key: str = "full",
    segment_kwargs: dict | None = None,
    expand_um: float = 4.0,
    tie_break: str = "pca",
    n_pcs: int = 15,
    min_bin_counts: int = 1,
    key_added: str = "visium_hd_cells",
) -> None:
    """Recipe: segment nuclei/cells then assign Visium HD bins to them (bin2cell).

    Automates the common case — run :func:`~astrolabe.seg.segment_stardist`
    (default) or :func:`~astrolabe.seg.segment_cellpose`, then
    :func:`assign_bins` — and attaches the result as ``data[key_added]``,
    setting it as the primary (default) modality. This is not a new
    segmentation algorithm; it only wires together existing, unmodified
    public functions. For full control (a custom segmentation, or inspecting
    the intermediate segmentation before assigning bins), call
    :func:`~astrolabe.seg.segment_stardist`/:func:`~astrolabe.seg.segment_cellpose`
    and :func:`assign_bins` yourself instead.

    Parameters
    ----------
    data
        :class:`~mudata.MuData` returned by :func:`~astrolabe.io.read_visium_hd`.
    use_he
        If ``True`` (default), segment on the H&E image (``data[he_modality]``)
        — the literal "include H&E in the segmentation" toggle. Exactly one
        of *use_he*/*use_expression* must be ``True``.
    use_expression
        If ``True``, segment on a synthetic raster of per-bin total counts
        instead of H&E (bin2cell also supports this mode, for runs with no
        usable H&E image). Rasterized onto the native bin grid (one pixel
        per 2µm bin, exact — no interpolation).
    backend
        ``'stardist'`` (default) or ``'cellpose'``.
    he_modality
        Key of the H&E modality in *data*, when *use_he=True*.
    bins_modality
        Key of the lazy bins modality in *data*.
    image_key
        Image key to segment on (passed through to the segmentation function;
        also the key the synthetic expression raster is attached under when
        *use_expression=True*).
    segment_kwargs
        Extra keyword arguments forwarded to the segmentation function (e.g.
        ``{'model_name': '2D_versatile_fluo'}``). Defaults to StarDist's
        ``'2D_versatile_he'`` model when *use_he=True* (StarDist backend
        only) — the model bin2cell itself uses for H&E nuclei segmentation.
    expand_um, tie_break, n_pcs, min_bin_counts
        Forwarded to :func:`assign_bins`.
    key_added
        Modality key for the resulting cell x gene data. Default
        ``'visium_hd_cells'``.

    Examples
    --------
    >>> mdata = ab.io.read_visium_hd("Visium_HD_run/outs/")  # no native segmentation
    >>> ab.seg.segment_bin2cell(mdata, use_he=True)
    >>> mdata.uns['default_modality']
    'visium_hd_cells'
    """
    if use_he == use_expression:
        raise ValueError(
            "segment_bin2cell: exactly one of use_he/use_expression must be True "
            f"(got use_he={use_he}, use_expression={use_expression})."
        )
    if backend not in ("stardist", "cellpose"):
        raise ValueError(f"backend must be 'stardist' or 'cellpose', got {backend!r}")

    bins_adata = data[bins_modality]
    segment_kwargs = dict(segment_kwargs or {})

    if use_he:
        if he_modality not in data.mod:
            raise KeyError(
                f"segment_bin2cell: modality '{he_modality}' not found. Load H&E "
                "first (ab.io.read_visium_hd loads it by default) or pass "
                "use_expression=True to segment on bin expression instead."
            )
        source = data[he_modality]
        if backend == "stardist":
            segment_kwargs.setdefault("model_name", "2D_versatile_he")
        cell_seg = _run_segmentation(backend, source, image_key, segment_kwargs)
    else:
        img, affine = _rasterize_bin_expression(bins_adata)
        cell_seg = _segment_on_raster(backend, img, affine, image_key, segment_kwargs, bins_adata)

    assign_bins(
        bins_adata, cell_seg,
        expand_um=expand_um, tie_break=tie_break, n_pcs=n_pcs,
        min_bin_counts=min_bin_counts, inplace=True,
    )

    attach_modality(data, cell_seg, key_added, set_default=True)
    log.info("segment_bin2cell: attached '%s' as the primary modality", key_added)


def _run_segmentation(backend: str, source, image_key: str, segment_kwargs: dict) -> ad.AnnData:
    if backend == "stardist":
        return segment_stardist(source, image_key=image_key, **segment_kwargs)
    return segment_cellpose(source, image_key=image_key, **segment_kwargs)


def _segment_on_raster(
    backend: str, img: np.ndarray, affine: np.ndarray, image_key: str,
    segment_kwargs: dict, bins_adata: ad.AnnData,
) -> ad.AnnData:
    """Write *img* to a temp TIFF, wrap it in a throwaway ImageSpec, segment, clean up."""
    library_id = next(iter(bins_adata.uns["spatial"]))
    bin_size_um = bins_adata.uns["spatial"][library_id]["bin_matrix"].bin_size_um

    fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        tifffile.imwrite(tmp_path, img)
        spec = ImageSpec(
            path=tmp_path,
            pixel_size_um=bin_size_um,
            dimension_order="YX",
            full_shape=img.shape,
            level_shapes=[img.shape],
            level_scales=[(1.0, 1.0)],
            registration_affine=affine,
        )
        throwaway = ad.AnnData(obs=pd.DataFrame(index=[]))
        throwaway.uns["spatial"] = {library_id: {"images": {image_key: spec}}}
        return _run_segmentation(backend, throwaway, image_key, segment_kwargs)
    finally:
        os.unlink(tmp_path)
