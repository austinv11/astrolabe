"""
Segmentation evaluation metrics.

Compare a predicted segmentation (boundary polygons in obsm) against a
ground-truth label image to compute precision, recall, F1, and Dice at a
given IoU threshold.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = ["evaluate_segmentation"]

log = logging.getLogger(__name__)


def evaluate_segmentation(
    pred_boundaries: "np.ndarray | object",
    true_masks: np.ndarray,
    pixel_size_um: float,
    iou_threshold: float = 0.5,
    origin_um: tuple[float, float] = (0.0, 0.0),
) -> dict[str, float]:
    """Evaluate a predicted segmentation against a ground-truth label image.

    Predicted and ground-truth instances are matched by **optimal bipartite
    assignment** (maximizing total IoU, via the Hungarian algorithm) over
    every pair with IoU >= *iou_threshold* — not a greedy first-come match,
    which can under-report true positives when processing predictions in an
    order that lets an earlier, worse match claim an instance a later
    prediction would have matched better.

    Scales to whole-slide inputs: pairwise IoU is accumulated sparsely (only
    pairs that actually share pixels), and the optimal matching runs
    independently per connected component of the overlap graph — never on a
    dense ``(n_pred, n_true)`` matrix.

    Parameters
    ----------
    pred_boundaries
        Predicted cell boundaries — either an :class:`awkward.Array` of
        variable-length ``(n_verts, 2)`` polygons in µm (as stored in
        ``obsm['cell_boundaries']``), or a plain ``(n_cells, n_verts, 2)``
        numpy array. Coordinates are in the same µm frame as the *source*
        image *before* any crop — see *origin_um*.
    true_masks
        ``(H, W)`` integer label image where 0 = background and each positive
        integer is a distinct ground-truth instance. Pixel ``(0, 0)``
        corresponds to *origin_um* in the predicted boundaries' µm frame —
        pass *origin_um* when *true_masks* is a crop of a larger image the
        predictions were computed on, rather than the whole tissue.
    pixel_size_um
        Physical size of one pixel in microns; used to rasterise predicted
        boundaries into pixel space.
    iou_threshold
        Minimum IoU for a predicted / ground-truth pair to count as a match
        (default ``0.5`` — COCO-style).
    origin_um
        ``(x0, y0)`` µm offset of ``true_masks``' pixel ``(0, 0)`` in the
        same frame as *pred_boundaries*. ``(0.0, 0.0)`` (default) assumes
        *true_masks* covers the same origin as the predictions -- the
        common case of evaluating against a ground-truth label image for
        the whole predicted region. Set this when *true_masks* is instead a
        cropped sub-region starting elsewhere in a larger tissue.

    Returns
    -------
    dict
        ``{'precision': float, 'recall': float, 'f1': float,
        'dice_pixel': float, 'dice_matched': float,
        'n_pred': int, 'n_true': int, 'n_tp': int}``

        Two distinct Dice statistics are reported:

        * ``dice_pixel`` — global **pixel-level** Dice between the union of
          all predicted foreground and all ground-truth foreground. It is
          invariant to instance splits/merges (one prediction covering two
          true cells scores the same as two correct predictions), so it
          measures foreground *coverage*, not instance agreement.
        * ``dice_matched`` — mean **per-instance** Dice over matched
          (prediction, ground truth) pairs only
          (``2·|overlap| / (|pred| + |true|)`` per pair). It measures how
          well matched instances agree in shape, and is ``0.0`` when
          nothing matched.

    Examples
    --------
    >>> metrics = ab.seg.evaluate_segmentation(
    ...     mdata["xenium"].obsm["cell_boundaries"],
    ...     true_label_image,
    ...     pixel_size_um=0.2125,
    ... )
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    H, W = true_masks.shape
    scale = 1.0 / pixel_size_um  # µm → pixels
    origin_px = (float(origin_um[0]) * scale, float(origin_um[1]) * scale)

    pred_polys_px = _to_polygon_list(pred_boundaries, scale, origin_px)

    n_pred = len(pred_polys_px)
    true_ids = np.unique(true_masks)
    true_ids = true_ids[true_ids > 0]
    n_true = len(true_ids)

    # Per-instance areas from a single pass over the label image (bincount),
    # not one full-image `true_masks == tid` scan per instance.
    id_counts = np.bincount(true_masks.ravel().astype(np.int64))
    true_areas = id_counts[true_ids].astype(np.float64)
    true_row_of_id = np.full(int(true_ids.max()) + 1 if n_true else 1, -1, dtype=np.int64)
    true_row_of_id[true_ids] = np.arange(n_true)

    log.info(
        "evaluate_segmentation: %d predicted vs %d ground-truth instances",
        n_pred, n_true,
    )

    pred_masks = [_rasterize(poly_px, H, W) for poly_px in pred_polys_px]
    pred_areas = np.array([len(rr) for rr, _ in pred_masks], dtype=np.float64)

    # Sparse COO IoU accumulation: only (prediction, true-instance) pairs
    # that actually share pixels ever get an entry — the dense
    # (n_pred x n_true) matrix this replaces is ~200 GB at slide scale.
    pi_list: list[np.ndarray] = []
    tj_list: list[np.ndarray] = []
    ov_list: list[np.ndarray] = []
    for i, (rr, cc) in enumerate(pred_masks):
        if len(rr) == 0:
            continue
        covered = true_masks[rr, cc]
        covered = covered[covered > 0]
        if covered.size == 0:
            continue
        unique, counts = np.unique(covered, return_counts=True)
        pi_list.append(np.full(len(unique), i, dtype=np.int64))
        tj_list.append(true_row_of_id[unique.astype(np.int64)])
        ov_list.append(counts.astype(np.float64))

    n_tp = 0
    matched_dice: list[float] = []
    if pi_list:
        pi = np.concatenate(pi_list)
        tj = np.concatenate(tj_list)
        ov = np.concatenate(ov_list)
        union = pred_areas[pi] + true_areas[tj] - ov
        iou = np.where(union > 0, ov / union, 0.0)

        keep = iou >= iou_threshold
        pi, tj, ov, iou = pi[keep], tj[keep], ov[keep], iou[keep]
        if len(pi) > 0:
            n_tp, matched_dice = _match_components(
                pi, tj, iou, ov, pred_areas, true_areas
            )

    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall = n_tp / n_true if n_true > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    dice_pixel = _pixel_dice(pred_masks, true_masks, H, W)
    dice_matched = float(np.mean(matched_dice)) if matched_dice else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice_pixel": float(dice_pixel),
        "dice_matched": dice_matched,
        "n_pred": n_pred,
        "n_true": n_true,
        "n_tp": n_tp,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _match_components(
    pi: np.ndarray,
    tj: np.ndarray,
    iou: np.ndarray,
    ov: np.ndarray,
    pred_areas: np.ndarray,
    true_areas: np.ndarray,
) -> tuple[int, list[float]]:
    """Optimal bipartite matching over above-threshold IoU pairs.

    Runs Hungarian matching independently per connected component of the
    bipartite overlap graph — components are tiny in practice (a cell
    overlaps a handful of neighbors), so each dense sub-problem stays a few
    rows/columns even when the slide has 100k+ instances.

    Returns ``(n_tp, per-matched-pair Dice list)``.
    """
    import scipy.sparse as sp
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components

    # Compact node ids: preds first, then true instances.
    upi, pi_c = np.unique(pi, return_inverse=True)
    utj, tj_c = np.unique(tj, return_inverse=True)
    n_p, n_t = len(upi), len(utj)
    n_nodes = n_p + n_t

    adj = sp.coo_matrix(
        (np.ones(len(pi_c)), (pi_c, tj_c + n_p)), shape=(n_nodes, n_nodes)
    )
    n_comp, labels = connected_components(adj, directed=False)

    # Group edges by the component of their prediction node.
    edge_comp = labels[pi_c]
    order = np.argsort(edge_comp, kind="stable")
    comp_sorted = edge_comp[order]
    boundaries = np.flatnonzero(np.diff(comp_sorted)) + 1
    edge_groups = np.split(order, boundaries)

    n_tp = 0
    matched_dice: list[float] = []
    for edges in edge_groups:
        rows = pi_c[edges]
        cols = tj_c[edges]
        urows, rlocal = np.unique(rows, return_inverse=True)
        ucols, clocal = np.unique(cols, return_inverse=True)
        # Dense sub-matrix restricted to this component. Cost 1.0 for
        # non-overlapping pairs is prohibitive vs. any negative -IoU, so
        # the optimal assignment never picks an absent pair over leaving
        # a row unmatched.
        cost = np.ones((len(urows), len(ucols)), dtype=np.float64)
        cost[rlocal, clocal] = -iou[edges]
        ov_sub = np.zeros_like(cost)
        ov_sub[rlocal, clocal] = ov[edges]
        r_ind, c_ind = linear_sum_assignment(cost)
        hit = cost[r_ind, c_ind] < 0
        n_tp += int(hit.sum())
        for r, c in zip(r_ind[hit], c_ind[hit]):
            p_global = upi[urows[r]]
            t_global = utj[ucols[c]]
            overlap = ov_sub[r, c]
            denom = pred_areas[p_global] + true_areas[t_global]
            matched_dice.append(2.0 * overlap / denom if denom > 0 else 0.0)

    return n_tp, matched_dice


def _to_polygon_list(boundaries, scale: float, origin_px: tuple[float, float]) -> list[np.ndarray]:
    """Convert obsm boundaries to a list of (n_verts, 2) float64 pixel arrays,
    offset so *origin_px* (in the same µm->px scale) maps to pixel (0, 0)."""
    ox, oy = origin_px
    offset = np.array([ox, oy], dtype=np.float64)

    def _to_px(verts: np.ndarray) -> np.ndarray:
        return np.asarray(verts, dtype=np.float64) * scale - offset

    polys: list[np.ndarray] = []
    try:
        import awkward as ak
        if isinstance(boundaries, ak.Array):
            for i in range(len(boundaries)):
                polys.append(_to_px(np.asarray(boundaries[i])))
            return polys
    except ImportError:
        pass

    arr = np.asarray(boundaries, dtype=np.float64)
    if arr.ndim == 3:
        for i in range(arr.shape[0]):
            polys.append(_to_px(arr[i]))
    elif arr.ndim == 2:
        polys.append(_to_px(arr))
    return polys


def _rasterize(poly_px: np.ndarray, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one (n_verts, 2) [x, y] pixel-space polygon to (rows, cols).

    Uses ``skimage.draw.polygon2mask``, which treats vertex coordinates as
    the polygon's actual boundary (a proper point-in-polygon containment
    test), not ``skimage.draw.polygon``'s pixel-*center*-inclusive
    convention -- the latter systematically dilates a rasterized polygon by
    about one pixel per side, so even a pixel-perfect prediction (identical
    polygon to whatever produced the ground truth) could never score
    IoU=1.0 against it.

    Rasterization happens inside the polygon's own bounding box intersected
    with the frame (never a full (H, W) mask per instance), and vertices are
    **not** clipped to the frame: ``polygon2mask`` handles out-of-frame
    vertices correctly, whereas clipping used to squash a border-straddling
    cell's shape onto the frame edge and distort its IoU.
    """
    from skimage.draw import polygon2mask

    empty = (np.array([], dtype=np.intp), np.array([], dtype=np.intp))
    if len(poly_px) < 3:
        return empty

    rows = poly_px[:, 1]
    cols = poly_px[:, 0]
    row0 = max(0, int(np.floor(rows.min())))
    row1 = min(H, int(np.ceil(rows.max())) + 1)
    col0 = max(0, int(np.floor(cols.min())))
    col1 = min(W, int(np.ceil(cols.max())) + 1)
    if row0 >= row1 or col0 >= col1:
        return empty

    local = np.column_stack([rows - row0, cols - col0])
    mask = polygon2mask((row1 - row0, col1 - col0), local)
    rr, cc = np.nonzero(mask)
    return rr + row0, cc + col0


def _pixel_dice(
    pred_masks: list[tuple[np.ndarray, np.ndarray]],
    true_masks: np.ndarray,
    H: int,
    W: int,
) -> float:
    """Pixel-level Dice coefficient between union of predicted and true foreground.

    Split/merge-invariant by construction — see the ``dice_pixel`` vs.
    ``dice_matched`` distinction in :func:`evaluate_segmentation`'s docs.
    """
    pred_fg = np.zeros((H, W), dtype=bool)
    for rr, cc in pred_masks:
        if len(rr) == 0:
            continue
        pred_fg[rr, cc] = True

    true_fg = true_masks > 0
    intersection = int((pred_fg & true_fg).sum())
    denom = int(pred_fg.sum()) + int(true_fg.sum())
    return 2.0 * intersection / denom if denom > 0 else 0.0
