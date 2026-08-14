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
        ``{'precision': float, 'recall': float, 'f1': float, 'dice': float,
        'n_pred': int, 'n_true': int, 'n_tp': int}``

    Examples
    --------
    >>> metrics = ab.seg.evaluate_segmentation(
    ...     mdata["xenium"].obsm["cell_boundaries"],
    ...     true_label_image,
    ...     pixel_size_um=0.2125,
    ... )
    >>> print(f"F1: {metrics['f1']:.3f}")
    """
    from scipy.optimize import linear_sum_assignment

    H, W = true_masks.shape
    scale = 1.0 / pixel_size_um  # µm → pixels
    origin_px = (float(origin_um[0]) * scale, float(origin_um[1]) * scale)

    pred_polys_px = _to_polygon_list(pred_boundaries, scale, origin_px)

    n_pred = len(pred_polys_px)
    true_ids = np.unique(true_masks)
    true_ids = true_ids[true_ids > 0]
    n_true = len(true_ids)
    true_id_to_row = {int(tid): i for i, tid in enumerate(true_ids)}
    true_areas = np.array([int((true_masks == tid).sum()) for tid in true_ids], dtype=np.float64)

    log.info(
        "evaluate_segmentation: %d predicted vs %d ground-truth instances",
        n_pred, n_true,
    )

    pred_masks = [_rasterize(poly_px, H, W) for poly_px in pred_polys_px]

    # Full (n_pred x n_true) IoU matrix, restricted to instances each
    # prediction's rasterized mask actually overlaps (sparse in practice).
    iou = np.zeros((n_pred, n_true), dtype=np.float64)
    for i, (rr, cc) in enumerate(pred_masks):
        if len(rr) == 0:
            continue
        covered = true_masks[rr, cc]
        unique, counts = np.unique(covered[covered > 0], return_counts=True)
        if len(unique) == 0:
            continue
        pred_area = len(rr)
        for tid, overlap in zip(unique, counts):
            j = true_id_to_row[int(tid)]
            union = pred_area + true_areas[j] - overlap
            iou[i, j] = overlap / union if union > 0 else 0.0

    n_tp = 0
    if n_pred > 0 and n_true > 0 and iou.max(initial=0.0) > 0:
        # linear_sum_assignment minimizes cost -> maximize IoU by negating.
        # Only pairs meeting iou_threshold count as candidate matches;
        # everything else is given a prohibitive cost so the optimal
        # assignment never picks a below-threshold pair over leaving a row
        # unmatched.
        cost = np.where(iou >= iou_threshold, -iou, 1.0)
        row_ind, col_ind = linear_sum_assignment(cost)
        n_tp = int((cost[row_ind, col_ind] < 0).sum())

    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall = n_tp / n_true if n_true > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    dice = _pixel_dice(pred_masks, true_masks, H, W)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "n_pred": n_pred,
        "n_true": n_true,
        "n_tp": n_tp,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    """
    from skimage.draw import polygon2mask

    if len(poly_px) < 3:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    rows = np.clip(poly_px[:, 1], 0, H - 1)
    cols = np.clip(poly_px[:, 0], 0, W - 1)
    mask = polygon2mask((H, W), np.column_stack([rows, cols]))
    rr, cc = np.nonzero(mask)
    return rr, cc


def _pixel_dice(
    pred_masks: list[tuple[np.ndarray, np.ndarray]],
    true_masks: np.ndarray,
    H: int,
    W: int,
) -> float:
    """Pixel-level Dice coefficient between union of predicted and true foreground."""
    pred_fg = np.zeros((H, W), dtype=bool)
    for rr, cc in pred_masks:
        if len(rr) == 0:
            continue
        pred_fg[rr, cc] = True

    true_fg = true_masks > 0
    intersection = int((pred_fg & true_fg).sum())
    denom = int(pred_fg.sum()) + int(true_fg.sum())
    return 2.0 * intersection / denom if denom > 0 else 0.0
