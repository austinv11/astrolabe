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
) -> dict[str, float]:
    """Evaluate a predicted segmentation against a ground-truth label image.

    For each predicted boundary polygon, the best-matching ground-truth
    instance is found by IoU overlap.  A prediction is a true positive if
    its IoU with the matched ground-truth instance exceeds *iou_threshold*.

    Parameters
    ----------
    pred_boundaries
        Predicted cell boundaries — either an :class:`awkward.Array` of
        variable-length ``(n_verts, 2)`` polygons in µm (as stored in
        ``obsm['cell_boundaries']``), or a plain ``(n_cells, n_verts, 2)``
        numpy array.  Coordinates must be in the same µm frame as the image.
    true_masks
        ``(H, W)`` integer label image where 0 = background and each positive
        integer is a distinct ground-truth instance.  Must be in pixel
        coordinates; the conversion to µm is done via *pixel_size_um*.
    pixel_size_um
        Physical size of one pixel in microns; used to rasterise predicted
        boundaries into pixel space.
    iou_threshold
        Minimum IoU for a predicted / ground-truth pair to count as a match
        (default ``0.5`` — COCO-style).

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
    from skimage.draw import polygon as sk_polygon

    H, W = true_masks.shape
    scale = 1.0 / pixel_size_um  # µm → pixels

    # ------------------------------------------------------------------ #
    # Convert pred_boundaries to list of (n_verts, 2) arrays in pixels   #
    # ------------------------------------------------------------------ #
    pred_polys_px = _to_polygon_list(pred_boundaries, scale)

    # ------------------------------------------------------------------ #
    # Rasterise each predicted polygon → binary mask                      #
    # ------------------------------------------------------------------ #
    n_pred = len(pred_polys_px)
    true_ids = np.unique(true_masks)
    true_ids = true_ids[true_ids > 0]
    n_true = len(true_ids)

    log.info(
        "evaluate_segmentation: %d predicted vs %d ground-truth instances",
        n_pred, n_true,
    )

    matched_true: set[int] = set()
    n_tp = 0

    for poly_px in pred_polys_px:
        if len(poly_px) < 3:
            continue

        rows = np.clip(poly_px[:, 1], 0, H - 1)
        cols = np.clip(poly_px[:, 0], 0, W - 1)
        rr, cc = sk_polygon(rows, cols, shape=(H, W))
        if len(rr) == 0:
            continue

        # Unique true labels under this predicted mask
        covered = true_masks[rr, cc]
        unique, counts = np.unique(covered[covered > 0], return_counts=True)
        if len(unique) == 0:
            continue

        # Match to true instance with highest IoU
        best_iou = 0.0
        best_id = -1
        pred_area = len(rr)
        for tid, overlap in zip(unique, counts):
            true_area = int((true_masks == tid).sum())
            union = pred_area + true_area - overlap
            iou = overlap / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_id = int(tid)

        if best_iou >= iou_threshold and best_id not in matched_true:
            n_tp += 1
            matched_true.add(best_id)

    precision = n_tp / n_pred if n_pred > 0 else 0.0
    recall = n_tp / n_true if n_true > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # Dice over the pixel-level union (Sørensen–Dice coefficient)
    dice = _pixel_dice(pred_polys_px, true_masks, H, W, sk_polygon)

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


def _to_polygon_list(boundaries, scale: float) -> list[np.ndarray]:
    """Convert obsm boundaries to a list of (n_verts, 2) float64 pixel arrays."""
    polys: list[np.ndarray] = []
    try:
        import awkward as ak
        if isinstance(boundaries, ak.Array):
            for i in range(len(boundaries)):
                verts = np.asarray(boundaries[i], dtype=np.float64) * scale
                polys.append(verts)
            return polys
    except ImportError:
        pass

    arr = np.asarray(boundaries, dtype=np.float64)
    if arr.ndim == 3:
        for i in range(arr.shape[0]):
            polys.append(arr[i] * scale)
    elif arr.ndim == 2:
        polys.append(arr * scale)
    return polys


def _pixel_dice(
    pred_polys_px: list[np.ndarray],
    true_masks: np.ndarray,
    H: int,
    W: int,
    sk_polygon,
) -> float:
    """Pixel-level Dice coefficient between union of predicted and true foreground."""
    pred_fg = np.zeros((H, W), dtype=bool)
    for poly_px in pred_polys_px:
        if len(poly_px) < 3:
            continue
        rr, cc = sk_polygon(
            np.clip(poly_px[:, 1], 0, H - 1),
            np.clip(poly_px[:, 0], 0, W - 1),
            shape=(H, W),
        )
        pred_fg[rr, cc] = True

    true_fg = true_masks > 0
    intersection = int((pred_fg & true_fg).sum())
    denom = int(pred_fg.sum()) + int(true_fg.sum())
    return 2.0 * intersection / denom if denom > 0 else 0.0
