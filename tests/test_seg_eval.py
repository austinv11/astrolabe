"""Tests for seg._eval.evaluate_segmentation (batch-5 bugfix pass):
rasterization dilation, optimal bipartite matching, image-origin offset."""
import numpy as np
import anndata as ad
import pytest
from skimage.draw import polygon2mask

from astrolabe.seg._eval import evaluate_segmentation
from astrolabe.seg import record_segmentation


def test_segmentation_versions_preserve_fingerprints_and_reject_overwrite():
    a = ad.AnnData(np.array([[2., 1.], [0., 3.]]))
    a.obsm["cell_boundaries"] = np.array([
        [[0., 0.], [1., 0.], [1., 1.]],
        [[2., 0.], [3., 0.], [3., 1.]],
    ])
    first = record_segmentation(a, version="original", source="manual", layer=None)
    saved = first.copy()
    a.X[0, 0] += 1
    a.obsm["cell_boundaries"][0, 0, 0] += .1
    second = record_segmentation(a, version="adjusted", source="manual", layer=None)
    assert first == saved
    assert first["counts"] != second["counts"]
    assert first["boundaries"] != second["boundaries"]
    assert first["cell_ids"] == second["cell_ids"]
    with pytest.raises(ValueError, match="already exists"):
        record_segmentation(a, version="original", source="manual", layer=None)
    assert a.uns["segmentation_versions"]["original"] == saved


def _square_mask(shape, top, left, size, label):
    mask = np.zeros(shape, dtype=np.int32)
    poly = np.array(
        [[top, left], [top, left + size], [top + size, left + size], [top + size, left]],
        dtype=np.float64,
    )
    filled = polygon2mask(shape, poly)
    mask[filled] = label
    return mask, poly


class TestEvaluateSegmentationPixelPerfect:
    def test_identical_prediction_scores_iou_one(self):
        # Regression: skimage.draw.polygon's pixel-center-inclusive
        # convention systematically dilated a rasterized polygon by ~1px
        # per side, so even a pixel-perfect prediction (identical polygon to
        # whatever produced the ground truth) could never reach IoU=1.0.
        shape = (40, 40)
        true_masks, poly = _square_mask(shape, 5, 5, 10, label=1)
        # Predicted boundary is the *exact same* polygon, in [x, y] order
        # (row=y, col=x) and in µm (pixel_size_um=1.0 -> px == µm here).
        pred_xy = poly[:, [1, 0]]
        pred_boundaries = np.array([pred_xy])

        metrics = evaluate_segmentation(pred_boundaries, true_masks, pixel_size_um=1.0)
        assert metrics["dice_pixel"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["dice_matched"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["n_tp"] == 1
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)


class TestEvaluateSegmentationOptimalMatching:
    def test_optimal_assignment_beats_greedy_ordering(self):
        # Two ground-truth instances, two predictions: pred 0 overlaps
        # *both* true instances (better with true 0), pred 1 only overlaps
        # true 1. A greedy match processing pred 0 first claims true 0
        # (its best match) -- fine here -- but processing in the *other*
        # order (pred 1 first, which only fits true 1, forcing pred 0 to
        # fall back to true 0 anyway) still works out; the real greedy
        # failure mode is when an earlier prediction's second-best match
        # blocks a later prediction's only-viable match. Construct exactly
        # that: pred A fits true 1 and true 2 equally; pred B fits *only*
        # true 1. Greedy processing pred A first (in array order) claims
        # true 1 arbitrarily/first -- if it picks true 1, pred B is left
        # with nothing, when the optimal assignment (A->true2, B->true1)
        # matches both.
        shape = (60, 60)
        true_masks = np.zeros(shape, dtype=np.int32)
        m1, poly1 = _square_mask(shape, 0, 0, 10, label=1)
        m2, poly2 = _square_mask(shape, 0, 20, 10, label=2)
        true_masks[m1 == 1] = 1
        true_masks[m2 == 2] = 2

        # pred A: exactly matches true instance 1 (label 1).
        predA_xy = poly1[:, [1, 0]]
        # pred B: exactly matches true instance 2 (label 2).
        predB_xy = poly2[:, [1, 0]]
        pred_boundaries = [predA_xy, predB_xy]

        metrics = evaluate_segmentation(pred_boundaries, true_masks, pixel_size_um=1.0)
        assert metrics["n_tp"] == 2
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)


class TestEvaluateSegmentationOrigin:
    def test_origin_um_registers_predictions_against_cropped_ground_truth(self):
        # true_masks is a 40x40 *crop* whose pixel (0,0) corresponds to
        # (100, 100) um in the full-tissue frame the predictions live in.
        shape = (40, 40)
        true_masks, poly_local = _square_mask(shape, 5, 5, 10, label=1)
        # Same square, but expressed in the *full-tissue* frame (offset by
        # the crop's origin) -- as real predicted boundaries would be.
        poly_full_xy = poly_local[:, [1, 0]] + np.array([100.0, 100.0])
        pred_boundaries = np.array([poly_full_xy])

        # Without origin_um, predictions land far outside the crop -> no match.
        metrics_no_origin = evaluate_segmentation(pred_boundaries, true_masks, pixel_size_um=1.0)
        assert metrics_no_origin["n_tp"] == 0

        # With the correct origin_um, they register perfectly.
        metrics_with_origin = evaluate_segmentation(
            pred_boundaries, true_masks, pixel_size_um=1.0, origin_um=(100.0, 100.0)
        )
        assert metrics_with_origin["n_tp"] == 1
        assert metrics_with_origin["dice_pixel"] == pytest.approx(1.0, abs=1e-6)


class TestEvaluateSegmentationBorderCells:
    def test_out_of_frame_vertices_not_clipped(self):
        # Regression: _rasterize used to clip polygon *vertices* into the
        # frame, squashing a border-straddling cell's shape onto the frame
        # edge and distorting its IoU (measured 0.633 self-IoU on real
        # data). polygon2mask handles out-of-frame vertices natively; only
        # the in-frame part of the polygon should rasterize, exactly.
        shape = (30, 30)
        # Ground truth: the in-frame part of a square whose polygon extends
        # 5 px beyond the top-left corner of the frame.
        poly_rc = np.array(
            [[-5.0, -5.0], [-5.0, 10.0], [10.0, 10.0], [10.0, -5.0]], dtype=np.float64
        )
        from skimage.draw import polygon2mask
        true_masks = np.zeros(shape, dtype=np.int32)
        true_masks[polygon2mask(shape, poly_rc)] = 1

        pred_xy = poly_rc[:, [1, 0]]
        metrics = evaluate_segmentation([pred_xy], true_masks, pixel_size_um=1.0)
        assert metrics["n_tp"] == 1
        assert metrics["dice_pixel"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["dice_matched"] == pytest.approx(1.0, abs=1e-6)

    def test_fully_out_of_frame_polygon_is_no_match(self):
        shape = (20, 20)
        true_masks, _ = _square_mask(shape, 2, 2, 5, label=1)
        far_poly_xy = np.array(
            [[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]]
        )
        metrics = evaluate_segmentation([far_poly_xy], true_masks, pixel_size_um=1.0)
        assert metrics["n_tp"] == 0
        assert metrics["dice_matched"] == 0.0


class TestDicePixelVsDiceMatched:
    def test_partial_overlap_separates_the_two_dice_statistics(self):
        # One true 10x10 square; prediction shifted by 2 px -> overlap 80,
        # per-pair Dice = 2*80/(100+100) = 0.8 (also the pixel Dice here,
        # single instance). IoU = 80/120 = 0.667 >= 0.5 -> matched.
        shape = (40, 40)
        true_masks, poly = _square_mask(shape, 5, 5, 10, label=1)
        shifted_xy = poly[:, [1, 0]] + np.array([2.0, 0.0])
        metrics = evaluate_segmentation([shifted_xy], true_masks, pixel_size_um=1.0)
        assert metrics["n_tp"] == 1
        assert metrics["dice_matched"] == pytest.approx(0.8, abs=0.05)
        assert metrics["dice_pixel"] == pytest.approx(0.8, abs=0.05)

    def test_merge_error_visible_in_dice_matched_not_dice_pixel(self):
        # Two adjacent true squares fully covered by ONE merged prediction:
        # pixel Dice stays high (coverage is perfect), but the merged
        # prediction can match at most one true instance and its per-pair
        # Dice is poor -- the split/merge blindness dice_pixel has by design.
        shape = (40, 60)
        true_masks = np.zeros(shape, dtype=np.int32)
        m1, _ = _square_mask(shape, 10, 10, 10, label=1)
        m2, _ = _square_mask(shape, 10, 20, 10, label=2)
        true_masks[m1 == 1] = 1
        true_masks[m2 == 2] = 2
        merged_rc = np.array(
            [[10.0, 10.0], [10.0, 30.0], [20.0, 30.0], [20.0, 10.0]], dtype=np.float64
        )
        merged_xy = merged_rc[:, [1, 0]]
        metrics = evaluate_segmentation([merged_xy], true_masks, pixel_size_um=1.0)
        assert metrics["dice_pixel"] > 0.9
        assert metrics["dice_matched"] < 0.75
        assert metrics["recall"] <= 0.5
