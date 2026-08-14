"""Tests for pp._if's Z-stack channel-axis handling (batch-5 bugfix pass)."""
import numpy as np
import pytest

from astrolabe.pp._if import _resolve_channel_axis, _read_tile


class TestResolveChannelAxis:
    def test_explicit_c_axis_used(self):
        assert _resolve_channel_axis("CYX", 3) == 0
        assert _resolve_channel_axis("YXC", 3) == 2

    def test_z_stack_with_no_channel_returns_none(self):
        # Regression: a real Xenium morphology.ome.tif is 'ZYX' (multiple
        # DAPI z-planes, no antibody panel at all) -- this used to fall
        # through to the ndim==3 default (axis 0), silently treating the Z
        # axis as if it were a channel axis and reporting 11 spurious
        # "channels" for a single-stain z-stack.
        assert _resolve_channel_axis("ZYX", 3) is None

    def test_t_stack_with_no_channel_returns_none(self):
        assert _resolve_channel_axis("TYX", 3) is None

    def test_ambiguous_3d_with_no_labels_falls_back_to_axis_0(self):
        # No 'C', 'Z', or 'T' label at all -- preserve the prior best-guess
        # fallback for genuinely unlabeled arrays.
        assert _resolve_channel_axis("QYX", 3) == 0


class TestReadTileCollapsesZWhenNoChannel:
    def test_z_stack_max_projects_to_single_channel(self):
        # 5-plane 'ZYX' stack; channel_axis=None should collapse Z via
        # max-projection to a single implicit channel, not report 5.
        z, h, w = 5, 20, 20
        rng = np.random.default_rng(0)
        image = rng.uniform(0, 1, (z, h, w)).astype(np.float32)

        tile = _read_tile(image, None, row_min=2, row_max=10, col_min=3, col_max=12)

        assert tile.shape[0] == 1  # collapsed to a single channel
        assert tile.shape[1:] == (8, 9)
        expected = image[:, 2:10, 3:12].max(axis=0)
        np.testing.assert_allclose(tile[0], expected)

    def test_explicit_channel_axis_still_works_unchanged(self):
        c, h, w = 3, 20, 20
        rng = np.random.default_rng(1)
        image = rng.uniform(0, 1, (c, h, w)).astype(np.float32)
        tile = _read_tile(image, 0, row_min=0, row_max=5, col_min=0, col_max=5)
        assert tile.shape == (3, 5, 5)
        np.testing.assert_allclose(tile, image[:, 0:5, 0:5])
