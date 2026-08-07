"""Tests for _transform_coords dtype promotion (audit fix 2.9)."""
import numpy as np
import pytest

from astrolabe._utils import _transform_coords


def test_int_coords_promoted_to_float():
    coords = np.array([[100, 200], [300, 400]], dtype=np.int32)
    M = np.eye(3)
    M[0, 2] = 0.5  # sub-pixel shift
    out = _transform_coords(coords, M)
    assert np.issubdtype(out.dtype, np.floating), f"expected float, got {out.dtype}"
    assert abs(out[0, 0] - 100.5) < 1e-5, "sub-pixel precision must be preserved for int input"


def test_int64_coords_promoted_to_float():
    coords = np.array([[0, 0], [1000, 2000]], dtype=np.int64)
    out = _transform_coords(coords, np.eye(3))
    assert np.issubdtype(out.dtype, np.floating)


def test_float32_coords_preserved():
    coords = np.array([[1.5, 2.5], [3.7, 4.2]], dtype=np.float32)
    out = _transform_coords(coords, np.eye(3))
    assert out.dtype == np.float32


def test_float64_coords_preserved():
    coords = np.array([[1.5, 2.5]], dtype=np.float64)
    out = _transform_coords(coords, np.eye(3))
    assert out.dtype == np.float64


def test_identity_transform_no_change():
    coords = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = _transform_coords(coords, np.eye(3))
    np.testing.assert_allclose(out, coords, rtol=1e-6)


def test_translation_applied_correctly():
    coords = np.array([[0.0, 0.0]], dtype=np.float32)
    M = np.eye(3)
    M[0, 2] = 100.0
    M[1, 2] = 200.0
    out = _transform_coords(coords, M)
    np.testing.assert_allclose(out[0], [100.0, 200.0], rtol=1e-6)
