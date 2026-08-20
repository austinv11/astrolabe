"""Tests for pl._filters: µm-unit filter parameters and the FFT notch fix.

Two concerns are covered here:

1. **Physical-unit (µm) filter parameters.** The same ``filter_chain`` is
   applied at full resolution by ``pp.extract_if_intensities`` and at a
   zoom-dependent downsampled resolution by the renderer, so filters take
   their bandwidths in microns (``sigma_um`` / ``kernel_um`` / ``cutoff_um``
   / ``fov_size_um``) and convert to pixels per call via the ``um_per_px``
   keyword. Legacy pixel-unit parameters remain and are resolution-locked.

2. **FFTFilter per-axis notch width** (batch-5 bugfix). Note: the number of
   harmonics notched scales with ``floor(0.5 * fov_size_px)`` (the
   fundamental frequency 1/fov_size_px gets *more* sub-Nyquist harmonics for
   a *larger* fov_size_px) -- these tests deliberately use a small
   fov_size_px (a handful of pixels) purely to keep the notch to a single,
   precisely-known harmonic for a controlled unit test, not because that's a
   realistic tile size.
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter as scipy_gaussian

from astrolabe._utils import _call_image_filter
from astrolabe.io._io import _filter_chain_from_json, _filter_chain_to_json
from astrolabe.pl._filters import CLAHEFilter, FFTFilter, GaussianFilter


def _sinusoid(period_px: float, shape: tuple[int, int]) -> np.ndarray:
    """Zero-mean horizontal sinusoid with the given period in pixels."""
    x = np.arange(shape[1], dtype=np.float32)
    row = np.sin(2 * np.pi * x / period_px).astype(np.float32)
    return np.broadcast_to(row, shape).copy()


class TestGaussianFilterMicronUnits:
    def test_sigma_um_gives_same_physical_smoothing_across_resolutions(self):
        # The core defect: a pixel-unit sigma has a physical bandwidth that
        # changes with the pyramid level being filtered. A µm-unit sigma must
        # not. We measure smoothing via the attenuation of a sinusoid whose
        # *physical* period is fixed (32 µm): a Gaussian of physical sigma
        # 4 µm attenuates it by exp(-2 pi^2 (sigma/T)^2) ~= 0.735 regardless
        # of the pixel grid it is computed on.
        period_um = 32.0
        sigma_um = 4.0
        expected = np.exp(-2 * np.pi ** 2 * (sigma_um / period_um) ** 2)

        # Full resolution: 0.5 µm/px -> period 64 px, sigma 8 px.
        full = _sinusoid(period_um / 0.5, (256, 256))
        # 4x downsample: 2.0 µm/px -> period 16 px, sigma 2 px.
        down = _sinusoid(period_um / 2.0, (64, 64))

        filt = GaussianFilter(sigma_um=sigma_um)
        ratio_full = float(filt(full, um_per_px=0.5).std() / full.std())
        ratio_down = float(filt(down, um_per_px=2.0).std() / down.std())

        # Same physical smoothing at both resolutions...
        assert abs(ratio_full - ratio_down) < 0.03
        # ...and matching the analytic attenuation for a physical 4 µm sigma.
        assert abs(ratio_full - expected) < 0.08
        assert abs(ratio_down - expected) < 0.08

    def test_pixel_sigma_is_resolution_locked_demonstrating_old_defect(self):
        # Control for the test above: the legacy pixel-unit path really does
        # smooth by a different physical amount at each resolution (this is
        # the behaviour the µm units exist to fix).
        period_um = 32.0
        full = _sinusoid(period_um / 0.5, (256, 256))
        down = _sinusoid(period_um / 2.0, (64, 64))

        filt = GaussianFilter(sigma_px=8.0)
        ratio_full = float(filt(full).std() / full.std())
        ratio_down = float(filt(down).std() / down.std())
        assert abs(ratio_full - ratio_down) > 0.5

    def test_legacy_sigma_px_matches_scipy_and_ignores_um_per_px(self):
        rng = np.random.default_rng(0)
        img = rng.normal(100.0, 10.0, (40, 50)).astype(np.float32)
        filt = GaussianFilter(sigma_px=2.0)
        expected = scipy_gaussian(img.astype(np.float32), sigma=2.0)
        np.testing.assert_allclose(filt(img), expected)
        # um_per_px is irrelevant for a pixel-constructed filter.
        np.testing.assert_allclose(filt(img, um_per_px=7.3), expected)

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="exactly one"):
            GaussianFilter()
        with pytest.raises(ValueError, match="exactly one"):
            GaussianFilter(sigma_um=1.0, sigma_px=1.0)

    def test_um_sigma_without_um_per_px_raises(self):
        img = np.zeros((8, 8), dtype=np.float32)
        filt = GaussianFilter(sigma_um=2.0)
        with pytest.raises(ValueError, match="um_per_px"):
            filt(img)
        with pytest.raises(ValueError, match="um_per_px"):
            filt(img, um_per_px=float("nan"))


class TestCLAHEFilterMicronUnits:
    def test_kernel_um_matches_equivalent_kernel_px(self):
        rng = np.random.default_rng(1)
        img = rng.normal(100.0, 25.0, (128, 128)).astype(np.float32)
        # 32 µm at 0.5 µm/px == 64 px.
        out_um = CLAHEFilter(clip_limit=0.02, kernel_um=32.0)(img, um_per_px=0.5)
        out_px = CLAHEFilter(clip_limit=0.02, kernel_px=64)(img)
        np.testing.assert_allclose(out_um, out_px)

    def test_kernel_um_tuple_supported(self):
        rng = np.random.default_rng(2)
        img = rng.normal(0.0, 1.0, (96, 128)).astype(np.float32)
        out_um = CLAHEFilter(kernel_um=(16.0, 32.0))(img, um_per_px=0.5)
        out_px = CLAHEFilter(kernel_px=(32, 64))(img)
        np.testing.assert_allclose(out_um, out_px)

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="at most one"):
            CLAHEFilter(kernel_um=32.0, kernel_px=64)

    def test_default_falls_back_to_legacy_64px(self):
        rng = np.random.default_rng(3)
        img = rng.normal(0.0, 1.0, (128, 128)).astype(np.float32)
        np.testing.assert_allclose(
            CLAHEFilter()(img), CLAHEFilter(kernel_px=64)(img)
        )


class TestFFTFilterMicronUnits:
    def test_cutoff_um_matches_equivalent_nyquist_fraction(self):
        rng = np.random.default_rng(4)
        img = rng.normal(100.0, 10.0, (64, 96)).astype(np.float32)
        # cutoff_um=40 at 2 µm/px -> period 20 px -> 0.05 cycles/px.
        out_um = FFTFilter(cutoff_um=40.0)(img, um_per_px=2.0)
        out_px = FFTFilter(cutoff=0.05)(img)
        np.testing.assert_allclose(out_um, out_px, rtol=1e-5, atol=1e-4)

    def test_fov_size_um_matches_equivalent_fov_size_px(self):
        rng = np.random.default_rng(5)
        img = rng.normal(100.0, 10.0, (60, 60)).astype(np.float32)
        out_um = FFTFilter(fov_size_um=12.0)(img, um_per_px=2.0)
        out_px = FFTFilter(cutoff=0.0, fov_size_px=6)(img)
        np.testing.assert_allclose(out_um, out_px, rtol=1e-5, atol=1e-4)

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="at most one"):
            FFTFilter(cutoff_um=100.0, cutoff=0.05)
        with pytest.raises(ValueError, match="at most one"):
            FFTFilter(fov_size_um=450.0, fov_size_px=220)
        with pytest.raises(ValueError, match="at least one"):
            FFTFilter()

    def test_um_params_without_um_per_px_raise(self):
        img = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="um_per_px"):
            FFTFilter(cutoff_um=100.0)(img)
        with pytest.raises(ValueError, match="um_per_px"):
            FFTFilter(fov_size_um=450.0)(img)


class TestFilterChainJSONRoundTrip:
    def test_um_parameters_survive_round_trip(self):
        chain = [
            GaussianFilter(sigma_um=1.5),
            CLAHEFilter(clip_limit=0.02, kernel_um=(20.0, 30.0)),
            FFTFilter(cutoff_um=150.0, fov_size_um=450.0, order=3),
        ]
        restored = _filter_chain_from_json(_filter_chain_to_json(chain))
        assert [type(f).__name__ for f in restored] == [
            "GaussianFilter", "CLAHEFilter", "FFTFilter",
        ]
        g, c, f = restored
        assert g.sigma_um == 1.5 and g.sigma_px is None
        assert c.kernel_um == (20.0, 30.0) and c.kernel_px is None
        assert c.clip_limit == 0.02
        assert f.cutoff_um == 150.0 and f.fov_size_um == 450.0
        assert f.order == 3 and f.cutoff is None and f.fov_size_px is None

    def test_legacy_pixel_parameters_survive_round_trip(self):
        chain = [
            GaussianFilter(sigma_px=2.0),
            CLAHEFilter(kernel_px=64),
            FFTFilter(cutoff=0.05, fov_size_px=220),
        ]
        restored = _filter_chain_from_json(_filter_chain_to_json(chain))
        g, c, f = restored
        assert g.sigma_px == 2.0 and g.sigma_um is None
        assert c.kernel_px == 64
        assert f.cutoff == 0.05 and f.fov_size_px == 220
        assert f.cutoff_um is None and f.fov_size_um is None


class TestCallImageFilterHelper:
    def test_forwards_um_per_px_to_builtin_filters(self):
        img = _sinusoid(64.0, (128, 128))
        filt = GaussianFilter(sigma_um=4.0)
        direct = filt(img, um_per_px=0.5)
        via_helper = _call_image_filter(filt, img, 0.5)
        np.testing.assert_allclose(via_helper, direct)

    def test_plain_callable_without_kwarg_still_works(self):
        img = np.ones((8, 8), dtype=np.float32)
        calls = []

        def doubler(channel):
            calls.append(channel.shape)
            return channel * 2.0

        out = _call_image_filter(doubler, img, 0.5)
        np.testing.assert_allclose(out, img * 2.0)
        assert calls == [(8, 8)]


class TestFFTFilterNotchWidth:
    def test_preserves_signal_on_non_square_image(self):
        # Regression: notch_width was converted to cycles/pixel using a
        # single `nw = notch_width / max(h, w)` shared by both axes --
        # correct only for whichever axis happens to be the larger
        # dimension. On a strongly non-square image this made the notch far
        # too wide (or too narrow) for the other axis, in the worst
        # documented case zeroing 36,738 of 36,740 frequency bins and
        # destroying the image instead of flat-fielding it.
        h, w = 60, 240  # 4x non-square
        rng = np.random.default_rng(0)
        img = (100.0 + rng.normal(0, 5.0, (h, w))).astype(np.float32)

        # fov_size_px=3 -> f_tile=1/3, exactly one harmonic (k=1, f0=1/3)
        # falls below Nyquist -- a single, narrow (notch_width=2 bin) notch
        # per axis, nowhere near enough to gut the whole spectrum on its own
        # regardless of the per-axis-vs-shared-width fix.
        filt = FFTFilter(cutoff=0.0, fov_size_px=3, notch_width=2.0)
        result = filt(img)

        assert np.abs(result.mean() - img.mean()) < 5.0
        assert result.std() > 0.5 * img.std()

    def test_notch_targets_each_axis_own_bin_spacing(self):
        # Direct check of the actual fix: a pure sinusoid at the harmonic
        # frequency along the *short* axis (whose correct notch width in
        # cycles/pixel is larger than the old shared max(h,w)-based value)
        # must still be removed -- the old code under-notched the short
        # axis on a wide image, leaving such a component largely intact.
        h, w = 40, 400  # 10x non-square
        fov = 4  # f_tile = 0.25, one harmonic at f0=0.25 on both axes
        yy, xx = np.mgrid[0:h, 0:w]
        # Sinusoid along the short (row/y) axis only, at exactly the
        # harmonic frequency f0=0.25 cycles/pixel (period 4 rows).
        img = (100.0 + 30.0 * np.sin(2 * np.pi * 0.25 * yy)).astype(np.float32)

        filt = FFTFilter(cutoff=0.0, fov_size_px=fov, notch_width=2.0)
        result = filt(img)

        # The row-axis sinusoid's power must be substantially removed.
        assert result.std() < 0.3 * img.std()

    def test_square_image_unaffected_by_the_fix(self):
        # On a square image h==w, per-axis and shared notch width were
        # always numerically identical -- confirm the fix is a true no-op
        # here (no regression for the previously-working case).
        n = 128
        rng = np.random.default_rng(1)
        img = (100.0 + rng.normal(0, 10.0, (n, n))).astype(np.float32)
        filt = FFTFilter(cutoff=0.0, fov_size_px=3, notch_width=2.0)
        result = filt(img)
        assert np.isfinite(result).all()
        assert result.std() > 0.5 * img.std()
