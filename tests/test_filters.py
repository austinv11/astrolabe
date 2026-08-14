"""Tests for pl._filters.FFTFilter (batch-5 bugfix: per-axis notch width).

Note: the number of harmonics notched scales with `floor(0.5 * fov_size_px)`
(the fundamental frequency 1/fov_size_px gets *more* sub-Nyquist harmonics
for a *larger* fov_size_px) -- these tests deliberately use a small
fov_size_px (a handful of pixels) purely to keep the notch to a single,
precisely-known harmonic for a controlled unit test, not because that's a
realistic tile size.
"""
import numpy as np

from astrolabe.pl._filters import FFTFilter


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
