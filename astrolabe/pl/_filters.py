"""
Image filter classes for per-channel pixel processing.

Filters are stored in ``ImageSpec.filter_chain`` and applied lazily at render
time after pixel data is loaded from disk and before normalisation.  Each
filter is a callable ``(channel_2d: np.ndarray) -> np.ndarray`` that takes
and returns a 2-D ``float32`` array.

Classes
-------
GaussianFilter
    Gaussian spatial blur.
CLAHEFilter
    Contrast-Limited Adaptive Histogram Equalization.
FFTFilter
    Frequency-domain high-pass (flat-field correction) with optional
    notch filter to remove periodic stitching tile artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GaussianFilter", "CLAHEFilter", "FFTFilter"]


@dataclass
class GaussianFilter:
    """Gaussian spatial blur applied independently to each image channel.

    Parameters
    ----------
    sigma
        Standard deviation of the Gaussian kernel in pixels.
    """

    sigma: float = 1.0

    def __call__(self, channel: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(channel.astype(np.float32), sigma=self.sigma)


@dataclass
class CLAHEFilter:
    """Contrast-Limited Adaptive Histogram Equalisation (CLAHE).

    Enhances local contrast while limiting noise amplification.  The channel
    is normalised to ``[0, 1]`` for processing (as required by
    ``skimage.exposure.equalize_adapthist``) then rescaled back to the
    original intensity range.

    Parameters
    ----------
    clip_limit
        Normalised clip limit in ``[0, 1]`` (skimage convention).  Higher
        values allow stronger contrast enhancement.
    kernel_size
        Size of the contextual region for histogram equalisation.  Can be
        an ``int`` (square tiles) or ``(rows, cols)`` tuple.
    """

    clip_limit: float = 0.01
    kernel_size: int | tuple[int, int] = 64

    def __call__(self, channel: np.ndarray) -> np.ndarray:
        from skimage.exposure import equalize_adapthist

        arr = channel.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        result = equalize_adapthist(
            arr, kernel_size=self.kernel_size, clip_limit=self.clip_limit
        )
        return (result.astype(np.float32) * (hi - lo) + lo)


@dataclass
class FFTFilter:
    """Frequency-domain filter for flat-field correction and stitching artifact removal.

    Applies two independent operations (either or both may be active):

    **High-pass (flat-field correction):** Butterworth high-pass filter
    attenuates low spatial frequencies that represent slowly-varying
    illumination non-uniformity (e.g. uneven illumination across an FOV).

    **Notch filter (stitching artifact removal):** Zeros frequency components
    at integer harmonics of the tile repeat frequency ``1 / fov_size_px`` in
    both the x and y axes.  This removes the periodic intensity pattern that
    appears at tile boundaries after stitching.

    Parameters
    ----------
    cutoff
        High-pass cutoff frequency as a fraction of the Nyquist frequency
        (``0`` – ``0.5``).  Frequencies below this are attenuated according
        to the Butterworth roll-off.  Set to ``0`` to disable high-pass.
    order
        Butterworth filter order.  Higher values give a steeper roll-off.
    fov_size_px
        Tile size in pixels used to compute the stitching harmonic
        frequencies.  If ``None``, the notch filter is disabled.
    notch_width
        Half-width of each frequency notch in FFT index units (pixels in
        the frequency domain).  Default ``2.0`` notches ±2 bins around each
        harmonic.

    Examples
    --------
    Flat-field only:

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff=0.05))

    Stitching artifact removal only (e.g. 220-pixel FOV tiles):

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff=0, fov_size_px=220))

    Both:

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff=0.05, fov_size_px=220))
    """

    cutoff: float = 0.05
    order: int = 2
    fov_size_px: int | None = None
    notch_width: float = 2.0

    def __call__(self, channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        arr = channel.astype(np.float32)
        F = np.fft.rfft2(arr)

        fy = np.fft.fftfreq(h)[:, None]   # (H, 1) — cycles/pixel
        fx = np.fft.rfftfreq(w)[None, :]  # (1, W//2+1) — cycles/pixel

        mask = np.ones((h, w // 2 + 1), dtype=np.float32)

        # Butterworth high-pass
        if self.cutoff > 0:
            freq = np.sqrt(fx ** 2 + fy ** 2)
            mask *= 1.0 / np.sqrt(
                1.0 + (self.cutoff / (freq + 1e-10)) ** (2 * self.order)
            )

        # Notch filter at tile harmonics
        if self.fov_size_px is not None:
            f_tile = 1.0 / self.fov_size_px
            # Convert notch_width from FFT bins to cycles/pixel
            nw = self.notch_width / max(h, w)
            k = 1
            while (f0 := k * f_tile) <= 0.5:
                # Horizontal harmonics (rows): target rows where |fy| ≈ f0
                mask[np.abs(np.abs(fy[:, 0]) - f0) < nw, :] = 0.0
                # Vertical harmonics (cols): target cols where |fx| ≈ f0
                mask[:, np.abs(np.abs(fx[0, :]) - f0) < nw] = 0.0
                k += 1

        result = np.fft.irfft2(F * mask, s=(h, w))
        return result.astype(np.float32)
