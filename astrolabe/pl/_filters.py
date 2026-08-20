"""
Image filter classes for per-channel pixel processing.

Filters are stored in ``ImageSpec.filter_chain`` and applied per-channel in
two places: lazily at render time (``pl.image``, before normalization) and
eagerly at analysis time (``pp.extract_if_intensities``, before per-cell
reduction and stain deconvolution).  Each filter is a callable
``(channel_2d: np.ndarray, *, um_per_px: float | None = None) -> np.ndarray``
that takes and returns a 2-D ``float32`` array.

Unit convention
---------------
Filter bandwidths are specified in **physical microns** (``sigma_um``,
``kernel_um``, ``cutoff_um``, ``fov_size_um``).  The two consumers apply the
same ``filter_chain`` at very different pixel resolutions — extraction runs
on pyramid level 0 (full resolution) while the renderer runs on a
downsampled level that additionally changes with zoom — so a pixel-unit
parameter would have a physical bandwidth that differs by up to ~8× between
analysis and display.  Callers therefore pass the effective µm-per-pixel of
the array being filtered via the ``um_per_px`` keyword, and each filter
converts its µm parameters to pixels for that array.

Legacy pixel-unit parameters (``sigma_px``, ``kernel_px``, ``cutoff`` as a
Nyquist fraction, ``fov_size_px``) are retained for resolution-locked use
cases and tests; they are applied verbatim regardless of ``um_per_px``.

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

from dataclasses import dataclass

import numpy as np

__all__ = ["GaussianFilter", "CLAHEFilter", "FFTFilter"]


def _um_to_px(value_um: float, um_per_px: float | None, owner: str, param: str) -> float:
    """Convert a µm-specified bandwidth to pixels for the array being filtered."""
    if um_per_px is None or not np.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError(
            f"{owner} was constructed with {param}={value_um} (microns) but the "
            f"caller did not supply a valid um_per_px (got {um_per_px!r}). "
            f"Either apply the filter through astrolabe (pl.image / "
            f"pp.extract_if_intensities pass um_per_px automatically, provided "
            f"the ImageSpec has a pixel size), or construct the filter with the "
            f"legacy pixel-unit parameter instead."
        )
    return value_um / um_per_px


@dataclass
class GaussianFilter:
    """Gaussian spatial blur applied independently to each image channel.

    Parameters
    ----------
    sigma_um
        Standard deviation of the Gaussian kernel in **microns** (preferred).
        Converted to pixels at apply time using the ``um_per_px`` of the
        array being filtered, so the physical bandwidth is identical at any
        pyramid level / zoom.
    sigma_px
        Legacy standard deviation in **pixels**, applied verbatim regardless
        of resolution.  Mutually exclusive with ``sigma_um``.
    """

    sigma_um: float | None = None
    sigma_px: float | None = None

    def __post_init__(self) -> None:
        if (self.sigma_um is None) == (self.sigma_px is None):
            raise ValueError(
                "GaussianFilter: pass exactly one of sigma_um (microns, preferred) "
                "or sigma_px (pixels, legacy)."
            )

    def __call__(self, channel: np.ndarray, *, um_per_px: float | None = None) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

        if self.sigma_um is not None:
            sigma = _um_to_px(self.sigma_um, um_per_px, "GaussianFilter", "sigma_um")
        else:
            sigma = self.sigma_px
        return gaussian_filter(channel.astype(np.float32), sigma=sigma)


@dataclass
class CLAHEFilter:
    """Contrast-Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast while limiting noise amplification.  The channel
    is normalized to ``[0, 1]`` for processing (as required by
    ``skimage.exposure.equalize_adapthist``) then rescaled back to the
    original intensity range.

    Parameters
    ----------
    clip_limit
        Normalized clip limit in ``[0, 1]`` (skimage convention).  Higher
        values allow stronger contrast enhancement.  Unit-free.
    kernel_um
        Size of the contextual region for histogram equalization in
        **microns** (preferred).  A ``float`` (square tiles) or
        ``(rows_um, cols_um)`` tuple.  Converted to pixels at apply time
        using ``um_per_px``.
    kernel_px
        Legacy contextual-region size in **pixels** — an ``int`` or
        ``(rows, cols)`` tuple, applied verbatim regardless of resolution.
        Mutually exclusive with ``kernel_um``.  When neither is given the
        legacy default of 64 pixels is used.
    """

    clip_limit: float = 0.01
    kernel_um: float | tuple[float, float] | None = None
    kernel_px: int | tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.kernel_um, list):
            self.kernel_um = tuple(self.kernel_um)
        if isinstance(self.kernel_px, list):
            self.kernel_px = tuple(self.kernel_px)
        if self.kernel_um is not None and self.kernel_px is not None:
            raise ValueError(
                "CLAHEFilter: pass at most one of kernel_um (microns, preferred) "
                "or kernel_px (pixels, legacy)."
            )

    def _resolve_kernel(self, um_per_px: float | None) -> int | tuple[int, int]:
        if self.kernel_um is not None:
            if isinstance(self.kernel_um, tuple):
                return tuple(
                    max(1, round(_um_to_px(k, um_per_px, "CLAHEFilter", "kernel_um")))
                    for k in self.kernel_um
                )
            return max(1, round(_um_to_px(self.kernel_um, um_per_px, "CLAHEFilter", "kernel_um")))
        if self.kernel_px is not None:
            return self.kernel_px
        return 64  # legacy pixel-unit default

    def __call__(self, channel: np.ndarray, *, um_per_px: float | None = None) -> np.ndarray:
        from skimage.exposure import equalize_adapthist

        kernel_size = self._resolve_kernel(um_per_px)
        arr = channel.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        result = equalize_adapthist(
            arr, kernel_size=kernel_size, clip_limit=self.clip_limit
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
    at integer harmonics of the tile repeat frequency (``1 / fov_size``) in
    both the x and y axes.  This removes the periodic intensity pattern that
    appears at tile boundaries after stitching.

    Parameters
    ----------
    cutoff_um
        High-pass cutoff as a spatial **period in microns** (preferred):
        intensity variation on length scales longer than ``cutoff_um`` is
        attenuated according to the Butterworth roll-off.  Converted to a
        cycles-per-pixel frequency at apply time using ``um_per_px``.
    order
        Butterworth filter order.  Higher values give a steeper roll-off.
    fov_size_um
        Tile size in **microns** (preferred) used to compute the stitching
        harmonic frequencies.  Converted to pixels at apply time.
    notch_width
        Half-width of each frequency notch in FFT index units (bins in the
        frequency domain).  Default ``2.0`` notches ±2 bins around each
        harmonic.  This tolerance is deliberately kept in FFT bins (not µm):
        it absorbs spectral leakage, whose width is set by the array itself.
    cutoff
        Legacy high-pass cutoff as a fraction of the Nyquist frequency
        (``0`` – ``0.5`` cycles/pixel), applied verbatim regardless of
        resolution.  Mutually exclusive with ``cutoff_um``.  ``0`` or
        ``None`` disables the high-pass.
    fov_size_px
        Legacy tile size in **pixels**, applied verbatim regardless of
        resolution.  Mutually exclusive with ``fov_size_um``.  ``None``
        disables the notch filter (together with ``fov_size_um=None``).

    Examples
    --------
    Flat-field only (attenuate variation slower than ~200 µm):

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff_um=200))

    Stitching artifact removal only (e.g. 450-µm FOV tiles):

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(fov_size_um=450))

    Both:

    >>> ab.pp.add_image_filter(mdata, ab.pl.FFTFilter(cutoff_um=200, fov_size_um=450))
    """

    cutoff_um: float | None = None
    order: int = 2
    fov_size_um: float | None = None
    notch_width: float = 2.0
    cutoff: float | None = None
    fov_size_px: int | None = None

    def __post_init__(self) -> None:
        if self.cutoff_um is not None and self.cutoff is not None:
            raise ValueError(
                "FFTFilter: pass at most one of cutoff_um (microns, preferred) "
                "or cutoff (Nyquist fraction, legacy)."
            )
        if self.fov_size_um is not None and self.fov_size_px is not None:
            raise ValueError(
                "FFTFilter: pass at most one of fov_size_um (microns, preferred) "
                "or fov_size_px (pixels, legacy)."
            )
        if (
            self.cutoff_um is None
            and self.cutoff is None
            and self.fov_size_um is None
            and self.fov_size_px is None
        ):
            raise ValueError(
                "FFTFilter: configure at least one operation — a high-pass "
                "(cutoff_um / cutoff) and/or a notch filter (fov_size_um / fov_size_px)."
            )

    def __call__(self, channel: np.ndarray, *, um_per_px: float | None = None) -> np.ndarray:
        # Resolve µm parameters to the pixel grid of *this* array.
        if self.cutoff_um is not None:
            # spatial period (µm) → period in pixels → frequency (cycles/pixel)
            period_px = _um_to_px(self.cutoff_um, um_per_px, "FFTFilter", "cutoff_um")
            cutoff = 1.0 / period_px
        else:
            cutoff = self.cutoff or 0.0

        if self.fov_size_um is not None:
            fov_size_px = _um_to_px(self.fov_size_um, um_per_px, "FFTFilter", "fov_size_um")
        else:
            fov_size_px = self.fov_size_px

        h, w = channel.shape
        arr = channel.astype(np.float32)
        F = np.fft.rfft2(arr)

        fy = np.fft.fftfreq(h)[:, None]   # (H, 1) — cycles/pixel
        fx = np.fft.rfftfreq(w)[None, :]  # (1, W//2+1) — cycles/pixel

        mask = np.ones((h, w // 2 + 1), dtype=np.float32)

        # Butterworth high-pass
        if cutoff > 0:
            freq = np.sqrt(fx ** 2 + fy ** 2)
            mask *= 1.0 / np.sqrt(
                1.0 + (cutoff / (freq + 1e-10)) ** (2 * self.order)
            )

        # Notch filter at tile harmonics
        if fov_size_px is not None and fov_size_px > 0:
            f_tile = 1.0 / fov_size_px
            # Convert notch_width from FFT bins to cycles/pixel -- separately
            # per axis, since fy's bin spacing is 1/h and fx's is 1/w. Using
            # a single width against max(h,w) for both (the previous
            # behavior) is only correct for whichever axis happens to be the
            # larger dimension; on a non-square image the other axis's
            # notch width is wrong by a factor of max(h,w)/min(h,w), in
            # either direction depending on which axis is smaller --
            # observed to destroy nearly the entire frequency domain
            # (36,738 of 36,740 bins zeroed) at documented/demonstrated
            # parameters rather than flat-fielding the image.
            nw_y = self.notch_width / h
            nw_x = self.notch_width / w
            k = 1
            while (f0 := k * f_tile) <= 0.5:
                # Horizontal harmonics (rows): target rows where |fy| ≈ f0
                mask[np.abs(np.abs(fy[:, 0]) - f0) < nw_y, :] = 0.0
                # Vertical harmonics (cols): target cols where |fx| ≈ f0
                mask[:, np.abs(np.abs(fx[0, :]) - f0) < nw_x] = 0.0
                k += 1

        result = np.fft.irfft2(F * mask, s=(h, w))
        return result.astype(np.float32)
