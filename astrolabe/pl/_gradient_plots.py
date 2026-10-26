"""
Summary plots for gradient/vector-field analysis results.
"""

from __future__ import annotations

from astrolabe._dependencies import _get_result

from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd

from astrolabe._utils import _resolve_adata
from astrolabe.pl._spatial import (
    _DEFAULT_DIVERGING_CMAP,
    FigurePanel,
    PlotGrid,
    hstack,
    points,
)
from astrolabe.pl._theme import _style_axes, _style_colorbar, _style_legend, get_theme

__all__ = [
    "flux_plot",
    "direction_histogram",
    "diffusion_comparison",
    "decay_length_plot",
]


def flux_plot(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    statistic: str = "integrated_flux",
    triangle: str = "lower",
    min_interface_um: float = 0.0,
    uns_key: str = "interface_flux",
    cmap: str = _DEFAULT_DIVERGING_CMAP,
    figsize: tuple[float, float] = (6, 5),
    annotate: bool = True,
    fmt: str = ".2f",
) -> FigurePanel:
    """Heatmap of expression flux across compartment interfaces.

    Reads the results of :func:`astrolabe.tl.interface_flux` and renders a
    diverging-colormap heatmap where cell ``(i, j)`` shows how expression
    changes across the *i*|*j* interface, oriented from *i* toward *j*.

    Positive values (warm) mean expression **rises** crossing from
    compartment *i* into compartment *j*; negative values (cool) mean it
    falls. The matrix is exactly antisymmetric by construction
    (``Φ(B→A) = −Φ(A→B)``) — only one triangle carries information, so by
    default only the lower triangle is drawn (``triangle='lower'``); reading
    "A→B is positive AND B→A is negative" as two findings double-counts one
    number.

    The returned :class:`~astrolabe.pl.FigurePanel` composes with spatial
    panels via ``|`` and ``/``::

        (ab.pl.points(mdata, 'leiden') | ab.pl.flux_plot(mdata, 'EPCAM')).show()

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key whose flux to display.  Must be one of the keys passed to
        :func:`astrolabe.tl.interface_flux`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    statistic
        ``'integrated_flux'`` (default) — normal derivative integrated over
        the interface length (units: feature scale); large interfaces weigh
        more. ``'mean_normal_derivative'`` — length-weighted mean derivative
        (units: per µm); interface-size independent.
    triangle
        ``'lower'`` (default) — draw and annotate only the informative lower
        triangle; the redundant upper half and diagonal are greyed out with a
        note. ``'full'`` — draw both triangles.
    min_interface_um
        Mask (grey out) compartment pairs whose shared interface is shorter
        than this many µm. A three-cell contact's flux is noise, but it
        renders exactly as loudly as a millimetre-long interface without this
        gate. ``0`` (default) shows all pairs.
    uns_key
        ``uns`` key written by :func:`astrolabe.tl.interface_flux` — pass the
        same value as that call's ``key_added``.
    cmap
        Diverging colormap; defaults to the active theme's diverging colormap.
    figsize
        Figure size in inches.
    annotate
        If ``True``, write numeric values in each cell.
    fmt
        ``str.format``-compatible format string for cell annotations.

    Returns
    -------
    FigurePanel
    """
    adata = _resolve_adata(data, modality)
    if uns_key not in adata.uns:
        raise KeyError(
            f"uns['{uns_key}'] not found. Run ab.tl.interface_flux() first."
        )
    flux_data = _get_result(adata, uns_key)
    if statistic not in ("integrated_flux", "mean_normal_derivative"):
        raise ValueError(
            "statistic must be 'integrated_flux' or 'mean_normal_derivative', "
            f"got {statistic!r}"
        )
    if statistic not in flux_data or key not in flux_data[statistic]:
        available = list(flux_data.get(statistic, {}).keys())
        raise KeyError(
            f"Key '{key}' not found in uns['{uns_key}']['{statistic}']. "
            f"Available: {available}"
        )

    if triangle not in ("lower", "full"):
        raise ValueError(f"triangle must be 'lower' or 'full', got {triangle!r}")

    df: pd.DataFrame = flux_data[statistic][key]
    groups = df.index.tolist()
    mat = df.values.astype(float)

    if min_interface_um > 0 and "interface_length" in flux_data:
        lengths = (
            flux_data["interface_length"]
            .reindex(index=df.index, columns=df.columns)
            .values.astype(float)
        )
        with np.errstate(invalid="ignore"):
            mat = np.where(
                np.isfinite(lengths) & (lengths >= min_interface_um), mat, np.nan
            )

    # masked entries (no interface, below the length gate, or the redundant
    # triangle) render in the theme's neutral instead of a colormap pole
    shown = np.isfinite(mat)
    if triangle == "lower":
        shown &= np.tri(len(groups), k=-1, dtype=bool)

    abs_max = float(np.nanmax(np.abs(mat[shown]))) if shown.any() else 1.0
    if abs_max == 0:
        abs_max = 1.0

    unit = "feature units" if statistic == "integrated_flux" else "per µm"
    label = (
        f"integrated flux ({unit})"
        if statistic == "integrated_flux"
        else f"mean normal derivative ({unit})"
    )

    t = get_theme()
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(t.div_neutral)
    im = ax.imshow(
        np.ma.masked_where(~shown, mat),
        cmap=cmap_obj,
        vmin=-abs_max,
        vmax=abs_max,
        aspect="auto",
    )
    _style_colorbar(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label))
    if triangle == "lower":
        ax.text(
            0.98, 0.98, "upper half = -(lower) by identity",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=t.muted,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_xlabel("into compartment")
    ax.set_ylabel("out of compartment")
    ax.set_title(f"{key} interface flux")

    if annotate:
        for i in range(len(groups)):
            for j in range(len(groups)):
                val = mat[i, j]
                if shown[i, j] and np.isfinite(val):
                    # Contrast against the cell's own fill: strong (saturated
                    # pole) values get the surface color, weak (near-neutral-
                    # midpoint) values get the primary ink -- these are each
                    # theme's own opposite extremes, so this flips correctly
                    # for dark mode rather than hardcoding white/black.
                    text_color = t.surface if abs(val) > 0.6 * abs_max else t.primary_ink
                    ax.text(
                        j, i,
                        format(val, fmt),
                        ha="center", va="center",
                        color=text_color, fontsize=8,
                    )

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


def direction_histogram(
    data: ad.AnnData | md.MuData,
    keys: str | Sequence[str],
    *,
    modality: str | None = None,
    weights: str = "magnitude",
    exclude_unreliable: bool = True,
    relative_to: str | tuple[float, float] | np.ndarray | None = "cell_axis",
    spatial_key: str = "spatial",
    n_bins: int = 36,
    magnitude_eps: float = 1e-8,
    gradient_key_added: str | None = None,
    figsize: tuple[float, float] = (4, 4),
    alpha: float = 0.7,
) -> FigurePanel:
    """Polar rose diagram of expression gradient directions.

    Reads ``obs[f'gradient_{key}_direction']`` written by
    :func:`astrolabe.tl.expression_gradient` (radians, arctan2 convention)
    and renders a polar bar chart.  One bar series per key is overlaid on
    the same polar axes.

    **Absolute bearings are usually meaningless**: the tissue section could
    have been mounted at any rotation, so a rose over raw image-frame angles
    only answers "is there one tissue-wide direction?" (rarely). The
    convention in the tissue-polarity literature (e.g. planar cell polarity
    quantification; Aigouy et al. 2010) is to measure directions **relative
    to a landmark or tissue axis**. Three reference frames are available via
    ``relative_to``:

    - ``'cell_axis'`` (default) — each gradient is measured in its own
      cell's **nucleus-anchored axis frame**
      (``obs['cell_polarity_major_angle'/'cell_polarity_minor_angle']``
      from :func:`~astrolabe.pp.calculate_shape_metrics`): the bbox axes
      directed so each positive end is the half of the cell farther from
      the nucleus. The rose is a full circle where every quadrant means
      something — 0° (right) = the gradient climbs along the major axis
      *toward* the nucleus (the proximal pole), 180° = away toward the
      distal pole, +90°/−90° = along the minor axis toward/away from the
      nucleus. Mounting-rotation invariant, and it asks the planar-cell-
      polarity question with nucleus-displacement polarity built in.
    - a landmark (``obs`` column name, ``(x, y)`` point, or array) — each
      gradient is measured against a per-cell landmark bearing (e.g. from
      :func:`~astrolabe.tl.bearing_to_group`): a spike at "toward" (top)
      reads "gradients climb toward the landmark", at "away" (bottom)
      "away from it".
    - ``None`` — raw image-frame bearings (0° = +x, counter-clockwise),
      labeled in degrees. Rarely the right frame; kept for completeness.

    Each cell's bar contribution is weighted by its gradient magnitude by
    default (``weights='magnitude'``), so the rose shows *where the steepness
    points*, not merely how many (possibly near-flat) cells point each way.
    The title/legend reports the matching mean resultant length: the
    magnitude-weighted R̄ (``|Σ m·e^{iθ}| / Σ m``) or the unweighted R̄ for
    ``weights='count'``.

    Cells with a below-``magnitude_eps`` gradient or (by default) flagged
    ``obs['gradient_unreliable']`` are excluded — the same exclusions
    :func:`astrolabe.tl.gradient_significance` applies.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more feature keys.  Each must have corresponding
        ``obs['gradient_{key}_direction']`` / ``'_magnitude'`` columns.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    weights
        ``'magnitude'`` (default) — each cell contributes its gradient
        magnitude, so bars show the fraction of total steepness per bearing.
        ``'count'`` — each cell contributes 1.
    exclude_unreliable
        Exclude cells flagged ``obs['gradient_unreliable']`` (tissue
        boundary; the default across all gradient consumers).
    relative_to
        Reference frame for the rose:

        - ``'cell_axis'`` (default) — per-cell nucleus-anchored axis frame
          from ``obs['cell_polarity_major_angle'/'cell_polarity_minor_angle']``;
          requires :func:`~astrolabe.pp.calculate_shape_metrics` to have
          been run on data with nucleus boundaries. Cells without a usable
          nucleus (NaN polarity) are excluded; a near-centered nucleus
          makes a cell's frame sign noise-driven (filter on
          ``obs['cell_nucleus_offset_um']`` if that matters).
        - any other ``str`` — name of an ``obs`` column of reference
          bearings in radians (e.g. written by
          :func:`~astrolabe.tl.bearing_to_group`);
        - ``(x, y)`` tuple — a fixed µm point; each cell's reference is the
          bearing from the cell toward that point;
        - array of shape ``(n_cells,)`` — explicit per-cell bearings;
        - ``None`` — raw image-frame bearings with degree labels.

        With a landmark reference, "toward" renders at the top of the rose
        and "away" at the bottom; in the ``'cell_axis'`` frame the "toward"
        poles are at the right (major) and top (minor), with their "away"
        counterparts opposite. Cells with a NaN reference are excluded.
    spatial_key
        ``obsm`` key for µm coordinates (only used for the tuple form of
        ``relative_to``).
    n_bins
        Number of angular bins (default 36 = 10° per bin).
    magnitude_eps
        Gradient-magnitude threshold below which cells are excluded.
    gradient_key_added
        ``key_added`` used in the :func:`astrolabe.tl.expression_gradient`
        call this reads from, if it was given a custom one.
    figsize
        Figure size in inches.
    alpha
        Bar opacity.

    Returns
    -------
    FigurePanel
    """
    adata = _resolve_adata(data, modality)
    if isinstance(keys, str):
        keys = [keys]
    if weights not in ("magnitude", "count"):
        raise ValueError(f"weights must be 'magnitude' or 'count', got {weights!r}")

    # resolve the per-cell reference frame/bearing, if any
    cell_axis = isinstance(relative_to, str) and relative_to == "cell_axis"
    reference = None
    maj_ref = min_ref = None
    if cell_axis:
        needed = ("cell_polarity_major_angle", "cell_polarity_minor_angle")
        if any(c not in adata.obs for c in needed):
            raise KeyError(
                "obs['cell_polarity_major_angle'/'cell_polarity_minor_angle'] "
                "not found — the default relative_to='cell_axis' measures "
                "each gradient in its own cell's nucleus-anchored axis frame. "
                "Run ab.pp.calculate_shape_metrics() first (needs "
                "obsm['nucleus_boundaries'] to direct the axes); on data "
                "without nucleus boundaries pass relative_to= a landmark "
                "bearing column / point (see ab.tl.bearing_to_group) or "
                "relative_to=None for raw image-frame bearings."
            )
        maj_ref = np.asarray(adata.obs["cell_polarity_major_angle"], dtype=float)
        min_ref = np.asarray(adata.obs["cell_polarity_minor_angle"], dtype=float)
    elif relative_to is not None:
        if isinstance(relative_to, str):
            if relative_to not in adata.obs:
                raise KeyError(
                    f"obs['{relative_to}'] not found — run "
                    f"ab.tl.bearing_to_group() first or pass an array/point."
                )
            reference = np.asarray(adata.obs[relative_to], dtype=float)
        elif isinstance(relative_to, tuple) and len(relative_to) == 2:
            coords = np.asarray(adata.obsm[spatial_key][:, :2], dtype=float)
            reference = np.arctan2(
                relative_to[1] - coords[:, 1], relative_to[0] - coords[:, 0]
            )
        else:
            reference = np.asarray(relative_to, dtype=float)
            if reference.shape != (adata.n_obs,):
                raise ValueError(
                    f"relative_to array must have shape ({adata.n_obs},), "
                    f"got {reference.shape}"
                )

    unreliable = None
    if exclude_unreliable and "gradient_unreliable" in adata.obs:
        unreliable = adata.obs["gradient_unreliable"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [p["color"] for p in prop_cycle]

    rbar_label = "weighted R̄" if weights == "magnitude" else "R̄"
    rbar_single = None
    for idx, key in enumerate(keys):
        base = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
        dir_col = f"{base}_direction"
        mag_col = f"{base}_magnitude"
        if dir_col not in adata.obs:
            raise KeyError(
                f"obs['{dir_col}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
            )
        directions = np.asarray(adata.obs[dir_col], dtype=float)
        valid = np.isfinite(directions)
        if mag_col in adata.obs:
            magnitudes = np.asarray(adata.obs[mag_col], dtype=float)
            valid &= magnitudes > magnitude_eps
        else:
            magnitudes = np.ones_like(directions)
        if unreliable is not None:
            valid &= ~unreliable
        if cell_axis:
            # Components along the two nucleus-directed axes give the display
            # angle. The axes stored by calculate_shape_metrics point their
            # positive end AWAY from the nucleus, but the rose is drawn in the
            # opposite (nucleus-proximal) frame so that "toward" matches the
            # landmark rose's convention of sitting at the primary poles:
            # 0 = climbing along the major axis toward the nucleus (proximal
            # pole), ±180° = away toward the distal pole, +90/−90 = toward/away
            # along the minor axis. Hence the negated cosine components; atan2
            # of the two handles per-cell frames of either handedness.
            valid &= np.isfinite(maj_ref) & np.isfinite(min_ref)
            safe_maj = np.where(np.isfinite(maj_ref), maj_ref, 0.0)
            safe_min = np.where(np.isfinite(min_ref), min_ref, 0.0)
            directions = np.arctan2(
                -np.cos(directions - safe_min), -np.cos(directions - safe_maj)
            )
        elif reference is not None:
            valid &= np.isfinite(reference)
            directions = directions - np.where(np.isfinite(reference), reference, 0.0)
            # wrap back to [-pi, pi]
            directions = np.arctan2(np.sin(directions), np.cos(directions))
        directions = directions[valid]
        w = magnitudes[valid] if weights == "magnitude" else np.ones(valid.sum())

        rbar = np.nan
        if len(directions) and w.sum() > 0:
            rbar = float(
                np.abs(np.sum(w * np.exp(1j * directions))) / w.sum()
            )
        if len(keys) == 1:
            rbar_single = rbar

        counts, _ = np.histogram(directions, bins=bin_edges, weights=w)
        total = counts.sum()
        fracs = counts / total if total > 0 else counts.astype(float)

        color = colors[idx % len(colors)]
        label = key if np.isnan(rbar) else f"{key} ({rbar_label}={rbar:.3f})"
        ax.bar(
            bin_centers,
            fracs,
            width=bin_width * 0.9,
            bottom=0,
            alpha=alpha,
            color=color,
            label=label,
            align="center",
        )

    ax.set_theta_direction(1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    if cell_axis:
        # Nucleus-anchored frame: major axis horizontal with the
        # toward-the-nucleus (proximal) pole on the right, minor axis vertical
        # with its toward pole on top. "away"/"toward" are relative to the
        # nucleus.
        ax.set_theta_zero_location("E")
        ax.set_xticklabels([
            "toward\n(major)", "45°", "toward\n(minor)", "135°",
            "away\n(major)", "-135°", "away\n(minor)", "-45°",
        ])
    elif reference is not None:
        # Landmark frame: "toward" at the top, "away" at the bottom.
        ax.set_theta_zero_location("N")
        ax.set_xticklabels(
            ["toward", "45°", "90°", "135°", "away", "-135°", "-90°", "-45°"]
        )
    else:
        # Raw image frame: 0° = +x, counter-clockwise (the arctan2 frame).
        ax.set_theta_zero_location("E")
        ax.set_xticklabels(
            ["0°", "45°", "90°", "135°", "180°", "-135°", "-90°", "-45°"]
        )
    ax.set_ylabel("")

    if cell_axis:
        suffix = " (nucleus-anchored cell frame)"
    elif reference is not None:
        suffix = " (relative to landmark)"
    else:
        suffix = " (raw image frame)"
    if len(keys) == 1:
        title = f"{keys[0]} gradient direction{suffix}"
        if rbar_single is not None and not np.isnan(rbar_single):
            title += f"  ({rbar_label} = {rbar_single:.3f})"
        ax.set_title(title, pad=12)
    else:
        _style_legend(ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.12), fontsize=8))
        ax.set_title(f"gradient directions{suffix}", pad=12)

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


def diffusion_comparison(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    steps: Sequence[int] = (1, 3, 5),
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    normalize: str = "coifman_lafon",
    spatial_key: str = "spatial",
    size: float | None = None,
    alpha: float = 0.7,
    cmap: str | None = None,
    figsize: tuple[float, float] = (4, 4),
) -> PlotGrid:
    """Raw vs. graph-diffused expression, side by side on a shared color scale.

    The practical rule for choosing :func:`~astrolabe.tl.diffuse_expression`'s
    ``n_steps`` (MAGIC's own; van Dijk et al. 2018): plot raw against smoothed
    and stop increasing ``n_steps`` as soon as the structure you care about
    starts to fade — each step spreads information roughly one neighbor ring,
    so ``n_steps`` is approximately the smoothing radius in cell diameters.
    This function is that comparison: one spatial panel for the raw values
    plus one per entry of *steps*, all sharing **one color scale** (the
    1st–99.5th percentile of the raw values). Per-panel autoscaling would
    hide exactly the fading you are looking for, because diffusion compresses
    the value range — which is why composing plain
    :func:`~astrolabe.pl.points` panels with ``|`` is not an honest
    comparison.

    Nothing is written to the object — smoothing happens on a throwaway
    copy. Once a step count is chosen, persist it with
    :func:`~astrolabe.tl.diffuse_expression`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Gene name or ``obs`` column to smooth and display.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    steps
        Diffusion step counts to show alongside the raw panel
        (deduplicated and sorted; each must be ≥ 1).
    layer
        Layer in ``adata.layers`` to read *key* from instead of ``X``.
    neighbors_key
        Spatial graph to diffuse on (from
        :func:`~astrolabe.tl.spatial_neighbors`).
    normalize
        ``'coifman_lafon'`` (default, density-corrected) or ``'row'`` —
        matches :func:`~astrolabe.tl.diffuse_expression`.
    spatial_key
        ``obsm`` key with ``(n_cells, 2)`` µm coordinates.
    size, alpha, cmap, figsize
        Forwarded to each :func:`~astrolabe.pl.points` panel.

    Returns
    -------
    PlotGrid
        Lazy multi-panel figure; call ``.show()`` or compose further.

    Examples
    --------
    >>> ab.pl.diffusion_comparison(mdata, 'EPCAM', steps=(1, 3, 5)).show()
    >>> ab.tl.diffuse_expression(mdata, 'EPCAM', n_steps=1)   # chosen step
    """
    from astrolabe.tl._gradient import _diffusion_operator, _get_neighbors_keys
    from astrolabe.tl._spatial_stats import _extract_values

    adata = _resolve_adata(data, modality)
    step_list = sorted({int(s) for s in steps})
    if not step_list or step_list[0] < 1:
        raise ValueError(
            f"steps must be a non-empty collection of integers >= 1, got {steps!r} "
            "(the raw panel is always included)."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found.")

    f = np.asarray(_extract_values(adata, key, layer), dtype=np.float64)
    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    M, _ = _diffusion_operator(adata.obsp[conn_key], normalize)

    # One shared color scale, anchored on the raw values: diffusion shrinks
    # the value range, and per-panel autoscaling would mask the fading.
    finite = f[np.isfinite(f)]
    vmin = vmax = None
    if finite.size:
        vmin, vmax = (float(v) for v in np.percentile(finite, [1.0, 99.5]))
        if vmin >= vmax:  # near-constant field: let matplotlib autoscale
            vmin = vmax = None

    coords = np.asarray(adata.obsm[spatial_key][:, :2], dtype=np.float32)

    def _panel(values: np.ndarray, title: str):
        tmp = ad.AnnData(X=np.zeros((adata.n_obs, 0), dtype=np.float32))
        tmp.obs[key] = np.asarray(values, dtype=np.float32)
        tmp.obsm[spatial_key] = coords
        return points(
            tmp,
            color=key,
            spatial_key=spatial_key,
            size=size,
            alpha=alpha,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            figsize=figsize,
        )

    panels = [_panel(f, f"{key} (raw)")]
    cur = f.copy()
    done = 0
    for s in step_list:
        for _ in range(s - done):
            cur = M @ cur
        done = s
        panels.append(_panel(cur, f"{key} (n_steps={s})"))
    return hstack(*panels)


def decay_length_plot(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    uns_key: str | None = None,
    show_background: bool = True,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
) -> FigurePanel:
    """Radial decay profile with the fitted exponential from :func:`astrolabe.tl.decay_length`.

    Draws the binned mean expression against distance from the source, with
    marker area proportional to the square root of each bin's cell count (the
    same weights the fit itself used, so a bin the fit barely trusted looks
    small), the fitted ``f0 * exp(-r/lambda) + c`` curve, and the fitted
    background floor ``c`` as a dashed line.

    The title reports lambda with its standard error; when the fit is poorly
    identified (SE > 50% of lambda, lambda beyond the fitting window, or a
    failed fit) an in-axes caution note is shown — mirroring the warnings
    :func:`astrolabe.tl.decay_length` itself logs. A profile whose fitted
    ``f0`` is comparable to ``c`` is mostly floor: the honest conclusion is
    "not a radial morphogen-style gradient from this source", not a decay
    length.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key passed to :func:`astrolabe.tl.decay_length`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    uns_key
        ``uns`` key holding the fit results. ``None`` (default) reads
        ``f'decay_length_{key}'`` — pass the same value as the tl call's
        ``key_added`` if you overrode it.
    show_background
        Draw the fitted background floor as a dashed reference line.
    figsize
        Figure size in inches.
    title
        Axes title. Auto-generated with the fit summary when ``None``.

    Returns
    -------
    FigurePanel
    """
    adata = _resolve_adata(data, modality)
    resolved_key = uns_key or f"decay_length_{key}"
    if resolved_key not in adata.uns:
        raise KeyError(
            f"uns['{resolved_key}'] not found. Run ab.tl.decay_length() first."
        )
    res = _get_result(adata, resolved_key)

    r = np.asarray(res["bin_centers"], dtype=float)
    y = np.asarray(res["bin_means"], dtype=float)
    counts = np.asarray(res["bin_counts"], dtype=float)
    lam = float(res.get("lambda_um", np.nan))
    lam_se = float(res.get("lambda_se_um", np.nan))
    f0 = float(res.get("f0", np.nan))
    background = float(res.get("background", np.nan))
    r2 = float(res.get("r_squared", np.nan))
    max_dist = float(res.get("params", {}).get("max_dist_um", np.nanmax(r)))

    t = get_theme()
    fig, ax = plt.subplots(figsize=figsize)

    # marker area ~ sqrt(count): the fit weighted bins by 1/sqrt(count), so
    # visually large points are the ones the fit actually trusted
    finite_bins = np.isfinite(y)
    max_count = np.nanmax(counts[finite_bins]) if finite_bins.any() else 1.0
    sizes = 8.0 + 60.0 * np.sqrt(np.clip(counts, 0, None) / max(max_count, 1.0))
    ax.scatter(
        r[finite_bins], y[finite_bins], s=sizes[finite_bins],
        color=t.cat_core[0], alpha=0.8, edgecolors="none", zorder=3,
        label="binned mean (size ~ cells/bin)",
    )

    fit_ok = np.isfinite(lam) and np.isfinite(f0) and np.isfinite(background)
    if fit_ok:
        r_dense = np.linspace(0, float(np.nanmax(r[finite_bins])) if finite_bins.any() else max_dist, 200)
        ax.plot(
            r_dense, f0 * np.exp(-r_dense / lam) + background,
            color=t.cat_core[1], linewidth=1.8, zorder=2,
            label="fit: f0 exp(-r/lambda) + c",
        )
        if show_background:
            ax.axhline(
                background, linestyle=":", color=t.muted, linewidth=1.2,
                zorder=1, label=f"background c = {background:.2f}",
            )

    cautions = []
    if not fit_ok:
        cautions.append("fit failed — no decay length")
    else:
        if np.isfinite(lam_se) and lam_se > 0.5 * lam:
            cautions.append("poorly identified: SE > 50% of lambda")
        if lam > max_dist:
            cautions.append("lambda exceeds fitting window (lower bound only)")
        if np.isfinite(f0) and np.isfinite(background) and background > 0 and f0 < 0.5 * background:
            cautions.append("f0 << background: profile is mostly floor")
    if cautions:
        ax.text(
            0.97, 0.80, "caution: " + "; ".join(cautions),
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=t.muted, style="italic",
        )

    ax.set_xlabel("distance from source (µm)")
    ax.set_ylabel(f"mean {key}")
    if title is None:
        if fit_ok:
            se_txt = f" ± {lam_se:.0f}" if np.isfinite(lam_se) else ""
            title = (
                f"{key} radial decay  (lambda = {lam:.0f}{se_txt} µm, "
                f"R² = {r2:.2f})"
            )
        else:
            title = f"{key} radial decay (fit failed)"
    ax.set_title(title, fontsize=10)
    _style_legend(ax.legend(fontsize=7.5, loc="upper right"))

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)
