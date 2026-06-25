"""
Summary plots for gradient/vector-field analysis results.
"""

from __future__ import annotations

from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd

from astrolabe._utils import _resolve_adata
from astrolabe.pl._spatial import FigurePanel

__all__ = ["flux_plot", "direction_histogram"]


def flux_plot(
    data: ad.AnnData | md.MuData,
    key: str,
    *,
    modality: str | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (6, 5),
    annotate: bool = True,
    fmt: str = ".2f",
) -> FigurePanel:
    """Heatmap of cross-compartment expression flux.

    Reads ``uns['compartment_flux']`` written by
    :func:`astrolabe.tl.compartment_flux` and renders a diverging-colormap
    heatmap where cell ``(i, j)`` shows the mean projected gradient flux from
    compartment *i* toward compartment *j*.

    Positive values (red) indicate that the expression gradient at A–B
    boundary edges points *from A toward B*; negative values (blue) indicate
    the opposite.

    The returned :class:`~astrolabe.pl.FigurePanel` composes with spatial
    panels via ``|`` and ``/``::

        (ab.pl.points(mdata, 'leiden') | ab.pl.flux_plot(mdata, 'EPCAM')).show()

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    key
        Feature key whose flux to display.  Must be one of the keys passed to
        :func:`astrolabe.tl.compartment_flux`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    cmap
        Diverging colormap; defaults to ``'RdBu_r'``.
    figsize
        Figure size in inches.
    annotate
        If ``True``, write numeric flux values in each cell.
    fmt
        ``str.format``-compatible format string for cell annotations.

    Returns
    -------
    FigurePanel
    """
    adata = _resolve_adata(data, modality)
    uns_key = "compartment_flux"
    if uns_key not in adata.uns:
        raise KeyError(
            f"uns['{uns_key}'] not found. "
            "Run ab.tl.compartment_flux() first."
        )
    flux_data = adata.uns[uns_key]
    if "flux" not in flux_data or key not in flux_data["flux"]:
        available = list(flux_data.get("flux", {}).keys())
        raise KeyError(
            f"Key '{key}' not found in uns['compartment_flux']['flux']. "
            f"Available: {available}"
        )

    df: pd.DataFrame = flux_data["flux"][key]
    groups = df.index.tolist()
    mat = df.values.astype(float)

    abs_max = float(np.nanmax(np.abs(mat[np.isfinite(mat)]))) if np.any(np.isfinite(mat)) else 1.0
    if abs_max == 0:
        abs_max = 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        mat,
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean flux (a.u.)")

    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_xlabel("target compartment")
    ax.set_ylabel("source compartment")
    ax.set_title(f"∇{key} compartment flux")

    if annotate:
        for i in range(len(groups)):
            for j in range(len(groups)):
                val = mat[i, j]
                if np.isfinite(val):
                    text_color = "white" if abs(val) > 0.6 * abs_max else "black"
                    ax.text(
                        j, i,
                        format(val, fmt),
                        ha="center", va="center",
                        color=text_color, fontsize=8,
                    )

    fig.tight_layout()
    return FigurePanel(fig)


def direction_histogram(
    data: ad.AnnData | md.MuData,
    keys: str | Sequence[str],
    *,
    modality: str | None = None,
    n_bins: int = 36,
    figsize: tuple[float, float] = (4, 4),
    alpha: float = 0.7,
) -> FigurePanel:
    """Polar rose diagram of expression gradient directions.

    Reads ``obs[f'gradient_{key}_direction']`` written by
    :func:`astrolabe.tl.expression_gradient` (radians, arctan2 convention)
    and renders a polar bar chart.  One bar series per key is overlaid on
    the same polar axes.

    The mean resultant length (circular concentration) from
    :func:`astrolabe.tl.gradient_significance` is printed in the title when a
    single key is given.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more feature keys.  Each must have a corresponding
        ``obs['gradient_{key}_direction']`` column.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    n_bins
        Number of angular bins (default 36 = 10° per bin).
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

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [p["color"] for p in prop_cycle]

    for idx, key in enumerate(keys):
        dir_col = f"gradient_{key}_direction"
        if dir_col not in adata.obs:
            raise KeyError(
                f"obs['{dir_col}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
            )
        directions = np.asarray(adata.obs[dir_col], dtype=float)
        directions = directions[np.isfinite(directions)]

        counts, _ = np.histogram(directions, bins=bin_edges)
        # Normalize to fraction
        total = counts.sum()
        fracs = counts / total if total > 0 else counts.astype(float)

        color = colors[idx % len(colors)]
        ax.bar(
            bin_centers,
            fracs,
            width=bin_width * 0.9,
            bottom=0,
            alpha=alpha,
            color=color,
            label=key,
            align="center",
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"])
    ax.set_ylabel("")

    if len(keys) == 1:
        ax.set_title(f"∇{keys[0]} direction", pad=12)
    else:
        ax.legend(loc="lower right", bbox_to_anchor=(1.3, -0.1), fontsize=8)
        ax.set_title("gradient directions", pad=12)

    fig.tight_layout()
    return FigurePanel(fig)
