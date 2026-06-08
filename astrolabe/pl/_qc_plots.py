"""
QC and expression distribution plots.
"""

from __future__ import annotations

from typing import Any

import anndata as ad
import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from astrolabe._utils import with_adata
from astrolabe.pl._spatial import FigurePanel

__all__ = [
    "violin",
    "gene_qv_scatter",
    "cell_area_histogram",
    "nucleus_area_histogram",
    "genes_per_cell_histogram",
    "transcripts_per_cell_histogram",
]

# Default colours for known feature types (controls visually distinct from Gene Expression).
_FEATURE_TYPE_PALETTE: dict[str, str] = {
    "Gene Expression": "#AAAAAA",
    "Negative Control Probe": "#D94F3D",
    "Negative Control Codeword": "#F4883B",
    "Unassigned Codeword": "#6A7FC1",
    "Deprecated Codeword": "#5BAD6F",
    "Genomic Control": "#B05DBF",
}


@with_adata
def violin(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    groupby: str | None = None,
    log: bool = False,
    use_raw: bool = False,
    stripplot: bool = True,
    jitter: float | bool = 0.4,
    size: float = 1.0,
    layer: str | None = None,
    show: bool | None = None,
    save: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> "matplotlib.figure.Figure | matplotlib.axes.Axes | None":
    """Per-cluster or per-dataset violin plots of gene expression or QC metrics.

    Wraps :func:`scanpy.pl.violin` with astrolabe-idiomatic defaults.

    Parameters
    ----------
    adata
        AnnData.
    keys
        Gene names or ``obs`` column names to plot.
    groupby
        ``obs`` column to group by (e.g. ``'leiden'``).  ``None`` plots
        the full dataset as a single violin.
    log
        Log-transform the y-axis.
    use_raw
        Use ``adata.raw`` instead of ``adata.X``.
    stripplot
        Overlay a stripplot (individual cell dots) on the violin.
    jitter
        Amount of jitter for the stripplot.
    size
        Dot size for the stripplot.
    layer
        Layer to use instead of ``X``.
    show, save, ax, return_fig, figsize
        Standard output control arguments.
    **kwargs
        Additional arguments forwarded to ``scanpy.pl.violin``.

    Returns
    -------
    Figure, Axes, or None.
    """
    return sc.pl.violin(
        adata,
        keys=keys,
        groupby=groupby,
        log=log,
        use_raw=use_raw,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        layer=layer,
        show=show,
        save=save,
        ax=ax,
        **kwargs,
    )


@with_adata
def gene_qv_scatter(
    adata: ad.AnnData,
    transcripts: Any = None,
    *,
    qv_col: str = "qv",
    gene_col: str = "feature_name",
    feature_type_key: str = "feature_type",
    total_counts_key: str = "total_counts",
    mean_qv_key: str | None = None,
    palette: dict[str, str] | None = None,
    size: float = 10.0,
    alpha: float = 0.7,
    title: str = "Calibrated Q-score vs. transcript abundance",
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Scatter plot of mean calibrated Q-score vs. log-scaled total transcripts per gene.

    Replicates the per-gene QC plot in the 10x Xenium analysis report.  Each
    point is a gene (or control probe).  Negative controls should cluster at
    the bottom-left, well separated from true gene expression probes.

    Mean Q-score per gene is computed from *transcripts* (grouped by
    *gene_col*) or read from a pre-computed ``var`` column (*mean_qv_key*).

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`~mudata.MuData`.  Must have
        ``var['total_counts']`` (from :func:`~astrolabe.pp.calculate_qc_metrics`)
        and ``var['feature_type']`` (from :func:`~astrolabe.io.read_xenium`).
    transcripts
        :class:`pyarrow.Table` or :class:`pandas.DataFrame` returned by
        :func:`~astrolabe.io.read_xenium_transcripts`.  Used to compute mean
        Q-score per gene.  Mutually exclusive with *mean_qv_key*.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    qv_col
        Column in *transcripts* containing per-transcript quality values.
    gene_col
        Column in *transcripts* containing gene/feature names.
    feature_type_key
        Column in ``var`` containing feature type labels.
    total_counts_key
        Column in ``var`` containing total transcript counts per gene.
        Plotted on the x-axis with a log scale.
    mean_qv_key
        If *transcripts* is ``None``, read mean Q-score from this ``var``
        column instead.
    palette
        Mapping of feature-type label → matplotlib colour string.  Overrides
        the built-in Xenium-matched palette for any keys supplied.
    size
        Marker size in points².
    alpha
        Marker opacity.
    title
        Axes title.
    ax
        Existing :class:`~matplotlib.axes.Axes` to draw into.  A new figure
        is created when ``None``.
    figsize
        ``(width, height)`` in inches, used only when *ax* is ``None``.
    **scatter_kwargs
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> tx = ab.io.read_xenium_transcripts("Xenium_run/")
    >>> ab.pl.gene_qv_scatter(mdata, transcripts=tx)

    >>> # Using a pre-computed var column
    >>> ab.pl.gene_qv_scatter(mdata, mean_qv_key="mean_qv")
    """
    if total_counts_key not in adata.var.columns:
        raise KeyError(
            f"var['{total_counts_key}'] not found. "
            "Run ab.pp.calculate_qc_metrics(data) first."
        )
    if feature_type_key not in adata.var.columns:
        raise KeyError(
            f"var['{feature_type_key}'] not found. "
            "Ensure var contains feature-type labels before calling gene_qv_scatter."
        )

    # --- compute mean QV per gene ---
    if transcripts is not None and mean_qv_key is not None:
        raise ValueError("Provide either 'transcripts' or 'mean_qv_key', not both.")

    if transcripts is None and mean_qv_key is None:
        transcripts = adata.uns.get("transcripts")

    if transcripts is not None:
        # Convert pyarrow → pandas if needed
        try:
            import pyarrow as pa
            if isinstance(transcripts, pa.Table):
                transcripts = transcripts.select([gene_col, qv_col]).to_pandas()
        except ImportError:
            pass
        if not isinstance(transcripts, pd.DataFrame):
            raise TypeError(
                f"transcripts must be a pyarrow.Table or pandas.DataFrame, "
                f"got {type(transcripts).__name__!r}."
            )
        mean_qv_series = transcripts.groupby(gene_col)[qv_col].mean()
        mean_qv = adata.var_names.map(mean_qv_series).to_numpy(dtype=np.float64)
    elif mean_qv_key is not None:
        if mean_qv_key not in adata.var.columns:
            raise KeyError(
                f"var['{mean_qv_key}'] not found. "
                "Either pass transcripts= or pre-compute the column."
            )
        mean_qv = adata.var[mean_qv_key].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            "No Q-score source found. Either: run ab.io.read_xenium_transcripts(mdata) "
            "so the table is available in uns['transcripts'], pass transcripts= explicitly, "
            "or supply mean_qv_key= if the column is pre-computed in var."
        )

    counts = adata.var[total_counts_key].to_numpy(dtype=np.float64) + 1
    feature_types = adata.var[feature_type_key].astype(str).to_numpy()

    # Augment with per-gene control stats stashed by calculate_xenium_qc_metrics.
    # This restores individual control probe points even after controls are removed from var.
    ctrl_stash = adata.uns.get('control_probe_summary', {})
    if ctrl_stash and 'total_counts' in ctrl_stash:
        ctrl_counts = np.array(ctrl_stash['total_counts'], dtype=np.float64) + 1
        ctrl_qv = np.array(
            ctrl_stash.get('mean_qv', [np.nan] * len(ctrl_counts)), dtype=np.float64
        )
        ctrl_types = np.array(ctrl_stash['feature_type'])
        counts = np.concatenate([counts, ctrl_counts])
        mean_qv = np.concatenate([mean_qv, ctrl_qv])
        feature_types = np.concatenate([feature_types, ctrl_types])

    # --- resolve palette ---
    effective_palette = {**_FEATURE_TYPE_PALETTE, **(palette or {})}
    unique_types = sorted(set(feature_types))

    # Assign a colour to any feature type not in the palette
    fallback_cmap = plt.get_cmap("tab10")
    known = set(effective_palette)
    extra = [ft for ft in unique_types if ft not in known]
    for i, ft in enumerate(extra):
        effective_palette[ft] = mcolors.to_hex(fallback_cmap(i % 10))

    # --- plot ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw Gene Expression first (background), controls on top
    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]
    for ft in draw_order:
        mask = feature_types == ft
        if not mask.any():
            continue
        ax.scatter(
            counts[mask],
            mean_qv[mask],
            c=effective_palette.get(ft, "#888888"),
            s=size,
            alpha=alpha,
            label=ft,
            rasterized=mask.sum() > 5_000,
            **scatter_kwargs,
        )

    ax.set_xscale("log")
    ax.set_xlabel("total transcripts + 1")
    ax.set_ylabel("Mean calibrated Q-score")
    ax.set_title(title)

    legend_handles = [
        mpatches.Patch(color=effective_palette.get(ft, "#888888"), label=ft)
        for ft in unique_types
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _obs_histogram(
    adata: ad.AnnData,
    key: str,
    *,
    bins: int | np.ndarray = 50,
    log: bool = False,
    color: str = "#4C72B0",
    xlabel: str | None = None,
    ylabel: str = "Number of cells",
    title: str | None = None,
    vlines: list[float] | None = None,
    vline_color: str = "#D94F3D",
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Shared implementation for per-cell obs-column histograms."""
    if key not in adata.obs.columns:
        raise KeyError(
            f"obs['{key}'] not found. Run ab.pp.calculate_qc_metrics(data) first."
        )
    values = adata.obs[key].to_numpy(dtype=np.float64)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if log and isinstance(bins, int):
        pos = values[values > 0]
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1) if len(pos) else bins

    ax.hist(values, bins=bins, color=color, edgecolor="none", **hist_kwargs)

    if log:
        ax.set_xscale("log")

    if vlines:
        for v in vlines:
            ax.axvline(v, color=vline_color, linestyle="--", linewidth=1.2)

    ax.set_xlabel(xlabel or key)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def cell_area_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str = "#4C72B0",
    title: str = "Cell area",
    vlines: list[float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of cell area in µm².

    Requires ``obs['cell_area']``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    bins
        Number of histogram bins or an explicit bin-edge array.
    color
        Bar colour.
    title
        Axes title.
    vlines
        X-positions of vertical reference lines (e.g. filter thresholds).
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _obs_histogram(
        adata, "cell_area",
        bins=bins, color=color, xlabel="Cell area (µm²)", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def nucleus_area_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str = "#4C72B0",
    title: str = "Nucleus area",
    vlines: list[float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of nucleus area in µm².

    Requires ``obs['nucleus_area']``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    bins
        Number of histogram bins or an explicit bin-edge array.
    color
        Bar colour.
    title
        Axes title.
    vlines
        X-positions of vertical reference lines (e.g. filter thresholds).
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    FigurePanel
    """
    return _obs_histogram(
        adata, "nucleus_area",
        bins=bins, color=color, xlabel="Nucleus area (µm²)", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def genes_per_cell_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str = "#4C72B0",
    title: str = "Genes per cell",
    vlines: list[float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of unique genes detected per cell.

    Requires ``obs['n_genes_by_counts']`` (populated by
    :func:`~astrolabe.pp.calculate_qc_metrics`).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    bins
        Number of histogram bins or an explicit bin-edge array.
    color
        Bar colour.
    title
        Axes title.
    vlines
        X-positions of vertical reference lines (e.g. filter thresholds).
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _obs_histogram(
        adata, "n_genes_by_counts",
        bins=bins, color=color, xlabel="Genes per cell", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def transcripts_per_cell_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    log: bool = True,
    color: str = "#4C72B0",
    title: str = "Transcripts per cell",
    vlines: list[float] | None = None,
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of total transcript counts per cell.

    Uses a log x-axis by default because transcript counts span several orders
    of magnitude.  Requires ``obs['total_counts']`` (populated by
    :func:`~astrolabe.pp.calculate_qc_metrics`).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    bins
        Number of histogram bins or an explicit bin-edge array.  When *log*
        is ``True`` and *bins* is an integer the edges are log-spaced
        automatically.
    log
        Use a log x-axis with log-spaced bins (default ``True``).
    color
        Bar colour.
    title
        Axes title.
    vlines
        X-positions of vertical reference lines (e.g. filter thresholds).
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _obs_histogram(
        adata, "total_counts",
        bins=bins, log=log, color=color, xlabel="Transcripts per cell", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )