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
    "dotplot",
    "heatmap",
    "embedding_scatter",
    "volcano_plot",
    "paga_plot",
    "ripley_plot",
    "gene_qv_scatter",
    "gene_detection_scatter",
    "mean_variance_plot",
    "cell_area_histogram",
    "nucleus_area_histogram",
    "genes_per_cell_histogram",
    "transcripts_per_cell_histogram",
    "intensity_histogram",
    "intensity_histograms",
    "scatter_gate",
    "shape_distribution_plot",
    "shape_ratio_plot",
    "moran_plot",
    "nhood_enrichment_plot",
    "co_occurrence_plot",
    "centrality_scores_plot",
    "expression_by_distance_plot",
    "ligrec_plot",
    "som_umatrix",
    "som_component_plane",
    "som_hits",
    "som_subcellular_plot",
]

# Default colors for known feature types (controls visually distinct from Gene Expression).
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
    save: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> FigurePanel:
    """Per-cluster or per-dataset violin plots of gene expression or QC metrics.

    Wraps :func:`scanpy.pl.violin` with astrolabe-idiomatic defaults and
    returns a composable :class:`~astrolabe.pl.FigurePanel`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    keys
        Gene names or ``obs`` column names to plot
    groupby
        ``obs`` column to group by (e.g. ``'leiden'``).  ``None`` plots
        the full dataset as a single violin
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
    save
        Path to save the figure.
    ax
        Existing axes to draw into.
    **kwargs
        Additional arguments forwarded to ``scanpy.pl.violin``.

    Returns
    -------
    FigurePanel
    """
    result = sc.pl.violin(
        adata,
        keys=keys,
        groupby=groupby,
        log=log,
        use_raw=use_raw,
        stripplot=stripplot,
        jitter=jitter,
        size=size,
        layer=layer,
        show=False,
        save=save,
        ax=ax,
        **kwargs,
    )
    return FigurePanel.from_obj(result)


@with_adata
def dotplot(
    adata: ad.AnnData,
    var_names: list[str] | dict[str, list[str]],
    groupby: str,
    *,
    layer: str | None = None,
    use_raw: bool | None = None,
    log: bool = False,
    num_categories: int = 7,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> FigurePanel:
    """Dot plot of gene expression across cluster groups.

    Dot size encodes the fraction of cells expressing each gene; color
    encodes mean expression.  Returns a :class:`~astrolabe.pl.FigurePanel`
    so the result can be composed with spatial panels using ``|`` and ``/``.

    Wraps :func:`scanpy.pl.dotplot`.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`mudata.MuData`.
    var_names
        Genes to display.  Pass a list for a flat view, or a dict mapping
        group label → list of genes to display genes with group separators::

            {"T cells": ["CD3D", "CD8A"], "Epithelial": ["EPCAM", "KRT8"]}

    groupby
        ``obs`` column to group cells by (e.g. ``'leiden'`` or
        ``'leiden_annotation'``).
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer.  ``None`` uses ``X``.
    use_raw
        Use ``adata.raw``.  ``None`` defers to scanpy's default.
    log
        Log-transform expression values before plotting.
    num_categories
        Maximum number of categories shown on the groupby axis before
        labels are truncated.
    figsize
        ``(width, height)`` in inches.  ``None`` lets scanpy auto-size.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pl.dotplot`.

    Returns
    -------
    FigurePanel
        Composable panel — use ``|`` to place next to a spatial plot::

            ab.pl.dotplot(mdata, markers, "leiden") | ab.pl.points(mdata, "leiden")

    Examples
    --------
    >>> markers = {"T cell": ["CD3D", "CD8A"], "Epithelial": ["EPCAM", "KRT8"]}
    >>> ab.pl.dotplot(mdata, markers, groupby="leiden_annotation").show()
    >>> (ab.pl.dotplot(mdata, markers, "leiden") | ab.pl.points(mdata, "leiden")).show()
    """
    dp = sc.pl.dotplot(
        adata,
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        figsize=figsize,
        show=False,
        return_fig=True,
        **kwargs,
    )

    return FigurePanel.from_obj(dp)


@with_adata
def heatmap(
    adata: ad.AnnData,
    var_names: "list[str] | dict[str, list[str]]",
    groupby: str,
    *,
    layer: "str | None" = None,
    use_raw: "bool | None" = None,
    log: bool = False,
    num_categories: int = 7,
    figsize: "tuple[float, float] | None" = None,
    swap_axes: bool = False,
    show_gene_labels: "bool | None" = None,
    **kwargs: Any,
) -> FigurePanel:
    """Heatmap of mean gene expression across cluster groups.

    Each row is a gene, each column is a cluster group (or swapped with
    ``swap_axes=True``).  Returns a :class:`~astrolabe.pl.FigurePanel` so
    the result can be composed with spatial panels using ``|`` and ``/``.

    Wraps :func:`scanpy.pl.heatmap`.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`mudata.MuData`.
    var_names
        Genes to display.  Pass a list for a flat view, or a dict mapping
        group label → list of genes to display genes with group separators::

            {"T cells": ["CD3D", "CD8A"], "Epithelial": ["EPCAM", "KRT8"]}

    groupby
        ``obs`` column to group cells by (e.g. ``'leiden'`` or
        ``'leiden_annotation'``).
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer.  ``None`` uses ``X``.
    use_raw
        Use ``adata.raw``.  ``None`` defers to scanpy's default.
    log
        Log-transform expression values before plotting.
    num_categories
        Maximum number of categories shown before labels are truncated.
    figsize
        ``(width, height)`` in inches.  ``None`` lets scanpy auto-size.
    swap_axes
        Transpose the heatmap (genes on x-axis, groups on y-axis).
    show_gene_labels
        Show gene labels on the gene axis.  ``None`` auto-decides based
        on the number of genes.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pl.heatmap`.

    Returns
    -------
    FigurePanel
        Composable panel — use ``|`` or ``/`` to combine with spatial plots::

            ab.pl.heatmap(mdata, markers, "leiden") | ab.pl.points(mdata, "leiden")

    Examples
    --------
    >>> markers = {"T cell": ["CD3D", "CD8A"], "Epithelial": ["EPCAM", "KRT8"]}
    >>> ab.pl.heatmap(mdata, markers, groupby="leiden_annotation").show()
    """
    import matplotlib.pyplot as plt

    sc.pl.heatmap(
        adata,
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        use_raw=use_raw,
        log=log,
        num_categories=num_categories,
        figsize=figsize,
        swap_axes=swap_axes,
        show_gene_labels=show_gene_labels,
        show=False,
        **kwargs,
    )

    return FigurePanel.from_obj(plt.gcf())


@with_adata
def embedding_scatter(
    adata: ad.AnnData,
    *,
    obsm_key: str = "X_umap",
    color: "str | list[str] | None" = None,
    title: "str | list[str] | None" = None,
    palette: "str | dict | None" = None,
    size: "float | None" = None,
    alpha: float = 0.7,
    frameon: bool = False,
    figsize: "tuple[float, float] | None" = None,
    ax: "matplotlib.axes.Axes | None" = None,
    **kwargs: Any,
) -> FigurePanel:
    """Scatter plot of a 2-D embedding (UMAP, PCA, spatial_umap, etc.).

    Wraps :func:`scanpy.pl.embedding` and returns a composable
    :class:`~astrolabe.pl.FigurePanel`.  When multiple ``color`` values are
    requested, scanpy draws one sub-panel per color.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    obsm_key
        Key in ``obsm`` to use as the 2-D coordinates (e.g. ``'X_umap'``,
        ``'X_pca'``, ``'X_spatial_umap'``).  The ``'X_'`` prefix is stripped
        when passing to scanpy's ``basis`` argument.
    color
        One or more ``obs`` column names or gene names to color cells by.
        ``None`` draws all cells in a single grey color.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    title
        Axes title(s).  ``None`` defaults to the color key.
    palette
        Color palette for categorical colors.
    size
        Scatter marker size in points².  ``None`` (default) lets scanpy
        auto-scale as ``120 000 / n_cells``, which is appropriate for most
        UMAP and PCA embeddings.  Pass an explicit float to override.
    alpha
        Marker opacity.
    frameon
        Show axes frame (default ``False``).
    figsize
        ``(width, height)`` in inches.  ``None`` lets scanpy auto-size.
    ax
        Existing :class:`~matplotlib.axes.Axes` to draw into.  When provided,
        scanpy draws into this axes and the figure is extracted from it.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pl.embedding`.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.umap(mdata)
    >>> ab.pl.embedding_scatter(mdata, color="leiden").show()
    >>> (ab.pl.embedding_scatter(mdata, color="EPCAM") | ab.pl.points(mdata, "EPCAM")).show()
    """
    # scanpy's embedding() takes basis without the 'X_' prefix
    basis = obsm_key[2:] if obsm_key.startswith("X_") else obsm_key

    # sc.pl.embedding has no figsize param in current scanpy; create the
    # axes ourselves so figsize is honored instead of leaking into **kwargs
    # (where it would reach Axes.scatter() and raise an AttributeError).
    made_ax = ax is None
    if made_ax:
        fig, ax = plt.subplots(figsize=figsize)

    result = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        title=title,
        palette=palette,
        **({"size": size} if size is not None else {}),
        alpha=alpha,
        frameon=frameon,
        ax=ax,
        show=False,
        **kwargs,
    )

    fig = ax.get_figure() if not made_ax else FigurePanel.from_obj(result).figure
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def volcano_plot(
    adata: ad.AnnData,
    group: "str | None" = None,
    *,
    key: str = "rank_genes_groups",
    pval_col: str = "pvals_adj",
    lfc_col: str = "logfoldchanges",
    pval_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    top_n: int = 10,
    point_size: float = 5.0,
    alpha: float = 0.5,
    up_color: str = "#D94F3D",
    down_color: str = "#6A7FC1",
    ns_color: str = "#AAAAAA",
    title: "str | None" = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: "tuple[float, float]" = (6, 5),
) -> FigurePanel:
    """Volcano plot of differential expression results.

    Reads results stored by :func:`~astrolabe.tl.rank_genes_groups` in
    ``uns['rank_genes_groups']`` and plots log₂FC vs −log₁₀(p-value).
    Up-regulated genes (log₂FC > *lfc_threshold* and p < *pval_threshold*)
    are shown in red; down-regulated in blue; non-significant in grey.
    Top-*n* most significant genes in each direction are labeled.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    group
        Name of the comparison group.  When ``None`` and only one group
        exists in the results, it is used automatically.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.rank_genes_groups`.
    pval_col
        Column to use for significance: ``'pvals_adj'`` (default) or
        ``'pvals'``.
    lfc_col
        Column containing log₂ fold changes.
    pval_threshold
        Significance threshold (dashed horizontal line).
    lfc_threshold
        Fold-change threshold (dashed vertical lines).
    top_n
        Number of extreme genes to label per direction.
    point_size, alpha
        Scatter marker parameters.
    up_color, down_color, ns_color
        Colors for up-regulated, down-regulated, and non-significant genes.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.tl.rank_genes_groups(mdata, groupby="leiden")
    >>> ab.pl.volcano_plot(mdata, group="0").show()
    >>> (ab.pl.volcano_plot(mdata, "0") | ab.pl.points(mdata, "leiden")).show()
    """
    if key not in adata.uns:
        raise KeyError(
            f"uns['{key}'] not found. Run ab.tl.rank_genes_groups() first"
        )

    rgg = adata.uns[key]
    groups_stored = list(rgg["names"].dtype.names)

    if group is None:
        if len(groups_stored) == 1:
            group = groups_stored[0]
        else:
            raise ValueError(
                f"Multiple groups in uns['{key}']: {groups_stored}. "
                "Specify group= explicitly"
            )
    if group not in groups_stored:
        raise KeyError(
            f"Group '{group}' not found in uns['{key}']. Available: {groups_stored}"
        )

    genes = np.array(rgg["names"][group]).astype(str)
    lfc = np.array(rgg[lfc_col][group], dtype=np.float64)

    pval_source = pval_col if pval_col in rgg else "pvals"
    pvals_raw = np.array(rgg[pval_source][group], dtype=np.float64)
    pvals_raw = np.clip(pvals_raw, 1e-300, 1.0)
    neg_log_p = -np.log10(pvals_raw)

    # Categorize points
    sig_up   = (pvals_raw < pval_threshold) & (lfc >  lfc_threshold)
    sig_down = (pvals_raw < pval_threshold) & (lfc < -lfc_threshold)
    colors = np.where(sig_up, up_color, np.where(sig_down, down_color, ns_color))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Non-significant first (background), then colored points on top
    for mask, c in [(~(sig_up | sig_down), ns_color), (sig_up, up_color), (sig_down, down_color)]:
        if mask.any():
            ax.scatter(
                lfc[mask], neg_log_p[mask],
                c=c, s=point_size, alpha=alpha, linewidths=0,
                rasterized=int(mask.sum()) > 5_000,
            )

    # Reference lines
    ax.axhline(-np.log10(pval_threshold), color="#555555", linestyle="--", linewidth=0.9)
    ax.axvline( lfc_threshold, color="#555555", linestyle="--", linewidth=0.9)
    ax.axvline(-lfc_threshold, color="#555555", linestyle="--", linewidth=0.9)

    # Label top-n significant genes per direction
    def _label_top(mask: np.ndarray, n: int) -> None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        top = idx[np.argsort(pvals_raw[idx])[:n]]
        for i in top:
            ax.annotate(
                genes[i], (lfc[i], neg_log_p[i]),
                fontsize=6, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
            )

    _label_top(sig_up, top_n)
    _label_top(sig_down, top_n)

    ax.set_xlabel("log₂ fold change", fontsize=9)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=9)
    ax.set_title(title or f"Volcano — group {group}", fontsize=10)

    handles = [
        mpatches.Patch(color=up_color, label=f"Up (n={sig_up.sum()})"),
        mpatches.Patch(color=down_color, label=f"Down (n={sig_down.sum()})"),
        mpatches.Patch(color=ns_color, label="NS"),
    ]
    ax.legend(handles=handles, fontsize=7, framealpha=0.7)

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def paga_plot(
    adata: ad.AnnData,
    *,
    color: "str | list[str] | None" = None,
    threshold: float = 0.01,
    layout: "str | None" = None,
    title: "str | None" = None,
    figsize: "tuple[float, float]" = (5, 5),
    **kwargs: Any,
) -> FigurePanel:
    """PAGA trajectory graph.

    Thin wrapper around :func:`scanpy.pl.paga` that returns a composable
    :class:`~astrolabe.pl.FigurePanel`.  Requires
    :func:`~astrolabe.tl.paga` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    color
        ``obs`` column or gene name used to color cluster nodes.
        ``None`` uses the groupby column from the PAGA run.
    threshold
        Minimum edge weight to draw.
    layout
        Graph layout algorithm (e.g. ``'fr'``, ``'eq_tree'``).
    title
        Axes title.
    figsize
        Figure size in inches.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pl.paga`.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.tl.paga(mdata, groups="leiden")
    >>> ab.pl.paga_plot(mdata, color="leiden").show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.paga(
        adata,
        color=color,
        threshold=threshold,
        layout=layout,
        title=title,
        show=False,
        ax=ax,
        **kwargs,
    )
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def ripley_plot(
    adata: ad.AnnData,
    *,
    key: str = "ripley_l",
    cell_types: "list[str] | None" = None,
    cmap: str = "tab10",
    title: "str | None" = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: "tuple[float, float]" = (6, 4),
) -> FigurePanel:
    """Ripley's L function plot (L(d) − d vs. distance).

    Visualizes the output of :func:`~astrolabe.tl.ripley_l`.  Values above
    zero indicate spatial clustering; values below indicate regularity.  A
    dashed reference line at zero marks complete spatial randomness (CSR).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.ripley_l`.
    cell_types
        Cell types to plot.  ``None`` plots all available.
    cmap
        Colormap name or instance used to cycle line colors.  Default ``"tab10"``.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.ripley_l(mdata, cluster_key="leiden")
    >>> ab.pl.ripley_plot(mdata).show()
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.ripley_l() first")

    result = adata.uns[key]
    # result is a dict keyed by cell-type string, each value has 'bins' and 'stats'
    if cell_types is None:
        cell_types = [k for k in result if not k.startswith("_")]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap_obj = plt.get_cmap(cmap)
    for i, ct in enumerate(cell_types):
        if ct not in result:
            continue
        entry = result[ct]
        # entry is a DataFrame with index=distance_um and columns L_obs, L_exp, L_hi, L_lo
        if hasattr(entry, "index"):
            bins = np.asarray(entry.index, dtype=float)
            l_col = "L_obs" if "L_obs" in entry.columns else entry.columns[0]
            l_vals = entry[l_col].to_numpy(dtype=float)
        else:
            bins = np.asarray(entry.get("bins", []))
            stats = entry.get("stats")
            l_vals = stats["L"].to_numpy(dtype=float) if hasattr(stats, "columns") else np.asarray(stats, dtype=float)

        l_d = l_vals - bins[:len(l_vals)]
        color = cmap_obj(i % 10)
        ax.plot(bins[:len(l_d)], l_d, label=ct, color=color)
        # Draw simulation envelope if available
        if hasattr(entry, "columns") and "L_hi" in entry.columns and "L_lo" in entry.columns:
            l_hi = entry["L_hi"].to_numpy(dtype=float) - bins[:len(l_vals)]
            l_lo = entry["L_lo"].to_numpy(dtype=float) - bins[:len(l_vals)]
            ax.fill_between(bins[:len(l_d)], l_lo, l_hi, alpha=0.15, color=color)

    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance d (µm)", fontsize=9)
    ax.set_ylabel("L(d) − d", fontsize=9)
    ax.set_title(title or "Ripley's L", fontsize=10)

    n = len(cell_types)
    if n <= 15:
        ax.legend(fontsize=7, framealpha=0.7, loc="upper left")

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


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
        Mapping of feature-type label → matplotlib color string.  Overrides
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
            "Run ab.pp.calculate_qc_metrics(data) first"
        )
    if feature_type_key not in adata.var.columns:
        raise KeyError(
            f"var['{feature_type_key}'] not found. "
            "Ensure var contains feature-type labels before calling gene_qv_scatter"
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
                f"got {type(transcripts).__name__!r}"
            )
        mean_qv_series = transcripts.groupby(gene_col)[qv_col].mean()
        mean_qv = adata.var_names.map(mean_qv_series).to_numpy(dtype=np.float64)
    elif mean_qv_key is not None:
        if mean_qv_key not in adata.var.columns:
            raise KeyError(
                f"var['{mean_qv_key}'] not found. "
                "Either pass transcripts= or pre-compute the column"
            )
        mean_qv = adata.var[mean_qv_key].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            "No Q-score source found. Either: run ab.io.read_xenium_transcripts(mdata) "
            "so the table is available in uns['transcripts'], pass transcripts= explicitly, "
            "or supply mean_qv_key= if the column is pre-computed in var"
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

    # Assign a color to any feature type not in the palette
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
# Gene detection rate scatter
# ---------------------------------------------------------------------------


@with_adata
def gene_detection_scatter(
    adata: ad.AnnData,
    *,
    feature_type_key: str = "feature_type",
    control_key: str = "is_control",
    total_counts_key: str = "total_counts",
    pct_dropout_key: str = "pct_dropout_by_counts",
    palette: dict[str, str] | None = None,
    size: float = 10.0,
    alpha: float = 0.7,
    title: str = "Gene detection rate vs. transcript abundance",
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Scatter plot of gene detection rate vs. log-scaled total transcripts per gene.

    Platform-agnostic analog of :func:`gene_qv_scatter`.  Each point is a gene
    (or control probe).  Negative / blank controls should cluster at the
    bottom-left (low counts, low detection rate), well separated from true
    genes which typically show higher detection rates at a given abundance.

    Requires :func:`~astrolabe.pp.calculate_qc_metrics` to have been called
    first (provides ``var['total_counts']`` and ``var['pct_dropout_by_counts']``).

    Works on all platforms:

    - **Xenium** — colored by ``var['feature_type']`` (full control-type palette)
    - **CosMx / MERSCOPE** — colored by ``var['is_control_probe']`` (boolean)

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`~mudata.MuData`.  Must have
        ``var['total_counts']`` and ``var['pct_dropout_by_counts']`` from
        :func:`~astrolabe.pp.calculate_qc_metrics`.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    feature_type_key
        Column in ``var`` containing feature-type labels (Xenium).  If absent,
        falls back to *control_key* or ``'is_control_probe'``.
    control_key
        Boolean ``var`` column used when *feature_type_key* is absent.
        Falls back to ``'is_control_probe'`` if this column is also absent.
    total_counts_key
        Column in ``var`` containing total transcript counts.  X-axis (log).
    pct_dropout_key
        Column in ``var`` containing the percentage of cells with zero counts.
        Detection rate = ``1 - pct_dropout / 100``.
    palette
        Mapping of feature-type label → matplotlib color.  Overrides the
        built-in palette for any keys supplied.
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
    FigurePanel

    Examples
    --------
    >>> ab.pp.calculate_qc_metrics(mdata)
    >>> ab.pl.gene_detection_scatter(mdata)            # CosMx / MERSCOPE / Xenium

    >>> ab.pl.gene_detection_scatter(mdata, palette={"Control Probe": "blue"})
    """
    for key in (total_counts_key, pct_dropout_key):
        if key not in adata.var.columns:
            raise KeyError(
                f"var['{key}'] not found. "
                "Run ab.pp.calculate_qc_metrics(data) first."
            )

    counts = adata.var[total_counts_key].to_numpy(dtype=np.float64) + 1
    detection_rate = 1.0 - adata.var[pct_dropout_key].to_numpy(dtype=np.float64) / 100.0

    # --- resolve feature labels ---
    if feature_type_key in adata.var.columns:
        feature_types = adata.var[feature_type_key].astype(str).to_numpy()
        effective_palette = {**_FEATURE_TYPE_PALETTE, **(palette or {})}

        # Restore stashed controls (Xenium: after calculate_xenium_qc_metrics + filter_genes)
        ctrl_stash = adata.uns.get("control_probe_summary", {})
        if ctrl_stash and "total_counts" in ctrl_stash:
            ctrl_counts = np.array(ctrl_stash["total_counts"], dtype=np.float64) + 1
            ctrl_types = np.array(ctrl_stash["feature_type"])
            ctrl_detection = np.full(len(ctrl_counts), np.nan)
            counts = np.concatenate([counts, ctrl_counts])
            detection_rate = np.concatenate([detection_rate, ctrl_detection])
            feature_types = np.concatenate([feature_types, ctrl_types])
    else:
        _ctrl_col = control_key if control_key in adata.var.columns else "is_control_probe"
        if _ctrl_col in adata.var.columns:
            is_ctrl = adata.var[_ctrl_col].astype(bool).to_numpy()
            feature_types = np.where(is_ctrl, "Control Probe", "Gene Expression")
        else:
            feature_types = np.full(len(counts), "Gene Expression")
        effective_palette = {
            "Gene Expression": "#AAAAAA",
            "Control Probe": "#D94F3D",
            **(palette or {}),
        }

    # Assign fallback colors for any unlisted types
    unique_types = sorted(set(feature_types))
    fallback_cmap = plt.get_cmap("tab10")
    known = set(effective_palette)
    for i, ft in enumerate(ft for ft in unique_types if ft not in known):
        effective_palette[ft] = mcolors.to_hex(fallback_cmap(i % 10))

    # --- plot ---
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]
    for ft in draw_order:
        mask = feature_types == ft
        if not mask.any():
            continue
        ax.scatter(
            counts[mask],
            detection_rate[mask],
            c=effective_palette.get(ft, "#888888"),
            s=size,
            alpha=alpha,
            label=ft,
            rasterized=mask.sum() > 5_000,
            **scatter_kwargs,
        )

    ax.set_xscale("log")
    ax.set_xlabel("total transcripts + 1")
    ax.set_ylabel("Detection rate (fraction of cells)")
    ax.set_ylim(-0.05, 1.05)
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
# Mean-variance plot
# ---------------------------------------------------------------------------


@with_adata
def mean_variance_plot(
    adata: ad.AnnData,
    *,
    feature_type_key: str = "feature_type",
    control_key: str = "is_control",
    palette: dict[str, str] | None = None,
    size: float = 8.0,
    alpha: float = 0.6,
    title: str = "Mean-variance relationship",
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Per-gene mean vs. variance scatter plot with a Poisson reference line.

    On a log-log scale, genes following a Poisson distribution (e.g. negative
    control probes) should fall along the diagonal ``y = x`` (variance =
    mean).  Biologically expressed genes are overdispersed — their variance
    exceeds their mean and they lie **above** the reference line.  Systematic
    departure of control probes above ``y = x`` indicates technical noise.

    .. note::
        Call this on raw counts (before :func:`~astrolabe.pp.normalize_total`
        or :func:`~astrolabe.pp.log1p`).  After transformation the
        mean-variance relationship no longer follows Poisson statistics.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  ``X`` must
        contain raw integer counts.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    feature_type_key
        ``var`` column containing feature-type labels (e.g. ``'feature_type'``
        from :func:`~astrolabe.io.read_xenium`).  Used to color points by
        probe type.  Falls back to *control_key* when absent.
    control_key
        ``var`` boolean column set by :func:`~astrolabe.pp.flag_controls`
        (``'is_control'``).  Used when *feature_type_key* is not present.
    palette
        Color overrides for feature-type labels.  Merged with the built-in
        Xenium palette.
    size
        Marker size in points².
    alpha
        Marker opacity.
    title
        Axes title.
    ax
        Existing :class:`~matplotlib.axes.Axes` to draw into.
    figsize
        Figure size in inches when *ax* is ``None``.
    **scatter_kwargs
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    FigurePanel
    """
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel().astype(np.float64)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        mean = Xd.mean(axis=0)
        mean_sq = (Xd ** 2).mean(axis=0)

    variance = mean_sq - mean ** 2

    # ---- feature-type grouping ----------------------------------------
    if feature_type_key in adata.var.columns:
        feature_types = adata.var[feature_type_key].astype(str).to_numpy()
    elif control_key in adata.var.columns:
        is_ctrl = adata.var[control_key].to_numpy(dtype=bool)
        feature_types = np.where(is_ctrl, "Negative Control", "Gene Expression")
    elif "is_control_probe" in adata.var.columns:
        is_ctrl = adata.var["is_control_probe"].to_numpy(dtype=bool)
        feature_types = np.where(is_ctrl, "Negative Control Probe", "Gene Expression")
    else:
        feature_types = np.full(adata.n_vars, "Gene Expression", dtype=object)

    effective_palette = {**_FEATURE_TYPE_PALETTE, **(palette or {})}
    unique_types = sorted(set(feature_types))

    fallback_cmap = plt.get_cmap("tab10")
    extra = [ft for ft in unique_types if ft not in effective_palette]
    for i, ft in enumerate(extra):
        effective_palette[ft] = mcolors.to_hex(fallback_cmap(i % 10))

    # ---- plot ------------------------------------------------------------
    own_fig = ax is None
    if own_fig:
        fig, ax_plot = plt.subplots(figsize=figsize)
    else:
        ax_plot = ax
        fig = ax_plot.get_figure()

    # Gene Expression first (background), controls on top
    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]
    for ft in draw_order:
        mask = feature_types == ft
        if not mask.any():
            continue
        ax_plot.scatter(
            mean[mask],
            variance[mask],
            c=effective_palette.get(ft, "#888888"),
            s=size,
            alpha=alpha,
            label=ft,
            rasterized=int(mask.sum()) > 5_000,
            **scatter_kwargs,
        )

    ax_plot.set_xscale("log")
    ax_plot.set_yscale("log")

    # Poisson reference line: variance = mean  (y = x on log-log)
    pos = mean[mean > 0]
    if len(pos) > 0:
        lo, hi = float(pos.min()), float(pos.max())
        x_ref = np.logspace(np.log10(lo), np.log10(hi), 300)
        ax_plot.plot(
            x_ref, x_ref,
            color="#D94F3D", linestyle="--", linewidth=1.2,
            zorder=5, label="Poisson (var = mean)",
        )

    ax_plot.set_xlabel("Mean expression (counts per cell)")
    ax_plot.set_ylabel("Variance")
    ax_plot.set_title(title)

    legend_handles = [
        mpatches.Patch(color=effective_palette.get(ft, "#888888"), label=ft)
        for ft in unique_types
    ] + [
        plt.Line2D(
            [0], [0],
            color="#D94F3D", linestyle="--", linewidth=1.2,
            label="Poisson (var = mean)",
        )
    ]
    ax_plot.legend(
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
            f"obs['{key}'] not found. Run ab.pp.calculate_qc_metrics(data) first"
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
        Bar color.
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
        Bar color.
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
        Bar color.
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
        Bar color.
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


# ---------------------------------------------------------------------------
# Shape feature distribution plots
# ---------------------------------------------------------------------------

_SHAPE_METRIC_SUFFIXES: frozenset[str] = frozenset({
    "area", "perimeter", "span_x", "span_y", "equivalent_radius",
    "major_axis_length", "minor_axis_length", "aspect_ratio", "rna_density",
    "circularity", "solidity", "elongation",
})


def _array_histogram(
    values: np.ndarray,
    *,
    bins: int | np.ndarray = 50,
    log: bool = False,
    color: str = "#4C72B0",
    xlabel: str = "",
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of an arbitrary float array (does not read from obs)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if log and isinstance(bins, int) and len(vals) > 0:
        pos = vals[vals > 0]
        if len(pos):
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1)

    ax.hist(vals, bins=bins, color=color, edgecolor="none", **hist_kwargs)

    if log:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of cells")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def shape_distribution_plot(
    adata: ad.AnnData,
    keys: list[str] | None = None,
    *,
    prefix: str | None = None,
    ncols: int = 3,
    bins: int | np.ndarray = 50,
    log: bool = False,
    color: str = "#4C72B0",
    figsize_per_panel: tuple[float, float] = (4, 3),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram grid of shape feature distributions from ``obs``.

    Designed to visualize the output of
    :func:`~astrolabe.pp.calculate_shape_metrics`.  When *keys* is ``None``
    the function auto-detects relevant columns from *prefix* or from the
    known set of shape metric suffixes.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    keys
        Explicit list of ``obs`` column names to plot.  When ``None``,
        columns are detected automatically.
    prefix
        If *keys* is ``None``, restrict auto-detection to columns that
        start with this prefix (e.g. ``'cell_'`` or ``'nucleus_'``).
    ncols
        Number of histogram panels per row.
    bins
        Number of bins or explicit bin-edge array passed to each histogram.
    log
        Use log-spaced bins and a log x-axis.
    color
        Bar fill color.
    figsize_per_panel
        ``(width, height)`` in inches for each individual histogram panel.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pp.calculate_shape_metrics(mdata)
    >>> ab.pl.shape_distribution_plot(mdata).show()
    >>> ab.pl.shape_distribution_plot(mdata, prefix='nucleus_').show()
    >>> ab.pl.shape_distribution_plot(mdata, keys=['cell_area', 'cell_aspect_ratio']).show()
    """
    import math

    if keys is None:
        if prefix is not None:
            keys = [c for c in adata.obs.columns if c.startswith(prefix)]
        else:
            keys = [
                c for c in adata.obs.columns
                if any(c.endswith(f"_{s}") for s in _SHAPE_METRIC_SUFFIXES)
            ]
        if not keys:
            raise ValueError(
                "No shape metric columns found in obs. "
                "Run ab.pp.calculate_shape_metrics() first, "
                "or pass keys= or prefix= explicitly."
            )

    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, key in zip(axes_flat, keys):
        _obs_histogram(
            adata, key,
            bins=bins, log=log, color=color, ax=ax, **hist_kwargs,
        )
        ax.set_title(key.replace("_", " "))

    for ax in axes_flat[len(keys):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def shape_ratio_plot(
    adata: ad.AnnData,
    pairs: list[tuple[str, str]],
    *,
    ncols: int = 3,
    bins: int | np.ndarray = 50,
    log: bool = False,
    color: str = "#4C72B0",
    figsize_per_panel: tuple[float, float] = (4, 3),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram grid of per-cell ratios of pairs of ``obs`` columns.

    For each ``(numerator, denominator)`` pair, computes
    ``obs[numerator] / obs[denominator]`` cell-by-cell and shows its
    distribution.  Zero denominators produce ``NaN`` and are excluded.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    pairs
        List of ``(numerator_key, denominator_key)`` obs column pairs.
    ncols
        Number of histogram panels per row.
    bins
        Number of bins or explicit bin-edge array.
    log
        Use log-spaced bins and a log x-axis.
    color
        Bar fill color.
    figsize_per_panel
        ``(width, height)`` in inches per panel.
    **hist_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.hist`.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.pp.calculate_shape_metrics(mdata)
    >>> ab.pl.shape_ratio_plot(
    ...     mdata,
    ...     pairs=[
    ...         ("cell_area", "cell_perimeter"),
    ...         ("cell_span_x", "cell_span_y"),
    ...         ("cell_major_axis_length", "cell_minor_axis_length"),
    ...     ],
    ... ).show()
    """
    import math

    if not pairs:
        raise ValueError("pairs must be a non-empty list of (numerator, denominator) tuples.")

    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (num_key, den_key) in zip(axes_flat, pairs):
        for key in (num_key, den_key):
            if key not in adata.obs.columns:
                raise KeyError(
                    f"obs['{key}'] not found. "
                    "Run ab.pp.calculate_shape_metrics() first."
                )
        num = adata.obs[num_key].to_numpy(dtype=np.float64)
        den = adata.obs[den_key].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den != 0, num / den, np.nan)
        xlabel = f"{num_key} / {den_key}"
        _array_histogram(ratio, bins=bins, log=log, color=color, xlabel=xlabel, ax=ax, **hist_kwargs)
        ax.set_title(xlabel.replace("_", " "))

    for ax in axes_flat[len(pairs):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Intensity histograms (morphology / IF modalities)
# ---------------------------------------------------------------------------


def _resolve_channel(adata: ad.AnnData, channel: int | str) -> tuple[np.ndarray, str]:
    """Return ``(values, channel_name)`` for a channel index or name."""
    import scipy.sparse as sp

    if isinstance(channel, str):
        if channel not in adata.var_names:
            raise KeyError(
                f"Channel '{channel}' not found in var_names. "
                f"Available: {list(adata.var_names)}"
            )
        idx = adata.var_names.get_loc(channel)
        name = channel
    else:
        idx = int(channel)
        name = adata.var_names[idx] if idx < len(adata.var_names) else f"ch{idx}"

    col = adata.X[:, idx]
    if sp.issparse(col):
        col = np.asarray(col.todense()).ravel()
    return col.astype(np.float64), name


@with_adata
def intensity_histogram(
    adata: ad.AnnData,
    channel: int | str = 0,
    *,
    bins: int = 60,
    color: str = "#4C72B0",
    otsu: bool = True,
    vlines: list[float] | None = None,
    vline_color: str = "#D94F3D",
    log: bool = False,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 4),
) -> FigurePanel:
    """Histogram of per-cell intensities for a single image channel.

    Designed for use with the ``'morphology'`` or ``'if'`` modality whose
    ``X`` matrix contains per-cell mean pixel intensities (from
    :func:`~astrolabe.io.extract_morphology_intensities` or
    :func:`~astrolabe.pp.extract_if_intensities`).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` for the
        intensity modality (e.g. ``mdata['morphology']``).
    channel
        Channel to plot: integer index or ``var_names`` string.
    bins
        Number of histogram bins.
    color
        Bar color.
    otsu
        If ``True``, compute an Otsu threshold on the intensity values and
        overlay it as a vertical dashed line.  Useful for identifying a
        positive/negative gate.
    vlines
        Additional vertical reference lines (e.g. manual gates).
    vline_color
        Color for Otsu and *vlines* reference lines.
    log
        Use a log x-axis.
    title
        Axes title.  Defaults to the channel name.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.

    Returns
    -------
    FigurePanel
    """
    values, ch_name = _resolve_channel(adata, channel)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    finite = values[np.isfinite(values)]
    if log and isinstance(bins, int):
        pos = finite[finite > 0]
        bin_edges = (
            np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1)
            if len(pos) > 1
            else bins
        )
    else:
        bin_edges = bins

    ax.hist(finite, bins=bin_edges, color=color, edgecolor="none")

    all_vlines: list[tuple[float, str]] = []

    if otsu and len(finite) > 1:
        try:
            from skimage.filters import threshold_otsu
            thresh = float(threshold_otsu(finite))
            all_vlines.append((thresh, f"Otsu = {thresh:.1f}"))
        except Exception:
            pass

    for v in (vlines or []):
        all_vlines.append((v, f"{v:.1f}"))

    for v, label in all_vlines:
        ax.axvline(v, color=vline_color, linestyle="--", linewidth=1.2, label=label)

    if all_vlines:
        ax.legend(frameon=False, fontsize=8)

    if log:
        ax.set_xscale("log")

    ax.set_xlabel(f"Intensity — {ch_name}")
    ax.set_ylabel("Number of cells")
    ax.set_title(title or ch_name)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


@with_adata
def intensity_histograms(
    adata: ad.AnnData,
    *,
    channels: "list[int | str] | None" = None,
    bins: int = 60,
    color: str = "#4C72B0",
    otsu: bool = True,
    vlines: "list[float] | None" = None,
    vline_color: str = "#D94F3D",
    log: bool = False,
    ncols: int = 4,
    figsize_per_panel: tuple[float, float] = (4, 3),
) -> FigurePanel:
    """Grid of per-channel intensity histograms with optional Otsu gates.

    Convenience wrapper around :func:`intensity_histogram` that plots all
    channels (or a subset) in a single figure.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` for the
        intensity modality.
    channels
        List of channel indices or names to plot.  ``None`` plots all
        ``var_names`` in order.
    bins, color, otsu, vlines, vline_color, log
        Forwarded to :func:`intensity_histogram` for every panel.
    ncols
        Maximum number of columns in the grid.
    figsize_per_panel
        ``(width, height)`` in inches for each subplot.

    Returns
    -------
    FigurePanel
    """
    if channels is None:
        channels = list(range(adata.n_vars))

    n = len(channels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for i, ch in enumerate(channels):
        r, c = divmod(i, ncols)
        ax_i = axes[r][c]
        values, ch_name = _resolve_channel(adata, ch)
        finite = values[np.isfinite(values)]

        if log and isinstance(bins, int):
            pos = finite[finite > 0]
            bin_edges: Any = (
                np.logspace(np.log10(pos.min()), np.log10(pos.max()), bins + 1)
                if len(pos) > 1
                else bins
            )
        else:
            bin_edges = bins

        ax_i.hist(finite, bins=bin_edges, color=color, edgecolor="none")

        panel_vlines: list[tuple[float, str]] = []
        if otsu and len(finite) > 1:
            try:
                from skimage.filters import threshold_otsu
                thresh = float(threshold_otsu(finite))
                panel_vlines.append((thresh, f"Otsu={thresh:.1f}"))
            except Exception:
                pass
        for v in (vlines or []):
            panel_vlines.append((v, f"{v:.1f}"))
        for v, label in panel_vlines:
            ax_i.axvline(v, color=vline_color, linestyle="--", linewidth=1.0, label=label)
        if panel_vlines:
            ax_i.legend(frameon=False, fontsize=6)

        if log:
            ax_i.set_xscale("log")
        ax_i.set_title(ch_name, fontsize=9)
        ax_i.set_xlabel("Intensity", fontsize=8)
        ax_i.set_ylabel("Cells", fontsize=8)
        ax_i.tick_params(labelsize=7)
        ax_i.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax_i.set_axisbelow(True)

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# FACS-like scatter gate plot
# ---------------------------------------------------------------------------


def _density_values(xv: np.ndarray, yv: np.ndarray, bins: int) -> np.ndarray:
    """Map each (x, y) point to the count in its 2-D histogram bin."""
    h, xe, ye = np.histogram2d(xv, yv, bins=bins)
    xi = np.clip(np.searchsorted(xe, xv, side="right") - 1, 0, bins - 1)
    yi = np.clip(np.searchsorted(ye, yv, side="right") - 1, 0, bins - 1)
    return h[xi, yi]


def _otsu_or_none(values: np.ndarray) -> float | None:
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(values[np.isfinite(values)]))
    except Exception:
        return None


@with_adata
def scatter_gate(
    adata: ad.AnnData,
    x: int | str,
    y: int | str,
    *,
    gate_x: float | None = None,
    gate_y: float | None = None,
    otsu: bool = True,
    color: "int | str | None" = None,
    density_bins: int = 50,
    marginal_bins: int = 60,
    log_x: bool = False,
    log_y: bool = False,
    cmap: str = "viridis",
    point_size: float = 2.0,
    alpha: float = 0.4,
    show_counts: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> FigurePanel:
    """FACS-style scatter plot with marginal histograms and optional quadrant gates.

    Designed for use with the ``'morphology'`` or ``'if'`` modality whose
    ``X`` matrix contains per-cell intensities.  Each cell is one point;
    color encodes local density (default) or a third channel.  Marginal
    histograms are drawn on the top and right edges sharing axes with the
    central scatter.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` for the
        intensity modality.
    x, y
        Channel for the x- and y-axes: integer index or ``var_names`` string.
    gate_x, gate_y
        Explicit gate threshold values.  When supplied, quadrant dividers are
        drawn and cell counts per quadrant are annotated.
    otsu
        If ``True`` and a gate is not explicitly provided for an axis, the
        Otsu threshold is computed from that channel's distribution and used
        as the gate.
    color
        Third channel (index or name) used to color points.  ``None``
        colors by local 2-D density.
    density_bins
        Number of bins for the 2-D density histogram (ignored when *color*
        is set).
    marginal_bins
        Number of bins for the marginal histograms.
    log_x, log_y
        Use log x / y axes.
    cmap
        Colormap for density or third-channel coloring.
    point_size
        Scatter point size in points².
    alpha
        Scatter point opacity.
    show_counts
        Annotate each quadrant with its cell count and percentage.
    title
        Figure title.
    figsize
        ``(width, height)`` in inches.

    Returns
    -------
    FigurePanel
    """
    import scipy.sparse as sp
    from matplotlib.gridspec import GridSpec

    xv, x_name = _resolve_channel(adata, x)
    yv, y_name = _resolve_channel(adata, y)

    finite = np.isfinite(xv) & np.isfinite(yv)
    xf, yf = xv[finite], yv[finite]

    # ------------------------------------------------------------------ #
    # Gates                                                               #
    # ------------------------------------------------------------------ #
    if gate_x is None and otsu:
        gate_x = _otsu_or_none(xf)
    if gate_y is None and otsu:
        gate_y = _otsu_or_none(yf)

    # ------------------------------------------------------------------ #
    # Point colors                                                       #
    # ------------------------------------------------------------------ #
    if color is not None:
        cv, c_name = _resolve_channel(adata, color)
        c_scatter = cv[finite]
        norm = mcolors.Normalize(
            vmin=float(np.nanpercentile(c_scatter, 1)),
            vmax=float(np.nanpercentile(c_scatter, 99)),
        )
        scatter_c = plt.get_cmap(cmap)(norm(c_scatter))
        colorbar_label = c_name
    else:
        density = _density_values(xf, yf, density_bins)
        order = np.argsort(density)          # draw dense clusters on top
        xf, yf = xf[order], yf[order]
        density = density[order]
        norm = mcolors.Normalize(vmin=density.min(), vmax=density.max())
        scatter_c = plt.get_cmap(cmap)(norm(density))
        colorbar_label = "density"
        if color is not None:
            c_scatter = c_scatter[order]

    # ------------------------------------------------------------------ #
    # Layout: 2×2 GridSpec — top-left = x marginal, bottom-left = scatter,
    # bottom-right = y marginal, top-right = colorbar                    #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1, 4],
        width_ratios=[4, 1],
        hspace=0.05,
        wspace=0.05,
    )
    ax_top   = fig.add_subplot(gs[0, 0])   # x marginal
    ax_main  = fig.add_subplot(gs[1, 0])   # scatter
    ax_right = fig.add_subplot(gs[1, 1])   # y marginal
    ax_cb    = fig.add_subplot(gs[0, 1])   # colorbar

    # ------------------------------------------------------------------ #
    # Main scatter                                                        #
    # ------------------------------------------------------------------ #
    sc = ax_main.scatter(
        xf, yf,
        c=scatter_c,
        s=point_size,
        alpha=alpha,
        linewidths=0,
        rasterized=len(xf) > 5_000,
    )
    if log_x:
        ax_main.set_xscale("log")
    if log_y:
        ax_main.set_yscale("log")
    ax_main.set_xlabel(x_name)
    ax_main.set_ylabel(y_name)

    # ------------------------------------------------------------------ #
    # Marginals                                                           #
    # ------------------------------------------------------------------ #
    marginal_color = "#4C72B0"

    def _marginal_bins(vals, n):
        if log_x or log_y:
            pos = vals[vals > 0]
            if len(pos) > 1:
                return np.logspace(np.log10(pos.min()), np.log10(pos.max()), n + 1)
        return n

    ax_top.hist(xf, bins=_marginal_bins(xf, marginal_bins),
                color=marginal_color, edgecolor="none")
    ax_right.hist(yf, bins=_marginal_bins(yf, marginal_bins),
                  color=marginal_color, edgecolor="none",
                  orientation="horizontal")

    if log_x:
        ax_top.set_xscale("log")
    if log_y:
        ax_right.set_yscale("log")

    # Share axes with main scatter
    ax_top.sharex(ax_main)
    ax_right.sharey(ax_main)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_top.set_ylabel("count", fontsize=8)
    ax_right.set_xlabel("count", fontsize=8)
    ax_top.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax_right.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax_top.set_axisbelow(True)
    ax_right.set_axisbelow(True)

    # ------------------------------------------------------------------ #
    # Gate lines                                                          #
    # ------------------------------------------------------------------ #
    gate_kw = dict(color="#D94F3D", linestyle="--", linewidth=1.2, zorder=5)
    if gate_x is not None:
        ax_main.axvline(gate_x, **gate_kw)
        ax_top.axvline(gate_x, **gate_kw)
    if gate_y is not None:
        ax_main.axhline(gate_y, **gate_kw)
        ax_right.axhline(gate_y, **gate_kw)

    # ------------------------------------------------------------------ #
    # Quadrant counts                                                     #
    # ------------------------------------------------------------------ #
    if show_counts and gate_x is not None and gate_y is not None:
        n_total = len(xf)
        gx, gy = gate_x, gate_y
        # Q1 top-left, Q2 top-right, Q3 bottom-left, Q4 bottom-right
        quads = {
            "Q3\n(x−, y−)": (~(xf >= gx)) & (~(yf >= gy)),
            "Q4\n(x+, y−)":  (xf >= gx)   & (~(yf >= gy)),
            "Q1\n(x−, y+)": (~(xf >= gx)) &  (yf >= gy),
            "Q2\n(x+, y+)":  (xf >= gx)   &  (yf >= gy),
        }
        # Corner positions as (ha, va, x_rel, y_rel) using axis fraction
        corners = [
            ("left",  "bottom", 0.02, 0.02),   # Q3 bottom-left
            ("right", "bottom", 0.98, 0.02),   # Q4 bottom-right
            ("left",  "top",    0.02, 0.98),   # Q1 top-left
            ("right", "top",    0.98, 0.98),   # Q2 top-right
        ]
        for (label, mask), (ha, va, xr, yr) in zip(quads.items(), corners):
            n = int(mask.sum())
            pct = n / n_total * 100 if n_total else 0.0
            ax_main.text(
                xr, yr, f"{n:,}\n({pct:.1f}%)",
                transform=ax_main.transAxes,
                ha=ha, va=va, fontsize=7,
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
            )

    # ------------------------------------------------------------------ #
    # Colorbar                                                            #
    # ------------------------------------------------------------------ #
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cb.set_label(colorbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, y=1.01, fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Moran scatter plot
# ---------------------------------------------------------------------------

_MORAN_QUAD_PALETTE: dict[str, str] = {
    "HH": "#D94F3D",
    "LL": "#6A7FC1",
    "HL": "#F4883B",
    "LH": "#5BAD6F",
}


@with_adata
def moran_plot(
    adata: ad.AnnData,
    key: str,
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    moran_key: str | None = "moran",
    lisa_key: str | None = "moran_local",
    palette: dict[str, str] | None = None,
    point_size: float = 4.0,
    alpha: float = 0.4,
    title: str | None = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (5, 5),
) -> FigurePanel:
    """Moran scatter plot: standardised values vs. their spatial lag.

    The slope of the fit line through the origin equals Moran's I.  Points
    are optionally colored by LISA quadrant (HH / LL / HL / LH) when
    pre-computed local results are available in ``uns``.

    Parameters
    ----------
    key
        Gene name or ``obs`` column to visualize.
    layer
        Expression layer for gene keys.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    moran_key
        ``uns`` key of pre-computed :func:`~astrolabe.tl.moran` results;
        used to read the global I value.  ``None`` computes it on the fly.
    lisa_key
        ``uns`` key of pre-computed :func:`~astrolabe.tl.moran_local`
        results for quadrant coloring.  ``None`` = no coloring.
    palette
        Quadrant color overrides (keys: ``'HH'``, ``'LL'``, ``'HL'``,
        ``'LH'``).
    point_size, alpha
        Scatter marker size and opacity.
    title
        Axes title; defaults to ``'Moran plot — <key>'``.
    ax
        Existing axes to draw into.  ``None`` creates a new figure.
    figsize
        Figure size when creating a new figure.

    Returns
    -------
    :class:`~astrolabe.pl.FigurePanel`

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> ab.tl.moran(mdata, "EPCAM")
    >>> ab.tl.moran_local(mdata, "EPCAM")
    >>> ab.pl.moran_plot(mdata, "EPCAM")
    """
    import scipy.sparse as _sp

    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize)
    else:
        ax_plot = ax
        fig = ax.get_figure()

    # ------------------------------------------------------------------ #
    # Resolve values and spatial lag                                      #
    # ------------------------------------------------------------------ #
    if key in adata.var_names:
        idx = adata.var_names.get_loc(key)
        mat = adata.layers[layer] if layer is not None else adata.X
        col = mat[:, idx]
        if _sp.issparse(col):
            col = col.toarray().ravel()
        y = np.asarray(col, dtype=np.float64)
    elif key in adata.obs.columns:
        y = adata.obs[key].to_numpy(dtype=np.float64)
    else:
        raise KeyError(f"Key '{key}' not found in var_names or obs columns.")

    std = y.std()
    z = (y - y.mean()) / std if std > 0 else np.zeros_like(y)

    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors() first"
        )
    W = adata.obsp[conn_key]
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W_norm = _sp.diags(1.0 / row_sums) @ W
    lag_z = np.asarray(W_norm @ z).ravel()

    # ------------------------------------------------------------------ #
    # Point colors (LISA quadrant)                                       #
    # ------------------------------------------------------------------ #
    pal = {**_MORAN_QUAD_PALETTE, **(palette or {})}
    no_quad_color = "#AAAAAA"

    quad_arr = None
    if lisa_key is not None and lisa_key in adata.uns:
        lisa_df = adata.uns[lisa_key]
        try:
            quad_arr = lisa_df.loc[key]["quadrant"].to_numpy(dtype=str)
        except (KeyError, TypeError):
            pass

    colors = [pal.get(q, no_quad_color) for q in quad_arr] if quad_arr is not None else no_quad_color
    ax_plot.scatter(z, lag_z, c=colors, s=point_size, alpha=alpha, linewidths=0, rasterized=True)

    # ------------------------------------------------------------------ #
    # Moran's I slope line                                                #
    # ------------------------------------------------------------------ #
    slope: float | None = None
    if moran_key is not None and moran_key in adata.uns:
        try:
            slope = float(adata.uns[moran_key].loc[key, "I"])
        except (KeyError, TypeError):
            pass

    if slope is None:
        var_z = float(z.var())
        slope = float(np.cov(z, lag_z)[0, 1] / var_z) if var_z > 0 else 0.0

    x_line = np.array([z.min(), z.max()])
    ax_plot.plot(x_line, slope * x_line, color="#333333", linewidth=1.2, zorder=5)

    # ------------------------------------------------------------------ #
    # Reference lines and legend                                          #
    # ------------------------------------------------------------------ #
    ax_plot.axhline(0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)
    ax_plot.axvline(0, color="#CCCCCC", linewidth=0.8, linestyle="--", zorder=1)

    handles = [plt.Line2D([0], [0], color="#333333", linewidth=1.2, label=f"I = {slope:.3f}")]
    if quad_arr is not None:
        for quad, color in pal.items():
            if quad in quad_arr:
                handles.append(mpatches.Patch(facecolor=color, label=quad))
    ax_plot.legend(handles=handles, fontsize=7, framealpha=0.7)

    ax_plot.set_xlabel(f"z({key})", fontsize=9)
    ax_plot.set_ylabel(f"Spatial lag of z({key})", fontsize=9)
    ax_plot.set_title(title or f"Moran plot — {key}", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Neighborhood analysis companion plots
# ---------------------------------------------------------------------------

_DIVNORM_CMAP = "RdBu_r"


@with_adata
def nhood_enrichment_plot(
    adata: ad.AnnData,
    groupby: str,
    *,
    key: str = "nhood_enrichment",
    metric: str = "zscore",
    cmap: str = _DIVNORM_CMAP,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> FigurePanel:
    """Annotated heatmap of neighborhood enrichment z-scores.

    Visualizes the result of :func:`~astrolabe.tl.nhood_enrichment`.
    Warm colors = enriched co-localization; cool colors = avoidance.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column originally passed to
        :func:`~astrolabe.tl.nhood_enrichment`.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.nhood_enrichment`.
    metric
        ``'zscore'`` (default) or ``'counts'``.
    cmap
        Diverging colormap for the heatmap.  Default ``"RdBu_r"``.
    vmax
        Color-scale maximum.  Defaults to ``max(|values|)``.
    figsize
        Figure size.
    title
        Axes title.
    ax
        Existing axes to draw into.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.nhood_enrichment() first")

    result = adata.uns[key]
    groups = result["groups"]
    mat = result[metric].astype(float)
    n = len(groups)

    if figsize is None:
        figsize = (max(4.0, n * 0.7 + 1.5), max(3.5, n * 0.7 + 1.0))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    v = float(vmax) if vmax is not None else float(np.abs(mat).max())
    if v == 0:
        v = 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")

    fs = max(5, min(9, 80 // max(n, 1)))
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:.1f}",
                ha="center", va="center", fontsize=fs,
                color="white" if abs(val) > 0.6 * v else "black",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title(title or f"Neighborhood enrichment ({metric})", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label=metric)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def co_occurrence_plot(
    adata: ad.AnnData,
    groupby: str,
    *,
    key: str = "co_occurrence",
    source_groups: list[str] | None = None,
    target_groups: list[str] | None = None,
    cmap: str = "tab20",
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> FigurePanel:
    """Line plot of co-occurrence probability ratio vs. distance.

    Visualizes the result of :func:`~astrolabe.tl.co_occurrence`.
    Dashed reference line at ratio = 1 marks random co-occurrence.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column originally passed to
        :func:`~astrolabe.tl.co_occurrence`.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.co_occurrence`.
    source_groups
        Source cell types to show.  ``None`` shows all.
    target_groups
        Target cell types to show.  ``None`` shows all.
    cmap
        Colormap name or instance used to cycle line colors.  Default ``"tab20"``.
    figsize
        Figure size.
    title
        Axes title.
    ax
        Existing axes.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.co_occurrence() first")

    result = adata.uns[key]
    occ = result["occ"]
    interval = result["interval"]
    groups = result["groups"]

    src_idx = [groups.index(g) for g in source_groups] if source_groups is not None else range(len(groups))
    tgt_idx = [groups.index(g) for g in target_groups] if target_groups is not None else range(len(groups))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap)
    n_colors = cmap_obj.N if hasattr(cmap_obj, "N") else 20
    for line_i, (a, b) in enumerate([(a, b) for a in src_idx for b in tgt_idx]):
        ax.plot(interval, occ[a, b], label=f"{groups[a]} → {groups[b]}", color=cmap_obj(line_i % n_colors))

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Co-occurrence ratio")
    ax.set_title(title or "Co-occurrence probability ratio", fontsize=10)
    n_lines = len(list(src_idx)) * len(list(tgt_idx))
    if n_lines <= 15:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.7)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def centrality_scores_plot(
    adata: ad.AnnData,
    groupby: str,
    *,
    key: str = "centrality_scores",
    metrics: list[str] | None = None,
    cmap: str = "tab10",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Horizontal bar charts of per-cluster centrality metrics.

    Visualizes the result of :func:`~astrolabe.tl.centrality_scores`.
    Each non-NaN metric gets its own subplot.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column passed to :func:`~astrolabe.tl.centrality_scores`.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.centrality_scores`.
    metrics
        Columns to plot.  ``None`` uses all non-NaN columns.
    cmap
        Colormap name or instance for bar colors.  Default ``"tab10"``.
    figsize
        Figure size.
    title
        Figure suptitle.
    """
    if key not in adata.uns:
        raise KeyError(
            f"uns['{key}'] not found. Run ab.tl.centrality_scores() first"
        )

    df: pd.DataFrame = adata.uns[key]
    if metrics is None:
        metrics = [c for c in df.columns if not df[c].isna().all()]

    n_metrics = len(metrics)
    n_groups = len(df)
    if figsize is None:
        figsize = (4.0 * n_metrics, max(3.0, n_groups * 0.35 + 1.0))

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=True)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, n_groups))
    for ax, metric in zip(axes, metrics):
        vals = df[metric].fillna(0).to_numpy()
        cats = df.index.tolist()
        ax.barh(cats, vals, color=colors)
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=8)
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def expression_by_distance_plot(
    adata: ad.AnnData,
    *genes: str,
    key: str = "expression_by_distance",
    cmap: str = "tab10",
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> FigurePanel:
    """Expression gradient with ±SE ribbon vs. distance to anchor.

    Visualizes the result of :func:`~astrolabe.tl.expression_by_distance`.
    One line per gene with a shaded ±1 SE band.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    *genes
        Gene names to plot.  Required — pass at least one.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key
        ``uns`` key written by
        :func:`~astrolabe.tl.expression_by_distance`.
    cmap
        Colormap name or instance used to cycle line colors.  Default ``"tab10"``.
    figsize
        Figure size.
    title
        Axes title.
    ax
        Existing axes.
    """
    if key not in adata.uns:
        raise KeyError(
            f"uns['{key}'] not found. Run ab.tl.expression_by_distance() first"
        )

    result = adata.uns[key]
    mean_df: pd.DataFrame = result["mean"]
    se_df: pd.DataFrame = result["se"]
    anchor_groups = result.get("anchor_groups", [])
    anchor_key = result.get("anchor_key", "")

    plot_genes = list(genes) if genes else mean_df.columns.tolist()
    missing = [g for g in plot_genes if g not in mean_df.columns]
    if missing:
        raise KeyError(f"Genes not in result: {missing}")

    distances = mean_df.index.to_numpy(dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap)
    for i, gene in enumerate(plot_genes):
        mean = mean_df[gene].to_numpy(dtype=float)
        se = se_df[gene].to_numpy(dtype=float)
        color = cmap_obj(i % 10)
        ax.plot(distances, mean, label=gene, color=color)
        ax.fill_between(distances, mean - se, mean + se, alpha=0.2, color=color)

    ax.set_xlabel(f"Distance to {anchor_key}={anchor_groups} (µm)")
    ax.set_ylabel("Mean expression")
    ax.set_title(title or "Expression by distance", fontsize=10)
    if len(plot_genes) <= 12:
        ax.legend(fontsize=8, framealpha=0.7)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def ligrec_plot(
    adata: ad.AnnData,
    *,
    key: str = "ligrec",
    sender: str | list[str] | None = None,
    receiver: str | list[str] | None = None,
    p_threshold: float = 0.05,
    top_n: int = 50,
    cmap: str = "Reds",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Dot plot of ligand-receptor interaction scores.

    Visualizes the result of :func:`~astrolabe.tl.ligrec`.  Columns are
    (sender, receiver) cell-type pairs; rows are LR pairs with at least one
    significant interaction.  Dot size encodes mean LR product score; color
    encodes -log₁₀ p-value.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is :class:`~mudata.MuData`.
    key
        ``uns`` key written by :func:`~astrolabe.tl.ligrec`.
    sender
        Sender cell type(s) to show.  ``None`` shows all.
    receiver
        Receiver cell type(s) to show.  ``None`` shows all.
    p_threshold
        Only LR pairs with at least one p-value below this are shown.
    top_n
        Maximum rows to display (ranked by minimum p-value).
    cmap
        Colormap for dot color (encodes −log₁₀ p-value).  Default ``"Reds"``.
    figsize
        Figure size.  Auto-sized if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.ligrec() first")

    result = adata.uns[key]
    means_df: pd.DataFrame = result["means"]
    pval_df: pd.DataFrame = result["pvalues"]

    ct_pairs = list(pval_df.columns)
    if sender is not None:
        senders = [sender] if isinstance(sender, str) else sender
        ct_pairs = [p for p in ct_pairs if p[0] in senders]
    if receiver is not None:
        receivers = [receiver] if isinstance(receiver, str) else receiver
        ct_pairs = [p for p in ct_pairs if p[1] in receivers]

    if not ct_pairs:
        raise ValueError("No cell-type pairs remain after filtering")

    means_sub = means_df[ct_pairs]
    pval_sub = pval_df[ct_pairs]

    sig_mask = (pval_sub < p_threshold).any(axis=1)
    if not sig_mask.any():
        raise ValueError(
            f"No LR pairs significant at p < {p_threshold}. "
            "Try a higher p_threshold or increase n_perms"
        )

    min_pval = pval_sub[sig_mask].min(axis=1)
    top_idx = min_pval.nsmallest(min(top_n, sig_mask.sum())).index
    means_plot = means_sub.loc[top_idx]
    pval_plot = pval_sub.loc[top_idx]

    n_rows, n_cols = means_plot.shape
    col_labels = [f"{s}|{r}" for s, r in ct_pairs]
    row_labels = [f"{lig}:{rec}" for lig, rec in means_plot.index]

    if figsize is None:
        figsize = (max(4.0, n_cols * 0.8 + 2.0), max(3.0, n_rows * 0.35 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    score_vals = means_plot.values.astype(float)
    max_score = float(score_vals.max()) or 1.0
    log_pval = -np.log10(pval_plot.values.astype(float) + 1e-10)
    max_log_pval = float(log_pval.max()) or 1.0

    xs, ys, sizes, colors = [], [], [], []
    for j in range(n_cols):
        for i in range(n_rows):
            xs.append(j)
            ys.append(i)
            sizes.append(300 * score_vals[i, j] / max_score)
            colors.append(log_pval[i, j] / max_log_pval)

    sc_plot = ax.scatter(
        xs, ys,
        s=sizes, c=colors,
        cmap=cmap, vmin=0, vmax=1,
        edgecolors="grey", linewidths=0.3, alpha=0.9,
    )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Sender | Receiver", fontsize=8)
    ax.set_ylabel("Ligand : Receptor", fontsize=8)
    ax.set_title(title or "Ligand–receptor interactions", fontsize=10)

    cbar = fig.colorbar(sc_plot, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("-log₁₀(p)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    for frac, label in [(0.25, "25 %"), (0.75, "75 %")]:
        ax.scatter([], [], s=300 * frac, c="grey", alpha=0.6, label=f"score {label}")
    ax.legend(
        title="Score (rel.)", fontsize=6, title_fontsize=6,
        loc="upper left", bbox_to_anchor=(1.12, 1.0), framealpha=0.7,
    )

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_umatrix(
    adata: ad.AnnData,
    *,
    key: str = "som",
    cmap: str = "viridis",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Plot the SOM unified distance matrix (U-matrix).

    High U-matrix values indicate cluster boundaries (nodes whose weights are
    dissimilar to their neighbors); low values mark cluster centers. Use this
    to identify natural cluster structure in the SOM without external labels.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with SOM results
        from :func:`~astrolabe.tl.som`.
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key written by :func:`~astrolabe.tl.som`.
    cmap
        Colormap. Default ``'viridis'``.
    annotate
        Overlay ``"{col}_{row}"`` node labels on each cell.
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")

    umatrix: np.ndarray = adata.uns[key]["umatrix"]
    sx, sy = umatrix.shape

    if figsize is None:
        figsize = (max(4.0, sy * 0.6), max(3.5, sx * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(umatrix.T, cmap=cmap, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="U-matrix distance")

    if annotate:
        for xi in range(sx):
            for yi in range(sy):
                ax.text(
                    xi, yi, f"{xi}_{yi}",
                    ha="center", va="center", fontsize=6, color="white",
                    fontweight="bold",
                )

    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.set_xticks(range(sx))
    ax.set_yticks(range(sy))
    ax.set_title(title or "SOM U-matrix", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_component_plane(
    adata: ad.AnnData,
    gene: str,
    *,
    key: str = "som",
    cmap: str = "viridis",
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Plot a SOM component plane for one gene.

    Shows how strongly each SOM node is activated by a single gene — the weight
    value for that gene across all nodes. Nodes where the gene has high weight
    are those whose prototype vector includes high expression of that gene.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with SOM results
        from :func:`~astrolabe.tl.som` (trained on a representation that
        includes gene expression, e.g. ``use_rep=None``).
    modality
        Modality key when *data* is a MuData.
    gene
        Gene name. Must be in ``adata.var_names``.
    key
        ``uns`` key written by :func:`~astrolabe.tl.som`.
    cmap
        Colormap. Default ``'viridis'``.
    annotate
        Overlay node labels.
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")
    if gene not in adata.var_names:
        raise KeyError(f"'{gene}' not found in var_names")

    weights: np.ndarray = adata.uns[key]["weights"]
    sx, sy, n_feat = weights.shape

    if n_feat != adata.n_vars:
        raise ValueError(
            f"SOM was trained on {n_feat} features but adata has {adata.n_vars} vars. "
            "Component planes require the SOM to be trained with use_rep=None."
        )

    gene_idx = adata.var_names.get_loc(gene)
    plane = weights[:, :, gene_idx]

    if figsize is None:
        figsize = (max(4.0, sy * 0.6), max(3.5, sx * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plane.T, cmap=cmap, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Weight")

    if annotate:
        for xi in range(sx):
            for yi in range(sy):
                ax.text(
                    xi, yi, f"{xi}_{yi}",
                    ha="center", va="center", fontsize=6, color="white",
                )

    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.set_xticks(range(sx))
    ax.set_yticks(range(sy))
    ax.set_title(title or f"Component plane: {gene}", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_hits(
    adata: ad.AnnData,
    *,
    key: str = "som",
    cmap: str = "Blues",
    annotate: bool = True,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Plot the SOM hit-count map (cells per node).

    Shows how many cells map to each node. Dense nodes are common expression
    states; sparse nodes are rare or transitional. Used alongside the U-matrix
    to distinguish cluster boundaries from low-density transition zones.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with SOM results
        from :func:`~astrolabe.tl.som`.
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key written by :func:`~astrolabe.tl.som`.
    cmap
        Colormap. Default ``'Blues'``.
    annotate
        Overlay cell count in each node.
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")

    hit_counts: np.ndarray = adata.uns[key]["hit_counts"]
    sx, sy = hit_counts.shape

    if figsize is None:
        figsize = (max(4.0, sy * 0.6), max(3.5, sx * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(hit_counts.T, cmap=cmap, origin="upper", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cell count")

    if annotate:
        for xi in range(sx):
            for yi in range(sy):
                val = int(hit_counts[xi, yi])
                color = "white" if hit_counts[xi, yi] > hit_counts.max() * 0.5 else "black"
                ax.text(
                    xi, yi, str(val),
                    ha="center", va="center", fontsize=6, color=color,
                )

    ax.set_xlabel("SOM column")
    ax.set_ylabel("SOM row")
    ax.set_xticks(range(sx))
    ax.set_yticks(range(sy))
    ax.set_title(title or "SOM hit-count map", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_subcellular_plot(
    adata: ad.AnnData,
    transcript_domains: pd.DataFrame,
    *,
    max_cells: int = 200,
    key: str = "som_subcellular",
    palette: dict | None = None,
    cell_boundaries_key: str = "cell_boundaries",
    spatial_key: str = "spatial",
    point_size: float = 2.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Render subcellular domain patterns by drawing transcript positions within cell outlines.

    For each sampled cell, draws the boundary polygon as a thin outline and
    scatters its transcript positions colored by their SOM domain label. The
    result shows whether the SOM domains correspond to recognisable intracellular
    structures (nuclear regions, peripheral rings, polar accumulations, etc.).

    For a tissue-level spatial overview of dominant domains, use
    ``ab.pl.points(mdata, color='som_subcellular_dominant')`` or
    ``ab.pl.boundaries(mdata, color='som_subcellular_dominant')`` instead.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with subcellular
        SOM results from :func:`~astrolabe.tl.som_subcellular_domains`.
    transcript_domains
        DataFrame returned by :func:`~astrolabe.tl.som_subcellular_domains`
        with columns ``cell_id``, ``x``, ``y``, ``gene``, ``domain``.
    modality
        Modality key when *data* is a MuData.
    max_cells
        Maximum number of cells to render (random sample, reproducible).
    key
        ``uns`` key written by :func:`~astrolabe.tl.som_subcellular_domains`.
    palette
        Dict mapping domain label → color. Auto-assigned from ``tab20`` if
        ``None``.
    cell_boundaries_key
        ``obsm`` key holding cell boundary polygons (awkward array).
    spatial_key
        ``obsm`` key for cell centroids.
    point_size
        Scatter point size in points².
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    import awkward as ak
    import matplotlib.patches as mp_patches
    import matplotlib.collections as mc

    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som_subcellular_domains() first")
    if cell_boundaries_key not in adata.obsm:
        raise KeyError(f"obsm['{cell_boundaries_key}'] not found")

    domain_names: list[str] = adata.uns[key]["domain_names"]
    n_domains = len(domain_names)

    if palette is None:
        cmap_tab = plt.get_cmap("tab20")
        palette = {d: cmap_tab(i % 20) for i, d in enumerate(domain_names)}

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    boundaries_ak = adata.obsm[cell_boundaries_key]

    cells_with_tx = transcript_domains["cell_id"].astype(str).unique()
    cells_with_tx = [c for c in cells_with_tx if c in obs_index]
    rng = np.random.default_rng(0)
    sampled = rng.choice(
        cells_with_tx,
        size=min(max_cells, len(cells_with_tx)),
        replace=False,
    )
    sampled_set = set(sampled.tolist())

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    cell_indices = [obs_index[c] for c in sampled]
    cx_vals = centroids[cell_indices, 0]
    cy_vals = centroids[cell_indices, 1]
    pad = 20.0
    xlim = (float(cx_vals.min()) - pad, float(cx_vals.max()) + pad)
    ylim = (float(cy_vals.min()) - pad, float(cy_vals.max()) + pad)

    if figsize is None:
        w = max(6.0, min(12.0, (xlim[1] - xlim[0]) / 50))
        h = max(5.0, min(10.0, (ylim[1] - ylim[0]) / 50))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    poly_patches = []
    for cid in sampled:
        ci = obs_index[cid]
        try:
            verts = np.asarray(ak.to_regular(boundaries_ak[ci]))
        except Exception:
            try:
                verts = np.asarray(boundaries_ak[ci])
            except Exception:
                continue
        if verts.ndim == 2 and verts.shape[1] == 2:
            poly_patches.append(mp_patches.Polygon(verts, closed=True, fill=False))

    if poly_patches:
        ax.add_collection(mc.PatchCollection(
            poly_patches, match_original=True,
            edgecolors="gray", linewidths=0.5, facecolors="none",
        ))

    tx_sub = transcript_domains[transcript_domains["cell_id"].astype(str).isin(sampled_set)]
    for domain in domain_names:
        mask = tx_sub["domain"] == domain
        if mask.any():
            sub = tx_sub[mask]
            ax.scatter(
                sub["x"].to_numpy(dtype=np.float32),
                sub["y"].to_numpy(dtype=np.float32),
                s=point_size, c=[palette[domain]], label=domain,
                linewidths=0, alpha=0.7, zorder=2,
            )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(title or f"Subcellular domains ({len(sampled)} cells sampled)", fontsize=10)

    if n_domains <= 20:
        ax.legend(
            title="Domain", fontsize=6, title_fontsize=7,
            loc="upper right", framealpha=0.7, markerscale=2.0,
        )

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)