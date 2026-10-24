"""
QC and expression distribution plots.
"""

from __future__ import annotations

from astrolabe._dependencies import _get_result

import logging
from typing import Any

import anndata as ad
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from astrolabe._utils import _resolve_field, with_adata
from astrolabe.pl._generic import histogram as _generic_histogram, scatter as _generic_scatter
from astrolabe.pl._spatial import _DEFAULT_CONTINUOUS_CMAP, _DEFAULT_DIVERGING_CMAP, FigurePanel
from astrolabe.pl._theme import (
    TITLE_FONT,
    _extended_cat_colors,
    _legend_handles,
    _oklch_to_hex,
    _style_axes,
    _style_colorbar,
    _style_legend,
    get_theme,
)

__all__ = [
    "violin",
    "dotplot",
    "expression_heatmap",
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
    "genes_vs_area_scatter",
    "transcripts_vs_area_scatter",
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
    "som_quality_plot",
    "som_stability_plot",
]

log = logging.getLogger(__name__)


def _feature_type_palette() -> dict[str, str]:
    """Default colors for known feature types, from the active theme's core palette.

    Gene Expression (the expected/background category) gets the theme's
    muted gray so controls read as the visually distinct ones.

    CosMx/MERSCOPE control labels share the hue of their Xenium conceptual
    analog, so the same failure mode is the same color on every platform:
    nonspecific-binding probes (Negative Probe ↔ Negative Control Probe),
    decoding-error codes (System Control ↔ Negative Control Codeword), and
    probe-less codewords (Blank Barcode ↔ Unassigned Codeword). The platforms
    never co-occur in one plot, so reuse costs nothing within a panel.
    """
    t = get_theme()
    return {
        "Gene Expression": t.muted,
        "Negative Control Probe": t.cat_core[7],
        "Negative Control Codeword": t.cat_core[1],
        "Unassigned Codeword": t.cat_core[0],
        "Deprecated Codeword": t.cat_core[2],
        "Genomic Control": t.cat_core[6],
        # CosMx (read_cosmx)
        "Negative Probe": t.cat_core[7],
        "System Control": t.cat_core[1],
        # MERSCOPE (read_merscope)
        "Blank Barcode": t.cat_core[0],
    }


def _categorical_colors(cmap: str | None, n: int) -> list:
    """Resolve *n* distinguishable categorical colors for a line/bar-per-category plot.

    Mirrors `_build_cat_colors`'s (`pl/_spatial.py`) >8-category fallback for
    astrolabe's own default theme palette (`cmap=None`): naively indexing the
    8-hue `cat_core` with `i % 8` silently collides two categories onto one
    color past 8 groups (routine with e.g. 9-10 Leiden clusters), so this
    switches to the OKLCH-hue-sweep `_extended_cat_colors` ramp instead once
    `n > 8`. An explicitly named `cmap` is respected as-is and still cycled
    via modulo -- that's the user's own colormap choice, not astrolabe's
    default palette, so the >8 fallback doesn't apply.
    """
    if cmap is None:
        t = get_theme()
        return list(t.cat_core if n <= 8 else _extended_cat_colors(n, t.name))
    cmap_obj = plt.get_cmap(cmap)
    n_colors = cmap_obj.N if hasattr(cmap_obj, "N") else 10
    return [cmap_obj(i % n_colors) for i in range(n)]


def _validate_groupby_against_result(
    adata: ad.AnnData,
    groupby: str,
    stored_groups,
    func_name: str,
) -> None:
    """Warn when *groupby* doesn't plausibly describe a stored result.

    The neighborhood-analysis plot functions take ``groupby`` for labeling
    only — the values plotted come from the stored ``uns`` result. A wrong
    ``groupby`` therefore silently mislabels the figure. This checks that
    the obs column exists and that its categories cover the stored groups,
    logging a warning (never raising) on mismatch.
    """
    if groupby not in adata.obs.columns:
        log.warning(
            "%s: obs column '%s' not found — groupby only labels the figure "
            "here, but make sure it names the column the stored result was "
            "computed from.",
            func_name, groupby,
        )
        return
    if stored_groups is None:
        return
    obs_groups = set(map(str, pd.unique(adata.obs[groupby])))
    missing = [g for g in map(str, stored_groups) if g not in obs_groups]
    if missing:
        log.warning(
            "%s: stored result contains group(s) %s not present in "
            "obs['%s'] — was the result computed with a different groupby?",
            func_name, missing, groupby,
        )


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
    panel = FigurePanel.from_obj(result)
    if stripplot and adata.n_obs > 5_000:
        # Density-scaled jitter transparency: at 10^5 cells the solid-black
        # stripplot dots fully occlude the violin bodies for large groups.
        # Scale alpha down with cell count (floor 0.02) so dense groups read
        # as a translucent cloud instead of a black bar.
        jitter_alpha = float(np.clip(5_000.0 / adata.n_obs, 0.02, 1.0))
        for ax_ in panel.figure.axes:
            for coll in ax_.collections:
                if isinstance(coll, matplotlib.collections.PathCollection):
                    coll.set_alpha(jitter_alpha)
    for ax_ in panel.figure.axes:
        _style_axes(ax_)
    if ax is None:
        plt.close(panel.figure)
    return panel


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
        Only used when the *groupby* column is **not** categorical: the
        continuous column is subdivided into this many bins (scanpy
        semantics).  Ignored for categorical columns.
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

    panel = FigurePanel.from_obj(dp)
    for ax_ in panel.figure.axes:
        _style_axes(ax_)
    plt.close(panel.figure)
    return panel


@with_adata
def expression_heatmap(
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
        Only used when the *groupby* column is **not** categorical: the
        continuous column is subdivided into this many bins (scanpy
        semantics).  Ignored for categorical columns.
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

            ab.pl.expression_heatmap(mdata, markers, "leiden") | ab.pl.points(mdata, "leiden")

    Examples
    --------
    >>> markers = {"T cell": ["CD3D", "CD8A"], "Epithelial": ["EPCAM", "KRT8"]}
    >>> ab.pl.expression_heatmap(mdata, markers, groupby="leiden_annotation").show()
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

    panel = FigurePanel.from_obj(plt.gcf())
    for ax_ in panel.figure.axes:
        _style_axes(ax_)
    plt.close(panel.figure)
    return panel


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
    """Scatter plot of a 2-D embedding (UMAP, PCA, a cospatial embedding, etc.).

    Wraps :func:`scanpy.pl.embedding` and returns a composable
    :class:`~astrolabe.pl.FigurePanel`.  When multiple ``color`` values are
    requested, scanpy draws one sub-panel per color.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    obsm_key
        Key in ``obsm`` to use as the 2-D coordinates (e.g. ``'X_umap'``,
        ``'X_pca'``, or a cospatial embedding from
        ``tl.blend_neighbors`` + ``tl.umap``).  The ``'X_'`` prefix is
        stripped when passing to scanpy's ``basis`` argument.
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

    # Default categorical colors must match pl.points()' category→color
    # assignment (theme palette over natural-sorted categories) — otherwise a
    # composed `embedding_scatter(...) | points(...)` figure colors the SAME
    # cluster differently in its two panels, defeating cross-referencing.
    if palette is None:
        from astrolabe.pl._spatial import _build_cat_colors, _sorted_categories

        _color_keys = color if isinstance(color, (list, tuple)) else [color]
        _pal: dict[str, Any] = {}
        for _ck in _color_keys:
            if _ck is None or _ck not in adata.obs.columns:
                continue
            _col = adata.obs[_ck]
            if not isinstance(_col.dtype, pd.CategoricalDtype):
                continue
            _cats = _sorted_categories(list(_col.cat.categories))
            _pal.update(_build_cat_colors(_cats, None, "lightgray"))
        if _pal:
            palette = _pal

    # sc.pl.embedding has no figsize param in current scanpy; create the
    # axes ourselves so figsize is honored instead of leaking into **kwargs
    # (where it would reach Axes.scatter() and raise an AttributeError).
    # Exception: a multi-entry color list makes scanpy draw one sub-panel
    # per color, which a single pre-created axes cannot hold -- let scanpy
    # build its own multi-panel figure then (figsize is ignored there).
    multi_color = isinstance(color, (list, tuple)) and len(color) > 1
    made_ax = ax is None
    if made_ax and multi_color:
        if figsize is not None:
            log.debug(
                "embedding_scatter: figsize is ignored when color is a list "
                "(scanpy sizes its own multi-panel figure)."
            )
        result = sc.pl.embedding(
            adata,
            basis=basis,
            color=color,
            title=title,
            palette=palette,
            **({"size": size} if size is not None else {}),
            alpha=alpha,
            frameon=frameon,
            show=False,
            **kwargs,
        )
        fig = FigurePanel.from_obj(result).figure
        for ax_ in fig.axes:
            _style_axes(ax_)
        plt.close(fig)
        return FigurePanel(fig)

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
    for ax_ in fig.axes:
        _style_axes(ax_)
    if made_ax:
        # Only close a figure this function itself created -- closing a
        # caller-supplied axes' figure mid-composition breaks the caller.
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
    up_color: str | None = None,
    down_color: str | None = None,
    ns_color: str | None = None,
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
        ``'pvals'``.  A :class:`KeyError` is raised if the column is missing
        from the stored results — raw p-values are never substituted
        silently.
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

    rgg = _get_result(adata, key)
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

    if len(genes) < adata.n_vars:
        log.warning(
            "volcano_plot: uns['%s'] holds %d genes but adata has %d — the "
            "volcano is truncated (tl.rank_genes_groups was run with a "
            "limited n_genes). Re-run with n_genes=adata.n_vars for a full "
            "volcano.",
            key, len(genes), adata.n_vars,
        )

    if pval_col not in rgg:
        raise KeyError(
            f"'{pval_col}' not found in uns['{key}'] "
            f"(available: {sorted(k for k in rgg if not k.startswith('_'))}). "
            "Multiple-testing-adjusted p-values are the default significance "
            "source; pass pval_col='pvals' explicitly to plot raw p-values "
            "instead."
        )
    pvals_raw = np.array(rgg[pval_col][group], dtype=np.float64)
    pvals_raw = np.clip(pvals_raw, 1e-300, 1.0)
    neg_log_p = -np.log10(pvals_raw)

    t = get_theme()
    up_color = up_color if up_color is not None else t.cat_core[7]
    down_color = down_color if down_color is not None else t.cat_core[0]
    ns_color = ns_color if ns_color is not None else t.muted

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
    ax.axhline(-np.log10(pval_threshold), color=t.muted, linestyle="--", linewidth=0.9)
    ax.axvline( lfc_threshold, color=t.muted, linestyle="--", linewidth=0.9)
    ax.axvline(-lfc_threshold, color=t.muted, linestyle="--", linewidth=0.9)

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
    ylabel = "−log₁₀(adjusted p)" if pval_col == "pvals_adj" else f"−log₁₀({pval_col})"
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title or f"Volcano — group {group}", fontsize=10)

    handles = _legend_handles(
        [f"Up (n={sig_up.sum()})", f"Down (n={sig_down.sum()})", "NS"],
        [up_color, down_color, ns_color],
    )
    _style_legend(ax.legend(handles=handles, fontsize=7, framealpha=0.7))

    _style_axes(ax)
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
    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def ripley_plot(
    adata: ad.AnnData,
    *,
    key: str = "ripley_l",
    cell_types: "list[str] | None" = None,
    cmap: str | None = None,
    title: "str | None" = None,
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: "tuple[float, float]" = (6, 4),
) -> FigurePanel:
    """Ripley's L function plot (L(d) − d vs. distance).

    Visualizes the output of :func:`~astrolabe.tl.ripley_l`.  Each cell
    type's stored CSR expectation ``L_exp`` (the simulated null mean, which
    bows below zero at large *d* because the estimator applies no edge
    correction) is drawn as a dashed reference in the same color — read
    clustering/dispersion as ``L_obs`` above/below *that* curve, not above
    or below zero.  A dashed line at zero is drawn only as a fallback when
    no simulations were run (``n_perms=0``, ``L_exp`` all-NaN).

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
        Colormap name or instance used to cycle line colors.  ``None``
        (default) uses the active theme's categorical palette.
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

    result = _get_result(adata, key)
    # result is a dict keyed by cell-type string, each value has 'bins' and 'stats'
    if cell_types is None:
        cell_types = [k for k in result if not k.startswith("_")]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = _categorical_colors(cmap, len(cell_types))
    drew_l_exp = False
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

        # entry's L_obs/L_exp/L_hi/L_lo columns (`_ripley_l_kdtree`'s `_l()`)
        # are already the detrended L(d) - d quantity plotted here -- do not
        # subtract `bins` again, that produces L(d) - 2d and fabricates a
        # runaway decline even under pure CSR (verified: a synthetic
        # CSR-labeled dataset showed real L(d)-d ~ -56 at d=800 but plotted
        # as ~ -856).
        l_d = l_vals
        color = colors[i]
        ax.plot(bins[:len(l_d)], l_d, label=ct, color=color)
        if hasattr(entry, "columns") and "L_exp" in entry.columns:
            # The stored CSR expectation is the simulated null mean, NOT a
            # constant 0: with no edge correction the null curve bows below
            # zero at large d, and judging L_obs against y=0 there inverts
            # the clustered-vs-dispersed read.
            l_exp = entry["L_exp"].to_numpy(dtype=float)
            if np.isfinite(l_exp).any():
                ax.plot(
                    bins[:len(l_exp)], l_exp,
                    color=color, linestyle="--", linewidth=0.9, alpha=0.7,
                    label=f"_{ct}_L_exp",
                )
                drew_l_exp = True
        # Draw simulation envelope if available
        if hasattr(entry, "columns") and "L_hi" in entry.columns and "L_lo" in entry.columns:
            l_hi = entry["L_hi"].to_numpy(dtype=float)
            l_lo = entry["L_lo"].to_numpy(dtype=float)
            ax.fill_between(bins[:len(l_d)], l_lo, l_hi, alpha=0.15, color=color)

    if not drew_l_exp:
        # n_perms=0 stores an all-NaN L_exp; fall back to the naive y=0 CSR
        # reference (with the no-edge-correction caveat in the docstring).
        ax.axhline(0, color=get_theme().muted, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Distance d (µm)", fontsize=9)
    ax.set_ylabel("L(d) − d", fontsize=9)
    ax.set_title(title or "Ripley's L", fontsize=10)

    n = len(cell_types)
    if n <= 15:
        _style_legend(ax.legend(fontsize=7, framealpha=0.7, loc="upper left"))

    _style_axes(ax)
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

    .. note::
        ``read_xenium``'s default auto-loaded ``uns['transcripts']`` includes
        control-probe transcripts (``include_controls=True``), so control
        feature types get a real Q-score by default. Control rows whose
        ``var`` QC columns are NaN (``pp.calculate_qc_metrics`` computes
        per-gene metrics on non-control genes only) are backfilled from the
        ``uns['control_probe_summary']`` stash or recomputed directly from
        ``X`` (with a warning), so controls always render while still
        present. If a control type shows no points, its **Q-scores** are
        missing: ``uns['transcripts']`` was reloaded with
        ``include_controls=False``, or the controls were removed by
        ``pipeline.filter_xenium`` before
        ``pp.calculate_xenium_qc_metrics`` could stash them.

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
        mean_qv, _ = _resolve_field(adata, mean_qv_key, axis="var")
    else:
        raise ValueError(
            "No Q-score source found. Either: run ab.io.read_xenium_transcripts(mdata) "
            "so the table is available in uns['transcripts'], pass transcripts= explicitly, "
            "or supply mean_qv_key= if the column is pre-computed in var"
        )

    counts, _ = _resolve_field(adata, total_counts_key, axis="var")
    counts = counts + 1
    feature_types, _ = _resolve_field(adata, feature_type_key, axis="var")

    # Merge in per-gene control stats stashed by calculate_xenium_qc_metrics
    # (computed independently from raw X, so always populated regardless of
    # what mean_qv/total_counts_key ended up NaN for): backfill control genes
    # still present in var whose own count/QV came out NaN — e.g.
    # calculate_qc_metrics(exclude_controls=True) leaves control rows' own
    # total_counts NaN, or the transcripts table used for QV excluded some
    # control types — and append stash-only rows for control genes already
    # removed from var entirely (e.g. after pipeline.filter_xenium). Genes
    # still in var are backfilled in place rather than appended again, to
    # avoid plotting every control gene twice.
    ctrl_stash = adata.uns.get('control_probe_summary', {})
    if ctrl_stash and 'total_counts' in ctrl_stash:
        name_key = next(
            k for k in ctrl_stash
            if k not in ('feature_type', 'total_counts', 'mean_qv',
                         'n_cells_by_counts', 'n_obs')
        )
        ctrl_names = np.asarray(ctrl_stash[name_key], dtype=object)
        ctrl_counts = np.array(ctrl_stash['total_counts'], dtype=np.float64) + 1
        ctrl_qv = np.array(
            ctrl_stash.get('mean_qv', [np.nan] * len(ctrl_counts)), dtype=np.float64
        )
        ctrl_types = np.array(ctrl_stash['feature_type'], dtype=object)

        var_pos = {name: i for i, name in enumerate(adata.var_names)}
        still_present = np.array([n in var_pos for n in ctrl_names])
        for i in np.flatnonzero(still_present):
            j = var_pos[ctrl_names[i]]
            if not np.isfinite(counts[j]):
                counts[j] = ctrl_counts[i]
            if not np.isfinite(mean_qv[j]):
                mean_qv[j] = ctrl_qv[i]

        removed = ~still_present
        counts = np.concatenate([counts, ctrl_counts[removed]])
        mean_qv = np.concatenate([mean_qv, ctrl_qv[removed]])
        feature_types = np.concatenate([feature_types, ctrl_types[removed]])

    # Last-resort backfill straight from X: with no stash (only
    # calculate_qc_metrics ran, which leaves control rows' total_counts NaN
    # by design), the finite filter would otherwise silently drop every
    # control point from the plot.
    nan_counts = ~np.isfinite(counts[: adata.n_vars])
    if nan_counts.any():
        col_sums = np.asarray(adata.X[:, nan_counts].sum(axis=0)).ravel()
        counts[: adata.n_vars][nan_counts] = col_sums + 1
        log.warning(
            "gene_qv_scatter: var['%s'] was NaN for %d gene(s) — typically "
            "control probes, which pp.calculate_qc_metrics(exclude_controls="
            "True) skips by design. Their counts were recomputed from X so "
            "they still render.",
            total_counts_key, int(nan_counts.sum()),
        )

    # --- resolve palette ---
    effective_palette = {**_feature_type_palette(), **(palette or {})}
    unique_types = sorted(set(feature_types))

    # Assign a color to any feature type not in the palette
    known = set(effective_palette)
    extra = [ft for ft in unique_types if ft not in known]
    fallback_colors = _categorical_colors(None, len(extra))
    for i, ft in enumerate(extra):
        effective_palette[ft] = mcolors.to_hex(fallback_colors[i])

    # Draw Gene Expression first (background), controls on top
    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]

    return _generic_scatter(
        adata, x=counts, y=mean_qv, color=feature_types, axis="var",
        palette=effective_palette, category_order=draw_order,
        log_x=True, size=size, alpha=alpha,
        xlabel="total transcripts + 1", ylabel="Mean calibrated Q-score",
        title=title, ax=ax, figsize=figsize, **scatter_kwargs,
    )



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
    show_noise_floor: bool = True,
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

    Works on all platforms. ``read_xenium``, ``read_cosmx``, and
    ``read_merscope`` all write ``var['feature_type']``, so points are colored
    by control type everywhere (Xenium's probe/codeword classes; CosMx's
    Negative Probe vs. System Control; MERSCOPE's Blank Barcode). For data
    without a ``feature_type`` column (e.g. a custom reader +
    :func:`~astrolabe.pp.flag_controls`), coloring falls back to the boolean
    *control_key* column as a single "Control Probe" class.

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
    show_noise_floor
        If ``True`` (default) and ``uns['noise_floor']`` is present (written
        by :func:`~astrolabe.pp.calculate_noise_floor`), draw the
        control-derived noise-floor threshold as a dashed vertical line —
        genes left of it are statistically indistinguishable from background.
        Silently skipped when the threshold has not been computed.
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

    counts, _ = _resolve_field(adata, total_counts_key, axis="var")
    counts = counts + 1
    dropout, _ = _resolve_field(adata, pct_dropout_key, axis="var")
    detection_rate = 1.0 - dropout / 100.0

    # Backfill NaN rows straight from X: pp.calculate_qc_metrics
    # (exclude_controls=True, the default) deliberately leaves control rows'
    # per-gene metrics NaN, and the finite filter downstream would then drop
    # every control point — the exact population this plot exists to show.
    nan_rows = ~(np.isfinite(counts) & np.isfinite(detection_rate))
    if nan_rows.any():
        X_sub = adata.X[:, nan_rows]
        col_sums = np.asarray(X_sub.sum(axis=0)).ravel()
        n_cells_pos = np.asarray((X_sub > 0).sum(axis=0)).ravel()
        counts[nan_rows] = col_sums + 1
        detection_rate[nan_rows] = n_cells_pos / max(adata.n_obs, 1)
        log.info(
            "gene_detection_scatter: recomputed counts/detection rate from X "
            "for %d gene(s) whose QC columns were NaN (controls are excluded "
            "from pp.calculate_qc_metrics per-gene metrics by design).",
            int(nan_rows.sum()),
        )

    # --- resolve feature labels ---
    if feature_type_key in adata.var.columns:
        feature_types, _ = _resolve_field(adata, feature_type_key, axis="var")
        effective_palette = {**_feature_type_palette(), **(palette or {})}

        # Restore stashed controls (Xenium: after calculate_xenium_qc_metrics
        # + filter_genes). Only genes already REMOVED from var are appended —
        # controls still present were backfilled in place above, and
        # appending them again would draw every control point twice.
        ctrl_stash = adata.uns.get("control_probe_summary", {})
        if ctrl_stash and "total_counts" in ctrl_stash:
            name_key = next(
                k for k in ctrl_stash
                if k not in ("feature_type", "total_counts", "mean_qv",
                             "n_cells_by_counts", "n_obs")
            )
            ctrl_names = np.asarray(ctrl_stash[name_key], dtype=object)
            ctrl_counts = np.array(ctrl_stash["total_counts"], dtype=np.float64) + 1
            ctrl_types = np.array(ctrl_stash["feature_type"])
            if "n_cells_by_counts" in ctrl_stash and ctrl_stash.get("n_obs"):
                ctrl_detection = (
                    np.array(ctrl_stash["n_cells_by_counts"], dtype=np.float64)
                    / float(ctrl_stash["n_obs"])
                )
            else:
                ctrl_detection = np.full(len(ctrl_counts), np.nan)
            var_set = set(adata.var_names)
            removed = np.array([n not in var_set for n in ctrl_names])
            counts = np.concatenate([counts, ctrl_counts[removed]])
            detection_rate = np.concatenate([detection_rate, ctrl_detection[removed]])
            feature_types = np.concatenate([feature_types, ctrl_types[removed]])
    else:
        _ctrl_col = control_key if control_key in adata.var.columns else "is_control_probe"
        if _ctrl_col in adata.var.columns:
            is_ctrl_arr, _ = _resolve_field(adata, _ctrl_col, axis="var")
            is_ctrl = is_ctrl_arr.astype(bool)
            feature_types = np.where(is_ctrl, "Control Probe", "Gene Expression")
        else:
            feature_types = np.full(len(counts), "Gene Expression")
        t = get_theme()
        effective_palette = {
            "Gene Expression": t.muted,
            "Control Probe": t.cat_core[7],
            **(palette or {}),
        }

    # Assign fallback colors for any unlisted types
    unique_types = sorted(set(feature_types))
    known = set(effective_palette)
    extra = [ft for ft in unique_types if ft not in known]
    fallback_colors = _categorical_colors(None, len(extra))
    for i, ft in enumerate(extra):
        effective_palette[ft] = mcolors.to_hex(fallback_colors[i])

    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]

    panel = _generic_scatter(
        adata, x=counts, y=detection_rate, color=feature_types, axis="var",
        palette=effective_palette, category_order=draw_order,
        log_x=True, size=size, alpha=alpha,
        xlabel="total transcripts + 1", ylabel="Detection rate (fraction of cells)",
        title=title, ax=ax, figsize=figsize, **scatter_kwargs,
    )
    # With a caller-supplied ax, figure.axes[0] may be a *different* axes of
    # a multi-panel figure — target the axes actually drawn into.
    ax_plot = ax if ax is not None else panel.figure.axes[0]
    ax_plot.set_ylim(-0.05, 1.05)

    noise_floor = adata.uns.get("noise_floor") or {}
    if show_noise_floor and "threshold_mean_counts" in noise_floor:
        # The threshold is a mean-counts-per-cell statistic; this x-axis is
        # total counts (+1), so convert through the current cell count.
        x_thr = noise_floor["threshold_mean_counts"] * adata.n_obs + 1
        t = get_theme()
        ax_plot.axvline(
            x_thr, color=t.muted, linestyle="--", linewidth=1.0, zorder=1.5
        )
        ax_plot.text(
            x_thr, 0.97, " noise floor",
            transform=ax_plot.get_xaxis_transform(),
            ha="left", va="top", fontsize=8, color=t.muted,
        )
    return panel


# ---------------------------------------------------------------------------
# Mean-variance plot
# ---------------------------------------------------------------------------


@with_adata
def mean_variance_plot(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    feature_type_key: str = "feature_type",
    control_key: str = "is_control",
    palette: dict[str, str] | None = None,
    size: float = 8.0,
    alpha: float = 0.6,
    log_x: bool = True,
    log_y: bool = True,
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
        or :func:`~astrolabe.pp.log1p`) — either by calling it before those
        run, or by pointing *layer* at a layer holding a raw-count copy
        stashed beforehand.  After transformation the mean-variance
        relationship no longer follows Poisson statistics.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  ``X`` (or
        *layer*) must contain raw integer counts.
    modality
        Modality key when *data* is a :class:`~mudata.MuData`.
    layer
        Layer holding raw counts.  ``None`` reads ``X`` directly — only
        correct if ``X`` has not yet been normalized/log-transformed.
    feature_type_key
        ``var`` column containing feature-type labels (``'feature_type'``,
        set by :func:`~astrolabe.io.read_xenium`,
        :func:`~astrolabe.io.read_cosmx`, and
        :func:`~astrolabe.io.read_merscope`).  Used to color points by
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
    log_x, log_y
        Log-scale the mean / variance axis (both default ``True``).  Genes
        with zero mean or zero variance cannot be drawn on a log axis; a
        warning reports how many were dropped (and how many of those are
        controls).  Pass ``False`` to show them on a linear axis instead.
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

    X = adata.layers[layer] if layer is not None else adata.X
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
        feature_types, _ = _resolve_field(adata, feature_type_key, axis="var")
    elif control_key in adata.var.columns:
        is_ctrl_arr, _ = _resolve_field(adata, control_key, axis="var")
        feature_types = np.where(is_ctrl_arr.astype(bool), "Negative Control", "Gene Expression")
    elif "is_control_probe" in adata.var.columns:
        is_ctrl_arr, _ = _resolve_field(adata, "is_control_probe", axis="var")
        feature_types = np.where(is_ctrl_arr.astype(bool), "Negative Control Probe", "Gene Expression")
    else:
        feature_types = np.full(adata.n_vars, "Gene Expression", dtype=object)

    effective_palette = {**_feature_type_palette(), **(palette or {})}
    unique_types = sorted(set(feature_types))

    extra = [ft for ft in unique_types if ft not in effective_palette]
    fallback_colors = _categorical_colors(None, len(extra))
    for i, ft in enumerate(extra):
        effective_palette[ft] = mcolors.to_hex(fallback_colors[i])

    # Gene Expression first (background), controls on top
    draw_order = ["Gene Expression"] + [ft for ft in unique_types if ft != "Gene Expression"]

    # Report what the log-log finite filter is about to drop: zero-mean /
    # zero-variance genes are disproportionately the very control probes
    # this plot exists to show. The statistic itself is unchanged.
    dropped_mask = np.zeros(len(mean), dtype=bool)
    if log_x:
        dropped_mask |= ~(mean > 0)
    if log_y:
        dropped_mask |= ~(variance > 0)
    if dropped_mask.any():
        n_ctrl_dropped = int((np.asarray(feature_types)[dropped_mask] != "Gene Expression").sum())
        log.warning(
            "mean_variance_plot: %d gene(s) with zero mean or zero variance "
            "cannot be drawn on the log axes and were dropped (%d of them "
            "control probes/codewords). Pass log_x=False/log_y=False to show "
            "them on linear axes.",
            int(dropped_mask.sum()), n_ctrl_dropped,
        )

    panel = _generic_scatter(
        adata, x=mean, y=variance, color=feature_types, axis="var",
        palette=effective_palette, category_order=draw_order,
        log_x=log_x, log_y=log_y, size=size, alpha=alpha,
        xlabel="Mean expression (counts per cell)", ylabel="Variance",
        title=title, ax=ax, figsize=figsize, **scatter_kwargs,
    )
    ax_plot = ax if ax is not None else panel.figure.axes[0]

    # Poisson reference line: variance = mean  (y = x on log-log).
    # Uses primary_ink rather than muted -- muted is also "Gene Expression"'s
    # color (_feature_type_palette), and the two would otherwise be indistinguishable.
    pos = mean[mean > 0]
    if len(pos) > 0:
        lo, hi = float(pos.min()), float(pos.max())
        x_ref = np.logspace(np.log10(lo), np.log10(hi), 300)
        ax_plot.plot(
            x_ref, x_ref,
            color=get_theme().primary_ink, linestyle="--", linewidth=1.2,
            zorder=5, label="Poisson (var = mean)",
        )

        legend_handles = _legend_handles(
            unique_types, [effective_palette.get(ft, get_theme().muted) for ft in unique_types]
        ) + [
            plt.Line2D(
                [0], [0],
                color=get_theme().primary_ink, linestyle="--", linewidth=1.2,
                label="Poisson (var = mean)",
            )
        ]
        leg = ax_plot.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            frameon=False,
            fontsize=8,
        )
        _style_legend(leg)

    return panel


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

@with_adata
def cell_area_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str | None = None,
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
    return _generic_histogram(
        adata, "cell_area", axis="obs",
        bins=bins, color=color, xlabel="Cell area (µm²)", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def nucleus_area_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str | None = None,
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
    return _generic_histogram(
        adata, "nucleus_area", axis="obs",
        bins=bins, color=color, xlabel="Nucleus area (µm²)", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def genes_per_cell_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    color: str | None = None,
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
    return _generic_histogram(
        adata, "n_genes_by_counts", axis="obs",
        bins=bins, color=color, xlabel="Genes per cell", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def transcripts_per_cell_histogram(
    adata: ad.AnnData,
    *,
    bins: int | np.ndarray = 50,
    log: bool = True,
    color: str | None = None,
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
    return _generic_histogram(
        adata, "total_counts", axis="obs",
        bins=bins, log=log, color=color, xlabel="Transcripts per cell", title=title,
        vlines=vlines, ax=ax, figsize=figsize, **hist_kwargs,
    )


@with_adata
def genes_vs_area_scatter(
    adata: ad.AnnData,
    *,
    area_key: str = "cell_area",
    log_x: bool = False,
    log_y: bool = False,
    density: bool = True,
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    point_size: float = 4.0,
    alpha: float = 0.5,
    annotate_corr: bool = True,
    title: str = "Genes detected vs. area area",
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Scatter of genes detected per cell vs. cell (or nucleus) area.

    A positive relationship is expected biologically (bigger cells hold more
    mRNA), but a suspiciously tight/linear one can flag over-segmentation —
    cell size driving detected gene counts rather than genuine expression
    differences. Points are colored by local 2-D density by default so
    overplotting stays readable at high cell counts.

    Requires ``obs['n_genes_by_counts']`` and ``obs[area_key]`` — both
    populated by :func:`~astrolabe.pp.calculate_qc_metrics` (the counts) and
    the platform reader (the area).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    area_key
        ``obs`` column with the area measurement (µm²). Default
        ``'cell_area'``; pass ``'nucleus_area'`` to compare against nucleus
        size instead.
    log_x, log_y
        Log-scale the area / gene-count axis.
    density
        Color points by local 2-D density (default) rather than a flat color.
    cmap
        Colormap used for density coloring.
    point_size
        Marker size in points².
    alpha
        Marker opacity.
    annotate_corr
        Annotate the Spearman correlation coefficient in the corner.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **scatter_kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    FigurePanel
    """
    return _generic_scatter(
        adata, x=area_key, y="n_genes_by_counts", axis="obs",
        xlabel=f"{area_key.replace('_', ' ').capitalize()} (µm²)",
        ylabel="Genes per cell",
        log_x=log_x, log_y=log_y, density=density, cmap=cmap,
        size=point_size, alpha=alpha, annotate_corr=annotate_corr,
        title=title, ax=ax, figsize=figsize, **scatter_kwargs,
    )


@with_adata
def transcripts_vs_area_scatter(
    adata: ad.AnnData,
    *,
    area_key: str = "cell_area",
    log_x: bool = False,
    log_y: bool = True,
    density: bool = True,
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    point_size: float = 4.0,
    alpha: float = 0.5,
    annotate_corr: bool = True,
    title: str = "Transcripts vs. area",
    ax: matplotlib.axes.Axes | None = None,
    figsize: tuple[float, float] = (6, 5),
    **scatter_kwargs: Any,
) -> FigurePanel:
    """Scatter of total transcript counts per cell vs. cell (or nucleus) area.

    Uses a log y-axis by default since transcript counts span several orders
    of magnitude, matching :func:`transcripts_per_cell_histogram`. As with
    :func:`genes_vs_area_scatter`, a suspiciously tight relationship between
    counts and area can indicate over-segmentation rather than genuine
    biology.

    Requires ``obs['total_counts']`` and ``obs[area_key]`` — both populated
    by :func:`~astrolabe.pp.calculate_qc_metrics` (the counts) and the
    platform reader (the area).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    area_key
        ``obs`` column with the area measurement (µm²). Default
        ``'cell_area'``; pass ``'nucleus_area'`` to compare against nucleus
        size instead.
    log_x, log_y
        Log-scale the area / transcript-count axis.
    density
        Color points by local 2-D density (default) rather than a flat color.
    cmap
        Colormap used for density coloring.
    point_size
        Marker size in points².
    alpha
        Marker opacity.
    annotate_corr
        Annotate the Spearman correlation coefficient in the corner.
    title
        Axes title.
    ax
        Existing axes to draw into.
    figsize
        Figure size when *ax* is ``None``.
    **scatter_kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    FigurePanel
    """
    return _generic_scatter(
        adata, x=area_key, y="total_counts", axis="obs",
        xlabel=f"{area_key.replace('_', ' ').capitalize()} (µm²)",
        ylabel="Transcripts per cell",
        log_x=log_x, log_y=log_y, density=density, cmap=cmap,
        size=point_size, alpha=alpha, annotate_corr=annotate_corr,
        title=title, ax=ax, figsize=figsize, **scatter_kwargs,
    )


# ---------------------------------------------------------------------------
# Shape feature distribution plots
# ---------------------------------------------------------------------------

_SHAPE_METRIC_SUFFIXES: frozenset[str] = frozenset({
    "area", "perimeter", "span_x", "span_y", "equivalent_radius",
    "bbox_major_axis_length", "bbox_minor_axis_length", "aspect_ratio", "rna_density",
    "circularity", "solidity",
})


def _array_histogram(
    values: np.ndarray,
    *,
    bins: int | np.ndarray = 50,
    log: bool = False,
    color: str | None = None,
    xlabel: str = "",
    ax: "matplotlib.axes.Axes | None" = None,
    figsize: tuple[float, float] = (6, 4),
    **hist_kwargs: Any,
) -> FigurePanel:
    """Histogram of an arbitrary float array (does not read from obs)."""
    t = get_theme()
    color = color if color is not None else t.cat_core[0]
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
    ax.yaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax.set_axisbelow(True)

    _style_axes(ax)
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
    color: str | None = None,
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
        _generic_histogram(
            adata, key, axis="obs",
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
    color: str | None = None,
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
    ...         ("cell_bbox_major_axis_length", "cell_bbox_minor_axis_length"),
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


def _channel_name(adata: ad.AnnData, channel: int | str) -> str:
    """Resolve a channel index or name to its ``var_names`` string label."""
    if isinstance(channel, str):
        if channel not in adata.var_names:
            raise KeyError(
                f"Channel '{channel}' not found in var_names. "
                f"Available: {list(adata.var_names)}"
            )
        return channel
    return adata.var_names[int(channel)]


def _resolve_channel(adata: ad.AnnData, channel: int | str) -> tuple[np.ndarray, str]:
    """Return ``(values, channel_name)`` for a channel index or name."""
    name = _channel_name(adata, channel)
    values, _ = _resolve_field(adata, name, axis="obs")
    return values, name


@with_adata
def intensity_histogram(
    adata: ad.AnnData,
    channel: int | str = 0,
    *,
    bins: int = 60,
    color: str | None = None,
    otsu: bool = True,
    vlines: list[float] | None = None,
    vline_color: str | None = None,
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
    t = get_theme()
    color = color if color is not None else t.cat_core[0]
    vline_color = vline_color if vline_color is not None else t.muted

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
        thresh = _otsu_threshold(finite, log=log)
        if thresh is not None:
            all_vlines.append((thresh, f"Otsu = {thresh:.1f}"))

    for v in (vlines or []):
        all_vlines.append((v, f"{v:.1f}"))

    for v, label in all_vlines:
        ax.axvline(v, color=vline_color, linestyle="--", linewidth=1.2, label=label)

    if all_vlines:
        _style_legend(ax.legend(frameon=False, fontsize=8))

    if log:
        ax.set_xscale("log")

    ax.set_xlabel(f"Intensity — {ch_name}")
    ax.set_ylabel("Number of cells")
    ax.set_title(title or ch_name)
    ax.yaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax.set_axisbelow(True)

    _style_axes(ax)
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
    color: str | None = None,
    otsu: bool = True,
    vlines: "list[float] | None" = None,
    vline_color: str | None = None,
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

    t = get_theme()
    color = color if color is not None else t.cat_core[0]
    vline_color = vline_color if vline_color is not None else t.muted

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
            thresh = _otsu_threshold(finite, log=log)
            if thresh is not None:
                panel_vlines.append((thresh, f"Otsu={thresh:.1f}"))
        for v in (vlines or []):
            panel_vlines.append((v, f"{v:.1f}"))
        for v, label in panel_vlines:
            ax_i.axvline(v, color=vline_color, linestyle="--", linewidth=1.0, label=label)
        if panel_vlines:
            _style_legend(ax_i.legend(frameon=False, fontsize=6))

        if log:
            ax_i.set_xscale("log")
        ax_i.set_title(ch_name, fontsize=9)
        ax_i.set_xlabel("Intensity", fontsize=8)
        ax_i.set_ylabel("Cells", fontsize=8)
        ax_i.tick_params(labelsize=7)
        ax_i.yaxis.grid(True, color=t.gridline, linewidth=0.6)
        ax_i.set_axisbelow(True)
        _style_axes(ax_i)

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


def _otsu_threshold(values: np.ndarray, *, log: bool = False) -> float | None:
    """Otsu threshold matched to how the axis will be displayed.

    When the axis is log-scaled, the between-class-variance optimum must be
    found in log10 space (``10**threshold_otsu(log10(x))``) — Otsu on the
    linear values of a log-normal-ish intensity distribution lands the gate
    far into the right tail of what the viewer actually sees.  On linear
    axes, the values are percentile-clipped (0.1–99.9) first so a handful of
    extreme outliers can't drag the optimum.  Returns ``None`` when a
    threshold can't be computed (missing skimage, constant data, ...).
    """
    try:
        from skimage.filters import threshold_otsu
    except ImportError:
        return None
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    try:
        if log:
            pos = finite[finite > 0]
            if pos.size < 2:
                return None
            return float(10 ** threshold_otsu(np.log10(pos)))
        lo, hi = np.percentile(finite, [0.1, 99.9])
        return float(threshold_otsu(np.clip(finite, lo, hi)))
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
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    point_size: float | None = None,
    alpha: float | None = None,
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
        Scatter point size in points². ``None`` (default) picks ``2.0`` for
        large datasets (> 5,000 cells, where overplotting is the risk) or
        ``6.0`` for smaller ones (where the risk is invisible sparse points
        instead).
    alpha
        Scatter point opacity. ``None`` (default) picks ``0.4``/``0.7`` using
        the same size threshold as ``point_size``.
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

    # Small datasets need visibly larger, more opaque points -- the
    # opposite risk of the large-dataset overplotting these defaults are
    # normally tuned for. Reuses the same 5,000-cell threshold as the
    # rasterization switch below for one consistent "small vs. large
    # dataset" mental model.
    small_dataset = len(xf) <= 5_000
    if point_size is None:
        point_size = 6.0 if small_dataset else 2.0
    if alpha is None:
        alpha = 0.7 if small_dataset else 0.4

    # ------------------------------------------------------------------ #
    # Gates                                                               #
    # ------------------------------------------------------------------ #
    if gate_x is None and otsu:
        gate_x = _otsu_threshold(xf, log=log_x)
    if gate_y is None and otsu:
        gate_y = _otsu_threshold(yf, log=log_y)

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
        point_cmap = plt.get_cmap(cmap)
        scatter_c = point_cmap(norm(c_scatter))
        colorbar_label = c_name
    else:
        density = _density_values(xf, yf, density_bins)
        order = np.argsort(density)          # draw dense clusters on top
        xf, yf = xf[order], yf[order]
        density = density[order]
        norm = mcolors.Normalize(vmin=density.min(), vmax=density.max())
        # Every plotted point has density >= 1 (it's a member of its own
        # bin), so the *palest* end of the colormap -- near-white -- is
        # what most points get whenever density is low almost everywhere,
        # e.g. a small or diffuse dataset. Combined with scatter alpha,
        # that renders the majority of points invisible against a light
        # background. Drop the palest ~35% of the ramp so even the
        # lowest-density points stay visibly colored; reused for the
        # colorbar below so dots and colorbar stay consistent.
        base_cmap = plt.get_cmap(cmap)
        point_cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{cmap}_density", base_cmap(np.linspace(0.35, 1.0, 256))
        )
        scatter_c = point_cmap(norm(density))
        colorbar_label = "density"

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
    t = get_theme()
    marginal_color = t.cat_core[0]

    def _marginal_bin_edges(vals, n, use_log):
        # Per-axis: log-spaced bins only for the axis actually drawn on a
        # log scale — one shared predicate put log-spaced bins on a linear
        # marginal axis whenever log_x != log_y.
        if use_log:
            pos = vals[vals > 0]
            if len(pos) > 1:
                return np.logspace(np.log10(pos.min()), np.log10(pos.max()), n + 1)
        return n

    ax_top.hist(xf, bins=_marginal_bin_edges(xf, marginal_bins, log_x),
                color=marginal_color, edgecolor="none")
    ax_right.hist(yf, bins=_marginal_bin_edges(yf, marginal_bins, log_y),
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
    ax_top.yaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax_right.xaxis.grid(True, color=t.gridline, linewidth=0.6)
    ax_top.set_axisbelow(True)
    ax_right.set_axisbelow(True)

    # ------------------------------------------------------------------ #
    # Gate lines                                                          #
    # ------------------------------------------------------------------ #
    gate_kw = dict(color=t.cat_core[7], linestyle="--", linewidth=1.2, zorder=5)
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
                color=t.primary_ink,
                bbox=dict(boxstyle="round,pad=0.2", fc=t.surface, alpha=0.7, ec="none"),
            )

    # ------------------------------------------------------------------ #
    # Colorbar                                                            #
    # ------------------------------------------------------------------ #
    sm = plt.cm.ScalarMappable(cmap=point_cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cb.set_label(colorbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style_colorbar(cb)

    _style_axes(ax_main)
    if title:
        fig.suptitle(title, y=1.01, fontsize=10, fontfamily=TITLE_FONT, fontweight=600, color=t.primary_ink)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Moran scatter plot
# ---------------------------------------------------------------------------

def _moran_quad_palette() -> dict[str, str]:
    """LISA quadrant colors from the active theme's core palette."""
    t = get_theme()
    return {"HH": t.cat_core[7], "LL": t.cat_core[0], "HL": t.cat_core[1], "LH": t.cat_core[5]}


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
        No provenance is stored with the result — make sure the stored run
        used the same *neighbors_key* as this plot, or the slope line will
        not match the scatter.
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

    t = get_theme()
    if ax is None:
        fig, ax_plot = plt.subplots(figsize=figsize)
    else:
        ax_plot = ax
        fig = ax.get_figure()

    # ------------------------------------------------------------------ #
    # Resolve values and spatial lag                                      #
    # ------------------------------------------------------------------ #
    y, _ = _resolve_field(adata, key, axis="obs", layer=layer)

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
    pal = {**_moran_quad_palette(), **(palette or {})}
    no_quad_color = t.muted

    quad_arr = None
    if lisa_key is not None and lisa_key in adata.uns:
        lisa_df = _get_result(adata, lisa_key)
        try:
            # Align by obs_names, not position: the stored LISA frame keeps
            # the full run's cell index, which desyncs positionally once the
            # AnnData has been subset after tl.moran_local ran. Cells absent
            # from the stored result get the neutral no-quadrant color.
            quad_series = lisa_df.loc[key]["quadrant"].reindex(adata.obs_names)
            quad_arr = quad_series.fillna("NS").to_numpy(dtype=str)
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
    ax_plot.plot(x_line, slope * x_line, color=t.primary_ink, linewidth=1.2, zorder=5)

    # ------------------------------------------------------------------ #
    # Reference lines and legend                                          #
    # ------------------------------------------------------------------ #
    ax_plot.axhline(0, color=t.gridline, linewidth=0.8, linestyle="--", zorder=1)
    ax_plot.axvline(0, color=t.gridline, linewidth=0.8, linestyle="--", zorder=1)

    handles = [plt.Line2D([0], [0], color=t.primary_ink, linewidth=1.2, label=f"I = {slope:.3f}")]
    if quad_arr is not None:
        quads_present = [(quad, color) for quad, color in pal.items() if quad in quad_arr]
        handles += _legend_handles([q for q, _ in quads_present], [c for _, c in quads_present])
    _style_legend(ax_plot.legend(handles=handles, fontsize=7, framealpha=0.7))

    ax_plot.set_xlabel(f"z({key})", fontsize=9)
    ax_plot.set_ylabel(f"Spatial lag of z({key})", fontsize=9)
    ax_plot.set_title(title or f"Moran plot — {key}", fontsize=10)

    _style_axes(ax_plot)
    if ax is None:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


# ---------------------------------------------------------------------------
# Neighborhood analysis companion plots
# ---------------------------------------------------------------------------

_DIVNORM_CMAP = _DEFAULT_DIVERGING_CMAP


@with_adata
def nhood_enrichment_plot(
    adata: ad.AnnData,
    groupby: str,
    *,
    key: str = "nhood_enrichment",
    metric: str = "zscore",
    cmap: str | None = None,
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
        ``'zscore'`` (default) or ``'counts'``.  Z-scores are drawn on a
        zero-centered diverging scale; counts (non-negative) on a
        sequential ``Normalize(0, max)`` scale with integer annotations.
    cmap
        Colormap for the heatmap.  ``None`` (default) uses the theme's
        diverging colormap for ``'zscore'`` and its sequential colormap for
        ``'counts'``.
    vmax
        Color-scale maximum.  Defaults to ``max(|values|)`` (zscore) or
        ``max(values)`` (counts).
    figsize
        Figure size.
    title
        Axes title.
    ax
        Existing axes to draw into.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.nhood_enrichment() first")

    result = _get_result(adata, key)
    groups = result["groups"]
    mat = result[metric].astype(float)
    n = len(groups)
    _validate_groupby_against_result(adata, groupby, groups, "nhood_enrichment_plot")

    if figsize is None:
        figsize = (max(4.0, n * 0.7 + 1.5), max(3.5, n * 0.7 + 1.0))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if metric == "zscore":
        # Signed statistic: zero-centered diverging scale.
        v = float(vmax) if vmax is not None else float(np.abs(mat).max())
        if v == 0:
            v = 1.0
        norm: mcolors.Normalize = mcolors.TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v)
        cmap_used = cmap if cmap is not None else _DIVNORM_CMAP
        fmt = "{:.1f}"
        strength = lambda val: abs(val) / v  # noqa: E731
    else:
        # Non-negative counts: a diverging map centered on 0 wastes half its
        # range and washes every cell into the warm half — use a sequential
        # scale anchored at 0, with integer annotations.
        v = float(vmax) if vmax is not None else float(np.nanmax(mat))
        if v == 0:
            v = 1.0
        norm = mcolors.Normalize(vmin=0.0, vmax=v)
        cmap_used = cmap if cmap is not None else _DEFAULT_CONTINUOUS_CMAP
        fmt = "{:.0f}"
        strength = lambda val: val / v  # noqa: E731

    im = ax.imshow(mat, cmap=cmap_used, norm=norm, aspect="auto")

    fs = max(5, min(9, 80 // max(n, 1)))
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            ax.text(
                j, i, fmt.format(val),
                ha="center", va="center", fontsize=fs,
                color="white" if strength(val) > 0.6 else "black",
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    # The matrix is symmetric — "Sender"/"Receiver" implied a directionality
    # it doesn't have. Both axes are simply the cell-type grouping.
    ax.set_xlabel(groupby)
    ax.set_ylabel(groupby)
    ax.set_title(title or f"Neighborhood enrichment ({metric})", fontsize=10)
    _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.7, label=metric))

    _style_axes(ax)
    if own_fig:
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
    cmap: str | None = None,
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
        Colormap name or instance used to cycle line colors.  ``None``
        (default) uses the active theme's categorical palette.
    figsize
        Figure size.
    title
        Axes title.
    ax
        Existing axes.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.co_occurrence() first")

    result = _get_result(adata, key)
    occ = result["occ"]
    interval = np.asarray(result["interval"], dtype=float)
    groups = result["groups"]
    _validate_groupby_against_result(adata, groupby, groups, "co_occurrence_plot")

    # x positions: when the stored interval has one more entry than the occ
    # curves (edges, the GPU/squidpy convention of n edges -> n-1 bins), plot
    # each ring at its midpoint instead of raising a length-mismatch error.
    n_rings = occ.shape[2]
    if len(interval) == n_rings + 1:
        x_vals = (interval[:-1] + interval[1:]) / 2.0
    else:
        x_vals = interval

    src_idx = [groups.index(g) for g in source_groups] if source_groups is not None else range(len(groups))
    tgt_idx = [groups.index(g) for g in target_groups] if target_groups is not None else range(len(groups))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    pairs = [(a, b) for a in src_idx for b in tgt_idx]
    colors = _categorical_colors(cmap, len(pairs))
    for line_i, (a, b) in enumerate(pairs):
        ax.plot(x_vals, occ[a, b], label=f"{groups[a]} → {groups[b]}", color=colors[line_i])

    ax.axhline(1.0, color=get_theme().muted, linestyle="--", linewidth=0.8, zorder=0)
    ax.set_xlabel("Distance (µm)")
    ax.set_ylabel("Co-occurrence ratio")
    ax.set_title(title or "Co-occurrence probability ratio", fontsize=10)
    n_lines = len(list(src_idx)) * len(list(tgt_idx))
    if n_lines <= 15:
        _style_legend(ax.legend(fontsize=7, loc="upper right", framealpha=0.7))

    _style_axes(ax)
    if own_fig:
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
    cmap: str | None = None,
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
        Colormap name or instance for bar colors.  ``None`` (default) uses
        the active theme's categorical palette.
    figsize
        Figure size.
    title
        Figure suptitle.
    """
    if key not in adata.uns:
        raise KeyError(
            f"uns['{key}'] not found. Run ab.tl.centrality_scores() first"
        )

    df: pd.DataFrame = _get_result(adata, key)
    if metrics is None:
        metrics = [c for c in df.columns if not df[c].isna().all()]
    _validate_groupby_against_result(
        adata, groupby, df.index.tolist(), "centrality_scores_plot"
    )

    n_metrics = len(metrics)
    n_groups = len(df)
    if figsize is None:
        figsize = (4.0 * n_metrics, max(3.0, n_groups * 0.35 + 1.0))

    fig, axes = plt.subplots(1, n_metrics, figsize=figsize, sharey=True)
    if n_metrics == 1:
        axes = [axes]

    colors = _categorical_colors(cmap, n_groups)
    cats = df.index.tolist()
    muted = get_theme().muted
    for ax, metric in zip(axes, metrics):
        # NaN scores stay NaN: matplotlib draws no bar for them, and an
        # explicit 'n/a' marks the absence. fillna(0) rendered a missing
        # score as a real zero — indistinguishable from a true 0.0.
        vals = df[metric].to_numpy(dtype=float)
        ax.barh(cats, vals, color=colors)
        for yi, v in enumerate(vals):
            if not np.isfinite(v):
                ax.text(
                    0.02, yi, "n/a",
                    transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6, color=muted,
                )
        ax.set_xlabel(metric.replace("_", " ").title(), fontsize=8)
        ax.tick_params(labelsize=7)
        _style_axes(ax)

    # Invert exactly once, outside the loop: the axes share their y-axis
    # (sharey=True), so per-axes inversion inside the loop flipped the
    # shared axis back and forth — final ordering depended on whether the
    # metric count was odd or even.
    axes[0].invert_yaxis()

    if title:
        fig.suptitle(title, fontsize=10, fontfamily=TITLE_FONT, fontweight=600, color=get_theme().primary_ink)

    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def expression_by_distance_plot(
    adata: ad.AnnData,
    *genes: str,
    key: str = "expression_by_distance",
    cmap: str | None = None,
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
        Colormap name or instance used to cycle line colors.  ``None``
        (default) uses the active theme's categorical palette.
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

    result = _get_result(adata, key)
    mean_df: pd.DataFrame = result["mean"]
    se_df: pd.DataFrame = result["se"]
    anchor_groups = result.get("anchor_groups", [])
    anchor_key = result.get("anchor_key", "")

    plot_genes = list(genes) if genes else mean_df.columns.tolist()
    missing = [g for g in plot_genes if g not in mean_df.columns]
    if missing:
        raise KeyError(f"Genes not in result: {missing}")

    distances = mean_df.index.to_numpy(dtype=float)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = _categorical_colors(cmap, len(plot_genes))
    for i, gene in enumerate(plot_genes):
        mean = mean_df[gene].to_numpy(dtype=float)
        se = se_df[gene].to_numpy(dtype=float)
        color = colors[i]
        ax.plot(distances, mean, label=gene, color=color)
        ax.fill_between(distances, mean - se, mean + se, alpha=0.2, color=color)

    if isinstance(anchor_groups, str):
        anchor_label = anchor_groups
    else:
        anchor_label = ", ".join(str(g) for g in anchor_groups)
    ax.set_xlabel(f"Distance to {anchor_key} = {anchor_label} (µm)")
    ax.set_ylabel("Mean expression")
    ax.set_title(title or "Expression by distance", fontsize=10)
    if len(plot_genes) <= 12:
        _style_legend(ax.legend(fontsize=8, framealpha=0.7))

    _style_axes(ax)
    if own_fig:
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
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
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
        Colormap for dot color (encodes −log₁₀ p-value, clipped at the
        permutation resolution ``1/(n_perms+1)``).  Defaults to the active
        theme's sequential colormap.
    figsize
        Figure size.  Auto-sized if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.ligrec() first")

    result = _get_result(adata, key)
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

    # Rows are already restricted to significant LR pairs above; mirror that
    # for columns too. With many cell types, ct_pairs is n_types² and most
    # (sender, receiver) combinations carry no signal for the LR pairs being
    # shown — rendering all of them anyway makes the figure balloon in width
    # (e.g. 100 columns for 10 cell types) even when only a handful are ever
    # colored in. Keep only columns significant for at least one shown row.
    col_sig = (pval_plot < p_threshold).any(axis=0)
    ct_pairs = [p for p, keep in zip(ct_pairs, col_sig) if keep]
    means_plot = means_plot.loc[:, col_sig]
    pval_plot = pval_plot.loc[:, col_sig]

    n_rows, n_cols = means_plot.shape
    col_labels = [f"{s}|{r}" for s, r in ct_pairs]
    row_labels = [f"{lig}:{rec}" for lig, rec in means_plot.index]

    if figsize is None:
        # Cap width even when many columns survive the significance filter
        # above (e.g. many cell types) rather than growing the figure
        # unboundedly — per-column spacing shrinks instead past this point.
        figsize = (min(24.0, max(4.0, n_cols * 0.8 + 2.0)), max(3.0, n_rows * 0.35 + 1.5))

    fig, ax = plt.subplots(figsize=figsize)

    score_vals = means_plot.values.astype(float)
    max_score = float(score_vals.max()) or 1.0
    # Clip p at the permutation resolution 1/(n_perms + 1) — the smallest
    # empirical p the stored test can actually produce — so the color scale
    # reflects real resolution rather than an arbitrary 1e-10 floor. The
    # colorbar then shows RAW −log10(p) with true vmin/vmax (0 to the
    # clipped maximum), not values silently normalized to the per-figure max.
    n_perms = result.get("metadata", {}).get("n_perms")
    p_floor = 1.0 / (float(n_perms) + 1.0) if n_perms else 1e-10
    log_pval = -np.log10(np.clip(pval_plot.values.astype(float), p_floor, 1.0))
    max_log_pval = float(log_pval.max()) or 1.0

    xs, ys, sizes, colors = [], [], [], []
    for j in range(n_cols):
        for i in range(n_rows):
            xs.append(j)
            ys.append(i)
            sizes.append(300 * score_vals[i, j] / max_score)
            colors.append(log_pval[i, j])

    t = get_theme()
    sc_plot = ax.scatter(
        xs, ys,
        s=sizes, c=colors,
        cmap=cmap, vmin=0.0, vmax=max_log_pval,
        edgecolors=t.muted, linewidths=0.3, alpha=0.9,
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
    floor_note = f", floor 1/(n_perms+1) = {p_floor:.2g}" if n_perms else ""
    cbar.set_label(f"−log₁₀(p){floor_note}", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    _style_colorbar(cbar)

    for frac, label in [(0.25, "25 %"), (0.75, "75 %")]:
        ax.scatter([], [], s=300 * frac, c=t.muted, alpha=0.6, label=f"score {label}")
    _style_legend(ax.legend(
        title="Score (rel.)", fontsize=6, title_fontsize=6,
        loc="upper left", bbox_to_anchor=(1.12, 1.0), framealpha=0.7,
    ))

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_umatrix(
    adata: ad.AnnData,
    *,
    key: str = "som",
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Plot the SOM unified distance matrix (U-matrix).

    High U-matrix values indicate cluster boundaries (nodes whose weights are
    dissimilar to their neighbors); low values mark cluster centers.

    The plotted values are **unnormalized** mean codebook distances to the
    8-connected grid neighbors, in the training feature space (z-scored units
    when the SOM was trained with ``scale=True``). A min-max-normalized
    U-matrix (minisom's ``distance_map()`` convention) always has a maximum
    of exactly 1.0, which makes a collapsed, over-smoothed codebook visually
    indistinguishable from a well-separated one — real units keep that
    difference visible. The title reports the map's quantization and
    topographic error when available.

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
        Colormap. Defaults to the active theme's sequential colormap
        (``'astrolabe_seq'``).
    annotate
        Overlay ``"{col}_{row}"`` node labels on each cell. Off by default —
        hundreds of labels on a 20×20+ grid are unreadable; turn on for small
        maps (e.g. the 5×5 subcellular grid).
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")

    umatrix: np.ndarray = _get_result(adata, key)["umatrix"]
    sx, sy = umatrix.shape

    if figsize is None:
        figsize = (max(4.0, sx * 0.6), max(3.5, sy * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(umatrix.T, cmap=cmap, origin="upper", aspect="equal")
    _style_colorbar(
        fig.colorbar(im, ax=ax, shrink=0.8, label="mean neighbor codebook distance")
    )

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
    if title is None:
        params = _get_result(adata, key).get("params", {})
        qe, te = params.get("quantization_error"), params.get("topographic_error")
        title = "SOM U-matrix"
        if qe is not None and te is not None:
            title += f"  (QE={qe:.3f}, TE={te:.3f})"
    ax.set_title(title, fontsize=10)

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_component_plane(
    adata: ad.AnnData,
    gene: str,
    *,
    key: str = "som",
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    annotate: bool = False,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Plot a SOM component plane for one gene.

    Shows how strongly each SOM node is activated by a single gene — the weight
    value for that gene across all nodes. Nodes where the gene has high weight
    are those whose prototype vector includes high expression of that gene.

    When the SOM was trained with ``scale=True`` (the default), weights live
    in per-feature z-scored space, so plane amplitudes are directly
    comparable across genes rather than being dominated by each gene's raw
    expression magnitude.

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

    weights: np.ndarray = _get_result(adata, key)["weights"]
    sx, sy, n_feat = weights.shape

    if n_feat != adata.n_vars:
        raise ValueError(
            f"SOM was trained on {n_feat} features but adata has {adata.n_vars} vars. "
            "Component planes require the SOM to be trained with use_rep=None."
        )

    gene_idx = adata.var_names.get_loc(gene)
    plane = weights[:, :, gene_idx]

    if figsize is None:
        figsize = (max(4.0, sx * 0.6), max(3.5, sy * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(plane.T, cmap=cmap, origin="upper", aspect="equal")
    _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.8, label="Weight"))

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

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_hits(
    adata: ad.AnnData,
    *,
    key: str = "som",
    cmap: str = _DEFAULT_CONTINUOUS_CMAP,
    annotate: bool = False,
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
        Colormap. Defaults to the active theme's sequential colormap
        (``'astrolabe_seq'``).
    annotate
        Overlay cell count in each node. Off by default — unreadable on
        large grids; turn on for small maps.
    figsize
        Figure size. Auto-computed if ``None``.
    title
        Axes title.
    """
    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")

    hit_counts: np.ndarray = _get_result(adata, key)["hit_counts"]
    sx, sy = hit_counts.shape

    if figsize is None:
        figsize = (max(4.0, sx * 0.6), max(3.5, sy * 0.6))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(hit_counts.T, cmap=cmap, origin="upper", aspect="equal")
    _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.8, label="Cell count"))

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

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_quality_plot(
    adata: ad.AnnData,
    *,
    key: str = "som",
    local_radius: float = 3.0,
    max_pairs: int = 50_000,
    seed: int = 0,
    figsize: tuple[float, float] = (5.5, 5.0),
    title: str | None = None,
) -> FigurePanel:
    """Topology-preservation diagnostic: grid distance vs. codebook distance.

    For pairs of SOM nodes, plots the distance between the nodes on the grid
    against the Euclidean distance between their codebook vectors.

    **The headline statistic is deliberately local.** Topology preservation is
    a claim about *neighbourhoods* — that adjacent nodes hold similar cells —
    and a Spearman correlation over all node pairs cannot test it: the global
    trend dominates, so even a badly folded map scores 0.7–0.9 simply because
    opposite corners really are far apart in feature space. The reported
    ``rho_local`` is therefore restricted to pairs within *local_radius* grid
    steps, which is the range the U-matrix, hit map and component planes
    actually rely on. The global value is shown alongside for reference, and a
    large gap between them (global high, local low) is the signature of a
    folded map.

    Also reported are **trustworthiness** and **continuity** at that radius
    (Venna & Kaski 2001, ICANN) — the standard pair of local
    neighbourhood-preservation measures. Trustworthiness penalises nodes that
    are grid-neighbours but not codebook-neighbours (false adjacency, the
    failure that makes a map figure lie); continuity penalises the reverse
    (codebook-neighbours torn apart on the grid). Both run 0–1, higher better.

    A near-zero codebook-distance band across all grid distances means the
    codebook collapsed (over-smoothed, e.g. a neighbourhood radius that never
    annealed) — the failure the U-matrix alone can hide.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with SOM results
        from :func:`~astrolabe.tl.som` (or the other SOM functions).
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key written by the SOM function.
    local_radius
        Grid-distance cutoff defining "local" for ``rho_local``,
        trustworthiness and continuity. The default 3 covers the 8-connected
        neighbourhood plus two rings.
    max_pairs
        Maximum node pairs to sample (seeded) for the scatter and the global
        rho. The local statistics always use every pair within *local_radius*.
    seed
        Random seed for pair sampling.
    figsize
        ``(width, height)`` in inches.
    title
        Axes title. Auto-generated with the statistics when ``None``.

    References
    ----------
    Venna & Kaski (2001) Neighborhood preservation in nonlinear projection
    methods, ICANN 2001, 485-491.
    """
    from scipy.stats import spearmanr

    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")
    if "weights" not in adata.uns[key]:
        raise KeyError(
            f"no 'weights' stored under uns['{key}'] — this result predates "
            "the codebook being saved; re-run the tl function that wrote it"
        )

    weights: np.ndarray = _get_result(adata, key)["weights"]
    sx, sy, n_feat = weights.shape
    codebook = weights.reshape(-1, n_feat).astype(np.float64)
    grid = np.array([(i, j) for i in range(sx) for j in range(sy)], dtype=np.float64)
    n_nodes = len(codebook)

    rng = np.random.default_rng(seed)
    n_all = n_nodes * (n_nodes - 1) // 2
    if n_all <= max_pairs:
        ii, jj = np.triu_indices(n_nodes, k=1)
    else:
        ii = rng.integers(0, n_nodes, max_pairs)
        jj = rng.integers(0, n_nodes, max_pairs)
        keep = ii != jj
        ii, jj = ii[keep], jj[keep]

    grid_d = np.linalg.norm(grid[ii] - grid[jj], axis=1)
    code_d = np.linalg.norm(codebook[ii] - codebook[jj], axis=1)
    rho_global = spearmanr(grid_d, code_d).statistic
    near = grid_d <= local_radius
    rho_local = (
        spearmanr(grid_d[near], code_d[near]).statistic if near.sum() > 2 else np.nan
    )

    # ---- trustworthiness / continuity at k = the local neighbourhood size ---
    # Full pairwise ranks on the node set (a few thousand nodes at most, so a
    # dense (n_nodes, n_nodes) matrix is fine here even though it never would be
    # on cells).
    from astrolabe.tl._som import _self_dists
    gd = _self_dists(grid)
    cd = _self_dists(codebook)
    np.fill_diagonal(gd, np.inf)
    np.fill_diagonal(cd, np.inf)
    k = max(1, int(np.median((gd <= local_radius).sum(axis=1))))
    k = min(k, n_nodes - 2)
    rank_g = np.argsort(np.argsort(gd, axis=1), axis=1)
    rank_c = np.argsort(np.argsort(cd, axis=1), axis=1)
    knn_g = rank_g < k
    knn_c = rank_c < k
    norm = n_nodes * k * (2 * n_nodes - 3 * k - 1) / 2.0
    trust = 1.0 - 2.0 * np.clip(rank_c[knn_g] - k, 0, None).sum() / norm
    cont = 1.0 - 2.0 * np.clip(rank_g[knn_c] - k, 0, None).sum() / norm

    params = _get_result(adata, key).get("params", {})
    qe, te = params.get("quantization_error"), params.get("topographic_error")

    fig, ax = plt.subplots(figsize=figsize)
    t = get_theme()
    ax.scatter(grid_d[~near], code_d[~near], s=4, alpha=0.10, color=t.muted,
               edgecolors="none", rasterized=True, label="distant pairs")
    ax.scatter(grid_d[near], code_d[near], s=6, alpha=0.25, color=t.cat_core[0],
               edgecolors="none", rasterized=True,
               label=f"pairs within {local_radius:g} nodes")
    ax.axvline(local_radius, color=t.primary_ink, lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("grid distance (nodes)")
    ax.set_ylabel("codebook distance")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=False, markerscale=2.0)
    _style_legend(leg)
    if title is None:
        title = (
            f"SOM topology: local rho = {rho_local:.2f} "
            f"(global {rho_global:.2f})\n"
            f"trustworthiness {trust:.3f}, continuity {cont:.3f}  (k={k})"
        )
        if qe is not None and te is not None:
            title += f"\nQE={qe:.3f}, TE={te:.3f}"
    ax.set_title(title, fontsize=9)

    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


@with_adata
def som_stability_plot(
    adata: ad.AnnData,
    *,
    key: str = "som_stability",
    figsize: tuple[float, float] = (9.5, 4.0),
    title: str | None = None,
) -> FigurePanel:
    """How many metaclusters does the map actually support?

    Draws the two curves that decide whether a reported cluster count is a
    finding or an arbitrary cut, from :func:`~astrolabe.tl.som_stability`:

    * **Left — reproducibility.** Mean pairwise adjusted Rand index between
      independently seeded SOM runs, per ``n_metaclusters``, with the ``k`` the
      run actually reports ringed and the flat region shaded. The curve almost
      always **declines** with ``k`` and then flattens -- more pieces means more
      boundaries the data does not fix -- so its maximum sits at the smallest
      ``k`` tested and carries no information. Read instead the ARI at your
      chosen ``k`` (is the partition you are about to quote reproducible?) and
      where the curve stops changing (beyond which extra clusters buy nothing
      stable).
    * **Right — Ward merge heights.** The variance cost of each successive
      merge of the node codebook. A pronounced jump means the merge below it
      joined genuinely distinct prototypes, so the count just above the jump is
      a natural stopping point; a smooth curve means the codebook is a
      continuum and no ``k`` is privileged — which is itself worth knowing
      before quoting one.

    Nothing in the SOM literature makes a single Ward cut self-justifying;
    FlowSOM's own protocol (Quintelier et al. 2021, *Nat Protoc*) inspects a
    range of ``k`` for exactly this reason.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key written by :func:`~astrolabe.tl.som_stability`.
    figsize
        ``(width, height)`` in inches.
    title
        Suptitle. Auto-generated when ``None``.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> ab.tl.som_stability(mdata, k_range=range(5, 26), n_seeds=3)
    >>> ab.pl.som_stability_plot(mdata).show()
    """
    if key not in adata.uns:
        raise KeyError(
            f"uns['{key}'] not found. Run ab.tl.som_stability() first"
        )
    st = _get_result(adata, key)
    ks = np.asarray(st["k"], dtype=float)
    ari = np.asarray(st["mean_ari"], dtype=float)
    ari_sd = np.asarray(st.get("sd_ari", np.zeros_like(ari)), dtype=float)
    merge_k = np.asarray(st["merge_k"], dtype=float)
    heights = np.asarray(st["merge_heights"], dtype=float)

    t = get_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    ax.errorbar(ks, ari, yerr=ari_sd, marker="o", ms=4, lw=1.6, capsize=2,
                color=t.cat_core[0])
    flat_a, flat_b = st.get("flat_from"), st.get("flat_to")
    if flat_a is not None and flat_b is not None and flat_b > flat_a:
        ax.axvspan(flat_a, flat_b, color=t.cat_core[0], alpha=0.08, lw=0,
                   label=f"flat: k={int(flat_a)}-{int(flat_b)}")
    ref_k, ref_ari = st.get("reference_k"), st.get("ari_at_reference_k")
    if ref_k is not None and ref_ari is not None and np.isfinite(ref_ari):
        # mark the k the run actually reports, so the figure answers the reader's
        # question instead of one about where the curve peaks
        ax.axvline(ref_k, color=t.primary_ink, lw=1.0, ls="--", alpha=0.55)
        ax.plot([ref_k], [ref_ari], marker="o", ms=9, mfc="none",
                mec=t.primary_ink, mew=1.5, zorder=5,
                label=f"reported k={int(ref_k)}: ARI {ref_ari:.2f}")
    if ax.get_legend_handles_labels()[1]:
        _style_legend(ax.legend(loc="upper right", fontsize=7.5, frameon=False))
    ax.set_xlabel("n_metaclusters")
    ax.set_ylabel("mean pairwise ARI across seeds")
    ax.set_ylim(0, 1.02)
    ax.set_title("Reproducibility of the cut", fontsize=10)
    _style_axes(ax, grid=True)

    ax = axes[1]
    kmax = (int(ks.max()) + 5) if len(ks) else int(merge_k.max())
    show = merge_k <= kmax
    ax.plot(merge_k[show], heights[show], marker="o", ms=3, lw=1.4,
            color=t.cat_core[1])
    ax.set_xlabel("clusters remaining after the merge")
    ax.set_ylabel("Ward merge height")
    ax.set_title("Cost of each merge", fontsize=10)
    _style_axes(ax, grid=True)

    if title is None:
        title = "SOM metacluster stability"
        ref_k, ref_ari = st.get("reference_k"), st.get("ari_at_reference_k")
        if ref_k is not None and ref_ari is not None and np.isfinite(ref_ari):
            title += f"  —  ARI {ref_ari:.2f} at the reported k = {int(ref_k)}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)
