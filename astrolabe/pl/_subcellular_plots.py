"""Plots for geometry-anchored subcellular localization results.

Three figures, matching the three things :mod:`astrolabe.tl._subcellular`
produces: the continuous radial profile (what a gene's localization actually
looks like), the gene x compartment effect matrix (which genes and where), and
the transcripts-inside-outlines view (the raw evidence, one cell at a time).
"""

from __future__ import annotations

from astrolabe._dependencies import _get_result

import logging

from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import pandas as pd

from astrolabe._utils import _resolve_adata
from astrolabe.pl._spatial import _DEFAULT_DIVERGING_CMAP, FigurePanel
from astrolabe.pl._theme import (
    _style_axes,
    _style_colorbar,
    _style_legend,
    get_theme,
)
from astrolabe.tl._subcellular import COMPARTMENTS

log = logging.getLogger(__name__)

__all__ = [
    "subcellular_profile",
    "subcellular_enrichment_plot",
    "subcellular_plot",
]

def _compartment_colors(n: int = 4) -> list[str]:
    """Sequential ramp for the compartments.

    The four compartments are an *ordered* variable (nucleus -> periphery), so
    they get a sequential ramp rather than a categorical palette -- a
    categorical palette would imply the order carries no meaning, which is
    exactly backwards here.
    """
    import matplotlib as mpl

    cmap = mpl.colormaps["astrolabe_seq"]
    return [mpl.colors.to_hex(cmap(t)) for t in np.linspace(0.18, 0.92, n)]


def _shade_compartments(ax, bins, colors, alpha=0.045):
    """Tint the u axis by compartment so bands are readable without a legend."""
    edges = [0.0, *bins, 2.0]
    for i, name in enumerate(COMPARTMENTS):
        ax.axvspan(edges[i], edges[i + 1], color=colors[i], alpha=alpha,
                   lw=0, zorder=0)


def _compact(n) -> str:
    """Short transcript count for an axis label: 1200 -> 1.2k, 342404 -> 342k."""
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(n)}"


def _null_density(adata, coords_key, bins):
    """Uniform-in-polygon density of ``u``, resampled onto *bins*.

    Read from ``uns[coords_key]['null_profile']``, which
    :func:`~astrolabe.tl.subcellular_coords` accumulates over every interior
    pixel of every cell -- the same coordinate transform the transcripts went
    through, applied to exactly uniform placement.
    """
    try:
        prof = np.asarray(adata.uns[coords_key]["null_profile"], dtype=np.float64)
    except (KeyError, TypeError):
        raise KeyError(
            f"uns['{coords_key}']['null_profile'] not found -- it is written by "
            "ab.tl.subcellular_coords(). Re-run it, or pass reference='panel' "
            "to compare against the pooled panel instead."
        ) from None
    # Integrate, do not point-sample. The stored profile is a fine-binned
    # histogram; evaluating it at each target bin's centre with np.interp reads
    # one noisy source bin instead of the average over the target bin, which
    # leaves a ratio that is neither flat nor centred on 1. Differencing the
    # source CDF at the target edges gives the exact mass per target bin for any
    # binning.
    src_edges = np.linspace(0.0, 2.0, len(prof) + 1)
    cdf = np.concatenate([[0.0], np.cumsum(prof)])
    mass = np.diff(np.interp(bins, src_edges, cdf))
    width = np.diff(bins)
    # Bins the null barely reaches (the innermost slivers of the nucleus, where
    # the area element vanishes) would divide two near-zero numbers and throw a
    # spike across the whole y-axis. Blank them instead: there is no evidence
    # there to plot either way.
    floor = 0.002 * float(mass.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(width > 0, mass / width, np.nan)
    return np.where(mass >= floor, dens, np.nan)


def _require_coords(coords):
    if coords is None:
        raise ValueError(
            "`coords` is required -- pass the DataFrame returned by "
            "ab.tl.subcellular_coords(). It is not stored on the object "
            "because it has one row per transcript."
        )
    return coords


def subcellular_profile(
    data: ad.AnnData | md.MuData,
    coords: pd.DataFrame,
    *,
    modality: str | None = None,
    genes: Sequence[str] | None = None,
    group: str | None = None,
    reference: str = "null",
    n_bins: int = 60,
    show_baseline: bool = True,
    coords_key: str = "subcellular",
    palette: Sequence[str] | None = None,
    figsize: tuple[float, float] = (5.6, 4.6),
    title: str | None = None,
    ax=None,
) -> FigurePanel:
    """Radial enrichment along the intracellular coordinate, per gene.

    The x axis *is* the cell, running from the nucleus centre (``u = 0``)
    through the nuclear envelope (``u = 1``) to the plasma membrane
    (``u = 2``), and each curve is one gene's transcript distribution along it.

    **What is plotted depends on ``reference``, and the default is a ratio, not
    a density.** ``u`` is a *radial fraction*, so its area element is not
    constant: inside the nucleus, area grows with ``u``, and transcripts placed
    uniformly at random already produce a curve that rises to a peak at the
    envelope. Plotting raw density therefore invites reading that peak as
    perinuclear accumulation when it is geometry. Dividing by the
    uniform-in-polygon expectation cancels it exactly, so a flat line at 1.0
    means "no different from uniform" and every departure is real.

    The grey curve is the **whole panel** pooled. It is the reference the
    headline test compares against, and it is worth reading in its own right:
    where it departs from 1.0, the panel as a whole departs from uniform --
    partly real nuclear retention, partly the 3-D-to-2-D projection and
    segmentation truncation that no per-cell geometry can undo. A gene tracking
    the grey curve is unlocalized *relative to the panel*, which is what the
    cell-stratified test measures, and those shared distortions cancel there.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    coords
        The DataFrame returned by :func:`~astrolabe.tl.subcellular_coords`.
    modality
        Modality key when *data* is a MuData.
    genes
        Restrict the drawn transcripts to these genes. The cells drawn do not
        depend on this, so two panels with different `genes` and the same
        `seed`/`max_cells` show the same cells and are directly comparable.
        Genes to draw. ``None`` draws only the pooled baseline, which is a
        useful figure in its own right (it shows where transcripts sit in
        general, and how thin the cytoplasmic shell is).
    reference
        What each curve is divided by.

        - ``'null'`` (default) -- the uniform-in-polygon expectation stored by
          :func:`~astrolabe.tl.subcellular_coords` in
          ``uns[coords_key]['null_profile']``. A flat line at 1.0 is uniform.
        - ``'panel'`` -- the pooled all-gene density, so the baseline is flat at
          1.0 by construction and each curve is exactly the quantity the
          cell-stratified test evaluates.
        - ``'density'`` -- no division; raw transcript density, dominated by the
          area element described above.
    group
        When *coords* carries a ``group`` column, restrict to this group. Both
        the gene curves and the baseline are then computed within it, so the
        comparison stays internal to one cell type. Compose panels with ``|``
        to put two cell types side by side.
    n_bins
        Histogram bins across ``[0, 2]``.
    show_baseline
        Draw the pooled all-gene density.
    coords_key
        ``uns`` key written by :func:`~astrolabe.tl.subcellular_coords`, read
        for the compartment cut points used to shade the bands.
    palette
        Colors for the gene curves. ``None`` uses the theme categorical palette.
    figsize, title, ax
        Standard figure controls. Passing *ax* draws into an existing axes and
        the returned panel wraps its figure.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> coords = ab.tl.subcellular_coords(mdata)
    >>> ab.pl.subcellular_profile(mdata, coords, genes=['ACTA2', 'CAPN8']).show()
    >>> (ab.pl.subcellular_profile(mdata, coords, genes=['ACTA2'], group='0')
    ...  | ab.pl.subcellular_profile(mdata, coords, genes=['ACTA2'], group='1')).show()
    """
    from astrolabe.pl._theme import _extended_cat_colors

    adata = _resolve_adata(data, modality)
    _require_coords(coords)

    if group is not None:
        if "group" not in coords.columns:
            raise ValueError(
                "group= was given but `coords` has no 'group' column -- pass "
                "groupby= to ab.tl.subcellular_coords()."
            )
        coords = coords[coords["group"].astype(str) == str(group)]
        if len(coords) == 0:
            raise ValueError(f"group {group!r} has no transcripts")

    if reference not in ("null", "panel", "density"):
        raise ValueError(
            f"reference must be 'null', 'panel' or 'density', got {reference!r}"
        )

    bins = np.linspace(0.0, 2.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    t = get_theme()
    comp_colors = _compartment_colors(len(COMPARTMENTS))

    cut = (0.7, 1.3, 1.7)
    if coords_key in adata.uns:
        cut = tuple(adata.uns[coords_key]["params"].get("compartment_bins", cut))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _shade_compartments(ax, cut, comp_colors)

    u_all = coords["u"].to_numpy(dtype=np.float64)
    pooled_h, _ = np.histogram(u_all, bins=bins, density=True)

    denom = np.ones_like(pooled_h)
    if reference == "null":
        denom = _null_density(adata, coords_key, bins)
    elif reference == "panel":
        denom = pooled_h
    safe = np.where(denom > 1e-12, denom, np.nan)

    if reference != "density":
        ax.axhline(1.0, color=t.primary_ink, lw=0.9, alpha=0.4, zorder=1)

    if show_baseline and reference != "panel":
        y = pooled_h / safe
        ax.fill_between(centers, np.ones_like(y), y, color=t.muted, alpha=0.30,
                        lw=0, zorder=1,
                        label=f"all genes (n={len(u_all):,})")
        ax.plot(centers, y, color=t.muted, lw=1.4, zorder=2)

    gene_list = list(genes) if genes is not None else []
    if gene_list:
        if palette is None:
            palette = (
                list(t.cat_core) if len(gene_list) <= 8
                else _extended_cat_colors(len(gene_list), t.name)
            )
        gcol = coords["gene"].astype(str).to_numpy()
        for i, g in enumerate(gene_list):
            m = gcol == g
            if not m.any():
                raise ValueError(f"gene {g!r} has no transcripts in `coords`")
            h, _ = np.histogram(u_all[m], bins=bins, density=True)
            ax.plot(centers, h / safe, color=palette[i % len(palette)], lw=2.0,
                    zorder=3, label=f"{g} (n={int(m.sum()):,})")

    # the nuclear envelope is the one interior landmark; 0 and 2 are the ends
    ax.axvline(1.0, color=t.primary_ink, lw=1.0, ls="--", alpha=0.55, zorder=4)

    ax.set_xlim(0.0, 2.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("radial position u   (0 nucleus centre, 1 envelope, 2 membrane)")
    ax.set_ylabel(
        {
            "null": "enrichment vs uniform placement",
            "panel": "enrichment vs the whole panel",
            "density": "transcript density",
        }[reference]
    )
    # Compartment names go on a secondary top axis rather than inside the data
    # area: they are axis annotations, and drawn in-axes they collide with the
    # legend on exactly the curves whose only free space is an upper corner.
    edges = [0.0, *cut, 2.0]
    top = ax.secondary_xaxis("top")
    top.set_xticks([(edges[i] + edges[i + 1]) / 2.0
                    for i in range(len(COMPARTMENTS))])
    top.set_xticklabels([n.replace("_", " ") for n in COMPARTMENTS],
                        fontsize=6.5, color=t.muted)
    top.tick_params(axis="x", length=0, pad=1.5)
    for side in top.spines.values():
        side.set_visible(False)
    if title is None:
        title = "Subcellular radial profile"
        if group is not None:
            title += f"  ({group})"
    ax.set_title(title, fontsize=10)
    # 'best' rather than a fixed corner: null-referenced curves can sit anywhere
    # from 0 to well above 1, so no single corner is reliably empty
    leg = ax.legend(loc="best", fontsize=7.5, framealpha=0.75,
                    facecolor=t.surface, edgecolor="none")
    _style_legend(leg)
    _style_axes(ax)
    if own_fig:
        fig.tight_layout()
        plt.close(fig)
    return FigurePanel(fig)


def subcellular_enrichment_plot(
    data: ad.AnnData | md.MuData,
    result: pd.DataFrame | None = None,
    *,
    modality: str | None = None,
    key: str = "subcellular_enrichment",
    genes: Sequence[str] | None = None,
    n_top: int = 16,
    min_transcripts: int = 2000,
    value: str = "cmh_log_or",
    alpha: float = 0.05,
    group: str | None = None,
    clip_percentile: float = 98.0,
    annotate_significant: bool = True,
    cmap: str = _DEFAULT_DIVERGING_CMAP,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Gene x compartment effect matrix from :func:`~astrolabe.tl.subcellular_enrichment`.

    Four named columns, so there is nothing to decode and no room for a
    position-bin artifact to masquerade as a per-gene pattern.

    Three choices worth knowing about, because each prevents a specific way of
    misreading this figure:

    * **Genes are ranked by effect, not by p-value.** With millions of
      transcripts the smallest p-values belong to the most abundant genes, so a
      p-ranked figure is a picture of the panel's abundance distribution.
    * **Both tails are shown.** Half the rows come from each end of the
      *peripheral-minus-nuclear* contrast, and rows are ordered along it. Taking
      the top ``n_top`` by *absolute* effect would fill the figure from whichever
      tail happens to be stronger -- on a typical panel the nuclear one -- and
      every row would then look alike, which reads as though nothing
      gene-specific had been found.
    * **Color limits are a robust percentile, not the maximum**, so one extreme
      cell cannot flatten everything else to the midpoint.

    ``min_transcripts`` is the fourth: the largest odds ratios often belong to
    barely-detected genes, and one with a few hundred transcripts leading the
    figure is a distraction rather than a finding.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    result
        The DataFrame from :func:`~astrolabe.tl.subcellular_enrichment`. ``None``
        reads ``uns[key]``.
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key holding the result when *result* is ``None``.
    genes
        Explicit genes to show, in the given order. Overrides *n_top*.
    n_top
        Number of genes to show when *genes* is ``None``, split evenly between
        the nucleus-biased and membrane-biased tails.
    min_transcripts
        Ignore genes with fewer transcripts than this when choosing rows
        (explicit *genes* are always honoured). ``0`` disables it.
    value
        ``'cmh_log_or'`` (default) plots the cell-stratified Mantel--Haenszel
        log odds ratio -- the headline effect size. ``'log2_vs_pooled'`` plots
        the simple pooled fold-change over the panel, which ignores the cell
        stratification and is included for comparison.
    alpha
        Significance threshold on ``p_adj_omnibus`` for gene selection, and for
        the per-cell significance markers.
    group
        Restrict to one ``group`` value when the result carries groups.
    clip_percentile
        Percentile of ``|value|`` used for the symmetric color limits.
    annotate_significant
        Ring the cells that **fail** ``p_cmh_* < alpha``. Marking the passes would
        put a dot on nearly every cell at this scale; the exceptions are the
        informative ones.
    cmap, figsize, title
        Standard figure controls.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> res = ab.tl.subcellular_enrichment(mdata, coords)
    >>> ab.pl.subcellular_enrichment_plot(mdata).show()
    """
    adata = _resolve_adata(data, modality)
    if result is None:
        if key not in adata.uns:
            raise KeyError(
                f"uns['{key}'] not found. Run ab.tl.subcellular_enrichment() first"
            )
        result = _get_result(adata, key)
    res = pd.DataFrame(result).copy()
    if group is not None:
        if "group" not in res.columns:
            raise ValueError("group= was given but the result has no 'group' column")
        res = res[res["group"].astype(str) == str(group)]

    if value not in ("cmh_log_or", "log2_vs_pooled"):
        raise ValueError(
            f"value must be 'cmh_log_or' or 'log2_vs_pooled', got {value!r}"
        )
    cols = [f"{value}_{c}" for c in COMPARTMENTS]
    missing = [c for c in cols if c not in res.columns]
    if missing:
        raise KeyError(f"result is missing columns {missing}")

    M = res[cols].to_numpy(dtype=np.float64)
    axis = M[:, -1] - M[:, 0]  # peripheral minus nuclear
    res = res.assign(_axis=axis)

    if genes is not None:
        want = [str(g) for g in genes]
        sel = res.set_index(res["gene"].astype(str)).reindex(want)
        present = sel[cols[0]].notna().to_numpy()
        missing = [g for g, ok in zip(want, present) if not ok]
        if len(missing) == len(want):
            raise ValueError("none of the requested genes are in the result")
        if missing:
            log.warning(
                "subcellular_enrichment_plot: dropping %d requested gene(s) "
                "absent from the result (below min_gene_transcripts, or in a "
                "different group): %s",
                len(missing), ", ".join(missing[:8]),
            )
            sel = sel.loc[[g for g in want if g not in set(missing)]]
        order = sel
    else:
        pool = res[res["p_adj_omnibus"] < alpha] if "p_adj_omnibus" in res else res
        if min_transcripts and "n_transcripts" in pool.columns:
            powered = pool[pool["n_transcripts"] >= min_transcripts]
            if len(powered) >= 2:
                pool = powered
            else:
                log.warning(
                    "subcellular_enrichment_plot: only %d gene(s) clear "
                    "min_transcripts=%d; showing the unfiltered set instead",
                    len(powered), min_transcripts,
                )
        pool = pool[np.isfinite(pool["_axis"].to_numpy())]
        if len(pool) == 0:
            raise ValueError(
                f"no genes pass p_adj_omnibus < {alpha} with a finite effect; "
                f"pass genes= explicitly"
            )
        # half from each tail: one tail alone makes every row look alike
        half = max(1, n_top // 2)
        low = pool.nsmallest(min(half, len(pool)), "_axis")
        high = pool.nlargest(min(half, len(pool)), "_axis")
        picked = pd.concat([low, high]).drop_duplicates(subset=["gene"])
        order = picked.sort_values("_axis")
        order = order.set_index(order["gene"].astype(str))

    M = order[cols].to_numpy(dtype=np.float64)
    labels = list(order.index)
    n_g = len(labels)

    finite = M[np.isfinite(M)]
    lim = float(np.percentile(np.abs(finite), clip_percentile)) if finite.size else 1.0
    lim = max(lim, 1e-6)
    # +-inf is a real result (a gene absent from, or confined to, a compartment);
    # render it at the limit rather than as a hole, and hatch it so it reads as
    # saturated rather than merely large.
    inf_mask = np.isinf(M)
    Mp = np.clip(np.nan_to_num(M, nan=np.nan, posinf=lim, neginf=-lim), -lim, lim)

    if figsize is None:
        figsize = (5.2, max(3.4, 0.32 * n_g + 1.8))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Mp, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                   origin="upper", interpolation="nearest")
    t = get_theme()
    label = ("MH log odds ratio vs other transcripts in the same cell"
             if value == "cmh_log_or" else "log2 observed / pooled panel")
    _style_colorbar(fig.colorbar(im, ax=ax, shrink=0.85, label=label))

    ax.set_xticks(range(len(COMPARTMENTS)))
    ax.set_xticklabels([c.replace("_", "\n") for c in COMPARTMENTS], fontsize=8)
    ax.set_yticks(range(n_g))
    if "n_transcripts" in order.columns:
        nt = order["n_transcripts"].to_numpy()
        ylabels = [f"{g}  ({_compact(n)})" for g, n in zip(labels, nt)]
    else:
        ylabels = labels
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel("compartment")

    for i in range(n_g):
        for j in range(len(COMPARTMENTS)):
            if inf_mask[i, j]:
                ax.text(j, i, "∞" if M[i, j] > 0 else "0", ha="center",
                        va="center", fontsize=7, color=t.surface, fontweight="bold")
            elif not np.isfinite(M[i, j]):
                ax.text(j, i, "·", ha="center", va="center", fontsize=9,
                        color=t.muted)
    if annotate_significant:
        pcols = [f"p_cmh_{c}" for c in COMPARTMENTS]
        if all(c in order.columns for c in pcols):
            P = order[pcols].to_numpy(dtype=np.float64)
            # Mark the cells that DO NOT pass. At this scale almost everything is
            # significant, so marking the passes would put a dot on every cell and
            # carry no information; the exceptions are what a reader needs to see.
            yy, xx = np.where(~(np.isfinite(P) & (P < alpha)) & ~inf_mask)
            ax.scatter(xx, yy, s=14, facecolors="none", edgecolors=t.surface,
                       linewidths=0.9, zorder=3)

    if title is None:
        title = "Subcellular localization"
        if group is not None:
            title += f"  ({group})"
    ax.set_title(title, fontsize=10)
    _style_axes(ax)
    ax.grid(False)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)


def subcellular_plot(
    data: ad.AnnData | md.MuData,
    coords: pd.DataFrame,
    *,
    modality: str | None = None,
    genes: Sequence[str] | None = None,
    max_cells: int | None = None,
    cell_boundaries_key: str = "cell_boundaries",
    nucleus_boundaries_key: str = "nucleus_boundaries",
    spatial_key: str = "spatial",
    point_size: float = 6.0,
    palette: Sequence[str] | None = None,
    legend: bool = True,
    seed: int = 0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> FigurePanel:
    """Transcripts inside their cell and nucleus outlines, colored by compartment.

    The raw evidence behind the other two figures. Both outlines are drawn:
    without the nucleus there is no way to see whether a "nuclear" call is
    really nuclear, which makes the nucleus the single most important element
    on this figure.

    Pass *genes* to see one gene's localization. Drawing every gene at once
    shows only the aggregate radial gradient -- decorative, and easy to
    over-read -- so the default is to require a choice for anything but a
    general look at the data.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    coords
        The DataFrame returned by :func:`~astrolabe.tl.subcellular_coords`.
    modality
        Modality key when *data* is a MuData.
    genes
        Restrict to these genes. ``None`` draws all of them.
    max_cells
        Randomly sample at most this many cells (seeded). ``None`` draws every
        cell present in *coords*, which is only readable on a small crop.
    cell_boundaries_key, nucleus_boundaries_key
        ``obsm`` keys for the two polygon sets.
    spatial_key
        ``obsm`` key for cell centroids (um), used to frame the axes.
    point_size
        Transcript marker size.
    palette
        Four colors for the compartments. ``None`` uses the sequential ramp.
    legend
        Draw the compartment legend.
    seed
        Cell-sampling seed.
    figsize, title
        Standard figure controls.

    Returns
    -------
    FigurePanel

    Examples
    --------
    >>> crop = ab.pp.crop(mdata, rect, inplace=False)
    >>> ab.pl.subcellular_plot(crop, coords, genes=['ACTA2']).show()
    """
    import awkward as ak
    import matplotlib.collections as mc
    import matplotlib.patches as mp_patches

    adata = _resolve_adata(data, modality)
    for k in (cell_boundaries_key, spatial_key):
        if k not in adata.obsm:
            raise KeyError(f"obsm['{k}'] not found")
    has_nuc = nucleus_boundaries_key in adata.obsm

    # Choose the cells from the WHOLE table, then restrict the transcripts to
    # `genes`. Sampling after the gene filter would give every gene its own cell
    # set and its own axis limits, so two panels could not be compared -- which
    # is the main thing this figure is composed to do.
    cd = coords
    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    present = np.array([c for c in pd.unique(cd["cell_id"].astype(str))
                        if c in obs_index])
    if len(present) == 0:
        raise ValueError(
            "no cell_id in `coords` matches obs_names -- if you cropped the "
            "object, crop it before computing coords or pass the full object"
        )
    if max_cells and len(present) > max_cells:
        rng = np.random.default_rng(seed)
        present = rng.choice(present, size=max_cells, replace=False)
    keep = set(present.tolist())
    cd = cd[cd["cell_id"].astype(str).isin(keep)]
    idx = np.array([obs_index[c] for c in present])

    if genes is not None:
        want = {str(g) for g in genes}
        cd = cd[cd["gene"].astype(str).isin(want)]
        if len(cd) == 0:
            raise ValueError(
                f"none of the {len(present)} cells drawn hold a transcript of "
                f"{sorted(want)!r}; raise max_cells, or crop to a region where "
                "the gene is detected"
            )

    t = get_theme()
    colors = list(palette) if palette is not None else _compartment_colors(
        len(COMPARTMENTS)
    )

    cent = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[idx]
    pad = 8.0
    xlim = (cent[:, 0].min() - pad, cent[:, 0].max() + pad)
    ylim = (cent[:, 1].min() - pad, cent[:, 1].max() + pad)
    if figsize is None:
        span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
        figsize = (max(5.0, min(11.0, span_x / 12.0)),
                   max(4.5, min(10.0, span_y / 12.0)))

    fig, ax = plt.subplots(figsize=figsize)

    def _rings(key, color, lw, ls):
        patches = []
        raw = adata.obsm[key]
        for i in idx:
            try:
                verts = np.asarray(ak.to_regular(raw[i]))
            except Exception:
                verts = np.asarray(raw[i])
            if verts.ndim != 2 or verts.shape[0] < 3:
                continue
            patches.append(mp_patches.Polygon(verts[:, :2], closed=True))
        if patches:
            ax.add_collection(mc.PatchCollection(
                patches, facecolors="none", edgecolors=color, linewidths=lw,
                linestyles=ls, zorder=1,
            ))

    _rings(cell_boundaries_key, t.primary_ink, 0.8, "solid")
    if has_nuc:
        _rings(nucleus_boundaries_key, t.muted, 0.8, (0, (2, 1.5)))

    comp = cd["compartment"].astype(str).to_numpy()
    xs = cd["x"].to_numpy(dtype=np.float64)
    ys = cd["y"].to_numpy(dtype=np.float64)
    for i, name in enumerate(COMPARTMENTS):
        m = comp == name
        if m.any():
            ax.scatter(xs[m], ys[m], s=point_size, c=[colors[i]], alpha=0.85,
                       linewidths=0, zorder=2, label=name)

    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    if title is None:
        gtxt = ", ".join(str(g) for g in genes) if genes else "all genes"
        title = f"Subcellular transcripts — {gtxt} ({len(present)} cells)"
    ax.set_title(title, fontsize=10)
    if legend:
        # below the axes: cells fill the frame, so any in-axes corner overlaps
        # either the outlines or the transcripts
        leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
                        ncol=len(COMPARTMENTS), fontsize=7, frameon=False,
                        markerscale=2.0, columnspacing=1.2, handletextpad=0.4)
        _style_legend(leg)
    _style_axes(ax)
    fig.tight_layout()
    plt.close(fig)
    return FigurePanel(fig)
