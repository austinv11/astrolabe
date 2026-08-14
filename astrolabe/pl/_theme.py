"""
Light/dark plot theming for astrolabe.

Provides a colorblind-checked categorical palette (validated against the
project's dataviz methodology via ``node scripts/validate_palette.js`` --
see the plan history for the actual check output), sequential/diverging
colormaps built from the same palette, a bundled Inter typeface, and the
matplotlib rcParams that apply them -- so plots read as one system instead
of stock matplotlib defaults.

Usage
-----
>>> import astrolabe as ab
>>> ab.pl.set_theme("dark")            # global toggle, mirrors ab.use_gpu()
>>> with ab.pl.theme("dark"):          # scoped override, mirrors ab.gpu()
...     ab.pl.points(mdata, "leiden").show()
>>> ab.pl.get_theme().name
'light'
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend import Legend

__all__ = ["Theme", "set_theme", "get_theme", "theme"]

log = logging.getLogger(__name__)

_FONT_DIR = Path(__file__).parent / "_fonts"
_FONTS_REGISTERED = False

_SEQ_CMAP_NAME = "astrolabe_seq"
_DIV_CMAP_NAME = "astrolabe_div"

# Validated 8-hue categorical core (dataviz skill reference palette,
# remapped onto astrolabe's semantic roles -- see plan history for the
# node scripts/validate_palette.js runs that confirmed both modes pass all
# six checks: worst adjacent CVD dE 9.1 light / 8.4 dark, normal-vision
# floor 19.6 / 19.3, first 3 slots also clear --pairs all in both modes).
# Order is fixed and must never be reshuffled per-chart (see color-formula
# check 1); the astrolabe-specific slot roles are:
#   0 blue    primary / "down" / Moran LL
#   1 orange  Moran HL
#   2 aqua    tertiary
#   3 yellow  quaternary
#   4 magenta quinary
#   5 green   Moran LH
#   6 violet  senary
#   7 red     "up" / critical / Moran HH
#
# Slot 1 (orange) was re-picked from the skill's original reference value
# (light #eb6834 / dark #d95926): although the adjacent-pair sweep above
# passed, orange sat only ~16 degrees of OKLCH hue and ~0.05 L away from
# slot 7 (red) -- fine in isolation, but charts that plot both roles at
# once (gene_qv_scatter's control-probe categories, moran_plot's HH/HL
# quadrants) read as visually identical. `node scripts/validate_palette.js
# "<orange>,<yellow>,<red>" --mode <mode> --pairs all` against slot 1 vs.
# slots 3 and 7 (the two non-adjacent warm-cluster neighbors actually
# co-plotted) confirmed the collision (orange-vs-red normal-vision dE 7.1
# light / 7.1 dark, both below the 15 floor) and validated the fix: hue
# shifted to 58 deg light / 48 deg dark (still unambiguously orange, roughly
# equidistant from red and yellow) raised orange-vs-red normal-vision dE to
# 10.9 light / 11.4 dark and CVD dE to 6.8 light (WARN floor, legal since
# astrolabe legends always carry a text label) / 10.2 dark (clears the 8.0
# target outright). Re-ran full adjacent-pair and first-3-slots --pairs-all
# checks with the new value in place: both modes still ALL CHECKS PASS,
# identical worst-adjacent/normal-vision-floor numbers as before (9.1/8.4,
# 19.6/19.3) since slot 1's neighbors in the fixed order (blue, aqua) were
# unaffected.
_CAT_CORE_LIGHT: tuple[str, ...] = (
    "#2a78d6", "#e07500", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
_CAT_CORE_DARK: tuple[str, ...] = (
    "#3987e5", "#c25500", "#199e70", "#c98500",
    "#d55181", "#008300", "#9085e9", "#e66767",
)

# Sequential ramp (single blue hue, steps 100->700, light->dark). Mode-
# invariant: it fills its own colorbar rather than sitting on chart chrome.
_SEQ_STOPS: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)


@dataclass(frozen=True, slots=True)
class Theme:
    """A named light/dark chart theme: chrome colors + the categorical core.

    Not normally constructed directly -- use :func:`set_theme`/:class:`theme`
    to activate one and :func:`get_theme` to read the active instance.
    """

    name: Literal["light", "dark"]
    surface: str
    primary_ink: str
    secondary_ink: str
    muted: str
    gridline: str
    baseline: str
    div_neutral: str
    cat_core: tuple[str, ...]
    seq_cmap: str = _SEQ_CMAP_NAME
    div_cmap: str = _DIV_CMAP_NAME


_LIGHT = Theme(
    name="light", surface="#fcfcfb", primary_ink="#0b0b0b", secondary_ink="#52514e",
    muted="#898781", gridline="#e1e0d9", baseline="#c3c2b7", div_neutral="#f0efec",
    cat_core=_CAT_CORE_LIGHT,
)
_DARK = Theme(
    name="dark", surface="#1a1a19", primary_ink="#ffffff", secondary_ink="#c3c2b7",
    muted="#898781", gridline="#2c2c2a", baseline="#383835", div_neutral="#383835",
    cat_core=_CAT_CORE_DARK,
)

_CURRENT_THEME: Theme = _LIGHT


# ---------------------------------------------------------------------------
# Registration -- lazy so plain `import astrolabe` stays cheap
# ---------------------------------------------------------------------------


def _register_fonts_once() -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    for path in sorted(_FONT_DIR.glob("*.ttf")):
        fm.fontManager.addfont(str(path))
    _FONTS_REGISTERED = True
    log.info("Registered bundled Inter font files with matplotlib")


def _register_cmaps(t: Theme) -> None:
    """Register astrolabe's sequential/diverging colormaps for the active theme.

    Re-registered (``force=True``) on every :func:`set_theme` call rather
    than once, since the diverging ramp's neutral midpoint is mode-specific
    (blends with each theme's chart surface) while the sequential ramp is
    reused unmodified across modes.
    """
    seq = LinearSegmentedColormap.from_list(_SEQ_CMAP_NAME, _SEQ_STOPS)
    div = LinearSegmentedColormap.from_list(
        _DIV_CMAP_NAME, [t.cat_core[0], t.div_neutral, t.cat_core[7]]
    )
    # force=True lets set_theme() be called repeatedly (theme switches,
    # multiple kernels importing astrolabe); matplotlib still warns on
    # every re-registration even with force=True, which is expected here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Overwriting the cmap")
        mpl.colormaps.register(seq, name=_SEQ_CMAP_NAME, force=True)
        mpl.colormaps.register(div, name=_DIV_CMAP_NAME, force=True)


def _rcparams_for(t: Theme) -> dict:
    return {
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "text.color": t.primary_ink,
        "axes.labelcolor": t.primary_ink,
        "axes.titlecolor": t.primary_ink,
        "xtick.color": t.secondary_ink,
        "ytick.color": t.secondary_ink,
        "xtick.labelcolor": t.secondary_ink,
        "ytick.labelcolor": t.secondary_ink,
        "axes.edgecolor": t.baseline,
        "axes.linewidth": 0.8,
        "grid.color": t.gridline,
        "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=list(t.cat_core)),
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 300 is a floor, not just a default -- a FigurePanel's wrapped
        # content is rasterized once (see _render_figure_panel in
        # _spatial.py) and can end up displayed considerably larger than
        # its own declared figsize once dynamic sizing or a wide sibling
        # panel grows the grid; 150 visibly blurred under that stretch.
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }


# ---------------------------------------------------------------------------
# Public API -- mirrors astrolabe._gpu's use_gpu / gpu / gpu_enabled pattern
# ---------------------------------------------------------------------------


def set_theme(name: Literal["light", "dark"] = "light") -> None:
    """Activate *name* globally: rcParams, colormaps, and the bundled font.

    Parameters
    ----------
    name
        ``"light"`` or ``"dark"``.

    See Also
    --------
    theme : Context manager that restores the prior theme on exit.
    get_theme : Return the currently active :class:`Theme`.
    """
    global _CURRENT_THEME
    t = _LIGHT if name == "light" else _DARK
    _register_fonts_once()
    _register_cmaps(t)
    plt.rcParams.update(_rcparams_for(t))
    _CURRENT_THEME = t
    log.info("astrolabe plot theme set to %r", name)


def get_theme() -> Theme:
    """Return the currently active :class:`Theme`."""
    return _CURRENT_THEME


class theme:
    """Context manager: activate a theme for the duration of a ``with`` block.

    Restores the prior theme and rcParams on exit, including when nested,
    the same way :class:`astrolabe.gpu` restores the prior GPU flag state.

    Examples
    --------
    >>> with ab.pl.theme("dark"):
    ...     ab.pl.points(mdata, "leiden").show()
    """

    def __init__(self, name: Literal["light", "dark"] = "light"):
        self._name = name
        self._prev_theme: Theme | None = None
        self._prev_rcparams: dict | None = None

    def __enter__(self) -> "theme":
        self._prev_theme = _CURRENT_THEME
        self._prev_rcparams = dict(plt.rcParams)
        set_theme(self._name)
        return self

    def __exit__(self, *_exc) -> None:
        global _CURRENT_THEME
        plt.rcParams.update(self._prev_rcparams)
        _CURRENT_THEME = self._prev_theme


# ---------------------------------------------------------------------------
# High-cardinality categorical ramp -- best-effort, not CVD-validated
# ---------------------------------------------------------------------------


def _oklch_to_hex(L: float, C: float, h_deg: float) -> str:
    """Convert an OKLCH color to a ``'#rrggbb'`` hex string.

    Inverts Bjorn Ottosson's OKLab matrices (the same forward transform the
    dataviz skill's validator uses) to go OKLab -> linear sRGB, then gamma-
    encodes to sRGB. Out-of-gamut components are clamped rather than
    gamut-mapped, which is acceptable here since callers pick L/C values
    already known to be in-gamut across the hue circle (see
    :func:`_extended_cat_colors`).
    """
    h = math.radians(h_deg)
    a = C * math.cos(h)
    b = C * math.sin(h)

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3

    r_lin = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    def _encode(c: float) -> int:
        c = min(max(c, 0.0), 1.0)
        c = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(min(max(c, 0.0), 1.0) * 255)

    return "#{:02x}{:02x}{:02x}".format(_encode(r_lin), _encode(g_lin), _encode(b_lin))


def _extended_cat_colors(n: int, mode: Literal["light", "dark"]) -> list[str]:
    """Best-effort visually-distinct categorical colors for n > 8 categories.

    Sweeps OKLCH hue evenly at a fixed lightness/chroma per mode. This is
    NOT run through the six-check CVD validator (``scripts/validate_palette.js``)
    -- past ~8 simultaneous categories no hue assignment clears the
    colorblind-separation floor at all, so this ramp trades a validated
    guarantee for graceful degradation at high cardinality (e.g. Leiden
    clusters with 20-30+ groups). Prefer faceting / small multiples over
    relying on hue alone once ``n`` is large.
    """
    L, C = (0.62, 0.13) if mode == "light" else (0.72, 0.13)
    offset = 15.0  # avoid opening on the core palette's blue (~255 deg)
    return [_oklch_to_hex(L, C, (offset + i * 360.0 / n) % 360.0) for i in range(n)]


# ---------------------------------------------------------------------------
# Shared styling helpers -- for chrome rcParams can't reach
# ---------------------------------------------------------------------------

# Finalized chrome spec (user-selected comparison, see plan history):
# Structured's open frame + gridlines on statistical (non-spatial) axes,
# combined with Minimal's dot-marker legend swatches and SemiBold titles.
#
# A *list*, not a bare string: many astrolabe titles use symbols Inter
# doesn't ship (nabla, times, etc.). matplotlib only does per-glyph font
# fallback across a family *list* passed to set_fontfamily/fontfamily= --
# a single explicit family name bypasses the font.sans-serif rcParam
# fallback chain entirely and renders those glyphs as tofu/missing.
TITLE_FONT = ["Inter SemiBold", "Segoe UI", "Arial", "DejaVu Sans"]


def _style_axes(ax, *, grid: bool = False) -> None:
    """Apply theme chrome that active rcParams can't express.

    Facecolor, spine color/visibility, tick/label color, and body font
    already apply automatically at axes-creation time via the active
    rcParams; this covers what isn't rcParam-driven:

    - re-fonts any existing title to :data:`TITLE_FONT` (matplotlib has no
      rcParam for a title-specific font family distinct from body text),
      so callers keep writing plain ``ax.set_title(...)`` and get the
      SemiBold treatment for free by calling this helper afterward;
    - normalizes tick length/width;
    - optionally adds a faint y-axis gridline, for statistical plots
      (histograms, embedding scatters, dot/heat maps) where it aids reading
      values -- left off by default since spatial (micron-space) panels
      intentionally have none, matching the library's existing convention.
    """
    t = get_theme()
    if ax.get_title():
        ax.title.set_fontfamily(TITLE_FONT)
        ax.title.set_color(t.primary_ink)
    ax.tick_params(length=3, width=0.8)
    if grid:
        ax.yaxis.grid(True, color=t.gridline, linewidth=0.6)
        ax.set_axisbelow(True)


def _legend_handles(labels, colors) -> list:
    """Build dot-marker legend handles (the finalized swatch style).

    Used in place of ``matplotlib.patches.Patch`` swatches everywhere
    astrolabe builds a categorical legend.
    """
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6, color=c, label=str(lbl))
        for lbl, c in zip(labels, colors)
    ]


def _style_legend(leg) -> None:
    """Theme a Legend built with an explicit ``frameon=False`` call.

    Every astrolabe legend call passes ``frameon=False`` explicitly, which
    bypasses ``legend.*`` rcParam defaults for the frame itself -- so text
    color has to be set imperatively too.
    """
    if leg is None:
        return
    t = get_theme()
    for text in leg.get_texts():
        text.set_color(t.primary_ink)


def _style_colorbar(cbar) -> None:
    """Theme a Colorbar (no rcParam controls its outline/tick chrome)."""
    if cbar is None:
        return
    t = get_theme()
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=t.secondary_ink, labelcolor=t.secondary_ink, length=3, width=0.8)
    # matplotlib's default vertical-colorbar label rotation (90 deg) reads
    # bottom-to-top; flipped 180 deg (270 deg) reads top-to-bottom instead.
    cbar.ax.yaxis.label.set_rotation(270)
    cbar.ax.yaxis.label.set_va("bottom")


# ---------------------------------------------------------------------------
# Colorbar/legend coordination -- dodge a legend past any colorbar already
# claiming the axes' right margin, and fix the "two overlapping legends" bug
# ---------------------------------------------------------------------------

# Fallback dodge (axes-fraction) used only if a real measurement isn't
# available (e.g. a backend with no renderer yet). Deliberately conservative
# -- a fixed guess can't know about a colorbar's label text (which can be
# rotated 270 degrees and extend well past the tick labels, as astrolabe's
# colorbars are -- see _style_colorbar) or long tick-label content, so it
# previously produced real overlap on real data (a colorbar labeled with a
# gene name next to a boundaries legend). The real path below measures the
# colorbar's actual rendered extent, label included, instead of guessing.
_COLORBAR_DODGE_FALLBACK = 0.3
_LEGEND_COLORBAR_GAP_PT = 8.0  # breathing room between colorbar label and legend


def _register_colorbar(ax, cbar):
    """Style *cbar* and record it so a legend placed later via
    :func:`_place_legend` can measure its real extent and dodge past it.

    Safe to call more than once per axes (e.g. two continuous layers
    stacked with ``+``); each call accumulates, matching how matplotlib's
    own automatic ``ax=`` placement chains multiple colorbars further right
    of each other.
    """
    _style_colorbar(cbar)
    state = getattr(ax, "_ab_layout_state", None)
    if state is None:
        state = {}
        ax._ab_layout_state = state
    state.setdefault("colorbars", []).append(cbar)
    return cbar


_MAX_LEGEND_ENTRIES = 50  # above this, a legend is almost certainly a mis-chosen high-cardinality column, not a usable key


def _place_legend(ax, handles, labels):
    """Build the single legend for *ax*, dodged past any registered colorbar.

    Callers must accumulate all ``(handle, label)`` pairs for an axes and
    call this exactly once, after every layer/colorbar on that axes has been
    drawn -- see ``_render_panel`` in ``_spatial.py``. Having one call site
    for legend construction (rather than each layer type building its own)
    is what prevents two layers from independently drawing overlapping
    legends on the same axes.

    The dodge amount is measured, not guessed: a colorbar's true rendered
    width (label included -- which for astrolabe's rotated label can extend
    well past its tick labels) is only known after a layout pass.

    Every anchor-based approach that left the legend participating in
    ``constrained_layout`` (used by both ``SpatialPanel.to_figure`` and
    ``PlotGrid.to_figure``) failed, including iterating toward the target
    with damped corrections: an artist anchored via ``ax.transAxes`` at a
    fraction past the axes' own box gets read by constrained_layout as "the
    axes needs to be this wide," which inflates padding *between the axes
    and its own colorbar* to compensate -- observed directly as the main
    plot pulled ~10x further from its colorbar than the legend gap actually
    required, regardless of how carefully the target anchor was computed.

    The fix is to stop asking constrained_layout to account for the legend
    at all: ``set_in_layout(False)`` excludes it from that bookkeeping, so
    positioning it wherever the measurement says (even past the axes'
    edge) has zero effect on anything else's layout. The tradeoff is that
    ``in_layout=False`` also excludes it from ``bbox_inches="tight"``
    cropping (used by both ``.save()`` and Jupyter's SVG display), which
    would otherwise cut the legend out of the saved/displayed image -- so
    it's registered on the figure for callers to pass via
    ``bbox_extra_artists`` instead (see ``_Renderable.save``/
    ``_repr_mimebundle_`` in ``_spatial.py``).
    """
    if not handles:
        return None
    if len(handles) > _MAX_LEGEND_ENTRIES:
        log.warning(
            "Skipping legend with %d entries (> %d) "
            "Pass a coarser grouping, or build a custom legend, "
            "if you need one here.",
            len(handles), _MAX_LEGEND_ENTRIES,
        )
        return None
    state = getattr(ax, "_ab_layout_state", {})
    colorbars = state.get("colorbars") or []
    anchor_x = 1.01

    if colorbars:
        fig = ax.get_figure()
        try:
            # Two passes: constrained_layout's axes/colorbar positions are
            # usually stable by the second draw (confirmed empirically --
            # see plan history). No legend exists yet at this point, so
            # there's nothing here to trigger the runaway described above.
            for _ in range(2):
                fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_bbox = ax.get_window_extent(renderer)
            gap_px = _LEGEND_COLORBAR_GAP_PT * fig.dpi / 72.0
            max_right_px = max(cbar.ax.get_tightbbox(renderer).x1 for cbar in colorbars)
            anchor_x = 1.0 + (max_right_px + gap_px - ax_bbox.x1) / ax_bbox.width
        except Exception:
            log.debug("Could not measure colorbar extent for legend dodge; using fallback.", exc_info=True)
            anchor_x = 1.01 + _COLORBAR_DODGE_FALLBACK

    leg = Legend(
        ax, handles, labels,
        bbox_to_anchor=(anchor_x, 1),
        bbox_transform=ax.transAxes,
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    leg.set_clip_on(False)
    if colorbars:
        leg.set_in_layout(False)
        fig = ax.get_figure()
        fig._ab_extra_artists = getattr(fig, "_ab_extra_artists", [])
        fig._ab_extra_artists.append(leg)
    ax.add_artist(leg)
    _style_legend(leg)
    return leg
