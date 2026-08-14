"""Tests for astrolabe.pl._spatial: aspect-aware PlotGrid sizing, colorbar
labeling, and legend/colorbar coordination (dodge + duplicate-legend fix).
"""
import matplotlib
matplotlib.use("Agg")  # headless backend: avoid Tk backend flakiness in CI/sandboxes

import awkward as ak
import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import pytest
import tifffile

import astrolabe as ab
from astrolabe._utils import ImageSpec
from astrolabe.pl._spatial import (
    FigurePanel,
    PlotGrid,
    _apply_registration_rotation,
    _auto_point_size,
    _cell_aspect,
    _layer_extent,
    _load_tiff_level,
    _pad_to_aspect,
    _panel_extent,
    _render_figure_panel,
)


def _make_adata(rng, n=30, wide=False):
    xy = (
        np.stack([rng.uniform(0, 3000, n), rng.uniform(0, 500, n)], axis=1)
        if wide
        else rng.uniform(0, 500, (n, 2))
    ).astype(np.float32)
    obs = pd.DataFrame(
        {
            "leiden": rng.choice(["0", "1", "2"], n).astype(str),
            "total_counts": rng.uniform(1, 100, n).astype(np.float32),
        },
        index=[f"c{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    import scipy.sparse as sp
    X = sp.random(n, 5, density=0.5, format="csr", random_state=0).astype(np.float32)
    adata = ad.AnnData(X=X, obs=obs, obsm={"spatial": xy}, var=var)
    return adata


def _add_boundaries(adata, key="cell_boundaries", size=5.0):
    xy = adata.obsm["spatial"]
    polys = [
        [[cx - size, cy - size], [cx + size, cy - size], [cx + size, cy + size], [cx - size, cy + size]]
        for cx, cy in xy
    ]
    adata.obsm[key] = ak.Array(polys)


@pytest.fixture
def wide_adata():
    return _make_adata(np.random.default_rng(0), wide=True)


@pytest.fixture
def square_adata():
    return _make_adata(np.random.default_rng(1), wide=False)


# ---------------------------------------------------------------------------
# Extent / aspect-ratio helpers
# ---------------------------------------------------------------------------

def test_panel_extent_wide_vs_square_aspect(wide_adata, square_adata):
    wide_panel = ab.pl.points(wide_adata, "total_counts")
    square_panel = ab.pl.points(square_adata, "total_counts")

    wide_aspect = _cell_aspect(wide_panel)
    square_aspect = _cell_aspect(square_panel)

    assert wide_aspect > 3.0
    assert 0.5 < square_aspect < 2.0


def test_layer_extent_matches_obsm_bounds(square_adata):
    lay = ab.pl.points(square_adata, "total_counts").layers[0]
    ext = _layer_extent(lay)
    coords = square_adata.obsm["spatial"]
    assert ext == (
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
    )


def test_panel_extent_respects_crop_bounds(square_adata):
    square_adata.uns["spatial_bounds"] = {"xmin": 10.0, "xmax": 20.0, "ymin": 30.0, "ymax": 35.0}
    panel = ab.pl.points(square_adata, "total_counts")
    assert _panel_extent(panel) == (10.0, 20.0, 30.0, 35.0)


def test_panel_extent_unions_multiple_layers(square_adata):
    _add_boundaries(square_adata)
    combo = ab.pl.points(square_adata, "total_counts") + ab.pl.boundaries(square_adata, alpha=0.3)
    ext = _panel_extent(combo)
    assert ext is not None


def test_cell_aspect_none_for_none_cell():
    assert _cell_aspect(None) is None


def test_cell_aspect_figure_panel_assumed_equal_regardless_of_figsize():
    """FigurePanel is deliberately NOT introspected for aspect -- its declared
    figsize isn't a trustworthy read on the actual plotted content (e.g. a
    legend can widen the exported raster well past a square axes), so the
    grid falls back to assuming 1.0 for it rather than reading get_size_inches().
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 2))
    panel = FigurePanel(fig)
    assert _cell_aspect(panel) is None
    plt.close(fig)


def test_figure_panels_only_grid_gets_equal_boxes(square_adata):
    """A row of only FigurePanel cells (no SpatialPanel) falls back to
    equal-aspect boxes, matching the 'assume equal aspect ratio' rule.
    """
    import matplotlib.pyplot as plt
    fig_a, _ = plt.subplots(figsize=(8, 2))
    fig_b, _ = plt.subplots(figsize=(2, 8))
    grid = ab.pl.as_panel(fig_a) | ab.pl.as_panel(fig_b)
    fig = grid.to_figure()
    width_ratios = fig.axes[0].get_gridspec().get_width_ratios()
    assert width_ratios[0] == pytest.approx(width_ratios[1])
    plt.close(fig)
    plt.close(fig_a)
    plt.close(fig_b)


# ---------------------------------------------------------------------------
# FigurePanel rasterization -- pad to fit, never stretch
# ---------------------------------------------------------------------------

def test_pad_to_aspect_widens_a_too_tall_image():
    img = np.ones((100, 100, 4), dtype=np.float32)  # square source
    padded = _pad_to_aspect(img, target_aspect=2.0)  # needs to become wide
    h, w = padded.shape[:2]
    assert w / h == pytest.approx(2.0, rel=0.02)
    # original square content preserved unstretched in the center
    assert h == 100


def test_pad_to_aspect_heightens_a_too_wide_image():
    img = np.ones((50, 200, 4), dtype=np.float32)  # aspect 4.0
    padded = _pad_to_aspect(img, target_aspect=1.0)
    h, w = padded.shape[:2]
    assert w / h == pytest.approx(1.0, rel=0.02)
    assert w == 200


def test_pad_to_aspect_noop_for_matching_aspect():
    img = np.ones((100, 150, 4), dtype=np.float32)  # aspect 1.5
    padded = _pad_to_aspect(img, target_aspect=1.5)
    assert padded.shape == img.shape


def test_pad_to_aspect_uses_supplied_fill_color_not_transparent():
    """A transparent pad reveals the destination axes' own facecolor behind
    it; if that doesn't exactly match the raster's own (opaque) background,
    or antialiasing blurs the seam under stretching, the boundary shows as
    a visible line. Padding with the theme surface color as an opaque fill
    avoids that regardless of what's behind the axes.
    """
    img = np.ones((100, 100, 4), dtype=np.float32)
    fill = (0.2, 0.4, 0.6, 1.0)
    padded = _pad_to_aspect(img, target_aspect=2.0, fill_rgba=fill)
    # New columns on the padded sides must be the opaque fill, not transparent.
    assert tuple(padded[0, 0]) == pytest.approx(fill)
    assert padded[0, 0, 3] == 1.0  # fully opaque, not alpha=0


def test_render_figure_panel_pads_legend_widened_figure_instead_of_stretching():
    """A figure with an internal legend gets bbox_inches='tight'-cropped wider
    than its declared square figsize; rendered into a square grid cell, the
    plot itself must not be squeezed -- the raster should be padded to the
    assigned cell's aspect, not stretched.
    """
    import matplotlib.pyplot as plt

    src_fig, src_ax = plt.subplots(figsize=(5, 5))
    for cat, color in zip(["0", "1", "2", "3"], ["C0", "C1", "C2", "C3"]):
        src_ax.scatter([0, 1], [0, 1], color=color, label=cat)
    src_ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)

    dest_fig, dest_ax = plt.subplots(figsize=(5, 5))
    _render_figure_panel(dest_ax, src_fig, target_aspect=1.0)
    img = dest_ax.images[0].get_array()
    h, w = img.shape[:2]
    assert w / h == pytest.approx(1.0, rel=0.02)

    plt.close(src_fig)
    plt.close(dest_fig)


# ---------------------------------------------------------------------------
# PlotGrid.to_figure sizing
# ---------------------------------------------------------------------------

def test_dynamic_sizing_gives_wide_panel_more_width(wide_adata, square_adata):
    grid = ab.pl.points(wide_adata, "total_counts") | ab.pl.points(square_adata, "total_counts")
    fig = grid.to_figure()
    width_ratios = fig.axes[0].get_gridspec().get_width_ratios()
    assert width_ratios[0] > width_ratios[1] * 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dynamic_sizing_false_reproduces_equal_boxes(wide_adata, square_adata):
    grid = ab.pl.points(wide_adata, "total_counts") | ab.pl.points(square_adata, "total_counts")
    fig = grid.to_figure(dynamic_sizing=False)
    width_ratios = fig.axes[0].get_gridspec().get_width_ratios()
    assert width_ratios[0] == width_ratios[1] == 1.0
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_explicit_figsize_override_wins(wide_adata, square_adata):
    grid = ab.pl.points(wide_adata, "total_counts") | ab.pl.points(square_adata, "total_counts")
    fig = grid.to_figure(figsize=(12.0, 4.0))
    assert tuple(fig.get_size_inches()) == pytest.approx((12.0, 4.0))
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_single_row_grid_grows_figure_instead_of_distorting_square_panel(wide_adata):
    """A wide panel next to a square FigurePanel: the square one must render
    at true 1:1 aspect (no stretch), so the figure's total size grows rather
    than squeezing the square panel's column into a fixed total width.
    """
    import matplotlib.pyplot as plt

    square_fig, square_ax = plt.subplots(figsize=(5, 5))
    square_ax.scatter([0, 1], [0, 1])
    plt.close(square_fig)

    grid = ab.pl.points(wide_adata, "total_counts") | ab.pl.as_panel(square_fig)
    fig = grid.to_figure()
    gridspec = fig.axes[0].get_gridspec()
    width_ratios = gridspec.get_width_ratios()
    height_ratios = gridspec.get_height_ratios()

    fig_w, fig_h = fig.get_size_inches()
    square_col_width = fig_w * width_ratios[1] / sum(width_ratios)
    square_col_height = fig_h * height_ratios[0] / sum(height_ratios)
    assert square_col_width == pytest.approx(square_col_height, rel=0.02)

    # The legacy (equal-box) total width for a 1x2 grid at the default
    # per-panel scale is ncols * per_w == 10; the figure must grow past
    # that to give the wide panel real proportional width without
    # shrinking the square panel below its own true aspect.
    assert fig_w > 10.0
    plt.close(fig)


def test_colorbar_label_rotated_180_from_default(square_adata):
    fig = ab.pl.points(square_adata, "total_counts").to_figure()
    cbar_ax = fig.axes[-1]
    assert cbar_ax.yaxis.label.get_rotation() == 270
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_uniform_aspect_grid_matches_legacy_total_size():
    """Two panels with an exactly-square extent should render at the same
    total size regardless of dynamic_sizing (aspect 1.0 either way).
    """
    rng = np.random.default_rng(42)
    exact_square = _make_adata(rng, n=30, wide=False)
    # Pin the extent to exactly [0, 500] x [0, 500] so aspect is exactly 1.0,
    # not just approximately (random uniform coords rarely span the full
    # range on both axes identically).
    exact_square.obsm["spatial"][0] = [0.0, 0.0]
    exact_square.obsm["spatial"][1] = [500.0, 500.0]

    grid = ab.pl.points(exact_square, "total_counts") | ab.pl.points(exact_square, "total_counts")
    fig_dynamic = grid.to_figure()
    fig_static = grid.to_figure(dynamic_sizing=False)
    assert tuple(fig_dynamic.get_size_inches()) == pytest.approx(tuple(fig_static.get_size_inches()))
    import matplotlib.pyplot as plt
    plt.close(fig_dynamic)
    plt.close(fig_static)


# ---------------------------------------------------------------------------
# Colorbar labels
# ---------------------------------------------------------------------------

def test_points_colorbar_is_labeled(square_adata):
    fig = ab.pl.points(square_adata, "total_counts").to_figure()
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_ylabel() == "total_counts"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_boundaries_continuous_colorbar_is_labeled(square_adata):
    _add_boundaries(square_adata)
    fig = ab.pl.boundaries(square_adata, color="total_counts").to_figure()
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_ylabel() == "total_counts"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_raster_aggregate_colorbar_is_labeled(square_adata):
    fig = ab.pl.raster(square_adata, color="total_counts", bins=4).to_figure()
    cbar_ax = fig.axes[-1]
    assert cbar_ax.get_ylabel() == "total_counts"
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Legend/colorbar dodge + duplicate-legend fix
# ---------------------------------------------------------------------------

def test_colorbar_and_legend_do_not_share_anchor(square_adata):
    _add_boundaries(square_adata)
    combo_with_cbar = ab.pl.points(square_adata, color="total_counts") + ab.pl.boundaries(square_adata, alpha=0.3)
    combo_without_cbar = ab.pl.boundaries(square_adata, alpha=0.3)

    fig_with = combo_with_cbar.to_figure()
    fig_without = combo_without_cbar.to_figure()

    def legend_anchor_x(fig):
        ax = fig.axes[0]
        legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
        assert len(legends) == 1
        return legends[0].get_bbox_to_anchor().transformed(ax.transAxes.inverted()).x0

    anchor_with_cbar = legend_anchor_x(fig_with)
    anchor_without_cbar = legend_anchor_x(fig_without)
    assert anchor_with_cbar > anchor_without_cbar, "legend should dodge further right when a colorbar is present"

    import matplotlib.pyplot as plt
    plt.close(fig_with)
    plt.close(fig_without)


def test_legend_actually_clears_a_labeled_colorbar_in_pixel_space(square_adata):
    """Regression test for a real overlap seen with real data: a colorbar
    labeled with a long field name (rotated 270 degrees, so it extends past
    the tick labels -- see _style_colorbar) needs more dodge room than a
    fixed heuristic constant could know about. The dodge must be measured
    from the colorbar's actual rendered extent (get_tightbbox, label
    included), not guessed -- check the legend and colorbar bounding boxes
    don't overlap in real pixel space, not just that the anchor moved.

    Note: matplotlib's constrained_layout solver can emit a benign
    "collapsed to zero" UserWarning on an intermediate internal pass while
    reserving room for a long rotated label in a small default figsize --
    confirmed via direct inspection that the final resolved axes width is
    always healthy regardless (this is solver noise, not a real collapse).
    """
    _add_boundaries(square_adata)
    square_adata.obs["a_long_field_name_like_a_real_gene"] = square_adata.obs["total_counts"]
    combo = (
        ab.pl.points(square_adata, color="a_long_field_name_like_a_real_gene")
        + ab.pl.boundaries(square_adata, alpha=0.3)
    )
    fig = combo.to_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legends) == 1
    legend_bbox = legends[0].get_window_extent(renderer)
    cbar_ax = fig.axes[-1]
    cbar_bbox = cbar_ax.get_tightbbox(renderer)

    assert legend_bbox.x0 >= cbar_bbox.x1, "legend must clear the colorbar's full extent, including its label"

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dodged_legend_survives_a_real_show_call(square_adata):
    """Regression test for a real bug: a legend dodged past a colorbar is
    excluded from constrained_layout via set_in_layout(False) (see
    _place_legend), which also excludes it from bbox_inches="tight" cropping
    unless bbox_extra_artists is passed. .save()/_repr_mimebundle_ do this,
    but .show() originally called plain plt.show() -- which in a real
    Jupyter kernel goes through IPython's own inline-backend rendering,
    bypassing that fix entirely and silently cropping the legend out of
    every .show()-displayed figure with a colorbar. Simulates the actual
    notebook path by intercepting IPython.display.publish_display_data.
    """
    import IPython.display

    _add_boundaries(square_adata)
    combo = ab.pl.points(square_adata, color="total_counts") + ab.pl.boundaries(square_adata, alpha=0.3)

    captured = {}

    def fake_publish(data=None, **kwargs):
        captured["data"] = data

    original = IPython.display.publish_display_data
    IPython.display.publish_display_data = fake_publish
    try:
        combo.show()
    finally:
        IPython.display.publish_display_data = original

    assert "data" in captured, "show() must publish display data when IPython is available"
    svg = captured["data"]["image/svg+xml"]
    assert "Cell Boundaries" in svg, "the dodged legend must survive the actual .show() display path"


def test_no_colorbar_no_dodge_needed(square_adata):
    _add_boundaries(square_adata)
    panel = ab.pl.boundaries(square_adata, alpha=0.3)
    fig = panel.to_figure()
    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legends) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_categorical_points_plus_boundaries_single_legend(square_adata):
    _add_boundaries(square_adata)
    combo = ab.pl.points(square_adata, color="leiden") + ab.pl.boundaries(square_adata, alpha=0.3)
    fig = combo.to_figure()
    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legends) == 1, "categorical points + boundaries must not double-draw the legend"
    labels = [t.get_text() for t in legends[0].get_texts()]
    assert "Cell Boundaries" in labels
    assert {"0", "1", "2"}.issubset(set(labels))
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_points_categorical_alone_still_has_one_legend(square_adata):
    fig = ab.pl.points(square_adata, color="leiden").to_figure()
    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legends) == 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_points_categorical_draws_one_interleaved_collection_not_per_category():
    # Regression: _plot_categorical used to issue one ax.scatter() call per
    # category in `sorted(set(categories))` (alphabetical) order, so later
    # calls painted directly over earlier ones wherever points overlapped on
    # screen. On a dense plot this made whichever category sorted last (e.g.
    # "LL" among HH/HL/LH/LL quadrant labels) look artificially dominant, and
    # could make a category with a large true share of points (e.g. ~28%)
    # look almost entirely absent just because it sorted before others drawn
    # on top of it -- independent of actual abundance. Now all categories are
    # interleaved into a single shuffled ax.scatter() call, so no category
    # systematically occludes another. Pinning "exactly one collection" (not
    # one per category) is a direct proxy for "no fixed per-category
    # z-order."
    n = 200
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    obs = pd.DataFrame({"cat": rng.choice(["A", "B", "C", "D"], n).astype(str)})
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = xy

    fig = ab.pl.points(adata, color="cat").to_figure()
    ax = fig.axes[0]
    scatter_collections = [
        c for c in ax.collections if hasattr(c, "get_offsets") and len(c.get_offsets()) > 0
    ]
    assert len(scatter_collections) == 1
    assert len(scatter_collections[0].get_offsets()) == n
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Vertical stacking (/) -- same sizing logic, orthogonal axis
# ---------------------------------------------------------------------------

def test_vstack_figure_panel_content_not_stretched(wide_adata):
    """Stacking a spatial panel above a legend-bearing FigurePanel with '/'
    must not stretch the FigurePanel's content, even though (unlike the '|'
    case) matplotlib's constrained_layout doesn't leave it perfectly square --
    the padding step must use the axes' true resolved aspect, not a nominal
    estimate, so the raster is padded to fit rather than squeezed/stretched.
    """
    import matplotlib.pyplot as plt

    umap_fig, umap_ax = plt.subplots(figsize=(5, 5))
    umap_ax.scatter([0, 1, 2], [0, 1, 2], label="0")
    umap_ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)

    grid = ab.pl.points(wide_adata, "total_counts") / ab.pl.as_panel(umap_fig)
    fig = grid.to_figure()

    figure_panel_ax = fig.axes[1]
    box = figure_panel_ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    box_aspect = (box.width * fig_w) / (box.height * fig_h)

    img = figure_panel_ax.images[0].get_array()
    h, w = img.shape[:2]
    img_aspect = w / h

    # The rasterized content's aspect must match the box it's displayed in
    # (imshow(aspect="auto") always fills the box) -- so equality here means
    # the padding step correctly targeted the box's real aspect and no
    # stretching occurred, regardless of how close that box is to square.
    assert img_aspect == pytest.approx(box_aspect, rel=0.01)

    plt.close(fig)
    plt.close(umap_fig)


# ---------------------------------------------------------------------------
# Point marker size -- must scale with the axes' actual physical size, not
# just n_cells (a panel compressed by dynamic sizing renders at a very
# different µm-per-point scale than a standalone default panel)
# ---------------------------------------------------------------------------

def test_auto_point_size_shrinks_for_a_compressed_axes_box():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    coords = np.stack([rng.uniform(0, 3200, 600), rng.uniform(0, 550, 600)], axis=1)

    fig_normal, ax_normal = plt.subplots(figsize=(5, 5))
    size_normal = _auto_point_size(ax_normal, coords, 600)

    fig_compressed, ax_compressed = plt.subplots(figsize=(5, 6))
    ax_compressed.set_position([0.1, 0.1, 0.7, 0.13])  # a short, wide box
    size_compressed = _auto_point_size(ax_compressed, coords, 600)

    assert size_compressed < size_normal / 2
    plt.close(fig_normal)
    plt.close(fig_compressed)


def test_points_layer_markers_shrink_in_a_vstack_next_to_a_figure_panel():
    """Regression test for the bug this fixes directly: a wide spatial panel
    compressed short by '/' next to a square FigurePanel used to render with
    markers sized for a standalone panel, completely overlapping and making
    the plot uninterpretable. Uses a denser dataset than the wide_adata
    fixture so marker size isn't dominated by the upper clamp in both cases.
    """
    import matplotlib.pyplot as plt

    dense_wide = _make_adata(np.random.default_rng(3), n=600, wide=True)

    umap_fig, umap_ax = plt.subplots(figsize=(5, 5))
    umap_ax.scatter([0, 1, 2], [0, 1, 2])
    plt.close(umap_fig)

    standalone = ab.pl.points(dense_wide, "total_counts").to_figure()
    standalone_size = standalone.axes[0].collections[0].get_sizes()[0]
    plt.close(standalone)

    grid = ab.pl.points(dense_wide, "total_counts") / ab.pl.as_panel(umap_fig)
    fig = grid.to_figure()
    compressed_size = fig.axes[0].collections[0].get_sizes()[0]

    assert compressed_size < standalone_size / 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# _load_tiff_level channel-axis normalization
# ---------------------------------------------------------------------------

def _write_rgb_tiff(path, shape=(48, 32, 3)):
    """A channels-last (YXC) RGB tiff, matching H&E's on-disk layout."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=shape, dtype=np.uint8)
    tifffile.imwrite(str(path), arr, photometric="rgb")


def test_load_tiff_level_normalizes_channels_last_to_channel_first(tmp_path):
    # Regression test: dimension_order='YXC' (H&E's on-disk layout) used to be
    # returned as-is by _load_tiff_level despite its own contract claiming
    # channel-first -- downstream code (_apply_registration_rotation etc.)
    # then treated image ROWS as channels.
    path = tmp_path / "rgb.tif"
    _write_rgb_tiff(path, shape=(48, 32, 3))

    raw, ch_ax = _load_tiff_level(str(path), "YXC", level_idx=0)

    assert ch_ax == 0
    assert raw.shape == (3, 48, 32)


def test_load_tiff_level_channel_first_input_unaffected(tmp_path):
    # dimension_order already CYX (fluorescence/morphology convention): no
    # move should be needed, and the result must be unchanged.
    path = tmp_path / "cyx.tif"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(4, 48, 32), dtype=np.uint8)
    tifffile.imwrite(str(path), arr, photometric="minisblack")

    raw, ch_ax = _load_tiff_level(str(path), "CYX", level_idx=0)

    assert ch_ax == 0
    assert raw.shape == (4, 48, 32)


# ---------------------------------------------------------------------------
# Registration rotation (rotated registration_affine, e.g. H&E images)
# ---------------------------------------------------------------------------


def _rotated_spec(angle_deg=90.0, pixel_size_um=1.0):
    # A pure-rotation-by-angle_deg (2, 2) linear part, no scale/reflection --
    # mirrors the real H&E imagealignment.csv structure used in production
    # (verified against the public Human Lung Cancer FFPE sample).
    theta = np.radians(angle_deg)
    L = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.eye(3)
    A[:2, :2] = L
    return ImageSpec(
        path="unused.tif",
        pixel_size_um=pixel_size_um,
        dimension_order="CYX",
        full_shape=(1, 40, 30),
        level_shapes=[(1, 40, 30)],
        level_scales=[(1.0, 1.0)],
        registration_affine=A,
    )


def test_registration_rotation_places_marker_at_correct_real_world_position():
    # Regression test: scipy.ndimage.rotate's rotation-direction convention
    # (row-down array indexing) is the opposite sense from this renderer's
    # origin='lower' (row-up) display convention. Without negating the angle,
    # a marker pixel lands at the diagonally opposite corner of the viewport
    # from where the affine says it should (a 180 degree point-reflection).
    from astrolabe._utils import _transform_coords

    spec = _rotated_spec(angle_deg=90.0)
    raw = np.zeros((1, 40, 30), dtype=np.float64)
    marker_row, marker_col = 2, 2  # near the (col=0, row=0) corner
    raw[:, marker_row, marker_col] = 100.0

    A = spec.registration_affine
    gt_um = _transform_coords(np.array([[float(marker_col), float(marker_row)]]), A)[0]

    rotated, extent = _apply_registration_rotation(raw, spec, level_idx=0, crop_pixels=None)
    assert extent is not None

    idx = np.unravel_index(np.argmax(rotated[0]), rotated[0].shape)
    rh, rw = rotated.shape[1], rotated.shape[2]
    xmin, xmax, ymin, ymax = extent
    x_disp = xmin + (idx[1] + 0.5) / rw * (xmax - xmin)
    y_disp = ymin + (idx[0] + 0.5) / rh * (ymax - ymin)

    assert x_disp == pytest.approx(gt_um[0], abs=2.0)
    assert y_disp == pytest.approx(gt_um[1], abs=2.0)


def test_registration_rotation_no_crop_extent_matches_full_image_footprint():
    # Regression test: the no-crop branch used to multiply already
    # full-resolution h_full/w_full by the pyramid level's yscale/xscale a
    # second time, wildly inflating the extent whenever the whole image was
    # rendered without a prior pp.crop() (spatial_bounds unset).
    from astrolabe.pl._spatial import _compute_extent, _spec_yx_full_shape

    spec = _rotated_spec(angle_deg=90.0, pixel_size_um=0.5)
    h_full, w_full = _spec_yx_full_shape(spec)
    expected_extent = _compute_extent(spec, h_full, w_full)

    raw = np.zeros((1, 40, 30), dtype=np.float64)
    _, extent = _apply_registration_rotation(raw, spec, level_idx=0, crop_pixels=None)

    for got, want in zip(extent, expected_extent):
        assert got == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------------------
# Brightfield RGB rendering
# ---------------------------------------------------------------------------

def _brightfield_adata(tmp_path):
    path = tmp_path / "he.tif"
    _write_rgb_tiff(path, shape=(64, 64, 3))
    spec = ImageSpec(
        path=str(path),
        pixel_size_um=0.5,
        dimension_order="YXC",
        full_shape=(64, 64, 3),
        level_shapes=[(64, 64, 3)],
        level_scales=[(1.0, 1.0)],
        channel_names=["hematoxylin", "eosin", "residual"],
        is_brightfield=True,
    )
    adata = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        var=pd.DataFrame(index=pd.Index(["hematoxylin", "eosin", "residual"])),
        uns={"spatial": {"sample": {"images": {"full": spec}}}},
    )
    return adata


def test_brightfield_image_default_has_no_dark_background(tmp_path):
    adata = _brightfield_adata(tmp_path)
    fig = ab.pl.image(adata).to_figure()
    ax = fig.axes[0]
    # _apply_dark_theme sets both the axes and figure patch to black.
    assert ax.get_facecolor()[:3] != (0.0, 0.0, 0.0)


def test_brightfield_image_default_has_no_channel_legend(tmp_path):
    adata = _brightfield_adata(tmp_path)
    fig = ab.pl.image(adata).to_figure()
    ax = fig.axes[0]
    legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert legends == []


def test_dark_bg_explicit_override_still_wins_for_brightfield(tmp_path):
    adata = _brightfield_adata(tmp_path)
    fig = ab.pl.image(adata, dark_bg=True).to_figure()
    ax = fig.axes[0]
    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)


def test_non_brightfield_image_keeps_dark_background_default():
    rng = np.random.default_rng(0)
    spec = ImageSpec(
        path="unused.tif",
        pixel_size_um=0.5,
        dimension_order="CYX",
        full_shape=(2, 64, 64),
        level_shapes=[(2, 64, 64)],
        level_scales=[(1.0, 1.0)],
        channel_names=["ch0", "ch1"],
    )
    adata = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        var=pd.DataFrame(index=pd.Index(["ch0", "ch1"])),
        uns={"spatial": {"sample": {"images": {"full": spec}}}},
    )
    # Don't actually render (no real tiff on disk) -- just check the resolved
    # dark_bg default that image() computes before drawing.
    from astrolabe.pl._spatial import image as pl_image
    panel = pl_image(adata)
    assert panel.layers[0].dark_bg is True


def _non_brightfield_adata(tmp_path, name="ch"):
    path = tmp_path / f"{name}.tif"
    rng = np.random.default_rng(0)
    tifffile.imwrite(str(path), rng.integers(0, 255, size=(64, 64), dtype=np.uint8))
    spec = ImageSpec(
        path=str(path),
        pixel_size_um=0.5,
        dimension_order="YX",
        full_shape=(64, 64),
        level_shapes=[(64, 64)],
        level_scales=[(1.0, 1.0)],
    )
    return ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        uns={"spatial": {"sample": {"images": {"full": spec}}}},
    )


def _dark_background_rects(fig):
    """Black Rectangle artists added by _add_dark_panel_backgrounds."""
    import matplotlib.patches

    return [
        a for a in fig.artists
        if isinstance(a, matplotlib.patches.Rectangle) and a.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    ]


class TestDarkThemeDoesNotLeakBetweenComposedPanels:
    # Regression: _apply_dark_theme used to set fig.patch.set_facecolor("black")
    # unconditionally -- correct for a single dark-themed image panel, but a
    # composed `|` figure shares one fig across multiple axes, so this leaked
    # a black figure background onto every sibling panel too, stranding a
    # light-themed sibling's title/tick/label text (styled for a light
    # surface) on a black background. A first fix (stop touching fig.patch
    # in the composed case) then exposed a second problem: the dark panel's
    # *own* title/tick/legend text sits outside its axes' own black plot
    # rectangle, in shared figure-level space, so it stranded itself on the
    # now-light gutter the same way. Per-element text bboxes (a second,
    # since-reverted fix) covered title/ticks but not the legend, and looked
    # like literal black boxes floating on a light background rather than
    # one coherent dark panel. The real fix: one black rectangle behind each
    # dark axes' full *tight* bbox (title + ticks + labels + legend), added
    # to the figure after layout is resolved -- each panel renders with its
    # own complete, correct style, independent of what any sibling panel does.
    def test_light_panel_keeps_light_background_and_dark_text_next_to_dark_panel(self, tmp_path):
        dark_adata = _non_brightfield_adata(tmp_path, "morph")
        light_adata = _brightfield_adata(tmp_path)

        panel = ab.pl.image(dark_adata) | ab.pl.image(light_adata)
        fig = panel.to_figure()

        dark_ax, light_ax = fig.axes
        assert dark_ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
        assert light_ax.get_facecolor()[:3] != (0.0, 0.0, 0.0)
        # Light panel's own chrome must stay dark-on-light (legible), not
        # inherit the dark panel's white-on-black styling.
        assert light_ax.title.get_color() != "white"
        assert fig.get_facecolor()[:3] != (0.0, 0.0, 0.0)

    def test_dark_panel_gets_a_background_rectangle_covering_its_full_chrome(self, tmp_path):
        dark_adata = _non_brightfield_adata(tmp_path, "morph")
        light_adata = _brightfield_adata(tmp_path)

        panel = ab.pl.image(dark_adata) | ab.pl.image(light_adata)
        fig = panel.to_figure()
        dark_ax, light_ax = fig.axes

        rects = _dark_background_rects(fig)
        assert len(rects) == 1, "expected exactly one dark-panel background rectangle"
        rect = rects[0]

        # The rectangle (in figure-fraction coords) must fully contain the
        # dark axes' own title -- i.e. its chrome, not just its plot area --
        # and must not extend into the light panel's own position.
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        title_bbox = dark_ax.title.get_window_extent(renderer).transformed(inv)
        rx0, ry0 = rect.get_xy()
        rx1, ry1 = rx0 + rect.get_width(), ry0 + rect.get_height()
        assert rx0 <= title_bbox.x0 and title_bbox.x1 <= rx1
        assert ry0 <= title_bbox.y0 and title_bbox.y1 <= ry1

        light_pos = light_ax.get_position()
        assert rx1 <= light_pos.x0 + 1e-6

    def test_single_dark_panel_gets_a_background_rectangle_too(self, tmp_path):
        dark_adata = _non_brightfield_adata(tmp_path, "morph")
        fig = ab.pl.image(dark_adata).to_figure()
        assert len(_dark_background_rects(fig)) == 1

    def test_no_dark_panel_means_no_background_rectangle(self, tmp_path):
        light_adata = _brightfield_adata(tmp_path)
        fig = ab.pl.image(light_adata).to_figure()
        assert _dark_background_rects(fig) == []


# ---------------------------------------------------------------------------
# pl.annotation() categorical/continuous coloring
# ---------------------------------------------------------------------------

def _annotation_gdf():
    import geopandas as gpd
    from shapely.geometry import Polygon

    return gpd.GeoDataFrame(
        {
            "name": ["Tumor", "Tumor", "Stroma"],
            "score": [1.0, 5.0, 3.0],
        },
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
            Polygon([(40, 40), (50, 40), (50, 50), (40, 50)]),
        ],
    )


def test_annotation_literal_color_list_of_arrays_unchanged():
    roi = ab.pl.rectangle((5, 5), 4, 4)
    fig = ab.pl.annotation([roi], color="yellow").to_figure()
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    assert ax.get_legend() is None


def test_annotation_categorical_column_produces_legend_per_category():
    gdf = _annotation_gdf()
    fig = (ab.pl.annotation(gdf, color="name")).to_figure()
    ax = fig.axes[0]
    legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert len(legends) == 1
    labels = {t.get_text() for t in legends[0].get_texts()}
    assert labels == {"Tumor", "Stroma"}


def test_annotation_continuous_column_produces_colorbar():
    gdf = _annotation_gdf()
    fig = (ab.pl.annotation(gdf, color="score")).to_figure()
    ax = fig.axes[0]
    assert len(fig.axes) == 2  # main axes + colorbar axes


def test_annotation_multipolygon_row_gets_one_legend_entry():
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    gdf = gpd.GeoDataFrame(
        {"name": ["Vessel"]},
        geometry=[MultiPolygon([
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (30, 20), (30, 30), (20, 30)]),
        ])],
    )
    fig = ab.pl.annotation(gdf, color="name").to_figure()
    ax = fig.axes[0]
    legends = [c for c in ax.get_children() if isinstance(c, matplotlib.legend.Legend)]
    assert len(legends) == 1
    assert len(legends[0].get_texts()) == 1
    assert legends[0].get_texts()[0].get_text() == "Vessel"


def test_annotation_gdf_with_nonmatching_color_falls_back_to_literal():
    gdf = _annotation_gdf()
    fig = ab.pl.annotation(gdf, color="magenta").to_figure()
    ax = fig.axes[0]
    assert ax.get_legend() is None
    assert len(ax.collections) == 1


def test_brightfield_named_channel_applies_real_deconvolution_not_raw_channel(tmp_path):
    # Regression test: requesting channel='hematoxylin' on an H&E image with
    # a stain_matrix set used to just display the raw (mislabeled) RGB plane
    # at that index. It should now differ from the raw channel.
    path = tmp_path / "he_stain.tif"
    _write_rgb_tiff(path, shape=(48, 48, 3))
    # A stain matrix that's clearly not the identity, so deconvolved output
    # provably differs from any single raw RGB channel.
    stain_matrix = np.array([
        [1.8, -1.0, -0.5],
        [-0.1, 1.1, -0.1],
        [-0.6, -0.5, 1.6],
    ])
    spec = ImageSpec(
        path=str(path),
        pixel_size_um=0.5,
        dimension_order="YXC",
        full_shape=(48, 48, 3),
        level_shapes=[(48, 48, 3)],
        level_scales=[(1.0, 1.0)],
        channel_names=["hematoxylin", "eosin", "residual"],
        is_brightfield=True,
        stain_matrix=stain_matrix,
    )
    adata = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        var=pd.DataFrame(index=pd.Index(["hematoxylin", "eosin", "residual"])),
        uns={"spatial": {"sample": {"images": {"full": spec}}}},
    )

    fig_deconv = ab.pl.image(adata, channel="hematoxylin").to_figure()
    fig_raw = ab.pl.image(adata, channel=0).to_figure()

    img_deconv = fig_deconv.axes[0].images[0].get_array()
    img_raw = fig_raw.axes[0].images[0].get_array()
    assert not np.allclose(np.asarray(img_deconv), np.asarray(img_raw))


def test_default_title_uses_channel_name_not_ch_prefix_for_string_channel(tmp_path):
    # Regression test: requesting channel='hematoxylin' used to produce a
    # default title of "sample/full/chhematoxylin" (the "ch{channel}" format
    # meant for integer channel indices was applied unconditionally). A
    # string channel name should appear in the title as-is.
    path = tmp_path / "he_stain.tif"
    _write_rgb_tiff(path, shape=(48, 48, 3))
    spec = ImageSpec(
        path=str(path),
        pixel_size_um=0.5,
        dimension_order="YXC",
        full_shape=(48, 48, 3),
        level_shapes=[(48, 48, 3)],
        level_scales=[(1.0, 1.0)],
        channel_names=["hematoxylin", "eosin", "residual"],
        is_brightfield=True,
    )
    adata = ad.AnnData(
        obs=pd.DataFrame(index=pd.Index([], name="cell_id", dtype=str)),
        var=pd.DataFrame(index=pd.Index(["hematoxylin", "eosin", "residual"])),
        uns={"spatial": {"sample": {"images": {"full": spec}}}},
    )

    panel_str = ab.pl.image(adata, channel="hematoxylin")
    assert panel_str.title == "sample/full/hematoxylin"

    panel_int = ab.pl.image(adata, channel=0)
    assert panel_int.title == "sample/full/ch0"


def test_points_legend_skipped_above_cardinality_guard(caplog):
    # Regression: a categorical column with very high cardinality (e.g. SOM
    # node labels) used to get a legend with one entry per category and no
    # size cap at all -- a real run produced an unusable 526x6155px figure
    # from a 400-entry legend. Above the guard threshold, no legend should be
    # built at all (better than an unreadable one).
    n = 400
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    obs = pd.DataFrame({"node": [f"node_{i}" for i in range(n)]})
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = xy

    fig = ab.pl.points(adata, color="node").to_figure()
    ax = fig.axes[0]
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    assert len(legends) == 0
    plt.close(fig)


def test_points_continuous_color_is_percentile_clipped():
    # Regression: pl.points had no outlier-robust color-scale clipping, unlike
    # pl.image's 1st/99.8th percentile stretch -- a handful of outlier cells
    # setting vmax washed out the whole figure (total_counts, cell_area, etc.
    # in several tutorial notebooks). A small minority of extreme outliers
    # (well under the 0.2% the percentile clip is designed to exclude) among
    # otherwise small values should no longer set vmax to their raw value.
    n = 2000
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 500, (n, 2)).astype(np.float32)
    values = rng.uniform(1, 10, n).astype(np.float32)
    values[:2] = 10_000.0  # 0.1% extreme outliers
    obs = pd.DataFrame({"val": values})
    adata = ad.AnnData(X=np.zeros((n, 1), dtype=np.float32), obs=obs)
    adata.obsm["spatial"] = xy

    fig = ab.pl.points(adata, color="val").to_figure()
    ax = fig.axes[0]
    sc = next(c for c in ax.collections if hasattr(c, "get_offsets") and len(c.get_offsets()) > 0)
    assert sc.norm.vmax < 100.0  # nowhere near the raw 10_000 outlier
    plt.close(fig)
