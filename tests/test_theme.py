"""Tests for astrolabe.pl._theme (light/dark plot theme system)."""
import matplotlib.pyplot as plt
import pytest

import astrolabe as ab
from astrolabe.pl import _theme as th


@pytest.fixture(autouse=True)
def _reset_theme():
    """Every test starts and ends on the light theme, regardless of order."""
    ab.pl.set_theme("light")
    yield
    ab.pl.set_theme("light")


def test_set_theme_get_theme_roundtrip():
    ab.pl.set_theme("dark")
    assert ab.pl.get_theme().name == "dark"
    ab.pl.set_theme("light")
    assert ab.pl.get_theme().name == "light"


def test_theme_context_manager_restores_prior_state():
    ab.pl.set_theme("light")
    with ab.pl.theme("dark"):
        assert ab.pl.get_theme().name == "dark"
        assert plt.rcParams["axes.facecolor"] == th._DARK.surface
    assert ab.pl.get_theme().name == "light"
    assert plt.rcParams["axes.facecolor"] == th._LIGHT.surface


def test_theme_context_manager_nesting_restores_outer_theme():
    ab.pl.set_theme("light")
    with ab.pl.theme("dark"):
        with ab.pl.theme("light"):
            assert ab.pl.get_theme().name == "light"
        # inner context must restore to "dark", not the original "light"
        assert ab.pl.get_theme().name == "dark"
    assert ab.pl.get_theme().name == "light"


def test_extended_cat_colors_returns_unique_valid_hex():
    colors = th._extended_cat_colors(24, "light")
    assert len(colors) == 24
    assert len(set(colors)) == 24
    for c in colors:
        assert c.startswith("#") and len(c) == 7
        int(c[1:], 16)  # raises ValueError if not valid hex


def test_extended_cat_colors_dark_mode_also_valid():
    colors = th._extended_cat_colors(12, "dark")
    assert len(colors) == 12
    assert len(set(colors)) == 12


def test_build_cat_colors_uses_core_palette_at_or_below_eight():
    from astrolabe.pl._spatial import _build_cat_colors

    ab.pl.set_theme("light")
    cats = [f"c{i}" for i in range(8)]
    colors = _build_cat_colors(cats, None, "#000000")
    assert set(colors.values()) == set(th._LIGHT.cat_core)


def test_build_cat_colors_uses_extended_ramp_above_eight():
    from astrolabe.pl._spatial import _build_cat_colors

    ab.pl.set_theme("light")
    cats = [f"c{i}" for i in range(9)]
    colors = _build_cat_colors(cats, None, "#000000")
    assert len(set(colors.values())) == 9
    # Past 8 categories, colors are NOT limited to the validated 8-hue core.
    assert not set(colors.values()).issubset(set(th._LIGHT.cat_core))


def test_colormap_registration_idempotent_across_repeated_set_theme():
    for _ in range(3):
        ab.pl.set_theme("dark")
        ab.pl.set_theme("light")
    assert "astrolabe_seq" in plt.colormaps()
    assert "astrolabe_div" in plt.colormaps()


def test_set_theme_registers_bundled_font():
    ab.pl.set_theme("light")
    families = {f.name for f in __import__("matplotlib.font_manager", fromlist=["fontManager"]).fontManager.ttflist}
    assert "Inter" in families


def test_theme_dpi_floor_is_300():
    """A FigurePanel's wrapped content is rasterized once at savefig.dpi and
    can end up displayed considerably larger than its own declared figsize
    (dynamic sizing, a wide sibling panel) -- 150 visibly blurred under that
    stretch, so 300 is a floor for both modes, not just a default.
    """
    for name in ("light", "dark"):
        ab.pl.set_theme(name)
        assert plt.rcParams["figure.dpi"] >= 300
        assert plt.rcParams["savefig.dpi"] >= 300
