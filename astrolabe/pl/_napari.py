"""napari integration for astrolabe spatial transcriptomics data.

Requires napari (optional dependency): ``uv add napari``

The :func:`view_napari` function opens an interactive viewer and returns a
:class:`NapariSession` that holds the viewer reference and exposes annotation
import helpers:

- :meth:`NapariSession.annotations` — extract shapes drawn in napari as µm polygons
- :meth:`NapariSession.label_cells` — point-in-polygon test → obs column
- :meth:`NapariSession.get_affine` — read a layer's current napari affine in astrolabe convention
- :meth:`NapariSession.update_registration` — write napari affines back to ``ImageSpec.registration_affine``

Coordinate convention
---------------------
astrolabe stores spatial data as **(x, y)** in µm.  napari layers expect
**(y, x)** (row, col) world coordinates.  All conversions are handled
internally; the napari viewer and the matplotlib plots share the same µm
extents.

Image loading
-------------
Images are opened as multiscale zarr stores via ``tifffile.aszarr()``.  napari
fetches tiles lazily — no full-resolution image is loaded into memory.  The
``tifffile.TiffFile`` and zarr store are kept alive in each Image layer's
``.metadata`` dict until the viewer is closed.

Multi-modality display
----------------------
When a MuData is passed, images from **all** modalities that carry
``uns['spatial'][lib_id]['images']`` entries are added as separate napari
Image layers.  Because each ``ImageSpec`` positions its image in the shared
µm frame via ``registration_affine`` (or ``pixel_size_um``), the layers align
automatically.

Interactive registration
------------------------
napari's built-in TRANSFORM mode (press ``T``) lets users drag, rotate, and
scale any Image layer interactively.  Use :meth:`NapariSession.update_registration`
after closing the viewer to write the adjusted affines back to the
``ImageSpec`` objects — and then :func:`astrolabe.io.save` to persist them.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from astrolabe._utils import ImageSpec, _resolve_adata
from astrolabe.pl._spatial import _build_cat_colors, _resolve_color_values

if TYPE_CHECKING:
    import napari

log = logging.getLogger(__name__)

__all__ = ["view_napari", "NapariSession"]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_napari():
    try:
        import napari
        return napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for interactive viewing: uv add napari"
        ) from exc


def _xy_to_yx(coords: np.ndarray) -> np.ndarray:
    """Flip (n, 2) xy µm → yx for napari layer coordinates."""
    return np.ascontiguousarray(coords[:, ::-1], dtype=np.float64)


def _yx_to_xy(coords: np.ndarray) -> np.ndarray:
    """Flip (n, 2) yx napari → xy µm for astrolabe coordinates."""
    return np.ascontiguousarray(coords[:, ::-1], dtype=np.float64)


def _spec_to_napari_affine(spec: ImageSpec) -> np.ndarray:
    """Convert ``registration_affine`` to a napari Image affine.

    ImageSpec convention: ``[x_µm, y_µm, 1]^T = M @ [col, row, 1]^T``
    napari convention:    ``[y_µm, x_µm, 1]^T = A @ [row, col, 1]^T``
    """
    M = spec.registration_affine
    return np.array(
        [
            [M[1, 1], M[1, 0], M[1, 2]],  # y_µm from row, col
            [M[0, 1], M[0, 0], M[0, 2]],  # x_µm from row, col
            [0.0,     0.0,     1.0     ],
        ],
        dtype=np.float64,
    )


def _napari_affine_to_spec(A: np.ndarray) -> np.ndarray:
    """Convert a napari Image affine back to astrolabe ``registration_affine`` convention.

    Exact inverse of :func:`_spec_to_napari_affine`.

    napari convention:    ``[y_µm, x_µm, 1]^T = A @ [row, col, 1]^T``
    ImageSpec convention: ``[x_µm, y_µm, 1]^T = M @ [col, row, 1]^T``
    """
    return np.array(
        [
            [A[1, 1], A[1, 0], A[1, 2]],  # x_µm from col, row
            [A[0, 1], A[0, 0], A[0, 2]],  # y_µm from col, row
            [0.0,     0.0,     1.0     ],
        ],
        dtype=np.float64,
    )


def _awkward_to_polygons_yx(ak_array) -> list[np.ndarray]:
    """Convert an awkward boundary array to a list of (n_verts, 2) float64 **yx** arrays."""
    result = []
    for i in range(len(ak_array)):
        pts = np.asarray(ak_array[i].to_numpy(), dtype=np.float64)  # (n_verts, 2) xy
        if pts.ndim == 2 and len(pts) >= 3:
            result.append(_xy_to_yx(pts))
    return result


def _open_multiscale_zarr(spec: ImageSpec) -> tuple[Any, Any, list[Any]]:
    """Open a pyramidal TIFF as a list of lazy zarr arrays.

    Returns ``(tif, store, levels)`` where ``tif`` and ``store`` must be kept
    alive (store them in ``layer.metadata``) as long as napari accesses tiles.
    """
    import zarr
    import tifffile

    tif = tifffile.TiffFile(spec.path)
    store = tif.aszarr()
    z = zarr.open(store, mode="r")
    if isinstance(z, zarr.Group):
        levels = [z[i] for i in range(spec.n_levels)]
    else:
        levels = [z]
    return tif, store, levels


def _auto_point_size(n_cells: int) -> float:
    return max(1.0, 300_000.0 / n_cells)


def _collect_images(
    data: ad.AnnData | md.MuData,
    library_id: str | None,
    image_key: str,
) -> list[tuple[str, str, ImageSpec]]:
    """Return all image specs across modalities that should be shown.

    Returns a list of ``(base_name, resolved_key, spec)`` tuples where
    ``base_name`` is the napari layer name prefix used for that image (e.g.
    ``"if: full"`` or ``"xenium: morphology"``).
    """
    results: list[tuple[str, str, ImageSpec]] = []

    if isinstance(data, md.MuData):
        sources: list[tuple[str | None, ad.AnnData]] = list(data.mod.items())
    else:
        sources = [(None, data)]

    for mod_key, mod_adata in sources:
        for lib_id, lib_uns in mod_adata.uns.get("spatial", {}).items():
            if library_id is not None and lib_id != library_id:
                continue
            images: dict[str, ImageSpec] = lib_uns.get("images", {})
            if not images:
                continue
            # Prefer the requested key, then 'full', then first available
            key = (
                image_key if image_key in images
                else ("full" if "full" in images else next(iter(images)))
            )
            spec = images[key]
            prefix = f"{mod_key}: " if mod_key is not None else f"{lib_id}: "
            results.append((prefix + key, key, spec))

    return results


def _add_image_layer(
    viewer: napari.Viewer,
    spec: ImageSpec,
    base_name: str,
    *,
    channel: int | str | None,
) -> None:
    """Add one ImageSpec to the napari viewer as (possibly multi-channel) Image layer(s).

    Layer names: for multichannel IF images with ``spec.channel_names``, each
    channel layer is renamed ``"{base_name} [{ch_name}]"``.  For H&E images
    (detected by ``spec.stain_matrix is not None``), ``channel_axis`` is
    suppressed so the image renders as a composite.
    """
    tif, store, levels = _open_multiscale_zarr(spec)

    # Suppress channel splitting for H&E (display as RGB composite)
    if spec.stain_matrix is not None:
        ch_ax: int | None = None
    else:
        idx = spec.dimension_order.find("C")
        ch_ax = idx if idx >= 0 else None

    if spec.registration_affine is not None:
        spatial_kw: dict[str, Any] = {"affine": _spec_to_napari_affine(spec)}
    else:
        px = spec.pixel_size_um
        spatial_kw = {"scale": (px, px)}

    multiscale_data = levels if len(levels) > 1 else levels[0]
    added = viewer.add_image(
        multiscale_data,
        multiscale=len(levels) > 1,
        channel_axis=ch_ax,
        name=base_name,
        blending="additive",
        metadata={"_tif": tif, "_store": store},
        **spatial_kw,
    )

    layer_list: list[Any] = added if isinstance(added, list) else [added]

    # Rename per-channel layers with channel names if available
    if ch_ax is not None and spec.channel_names and len(spec.channel_names) == len(layer_list):
        for lyr, ch_name in zip(layer_list, spec.channel_names):
            lyr.name = f"{base_name} [{ch_name}]"

    # Hide non-selected channels when channel= is specified
    if channel is not None and ch_ax is not None:
        if isinstance(channel, str) and spec.channel_names:
            try:
                ch_idx = spec.channel_names.index(channel)
            except ValueError:
                ch_idx = None
        else:
            ch_idx = int(channel) if isinstance(channel, int) else None

        if ch_idx is not None:
            for i, lyr in enumerate(layer_list):
                lyr.visible = i == ch_idx


def _add_shapes_layer(
    viewer: napari.Viewer,
    polys: list[np.ndarray],
    values: np.ndarray | None,
    is_cat: bool,
    *,
    name: str,
) -> None:
    """Add polygons as a napari Shapes layer with optional per-shape color."""
    if not polys:
        return

    if values is None:
        viewer.add_shapes(
            polys,
            shape_type="polygon",
            edge_color="#aaaaaa",
            face_color="transparent",
            name=name,
        )
    elif is_cat:
        unique_cats = sorted(set(values))
        cat_colors = _build_cat_colors(unique_cats, None, "#d3d3d3")
        edge_colors = [cat_colors.get(str(v), "#d3d3d3") for v in values]
        viewer.add_shapes(
            polys,
            shape_type="polygon",
            edge_color=edge_colors,
            face_color="transparent",
            name=name,
        )
    else:
        import matplotlib.pyplot as plt

        vmin, vmax = float(np.nanpercentile(values, 2)), float(np.nanpercentile(values, 98))
        normed = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        cmap = plt.get_cmap("viridis")
        edge_colors = [cmap(float(v)) for v in normed]
        viewer.add_shapes(
            polys,
            shape_type="polygon",
            edge_color=edge_colors,
            face_color="transparent",
            name=name,
        )


# ---------------------------------------------------------------------------
# NapariSession
# ---------------------------------------------------------------------------

_CHANNEL_SUFFIX_RE = re.compile(r"\s*\[[^\]]*\]$")


class NapariSession:
    """Wraps a :class:`napari.Viewer` with annotation and registration import helpers.

    Returned by :func:`view_napari`; not directly exported from ``astrolabe.pl``.

    Examples
    --------
    >>> session = ab.pl.view_napari(mdata, color="leiden")
    >>> session.run()                                         # block until closed
    >>> polys = session.annotations("ROI layer")             # list of xy µm arrays
    >>> session.label_cells("ROI layer", obs_key="region")   # writes obs column
    >>> session.update_registration("if: full")              # save napari alignment back
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        data: ad.AnnData | md.MuData,
        adata: ad.AnnData,
        layer_to_spec: dict[str, ImageSpec],
    ) -> None:
        self._viewer = viewer
        self._data = data
        self._adata = adata
        self._layer_to_spec = layer_to_spec  # base_name → ImageSpec

    @property
    def viewer(self) -> napari.Viewer:
        """The underlying :class:`napari.Viewer`."""
        return self._viewer

    def run(self) -> None:
        """Block until all napari viewers are closed (calls ``napari.run()``)."""
        _require_napari().run()

    # ── Annotation import ────────────────────────────────────────────────────

    def _get_shapes_layer(self, layer_name: str | None):
        napari_mod = _require_napari()
        for layer in self._viewer.layers:
            if not isinstance(layer, napari_mod.layers.Shapes):
                continue
            if layer_name is None or layer.name == layer_name:
                return layer
        suffix = "" if layer_name is None else f" named {layer_name!r}"
        raise KeyError(
            f"No Shapes layer{suffix} found. Draw shapes in napari first."
        )

    def annotations(self, layer_name: str | None = None) -> list[np.ndarray]:
        """Extract polygons from a napari Shapes layer as µm **(x, y)** arrays.

        Parameters
        ----------
        layer_name
            Name of the Shapes layer to read. Uses the first Shapes layer if ``None``.

        Returns
        -------
        list of ``(n_verts, 2)`` float64 arrays in µm **(x, y)** — compatible
        with :func:`astrolabe.pl.annotation` and :func:`astrolabe.pp.crop`.
        """
        layer = self._get_shapes_layer(layer_name)
        return [_yx_to_xy(poly) for poly in layer.data]

    def label_cells(
        self,
        layer_name: str | None = None,
        obs_key: str = "region",
        modality: str | None = None,
    ) -> None:
        """Assign cells to annotated regions via point-in-polygon and write to obs.

        For each polygon drawn in the napari Shapes layer, tests which cell
        centroids (``obsm['spatial']``) fall inside it and writes the region
        name to ``adata.obs[obs_key]`` as a Categorical.  Cells outside all
        polygons receive ``pd.NA``.  Shapes are named
        ``{layer_name}_0``, ``{layer_name}_1``, … by default.

        Parameters
        ----------
        layer_name
            Name of the Shapes layer to read. Uses the first Shapes layer if ``None``.
        obs_key
            Column to write in ``adata.obs``.
        modality
            Modality override when ``data`` is a MuData.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon as ShapelyPolygon

        adata = _resolve_adata(self._data, modality) if modality is not None else self._adata
        layer = self._get_shapes_layer(layer_name)
        polys_xy = self.annotations(layer_name)

        if not polys_xy:
            raise ValueError("Shapes layer contains no polygons.")

        base_name = layer.name
        shape_names = [f"{base_name}_{i}" for i in range(len(polys_xy))]
        shapely_polys = [ShapelyPolygon(p) for p in polys_xy]

        coords = adata.obsm["spatial"]
        cells_gs = gpd.GeoSeries.from_xy(
            coords[:, 0].astype(np.float64),
            coords[:, 1].astype(np.float64),
        )

        labels: list[Any] = [pd.NA] * len(coords)
        for name, poly in zip(shape_names, shapely_polys):
            mask = cells_gs.within(poly).to_numpy()
            for idx in np.flatnonzero(mask):
                labels[idx] = name

        adata.obs[obs_key] = pd.Categorical(labels)
        log.info(
            "label_cells: wrote %d region labels to obs[%r]",
            sum(l is not pd.NA for l in labels),
            obs_key,
        )

    # ── Registration import ──────────────────────────────────────────────────

    def _find_layer(self, layer_name: str):
        """Return the first napari layer whose name matches exactly or starts with ``layer_name``."""
        for layer in self._viewer.layers:
            if layer.name == layer_name or layer.name.startswith(layer_name):
                return layer
        raise KeyError(f"No napari layer found matching {layer_name!r}.")

    def get_affine(self, layer_name: str) -> np.ndarray:
        """Read the current napari affine of an Image layer in astrolabe convention.

        Converts napari's (row, col) → (y_µm, x_µm) matrix back to astrolabe's
        (col, row) → (x_µm, y_µm) ``registration_affine`` format.

        Parameters
        ----------
        layer_name
            Exact napari layer name, or a prefix (e.g. ``"if: full"`` matches
            ``"if: full [DAPI]"``).

        Returns
        -------
        (3, 3) float64 homogeneous affine in the same convention as
        :attr:`ImageSpec.registration_affine`.
        """
        layer = self._find_layer(layer_name)
        return _napari_affine_to_spec(layer.affine.affine_matrix)

    def update_registration(self, layer_name: str | None = None) -> None:
        """Write napari Image layer affines back to the corresponding ``ImageSpec`` objects.

        After a user aligns modalities in napari's TRANSFORM mode (press ``T``),
        call this method to update ``ImageSpec.registration_affine`` in-place.
        The change propagates automatically to the parent MuData since the
        ImageSpec is the same object.  Persist with :func:`astrolabe.io.save`.

        Parameters
        ----------
        layer_name
            Name (or prefix) of the napari Image layer to update.  If ``None``,
            updates all tracked image layers.  A channel suffix like
            ``" [DAPI]"`` is stripped before looking up the base ImageSpec.

        Examples
        --------
        >>> session.update_registration("if: full")   # strip " [DAPI]" etc. automatically
        >>> session.update_registration()              # update all image layers
        >>> ab.io.save(mdata, "results/aligned")       # persist
        """
        if layer_name is None:
            targets = list(self._layer_to_spec.keys())
        else:
            # Strip " [channel]" suffix to get the base name
            base = _CHANNEL_SUFFIX_RE.sub("", layer_name).strip()
            targets = [base]

        for base in targets:
            if base not in self._layer_to_spec:
                raise KeyError(
                    f"No tracked image layer with base name {base!r}. "
                    f"Available: {list(self._layer_to_spec)}"
                )
            spec = self._layer_to_spec[base]
            new_affine = self.get_affine(base)
            spec.registration_affine = new_affine
            log.info("update_registration: updated %r\n%s", base, new_affine)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def view_napari(
    data: ad.AnnData | md.MuData,
    *,
    modality: str | None = None,
    image: bool = True,
    image_key: str = "full",
    channel: int | str | None = None,
    library_id: str | None = None,
    color: str | None = None,
    boundaries: bool | str = False,
    nucleus_boundaries: bool = False,
    ndisplay: int = 2,
) -> NapariSession:
    """Open an interactive napari viewer for spatial transcriptomics data.

    Requires napari (optional dependency): ``uv add napari``.

    When *data* is a :class:`~mudata.MuData`, images from **all** modalities
    (Xenium morphology, IF, H&E, …) are added as separate napari Image layers.
    They align automatically because each :class:`~astrolabe._utils.ImageSpec`
    positions its image in the shared µm frame via ``registration_affine``.

    Parameters
    ----------
    data
        AnnData or MuData.  MuData resolves the primary modality (for Points
        and Shapes layers) via ``modality`` kwarg or ``uns['default_modality']``.
        Image layers are collected from **all** modalities regardless.
    modality
        Modality key for MuData (e.g. ``'xenium'``).  Affects Points and
        boundary layers only.
    image
        Add Image layers from all modalities that carry ``uns['spatial']`` images.
    image_key
        Preferred image key to show per modality (default ``'full'``).  Falls
        back to ``'full'`` then the first available key for each modality.
    channel
        Single channel to show (int index or channel name string).  When
        ``None`` all channels are added as separate napari Image layers per
        modality.
    library_id
        Library identifier filter; defaults to including all libraries.
    color
        Gene name or obs column to colour the Points layer.
    boundaries
        If ``True``, add cell boundaries as a Shapes layer.  If a string, use
        that obs/gene column for per-boundary colour.
    nucleus_boundaries
        If ``True``, add nucleus boundaries as a Shapes layer.
    ndisplay
        Number of displayed dimensions (``2`` or ``3``).

    Returns
    -------
    NapariSession
        Holds the viewer and exposes annotation / registration import methods.
        Call ``.run()`` to block until the viewer window is closed, then
        ``.update_registration()`` to write back any affine adjustments.
    """
    napari_mod = _require_napari()
    adata = _resolve_adata(data, modality)
    viewer = napari_mod.Viewer(ndisplay=ndisplay)
    viewer.dims.axis_labels = ("y (µm)", "x (µm)")

    # ── Image layers (all modalities) ─────────────────────────────────────────
    layer_to_spec: dict[str, ImageSpec] = {}
    if image:
        image_sources = _collect_images(data, library_id, image_key)
        if not image_sources:
            log.warning("view_napari: no images found in any modality; skipping Image layers.")
        for base_name, _key, spec in image_sources:
            _add_image_layer(viewer, spec, base_name, channel=channel)
            layer_to_spec[base_name] = spec

    # ── Points layer ─────────────────────────────────────────────────────────
    coords_yx = _xy_to_yx(adata.obsm["spatial"])
    size = _auto_point_size(len(coords_yx))
    values, is_cat = _resolve_color_values(adata, color)
    pt_name = f"cells ({color})" if color else "cells"

    if values is None:
        viewer.add_points(coords_yx, name=pt_name, size=size)
    elif is_cat:
        unique_cats = sorted(set(values))
        cat_colors = _build_cat_colors(unique_cats, None, "#d3d3d3")
        face_colors = np.array([cat_colors.get(str(v), "#d3d3d3") for v in values])
        viewer.add_points(coords_yx, face_color=face_colors, name=pt_name, size=size)
    else:
        viewer.add_points(
            coords_yx,
            features={"value": values},
            face_color="value",
            face_colormap="viridis",
            name=pt_name,
            size=size,
        )

    # ── Boundary Shapes layers ────────────────────────────────────────────────
    if boundaries is not False and "cell_boundaries" in adata.obsm:
        bcolor = boundaries if isinstance(boundaries, str) else None
        bvals, bcat = _resolve_color_values(adata, bcolor)
        polys = _awkward_to_polygons_yx(adata.obsm["cell_boundaries"])
        _add_shapes_layer(viewer, polys, bvals, bcat, name="cell boundaries")

    if nucleus_boundaries and "nucleus_boundaries" in adata.obsm:
        polys = _awkward_to_polygons_yx(adata.obsm["nucleus_boundaries"])
        _add_shapes_layer(viewer, polys, None, False, name="nucleus boundaries")

    return NapariSession(viewer, data, adata, layer_to_spec)
