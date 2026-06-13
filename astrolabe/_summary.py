"""
Summary — rich human-readable overview of a MuData object.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import anndata as ad
import mudata as md
import numpy as np

if TYPE_CHECKING:
    pass

__all__ = ["summary", "Summary"]

# ── CSS (scoped with ab- prefix to avoid notebook conflicts) ──────────────────

_HTML_CSS = """
<style>
.ab-summary { font-family: var(--jp-ui-font-family, sans-serif); font-size: 13px;
              color: var(--jp-ui-font-color1, #111); max-width: 860px; }
.ab-summary h3 { margin: 0 0 6px 0; font-size: 15px; font-weight: 600; color: #333; }
.ab-card-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
.ab-card { background: var(--jp-layout-color2, #f7f7f7); border: 1px solid #ddd;
           border-radius: 6px; padding: 10px 14px; min-width: 130px; flex: 1 1 130px; }
.ab-card-title { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
                 color: #888; margin-bottom: 4px; }
.ab-card-value { font-size: 22px; font-weight: 700; color: #2a7ae2; line-height: 1.1; }
.ab-card-sub { font-size: 11px; color: #666; margin-top: 2px; }
.ab-section { margin-bottom: 14px; }
.ab-section-title { font-size: 12px; font-weight: 700; text-transform: uppercase;
                    letter-spacing: .08em; color: #555; border-bottom: 1px solid #e0e0e0;
                    padding-bottom: 3px; margin-bottom: 6px; }
.ab-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.ab-table th { background: #f0f0f0; text-align: left; padding: 3px 8px;
               font-weight: 600; border: 1px solid #ddd; }
.ab-table td { padding: 2px 8px; border: 1px solid #e8e8e8; vertical-align: top; }
.ab-table tr:nth-child(even) td { background: #fafafa; }
.ab-pill { display: inline-block; background: #e8f0fe; color: #1a56db; border-radius: 10px;
           padding: 0 7px; font-size: 11px; margin: 1px 2px; }
.ab-pill-green  { background: #e6f4ea; color: #137333; }
.ab-pill-orange { background: #fef3e2; color: #b45309; }
.ab-pill-purple { background: #f3e8ff; color: #6b21a8; }
.ab-pill-teal   { background: #e0f2fe; color: #0369a1; }
.ab-pill-grey   { background: #f1f3f4; color: #555; }
.ab-none { color: #999; font-style: italic; }
.ab-workflow { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px;
               padding: 8px 10px; background: var(--jp-layout-color2, #f7f7f7);
               border: 1px solid #ddd; border-radius: 6px; }
.ab-step { font-size: 11px; padding: 2px 9px; border-radius: 10px; font-weight: 500; }
.ab-step-done    { background: #e6f4ea; color: #137333; }
.ab-step-pending { background: #f1f3f4; color: #999; }
</style>
"""

# ── Classification helpers ────────────────────────────────────────────────────

# Explicit QC column names that have no recognisable naming pattern
_QC_EXPLICIT = frozenset({
    "total_counts", "n_genes", "n_counts",
    "cell_area", "nucleus_area", "nucleus_to_cell_area_ratio",
    "has_nucleus", "nucleus_count", "transcript_counts",
})

# uns keys that belong to other summary sections, not "Stored results"
_UNS_SYSTEM_KEYS = frozenset({
    "spatial",                   # shown in Libraries section
    "default_modality",          # internal routing
    "astrolabe_schema_version",  # internal versioning
    "neighbors",                 # shown in Graph/segmentation section
    "spatial_neighbors",         # shown in Graph/segmentation section
})

# Ordered display groups for obs columns
_OBS_GROUP_ORDER = ["QC", "Clustering", "Annotation", "Gene scores", "Spatial", "Other"]

# Ordered display categories for stored results
_RESULT_CAT_ORDER = [
    "Embeddings", "Clustering", "Trajectory", "DE",
    "Spatial stats", "Neighborhood", "Other",
]


def _classify_obs_col(col: str) -> str:
    """Classify an obs column name into a display group using naming conventions."""
    cl = col.lower()
    if (
        col in _QC_EXPLICIT
        or col.startswith(("log1p_", "pct_counts_", "n_transcripts_", "pct_transcripts_"))
        or (col.startswith("n_") and col.endswith("_by_counts"))
        or col.endswith(("_total", "_circularity", "_solidity", "_elongation"))
        or (col.startswith(("n_", "pct_")) and col.endswith(("_probes", "_codewords")))
    ):
        return "QC"
    if col.startswith(("leiden", "louvain", "banksy_sweep_")):
        return "Clustering"
    if col.endswith(("_annotation", "_celltype")):
        return "Annotation"
    if col.endswith(("_score", "_Score")):
        return "Gene scores"
    if "moran" in cl or "lisa" in cl or col.endswith(("_cluster", "_domain")):
        return "Spatial"
    return "Other"


def _classify_result_key(key: str) -> str:
    """Classify a uns result key into a display category."""
    kl = key.lower()
    if key in ("pca", "umap") or key.endswith(("_pca", "_umap")):
        return "Embeddings"
    if key in ("leiden", "louvain") or key.startswith("banksy_sweep"):
        return "Clustering"
    if key in ("paga", "dpt"):
        return "Trajectory"
    if key == "rank_genes_groups" or key.endswith("_de"):
        return "DE"
    if any(x in kl for x in ("moran", "geary", "getis", "ripley", "spatially_variable", "spatial_lag")):
        return "Spatial stats"
    if key in ("nhood_enrichment", "co_occurrence", "centrality_scores",
               "expression_by_distance", "ligrec", "composition", "icg"):
        return "Neighborhood"
    return "Other"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _h(text: str) -> str:
    return html.escape(str(text))


# ── Data collection ───────────────────────────────────────────────────────────

def _collect(data: ad.AnnData | md.MuData) -> dict:
    info: dict = {}

    # ── modalities ────────────────────────────────────────────────────────────
    if isinstance(data, md.MuData):
        mods = {}
        for key, adata in data.mod.items():
            n_ctrl = int(adata.var.get("is_control_probe", np.zeros(adata.n_vars, bool)).sum())
            mods[key] = {
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "n_ctrl": n_ctrl,
                "is_default": data.uns.get("default_modality") == key,
            }
        info["modalities"] = mods
        info["default_modality"] = data.uns.get("default_modality")
        primary_mod = info["default_modality"] or next(iter(mods))
        adata = data.mod[primary_mod]
        uns = data.uns
    else:
        info["modalities"] = None
        adata = data
        uns = data.uns

    # ── top-level counts ──────────────────────────────────────────────────────
    info["n_obs"] = adata.n_obs
    info["n_vars"] = adata.n_vars

    # ── spatial info ──────────────────────────────────────────────────────────
    spatial_info: dict = {}
    if "spatial" in uns:
        for lib_id, lib_meta in uns["spatial"].items():
            images = {}
            if "images" in lib_meta:
                for img_key, spec in lib_meta["images"].items():
                    ch_names = getattr(spec, "channel_names", None)
                    n_ch = len(ch_names) if ch_names else None
                    dim = getattr(spec, "dimension_order", None)
                    full_shape = getattr(spec, "full_shape", None)
                    n_levels = len(getattr(spec, "level_shapes", None) or []) or 1
                    px_um = getattr(spec, "pixel_size_um", None)
                    images[img_key] = {
                        "shape": full_shape,
                        "n_levels": n_levels,
                        "pixel_size_um": px_um,
                        "dimension_order": dim,
                        "channel_names": list(ch_names) if ch_names else None,
                        "n_channels": n_ch,
                    }
            spatial_info[lib_id] = {
                "source_path": lib_meta.get("source_path"),
                "coordinate_unit": lib_meta.get("coordinate_unit", "µm"),
                "images": images,
            }

    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        spatial_info["_extent"] = {
            "x_min": float(coords[:, 0].min()),
            "x_max": float(coords[:, 0].max()),
            "y_min": float(coords[:, 1].min()),
            "y_max": float(coords[:, 1].max()),
        }
    info["spatial"] = spatial_info

    # ── library IDs in obs ────────────────────────────────────────────────────
    if "library_id" in adata.obs.columns:
        lib_ids = sorted(adata.obs["library_id"].unique().tolist())
        info["library_ids"] = lib_ids
        info["n_samples"] = len(lib_ids)
    else:
        info["library_ids"] = None
        info["n_samples"] = None

    # ── features ──────────────────────────────────────────────────────────────
    info["n_hvg"] = int(adata.var["highly_variable"].sum()) if "highly_variable" in adata.var.columns else None
    info["n_svg"] = int(adata.var["sv_spatially_variable"].sum()) if "sv_spatially_variable" in adata.var.columns else None

    # ── obs columns (classified) ──────────────────────────────────────────────
    obs_groups: dict[str, list[str]] = {}
    for col in adata.obs.columns:
        g = _classify_obs_col(col)
        obs_groups.setdefault(g, []).append(col)
    info["obs_groups"] = obs_groups

    # ── summary statistics (from obs, if available) ───────────────────────────
    def _obs_stat(col: str, fn) -> float | None:
        if col in adata.obs.columns:
            try:
                return float(fn(adata.obs[col]))
            except Exception:
                pass
        return None

    info["median_counts"] = _obs_stat("total_counts", lambda s: s.median())
    info["median_genes"] = _obs_stat("n_genes_by_counts", lambda s: s.median())
    info["mean_cell_area"] = _obs_stat("cell_area", lambda s: s.mean())
    info["pct_with_nucleus"] = _obs_stat("has_nucleus", lambda s: s.mean() * 100.0)

    # ── clustering runs ───────────────────────────────────────────────────────
    cluster_info = []
    for col in obs_groups.get("Clustering", []):
        try:
            n = int(adata.obs[col].nunique())
            cluster_info.append({"col": col, "n_clusters": n})
        except Exception:
            pass
    info["cluster_info"] = cluster_info

    # ── segmentation / graph state ────────────────────────────────────────────
    info["segmentation_keys"] = [k for k in ("cell_boundaries", "nucleus_boundaries") if k in adata.obsm]
    info["has_spatial_graph"] = (
        "spatial_neighbors" in uns or "spatial_neighbors_connectivities" in adata.obsp
    )

    # ── uns results (auto-discovered: everything not in system blocklist) ─────
    present_results = sorted(k for k in uns if k not in _UNS_SYSTEM_KEYS)
    info["uns_results"] = present_results

    results_by_category: dict[str, list[str]] = {}
    for key in present_results:
        cat = _classify_result_key(key)
        results_by_category.setdefault(cat, []).append(key)
    info["results_by_category"] = results_by_category

    # ── workflow progress ─────────────────────────────────────────────────────
    info["workflow_steps"] = [
        ("QC",         "total_counts" in adata.obs.columns),
        ("Normalized", "total_counts_norm" in adata.obs.columns),
        ("log1p",      "log1p_total_counts" in adata.obs.columns),
        ("HVGs",       "highly_variable" in adata.var.columns),
        ("PCA",        "X_pca" in adata.obsm),
        ("Neighbors",  "neighbors" in uns),
        ("UMAP",       "X_umap" in adata.obsm),
        ("Clustered",  bool(obs_groups.get("Clustering"))),
        ("SVGs",       "sv_spatially_variable" in adata.var.columns),
    ]

    return info


# ── Plain-text repr ───────────────────────────────────────────────────────────

def _build_repr(info: dict) -> str:
    lines = ["MuData Summary" if info["modalities"] else "AnnData Summary",
             "=" * 50]

    if info["modalities"]:
        lines.append("Modalities:")
        for key, m in info["modalities"].items():
            star = " (default)" if m["is_default"] else ""
            ctrl = f", {m['n_ctrl']} control probes" if m["n_ctrl"] else ""
            lines.append(f"  {key}{star}: {m['n_obs']:,} cells × {m['n_vars']:,} vars{ctrl}")
    else:
        lines.append(f"Cells: {info['n_obs']:,}   Vars: {info['n_vars']:,}")

    ext = info["spatial"].get("_extent")
    if ext:
        lines.append(
            f"Extent: x=[{ext['x_min']:.0f}, {ext['x_max']:.0f}] µm  "
            f"y=[{ext['y_min']:.0f}, {ext['y_max']:.0f}] µm"
        )

    libs = {k: v for k, v in info["spatial"].items() if k != "_extent"}
    if libs:
        lines.append(f"Libraries ({len(libs)}):")
        for lid, lv in libs.items():
            n_img = len(lv["images"])
            lines.append(f"  {lid}: {lv['source_path'] or '—'}  ({n_img} image(s))")

    stat_parts = []
    if info["median_counts"] is not None:
        stat_parts.append(f"median counts={info['median_counts']:.0f}")
    if info["median_genes"] is not None:
        stat_parts.append(f"median genes={info['median_genes']:.0f}")
    if info["mean_cell_area"] is not None:
        stat_parts.append(f"mean area={info['mean_cell_area']:.1f} µm²")
    if stat_parts:
        lines.append("Stats: " + ", ".join(stat_parts))

    feat_parts = []
    if info["n_hvg"] is not None:
        feat_parts.append(f"{info['n_hvg']:,} HVGs")
    if info["n_svg"] is not None:
        feat_parts.append(f"{info['n_svg']:,} SVGs")
    if feat_parts:
        lines.append("Features: " + ", ".join(feat_parts))

    if info["cluster_info"]:
        lines.append("Clustering:")
        for ci in info["cluster_info"]:
            lines.append(f"  {ci['col']}: {ci['n_clusters']} clusters")

    if info["obs_groups"]:
        lines.append("Obs columns:")
        for grp in _OBS_GROUP_ORDER:
            cols = info["obs_groups"].get(grp)
            if cols:
                lines.append(f"  {grp}: {', '.join(cols)}")

    seg_keys = info.get("segmentation_keys", [])
    graph = info.get("has_spatial_graph", False)
    seg_parts = list(seg_keys) + (["spatial_graph"] if graph else [])
    if seg_parts:
        lines.append("Graph/seg: " + ", ".join(seg_parts))

    done = [label for label, ok in info.get("workflow_steps", []) if ok]
    if done:
        lines.append("Workflow: " + " → ".join(done))

    if info["uns_results"]:
        lines.append("Results: " + ", ".join(info["uns_results"]))

    return "\n".join(lines)


# ── HTML repr ─────────────────────────────────────────────────────────────────

_RESULT_CAT_PILL = {
    "Embeddings":   "ab-pill-teal",
    "Clustering":   "ab-pill",
    "Trajectory":   "ab-pill-purple",
    "DE":           "ab-pill-orange",
    "Spatial stats": "ab-pill-green",
    "Neighborhood": "ab-pill-green",
    "Other":        "ab-pill-grey",
}


def _build_html(info: dict) -> str:
    parts = [_HTML_CSS, '<div class="ab-summary">']

    # ── top stat cards ────────────────────────────────────────────────────────
    parts.append('<div class="ab-card-grid">')

    if info["modalities"]:
        for key, m in info["modalities"].items():
            star = " ★" if m["is_default"] else ""
            parts.append(
                f'<div class="ab-card">'
                f'<div class="ab-card-title">{_h(key)}{_h(star)}</div>'
                f'<div class="ab-card-value">{_fmt_int(m["n_obs"])}</div>'
                f'<div class="ab-card-sub">{_fmt_int(m["n_vars"])} vars'
                + (f' · {_fmt_int(m["n_ctrl"])} ctrl' if m["n_ctrl"] else "")
                + "</div></div>"
            )
    else:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Cells</div>'
            f'<div class="ab-card-value">{_fmt_int(info["n_obs"])}</div></div>'
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Features</div>'
            f'<div class="ab-card-value">{_fmt_int(info["n_vars"])}</div></div>'
        )

    ext = info["spatial"].get("_extent")
    if ext:
        w = ext["x_max"] - ext["x_min"]
        h = ext["y_max"] - ext["y_min"]
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Tissue extent</div>'
            f'<div class="ab-card-value">{w:.0f} µm</div>'
            f'<div class="ab-card-sub">× {h:.0f} µm</div></div>'
        )

    if info.get("median_counts") is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Median counts</div>'
            f'<div class="ab-card-value">{info["median_counts"]:.0f}</div>'
            f'<div class="ab-card-sub">per cell</div></div>'
        )

    if info.get("median_genes") is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Median genes</div>'
            f'<div class="ab-card-value">{info["median_genes"]:.0f}</div>'
            f'<div class="ab-card-sub">per cell</div></div>'
        )

    if info.get("mean_cell_area") is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Mean cell area</div>'
            f'<div class="ab-card-value">{info["mean_cell_area"]:.0f}</div>'
            f'<div class="ab-card-sub">µm²</div></div>'
        )

    if info.get("pct_with_nucleus") is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Has nucleus</div>'
            f'<div class="ab-card-value">{info["pct_with_nucleus"]:.0f}%</div>'
            f'<div class="ab-card-sub">of cells</div></div>'
        )

    if info.get("n_samples") is not None and info["n_samples"] > 1:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Samples</div>'
            f'<div class="ab-card-value">{info["n_samples"]}</div>'
            f'<div class="ab-card-sub">libraries</div></div>'
        )

    if info["n_hvg"] is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">HVGs</div>'
            f'<div class="ab-card-value">{_fmt_int(info["n_hvg"])}</div></div>'
        )

    if info["n_svg"] is not None:
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">SVGs</div>'
            f'<div class="ab-card-value">{_fmt_int(info["n_svg"])}</div></div>'
        )

    # Clustering cards — one per run
    for ci in info.get("cluster_info", []):
        parts.append(
            f'<div class="ab-card">'
            f'<div class="ab-card-title">Clusters</div>'
            f'<div class="ab-card-value">{ci["n_clusters"]}</div>'
            f'<div class="ab-card-sub">{_h(ci["col"])}</div></div>'
        )

    parts.append("</div>")  # end card-grid

    # ── workflow progress ─────────────────────────────────────────────────────
    steps = info.get("workflow_steps", [])
    if steps:
        parts.append('<div class="ab-workflow">')
        for label, done in steps:
            cls = "ab-step-done" if done else "ab-step-pending"
            icon = "✓" if done else "○"
            parts.append(f'<span class="ab-step {cls}">{icon} {_h(label)}</span>')
        parts.append("</div>")

    # ── libraries + images ────────────────────────────────────────────────────
    libs = {k: v for k, v in info["spatial"].items() if k != "_extent"}
    if libs:
        parts.append('<div class="ab-section"><div class="ab-section-title">Libraries &amp; images</div>')
        parts.append('<table class="ab-table"><tr><th>Library ID</th><th>Source</th>'
                     '<th>Images</th></tr>')
        for lid, lv in libs.items():
            img_pills = ""
            for img_key, im in lv["images"].items():
                ch = im.get("n_channels")
                lvls = im.get("n_levels", 1)
                shape = im.get("shape")
                tip = f"{img_key}"
                if ch:
                    tip += f" · {ch}ch"
                if shape:
                    tip += f" · {'×'.join(str(s) for s in shape)}"
                if lvls > 1:
                    tip += f" · {lvls} levels"
                img_pills += f'<span class="ab-pill">{_h(tip)}</span>'
            if not img_pills:
                img_pills = '<span class="ab-none">none</span>'
            src = _h(lv["source_path"] or "—")
            parts.append(f"<tr><td><b>{_h(lid)}</b></td><td>{src}</td><td>{img_pills}</td></tr>")
        parts.append("</table></div>")

    # ── obs columns ───────────────────────────────────────────────────────────
    if info["obs_groups"]:
        parts.append('<div class="ab-section"><div class="ab-section-title">Obs columns</div>')
        parts.append('<table class="ab-table"><tr><th>Group</th><th>Columns</th></tr>')
        for grp in _OBS_GROUP_ORDER:
            cols = info["obs_groups"].get(grp)
            if not cols:
                continue
            pills = "".join(f'<span class="ab-pill ab-pill-grey">{_h(c)}</span>' for c in cols)
            parts.append(f"<tr><td>{_h(grp)}</td><td>{pills}</td></tr>")
        parts.append("</table></div>")

    # ── segmentation / graph state ────────────────────────────────────────────
    seg_keys = info.get("segmentation_keys", [])
    graph = info.get("has_spatial_graph", False)
    if seg_keys or graph:
        parts.append('<div class="ab-section"><div class="ab-section-title">Graph &amp; segmentation</div><div>')
        for k in seg_keys:
            parts.append(f'<span class="ab-pill ab-pill-green">{_h(k)}</span>')
        if graph:
            parts.append('<span class="ab-pill ab-pill-green">spatial_graph</span>')
        parts.append("</div></div>")

    # ── stored results (categorised) ─────────────────────────────────────────
    rbc = info.get("results_by_category", {})
    if rbc:
        parts.append('<div class="ab-section"><div class="ab-section-title">Stored results</div>')
        parts.append('<table class="ab-table"><tr><th>Category</th><th>Keys</th></tr>')
        for cat in _RESULT_CAT_ORDER:
            keys = rbc.get(cat)
            if not keys:
                continue
            pill_cls = _RESULT_CAT_PILL.get(cat, "ab-pill-grey")
            pills = "".join(f'<span class="ab-pill {pill_cls}">{_h(k)}</span>' for k in keys)
            parts.append(f"<tr><td>{_h(cat)}</td><td>{pills}</td></tr>")
        parts.append("</table></div>")

    parts.append("</div>")  # end ab-summary
    return "".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

class Summary:
    """Rich overview of an AnnData or MuData object.

    Renders as a styled HTML card grid in Jupyter notebooks and as a
    plain-text table in the terminal.  Returned by :func:`summary`.
    """

    def __init__(self, data: ad.AnnData | md.MuData) -> None:
        self._info = _collect(data)

    def __repr__(self) -> str:
        return _build_repr(self._info)

    def _repr_html_(self) -> str:
        return _build_html(self._info)


def summary(data: ad.AnnData | md.MuData) -> Summary:
    """Return a rich summary of an AnnData or MuData object.

    In a Jupyter notebook the result renders as a styled HTML card layout.
    In the terminal it prints as a plain-text table.

    Parameters
    ----------
    data
        :class:`anndata.AnnData` or :class:`mudata.MuData`.

    Returns
    -------
    Summary
        A :class:`Summary` instance with ``__repr__`` and ``_repr_html_``
        for rich display.

    Examples
    --------
    >>> ab.summary(mdata)
    """
    return Summary(data)
