"""Spatial statistical tests.

Moran's I, bivariate variants, Geary's C, Getis-Ord G*, spatial lag, and
spatial silhouette delegate to the PySAL ecosystem (``esda``, ``libpysal``).
Ripley's L uses a KDTree-based range-counting implementation (scipy) that is
O(n log n) in time and memory — no pointpats dependency.
"""

from __future__ import annotations

import logging
import threading

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import _spawn_rngs, from_kwarg, with_adata

__all__ = [
    "moran",
    "moran_local",
    "moran_bv",
    "moran_local_bv",
    "getis_ord",
    "spatial_lag",
    "spatial_silhouette",
    "ripley_l",
    "spatially_variable_genes",
    "geary_c",
    "colocation_quotient",
]

log = logging.getLogger(__name__)

_QUAD_LABELS: dict[int, str] = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_weights(adata: ad.AnnData, neighbors_key: str):
    """Build a row-standardized libpysal W from the stored obsp connectivity."""
    from libpysal.weights import WSP

    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors() first"
        )
    w = WSP(adata.obsp[conn_key]).to_W()
    w.transform = "r"
    return w


def _thread_local_weights_getter(adata: ad.AnnData, neighbors_key: str):
    """Return a zero-arg getter yielding a per-thread-exclusive libpysal W.

    A single ``W`` instance must never be handed to more than one thread at
    once: ``esda.Moran``/``esda.Geary`` reassign ``w.transform`` on *every*
    construction, and libpysal's transform setter unconditionally invalidates
    and rebuilds cached derived state (``w.sparse`` and friends) with no
    locking. Concurrent constructions sharing one ``W`` race on that rebuild
    and can hand back arrays with mismatched shapes. Building a fresh ``W`` costs
    ~1-1.5s here, too slow to redo per gene, so this builds one independent
    ``W`` per worker *thread* (lazily, on that thread's first call) via
    ``threading.local`` — bounding the rebuild cost to `n_jobs`, not `n_genes`,
    while guaranteeing no two genes running concurrently ever touch the same
    ``W`` instance.
    """
    local = threading.local()

    def _get() -> object:
        w = getattr(local, "w", None)
        if w is None:
            w = _build_weights(adata, neighbors_key)
            local.w = w
        return w

    return _get


def _boundary_silhouette_graph(attr: np.ndarray, labels: np.ndarray, w) -> np.ndarray:
    """Graph-efficient boundary silhouette: O(n×k) instead of O(n²).

    esda.boundary_silhouette defaults to computing all-pairs distances via
    sklearn.metrics.pairwise.euclidean_distances, which requires O(n²) memory.
    This implementation computes distances only for cells connected by an edge
    in the spatial weight matrix W, using pandas groupby for vectorised
    aggregation rather than a Python-level cell loop.
    """
    adj = w.to_adjlist(remove_symmetric=False)
    n = len(labels)
    scores = np.zeros(n, dtype=np.float64)

    if len(adj) == 0:
        return scores

    id_order = list(w.id_order)
    id_to_row: dict = {obs_id: i for i, obs_id in enumerate(id_order)}

    focal_idx = np.array([id_to_row[f] for f in adj["focal"]], dtype=np.intp)
    neighbor_idx = np.array([id_to_row[nb] for nb in adj["neighbor"]], dtype=np.intp)

    diffs = attr[focal_idx] - attr[neighbor_idx]
    edge_dists = np.sqrt((diffs * diffs).sum(axis=1))

    edge_df = pd.DataFrame(
        {
            "focal": focal_idx,
            "nb_label": labels[neighbor_idx],
            "focal_label": labels[focal_idx],
            "dist": edge_dists,
        }
    )
    edge_df["same"] = edge_df["focal_label"] == edge_df["nb_label"]

    # a(i): mean distance to same-cluster spatial neighbors
    a_vals = (
        edge_df[edge_df["same"]]
        .groupby("focal")["dist"]
        .mean()
        .rename("a")
    )

    # b(i): mean distance to nearest *other* border cluster, per focal cell
    cross = edge_df[~edge_df["same"]].copy()
    if len(cross) == 0:
        return scores  # no cross-cluster edges; all scores remain 0

    b_vals = (
        cross.groupby(["focal", "nb_label"])["dist"]
        .mean()
        .groupby(level="focal")
        .min()
        .rename("b")
    )

    sil_df = pd.concat([a_vals, b_vals], axis=1)
    # Cells not on any cluster boundary (b is NaN) score 0 by convention.
    no_boundary = sil_df["b"].isna()
    # Cells with no same-cluster spatial neighbor at all (a is NaN) are
    # spatially isolated from their own label -- the esda.boundary_silhouette
    # analogue is a singleton cluster, which it also scores 0. Filling a with
    # 0.0 below and leaving these in would instead make them read as maximally
    # *compact* (a=0), driving sil to +1 -- the opposite of what "isolated
    # from its own cluster" should mean.
    no_same_cluster_neighbor = sil_df["a"].isna()
    sil_df = sil_df.fillna(0.0)
    mx = sil_df[["a", "b"]].max(axis=1)
    sil = (sil_df["b"] - sil_df["a"]) / mx.where(mx > 0, other=1.0)
    sil[no_boundary] = 0.0
    sil[no_same_cluster_neighbor] = 0.0
    scores[sil_df.index.to_numpy(dtype=np.intp)] = sil.to_numpy()
    return scores


def _get_conn_matrix(adata: ad.AnnData, neighbors_key: str) -> sp.spmatrix:
    """Return the raw sparse connectivity matrix from obsp (no libpysal)."""
    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors() first"
        )
    return adata.obsp[conn_key]


def _extract_values(adata: ad.AnnData, key: str, layer: str | None) -> np.ndarray:
    """Return a 1-D float64 array for *key*, checking var_names then obs columns."""
    if key in adata.var_names:
        idx = adata.var_names.get_loc(key)
        mat = adata.layers[layer] if layer is not None else adata.X
        col = mat[:, idx]
        if sp.issparse(col):
            col = col.toarray().ravel()
        return np.asarray(col, dtype=np.float64)
    if key in adata.obs.columns:
        return adata.obs[key].to_numpy(dtype=np.float64)
    raise KeyError(f"Key '{key}' not found in var_names or obs columns")


def _normalize_pairs(
    x: str | list[str], y: str | list[str]
) -> tuple[list[str], list[str]]:
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length; got {len(x)} and {len(y)}"
        )
    return x, y


def _adj_pvalues(p: np.ndarray, method: str) -> np.ndarray:
    from statsmodels.stats.multitest import multipletests
    _, p_adj, _, _ = multipletests(p, method=method)
    return p_adj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@with_adata(mirror_uns=from_kwarg("key_added"))
def moran(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    key_added: str = "moran",
) -> pd.DataFrame:
    """Global Moran's I for one or more expression or obs keys.

    Each key is first looked up in ``var_names`` (gene expression); if absent,
    it falls back to ``obs`` columns (QC metrics, cell area, etc.).

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more gene / obs column names to test.
    modality
        For :class:`~mudata.MuData`: modality key (defaults to
        ``uns['default_modality']``).
    layer
        Expression layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Permutations for empirical p-values (``0`` uses normality approximation only).
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per key with columns ``I``, ``EI``, ``z_norm``, ``p_norm``,
        and (when ``n_perms > 0``) ``z_sim``, ``p_sim``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> result = ab.tl.moran(mdata, ["EPCAM", "CD3D", "MKI67"])
    >>> result.sort_values("I", ascending=False)
    """
    import esda

    if isinstance(keys, str):
        keys = [keys]

    w = _build_weights(adata, neighbors_key)
    log.info("moran: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    rows = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        mi = esda.Moran(y, w, permutations=n_perms, two_tailed=True)
        row: dict[str, float] = {
            "I": mi.I,
            "EI": mi.EI,
            "z_norm": mi.z_norm,
            "p_norm": mi.p_norm,
        }
        if n_perms > 0:
            row["z_sim"] = mi.z_sim
            row["p_sim"] = mi.p_sim
        rows.append(row)

    result = pd.DataFrame(rows, index=pd.Index(keys, name="key"))
    adata.uns[key_added] = result
    log.info("moran: done — top I=%.3f (%s)", result["I"].max(), result["I"].idxmax())
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def moran_local(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    p_adjust_method: str = "fdr_bh",
    n_jobs: int = 1,
    key_added: str = "moran_local",
) -> pd.DataFrame:
    """Local Moran's I (LISA) for one or more expression or obs keys.

    Identifies per-cell spatial association patterns:

    - **HH** — high value surrounded by high-value neighbors (hot spot)
    - **LL** — low value surrounded by low-value neighbors (cold spot)
    - **HL** — high value surrounded by low-value neighbors (spatial outlier)
    - **LH** — low value surrounded by high-value neighbors (spatial outlier)

    Each key is first looked up in ``var_names``; if absent it falls back to
    ``obs`` columns.  P-values are FDR-adjusted across all cells per key.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more gene / obs column names to test.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Conditional randomisation permutations for per-cell p-values.
        ``0`` skips permutation; ``p_sim`` and ``p_adj`` will be ``NaN``.
    p_adjust_method
        Multiple-testing correction (e.g. ``'fdr_bh'``, ``'bonferroni'``).
        Applied independently per key across all cells.
    n_jobs
        Parallel workers for the conditional-randomisation permutations,
        forwarded to ``esda.Moran_Local``. ``1`` (default) runs sequentially.
        ``-1`` uses all available cores.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(key, obs_name)`` rows with columns ``Ii``, ``z_sim``,
        ``p_sim``, ``p_adj``, ``quadrant``.
        Access per-key results with ``result.loc['<key>']``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> lisa = ab.tl.moran_local(mdata, ["EPCAM", "CD3D"])
    >>> hh_epcam = lisa.loc["EPCAM"].query("quadrant == 'HH' and p_adj < 0.05")
    """
    import esda

    if isinstance(keys, str):
        keys = [keys]

    w = _build_weights(adata, neighbors_key)
    log.info("moran_local: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    frames = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        lmi = esda.Moran_Local(y, w, permutations=n_perms, n_jobs=n_jobs)

        if n_perms > 0:
            p_sim = lmi.p_sim
            z_sim = lmi.z_sim
            p_adj = _adj_pvalues(p_sim, p_adjust_method)
        else:
            p_sim = np.full(adata.n_obs, np.nan)
            z_sim = np.full(adata.n_obs, np.nan)
            p_adj = np.full(adata.n_obs, np.nan)

        quadrant = pd.array(
            [_QUAD_LABELS.get(int(q), "NS") for q in lmi.q], dtype="string"
        )

        frames.append(
            pd.DataFrame(
                {"Ii": lmi.Is, "z_sim": z_sim, "p_sim": p_sim, "p_adj": p_adj, "quadrant": quadrant},
                index=adata.obs_names,
            )
        )

    result = pd.concat(frames, keys=keys, names=["key", "obs_name"])
    adata.uns[key_added] = result
    log.info("moran_local: done")
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def moran_bv(
    adata: ad.AnnData,
    x: str | list[str],
    y: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    key_added: str = "moran_bv",
) -> pd.DataFrame:
    """Global bivariate Moran's I for pairs of expression / obs keys.

    Measures the degree to which high values of *x* are surrounded by
    high values of *y* in space — e.g. ligand-receptor co-localisation
    or tumor-immune co-expression.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.
    Each key is looked up in ``var_names`` first, then ``obs`` columns.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    x, y
        One or more paired gene / obs column names.  Lengths must match.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Permutations used to obtain ``EI``/``z_sim``/``p_sim``.  Must be > 0:
        unlike univariate Moran's I, ``esda.Moran_BV`` has no analytic
        normal-approximation null, so permutations are required.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(x, y)`` rows with columns ``I``, ``EI``, ``z_sim``,
        ``p_sim``.  Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> result = ab.tl.moran_bv(mdata, ["EPCAM", "CD274"], ["CXCL12", "PDCD1"])
    >>> result.sort_values("I", ascending=False)
    """
    import esda

    x_keys, y_keys = _normalize_pairs(x, y)

    w = _build_weights(adata, neighbors_key)
    log.info("moran_bv: %d pairs, %d cells, n_perms=%d", len(x_keys), adata.n_obs, n_perms)

    if n_perms <= 0:
        raise ValueError(
            "moran_bv: esda.Moran_BV has no analytic normal approximation; "
            "n_perms must be > 0 to obtain z/p-values."
        )

    rows = []
    for xk, yk in zip(x_keys, y_keys):
        xv = _extract_values(adata, xk, layer)
        yv = _extract_values(adata, yk, layer)
        mi = esda.Moran_BV(xv, yv, w, permutations=n_perms)
        row: dict[str, float] = {
            "I": mi.I,
            "EI": mi.EI_sim,
            "z_sim": mi.z_sim,
            "p_sim": mi.p_sim,
        }
        rows.append(row)

    idx = pd.MultiIndex.from_tuples(list(zip(x_keys, y_keys)), names=["x", "y"])
    result = pd.DataFrame(rows, index=idx)
    adata.uns[key_added] = result
    log.info("moran_bv: done")
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def moran_local_bv(
    adata: ad.AnnData,
    x: str | list[str],
    y: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    p_adjust_method: str = "fdr_bh",
    n_jobs: int = 1,
    key_added: str = "moran_local_bv",
) -> pd.DataFrame:
    """Local bivariate Moran's I (LISA) for pairs of expression / obs keys.

    Per-cell spatial co-association of *x* with the spatial lag of *y*.
    Quadrant labels (HH/LL/HL/LH) indicate whether a cell has a high/low
    value of *x* surrounded by high/low values of *y*.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.
    Each key is looked up in ``var_names`` first, then ``obs`` columns.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    x, y
        One or more paired gene / obs column names.  Lengths must match.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Conditional randomisation permutations (``0`` → NaN p-values).
    p_adjust_method
        Multiple-testing correction applied per pair across all cells.
    n_jobs
        Parallel workers for the permutations, forwarded to
        ``esda.Moran_Local_BV``. ``1`` (default) runs sequentially. ``-1``
        uses all available cores.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(pair, obs_name)`` rows where ``pair = 'x:y'``, with
        columns ``Ii``, ``z_sim``, ``p_sim``, ``p_adj``, ``quadrant``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.
    """
    import esda

    x_keys, y_keys = _normalize_pairs(x, y)

    w = _build_weights(adata, neighbors_key)
    log.info("moran_local_bv: %d pairs, %d cells, n_perms=%d", len(x_keys), adata.n_obs, n_perms)

    frames = []
    pair_labels = []
    for xk, yk in zip(x_keys, y_keys):
        xv = _extract_values(adata, xk, layer)
        yv = _extract_values(adata, yk, layer)
        lmi = esda.Moran_Local_BV(xv, yv, w, permutations=n_perms, n_jobs=n_jobs)

        if n_perms > 0:
            p_sim = np.asarray(getattr(lmi, "p_sim", np.full(adata.n_obs, np.nan)))
            z_sim = np.asarray(getattr(lmi, "z_sim", np.full(adata.n_obs, np.nan)))
            p_adj = _adj_pvalues(p_sim, p_adjust_method)
        else:
            p_sim = np.full(adata.n_obs, np.nan)
            z_sim = np.full(adata.n_obs, np.nan)
            p_adj = np.full(adata.n_obs, np.nan)

        quadrant = pd.array(
            [_QUAD_LABELS.get(int(q), "NS") for q in lmi.q], dtype="string"
        )

        frames.append(
            pd.DataFrame(
                {"Ii": lmi.Is, "z_sim": z_sim, "p_sim": p_sim, "p_adj": p_adj, "quadrant": quadrant},
                index=adata.obs_names,
            )
        )
        pair_labels.append(f"{xk}:{yk}")

    result = pd.concat(frames, keys=pair_labels, names=["key", "obs_name"])
    adata.uns[key_added] = result
    log.info("moran_local_bv: done")
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def getis_ord(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    p_adjust_method: str = "fdr_bh",
    alpha: float = 0.05,
    n_jobs: int = 1,
    key_added: str = "getis_ord",
) -> pd.DataFrame:
    """Local Getis-Ord G* hot/cold spot detection.

    Unlike LISA, G* has a single interpretable axis: positive z-score = local
    high-expression cluster (hot spot), negative = cold spot.  Spatial outliers
    (HL/LH) are not distinguished.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.
    Each key is looked up in ``var_names`` first, then ``obs`` columns.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more gene / obs column names to test.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer to use.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Permutations for empirical p-values (``0`` uses normality).
    p_adjust_method
        Multiple-testing correction applied per key across all cells.
    alpha
        Significance threshold used to assign ``type``
        (``'hot'`` / ``'cold'`` / ``'NS'``).
    n_jobs
        Parallel workers for the permutations, forwarded to
        ``esda.G_Local``. ``1`` (default) runs sequentially. ``-1`` uses all
        available cores.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(key, obs_name)`` rows with columns ``Gs``, ``z_sim``,
        ``p_sim``, ``p_adj``, ``type``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> go = ab.tl.getis_ord(mdata, ["EPCAM", "CD3D"])
    >>> go.loc["EPCAM"].query("type == 'hot'")
    """
    import esda

    if isinstance(keys, str):
        keys = [keys]

    w = _build_weights(adata, neighbors_key)
    log.info("getis_ord: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    frames = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        # star=1 (rather than star=True): our spatial graph has a zero
        # diagonal (no self-loops) and is already row-standardized, so
        # esda.G_Local can't infer a self-weight for Gi* on its own --
        # star=True leaves it to guess one *per row* (defaults to that row's
        # max neighbor weight, with a UserWarning on every call). Passing an
        # explicit self-weight of 1 (the standard Ord & Getis / GeoDa
        # convention: the point counts as an ordinary neighbor of itself)
        # gives every cell the same, reproducible self-weight instead of one
        # that silently varies with local connectivity density, and avoids
        # the warning.
        lG = esda.G_Local(y, w, permutations=n_perms, transform="r", star=1, n_jobs=n_jobs)

        if n_perms > 0:
            z_vals = np.asarray(lG.z_sim)
            p_vals = np.asarray(lG.p_sim)
            p_adj = _adj_pvalues(p_vals, p_adjust_method)
        else:
            z_vals = np.asarray(lG.Zs)
            p_vals = np.asarray(lG.p_norm)
            p_adj = p_vals  # no adjustment without permutations

        spot_type = pd.array(
            [
                "hot" if z > 0 and p < alpha else "cold" if z < 0 and p < alpha else "NS"
                for z, p in zip(z_vals, p_adj)
            ],
            dtype="string",
        )

        frames.append(
            pd.DataFrame(
                {"Gs": np.asarray(lG.Gs), "z_sim": z_vals, "p_sim": p_vals, "p_adj": p_adj, "type": spot_type},
                index=adata.obs_names,
            )
        )

    result = pd.concat(frames, keys=keys, names=["key", "obs_name"])
    adata.uns[key_added] = result
    log.info("getis_ord: done")
    return result


@with_adata
def spatial_lag(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    suffix: str = "_lag",
) -> None:
    """Compute the spatial lag of expression or obs variables.

    For each key, writes the weighted average of neighbors' values as a new
    ``obs`` column.  Numeric variables use the row-standardised spatial
    average; categorical variables use the most-common neighbor label.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.
    Each key is looked up in ``var_names`` first, then ``obs`` columns.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more gene / obs column names.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer for gene keys.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    suffix
        Appended to each key name to form the new ``obs`` column name
        (default ``'_lag'``).

    Examples
    --------
    >>> ab.tl.spatial_lag(mdata, "EPCAM")
    >>> ab.tl.spatial_lag(mdata, "leiden")   # categorical
    """
    from libpysal.weights import lag_categorical, lag_spatial

    if isinstance(keys, str):
        keys = [keys]

    w = _build_weights(adata, neighbors_key)

    for key in keys:
        col_name = f"{key}{suffix}"
        if key in adata.var_names:
            y = _extract_values(adata, key, layer)
            adata.obs[col_name] = lag_spatial(w, y)
        elif key in adata.obs.columns:
            dtype = adata.obs[key].dtype
            is_cat = isinstance(dtype, pd.CategoricalDtype) or dtype.kind in ("O", "U", "S")
            if is_cat:
                labels = adata.obs[key].astype(str).to_numpy()
                adata.obs[col_name] = lag_categorical(w, labels)
            else:
                y = adata.obs[key].to_numpy(dtype=np.float64)
                adata.obs[col_name] = lag_spatial(w, y)
        else:
            raise KeyError(f"Key '{key}' not found in var_names or obs columns")

    log.info("spatial_lag: wrote %d lag columns", len(keys))


@with_adata
def spatial_silhouette(
    adata: ad.AnnData,
    groupby: str,
    *,
    use_rep: str = "X_pca",
    neighbors_key: str = "spatial_neighbors",
    key_added: str = "spatial_silhouette",
) -> None:
    """Boundary silhouette score for spatially-defined clusters.

    Measures how well each cell separates from its spatial neighbors in
    attribute space.  Cells at cluster boundaries score lower than cells deep
    within a cluster.  Differs from the standard sklearn silhouette, which
    uses global pairwise distances.

    Values are signed and centered at 0, like a z-score: positive means a
    boundary cell fits its own cluster better than the neighboring one it
    borders, negative means the opposite (a likely-noisy assignment); cells
    with no same-cluster spatial neighbor at all, or with no cross-cluster
    neighbor at all (not on any boundary), score exactly 0. Plot with a
    diverging colormap centered at 0 (e.g. ``cmap="astrolabe_div",
    vcenter=0.0``) -- the default sequential colormap makes real but modest
    variation (typically within roughly ±0.5 on real data, not the full ±1
    range) hard to read, since it doesn't distinguish "near 0" from "clearly
    negative" the way a diverging scale does.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column containing cluster labels (e.g. ``'leiden'``).
    modality
        For :class:`~mudata.MuData`: modality key.
    use_rep
        ``obsm`` key for the attribute embedding (default ``'X_pca'``).
        Falls back to ``X`` if not present.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    key_added
        ``obs`` key where per-cell scores are written.

    Examples
    --------
    >>> ab.tl.spatial_silhouette(mdata, groupby="leiden")
    >>> ab.pl.points(mdata, color="spatial_silhouette", cmap="astrolabe_div", vcenter=0.0)
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"obs column '{groupby}' not found")

    w = _build_weights(adata, neighbors_key)
    labels = adata.obs[groupby].to_numpy()

    if use_rep in adata.obsm:
        attr = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    else:
        X = adata.X
        if sp.issparse(X):
            attr = np.asarray(X.todense(), dtype=np.float64)
        else:
            attr = np.asarray(X, dtype=np.float64)

    log.info("spatial_silhouette: %d cells, groupby='%s', rep='%s'", adata.n_obs, groupby, use_rep)
    scores = _boundary_silhouette_graph(attr, labels, w)
    adata.obs[key_added] = scores
    log.info("spatial_silhouette: done")


def _ripley_l_kdtree(
    coords: np.ndarray,
    radii: np.ndarray,
    n_perms: int,
    rng: "np.random.Generator",
    n_jobs: int = 1,
) -> "tuple[np.ndarray, np.ndarray]":
    """KDTree-based Ripley's L — O(n log n × m) time, O(n log n) memory.

    Uses ``scipy.spatial.cKDTree.count_neighbors`` to count pairs within each
    radius without materialising an O(n²) distance matrix.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    xy_min = coords.min(axis=0)
    xy_max = coords.max(axis=0)
    area = max(float((xy_max[0] - xy_min[0]) * (xy_max[1] - xy_min[1])), 1.0)

    def _k(pts: np.ndarray) -> np.ndarray:
        tree = cKDTree(pts)
        # count_neighbors counts ordered pairs (i,j) with dist(i,j) < r,
        # including i==j self-pairs.  Subtracting len(pts) removes self-pairs;
        # the remaining count is already doubled ((i,j) and (j,i) both counted),
        # which is the correct numerator for K(r) = A/(n(n-1)) × Σ_{i≠j} I(d<r).
        # count_neighbors accepts the whole `radii` array and does a *single*
        # tree traversal for all of them ("Multiple radii are searched with a
        # single tree traversal" per scipy's docs) -- calling it once per
        # radius instead (as this used to) redundantly re-traverses the tree
        # `len(radii)` times for the same result; ~6x slower on a
        # 15k-point/200-radius benchmark.
        raw = tree.count_neighbors(tree, radii).astype(np.float64) - len(pts)
        return area * raw / (n * (n - 1))

    def _l(k: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(k, 0.0) / np.pi) - radii

    l_obs = _l(_k(coords))

    def _one_sim(child_rng: "np.random.Generator") -> np.ndarray:
        sim = child_rng.uniform(xy_min, xy_max, size=(n, 2))
        return _l(_k(sim))

    if n_perms > 0:
        # Derive independent child streams from the shared rng so multiple
        # groups (each calling this function once) still advance through a
        # single deterministic sequence, while perms within a group are
        # safe to run concurrently.
        child_rngs = _spawn_rngs(int(rng.integers(0, 2**63 - 1)), n_perms)
        sims = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_sim)(child_rng) for child_rng in child_rngs
        )
        l_sims = np.stack(sims)
    else:
        l_sims = np.empty((0, len(radii)), dtype=np.float64)

    return l_obs, l_sims


@with_adata(mirror_uns=from_kwarg("key_added"))
def ripley_l(
    adata: ad.AnnData,
    groupby: str | None = None,
    *,
    spatial_key: str = "spatial",
    support: int = 200,
    n_perms: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    key_added: str = "ripley_l",
) -> dict[str, pd.DataFrame]:
    """Ripley's L function for cell-type point patterns.

    Tests whether cells of a given type are spatially clustered beyond what
    is expected under complete spatial randomness (CSR).

    The returned ``L_obs``/``L_hi``/``L_lo`` columns hold the *detrended*
    quantity L(d) − d, not raw L(d) — this is why ``L_exp`` is a constant 0:
    raw L(d) equals *d* under CSR (a diagonal reference line), while the
    detrended L(d) − d equals 0 under CSR (a flat reference line), which is
    what :func:`~astrolabe.pl.ripley_plot` plots. Positive values indicate
    clustering at distance *d* (more same-type neighbors within *d* than
    random chance predicts); negative values indicate regularity/dispersion
    (fewer).

    Unlike Moran's I (which tests autocorrelation of a continuous variable),
    Ripley's L treats cells of a given type as a binary point pattern and
    characterizes the *scale* of clustering.

    Does not require :func:`~astrolabe.tl.spatial_neighbors`.

    Uses a KDTree-based range-counting algorithm — O(n log n × support) time
    and O(n log n) memory — so it scales to tens of thousands of cells per
    group without subsampling.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column to stratify by (e.g. ``'leiden'``, ``'cell_type'``).
        ``None`` treats all cells as a single group labeled ``'all'``.
    modality
        For :class:`~mudata.MuData`: modality key.
    spatial_key
        ``obsm`` key for centroid coordinates (in µm).
    support
        Number of distance steps at which to evaluate L(d).
    n_perms
        Simulation permutations for the CSR envelope
        (``0`` → no envelope; ``L_hi``/``L_lo`` will be ``NaN``).
    seed
        Random seed for reproducible CSR simulations.
    n_jobs
        Parallel workers for the CSR simulations within each group (joblib,
        thread-backed — no extra memory copies since arrays are shared, not
        pickled). ``1`` (default) runs sequentially. ``-1`` uses all
        available cores.
    key_added
        ``uns`` key where results are stored.

    Returns
    -------
    dict[str, pd.DataFrame]
        One entry per group.  Each DataFrame has distances (µm) as index and
        columns ``L_obs``, ``L_exp``, ``L_hi``, ``L_lo``, ``p_value``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> curves = ab.tl.ripley_l(mdata, groupby="leiden")
    >>> curves["0"].plot(y=["L_obs", "L_hi", "L_lo"])   # cluster 0
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Radii computed from full-dataset extent so all groups share the same x-axis.
    # max_r = min(x_range, y_range) / 4 follows Ripley's rule of thumb.
    xy_range = coords.max(axis=0) - coords.min(axis=0)
    max_r = float(min(xy_range)) / 4.0
    radii = np.linspace(0.0, max_r, support)

    if groupby is None:
        groups: dict[str, np.ndarray] = {"all": np.ones(adata.n_obs, dtype=bool)}
    else:
        if groupby not in adata.obs.columns:
            raise KeyError(f"obs column '{groupby}' not found")
        unique_vals = adata.obs[groupby].unique()
        groups = {str(v): adata.obs[groupby].values == v for v in unique_vals}

    log.info("ripley_l: %d groups, support=%d, n_perms=%d", len(groups), support, n_perms)
    results: dict[str, pd.DataFrame] = {}

    for name, mask in groups.items():
        group_coords = coords[mask]
        if group_coords.shape[0] < 3:
            log.warning("Group '%s' has fewer than 3 cells; skipping.", name)
            continue

        l_obs, l_sims = _ripley_l_kdtree(group_coords, radii, n_perms, rng, n_jobs=n_jobs)

        df = pd.DataFrame(
            {"L_obs": l_obs, "L_exp": np.zeros_like(l_obs)},
            index=pd.Index(radii, name="distance_um"),
        )

        if n_perms > 0:
            df["L_hi"] = np.percentile(l_sims, 97.5, axis=0)
            df["L_lo"] = np.percentile(l_sims, 2.5, axis=0)
            # empirical one-sided p: fraction of CSR sims with L(max_r) >= observed
            df["p_value"] = float((l_sims[:, -1] >= l_obs[-1]).mean())
        else:
            df["L_hi"] = np.nan
            df["L_lo"] = np.nan
            df["p_value"] = np.nan

        results[name] = df

    adata.uns[key_added] = results
    log.info("ripley_l: done — %d groups", len(results))
    return results


# ---------------------------------------------------------------------------
# Spatially variable genes
# ---------------------------------------------------------------------------


def _svg_moran(
    adata: ad.AnnData,
    genes: list[str],
    layer: str | None,
    neighbors_key: str,
    n_perms: int,
    p_adjust_method: str,
    n_jobs: int = 1,
) -> pd.DataFrame:
    import esda

    get_weights = _thread_local_weights_getter(adata, neighbors_key)

    def _one_gene(g: str) -> dict:
        y = _extract_values(adata, g, layer)
        mi = esda.Moran(y, get_weights(), permutations=n_perms, two_tailed=True)
        p = float(mi.p_sim if n_perms > 0 else mi.p_norm)
        return {"gene": g, "score": float(mi.I), "pval": p}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one_gene)(g) for g in genes)

    df = pd.DataFrame(rows).set_index("gene")
    df["pval_adj"] = _adj_pvalues(df["pval"].to_numpy(), p_adjust_method)
    return df


def _svg_geary(
    adata: ad.AnnData,
    genes: list[str],
    layer: str | None,
    neighbors_key: str,
    n_perms: int,
    p_adjust_method: str,
    n_jobs: int = 1,
) -> pd.DataFrame:
    import esda

    get_weights = _thread_local_weights_getter(adata, neighbors_key)

    def _one_gene(g: str) -> dict:
        y = _extract_values(adata, g, layer)
        gc = esda.Geary(y, get_weights(), permutations=n_perms)
        # Geary's C: C < 1 = positive autocorrelation, C > 1 = negative.
        # p_norm is the z-score-based p-value under the normality assumption;
        # p_sim is the permutation-based p-value when n_perms > 0.
        p = float(gc.p_sim if n_perms > 0 else gc.p_norm)
        return {"gene": g, "score": float(gc.C), "pval": p}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one_gene)(g) for g in genes)

    df = pd.DataFrame(rows).set_index("gene")
    df["pval_adj"] = _adj_pvalues(df["pval"].to_numpy(), p_adjust_method)
    return df


@with_adata(mirror_uns=from_kwarg("key_added"))
def spatially_variable_genes(
    adata: ad.AnnData,
    *,
    flavor: str = "moran",
    n_top_genes: int | None = None,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 0,
    p_adjust_method: str = "fdr_bh",
    n_jobs: int = 1,
    key_added: str = "spatially_variable",
) -> pd.DataFrame:
    """Identify spatially variable genes.

    Tests whether each gene's expression varies non-randomly across space,
    using one of two methods selected via ``flavor``.

    Results are written to ``var`` and stored in ``uns[key_added]``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    flavor
        Statistical method:

        ``'moran'`` (default)
            Global Moran's I.  High positive I → expression clusters in space.
            Fast and interpretable; the standard choice for most workflows.

        ``'geary'``
            Geary's C.  C < 1 indicates positive spatial autocorrelation.
            More sensitive to *local* spatial variation than Moran's I; less
            affected by global gradients.

        Both flavors require :func:`~astrolabe.tl.spatial_neighbors` to have
        been run first.

    n_top_genes
        Mark the top *n* genes as spatially variable in
        ``var['sv_spatially_variable']``.  ``None`` marks all genes with
        ``pval_adj < 0.05``.
    layer
        Expression layer.  ``None`` uses ``X``.
    neighbors_key
        Key in ``uns`` written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Permutations for empirical p-values.  ``0`` (default) uses an analytic
        approximation, which is strongly recommended for genome-wide scans —
        permutation testing across hundreds of genes is prohibitively slow.
    p_adjust_method
        Multiple-testing correction (e.g. ``'fdr_bh'``, ``'bonferroni'``).
    n_jobs
        Parallel workers for the per-gene loop (joblib, thread-backed — no
        extra memory copies since arrays are shared, not pickled). ``1``
        (default) runs sequentially. ``-1`` uses all available cores.
        Ignored on the GPU path (``ab.use_gpu(True)``), which parallelizes
        across genes on-device instead. Applies regardless of ``n_perms``,
        since the per-gene loop is the dominant cost even at ``n_perms=0``.
    modality
        For :class:`~mudata.MuData`: modality key.
    key_added
        ``uns`` key where the full result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per gene with columns ``score``, ``pval``, ``pval_adj``,
        ``spatially_variable`` (bool), ``flavor``.
        ``score`` is Moran's I for ``'moran'`` and Geary's C for ``'geary'``.
        Also stored in ``adata.uns[key_added]`` and written to ``adata.var``
        as ``sv_score``, ``sv_pval``, ``sv_pval_adj``, ``sv_spatially_variable``.
        The ``uns[key_added]`` DataFrame is mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData
        (the ``var`` columns are not mirrored — they stay adata-resident).

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> svgs = ab.tl.spatially_variable_genes(mdata, n_top_genes=200)
    >>> svgs.sort_values("score", ascending=False).head(10)

    >>> # Geary's C — more sensitive to local variation
    >>> svgs = ab.tl.spatially_variable_genes(mdata, flavor="geary", n_top_genes=200)
    """
    _FLAVORS = ("moran", "geary")
    if flavor not in _FLAVORS:
        raise ValueError(f"flavor must be one of {_FLAVORS!r}, got {flavor!r}")

    if gpu_enabled():
        rsc = _require_rapids()
        conn_key = f"{neighbors_key}_connectivities"
        rsc_uns_key = "moranI" if flavor == "moran" else "gearyC"
        score_col = "I" if flavor == "moran" else "C"
        with _gpu_X(adata):
            rsc.gr.spatial_autocorr(
                adata,
                connectivity_key=conn_key,
                mode=flavor,
                layer=layer,
                n_perms=n_perms if n_perms > 0 else None,
                corr_method=p_adjust_method,
            )
        # Remap squidpy output to astrolabe var columns
        rsc_df = adata.uns.pop(rsc_uns_key)  # remove squidpy key
        df = pd.DataFrame(index=adata.var_names)
        df["score"] = rsc_df[score_col].reindex(adata.var_names)
        df["pval"] = rsc_df["pval_norm"].reindex(adata.var_names)
        df["pval_adj"] = rsc_df["pval_norm_fdr_bh"].reindex(adata.var_names)
        df["flavor"] = flavor
        if n_top_genes is not None:
            ascending = flavor == "geary"
            top_idx = df["score"].sort_values(ascending=ascending).head(n_top_genes).index
            df["spatially_variable"] = df.index.isin(top_idx)
        else:
            df["spatially_variable"] = df["pval_adj"] < 0.05
        for col in ("score", "pval", "pval_adj", "spatially_variable", "flavor"):
            adata.var[f"sv_{col}"] = df[col].reindex(adata.var_names).to_numpy()
        adata.uns[key_added] = df
        n_sv = int(df["spatially_variable"].sum())
        log.info("spatially_variable_genes (GPU): %d SVGs identified", n_sv)
        return df

    genes = list(adata.var_names)
    log.info(
        "spatially_variable_genes: flavor='%s', %d genes, %d cells",
        flavor,
        len(genes),
        adata.n_obs,
    )

    if flavor == "moran":
        df = _svg_moran(adata, genes, layer, neighbors_key, n_perms, p_adjust_method, n_jobs=n_jobs)
    else:
        df = _svg_geary(adata, genes, layer, neighbors_key, n_perms, p_adjust_method, n_jobs=n_jobs)

    df["flavor"] = flavor

    if n_top_genes is not None:
        # For Geary's C, lower score = more spatial; for all others, higher is more spatial.
        ascending = flavor == "geary"
        top_idx = df["score"].sort_values(ascending=ascending).head(n_top_genes).index
        df["spatially_variable"] = df.index.isin(top_idx)
    else:
        df["spatially_variable"] = df["pval_adj"] < 0.05

    # Write per-gene columns back to var
    for col in ("score", "pval", "pval_adj", "spatially_variable", "flavor"):
        adata.var[f"sv_{col}"] = df[col].reindex(adata.var_names).to_numpy()

    adata.uns[key_added] = df
    n_sv = int(df["spatially_variable"].sum())
    log.info("spatially_variable_genes: %d SVGs identified", n_sv)
    return df


@with_adata(mirror_uns=from_kwarg("key_added"))
def geary_c(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 1000,
    key_added: str = "geary_c",
) -> pd.DataFrame:
    """Global Geary's C spatial autocorrelation for one or more keys.

    Geary's C is a complementary measure to Moran's I.  Values significantly
    below 1 indicate positive spatial autocorrelation (nearby cells are
    similar); values above 1 indicate negative autocorrelation.

    Requires :func:`~astrolabe.tl.spatial_neighbors` to have been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    keys
        One or more gene / obs column names to test.
    modality
        For :class:`~mudata.MuData`: modality key.
    layer
        Expression layer.  ``None`` uses ``X``.
    neighbors_key
        ``uns`` key written by :func:`~astrolabe.tl.spatial_neighbors`.
    n_perms
        Permutations for empirical p-values (``0`` uses normality
        approximation only).
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per key with columns ``C``, ``EC``, ``z_norm``, ``p_norm``,
        and (when ``n_perms > 0``) ``z_sim``, ``p_sim``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata)
    >>> result = ab.tl.geary_c(mdata, ["EPCAM", "CD3D"])
    >>> result.sort_values("C")
    """
    import esda

    if isinstance(keys, str):
        keys = [keys]

    w = _build_weights(adata, neighbors_key)
    log.info("geary_c: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    rows = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        gc = esda.Geary(y, w, permutations=n_perms)
        row: dict[str, float] = {
            "C": gc.C,
            "EC": gc.EC,
            "z_norm": gc.z_norm,
            "p_norm": gc.p_norm,
        }
        if n_perms > 0:
            row["z_sim"] = gc.z_sim
            row["p_sim"] = gc.p_sim
        rows.append(row)

    result = pd.DataFrame(rows, index=pd.Index(keys, name="key"))
    adata.uns[key_added] = result
    log.info("geary_c: done — min C=%.3f (%s)", result["C"].min(), result["C"].idxmin())
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def colocation_quotient(
    adata: ad.AnnData,
    *,
    mode: str = "cell",
    layer: str | None = None,
    transcripts: "object | None" = None,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    pixel_size_um: float = 10.0,
    genes: list[str] | None = None,
    pairs: list[tuple[str, str]] | None = None,
    threshold: float = 0.0,
    min_locs: int = 5,
    key_added: str = "clq",
) -> pd.DataFrame:
    """Compute gene-pair co-location quotients (CLQ).

    The CLQ measures whether two genes co-occur in the same spatial location
    more or less than expected by chance:

    ``CLQ(A, B) = [n_locs(A ∩ B) × N_total] / [n_locs(A) × n_locs(B)]``

    * CLQ > 1 — genes co-occur more than expected (co-enriched)
    * CLQ = 1 — independent
    * CLQ < 1 — genes avoid co-occurring

    Two modes are supported:

    ``'cell'`` (default)
        Locations are cells (or rows of any AnnData layer).  The expression
        matrix is binarized at *threshold* and the CLQ is computed from the
        resulting presence/absence matrix.

    ``'pixel'``
        Segmentation-free.  The spatial domain is divided into square bins of
        *pixel_size_um* µm; each bin is treated as one location.  Requires the
        *transcripts* argument.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        For :class:`~mudata.MuData`: modality key.
    mode
        ``'cell'`` or ``'pixel'``.
    layer
        Expression layer for ``mode='cell'``.  ``None`` uses ``X``.
    transcripts
        Transcript table (:class:`pyarrow.Table` or :class:`pandas.DataFrame`)
        required for ``mode='pixel'``.
    x_col, y_col
        Coordinate column names (``mode='pixel'``).
    feature_col
        Gene/feature column name (``mode='pixel'``).
    pixel_size_um
        Bin edge length in µm (``mode='pixel'``).
    genes
        Restrict to this gene subset.  ``None`` includes all genes / features.
    pairs
        Compute CLQ only for these specific (gene_a, gene_b) pairs.  ``None``
        computes all pairs satisfying *min_locs*.
    threshold
        Binarisation cutoff for ``mode='cell'``: a cell counts as expressing
        a gene when its count exceeds *threshold* (default 0.0 = any nonzero).
    min_locs
        Skip genes present in fewer than this many cells / bins.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per pair with columns ``gene_a``, ``gene_b``, ``clq``,
        ``n_a``, ``n_b``, ``n_ab``, ``n_locs``, ``mode``, ``pixel_size_um``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> clq = ab.tl.colocation_quotient(mdata)
    >>> clq.sort_values('clq', ascending=False).head(10)

    >>> # Nucleus-only CLQ (requires create_compartment_layers first)
    >>> ab.tl.colocation_quotient(mdata, layer='nucleus', key_added='clq_nucleus')

    >>> # Segmentation-free pixel CLQ
    >>> transcripts = ab.io.read_xenium_transcripts(mdata)
    >>> ab.tl.colocation_quotient(mdata, mode='pixel', transcripts=transcripts,
    ...                            pixel_size_um=10.0, key_added='clq_pixel')
    """
    _MODES = ("cell", "pixel")
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES!r}, got {mode!r}")

    gene_set: set[str] | None = set(genes) if genes is not None else None
    empty_df = pd.DataFrame(
        columns=["gene_a", "gene_b", "clq", "n_a", "n_b", "n_ab", "n_locs", "mode", "pixel_size_um"]
    )

    # ── Build binary (locations × genes) sparse matrix B ────────────────────

    if mode == "cell":
        if layer is not None and layer not in adata.layers:
            raise KeyError(
                f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}"
            )
        M = adata.layers[layer] if layer is not None else adata.X
        if not sp.issparse(M):
            M = sp.csr_matrix(M)

        if gene_set is not None:
            keep = np.array([g in gene_set for g in adata.var_names], dtype=bool)
            M = M[:, keep]
            gene_list = [g for g in adata.var_names if g in gene_set]
        else:
            gene_list = list(adata.var_names)

        B = (M > threshold).astype(np.float32)  # (n_obs × n_genes)
        N = adata.n_obs

    else:  # pixel
        if transcripts is None:
            raise ValueError("mode='pixel' requires the transcripts argument.")

        try:
            import pyarrow as _pa
            df_pix: pd.DataFrame = (
                transcripts.to_pandas()  # type: ignore[union-attr]
                if isinstance(transcripts, _pa.Table)
                else transcripts  # type: ignore[assignment]
            )
        except ImportError:
            df_pix = transcripts  # type: ignore[assignment]

        for col in (x_col, y_col, feature_col):
            if col not in df_pix.columns:
                raise KeyError(f"Column '{col}' not found in transcript table.")

        x = df_pix[x_col].to_numpy(dtype=np.float64)
        y = df_pix[y_col].to_numpy(dtype=np.float64)
        feat_all = df_pix[feature_col].to_numpy()

        x_min, y_min = x.min(), y.min()
        bin_x = np.floor((x - x_min) / pixel_size_um).astype(np.int64)
        bin_y = np.floor((y - y_min) / pixel_size_um).astype(np.int64)
        n_y_bins = int(bin_y.max()) + 1
        bin_id = bin_x * n_y_bins + bin_y

        # N_bins = unique bins occupied by any transcript (before gene filter)
        _, bin_remapped = np.unique(bin_id, return_inverse=True)
        N = int(bin_remapped.max()) + 1

        if gene_set is not None:
            mask = np.array([f in gene_set for f in feat_all], dtype=bool)
            feat_sub = feat_all[mask]
            bin_sub = bin_remapped[mask]
        else:
            feat_sub = feat_all
            bin_sub = bin_remapped

        unique_genes = sorted(set(feat_sub.tolist()))
        if genes is not None:
            gene_list = [g for g in genes if g in set(unique_genes)]
        else:
            gene_list = unique_genes

        if not gene_list:
            adata.uns[key_added] = empty_df
            return empty_df

        gene_to_idx_pix = {g: i for i, g in enumerate(gene_list)}
        n_g = len(gene_list)
        gene_col_arr = np.array(
            [gene_to_idx_pix.get(f, -1) for f in feat_sub], dtype=np.int32
        )
        known = gene_col_arr >= 0
        B = sp.coo_matrix(
            (
                np.ones(known.sum(), dtype=np.float32),
                (bin_sub[known], gene_col_arr[known]),
            ),
            shape=(N, n_g),
        ).tocsr()
        B.data[:] = 1.0  # binarize (sum duplicates → 1)

    # ── Common CLQ computation ───────────────────────────────────────────────

    if not gene_list:
        adata.uns[key_added] = empty_df
        return empty_df

    co = B.T @ B  # (n_genes × n_genes), entry [i,j] = n_locs with both i and j
    freq = np.asarray(co.diagonal()).ravel()  # n_locs per gene (diagonal)
    co_dense = co.toarray()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    if pairs is not None:
        pair_indices: list[tuple[int, int]] = []
        for a, b in pairs:
            if a not in gene_to_idx:
                log.warning("colocation_quotient: gene '%s' not found; skipping pair.", a)
                continue
            if b not in gene_to_idx:
                log.warning("colocation_quotient: gene '%s' not found; skipping pair.", b)
                continue
            pair_indices.append((gene_to_idx[a], gene_to_idx[b]))
    else:
        eligible = np.where(freq >= min_locs)[0]
        pair_indices = [(int(i), int(j)) for i in eligible for j in eligible if i < j]

    rows_out = []
    for i_a, i_b in pair_indices:
        n_a = int(freq[i_a])
        n_b = int(freq[i_b])
        n_ab = int(co_dense[i_a, i_b])
        denom = n_a * n_b
        clq = float(n_ab * N / denom) if denom > 0 else np.nan
        rows_out.append({
            "gene_a": gene_list[i_a],
            "gene_b": gene_list[i_b],
            "clq": clq,
            "n_a": n_a,
            "n_b": n_b,
            "n_ab": n_ab,
            "n_locs": N,
            "mode": mode,
            "pixel_size_um": pixel_size_um if mode == "pixel" else np.nan,
        })

    result = pd.DataFrame(rows_out) if rows_out else empty_df
    adata.uns[key_added] = result
    log.info(
        "colocation_quotient: %d pairs computed (mode='%s', N=%d)",
        len(result), mode, N,
    )
    return result
