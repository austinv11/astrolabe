"""Spatial statistical tests.

Global and local Moran's I, bivariate variants, Getis-Ord G*, spatial lag,
spatial silhouette, and Ripley's L all delegate to the PySAL ecosystem
(``esda``, ``libpysal``, ``pointpats``) for correctness and maintainability.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import with_adata

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


def _normalise_pairs(
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


@with_adata
def moran(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
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
        Also stored in ``adata.uns[key_added]``.

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


@with_adata
def moran_local(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
    p_adjust_method: str = "fdr_bh",
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
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(key, obs_name)`` rows with columns ``Ii``, ``z_sim``,
        ``p_sim``, ``p_adj``, ``quadrant``.
        Access per-key results with ``result.loc['<key>']``.
        Also stored in ``adata.uns[key_added]``.

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
        lmi = esda.Moran_Local(y, w, permutations=n_perms)

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


@with_adata
def moran_bv(
    adata: ad.AnnData,
    x: str | list[str],
    y: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
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
        Permutations (``0`` = normality approximation only).
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(x, y)`` rows with columns ``I``, ``EI``, ``z_norm``,
        ``p_norm``, and (when ``n_perms > 0``) ``z_sim``, ``p_sim``.
        Also stored in ``adata.uns[key_added]``.

    Examples
    --------
    >>> result = ab.tl.moran_bv(mdata, ["EPCAM", "CD274"], ["CXCL12", "PDCD1"])
    >>> result.sort_values("I", ascending=False)
    """
    import esda

    x_keys, y_keys = _normalise_pairs(x, y)

    w = _build_weights(adata, neighbors_key)
    log.info("moran_bv: %d pairs, %d cells, n_perms=%d", len(x_keys), adata.n_obs, n_perms)

    rows = []
    for xk, yk in zip(x_keys, y_keys):
        xv = _extract_values(adata, xk, layer)
        yv = _extract_values(adata, yk, layer)
        mi = esda.Moran_BV(xv, yv, w, permutations=n_perms)
        row: dict[str, float] = {
            "I": mi.I,
            "EI": mi.EI_sim if hasattr(mi, "EI_sim") and n_perms > 0 else getattr(mi, "EI", np.nan),
            "z_norm": mi.z_norm,
            "p_norm": mi.p_norm,
        }
        if n_perms > 0:
            row["z_sim"] = mi.z_sim
            row["p_sim"] = mi.p_sim
        rows.append(row)

    idx = pd.MultiIndex.from_tuples(list(zip(x_keys, y_keys)), names=["x", "y"])
    result = pd.DataFrame(rows, index=idx)
    adata.uns[key_added] = result
    log.info("moran_bv: done")
    return result


@with_adata
def moran_local_bv(
    adata: ad.AnnData,
    x: str | list[str],
    y: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
    p_adjust_method: str = "fdr_bh",
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
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(pair, obs_name)`` rows where ``pair = 'x:y'``, with
        columns ``Ii``, ``z_sim``, ``p_sim``, ``p_adj``, ``quadrant``.
        Also stored in ``adata.uns[key_added]``.
    """
    import esda

    x_keys, y_keys = _normalise_pairs(x, y)

    w = _build_weights(adata, neighbors_key)
    log.info("moran_local_bv: %d pairs, %d cells, n_perms=%d", len(x_keys), adata.n_obs, n_perms)

    frames = []
    pair_labels = []
    for xk, yk in zip(x_keys, y_keys):
        xv = _extract_values(adata, xk, layer)
        yv = _extract_values(adata, yk, layer)
        lmi = esda.Moran_Local_BV(xv, yv, w, permutations=n_perms)

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


@with_adata
def getis_ord(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
    p_adjust_method: str = "fdr_bh",
    alpha: float = 0.05,
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
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        MultiIndex ``(key, obs_name)`` rows with columns ``Gs``, ``z_sim``,
        ``p_sim``, ``p_adj``, ``type``.
        Also stored in ``adata.uns[key_added]``.

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
        lG = esda.G_Local(y, w, permutations=n_perms, transform="r", star=True)

        if n_perms > 0:
            z_vals = np.asarray(lG.z_sim)
            p_vals = np.asarray(lG.p_sim)
            p_adj = _adj_pvalues(p_vals, p_adjust_method)
        else:
            z_vals = np.asarray(lG.z_norm)
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
    >>> ab.pl.points(mdata, color="spatial_silhouette")
    """
    import esda

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
    scores_df = esda.boundary_silhouette(attr, labels, w)

    if isinstance(scores_df, pd.DataFrame):
        score_col = next(
            (c for c in scores_df.columns if "score" in c.lower()), scores_df.columns[0]
        )
        adata.obs[key_added] = scores_df[score_col].values
    else:
        adata.obs[key_added] = np.asarray(scores_df).ravel()

    log.info("spatial_silhouette: done")


@with_adata
def ripley_l(
    adata: ad.AnnData,
    groupby: str | None = None,
    *,
    spatial_key: str = "spatial",
    support: int = 200,
    n_perms: int = 99,
    key_added: str = "ripley_l",
) -> dict[str, pd.DataFrame]:
    """Ripley's L function for cell-type point patterns.

    Tests whether cells of a given type are spatially clustered beyond what
    is expected under complete spatial randomness (CSR).  Positive L(d) values
    indicate clustering at distance *d*; L(d) = 0 under CSR.

    Unlike Moran's I (which tests autocorrelation of a continuous variable),
    Ripley's L treats cells of a given type as a binary point pattern and
    characterises the *scale* of clustering.

    Does not require :func:`~astrolabe.tl.spatial_neighbors`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column to stratify by (e.g. ``'leiden'``, ``'cell_type'``).
        ``None`` treats all cells as a single group labelled ``'all'``.
    modality
        For :class:`~mudata.MuData`: modality key.
    spatial_key
        ``obsm`` key for centroid coordinates (in µm).
    support
        Number of distance steps at which to evaluate L(d).
    n_perms
        Simulation permutations for the CSR envelope
        (``0`` → no envelope; ``L_hi``/``L_lo`` will be ``NaN``).
    key_added
        ``uns`` key where results are stored.

    Returns
    -------
    dict[str, pd.DataFrame]
        One entry per group.  Each DataFrame has distances (µm) as index and
        columns ``L_obs``, ``L_exp``, ``L_hi``, ``L_lo``, ``p_value``.
        Also stored in ``adata.uns[key_added]``.

    Examples
    --------
    >>> curves = ab.tl.ripley_l(mdata, groupby="leiden")
    >>> curves["0"].plot(y=["L_obs", "L_hi", "L_lo"])   # cluster 0
    """
    try:
        from pointpats.distance_statistics import l_test
    except ImportError:
        from pointpats import l_test  # older API

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

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
        try:
            res = l_test(
                group_coords,
                support=support,
                permutations=n_perms,
                keep_simulations=(n_perms > 0),
            )
        except Exception as exc:
            log.warning("ripley_l failed for group '%s': %s", name, exc)
            continue

        distances = np.asarray(res.support)
        l_obs = np.asarray(res.statistic)
        l_exp = np.zeros_like(l_obs)

        df = pd.DataFrame(
            {"L_obs": l_obs, "L_exp": l_exp},
            index=pd.Index(distances, name="distance_um"),
        )

        sims = getattr(res, "simulations", None)
        if n_perms > 0 and sims is not None:
            sims = np.asarray(sims)
            if sims.ndim == 2:
                # normalize to (n_perms, n_support)
                if sims.shape[0] == len(distances):
                    sims = sims.T
                df["L_hi"] = np.percentile(sims, 97.5, axis=0)
                df["L_lo"] = np.percentile(sims, 2.5, axis=0)
            else:
                df["L_hi"] = np.nan
                df["L_lo"] = np.nan
        else:
            df["L_hi"] = np.nan
            df["L_lo"] = np.nan

        pv = getattr(res, "pvalue", None)
        df["p_value"] = np.asarray(pv) if pv is not None else np.nan

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
) -> pd.DataFrame:
    import esda

    w = _build_weights(adata, neighbors_key)
    rows = []
    for g in genes:
        y = _extract_values(adata, g, layer)
        mi = esda.Moran(y, w, permutations=n_perms, two_tailed=True)
        p = float(mi.p_sim if n_perms > 0 else mi.p_norm)
        rows.append({"gene": g, "score": float(mi.I), "pval": p})

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
) -> pd.DataFrame:
    import esda

    w = _build_weights(adata, neighbors_key)
    rows = []
    for g in genes:
        y = _extract_values(adata, g, layer)
        gc = esda.Geary(y, w, permutations=n_perms)
        # Geary's C: C < 1 = positive autocorrelation, C > 1 = negative.
        # p_z_sim is the z-score-based p-value under the normality assumption;
        # p_sim is the permutation-based p-value when n_perms > 0.
        p = float(gc.p_sim if n_perms > 0 else gc.p_z_sim)
        rows.append({"gene": g, "score": float(gc.C), "pval": p})

    df = pd.DataFrame(rows).set_index("gene")
    df["pval_adj"] = _adj_pvalues(df["pval"].to_numpy(), p_adjust_method)
    return df


@with_adata
def spatially_variable_genes(
    adata: ad.AnnData,
    *,
    flavor: str = "moran",
    n_top_genes: int | None = None,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 0,
    p_adjust_method: str = "fdr_bh",
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
        df = _svg_moran(adata, genes, layer, neighbors_key, n_perms, p_adjust_method)
    else:
        df = _svg_geary(adata, genes, layer, neighbors_key, n_perms, p_adjust_method)

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


@with_adata
def geary_c(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    n_perms: int = 999,
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
        Also stored in ``adata.uns[key_added]``.

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
