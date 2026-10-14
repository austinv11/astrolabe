"""Spatial statistical tests.

Moran's I, bivariate variants, Geary's C, Getis-Ord G*, spatial lag, and
spatial silhouette delegate to the PySAL ecosystem (``esda``, ``libpysal``).
Ripley's L uses a KDTree-based range-counting implementation (scipy) that is
O(n log n) in time and memory — no pointpats dependency.
"""

from __future__ import annotations

import logging
import threading
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._statistics import (permutation_p, adjust_p, annotate_permutation_resolution,
                                   PermutationResolutionWarning)
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


def _perm_normal_p(z_sim):
    """Normal approximation diagnostic; never used to control local FDR."""
    from scipy.stats import norm
    return 2 * norm.sf(np.abs(z_sim))


def _gated_quadrants(q, p, alpha):
    return pd.array([
        _QUAD_LABELS.get(int(qi), "NS") if np.isfinite(pi) and pi < alpha
        else "NS" if np.isfinite(pi) else "not_tested"
        for qi, pi in zip(q, p)
    ], dtype="string")


def _two_sided_perm_p(stat, sims):
    return permutation_p(stat, sims)[0]


def _directed_perm_p(stat, sims):
    """Exploratory smaller tail, without two-sided significance semantics."""
    return np.minimum(permutation_p(stat, sims, alternative="greater")[0],
                      permutation_p(stat, sims, alternative="less")[0])


def _local_family(result, method, alpha, *, getis=False):
    result["p_normal_diagnostic"] = _perm_normal_p(result["z_sim"].to_numpy())
    result["p_adj"] = adjust_p(result["p_sim"].to_numpy(), method)
    result["status"] = np.where(result.p_sim.notna(), "tested", "not_tested")
    if getis:
        result["type"] = pd.array([
            "hot" if z > 0 else "cold" for z in result["z_sim"]
        ], dtype="string")
        result.loc[result.p_adj.ge(alpha), "type"] = "NS"
        result.loc[result.p_adj.isna(), "type"] = "not_tested"
    else:
        result["quadrant_raw"] = result["quadrant"].where(np.isfinite(result["Ii"]), pd.NA)
        result.loc[result.p_adj.ge(alpha), "quadrant"] = "NS"
        result.loc[result.p_adj.isna(), "quadrant"] = "not_tested"
    result.attrs.update(p_adjust_method=method, fdr_family="all requested cells and fields",
                        null="conditional randomization", inference="empirical equal tails")
    annotate_permutation_resolution(result, "p_sim", "n_valid", method=method,
                                   alpha=alpha, label="local spatial tests")
    return result


def _permutation_blocks(adata, neighbors_key):
    """Use the graph's library grouping for exchangeable expression draws."""
    library_key = adata.uns.get(neighbors_key, {}).get("params", {}).get("library_key", "library_id")
    if library_key is None or library_key not in adata.obs:
        return [np.arange(adata.n_obs)], library_key
    labels = adata.obs[library_key]
    if labels.isna().any():
        raise ValueError("library labels must be present for spatial inference")
    codes, levels = pd.factorize(labels, sort=False)
    if len(levels) > 1:
        graph = _get_conn_matrix(adata, neighbors_key).tocoo()
        if ((graph.data != 0) & (codes[graph.row] != codes[graph.col])).any():
            raise ValueError("spatial inference graph contains edges across libraries")
    return [np.flatnonzero(codes == i) for i in range(len(levels))], library_key


def _local_by_library(adata, function, args, kwargs, *, getis=False):
    """Fit conditional local nulls per library, then correct one pooled family."""
    blocks, library_key = _permutation_blocks(adata, kwargs["neighbors_key"])
    if len(blocks) <= 1:
        return None
    if any(len(block) < 3 for block in blocks):
        raise ValueError("local spatial inference requires at least three cells per library")
    parts = []
    for block in blocks:
        # A writable subset prevents the function's result store from mutating
        # the original; each library is processed and released in turn.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PermutationResolutionWarning)
            parts.append(function(adata[block].copy(), *args, **kwargs))
    result = pd.concat(parts)
    keys = result.index.get_level_values("key").unique()
    result = result.reindex(pd.MultiIndex.from_product([keys, adata.obs_names], names=["key", "obs_name"]))
    if not getis:
        result["quadrant"] = result["quadrant_raw"].copy()
    result = _local_family(result, kwargs["p_adjust_method"], kwargs["alpha"], getis=getis)
    result.attrs.update(library_key=library_key, n_libraries=len(blocks),
                        null="within-library conditional randomization", n_perms=int(kwargs["n_perms"]))
    adata.uns[kwargs["key_added"]] = result
    return result


def _shuffle_blocks(values, rng, blocks):
    if len(blocks) == 1:
        return rng.permutation(values)
    shuffled = values.copy()
    for block in blocks:
        shuffled[block] = rng.permutation(values[block])
    return shuffled


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_weights(adata: ad.AnnData, neighbors_key: str, *, row_standardize: bool = True):
    """Build a libpysal W from the stored obsp connectivity.

    ``row_standardize=False`` leaves the raw (zero-diagonal, unstandardized)
    weights untouched -- required by :func:`getis_ord`, whose ``esda.G_Local``
    call fills in a Gi* self-weight on the *diagonal* and then row-standardizes
    once, itself, via its own ``transform=`` argument. Row-standardizing here
    first (as every other caller wants) and letting esda standardize *again*
    is not a harmless no-op for G_Local specifically: esda's diagonal fill
    happens between the two standardizations, against a matrix whose
    off-diagonal entries already sum to 1 -- so the second pass renormalizes
    `1 (self) + 1 (already-normalized neighbor total)` down to a self-weight
    of exactly 0.5 for every cell, regardless of neighbor count, instead of
    the intended ~1/(k+1). Every other caller here (moran/geary/etc.) doesn't
    add a diagonal entry, so a redundant second row-standardization on an
    already-row-standardized matrix is a true no-op for them.
    """
    from libpysal.weights import WSP

    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors() first"
        )
    conn = _get_conn_matrix(adata, neighbors_key).tocsr(copy=True)
    conn.setdiag(0)
    conn.eliminate_zeros()
    w = WSP(conn).to_W()
    if row_standardize:
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


def _get_conn_matrix(adata: ad.AnnData, neighbors_key: str) -> sp.spmatrix:
    """Return the raw sparse connectivity matrix from obsp (no libpysal)."""
    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors() first"
        )
    conn = adata.obsp[conn_key]
    return conn.get() if hasattr(conn, "get") else conn


def _chunked_distance_sum(
    attr: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    *,
    max_block_elements: int,
    zero_self_pairs: bool,
) -> np.ndarray:
    """Sum of Euclidean distances from ``attr[row_idx]`` to every ``attr[col_idx]``.

    Row-chunked so no single dense block exceeds ~``max_block_elements``
    entries. Uses :func:`sklearn.metrics.pairwise.euclidean_distances` --
    the same function :func:`esda.boundary_silhouette` uses internally on
    its single dense (N, N) matrix -- so summed values match slicing that
    full matrix, bit-for-bit, with one exception: sklearn only force-zeroes
    the diagonal when its two arguments are the *same object* (``X is Y``),
    which never holds here even when ``row_idx``/``col_idx`` overlap (the
    own-cluster case). ``zero_self_pairs=True`` corrects for that explicitly
    so a same-cluster self-distance is exactly 0, not a tiny formula
    cancellation epsilon.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    n_rows, n_cols = len(row_idx), len(col_idx)
    row_chunk = max(1, max_block_elements // max(1, n_cols))
    cols = attr[col_idx]
    out = np.zeros(n_rows, dtype=np.float64)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        rows = attr[row_idx[r0:r1]]
        d = euclidean_distances(rows, cols)
        if zero_self_pairs:
            matches = row_idx[r0:r1][:, None] == col_idx[None, :]
            if matches.any():
                d[matches] = 0.0
        out[r0:r1] = d.sum(axis=1)
    return out


def _boundary_silhouette(
    attr: np.ndarray,
    labels: np.ndarray,
    conn: sp.spmatrix,
    *,
    max_block_mb: float = 256.0,
) -> np.ndarray:
    """Memory-chunked reimplementation of :func:`esda.boundary_silhouette`.

    Replicates esda's exact algorithm (see that function's docstring) --
    same boundary detection, same singleton-cluster and island handling,
    same divisors -- but never materializes a dense (N, N) distance matrix.
    Only the distance sums the formula actually needs (same-cluster mean
    dissimilarity, per-candidate-neighbor-cluster mean distance) are
    computed, via :func:`_chunked_distance_sum`, bounding peak memory to
    roughly ``max_block_mb`` regardless of ``N``.
    """
    n = len(labels)
    max_block_elements = max(1, int(max_block_mb * 1e6 / 8))

    coo = conn.tocoo()
    nz = coo.data != 0
    row_nz, col_nz = coo.row[nz], coo.col[nz]

    # Island check over *all* nonzero entries, including the diagonal --
    # tl.spatial_neighbors(set_diag=True) sets self-loops, which esda's own
    # to_adjlist(drop_islands=True) would treat as "has an edge" too.
    has_edge = np.zeros(n, dtype=bool)
    has_edge[row_nz] = True
    if not has_edge.all():
        raise ValueError(
            "spatial_silhouette requires every cell to have at least one graph "
            "neighbor (no islands). Run pp.filter_isolated(), or pass "
            "set_diag=True to tl.spatial_neighbors(), first."
        )

    # Self-loops are never boundary edges (same cell -> same label), so drop
    # them here -- but only after the island check above.
    real_edge = row_nz != col_nz
    focal_idx, neighbor_idx = row_nz[real_edge], col_nz[real_edge]

    labels_arr = np.asarray(labels)
    if pd.isna(labels_arr).any():
        raise ValueError("groupby column must not contain NaN/missing labels")
    unique_labels, encoded = np.unique(labels_arr, return_inverse=True)
    n_labels = len(unique_labels)
    cluster_members = [np.nonzero(encoded == k)[0] for k in range(n_labels)]
    cluster_size = np.array([len(m) for m in cluster_members])

    boundary_mask = encoded[focal_idx] != encoded[neighbor_idx]
    scores = np.zeros(n, dtype=np.float64)
    if not boundary_mask.any():
        return scores

    b_focal = focal_idx[boundary_mask]
    b_neighbor_label = encoded[neighbor_idx][boundary_mask]
    boundary_focals = np.unique(b_focal)

    mean_dissim = np.full(n, np.nan)
    neighbor_score = np.full(n, np.inf)
    valid = np.zeros(n, dtype=bool)  # own cluster has > 1 member

    own_label = encoded[boundary_focals]
    for k in range(n_labels):
        if cluster_size[k] <= 1:
            continue
        group = boundary_focals[own_label == k]
        if len(group) == 0:
            continue
        members = cluster_members[k]
        s = _chunked_distance_sum(
            attr, group, members,
            max_block_elements=max_block_elements, zero_self_pairs=True,
        )
        mean_dissim[group] = s / (len(members) - 1)
        valid[group] = True

    # Group boundary focal cells by candidate neighbor cluster label via a
    # sparse presence matrix -- memory bounded by the number of boundary
    # edges, not N x n_labels. Duplicate (focal, label) pairs (a focal cell
    # with several same-label graph edges) collapse for free via the
    # (data, (row, col)) constructor's summing-on-duplicates behavior.
    candidate = sp.csr_matrix(
        (np.ones(len(b_focal), dtype=np.int8), (b_focal, b_neighbor_label)),
        shape=(n, n_labels),
    ).tocsc()
    for m in range(n_labels):
        start, end = candidate.indptr[m], candidate.indptr[m + 1]
        focal_group = candidate.indices[start:end]
        focal_group = focal_group[valid[focal_group]]
        if len(focal_group) == 0:
            continue
        members = cluster_members[m]
        s = _chunked_distance_sum(
            attr, focal_group, members,
            max_block_elements=max_block_elements, zero_self_pairs=False,
        )
        neighbor_score[focal_group] = np.minimum(neighbor_score[focal_group], s / len(members))

    active = valid & np.isfinite(neighbor_score)
    a, b = mean_dissim[active], neighbor_score[active]
    scores[active] = (b - a) / np.maximum(a, b)
    return scores


def _extract_values(adata: ad.AnnData, key: str, layer: str | None) -> np.ndarray:
    """Return a 1-D float64 array for *key*, checking var_names then obs columns."""
    if key in adata.var_names:
        idx = adata.var_names.get_loc(key)
        mat = adata.layers[layer] if layer is not None else adata.X
        col = mat[:, idx]
        if hasattr(col, "get"):
            col = col.get()
        if sp.issparse(col):
            col = col.toarray().ravel()
        return np.asarray(col, dtype=np.float64).ravel()
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
    """Multiple-testing correction, tolerant of NaN entries.

    A single NaN p-value (routine: e.g. a zero-variance/undetected gene's
    ``esda.Moran``/``Geary`` return ``I``/``C`` = ``p_norm`` = NaN) used to
    silently NaN-poison ``statsmodels.multipletests`` across the *entire*
    array -- every other gene's adjusted p-value became NaN too, and since
    ``spatially_variable = pval_adj < 0.05`` a NaN compares False everywhere,
    so the whole call quietly reported zero significant genes with no error
    or warning. NaN entries are excluded from the correction and re-inserted
    as NaN afterward, so they neither pass nor corrupt the other genes' results.
    """
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(p, dtype=np.float64)
    valid = np.isfinite(p)
    p_adj = np.full_like(p, np.nan)
    if valid.any():
        _, p_adj[valid], _, _ = multipletests(p[valid], method=method)
    return p_adj


class _seeded_numpy_state:
    """Temporarily seed the legacy global ``np.random`` state.

    ``esda.Moran``/``Moran_BV``/``Geary`` permute via ``np.random.permutation``
    (the legacy global ``RandomState``) and accept no ``seed=`` argument of
    their own, unlike ``Moran_Local``/``Moran_Local_BV``/``G_Local`` which do.
    This is the only way to make their permutation p-values reproducible; the
    caller's own global RNG state is saved and restored around the call so a
    seeded run here doesn't perturb unrelated code using the global RNG.
    A ``None`` seed is a no-op (existing global-RNG behavior, unseeded).
    """

    def __init__(self, seed: int | None):
        self._seed = seed
        self._state = None

    def __enter__(self):
        if self._seed is not None:
            self._state = np.random.get_state()
            np.random.seed(self._seed)
        return self

    def __exit__(self, *exc):
        if self._state is not None:
            np.random.set_state(self._state)


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
    seed: int | None = None,
    key_added: str = "moran",
) -> pd.DataFrame:
    """Global Moran's I (Moran 1950) for one or more expression or obs keys.

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
    seed
        Seed for the permutation RNG (reproducible ``p_sim``). ``None``
        (default) uses numpy's unseeded global state, matching prior behavior.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per key with columns ``I``, ``EI``, ``z_norm``, ``p_norm``,
        and (when ``n_perms > 0``) ``z_sim``, ``p_sim``.

        .. note::
            The two p-value columns use different conventions, inherited
            from ``esda.Moran``: ``p_norm`` is a **two-tailed** analytic
            (normal-approximation) p-value, while ``p_sim`` is esda's
            **one-tailed, data-directed** permutation p-value (the tail is
            picked from the sign of the observed deviation, never doubled).
            The two are therefore not directly comparable at the same
            nominal alpha — ``p_sim`` is roughly half of a two-tailed
            equivalent for the same evidence.

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
        with _seeded_numpy_state(seed):
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
    alpha: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
    key_added: str = "moran_local",
) -> pd.DataFrame:
    """Local spatial association with empirical two-sided permutation p-values.

    BH (default) corrects one family across all requested cells and fields.
    BH assumes independent or positively dependent tests; use ``fdr_by`` for
    arbitrary dependence. Labels use adjusted p-values at ``alpha``; raw
    quadrants and directed smaller tails are exploratory diagnostics.
    Zero permutations and undefined statistics are explicitly not tested.
    """
    import esda
    if not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    if isinstance(keys, str):
        keys = [keys]

    scoped = _local_by_library(adata, moran_local, (keys,), dict(
        layer=layer, neighbors_key=neighbors_key, n_perms=n_perms, p_adjust_method=p_adjust_method,
        alpha=alpha, n_jobs=n_jobs, seed=seed, key_added=key_added))
    if scoped is not None:
        return scoped
    w = _build_weights(adata, neighbors_key)
    log.info("moran_local: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    frames = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        lmi = esda.Moran_Local(
            y, w, permutations=n_perms, n_jobs=n_jobs, seed=seed, keep_simulations=True
        )

        if n_perms > 0:
            p_sim, n_valid = permutation_p(lmi.Is, lmi.rlisas)
            p_dir = _directed_perm_p(lmi.Is, lmi.rlisas)
            z_sim = lmi.z_sim
            p_adj = _adj_pvalues(p_sim, p_adjust_method)
        else:
            n_valid = np.zeros(adata.n_obs, dtype=int)
            p_sim = np.full(adata.n_obs, np.nan)
            p_dir = np.full(adata.n_obs, np.nan)
            z_sim = np.full(adata.n_obs, np.nan)
            p_adj = np.full(adata.n_obs, np.nan)

        if (np.ptp(y) == 0):
            p_sim[:] = np.nan
            p_dir[:] = np.nan
            z_sim[:] = np.nan

        quadrant = pd.array([_QUAD_LABELS.get(int(q), "NS") for q in lmi.q], dtype="string")

        frames.append(
            pd.DataFrame(
                {"Ii": lmi.Is, "z_sim": z_sim, "p_sim": p_sim, "p_dir": p_dir, "p_adj": p_adj, "quadrant": quadrant,
                 "n_valid": n_valid},
                index=adata.obs_names,
            )
        )

    result = pd.concat(frames, keys=keys, names=["key", "obs_name"])
    result = _local_family(result, p_adjust_method, alpha)
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
    seed: int | None = None,
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
    seed
        Seed for the permutation RNG (``esda.Moran_BV`` accepts no ``seed=``
        of its own; the global numpy RNG is seeded around the call and
        restored after). ``None`` (default) is unseeded, matching prior
        behavior.
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
        with _seeded_numpy_state(seed):
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
    alpha: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
    key_added: str = "moran_local_bv",
) -> pd.DataFrame:
    """Local spatial association with empirical two-sided permutation p-values.

    BH (default) corrects one family across all requested cells and fields.
    BH assumes independent or positively dependent tests; use ``fdr_by`` for
    arbitrary dependence. Labels use adjusted p-values at ``alpha``; raw
    quadrants and directed smaller tails are exploratory diagnostics.
    Zero permutations and undefined statistics are explicitly not tested.
    """
    import esda
    if not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    x_keys, y_keys = _normalize_pairs(x, y)

    scoped = _local_by_library(adata, moran_local_bv, (x_keys, y_keys), dict(
        layer=layer, neighbors_key=neighbors_key, n_perms=n_perms, p_adjust_method=p_adjust_method,
        alpha=alpha, n_jobs=n_jobs, seed=seed, key_added=key_added))
    if scoped is not None:
        return scoped
    w = _build_weights(adata, neighbors_key)
    log.info("moran_local_bv: %d pairs, %d cells, n_perms=%d", len(x_keys), adata.n_obs, n_perms)

    frames = []
    pair_labels = []
    for xk, yk in zip(x_keys, y_keys):
        xv = _extract_values(adata, xk, layer)
        yv = _extract_values(adata, yk, layer)
        lmi = esda.Moran_Local_BV(
            xv, yv, w, permutations=n_perms, n_jobs=n_jobs, seed=seed, keep_simulations=True
        )

        if n_perms > 0:
            p_sim, n_valid = permutation_p(lmi.Is, lmi.rlisas)
            p_dir = _directed_perm_p(lmi.Is, lmi.rlisas)
            z_sim = np.asarray(getattr(lmi, "z_sim", np.full(adata.n_obs, np.nan)))
            p_adj = _adj_pvalues(p_sim, p_adjust_method)
        else:
            n_valid = np.zeros(adata.n_obs, dtype=int)
            p_sim = np.full(adata.n_obs, np.nan)
            p_dir = np.full(adata.n_obs, np.nan)
            z_sim = np.full(adata.n_obs, np.nan)
            p_adj = np.full(adata.n_obs, np.nan)

        if (np.ptp(xv) == 0 or np.ptp(yv) == 0):
            p_sim[:] = np.nan
            p_dir[:] = np.nan
            z_sim[:] = np.nan

        quadrant = pd.array([_QUAD_LABELS.get(int(q), "NS") for q in lmi.q], dtype="string")

        frames.append(
            pd.DataFrame(
                {"Ii": lmi.Is, "z_sim": z_sim, "p_sim": p_sim, "p_dir": p_dir, "p_adj": p_adj, "quadrant": quadrant,
                 "n_valid": n_valid},
                index=adata.obs_names,
            )
        )
        pair_labels.append(f"{xk}:{yk}")

    result = pd.concat(frames, keys=pair_labels, names=["key", "obs_name"])
    result = _local_family(result, p_adjust_method, alpha)
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
    seed: int | None = None,
    key_added: str = "getis_ord",
) -> pd.DataFrame:
    """Local spatial association with empirical two-sided permutation p-values.

    BH (default) corrects one family across all requested cells and fields.
    BH assumes independent or positively dependent tests; use ``fdr_by`` for
    arbitrary dependence. Labels use adjusted p-values at ``alpha``; raw
    quadrants and directed smaller tails are exploratory diagnostics.
    Zero permutations and undefined statistics are explicitly not tested.
    """
    import esda
    if not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")

    if isinstance(keys, str):
        keys = [keys]

    scoped = _local_by_library(adata, getis_ord, (keys,), dict(
        layer=layer, neighbors_key=neighbors_key, n_perms=n_perms, p_adjust_method=p_adjust_method,
        alpha=alpha, n_jobs=n_jobs, seed=seed, key_added=key_added), getis=True)
    if scoped is not None:
        return scoped
    w = _build_weights(adata, neighbors_key, row_standardize=False)
    conn = _get_conn_matrix(adata, neighbors_key)
    nz = conn.data[conn.data != 0]
    if nz.size and not np.allclose(nz, 1.0):
        log.warning(
            "getis_ord: obsp['%s'-derived] graph weights are not binary (0/1); "
            "the Gi* self-weight star=1 assumes a binary graph, so results on "
            "a weighted graph (e.g. 'joint_neighbors') over-weight each cell's "
            "own value. Use an unweighted spatial graph from tl.spatial_neighbors().",
            neighbors_key,
        )
    log.info("getis_ord: %d keys, %d cells, n_perms=%d", len(keys), adata.n_obs, n_perms)

    from scipy import stats as sp_stats

    frames = []
    for key in keys:
        y = _extract_values(adata, key, layer)
        lG = esda.G_Local(
            y, w, permutations=n_perms, transform="r", star=1,
            n_jobs=n_jobs, seed=seed, keep_simulations=True,
        )

        z_norm = np.asarray(lG.Zs, dtype=np.float64)
        p_norm = np.minimum(2.0 * sp_stats.norm.sf(np.abs(z_norm)), 1.0)

        if n_perms > 0:
            z_sim = np.asarray(lG.z_sim)
            p_sim, n_valid = permutation_p(np.asarray(lG.Gs), np.asarray(lG.sim).T)
            p_dir = _directed_perm_p(np.asarray(lG.Gs), np.asarray(lG.sim).T)
        else:
            n_valid = np.zeros(adata.n_obs, dtype=int)
            z_sim = np.full(adata.n_obs, np.nan)
            p_sim = np.full(adata.n_obs, np.nan)
            p_dir = np.full(adata.n_obs, np.nan)


        if (np.ptp(y) == 0):
            p_sim[:] = np.nan
            p_dir[:] = np.nan
            z_sim[:] = np.nan
        z_vals = z_sim if n_perms > 0 else z_norm
        p_adj = _adj_pvalues(
            p_sim, p_adjust_method
        )

        p_gate = p_dir if n_perms > 0 else p_norm
        spot_type = pd.array(
            [
                "hot" if z > 0 and p < alpha else "cold" if z < 0 and p < alpha else "NS"
                for z, p in zip(z_vals, p_gate)
            ],
            dtype="string",
        )

        frames.append(
            pd.DataFrame(
                {
                    "Gs": np.asarray(lG.Gs),
                    "n_valid": n_valid,
                    "z_norm": z_norm,
                    "p_norm": p_norm,
                    "z_sim": z_sim,
                    "p_sim": p_sim,
                    "p_dir": p_dir,
                    "p_adj": p_adj,
                    "type": spot_type,
                },
                index=adata.obs_names,
            )
        )

    result = pd.concat(frames, keys=keys, names=["key", "obs_name"])
    result = _local_family(result, p_adjust_method, alpha, getis=True)
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
    average; categorical variables use the most-common neighbor label
    (ties broken by libpysal's default ``'tryself'`` rule: the cell's own
    label when it is among the tied modes, otherwise an arbitrary tied
    label). Cells with no graph neighbors at all (islands) get ``NaN``
    rather than a fabricated ``0.0`` / arbitrary label.

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
    # Islands (zero-degree cells): a row-standardized all-zero weight row
    # makes lag_spatial return 0.0 and lag_categorical an arbitrary label --
    # both fabricated values, not "no information". NaN them explicitly.
    islands = np.asarray(_get_conn_matrix(adata, neighbors_key).sum(axis=1)).ravel() == 0

    def _numeric_lag(y: np.ndarray) -> np.ndarray:
        lagged = np.asarray(lag_spatial(w, y), dtype=np.float64)
        lagged[islands] = np.nan
        return lagged

    for key in keys:
        col_name = f"{key}{suffix}"
        if key in adata.var_names:
            y = _extract_values(adata, key, layer)
            adata.obs[col_name] = _numeric_lag(y)
        elif key in adata.obs.columns:
            dtype = adata.obs[key].dtype
            is_cat = isinstance(dtype, pd.CategoricalDtype) or dtype.kind in ("O", "U", "S")
            if is_cat:
                labels = adata.obs[key].astype(str).to_numpy()
                lagged = pd.Series(
                    lag_categorical(w, labels), index=adata.obs_names, dtype="category"
                )
                if islands.any():
                    lagged[islands] = np.nan
                adata.obs[col_name] = lagged
            else:
                y = adata.obs[key].to_numpy(dtype=np.float64)
                adata.obs[col_name] = _numeric_lag(y)
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
    max_block_mb: float = 256.0,
) -> None:
    """Boundary silhouette score for spatially-defined clusters (Wolf, Knaap & Rey 2019).

    Reimplements :func:`esda.boundary_silhouette` (PySAL)'s exact algorithm
    rather than delegating to it at runtime. An earlier graph-restricted
    version here computed a different statistic than the one it claimed to
    speed up -- it averaged distances only over cells sharing a *graph edge*
    with the focal cell, whereas esda's own algorithm uses the spatial graph
    only to decide which clusters are *candidate* neighbors of a boundary
    cell, and then averages distances over *every* cell in each relevant
    cluster, graph-connected or not. Those are genuinely different numbers,
    not just a faster/slower path to the same one -- this function reproduces
    esda's own algorithm, branch for branch (see :func:`_boundary_silhouette`),
    verified numerically consistent with it in ``tests/`` and
    ``benchmarks/esda/spatial_silhouette.ipynb``, rather than delegating to it
    directly: esda's implementation materializes a dense ``(n_cells, n_cells)``
    distance matrix regardless of how many cells are actually on a boundary,
    which runs out of memory well before real dataset scale (e.g. ~196 GB at
    162k cells). This computes only the distance sums the formula needs, in
    row-chunked blocks bounded by ``max_block_mb``, so peak memory no longer
    scales with the square of the cell count.

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
    max_block_mb
        Upper bound, in MB, on the size of any single dense distance block
        materialized during computation. Lower this if memory is still
        tight; raise it for speed on a machine with more RAM.

    Examples
    --------
    >>> ab.tl.spatial_silhouette(mdata, groupby="leiden")
    >>> ab.pl.points(mdata, color="spatial_silhouette", cmap="astrolabe_div", vcenter=0.0)
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"obs column '{groupby}' not found")

    conn = _get_conn_matrix(adata, neighbors_key)
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

    scores = _boundary_silhouette(attr, labels, conn, max_block_mb=max_block_mb)
    adata.obs[key_added] = scores
    log.info("spatial_silhouette: done")


def _ripley_l_kdtree(
    coords: np.ndarray,
    radii: np.ndarray,
    n_perms: int,
    rng: "np.random.Generator",
    n_jobs: int = 1,
    *,
    window_min: np.ndarray | None = None,
    window_max: np.ndarray | None = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """KDTree-based Ripley's L — O(n log n × m) time, O(n log n) memory.

    Uses ``scipy.spatial.cKDTree.count_neighbors`` to count pairs within each
    radius without materialising an O(n²) distance matrix.

    ``window_min``/``window_max``, when given, define the study region (used
    for both the K(r) area normalization and the CSR null's uniform sampling
    bounds) as the tissue-wide extent rather than this call's own ``coords``
    bounding box. A cell type confined to a small region of the tissue would
    otherwise get its own small, artificial null window -- CSR *within that
    small region* is a different (and not directly comparable, across
    groups) null hypothesis than CSR within the actual observed tissue.
    Defaults to this call's own ``coords`` extent when not given.
    """
    from scipy.spatial import cKDTree

    n = len(coords)
    xy_min = window_min if window_min is not None else coords.min(axis=0)
    xy_max = window_max if window_max is not None else coords.max(axis=0)
    area = float((xy_max[0] - xy_min[0]) * (xy_max[1] - xy_min[1]))
    if area <= 0:
        # The old max(area, 1.0) floor silently substituted a unit-dependent
        # fake area (1 µm²) for degenerate windows (all points collinear /
        # coincident), producing a meaningless K(r). Fail loudly instead.
        raise ValueError(
            "ripley_l: study window has zero area (all coordinates collinear "
            "or identical along one axis); Ripley's K/L is undefined."
        )

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
    """Ripley's L function (Ripley 1976; Besag 1977) for cell-type point patterns.

    Tests whether cells of a given type are spatially clustered beyond what
    is expected under complete spatial randomness (CSR).

    The returned ``L_obs``/``L_exp``/``L_hi``/``L_lo`` columns all hold the
    *detrended* quantity L(d) − d, not raw L(d): raw L(d) equals *d* under
    ideal CSR (a diagonal reference line), while detrended L(d) − d is flat.
    Positive values indicate clustering at distance *d* (more same-type
    neighbors within *d* than random chance predicts); negative values
    indicate regularity/dispersion (fewer). Read ``L_obs`` against
    ``L_exp``, **not against 0**: ``L_exp`` is the *mean of the simulated
    CSR curves* (NaN when ``n_perms=0``), not a constant 0 — this estimator
    applies **no edge correction**, so boundary cells are systematically
    undercounted at large *d* and the true CSR expectation of L(d) − d bows
    below zero as *d* grows. The simulated mean carries the same
    finite-sample/no-edge-correction bias as ``L_obs``, making it the
    correct apples-to-apples reference; comparing ``L_obs`` to 0 at large
    *d* can invert the clustered-vs-dispersed conclusion.

    The CSR null is uniform sampling in the tissue's **axis-aligned bounding
    rectangle** (shared by every group). For markedly non-rectangular
    tissue (an L-shaped or annular section), parts of that rectangle contain
    no tissue at all, which biases the null toward apparent clustering of
    the observed points — interpret marginal results cautiously there.

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
        ``p_value`` is a **per-radius, two-sided** Monte Carlo rank p-value
        against the CSR simulations at that same radius,
        ``2·(min(#sims ≥ L_obs, #sims ≤ L_obs) + 1) / (n_perms + 1)`` —
        significant for clustering *and* for dispersion, evaluated
        independently at every distance. These pointwise p-values are not
        corrected for testing across the ``support`` radii; treat "any
        radius below alpha" screening accordingly, or read the envelope
        (``L_hi``/``L_lo``) as a whole (see Baddeley et al. 2014 on the
        multiple-testing pitfall of pointwise envelopes).
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> curves = ab.tl.ripley_l(mdata, groupby="leiden")
    >>> curves["0"].plot(y=["L_obs", "L_hi", "L_lo"])   # cluster 0
    """
    from astrolabe._utils import _check_single_library

    _check_single_library(adata)

    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")

    # Only x/y: a 3-D obsm (e.g. z-stacked coordinates) would silently break
    # the planar area normalization and the 2-D CSR null.
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]
    rng = np.random.default_rng(seed)

    # Tissue-wide window, shared by every group -- both the radii (below) and
    # each group's K(r) area normalization / CSR null window use this same
    # extent, not any individual group's own (potentially much smaller and
    # incomparable) occupied bounding box.
    window_min = coords.min(axis=0)
    window_max = coords.max(axis=0)

    # Radii computed from full-dataset extent so all groups share the same x-axis.
    # max_r = min(x_range, y_range) / 4 follows Ripley's rule of thumb.
    xy_range = window_max - window_min
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

        l_obs, l_sims = _ripley_l_kdtree(
            group_coords, radii, n_perms, rng, n_jobs=n_jobs,
            window_min=window_min, window_max=window_max,
        )

        if n_perms > 0:
            # L_exp is the *simulated* mean CSR curve, not a hardcoded 0.
            # This estimator has no edge correction (boundary cells are
            # systematically undercounted at large r), so the true expected
            # L(d) - d under CSR is not flat at 0 -- it bows below zero as d
            # grows, and reading L_obs against a hardcoded L_exp=0 can invert
            # the clustered-vs-dispersed conclusion at large d. The simulated
            # mean already carries the same finite-sample/no-edge-correction
            # bias as L_obs, so it's the correct apples-to-apples reference.
            l_exp = l_sims.mean(axis=0)
            l_hi = np.percentile(l_sims, 97.5, axis=0)
            l_lo = np.percentile(l_sims, 2.5, axis=0)
            # Per-radius two-sided Monte Carlo rank p -- one p-value at every
            # distance, sensitive to dispersion as well as clustering. (An
            # earlier version tested only the largest radius, one-sided for
            # clustering, and broadcast that single scalar into every row of
            # the distance-indexed column.) The +1 floor keeps an exact 0
            # from breaking downstream -log10(p)/FDR steps. Pointwise, not
            # corrected across radii -- see the docstring.
            p_value, n_valid = permutation_p(l_obs, l_sims.T)
        else:
            l_exp = np.full_like(l_obs, np.nan)
            l_hi = np.full_like(l_obs, np.nan)
            l_lo = np.full_like(l_obs, np.nan)
            p_value = np.full_like(l_obs, np.nan)
            n_valid = np.zeros_like(l_obs, dtype=int)

        df = pd.DataFrame(
            {"L_obs": l_obs, "L_exp": l_exp, "L_hi": l_hi, "L_lo": l_lo, "p_value": p_value,
             "n_valid": n_valid},
            index=pd.Index(radii, name="distance_um"),
        )

        annotate_permutation_resolution(df, "p_value", "n_valid", method=None,
                                       label=f"pointwise Ripley {name}")
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
    seed: int | None = None,
) -> pd.DataFrame:
    import esda
    from scipy import stats as sp_stats

    get_weights = _thread_local_weights_getter(adata, neighbors_key)
    blocks, library_key = _permutation_blocks(adata, neighbors_key)
    if len(blocks) > 1 and n_perms == 0:
        raise ValueError("multiple libraries require n_perms > 0 for SVG inference; the null shuffles within each library")

    rngs = _spawn_rngs(seed, len(genes))

    def _one_gene(g: str, rng) -> dict:
        y = _extract_values(adata, g, layer)
        mi = esda.Moran(y, get_weights(), permutations=0, two_tailed=False)
        n_valid = 0
        # One-sided upper-tail p ("is this gene POSITIVELY spatially
        # autocorrelated?") on both code paths -- matching squidpy's
        # convention for this scan. Neither of esda's own p-values gives
        # that: p_norm (even with two_tailed=False) and p_sim are both
        # sign-matched/folded, so a strongly *negatively* autocorrelated
        # gene (e.g. a checkerboard artifact) used to pass as an "SVG",
        # and the convention silently flipped between n_perms=0 and >0.
        if not np.isfinite(mi.I):
            p = np.nan
        elif n_perms > 0:
            centered = y-y.mean()
            denominator = centered@centered
            weights = get_weights().sparse
            sims = np.empty(n_perms)
            for permutation in range(n_perms):
                shuffled = _shuffle_blocks(centered, rng, blocks)
                sims[permutation] = len(y)/weights.sum() * (shuffled@(weights@shuffled))/denominator
            p, n_valid = permutation_p(np.asarray(mi.I), sims, alternative="greater")
            p = float(p)
        else:
            p = float(sp_stats.norm.sf(mi.z_norm))
        return {"gene": g, "score": float(mi.I), "pval": p, "n_valid": int(n_valid)}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one_gene)(g, rng) for g, rng in zip(genes, rngs))

    df = pd.DataFrame(rows).set_index("gene")
    df["pval_adj"] = _adj_pvalues(df["pval"].to_numpy(), p_adjust_method)
    annotate_permutation_resolution(df, "pval", "n_valid", alternative="greater",
                                   method=p_adjust_method, label="Moran SVG")
    df.attrs.update(library_key=library_key, null="within-library expression permutations" if n_perms else "analytic normality",
                    n_perms=int(n_perms))
    return df


def _svg_geary(
    adata: ad.AnnData,
    genes: list[str],
    layer: str | None,
    neighbors_key: str,
    n_perms: int,
    p_adjust_method: str,
    n_jobs: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    import esda
    from scipy import stats as sp_stats

    get_weights = _thread_local_weights_getter(adata, neighbors_key)
    blocks, library_key = _permutation_blocks(adata, neighbors_key)
    if len(blocks) > 1 and n_perms == 0:
        raise ValueError("multiple libraries require n_perms > 0 for SVG inference; the null shuffles within each library")

    rngs = _spawn_rngs(seed, len(genes))

    def _one_gene(g: str, rng) -> dict:
        y = _extract_values(adata, g, layer)
        gc = esda.Geary(y, get_weights(), permutations=0)
        n_valid = 0
        # Geary's C: C < 1 = positive autocorrelation, C > 1 = negative.
        # One-sided LOWER-tail p ("is C significantly below 1?"), the Geary
        # mirror of _svg_moran's one-sided upper-tail convention -- z_norm is
        # negative for positive autocorrelation, so cdf(z_norm) is the
        # matching analytic tail, and (sims <= C) the matching permutation
        # tail. No doubling: a doubled/folded p would again let negatively
        # autocorrelated genes pass, and disagree with the 'moran' flavor.
        if not np.isfinite(gc.C):
            p = np.nan
        elif n_perms > 0:
            weights = get_weights().sparse.tocoo()
            denominator = ((y-y.mean())**2).sum()
            sims = np.empty(n_perms)
            for permutation in range(n_perms):
                shuffled = _shuffle_blocks(y, rng, blocks)
                sims[permutation] = (len(y)-1)/(2*weights.data.sum()*denominator) * np.sum(weights.data*(shuffled[weights.row]-shuffled[weights.col])**2)
            p, n_valid = permutation_p(np.asarray(gc.C), sims, alternative="less")
            p = float(p)
        else:
            p = float(sp_stats.norm.cdf(gc.z_norm))
        return {"gene": g, "score": float(gc.C), "pval": p, "n_valid": int(n_valid)}

    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one_gene)(g, rng) for g, rng in zip(genes, rngs))

    df = pd.DataFrame(rows).set_index("gene")
    df["pval_adj"] = _adj_pvalues(df["pval"].to_numpy(), p_adjust_method)
    annotate_permutation_resolution(df, "pval", "n_valid", alternative="less",
                                   method=p_adjust_method, label="Geary SVG")
    df.attrs.update(library_key=library_key, null="within-library expression permutations" if n_perms else "analytic normality",
                    n_perms=int(n_perms))
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
    seed: int | None = None,
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
        GPU-resident inputs use the same CPU statistic. Applies regardless of ``n_perms``,
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
        ``pval`` is **one-sided** for positive spatial autocorrelation
        (upper tail of I for ``'moran'``, lower tail of C — i.e. C < 1 —
        for ``'geary'``), on every code path (analytic or permutation, CPU
        or GPU): "spatially variable" here means expression *clusters* in
        space, so a negatively autocorrelated pattern (e.g. a checkerboard
        artifact) is not significant. This matches
        ``squidpy.gr.spatial_autocorr``'s convention.
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
    if not isinstance(n_perms, (int, np.integer)) or n_perms < 0:
        raise ValueError("n_perms must be a nonnegative integer")

    genes = list(adata.var_names)
    log.info(
        "spatially_variable_genes: flavor='%s', %d genes, %d cells",
        flavor,
        len(genes),
        adata.n_obs,
    )

    if flavor == "moran":
        df = _svg_moran(adata, genes, layer, neighbors_key, n_perms, p_adjust_method, n_jobs=n_jobs, seed=seed)
    else:
        df = _svg_geary(adata, genes, layer, neighbors_key, n_perms, p_adjust_method, n_jobs=n_jobs, seed=seed)

    df["flavor"] = flavor

    if n_top_genes is not None:
        # For Geary's C, lower score = more spatial; for all others, higher is more spatial.
        ascending = flavor == "geary"
        top_idx = df["score"].sort_values(ascending=ascending).head(n_top_genes).index
        df["spatially_variable"] = df.index.isin(top_idx)
    else:
        df["spatially_variable"] = df["pval_adj"] < 0.05

    # Write per-gene columns back to var
    for col in ("score", "pval", "pval_adj", "spatially_variable", "flavor",
                "n_valid", "p_min", "resolution_limited"):
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
    seed: int | None = None,
    key_added: str = "geary_c",
) -> pd.DataFrame:
    """Global Geary's C (Geary 1954) spatial autocorrelation for one or more keys.

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
    seed
        Seed for the permutation RNG (``esda.Geary`` accepts no ``seed=`` of
        its own; the global numpy RNG is seeded around the call and restored
        after). ``None`` (default) is unseeded, matching prior behavior.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per key with columns ``C``, ``EC``, ``z_norm``, ``p_norm``,
        and (when ``n_perms > 0``) ``z_sim``, ``p_sim``.

        .. note::
            As in :func:`moran`, the two p-value columns use different
            conventions: ``p_norm`` is **two-tailed** (esda's one-tailed
            analytic value, doubled here to align with ``esda.Moran``'s
            ``two_tailed=True``), while ``p_sim`` is esda's **one-tailed,
            data-directed** permutation p-value (tail picked from the sign
            of the observed deviation, never doubled) — roughly half the
            two-tailed equivalent for the same evidence.

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
        with _seeded_numpy_state(seed):
            gc = esda.Geary(y, w, permutations=n_perms)
        row: dict[str, float] = {
            "C": gc.C,
            "EC": gc.EC,
            "z_norm": gc.z_norm,
            # esda.Geary.p_norm is one-tailed by esda's own design (only the
            # sign-matching tail, never doubled) -- esda.Moran.p_norm is
            # two-tailed (moran() above passes two_tailed=True). Left as-is,
            # the same nominal alpha would be ~2x more lenient for geary_c
            # than for moran. Doubled here (capped at 1.0) to match moran's
            # two-tailed p_norm. (spatially_variable_genes uses its own
            # one-sided convention on both flavors -- see _svg_geary.)
            "p_norm": min(2.0 * gc.p_norm, 1.0),
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
    groupby: str,
    *,
    spatial_key: str = "spatial",
    k: int = 1,
    categories: list[str] | None = None,
    min_count: int = 5,
    n_perms: int = 0,
    n_jobs: int = 1,
    seed: int | None = None,
    key_added: str = "clq",
) -> pd.DataFrame:
    """Nearest-neighbor colocation quotient (Leslie & Kronenfeld 2011).

    ``CLQ_{A→B}`` measures whether category *B* occurs among category *A*'s
    ``k`` nearest spatial neighbors more or less often than expected under
    random label assignment over the same point locations:

    ``CLQ_{A→B} = (C_{A→B} / (N_A × k)) / (N_B / (N - 1))``   (A ≠ B)

    ``CLQ_{A→A} = (C_{A→A} / (N_A × k)) / ((N_A - 1) / (N - 1))``

    where ``C_{A→B}`` is the total count of type-*B* points among the ``k``
    nearest neighbors of every type-*A* point, ``N_A``/``N_B`` are category
    counts, and ``N`` is the total point count. The diagonal (``A == B``)
    expectation is ``(N_A − 1)/(N − 1)``, not ``N_A/(N − 1)`` — a point is
    never its own neighbor, so only the *other* ``N_A − 1`` same-type points
    are available among the ``N − 1`` candidate neighbors (Leslie &
    Kronenfeld 2011). ``CLQ > 1``: *B* is over-represented near *A*
    (attraction); ``CLQ < 1``: under-represented (avoidance); ``CLQ ≈ 1``:
    no spatial association.

    This is a genuinely spatial, **asymmetric** statistic (``CLQ_{A→B} !=
    CLQ_{B→A}`` in general, and shuffling point coordinates changes the
    result) — unlike this function's previous implementation, which computed
    a symmetric, coordinate-*invariant* same-location co-occurrence lift
    between gene pairs (whether two genes were ever detected in the same
    cell/bin, independent of any cell's position). That statistic is not the
    published CLQ and has been replaced rather than kept alongside it.
    **This is a breaking change**: the function now takes a categorical
    ``groupby`` column (e.g. cell type, Leiden cluster) instead of gene
    pairs, and no longer has a ``mode='pixel'``/``transcripts=`` path — a
    segmentation-free, gene-level analog of nearest-neighbor CLQ has no
    natural single "location" for a transcript the way a same-location
    lift did, so it isn't offered here.

    Requires ``obsm[spatial_key]`` (no separate ``spatial_neighbors`` graph
    needed — the k-NN graph is derived internally from ``k``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    groupby
        ``obs`` column of categorical point labels (e.g. cell type, cluster).
    modality
        For :class:`~mudata.MuData`: modality key.
    spatial_key
        ``obsm`` key for cell centroids (µm).
    k
        Number of nearest spatial neighbors per point. ``1`` (default)
        matches Leslie & Kronenfeld's original single-nearest-neighbor CLQ;
        ``k > 1`` uses their mean-fraction generalization.
    categories
        Restrict the *output* to CLQ values involving these categories.
        ``None`` (default) includes every category with at least
        *min_count* points. Every category's points still participate in
        the underlying k-NN graph regardless of this filter — this only
        narrows which rows are returned.
    min_count
        Skip categories with fewer than this many points when *categories*
        is ``None``.
    n_perms
        Label-permutation null (coordinates and the k-NN graph fixed,
        category labels reshuffled) for an empirical two-sided p-value:
        the rank-based ``(2·min(#perms ≥ obs, #perms ≤ obs) + 1) /
        (n_perms + 1)``, computed against the permutation distribution
        itself rather than against deviation from a hardcoded 1.0 (the
        null mean of CLQ is not exactly 1 in finite samples — especially
        on the diagonal — and the distribution is right-skewed).
        ``0`` (default) skips the null and omits ``p_value``.
    n_jobs
        Parallel workers for the permutation loop (joblib, thread-backed).
        ``1`` (default) runs sequentially. ``-1`` uses all available cores.
    seed
        Seed for the permutation RNG. ``None`` is unseeded.
    key_added
        ``uns`` key where the result DataFrame is stored.

    Returns
    -------
    pd.DataFrame
        One row per ordered ``(category_a, category_b)`` pair (including
        ``a == b``, meaningful as same-type spatial clustering) with columns
        ``category_a``, ``category_b``, ``clq``, ``n_a``, ``n_b``, ``k``,
        and (when ``n_perms > 0``) ``p_value``.
        Also stored in ``adata.uns[key_added]``, and mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.

    Examples
    --------
    >>> clq = ab.tl.colocation_quotient(mdata, groupby="cell_type", n_perms=200)
    >>> clq.query("category_a != category_b").sort_values("clq", ascending=False).head(10)
    """
    from astrolabe._utils import _check_single_library

    _check_single_library(adata)

    if groupby not in adata.obs.columns:
        raise KeyError(f"obs['{groupby}'] not found")
    if spatial_key not in adata.obsm:
        raise KeyError(f"obsm['{spatial_key}'] not found")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    labels_raw = adata.obs[groupby]
    valid = labels_raw.notna().to_numpy()
    if not valid.all():
        log.warning(
            "colocation_quotient: dropping %d cells with missing '%s'.",
            int((~valid).sum()), groupby,
        )
    labels = labels_raw[valid].astype("category")
    all_categories = labels.cat.categories.tolist()
    codes = labels.cat.codes.to_numpy()
    # Only x/y: a 3-D obsm would silently change the k-NN graph geometry.
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[valid][:, :2]
    n = len(codes)
    n_types = len(all_categories)
    cat_idx = {c: i for i, c in enumerate(all_categories)}
    counts = np.bincount(codes, minlength=n_types).astype(np.float64)

    if n <= k:
        raise ValueError(f"k={k} nearest neighbors requested but only {n} valid points.")

    if categories is not None:
        unknown = sorted(set(categories) - set(all_categories))
        if unknown:
            raise KeyError(f"Categories not found in obs['{groupby}']: {unknown}")
        cat_list = categories
    else:
        cat_list = [c for c in all_categories if counts[cat_idx[c]] >= min_count]

    if not cat_list:
        empty_df = pd.DataFrame(columns=["category_a", "category_b", "clq", "n_a", "n_b", "k"])
        adata.uns[key_added] = empty_df
        return empty_df

    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # k+1 is always >= 2, so cKDTree.query always returns a 2-D (n, k+1)
    # array here (it only collapses to 1-D when the scalar k=1 is passed
    # directly). Column 0 is each point itself (distance 0); drop it.
    _, nn_idx = tree.query(coords, k=k + 1)
    nn_idx = nn_idx[:, 1:]

    def _clq_matrix(codes_arr: np.ndarray) -> np.ndarray:
        """Full (n_types x n_types) CLQ matrix for one label assignment.

        `nn_idx` (the coordinate-derived k-NN graph) is fixed; only the
        label-to-point assignment `codes_arr` varies, e.g. across
        permutations. Category counts are identical for any permutation of
        the same labels, so `counts` (computed once, above) is reused.
        """
        H = np.zeros((n, n_types), dtype=np.float64)
        H[np.arange(n), codes_arr] = 1.0
        neighbor_type_counts = H[nn_idx].sum(axis=1)  # (n, n_types)
        C = H.T @ neighbor_type_counts  # (n_types, n_types): C[A, B]
        with np.errstate(divide="ignore", invalid="ignore"):
            c_ab = C / (counts[:, None] * k)
            # Expected fraction of type-B points among a point's N-1
            # candidate neighbors. On the diagonal (A == B) the point itself
            # is never its own neighbor, so only counts[A] - 1 same-type
            # points are available: Leslie & Kronenfeld (2011) use
            # (N_A - 1)/(N - 1) there, not N_A/(N - 1) -- the latter biases
            # diagonal CLQ low by (N_A - 1)/N_A (20% at the min_count=5
            # floor).
            expected = np.broadcast_to(counts[None, :], (n_types, n_types)).copy()
            np.fill_diagonal(expected, counts - 1.0)
            expected /= n - 1
            clq = c_ab / expected
        return clq

    clq_obs = _clq_matrix(codes)

    perm_stack = None
    if n_perms > 0:
        rngs = _spawn_rngs(seed if seed is not None else np.random.default_rng().integers(0, 2**63 - 1), n_perms)

        def _one_perm(rng: "np.random.Generator") -> np.ndarray:
            return _clq_matrix(codes[rng.permutation(n)])

        perm_mats = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_perm)(r) for r in rngs
        )
        perm_stack = np.stack(perm_mats)  # (n_perms, n_types, n_types)

    rows_out = []
    for a in cat_list:
        for b in cat_list:
            ia, ib = cat_idx[a], cat_idx[b]
            row = {
                "category_a": a,
                "category_b": b,
                "clq": float(clq_obs[ia, ib]),
                "n_a": int(counts[ia]),
                "n_b": int(counts[ib]),
                "k": k,
            }
            if perm_stack is not None:
                # Rank-based two-sided p against the permutation distribution
                # itself: 2*(min(#>=, #<=)+1)/(n_valid+1), capped at 1. (An
                # earlier version measured |CLQ - 1| deviation from a
                # hardcoded 1.0 -- but the finite-sample null mean of CLQ is
                # not 1, especially on the diagonal, and the null is
                # right-skewed, so that deviation test was miscalibrated.)
                perm_vals = perm_stack[:, ia, ib]
                obs_val = clq_obs[ia, ib]
                p, n_valid = permutation_p(np.asarray(obs_val), perm_vals)
                row["p_value"] = float(p)
                row["n_valid"] = int(n_valid)
            rows_out.append(row)

    result = pd.DataFrame(rows_out)
    if perm_stack is not None:
        annotate_permutation_resolution(result, "p_value", "n_valid", method=None,
                                       label="uncorrected CLQ")
    adata.uns[key_added] = result
    log.info(
        "colocation_quotient: %d category pairs, %d points, k=%d, n_perms=%d",
        len(result), n, k, n_perms,
    )
    return result
