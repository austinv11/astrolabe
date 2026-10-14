"""Gene expression gradient and vector field analysis.

This module implements:

- **expression_gradient** — per-cell gradient vectors via graph-based Moving
  Least Squares (MLS).  Fits a local linear plane (free intercept) to each
  cell's spatial neighborhood, giving a (gx, gy) gradient vector in
  feature-units per µm.  Fan & Gijbels 1996 (Local Polynomial Modelling);
  Lancaster & Salkauskas 1981 (Math. Comput.); Mirzaei et al. 2012
  (IMA J. Numer. Anal. — "diffuse derivatives").

- **gradient_field** — interpolates per-cell vectors onto a masked regular
  grid and computes divergence/curl via finite differences.  The masked
  streamline field represents a spatial ODE dx/dt = ∇f(x) over tissue
  microns — not RNA velocity (which is temporal/spliced-unspliced).
  Qiu et al. 2022 (Cell, Dynamo) provide the template.

- **gradient_significance** — permutation test on mean gradient magnitude and
  directional coherence (mean resultant length R̄).  Prefer over the
  parametric Rayleigh test because adjacent-cell gradient directions are
  spatially autocorrelated.  Mardia & Jupp 2000 (Directional Statistics,
  Wiley) define R̄ and the Rayleigh test.

- **interface_flux** — integrates the boundary-normal expression derivative
  over the Voronoi facets separating cell-type compartments: a genuine
  surface integral (facet length × normal derivative), plus the
  length-weighted mean derivative.  A novel descriptive statistic with no
  published analog; it involves no transport model (see its docstring).

- **spatial_diffusion** — heat equation on the cell graph ∂f/∂t = −Lf,
  approximated by iterative multiplication with the Coifman–Lafon
  density-normalized diffusion operator.  Coifman & Lafon 2006 (Appl.
  Comput. Harmon. Anal.); Andersson & Lundeberg 2021 (Bioinformatics,
  Sepal).

- **decay_length** — fits f(r) = f₀ e^(−r/λ) around a user-specified
  source.  λ is a phenomenological decay length in µm, *not* a physical
  diffusion constant.  Wartlick, Kicheva & González-Gaitán 2009 (Cold
  Spring Harb. Perspect. Biol.) frame morphogen-gradient decay lengths.

Important: these methods operate on static spatial expression fields.  The
"direction" they reveal is the steepest ascent of expression in physical
tissue space, not a temporal trajectory.  Never interpret gradient directions
as cell movement or RNA velocity.
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control

from astrolabe._statistics import permutation_p, adjust_p, annotate_permutation_resolution
from astrolabe._utils import _check_single_library, _spawn_rngs, from_kwarg, with_adata
from astrolabe.tl._spatial_stats import _extract_values

__all__ = [
    "expression_gradient",
    "gradient_field",
    "gradient_significance",
    "interface_flux",
    "bearing_to_group",
    "diffuse_expression",
    "spatial_diffusion",
    "decay_length",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _source_fingerprint(adata, key, layer, spatial_key, neighbors_key, groupby=None):
    """Hash the actual source values, coordinates, graph and row identity."""
    import hashlib
    digest = hashlib.sha256()
    digest.update("\0".join(adata.obs_names).encode())
    for values in (np.asarray(adata.obsm[spatial_key], dtype=np.float64),
                   np.asarray(_extract_values(adata, key, layer), dtype=np.float64)):
        digest.update(np.ascontiguousarray(values).tobytes())
    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    conn = adata.obsp[conn_key].tocsr()
    for values in (conn.data, conn.indices, conn.indptr):
        digest.update(np.ascontiguousarray(values).tobytes())
    if groupby is not None:
        digest.update("\0".join(adata.obs[groupby].astype(str)).encode())
    return digest.hexdigest()


def _get_neighbors_keys(adata: ad.AnnData, neighbors_key: str) -> tuple[str, str]:
    """Return (conn_key, dist_key) stored in uns; raise if graph is missing."""
    uns_entry = adata.uns.get(neighbors_key, {})
    conn_key = uns_entry.get("connectivities_key", neighbors_key + "_connectivities")
    dist_key = uns_entry.get("distances_key", neighbors_key + "_distances")
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. "
            "Run ab.tl.spatial_neighbors() first."
        )
    return conn_key, dist_key


def _diffusion_operator(
    conn_mat: sp.spmatrix,
    normalize: str,
) -> tuple[sp.spmatrix, sp.spmatrix]:
    """Build the row-stochastic diffusion operator ``M`` from a connectivity
    matrix, plus the (possibly density-normalized) kernel ``A_hat`` it was
    row-normalized from.

    ``normalize='coifman_lafon'`` first applies the α = 1 density correction
    ``A_hat = D_inv @ A @ D_inv`` (Coifman & Lafon 2006) so the operator
    follows tissue geometry rather than cell density; ``'row'`` row-normalizes
    the raw kernel directly. The single construction shared by
    :func:`spatial_diffusion` and :func:`diffuse_expression`.
    """
    A = conn_mat.astype(np.float64)
    if normalize == "coifman_lafon":
        d = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
        d[d == 0] = 1.0
        # alpha=1: full 1/(q_i q_j), not 1/sqrt(q_i q_j) (alpha=1/2) --
        # D_inv (not D_inv_sqrt) applied on both sides.
        D_inv = sp.diags(1.0 / d)
        A_hat = D_inv @ A @ D_inv
    elif normalize == "row":
        A_hat = A.copy()
    else:
        raise ValueError(
            f"normalize must be 'coifman_lafon' or 'row', got {normalize!r}"
        )
    row_sums = np.asarray(A_hat.sum(axis=1), dtype=np.float64).ravel()
    row_sums[row_sums == 0] = 1.0
    M = sp.diags(1.0 / row_sums) @ A_hat
    return M, A_hat


def _diffuse_expression(
    f: np.ndarray,
    conn_mat: sp.spmatrix,
    n_steps: int,
) -> np.ndarray:
    """Row-normalize conn_mat then apply f ← M^n f (in-place on a copy).

    Deliberately the plain row-stochastic (``normalize='row'``) operator: this
    is the *local denoiser* used by ``expression_gradient``'s pre-smoothing
    and the surrogate machinery, where changing the operator would shift every
    stored gradient. The density-aware public smoother is
    :func:`diffuse_expression`.
    """
    if n_steps == 0:
        return f.copy()
    M, _ = _diffusion_operator(conn_mat, "row")
    f = np.asarray(f, dtype=np.float64)
    for _ in range(n_steps):
        f = M @ f
    return f


def _resolve_groups(adata, groupby, groups):
    """Validate *groupby* / *groups* and return ``(labels, levels)``.

    ``labels`` is a string array over cells (``None`` when *groupby* is
    ``None``); ``levels`` is the ordered list of groups to process.
    """
    if groupby is None:
        if groups is not None:
            raise ValueError("groups= requires groupby=")
        return None, [None]
    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")
    labels = adata.obs[groupby].astype(str).to_numpy()
    if groups is None:
        levels = sorted(pd.unique(labels).tolist())
    else:
        levels = [str(g) for g in groups]
        missing = set(levels) - set(labels.tolist())
        if missing:
            raise ValueError(f"groups not present in obs['{groupby}']: {sorted(missing)}")
    return labels, levels


def _restrict_conn_union(conn_mat, labels, levels):
    """Keep only edges whose two endpoints share a group in *levels*.

    The union over groups, so one pass builds the graph every group needs at
    once -- each cell then fits against its own group's neighbours only, and
    cells outside every selected group keep no edges at all (handled downstream
    by the existing ``obs['gradient_unreliable']`` machinery). Returns
    ``(conn_restricted, n_dropped_edges)``.
    """
    coo = conn_mat.tocoo()
    keep = np.zeros(conn_mat.nnz, dtype=bool)
    for g in levels:
        in_g = labels == str(g)
        keep |= in_g[coo.row] & in_g[coo.col]
    restricted = sp.coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=conn_mat.shape
    ).tocsr()
    return restricted, int((~keep).sum())


def _restrict_conn(conn_mat, labels, group):
    """Drop edges that leave *group*, and report which cells stay in scope.

    An expression gradient fitted across a cell-type boundary is largely
    measuring *composition* — a lineage-restricted gene changes value at the
    boundary because the cell type changes, not because the gene has a gradient
    within either type — so restricting the neighbourhood to within-group edges
    is what makes the answer a within-type statement. Cells outside the group
    keep no edges, so every downstream per-cell quantity is left at the value
    the existing reliability machinery already assigns to a cell with no
    neighbours.

    Returns ``(conn_restricted, in_group)``. With ``group is None`` the input is
    returned untouched, so the ungrouped path is bit-for-bit unchanged.
    """
    if group is None:
        return conn_mat, np.ones(conn_mat.shape[0], dtype=bool)
    in_group = labels == str(group)
    coo = conn_mat.tocoo()
    keep = in_group[coo.row] & in_group[coo.col]
    restricted = sp.coo_matrix(
        (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=conn_mat.shape
    ).tocsr()
    return restricted, in_group


def _edge_weights(
    coords: np.ndarray,
    adata: ad.AnnData,
    conn_key: str,
    weight_by_distance: bool,
    conn_mat=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (src, dst, weights) from the spatial graph COO representation.

    Distances are recomputed directly from *coords*, not read from a stored
    ``obsp`` distance matrix. That matrix is only guaranteed to carry an
    entry for every edge under `tl.spatial_neighbors`' own default graph --
    under `tl.blend_neighbors` (a joint spatial+expression graph CLAUDE.md
    itself documents and recommends), a connectivity edge frequently has no
    matching distance-matrix entry at all. A missing sparse entry reads as
    a stored `0.0`, indistinguishable from two literally-coincident cells,
    so `1.0 / (dist + 1e-8)` silently assigned those edges a weight of 1e8
    -- six orders of magnitude too large -- corrupting 54% of edges in that
    workflow. Recomputing Euclidean distance from coordinates has no such
    "missing vs. genuinely zero" ambiguity and is O(n_edges), cheap relative
    to the MLS solve this feeds.
    """
    conn_coo = (adata.obsp[conn_key] if conn_mat is None else conn_mat).tocoo()
    src = conn_coo.row.astype(np.intp)
    dst = conn_coo.col.astype(np.intp)

    if weight_by_distance:
        # Gaussian kernel with a per-cell bandwidth (each source cell's own
        # median neighbor distance) — the compactly-supported-style weight
        # classical MLS specifies (Lancaster & Salkauskas 1981; same scheme
        # as BANKSY's 'scaled_gaussian'). The previous 1/(d + 1e-8) weight is
        # unbounded as d→0 (one near-coincident neighbor dominates the whole
        # fit) and does not decay toward the neighborhood boundary, which
        # makes the estimator discontinuous in the neighbor set: adding or
        # removing one long Delaunay edge produced a jump in the gradient.
        dists = np.linalg.norm(coords[dst] - coords[src], axis=1)
        order = np.lexsort((dists, src))
        s_sorted = src[order]
        d_sorted = dists[order]
        boundaries = np.flatnonzero(np.diff(s_sorted)) + 1
        starts = np.concatenate([[0], boundaries])
        stops = np.concatenate([boundaries, [len(s_sorted)]])
        mid = starts + (stops - starts - 1) // 2  # lower median, vectorized
        h = np.full(coords.shape[0], np.nan)
        h[s_sorted[starts]] = d_sorted[mid]
        global_h = np.nanmedian(h)
        h = np.where(np.isfinite(h) & (h > 1e-9), h, global_h if global_h > 0 else 1.0)
        w = np.exp(-0.5 * (dists / h[src]) ** 2)
    else:
        w = np.ones(len(src), dtype=np.float64)
    return src, dst, w


def _mls_gradient(
    coords: np.ndarray,
    f: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Vectorised free-intercept Moving Least Squares gradient per cell.

    Fits the weighted local linear model with a free intercept

        Δf_ij ≈ β0_i + g_i · Δx_ij

    per cell via the batched 3×3 normal equations S_i θ_i = b_i with

        S_i = Σ_j w_ij [1, Δx, Δy]⊗[1, Δx, Δy]   (3×3)
        b_i = Σ_j w_ij Δf [1, Δx, Δy]            (3-vector)

    The center cell enters as one more kernel-weighted sample — its weight
    is exactly exp(0) = 1, and in the Δf parameterization it is the row
    (w=1, Δx=0, Δf=0), touching only S[0,0]. The free intercept β0 absorbs
    the center cell's own noise: a fit constrained through the center value
    (the intercept-free structure-tensor form) leaks that noise into the
    gradient as −A⁻¹(Σ w Δx)ε_i, which is ≈0 in symmetric interiors but
    large and inward-pointing at one-sided tissue-edge neighborhoods —
    magnitude-inflated, boundary-normal-biased edge gradients. β0 is
    discarded after the solve.

    A ridge of 1e-8 on the gradient block only (S[1,1]/S[2,2]) guards
    collinear neighborhoods; S[0,0] ≥ 1 by the center sample and never
    needs one.

    Reference: Fan & Gijbels 1996 (Local Polynomial Modelling, Chapman &
    Hall) — the standard local-linear estimator; Lancaster & Salkauskas
    1981 (Math. Comput.) for the MLS weighting.
    """
    n_cells = coords.shape[0]
    S = np.zeros((n_cells, 3, 3), dtype=np.float64)
    b = np.zeros((n_cells, 3), dtype=np.float64)

    dx = coords[dst] - coords[src]       # (n_edges, 2)  µm displacement
    df = f[dst] - f[src]                 # (n_edges,)    expression diff
    wdx = w[:, None] * dx                # (n_edges, 2)

    np.add.at(S[:, 0, 0], src, w)                       # Σ w
    np.add.at(S[:, 0, 1], src, wdx[:, 0])               # Σ w Δx
    np.add.at(S[:, 0, 2], src, wdx[:, 1])               # Σ w Δy
    np.add.at(S[:, 1, 1], src, wdx[:, 0] * dx[:, 0])    # Σ w Δx²
    np.add.at(S[:, 1, 2], src, wdx[:, 0] * dx[:, 1])    # Σ w ΔxΔy
    np.add.at(S[:, 2, 2], src, wdx[:, 1] * dx[:, 1])    # Σ w Δy²
    S[:, 1, 0] = S[:, 0, 1]
    S[:, 2, 0] = S[:, 0, 2]
    S[:, 2, 1] = S[:, 1, 2]

    # Center cell as a sample (kernel weight exp(0)=1, Δx=0, Δf=0): anchors
    # the intercept without forcing the fit through the center's value.
    S[:, 0, 0] += 1.0
    # Ridge on the gradient block only.
    S[:, 1, 1] += 1e-8
    S[:, 2, 2] += 1e-8

    wdf = w * df
    np.add.at(b[:, 0], src, wdf)
    np.add.at(b[:, 1], src, wdf * dx[:, 0])
    np.add.at(b[:, 2], src, wdf * dx[:, 1])

    # Batched 3×3 solve.  numpy's gufunc signature is (m,m),(m,n)->(m,n), so
    # b must have 2D core dims: reshape to (n_cells, 3, 1) and squeeze after.
    theta = np.linalg.solve(S, b[:, :, None])[:, :, 0]   # (n_cells, 3)
    return theta[:, 1:3].astype(np.float32)


def _neighborhood_quality(
    coords: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell neighborhood diagnostics for the MLS fit.

    Returns ``(n_neighbors, anisotropy, onesidedness)``:

    - ``anisotropy`` — λ_min/λ_max eigenvalue ratio of the **centered**
      weighted structure tensor Ã = M − m mᵀ/(Σw + 1) (the Schur complement
      the free-intercept fit actually solves against, with the +1 matching
      the center sample in ``_mls_gradient``'s S[0,0]). Near 0 means the
      neighborhood is (near-)collinear, where the local linear fit is
      poorly conditioned.
    - ``onesidedness`` — ‖Σ w Δx‖ / Σ w ‖Δx‖ ∈ [0, 1], the kernel- and
      radius-weighted mean resultant of neighbor bearings. The uncentered
      eigenvalue ratio is *blind* to one-sided neighborhoods (a uniform
      half-disc has a perfectly isotropic uncentered tensor), which is why
      this is a separate axis: a full disc or symmetric interior gives ≈ 0,
      a 240°-span neighborhood ≈ 0.41, a half-plane 2/π ≈ 0.64, a corner
      ≈ 0.8–0.9. It is exactly the normalized moment m that controls both
      the constrained fit's center-noise bias and the free-intercept fit's
      conditioning along the boundary normal.
    """
    n_cells = coords.shape[0]
    M = np.zeros((n_cells, 2, 2), dtype=np.float64)
    m = np.zeros((n_cells, 2), dtype=np.float64)
    sw = np.zeros(n_cells, dtype=np.float64)
    swr = np.zeros(n_cells, dtype=np.float64)

    dx = coords[dst] - coords[src]
    wdx = w[:, None] * dx
    np.add.at(M[:, 0, 0], src, wdx[:, 0] * dx[:, 0])
    np.add.at(M[:, 0, 1], src, wdx[:, 0] * dx[:, 1])
    np.add.at(M[:, 1, 1], src, wdx[:, 1] * dx[:, 1])
    M[:, 1, 0] = M[:, 0, 1]
    np.add.at(m[:, 0], src, wdx[:, 0])
    np.add.at(m[:, 1], src, wdx[:, 1])
    np.add.at(sw, src, w)
    np.add.at(swr, src, w * np.linalg.norm(dx, axis=1))

    n_nbrs = np.bincount(src, minlength=n_cells)

    with np.errstate(divide="ignore", invalid="ignore"):
        onesided = np.where(swr > 0, np.linalg.norm(m, axis=1) / swr, 1.0)

    A = M - m[:, :, None] * m[:, None, :] / (sw + 1.0)[:, None, None]

    # closed-form eigenvalues of symmetric 2x2
    tr = A[:, 0, 0] + A[:, 1, 1]
    det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] ** 2
    disc = np.sqrt(np.maximum(tr**2 / 4.0 - det, 0.0))
    lam_max = tr / 2.0 + disc
    lam_min = tr / 2.0 - disc
    with np.errstate(divide="ignore", invalid="ignore"):
        aniso = np.where(lam_max > 0, np.maximum(lam_min, 0.0) / lam_max, 0.0)
    return n_nbrs, aniso, onesided


def _mean_resultant_length(grad: np.ndarray, magnitude_eps: float = 1e-8) -> float:
    """Circular mean resultant length R̄ ∈ [0, 1] of gradient vector directions.

    R̄ = 1 indicates perfectly aligned directions; R̄ ≈ 0 indicates uniform.
    Mardia & Jupp 2000 (Directional Statistics, Wiley), eq. 2.2.

    Takes the raw ``(n, 2)`` gradient vectors, not pre-computed angles, and
    excludes near-zero-magnitude ones before computing directions:
    ``arctan2(0, 0) = 0`` is a real, defined numpy return value, not
    NaN/undefined, so degenerate zero gradients (routine in
    diffusion-smoothed sparse/zero-inflated regions) would otherwise all
    pile up at angle exactly 0 and artificially inflate R̄ -- pure numerical
    noise reading as strong directional coherence (measured: real signal
    R̄=0.48 inflated to a reported 0.92 on affected data).
    """
    magnitude = np.linalg.norm(grad, axis=1)
    valid = magnitude > magnitude_eps
    if not np.any(valid):
        return float("nan")
    angles = np.arctan2(grad[valid, 1], grad[valid, 0])
    return float(np.abs(np.mean(np.exp(1j * angles))))


def _graph_moran(f: np.ndarray, conn_mat: sp.spmatrix) -> float:
    """Moran's I of *f* on the (binarized) graph — cheap autocorrelation gauge."""
    z = f - f.mean()
    denom = float(z @ z)
    if denom <= 0:
        return float("nan")
    W = conn_mat.astype(bool).astype(np.float64)
    s0 = W.sum()
    if s0 == 0:
        return float("nan")
    return float((len(f) / s0) * (z @ (W @ z)) / denom)


def _calibrate_surrogate_smoothing(
    f: np.ndarray,
    conn_mat: sp.spmatrix,
    rng: "np.random.Generator",
    *,
    candidates: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    n_probe: int = 3,
) -> int:
    """Pick the diffusion-step count whose rank-mapped noise surrogates best
    match the observed field's graph Moran's I.

    Surrogates are built by rank-mapping the observed values onto smoothed
    white noise (Viladomat et al. 2014, Ann. Appl. Stat.; Burt et al. 2020,
    NeuroImage — BrainSMASH), so they preserve the exact value distribution
    (including zero inflation) and approximately preserve the spatial
    autocorrelation. Smoothed-noise Moran's I increases monotonically with
    the step count, so the scan stops at the first candidate that overshoots.
    """
    i_obs = _graph_moran(f, conn_mat)
    if not np.isfinite(i_obs):
        return 0
    f_sorted = np.sort(f)
    best_k, best_gap = 0, np.inf
    for k in candidates:
        i_vals = []
        for _ in range(n_probe):
            noise = rng.standard_normal(len(f))
            smooth = _diffuse_expression(noise, conn_mat, k)
            surr = np.empty_like(f_sorted)
            surr[np.argsort(smooth)] = f_sorted
            i_vals.append(_graph_moran(surr, conn_mat))
        i_k = float(np.nanmean(i_vals))
        gap = abs(i_k - i_obs)
        if gap < best_gap:
            best_k, best_gap = k, gap
        if i_k > i_obs:
            break
    return best_k


def _gradient_field_uns_key(ba: inspect.BoundArguments) -> str:
    """Mirrors gradient_field's own ``out_key = key_added or f"gradient_field_{key}"``."""
    return ba.arguments["key_added"] or f"gradient_field_{ba.arguments['key']}"


def _decay_length_uns_key(ba: inspect.BoundArguments) -> str:
    """Mirrors decay_length's own ``out_key = key_added or f"decay_length_{key}"``."""
    return ba.arguments["key_added"] or f"decay_length_{ba.arguments['key']}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@with_adata
def expression_gradient(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    layer: str | None = None,
    spatial_key: str = "spatial",
    neighbors_key: str = "spatial_neighbors",
    n_diffusion_steps: int = 1,
    weight_by_distance: bool = True,
    groupby: str | None = None,
    groups: list[str] | None = None,
    key_added: str | None = None,
) -> None:
    """Compute per-cell gene expression gradient vectors via graph-based MLS.

    For each cell *i*, fits a local linear plane to expression values in
    its spatial neighborhood using weighted least squares (Moving Least
    Squares / diffuse derivative estimator):

    .. math::

        \\Delta f_{ij} \\approx \\beta_{0,i}
        + \\hat{\\mathbf{g}}_i \\cdot \\Delta\\mathbf{x}_{ij}

    solved per cell from the batched 3×3 normal equations with a **free
    intercept** (the standard local-linear estimator, Fan & Gijbels 1996);
    the center cell enters as one more kernel-weighted sample and the
    intercept absorbs its own noise instead of leaking it into the gradient
    (a fit constrained through the center value is magnitude-inflated and
    inward-biased at one-sided tissue-edge neighborhoods).

    Weights follow classical MLS practice: a Gaussian kernel over the
    neighbor distance, with each cell's own median neighbor distance as the
    bandwidth (distances are recomputed from coordinates, never read from a
    stored ``obsp`` distance matrix).  An optional graph-diffusion
    pre-smoothing step (multiply expression by the row-normalized adjacency
    matrix ``n_diffusion_steps`` times) suppresses single-gene sparsity noise
    before differentiation.

    .. note::

        **Magnitudes are relative, not physical.** Diffusion pre-smoothing
        shrinks gradient magnitudes by a graph-dependent factor, so
        magnitudes are only comparable between runs with the same
        ``n_diffusion_steps``, the same spatial graph, and inputs on the same
        scale.  The "per µm" unit below applies to whatever scale *keys* is
        on (log1p expression for genes from a log-normalized ``X``, score
        units for ``obs`` columns).

    .. note::

        **Tissue-boundary cells are mitigated and flagged.** The
        free-intercept local plane fit removes the center-anchored inward
        bias that a constrained fit has at one-sided neighborhoods; what
        cannot be fixed — reduced information along the boundary normal —
        is flagged. ``obs['gradient_unreliable']`` marks cells with fewer
        than 3 neighbors, a centered structure-tensor eigenvalue ratio
        below 0.05 (collinear), or a neighbor-bearing one-sidedness above
        0.45 (half-plane or worse: a full disc gives 0, a half-plane
        2/π ≈ 0.64). The continuous statistic is stored in
        ``obs['gradient_onesidedness']``; exclude or down-weight flagged
        cells in downstream summaries (every consumer in this library
        does so by default).

    .. note::

        Gradient directions describe the steepest ascent of *expression* in
        physical tissue space.  They are **not** RNA velocity (which is a
        temporal derivative from spliced/unspliced ratios) and should never be
        interpreted as cell movement.

    .. note::

        Recommend ``ab.tl.spatial_neighbors(mdata, coord_type='delaunay',
        delaunay_max_dist=200)`` for the most isotropic neighborhoods.
        KNN graphs can introduce directional bias where cell density varies.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    keys
        Gene name(s) or ``obs`` column name(s) to compute gradients for.
        Module / signature scores (from ``tl.score_genes``, ``tl.pca``
        components, etc.) give much higher SNR than sparse single genes.
    layer
        Layer in ``adata.layers`` to use instead of ``X``.
    spatial_key
        Key in ``obsm`` with ``(n_cells, 2)`` µm coordinates.
    neighbors_key
        Key in ``uns`` / ``obsp`` for the spatial graph.  Must have been built
        by :func:`~astrolabe.tl.spatial_neighbors`.
    n_diffusion_steps
        Number of graph-diffusion smoothing steps applied to expression before
        computing gradients.  Default ``1`` is usually sufficient for sparse
        single genes; set ``0`` to skip for dense module scores.
    weight_by_distance
        If ``True`` (default), weight neighbors with a Gaussian kernel whose
        bandwidth is each cell's own median neighbor distance.
        If ``False``, all neighbors receive equal weight.
    groupby
        ``obs`` column (e.g. a cell type) restricting each cell's local
        neighborhood to **within-group edges**, making the fitted gradient a
        within-type statement. This matters more than it sounds: a gradient
        fitted across a cell-type boundary largely measures *composition*,
        because a lineage-restricted gene changes value at the boundary through
        the cell type changing, not through any gradient within either type. On
        planted data a flat-within-type lineage gene shows a pooled magnitude
        ~3.6x its within-type magnitude, while a genuine within-type gradient is
        unchanged. Cells outside the selected groups keep no edges and are
        handled by the existing ``obs['gradient_unreliable']`` machinery.
        Recorded in ``uns['gradient_params']`` so :func:`gradient_field` and
        :func:`gradient_significance` reproduce the same restriction.
    groups
        Restrict to these *groupby* values. Requires *groupby*.
    key_added
        Prefix for output keys, combined with each entry of *keys* as
        ``'{key_added}_{key}'``. ``None`` (default) uses the plain
        ``'gradient_{key}'`` naming below. Pass this same value as
        ``gradient_key_added`` to :func:`~astrolabe.tl.gradient_field`,
        :func:`~astrolabe.tl.gradient_significance`, or
        :func:`~astrolabe.tl.interface_flux` to read gradients computed
        with a custom prefix.

    Writes
    ------
    ``obsm['gradient_{key}']``
        ``float32`` ``(n_cells, 2)`` — gradient vectors ``[gx, gy]`` in
        units of (feature scale) per µm — log1p expression per µm for genes
        from a log-normalized ``X``; see the magnitude note above.
    ``obs['gradient_{key}_magnitude']``
        ``float32`` ``(n_cells,)`` — Euclidean magnitude ‖g‖.
    ``obs['gradient_{key}_direction']``
        ``float32`` ``(n_cells,)`` — direction angle ``arctan2(gy, gx)``
        in radians; ``NaN`` where the gradient magnitude is ~0 (a zero
        vector has no direction).
    ``obs['gradient_unreliable']``
        ``bool`` — cells whose local fit is poorly conditioned (tissue
        boundary / island); key-independent, written once per call.
    ``obs['gradient_onesidedness']``
        ``float32`` — neighbor-bearing one-sidedness in [0, 1] (0 =
        symmetric surround, 2/π ≈ 0.64 = half-plane); the continuous
        statistic behind the ``> 0.45`` term of the flag, useful for QC
        maps. Key-independent, written once per call.
    ``uns['gradient_params'][<obsm key>]``
        Dict of parameters used to compute each stored gradient, keyed by
        its ``obsm`` name — so multiple runs with different smoothing or
        graphs never overwrite each other's provenance.
    """
    _check_single_library(adata)

    if isinstance(keys, str):
        keys = [keys]

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found. "
            "Ensure ab.tl.spatial_neighbors() has been run."
        )
    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)

    conn_key, _dist_key = _get_neighbors_keys(adata, neighbors_key)
    conn_mat = adata.obsp[conn_key]
    labels, levels = _resolve_groups(adata, groupby, groups)
    if groupby is not None:
        conn_mat, n_dropped = _restrict_conn_union(conn_mat, labels, levels)
        n_out = int((~np.isin(labels, [str(g) for g in levels])).sum())
        log.info(
            "expression_gradient: restricted to within-'%s' edges over %d group(s); "
            "%d edge(s) dropped, %d cell(s) outside the selected groups",
            groupby, len(levels), n_dropped, n_out,
        )
    src, dst, w = _edge_weights(
        coords, adata, conn_key, weight_by_distance, conn_mat=conn_mat
    )

    # Key-independent fit-quality flag: tissue-boundary / island cells whose
    # local linear fit is one-sided or near-collinear. Thresholds are fixed
    # constants: onesidedness 0.45 sits between a tolerable 240°-span
    # neighborhood (0.41) and the mildest real half-plane case (a kernel-
    # weighted lattice-edge cell, ≈0.52); aniso 0.05 catches collinearity.
    n_nbrs, aniso, onesided = _neighborhood_quality(coords, src, dst, w)
    unreliable = (n_nbrs < 3) | (aniso < 0.05) | (onesided > 0.45)
    adata.obs["gradient_unreliable"] = unreliable
    adata.obs["gradient_onesidedness"] = onesided.astype(np.float32)
    if unreliable.any():
        log.info(
            "expression_gradient: %d/%d cells flagged gradient_unreliable "
            "(boundary/one-sided neighborhoods)",
            int(unreliable.sum()), adata.n_obs,
        )

    # Warn on very sparse inputs
    for key in keys:
        if key in adata.var_names:
            idx = adata.var_names.get_loc(key)
            mat = adata.layers[layer] if layer is not None else adata.X
            col = mat[:, idx]
            if sp.issparse(col):
                col = col.toarray().ravel()
            frac_nonzero = np.count_nonzero(col) / len(col)
            if frac_nonzero < 0.2 and n_diffusion_steps == 0:
                warnings.warn(
                    f"expression_gradient: '{key}' has {frac_nonzero:.1%} non-zero "
                    "values. Consider n_diffusion_steps=1 to smooth before "
                    "differentiating.",
                    stacklevel=3,
                )

    for key in keys:
        f = _extract_values(adata, key, layer)
        if n_diffusion_steps > 0:
            f = _diffuse_expression(f, conn_mat, n_diffusion_steps)

        grad = _mls_gradient(coords, f.astype(np.float64), src, dst, w)

        # A plain `key_added or f"gradient_{key}"` used to reuse the exact
        # same literal `key_added` string for every key in a multi-key call,
        # so obsm[base] was silently overwritten by each subsequent key --
        # only the last gene's gradient survived. `key_added` is now always
        # combined with `key` as a prefix, so it stays unique per key; the
        # `key_added=None` default naming (by far the common case) is
        # unchanged.
        base = f"{key_added}_{key}" if key_added else f"gradient_{key}"
        adata.obsm[base] = grad
        magnitude = np.linalg.norm(grad, axis=1).astype(np.float32)
        adata.obs[f"{base}_magnitude"] = magnitude
        # A zero gradient has no direction: arctan2(0, 0) = 0 would pile the
        # degenerate cells onto bearing 0 (e.g. a `.describe()` median of
        # exactly 0.0 for sparse genes). NaN is the honest value; every
        # consumer (direction_histogram, R̄) already filters non-finite.
        direction = np.arctan2(grad[:, 1], grad[:, 0]).astype(np.float32)
        direction[magnitude <= 1e-8] = np.nan
        adata.obs[f"{base}_direction"] = direction

        log.info(
            "expression_gradient: '%s' — mean magnitude %.4f, "
            "directional coherence R̄=%.3f",
            key,
            float(adata.obs[f"{base}_magnitude"].mean()),
            _mean_resultant_length(grad),
        )

    # Per-obsm-key provenance: a later call with different smoothing or a
    # different graph must not silently change what gradient_significance /
    # gradient_field reproduce for an earlier key.
    params_store = adata.uns.setdefault("gradient_params", {})
    for key in keys:
        base = f"{key_added}_{key}" if key_added else f"gradient_{key}"
        params_store[base] = {
            "key": key,
            "source_fingerprint": _source_fingerprint(adata, key, layer, spatial_key, neighbors_key, groupby),
            "layer": layer,
            "spatial_key": spatial_key,
            "neighbors_key": neighbors_key,
            "n_diffusion_steps": n_diffusion_steps,
            "weight_by_distance": weight_by_distance,
            "groupby": groupby,
            "groups": (None if groups is None else [str(g) for g in groups]),
        }


@with_adata(mirror_uns=_gradient_field_uns_key)
def gradient_field(
    adata: ad.AnnData,
    key: str,
    *,
    spatial_key: str = "spatial",
    n_grid: int = 50,
    grid_smooth_sigma: float = 1.0,
    bandwidth_um: float | None = None,
    mask_threshold: float = 1.0,
    gradient_key_added: str | None = None,
    key_added: str | None = None,
) -> None:
    """Grid the expression field and derive its gradient/divergence maps.

    Interpolates the *scalar* expression field (with the same diffusion
    pre-smoothing recorded by :func:`~astrolabe.tl.expression_gradient` for
    this key) onto an ``n_grid × n_grid`` uniform grid by binned
    averaging with a normalized Gaussian smoothing kernel
    (``grid_smooth_sigma`` grid cells), then differentiates the **masked**
    grid with second-order central differences to obtain U/V and the
    divergence ``∂U/∂x + ∂V/∂y = ∇²f``.

    Two deliberate design points:

    - **The scalar field is interpolated, not the vector components.**
      Interpolating U and V independently does not produce the gradient of
      any scalar function — the result has nonzero curl purely from the
      interpolator, and any "rotation" seen in streamlines would be
      interpolation artifact. Differentiating one interpolated scalar keeps
      the gridded field curl-free by construction (discrete central-difference
      mixed partials commute), so no ``curl`` output exists — for a gradient
      field it is zero by definition.
    - **The mask is applied before differencing.** Finite differences taken
      against RBF values extrapolated outside the tissue would manufacture a
      bright divergence rim exactly at the tissue edge; masking first means
      derivative stencils touching a void grid point are NaN instead.

    **Sign convention:** since the field is a gradient, its divergence is the
    Laplacian of expression — **positive divergence marks a local expression
    minimum** (gradients point outward, ascending away from the low point);
    **negative divergence marks a local maximum / peak** (gradients converge
    into it, in the same sense that streamlines terminate on peaks).

    A tissue-support mask is derived from the log-density of a Gaussian KDE
    fit to cell positions; grid points in tissue voids have U/V/divergence/curl
    set to ``NaN``. The mask cutoff is a **percentile of the same KDE's
    log-density evaluated at the actual cell locations**, not an absolute
    log-density value — an absolute cutoff has no fixed meaning across
    datasets (KDE log-density scale depends on tissue footprint, cell count,
    and bandwidth; real Xenium runs typically sit around −12 to −18, so a
    fixed threshold anywhere near 0 masks out every grid point, silently
    returning an all-NaN field with no error).

    The grid field is the prerequisite for :func:`~astrolabe.pl.gradient_streamlines`
    and :func:`~astrolabe.pl.divergence_plot`.  The streamlines represent
    integral curves of the spatial ODE dx/dt = ∇f(x) — descriptive field
    lines of the static expression field, not cell trajectories.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    key
        Feature key used in :func:`~astrolabe.tl.expression_gradient`
        (e.g. ``'EPCAM'`` → reads ``obsm['gradient_EPCAM']``, or
        ``obsm['{gradient_key_added}_EPCAM']`` if that call used a custom
        ``key_added``).
    spatial_key
        Key in ``obsm`` with ``(n_cells, 2)`` µm coordinates.
    n_grid
        Number of grid points along each axis.  ``50`` is a good default;
        increase for larger FOVs or finer features.
    grid_smooth_sigma
        Gaussian smoothing bandwidth in grid-cell units applied during
        gridding (normalized convolution: empty bins are filled from their
        neighborhood rather than biasing toward zero). Larger values give a
        smoother field and divergence at the cost of spatial resolution.
    bandwidth_um
        KDE bandwidth for the tissue mask (µm).  Defaults to Scott's rule
        applied to the coordinate range.
    mask_threshold
        Percentile (0-100) of the cell-location KDE log-density used as the
        tissue-void cutoff.  Grid points below this percentile are set to
        ``NaN``.  ``1.0`` (default) masks out only the sparsest ~1% of the
        density range actually observed at real cell locations. Increase to
        shrink the supported region (stricter); decrease to expand it.
    gradient_key_added
        ``key_added`` used in the :func:`~astrolabe.tl.expression_gradient`
        call this reads from, if it was given a custom one. ``None``
        (default) reads ``obsm['gradient_{key}']``, matching
        :func:`~astrolabe.tl.expression_gradient`'s own default.

    Writes ``uns[key_added or f'gradient_field_{key}']``, mirrored to
    ``mdata.uns[<that key>][<modality>]`` when called on a MuData.
    key_added
        Key for ``uns`` output.  Default: ``'gradient_field_{key}'``.

    Writes
    ------
    ``uns['gradient_field_{key}']``
        Dict with keys: ``F`` (interpolated scalar field), ``U``, ``V``
        (2-D float64 grids, shape (n_grid, n_grid)), ``X_grid``, ``Y_grid``
        (1-D µm coordinates), ``divergence`` (2-D, = ∇²f; NaN outside the
        tissue mask and at its one-cell rim), ``mask`` (bool), ``params``.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde

    _check_single_library(adata)

    grad_key = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
    if grad_key not in adata.obsm:
        key_added_hint = f", key_added='{gradient_key_added}'" if gradient_key_added else ""
        raise KeyError(
            f"obsm['{grad_key}'] not found. "
            f"Run ab.tl.expression_gradient(mdata, keys=['{key}']{key_added_hint}) first."
        )

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)

    # Rebuild the same smoothed scalar field the per-cell gradients were
    # computed from, using that gradient's own stored provenance.
    key_params = adata.uns.get("gradient_params", {}).get(grad_key, {})
    layer = key_params.get("layer")
    n_diffusion_steps = key_params.get("n_diffusion_steps", 1)
    neighbors_key = key_params.get("neighbors_key", "spatial_neighbors")
    f_cell = _extract_values(adata, key, layer).astype(np.float64)
    if n_diffusion_steps > 0:
        conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
        conn_f = adata.obsp[conn_key]
        grad_groupby = key_params.get("groupby")
        grad_groups = key_params.get("groups")
        if grad_groupby is not None:
            lab_f, lev_f = _resolve_groups(adata, grad_groupby, grad_groups)
            conn_f, _ = _restrict_conn_union(conn_f, lab_f, lev_f)
        f_cell = _diffuse_expression(f_cell, conn_f, n_diffusion_steps)

    # Regular grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Grid the SCALAR field (see docstring: interpolating U/V independently
    # is not the gradient of anything) by binned averaging + normalized
    # Gaussian convolution. NOT a per-point local interpolator (e.g.
    # RBFInterpolator with neighbors=): those evaluate each grid point with
    # its own neighbor set, so adjacent grid points use different local fits
    # and the resulting field is discontinuous at grid scale — its finite
    # differences flip sign cell-to-cell (streamlines terminate immediately)
    # and its Laplacian is a checkerboard. Binning + one shared Gaussian
    # kernel gives a genuinely smooth grid whose first and second
    # differences are meaningful.
    log.info("gradient_field: gridding scalar field to %d × %d …", n_grid, n_grid)
    dx_spacing = (x_max - x_min) / (n_grid - 1)
    dy_spacing = (y_max - y_min) / (n_grid - 1)
    xi = np.clip(np.rint((coords[:, 0] - x_min) / dx_spacing).astype(int), 0, n_grid - 1)
    yi = np.clip(np.rint((coords[:, 1] - y_min) / dy_spacing).astype(int), 0, n_grid - 1)
    f_sum = np.zeros((n_grid, n_grid), dtype=np.float64)   # rows = y, cols = x
    f_cnt = np.zeros((n_grid, n_grid), dtype=np.float64)
    np.add.at(f_sum, (yi, xi), f_cell)
    np.add.at(f_cnt, (yi, xi), 1.0)
    # normalized (mask-aware) Gaussian convolution: blur(values)/blur(weights)
    # — empty bins are filled from their neighborhood instead of biasing
    # toward zero, and occupied bins are smoothed with one shared kernel.
    sigma = max(grid_smooth_sigma, 1e-6)
    num = gaussian_filter(f_sum, sigma=sigma, mode="nearest")
    den = gaussian_filter(f_cnt, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(den > 1e-12, num / den, np.nan)

    # Tissue mask via Gaussian KDE log-density. The cutoff is a percentile of
    # the *same* KDE's log-density evaluated at actual cell locations, not an
    # absolute log-density value -- that scale depends on tissue footprint,
    # cell count, and bandwidth, so a fixed absolute cutoff has no portable
    # meaning across datasets (see docstring).
    bw = bandwidth_um if bandwidth_um is not None else "scott"
    cutoff = None
    try:
        kde = gaussian_kde(coords.T, bw_method=bw)
        cell_log_dens = np.log(kde(coords.T))
        cutoff = float(np.percentile(cell_log_dens, mask_threshold))
        log_dens = np.log(kde(grid_pts.T)).reshape(n_grid, n_grid)
        mask = log_dens > cutoff
    except Exception:
        log.warning("gradient_field: KDE tissue mask failed; using all-True mask.")
        mask = np.ones((n_grid, n_grid), dtype=bool)

    # Mask BEFORE differencing: stencils that would reach outside the tissue
    # propagate NaN instead of differencing against extrapolated values.
    F_masked = np.where(mask, F, np.nan)

    dF_dy, dF_dx = np.gradient(F_masked, dy_spacing, dx_spacing)
    U = dF_dx
    V = dF_dy
    dU_dy, dU_dx = np.gradient(U, dy_spacing, dx_spacing)
    dV_dy, dV_dx = np.gradient(V, dy_spacing, dx_spacing)
    divergence = dU_dx + dV_dy  # = ∇²f; NaN-rimmed by construction

    out_key = key_added or f"gradient_field_{key}"
    adata.uns[out_key] = {
        "F": F_masked,
        "U": U,
        "V": V,
        "X_grid": x_grid,
        "Y_grid": y_grid,
        "divergence": divergence,
        "mask": mask,
        "params": {
            "key": key,
            "spatial_key": spatial_key,
            "n_grid": n_grid,
            "grid_smooth_sigma": grid_smooth_sigma,
            "n_diffusion_steps": n_diffusion_steps,
            "mask_threshold": mask_threshold,
            "mask_log_density_cutoff": cutoff,
        },
    }
    log.info(
        "gradient_field: '%s' — grid %d×%d, tissue coverage %.1f%%",
        key,
        n_grid,
        n_grid,
        100 * mask.mean(),
    )


@with_adata(mirror_uns=from_kwarg("key_added"))
def gradient_significance(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    spatial_key: str | None = None,
    neighbors_key: str | None = None,
    layer: str | None = None,
    null: Literal["smooth", "csr"] = "smooth",
    n_perms: int = 200,
    random_state: int = 0,
    n_jobs: int = 1,
    exclude_unreliable: bool = True,
    groupby: str | None = None,
    groups: list[str] | None = None,
    gradient_key_added: str | None = None,
    key_added: str = "gradient_significance",
) -> None:
    """Surrogate test for tissue-wide gradient directional coherence.

    For each feature key, compares the observed gradient field against
    ``n_perms`` surrogate expression fields:

    1. **Coherence** (the primary test): do gradient directions share a
       tissue-wide bearing more than the null?  Measured by the **mean
       resultant length** R̄ ∈ [0, 1] over gradient angles (Mardia & Jupp
       2000, Directional Statistics, eq. 2.2).  One-sided: larger R̄ is
       significant.
    2. **Magnitude**: is the mean gradient steeper or shallower than the
       null?  Two-sided.

    .. note::

        **R̄ detects one global direction, nothing else.** It is provably
        blind to radially convergent/divergent structure: gradients pointing
        into many separate hotspots span all bearings and give R̄ ≈ 0
        regardless of how strong the local structure is. Use
        :func:`~astrolabe.tl.gradient_field`'s divergence for hotspot-style
        patterns. Also report the effect size, not just p: at large *n* even
        R̄ far below ``coherence_null_mean + a few percent`` can reach the
        permutation floor.

    **The null model** (``null='smooth'``, default) preserves both the exact
    value distribution and the spatial autocorrelation of the observed
    field: white noise is diffused on the spatial graph until its graph
    Moran's I matches the observed field's, then the observed values are
    rank-mapped onto it (Viladomat et al. 2014, Ann. Appl. Stat. 8(3);
    Burt et al. 2020, NeuroImage 220 — "BrainSMASH" surrogates).  This
    matters because a full label shuffle (``null='csr'``) destroys spatial
    autocorrelation entirely, so testing against it only answers "is this
    feature spatially structured at all?" — any smooth field, coherent or
    not, rejects it.  The parametric Rayleigh test is unsuitable for the
    same reason (it assumes independent angles). ``null='csr'`` is retained
    for exactly that weaker question and for comparison.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    keys
        Feature keys (gene names or ``obs`` columns).  Must match the
        ``keys`` argument used in a prior :func:`~astrolabe.tl.expression_gradient`
        call (pre-computed gradients are read from ``obsm``).
    spatial_key
        ``obsm`` key for µm coordinates — must match the
        :func:`~astrolabe.tl.expression_gradient` call (checked against the
        stored per-key params when available).
    neighbors_key
        Spatial graph key.
    layer
        Expression layer for re-computing null gradients.
    null
        ``'smooth'`` (default) — autocorrelation-matched rank-mapped noise
        surrogates (see above). ``'csr'`` — full label shuffle; only tests
        "any spatial structure at all".
    n_perms
        Number of surrogates.  The smallest attainable p is
        ``1/(n_perms + 1)`` (≈0.005 at the default ``200``); use ``1000``
        for finer resolution.
    random_state
        Seed for reproducibility.
    n_jobs
        Parallel workers for the permutation loop (joblib, thread-backed —
        no extra memory copies since arrays are shared, not pickled). ``1``
        (default) runs sequentially; ``-1`` uses all available cores.
        Per-surrogate progress is logged at DEBUG level.
    exclude_unreliable
        Exclude cells flagged ``obs['gradient_unreliable']`` (tissue
        boundary / one-sided neighborhoods, where the local linear fit is
        under-determined along the boundary normal) from both the observed
        statistics and every surrogate's —
        the flag is pure neighborhood geometry, identical under the null, so
        exclusion is null-consistent. Matches the default of
        :func:`~astrolabe.tl.interface_flux`,
        :func:`~astrolabe.pl.gradient_quiver`, and
        :func:`~astrolabe.pl.direction_histogram` — all four gradient
        consumers exclude these cells by default.
    groupby
        ``obs`` column (e.g. a cell type). The test then runs **within each
        group**, and the result gains a ``group`` index level and an ``n_cells``
        column. This is where grouping matters most: coherence is a
        *directional* statistic, so two lineages with opposite bearings cancel
        when pooled, and the pooled answer is not a weaker version of the truth
        but a different one. The surrogate null is generated on the group's own
        restricted graph so it stays autocorrelation-matched to the field being
        tested. A warning is emitted if a gradient computed with a *groupby* is
        then tested pooled.
    groups
        Restrict to these *groupby* values. Requires *groupby*.
    gradient_key_added
        ``key_added`` used in the :func:`~astrolabe.tl.expression_gradient`
        call this reads from, if it was given a custom one. ``None``
        (default) reads ``obsm['gradient_{key}']``.
    key_added
        Key in ``uns`` for results.

    Writes
    ------
    ``uns[key_added]``
        :class:`pandas.DataFrame` with index = keys and columns:

        - ``magnitude`` — observed mean gradient magnitude
        - ``coherence`` — observed R̄ (mean resultant length)
        - ``coherence_null_mean`` — mean R̄ across the surrogates (the effect
          size baseline: report observed R̄ *relative to this*, not alone)
        - ``p_magnitude`` — empirical p-value (two-sided)
        - ``p_coherence`` — empirical p-value (one-sided: larger R̄)
        - ``p_magnitude_adj`` / ``p_coherence_adj`` — Benjamini–Hochberg
          adjusted p-values
        - ``null`` / ``null_smoothing_steps`` — the null model used

        Mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.
    """
    _check_single_library(adata)

    if isinstance(keys, str):
        keys = [keys]

    if n_perms < 0:
        raise ValueError("n_perms must be nonnegative")
    all_params = adata.uns.get("gradient_params", {})
    provenance = []
    for key in keys:
        base = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
        if base not in all_params:
            raise ValueError("gradient provenance is missing; recompute expression_gradient")
        provenance.append(all_params[base])
    def resolve(name, requested, default):
        stored = [p.get(name, default) for p in provenance]
        if name == "groups":
            stored = [None if value is None else list(value) for value in stored]
            requested = None if requested is None else list(requested)
        if any(value != stored[0] for value in stored[1:]):
            raise ValueError(f"requested gradients have different {name} provenance; test separately")
        if requested is not None and requested != stored[0]:
            raise ValueError(f"{name} conflicts with fitted gradient provenance")
        return stored[0]
    neighbors_key = resolve("neighbors_key", neighbors_key, "spatial_neighbors")
    spatial_key = resolve("spatial_key", spatial_key, "spatial")
    groupby = resolve("groupby", groupby, None)
    groups = resolve("groups", groups, None)
    for key, parameters in zip(keys, provenance):
        if layer is not None and layer != parameters.get("layer"):
            raise ValueError("layer conflicts with fitted gradient provenance")
        actual = _source_fingerprint(adata, key, parameters.get("layer"), spatial_key, neighbors_key, groupby)
        if parameters.get("source_fingerprint") != actual:
            raise ValueError("gradient source changed; recompute expression_gradient before significance")

    conn_key, _dist_key = _get_neighbors_keys(adata, neighbors_key)
    conn_mat = adata.obsp[conn_key]
    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)
    n_cells = len(coords)

    rng = np.random.default_rng(random_state)
    all_params = adata.uns.get("gradient_params", {})

    # Reliability mask: geometry-only (same under the null), so applying it
    # to observed AND surrogate statistics is null-consistent.
    if exclude_unreliable and "gradient_unreliable" in adata.obs:
        reliable = ~adata.obs["gradient_unreliable"].to_numpy(dtype=bool)
    else:
        reliable = np.ones(n_cells, dtype=bool)

    rows = []
    group_labels, group_levels = _resolve_groups(adata, groupby, groups)
    for key, group in [(k, g) for k in keys for g in group_levels]:
        grad_key = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
        if grad_key not in adata.obsm:
            key_added_hint = f", key_added='{gradient_key_added}'" if gradient_key_added else ""
            raise KeyError(
                f"obsm['{grad_key}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']{key_added_hint}) first."
            )
        # A directional statistic pooled over cell types is a statement about
        # the mixture, not about any of them: two lineages with opposite
        # bearings cancel. Restricting the cells AND the graph the surrogates
        # diffuse on keeps the null autocorrelation-matched to the field
        # actually being tested.
        conn_mat, in_group = _restrict_conn(adata.obsp[conn_key], group_labels, group)
        scope = reliable & in_group
        if int(scope.sum()) < 10:
            log.warning(
                "gradient_significance: group %r has %d reliable cell(s); skipped",
                group, int(scope.sum()),
            )
            continue
        grad = adata.obsm[grad_key].astype(np.float64)[scope]
        obs_magnitude = float(np.linalg.norm(grad, axis=1).mean())
        obs_coherence = _mean_resultant_length(grad)

        # Reproduce exactly the smoothing/weighting this gradient was built
        # with, from its own stored provenance (never a global last-run dict).
        key_params = all_params.get(grad_key, {})
        if not key_params:
            log.warning(
                "gradient_significance: no stored gradient_params for '%s'; "
                "assuming defaults (n_diffusion_steps=1, weight_by_distance=True).",
                grad_key,
            )
        n_diffusion_steps = key_params.get("n_diffusion_steps", 1)
        weight_by_distance = key_params.get("weight_by_distance", True)
        stored_groupby = key_params.get("groupby")
        if stored_groupby is not None and groupby is None:
            log.warning(
                "gradient_significance: '%s' was computed with "
                "groupby='%s' (within-group edges only) but is being tested "
                "pooled. Pass the same groupby= so the null matches the field.",
                grad_key, stored_groupby,
            )
        stored_spatial = key_params.get("spatial_key", spatial_key)
        if stored_spatial != spatial_key:
            raise ValueError(
                f"'{grad_key}' was computed with spatial_key='{stored_spatial}' "
                f"but gradient_significance was called with spatial_key="
                f"'{spatial_key}' — the null would use different coordinates."
            )
        src, dst, w = _edge_weights(
            coords, adata, conn_key, weight_by_distance, conn_mat=conn_mat
        )

        stored_layer = key_params.get("layer")
        if layer is not None and layer != stored_layer:
            raise ValueError("layer conflicts with fitted gradient provenance")
        if key_params.get("neighbors_key", neighbors_key) != neighbors_key:
            raise ValueError("neighbors_key conflicts with fitted gradient provenance")
        f = _extract_values(adata, key, stored_layer)
        if null == "smooth":
            n_smooth = _calibrate_surrogate_smoothing(f, conn_mat, rng)
            log.info(
                "gradient_significance: '%s' surrogate smoothing calibrated "
                "to %d diffusion steps (graph Moran's I matched)",
                key, n_smooth,
            )
        else:
            n_smooth = 0
        f_sorted = np.sort(f)

        def _one_surrogate(child_rng: "np.random.Generator") -> tuple[float, float]:
            if null == "smooth":
                noise = child_rng.standard_normal(n_cells)
                base = _diffuse_expression(noise, conn_mat, n_smooth)
                surr = np.empty_like(f_sorted)
                surr[np.argsort(base)] = f_sorted
            else:  # csr: full label shuffle
                surr = f[child_rng.permutation(n_cells)]
            if n_diffusion_steps > 0:
                surr = _diffuse_expression(surr, conn_mat, n_diffusion_steps)
            surr_grad = _mls_gradient(coords, surr, src, dst, w).astype(np.float64)[scope]
            magnitude = float(np.linalg.norm(surr_grad, axis=1).mean())
            coherence = _mean_resultant_length(surr_grad)
            return magnitude, coherence

        # Each key consumes one draw from the shared rng to seed its own
        # independent surrogate substreams — keeps the overall sequence
        # deterministic across keys while surrogates within a key run
        # safely in parallel.
        child_rngs = _spawn_rngs(int(rng.integers(0, 2**63 - 1)), n_perms)
        log.info(
            "gradient_significance: '%s' running %d surrogates (n_jobs=%d)",
            key, n_perms, n_jobs,
        )
        if n_jobs == 1:
            log_every = max(1, n_perms // 10)
            results = []
            for i, child_rng in enumerate(child_rngs):
                if i % log_every == 0:
                    log.debug("gradient_significance: surrogate %d/%d", i + 1, n_perms)
                results.append(_one_surrogate(child_rng))
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_surrogate)(child_rng) for child_rng in child_rngs
            )

        null_magnitudes = np.array([r[0] for r in results], dtype=np.float64)
        null_coherences = np.array([r[1] for r in results], dtype=np.float64)

        # (k+1)/(n+1) floor -- an exact 0 isn't a valid Monte Carlo p-value
        # estimate and breaks downstream -log10(p)/FDR steps. Magnitude is
        # two-sided (steeper OR shallower than the matched null is a
        # deviation); coherence is one-sided (only "more aligned" is the
        # alternative of interest).
        p_mag, n_mag = permutation_p(np.asarray(obs_magnitude), null_magnitudes)
        p_coh, n_coh = permutation_p(np.asarray(obs_coherence), null_coherences, alternative="greater")
        rows.append(
            {
                "key": key,
                **({"group": str(group)} if group is not None else {}),
                "n_cells": int(scope.sum()),
                "magnitude": obs_magnitude,
                "coherence": obs_coherence,
                "coherence_null_mean": (float(np.nanmean(null_coherences)) if np.isfinite(null_coherences).any() else np.nan),
                "p_magnitude": float(p_mag),
                "p_coherence": float(p_coh),
                "n_valid_coherence": int(n_coh),
                "n_valid_magnitude": int(n_mag),
                "null": null,
                "null_smoothing_steps": n_smooth,
            }
        )
        log.info(
            "gradient_significance: '%s' magnitude=%.4f (p=%.3f, two-sided) "
            "coherence=%.3f vs null mean %.3f (p=%.3f)",
            key,
            obs_magnitude,
            float(p_mag),
            obs_coherence,
            (float(np.nanmean(null_coherences)) if np.isfinite(null_coherences).any() else np.nan),
            float(p_coh),
        )

    if not rows:
        raise ValueError(
            "no key/group combination had enough reliable cells to test"
        )
    result = pd.DataFrame(rows)
    # with groups the (key, group) pair is the unit, so a plain 'key' index
    # would carry duplicates
    result = result.set_index(["key", "group"] if groupby is not None else "key")
    result["p_magnitude_adj"] = adjust_p(
        result["p_magnitude"].to_numpy()
    )
    result["p_coherence_adj"] = adjust_p(
        result["p_coherence"].to_numpy()
    )
    annotate_permutation_resolution(result, "p_magnitude", "n_valid_magnitude",
        label="gradient magnitude", suffix="_magnitude")
    annotate_permutation_resolution(result, "p_coherence", "n_valid_coherence",
        alternative="greater", label="gradient coherence", suffix="_coherence")
    adata.uns[key_added] = result


@with_adata(mirror_uns=from_kwarg("key_added"))
def interface_flux(
    adata: ad.AnnData,
    groupby: str,
    keys: str | list[str],
    *,
    spatial_key: str = "spatial",
    max_pair_dist_um: float = 200.0,
    exclude_unreliable: bool = True,
    gradient_key_added: str | None = None,
    key_added: str = "interface_flux",
) -> None:
    """Expression flux integrated over cell-type interfaces (Voronoi facets).

    Builds the Voronoi tessellation of the cell centroids and, for every
    finite Voronoi facet separating two cells with different ``groupby``
    labels, evaluates the boundary-normal expression derivative
    ``df/dn = ½(∇f_i + ∇f_j) · n̂`` (the Voronoi facet is perpendicular to
    the centroid-to-centroid direction, so n̂ is that unit vector) and
    integrates it over the facet length:

    .. math::

        \\Phi(A \\to B) \\;=\\; \\sum_{\\text{facets } i \\in A,\\, j \\in B}
        L_{ij}\\; \\frac{\\partial f}{\\partial n}\\Big|_{ij}

    This is a genuine surface integral of the normal derivative over the
    A|B interface — units are (feature scale), i.e. per-µm derivative × µm
    of interface. A large interface contributes proportionally more than a
    three-cell contact; the previous per-edge arithmetic mean did not
    distinguish them. Also reported: the length-weighted **mean normal
    derivative** (units per µm) and the total **interface length** per pair.

    .. note::

        **The antisymmetry is an identity, not a finding.** By construction
        ``Φ(B→A) = −Φ(A→B)`` exactly (same facets, flipped normal). Only one
        triangle of the matrix carries information; a "directed" reading of
        both halves double-counts the same numbers.

    .. note::

        **This is a novel descriptive statistic with no published analog.**
        It quantifies how steeply (and in which direction) a static
        expression field changes across a compartment boundary. It involves
        no transport model, no time, and no ligand-receptor structure — for
        directional cell-cell *signalling* inference see COMMOT (Cang et al.
        2023, Nat. Methods), which is a different, optimal-transport-based
        method. Interfaces are also exactly where a local *linear* MLS fit
        is weakest (a step is not a plane), so treat magnitudes as relative
        comparisons between pairs within one dataset, not physical fluxes.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    groupby
        ``obs`` column with cell-type / compartment labels.
    keys
        Feature keys whose gradients to integrate.  Must have been computed
        by :func:`~astrolabe.tl.expression_gradient`.
    spatial_key
        ``obsm`` key for µm coordinates.
    max_pair_dist_um
        Facets whose two generating centroids are farther apart than this
        are skipped — Voronoi neighbors across tissue voids share a facet
        that does not correspond to any physical cell-cell interface.
    exclude_unreliable
        Skip facets touching a cell flagged ``obs['gradient_unreliable']``
        (tissue-boundary cells with poorly conditioned gradients). Default
        ``True``.
    gradient_key_added
        ``key_added`` used in the :func:`~astrolabe.tl.expression_gradient`
        call this reads from, if it was given a custom one. ``None``
        (default) reads ``obsm['gradient_{key}']``.
    key_added
        Key in ``uns`` for results.

    Writes
    ------
    ``uns[key_added]``
        Dict with:

        - ``integrated_flux`` — ``{key: DataFrame(groups × groups)}``;
          ``df.loc[A, B]`` = Φ(A→B) (exactly antisymmetric; NaN where no
          interface exists). Units: feature scale.
        - ``mean_normal_derivative`` — ``{key: DataFrame}``; Φ / interface
          length (per µm).
        - ``interface_length`` — DataFrame (symmetric), µm of shared Voronoi
          facet per pair.
        - ``params``.

        Mirrored to ``mdata.uns[key_added][<modality>]`` when called on a
        MuData.
    """
    from scipy.spatial import Voronoi

    _check_single_library(adata)

    if isinstance(keys, str):
        keys = [keys]

    if groupby not in adata.obs.columns:
        raise KeyError(f"obs['{groupby}'] not found.")

    labels = adata.obs[groupby].astype("category")
    codes = labels.cat.codes.to_numpy()
    cats = labels.cat.categories.tolist()
    n_groups = len(cats)

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)
    vor = Voronoi(coords)

    rp = vor.ridge_points  # (n_ridges, 2) generating point indices
    rv = np.asarray(vor.ridge_vertices)  # (n_ridges, 2) vertex indices, -1 = infinite
    finite = (rv >= 0).all(axis=1)

    i_pts = rp[:, 0]
    j_pts = rp[:, 1]

    # Facet filters: finite facet, both cells labeled (NaN labels get code -1
    # from pandas, and np.add.at would silently wrap a -1 index to the last
    # category), different labels, physically adjacent generating cells, and
    # optionally both gradients reliable.
    keep = finite
    keep &= (codes[i_pts] >= 0) & (codes[j_pts] >= 0)
    keep &= codes[i_pts] != codes[j_pts]
    pair_d = np.linalg.norm(coords[j_pts] - coords[i_pts], axis=1)
    keep &= (pair_d <= max_pair_dist_um) & (pair_d > 1e-8)
    if exclude_unreliable and "gradient_unreliable" in adata.obs.columns:
        unrel = adata.obs["gradient_unreliable"].to_numpy(dtype=bool)
        keep &= ~unrel[i_pts] & ~unrel[j_pts]

    i_k = i_pts[keep]
    j_k = j_pts[keep]
    rv_k = rv[keep]

    # Clip each facet segment to the disc of radius = pair distance around
    # the generating pair's midpoint. Hull-adjacent Voronoi facets extend far
    # outside the tissue (their closing vertices can sit hundreds of µm away),
    # so the raw segment length wildly overstates the physical contact — and
    # second-neighbor pairs at the hull can share a facet that lies entirely
    # outside the tissue, which the clip removes completely. For two adjacent
    # cells with spacing d, the true shared face lies within ~d of their
    # midpoint, so the clip leaves genuine interior facets untouched.
    v0 = vor.vertices[rv_k[:, 0]]
    v1 = vor.vertices[rv_k[:, 1]]
    mid = 0.5 * (coords[i_k] + coords[j_k])
    R = pair_d[keep]
    seg = v1 - v0
    seg_len = np.linalg.norm(seg, axis=1)
    seg_len_safe = np.where(seg_len > 1e-12, seg_len, 1.0)
    d_hat = seg / seg_len_safe[:, None]
    # segment point p(t) = v0 + t·d_hat, t ∈ [0, seg_len]; solve |p − mid| = R
    rel = v0 - mid
    b = np.sum(rel * d_hat, axis=1)
    c = np.sum(rel * rel, axis=1) - R**2
    disc = b**2 - c
    has_int = disc >= 0
    sqrt_disc = np.sqrt(np.where(has_int, disc, 0.0))
    t0 = np.clip(-b - sqrt_disc, 0.0, seg_len)
    t1 = np.clip(-b + sqrt_disc, 0.0, seg_len)
    facet_len = np.where(has_int, t1 - t0, 0.0)

    positive = facet_len > 1e-9
    i_k, j_k, facet_len = i_k[positive], j_k[positive], facet_len[positive]
    pair_d_kept = pair_d[keep][positive]
    n_hat = (coords[j_k] - coords[i_k]) / pair_d_kept[:, None]  # i→j unit normal

    ci = codes[i_k]
    cj = codes[j_k]

    # Interface length matrix (symmetric). Accumulate one triangle then
    # symmetrize by transpose-add: two separate np.add.at passes interleave
    # the float additions in different orders for [A, B] vs [B, A], leaving
    # the matrix symmetric only to ~1e-13; x + y == y + x is exact.
    length_mat = np.zeros((n_groups, n_groups), dtype=np.float64)
    np.add.at(length_mat, (ci, cj), facet_len)
    length_mat = length_mat + length_mat.T

    flux_res: dict[str, pd.DataFrame] = {}
    deriv_res: dict[str, pd.DataFrame] = {}
    for key in keys:
        grad_key = f"{gradient_key_added}_{key}" if gradient_key_added else f"gradient_{key}"
        if grad_key not in adata.obsm:
            key_added_hint = f", key_added='{gradient_key_added}'" if gradient_key_added else ""
            raise KeyError(
                f"obsm['{grad_key}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']{key_added_hint}) first."
            )
        grad = adata.obsm[grad_key].astype(np.float64)

        dfdn = np.sum(0.5 * (grad[i_k] + grad[j_k]) * n_hat, axis=1)
        contrib = facet_len * dfdn

        # Accumulate one triangle, then antisymmetrize by transpose-subtract.
        # Two separate np.add.at passes (one per triangle) interleave the
        # float additions in different orders, so Φ(B→A) landed within ~1e-16
        # of −Φ(A→B) but not bit-equal to it; round(x − y) == −round(y − x)
        # in IEEE arithmetic, so this form makes the documented identity
        # Φ(B→A) = −Φ(A→B) hold exactly.
        flux = np.zeros((n_groups, n_groups), dtype=np.float64)
        np.add.at(flux, (ci, cj), contrib)
        flux = flux - flux.T  # exact antisymmetry, by identity

        no_interface = length_mat == 0
        flux_masked = np.where(no_interface, np.nan, flux)
        with np.errstate(invalid="ignore"):
            deriv = np.where(no_interface, np.nan, flux / np.where(no_interface, 1.0, length_mat))

        df_flux = pd.DataFrame(flux_masked, index=cats, columns=cats)
        df_flux.index.name = "source"
        df_flux.columns.name = "target"
        flux_res[key] = df_flux
        df_deriv = pd.DataFrame(deriv, index=cats, columns=cats)
        df_deriv.index.name = "source"
        df_deriv.columns.name = "target"
        deriv_res[key] = df_deriv

    length_df = pd.DataFrame(
        np.where(length_mat == 0, np.nan, length_mat), index=cats, columns=cats
    )

    adata.uns[key_added] = {
        "integrated_flux": flux_res,
        "mean_normal_derivative": deriv_res,
        "interface_length": length_df,
        "params": {
            "groupby": groupby,
            "keys": keys,
            "spatial_key": spatial_key,
            "max_pair_dist_um": max_pair_dist_um,
            "exclude_unreliable": exclude_unreliable,
        },
    }
    log.info(
        "interface_flux: %d interface facets (%.0f µm total) across %d groups "
        "for %d features",
        len(facet_len),
        float(facet_len.sum()),
        n_groups,
        len(keys),
    )


@with_adata
def bearing_to_group(
    adata: ad.AnnData,
    groupby: str,
    group: str,
    *,
    spatial_key: str = "spatial",
    key_added: str | None = None,
) -> None:
    """Per-cell bearing toward the **nearest** cell of a landmark group.

    Directions in a tissue image are only meaningful relative to tissue
    landmarks (the section could have been mounted at any rotation). This
    writes, for every cell, the compass bearing (radians, arctan2 convention)
    from the cell toward the *nearest* member of ``obs[groupby] == group`` —
    nearest, not the group centroid, so a landmark split across multiple
    regions (e.g. several tumor nests) is handled correctly: each cell is
    referenced to its own closest nest.

    Use the resulting column as
    :func:`~astrolabe.pl.direction_histogram`'s ``relative_to=`` to ask "do
    gradients point toward (0°) or away from (180°) the landmark?", or
    subtract it from any stored gradient direction for custom analyses.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    groupby
        ``obs`` column holding the group labels.
    group
        The landmark label within ``obs[groupby]``.
    spatial_key
        ``obsm`` key for µm coordinates.
    key_added
        Output ``obs`` column. Default ``f'bearing_to_{group}'``.

    Writes
    ------
    ``obs[key_added]``
        float32 bearing in radians toward the nearest group member; ``NaN``
        for cells that are themselves in the group (their nearest member is
        themselves — the bearing is undefined).
    """
    from scipy.spatial import cKDTree

    _check_single_library(adata)

    if groupby not in adata.obs:
        raise KeyError(f"obs['{groupby}'] not found")
    labels = adata.obs[groupby].astype(str).to_numpy()
    member = labels == str(group)
    if not member.any():
        raise ValueError(
            f"no cells with obs['{groupby}'] == {group!r} "
            f"(available: {sorted(set(labels))[:20]})"
        )

    coords = np.asarray(adata.obsm[spatial_key][:, :2], dtype=np.float64)
    tree = cKDTree(coords[member])
    _, nearest = tree.query(coords[~member], k=1)
    member_coords = coords[member]

    bearing = np.full(adata.n_obs, np.nan, dtype=np.float32)
    delta = member_coords[nearest] - coords[~member]
    bearing[~member] = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32)

    col = key_added or f"bearing_to_{group}"
    adata.obs[col] = bearing
    log.info(
        "bearing_to_group: obs['%s'] — bearing toward nearest of %d '%s' cells "
        "(NaN for the %d member cells)",
        col, int(member.sum()), group, int(member.sum()),
    )


@with_adata
def diffuse_expression(
    adata: ad.AnnData,
    keys: str | list[str],
    *,
    n_steps: int = 3,
    layer: str | None = None,
    neighbors_key: str = "spatial_neighbors",
    normalize: Literal["coifman_lafon", "row"] = "coifman_lafon",
    groupby: str | None = None,
    groups: list[str] | None = None,
    suffix: str = "_diffused",
) -> None:
    """Smooth per-cell values by graph diffusion on the spatial neighbor graph.

    The intuition is a drop of dye: each step replaces every cell's value
    with a weighted average over its spatial neighbors, so single-cell noise
    (dropout zeros, segmentation errors) washes out while tissue-scale
    structure — which spans many neighborhoods — survives. This is the same
    idea MAGIC uses on expression graphs (van Dijk et al. 2018, Cell).

    **Choosing** ``n_steps``: each step spreads information roughly one
    neighbor ring (~one cell diameter on a Delaunay/KNN graph), so ``n_steps``
    is approximately the smoothing radius in cell diameters. Small values
    (1–3) denoise; large values start erasing real biology — structures
    smaller than ~``n_steps`` cell diameters blur away first. When in doubt,
    plot the raw and smoothed values side by side and stop increasing
    ``n_steps`` as soon as the structure you care about starts to fade
    (MAGIC's own guidance: pick the smallest ``t`` after which the picture
    stops changing). :func:`~astrolabe.pl.diffusion_comparison` renders
    exactly this comparison — raw plus any set of step counts on one shared
    color scale — without writing anything to the object.

    ``normalize='coifman_lafon'`` (default) applies the α = 1 density
    correction of Coifman & Lafon (2006) before row-normalizing: without it,
    dye pools where *cells* are dense (a dense tumor core acts as an
    attractor), not where expression is high — the correction makes the
    smoothing follow tissue geometry instead of sampling density, matching
    :func:`~astrolabe.tl.spatial_diffusion`'s operator exactly.

    Related functions: :func:`~astrolabe.tl.spatial_lag` is a single plain
    row-normalized step (no density correction);
    :func:`~astrolabe.tl.expression_gradient`'s internal
    ``n_diffusion_steps`` pre-smoothing deliberately keeps the plain
    row-stochastic operator (changing it would shift every stored gradient);
    :func:`~astrolabe.tl.spatial_diffusion` exposes the operator's
    eigenmodes and the Sepal diffusion score.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    keys
        Feature keys — gene names or numeric ``obs`` columns.
    n_steps
        Number of diffusion steps (≈ smoothing radius in cell diameters).
        ``0`` writes an unsmoothed copy.
    layer
        Expression layer to read gene values from (``None`` uses ``X``).
    neighbors_key
        Spatial graph key from :func:`~astrolabe.tl.spatial_neighbors`.
    normalize
        ``'coifman_lafon'`` (default) — density-corrected operator;
        ``'row'`` — plain row normalization (biased toward dense regions).
    groupby
        ``obs`` column restricting diffusion to **within-group edges**, so
        values never bleed across a cell-type or region boundary. Without it,
        smoothing a lineage-restricted feature across an interface manufactures
        an intermediate zone no cell actually occupies (measured on planted
        data: 6 diffusion steps cut the boundary contrast by more than half,
        while the within-group operator preserves it).
    groups
        Restrict to these *groupby* values. Requires *groupby*.
    suffix
        Output column suffix: values land in ``obs[f'{key}{suffix}']``.

    Writes
    ------
    ``obs[f'{key}{suffix}']``
        float32 smoothed values per key.

    Examples
    --------
    >>> ab.tl.spatial_neighbors(mdata, coord_type='delaunay')
    >>> ab.tl.diffuse_expression(mdata, 'EPCAM', n_steps=5)
    >>> ab.pl.points(mdata, color='EPCAM_diffused').show()
    """
    _check_single_library(adata)

    if isinstance(keys, str):
        keys = [keys]

    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    conn_d = adata.obsp[conn_key]
    labels, levels = _resolve_groups(adata, groupby, groups)
    if groupby is not None:
        conn_d, n_dropped = _restrict_conn_union(conn_d, labels, levels)
        log.info(
            "diffuse_expression: smoothing within '%s' only (%d group(s)); "
            "%d cross-group edge(s) dropped",
            groupby, len(levels), n_dropped,
        )
    M, _A_hat = _diffusion_operator(conn_d, normalize)

    for key in keys:
        f = np.asarray(_extract_values(adata, key, layer), dtype=np.float64)
        for _ in range(n_steps):
            f = M @ f
        adata.obs[f"{key}{suffix}"] = f.astype(np.float32)
        log.info(
            "diffuse_expression: '%s' → obs['%s%s'] (%d steps, %s operator)",
            key, key, suffix, n_steps, normalize,
        )


@with_adata(mirror_uns=from_kwarg("key_added"))
def spatial_diffusion(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    neighbors_key: str = "spatial_neighbors",
    normalize: Literal["coifman_lafon", "row"] = "coifman_lafon",
    compute_eigenmodes: bool = False,
    n_eigenmodes: int = 10,
    diffusion_score: bool = False,
    max_diffusion_steps: int = 500,
    diffusion_convergence_tol: float = 0.2,
    groupby: str | None = None,
    groups: list[str] | None = None,
    key_added: str = "diffusion",
) -> None:
    """Graph heat-equation diffusion on the spatial cell graph.

    Approximates the solution to the heat equation on the cell graph:

    .. math::

        \\frac{\\partial f}{\\partial t} = -\\mathbf{L}f,
        \\quad f(t) = \\exp(-t\\mathbf{L})\\,f_0

    by iterative multiplication with the diffusion operator M:

    .. math::

        f(t) \\approx M^n\\,f_0

    Uses the **Coifman–Lafon density normalization** (α = 1) which
    separates the geometry of the tissue from cell-density variation:

    .. math::

        \\hat{K}_{ij} = \\frac{K_{ij}}{q_i q_j},
        \\quad q_i = \\sum_j K_{ij}

    Without this normalization, naive row-normalization causes diffusion
    to track high-density tumor cores rather than expression geometry. α = 1
    (full `q_i q_j` in the denominator) gives the strongest density-invariance
    guarantee in Coifman & Lafon's general α-family (`K / (q_i^α q_j^α)`,
    α ∈ [0, 1]) — the intermediate α = 1/2 member (`K / sqrt(q_i q_j)`) only
    partially corrects for density and leaves non-uniform sampling density
    measurably entangled with the diffusion geometry, which is exactly the
    failure mode this normalization exists to avoid.  (Note this α-family is
    distinct from the symmetric normalized graph Laplacian's
    ``D^{-1/2} A D^{-1/2}``, which normalizes by *degree after* kernel
    construction rather than correcting the kernel for sampling density.)

    **References:** Coifman & Lafon 2006 (Appl. Comput. Harmon. Anal.) —
    diffusion maps and density normalization; Andersson & Lundeberg 2021
    (Bioinformatics, Sepal) — diffusion-simulation SVG ranking on ST data;
    Haghverdi et al. 2016 (Nature Methods, DPT) — exp(−tL) on single-cell
    data.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    spatial_key
        ``obsm`` key for µm coordinates.
    neighbors_key
        Spatial graph key.
    normalize
        Normalization strategy for the graph Laplacian:

        - ``'coifman_lafon'`` (default) — density-normalized diffusion maps
          operator.  Recommended for non-uniform cell densities.
        - ``'row'`` — plain row-normalization.  Faster but biased by density.
    groupby
        ``obs`` column restricting diffusion to **within-group edges**, so
        values never bleed across a cell-type or region boundary. Without it,
        smoothing a lineage-restricted feature across an interface manufactures
        an intermediate zone no cell actually occupies (measured on planted
        data: 6 diffusion steps cut the boundary contrast by more than half,
        while the within-group operator preserves it).
    groups
        Restrict to these *groupby* values. Requires *groupby*.
    compute_eigenmodes
        If ``True``, also compute the ``n_eigenmodes`` leading eigenvectors of
        the symmetric normalized Laplacian and store them in
        ``obsm['X_diffusion']``.  These form a spatial Fourier basis.
    n_eigenmodes
        Number of eigenmodes to compute when ``compute_eigenmodes=True``.
    diffusion_score
        If ``True``, compute a literal Sepal-style (Andersson & Lundeberg
        2021) **time-to-uniformity** score for every gene in ``X`` and
        store in ``var['diffusion_score']``: starting from each gene's
        observed expression, repeatedly apply the diffusion operator
        ``f ← M @ f`` and report the number of steps until the pattern's
        mean absolute deviation from the graph-wide mean, relative to
        where it started, drops below ``diffusion_convergence_tol``. Genes
        that never converge within ``max_diffusion_steps`` are
        right-censored at that value. Higher scores mean a spatial pattern
        that survives diffusion longer — i.e. more persistent / spatially
        coherent.
    max_diffusion_steps
        Upper bound on diffusion steps per gene when ``diffusion_score=True``.
        Genes that haven't converged by this step are censored at this
        value rather than run indefinitely. Has no effect otherwise.
    diffusion_convergence_tol
        Relative mean-absolute-deviation threshold (from the graph-wide
        mean) below which a gene's pattern is considered to have reached
        uniformity, when ``diffusion_score=True``. Has no effect otherwise.
    key_added
        Key in ``uns`` for metadata.

    Writes
    ------
    ``uns[key_added]``
        Dict with ``params`` and optionally ``eigenvalues``. Mirrored to
        ``mdata.uns[key_added][<modality>]`` when called on a MuData.
    ``obsm['X_diffusion']``
        ``(n_cells, n_eigenmodes)`` float64 — spatial Fourier modes
        (only when ``compute_eigenmodes=True``; adata-resident only).
    ``var['diffusion_score']``
        float64 — Sepal-style steps-to-uniformity persistence score per
        gene, right-censored at ``max_diffusion_steps``
        (only when ``diffusion_score=True``; adata-resident only).
    """
    from scipy.sparse.linalg import eigsh

    _check_single_library(adata)

    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    conn_s = adata.obsp[conn_key]
    labels, levels = _resolve_groups(adata, groupby, groups)
    if groupby is not None:
        conn_s, n_dropped = _restrict_conn_union(conn_s, labels, levels)
        log.info(
            "spatial_diffusion: diffusing within '%s' only (%d group(s)); "
            "%d cross-group edge(s) dropped",
            groupby, len(levels), n_dropped,
        )
    M, A_hat = _diffusion_operator(conn_s, normalize)

    uns_entry: dict = {
        "params": {
            "spatial_key": spatial_key,
            "neighbors_key": neighbors_key,
            "normalize": normalize,
        }
    }

    if compute_eigenmodes:
        # Symmetric Laplacian L_sym = I − (D_hat^{-1/2} A_hat D_hat^{-1/2})
        row_sums_hat = np.asarray(A_hat.sum(axis=1), dtype=np.float64).ravel()
        row_sums_hat[row_sums_hat == 0] = 1.0
        D_hat_inv_sqrt = sp.diags(1.0 / np.sqrt(row_sums_hat))
        L_sym = sp.eye(A_hat.shape[0]) - D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

        k = min(n_eigenmodes, A_hat.shape[0] - 2)
        log.info("spatial_diffusion: computing %d eigenmodes …", k)
        # Shift-invert around 0 rather than which="SM": ARPACK's smallest-
        # magnitude mode without shift-invert is notoriously slow and often
        # fails to converge on large sparse Laplacians (the smallest
        # eigenvalues are clustered near 0).
        eigenvalues, eigenvectors = eigsh(L_sym.tocsc(), k=k, sigma=0, which="LM")
        sort_idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sort_idx].astype(np.float64)
        adata.obsm["X_diffusion"] = eigenvectors
        uns_entry["eigenvalues"] = eigenvalues[sort_idx].tolist()

    if diffusion_score:
        log.info(
            "spatial_diffusion: computing per-gene steps-to-uniformity scores "
            "(max_diffusion_steps=%d, tol=%.3f) …",
            max_diffusion_steps, diffusion_convergence_tol,
        )
        mat = adata.X
        if sp.issparse(mat):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float64)
        n_genes = mat.shape[1]

        # Literal Sepal (Andersson & Lundeberg 2021) time-to-uniformity
        # definition: iterate f <- M @ f per gene until its mean absolute
        # deviation from the graph-wide mean, relative to where it started,
        # drops below `diffusion_convergence_tol`; report the step count.
        # Replaces an earlier fixed-step variance-retained-ratio score
        # (`final_var / initial_var` after a fixed n_steps) -- that score
        # and this one both increase with persistence, but this is the
        # reference metric itself rather than a fixed-horizon proxy for it.
        zero_var = mat.var(axis=0) == 0  # equivalently: mean-abs-deviation == 0
        target = mat.mean(axis=0)
        scale = np.abs(mat - target).mean(axis=0)

        scores = np.full(n_genes, float(max_diffusion_steps), dtype=np.float64)

        # Batched across all still-unconverged genes at once; only the
        # global loop (not each gene individually) exits early once every
        # gene has converged. zero_var genes are excluded here (nothing to
        # diffuse) and scored NaN below instead.
        remaining = ~zero_var
        f_t = mat.copy()
        for t in range(1, max_diffusion_steps + 1):
            if not remaining.any():
                break
            f_t = M @ f_t
            rel_dev = np.abs(f_t - target).mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_dev = np.where(scale > 0, rel_dev / scale, 0.0)
            newly_converged = remaining & (rel_dev < diffusion_convergence_tol)
            scores[newly_converged] = float(t)
            remaining &= ~newly_converged

        n_censored = int((remaining).sum())
        if n_censored:
            log.warning(
                "spatial_diffusion: %d/%d genes did not converge within "
                "max_diffusion_steps=%d; their scores are right-censored "
                "at that value.",
                n_censored, n_genes, max_diffusion_steps,
            )

        # A zero-variance (undetected/constant) gene has no real signal to
        # diffuse at all -- excluded via NaN rather than left to read as a
        # real (and, under this metric, minimum-possible) persistence score.
        scores[zero_var] = np.nan
        adata.var["diffusion_score"] = scores.astype(np.float64)
        uns_entry["params"]["max_diffusion_steps"] = max_diffusion_steps
        uns_entry["params"]["diffusion_convergence_tol"] = diffusion_convergence_tol

    adata.uns[key_added] = uns_entry
    log.info(
        "spatial_diffusion: %d cells, normalize='%s'",
        adata.n_obs,
        normalize,
    )


@with_adata(mirror_uns=_decay_length_uns_key)
def decay_length(
    adata: ad.AnnData,
    key: str,
    source: tuple[float, float] | int | str,
    *,
    spatial_key: str = "spatial",
    source_groupby: str | None = None,
    max_dist_um: float = 500.0,
    n_bins: int = 30,
    layer: str | None = None,
    key_added: str | None = None,
) -> None:
    """Fit a radial exponential decay length around a localised source.

    Fits the phenomenological model ``f(r) = f₀ · e^(−r/λ) + c`` where:

    - ``r`` is the distance from *source* in µm
    - ``λ`` is the **decay length** in µm (the quantity of biological interest)
    - ``f₀`` is the source-associated amplitude above background
    - ``c`` is the **background expression floor** — real (log-normalized)
      expression rarely decays to exactly zero, and a model without this
      offset can only absorb a nonzero floor by driving λ toward infinity,
      producing absurd fitted decay lengths far beyond the data window.

    Bins are weighted by ``1/√(cell count)`` in the fit (an outer annulus
    averaging thousands of cells constrains the curve more than a 3-cell
    inner bin), and parameter standard errors from the fit covariance are
    reported — a λ whose standard error rivals λ itself is unidentifiable
    (typically: the true decay is slower than the fitting window can
    resolve), and the function warns in that case as well as when the
    fitted λ exceeds ``max_dist_um``.

    .. warning::

        ``λ`` is a **phenomenological decay length**, not a physical diffusion
        constant D.  It is only interpretable if the observed spatial gradient
        approximately follows radial decay from a localised source — an
        assumption best validated visually before interpreting λ.  The
        reported ``r_squared`` is computed on the *binned means*, which
        removes most cell-level variance — read it as "how exponential is
        the radial profile", not "how much expression variance is explained".
        Gregor et al. 2007 (*Cell*) and Wartlick, Kicheva & González-Gaitán
        2009 (*Cold Spring Harb. Perspect. Biol.*) frame morphogen-gradient
        decay lengths in the appropriate developmental context.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    key
        Gene name or ``obs`` column to fit.
    source
        The localised expression source.  One of:

        - ``(float, float)`` — µm coordinate ``(x, y)``
        - ``int`` — cell index (uses that cell's centroid)
        - ``str`` — ``obs[source_groupby]`` label; uses the group centroid
    spatial_key
        ``obsm`` key for µm coordinates.
    source_groupby
        ``obs`` column to look up when *source* is a ``str``.
    max_dist_um
        Maximum radial distance for the fit.
    n_bins
        Number of distance bins.
    layer
        Expression layer.
    key_added
        Key in ``uns`` for results.  Default: ``'decay_length_{key}'``.

    Writes
    ------
    ``uns[key_added or f'decay_length_{key}']``
        Dict with ``lambda_um``, ``lambda_se_um``, ``f0``, ``background``,
        ``r_squared`` (binned-level; see warning), ``bin_centers``,
        ``bin_means``, ``bin_counts``, ``params``. Mirrored to
        ``mdata.uns[<that key>][<modality>]`` when called on a MuData.
    """
    from scipy.optimize import curve_fit

    _check_single_library(adata)

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)

    # Resolve source coordinate
    if isinstance(source, (tuple, list, np.ndarray)) and len(source) == 2:
        src_coord = np.array(source, dtype=np.float64)
    elif isinstance(source, (int, np.integer)):
        src_coord = coords[int(source)]
    elif isinstance(source, str):
        if source_groupby is None:
            raise ValueError(
                "source_groupby must be specified when source is a str label."
            )
        if source_groupby not in adata.obs.columns:
            raise KeyError(f"obs['{source_groupby}'] not found.")
        mask = adata.obs[source_groupby].astype(str) == source
        if not mask.any():
            raise ValueError(f"No cells found with {source_groupby}='{source}'.")
        src_coord = coords[mask].mean(axis=0)
        # A ring/crescent/multi-nest group's centroid can land outside the
        # group itself (in stroma) — the radial-decay model is then fit
        # around a point that is not a source of anything.
        nearest_member = float(
            np.min(np.linalg.norm(coords[mask] - src_coord, axis=1))
        )
        group_span = float(np.ptp(coords[mask], axis=0).max())
        if group_span > 0 and nearest_member > 0.1 * group_span:
            warnings.warn(
                f"decay_length: the centroid of {source_groupby}='{source}' lies "
                f"{nearest_member:.0f} µm from the nearest member cell — the group "
                "is likely ring-shaped or multi-nested, and its centroid is not a "
                "meaningful radial source. Pass an explicit (x, y) coordinate.",
                stacklevel=3,
            )
    else:
        raise TypeError(
            f"source must be a (x, y) tuple, cell index, or obs label; "
            f"got {type(source).__name__}"
        )

    dists = np.linalg.norm(coords - src_coord, axis=1)
    keep = dists <= max_dist_um
    f = _extract_values(adata, key, layer)[keep]
    r = dists[keep]

    # Bin by distance
    bin_edges = np.linspace(0, max_dist_um, n_bins + 1)
    bin_idx = np.digitize(r, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_means = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(bin_means, bin_idx, f)
    np.add.at(bin_counts, bin_idx, 1)
    with np.errstate(invalid="ignore"):
        bin_means = np.where(bin_counts > 0, bin_means / bin_counts, np.nan)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Fit exponential-plus-background to non-NaN bins
    valid = ~np.isnan(bin_means) & (bin_counts >= 3)
    if valid.sum() < 5:
        warnings.warn(
            f"decay_length: insufficient bins with ≥3 cells to fit '{key}'. "
            "Try increasing max_dist_um or n_bins.",
            stacklevel=3,
        )
        lambda_um, lambda_se, f0, background, r_sq = (np.nan,) * 5
    else:
        def _exp_decay(r, f0, lam, c):
            return f0 * np.exp(-r / lam) + c

        try:
            y = bin_means[valid]
            c0 = float(np.nanmin(y))
            p0 = [max(float(np.nanmax(y)) - c0, 1e-6), float(max_dist_um / 3), c0]
            # sigma ∝ 1/sqrt(count): an annulus averaging n cells has SE of
            # its mean ∝ 1/sqrt(n) — otherwise a 3-cell inner bin carries the
            # same weight as a 20 000-cell outer one.
            sigma = 1.0 / np.sqrt(bin_counts[valid].astype(np.float64))
            popt, pcov = curve_fit(
                _exp_decay,
                bin_centers[valid],
                y,
                p0=p0,
                sigma=sigma,
                bounds=([0, 1e-3, 0], [np.inf, max_dist_um * 10, np.inf]),
                maxfev=10000,
            )
            f0, lambda_um, background = (float(v) for v in popt)
            lambda_se = float(np.sqrt(pcov[1, 1])) if np.all(np.isfinite(pcov)) else np.nan
            fitted = _exp_decay(bin_centers[valid], *popt)
            ss_res = np.sum((y - fitted) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_sq = float(1 - ss_res / (ss_tot + 1e-12))

            if lambda_um > max_dist_um:
                warnings.warn(
                    f"decay_length: fitted λ={lambda_um:.0f} µm exceeds the "
                    f"{max_dist_um:.0f} µm fitting window — the decay is not "
                    "resolvable from these data (λ is unidentifiable, only a "
                    "lower bound). Do not interpret its value.",
                    stacklevel=3,
                )
            elif np.isfinite(lambda_se) and lambda_se > 0.5 * lambda_um:
                warnings.warn(
                    f"decay_length: λ={lambda_um:.0f} µm has standard error "
                    f"{lambda_se:.0f} µm (>50% of the estimate) — poorly "
                    "identified; treat with caution.",
                    stacklevel=3,
                )
        except RuntimeError:
            warnings.warn(
                f"decay_length: exponential fit did not converge for '{key}'.",
                stacklevel=3,
            )
            lambda_um, lambda_se, f0, background, r_sq = (np.nan,) * 5

    out_key = key_added or f"decay_length_{key}"
    adata.uns[out_key] = {
        "lambda_um": lambda_um,
        "lambda_se_um": lambda_se,
        "f0": f0,
        "background": background,
        "r_squared": r_sq,
        "bin_centers": bin_centers,
        "bin_means": bin_means,
        "bin_counts": bin_counts,
        "params": {
            "key": key,
            "source": list(src_coord),
            "max_dist_um": max_dist_um,
            "n_bins": n_bins,
        },
    }
    log.info(
        "decay_length: '%s' λ=%.1f µm, f₀=%.3f, R²=%.3f",
        key,
        lambda_um,
        f0,
        r_sq,
    )
