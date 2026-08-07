"""Gene expression gradient and vector field analysis.

This module implements:

- **expression_gradient** — per-cell gradient vectors via graph-based Moving
  Least Squares (MLS).  Fits a local linear plane to each cell's spatial
  neighborhood, giving a (gx, gy) gradient vector in µm⁻¹ units.
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

- **compartment_flux** — for each cross-boundary edge in the spatial graph,
  projects the per-cell gradient onto the A→B unit direction and aggregates
  by ordered label pair.  Conceptually related to COMMOT (Cang et al. 2023,
  Nature Methods) which models directional signalling flux via OT.

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

from astrolabe._utils import _spawn_rngs, from_kwarg, with_adata
from astrolabe.tl._spatial_stats import _extract_values, _get_conn_matrix

__all__ = [
    "expression_gradient",
    "gradient_field",
    "gradient_significance",
    "compartment_flux",
    "spatial_diffusion",
    "decay_length",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _diffuse_expression(
    f: np.ndarray,
    conn_mat: sp.spmatrix,
    n_steps: int,
) -> np.ndarray:
    """Row-normalize conn_mat then apply f ← M^n f (in-place on a copy)."""
    if n_steps == 0:
        return f.copy()
    row_sums = np.asarray(conn_mat.sum(axis=1), dtype=np.float64).ravel()
    row_sums[row_sums == 0] = 1.0
    M = sp.diags(1.0 / row_sums) @ conn_mat.astype(np.float64)
    f = np.asarray(f, dtype=np.float64)
    for _ in range(n_steps):
        f = M @ f
    return f


def _edge_weights(
    adata: ad.AnnData,
    conn_key: str,
    dist_key: str,
    weight_by_distance: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (src, dst, weights) from the spatial graph COO representation."""
    conn_coo = adata.obsp[conn_key].tocoo()
    src = conn_coo.row.astype(np.intp)
    dst = conn_coo.col.astype(np.intp)

    if weight_by_distance and dist_key in adata.obsp:
        dist_csr = adata.obsp[dist_key].tocsr()
        # Element-wise fancy indexing: (n_edges,) via matrix broadcast
        dists = np.asarray(dist_csr[src, dst], dtype=np.float64).ravel()
        w = 1.0 / (dists + 1e-8)
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
    """Vectorised Moving Least Squares gradient per cell.

    Solves the weighted local linear system:

        A_i g_i = b_i

    where A_i = Σ_j w_ij (Δx_ij ⊗ Δx_ij)  (2×2 structure tensor)
    and   b_i = Σ_j w_ij  Δx_ij  Δf_ij     (2-vector RHS)

    Using np.linalg.solve on batched (n_cells, 2, 2) arrays — no Python
    loop over cells.  A ridge of 1e-8 × I guards against singular structure
    tensors at boundary cells with truncated/collinear neighborhoods.

    Reference: Lancaster & Salkauskas 1981 (Math. Comput.);
    Brunsdon, Fotheringham & Charlton 1996 (Geogr. Anal.)
    """
    n_cells = coords.shape[0]
    A = np.zeros((n_cells, 2, 2), dtype=np.float64)
    b = np.zeros((n_cells, 2), dtype=np.float64)

    dx = coords[dst] - coords[src]       # (n_edges, 2)  µm displacement
    df = f[dst] - f[src]                 # (n_edges,)    expression diff

    # Accumulate structure tensor: A[src] += w_ij * outer(dx_ij, dx_ij)
    wdx = w[:, None] * dx               # (n_edges, 2)
    np.add.at(A[:, 0, 0], src, wdx[:, 0] * dx[:, 0])
    np.add.at(A[:, 0, 1], src, wdx[:, 0] * dx[:, 1])
    np.add.at(A[:, 1, 0], src, wdx[:, 1] * dx[:, 0])
    np.add.at(A[:, 1, 1], src, wdx[:, 1] * dx[:, 1])

    # Accumulate RHS: b[src] += w_ij * dx_ij * df_ij
    wdf = (w * df)[:, None] * dx        # (n_edges, 2)
    np.add.at(b[:, 0], src, wdf[:, 0])
    np.add.at(b[:, 1], src, wdf[:, 1])

    # Tikhonov regularization: prevents singular structure tensors at tissue
    # boundaries where neighbors are collinear or too few.
    A[:, 0, 0] += 1e-8
    A[:, 1, 1] += 1e-8

    # Batched 2×2 solve.  numpy's gufunc signature is (m,m),(m,n)->(m,n), so
    # b must have 2D core dims: reshape to (n_cells, 2, 1) and squeeze after.
    gradients = np.linalg.solve(A, b[:, :, None])[:, :, 0]   # (n_cells, 2)
    return gradients.astype(np.float32)


def _mean_resultant_length(angles: np.ndarray) -> float:
    """Circular mean resultant length R̄ ∈ [0, 1] from an array of angles (rad).

    R̄ = 1 indicates perfectly aligned directions; R̄ ≈ 0 indicates uniform.
    Mardia & Jupp 2000 (Directional Statistics, Wiley), eq. 2.2.
    """
    return float(np.abs(np.mean(np.exp(1j * angles))))


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
    key_added: str | None = None,
) -> None:
    """Compute per-cell gene expression gradient vectors via graph-based MLS.

    For each cell *i*, fits a local linear plane to expression values in
    its spatial neighborhood using weighted least squares (Moving Least
    Squares / diffuse derivative estimator):

    .. math::

        \\hat{\\mathbf{g}}_i =
        \\left(\\sum_{j} w_{ij}\\,\\Delta\\mathbf{x}_{ij}
        \\Delta\\mathbf{x}_{ij}^\\top\\right)^{-1}
        \\left(\\sum_{j} w_{ij}\\,\\Delta\\mathbf{x}_{ij}\\,\\Delta f_{ij}\\right)

    Weights are inverse-distance (from ``obsp[neighbors_key + '_distances']``)
    so nearer neighbors contribute more.  An optional graph-diffusion
    pre-smoothing step (multiply expression by the row-normalized adjacency
    matrix ``n_diffusion_steps`` times) suppresses single-gene sparsity noise
    before differentiation.

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
        If ``True`` (default), weight neighbors by inverse Euclidean distance.
        If ``False``, all neighbors receive equal weight.
    key_added
        Base name for output keys.  Default: ``'gradient_{key}'``.

    Writes
    ------
    ``obsm['gradient_{key}']``
        ``float32`` ``(n_cells, 2)`` — gradient vectors ``[gx, gy]`` in
        units of log₁p(count)-expression per µm (natural log; assumes upstream log1p normalization).
    ``obs['gradient_{key}_magnitude']``
        ``float32`` ``(n_cells,)`` — Euclidean magnitude ‖g‖.
    ``obs['gradient_{key}_direction']``
        ``float32`` ``(n_cells,)`` — direction angle ``arctan2(gy, gx)``
        in radians.
    ``uns['gradient_params']``
        Dict of parameters used for this run.
    """
    if isinstance(keys, str):
        keys = [keys]

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found. "
            "Ensure ab.tl.spatial_neighbors() has been run."
        )
    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)

    conn_key, dist_key = _get_neighbors_keys(adata, neighbors_key)
    conn_mat = adata.obsp[conn_key]
    src, dst, w = _edge_weights(adata, conn_key, dist_key, weight_by_distance)

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

        base = key_added or f"gradient_{key}"
        adata.obsm[base] = grad
        adata.obs[f"{base}_magnitude"] = np.linalg.norm(grad, axis=1).astype(np.float32)
        adata.obs[f"{base}_direction"] = np.arctan2(
            grad[:, 1], grad[:, 0]
        ).astype(np.float32)

        log.info(
            "expression_gradient: '%s' — mean magnitude %.4f, "
            "directional coherence R̄=%.3f",
            key,
            float(adata.obs[f"{base}_magnitude"].mean()),
            _mean_resultant_length(adata.obs[f"{base}_direction"].to_numpy()),
        )

    adata.uns["gradient_params"] = {
        "keys": keys,
        "spatial_key": spatial_key,
        "neighbors_key": neighbors_key,
        "n_diffusion_steps": n_diffusion_steps,
        "weight_by_distance": weight_by_distance,
    }


@with_adata(mirror_uns=_gradient_field_uns_key)
def gradient_field(
    adata: ad.AnnData,
    key: str,
    *,
    spatial_key: str = "spatial",
    n_grid: int = 50,
    bandwidth_um: float | None = None,
    mask_threshold: float = -5.0,
    key_added: str | None = None,
) -> None:
    """Interpolate per-cell gradient vectors onto a masked regular grid.

    Reads ``obsm['gradient_{key}']`` computed by
    :func:`~astrolabe.tl.expression_gradient` and interpolates U (x-component)
    and V (y-component) onto an ``n_grid × n_grid`` uniform grid using
    RBF interpolation (linear kernel, 50 nearest neighbors).

    Divergence ``∂U/∂x + ∂V/∂y`` and curl ``∂V/∂x − ∂U/∂y`` are computed
    via ``numpy.gradient`` (second-order central differences).

    A tissue-support mask is derived from the log-density of a Gaussian KDE
    fit to cell positions; grid points in tissue voids (log-density <
    *mask_threshold*) have U/V/divergence/curl set to ``NaN``.

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
        (e.g. ``'EPCAM'`` → reads ``obsm['gradient_EPCAM']``).
    spatial_key
        Key in ``obsm`` with ``(n_cells, 2)`` µm coordinates.
    n_grid
        Number of grid points along each axis.  ``50`` is a good default;
        increase for larger FOVs or finer features.
    bandwidth_um
        KDE bandwidth for the tissue mask (µm).  Defaults to Scott's rule
        applied to the coordinate range.
    mask_threshold
        Log-density threshold for the tissue mask.  Grid points below this
        value are set to ``NaN``.  Decrease (more negative) to expand the
        masked region; increase to shrink it.

    Writes ``uns[key_added or f'gradient_field_{key}']``, mirrored to
    ``mdata.uns[<that key>][<modality>]`` when called on a MuData.
    key_added
        Key for ``uns`` output.  Default: ``'gradient_field_{key}'``.

    Writes
    ------
    ``uns['gradient_field_{key}']``
        Dict with keys: ``U``, ``V`` (2-D float64 grids, shape (n_grid, n_grid)),
        ``X_grid``, ``Y_grid`` (1-D µm coordinates), ``divergence``, ``curl``
        (2-D, NaN outside tissue mask), ``mask`` (bool), ``params``.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.stats import gaussian_kde

    grad_key = f"gradient_{key}"
    if grad_key not in adata.obsm:
        raise KeyError(
            f"obsm['{grad_key}'] not found. "
            f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
        )

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)
    grad = adata.obsm[grad_key].astype(np.float64)
    U_cell = grad[:, 0]
    V_cell = grad[:, 1]

    # Regular grid
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    # RBF interpolation of U and V components
    log.info("gradient_field: RBF interpolation to %d × %d grid …", n_grid, n_grid)
    neighbors = min(50, len(coords) - 1)
    rbf_u = RBFInterpolator(coords, U_cell, kernel="linear", neighbors=neighbors)
    rbf_v = RBFInterpolator(coords, V_cell, kernel="linear", neighbors=neighbors)
    U = rbf_u(grid_pts).reshape(n_grid, n_grid)
    V = rbf_v(grid_pts).reshape(n_grid, n_grid)

    # Tissue mask via Gaussian KDE log-density
    bw = bandwidth_um if bandwidth_um is not None else "scott"
    try:
        kde = gaussian_kde(coords.T, bw_method=bw)
        log_dens = np.log(kde(grid_pts.T)).reshape(n_grid, n_grid)
        mask = log_dens > mask_threshold
    except Exception:
        log.warning("gradient_field: KDE tissue mask failed; using all-True mask.")
        mask = np.ones((n_grid, n_grid), dtype=bool)

    # Apply mask (NaN outside tissue)
    U_masked = np.where(mask, U, np.nan)
    V_masked = np.where(mask, V, np.nan)

    # Divergence and curl via finite differences
    dx_spacing = (x_max - x_min) / (n_grid - 1)
    dy_spacing = (y_max - y_min) / (n_grid - 1)
    dU_dy, dU_dx = np.gradient(U, dy_spacing, dx_spacing)
    dV_dy, dV_dx = np.gradient(V, dy_spacing, dx_spacing)
    divergence = np.where(mask, dU_dx + dV_dy, np.nan)
    curl = np.where(mask, dV_dx - dU_dy, np.nan)

    out_key = key_added or f"gradient_field_{key}"
    adata.uns[out_key] = {
        "U": U_masked,
        "V": V_masked,
        "X_grid": x_grid,
        "Y_grid": y_grid,
        "divergence": divergence,
        "curl": curl,
        "mask": mask,
        "params": {
            "key": key,
            "spatial_key": spatial_key,
            "n_grid": n_grid,
            "mask_threshold": mask_threshold,
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
    neighbors_key: str = "spatial_neighbors",
    layer: str | None = None,
    n_perms: int = 200,
    random_state: int = 0,
    n_jobs: int = 1,
    key_added: str = "gradient_significance",
) -> None:
    """Permutation test for gradient magnitude and directional coherence.

    For each feature key, tests two null hypotheses against a null where
    expression labels are shuffled randomly over cell positions (keeping
    the spatial graph fixed):

    1. **Magnitude**: is the mean gradient magnitude larger than random?
    2. **Coherence**: do gradient directions align more than random?
       Coherence is measured by the **mean resultant length** R̄ ∈ [0, 1]
       (Mardia & Jupp 2000, Directional Statistics, Wiley, eq. 2.2).

    .. note::

        The parametric Rayleigh test (Rayleigh 1880) assumes independent
        observations — violated here because adjacent-cell gradient
        directions are spatially autocorrelated.  The permutation null is
        preferred.

    Use :func:`~astrolabe.tl.moran` as a prerequisite screen: if a
    feature shows no spatial autocorrelation (Moran's I ≈ 0, p > 0.05),
    its gradient is likely noise and significance testing is not meaningful.

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
    neighbors_key
        Spatial graph key.
    layer
        Expression layer for re-computing null gradients.
    n_perms
        Number of permutations.  ``200`` gives ≈5 % resolution for
        p-values; use ``1000`` for publication-level precision.
    random_state
        Seed for reproducibility.
    n_jobs
        Parallel workers for the permutation loop (joblib, thread-backed —
        no extra memory copies since arrays are shared, not pickled). ``1``
        (default) runs sequentially, logging progress every ~10% of
        permutations; ``-1`` uses all available cores (per-iteration
        progress logging is skipped since iteration order isn't meaningful
        when parallel).
    key_added
        Key in ``uns`` for results.

    Writes
    ------
    ``uns[key_added]``
        :class:`pandas.DataFrame` with index = keys and columns:

        - ``magnitude`` — observed mean gradient magnitude
        - ``coherence`` — observed R̄ (mean resultant length)
        - ``p_magnitude`` — empirical p-value (one-sided: larger is significant)
        - ``p_coherence`` — empirical p-value

        Mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.
        - ``p_magnitude_adj`` — Benjamini–Hochberg adjusted p-value
        - ``p_coherence_adj`` — Benjamini–Hochberg adjusted p-value
    """
    if isinstance(keys, str):
        keys = [keys]

    conn_key, dist_key = _get_neighbors_keys(adata, neighbors_key)
    conn_mat = adata.obsp[conn_key]
    coords = adata.obsm["spatial"][:, :2].astype(np.float64)
    n_cells = len(coords)

    rng = np.random.default_rng(random_state)

    params = adata.uns.get("gradient_params", {})
    weight_by_distance = params.get("weight_by_distance", True)
    n_diffusion_steps = params.get("n_diffusion_steps", 1)
    src, dst, w = _edge_weights(adata, conn_key, dist_key, weight_by_distance)

    rows = []
    for key in keys:
        grad_key = f"gradient_{key}"
        if grad_key not in adata.obsm:
            raise KeyError(
                f"obsm['{grad_key}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
            )
        grad = adata.obsm[grad_key].astype(np.float64)
        obs_magnitude = float(np.linalg.norm(grad, axis=1).mean())
        obs_coherence = _mean_resultant_length(np.arctan2(grad[:, 1], grad[:, 0]))

        f = _extract_values(adata, key, layer)

        def _one_perm(child_rng: "np.random.Generator") -> tuple[float, float]:
            perm_idx = child_rng.permutation(n_cells)
            perm_f = f[perm_idx]
            if n_diffusion_steps > 0:
                perm_f = _diffuse_expression(perm_f, conn_mat, n_diffusion_steps)
            perm_grad = _mls_gradient(coords, perm_f, src, dst, w).astype(np.float64)
            magnitude = float(np.linalg.norm(perm_grad, axis=1).mean())
            coherence = _mean_resultant_length(np.arctan2(perm_grad[:, 1], perm_grad[:, 0]))
            return magnitude, coherence

        # Each key consumes one draw from the shared rng to seed its own
        # independent permutation substreams — keeps the overall sequence
        # deterministic across keys while permutations within a key run
        # safely in parallel.
        child_rngs = _spawn_rngs(int(rng.integers(0, 2**63 - 1)), n_perms)
        if n_jobs == 1:
            log_every = max(1, n_perms // 10)
            results = []
            for i, child_rng in enumerate(child_rngs):
                if i % log_every == 0:
                    log.info("gradient_significance: permutation %d/%d", i + 1, n_perms)
                results.append(_one_perm(child_rng))
        else:
            log.info(
                "gradient_significance: '%s' running %d permutations with n_jobs=%d",
                key, n_perms, n_jobs,
            )
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_one_perm)(child_rng) for child_rng in child_rngs
            )

        null_magnitudes = np.array([r[0] for r in results], dtype=np.float64)
        null_coherences = np.array([r[1] for r in results], dtype=np.float64)

        p_mag = float(np.mean(null_magnitudes >= obs_magnitude))
        p_coh = float(np.mean(null_coherences >= obs_coherence))
        rows.append(
            {
                "key": key,
                "magnitude": obs_magnitude,
                "coherence": obs_coherence,
                "p_magnitude": p_mag,
                "p_coherence": p_coh,
            }
        )
        log.info(
            "gradient_significance: '%s' magnitude=%.4f (p=%.3f) "
            "coherence=%.3f (p=%.3f)",
            key,
            obs_magnitude,
            p_mag,
            obs_coherence,
            p_coh,
        )

    result = pd.DataFrame(rows).set_index("key")
    result["p_magnitude_adj"] = false_discovery_control(
        result["p_magnitude"].clip(lower=1e-10), method="bh"
    )
    result["p_coherence_adj"] = false_discovery_control(
        result["p_coherence"].clip(lower=1e-10), method="bh"
    )
    adata.uns[key_added] = result


@with_adata(mirror_uns=from_kwarg("key_added"))
def compartment_flux(
    adata: ad.AnnData,
    groupby: str,
    keys: str | list[str],
    *,
    neighbors_key: str = "spatial_neighbors",
    spatial_key: str = "spatial",
    key_added: str = "compartment_flux",
) -> None:
    """Net expression flux across cell-type/compartment boundaries.

    For each edge in the spatial graph that crosses a cell-type boundary
    (``obs[groupby][src] ≠ obs[groupby][dst]``), projects the per-cell
    MLS gradient at the *source* cell onto the unit vector pointing from
    source to destination.  Aggregates the mean projected flux by ordered
    ``(label_src → label_dst)`` pair.

    A positive value means the gradient at the A–B interface points from A
    toward B (expression rises from A toward B); a negative value means
    the gradient points away from B.

    Conceptually related to COMMOT (Cang et al. 2023, Nature Methods)
    which models directional signalling flux across cell types via
    collective optimal transport.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData.
    groupby
        ``obs`` column with cell-type / compartment labels.
    keys
        Feature keys whose gradients to project.  Must have been computed
        by :func:`~astrolabe.tl.expression_gradient`.
    neighbors_key
        Spatial graph key.
    spatial_key
        ``obsm`` key for µm coordinates.
    key_added
        Key in ``uns`` for results.

    Writes
    ------
    ``uns[key_added]``
        Dict mapping ``key → DataFrame(groups × groups)``.  Entry
        ``df.loc[A, B]`` is the mean projected flux from group A toward
        group B (NaN if no cross-boundary A→B edges exist).
        Mirrored to ``mdata.uns[key_added][<modality>]`` when called on a MuData.
    """
    if isinstance(keys, str):
        keys = [keys]

    if groupby not in adata.obs.columns:
        raise KeyError(f"obs['{groupby}'] not found.")

    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    conn_coo = adata.obsp[conn_key].tocoo()
    src_all = conn_coo.row.astype(np.intp)
    dst_all = conn_coo.col.astype(np.intp)

    labels = adata.obs[groupby].astype("category")
    codes = labels.cat.codes.to_numpy()
    cats = labels.cat.categories.tolist()
    n_groups = len(cats)

    # Find cross-boundary edges
    cross_mask = codes[src_all] != codes[dst_all]
    src_cross = src_all[cross_mask]
    dst_cross = dst_all[cross_mask]

    coords = adata.obsm[spatial_key][:, :2].astype(np.float64)
    dx = coords[dst_cross] - coords[src_cross]
    norms = np.linalg.norm(dx, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    dx_unit = dx / norms  # unit vectors A→B

    results: dict[str, pd.DataFrame] = {}
    for key in keys:
        grad_key = f"gradient_{key}"
        if grad_key not in adata.obsm:
            raise KeyError(
                f"obsm['{grad_key}'] not found. "
                f"Run ab.tl.expression_gradient(mdata, keys=['{key}']) first."
            )
        grad = adata.obsm[grad_key].astype(np.float64)

        # Project source gradient onto A→B direction
        proj_flux = np.sum(grad[src_cross] * dx_unit, axis=1)  # (n_cross_edges,)

        # Accumulate per ordered pair
        flux_sum = np.zeros((n_groups, n_groups), dtype=np.float64)
        flux_count = np.zeros((n_groups, n_groups), dtype=np.int64)
        src_codes = codes[src_cross]
        dst_codes = codes[dst_cross]
        np.add.at(flux_sum, (src_codes, dst_codes), proj_flux)
        np.add.at(flux_count, (src_codes, dst_codes), 1)

        with np.errstate(invalid="ignore"):
            flux_mean = np.where(flux_count > 0, flux_sum / flux_count, np.nan)

        df = pd.DataFrame(flux_mean, index=cats, columns=cats)
        df.index.name = "source"
        df.columns.name = "target"
        results[key] = df

    adata.uns[key_added] = {
        "flux": results,
        "params": {
            "groupby": groupby,
            "keys": keys,
            "neighbors_key": neighbors_key,
            "spatial_key": spatial_key,
        },
    }
    log.info(
        "compartment_flux: %d cross-boundary edges across %d groups for %d features",
        int(cross_mask.sum()),
        n_groups,
        len(keys),
    )


@with_adata(mirror_uns=from_kwarg("key_added"))
def spatial_diffusion(
    adata: ad.AnnData,
    *,
    spatial_key: str = "spatial",
    neighbors_key: str = "spatial_neighbors",
    n_steps: int = 5,
    normalize: Literal["coifman_lafon", "row"] = "coifman_lafon",
    compute_eigenmodes: bool = False,
    n_eigenmodes: int = 10,
    diffusion_score: bool = False,
    key_added: str = "diffusion",
) -> None:
    """Graph heat-equation diffusion on the spatial cell graph.

    Approximates the solution to the heat equation on the cell graph:

    .. math::

        \\frac{\\partial f}{\\partial t} = -\\mathbf{L}f,
        \\quad f(t) = \\exp(-t\\mathbf{L})\\,f_0

    by iterative multiplication with the diffusion operator M:

    .. math::

        f(t) \\approx M^n\\,f_0, \\quad n = \\text{n\\_steps}

    Uses the **Coifman–Lafon density normalization** (α = 1) which
    separates the geometry of the tissue from cell-density variation:

    .. math::

        \\hat{K}_{ij} = \\frac{K_{ij}}{\\sqrt{q_i q_j}},
        \\quad q_i = \\sum_j K_{ij}

    Without this normalization, naive row-normalization causes diffusion
    to track high-density tumor cores rather than expression geometry.

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
    n_steps
        Number of diffusion steps (higher = more smoothing).
    normalize
        Normalization strategy for the graph Laplacian:

        - ``'coifman_lafon'`` (default) — density-normalized diffusion maps
          operator.  Recommended for non-uniform cell densities.
        - ``'row'`` — plain row-normalization.  Faster but biased by density.
    compute_eigenmodes
        If ``True``, also compute the ``n_eigenmodes`` leading eigenvectors of
        the symmetric normalized Laplacian and store them in
        ``obsm['X_diffusion']``.  These form a spatial Fourier basis.
    n_eigenmodes
        Number of eigenmodes to compute when ``compute_eigenmodes=True``.
    diffusion_score
        If ``True``, compute a Sepal-style time-to-uniformity score for every
        gene in ``X`` and store in ``var['diffusion_score']``.  This ranks genes
        by how long their spatial pattern persists under diffusion.
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
        float64 — Sepal-style persistence score per gene
        (only when ``diffusion_score=True``; adata-resident only).
    """
    from scipy.sparse.linalg import eigsh

    conn_key, _ = _get_neighbors_keys(adata, neighbors_key)
    A = adata.obsp[conn_key].astype(np.float64)

    if normalize == "coifman_lafon":
        # Step 1: density normalization — d_i = sum_j K_ij
        d = np.asarray(A.sum(axis=1), dtype=np.float64).ravel()
        d[d == 0] = 1.0
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(d))
        # Step 2: normalized kernel (symmetric)
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    else:
        A_hat = A.copy()

    # Row-normalize to get diffusion operator M
    row_sums = np.asarray(A_hat.sum(axis=1), dtype=np.float64).ravel()
    row_sums[row_sums == 0] = 1.0
    M = sp.diags(1.0 / row_sums) @ A_hat

    uns_entry: dict = {
        "params": {
            "spatial_key": spatial_key,
            "neighbors_key": neighbors_key,
            "n_steps": n_steps,
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
        eigenvalues, eigenvectors = eigsh(L_sym, k=k, which="SM")
        sort_idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, sort_idx].astype(np.float64)
        adata.obsm["X_diffusion"] = eigenvectors
        uns_entry["eigenvalues"] = eigenvalues[sort_idx].tolist()

    if diffusion_score:
        log.info("spatial_diffusion: computing per-gene diffusion scores …")
        mat = adata.X
        if sp.issparse(mat):
            mat = mat.toarray()
        mat = np.asarray(mat, dtype=np.float64)
        n_score_steps = max(n_steps, 5)
        initial_var = mat.var(axis=0)
        initial_var[initial_var == 0] = 1.0
        f_t = mat.copy()
        for _ in range(n_score_steps):
            f_t = (M @ f_t)
        final_var = f_t.var(axis=0)
        scores = 1.0 - (final_var / initial_var).clip(0.0, 1.0)
        adata.var["diffusion_score"] = scores.astype(np.float64)

    adata.uns[key_added] = uns_entry
    log.info(
        "spatial_diffusion: %d cells, %d steps, normalize='%s'",
        adata.n_obs,
        n_steps,
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

    Fits the phenomenological model ``f(r) = f₀ · e^(−r/λ)`` where:

    - ``r`` is the distance from *source* in µm
    - ``λ`` is the **decay length** in µm (the quantity of biological interest)
    - ``f₀`` is the expression at the source

    .. warning::

        ``λ`` is a **phenomenological decay length**, not a physical diffusion
        constant D.  It is only interpretable if the observed spatial gradient
        approximately follows radial decay from a localised source — an
        assumption best validated visually before interpreting λ.
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
        Dict with ``lambda_um``, ``f0``, ``r_squared``, ``bin_centers``,
        ``bin_means``, ``params``. Mirrored to
        ``mdata.uns[<that key>][<modality>]`` when called on a MuData.
    """
    from scipy.optimize import curve_fit

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

    # Fit exponential to non-NaN bins
    valid = ~np.isnan(bin_means) & (bin_counts >= 3)
    if valid.sum() < 4:
        warnings.warn(
            f"decay_length: insufficient bins with ≥3 cells to fit '{key}'. "
            "Try increasing max_dist_um or n_bins.",
            stacklevel=3,
        )
        lambda_um, f0, r_sq = np.nan, np.nan, np.nan
    else:
        def _exp_decay(r, f0, lam):
            return f0 * np.exp(-r / lam)

        try:
            p0 = [float(np.nanmax(bin_means)), float(max_dist_um / 3)]
            popt, _ = curve_fit(
                _exp_decay,
                bin_centers[valid],
                bin_means[valid],
                p0=p0,
                bounds=([0, 1e-3], [np.inf, max_dist_um * 10]),
                maxfev=5000,
            )
            f0, lambda_um = float(popt[0]), float(popt[1])
            fitted = _exp_decay(bin_centers[valid], f0, lambda_um)
            ss_res = np.sum((bin_means[valid] - fitted) ** 2)
            ss_tot = np.sum((bin_means[valid] - bin_means[valid].mean()) ** 2)
            r_sq = float(1 - ss_res / (ss_tot + 1e-12))
        except RuntimeError:
            warnings.warn(
                f"decay_length: exponential fit did not converge for '{key}'.",
                stacklevel=3,
            )
            lambda_um, f0, r_sq = np.nan, np.nan, np.nan

    out_key = key_added or f"decay_length_{key}"
    adata.uns[out_key] = {
        "lambda_um": lambda_um,
        "f0": f0,
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
