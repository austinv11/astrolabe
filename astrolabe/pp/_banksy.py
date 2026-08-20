"""
BANKSY spatially-augmented feature matrix.

Augments each cell's expression with a λ-weighted contribution from its
spatial neighborhood (M0 = neighbor mean; |M1| = azimuthal Gabor magnitude),
producing a concatenated feature matrix for PCA-based clustering.

Reference
---------
Singhal, V., Chou, N., Lee, J. et al. *BANKSY unifies cell typing and tissue
domain segmentation for scalable spatial omics data analysis.*
Nature Genetics 56, 431-441 (2024). doi:10.1038/s41588-024-01664-3
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scipy.sparse as sp

from astrolabe._utils import with_adata

__all__ = ["banksy_matrix"]

log = logging.getLogger(__name__)


def _build_weight_matrices(
    conn: sp.spmatrix,
    coords: np.ndarray,
    agf: bool,
) -> tuple[sp.csr_matrix, sp.csr_matrix | None, sp.csr_matrix | None]:
    """Build a distance-decayed, row-normalized M0 weight matrix and optional AGF weight matrices.

    Mirrors published BANKSY's default ``'scaled_gaussian'`` neighbor
    weighting (Singhal et al. 2024; banksy_py's
    ``generate_spatial_weights_fixed_nbrs``): within each cell's own
    k-neighbor set, weight decays with distance rather than being uniform.
    Each cell's own neighbor distances set a per-cell Gaussian bandwidth
    (the median neighbor distance), so weighting adapts to local cell
    density instead of using one global bandwidth; weights are then
    row-normalized to sum to 1, same as astrolabe's previous uniform
    ``1/k`` scheme.
    """
    csr = conn.tocsr()
    csr.sort_indices()
    n = csr.shape[0]
    indptr = csr.indptr
    cols = csr.indices.astype(np.int32)
    rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(indptr))

    dy = coords[cols, 1] - coords[rows, 1]
    dx = coords[cols, 0] - coords[rows, 0]
    dist = np.hypot(dx, dy)

    weights = np.ones_like(dist)
    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if end <= start:
            continue
        nbrs = dist[start:end]
        median_r = np.median(nbrs)
        if median_r > 0:
            weights[start:end] = np.exp(-((nbrs / median_r) ** 2))
        # else: all neighbors coincide with the cell itself -- fall back
        # to the uniform weight already set above.

    row_totals = np.bincount(rows, weights=weights, minlength=n)
    row_totals[row_totals == 0] = 1.0
    w = weights / row_totals[rows]

    W_norm = sp.csr_matrix((w, cols, indptr), shape=(n, n))

    if not agf:
        return W_norm, None, None

    theta = np.arctan2(dy, dx)
    W_cos = sp.csr_matrix((w * np.cos(theta), cols, indptr), shape=(n, n))
    W_sin = sp.csr_matrix((w * np.sin(theta), cols, indptr), shape=(n, n))
    return W_norm, W_cos, W_sin


def _zscore_columns(A: np.ndarray) -> np.ndarray:
    """Column-wise z-score (population std, ddof=0).

    Zero-variance columns (undetected genes, or an |M1| feature that is
    identically zero) become all-zero rather than NaN, so they contribute
    no variance to the block instead of poisoning the PCA downstream.
    """
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    zero_var = sd == 0
    sd_safe = np.where(zero_var, 1.0, sd)
    Z = (A - mu) / sd_safe
    if zero_var.any():
        Z[:, zero_var] = 0.0
    return Z


def _agf_magnitude(
    W_cos: sp.csr_matrix,
    W_sin: sp.csr_matrix,
    X: np.ndarray,
    M0: np.ndarray,
) -> np.ndarray:
    """Local-mean-subtracted azimuthal Gabor magnitude |M1|.

    ``M1ᵢ = Σ_{j∈N(i)} wᵢⱼ·(xⱼ − M0ᵢ)·exp(i·θᵢⱼ)`` — the weighted
    neighborhood mean M0ᵢ is subtracted from each neighbor's value *before*
    the phase-weighted sum, matching published BANKSY's AGF definition
    (Singhal et al. 2024). Without the subtraction, the DC (mean) component
    of the neighborhood leaks into |M1| through the nonzero phase-vector sum
    ``Σ wᵢⱼ·exp(iθᵢⱼ)``, making |M1| partly duplicate M0 rather than
    capture the orthogonal gradient information it is meant to. With the
    subtraction, a spatially-constant gene has |M1| exactly 0.
    """
    # Σ w·(x_j − M0_i)·cosθ = (W_cos @ X)_i − M0_i·Σ w·cosθ  (row sums)
    cos_row = np.asarray(W_cos.sum(axis=1), dtype=np.float64).ravel()
    sin_row = np.asarray(W_sin.sum(axis=1), dtype=np.float64).ravel()
    M1_real = np.asarray(W_cos @ X, dtype=np.float32) - cos_row[:, None].astype(np.float32) * M0
    M1_imag = np.asarray(W_sin @ X, dtype=np.float32) - sin_row[:, None].astype(np.float32) * M0
    return np.sqrt(M1_real ** 2 + M1_imag ** 2)


@with_adata
def banksy_matrix(
    adata: ad.AnnData,
    *,
    lambda_param: float = 0.2,
    agf: bool = False,
    agf_weight: float = 1.0 / 3.0,
    neighbors_key: str = "spatial_neighbors",
    spatial_key: str = "spatial",
    layer: str | None = None,
    key_added: str = "X_banksy",
) -> None:
    """Build the BANKSY spatially-augmented feature matrix.

    Augments each cell's normalized expression with a λ-weighted contribution
    from its spatial neighborhood and stores the result in ``obsm``.  Feed
    the output into :func:`~astrolabe.tl.pca` + :func:`~astrolabe.tl.cluster`
    for spatially-coherent cell typing (low λ) or tissue-domain segmentation
    (high λ).

    Neighbor contributions are weighted by distance rather than uniformly,
    matching published BANKSY's default ``'scaled_gaussian'`` scheme
    (Singhal et al. 2024): within each cell's own k-neighbor set, weight
    decays with distance from a per-cell Gaussian bandwidth set by that
    cell's median neighbor distance, then is row-normalized to sum to 1 —
    so closer neighbors within the set contribute more than distant ones,
    rather than every neighbor counting equally.

    Two neighborhood features are supported:

    **M0 — neighborhood mean** (always included):
        ``M0ᵢ = Σ_{j∈N(i)} wᵢⱼ·xⱼ``,  weights ``wᵢⱼ`` distance-decayed and
        summing to 1 over ``j∈N(i)`` (see above).

    **|M1| — azimuthal Gabor magnitude** (``agf=True``):
        ``M1ᵢ = Σ_{j∈N(i)} wᵢⱼ·(xⱼ − M0ᵢ) · exp(i·θᵢⱼ)``
        where θᵢⱼ is the angle from cell i to neighbor j and ``wᵢⱼ`` are the
        same distance-decayed weights used for M0.  The weighted neighborhood
        mean M0ᵢ is subtracted before the phase-weighted sum (published
        BANKSY's definition), so |M1| captures directional expression
        *gradients* orthogonal to M0's mean level rather than duplicating it.

    Each block (own expression X, M0, and |M1| when present) is **z-scored
    column-wise** (zero mean, unit variance per feature; zero-variance
    features become 0) *before* the √-weights are applied.  This is what
    makes λ a genuine **variance split**: after weighting, the own-expression
    block carries exactly a ``(1−λ)`` share of each feature's unit variance
    and the neighborhood blocks carry the remaining ``λ`` — matching
    published BANKSY, whose λ≈0.2 (cell typing) / λ≈0.8 (domain
    segmentation) guidance assumes this normalization.  Without it, λ's
    effective meaning would silently depend on the dataset's expression
    scale and neighborhood smoothing.

    The matrix stored in ``obsm[key_added]`` is (Z(·) = column z-score):

    - Without AGF: ``[√(1−λ)·Z(X),  √λ·Z(M0)]`` — shape ``(n_cells, 2·n_genes)``
    - With AGF:  ``[√(1−λ)·Z(X),  √(λ(1−κ))·Z(M0),  √(λκ)·Z(|M1|)]``
      — shape ``(n_cells, 3·n_genes)``  (κ = ``agf_weight``)

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    lambda_param
        Spatial weighting factor λ ∈ [0, 1].  ``0`` = pure expression,
        ``1`` = pure neighborhood.  ~0.2 for cell typing, ~0.8 for spatial
        domain segmentation (Singhal et al. 2024).
    agf
        Include the azimuthal Gabor magnitude (|M1|) as an additional
        neighborhood feature block.
    agf_weight
        Fraction κ of λ allocated to |M1| vs M0 when ``agf=True``.
        Default ``1/3`` matches banksy_py's ``2^-m`` harmonic split
        (harmonic m gets weight ∝ 2^-m, so M0 (m=0) receives ``2/3·λ`` and
        |M1| (m=1) receives ``1/3·λ`` after normalization).
    neighbors_key
        Prefix for the spatial graph in ``obsp``
        (``{neighbors_key}_connectivities``).
    spatial_key
        ``obsm`` key containing ``(n_cells, 2)`` micron coordinates.
        Always required — used to compute the distance-decayed neighbor
        weights for M0 (and |M1| when ``agf=True``).
    layer
        Expression layer to use.  ``None`` uses ``adata.X``.
    key_added
        Key in ``obsm`` for the output matrix.

    Divergences from published BANKSY
    ---------------------------------
    banksy_py builds a *larger* (2k) neighbor set for the AGF (|M1|) term
    than for M0, and uses raw *directed* kNN graphs for both; astrolabe
    reuses the single symmetrized k-neighbor graph from
    :func:`~astrolabe.tl.spatial_neighbors` for both blocks.  All other
    previously-divergent details (per-block z-scoring, AGF local-mean
    subtraction, and the 2^-m harmonic λ split) now follow the published
    method.  The shared symmetrized graph is a deliberate, validated choice
    (``benchmarks/banksy_py/banksy_matrix.ipynb``, final section): on
    planted domains with heterogeneous cell density it recovers domains at
    least as well as banksy_py's own directed k + 2k construction, while
    the naive alternative — a directed k-graph shared across both blocks —
    collapses when one domain is much denser than the rest.  The AGF term
    needs either banksy_py's larger 2k set or the symmetrized graph's
    variable-degree neighborhoods to stay stable under non-uniform density;
    symmetrization provides that with the one graph the rest of the library
    already shares.

    Examples
    --------
    >>> # cell typing
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.2)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.tl.transcriptional_neighbors(mdata)  # graph over the BANKSY-PCA embedding
    >>> ab.tl.cluster(mdata)

    >>> # tissue domain segmentation with AGF
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.8, agf=True)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.tl.transcriptional_neighbors(mdata)
    >>> ab.tl.cluster(mdata, key_added="spatial_domains")

    References
    ----------
    Singhal, V., Chou, N., Lee, J. et al. *BANKSY unifies cell typing and
    tissue domain segmentation for scalable spatial omics data analysis.*
    Nature Genetics 56, 431-441 (2024). doi:10.1038/s41588-024-01664-3
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found; required for distance-decayed "
            f"neighbor weighting."
        )
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError(f"lambda_param must be in [0, 1]; got {lambda_param}")
    if agf and not 0.0 <= agf_weight <= 1.0:
        raise ValueError(f"agf_weight must be in [0, 1]; got {agf_weight}")

    X = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(X):
        dense_gb = X.shape[0] * X.shape[1] * 4 / 1e9
        if dense_gb > 2.0:
            log.warning(
                "banksy_matrix: converting %d-cell × %d-feature sparse matrix to dense "
                "(%.1f GB); this may be slow or exhaust memory.",
                X.shape[0], X.shape[1], dense_gb,
            )
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    coords = adata.obsm[spatial_key].astype(np.float64)
    W_norm, W_cos, W_sin = _build_weight_matrices(
        adata.obsp[conn_key], coords, agf
    )

    M0 = np.asarray(W_norm @ X, dtype=np.float32)

    # Z-score every block BEFORE the sqrt weights so that λ splits variance
    # exactly (published BANKSY semantics; see docstring). Note: because
    # column z-scoring is affine-invariant and W_norm is row-normalized, the
    # result is identical whether M0/|M1| are computed from raw or z-scored
    # expression -- only the post-hoc per-block z-score matters.
    X_z = _zscore_columns(X)
    M0_z = _zscore_columns(M0)

    w_self = float(np.sqrt(1.0 - lambda_param))

    if not agf:
        w_nbr = float(np.sqrt(lambda_param))
        out = np.concatenate([w_self * X_z, w_nbr * M0_z], axis=1)
    else:
        w_m0 = float(np.sqrt(lambda_param * (1.0 - agf_weight)))
        w_m1 = float(np.sqrt(lambda_param * agf_weight))
        M1_z = _zscore_columns(_agf_magnitude(W_cos, W_sin, X, M0))
        out = np.concatenate([w_self * X_z, w_m0 * M0_z, w_m1 * M1_z], axis=1)

    adata.obsm[key_added] = out.astype(np.float32)
    log.info(
        "banksy_matrix: λ=%.2f  agf=%s  → obsm['%s']  shape=%s",
        lambda_param, agf, key_added, out.shape,
    )
