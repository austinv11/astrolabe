"""
BANKSY spatially-augmented feature matrix.

Augments each cell's expression with a λ-weighted contribution from its
spatial neighborhood (M0 = neighbor mean; |M1| = azimuthal Gabor magnitude),
producing a concatenated feature matrix for PCA-based clustering.

Reference
---------
Singhal et al., *Nature Genetics* 2024.
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
    coords: np.ndarray | None,
    agf: bool,
) -> tuple[sp.csr_matrix, sp.csr_matrix | None, sp.csr_matrix | None]:
    """Build row-normalized M0 weight matrix and optional AGF weight matrices."""
    coo = conn.tocoo()
    rows, cols = coo.row.astype(np.int32), coo.col.astype(np.int32)
    n = conn.shape[0]

    row_counts = np.bincount(rows, minlength=n).astype(np.float64)
    row_counts[row_counts == 0] = 1.0
    w = 1.0 / row_counts[rows]

    W_norm = sp.csr_matrix((w, (rows, cols)), shape=(n, n))

    if not agf:
        return W_norm, None, None

    dy = coords[cols, 1] - coords[rows, 1]
    dx = coords[cols, 0] - coords[rows, 0]
    theta = np.arctan2(dy, dx)

    W_cos = sp.csr_matrix((w * np.cos(theta), (rows, cols)), shape=(n, n))
    W_sin = sp.csr_matrix((w * np.sin(theta), (rows, cols)), shape=(n, n))
    return W_norm, W_cos, W_sin


@with_adata
def banksy_matrix(
    adata: ad.AnnData,
    *,
    lambda_param: float = 0.2,
    agf: bool = False,
    agf_weight: float = 0.5,
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

    Two neighborhood features are supported:

    **M0 — neighborhood mean** (always included):
        ``M0ᵢ = (1/k) Σ_{j∈N(i)} xⱼ``

    **|M1| — azimuthal Gabor magnitude** (``agf=True``):
        ``M1ᵢ = (1/k) Σ_{j∈N(i)} xⱼ · exp(i·θᵢⱼ)``
        where θᵢⱼ is the angle from cell i to neighbor j.  The magnitude
        |M1| captures directional expression gradients and improves tissue
        domain segmentation in structured tissues.

    The matrix stored in ``obsm[key_added]`` is:

    - Without AGF: ``[√(1−λ)·X,  √λ·M0]``  — shape ``(n_cells, 2·n_genes)``
    - With AGF:  ``[√(1−λ)·X,  √(λ(1−κ))·M0,  √(λκ)·|M1|]``
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
        domain segmentation.
    agf
        Include the azimuthal Gabor magnitude (|M1|) as an additional
        neighborhood feature block.
    agf_weight
        Fraction of λ allocated to |M1| vs M0 when ``agf=True``.
        Default 0.5 splits evenly.
    neighbors_key
        Prefix for the spatial graph in ``obsp``
        (``{neighbors_key}_connectivities``).
    spatial_key
        ``obsm`` key containing ``(n_cells, 2)`` micron coordinates.
        Required only when ``agf=True``.
    layer
        Expression layer to use.  ``None`` uses ``adata.X``.
    key_added
        Key in ``obsm`` for the output matrix.

    Examples
    --------
    >>> # cell typing
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.2)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.tl.cluster(mdata)

    >>> # tissue domain segmentation with AGF
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.8, agf=True)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.tl.cluster(mdata, key_added="spatial_domains")

    References
    ----------
    Singhal et al., *Nature Genetics* 2024.
    """
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )
    if agf and spatial_key not in adata.obsm:
        raise KeyError(
            f"obsm['{spatial_key}'] not found; required for AGF computation."
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

    coords = adata.obsm[spatial_key].astype(np.float64) if agf else None
    W_norm, W_cos, W_sin = _build_weight_matrices(
        adata.obsp[conn_key], coords, agf
    )

    M0 = np.asarray(W_norm @ X, dtype=np.float32)

    w_self = float(np.sqrt(1.0 - lambda_param))

    if not agf:
        w_nbr = float(np.sqrt(lambda_param))
        out = np.concatenate([w_self * X, w_nbr * M0], axis=1)
    else:
        w_m0 = float(np.sqrt(lambda_param * (1.0 - agf_weight)))
        w_m1 = float(np.sqrt(lambda_param * agf_weight))
        M1_real = np.asarray(W_cos @ X, dtype=np.float32)
        M1_imag = np.asarray(W_sin @ X, dtype=np.float32)
        M1_mag = np.sqrt(M1_real ** 2 + M1_imag ** 2)
        out = np.concatenate([w_self * X, w_m0 * M0, w_m1 * M1_mag], axis=1)

    adata.obsm[key_added] = out.astype(np.float32)
    log.info(
        "banksy_matrix: λ=%.2f  agf=%s  → obsm['%s']  shape=%s",
        lambda_param, agf, key_added, out.shape,
    )
