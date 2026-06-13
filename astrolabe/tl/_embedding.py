"""
Dimensionality reduction embeddings.  Thin wrappers around scanpy.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc

from astrolabe._gpu import _gpu_X, _require_rapids, _to_cpu_arr, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["pca", "umap", "spatial_umap"]

log = logging.getLogger(__name__)


@with_adata
def pca(
    adata: ad.AnnData,
    *,
    n_comps: int = 50,
    use_rep: str | None = None,
    use_highly_variable: bool | None = None,
    layer: str | None = None,
    zero_center: bool = True,
    svd_solver: str = "arpack",
    random_state: int = 0,
    key_added: str | None = None,
    **kwargs,
) -> None:
    """Compute PCA on the expression matrix or an arbitrary ``obsm`` embedding.

    When ``use_rep`` is ``None``, delegates to :func:`scanpy.tl.pca` and
    results are stored in ``obsm['X_pca']``, ``varm['PCs']``, ``uns['pca']``.

    When ``use_rep`` is set, runs scikit-learn PCA on ``obsm[use_rep]``
    (e.g. the BANKSY matrix) and stores the result in
    ``obsm[key_added or 'X_pca']``.

    If ``var['highly_variable']`` is present (written by
    :func:`~astrolabe.pp.highly_variable_genes`), PCA is restricted to those
    genes unless ``use_highly_variable=False`` is passed explicitly.
    (Ignored when ``use_rep`` is set.)

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should be
        normalized and log-transformed first.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_comps
        Number of principal components to compute.
    use_rep
        Key in ``obsm`` to use as input instead of ``X`` / ``layer``.
        Useful for running PCA on the BANKSY matrix or any other precomputed
        embedding.  When set, ``layer``, ``use_highly_variable``,
        ``zero_center``, and ``svd_solver`` are ignored.
    use_highly_variable
        Restrict PCA to ``var['highly_variable']`` genes.  ``None`` (default)
        uses HVGs automatically when the column exists.  Ignored when
        ``use_rep`` is set.
    layer
        Layer to use.  ``None`` uses ``X``.  Ignored when ``use_rep`` is set.
    zero_center
        Subtract the mean before computing SVD.
    svd_solver
        SVD solver.  ``'arpack'`` is stable; ``'randomized'`` is faster for
        very large datasets.
    random_state
        Random seed.
    key_added
        Output key in ``obsm`` (and prefix for ``varm``/``uns`` in the scanpy
        path).  ``None`` defaults to ``'X_pca'``.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.pca` (ignored when
        ``use_rep`` is set).

    Examples
    --------
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.pp.highly_variable_genes(mdata, n_top_genes=2000)
    >>> ab.tl.pca(mdata)

    >>> # PCA on the BANKSY matrix
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.8)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.tl.cluster(mdata)
    """
    if use_rep is None and gpu_enabled():
        rsc = _require_rapids()
        out_key = key_added or "X_pca"
        with _gpu_X(adata) as moved:
            rsc.pp.pca(
                adata,
                n_comps=n_comps,
                use_highly_variable=use_highly_variable,
                layer=layer,
                zero_center=zero_center,
                svd_solver=svd_solver,
                random_state=random_state,
                key_added=key_added,
                **kwargs,
            )
            if moved:
                if out_key in adata.obsm:
                    adata.obsm[out_key] = _to_cpu_arr(adata.obsm[out_key])
                if "PCs" in adata.varm:
                    adata.varm["PCs"] = _to_cpu_arr(adata.varm["PCs"])
        return

    if use_rep is not None:
        from sklearn.decomposition import PCA as _PCA

        if use_rep not in adata.obsm:
            raise KeyError(f"obsm['{use_rep}'] not found.")
        X_in = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        n_comps_eff = min(n_comps, X_in.shape[0] - 1, X_in.shape[1] - 1)
        embedding = _PCA(
            n_components=n_comps_eff, random_state=random_state
        ).fit_transform(X_in).astype(np.float32)
        adata.obsm[key_added or "X_pca"] = embedding
        log.info(
            "pca (use_rep='%s'): %d components → obsm['%s']",
            use_rep, n_comps_eff, key_added or "X_pca",
        )
        return

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=use_highly_variable,
        layer=layer,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
        key_added=key_added,
        **kwargs,
    )


@with_adata
def umap(
    adata: ad.AnnData,
    *,
    neighbors_key: str = "spatial_neighbors",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    key_added: str = "X_umap",
    **kwargs,
) -> None:
    """Compute UMAP embedding of the neighborhood graph.

    Stores the embedding in ``obsm[key_added]``.  Requires that
    :func:`astrolabe.tl.spatial_neighbors` or ``sc.pp.neighbors`` has
    been run first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with
        connectivity graph in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    neighbors_key
        Key in ``uns`` pointing to the neighborhood graph.
    min_dist
        UMAP minimum distance parameter.  Smaller values produce tighter
        clusters; larger values preserve more global structure.
    spread
        Effective scale of embedded points.
    n_components
        Number of UMAP dimensions.
    random_state
        Random seed.
    key_added
        Key in ``obsm`` for the embedding.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.umap`.
    """
    if gpu_enabled():
        rsc = _require_rapids()
        rsc.tl.umap(
            adata,
            neighbors_key=neighbors_key,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            random_state=random_state,
            key_added=key_added,
            **kwargs,
        )
        return
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        key_added=key_added,
        **kwargs,
    )


@with_adata
def spatial_umap(
    adata: ad.AnnData,
    *,
    alpha: float = 0.5,
    expr_neighbors_key: str = "neighbors",
    spatial_neighbors_key: str = "spatial_neighbors",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    joint_key: str = "joint_neighbors",
    key_added: str = "X_spatial_umap",
    **kwargs,
) -> None:
    """UMAP embedding using a weighted mix of expression and spatial neighbourhoods.

    Blends the transcriptomic neighbourhood graph (from ``sc.pp.neighbors``)
    with the spatial neighbourhood graph (from :func:`~astrolabe.tl.spatial_neighbors`)
    before running UMAP.  The result captures both cell-type identity and
    tissue organisation — cells of the same type that are also spatially
    co-localised cluster more tightly than cells that share identity but
    are scattered across the tissue.

    Both neighbour graphs must exist in ``obsp`` before calling this function.
    Run :func:`~astrolabe.tl.spatial_neighbors` and ``sc.pp.neighbors``
    (after :func:`~astrolabe.tl.pca`) first.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with both
        neighbourhood graphs in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    alpha
        Spatial weight in [0, 1].  ``alpha=0`` is a pure expression UMAP;
        ``alpha=1`` is a pure spatial layout; ``alpha=0.5`` (default) gives
        equal weight to both graphs.
    expr_neighbors_key
        Key in ``uns`` for the transcriptomic neighbourhood graph written
        by ``sc.pp.neighbors``.  Default ``'neighbors'``.
    spatial_neighbors_key
        Key in ``uns`` for the spatial neighbourhood graph written by
        :func:`~astrolabe.tl.spatial_neighbors`.  Default
        ``'spatial_neighbors'``.
    min_dist
        UMAP minimum distance.
    spread
        UMAP spread.
    n_components
        Number of UMAP dimensions.
    random_state
        Random seed.
    joint_key
        Key prefix used to store the blended graph in ``obsp`` and ``uns``.
        Cleaned up from ``uns`` after the UMAP run.
    key_added
        Key in ``obsm`` where the embedding is stored.

    Examples
    --------
    >>> sc.pp.neighbors(mdata["xenium"])          # expression graph
    >>> ab.tl.spatial_neighbors(mdata)            # spatial graph
    >>> ab.tl.spatial_umap(mdata, alpha=0.3)      # 30% spatial, 70% expression
    >>> ab.pl.points(mdata, color="leiden", embedding="X_spatial_umap").show()
    """
    import scipy.sparse as sp

    # ── Resolve connectivity matrices ─────────────────────────────────────────
    def _get_conn(key: str) -> sp.spmatrix:
        meta = adata.uns.get(key)
        if meta is None:
            raise KeyError(
                f"uns['{key}'] not found. "
                + (
                    "Run sc.pp.neighbors() first."
                    if key == expr_neighbors_key
                    else f"Run ab.tl.spatial_neighbors() first."
                )
            )
        conn_key = meta.get("connectivities_key", f"{key}_connectivities")
        if conn_key not in adata.obsp:
            raise KeyError(
                f"obsp['{conn_key}'] not found (expected from uns['{key}'])."
            )
        return adata.obsp[conn_key].astype(np.float32)

    conn_expr = _get_conn(expr_neighbors_key)
    conn_spatial = _get_conn(spatial_neighbors_key)

    if conn_expr.shape != conn_spatial.shape:
        raise ValueError(
            f"Expression ({conn_expr.shape}) and spatial ({conn_spatial.shape}) "
            "connectivity matrices have different shapes."
        )

    # ── Row-normalise so both graphs are on the same scale ────────────────────
    def _row_norm(mat: sp.spmatrix) -> sp.spmatrix:
        row_sums = np.asarray(mat.sum(axis=1), dtype=np.float32).ravel()
        row_sums[row_sums == 0] = 1.0
        return sp.diags(1.0 / row_sums, format="csr") @ mat.tocsr()

    conn_joint = alpha * _row_norm(conn_spatial) + (1.0 - alpha) * _row_norm(conn_expr)

    # ── Store blended graph temporarily for sc.tl.umap ───────────────────────
    conn_joint_key = f"{joint_key}_connectivities"
    adata.obsp[conn_joint_key] = conn_joint.astype(np.float32)
    adata.uns[joint_key] = {
        "connectivities_key": conn_joint_key,
        "distances_key": adata.uns[spatial_neighbors_key].get(
            "distances_key", f"{spatial_neighbors_key}_distances"
        ),
        "params": {
            "alpha": alpha,
            "expr_neighbors_key": expr_neighbors_key,
            "spatial_neighbors_key": spatial_neighbors_key,
        },
    }

    log.info(
        "spatial_umap: blended graph (alpha=%.2f spatial / %.2f expression), "
        "%d non-zeros",
        alpha, 1.0 - alpha, conn_joint.nnz,
    )

    # ── Run UMAP ──────────────────────────────────────────────────────────────
    sc.tl.umap(
        adata,
        neighbors_key=joint_key,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        random_state=random_state,
        **kwargs,
    )

    # Rename obsm['X_umap'] → obsm[key_added] if different
    if key_added != "X_umap" and "X_umap" in adata.obsm:
        adata.obsm[key_added] = adata.obsm.pop("X_umap")

    # Clean up the temporary joint graph from uns (keep obsp for reuse)
    del adata.uns[joint_key]

    log.info("spatial_umap: embedding written to obsm['%s']", key_added)