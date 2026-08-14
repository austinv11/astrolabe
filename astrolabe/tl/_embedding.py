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

__all__ = ["pca", "umap"]

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
    >>> ab.tl.transcriptional_neighbors(mdata)  # graph over the BANKSY-PCA embedding
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
            raise KeyError(f"obsm['{use_rep}'] not found")
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
    neighbors_key: str = "neighbors",
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    random_state: int = 0,
    key_added: str = "X_umap",
    **kwargs,
) -> None:
    """Compute UMAP embedding of the neighborhood graph.

    Stores the embedding in ``obsm[key_added]``.  Requires that
    :func:`astrolabe.tl.spatial_neighbors` or ``astrolabe.tl.transcriptional_neighbors`` has
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
