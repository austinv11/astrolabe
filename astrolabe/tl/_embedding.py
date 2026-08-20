"""
Dimensionality reduction embeddings.  Thin wrappers around scanpy.
"""

from __future__ import annotations

import logging

import anndata as ad
import scanpy as sc

from astrolabe._gpu import _gpu_X, _require_rapids, _to_cpu_arr, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["pca", "umap"]

log = logging.getLogger(__name__)


# Sentinel default for mask_var: "auto" defers to scanpy's own behavior
# (use var['highly_variable'] when the column exists, else no mask).
_MASK_VAR_AUTO = "auto"


def _is_auto(mask_var) -> bool:
    return isinstance(mask_var, str) and mask_var == _MASK_VAR_AUTO


@with_adata
def pca(
    adata: ad.AnnData,
    *,
    n_comps: int | None = None,
    use_rep: str | None = None,
    mask_var=_MASK_VAR_AUTO,
    layer: str | None = None,
    zero_center: bool = True,
    svd_solver: str | None = None,
    random_state: int = 0,
    key_added: str | None = None,
    **kwargs,
) -> None:
    """Compute PCA on the expression matrix or an arbitrary ``obsm`` embedding.

    Delegates to :func:`scanpy.pp.pca`.  When ``use_rep`` is ``None``,
    results are stored in ``obsm[key_added or 'X_pca']``, ``varm['PCs']``,
    ``uns[key_added or 'pca']``.  When ``use_rep`` is set (e.g. the BANKSY
    matrix), scanpy's ``obsm=`` input path is used instead: the embedding
    lands in ``obsm[key_added or 'X_pca']`` and the loadings /
    ``variance_ratio`` in ``uns[key_added or 'pca']`` (``varm`` is not
    written, since the input dimensions are not genes).

    If ``var['highly_variable']`` is present (written by
    :func:`~astrolabe.pp.highly_variable_genes`), PCA is restricted to those
    genes unless ``mask_var=None`` is passed explicitly.  (Never applied
    when ``use_rep`` is set.)

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should be
        normalized and log-transformed first.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_comps
        Number of principal components.  ``None`` (default) lets scanpy
        adapt: ``min(50, min_dim - 1)`` — safe for targeted panels with
        fewer than 50 genes, where a hardcoded 50 would raise.
    use_rep
        Key in ``obsm`` to use as input instead of ``X`` / ``layer``.
        Useful for running PCA on the BANKSY matrix or any other precomputed
        embedding.  ``zero_center``, ``svd_solver``, ``random_state``, and
        ``key_added`` are all honored on this path; ``layer`` and
        ``mask_var`` are gene-space concepts and raise if combined with it.
    mask_var
        Boolean var mask, ``var`` column name, or ``None``.  The default
        (``'auto'``) defers to scanpy: restrict to ``var['highly_variable']``
        when that column exists, otherwise use all genes.  Pass ``None`` to
        disable HVG masking explicitly.  (Replaces scanpy's deprecated
        ``use_highly_variable``.)
    layer
        Layer to use.  ``None`` uses ``X``.  Incompatible with ``use_rep``.
    zero_center
        Subtract the mean before computing SVD.
    svd_solver
        SVD solver.  ``None`` (default) lets scanpy choose automatically
        based on data size and sparsity; ``'arpack'`` / ``'randomized'``
        force a specific solver.  Ignored (with a warning) on the GPU path.
    random_state
        Random seed.
    key_added
        Output key in ``obsm`` (and prefix for ``varm``/``uns``).  ``None``
        defaults to ``'X_pca'`` / ``'PCs'`` / ``'pca'``.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pp.pca`.

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
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(f"obsm['{use_rep}'] not found")
        if layer is not None:
            raise ValueError("layer= and use_rep= are mutually exclusive")
        if not _is_auto(mask_var) and mask_var is not None:
            raise ValueError(
                "mask_var= is a gene-space mask and cannot be combined with "
                "use_rep= (obsm input columns are not genes)"
            )

    if use_rep is None and gpu_enabled():
        rsc = _require_rapids()
        out_key = key_added or "X_pca"
        if svd_solver is not None:
            log.warning("pca: svd_solver=%r is ignored on the GPU path", svd_solver)
        gpu_kwargs = dict(kwargs)
        if not _is_auto(mask_var):
            gpu_kwargs["mask_var"] = mask_var
        with _gpu_X(adata) as moved:
            rsc.pp.pca(
                adata,
                n_comps=n_comps,
                layer=layer,
                zero_center=zero_center,
                random_state=random_state,
                key_added=key_added,
                **gpu_kwargs,
            )
            if moved:
                if out_key in adata.obsm:
                    adata.obsm[out_key] = _to_cpu_arr(adata.obsm[out_key])
                if "PCs" in adata.varm:
                    adata.varm["PCs"] = _to_cpu_arr(adata.varm["PCs"])
        return

    pca_kwargs = dict(
        n_comps=n_comps,
        layer=layer,
        obsm=use_rep,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
        key_added=key_added,
        **kwargs,
    )
    if use_rep is not None:
        # An HVG mask over var makes no sense for obsm input — and scanpy's
        # auto-mask default would try to apply one if var['highly_variable']
        # exists, then fail on the dimension mismatch.
        pca_kwargs["mask_var"] = None
    elif not _is_auto(mask_var):
        pca_kwargs["mask_var"] = mask_var

    sc.pp.pca(adata, **pca_kwargs)

    if use_rep is not None:
        out_key = key_added or "X_pca"
        log.info(
            "pca (use_rep='%s'): %d components → obsm['%s']",
            use_rep, adata.obsm[out_key].shape[1], out_key,
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
        # rsc.tl.umap may not accept key_added= — always let it write the
        # default obsm['X_umap'], then move the result if a custom key was
        # requested.
        rsc.tl.umap(
            adata,
            neighbors_key=neighbors_key,
            min_dist=min_dist,
            spread=spread,
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )
        if key_added != "X_umap" and "X_umap" in adata.obsm:
            adata.obsm[key_added] = adata.obsm.pop("X_umap")
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
