"""
Harmony batch correction for PCA or BANKSY embeddings.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np

from astrolabe._gpu import _is_cupy, _require_rapids, _to_cpu_arr, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["harmony"]

log = logging.getLogger(__name__)


@with_adata
def harmony(
    adata: ad.AnnData,
    batch_key: str,
    *,
    basis: str = "X_pca",
    key_added: str = "X_pca_harmony",
    max_iter_harmony: int = 10,
    random_state: int = 0,
    **kwargs,
) -> None:
    """Harmony batch correction on a PCA or BANKSY embedding.

    Applies `Harmony <https://github.com/lilab-bcb/harmonypy>`_ to the
    embedding in ``obsm[basis]``, integrating cells across batches while
    preserving biological variation.  The corrected embedding is stored in
    ``obsm[key_added]`` and can be used directly for clustering or UMAP.

    Requires ``harmonypy`` (``uv add harmonypy``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    batch_key
        ``obs`` column containing batch / sample labels.
    basis
        ``obsm`` key for the embedding to correct.  Typically ``'X_pca'``
        (after :func:`~astrolabe.tl.pca`) or ``'X_pca'`` computed from the
        BANKSY matrix.
    key_added
        ``obsm`` key for the Harmony-corrected embedding.
    max_iter_harmony
        Maximum Harmony iterations.
    random_state
        Random seed passed to Harmony.
    **kwargs
        Additional keyword arguments forwarded to
        ``harmonypy.run_harmony``.

    Examples
    --------
    >>> ab.tl.pca(mdata)
    >>> ab.pp.harmony(mdata, batch_key="sample")
    >>> sc.pp.neighbors(mdata["xenium"], use_rep="X_pca_harmony")
    >>> ab.tl.cluster(mdata)

    >>> # BANKSY + Harmony for multi-sample domain segmentation
    >>> ab.pp.banksy_matrix(mdata, lambda_param=0.8, agf=True)
    >>> ab.tl.pca(mdata, use_rep="X_banksy")
    >>> ab.pp.harmony(mdata, batch_key="sample", basis="X_pca",
    ...               key_added="X_banksy_pca_harmony")
    """
    if gpu_enabled():
        rsc = _require_rapids()
        if basis not in adata.obsm:
            raise KeyError(f"obsm['{basis}'] not found. Run ab.tl.pca first.")
        if batch_key not in adata.obs:
            raise KeyError(f"obs['{batch_key}'] not found.")
        if random_state != 0:
            log.debug("harmony GPU: random_state=%d ignored (rsc.pp.harmony_integrate does not expose it)", random_state)
        import cupy as cp
        moved = not _is_cupy(adata.obsm[basis])
        if moved:
            adata.obsm[basis] = cp.asarray(adata.obsm[basis])
        try:
            rsc.pp.harmony_integrate(
                adata,
                key=batch_key,
                basis=basis,
                adjusted_basis=key_added,
                max_iter_harmony=max_iter_harmony,
            )
            if moved and key_added in adata.obsm:
                adata.obsm[key_added] = _to_cpu_arr(adata.obsm[key_added])
        finally:
            if moved:
                adata.obsm[basis] = _to_cpu_arr(adata.obsm[basis])
        log.info("harmony (GPU): done → obsm['%s']", key_added)
        return

    try:
        import harmonypy
    except ImportError as exc:
        raise ImportError(
            "harmonypy is required for batch correction: uv add harmonypy"
        ) from exc

    if basis not in adata.obsm:
        raise KeyError(f"obsm['{basis}'] not found. Run ab.tl.pca first.")
    if batch_key not in adata.obs:
        raise KeyError(f"obs['{batch_key}'] not found.")

    embedding = np.asarray(adata.obsm[basis], dtype=np.float64)

    log.info(
        "harmony: correcting '%s' (%d cells, batch='%s', n_components=%d)",
        basis, adata.n_obs, batch_key, embedding.shape[1],
    )
    ho = harmonypy.run_harmony(
        embedding,
        adata.obs[[batch_key]],
        batch_key,
        max_iter_harmony=max_iter_harmony,
        random_state=random_state,
        **kwargs,
    )

    adata.obsm[key_added] = ho.Z_corr.T.astype(np.float32)
    log.info("harmony: done → obsm['%s']", key_added)
