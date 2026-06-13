"""
BANKSY multi-λ / multi-resolution parameter sweep.

Runs the full BANKSY pipeline (banksy_matrix → PCA → neighbors → Leiden) at
every combination of lambda_params × resolutions, storing per-run cluster
labels in obs and a summary DataFrame in uns.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from astrolabe._utils import with_adata

__all__ = ["banksy_sweep"]

log = logging.getLogger(__name__)


@with_adata
def banksy_sweep(
    adata: ad.AnnData,
    *,
    lambda_params: list[float] = (0.0, 0.2, 0.5, 0.8),
    resolutions: list[float] = (0.3, 0.5, 1.0),
    n_pcs: int = 30,
    n_neighbors: int = 15,
    agf: bool = False,
    agf_weight: float = 0.5,
    neighbors_key: str = "spatial_neighbors",
    spatial_key: str = "spatial",
    layer: str | None = None,
    random_state: int = 0,
    key_added: str = "banksy_sweep",
) -> pd.DataFrame:
    """Run the BANKSY pipeline across a grid of λ values and Leiden resolutions.

    For each combination of ``lambda_params × resolutions``:

    1. :func:`~astrolabe.pp.banksy_matrix` — build the BANKSY feature matrix.
    2. PCA on the BANKSY matrix (``n_pcs`` components).
    3. ``sc.pp.neighbors`` on the BANKSY PCA embedding.
    4. ``sc.tl.leiden`` at the given resolution.

    Cluster labels are written to ``obs`` as
    ``{key_added}_L{lambda}_R{resolution}`` for each run.  A summary
    DataFrame is stored in ``uns[key_added]`` and returned.

    Low λ (≈0.0–0.2) recovers transcriptomic cell types; high λ (≈0.5–0.8)
    recovers spatially-coherent tissue domains.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with normalised
        expression and a spatial graph in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    lambda_params
        λ values to sweep.  Each controls the self-vs-neighbourhood weight.
    resolutions
        Leiden resolution values to sweep for each λ.
    n_pcs
        Number of PCA components for the BANKSY embedding.
    n_neighbors
        Neighbours used when building the KNN graph on the BANKSY PCA
        embedding for Leiden.
    agf
        Include azimuthal Gabor features in the BANKSY matrix.
    agf_weight
        Fraction of λ allocated to |M1| when ``agf=True``.
    neighbors_key
        Prefix for the spatial connectivity in ``obsp``.
    spatial_key
        ``obsm`` key for micron coordinates (required when ``agf=True``).
    layer
        Expression layer for the BANKSY matrix.  ``None`` uses ``adata.X``.
    random_state
        Random seed for PCA and Leiden.
    key_added
        Prefix for obs columns and the uns summary key.

    Returns
    -------
    pd.DataFrame
        One row per (λ, resolution) run with columns ``lambda``,
        ``resolution``, ``n_clusters``, ``obs_key``.

    Examples
    --------
    >>> summary = ab.tl.banksy_sweep(mdata)
    >>> summary
    # lambda  resolution  n_clusters  obs_key
    # 0.0     0.3         8           banksy_sweep_L0.0_R0.3
    # ...

    >>> # inspect a specific result
    >>> ab.pl.points(mdata, color="banksy_sweep_L0.8_R0.5")
    """
    from astrolabe.pp._banksy import banksy_matrix

    n_pcs_eff = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    records = []

    for lam in lambda_params:
        lam_tag = f"{lam:.2f}".rstrip("0").rstrip(".")
        tmp_key = f"_banksy_tmp_{lam_tag}"

        banksy_matrix(
            adata,
            lambda_param=lam,
            agf=agf,
            agf_weight=agf_weight,
            neighbors_key=neighbors_key,
            spatial_key=spatial_key,
            layer=layer,
            key_added=tmp_key,
        )

        from astrolabe.tl._embedding import pca as _pca

        n_pcs_run = min(n_pcs_eff, adata.obsm[tmp_key].shape[1] - 1)
        _pca(adata, n_comps=n_pcs_run, use_rep=tmp_key, random_state=random_state)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca", random_state=random_state)

        for res in resolutions:
            res_tag = f"{res:.2f}".rstrip("0").rstrip(".")
            obs_key = f"{key_added}_L{lam_tag}_R{res_tag}"

            sc.tl.leiden(
                adata,
                resolution=res,
                key_added=obs_key,
                random_state=random_state,
                flavor="igraph",
                directed=False,
                n_iterations=2,
            )
            n_clusters = adata.obs[obs_key].nunique()

            records.append(
                {
                    "lambda": lam,
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "obs_key": obs_key,
                }
            )
            log.info(
                "banksy_sweep: λ=%.2f  res=%.2f  → %d clusters  ('%s')",
                lam, res, n_clusters, obs_key,
            )

        # clean up temporary obsm key
        del adata.obsm[tmp_key]

    df = pd.DataFrame(records)
    adata.uns[key_added] = df
    log.info("banksy_sweep: done — %d runs", len(df))
    return df
