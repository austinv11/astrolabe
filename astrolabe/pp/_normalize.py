"""
Normalization helpers — thin wrappers around scanpy to keep the astrolabe
namespace consistent.  We re-export rather than re-implement to avoid
divergence with scanpy's well-tested implementations.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["normalize_total", "log1p", "normalize_by_area", "highly_variable_genes"]


@with_adata
def normalize_total(
    adata: ad.AnnData,
    *,
    target_sum: float | None = None,
    exclude_highly_expressed: bool = False,
    max_fraction: float = 0.05,
    key_added: str | None = "total_counts_norm",
    layer: str | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """Normalize each cell to the same total count.

    Delegates to :func:`scanpy.pp.normalize_total` with astrolabe-idiomatic
    defaults.  After normalization, ``obs['total_counts_norm']`` stores the
    per-cell normalization factor (library size).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with raw counts
        in ``X`` (or ``layer`` if specified).
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    target_sum
        Each cell is normalized to this total.  ``None`` normalizes to the
        median library size across cells, which preserves the scale of the
        original counts.
    exclude_highly_expressed
        Exclude highly expressed genes from the normalization factor
        computation (see scanpy docs).
    max_fraction
        If ``exclude_highly_expressed``, genes making up more than this
        fraction of total counts in any cell are excluded.
    key_added
        Key in ``obs`` to store the normalization factor per cell.
        ``None`` skips storing.
    layer
        Layer to normalize.  ``None`` normalizes ``X``.
    inplace
        Modify in place (default ``True``).

    Returns
    -------
    None or AnnData
    """
    if gpu_enabled():
        rsc = _require_rapids()
        with _gpu_X(adata):
            return rsc.pp.normalize_total(
                adata,
                target_sum=target_sum,
                exclude_highly_expressed=exclude_highly_expressed,
                max_fraction=max_fraction,
                key_added=key_added,
                layer=layer,
                inplace=inplace,
            )
    return sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
        exclude_highly_expressed=exclude_highly_expressed,
        max_fraction=max_fraction,
        key_added=key_added,
        layer=layer,
        inplace=inplace,
    )


@with_adata
def log1p(
    adata: ad.AnnData,
    *,
    base: float | None = None,
    layer: str | None = None,
    obsm: str | None = None,
) -> None:
    """Log-transform the data matrix in place: X = log(X + 1).

    Delegates to :func:`scanpy.pp.log1p`.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should be
        normalized first (e.g. via :func:`normalize_total`).
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    base
        Base of the logarithm.  ``None`` uses the natural log.
    layer
        Apply to this layer rather than ``X``.
    obsm
        Apply to ``obsm[obsm]`` rather than ``X``.
    """
    if gpu_enabled() and obsm is None:
        rsc = _require_rapids()
        with _gpu_X(adata):
            rsc.pp.log1p(adata, base=base, layer=layer)
        return
    sc.pp.log1p(adata, base=base, layer=layer, obsm=obsm)


@with_adata
def normalize_by_area(
    adata: ad.AnnData,
    *,
    area_key: str = "cell_area",
    layer: str | None = None,
    layer_added: str | None = None,
) -> None:
    """Divide log-transformed expression by cell area to correct for cell size.

    Large cells capture more transcripts simply because they have more
    cytoplasm.  This function divides each cell's log-expression vector by
    its area (in µm²), producing values that remain non-negative:

    .. math::

        X_{\\text{corr}} = X \\;/\\; \\text{area}

    .. note::
        Run **after** :func:`normalize_total` and :func:`log1p`.
        Requires ``obs[area_key]`` (written by
        :func:`~astrolabe.pp.calculate_qc_metrics`).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    area_key
        ``obs`` column containing cell area in µm²
        (default ``'cell_area'``).
    layer
        Layer to read from.  ``None`` reads ``X``.
    layer_added
        Layer to write corrected values into.  ``None`` overwrites the
        source (``X`` or *layer*).

    Examples
    --------
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.pp.normalize_by_area(mdata)
    """
    if area_key not in adata.obs.columns:
        raise KeyError(
            f"obs['{area_key}'] not found.  "
            "Run ab.pp.calculate_qc_metrics() first, or supply area_key=."
        )

    area = adata.obs[area_key].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(area) | (area <= 0)
    if bad.any():
        raise ValueError(
            f"{bad.sum()} cells have non-positive or NaN '{area_key}' values. "
            "Filter them before area normalization."
        )

    scale = area.astype(np.float32)

    mat = adata.layers[layer] if layer is not None else adata.X

    if sp.issparse(mat):
        # Divide each row by its scalar — stays sparse (zeros remain zero).
        # Use a diagonal matrix multiply: D^{-1} @ X.
        inv_diag = sp.diags(1.0 / scale, format="csr")
        result = inv_diag @ mat.tocsr()
    else:
        result = np.asarray(mat, dtype=np.float32) / scale[:, np.newaxis]

    target_layer = layer_added if layer_added is not None else layer
    if target_layer is None:
        adata.X = result
    else:
        adata.layers[target_layer] = result


@with_adata
def highly_variable_genes(
    adata: ad.AnnData,
    *,
    n_top_genes: int | None = 1000,
    flavor: str = "seurat",
    min_mean: float = 0.0125,
    max_mean: float = 3.0,
    min_disp: float = 0.5,
    layer: str | None = None,
    batch_key: str | None = None,
    subset: bool = False,
    inplace: bool = True,
    **kwargs,
) -> ad.AnnData | None:
    """Select highly variable genes.

    Delegates to :func:`scanpy.pp.highly_variable_genes`.  Results are
    written to ``var['highly_variable']``, ``var['means']``,
    ``var['dispersions']``, and ``var['dispersions_norm']``.

    Subsequent calls to :func:`~astrolabe.tl.pca` will automatically
    restrict to the selected genes via ``var['highly_variable']``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should be
        normalized and log-transformed first.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    n_top_genes
        Number of highly variable genes to keep.  When set, ``min_mean``,
        ``max_mean``, and ``min_disp`` are ignored.  Defaults to ``2000``.
    flavor
        Method for identifying variable genes.  ``'seurat'`` (default)
        expects normalized, log-transformed data — matching the
        ``normalize_total`` → ``log1p`` → ``highly_variable_genes`` order
        used throughout astrolabe.  ``'seurat_v3'`` fits a regularized
        model on **raw counts** instead and must be run before
        normalization/log-transformation (pass ``layer=`` pointing at a
        raw-counts layer, or call this before ``normalize_total``/``log1p``,
        if you want that flavor).  Other option: ``'cell_ranger'`` (also
        expects log-transformed data, like ``'seurat'``).
    min_mean
        Minimum mean expression.  Only used when ``n_top_genes`` is ``None``.
    max_mean
        Maximum mean expression.  Only used when ``n_top_genes`` is ``None``.
    min_disp
        Minimum normalized dispersion.  Only used when ``n_top_genes`` is ``None``.
    layer
        Layer to use for computing variability.  ``None`` uses ``X``.
    batch_key
        ``obs`` column to use for per-batch HVG selection.  Genes are ranked
        within each batch and the top ``n_top_genes`` are taken by union.
    subset
        Subset ``adata`` in place to highly variable genes.  Defaults to
        ``False`` (all genes kept; use ``var['highly_variable']`` to filter).
    inplace
        Write results to ``adata.var`` in place (default ``True``).
    **kwargs
        Additional arguments forwarded to :func:`scanpy.pp.highly_variable_genes`.

    Examples
    --------
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> ab.pp.highly_variable_genes(mdata, n_top_genes=1000)
    >>> ab.tl.pca(mdata)
    """
    if gpu_enabled():
        rsc = _require_rapids()
        with _gpu_X(adata):
            return rsc.pp.highly_variable_genes(
                adata,
                n_top_genes=n_top_genes,
                flavor=flavor,
                min_mean=min_mean,
                max_mean=max_mean,
                min_disp=min_disp,
                layer=layer,
                batch_key=batch_key,
                subset=subset,
                inplace=inplace,
                **kwargs,
            )
    return sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        min_mean=min_mean,
        max_mean=max_mean,
        min_disp=min_disp,
        layer=layer,
        batch_key=batch_key,
        subset=subset,
        inplace=inplace,
        **kwargs,
    )