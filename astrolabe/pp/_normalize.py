"""
Normalization helpers — thin wrappers around scanpy to keep the astrolabe
namespace consistent.  We re-export rather than re-implement to avoid
divergence with scanpy's well-tested implementations.
"""

from __future__ import annotations

import anndata as ad
import scanpy as sc

from astrolabe._utils import with_adata

__all__ = ["normalize_total", "log1p"]


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
    adata
        AnnData with raw counts in ``X`` (or ``layer`` if specified).
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
    """Logarithmize the data matrix: X = log(X + 1).

    Delegates to :func:`scanpy.pp.log1p`.  Operates in place.

    Parameters
    ----------
    adata
        AnnData.  Should be normalized first (e.g. via
        :func:`normalize_total`).
    base
        Base of the logarithm.  ``None`` uses the natural log.
    layer
        Apply to this layer rather than ``X``.
    obsm
        Apply to ``obsm[obsm]`` rather than ``X``.
    """
    sc.pp.log1p(adata, base=base, layer=layer, obsm=obsm)