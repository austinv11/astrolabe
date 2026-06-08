"""
Differential expression — thin wrapper around scanpy.
"""

from __future__ import annotations

from typing import Literal

import anndata as ad
import scanpy as sc

from astrolabe._utils import with_adata

__all__ = ["rank_genes_groups"]


@with_adata
def rank_genes_groups(
    adata: ad.AnnData,
    groupby: str,
    *,
    groups: list[str] | Literal["all"] = "all",
    reference: str = "rest",
    method: str | None = None,
    n_genes: int = 50,
    key_added: str = "rank_genes_groups",
    **kwargs,
) -> None:
    """Find marker genes for each cluster via differential expression.

    Wraps :func:`scanpy.tl.rank_genes_groups`.  Results are stored in
    ``adata.uns[key_added]`` in scanpy's standard format and can be
    retrieved with :func:`scanpy.get.rank_genes_groups_df`.

    Parameters
    ----------
    adata
        AnnData.  Should contain log-normalized counts in ``X``
        (run :func:`astrolabe.pp.normalize_total` + :func:`astrolabe.pp.log1p`
        first).
    groupby
        Column in ``obs`` to group by (e.g. ``'leiden'``).
    groups
        Groups to test.  ``'all'`` tests every group vs. the rest (or
        vs. ``reference``).
    reference
        Reference group.  ``'rest'`` compares each group against all
        other cells combined.
    method
        Statistical test.  ``None`` defaults to ``'wilcoxon'`` which is
        non-parametric and appropriate for count-like data.  Other options:
        ``'t-test'``, ``'logreg'``, ``'t-test_overestim_var'``.
    n_genes
        Number of top genes to report per group.
    key_added
        Key under ``uns`` for results.
    **kwargs
        Additional arguments forwarded to ``sc.tl.rank_genes_groups``.
    """
    method = method or "wilcoxon"
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        groups=groups,
        reference=reference,
        method=method,
        n_genes=n_genes,
        key_added=key_added,
        **kwargs,
    )