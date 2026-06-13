"""
Differential expression and cluster annotation — thin wrappers around scanpy.
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import scanpy as sc

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["rank_genes_groups", "score_genes", "annotate_clusters"]

log = logging.getLogger(__name__)


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
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should contain
        log-normalized counts in ``X`` (run
        :func:`astrolabe.pp.normalize_total` + :func:`astrolabe.pp.log1p`
        first).
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
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
    if gpu_enabled():
        rsc = _require_rapids()
        with _gpu_X(adata):
            rsc.tl.rank_genes_groups(
                adata,
                groupby=groupby,
                groups=groups,
                reference=reference,
                method=method,
                n_genes=n_genes,
                key_added=key_added,
                **kwargs,
            )
        return
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


@with_adata
def score_genes(
    adata: ad.AnnData,
    gene_list: list[str],
    *,
    ctrl_size: int = 50,
    gene_pool: list[str] | None = None,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int = 0,
    use_raw: bool | None = None,
    layer: str | None = None,
    **kwargs,
) -> None:
    """Score cells against a gene signature using the Seurat method.

    Wraps :func:`scanpy.tl.score_genes`.  For each cell, computes the
    average expression of *gene_list* minus the average expression of a
    randomly sampled control gene set of comparable expression level.
    Results are stored as ``obs[score_name]``.

    Typical use cases include scoring tumour hallmark gene sets, cell-cycle
    genes, or any published signature against your data before downstream
    clustering or plotting.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.  Should be
        log-normalized (run :func:`astrolabe.pp.normalize_total` +
        :func:`astrolabe.pp.log1p` first).
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    gene_list
        Genes in the signature to score.  Genes absent from ``var_names``
        are silently ignored by scanpy.
    ctrl_size
        Number of control genes sampled per expression bin for background
        estimation.
    gene_pool
        Universe of genes to sample controls from.  ``None`` uses all
        genes in ``var_names``.
    n_bins
        Number of expression bins for background control matching.
    score_name
        Name of the ``obs`` column written with per-cell scores.
    random_state
        Random seed for control gene sampling.
    use_raw
        Use ``adata.raw`` if available.  ``None`` defers to scanpy's default.
    layer
        Expression layer to use.  ``None`` uses ``X``.
    **kwargs
        Additional arguments forwarded to :func:`scanpy.tl.score_genes`.

    Examples
    --------
    >>> hallmark_prolif = ["MKI67", "TOP2A", "PCNA", "MCM2", "CDK1"]
    >>> ab.tl.score_genes(mdata, hallmark_prolif, score_name="proliferation_score")
    >>> ab.pl.points(mdata, color="proliferation_score").show()
    """
    if gpu_enabled():
        rsc = _require_rapids()
        with _gpu_X(adata):
            rsc.tl.score_genes(
                adata,
                gene_list=gene_list,
                ctrl_size=ctrl_size,
                gene_pool=gene_pool,
                n_bins=n_bins,
                score_name=score_name,
                random_state=random_state,
                use_raw=use_raw,
                layer=layer,
                **kwargs,
            )
        log.info("score_genes (GPU): wrote obs['%s']", score_name)
        return
    sc.tl.score_genes(
        adata,
        gene_list=gene_list,
        ctrl_size=ctrl_size,
        gene_pool=gene_pool,
        n_bins=n_bins,
        score_name=score_name,
        random_state=random_state,
        use_raw=use_raw,
        layer=layer,
        **kwargs,
    )
    log.info("score_genes: wrote obs['%s']", score_name)


@with_adata
def annotate_clusters(
    adata: ad.AnnData,
    groupby: str,
    annotation: dict[str, str],
    *,
    key_added: str | None = None,
    unknown_label: str = "unknown",
) -> None:
    """Map cluster IDs to cell type labels and write them to ``obs``.

    After running :func:`~astrolabe.tl.cluster` and identifying marker genes
    (e.g. via :func:`~astrolabe.tl.rank_genes_groups` or
    :func:`~astrolabe.pl.dotplot`), pass a dict of cluster → cell type to
    annotate all cells at once.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    groupby
        ``obs`` column containing cluster labels (e.g. ``'leiden'``).
    annotation
        Mapping of cluster label → cell type name.  Clusters absent from
        the dict receive *unknown_label*.
    key_added
        ``obs`` column name for the annotation.  Defaults to
        ``'{groupby}_annotation'``.
    unknown_label
        Label assigned to clusters not present in *annotation*.

    Examples
    --------
    >>> ab.tl.cluster(mdata, resolution=0.5)
    >>> ab.tl.rank_genes_groups(mdata, groupby="leiden")
    >>> ab.pl.dotplot(mdata, marker_genes, groupby="leiden").show()
    >>> ab.tl.annotate_clusters(mdata, "leiden", {
    ...     "0": "T cell",
    ...     "1": "Epithelial",
    ...     "2": "Macrophage",
    ... })
    >>> ab.pl.points(mdata, color="leiden_annotation").show()
    """
    key_added = key_added or f"{groupby}_annotation"

    if groupby not in adata.obs.columns:
        raise KeyError(
            f"obs column '{groupby}' not found.  "
            "Run ab.tl.cluster() first, or pass the correct groupby= key."
        )

    adata.obs[key_added] = (
        adata.obs[groupby]
        .astype(str)
        .map(annotation)
        .fillna(unknown_label)
        .astype("category")
    )

    n_annotated = (adata.obs[key_added] != unknown_label).sum()
    n_unknown = (adata.obs[key_added] == unknown_label).sum()
    log.info(
        "annotate_clusters: wrote obs['%s'] — %d annotated, %d '%s'",
        key_added,
        n_annotated,
        n_unknown,
        unknown_label,
    )