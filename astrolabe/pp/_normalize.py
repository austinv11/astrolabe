"""
Normalization helpers — thin wrappers around scanpy to keep the astrolabe
namespace consistent.  We re-export rather than re-implement to avoid
divergence with scanpy's well-tested implementations.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from astrolabe._gpu import _gpu_X, _require_rapids, gpu_enabled
from astrolabe._utils import with_adata

__all__ = ["normalize_total", "log1p", "normalize_by_area", "highly_variable_genes"]

log = logging.getLogger(__name__)


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
    log1p: bool = True,
) -> None:
    """Area-density normalization of raw counts — an *alternative* to
    :func:`normalize_total`, never a step after it.

    Divides each cell's **raw counts** by its segmented area (µm²), rescales
    by the median cell area so values stay in a familiar count-like range,
    and (by default) applies ``log1p``:

    .. math::

        X_{\\text{corr}} = \\log 1p\\!\\left(
            \\frac{\\text{counts}}{\\text{area}} \\times
            \\operatorname{median}(\\text{area})\\right)

    Use it when "how much transcript was captured per unit of segmented
    cell" is the quantity of interest — e.g. correlating expression with
    per-cell image intensities (which are themselves area densities), or
    comparing large vs. small cells on an equal footing.  The standard
    scanpy pipeline (:func:`normalize_total` + :func:`log1p` on ``X``)
    remains the right input for PCA/clustering/DE.

    .. warning::

        **Never apply this on top of library-size-normalized data.** After
        :func:`normalize_total`, every cell's linear-scale total is the same
        constant, so dividing by area forces *every* gene's value in a cell
        to scale as ``1/area`` — a pure artifact: a gene whose expression
        *fraction* is size-independent acquires a strong spurious negative
        area dependence.  Area normalization and library-size normalization
        are competing answers to the same question (what is each cell's
        "size" denominator?); pick one.  This function therefore raises if
        the input looks log1p-transformed (``uns['log1p']`` present) unless
        an explicit raw-counts ``layer`` is passed.

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
        Layer holding **raw counts** to read from.  ``None`` reads ``X``
        (which must then still be raw counts).
    layer_added
        Layer to write corrected values into.  ``None`` overwrites the
        source (``X`` or *layer*).
    log1p
        Apply ``log1p`` to the area-scaled counts (default ``True``).
        Pass ``False`` to keep linear densities.

    Examples
    --------
    >>> mdata['xenium'].layers['counts'] = mdata['xenium'].X.copy()  # before normalize_total
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> # area-normalized values from the RAW counts, in a separate layer:
    >>> ab.pp.normalize_by_area(mdata, layer='counts', layer_added='area_norm')
    """
    if area_key not in adata.obs.columns:
        raise KeyError(
            f"obs['{area_key}'] not found.  "
            "Run ab.pp.calculate_qc_metrics() first, or supply area_key=."
        )
    if layer is None and "log1p" in adata.uns:
        raise ValueError(
            "adata.X appears to be log1p-transformed (uns['log1p'] is set). "
            "normalize_by_area is an alternative to library-size "
            "normalization and must run on raw counts — stash them first "
            "(adata.layers['counts'] = adata.X.copy() before normalize_total) "
            "and pass layer='counts'."
        )

    area = adata.obs[area_key].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(area) | (area <= 0)
    if bad.any():
        raise ValueError(
            f"{bad.sum()} cells have non-positive or NaN '{area_key}' values. "
            "Filter them before area normalization."
        )

    # counts / area, rescaled by the median area so the output stays in a
    # count-like range ("counts per median-sized cell").
    scale = (area / np.median(area)).astype(np.float64)

    mat = adata.layers[layer] if layer is not None else adata.X

    if sp.issparse(mat):
        inv_diag = sp.diags(1.0 / scale, format="csr")
        result = (inv_diag @ mat.tocsr().astype(np.float64))
        if log1p:
            result = result.log1p()
        result = result.astype(np.float32)
    else:
        result = np.asarray(mat, dtype=np.float64) / scale[:, np.newaxis]
        if log1p:
            result = np.log1p(result)
        result = result.astype(np.float32)

    target_layer = layer_added if layer_added is not None else layer
    if target_layer is None:
        adata.X = result
    else:
        adata.layers[target_layer] = result


@with_adata
def highly_variable_genes(
    adata: ad.AnnData,
    *,
    n_top_genes: int | None = None,
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
        ``max_mean``, and ``min_disp`` are ignored.  Defaults to ``None``
        (scanpy's own default): genes are selected by the dispersion
        cutoffs instead of a fixed count — the right behavior for targeted
        spatial panels of a few hundred genes, where a fixed
        1000/2000-gene budget either selects everything (a silent no-op)
        or is meaningless.  If you pass a value at or above the panel
        size, every gene is flagged highly variable and a warning is
        logged.
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
    >>> ab.pp.highly_variable_genes(mdata)
    >>> ab.tl.pca(mdata)
    """
    if n_top_genes is not None and n_top_genes >= adata.n_vars:
        log.warning(
            "highly_variable_genes: n_top_genes=%d >= n_vars=%d — every gene "
            "will be flagged highly variable (selection is a no-op on this "
            "panel). Use n_top_genes=None to select by dispersion cutoffs "
            "instead.",
            n_top_genes, adata.n_vars,
        )
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