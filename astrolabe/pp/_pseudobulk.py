"""Raw-count aggregation with explicit biological replication."""
from __future__ import annotations
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from astrolabe._utils import with_adata


def _count_matrix(adata, layer):
    mat = adata.layers[layer] if layer is not None else adata.X
    values = mat.data if sp.issparse(mat) else np.asarray(mat)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("pseudobulk requires finite nonnegative integer raw counts")
    if values.size and values.max() > np.iinfo(np.int64).max:
        raise ValueError("counts exceed int64 range")
    return sp.csr_matrix(mat, dtype=np.int64)


@with_adata
def pseudobulk(adata: ad.AnnData, *, sample_key, groupby=None, layer="counts",
               metadata_keys=(), library_key="auto", min_cells=10):
    """Sum raw counts per biological sample and optional type/niche/ROI.

    Technical sections should share ``sample_key``. Metadata must be constant
    within each sample. ``layers['measured']`` distinguishes assayed features
    from structural zeros after outer concatenation; inference uses the common
    measured panel. Small groups are retained with ``obs['eligible']=False``.
    """
    groups = [] if groupby is None else [groupby] if isinstance(groupby, str) else list(groupby)
    keys = [sample_key, *groups]
    if min_cells < 1 or adata.obs[keys].isna().any().any():
        raise ValueError("grouping labels must be present and min_cells positive")
    for key in metadata_keys:
        if adata.obs[key].isna().any() or (adata.obs.groupby(sample_key, observed=True)[key].nunique() > 1).any():
            raise ValueError(f"{key} must be present and constant within biological samples")
    counts = _count_matrix(adata, layer)
    rows, metadata, coverage = [], [], []
    panel = adata.uns.get("measured_features")
    if library_key == "auto":
        library_key = adata.uns.get("measured_features_library_key", "library_id")
    if panel is not None:
        from astrolabe._coverage import _coverage
        _coverage(adata, adata.var_names)  # Validate metadata without requiring a common panel.
        if library_key not in adata.obs or adata.obs[library_key].isna().any():
            raise ValueError("measured panel metadata requires complete library labels")
        if not adata.obs[library_key].isin(panel.columns).all():
            raise ValueError("measured panel metadata is missing selected libraries")
    for labels, idx in adata.obs.groupby(keys, observed=True, sort=False).indices.items():
        labels = labels if isinstance(labels, tuple) else (labels,)
        values = np.asarray(counts[idx].sum(axis=0)).ravel()
        if (values < 0).any():
            raise OverflowError("pseudobulk count aggregation overflowed int64")
        rows.append(sp.csr_matrix(values))
        info = dict(zip(keys, labels)); info.update(n_cells=len(idx), eligible=len(idx) >= min_cells)
        for key in metadata_keys:
            info[key] = adata.obs.iloc[idx[0]][key]
        metadata.append(info)
        if panel is not None:
            if library_key not in adata.obs:
                raise ValueError("measured panel metadata requires library labels")
            libs = adata.obs.iloc[idx][library_key].unique()
            coverage.append(panel.reindex(adata.var_names).loc[:, libs].all(axis=1).to_numpy())
        else:
            coverage.append(np.ones(adata.n_vars, dtype=bool))
    if not rows:
        raise ValueError("no pseudobulk groups")
    obs = pd.DataFrame(metadata, index=[f"pb{i}" for i in range(len(rows))])
    result = ad.AnnData(sp.vstack(rows).tocsr(), obs=obs, var=adata.var.copy())
    result.layers["counts"] = result.X.copy()
    result.layers["measured"] = sp.csr_matrix(np.asarray(coverage, dtype=bool))
    result.uns["pseudobulk"] = {"sample_key": sample_key, "groupby": groups, "layer": layer,
                                "min_cells": min_cells, "unit": "biological sample"}
    return result
