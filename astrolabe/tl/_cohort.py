"""Replicate-level differential expression and neighborhood contrasts."""
from __future__ import annotations
import itertools
import importlib.metadata
import anndata as ad
import numpy as np
import pandas as pd
from astrolabe._statistics import permutation_p, adjust_p, annotate_permutation_resolution
from astrolabe._utils import with_adata, from_kwarg
from astrolabe.pp._pseudobulk import _count_matrix
from astrolabe.tl._spatial_stats import _get_conn_matrix
from astrolabe.tl._neighborhood import _category_codes


@with_adata(mirror_uns=from_kwarg("key_added"))
def rank_genes_pseudobulk(adata: ad.AnnData, *, condition_key, contrast,
                          covariates=(), paired_key=None, n_cpus=1, key_added="rank_genes_pseudobulk"):
    """PyDESeq2 contrasts on pseudobulk counts, separately for each stratum.

    ``contrast=(tested_level, reference_level)``. Require at least two
    biological replicates in each condition, full design rank and residual
    degrees of freedom. Paired designs include donor as a fixed effect.
    Cells and fields of view are never treated as biological replicates.
    """
    if "pseudobulk" not in adata.uns:
        raise ValueError("run pp.pseudobulk first")
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
        from pydeseq2.default_inference import DefaultInference
        from formulaic import model_matrix
    except ImportError as exc:
        raise ImportError("Install astrolabe[cohort] for pseudobulk DE") from exc
    if len(contrast) != 2 or contrast[0] == contrast[1]:
        raise ValueError("contrast must contain two distinct condition levels")
    sample_key = adata.uns["pseudobulk"]["sample_key"]
    groups = list(adata.uns["pseudobulk"]["groupby"])
    frame = adata.obs.copy()
    keys = [condition_key, *covariates, *([] if paired_key is None else [paired_key])]
    if frame[keys].isna().any().any():
        raise ValueError("design metadata contains missing values")
    selected = frame[condition_key].isin(contrast) & frame["eligible"]
    all_counts = _count_matrix(adata, "counts")
    parts = []
    grouped = frame[selected].groupby(groups, observed=True, sort=False) if groups else [((), frame[selected])]
    for labels, metadata in grouped:
        metadata = metadata.copy()
        for key in metadata:
            if isinstance(metadata[key].dtype, pd.CategoricalDtype):
                metadata[key] = metadata[key].cat.remove_unused_categories()
        if metadata[sample_key].duplicated().any():
            raise ValueError("a biological sample has multiple rows within a stratum")
        if any((metadata[condition_key] == c).sum() < 2 for c in contrast):
            raise ValueError("at least two biological replicates per condition are required")
        if paired_key is not None:
            by_donor = metadata.groupby(paired_key, observed=True)[condition_key].nunique()
            if (by_donor != 2).any() or metadata.groupby([paired_key, condition_key], observed=True).size().max() != 1:
                raise ValueError("paired contrasts require one sample per donor and condition")
        metadata[condition_key] = metadata[condition_key].astype(str)
        if paired_key is not None:
            metadata[paired_key] = metadata[paired_key].astype(str)
        if any(not str(k).isidentifier() for k in keys):
            raise ValueError("design column names must be identifiers")
        formula = "~ " + " + ".join(dict.fromkeys([*covariates, *([] if paired_key is None else [paired_key]), condition_key]))
        design = np.asarray(model_matrix(formula, metadata))
        if np.linalg.matrix_rank(design) != design.shape[1] or len(metadata) <= design.shape[1]:
            raise ValueError("confounded or saturated design has no estimable condition contrast")
        idx = adata.obs_names.get_indexer(metadata.index)
        measured = adata.layers["measured"][idx].toarray().all(axis=0)
        measured &= np.asarray(all_counts[idx].sum(axis=0)).ravel() > 0
        if not measured.any():
            raise ValueError("no commonly measured expressed genes")
        counts = pd.DataFrame(all_counts[idx][:, measured].toarray(), index=metadata.index, columns=adata.var_names[measured])
        inference = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(counts=counts, metadata=metadata, design=formula, refit_cooks=True,
                          size_factors_fit_type="poscounts", inference=inference)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[condition_key, str(contrast[0]), str(contrast[1])], inference=inference)
        stats.summary()
        result = stats.results_df.rename_axis("gene").reset_index()
        labels = labels if isinstance(labels, tuple) else (labels,)
        for key, label in zip(groups, labels):
            result[key] = label
        result["n_replicates_tested"] = int((metadata[condition_key] == str(contrast[0])).sum())
        result["n_replicates_reference"] = int((metadata[condition_key] == str(contrast[1])).sum())
        parts.append(result)
    if not parts:
        raise ValueError("no eligible pseudobulk strata")
    result = pd.concat(parts, ignore_index=True)
    result["padj_global"] = adjust_p(result.pvalue.to_numpy())
    result.attrs.update(backend="PyDESeq2", version=importlib.metadata.version("pydeseq2"),
                        condition_key=condition_key, contrast=list(contrast), paired_key=paired_key,
                        inference_unit="biological sample", panel="common measured within stratum")
    adata.uns[key_added] = result
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def diff_nhood_enrichment(adata: ad.AnnData, groupby: str, *, sample_key, condition_key,
                          contrast, neighbors_key="spatial_neighbors", paired_key=None,
                          block_key=None, n_perms=999, n_boot=999, seed=0,
                          key_added="diff_nhood_enrichment"):
    """Contrast sample-level contact fractions minus finite-population expectation.

    Each biological sample has equal weight. Permute sample conditions within
    blocks, or flip paired donor differences. Confidence intervals bootstrap
    biological samples/donors. This is an association test; supplied blocks
    must justify exchangeability. Technical sections share ``sample_key``.
    """
    if len(contrast) != 2 or contrast[0] == contrast[1] or n_perms < 0 or n_boot < 1:
        raise ValueError("invalid contrast or simulation counts")
    codes, categories = _category_codes(adata, groupby)
    keys = [sample_key, condition_key] + [k for k in (paired_key, block_key) if k is not None]
    if adata.obs[keys].isna().any().any():
        raise ValueError("replicate and condition labels must be present")
    sample_meta = adata.obs[keys].drop_duplicates()
    if sample_meta[sample_key].duplicated().any():
        raise ValueError("condition, donor and block must be constant within biological samples")
    sample_meta = sample_meta.set_index(sample_key)
    sample_meta = sample_meta[sample_meta[condition_key].isin(contrast)]
    for level in contrast:
        if (sample_meta[condition_key] == level).sum() < 2:
            raise ValueError("at least two biological replicates per condition are required")
    conn = _get_conn_matrix(adata, neighbors_key).tocsr(copy=True)
    conn.setdiag(0); conn.eliminate_zeros(); conn.data[:] = 1
    coo = conn.tocoo()
    sample_labels = adata.obs[sample_key].to_numpy()
    if (sample_labels[coo.row] != sample_labels[coo.col]).any():
        raise ValueError("graph contains edges across biological samples")
    pairs = list(itertools.product(range(len(categories)), repeat=2))
    scores = np.full((len(sample_meta), len(pairs)), np.nan)
    for si, sample in enumerate(sample_meta.index):
        idx = np.flatnonzero(sample_labels == sample)
        local = conn[idx][:, idx]
        local_codes = codes[idx]
        degree = np.asarray(local.sum(axis=1)).ravel()
        n = len(idx)
        for pi, (a, b) in enumerate(pairs):
            source = (local_codes == a) & (degree > 0)
            if source.any() and n > 1:
                contacts = np.asarray(local[:, local_codes == b].sum(axis=1)).ravel()
                expected = ((local_codes == b).sum() - (a == b)) / (n-1)
                scores[si, pi] = np.mean(contacts[source] / degree[source]) - expected
    tested = sample_meta[condition_key].to_numpy() == contrast[0]
    rng = np.random.default_rng(seed)
    rows = []
    for pi, (a, b) in enumerate(pairs):
        y = scores[:, pi]
        if paired_key is not None:
            donor_rows = []
            for donor, metadata in sample_meta.groupby(paired_key, observed=True):
                if len(metadata) != 2 or metadata[condition_key].nunique() != 2:
                    raise ValueError("paired test requires one sample per donor and condition")
                if block_key is not None and metadata[block_key].nunique() != 1:
                    raise ValueError("paired condition swaps require both samples in the same block")
                indices = sample_meta.index.get_indexer(metadata.index)
                t = indices[tested[indices]]; r = indices[~tested[indices]]
                donor_rows.append(y[t[0]] - y[r[0]])
            delta = np.asarray(donor_rows)
            valid = np.isfinite(delta)
            delta = delta[valid]
            nt = nr = len(delta)
            effect = delta.mean() if len(delta) >= 2 else np.nan
            null = np.array([(delta * rng.choice([-1, 1], len(delta))).mean() for _ in range(n_perms)]) if len(delta) else np.full(n_perms, np.nan)
            boot = np.array([rng.choice(delta, len(delta)).mean() for _ in range(n_boot)]) if len(delta) else np.full(n_boot, np.nan)
        else:
            mask = np.isfinite(y)
            yy, tt = y[mask], tested[mask]
            nt, nr = int(tt.sum()), int((~tt).sum())
            effect = yy[tt].mean()-yy[~tt].mean() if min(nt, nr) >= 2 else np.nan
            strata = sample_meta.loc[mask, block_key].to_numpy() if block_key else np.zeros(len(yy))
            if block_key and not any(np.unique(tt[strata == v]).size > 1 for v in np.unique(strata)):
                raise ValueError("condition is confounded with exchangeability blocks")
            null = []
            for _ in range(n_perms):
                shuffled = tt.copy()
                for block in np.unique(strata):
                    idx = np.flatnonzero(strata == block); shuffled[idx] = rng.permutation(shuffled[idx])
                null.append(yy[shuffled].mean()-yy[~shuffled].mean() if min(nt,nr) else np.nan)
            if block_key:
                blocks = [np.flatnonzero(strata == block) for block in np.unique(strata)]
                boot = []
                for _ in range(n_boot):
                    selected_blocks = rng.integers(0, len(blocks), len(blocks))
                    indices = np.concatenate([blocks[i] for i in selected_blocks])
                    values, conditions = yy[indices], tt[indices]
                    boot.append(values[conditions].mean()-values[~conditions].mean()
                                if conditions.any() and (~conditions).any() else np.nan)
            else:
                boot = [rng.choice(yy[tt], nt).mean()-rng.choice(yy[~tt], nr).mean() if min(nt,nr) else np.nan for _ in range(n_boot)]
            null = np.asarray(null)
        p, nv = permutation_p(np.asarray(effect), np.asarray(null))
        finite_boot = np.asarray(boot)[np.isfinite(boot)]
        ci = np.percentile(finite_boot, [2.5, 97.5]) if np.isfinite(effect) and finite_boot.size else [np.nan, np.nan]
        rows.append(dict(sender=categories[a], receiver=categories[b], effect=effect,
                         ci_low=ci[0], ci_high=ci[1], pvalue=float(p), n_valid=int(nv),
                         n_tested=nt, n_reference=nr))
    result = pd.DataFrame(rows)
    result["padj"] = adjust_p(result.pvalue.to_numpy())
    annotate_permutation_resolution(result, "pvalue", "n_valid", label="replicate neighborhood contrasts")
    result.attrs.update(sample_key=sample_key, condition_key=condition_key, contrast=list(contrast),
                        paired_key=paired_key, block_key=block_key, seed=seed,
                        statistic="sample contact fraction minus finite-population expectation")
    adata.uns[key_added] = {"results": result, "sample_scores": pd.DataFrame(scores, index=sample_meta.index,
                                 columns=pd.MultiIndex.from_tuples([(categories[a], categories[b]) for a,b in pairs]))}
    return result
