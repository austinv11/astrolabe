"""Donor-level niche abundance and reusable niche assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from astrolabe._statistics import adjust_p
from astrolabe._utils import from_kwarg, with_adata


@with_adata(mirror_uns=from_kwarg("key_added"))
def niche_abundance(adata, *, niche_key="niches", sample_key, donor_key, condition_key,
                     covariates=(), exposure="all_cells", area_by_sample=None, key_added="niche_abundance"):
    """Count every niche per sample with an explicit denominator, including zeros.

    ``all_cells`` includes unassigned cells in the denominator; ``assigned_cells``
    excludes them. ``area`` requires observed square-micron area per sample.
    Donor/condition/covariates must be constant within samples. This table is
    descriptive; use ``diff_niche_abundance`` for donor-level inference.
    """
    if exposure not in ("all_cells", "assigned_cells", "area"):
        raise ValueError("exposure must be all_cells, assigned_cells, or area")
    columns = list(dict.fromkeys([sample_key, donor_key, condition_key, *covariates]))
    if adata.obs[columns].isna().any().any():
        raise ValueError("Sample, donor, condition and covariates must be complete")
    meta = adata.obs[columns].drop_duplicates()
    if meta[sample_key].duplicated().any():
        raise ValueError("Metadata must be constant within each sample")
    labels = adata.obs[niche_key]
    levels = list(labels.cat.categories) if isinstance(labels.dtype, pd.CategoricalDtype) else list(pd.unique(labels.dropna()))
    if not levels:
        raise ValueError("No niche categories are available")
    rows = []
    for _, row in meta.iterrows():
        selected = adata.obs[sample_key] == row[sample_key]
        values = labels[selected]
        denominator = (area_by_sample.get(row[sample_key], np.nan) if area_by_sample is not None else np.nan) if exposure == "area" else len(values) if exposure == "all_cells" else values.notna().sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Every sample needs a positive finite exposure")
        counts = values.value_counts()
        for level in levels:
            rows.append(dict(row, niche=level, count=int(counts.get(level, 0)), exposure=float(denominator)))
    result = pd.DataFrame(rows)
    result["abundance"] = result["count"] / result.exposure
    result.attrs.update(sample_key=sample_key, donor_key=donor_key, condition_key=condition_key,
                        covariates=list(covariates), exposure=exposure, niche_key=niche_key)
    adata.uns[key_added] = result
    return result


def diff_niche_abundance(table, *, contrast, paired=False, covariates=(),
                         donor_key=None, condition_key=None, exposure=None):
    """Equal-donor linear contrasts of niche fractions/densities with HC3 SEs.

    Technical samples are pooled within donor/condition by their exposures.
    Unpaired donors must occur in only one condition; paired donors need both.
    Covariates are encoded numerically or with reference-coded categories.
    P-values use residual t degrees of freedom; this is an approximate linear
    model on donor summaries, not a cell-level binomial test. Discovery on the
    same cohort makes these contrasts exploratory.
    """
    from statsmodels.regression.linear_model import OLS
    donor = donor_key or table.attrs.get("donor_key")
    condition = condition_key or table.attrs.get("condition_key")
    exposure = exposure or table.attrs.get("exposure")
    if donor is None or condition is None or exposure not in ("all_cells", "assigned_cells", "area"):
        raise ValueError("Supply donor_key, condition_key and exposure when table metadata is unavailable")
    if table[[donor, condition, "niche", "count", "exposure"]].isna().any().any():
        raise ValueError("Counts, exposures and grouping labels must be complete")
    values = table[["count", "exposure"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any() or (values[:, 1] <= 0).any():
        raise ValueError("Counts must be finite and nonnegative; exposures finite and positive")
    if not np.equal(values[:, 0], np.floor(values[:, 0])).all():
        raise ValueError("Niche counts must be integers")
    if exposure != "area" and (values[:, 0] > values[:, 1]).any():
        raise ValueError("Niche counts cannot exceed the cell denominator")
    if len(contrast) != 2 or contrast[0] == contrast[1]:
        raise ValueError("contrast requires distinct tested/reference levels")
    selected = table[table[condition].isin(contrast)].copy()
    if selected.empty:
        raise ValueError("No selected conditions")
    metadata_keys = [donor, condition, *covariates]
    meta = selected[metadata_keys].drop_duplicates()
    if meta.duplicated([donor, condition]).any() or meta.isna().any().any():
        raise ValueError("Covariates must be complete and constant within donor/condition")
    n_conditions = meta.groupby(donor, observed=True)[condition].nunique()
    if (paired and (n_conditions != 2).any()) or (not paired and (n_conditions != 1).any()):
        raise ValueError("Design requires complete donor pairs or independent unpaired donors")
    if any((meta[condition] == level).sum() < 3 for level in contrast):
        raise ValueError("At least three independent donors per condition are required")
    pooled = selected.groupby([donor, condition, "niche"], observed=True, sort=False)[["count", "exposure"]].sum().reset_index()
    rows = []
    for niche, frame in pooled.groupby("niche", observed=True, sort=False):
        if len(frame) != len(meta):
            raise ValueError("Every niche needs an explicit count, including zero, for every donor/condition")
        frame = frame.merge(meta, on=[donor, condition], validate="many_to_one")
        frame["value"] = frame["count"]/frame.exposure
        nuisance = pd.get_dummies(frame[list(covariates)], drop_first=True, dtype=float) if covariates else pd.DataFrame(index=frame.index)
        if paired:
            tested = frame[frame[condition] == contrast[0]].set_index(donor)
            reference = frame[frame[condition] == contrast[1]].set_index(donor).reindex(tested.index)
            y = (tested.value-reference.value).to_numpy()
            nuisance.index = pd.MultiIndex.from_frame(frame[[donor, condition]])
            change = (nuisance.xs(contrast[0], level=condition).reindex(tested.index)
                      - nuisance.xs(contrast[1], level=condition).reindex(tested.index)).to_numpy()
            change = change[:, np.any(change != 0, axis=0)]
            design = np.column_stack([np.ones(len(y)), change]); effect_index = 0
        else:
            y = frame.value.to_numpy()
            design = np.column_stack([np.ones(len(frame)), frame[condition] == contrast[0], nuisance.to_numpy(dtype=float)])
            effect_index = 1
        rank = np.linalg.matrix_rank(design)
        if rank != design.shape[1] or len(y) <= rank:
            raise ValueError("Confounded or saturated donor design")
        fit = OLS(y, design).fit(cov_type="HC3", use_t=True)
        leverage = np.einsum("ij,ji->i", design, np.linalg.pinv(design))
        if (leverage >= 1-1e-12).any():
            raise ValueError("Insufficient residual information for HC3 inference")
        se = float(fit.bse[effect_index])
        effect = float(fit.params[effect_index]); df = fit.df_resid
        estimable = se > 0 and np.linalg.norm(fit.resid) > np.finfo(float).eps * max(1., np.linalg.norm(y)) * len(y)
        p = float(fit.pvalues[effect_index]) if estimable else np.nan
        low, high = fit.conf_int()[effect_index] if estimable else (np.nan, np.nan)
        rows.append({"niche": niche, "effect": effect, "se": se, "ci_low": low,
                     "ci_high": high, "pvalue": p,
                     "n_donors_tested": int((meta[condition] == contrast[0]).sum()),
                     "n_donors_reference": int((meta[condition] == contrast[1]).sum()),
                     "df_residual": df, "status": "tested" if np.isfinite(p) else "zero_residual_variance"})
    result = pd.DataFrame(rows)
    result["padj"] = adjust_p(result.pvalue.to_numpy())
    result.attrs.update(inference_unit="donor", paired=paired, contrast=list(contrast), covariates=list(covariates),
                        exposure=exposure, method="statsmodels OLS, HC3, residual t",
                        reference="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html")
    return result


@with_adata
def niche_pseudobulk(adata, *, donor_key, condition_key, niche_key="niches",
                     cell_type_key=None, covariates=(), layer="counts", min_cells=10):
    """Aggregate counts per donor/condition and niche for within-niche state DE.

    Technical sections of the same donor/condition are summed. Missing niche
    or cell-type assignments are excluded and counted in metadata. Use
    ``rank_genes_pseudobulk`` on the result; for paired conditions supply
    ``paired_key=donor_key`` there. Counts are never imputed for absent niches.
    """
    from astrolabe.pp._pseudobulk import pseudobulk
    groups = [niche_key] + ([] if cell_type_key is None else [cell_type_key])
    if adata.obs[[donor_key, condition_key]].isna().any().any():
        raise ValueError("Donor and condition labels must be complete")
    keep = adata.obs[groups].notna().all(axis=1).to_numpy()
    subset = adata[keep].copy()
    sample_key = "_niche_biological_sample"
    if sample_key in subset.obs:
        raise ValueError(f"Reserved aggregation column already exists: {sample_key}")
    codes, _ = pd.factorize(pd.MultiIndex.from_frame(subset.obs[[donor_key, condition_key]]))
    subset.obs[sample_key] = [f"sample_{code}" for code in codes]
    result = pseudobulk(subset, sample_key=sample_key, groupby=groups, layer=layer,
                        metadata_keys=list(dict.fromkeys([donor_key, condition_key, *covariates])), min_cells=min_cells)
    result.uns["niche_state"] = {"donor_key": donor_key, "condition_key": condition_key,
        "niche_key": niche_key, "n_cells_unassigned": int((~keep).sum()),
        "replication_unit": "donor", "technical_sections": "summed within donor/condition"}
    return result


@with_adata(mirror_uns=from_kwarg("key_added"))
def fit_niche_model(adata, *, niche_key="niches", profile_key=None, quantile=.99, key_added="niche_model"):
    """Freeze composition scaling, centroids and assignment-distance cutoffs.

    Call on discovery samples only, after ``niches``. Cutoffs are empirical
    training distances, not calibrated assignment probabilities. Inspect rare
    niches and held-out assignment rates before cohort inference.
    """
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in (0, 1]")
    profile_key = profile_key or adata.uns[niche_key]["profile_key"]
    profiles = adata.obsm[profile_key]
    profiles = np.asarray(profiles.toarray() if hasattr(profiles, "toarray") else profiles, dtype=float)
    valid = adata.obs[niche_key].notna().to_numpy()
    if not valid.any():
        raise ValueError("No assigned training cells")
    scale = profiles[valid].std(axis=0)
    scale[scale == 0] = 1
    levels = list(pd.unique(adata.obs.loc[valid, niche_key].astype(str)))
    centroids, cutoffs, counts = [], [], []
    for level in levels:
        values = profiles[adata.obs[niche_key].astype(str).to_numpy() == level]/scale
        centroid = values.mean(axis=0)
        centroids.append(centroid)
        cutoffs.append(float(np.quantile(np.linalg.norm(values-centroid, axis=1), quantile)))
        counts.append(len(values))
    result = {"categories": adata.uns[profile_key]["categories"], "niches": levels, "scale": scale,
              "centroids": np.asarray(centroids), "distance_cutoffs": np.asarray(cutoffs), "training_counts": np.asarray(counts),
              "quantile": quantile, "groupby": adata.uns[profile_key]["groupby"]}
    adata.uns[key_added] = result
    return result


@with_adata
def map_niches(adata, model, *, groupby=None, neighbors_key="spatial_neighbors", key_added="mapped_niche"):
    """Assign held-out compositions to frozen niche centroids; reject distant cells."""
    from astrolabe.tl._niches import niche_composition
    groupby = groupby or model["groupby"]
    unknown = set(adata.obs[groupby].dropna().astype(str))-set(model["categories"])
    if unknown:
        raise ValueError(f"Cell types absent from training model: {sorted(unknown)}")
    profile_key = key_added + "_composition"
    temporary_labels = adata.obs[groupby]
    try:
        adata.obs[groupby] = pd.Categorical(temporary_labels.astype(str), categories=model["categories"])
        profiles = niche_composition(adata, groupby, neighbors_key=neighbors_key, key_added=profile_key)
    finally:
        adata.obs[groupby] = temporary_labels
    from scipy.spatial.distance import cdist
    labels = np.full(adata.n_obs, None, dtype=object)
    distance = np.full(adata.n_obs, np.nan)
    for start in range(0, adata.n_obs, 8192):
        stop = min(start+8192, adata.n_obs)
        values = profiles[start:stop].toarray()/np.asarray(model["scale"])
        distances = cdist(values, np.asarray(model["centroids"]))
        nearest = distances.argmin(axis=1)
        closest = distances[np.arange(stop-start), nearest]
        usable = adata.obs[profile_key + "_valid"].to_numpy()[start:stop] & (closest <= np.asarray(model["distance_cutoffs"])[nearest] + 1e-12)
        if distances.shape[1] > 1:
            ordered = np.partition(distances, 1, axis=1)
            usable &= ordered[:, 1]-ordered[:, 0] > 1e-12
        local = np.full(stop-start, None, dtype=object)
        local[usable] = np.asarray(model["niches"])[nearest[usable]]
        labels[start:stop] = local; distance[start:stop] = closest
    adata.obs[key_added] = pd.Categorical(labels, categories=model["niches"])
    adata.obs[key_added + "_distance"] = distance
    return adata.obs[key_added]
