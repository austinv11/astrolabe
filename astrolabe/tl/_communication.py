"""Complex-aware spatial communication association scores."""
from __future__ import annotations
import importlib.metadata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from astrolabe._statistics import adjust_p, permutation_resolution
from astrolabe._utils import with_adata, from_kwarg


def import_ligrec_resource(resource, *, species, version=None):
    """Normalize a supplied table or an optional LIANA bundled resource.

    User tables use ligand/receptor IDs and explicit ligand_members /
    receptor_members (lists or semicolon-separated symbols). LIANA source /
    target complexes are expanded using that resource's underscore convention.
    Species and resource version are retained; no online resource is fetched.
    """
    if not species:
        raise ValueError("species must be explicit")
    name = "supplied"
    if isinstance(resource, str):
        try:
            import liana as li
        except ImportError as exc:
            raise ImportError("Install astrolabe[communication] for LIANA resources") from exc
        if species.lower() not in ("human", "homo sapiens"):
            raise ValueError("bundled LIANA resources use human symbols; supply a translated table explicitly")
        name = resource
        resource = li.rs.select_resource(resource)
        version = importlib.metadata.version("liana")
    frame = resource.copy()
    if {"source", "target"}.issubset(frame):
        frame = frame.rename(columns={"source": "ligand", "target": "receptor"})
        frame["ligand_members"] = frame.ligand.map(lambda v: str(v).split("_"))
        frame["receptor_members"] = frame.receptor.map(lambda v: str(v).split("_"))
    if name != "supplied":
        for side in ("ligand", "receptor"):
            frame[side + "_members"] = frame[side].map(lambda value: str(value).split("_"))
    if not {"ligand", "receptor"}.issubset(frame):
        raise ValueError("resource requires ligand/receptor or source/target columns")
    if frame[["ligand", "receptor"]].isna().any().any():
        raise ValueError("resource interaction IDs must be present")
    frame.attrs.update(resource=name, species=species, version=version or "user supplied")
    return frame


def _members(row, side):
    value = row.get(side + "_members", row[side])
    if isinstance(value, str):
        members = value.split(";")
    else:
        members = list(value)
    if not members or any(not isinstance(v, str) or not v for v in members):
        raise ValueError("complex members must be nonempty gene symbols")
    return list(dict.fromkeys(members))


def _ligrec(adata, groupby, lr_pairs, *, neighbors_key="spatial_neighbors", layer=None,
            n_perms=100, seed=None, expr_frac_threshold=.1, report_alpha=.05,
            corr_method="fdr_bh", n_jobs=1, min_valid_perms=20, library_key="auto",
            max_scratch_bytes=256*1024**2, coverage_policy="error", key_added="ligrec"):
    from astrolabe.tl._neighborhood import _category_codes
    from astrolabe.tl._spatial_stats import _get_conn_matrix, _permutation_blocks
    if not {"ligand", "receptor"}.issubset(lr_pairs):
        raise ValueError("lr_pairs must have ligand and receptor columns")
    if n_perms < 0 or min_valid_perms < 1 or not 0 <= expr_frac_threshold <= 1 or max_scratch_bytes < 1024:
        raise ValueError("invalid permutation, gate or memory parameters")
    codes, categories = _category_codes(adata, groupby)
    conn = _get_conn_matrix(adata, neighbors_key).tocsr()
    blocks, library_key = _permutation_blocks(adata, neighbors_key, library_key=library_key)
    library_codes = np.zeros(adata.n_obs, dtype=int)
    for code, block in enumerate(blocks):
        library_codes[block] = code
    mat = adata.layers[layer] if layer is not None else adata.X
    if hasattr(mat, "get"):
        mat = mat.get()  # CPU fallback retains the graph/layer statistic.
    present = set(adata.var_names)
    from astrolabe._coverage import _coverage
    if coverage_policy not in ("error", "eligible"):
        raise ValueError("coverage_policy must be error or eligible")
    pair_coverage = []
    coverage_labels = None
    for _, row in lr_pairs.iterrows():
        members = _members(row, "ligand") + _members(row, "receptor")
        if not set(members).issubset(present):
            pair_coverage.append(None)
            continue
        coverage, coverage_labels = _coverage(adata, members)
        if coverage.any() and not coverage.all() and coverage_policy == "error":
            raise ValueError("Partial measured-feature coverage for an interaction; "
                             "select a common panel or use coverage_policy='eligible'")
        pair_coverage.append(coverage)
    n_types = len(categories); n_ct = n_types**2; n_lr = len(lr_pairs)
    if not n_lr:
        raise ValueError("lr_pairs is empty")
    means = np.full((n_lr, n_ct), np.nan); pvalues = means.copy()
    n_valid = np.zeros((n_lr, n_ct), dtype=int)
    status = np.full((n_lr, n_ct), "not_tested", dtype=object)
    type_counts = np.bincount(codes, minlength=n_types)
    measured_pairs = sum(value is not None and value.any() for value in pair_coverage)
    family_size = int(measured_pairs*np.count_nonzero(type_counts)**2)
    seed = np.random.SeedSequence(seed)
    permutation_seeds = seed.spawn(n_perms)
    edge_chunk = max(1, min(conn.nnz or 1, max_scratch_bytes // 128))

    def edges(start):
        end = min(start+edge_chunk, conn.nnz)
        src = np.searchsorted(conn.indptr, np.arange(start,end), side="right")-1
        dst = conn.indices[start:end]
        keep = (src != dst) & (conn.data[start:end] != 0)
        return src[keep], dst[keep]

    for start in range(0, conn.nnz, edge_chunk):
        src, dst = edges(start)
        if (library_codes[src] != library_codes[dst]).any():
            raise ValueError("LR graph contains edges across libraries")

    def expression(members):
        result = np.full(adata.n_obs, np.inf)
        for member in members:
            values = mat[:, adata.var_names.get_loc(member)]
            if sp.issparse(values):
                values = values.toarray()
            values = np.asarray(values, dtype=float).ravel()
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("LR expression must be finite and nonnegative")
            np.minimum(result, values, out=result)
        return result

    def scores(ligand, receptor, labels, active):
        sums = np.zeros(n_ct); counts = np.zeros(n_ct)
        for start in range(0, conn.nnz, edge_chunk):
            src, dst = edges(start)
            keep = active[src] & active[dst]
            src, dst = src[keep], dst[keep]
            indices = labels[src]*n_types + labels[dst]
            values = (ligand[src] + receptor[dst]) / 2
            sums += np.bincount(indices, weights=values, minlength=n_ct)
            counts += np.bincount(indices, minlength=n_ct)
        return np.divide(sums, counts, out=np.full(n_ct, np.nan), where=counts > 0)

    def score_pair(pi, row):
        lm, rm = _members(row, "ligand"), _members(row, "receptor")
        if not set(lm+rm).issubset(present):
            status[pi] = "missing_subunit"
            return
        coverage = pair_coverage[pi]
        if not coverage.any():
            status[pi] = "unmeasured"
            return
        active = np.isin(coverage_labels, coverage.index[coverage])
        active_blocks = [block[active[block]] for block in blocks]
        active_counts = np.bincount(codes[active], minlength=n_types)
        ligand, receptor = expression(lm), expression(rm)
        lf = np.divide(np.bincount(codes[active], weights=ligand[active] > 0, minlength=n_types), active_counts,
                       out=np.zeros(n_types), where=active_counts > 0)
        rf = np.divide(np.bincount(codes[active], weights=receptor[active] > 0, minlength=n_types), active_counts,
                       out=np.zeros(n_types), where=active_counts > 0)
        gate = ((lf[:,None] >= expr_frac_threshold) & (rf[None,:] >= expr_frac_threshold)).ravel()
        observed = scores(ligand, receptor, codes, active)
        exceed = np.zeros(n_ct, dtype=int)
        for child_seed in permutation_seeds:
            rng = np.random.default_rng(child_seed)
            shuffled = codes.copy()
            for block in active_blocks:
                shuffled[block] = rng.permutation(shuffled[block])
            simulated = scores(ligand, receptor, shuffled, active)
            valid = np.isfinite(simulated)
            n_valid[pi] += valid
            tolerance = 100*np.finfo(float).eps*np.abs(observed)
            exceed += valid & (simulated >= observed-tolerance)
        eligible = gate & np.isfinite(observed)
        means[pi, eligible] = observed[eligible]
        tested = eligible & (n_valid[pi] >= min_valid_perms)
        pvalues[pi, tested] = (exceed[tested]+1.)/(n_valid[pi, tested]+1.)
        status[pi, ~gate] = "expression_gate"
        status[pi, ~np.isfinite(observed)] = "no_edges"
        status[pi, eligible] = "insufficient_permutations"
        status[pi, tested] = "tested"
    from joblib import Parallel, delayed
    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(score_pair)(pi, row) for pi, (_, row) in enumerate(lr_pairs.iterrows())
    )
    index = pd.MultiIndex.from_arrays([lr_pairs.ligand, lr_pairs.receptor], names=["ligand", "receptor"])
    columns = pd.MultiIndex.from_product([categories, categories], names=["sender", "receiver"])
    frame = lambda values: pd.DataFrame(values, index=index, columns=columns)
    coverage_frame = pd.DataFrame([{} if value is None else value.to_dict()
                                   for value in pair_coverage], index=index).fillna(False).astype(bool)
    floors, limited, resolution = permutation_resolution(
        pvalues, n_valid, alternative="greater", method=corr_method,
        alpha=report_alpha, label="ligand-receptor tests", family_size=family_size)
    result = {"means": frame(means), "pvalues": frame(pvalues), "n_valid": frame(n_valid),
              "eligible_libraries": coverage_frame,
              "p_min": frame(floors), "resolution_limited": frame(limited),
              "status": frame(status), "metadata": {"groupby": groupby, "layer": layer,
               "n_perms": n_perms, "min_valid_perms": min_valid_perms, "expr_frac_threshold": expr_frac_threshold,
               "report_alpha": report_alpha, "corr_method": corr_method, "neighbors_key": neighbors_key,
               "library_key": library_key, "n_libraries": len(blocks), "coverage_policy": coverage_policy,
               "spatial_restricted": True, "complex_rule": "minimum subunit",
               "resource": dict(lr_pairs.attrs), "max_scratch_bytes": max_scratch_bytes,
               "permutation_resolution": resolution}}
    result["metadata"].update(family_size=family_size,
        fdr_family="all measured interactions and present cell-type pairs; screened hypotheses retained")
    if corr_method is not None:
        result["pvalues_adj"] = frame(adjust_p(pvalues, corr_method, family_size=family_size))
    adata.uns[key_added] = result


@with_adata(mirror_uns=from_kwarg("key_added"))
def ligrec_local(adata, lr_pairs, *, layer=None, neighbors_key="spatial_neighbors",
                  n_perms=999, seed=0, key_added="ligrec_local"):
    """LIANA local cosine associations using a supplied layer and stored graph.

    Requires astrolabe[communication]. Local association is not a measurement
    of ligand transport or causal signaling. Complexes use LIANA conventions.
    """
    from astrolabe._coverage import _require_measured
    _require_measured(adata, [gene for _, row in lr_pairs.iterrows()
                             for side in ("ligand", "receptor") for gene in _members(row, side)])
    try:
        import liana as li
    except ImportError as exc:
        raise ImportError("Install astrolabe[communication] for local LR maps") from exc
    from astrolabe.tl._spatial_stats import _get_conn_matrix
    import anndata as ad
    mat = adata.layers[layer] if layer is not None else adata.X
    temporary = ad.AnnData(mat.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    conn = _get_conn_matrix(adata, neighbors_key).tocsr(copy=True)
    conn.setdiag(0); conn.eliminate_zeros()
    if "library_id" in adata.obs:
        coo = conn.tocoo(); libraries = adata.obs.library_id.to_numpy()
        if (libraries[coo.row] != libraries[coo.col]).any():
            raise ValueError("local LR graph crosses libraries")
    temporary.obsp["_astrolabe_lr_connectivity"] = conn
    resource = lr_pairs[["ligand", "receptor"]].copy()
    # Give LIANA explicit complex names derived from the same member lists.
    for side in ("ligand", "receptor"):
        resource[side] = lr_pairs.apply(lambda row: "_".join(_members(row, side)), axis=1)
    local = li.mt.bivariate(temporary, resource=resource, local_name="cosine", global_name=None,
                            connectivity_key="_astrolabe_lr_connectivity", use_raw=False,
                            n_perms=n_perms if n_perms > 0 else None, seed=seed, nz_prop=0,
                            remove_self_interactions=False)
    if not isinstance(local, ad.AnnData):
        raise RuntimeError("LIANA did not return local scores")
    local = local[adata.obs_names].copy()
    adata.obsm[key_added] = local.X
    adata.uns[key_added] = {"interactions": local.var.copy(),
                           "layers": dict(local.layers),
                           "metadata": {"backend": "LIANA local cosine", "version": importlib.metadata.version("liana"),
                             "layer": layer, "neighbors_key": neighbors_key, "seed": seed,
                             "resource": dict(lr_pairs.attrs)}}
    return adata.uns[key_added]
