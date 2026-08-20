"""
BANKSY multi-λ / multi-resolution parameter sweep.

Runs the full BANKSY pipeline (banksy_matrix → PCA → neighbors → Leiden) at
every combination of lambda_params × resolutions, storing per-run cluster
labels in obs and a summary DataFrame (with per-run quality metrics) in uns.
"""
from __future__ import annotations

import logging
import threading

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from joblib import Parallel, delayed

from astrolabe._utils import from_kwarg, with_adata

__all__ = ["banksy_sweep"]

log = logging.getLogger(__name__)

_SILHOUETTE_SUBSAMPLE = 10_000


@with_adata(mirror_uns=from_kwarg("key_added"))
def banksy_sweep(
    adata: ad.AnnData,
    *,
    lambda_params: list[float] = (0.0, 0.2, 0.5, 0.8),
    resolutions: list[float] = (0.3, 0.5, 1.0),
    n_pcs: int = 50,
    n_neighbors: int = 15,
    agf: bool = False,
    agf_weight: float = 1.0 / 3.0,
    neighbors_key: str = "spatial_neighbors",
    spatial_key: str = "spatial",
    layer: str | None = None,
    random_state: int = 0,
    n_jobs: int = 1,
    key_added: str = "banksy_sweep",
) -> pd.DataFrame:
    """Run the BANKSY pipeline across a grid of λ values and Leiden resolutions.

    For each combination of ``lambda_params × resolutions``:

    1. :func:`~astrolabe.pp.banksy_matrix` — build the BANKSY feature matrix.
    2. PCA on the BANKSY matrix (``n_pcs`` components).
    3. ``sc.pp.neighbors`` on the BANKSY PCA embedding.
    4. ``sc.tl.leiden`` at the given resolution.

    Cluster labels are written to ``obs`` as
    ``{key_added}_L{lambda}_R{resolution}`` for each run (adata-resident
    only).  A summary DataFrame is stored in ``uns[key_added]`` and
    returned; mirrored to ``mdata.uns[key_added][<modality>]`` when called
    on a MuData.

    Low λ (≈0.0–0.2) recovers transcriptomic cell types; high λ (≈0.5–0.8)
    recovers spatially-coherent tissue domains.

    **Per-run quality metrics** (summary columns):

    ``neighbor_agreement``
        Fraction of spatial-graph edges (``neighbors_key``) whose two
        endpoint cells share a cluster label — spatial coherence of the
        partition. **This is the column to rank domain-finding runs by**,
        with one caveat: it trivially increases as clusters merge (one giant
        cluster scores 1.0), so compare runs at similar ``n_clusters`` and
        read it alongside ``silhouette_expr``.
    ``silhouette_expr``
        Mean silhouette of the labels evaluated on a **pure-expression** PCA
        (computed once, independent of λ; subsampled to ≤10 000 cells for
        speed) — how transcriptionally distinct the clusters remain.  High-λ
        domain runs will legitimately score lower here than λ=0 cell-type
        runs; a near-zero value flags a partition with no transcriptional
        identity at all.  NaN when fewer than 2 clusters survive in the
        subsample.
    ``ari_to_prev_resolution``
        Adjusted Rand index between this run's labels and the previous
        (next-lower) resolution's labels *within the same λ* — a cheap
        stability probe (plateaus of high ARI across adjacent resolutions
        indicate a robust partition).  NaN for each λ's first resolution.

    Each λ iteration writes its intermediate BANKSY matrix / PCA / neighbors
    graph to keys namespaced by that λ (never the default ``obsm['X_pca']``
    or default neighbors keys), so it never touches — and does not need to
    save/restore — the caller's own PCA or neighbor graph. This also means
    λ iterations don't race on shared state and are safe to run concurrently
    via ``n_jobs``.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` with normalized
        expression and a spatial graph in ``obsp``.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    lambda_params
        λ values to sweep.  Each controls the self-vs-neighborhood weight.
    resolutions
        Leiden resolution values to sweep for each λ.  Sorted internally per
        λ so ``ari_to_prev_resolution`` always compares adjacent resolutions.
    n_pcs
        Number of PCA components for the BANKSY embedding. Matches
        :func:`~astrolabe.tl.pca`'s own default (``50``) -- the two used to
        silently diverge (this defaulted to ``30``), producing genuinely
        different cluster counts (13 vs 11 observed on real data at
        identical λ/resolution) between a manual BANKSY run and a sweep run
        with no indication anything differed.
    n_neighbors
        Neighbors used when building the KNN graph on the BANKSY PCA
        embedding for Leiden.
    agf
        Include azimuthal Gabor features in the BANKSY matrix.
    agf_weight
        Fraction of λ allocated to |M1| when ``agf=True``.  Default ``1/3``
        matches :func:`~astrolabe.pp.banksy_matrix`'s own default (banksy_py's
        ``2^-m`` harmonic split).
    neighbors_key
        Prefix for the spatial connectivity in ``obsp``.
    spatial_key
        ``obsm`` key for micron coordinates (required when ``agf=True``).
    layer
        Expression layer for the BANKSY matrix.  ``None`` uses ``adata.X``.
    random_state
        Random seed for PCA, Leiden, and the silhouette subsample.
    n_jobs
        Parallel workers for the outer λ loop (joblib, thread-backed — no
        extra memory copies since arrays are shared, not pickled). ``1``
        (default) runs sequentially. ``-1`` uses all available cores. The
        inner resolutions loop always runs sequentially within each λ
        worker.  Note: because ``adata.obs`` is a single shared DataFrame,
        the ``sc.tl.leiden`` call (the sole obs writer) is serialized behind
        a lock across λ workers — parallel speedup comes from the BANKSY
        matrix / PCA / neighbors stages, not from Leiden itself.
    key_added
        Prefix for obs columns and the uns summary key.

    Returns
    -------
    pd.DataFrame
        One row per (λ, resolution) run with columns ``lambda``,
        ``resolution``, ``n_clusters``, ``obs_key``, ``neighbor_agreement``,
        ``silhouette_expr``, ``ari_to_prev_resolution`` (see above).

    Examples
    --------
    >>> summary = ab.tl.banksy_sweep(mdata)
    >>> summary
    # lambda  resolution  n_clusters  obs_key                 neighbor_agreement ...
    # 0.0     0.3         8           banksy_sweep_L0.0_R0.3  0.61 ...

    >>> # best domain-finding run among the high-λ rows
    >>> high = summary[summary["lambda"] >= 0.5]
    >>> high.sort_values("neighbor_agreement", ascending=False).head()

    >>> # inspect a specific result
    >>> ab.pl.points(mdata, color="banksy_sweep_L0.8_R0.5")
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    from astrolabe.pp._banksy import banksy_matrix
    from astrolabe.tl._embedding import pca as _pca

    # Cap by n_obs only -- the per-run cap below uses the actual BANKSY
    # matrix width (2-3x n_vars), not n_vars itself.
    n_pcs_eff = min(n_pcs, adata.n_obs - 1)

    # ── Shared metric inputs (computed once, independent of λ) ────────────────
    conn_key = neighbors_key + "_connectivities"
    if conn_key not in adata.obsp:
        raise KeyError(
            f"obsp['{conn_key}'] not found. Run ab.tl.spatial_neighbors first."
        )
    conn = adata.obsp[conn_key].tocoo()
    edge_mask = conn.row != conn.col
    edge_row = conn.row[edge_mask]
    edge_col = conn.col[edge_mask]

    # Pure-expression PCA for the transcriptional-silhouette column
    # (equivalent to the λ=0 embedding; local sklearn PCA so the caller's
    # own obsm['X_pca'] / varm are never touched).
    from sklearn.decomposition import PCA

    X_expr = adata.layers[layer] if layer is not None else adata.X
    X_expr = X_expr.toarray() if sp.issparse(X_expr) else np.asarray(X_expr)
    X_expr = np.asarray(X_expr, dtype=np.float32)
    n_comp_expr = min(n_pcs_eff, adata.n_vars, adata.n_obs)
    expr_pca = PCA(n_components=n_comp_expr, random_state=random_state).fit_transform(X_expr)

    rng = np.random.default_rng(random_state)
    if adata.n_obs > _SILHOUETTE_SUBSAMPLE:
        sil_idx = rng.choice(adata.n_obs, _SILHOUETTE_SUBSAMPLE, replace=False)
    else:
        sil_idx = np.arange(adata.n_obs)
    expr_pca_sub = expr_pca[sil_idx]

    def _run_metrics(labels: np.ndarray) -> tuple[float, float]:
        """(neighbor_agreement, silhouette_expr) for one run's labels."""
        if len(edge_row):
            agreement = float(np.mean(labels[edge_row] == labels[edge_col]))
        else:
            agreement = float("nan")
        sub_labels = labels[sil_idx]
        n_sub_clusters = len(pd.unique(sub_labels))
        if 2 <= n_sub_clusters < len(sub_labels):
            sil = float(silhouette_score(expr_pca_sub, sub_labels))
        else:
            sil = float("nan")
        return agreement, sil

    # adata.obs is a single shared pandas DataFrame — concurrent column
    # inserts from different threads can race on its internal block
    # manager. adata.obsm/obsp/uns are per-key dict-likes where distinct-key
    # writes are safe, so only the sc.tl.leiden call (the sole obs writer)
    # needs to be serialized across λ workers. This deliberately serializes
    # the Leiden stage itself (documented above under ``n_jobs``).
    obs_lock = threading.Lock()

    # ari_to_prev_resolution compares *adjacent* resolutions, so iterate
    # them in ascending order regardless of the order passed in.
    resolutions_sorted = sorted(resolutions)

    def _run_lambda(lam_idx: int, lam: float) -> list[dict]:
        # Internal/ephemeral per-lambda keys are namespaced by loop *index*,
        # not a rounded-and-stripped "{lam:.2f}" string -- two distinct
        # lambda values that happen to round to the same 2 decimals (e.g.
        # 0.201 and 0.204 both -> "0.2") used to collide onto the *same*
        # obsm/obsp/uns keys, silently overwriting one run's intermediate
        # state with another's under n_jobs=1, and raising a KeyError when
        # one thread's cleanup deleted a key another thread was still using
        # under n_jobs>1. An index is unique by construction regardless of
        # how close two lambda values are.
        tmp_key = f"_banksy_tmp_{lam_idx}"
        pca_key = f"{tmp_key}_pca"
        neighbors_key_lam = f"{tmp_key}_neighbors"
        # The user-facing obs_key tag stays human-readable, but uses `:g`
        # (up to ~6 significant digits, no fixed-decimal truncation) instead
        # of the lossy `.2f`+strip -- collision-free for any lambda/
        # resolution values a user would plausibly pass as distinct.
        lam_tag = f"{lam:g}"

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

        n_pcs_run = min(n_pcs_eff, adata.obsm[tmp_key].shape[1] - 1)
        _pca(adata, n_comps=n_pcs_run, use_rep=tmp_key, random_state=random_state, key_added=pca_key)
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=pca_key,
            random_state=random_state,
            key_added=neighbors_key_lam,
        )

        lam_records = []
        prev_labels: np.ndarray | None = None
        for res in resolutions_sorted:
            res_tag = f"{res:g}"
            obs_key = f"{key_added}_L{lam_tag}_R{res_tag}"

            with obs_lock:
                sc.tl.leiden(
                    adata,
                    resolution=res,
                    key_added=obs_key,
                    random_state=random_state,
                    flavor="igraph",
                    directed=False,
                    n_iterations=2,
                    neighbors_key=neighbors_key_lam,
                )
                # leiden leaves a {'params': ...} provenance dict behind in
                # uns under the obs column's name -- clutter for sweep runs
                # (dozens of columns), so clean it up like the other
                # ephemeral keys.
                adata.uns.pop(obs_key, None)
                labels = adata.obs[obs_key].to_numpy(copy=True)

            n_clusters = len(pd.unique(labels))
            agreement, sil = _run_metrics(labels)
            ari_prev = (
                float(adjusted_rand_score(prev_labels, labels))
                if prev_labels is not None
                else float("nan")
            )
            prev_labels = labels

            lam_records.append(
                {
                    "lambda": lam,
                    "resolution": res,
                    "n_clusters": n_clusters,
                    "obs_key": obs_key,
                    "neighbor_agreement": agreement,
                    "silhouette_expr": sil,
                    "ari_to_prev_resolution": ari_prev,
                }
            )
            log.info(
                "banksy_sweep: λ=%.2f  res=%.2f  → %d clusters  "
                "agreement=%.3f  sil=%.3f  ('%s')",
                lam, res, n_clusters, agreement, sil, obs_key,
            )

        # clean up temporary per-λ keys
        del adata.obsm[tmp_key]
        del adata.obsm[pca_key]
        adata.uns.pop(pca_key, None)  # tl.pca (sc.pp.pca obsm= path) also writes uns
        adata.obsp.pop(f"{neighbors_key_lam}_distances", None)
        adata.obsp.pop(f"{neighbors_key_lam}_connectivities", None)
        adata.uns.pop(neighbors_key_lam, None)

        return lam_records

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_lambda)(i, lam) for i, lam in enumerate(lambda_params)
    )
    records = [row for lam_records in results for row in lam_records]

    df = pd.DataFrame(records)
    adata.uns[key_added] = df
    log.info("banksy_sweep: done — %d runs", len(df))
    return df
