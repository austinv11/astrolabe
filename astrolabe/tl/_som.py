"""
Self-organizing map analyses for spatial transcriptomics.

Methodology follows the established SOM-for-cytometry/omics workflow
(Kohonen 2001, *Self-Organizing Maps*; Vesanto & Alhoniemi 2000, *Clustering
of the self-organizing map*, IEEE TNN 11(3); Van Gassen et al. 2015, FlowSOM,
Cytometry A 87(7)):

1. **Scale features** before training, so Euclidean BMU assignment is not
   dominated by high-variance features.
2. **Train with shuffled, multi-epoch presentation** (never sequential
   single-pass), annealing the neighborhood radius to ~1 node and the
   learning rate toward 0.
3. **Meta-cluster the node codebook** (Ward linkage) into a small number of
   clusters — the SOM grid is a quantization, not a clustering; the reported
   cluster label is the metacluster, not the raw node.
4. **Report quantization error (QE) and topographic error (TE)** so map
   quality is measurable rather than assumed.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._utils import from_kwarg, with_adata

__all__ = ["som", "som_gene_modules", "som_stability"]

log = logging.getLogger(__name__)


def _require_minisom():
    try:
        from minisom import MiniSom
        return MiniSom
    except ImportError as exc:
        raise ImportError(
            "minisom is required for SOM analysis: uv add 'astrolabe[som]'"
        ) from exc


def _batch_bmu(som_obj, data: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Vectorized BMU assignment; returns flat index array of length n_samples.

    Squared Euclidean distance via ``||d - w||^2 = ||d||^2 + ||w||^2 - 2 d@w^T``
    -- a (chunk, n_nodes) matrix, not the (chunk, n_nodes, n_features)
    dense broadcast the direct `d[:, None, :] - w_flat[None, :, :]` form
    used to materialize. That intermediate scales with grid size *and*
    feature count simultaneously: measured 4.3GB at 20k cells on a modest
    grid, extrapolating to ~40GB at the 162k-cell/45x45-grid tutorial
    dataset -- an OOM on the library's own flagship example. Squared
    distance (not the sqrt'd distance) is used since argmin is invariant to
    the monotonic sqrt transform.
    """
    sx, sy, n_feat = som_obj.get_weights().shape
    w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float32)
    w_sq = np.sum(w_flat ** 2, axis=1)  # (n_nodes,)
    n = len(data)
    bmu_flat = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = data[start:end].astype(np.float32)
        d_sq = np.sum(d ** 2, axis=1, keepdims=True)  # (chunk, 1)
        dists_sq = d_sq + w_sq[None, :] - 2.0 * (d @ w_flat.T)  # (chunk, n_nodes)
        bmu_flat[start:end] = dists_sq.argmin(axis=1)
    return bmu_flat


def _default_sigma(sx: int, sy: int) -> float:
    """Grid-scaled initial neighborhood radius: a quarter of the smaller
    grid dimension (standard SOM ordering-phase heuristic; Kohonen 2001).

    Training passes ``sigma_decay_function='inverse_decay_to_one'`` so the
    radius anneals from this value down to 1 node by the final epoch --
    minisom's default ``asymptotic_decay`` only reaches a third of the
    initial value, which leaves large maps over-smoothed (adjacent nodes
    near-copies, flat U-matrix).
    """
    return max(1.0, min(sx, sy) / 4.0)


def _grid_sorted(labels) -> list[str]:
    """Sort ``"{col}_{row}"`` node labels by their integer grid position."""
    return sorted(labels, key=lambda s: tuple(int(p) for p in s.split("_")))


def _zscore(data: np.ndarray) -> np.ndarray:
    """Column z-score with zero-variance guard. Returns float32 copy."""
    mu = data.mean(axis=0)
    sd = data.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return ((data - mu) / sd).astype(np.float32)


def _unnormalized_umatrix(weights: np.ndarray) -> np.ndarray:
    """Mean Euclidean codebook distance to 8-connected grid neighbors.

    Unlike ``MiniSom.distance_map()`` -- which min-max normalizes so the
    maximum is always exactly 1.0, making a collapsed codebook visually
    indistinguishable from a well-separated one -- this keeps real units
    (distance in the training feature space).

    Rectangular topology only (the tl API enforces rectangular grids).
    """
    sx, sy, _ = weights.shape
    um = np.zeros((sx, sy), dtype=np.float64)
    counts = np.zeros((sx, sy), dtype=np.int32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            i0, i1 = max(0, di), sx + min(0, di)
            j0, j1 = max(0, dj), sy + min(0, dj)
            diff = weights[i0:i1, j0:j1] - weights[i0 - di:i1 - di, j0 - dj:j1 - dj]
            um[i0:i1, j0:j1] += np.linalg.norm(diff, axis=-1)
            counts[i0:i1, j0:j1] += 1
    return (um / counts).astype(np.float32)


def _metacluster_codebook(weights: np.ndarray, n_metaclusters: int, bmu_flat: np.ndarray):
    """Ward-linkage metaclustering of the node codebook (FlowSOM stage 3).

    Ward runs on the **populated** nodes only. An empty node's codebook is an
    interpolant of the neighborhood function in a region containing no cells,
    yet it would carry the same weight as a real prototype in Ward's variance
    criterion -- and on an auto-sized grid (45x45 = 2025 nodes) a substantial
    fraction of nodes are empty. Empty nodes are assigned afterwards to the
    metacluster of the nearest populated node in codebook space, so every node
    still has a label for plotting.

    Returns ``(node_meta, sample_meta)``: metacluster id per flat node index,
    and per sample via its BMU.
    """
    from sklearn.cluster import AgglomerativeClustering

    sx, sy, n_feat = weights.shape
    codebook = weights.reshape(-1, n_feat)
    n_nodes = len(codebook)

    hits = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(hits, bmu_flat, 1)
    pop = np.flatnonzero(hits > 0)

    k = int(min(n_metaclusters, len(pop)))
    if k < 2:
        return np.zeros(n_nodes, dtype=np.int32), np.zeros(len(bmu_flat), dtype=np.int32)

    lab_pop = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
        codebook[pop]
    )
    node_meta = np.empty(n_nodes, dtype=np.int32)
    node_meta[pop] = lab_pop
    empty = np.flatnonzero(hits == 0)
    if len(empty):
        d = (
            (codebook[empty] ** 2).sum(axis=1, keepdims=True)
            + (codebook[pop] ** 2).sum(axis=1)[None, :]
            - 2.0 * codebook[empty] @ codebook[pop].T
        )
        node_meta[empty] = lab_pop[d.argmin(axis=1)]
    return node_meta, node_meta[bmu_flat]


def _quality_metrics(som_obj, data: np.ndarray, seed: int, max_samples: int = 50_000):
    """Quantization + topographic error on a seeded subsample (cheap at scale)."""
    if len(data) > max_samples:
        rng = np.random.default_rng(seed)
        data = data[rng.choice(len(data), max_samples, replace=False)]
    qe = float(som_obj.quantization_error(data))
    te = float(som_obj.topographic_error(data))
    return qe, te


def _train_som(
    MiniSom, data: np.ndarray, sx: int, sy: int, *,
    sigma: float, learning_rate: float, n_epochs: int, seed: int,
    init: str = "pca", activation_distance: str = "euclidean",
):
    """Construct + train a MiniSom with the library's fixed conventions:

    - shuffled multi-epoch presentation (``random_order=True, use_epochs=True``)
      -- minisom's own ``train()`` default is *sequential single-order*
      presentation, which both introduces order effects and, whenever the
      iteration budget is below ``n_samples``, silently excludes the tail of
      the data from training entirely;
    - learning rate annealed to 0 (``inverse_decay_to_zero``) and sigma
      annealed to 1 node (``inverse_decay_to_one``) by the final epoch;
    - PCA weight initialization by default (faster, deterministic ordering).
    """
    som_obj = MiniSom(
        sx, sy, data.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        decay_function="inverse_decay_to_zero",
        sigma_decay_function="inverse_decay_to_one",
        topology="rectangular",
        activation_distance=activation_distance,
        random_seed=seed,
    )
    if init == "pca" and data.shape[1] >= 2:
        som_obj.pca_weights_init(data)
    else:
        som_obj.random_weights_init(data)
    som_obj.train(data, n_epochs, random_order=True, use_epochs=True, verbose=False)
    return som_obj


@with_adata(mirror_uns=from_kwarg("key_added"))
def som(
    adata: ad.AnnData,
    *,
    use_rep: str = "X_pca",
    x: int | None = None,
    y: int | None = None,
    n_metaclusters: int | None = 15,
    scale: bool | None = None,
    sigma: float | None = None,
    learning_rate: float = 0.5,
    n_epochs: int = 10,
    init: str = "pca",
    seed: int = 0,
    key_added: str = "som",
):
    """Train a SOM on an ``obsm`` embedding (or ``X``) and meta-cluster its nodes.

    Follows the FlowSOM-style workflow (Van Gassen et al. 2015): scaled
    features -> shuffled multi-epoch SOM training -> Ward meta-clustering of
    the node codebook into ``n_metaclusters`` clusters. The **metacluster**
    is the reported per-cell cluster label; the raw BMU node (stored
    separately) is a quantization, not a clustering -- a 45x45 grid has 2025
    nodes, which is a discretized embedding, not a set of cell populations.

    Map quality is measured, not assumed: quantization error (mean distance
    of each cell to its BMU) and topographic error (fraction of cells whose
    two best nodes are not grid-adjacent; Kohonen 2001) are computed on a
    seeded subsample and stored in ``uns[key_added]['params']``.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Training summary is stored in ``uns[key_added]`` and mirrored to
    ``mdata.uns[key_added][<modality>]`` when called on a MuData; the
    ``obsm``/``obs`` outputs stay adata-resident only.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    use_rep
        ``obsm`` key to use as training data. ``None`` uses ``X`` directly
        (densified if sparse).
    x
        Grid width (number of columns). Auto-computed from ``n_obs`` via the
        ``5 * sqrt(n)``-nodes heuristic if ``None``.
    y
        Grid height (number of rows). Defaults to ``x``.
    n_metaclusters
        Number of Ward metaclusters cut from the node codebook -- the
        reported cluster count. ``None`` skips meta-clustering and reports
        the raw node as the cluster (not recommended except for tiny grids).
    scale
        Z-score each training feature first. ``None`` (default) resolves
        automatically: ``True`` when training on raw features
        (``use_rep=None`` — Euclidean matching on unscaled expression is
        dominated by the highest-magnitude genes, the failure FlowSOM's
        scaling step exists to prevent), ``False`` when training on an
        ``obsm`` embedding like ``X_pca``. Pass a bool to override either way.

        The reason not to whiten an embedding is not that it disagrees with
        Leiden — tuning one clustering to agree with another and then citing
        the agreement proves nothing. It is that **whitening amplifies exactly
        the components PC truncation exists to discard**: the trailing
        eigenvalues of a cell x gene matrix sit at the Marchenko--Pastur noise
        edge, so rescaling every component to unit variance gives PC50's noise
        the same say in BMU assignment as PC1's signal. Keeping the embedding
        unscaled also puts the SOM in the *same metric* as the rest of the
        pipeline: :func:`scanpy.pp.neighbors` and UMAP both use unwhitened
        Euclidean distance on ``X_pca``, so a SOM trained the same way is
        measuring the same geometry rather than a different one.
    sigma
        Initial neighborhood radius. ``None`` (default) uses
        ``max(1.0, min(x, y) / 4)``; it anneals to 1 node by the final epoch.
    learning_rate
        Initial learning rate; anneals to 0 by the final epoch.
    n_epochs
        Full shuffled passes through the data (default 10). Replaces the old
        raw-draw ``n_iter``: every cell is guaranteed to participate in
        training every epoch.
    init
        ``'pca'`` (default) or ``'random'`` weight initialization.
    seed
        Random seed (training order, init, QE/TE subsample).
    key_added
        Prefix for all stored outputs.

    Stores
    ------
    ``obsm['X_' + key_added]``
        Float32 ``(n_obs, 2)`` BMU ``[col, row]`` coordinates.
    ``obs[key_added + '_cluster']``
        Categorical metacluster label per cell (``"0"``..``"k-1"``), or the
        raw node label when ``n_metaclusters=None``.
    ``obs[key_added + '_node']``
        Categorical raw BMU node ``"{col}_{row}"`` per cell.
    ``uns[key_added]``
        Dict with ``weights`` ``(x, y, n_features)`` codebook (in the scaled
        training space when ``scale=True``), ``umatrix`` ``(x, y)``
        **unnormalized** mean neighbor codebook distance, ``hit_counts``,
        ``node_metacluster`` (flat ``(x*y,)`` metacluster id per node), and
        ``params`` (including ``quantization_error`` / ``topographic_error``).

    Returns
    -------
    MiniSom
        The trained MiniSom object.

    References
    ----------
    Kohonen (2001) *Self-Organizing Maps*, 3rd ed., Springer.
    Vesanto & Alhoniemi (2000) IEEE Trans. Neural Netw. 11(3):586-600.
    Van Gassen et al. (2015) Cytometry A 87(7):636-645 (FlowSOM).

    Examples
    --------
    >>> ab.tl.pca(mdata)
    >>> ab.tl.som(mdata, n_metaclusters=15, seed=42)
    >>> ab.pl.som_umatrix(mdata).show()
    >>> ab.pl.points(mdata, color='som_cluster').show()
    """
    MiniSom = _require_minisom()

    if use_rep is None:
        import scipy.sparse as sp
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        data = np.asarray(X, dtype=np.float32)
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"obsm['{use_rep}'] not found. Run ab.tl.pca() first")
        data = np.asarray(adata.obsm[use_rep], dtype=np.float32)

    if scale is None:
        scale = use_rep is None  # z-score raw features; keep embeddings' variance ordering
    if scale:
        data = _zscore(data)

    n_obs, n_features = data.shape

    side = max(5, int(np.ceil(np.sqrt(5 * np.sqrt(n_obs)))))
    sx = x if x is not None else side
    sy = y if y is not None else (x if x is not None else side)
    sigma_eff = sigma if sigma is not None else _default_sigma(sx, sy)

    log.info(
        "som: training %d×%d grid on %d cells (%d features, %d epochs, sigma=%.2f→1, scale=%s)",
        sx, sy, n_obs, n_features, n_epochs, sigma_eff, scale,
    )

    som_obj = _train_som(
        MiniSom, data, sx, sy,
        sigma=sigma_eff, learning_rate=learning_rate,
        n_epochs=n_epochs, seed=seed, init=init,
    )

    bmu_flat = _batch_bmu(som_obj, data)
    bmu_x, bmu_y = np.unravel_index(bmu_flat, (sx, sy))
    bmu_coords = np.column_stack([bmu_x, bmu_y]).astype(np.float32)
    node_labels = [f"{bmu_x[i]}_{bmu_y[i]}" for i in range(n_obs)]

    weights = som_obj.get_weights().astype(np.float32)

    if n_metaclusters is not None:
        node_meta, sample_meta = _metacluster_codebook(weights, n_metaclusters, bmu_flat)
        cluster_labels = sample_meta.astype(str)
        cluster_categories = [str(i) for i in sorted(np.unique(sample_meta).tolist())]
    else:
        node_meta = np.arange(sx * sy, dtype=np.int32)
        cluster_labels = np.asarray(node_labels)
        cluster_categories = _grid_sorted(set(node_labels))

    qe, te = _quality_metrics(som_obj, data, seed)

    hit_counts = np.zeros(sx * sy, dtype=np.int32)
    np.add.at(hit_counts, bmu_flat, 1)
    hit_counts = hit_counts.reshape(sx, sy)

    adata.obsm["X_" + key_added] = bmu_coords
    # explicit numeric/grid category order: plain pd.Categorical sorts string
    # labels lexicographically ("0", "1", "10", "11", ..., "2"), which scrambles
    # value_counts().sort_index() and every legend
    adata.obs[key_added + "_cluster"] = pd.Categorical(
        cluster_labels, categories=cluster_categories
    )
    adata.obs[key_added + "_node"] = pd.Categorical(
        node_labels, categories=_grid_sorted(set(node_labels))
    )
    adata.uns[key_added] = {
        "weights": weights,
        "umatrix": _unnormalized_umatrix(weights),
        "hit_counts": hit_counts,
        "node_metacluster": node_meta,
        "params": {
            "x": sx, "y": sy, "sigma": sigma_eff, "learning_rate": learning_rate,
            "n_epochs": n_epochs, "use_rep": use_rep, "scale": scale, "init": init,
            "n_metaclusters": n_metaclusters, "seed": seed,
            "quantization_error": qe, "topographic_error": te,
        },
    }

    n_clusters = len(np.unique(cluster_labels))
    log.info(
        "som: done — %d metaclusters from %d nodes (QE=%.3f, TE=%.3f) → "
        "obs['%s_cluster'], obs['%s_node'], uns['%s']",
        n_clusters, sx * sy, qe, te, key_added, key_added, key_added,
    )
    return som_obj


def _self_dists(a):
    """Euclidean distance matrix of the rows of *a*, without a 3-D intermediate.

    ``a[:, None, :] - a[None, :, :]`` needs ``m**2 * n_features`` elements; here
    the features are cells, so that tensor reaches tens of GiB and (on Windows)
    is quietly served from the pagefile instead of raising. The Gram identity
    ``d^2 = |x|^2 + |y|^2 - 2 x.y`` keeps the peak allocation at ``m**2``.
    """
    a = np.asarray(a, dtype=np.float64)
    sq = np.einsum("ij,ij->i", a, a)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (a @ a.T)
    np.maximum(d2, 0.0, out=d2)      # kill round-off below zero before sqrt
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2)


@with_adata(mirror_uns=from_kwarg("key_added"))
def som_gene_modules(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    n_modules: int = 10,
    min_coherence: float = 0.05,
    n_pcs: int | None = 50,
    x: int | None = None,
    y: int | None = None,
    sigma: float | None = None,
    learning_rate: float = 0.5,
    n_epochs: int = 50,
    score_cells: bool = True,
    seed: int = 0,
    key_added: str = "som_gene_modules",
) -> pd.DataFrame:
    """Cluster genes into co-expression modules with a SOM + Ward meta-clustering.

    Each gene's expression profile across cells is the observation. Profiles
    are **z-scored per gene** and reduced to ``n_pcs`` principal components
    before training, so Euclidean distance in the training space approximates
    correlation distance between genes -- raw Euclidean distance between
    unscaled profiles reduces to overall expression magnitude, which produces
    one giant "highly expressed genes" module and noise.

    The identity is ``||a - b||^2 = 2n(1 - corr(a, b))`` for z-scored vectors,
    and it holds **exactly only before the PCA truncation**. Centering is an
    isometry, but projecting onto the top ``n_pcs`` of up to ``n_genes - 1``
    components shortens every distance by the discarded residual, so the
    training metric is correlation distance *restricted to the leading gene-space
    components*. That is a deliberate denoising choice, not an equivalence:
    ``uns[key_added]['params']` records ``pca_variance_retained`` and
    ``distance_fidelity`` (the Spearman correlation between PC-space pairwise
    distances and true correlation distances) so the approximation is
    measurable rather than assumed.

    The SOM nodes are then Ward meta-clustered into exactly ``n_modules``
    modules; a module's per-cell activity score uses
    :func:`scanpy.tl.score_genes` (signature mean minus a matched
    expression-bin control background), not a raw expression mean -- a raw
    mean of module members just re-measures the magnitude the clustering was
    built from.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    layer
        Expression layer for training. ``None`` uses ``X``. Should be
        log-normalized (the same convention :func:`scanpy.tl.score_genes`
        expects).
    n_modules
        Number of Ward metaclusters cut from the node codebook -- the exact
        number of reported modules (empty modules are dropped from outputs).
    min_coherence
        Mean within-module Pearson correlation below which a module is renamed
        ``'unassigned'`` rather than ``'M{k}'``, excluded from ``score_cells``,
        and reported separately. Per-gene z-scoring puts genes whose profile is
        Poisson noise on equal footing with real programs; having no
        correlation with anything, they carry almost no variance along the
        leading gene-space components, collapse toward the origin after PC
        truncation, and Ward hands the whole blob a single large cluster. That
        cluster is a **residual class**, not a co-expression module -- WGCNA's
        grey module (Langfelder & Horvath 2008, *BMC Bioinformatics* 9:559) --
        and naming it as one is how a 38%-of-the-panel catch-all gets reported
        as a finding. Set to ``0.0`` to disable and label every module ``M{k}``.
    n_pcs
        Number of principal components for the gene-profile reduction.
        ``None`` trains on the full z-scored profiles (only sensible for
        small cell counts).
    x
        Grid width. Auto-computed from ``n_genes`` via the ``5 * sqrt(n)``
        heuristic if ``None`` (e.g. ~10x10 for a 377-gene panel).
    y
        Grid height. Defaults to ``x``.
    sigma
        Initial neighborhood radius (anneals to 1). ``None`` uses
        ``max(1.0, min(x, y) / 4)``.
    learning_rate
        Initial learning rate (anneals to 0).
    n_epochs
        Full shuffled passes over the genes (default 50 -- gene counts are
        small, epochs are cheap).
    score_cells
        When ``True``, store per-cell module scores in
        ``obsm[key_added + '_scores']`` via :func:`scanpy.tl.score_genes`.
    seed
        Random seed.
    key_added
        Key prefix for stored outputs.

    Stores
    ------
    ``var['som_gene_module']``
        Categorical module label (``"M0"``..) per gene.
    ``var['som_gene_module_kme']``
        Each gene's correlation with its own module's eigengene (PC1 of the
        member z-scored profiles) -- WGCNA's kME. The high-|kME| members are
        the hub genes worth naming a module by; low-|kME| members are weakly
        attached. ``NaN`` for ``unassigned`` genes.
    ``uns[key_added]``
        Dict with ``gene_modules`` DataFrame (``gene``, ``node``, ``module``,
        ``kme``), ``module_names`` (ordered, non-empty, coherent modules only),
        ``module_coherence`` DataFrame (``module``,
        ``metacluster``, ``n_genes``, ``mean_within_r``, ``mean_between_r``,
        ``eigengene_var_explained``, ``coherent``) -- the table to print next to the module sizes, since a
        large module with ``mean_within_r`` near zero is a residual class --
        ``node_module`` (flat metacluster id per node), ``weights`` (the
        ``(x, y, n_dims)`` codebook, in gene-profile PCA space), ``umatrix``
        (unnormalized), ``hit_counts``, and ``params`` (incl. QE/TE,
        ``pca_variance_retained`` and ``distance_fidelity``).
    ``obsm[key_added + '_scores']``
        Float32 ``(n_obs, n_nonempty_modules)`` background-corrected module
        scores (only when ``score_cells=True``); column order matches
        ``uns[key_added]['module_names']``.

    Returns
    -------
    pd.DataFrame
        ``gene``, ``node``, ``module`` columns.

    Examples
    --------
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> gene_df = ab.tl.som_gene_modules(mdata, n_modules=10)
    >>> gene_df.groupby('module').size()
    """
    import scipy.sparse as sp

    MiniSom = _require_minisom()

    M = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=np.float32)

    gene_data = _zscore(M).T.copy()  # z-score per gene (column of M), genes as rows
    n_genes, n_cells = gene_data.shape

    var_retained, dist_fidelity = 1.0, 1.0
    if n_pcs is not None and n_pcs < n_cells:
        from sklearn.decomposition import PCA
        n_comps = min(n_pcs, n_genes - 1, n_cells - 1)
        pca = PCA(n_components=n_comps, random_state=seed)
        reduced = pca.fit_transform(gene_data).astype(np.float32)
        var_retained = float(pca.explained_variance_ratio_.sum())
        # How much of the correlation-distance geometry survived the
        # truncation? Measured on a seeded gene subsample so the cost is
        # bounded, and stored rather than asserted (see the docstring).
        rng_d = np.random.default_rng(seed)
        m = min(n_genes, 400)
        pick = rng_d.choice(n_genes, m, replace=False) if n_genes > m else np.arange(n_genes)
        d_true = _self_dists(gene_data[pick])
        d_pc = _self_dists(reduced[pick])
        iu = np.triu_indices(len(pick), 1)
        from scipy.stats import spearmanr
        dist_fidelity = float(spearmanr(d_true[iu], d_pc[iu]).statistic)
        gene_data = reduced

    side = max(3, int(np.ceil(np.sqrt(5 * np.sqrt(n_genes)))))
    sx = x if x is not None else side
    sy = y if y is not None else (x if x is not None else side)
    sigma_eff = sigma if sigma is not None else _default_sigma(sx, sy)

    log.info(
        "som_gene_modules: training %d×%d grid on %d genes "
        "(z-scored, %d dims, %d epochs, sigma=%.2f→1) → %d modules",
        sx, sy, n_genes, gene_data.shape[1], n_epochs, sigma_eff, n_modules,
    )

    som_obj = _train_som(
        MiniSom, gene_data, sx, sy,
        sigma=sigma_eff, learning_rate=learning_rate,
        n_epochs=n_epochs, seed=seed, init="pca",
    )

    bmu_flat = _batch_bmu(som_obj, gene_data, chunk=10_000)
    bmu_x, bmu_y = np.unravel_index(bmu_flat, (sx, sy))
    node_labels = [f"{bmu_x[i]}_{bmu_y[i]}" for i in range(n_genes)]

    weights = som_obj.get_weights().astype(np.float32)
    node_meta, gene_meta = _metacluster_codebook(weights, n_modules, bmu_flat)

    qe, te = _quality_metrics(som_obj, gene_data, seed)

    # non-empty modules, in metacluster-id order
    present = sorted(set(gene_meta.tolist()))
    n_empty = min(n_modules, sx * sy) - len(present)
    if n_empty > 0:
        log.warning("som_gene_modules: %d metacluster(s) captured no genes and were dropped", n_empty)

    # ---- coherence: is each module a program or a residual class? ---------
    zprof = _zscore(M).T  # genes x cells, z-scored per gene (pre-PCA)
    coh_rows = []
    kme = np.full(n_genes, np.nan)
    coherent: dict[int, bool] = {}
    for m in present:
        mem = np.flatnonzero(gene_meta == m)
        if len(mem) >= 2:
            R = np.corrcoef(zprof[mem])
            iu = np.triu_indices(len(mem), 1)
            within = float(np.nanmean(R[iu]))
            other = np.flatnonzero(gene_meta != m)
            between = (
                float(np.nanmean(np.corrcoef(zprof[mem], zprof[other])[:len(mem), len(mem):]))
                if len(other) else np.nan
            )
            # module eigengene = PC1 of the member profiles
            X = zprof[mem] - zprof[mem].mean(axis=1, keepdims=True)
            _, sv, vt = np.linalg.svd(X, full_matrices=False)
            eig = vt[0]
            # SVD fixes PC1 only up to sign; orient it with the module's mean
            # profile so kME is positive for typical members (WGCNA convention).
            if float(np.dot(eig, X.mean(axis=0))) < 0:
                eig = -eig
            ev = float(sv[0] ** 2 / np.maximum((sv ** 2).sum(), 1e-12))
            with np.errstate(invalid="ignore"):
                kme[mem] = [
                    float(np.corrcoef(zprof[g], eig)[0, 1]) for g in mem
                ]
        else:
            within = between = np.nan
            ev = np.nan
        # min_coherence <= 0 is the documented opt-out: label every module
        # M{k} and let the caller read module_coherence themselves.
        ok = min_coherence <= 0.0 or bool(
            np.isfinite(within) and within >= min_coherence
        )
        coherent[m] = ok
        coh_rows.append((
            f"M{m}" if ok else "unassigned", int(m), len(mem), within, between,
            ev, ok,
        ))
    module_coherence = pd.DataFrame(
        coh_rows,
        columns=["module", "metacluster", "n_genes", "mean_within_r",
                 "mean_between_r", "eigengene_var_explained", "coherent"],
    )

    label_of = {m: (f"M{m}" if coherent[m] else "unassigned") for m in present}
    module_labels = [label_of[m] for m in gene_meta]
    module_names = [f"M{m}" for m in present if coherent[m]]
    kme[[not coherent[m] for m in gene_meta]] = np.nan
    n_unassigned = int(sum(not coherent[m] for m in gene_meta))
    if n_unassigned:
        worst = module_coherence.loc[~module_coherence["coherent"]]
        log.warning(
            "som_gene_modules: %d/%d genes (%.0f%%) fell in %d module(s) with mean "
            "within-module r < %.2f and were labelled 'unassigned' rather than "
            "reported as programs (max size %d, r=%.3f) -- per-gene z-scoring puts "
            "noise-only genes on equal footing, and Ward collects them into one "
            "residual class (WGCNA's grey module)",
            n_unassigned, n_genes, 100.0 * n_unassigned / n_genes, len(worst),
            min_coherence, int(worst["n_genes"].max()),
            float(worst["mean_within_r"].max()),
        )

    # numeric category order (plain Categorical would sort "M10" before "M2");
    # 'unassigned' sorts last so it never leads a legend
    adata.var["som_gene_module"] = pd.Categorical(
        module_labels, categories=module_names + (["unassigned"] if n_unassigned else []),
    )
    adata.var["som_gene_module_kme"] = kme.astype(np.float32)

    gene_df = pd.DataFrame({
        "gene": adata.var_names.tolist(),
        "node": node_labels,
        "module": module_labels,
        "kme": kme.astype(np.float32),
    })

    if score_cells:
        import scanpy as sc
        scores = np.zeros((adata.n_obs, len(module_names)), dtype=np.float32)
        tmp_col = "_som_gene_module_score_tmp"
        for j, mod in enumerate(module_names):
            genes = gene_df.loc[gene_df["module"] == mod, "gene"].tolist()
            sc.tl.score_genes(adata, genes, score_name=tmp_col, random_state=seed)
            scores[:, j] = adata.obs[tmp_col].to_numpy(dtype=np.float32)
        adata.obs.drop(columns=[tmp_col], inplace=True)
        adata.obsm[key_added + "_scores"] = scores

    hit_counts = np.zeros(sx * sy, dtype=np.int32)
    np.add.at(hit_counts, bmu_flat, 1)

    adata.uns[key_added] = {
        "gene_modules": gene_df,
        "module_names": module_names,
        "module_coherence": module_coherence,
        "node_module": node_meta,
        "weights": weights,  # (x, y, n_dims) codebook; pl.som_quality_plot needs it
        "umatrix": _unnormalized_umatrix(weights),
        "hit_counts": hit_counts.reshape(sx, sy),
        "params": {
            "x": sx, "y": sy, "sigma": sigma_eff, "learning_rate": learning_rate,
            "n_epochs": n_epochs, "layer": layer, "n_pcs": n_pcs,
            "n_modules": n_modules, "seed": seed,
            "min_coherence": min_coherence,
            "pca_variance_retained": var_retained,
            "distance_fidelity": dist_fidelity,
            "quantization_error": qe, "topographic_error": te,
        },
    }

    log.info(
        "som_gene_modules: %d genes → %d coherent module(s) + %d unassigned "
        "(QE=%.3f, TE=%.3f; PCA retained %.0f%% of variance, distance fidelity "
        "%.2f) → var['som_gene_module'], var['som_gene_module_kme'], "
        "uns['%s']['module_coherence']",
        n_genes, len(module_names), n_unassigned, qe, te,
        100.0 * var_retained, dist_fidelity, key_added,
    )
    return gene_df


@with_adata(mirror_uns=from_kwarg("key_added"))
def som_stability(
    adata: ad.AnnData,
    *,
    key: str = "som",
    k_range: "range | list[int] | None" = None,
    n_seeds: int = 3,
    max_cells: int = 50_000,
    plateau_tol: float = 0.05,
    seed: int = 0,
    key_added: str = "som_stability",
) -> pd.DataFrame:
    """How reproducible is the metacluster count the map is cut at?

    A single Ward cut at ``n_metaclusters=k`` is not self-justifying: nothing
    about it says whether those ``k`` populations survive a change of random
    seed, or whether ``k`` sits anywhere special in the merge tree. FlowSOM's
    own protocol (Quintelier et al. 2021, *Nat Protoc* 16:3775-3801) inspects a
    range of ``k`` for exactly this reason, and this is the astrolabe analog.

    Two independent measures, both drawn by
    :func:`~astrolabe.pl.som_stability_plot`:

    - **Reproducibility.** ``n_seeds`` SOMs are trained on the same data with
      different random seeds (re-using the stored run's grid, epochs, scaling
      and representation), each is cut at every ``k``, and the mean pairwise
      adjusted Rand index between runs is reported per ``k``. Labels that do not
      survive a reseed are not a result.

      Expect this curve to **fall** with ``k`` and then flatten: cutting a
      codebook into more pieces necessarily puts more boundaries where the data
      does not fix them, so small ``k`` is always more reproducible than large
      ``k`` and the maximum is trivially at the smallest ``k`` tested. Read the
      ARI *at the ``k`` you intend to report* (``ari_at_reference_k``) and where
      the curve stops changing (``flat_from``/``flat_to``).
    - **Merge cost.** The Ward merge heights of the stored codebook. A
      pronounced jump means the merge below it fused genuinely distinct
      prototypes, so the count just above it is a natural stopping point; a
      smooth curve means the codebook is a continuum and no ``k`` is
      privileged, which is worth knowing before quoting one.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData` carrying a
        :func:`som` result under *key*.
    modality
        Modality key when *data* is a MuData.
    key
        ``uns`` key of the reference SOM run. Its ``params`` supply the grid
        size, epochs, ``use_rep``, ``scale``, ``sigma`` and ``init``, so the
        reseeded runs differ from it in nothing but the seed.
    k_range
        Metacluster counts to evaluate. ``None`` uses ``range(2, 2 * k0 + 1)``
        around the reference run's own ``n_metaclusters`` ``k0`` (or
        ``range(2, 21)`` if it had none).
    n_seeds
        Number of independently seeded SOMs. Each costs a full training run, so
        the default is 3 -- enough for three pairwise ARIs per ``k``.
    max_cells
        Train the stability runs on a seeded subsample of at most this many
        cells. Reproducibility of the *partition shape* is what is being
        measured, and it converges long before the full 150k; the ARI is
        computed on the same subsample for every seed so the comparison is
        exact.
    plateau_tol
        Successive-difference tolerance defining a *flat region* of the ARI curve,
        reported as ``uns[...]['flat_from']`` / ``['flat_to']``. Reproducibility
        typically declines with ``k`` and then levels off, so the informative
        summary is where the curve stops changing -- not proximity to its
        maximum, which on a declining curve always points at the smallest,
        trivially stable ``k``. The number to quote alongside a chosen ``k`` is
        ``ari_at_reference_k``.
    seed
        Base seed. Run ``i`` uses ``seed + i``, and the reference run's own seed
        is deliberately not reused so no run is privileged.
    key_added
        ``uns`` key for the result.

    Stores
    ------
    ``uns[key_added]``
        Dict with ``k``, ``mean_ari``, ``sd_ari``, ``merge_k``,
        ``merge_heights``, ``flat_from`` / ``flat_to`` / ``plateau_tol``,
        ``reference_k`` and ``ari_at_reference_k``, ``n_seeds``, ``n_cells``, and
        ``params``.
        Mirrored to ``mdata.uns[key_added][<modality>]`` for a MuData.

    Returns
    -------
    pd.DataFrame
        ``k``, ``mean_ari``, ``sd_ari`` -- one row per evaluated ``k``.

    Examples
    --------
    >>> ab.tl.som(mdata, x=20, y=20, n_metaclusters=15, seed=42)
    >>> ab.tl.som_stability(mdata)
    >>> ab.pl.som_stability_plot(mdata).show()
    """
    from itertools import combinations

    from scipy.cluster.hierarchy import linkage
    from sklearn.metrics import adjusted_rand_score

    MiniSom = _require_minisom()

    if key not in adata.uns:
        raise KeyError(f"uns['{key}'] not found. Run ab.tl.som() first")
    ref = adata.uns[key]
    p = ref.get("params", {})
    sx, sy = int(p["x"]), int(p["y"])
    use_rep = p.get("use_rep", "X_pca")
    scale = bool(p.get("scale", False))
    n_epochs = int(p.get("n_epochs", 10))
    init = p.get("init", "pca")
    sigma_eff = float(p.get("sigma", _default_sigma(sx, sy)))
    k0 = p.get("n_metaclusters") or 10
    ks = list(k_range) if k_range is not None else list(range(2, 2 * int(k0) + 1))
    ks = [int(k) for k in ks if k >= 2]
    if not ks:
        raise ValueError("k_range contains no value >= 2")

    if use_rep is None:
        import scipy.sparse as sp
        X = adata.X
        data = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
    else:
        if use_rep not in adata.obsm:
            raise KeyError(f"obsm['{use_rep}'] not found (the reference run used it)")
        data = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    if scale:
        data = _zscore(data)

    rng = np.random.default_rng(seed)
    if len(data) > max_cells:
        sub = rng.choice(len(data), max_cells, replace=False)
        data = data[sub]

    log.info(
        "som_stability: %d seeded %d×%d run(s) on %d cells, k in [%d, %d]",
        n_seeds, sx, sy, len(data), min(ks), max(ks),
    )

    # labels[i][k] = partition of the subsample from run i cut at k
    labels: list[dict[int, np.ndarray]] = []
    for i in range(n_seeds):
        som_obj = _train_som(
            MiniSom, data, sx, sy, sigma=sigma_eff,
            learning_rate=float(p.get("learning_rate", 0.5)),
            n_epochs=n_epochs, seed=seed + i, init=init,
        )
        bmu = _batch_bmu(som_obj, data)
        w = som_obj.get_weights().astype(np.float32)
        labels.append({k: _metacluster_codebook(w, k, bmu)[1] for k in ks})

    mean_ari, sd_ari = [], []
    for k in ks:
        vals = [
            adjusted_rand_score(labels[a][k], labels[b][k])
            for a, b in combinations(range(n_seeds), 2)
        ]
        mean_ari.append(float(np.mean(vals)) if vals else np.nan)
        sd_ari.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    # ---- Ward merge heights of the stored codebook, populated nodes only ----
    w_ref = np.asarray(ref["weights"])
    codebook = w_ref.reshape(-1, w_ref.shape[-1]).astype(np.float64)
    hits = np.asarray(ref.get("hit_counts", np.ones(len(codebook)))).ravel()
    pop = codebook[hits > 0] if len(hits) == len(codebook) else codebook
    Z = linkage(pop, method="ward")
    heights = Z[:, 2]
    merge_k = np.arange(len(pop) - 1, 0, -1)  # clusters left after each merge

    result = pd.DataFrame({"k": ks, "mean_ari": mean_ari, "sd_ari": sd_ari})
    adata.uns[key_added] = {
        "k": np.asarray(ks, dtype=np.int32),
        "mean_ari": np.asarray(mean_ari, dtype=np.float32),
        "sd_ari": np.asarray(sd_ari, dtype=np.float32),
        "merge_k": merge_k.astype(np.int32),
        "merge_heights": heights.astype(np.float32),
        "n_seeds": int(n_seeds),
        "n_cells": int(len(data)),
        "params": {
            "key": key, "x": sx, "y": sy, "use_rep": use_rep, "scale": scale,
            "n_epochs": n_epochs, "seed": int(seed), "max_cells": int(max_cells),
        },
    }
    arr = np.asarray(mean_ari, dtype=float)
    ks_arr = np.asarray(ks)

    # A *flat region*, not proximity to the maximum. Reproducibility usually
    # falls with k and then levels off, so "the largest k within tolerance of
    # the best" just returns wherever the maximum happens to sit -- k=2 on a
    # declining curve, which says nothing about the k anyone would report. What
    # is informative is where the curve stops changing.
    flat_start = flat_end = int(ks_arr[0])
    if len(arr) >= 3:
        d = np.abs(np.diff(arr))
        run_start, best = 0, (0, 0)
        for i in range(len(d)):
            if d[i] > plateau_tol:
                run_start = i + 1
            elif i + 1 - run_start > best[1] - best[0]:
                best = (run_start, i + 1)
        flat_start, flat_end = int(ks_arr[best[0]]), int(ks_arr[best[1]])

    ari_at_ref = (
        float(result.loc[result["k"] == int(k0), "mean_ari"].iloc[0])
        if int(k0) in ks else float("nan")
    )
    adata.uns[key_added]["flat_from"] = flat_start
    adata.uns[key_added]["flat_to"] = flat_end
    adata.uns[key_added]["plateau_tol"] = float(plateau_tol)
    adata.uns[key_added]["reference_k"] = int(k0)
    adata.uns[key_added]["ari_at_reference_k"] = ari_at_ref
    log.info(
        "som_stability: ARI at the reported k=%s is %.3f; the curve is flat "
        "(successive change < %.2f) from k=%d to k=%d; ARI runs %.3f at k=%d "
        "down to %.3f at k=%d -> uns['%s']",
        k0, ari_at_ref, plateau_tol, flat_start, flat_end,
        float(arr[0]), int(ks_arr[0]), float(arr[-1]), int(ks_arr[-1]), key_added,
    )
    return result
