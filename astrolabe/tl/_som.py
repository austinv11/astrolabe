"""
Self-organizing map analyses for spatial transcriptomics.
"""
from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd

from astrolabe._utils import from_kwarg, with_adata

__all__ = ["som", "som_gene_modules", "som_subcellular_domains"]

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
    """Vectorized BMU assignment; returns flat index array of length n_samples."""
    sx, sy, n_feat = som_obj.get_weights().shape
    w_flat = som_obj.get_weights().reshape(-1, n_feat).astype(np.float32)
    n = len(data)
    bmu_flat = np.empty(n, dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = data[start:end].astype(np.float32)
        dists = np.linalg.norm(d[:, None, :] - w_flat[None, :, :], axis=2)
        bmu_flat[start:end] = dists.argmin(axis=1)
    return bmu_flat


@with_adata(mirror_uns=from_kwarg("key_added"))
def som(
    adata: ad.AnnData,
    *,
    use_rep: str = "X_pca",
    x: int | None = None,
    y: int | None = None,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    n_iter: int | None = None,
    topology: str = "rectangular",
    seed: int = 0,
    key_added: str = "som",
):
    """Train a SOM on any ``obsm`` embedding or ``X`` and assign cells to BMU nodes.

    The SOM grid gives a discrete, topology-aware embedding. Nearby nodes contain
    similar cells; the U-matrix reveals natural cluster boundaries. Unlike PCA/UMAP,
    the grid topology is fixed and reproducible.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Training summary is stored in ``uns[key_added]`` and mirrored to
    ``mdata.uns[key_added][<modality>]`` when called on a MuData; the
    ``obsm``/``obs`` outputs (embedding, BMU cluster assignment) stay
    adata-resident only.

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
        Grid width (number of columns). Auto-computed from ``n_obs`` if ``None``.
    y
        Grid height (number of rows). Defaults to ``x`` if ``None``.
    sigma
        Neighborhood radius for weight updates.
    learning_rate
        Initial learning rate, decays to 0.01 during training.
    n_iter
        Total training iterations (random-sample draws). Defaults to
        ``100 * n_obs``.
    topology
        Grid topology: ``'rectangular'`` or ``'hexagonal'``.
    seed
        Random seed.
    key_added
        Prefix for all stored outputs.

    Stores
    ------
    ``obsm['X_' + key_added]``
        Float32 ``(n_obs, 2)`` BMU ``[col, row]`` coordinates.
    ``obs[key_added + '_cluster']``
        Categorical ``"{col}_{row}"`` cluster label per cell.
    ``uns[key_added]``
        Dict with ``weights`` ``(x, y, n_features)`` codebook,
        ``umatrix`` ``(x, y)`` unified distance matrix,
        ``hit_counts`` ``(x, y)`` cells-per-node counts, and ``params``.

    Returns
    -------
    MiniSom
        The trained MiniSom object (for ``quantization_error()``, etc.).

    Examples
    --------
    >>> ab.tl.pca(mdata)
    >>> ab.tl.som(mdata, x=8, y=8, seed=42)
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

    n_obs, n_features = data.shape

    side = max(5, int(np.ceil(np.sqrt(5 * np.sqrt(n_obs)))))
    sx = x if x is not None else side
    sy = y if y is not None else (x if x is not None else side)

    n_iter_eff = n_iter if n_iter is not None else 100 * n_obs

    log.info(
        "som: training %d×%d grid on %d cells (%d features, %d iterations)",
        sx, sy, n_obs, n_features, n_iter_eff,
    )

    som_obj = MiniSom(
        sx, sy, n_features,
        sigma=sigma,
        learning_rate=learning_rate,
        topology=topology,
        random_seed=seed,
    )
    som_obj.random_weights_init(data)
    som_obj.train(data, n_iter_eff, verbose=False)

    bmu_flat = _batch_bmu(som_obj, data)
    bmu_x, bmu_y = np.unravel_index(bmu_flat, (sx, sy))
    bmu_coords = np.column_stack([bmu_x, bmu_y]).astype(np.float32)
    cluster_labels = [f"{bmu_x[i]}_{bmu_y[i]}" for i in range(n_obs)]

    umatrix = som_obj.distance_map().astype(np.float32)
    # vectorized hit count from the flat BMU indices
    hit_counts = np.zeros(sx * sy, dtype=np.int32)
    np.add.at(hit_counts, bmu_flat, 1)
    hit_counts = hit_counts.reshape(sx, sy)

    adata.obsm["X_" + key_added] = bmu_coords
    adata.obs[key_added + "_cluster"] = pd.Categorical(cluster_labels)
    adata.uns[key_added] = {
        "weights": som_obj.get_weights().astype(np.float32),
        "umatrix": umatrix,
        "hit_counts": hit_counts,
        "params": {
            "x": sx, "y": sy, "sigma": sigma, "learning_rate": learning_rate,
            "topology": topology, "use_rep": use_rep, "n_iter": n_iter_eff,
        },
    }

    log.info(
        "som: done → obsm['X_%s'], obs['%s_cluster'], uns['%s']",
        key_added, key_added, key_added,
    )
    return som_obj


@with_adata(mirror_uns=from_kwarg("key_added"))
def som_gene_modules(
    adata: ad.AnnData,
    *,
    layer: str | None = None,
    n_modules: int = 25,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    n_iter: int | None = None,
    topology: str = "rectangular",
    score_cells: bool = True,
    seed: int = 0,
    key_added: str = "som_gene_modules",
) -> pd.DataFrame:
    """Cluster genes into co-expression modules using a SOM on the gene-by-cell matrix.

    Trains a SOM on the transposed expression matrix where each gene is an observation
    (its expression profile across cells) and cells are features. Genes landing on
    the same node share similar expression profiles. Unlike spatial gene modules,
    this requires no spatial graph.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Training summary is stored in ``uns[key_added]`` and mirrored to
    ``mdata.uns[key_added][<modality>]`` when called on a MuData; the
    ``var``/``obsm`` outputs (per-gene module assignment, per-module scores)
    stay adata-resident only.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    modality
        Modality key when *data* is a MuData; defaults to ``uns['default_modality']``.
    layer
        Expression layer for training. ``None`` uses ``X``.
    n_modules
        Approximate number of modules (SOM nodes). Grid is square:
        ``side = max(2, ceil(sqrt(n_modules)))``.
    sigma
        Neighborhood radius.
    learning_rate
        Initial learning rate.
    n_iter
        Training iterations. Defaults to ``100 * n_vars``.
    topology
        Grid topology: ``'rectangular'`` or ``'hexagonal'``.
    score_cells
        When ``True``, compute per-module mean expression for each cell and store in
        ``obsm[key_added + '_scores']``.
    seed
        Random seed.
    key_added
        Key prefix for stored outputs.

    Stores
    ------
    ``var['som_gene_module']``
        Categorical ``"{col}_{row}"`` module label per gene.
    ``uns[key_added]``
        Dict with ``gene_modules`` DataFrame, ``weights`` ``(x, y, n_cells)``
        codebook, and ``params``.
    ``obsm[key_added + '_scores']``
        Float32 ``(n_obs, n_modules)`` mean expression per module per cell
        (only when ``score_cells=True``).

    Returns
    -------
    pd.DataFrame
        ``gene``, ``module`` columns — same shape as :func:`spatial_gene_modules`.

    Examples
    --------
    >>> ab.pp.normalize_total(mdata)
    >>> ab.pp.log1p(mdata)
    >>> gene_df = ab.tl.som_gene_modules(mdata, n_modules=16)
    >>> gene_df.groupby('module').size()
    """
    import scipy.sparse as sp

    MiniSom = _require_minisom()

    M = adata.layers[layer] if layer is not None else adata.X
    if sp.issparse(M):
        M = M.toarray()
    M = np.asarray(M, dtype=np.float32)

    gene_data = M.T  # (n_genes, n_cells)
    n_genes, n_cells = gene_data.shape

    side = max(2, int(np.ceil(np.sqrt(n_modules))))
    sx, sy = side, side

    n_iter_eff = n_iter if n_iter is not None else 100 * n_genes

    log.info(
        "som_gene_modules: training %d×%d grid on %d genes (%d cells, %d iterations)",
        sx, sy, n_genes, n_cells, n_iter_eff,
    )

    som_obj = MiniSom(
        sx, sy, n_cells,
        sigma=sigma,
        learning_rate=learning_rate,
        topology=topology,
        random_seed=seed,
    )
    som_obj.random_weights_init(gene_data)
    som_obj.train(gene_data, n_iter_eff, verbose=False)

    bmu_flat = _batch_bmu(som_obj, gene_data, chunk=1000)
    bmu_x, bmu_y = np.unravel_index(bmu_flat, (sx, sy))
    bmu_labels = [f"{bmu_x[i]}_{bmu_y[i]}" for i in range(n_genes)]

    adata.var["som_gene_module"] = pd.Categorical(bmu_labels)

    gene_df = pd.DataFrame({"gene": adata.var_names.tolist(), "module": bmu_labels})

    if score_cells:
        all_modules = [f"{c}_{r}" for c in range(sx) for r in range(sy)]
        mod_idx = {m: i for i, m in enumerate(all_modules)}
        scores = np.zeros((n_cells, len(all_modules)), dtype=np.float32)
        for gene_i, mod in enumerate(bmu_labels):
            mi = mod_idx[mod]
            scores[:, mi] += M[:, gene_i]
        counts = np.zeros(len(all_modules), dtype=np.float32)
        for mod in bmu_labels:
            counts[mod_idx[mod]] += 1
        counts = np.where(counts > 0, counts, 1.0)
        scores /= counts[np.newaxis, :]
        adata.obsm[key_added + "_scores"] = scores

    adata.uns[key_added] = {
        "gene_modules": gene_df,
        "weights": som_obj.get_weights().astype(np.float32),
        "params": {
            "x": sx, "y": sy, "sigma": sigma, "learning_rate": learning_rate,
            "topology": topology, "layer": layer, "n_iter": n_iter_eff,
        },
    }

    log.info(
        "som_gene_modules: %d genes → %d modules in var['som_gene_module']",
        n_genes, sx * sy,
    )
    return gene_df


@with_adata(mirror_uns=from_kwarg("key_added"))
def som_subcellular_domains(
    adata: ad.AnnData,
    transcripts,
    *,
    x_col: str = "x_location",
    y_col: str = "y_location",
    feature_col: str = "feature_name",
    cell_id_col: str = "cell_id",
    spatial_key: str = "spatial",
    x: int | None = None,
    y: int | None = None,
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    n_iter: int | None = None,
    topology: str = "rectangular",
    seed: int = 0,
    key_added: str = "som_subcellular",
) -> pd.DataFrame:
    """Find recurring subcellular spatial patterns by training a SOM on intracellular transcript positions.

    Each transcript's ``(x, y)`` position is normalized relative to its cell's
    centroid and equivalent radius. The SOM discovers recurring spatial patterns
    across all cells — e.g. nuclear accumulation, peripheral enrichment, or
    polar asymmetry — without any prior annotation.

    Requires ``minisom`` (``uv add 'astrolabe[som]'``).

    Training summary is stored in ``uns[key_added]`` and mirrored to
    ``mdata.uns[key_added][<modality>]`` when called on a MuData; the
    ``obsm``/``obs`` outputs (per-domain fractions, dominant domain) stay
    adata-resident only.

    Parameters
    ----------
    data
        :class:`~anndata.AnnData` or :class:`~mudata.MuData`.
    transcripts
        Transcript table from :func:`~astrolabe.io.read_xenium_transcripts` or
        similar; accepts :class:`pyarrow.Table` or :class:`pandas.DataFrame`.
    modality
        Modality key when *data* is a MuData.
    x_col
        Column name for transcript x-coordinate in microns.
    y_col
        Column name for transcript y-coordinate in microns.
    feature_col
        Column name for gene/feature name.
    cell_id_col
        Column name identifying the cell each transcript belongs to.
        Values must match ``adata.obs_names``.
    spatial_key
        ``obsm`` key holding cell centroids ``(n_obs, 2)`` in µm.
    x
        Grid width. Defaults to 5.
    y
        Grid height. Defaults to ``x``.
    sigma
        Neighborhood radius.
    learning_rate
        Initial learning rate.
    n_iter
        Training iterations. Defaults to
        ``min(100 * n_transcripts, 1_000_000)``.
    topology
        Grid topology: ``'rectangular'`` or ``'hexagonal'``.
    seed
        Random seed.
    key_added
        Key prefix for stored outputs.

    Stores
    ------
    ``obs[key_added + '_dominant']``
        Categorical dominant domain (highest fraction) per cell.
    ``obsm[key_added + '_fractions']``
        Float32 ``(n_obs, x * y)`` fraction of transcripts in each domain
        per cell. Domain order matches ``uns[key_added]['domain_names']``.
    ``uns[key_added]``
        Dict with ``weights``, ``umatrix``, ``hit_counts``, ``domain_names``
        (ordered list), and ``params``.

    Returns
    -------
    pd.DataFrame
        Transcript-level domain assignments with columns
        ``cell_id``, ``x``, ``y``, ``gene``, ``domain``.
        Pass this to :func:`~astrolabe.pl.som_subcellular_plot`.

    Examples
    --------
    >>> transcripts = ab.io.read_xenium_transcripts("Xenium_run/")
    >>> tx_domains = ab.tl.som_subcellular_domains(mdata, transcripts, x=5, y=5)
    >>> ab.pl.som_subcellular_plot(mdata, tx_domains).show()
    >>> ab.pl.points(mdata, color='som_subcellular_dominant').show()
    """
    MiniSom = _require_minisom()

    # Convert PyArrow to pandas
    try:
        import pyarrow as pa
        if isinstance(transcripts, pa.Table):
            transcripts = transcripts.to_pandas()
    except ImportError:
        pass

    tx = transcripts[[cell_id_col, x_col, y_col, feature_col]].copy()
    tx[cell_id_col] = tx[cell_id_col].astype(str)

    obs_index = {cid: i for i, cid in enumerate(adata.obs_names)}
    mask = tx[cell_id_col].isin(obs_index)
    tx = tx[mask].reset_index(drop=True)

    if len(tx) == 0:
        raise ValueError(
            f"No transcripts matched adata.obs_names via '{cell_id_col}'. "
            "Check that cell IDs in the transcript table match obs_names."
        )

    centroids = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
    cell_idx_arr = tx[cell_id_col].map(obs_index).to_numpy(dtype=np.int32)

    cx = centroids[cell_idx_arr, 0]
    cy = centroids[cell_idx_arr, 1]
    dx = tx[x_col].to_numpy(dtype=np.float32) - cx
    dy = tx[y_col].to_numpy(dtype=np.float32) - cy

    if "cell_area" in adata.obs.columns:
        cell_areas = adata.obs["cell_area"].to_numpy(dtype=np.float32)
        r_all = np.sqrt(np.where(cell_areas > 0, cell_areas, 1.0) / np.pi)
        r_arr = r_all[cell_idx_arr]
    else:
        # per-cell bbox fallback: max(span_x, span_y) / 2
        tx_tmp = pd.DataFrame({"_cid": tx[cell_id_col].values, "_dx": dx, "_dy": dy})
        grp = tx_tmp.groupby("_cid").agg(
            _span=("_dx", lambda v: max(v.max() - v.min(), v.abs().max() * 2, 1.0))
        )
        grp2 = tx_tmp.groupby("_cid").agg(
            _span_y=("_dy", lambda v: max(v.max() - v.min(), v.abs().max() * 2, 1.0))
        )
        span = grp["_span"].combine(grp2["_span_y"], max) / 2
        r_arr = tx[cell_id_col].map(span.to_dict()).to_numpy(dtype=np.float32)
        r_arr = np.where(r_arr > 0, r_arr, 1.0)

    dx_norm = np.clip(dx / r_arr, -5.0, 5.0)
    dy_norm = np.clip(dy / r_arr, -5.0, 5.0)
    norm_data = np.column_stack([dx_norm, dy_norm]).astype(np.float32)

    sx = x if x is not None else 5
    sy = y if y is not None else (x if x is not None else 5)
    n_tx = len(norm_data)
    n_iter_eff = n_iter if n_iter is not None else min(100 * n_tx, 1_000_000)

    log.info(
        "som_subcellular_domains: training %d×%d grid on %d transcripts (%d cells, %d iter)",
        sx, sy, n_tx, tx[cell_id_col].nunique(), n_iter_eff,
    )

    som_obj = MiniSom(
        sx, sy, 2,
        sigma=sigma,
        learning_rate=learning_rate,
        topology=topology,
        random_seed=seed,
    )
    som_obj.random_weights_init(norm_data)
    som_obj.train(norm_data, n_iter_eff, verbose=False)

    bmu_flat = _batch_bmu(som_obj, norm_data)
    bmu_x_arr, bmu_y_arr = np.unravel_index(bmu_flat, (sx, sy))
    domains = [f"{bmu_x_arr[i]}_{bmu_y_arr[i]}" for i in range(n_tx)]
    tx = tx.copy()
    tx["domain"] = domains

    # ordered domain list
    all_domains = [f"{c}_{r}" for c in range(sx) for r in range(sy)]
    n_domains = len(all_domains)
    domain_idx = {d: i for i, d in enumerate(all_domains)}

    # per-cell fraction matrix using crosstab
    ct = pd.crosstab(tx[cell_id_col], tx["domain"])
    # reindex to all cells × all domains, fill missing with 0
    ct = ct.reindex(index=list(adata.obs_names), columns=all_domains, fill_value=0)
    row_sums = ct.sum(axis=1).to_numpy(dtype=np.float32)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    fractions = ct.to_numpy(dtype=np.float32) / row_sums[:, None]

    dominant_idx = fractions.argmax(axis=1)
    dominant = [all_domains[i] if fractions[j].sum() > 0 else "none"
                for j, i in enumerate(dominant_idx)]

    adata.obsm[key_added + "_fractions"] = fractions
    adata.obs[key_added + "_dominant"] = pd.Categorical(dominant)

    # vectorized hit counts
    hit_counts = np.zeros(sx * sy, dtype=np.int32)
    np.add.at(hit_counts, bmu_flat, 1)
    hit_counts = hit_counts.reshape(sx, sy)

    adata.uns[key_added] = {
        "weights": som_obj.get_weights().astype(np.float32),
        "umatrix": som_obj.distance_map().astype(np.float32),
        "hit_counts": hit_counts,
        "domain_names": all_domains,
        "params": {
            "x": sx, "y": sy, "sigma": sigma, "learning_rate": learning_rate,
            "topology": topology, "n_iter": n_iter_eff,
            "x_col": x_col, "y_col": y_col,
        },
    }

    result_df = pd.DataFrame({
        "cell_id": tx[cell_id_col].values,
        "x": tx[x_col].values,
        "y": tx[y_col].values,
        "gene": tx[feature_col].values,
        "domain": domains,
    })

    log.info(
        "som_subcellular_domains: %d domains → obs['%s_dominant'], obsm['%s_fractions']",
        n_domains, key_added, key_added,
    )
    return result_df
