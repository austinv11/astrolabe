"""Compositional spatial neighborhoods and joint niche discovery."""
from __future__ import annotations
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from astrolabe._utils import with_adata, from_kwarg
from astrolabe.tl._neighborhood import _category_codes, _one_hot
from astrolabe.tl._spatial_stats import _get_conn_matrix


@with_adata(mirror_uns=from_kwarg("key_added"))
def niche_composition(adata: ad.AnnData, groupby: str, *, neighbors_key="spatial_neighbors",
                      library_key="library_id", key_added="niche_composition"):
    """Store non-self, one-hop cell-type fractions as CSR in ``obsm``.

    Edges are binary and libraries are isolated even for a supplied graph.
    Isolated cells have a zero profile and are marked invalid in obs.
    Categories retain their declared order in uns metadata.
    """
    codes, categories = _category_codes(adata, groupby)
    conn = _get_conn_matrix(adata, neighbors_key).tocsr(copy=True)
    conn.setdiag(0); conn.eliminate_zeros()
    conn.data[:] = 1
    if library_key is not None and library_key in adata.obs:
        if adata.obs[library_key].isna().any():
            raise ValueError("missing library labels")
        coo = conn.tocoo()
        labels = adata.obs[library_key].to_numpy()
        keep = labels[coo.row] == labels[coo.col]
        conn = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=conn.shape)
    counts = (conn @ _one_hot(codes, len(categories))).tocsr()
    degree = np.asarray(counts.sum(axis=1)).ravel()
    fractions = (sp.diags(np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)) @ counts).tocsr()
    adata.obsm[key_added] = fractions
    adata.obs[key_added + "_valid"] = degree > 0
    adata.uns[key_added] = {"categories": np.asarray(categories, dtype=str), "groupby": groupby,
                            "neighbors_key": neighbors_key, "library_key": library_key,
                            "profile_key": key_added, "definition": "binary non-self one-hop fractions"}
    return fractions


@with_adata(mirror_uns=from_kwarg("key_added"))
def niches(adata: ad.AnnData, groupby: str, *, neighbors_key="spatial_neighbors",
           library_key="library_id", n_neighbors=15, resolution=1., seed=0,
           key_added="niches"):
    """Joint Leiden clustering of variance-scaled neighborhood compositions.

    Clusters are shared across libraries; isolated cells stay unassigned.
    The expression matrix and existing expression/spatial graphs are retained.
    """
    import scanpy as sc
    from sklearn.preprocessing import StandardScaler
    profile_key = key_added + "_composition"
    profiles = niche_composition(adata, groupby, neighbors_key=neighbors_key,
                                 library_key=library_key, key_added=profile_key)
    valid = adata.obs[profile_key + "_valid"].to_numpy()
    assigned = np.full(adata.n_obs, None, dtype=object)
    if valid.sum() == 1:
        assigned[valid] = "0"
    elif valid.sum() > 1:
        scaled = StandardScaler(with_mean=False).fit_transform(profiles[valid])
        temporary = ad.AnnData(scaled)
        if valid.sum() == 2:
            assigned[valid] = "0"
        else:
            sc.pp.neighbors(temporary, n_neighbors=min(n_neighbors, valid.sum()-1), use_rep="X", random_state=seed)
            sc.tl.leiden(temporary, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2)
            assigned[valid] = temporary.obs["leiden"].astype(str).to_numpy()
    adata.obs[key_added] = pd.Categorical(assigned)
    adata.uns[key_added] = {"profile_key": profile_key, "n_neighbors": n_neighbors,
                            "resolution": resolution, "seed": seed, "scaling": "variance"}
    return adata.obs[key_added]
