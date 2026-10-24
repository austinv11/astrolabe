"""Prespecified graph-scale sensitivity using existing spatial tests."""
from __future__ import annotations

import copy
import anndata as ad
import pandas as pd

from astrolabe._statistics import adjust_p
from astrolabe._utils import from_kwarg, with_adata


@with_adata(mirror_uns=from_kwarg("key_added"))
def spatial_scale_sensitivity(adata, keys, *, graph_specs, statistic="moran", layer=None,
                               spatial_key="spatial", library_key="auto", core_key="auto",
                               tissue_support="auto", n_perms=999, seed=0,
                               p_adjust_method="holm", key_added="spatial_scale_sensitivity"):
    """Repeat Moran's I or Geary's C over prespecified neighborhood graphs.

    ``graph_specs`` is a mapping of unique scale names to ``spatial_neighbors``
    parameters, e.g. ``{"r20": {"radius": 20}, "r40": {"radius": 40}}``.
    All scales use the same within-unit permutation seed and tissue boundaries.
    Return every field/scale, with multiplicity adjustment across the entire
    requested family. Holm is the default because scales are dependent. This
    reports sensitivity; it does not select a best scale or fit a new test.
    Original graphs and analysis results are retained.
    """
    from astrolabe.tl._graph import spatial_neighbors
    from astrolabe.tl._spatial_stats import moran, geary_c
    if statistic not in ("moran", "geary"):
        raise ValueError("statistic must be moran or geary")
    if not isinstance(graph_specs, dict) or not graph_specs:
        raise ValueError("Provide named prespecified graph specifications")
    if not isinstance(n_perms, int) or n_perms < 1:
        raise ValueError("Sensitivity inference requires positive n_perms")
    fields = [keys] if isinstance(keys, str) else list(keys)
    if not fields or len(set(fields)) != len(fields):
        raise ValueError("Select distinct nonempty fields")
    allowed = {"n_neighbors", "radius", "coord_type", "delaunay_max_dist", "contact_distance", "boundaries_key"}
    for name, spec in graph_specs.items():
        if not isinstance(name, str) or not name or not isinstance(spec, dict) or set(spec)-allowed:
            raise ValueError("Graph specifications must have nonempty string names and supported graph parameters")
    # Share read-only expression buffers; graph construction uses a separate
    # AnnData so it cannot invalidate the caller's graphs or derived results.
    temporary = ad.AnnData(X=adata.X, obs=adata.obs.copy(), var=adata.var.copy())
    if layer is not None:
        temporary.layers[layer] = adata.layers[layer]
    for key in adata.obsm:
        temporary.obsm[key] = adata.obsm[key]
    support = adata.uns.get("spatial_units", {}).get("support_key") if tissue_support == "auto" else tissue_support
    for key in ("spatial_units", support, "measured_features", "measured_features_library_key", "spatial"):
        if key in adata.uns:
            temporary.uns[key] = copy.deepcopy(adata.uns[key])
    parts = []
    for name, spec in graph_specs.items():
        spatial_neighbors(temporary, spatial_key=spatial_key, library_key=library_key,
                          core_key=core_key, tissue_support=tissue_support, **spec)
        frame = (moran if statistic == "moran" else geary_c)(temporary, fields, layer=layer,
                   n_perms=n_perms, seed=seed).copy()
        frame["scale"] = name
        frame["n_edges"] = temporary.obsp["spatial_neighbors_connectivities"].nnz
        parts.append(frame.rename_axis("field").reset_index())
    result = pd.concat(parts, ignore_index=True)
    result["p_adjusted_family"] = adjust_p(result.p_sim.to_numpy(), method=p_adjust_method)
    result.attrs.update(statistic=statistic, graph_specs=copy.deepcopy(graph_specs),
                        family="all requested fields and scales", p_adjust_method=p_adjust_method,
                        n_perms=n_perms, seed=seed, scale_selection="none")
    adata.uns[key_added] = result
    return result
