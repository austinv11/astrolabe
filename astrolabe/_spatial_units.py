"""Physical sampling units, independent of biological donor identity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _unit_keys(adata, library_key="auto", core_key="auto", neighbors_key=None):
    config = dict(adata.uns.get("spatial_units", {}))
    if neighbors_key is not None:
        config.update(adata.uns.get(neighbors_key, {}).get("params", {}))
    resolved = []
    for name, key, default in (("library_key", library_key, "library_id"), ("core_key", core_key, "core_id")):
        if key == "auto":
            key = config.get(name, default if default in adata.obs else None)
        if key is not None:
            if key not in adata.obs or adata.obs[key].isna().any():
                raise ValueError(f"Spatial library labels and core labels must be complete: obs[{key!r}]")
        resolved.append(key)
    return tuple(resolved)


def _units(adata, library_key="auto", core_key="auto", neighbors_key=None):
    library_key, core_key = _unit_keys(adata, library_key, core_key, neighbors_key)
    columns = list(dict.fromkeys(k for k in (library_key, core_key) if k is not None))
    if not columns:
        return [((), np.arange(adata.n_obs))], columns
    labels = pd.MultiIndex.from_frame(adata.obs[columns])
    codes, levels = pd.factorize(labels, sort=False)
    return [(tuple(level), np.flatnonzero(codes == code)) for code, level in enumerate(levels)], columns


def _support(adata, labels, columns, key="tissue_support"):
    import shapely
    record = adata.uns.get(key)
    if record is None:
        raise ValueError(f"No observed tissue support in uns[{key!r}]; use pp.set_tissue_support")
    table = record["regions"]
    keep = np.ones(len(table), dtype=bool)
    for column, label in zip(columns, labels):
        if column not in table:
            raise ValueError(f"Tissue support lacks spatial-unit column {column!r}")
        keep &= table[column].to_numpy() == label
    geometries = shapely.from_wkb(table.loc[keep, "wkb"].to_numpy())
    if not len(geometries):
        raise ValueError(f"No observed tissue support for spatial unit {labels}")
    return shapely.union_all(geometries)
