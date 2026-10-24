"""Content fingerprints and ownership for derived analysis results."""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

_REGISTRY = "_astrolabe_dependencies"


def _digest(value):
    """Hash array buffers incrementally without densifying sparse matrices."""
    h = hashlib.sha256()

    def add(item):
        if type(item).__module__.startswith("cupy"):
            item = item.get()
        if type(item).__module__.startswith("awkward"):
            import awkward as ak
            form, length, buffers = ak.to_buffers(ak.to_packed(item))
            add(form.to_json())
            add(length)
            add(buffers)
        elif sp.issparse(item):
            item = item.tocsr()
            add(item.shape)
            for part in (item.indptr, item.indices, item.data):
                add(part)
        elif isinstance(item, np.ndarray) and not item.dtype.hasobject:
            h.update(str((item.shape, item.dtype.str)).encode())
            # Iteration avoids an entire contiguous copy for strided inputs.
            for block in np.nditer(item, flags=["external_loop", "buffered", "zerosize_ok"],
                                   op_flags=["readonly"], order="C", buffersize=65536):
                h.update(block.tobytes())
        elif isinstance(item, (pd.DataFrame, pd.Series, pd.Index)):
            add(pd.util.hash_pandas_object(item, index=True).to_numpy())
            if isinstance(item, pd.DataFrame):
                add(list(item.columns))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        else:
            h.update(pickle.dumps(item, protocol=5))
    add(value)
    return h.hexdigest()


def _read(adata, source):
    attr, key = source.split(":", 1)
    if attr == "axis":
        return getattr(adata, key + "_names")
    if attr == "X":
        return adata.X
    if attr == "raw":
        return None if adata.raw is None else (adata.raw.var_names, adata.raw.X)
    return getattr(adata, attr).get(key)


def _fingerprints(adata, sources):
    return {source: _digest(_read(adata, source)) for source in sources}


def _inputs(adata, function, arguments):
    """Resolve existing input fields from the public call and graph metadata."""
    args = dict(arguments)
    args.update(args.pop("kwargs", {}))
    sources = {"axis:obs"}
    module = function.__module__.rsplit(".", 1)[-1]
    if "core_key" in args or "tissue_support" in args:
        from astrolabe._spatial_units import _unit_keys
        for key in _unit_keys(adata, args.get("library_key", "auto"), args.get("core_key", "auto"), args.get("neighbors_key")):
            if key is not None:
                sources.add("obs:" + key)
        sources.add("uns:spatial_units")
        support = args.get("tissue_support", "auto")
        if support == "auto":
            support = adata.uns.get("spatial_units", {}).get("support_key")
        if support is not None:
            sources.add("uns:" + support)
    if module == "_graph" and function.__name__ == "spatial_neighbors":
        from astrolabe._spatial_units import _unit_keys
        for key in _unit_keys(adata, args.get("library_key", "auto"), args.get("core_key", "auto")):
            if key is not None:
                sources.add("obs:" + key)
        sources.add("uns:spatial_units")
        support = args.get("tissue_support", "auto")
        if support == "auto":
            support = adata.uns.get("spatial_units", {}).get("support_key")
        if support is not None:
            sources.add("uns:" + support)
    expression = ("layer" in args or module in {"_gene_modules", "_cohort"}
                  or function.__name__ in {"rank_genes_groups", "score_genes", "transcriptional_neighbors"})
    if expression:
        sources.add("axis:var")
        if args.get("use_raw") or (module == "_de" and args.get("use_raw") is None and adata.raw is not None):
            sources.add("raw:")
        else:
            layer = args.get("layer")
            sources.add("X:" if layer is None else "layers:" + layer)
        for name in ("measured_features", "measured_features_library_key"):
            sources.add("uns:" + name)
        coverage_key = adata.uns.get("measured_features_library_key", "library_id")
        if coverage_key in adata.obs:
            sources.add("obs:" + coverage_key)
    for param, value in args.items():
        if param in {"adata", "key_added", "score_name"}:
            continue
        values = [value] if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
        for key in values:
            if not isinstance(key, str):
                continue
            if key in adata.obs:
                sources.add("obs:" + key)
            if key in adata.obsm:
                sources.add("obsm:" + key)
            if (param.endswith("key") and param not in {"spatial_key", "boundaries_key", "cell_boundaries_key", "nucleus_boundaries_key"}
                    and key in adata.uns):
                sources.add("uns:" + key)
                metadata = adata.uns[key]
                if isinstance(metadata, dict):
                    if metadata.get("profile_key") in adata.obsm:
                        sources.add("obsm:" + metadata["profile_key"])
                    for name in ("connectivities_key", "distances_key"):
                        if metadata.get(name) in adata.obsp:
                            sources.add("obsp:" + metadata[name])
                    for name in ("library_key", "core_key", "spatial_key"):
                        field = metadata.get("params", {}).get(name)
                        if field in adata.obs:
                            sources.add("obs:" + field)
                        if field in adata.obsm:
                            sources.add("obsm:" + field)
    if module in {"_subcellular", "_flux", "_regions"}:
        for key in ("transcripts", "regions", "tissue_support"):
            if key in adata.uns:
                sources.add("uns:" + key)
        for key in adata.obsm:
            if "boundar" in key:
                sources.add("obsm:" + key)
    if module == "_audit":
        sources.update({"uns:transcripts_inferred", "uns:reconstruction", "obs:original_cell_id"})
    return _fingerprints(adata, sorted(sources))


def _register(adata, token, sources, outputs):
    registry = adata.uns.setdefault(_REGISTRY, {})
    previous = registry.get(token, {}).get("outputs", {})
    outputs = {attr: sorted(set(keys) | set(previous.get(attr, []))) for attr, keys in outputs.items()}
    outputs["uns"] = [key for key in outputs.get("uns", []) if not key.startswith("_astrolabe_")]
    # A function may consume and update a shared parameter table. Its final
    # value, rather than its own prior output, becomes the dependency baseline.
    for attr, keys in outputs.items():
        for key in keys:
            sources.pop(attr + ":" + key, None)
    registry[token] = {"sources": sources, "outputs": outputs}


def _stale(adata):
    records = adata.uns.get(_REGISTRY, {})
    current = _fingerprints(adata, {source for record in records.values() for source in record["sources"]})
    return {token: [source for source, digest in record["sources"].items() if current[source] != digest]
            for token, record in records.items()
            if any(current[source] != digest for source, digest in record["sources"].items())}


def _assert_sources_fresh(adata, sources):
    """Check consumed results and their upstream owners without removing data."""
    records = adata.uns.get(_REGISTRY, {})
    owners = {}
    for token, record in records.items():
        for attr, keys in record["outputs"].items():
            for key in keys:
                owners.setdefault(attr + ":" + key, []).append(token)
    checked, current = set(), {}

    def visit(source):
        for token in owners.get(source, []):
            if token in checked:
                continue
            checked.add(token)
            record = records[token]
            for upstream in record["sources"]:
                visit(upstream)
            changed = []
            for upstream, digest in record["sources"].items():
                if upstream not in current:
                    current[upstream] = _digest(_read(adata, upstream))
                if current[upstream] != digest:
                    changed.append(upstream)
            if changed:
                raise ValueError(f"Stale result {token}: changed {changed}. Recompute it or use tl.validate_results(invalidate=True).")

    for source in sources:
        visit(source)


def _assert_fresh(adata, attr, key):
    """Reject consumption of an owned output whose upstream source changed."""
    _assert_sources_fresh(adata, [attr + ":" + key])


def _get_result(adata, key):
    """Read a stored result, rejecting stale registered dependencies."""
    _assert_fresh(adata, "uns", key)
    return adata.uns[key]


def _invalidate_changed(adata):
    """Remove changed results and then downstream consumers; keep unrelated data."""
    removed = {}
    while stale := _stale(adata):
        records = adata.uns[_REGISTRY]
        for token, reasons in stale.items():
            record = records.pop(token)
            removed[token] = reasons
            for attr, keys in record["outputs"].items():
                for key in keys:
                    container = getattr(adata, attr)
                    if key in container:
                        del container[key]
    return removed


def validate_results(data, *, modality=None, invalidate=False):
    """Check registered results after edits made outside Astrolabe.

    Return a table of stale results and changed input fields. With
    ``invalidate=True``, discard stale outputs and dependent results, including
    MuData mirrors. Unregistered external results are not certified. Validation
    reads input buffers (sparse matrices remain sparse); it does not refit models.
    """
    from astrolabe._utils import _resolve_adata_and_modality
    from astrolabe._state import _clear_spatial_mirrors, _spatial_result_keys
    adata, selected = _resolve_adata_and_modality(data, modality)
    before = _spatial_result_keys(adata)
    stale = _invalidate_changed(adata) if invalidate else _stale(adata)
    if invalidate and selected is not None:
        _clear_spatial_mirrors(data, selected, before - set(adata.uns))
    return pd.DataFrame([{"result": key, "changed_sources": ", ".join(reasons)}
                         for key, reasons in stale.items()], columns=["result", "changed_sources"])
