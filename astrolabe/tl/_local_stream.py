"""Bounded conditional local randomization using ESDA's seeded draw schedule."""
from __future__ import annotations

import numpy as np
from numba import njit

from astrolabe._statistics import permutation_p


@njit
def _simulate(start, end, x, y, indptr, weights, self_weights, draws, scaling, getis, output):
    for i in range(start, end):
        degree = indptr[i+1]-indptr[i]
        for p in range(len(draws)):
            lag = self_weights[i]*y[i] if degree else 0.
            for j in range(degree):
                position = draws[p, j]
                position += position >= i  # Sample all other cells; hold focal cell fixed.
                lag += y[position]*weights[indptr[i]+j]
            output[i-start, p] = lag/scaling if getis else x[i]*lag*scaling


def _conditional_summary(model, n_perms, seed, *, getis=False, chunk_memory_bytes=256*1024**2,
                         chunk_size=None):
    from esda.crand import vec_permutations
    from astrolabe.tl._spatial_stats import _directed_perm_p
    graph = model.w.sparse.tocsr(copy=True)
    self_weights = graph.diagonal().copy()
    graph.setdiag(0); graph.eliminate_zeros()
    n = graph.shape[0]
    max_card = int(np.diff(graph.indptr).max(initial=0))
    # Draw schedule plus one cell's simulations and reduction temporaries.
    fixed_bytes = n_perms*max_card*8 + (n*10 + graph.nnz)*8
    if not isinstance(chunk_memory_bytes, (int, np.integer)) or chunk_memory_bytes <= 0:
        raise ValueError("chunk_memory_bytes must be a positive integer")
    # This is a chunk-sizing target, never an allocation rejection threshold.
    # Fixed arrays or even a single row may exceed it; still process that row.
    allowed = max(1, (chunk_memory_bytes-fixed_bytes)//(max(1, n_perms)*48))
    if chunk_size is not None and (not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1):
        raise ValueError("chunk_size must be a positive integer")
    rows = min(n, allowed if chunk_size is None else chunk_size)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31-1))
    draws = vec_permutations(max_card, n, n_perms, seed)
    if getis:
        x = y = np.asarray(model.y, dtype=float)
        scaling = float(y.sum())
        observed = model.Gs
    else:
        x = np.asarray(model.zx if hasattr(model, "zx") else model.z)
        y = np.asarray(model.zy if hasattr(model, "zy") else model.z)
        scaling = (n-1)/float((x*x).sum()) if np.any(x) else np.nan
        observed = model.Is
    p = np.full(n, np.nan); directed = p.copy(); z = p.copy()
    valid = np.zeros(n, dtype=int)
    buffer = np.empty((rows, n_perms), dtype=np.float64)
    for start in range(0, n, rows):
        end = min(start+rows, n)
        sims = buffer[:end-start]
        if scaling == 0 or not np.isfinite(scaling):
            sims[:] = np.nan
        else:
            _simulate(start, end, x, y, graph.indptr, graph.data, self_weights, draws, scaling, getis, sims)
        p[start:end], valid[start:end] = permutation_p(observed[start:end], sims)
        directed[start:end] = _directed_perm_p(observed[start:end], sims)
        with np.errstate(divide="ignore", invalid="ignore"):
            z[start:end] = (observed[start:end]-sims.mean(axis=1))/sims.std(axis=1)
    return p, valid, directed, z
