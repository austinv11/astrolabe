"""Explicit working-memory budgets and opt-in disposable disk scratch."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@contextmanager
def _scratch(shape, *, disk_backed=False, scratch_dir=None, dtype=np.float64):
    if not disk_backed:
        if scratch_dir is not None:
            raise ValueError("scratch_dir requires disk_backed=True")
        yield np.empty(shape, dtype=dtype)
        return
    with TemporaryDirectory(prefix="astrolabe-", dir=scratch_dir) as directory:
        array = np.memmap(Path(directory) / "scratch.dat", mode="w+", dtype=dtype, shape=shape)
        try:
            yield array
        finally:
            array.flush()
            array._mmap.close()


def _budget(required, limit, description):
    if not isinstance(limit, (int, np.integer)) or limit <= 0:
        raise ValueError("max_working_bytes must be a positive integer")
    if required > limit:
        raise MemoryError(f"{description} requires approximately {required:,} working bytes; "
                          f"budget is {limit:,}. Reduce fields/permutations/workers or explicitly increase max_working_bytes. "
                          "Computation is never automatically spilled to disk.")
