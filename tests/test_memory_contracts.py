import anndata as ad
import numpy as np
import pandas as pd
import pytest

import astrolabe as ab
from astrolabe._memory import _scratch


@pytest.mark.parametrize("method", ["moran_local", "moran_local_bv", "getis_ord"])
def test_conditional_null_chunk_and_disk_equivalence(method, tmp_path):
    rng = np.random.default_rng(3)
    a = ad.AnnData(rng.poisson(3, (37, 2)).astype(float))
    a.var_names = ["a", "b"]
    a.obsm["spatial"] = rng.uniform(0, 10, (37, 2))
    ab.tl.spatial_neighbors(a, n_neighbors=5)
    args = ("a", "b") if method == "moran_local_bv" else ("a",)
    fn = getattr(ab.tl, method)
    reference = fn(a, *args, n_perms=39, seed=2, chunk_size=37)
    chunked = fn(a, *args, n_perms=39, seed=2, chunk_size=3)
    backed = fn(a, *args, n_perms=39, seed=2, chunk_size=3, disk_backed=True, scratch_dir=tmp_path)
    pd.testing.assert_frame_equal(reference, chunked)
    pd.testing.assert_frame_equal(reference, backed)
    assert not list(tmp_path.iterdir())
    with pytest.raises(MemoryError, match="never automatically"):
        fn(a, *args, n_perms=39, seed=2, max_working_bytes=100)


def test_disk_scratch_cleanup_after_error_and_no_default_disk_access(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError):
        with _scratch((10, 10), disk_backed=True, scratch_dir=tmp_path) as buffer:
            buffer[:] = 1
            raise RuntimeError("interrupted computation")
    assert not list(tmp_path.iterdir())
    def forbidden(*args, **kwargs):
        raise AssertionError("default computation attempted to use disk")
    monkeypatch.setattr("astrolabe._memory.TemporaryDirectory", forbidden)
    with _scratch((10, 10)) as buffer:
        assert isinstance(buffer, np.ndarray) and not isinstance(buffer, np.memmap)
