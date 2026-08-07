import numpy as np
import pandas as pd
import pytest
import anndata as ad
import mudata as md
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    """10 cells x 20 genes; sparse X; spatial coords; nucleus/cell area obs."""
    rng = np.random.default_rng(0)
    X = sp.random(10, 20, density=0.4, format="csr", random_state=0).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "nucleus_area": [5.0, 8.0, 3.0, 0.0, 7.0, 2.0, 6.0, 4.0, 5.5, 0.0],
            "cell_area": [20.0, 25.0, 18.0, 0.0, 22.0, 15.0, 21.0, 19.0, 23.0, 0.0],
        },
        index=[f"cell_{i}" for i in range(10)],
    )
    obsm = {"spatial": rng.uniform(0, 500, (10, 2)).astype(np.float32)}
    return ad.AnnData(X=X, obs=obs, obsm=obsm)


@pytest.fixture
def small_mdata(small_adata):
    mdata = md.MuData({"xenium": small_adata})
    mdata.uns["default_modality"] = "xenium"
    return mdata


@pytest.fixture
def small_mdata_two_mod(small_adata):
    """Same modality data under two different keys, for cross-modality mirroring tests."""
    mdata = md.MuData({"xenium": small_adata.copy(), "merscope": small_adata.copy()})
    mdata.uns["default_modality"] = "xenium"
    return mdata
