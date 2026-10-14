"""Real optional-backend comparisons (skipped when extras are not installed)."""
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
import astrolabe as ab
from tests.test_correctness_roadmap import cells


def test_liana_local_uses_layer_graph_and_matches_backend():
    li = pytest.importorskip("liana")
    a = cells(40); a.var_names = ["L","R"]
    a.layers["signal"] = np.random.default_rng(2).uniform(.1,2.,(40,2))
    pairs = ab.tl.import_ligrec_resource(pd.DataFrame({"ligand":["L"],"receptor":["R"]}),species="human")
    result = ab.tl.ligrec_local(a,pairs,layer="signal",n_perms=9,seed=3)
    ref = ad.AnnData(a.layers["signal"].copy(),obs=a.obs.copy(),var=a.var.copy())
    ref.obsp["graph"] = a.obsp["spatial_neighbors_connectivities"].copy()
    expected = li.mt.bivariate(ref,resource=pairs,local_name="cosine",global_name=None,
                               connectivity_key="graph",use_raw=False,n_perms=9,seed=3,
                               nz_prop=0,remove_self_interactions=False)
    actual = a.obsm["ligrec_local"]
    actual = actual.toarray() if sp.issparse(actual) else actual
    values = expected.X.toarray() if sp.issparse(expected.X) else expected.X
    np.testing.assert_allclose(actual,values)
    assert result["metadata"]["layer"] == "signal"


def test_liana_bundled_resource_expands_complexes_offline():
    pytest.importorskip("liana")
    table = ab.tl.import_ligrec_resource("consensus",species="human")
    complexes = table[table.ligand.str.contains("_")]
    assert len(complexes) > 0
    row = complexes.iloc[0]
    assert row.ligand_members == row.ligand.split("_")


def pseudobulk_fixture():
    rng = np.random.default_rng(23)
    counts = rng.negative_binomial(20,.3,(160,24))
    counts[80:,:3] *= 3
    a = ad.AnnData(sp.csr_matrix(counts)); a.var_names = [f"g{i}" for i in range(24)]
    a.layers["counts"] = a.X.copy()
    a.obs["sample"] = np.repeat([f"s{i}" for i in range(8)],20)
    a.obs["condition"] = ["ctrl"]*80+["case"]*80
    return ab.pp.pseudobulk(a,sample_key="sample",metadata_keys=["condition"])


def test_pydeseq2_adapter_matches_direct_backend():
    pytest.importorskip("pydeseq2")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    bulk = pseudobulk_fixture()
    result = ab.tl.rank_genes_pseudobulk(bulk,condition_key="condition",contrast=("case","ctrl"))
    inference = DefaultInference(n_cpus=1)
    counts = pd.DataFrame(bulk.X.toarray(),index=bulk.obs_names,columns=bulk.var_names)
    dds = DeseqDataSet(counts=counts,metadata=bulk.obs.copy(),design="~ condition",refit_cooks=True,
                       size_factors_fit_type="poscounts",inference=inference)
    dds.deseq2()
    stats = DeseqStats(dds,contrast=["condition","case","ctrl"],inference=inference)
    stats.summary()
    expected = stats.results_df
    actual = result.set_index("gene").reindex(expected.index)
    for key in ["log2FoldChange","pvalue","stat"]:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-7,equal_nan=True)
    assert actual.loc[["g0","g1","g2"],"log2FoldChange"].min() > 1


def test_pseudobulk_rejects_condition_confounding():
    pytest.importorskip("pydeseq2")
    bulk = pseudobulk_fixture(); bulk.obs["batch"] = bulk.obs.condition
    with pytest.raises(ValueError,match="confounded"):
        ab.tl.rank_genes_pseudobulk(bulk,condition_key="condition",contrast=("case","ctrl"),covariates=["batch"])
