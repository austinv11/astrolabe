"""Runnable synthetic examples for the five analysis extensions.

Run from the repository root with ``uv run --extra cohort --extra communication
python examples/analysis_extensions.py``. Without extras, the
core examples still run and optional examples print an installation hint.
"""
from pathlib import Path
import sys
import tempfile
import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely.geometry import box
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import astrolabe as ab
from astrolabe._utils import ImageSpec


def main():
    rng = np.random.default_rng(0)
    sample = np.repeat(np.arange(8),20)
    treatment = sample % 2 == 1
    counts = rng.negative_binomial(20,.3,(160,24))
    counts[treatment,:3] *= 3
    data = ad.AnnData(sp.csr_matrix(counts))
    data.var_names = [f"g{i}" for i in range(24)]
    data.layers["counts"] = data.X.copy()
    data.obs["sample"] = [f"sample{i}" for i in sample]
    data.obs["library_id"] = data.obs["sample"]
    data.obs["condition"] = np.where(treatment,"treated","control")
    data.obs["donor"] = [f"donor{i//2}" for i in sample]
    data.obs["cell_type"] = np.tile(["A"]*10+["B"]*10,8)
    data.obsm["spatial"] = rng.uniform(0,50,(160,2))
    ab.pp.normalize_total(data)
    ab.pp.log1p(data)
    data.layers["log_normalized"] = data.X.copy()
    ab.tl.spatial_neighbors(data,n_neighbors=4)

    ab.tl.niches(data,"cell_type",seed=0)
    print("Niche composition:")
    print(ab.tl.composition(data,groupby="cell_type",region_by="niches"))

    regions = gpd.GeoDataFrame({"region_id":["center"]*8,"library_id":[f"sample{i}" for i in range(8)]},
                               geometry=[box(10,10,40,40)]*8)
    ab.tl.assign_regions(data,regions)
    ab.tl.distance_to_region(data,regions,region="center")
    print("Region summary:")
    print(ab.tl.region_summary(data,sample_key="sample",groupby="cell_type"))

    bulk = ab.pp.pseudobulk(data,sample_key="sample",groupby="cell_type",metadata_keys=["condition","donor"])
    print("Pseudobulk shape:",bulk.shape)
    print(ab.tl.diff_nhood_enrichment(data,"cell_type",sample_key="sample",condition_key="condition",
                                     contrast=("treated","control"),paired_key="donor",n_perms=99,n_boot=99))
    try:
        print(ab.tl.rank_genes_pseudobulk(bulk,condition_key="condition",contrast=("treated","control"),paired_key="donor").head())
    except ImportError as exc:
        print(exc)

    resource = ab.tl.import_ligrec_resource(pd.DataFrame({"ligand":["g0_g1"],"receptor":["g2"],
               "ligand_members":[["g0","g1"]],"receptor_members":[["g2"]]}),species="human",version="synthetic-v1")
    ab.tl.ligrec(data,"cell_type",resource,layer="log_normalized",n_perms=99,seed=0)
    print("Complex LR means:",data.uns["ligrec"]["means"])
    try:
        ab.tl.ligrec_local(data,resource,layer="log_normalized",n_perms=19)
        print("Local LR shape:",data.obsm["ligrec_local"].shape)
    except ImportError as exc:
        print(exc)

    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory)/"texture.tif"
        image = ((np.indices((64,64)).sum(axis=0)%8)*32).astype(np.uint8)
        tifffile.imwrite(image_path,image)
        spec = ImageSpec(path=str(image_path),pixel_size_um=1.,dimension_order="YX",full_shape=(64,64),
                         level_shapes=[(64,64)],level_scales=[(1.,1.)])
        features = ab.pp.extract_image_features(data,spec=spec,library_id="sample0",mode="context",radius_um=3)
        print("Image features:",features.dropna().head())

if __name__ == "__main__":
    main()
