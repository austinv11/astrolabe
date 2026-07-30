"""
astrolabe.tl — analysis tools for spatial transcriptomics.

Functions
---------
spatial_neighbors
    Build a spatial neighborhood graph from cell centroid coordinates.
transcriptional_neighbors
    Build an expression-space neighborhood graph (thin wrapper around sc.pp.neighbors).
blend_neighbors
    Blend an expression graph and a spatial graph into one joint graph, usable
    as a neighbors_key for both umap() (cospatial embedding) and cluster()
    (cospatial cluster labels).
pca
    PCA on the expression matrix or an arbitrary obsm embedding (e.g. BANKSY).
cluster
    Leiden or Louvain community detection (thin wrapper around sc.tl.leiden/louvain).
umap
    UMAP embedding (thin wrapper around sc.tl.umap).
paga
    Partition-based graph abstraction (PAGA) trajectory inference.
dpt
    Diffusion pseudotime ordering from a user-specified root cell.
rank_genes_groups
    Differential expression between clusters (thin wrapper around sc.tl.rank_genes_groups).
score_genes
    Score cells against a gene signature (thin wrapper around sc.tl.score_genes).
annotate_clusters
    Map cluster IDs to cell type labels and write them to obs.
geary_c
    Global Geary's C spatial autocorrelation for one or more keys; returns a DataFrame.
colocation_quotient
    Gene-pair co-location quotients (CLQ) in cell mode (expression matrix) or pixel mode (transcript grid).
moran
    Global Moran's I spatial autocorrelation for one or more keys; returns a DataFrame.
moran_local
    Local Moran's I (LISA) with HH/LL/HL/LH quadrant assignments and FDR correction.
moran_bv
    Global bivariate Moran's I for pairs of keys (co-expression / co-localisation).
moran_local_bv
    Local bivariate Moran's I (LISA) for pairs of keys.
getis_ord
    Local Getis-Ord G* hot/cold spot detection.
spatial_lag
    Compute spatial lag (weighted neighbor average) of expression or obs variables.
spatial_silhouette
    Boundary silhouette score for spatial cluster quality.
ripley_l
    Ripley's L function for cell-type point-pattern clustering analysis.
spatially_variable_genes
    Identify genes with non-random spatial expression patterns (Moran or Geary).
identify_micronuclei
    Flag small nucleus segments as micronuclei by proximity to a larger primary.
merge_micronuclei
    Sum expression and union boundaries of micronuclei into their primary cells.
interaction_changed_genes
    Genes differentially expressed in cell type A when neighboring cell type B.
spatial_gene_modules
    Cluster genes into spatial co-expression modules.
banksy_sweep
    Run the BANKSY pipeline across a grid of λ values and Leiden resolutions.
composition
    Compute cell-type fraction per spatial domain; returns (regions × cell_types) DataFrame.
som
    Train a SOM on any obsm embedding or X; assigns cells to BMU nodes for topology-aware clustering.
som_gene_modules
    Cluster genes into co-expression modules by training a SOM on the gene-by-cell matrix.
som_subcellular_domains
    Discover recurring subcellular spatial patterns by training a SOM on intracellular transcript positions.
expression_gradient
    Per-cell Moving Least Squares gradient of a gene or obs variable over the spatial graph.
gradient_field
    Interpolate per-cell gradients onto a regular grid; compute divergence and curl.
gradient_significance
    Permutation test for gradient magnitude and directional coherence.
compartment_flux
    Mean projected gradient flux across cell-type label boundaries in the spatial graph.
spatial_diffusion
    Graph-Laplacian (Coifman–Lafon) diffusion smoothing and optional spectral decomposition.
decay_length
    Phenomenological radial decay-length fit f(r) = f₀·exp(−r/λ) from a source point.
"""

from astrolabe.tl._graph import spatial_neighbors, transcriptional_neighbors, blend_neighbors
from astrolabe.tl._cluster import cluster
from astrolabe.tl._embedding import pca, umap
from astrolabe.tl._trajectory import paga, dpt
from astrolabe.tl._de import rank_genes_groups, score_genes, annotate_clusters
from astrolabe.tl._spatial_stats import (
    moran,
    moran_local,
    moran_bv,
    moran_local_bv,
    getis_ord,
    spatial_lag,
    spatial_silhouette,
    ripley_l,
    spatially_variable_genes,
    geary_c,
    colocation_quotient,
)
from astrolabe.tl._micronuclei import identify_micronuclei, merge_micronuclei
from astrolabe.tl._icg import interaction_changed_genes
from astrolabe.tl._gene_modules import spatial_gene_modules
from astrolabe.tl._banksy_sweep import banksy_sweep
from astrolabe.tl._som import som, som_gene_modules, som_subcellular_domains
from astrolabe.tl._neighborhood import (
    nhood_enrichment,
    co_occurrence,
    centrality_scores,
    expression_by_distance,
    ligrec,
    composition,
)
from astrolabe.tl._gradient import (
    expression_gradient,
    gradient_field,
    gradient_significance,
    compartment_flux,
    spatial_diffusion,
    decay_length,
)

__all__ = [
    "spatial_neighbors",
    "transcriptional_neighbors",
    "blend_neighbors",
    "cluster",
    "pca",
    "umap",
    "paga",
    "dpt",
    "rank_genes_groups",
    "score_genes",
    "annotate_clusters",
    "moran",
    "moran_local",
    "moran_bv",
    "moran_local_bv",
    "getis_ord",
    "spatial_lag",
    "spatial_silhouette",
    "ripley_l",
    "spatially_variable_genes",
    "geary_c",
    "colocation_quotient",
    "identify_micronuclei",
    "merge_micronuclei",
    "nhood_enrichment",
    "co_occurrence",
    "centrality_scores",
    "expression_by_distance",
    "ligrec",
    "composition",
    "interaction_changed_genes",
    "spatial_gene_modules",
    "banksy_sweep",
    "som",
    "som_gene_modules",
    "som_subcellular_domains",
    "expression_gradient",
    "gradient_field",
    "gradient_significance",
    "compartment_flux",
    "spatial_diffusion",
    "decay_length",
]