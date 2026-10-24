# Conditional spatial inference: backend decision

## Decision and constraints

The user requires a **Python-only backend and established statistical methods**.
The custom PyMC proposal is withdrawn. The user approved the published BYM2/INLA
backend through optional pyINLA after reviewing its model, licensing, installation
and Bayesian outputs. `tl.conditional_expression` implements that approved choice;
no PyMC or R dependency is added. See the [backend specification and installation
guide](g5-inla-proposal.md) for its bounded scope and validation limitations.

The proposed model combined negative-binomial regression, independent spatial
Gaussian processes per core, and shared donor random effects. These are established
components, but the exact model, priors, and inferential procedure were assembled
for Astrolabe without identifying a published method validating that combination
for this application. PyMC supplies modeling machinery; using it does not establish
the statistical validity of a particular model.

## Evidence reviewed

| Candidate | Evidence and limitation | Decision |
| --- | --- | --- |
| BYM2/INLA through pyINLA | Published graph model and inference engine; supports scaled disconnected components and shared donor IID effects. Python wrapper is proprietary and currently constrained to Python 3.12 for Astrolabe. | Approved optional integration; explicit prior-serialization identity checks and external-environment support. |
| Custom PyMC model | Generic latent-GP support permits constructing the proposed model, but does not validate the complete analysis procedure. | Withdrawn; do not implement. |
| GPBoost | Provides Python count likelihoods and spatial/grouped random effects. Its documented `cluster_ids` separates both random effects and GP realizations. Setting it to core therefore does not directly express a donor effect shared across cores. | Not verified for the full G5 design; no custom workaround selected. |
| GPcounts | Published Python count-GP method with spatially variable gene and trajectory applications. Its documented spatial workflow does not establish support for the requested conditional contrasts and shared-donor/core hierarchy. | Not a verified replacement for G5. |
| smiDE | Published method addressing spatially correlated differential expression; supplied as an R package with a non-commercial code license. | Does not meet the Python-only requirement. |
| Ospina et al. spatial mixed models (2024) | Published spatial expression contrasts use Gaussian models with exponential spatial covariance, fitted with the R package spaMM. | Relevant established comparator, but the authors' implementation does not meet the Python-only requirement or verify the full requested count/donor/core design. |

Sources: [PyMC latent GP](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Latent.html),
[GPBoost model API](https://gpboost.readthedocs.io/en/latest/pythonapi/gpboost.GPModel.html),
[GPcounts authors' implementation and examples](https://github.com/ManchesterBioinference/GPcounts),
[smiDE paper and code availability](https://link.springer.com/article/10.1186/s13059-025-03867-1).
The additional spatial mixed-model comparator is documented in
[Ospina et al., methods and code availability](https://www.nature.com/articles/s41598-024-61758-0).

## Requirements for a replacement proposal

- Identify the published method, maintained implementation, exact estimand,
  assumptions, supported replication structure, and inferential outputs.
- Preserve the method's specified likelihood, covariance, inference, and testing
  procedure. Do not invent additional components to fill unsupported designs.
- Verify disconnected TMA units and donor replication explicitly. Reject
  unsupported designs; do not substitute cells or cores for independent donors.
- Compare an adapter against the authors' implementation and examples, then run
  the approved calibration checks. Simulations check implementation and applicability;
  they do not substitute for an established method.
- Present any narrower supported scope for review before adapter implementation.
  Existing pseudobulk remains limited to its supported sample-level questions.

## Exploratory check retained for audit

The isolated PyMC experiment fitted 48 simulated cells with four cores and two
donors. A planted coefficient of 0.5 yielded a posterior mean of 0.5444 and no
sampling divergences. This was only a computation smoke test, not evidence of
calibration, method validation, or an adequate donor count. It is not a basis for
shipping inference. Artifacts remain under
`analysis/spatial_gaps_2026_10_23/pymc_candidate.*` outside the package API.
