# INLA undercoverage investigation

The completed engineering checks do not resolve the 85–92% coverage observed
for nominal 95% intervals. This investigation keeps the published BYM2 model,
negative-binomial likelihood and disconnected-core/shared-donor structure.

## Prespecified diagnostic comparisons

Fresh seeds 7201–7220 cover unequal-core nulls, donor-condition nulls and planted
donor effects, cycling through 4/6/8/12 donors. Each dataset is fit with five
settings: the existing default, donor `pc.prec(1, .01)`, known donor precision 16,
all known variance/dispersion parameters, and full rather than simplified Laplace.
Oracle settings diagnose causes and cannot be used for real-data inference.
All settings and failures are retained; no interval multiplier or new estimator
is introduced. Native model files must confirm the requested settings.

The inherited donor log-gamma prior implies precision Gamma(shape=1, rate=5e-5).
Its probability of donor SD above the simulation value 0.25 is about 0.0008.
This is a concrete prior/data mismatch to investigate, not proof of the cause.
The PC comparator uses the same scale as the already specified spatial SD prior:
`P(SD > 1) = .01` on the log-expression scale, chosen before diagnostic fits.

The prior is the established [INLA PC precision prior](https://www.inla.r-inla-download.org/r-inla.org/doc/prior/pc.prec.pdf).
Full Laplace is an existing [INLA approximation setting](https://github.com/hrue/r-inla/blob/master/rinla/R/set.default.arguments.R).
The [negative-binomial reference](https://www.inla.r-inla-download.org/r-inla.org/doc/likelihood/nbinomial.pdf)
defines variance and the log-size parameter used by the oracle arm.

Any proposed implementation change must subsequently be checked on fresh,
prespecified seeds and retain explicit limitations for small donor cohorts.
Conditional frequentist coverage of fixed simulation parameters is a diagnostic;
Bayesian credible intervals do not automatically guarantee it for every design.

## Held-out comparison declared before diagnostic completion

Seeds 8201–8300 will compare the baseline and the single PC-donor candidate on
unequal-core nulls, donor-condition nulls, planted donor effects and adjusted
regional-confounding nulls (800 fits). Donor counts cycle through 4/6/8/12.
An independent seeded draw varies donor SD over 0/0.1/0.25/0.5, spatial SD over
0.15/0.3/0.6, phi over 0.2/0.5/0.8, and NB size over 5/10/30. The same dataset is
used for both fits. This avoids evaluating only the diagnostic variance values.
Report all failures, coverage uncertainty, donor strata, power and paired changes.
No further prior setting is selected using these held-out outcomes.

## Diagnostic findings and implementation

Of the original 100 donor-condition null fits, 30 had posterior median donor
precision above 1,000; 14 of the 15 false exclusions occurred in those 30 fits.
The simulation's true donor precision was 16. This association motivates the
controlled comparison; it does not itself identify causality.

The 300 fresh diagnostic fits complete with one retained full-Laplace backend
retry failure. Unequal-core coverage is 20/20 in every arm. Donor-condition null
coverage is 18/20 for the inherited prior and both oracle arms, 20/20 for PC,
and 17/19 for full Laplace. Planted-condition coverage is 18/20 for inherited
and full Laplace, and 20/20 for PC and both oracle arms. The PC arm detects 17/20
planted effects; inherited/full Laplace detect 19/20. Full Laplace changes the
approximation without resolving the prior's behavior. Twenty replicates are
insufficient to declare calibration; the separate held-out comparison is required.

The implementation now specifies the established PC donor prior explicitly and
exposes its `(u, alpha)` through `donor_sd_prior`. It verifies the actual native
donor prior and records the chosen values. The default uses the same prior scale
as the spatial field, not an optimized value from a parameter search. The model
remains experimental pending the held-out results and their design limitations.

All diagnostic outcomes: [null](inla-donor-diagnostic-null-results.json) and
[unequal-core/planted](inla-donor-diagnostic-other-results.json).
