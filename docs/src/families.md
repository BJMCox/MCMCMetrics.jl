# Diagnostic families

Every method consumes numeric arrays. The package owns estimator mathematics.
The caller owns warmup selection, chain identity, sampling design and stopping rules.

## Precision

`ess` and `iact` support Geyer's initial positive/monotone sequence and Sokal's
self-consistent window. Geyer's shape argument requires reversibility. Sokal's
window is a different estimator, with an explicit `window` setting.

`mcse(x; estimand=...)` supports the mean, variance, standard deviation,
quantile and median. Variance/SD use the centered-square influence process.
Quantile uncertainty uses indicator ESS, Beta probability bounds and type-7
empirical inversion. A tied interval that collapses returns unavailable precision.
These quantile methods assume regular continuous quantiles.

Mean MCSE supports nonoverlapping and overlapping batch means. Independent
chain mean variances combine using retained draw counts. A nonpositive LRV
is unavailable, including an alternating process with identical batch means.
No fallback substitutes nominal sample size or clamps ESS to the draw count.

`diagnostic_curve(x; prefixes, diagnostic=ess)` evaluates caller-chosen logical
prefixes. `cost_normalize(effective, cost)` uses a caller-supplied total cost.
The package neither measures time implicitly nor prescribes a stopping threshold.

## Single-chain checks and ranks

| Function | Question and assumptions |
|:--|:--|
| `geweke` | Do separated early/late means agree relative to their own MCSEs? |
| `heidelberger_welch` | Does a retained segment pass a Brownian-bridge stationarity check and relative half-width check? |
| `raftery_lewis` | How long would a fitted binary Markov model need for a chosen quantile-probability tolerance? |
| `rank_histogram`, `rank_ecdf` | How do pooled midranks differ across chains? |

These return evidence, never proof of convergence. Classical tests return
separate chain × parameter records and reject compressed inputs.
Geweke needs at least four draws in each selected window. HW uses the selected
MCSE estimator instead of coda's AR fit and checks truncations through 50%.
Raftery-Lewis diagnoses a selected quantile and does not recommend thinning output.
Rank summaries preserve exact count ordering and supply no IID confidence bands.

## Vector and structured chains

`mc_cov` estimates the covariance of a pooled vector mean. Parameter axes flatten
in Cartesian order. `ess_multivariate` uses log determinants of that covariance
and a within-chain pooled marginal covariance. Both accept batch means,
overlapping batches and optional lugsail settings. Singular estimates return NaN.
The documented Cholesky threshold treats roundoff-sized pivots as unresolved.

`rhat_multivariate` returns the square root of the Brooks–Gelman variance ratio,
the largest scale ratio over linear projections. It does not apply ranks.
`rhat_nested(x, superchain_ids)` requires grouped initialization: chains in a
group share an initial position and use conditionally independent randomness.
Assigning group IDs cannot create that design. One-draw subchains require
`split=false` and multiple subchains per group.

`categorical_summary` reports integer-state occupations and transitions,
including caller-supplied unvisited levels. `categorical_weiss` compares state
frequencies under a shared DAR(1) dependence model. It returns unavailable when
the fitted dependence lies outside that model. Sparse states weaken the
chi-square approximation. General categorical Markov chains need not be DAR(1).

## Sampler metadata and movement

`bfmi` requires full Hamiltonian energy, including kinetic energy.
`sampler_diagnostics` reports supplied divergence flags, acceptance probabilities,
depth saturation and leapfrog work. The configured depth maximum is explicit.
None of these fields can be recovered from a log-posterior column alone.

`squared_jump_distance` measures ordered vector movement with an optional SPD
metric and caller-supplied total cost. Repeated states add zero movement and
still count as transitions. `tempering_summary` consumes replica temperature
paths, reporting occupations and complete cold-hot-cold round trips. Replicas
are not assumed to be independent posterior chains.

## Importance weights and tails

`importance_diagnostics` reports raw weight ESS, largest normalized weight and
entropy. These describe concentration, not temporal MCMC efficiency. Log weights
may be finite or negative infinity; every batch series needs positive mass.
Counts repeat weight observations rather than multiply their importance mass.

`importance_mcse` requires `sampling=:iid` or `:mcmc`. Its MCMC path estimates
precision of the centered weighted ratio influence process while preserving
chain boundaries. A scalar weight ESS cannot replace that process.

`psis(logratios; reff)` smooths importance-ratio tails and returns owned normalized
log weights in the input layout. Supply relative efficiency explicitly.
It uses a focused generalized-Pareto fit, weak shape shrinkage and a ceiling
tail-length convention matching PSIS.jl. Failed/degenerate fits retain raw
normalized weights with an explicit status. Its reliability threshold uses
nominal sample size and can be optimistic under strong dependence.

`pareto_tail` fits tails of an actual estimand. Its k describes possible moment
problems and differs from importance-ratio k. Estimates near 1/2 or 1 warn about
variance or mean assumptions; they do not prove that those moments fail to exist.
`pareto_smoothed_minimum` is a heuristic for Pareto-smoothed expectations,
not a general raw-mean requirement or a stopping guarantee.

## Predictive scores

Supply pointwise log likelihood as draws × chains × observation axes, even
for one observation. A joint log likelihood does not define the factorization.

`waic` returns pointwise lppd, sample log-variance penalty and scores.
`loo` returns PSIS-LOO scores, complexity, Pareto diagnostics and weights.
Without explicit `reff`, LOO estimates efficiency from shifted raw likelihoods,
not their logarithms. It preserves chain identity.

Across-observation `se_elpd` concerns independent observation units. Posterior
`mcse_elpd` measures Monte Carlo uncertainty conditional on those observations.
LOO's total MCSE retains covariance between observation contributions and is
unavailable for unreliable Pareto fits. It excludes smoothing bias.

`compare_elpd` requires matching unique observation IDs in order and uses paired
pointwise differences. `loo_pit` integrates supplied predictive CDF values with
LOO weights. Continuous-outcome calibration differs from randomized discrete PIT.
Neither function infers missing predictive model information.
