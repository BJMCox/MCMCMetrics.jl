# API reference

## Batch diagnostics

Optional chain adapters support R-hat, ESS, MCSE, autocorrelation, IACT, aggregate
reports, and diagnostic curves. They return dictionaries keyed by parameter.
See [Sampler adapters](@ref) for examples and selection rules.

```@docs
rhat
ess
mcse
autocor
iact
diagnostics
diagnostic_curve
cost_normalize
```

## Single-chain checks and ranks

```@docs
geweke
heidelberger_welch
raftery_lewis
rank_histogram
rank_ecdf
```

## Sampler metadata and movement

```@docs
bfmi
sampler_diagnostics
squared_jump_distance
tempering_summary
```

## Vector and structured chains

```@docs
mc_cov
ess_multivariate
rhat_multivariate
rhat_nested
categorical_summary
categorical_weiss
```

## Importance weights and tails

```@docs
importance_diagnostics
importance_mcse
psis
pareto_tail
pareto_smoothed_minimum
```

## Predictive scores

```@docs
waic
loo
compare_elpd
loo_pit
```

## Online state

```@docs
OnlineDiagnostics
Base.push!(::OnlineDiagnostics, ::Any)
Base.append!(::OnlineDiagnostics, ::Any)
Base.empty!(::OnlineDiagnostics)
Base.merge!(::OnlineDiagnostics, ::OnlineDiagnostics)
```
