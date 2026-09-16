# Changelog

## Unreleased

- Add optional FlexiChains, MCMCChains, and InferenceObjects adapters for
  per-parameter diagnostics, preserving chain axes and each variable's precision.
- Add modern and basic R-hat, including exact compressed repetition counts.
- Add bulk, tail, raw-mean, quantile, and interval ESS, mean MCSE,
  autocorrelation, and integrated autocorrelation time.
- Add per-parameter StructArray reports with per-metric status.
- Add configurable concurrent online means, variances, and basic R-hat.
- Add compressed precision, batch/overlapping MCSE, quantile and variance MCSE,
  Sokal windows, diagnostic curves and optional FFT autocovariances.
- Add online dyadic scalar/vector precision, fixed indicators, bounded lags,
  sampler metadata, importance summaries, WAIC and jump distance.
- Add classical single-chain, multivariate, nested and categorical diagnostics,
  rank summaries, sampler metadata, movement and tempering summaries.
- Add importance MCSE, Pareto tails, PSIS, WAIC, PSIS-LOO, paired ELPD and LOO-PIT.
- Document sampler adapter contracts and test StructArray/ArraysOfArrays views.
- Preserve Float32 and Float64 arithmetic through batch and online methods.
- Add independent numerical tests, reference checks, benchmarks, and documentation.
- Create CI and Documenter setup. License the package under MIT.
