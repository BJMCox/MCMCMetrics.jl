# Array inputs and batch results

```@meta
DocTestSetup = :(using MCMCMetrics)
```

## Arrays and results

| Input | Meaning | Per-parameter result |
|:--|:--|:--|
| Vector | One scalar chain | Scalar |
| Matrix | Draws × independent chains | Scalar |
| Tensor | Draws × chains × parameter axes | Array with the parameter shape |
| Vector of matrices | One stored-draw × parameter matrix per chain | Parameter vector |

Use `drawdim` and `chaindim` to identify other layouts. Use `chaindim=nothing`
for one chain with parameter axes. All arrays must use one-based indexing.
Remove warmup explicitly, for example with a view.

```jldoctest
julia> using MCMCMetrics

julia> x = reshape(sin.(Float64.(1:256)), 32, 4, 2);

julia> size(rhat(x)), size(diagnostics(x))
((2,), (2,))

julia> y = permutedims(x, (3, 1, 2));

julia> rhat(y; drawdim=2, chaindim=3) ≈ rhat(x)
true

julia> eltype(rhat(Float32.(x)))
Float32
```

`diagnostics(x)` returns a StructArray with `index`, `rhat`, `ess_bulk`,
`ess_tail`, `mcse_mean`, `n_draws`, `n_chains`, and `status` columns.
`index` is a CartesianIndex into the original parameter shape.
`n_draws` counts all logical draws before splitting. Status is separate for
each metric, so an unavailable tail estimate does not hide a valid mean estimate.

Float32 and Float64 inputs retain their arithmetic and output type. Integer
observations use their Julia floating counterpart. Ranking preserves integer
order, and centering subtracts integer offsets before conversion. Other number
types need a separate numerical contract and are currently rejected.

## Estimators

| Call | Estimand or definition |
|:--|:--|
| `rhat(x)` | Maximum of rank-normalized split and folded split R-hat |
| `rhat(x; kind=:basic, split=false)` | Classical unsplit variance ratio |
| `ess(x)` | Bulk ESS from rank-normalized split draws |
| `ess(x; kind=:tail)` | Minimum ESS for the 5% and 95% quantile indicators |
| `ess(x; kind=:mean)` | Raw-draw ESS |
| `ess(x; kind=:quantile, prob=0.9)` | Indicator below the pooled empirical quantile |
| `ess(x; kind=:interval, interval=(0, 1))` | Indicator of the closed interval |
| `mcse(x)` | Raw pooled standard deviation divided by the square root of mean ESS |
| `autocor(x; lags=0:10)` | Autocorrelation separately within each chain |
| `iact(x)` | Integrated autocorrelation time separately within each unsplit chain |

Use `estimator=:sokal` for an explicit self-consistent spectral window.
Mean MCSE also supports `:batchmeans` and `:overlapping` estimators with an
explicit `batch_size`. Variance, standard deviation, quantile and median MCSE
use their own influence or indicator processes. See [Diagnostic families](@ref).

Splitting drops the center draw of odd-length chains before ranks, folding, or
quantile thresholds are computed. Mean MCSE uses all draws for the pooled
standard deviation. R-hat needs two independent chains. The package never treats
two halves of one chain as independent source chains for R-hat.

ESS uses direct autocovariance, divided by chain length at every lag. Geyer's
initial positive sequence discards pairs at the first nonpositive sum. The
initial monotone sequence reduces each retained pair to the preceding minimum.
This requires reversibility and finite variance of the diagnosed process.
Valid antithetic ESS can exceed the number of draws.

This implementation uses the plain paired estimator. It omits the extra terminal
lag and finite-sample ESS cap used by posterior and MCMCDiagnosticTools.
The tests use independent exact arithmetic as well as pinned reference checks.

Direct autocovariance can require quadratic work when many lags remain positive.
For dense inputs, load FFTW and select `autocov=:fft` explicitly to use the
optional extension. Loading FFTW does not change the default estimator.

## Repeated states

Pass exact integer repetition counts for compressed MCMC states. Counts describe
consecutive repeated observations, including rejected proposals. They do not
represent importance weights.

For array input, counts have shape stored draws × chains. A single-chain array
uses a count vector. Nested chain matrices use one count vector per chain.
Direct R-hat requires matching logical lengths, although the number of stored
runs may differ. The aggregate marks unequal-length combined estimates unavailable
with `:unequal_chain_lengths`. Counts must fit Int, and their sums are checked.

```jldoctest
julia> runs = [0.0 0.2; 1.0 1.2; 2.0 2.2; 3.0 3.2];

julia> counts = fill(3, 4, 2);

julia> rhat(runs; counts) ≈ rhat(repeat(runs; inner=(3, 1)))
true

julia> report = only(diagnostics(runs; counts));

julia> report.n_draws, report.status.ess_bulk
(24, :ok)
```

R-hat, ESS, MCSE, autocorrelation and IACT support exact repetitions without
expansion. Direct compressed spectral work has an explicit `max_lag_work`
budget. Exhausting that budget throws from a direct call; `diagnostics` reports
`:work_limit` for that metric and retains other available results.
Batch estimators also bound window and run-overlap work. FFT, PSIS, estimand-tail
fits and classical single-chain tests require uncompressed inputs.

## Invalid and unavailable results

Malformed axes, counts, shapes, unsupported methods, and positive-count nonfinite
observations throw. Zero-count observations are skipped, including nonfinite values.
Empty draw or parameter axes are rejected.

Statistically undefined estimates return a typed NaN. Aggregate status identifies
short chains, constant chains, unequal lengths, exhausted work budgets,
or estimator failure. A constant original, split, or transformed chain makes the
corresponding estimate unavailable. Finite output alone does not establish convergence.
The caller owns convergence thresholds and stopping rules.

## References

- Vehtari et al. (2021), [Rank-normalization, folding, and localization](https://doi.org/10.1214/20-BA1221).
- Geyer (1992), [Practical Markov Chain Monte Carlo](https://doi.org/10.1214/ss/1177011137).
- [Stan reference manual: analysis](https://mc-stan.org/docs/reference-manual/analysis.html).
