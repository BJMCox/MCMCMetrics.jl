# Online diagnostics

```@meta
DocTestSetup = :(using MCMCMetrics)
```

Choose metrics when creating the state. The defaults are `:mean`,
`:variance`, and `:rhat_basic`. They share private moments stored in a StructArray.
ArraysOfArrays supplies nested views of owned snapshot storage.

```jldoctest
julia> state = OnlineDiagnostics(Float32; nchains=2,
                                metrics=(:mean, :variance, :rhat_basic));

julia> append!(state, Float32[1 2; 3 4; 5 6; 7 8]);

julia> report = only(diagnostics(state));

julia> report.mean == Float32[4, 5], report.n_per_chain == [4, 4]
(true, true)

julia> report.rhat_basic ≈ rhat(Float32[1 2; 3 4; 5 6; 7 8]; kind=:basic, split=false)
true

julia> empty!(state);

julia> only(diagnostics(state)).n_draws
0
```

Use `parameter_shape=(p,)` for p parameters. `push!(state, draw; chain=j)`
accepts one draw. `append!(state, chunk; chain=j)` accepts draws × parameter
axes for one chain. Without `chain`, `append!` accepts every state chain using
the batch axis convention. Both methods support repetition counts.

Inputs must match the configured floating type. Convert integer observations
explicitly before an update. Input arrays must remain unchanged during the call.
The state validates a whole chunk before mutation and checks count additions
under locks. An invalid chunk contributes no observations.

One lock protects each chain. Different chains can update concurrently.
Callers preserve temporal order within each chain. Snapshots and resets acquire
all chain locks in ascending order. Snapshot results own their data. Finish all
producers from an old sampling cycle before calling `empty!`.

Online reports contain the selected metric columns. Mean and variance values
and statuses are per-chain vectors. Basic R-hat is one value per parameter.
`n_per_chain` holds the counts. `n_draws` is their checked Int sum.
Unequal chain lengths leave basic R-hat unavailable while per-chain moments remain
available. Online R-hat requires `kind=:basic, split=false` explicitly.

Select `:mcse_mean`, `:ess_mean` or `:iact_mean` for per-chain online precision.
The named `:dyadic_batchmeans` estimator keeps complete batch summaries at
powers-of-two widths. It uses O(parameters × chains × log(draws)) storage.
Each report states the chosen width, completed batches and discarded remainder.
The mean and MCSE denominator use every draw; the LRV estimate uses complete
batches centered on their own mean. This differs from batch `:batchmeans`,
which reports precision for the retained complete-batch mean.

Online precision needs stationarity and sufficient mixing/moments, as detailed
in its docstring. Finite batches can underestimate long-run variance. A nominal
95% interval is not an exact coverage or repeated-check stopping guarantee.
Online mean ESS is distinct from rank-based bulk ESS. Online rank/folded R-hat,
empirical-quantile precision and PSIS are not inferred from moment summaries.

Select `:indicator` with a fixed `threshold` for event-probability precision.
Select `:autocor` with explicit `max_lag` for a bounded lag buffer. Its globally
centered correlations match the batch definition; a fixed lag cap does not
provide an unrestricted long-run variance estimate.

## Auxiliary online metrics

Configure these through the same state. Read whole-chain families through their
own functions; `diagnostics(state)` keeps one row per scalar parameter.

| Metric choices | Reader and additional configuration |
|:--|:--|
| `:bfmi`, `:divergence`, `:acceptance`, `:tree_depth`, `:leapfrog_steps` | `sampler_diagnostics(state)`; depth requires `max_tree_depth` |
| `:importance` | `importance_diagnostics(state)` pools raw weight concentration |
| `:waic` | `waic(state)`; requires fixed `observation_shape` |
| `:jump_distance` | `squared_jump_distance(state)`; optional fixed SPD `metric` |
| `:mc_cov`, `:ess_multivariate` | `mc_cov(state)`, `ess_multivariate(state)` return per-chain vector precision records |

Supply exactly the configured metadata fields with each update. Floating
metadata uses the state's type. A repetition count repeats the entire supplied
observation, including its metadata.

```jldoctest
julia> state = OnlineDiagnostics(Float32; nchains=1, metrics=(:bfmi, :divergence));

julia> append!(state, Float32[0, 1, 2, 3]; chain=1,
               metadata=(energy=Float32[1, 2, 4, 7], divergent=Bool[0, 0, 1, 0]));

julia> only(sampler_diagnostics(state)).divergences
1
```

Scalar metadata arrays use stored draws × chains, independently of parameter
array axes. An addressed single chain omits the chain axis. Pointwise
`loglikelihood` adds the fixed observation axes. Nested chain inputs require one
metadata array per chain. See [Sampler adapters](@ref).

WAIC pools actual observations across unequal chain lengths. Its standard error
concerns observation units, not posterior Monte Carlo error. Vector precision
uses the same dyadic batch rule as scalar online precision. It returns separate
chain records with owned `mean_cov` matrices, ESS, status and batch metadata.
Its storage costs O(parameters² × chains × log(draws)); request it explicitly.
Other auxiliary storage stays fixed in the draw count.

`merge!(destination, source)` combines disjoint chunks of the same chains.
Both states must have identical type, shape, chain count, and metric choices.
Merging copies the source under locks before locking the destination.
The caller is responsible for disjoint observations and matching chain identity.
Ordered and auxiliary configurations reject `merge!` before changing the state. `empty!`
retains configuration and storage, while clearing all temporal boundaries.
