# MCMCMetrics.jl

```@meta
DocTestSetup = :(using MCMCMetrics)
```

Batch and online MCMC diagnostics for numeric arrays. Use retained sampler draws
directly, with explicit draw and chain axes. Julia 1.10 or later is required.

## Install

The package is not yet registered in General. Install it from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/MCMCMetrics.jl")
```

## Check a batch

The default layout is **draws × independent chains × parameter dimensions**.
Use `drawdim` and `chaindim` for other layouts. Replace this small demonstration
array with retained samples from your sampler.

```jldoctest
julia> x = reshape(sin.(Float64.(1:256)), 32, 4, 2);

julia> report = diagnostics(x);

julia> length(report), report.n_draws
(2, [128, 128])

julia> report.rhat ≈ rhat(x)
true
```

The report has one row per parameter and separate values/statuses for R-hat,
bulk ESS, tail ESS and mean MCSE. Read a metric's status before using its value.
These diagnostics report sampling problems and precision; they do not prove
that a chain has explored every important region.

## Update an online state

Choose the type, chain count, parameter shape and metrics once per sampling cycle.
Append new draws in chronological order, then request an owned snapshot.

```jldoctest
julia> state = OnlineDiagnostics(Float32; nchains=2,
                                metrics=(:mean, :variance, :mcse_mean));

julia> append!(state, Float32[1 2; 3 4; 5 6; 7 8]);

julia> only(diagnostics(state)).mean == Float32[4, 5]
true
```

Online mean precision uses dyadic batch means. Keep batch checkpoints when your
policy needs rank/folded R-hat or bulk/tail ESS. The sampler owns warmup selection,
chain identity, checkpoint timing and stopping thresholds.

## Guides

- [Array inputs and batch results](@ref): layouts, repetitions, precision and unavailable results.
- [Diagnostic families](@ref): available methods and their statistical assumptions.
- [Online diagnostics](@ref): configuration, metadata, storage and concurrent updates.
- [Sampler adapters](@ref): StructArrays/ArraysOfArrays storage and sampler ownership.
- [API reference](@ref): signatures, options and estimator definitions.
