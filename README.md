# MCMCMetrics.jl

[![CI](https://github.com/BJMCox/MCMCMetrics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BJMCox/MCMCMetrics.jl/actions/workflows/CI.yml)
[![Coverage](https://codecov.io/gh/BJMCox/MCMCMetrics.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/BJMCox/MCMCMetrics.jl)
[![Docs](https://img.shields.io/badge/docs-main-blue.svg)](https://bjmcox.github.io/MCMCMetrics.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

Batch and online MCMC diagnostics for Julia numeric arrays. Supports R-hat,
ESS, MCSE, sampler metadata, multivariate checks, PSIS and predictive scores.
Julia 1.10 or later.

## Install

```julia
using Pkg
Pkg.add(url="https://github.com/BJMCox/MCMCMetrics.jl")
```

The package is not yet registered in General. Installation uses the repository.

## Batch diagnostics

```julia
using MCMCMetrics, Random

# Replace these illustrative draws with retained samples from your sampler.
x = randn(Xoshiro(42), Float32, 1_000, 4, 3)  # draws × chains × parameters
rhat(x)                                      # rank/folded split R-hat
ess(x)                                       # bulk effective sample size
ess(x; kind=:tail)
mcse(x)                                      # standard error of the mean
report = diagnostics(x)                      # one StructArray row per parameter
```

Use `drawdim` and `chaindim` for other layouts. Float32 and Float64 inputs keep
their precision. Integer repetition counts support compressed states without
expansion. FFTW provides optional, explicitly selected autocovariances.

Load FlexiChains, MCMCChains, or InferenceObjects to enable optional chain
adapters. Calls such as `MCMCMetrics.diagnostics(chain)` return results keyed by
parameter, preserving component shapes, numeric precision, and metric statuses.
See the [adapter guide](https://bjmcox.github.io/MCMCMetrics.jl/adapters/).

## Online diagnostics

```julia
state = OnlineDiagnostics(Float32; nchains=4, parameter_shape=(3,),
                          metrics=(:mean, :variance, :rhat_basic, :mcse_mean))
append!(state, x)
snapshot = diagnostics(state)                # coherent, owned results
empty!(state)                                # start a new sampling cycle
```

Select metrics when creating the state. Online mean precision uses dyadic batch
means; modern rank/folded R-hat remains a batch calculation. Independent chains
can update concurrently. Preserve draw order within each chain and finish old
producers before resetting the state.

The runtime dependencies are StructArrays, ArraysOfArrays, Statistics,
LinearAlgebra, SpecialFunctions and LogExpFunctions. Samplers own warmup,
chain identity and stopping rules.

[Documentation](https://bjmcox.github.io/MCMCMetrics.jl/) ·
[Sampler adapters](https://bjmcox.github.io/MCMCMetrics.jl/adapters/) ·
[Development](CONTRIBUTING.md) · [MIT license](LICENSE.md)
