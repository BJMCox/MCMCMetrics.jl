module MCMCMetrics

import ArraysOfArrays, LinearAlgebra, LogExpFunctions, SpecialFunctions, Statistics, StructArrays
using StructArrays: StructArray

export rhat, ess, mcse, autocor, iact, diagnostics, OnlineDiagnostics
export diagnostic_curve, cost_normalize
export geweke, heidelberger_welch, raftery_lewis
export bfmi, sampler_diagnostics, rank_histogram, rank_ecdf
export importance_diagnostics, importance_mcse, psis, pareto_tail, pareto_smoothed_minimum
export waic, loo, compare_elpd, loo_pit
export rhat_nested, categorical_summary, categorical_weiss
export squared_jump_distance, tempering_summary
export mc_cov, ess_multivariate, rhat_multivariate

include("input.jl")
include("rhat.jl")
include("ess.jl")
include("repetitions.jl")
include("precision.jl")
include("diagnostics.jl")
include("online.jl")
include("single_chain.jl")
include("sampler.jl")
include("ranks.jl")
include("importance.jl")
include("pareto.jl")
include("predictive.jl")
include("grouped.jl")
include("categorical.jl")
include("movement.jl")
include("multivariate.jl")
include("chain_adapters.jl")

end
