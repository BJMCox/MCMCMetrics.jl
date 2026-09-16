using MCMCMetrics
using Test
using Statistics
using StructArrays: StructArray
import ArraysOfArrays, SpecialFunctions

@testset "MCMCMetrics" begin
    include("input.jl")
    include("rhat.jl")
    include("ess.jl")
    include("precision.jl")
    include("diagnostics.jl")
    include("online.jl")
    include("online_precision.jl")
    include("online_auxiliary.jl")
    include("single_chain.jl")
    include("sampler.jl")
    include("importance.jl")
    include("predictive.jl")
    include("structured.jl")
    include("multivariate.jl")
    include("adapters.jl")
end
