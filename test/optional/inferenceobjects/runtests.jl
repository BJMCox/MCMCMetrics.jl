using MCMCMetrics, Test
using Random: MersenneTwister, randn
import InferenceObjects, DimensionalData

@testset "MCMCMetrics with InferenceObjects" begin
    rng = MersenneTwister(734)
    x = randn(rng, Float32, 64, 4) .+ reshape(Float32[-5, 0, 7, 12], 1, :)
    beta = randn(rng, 64, 4, 2, 2)
    chain_ids, iter_ids = [41, 7, 9, 2], 11:2:137
    DD = DimensionalData
    # Physical axis order differs from both the default and between variables.
    xd = DD.DimArray(permutedims(x),
        (DD.Dim{:chain}(chain_ids), DD.Dim{:draw}(iter_ids)); name=:x)
    bd = DD.DimArray(permutedims(beta, (3, 1, 4, 2)),
        (DD.Dim{:row}([:a, :b]), DD.Dim{:draw}(iter_ids),
            DD.Dim{:column}([:c, :d]), DD.Dim{:chain}(chain_ids)); name=:beta)
    posterior = InferenceObjects.Dataset(xd, bd)
    ignored = InferenceObjects.namedtuple_to_dataset((x=fill(NaN, 64, 4),))
    data = InferenceObjects.InferenceData(; posterior,
        warmup_posterior=ignored, sample_stats=ignored)
    for (f, kwargs) in ((rhat, (;)), (diagnostics, (;)), (autocor, (; lags=0:3)))
        @test isequal(f(data; kwargs...),
            Dict(:x => f(x; kwargs...), :beta => f(beta; kwargs...)))
    end
    @test rhat(data)[:x] === rhat(x)
    @test rhat(posterior; parameters=(:beta,)) == Dict(:beta => rhat(beta))
end
