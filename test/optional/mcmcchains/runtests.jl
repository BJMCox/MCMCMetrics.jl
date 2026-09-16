using MCMCMetrics, Test
using Random: MersenneTwister, randn
import MCMCChains

@testset "MCMCMetrics with MCMCChains" begin
    x = randn(MersenneTwister(734), Float32, 64, 4) .+ reshape(Float32[-5, 0, 7, 12], 1, :)
    values = cat(reshape(x, 64, 1, 4), fill(Float32(NaN), 64, 1, 4); dims=2)
    chain = MCMCChains.Chains(values, [:x, :energy], (parameters=[:x], internals=[:energy]);
        iterations=collect(11:2:137))
    for (f, kwargs) in ((rhat, (;)), (diagnostics, (;)), (autocor, (; lags=0:3)))
        @test isequal(f(chain; kwargs...), Dict(:x => f(x; kwargs...)))
    end
    @test rhat(chain)[:x] === rhat(x)
    reordered = chain[:, :, [4, 2, 1, 3]]
    @test isequal(autocor(reordered; lags=0:3)[:x], autocor(x[:,[4,2,1,3]]; lags=0:3))
    @test isequal(diagnostics(chain[:, :, [2]])[:x], diagnostics(x[:,2:2]))

    stored = Array{Union{Missing,Float32}}(reshape(x, 64, 1, 4))
    @test rhat(MCMCChains.Chains(stored, [:x]))[:x] === rhat(x)
end
