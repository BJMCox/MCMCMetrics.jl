using MCMCMetrics, Test
using Random: MersenneTwister, randn
import FlexiChains

@testset "MCMCMetrics with FlexiChains" begin
    rng = MersenneTwister(734)
    x = randn(rng, Float32, 64, 4) .+ reshape(Float32[-5, 0, 7, 12], 1, :)
    beta = randn(rng, 64, 4, 2, 2)
    P, E = FlexiChains.Parameter, FlexiChains.Extra
    records = [Dict(P(:x) => x[d,c], P(:beta) => beta[d,c,:,:], E(:x) => NaN)
        for d in 1:64, c in 1:4]
    chain = FlexiChains.FlexiChain{Symbol}(64, 4, records;
        iter_indices=11:2:137, chain_indices=[41, 7, 9, 2])
    for (f, kwargs) in ((rhat, (;)), (diagnostics, (;)), (autocor, (; lags=0:3)))
        @test isequal(f(chain; kwargs...),
            Dict(:x => f(x; kwargs...), :beta => f(beta; kwargs...)))
    end
    @test rhat(chain)[:x] === rhat(x)
    @test rhat(chain; parameters=(:beta,)) == Dict(:beta => rhat(beta))
    counts = repeat([mod1(d, 3) for d in 1:64], 1, 4)
    @test rhat(chain; kind=:basic, split=false, counts)[:x] ===
        rhat(x; kind=:basic, split=false, counts)

    single = FlexiChains.FlexiChain{Symbol}(64, 1, records[:,2:2])
    @test isequal(diagnostics(single)[:x], diagnostics(x[:,2:2]))
    key = FlexiChains.@varname(theta)
    named = FlexiChains.VNChain(64, 4, [Dict(P(key) => x[d,c]) for d in 1:64, c in 1:4])
    @test rhat(named)[key] === rhat(x)
end
