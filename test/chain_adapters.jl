import FlexiChains, MCMCChains, InferenceObjects, DimensionalData
using Random: MersenneTwister, randn

@testset "Named chain adapters" begin
    rng = MersenneTwister(734)
    x = randn(rng, Float32, 64, 4) .+ reshape(Float32[-5, 0, 7, 12], 1, :)
    beta = randn(rng, 64, 4, 2, 2)
    chain_ids = [41, 7, 9, 2]
    iter_ids = 11:2:137
    metrics = (
        (rhat, (;)), (diagnostics, (;)), (autocor, (; lags=0:3)),
    )

    @testset "FlexiChains" begin
        P, E = FlexiChains.Parameter, FlexiChains.Extra
        records = [Dict(P(:x) => x[d,c], P(:beta) => beta[d,c,:,:], E(:x) => NaN)
            for d in 1:64, c in 1:4]
        chain = FlexiChains.FlexiChain{Symbol}(64, 4, records;
            iter_indices=iter_ids, chain_indices=chain_ids)
        for (f, kwargs) in metrics
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

    @testset "MCMCChains" begin
        values = cat(reshape(x, 64, 1, 4), fill(Float32(NaN), 64, 1, 4); dims=2)
        chain = MCMCChains.Chains(values, [:x, :energy], (parameters=[:x], internals=[:energy]);
            iterations=collect(iter_ids))
        for (f, kwargs) in metrics
            @test isequal(f(chain; kwargs...), Dict(:x => f(x; kwargs...)))
        end
        @test rhat(chain)[:x] === rhat(x)
        reordered = chain[:, :, [4, 2, 1, 3]]
        @test isequal(autocor(reordered; lags=0:3)[:x], autocor(x[:,[4,2,1,3]]; lags=0:3))
        @test isequal(diagnostics(chain[:, :, [2]])[:x], diagnostics(x[:,2:2]))

        stored = Array{Union{Missing,Float32}}(reshape(x, 64, 1, 4))
        @test rhat(MCMCChains.Chains(stored, [:x]))[:x] === rhat(x)
    end

    @testset "InferenceObjects" begin
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
        for (f, kwargs) in metrics
            @test isequal(f(data; kwargs...),
                Dict(:x => f(x; kwargs...), :beta => f(beta; kwargs...)))
        end
        @test rhat(data)[:x] === rhat(x)
        @test rhat(posterior; parameters=(:beta,)) == Dict(:beta => rhat(beta))
    end
end
