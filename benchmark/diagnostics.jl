using MCMCMetrics, BenchmarkTools, Random

function chain_updates!(state, x)
    @sync for chain in axes(x, 2)
        Threads.@spawn append!(state, view(x, :, chain, :); chain)
    end
    return state
end

function separate_metrics(x)
    (rhat(x), ess(x), ess(x; kind=:tail), mcse(x))
end

function benchmarks(; n=2048, chains=4, parameters=8)
    rng = Xoshiro(812)
    x = randn(rng, n, chains, parameters)
    parameter_first = permutedims(x, (3, 1, 2))
    runs = randn(rng, 32, chains, parameters)
    counts = fill(1_000_000, 32, chains)
    state = OnlineDiagnostics(Float64; nchains=chains, parameter_shape=(parameters,))
    draw = copy(view(x, 1, 1, :))
    seed = view(x, 1:min(n, 128), :, :)
    suite = BenchmarkGroup()
    suite["rhat"] = @benchmarkable rhat($x)
    suite["ess"] = @benchmarkable ess($x)
    suite["separate"] = @benchmarkable separate_metrics($x)
    suite["aggregate"] = @benchmarkable diagnostics($x)
    suite["strided"] = @benchmarkable diagnostics($parameter_first; drawdim=2, chaindim=3)
    suite["pack_and_compute"] = @benchmarkable diagnostics(permutedims($parameter_first, (2, 3, 1)))
    suite["reuse_packed_3"] = @benchmarkable foreach(_ -> diagnostics($x), 1:3)
    suite["reuse_strided_3"] = @benchmarkable foreach(_ -> diagnostics($parameter_first; drawdim=2, chaindim=3), 1:3)
    suite["compressed"] = @benchmarkable rhat($runs; counts=$counts)
    suite["push"] = @benchmarkable push!($state, $draw; chain=1) setup=(empty!($state); append!($state, $seed)) evals=1
    suite["chunk"] = @benchmarkable append!($state, $x) setup=(empty!($state)) evals=1
    suite["concurrent_chunks"] = @benchmarkable chain_updates!($state, $x) setup=(empty!($state)) evals=1
    return suite
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = run(benchmarks(); seconds=1)
    display(median(results))
end
