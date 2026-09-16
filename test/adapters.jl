@testset "Sampler record boundary" begin
    for T in (Float32, Float64)
        positions = [T[0 1 3 2 4 -1 5 2; 2 0 4 3 -1 5 1 6],
            T[2 0 4 3 -1 5 1 6; 0 1 3 2 4 -1 5 2]]
        counts = [[2,1,3,2,2,1,3,2], [1,2,2,3,1,2,2,3]]
        records = [StructArray((v=ArraysOfArrays.VectorOfSimilarVectors(x), weight=T.(k)))
            for (x,k) in zip(positions,counts)]
        saved_positions, saved_weights = deepcopy(positions), [copy(r.weight) for r in records]
        # BAT validates its floating weights before this exact conversion.
        repetitions = [Int.(r.weight) for r in records]
        views = [ArraysOfArrays.flatview(r.v) for r in records]
        expanded = cat((reshape(permutedims(x[:,vcat([fill(i,k) for (i,k) in enumerate(w)]...)]),
            sum(w),1,2) for (x,w) in zip(views,repetitions))...; dims=2)
        compressed, dense = diagnostics(views;drawdim=2,counts=repetitions), diagnostics(expanded)
        @test compressed.rhat ≈ dense.rhat
        @test compressed.ess_bulk ≈ dense.ess_bulk
        @test isapprox(compressed.ess_tail,dense.ess_tail;nans=true)
        @test compressed.status == dense.status
        @test compressed.mcse_mean ≈ dense.mcse_mean

        metrics = (:mean,:variance,:rhat_basic,:mcse_mean)
        state = OnlineDiagnostics(T;nchains=2,parameter_shape=(2,),metrics)
        for range in (1:3,4:8), c in 1:2
            append!(state,view(views[c],:,range);chain=c,drawdim=2,counts=view(repetitions[c],range))
        end
        direct = OnlineDiagnostics(T;nchains=2,parameter_shape=(2,),metrics)
        append!(direct,expanded)
        saved, expected = diagnostics(state), diagnostics(direct)
        @test saved.mean ≈ expected.mean
        @test saved.variance ≈ expected.variance
        @test saved.rhat_basic ≈ expected.rhat_basic
        @test mcse(state) ≈ mcse(direct)
        empty!(state)
        append!(state,expanded .+ T(100))
        @test diagnostics(state).mean ≈ [mu .+ T(100) for mu in saved.mean]
        @test saved.mean ≈ expected.mean
        @test positions == saved_positions && [r.weight for r in records] == saved_weights
    end
end
