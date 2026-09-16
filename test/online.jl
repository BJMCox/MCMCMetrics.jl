@testset "Online diagnostics" begin
    @testset "Moments and basic R-hat" begin
        for T in (Float32, Float64)
            x = T[1 4; 2 5; 3 6; 4 7]
            values = cat(x, -x; dims=3)
            state = OnlineDiagnostics(T; nchains=2, parameter_shape=(2,))
            append!(state, values)
            report = diagnostics(state)
            @test report.index == [CartesianIndex(1), CartesianIndex(2)]
            @test report.mean[1] == T[5 // 2, 11 // 2]
            @test report.mean[2] == -report.mean[1]
            @test report.variance[1] ≈ fill(T(5 // 3), 2)
            @test report.rhat_basic ≈ fill(T(sqrt(big(69) / 20)), 2)
            @test eltype(report.rhat_basic) == T && eltype(report.mean[1]) == T
            @test report.n_draws == [8, 8]
            @test report.n_per_chain[1] == [4, 4]
            @test all(s -> s.rhat_basic === :ok, report.status)
            @test rhat(state; kind=:basic, split=false) == report.rhat_basic
        end
    end

    @testset "Chunks, repetitions, and axes" begin
        state = OnlineDiagnostics(Float64; nchains=2, parameter_shape=(2,))
        x = reshape(Float64.(1:24), 4, 2, 3)
        target = permutedims(x[:, :, 1:2], (3, 1, 2))
        append!(state, target; drawdim=2, chaindim=3)
        partitioned = OnlineDiagnostics(Float64; nchains=2, parameter_shape=(2,))
        for c in 1:2
            push!(partitioned, target[:, 1, c]; chain=c)
            append!(partitioned, target[:, 2:4, c]; chain=c, drawdim=2)
        end
        @test diagnostics(partitioned).mean ≈ diagnostics(state).mean
        @test diagnostics(partitioned).variance ≈ diagnostics(state).variance
        @test rhat(partitioned; kind=:basic, split=false) ≈ rhat(state; kind=:basic, split=false)

        compressed = OnlineDiagnostics(Float64; nchains=2)
        append!(compressed, [1.0 4.0; 3.0 6.0]; counts=[2 2; 2 2])
        expanded = OnlineDiagnostics(Float64; nchains=2)
        append!(expanded, [1.0 4.0; 1.0 4.0; 3.0 6.0; 3.0 6.0])
        @test diagnostics(compressed).mean == diagnostics(expanded).mean
        @test diagnostics(compressed).variance ≈ diagnostics(expanded).variance
        @test rhat(compressed; kind=:basic, split=false) ≈ rhat(expanded; kind=:basic, split=false)
        push!(compressed, 9.0; chain=1)
        @test diagnostics(compressed).status[1].rhat_basic === :unequal_chain_lengths
    end

    @testset "Atomic rejected updates" begin
        state = OnlineDiagnostics(Float64; nchains=2)
        append!(state, [1.0 3.0; 2.0 4.0])
        before = diagnostics(state)
        @test_throws Exception append!(state, [5.0 7.0; 6.0 NaN])
        @test_throws Exception append!(state, [5.0 7.0; 6.0 8.0]; counts=[1 1; 1 -1])
        @test_throws Exception append!(state, Float32[5 7; 6 8])
        @test_throws Exception push!(state, [5.0]; chain=1)
        @test_throws Exception push!(state, NaN; chain=1)
        after = diagnostics(state)
        @test after.n_per_chain == before.n_per_chain
        @test after.mean == before.mean
        @test after.variance == before.variance

        overflow = OnlineDiagnostics(Float64; nchains=2)
        push!(overflow, 1.0; chain=2, count=typemax(Int))
        @test_throws Exception append!(overflow, [3.0 3.0])
        @test diagnostics(overflow).n_per_chain[1] == [0, typemax(Int)]
        @test isnan(diagnostics(overflow).mean[1][1])
        @test diagnostics(overflow).mean[1][2] == 1.0
        @test diagnostics(overflow).n_draws[1] == typemax(Int)

        zero = OnlineDiagnostics(Float64; nchains=1)
        append!(zero, [NaN, 2.0]; chain=1, counts=[0, 3])
        @test diagnostics(zero).mean[1] == [2.0]
        @test diagnostics(zero).n_per_chain[1] == [3]
    end

    @testset "Snapshot ownership and reset" begin
        state = OnlineDiagnostics(Float64; nchains=2)
        append!(state, [1.0 4.0; 3.0 6.0])
        snapshot = diagnostics(state)
        snapshot.mean[1][1] = 900.0
        snapshot.n_per_chain[1][1] = 900
        @test diagnostics(state).mean[1] == [2.0, 5.0]
        @test diagnostics(state).n_per_chain[1] == [2, 2]
        retained = diagnostics(state)
        empty!(state)
        @test retained.mean[1] == [2.0, 5.0]
        @test diagnostics(state).n_draws[1] == 0
        @test all(isnan, diagnostics(state).mean[1])
        append!(state, [1.0 4.0; 3.0 6.0])
        @test diagnostics(state).variance == retained.variance

        meanonly = OnlineDiagnostics(Float32; nchains=1, metrics=(:mean,))
        append!(meanonly, Float32[1, 2, 3]; chain=1)
        result = diagnostics(meanonly)
        @test result.mean[1] == Float32[2]
        @test propertynames(result) == (:index, :mean, :n_draws, :n_per_chain, :n_chains, :status)
    end

    @testset "Moment merges" begin
        left = OnlineDiagnostics(Float64; nchains=2)
        right = OnlineDiagnostics(Float64; nchains=2)
        append!(left, [1.0 4.0; 2.0 5.0])
        append!(right, [3.0 6.0; 4.0 7.0])
        merge!(left, right)
        @test diagnostics(left).mean[1] == [2.5, 5.5]
        @test diagnostics(left).variance[1] ≈ fill(5 / 3, 2)
        push!(right, 20.0; chain=1)
        @test diagnostics(left).n_per_chain[1] == [4, 4]
        @test_throws Exception merge!(left, left)
        @test_throws Exception merge!(left, OnlineDiagnostics(Float32; nchains=2))
        @test diagnostics(left).n_per_chain[1] == [4, 4]
        blank = OnlineDiagnostics(Float64; nchains=2)
        merge!(blank, left)
        @test diagnostics(blank).variance == diagnostics(left).variance

        almostfull = OnlineDiagnostics(Float64; nchains=2)
        push!(almostfull, 1.0; chain=2, count=typemax(Int))
        @test_throws Exception merge!(almostfull, left)
        @test diagnostics(almostfull).n_per_chain[1] == [0, typemax(Int)]
    end

    @testset "Concurrent producers and snapshots" begin
        state = OnlineDiagnostics(Float64; nchains=2)
        @sync for c in 1:2, _ in 1:4
            Threads.@spawn for _ in 1:20
                append!(state, [1.0, 3.0]; chain=c)
            end
        end
        report = diagnostics(state)
        @test report.n_per_chain[1] == [160, 160]
        @test report.mean[1] ≈ [2.0, 2.0]
        @test report.variance[1] ≈ fill(160 / 159, 2)

        coherent = OnlineDiagnostics(Float64; nchains=2)
        observations = Channel{Any}(100)
        @sync begin
            Threads.@spawn for _ in 1:100
                append!(coherent, [1.0 1.0; 3.0 3.0])
                yield()
            end
            Threads.@spawn for _ in 1:100
                put!(observations, diagnostics(coherent))
                yield()
            end
            Threads.@spawn for _ in 1:20
                try
                    append!(coherent, [9.0 9.0; 9.0 NaN])
                catch
                end
            end
        end
        close(observations)
        @test all(r -> r.n_per_chain[1][1] == r.n_per_chain[1][2], observations)
        @test diagnostics(coherent).n_per_chain[1] == [200, 200]
        @test diagnostics(coherent).mean[1] ≈ [2.0, 2.0]
    end

    @testset "Centered and scaled moments" begin
        for T in (Float32, Float64)
            values = T(2)^T(precision(T) - 1) .+ T[0, 1, 2, 3, 4, 5, 6, 7]
            state = OnlineDiagnostics(T; nchains=1)
            append!(state, values; chain=1)
            @test diagnostics(state).variance[1][1] ≈ T(var(BigFloat.(values)))

            scale = sqrt(floatmax(T) / T(2))
            extremes = scale .* T[-1 -1; -1 1; 1 -1; 1 1]
            large = OnlineDiagnostics(T; nchains=2)
            append!(large, extremes)
            @test diagnostics(large).variance[1][1] ≈ T(var(BigFloat.(extremes[:, 1])))
            @test rhat(large; kind=:basic, split=false) ≈ sqrt(T(3) / T(4))

            firsthalf = OnlineDiagnostics(T; nchains=1)
            secondhalf = OnlineDiagnostics(T; nchains=1)
            append!(firsthalf, values[1:4]; chain=1)
            append!(secondhalf, values[5:8]; chain=1)
            merge!(firsthalf, secondhalf)
            @test diagnostics(firsthalf).variance[1][1] ≈ T(6)
        end

        huge = OnlineDiagnostics(Float64; nchains=2)
        append!(huge, [-floatmax(Float64) -floatmax(Float64); floatmax(Float64) floatmax(Float64)])
        @test diagnostics(huge).mean[1] == [0.0, 0.0]
        @test rhat(huge; kind=:basic, split=false) ≈ sqrt(0.5)

        tiny = OnlineDiagnostics(Float64; nchains=2)
        append!(tiny, [-1e-200 -1e-200; 1e-200 1e-200])
        @test rhat(tiny; kind=:basic, split=false) ≈ sqrt(0.5)

        mixed = OnlineDiagnostics(Float64; nchains=2)
        x = [0.0 1e200; 1e-200 nextfloat(1e200); 2e-200 nextfloat(nextfloat(1e200))]
        append!(mixed, x)
        reference = sqrt(big(2) / 3 + var(vec(mean(BigFloat.(x); dims=1))) / mean(vec(var(BigFloat.(x); dims=1))))
        @test rhat(mixed; kind=:basic, split=false) ≈ Float64(reference)
    end

    @testset "Moving mean after an outlier" begin
        x = fill(1f0, 100000)
        x[1] = 1f4
        raw = OnlineDiagnostics(Float32; nchains=1)
        append!(raw, x; chain=1)
        compressed = OnlineDiagnostics(Float32; nchains=1)
        append!(compressed, Float32[1e4, 1]; chain=1, counts=[1, 99999])
        reference_mean = Float32(mean(BigFloat.(x)))
        reference_variance = Float32(var(BigFloat.(x)))
        @test diagnostics(raw).mean[1][1] ≈ reference_mean rtol=16eps(Float32)
        @test diagnostics(raw).variance[1][1] ≈ reference_variance rtol=16eps(Float32)
        @test diagnostics(compressed).mean[1][1] ≈ reference_mean rtol=16eps(Float32)
        @test diagnostics(compressed).variance[1][1] ≈ reference_variance

        values = Float32[1e8, 1, 3]
        counts = [1, 5 * 10^17, 5 * 10^17]
        n = sum(BigInt, counts)
        exact_mean = sum(BigFloat.(values) .* counts) / n
        exact_variance = sum((BigFloat.(values) .- exact_mean).^2 .* counts) / (n - 1)
        large = OnlineDiagnostics(Float32; nchains=1)
        append!(large, values; chain=1, counts)
        @test diagnostics(large).mean[1][1] ≈ Float32(exact_mean)
        @test diagnostics(large).variance[1][1] ≈ Float32(exact_variance)

        left = OnlineDiagnostics(Float32; nchains=1)
        right = OnlineDiagnostics(Float32; nchains=1)
        push!(left, values[1]; chain=1, count=counts[1])
        append!(right, values[2:3]; chain=1, counts=counts[2:3])
        merge!(left, right)
        @test diagnostics(left).mean[1][1] ≈ Float32(exact_mean)
        @test diagnostics(left).variance[1][1] ≈ Float32(exact_variance)

        reversed = OnlineDiagnostics(Float32; nchains=1)
        append!(reversed, reverse(values); chain=1, counts=reverse(counts))
        @test diagnostics(reversed).mean[1][1] ≈ Float32(exact_mean)
        @test diagnostics(reversed).variance[1][1] ≈ Float32(exact_variance)
    end
end
