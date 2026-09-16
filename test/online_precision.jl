# Independent oracle: derive complete-batch groups from interval overlaps, with
# BigFloat arithmetic only in tests. Runtime depends on run boundaries, not n.
function online_precision_oracle(values, counts=ones(Int, length(values)))
    setprecision(BigFloat, 256) do
        n = sum(counts)
        b = 1
        while BigInt(4) * b^2 <= n
            b *= 2
        end
        a, remainder = divrem(n, b)
        endpoints = cumsum(counts)
        starts = vcat(0, endpoints[1:end-1])
        special = unique([div(t, b) + 1 for t in endpoints[1:end-1] if rem(t, b) != 0 && t < a*b])
        groups = Tuple{BigFloat,Int}[]
        for (x, start, stop) in zip(values, starts, endpoints)
            firstbatch, lastbatch = cld(start, b) + 1, min(div(stop, b), a)
            multiplicity = max(0, lastbatch - firstbatch + 1)
            multiplicity > 0 && push!(groups, (BigFloat(x), multiplicity))
        end
        for j in special
            total = sum(BigFloat(x) * max(0, min(stop, j*b) - max(start, (j-1)*b))
                for (x, start, stop) in zip(values, starts, endpoints))
            push!(groups, (total / b, 1))
        end
        μ = sum(BigFloat(x) * k for (x, k) in zip(values, counts)) / n
        variance = sum((BigFloat(x)-μ)^2 * k for (x,k) in zip(values,counts)) / (n-1)
        batchmean = sum(x*k for (x,k) in groups) / a
        lrv = b * sum((x-batchmean)^2*k for (x,k) in groups) / (a-1)
        return (mean=μ, variance=variance, mcse=sqrt(lrv/n), ess=n*variance/lrv,
            iact=lrv/variance, width=b, batches=a, remainder=remainder)
    end
end

function online_lag_oracle(x, cap)
    setprecision(BigFloat, 256) do
        y = BigFloat.(x)
        z = y .- sum(y)/length(y)
        return [sum(z[1:end-k] .* z[1+k:end]) / sum(abs2,z) for k in 0:cap]
    end
end

@testset "Online ordered precision" begin
    metrics = (:mean, :variance, :mcse_mean, :ess_mean, :iact_mean)
    @testset "Dyadic prefixes and complete-batch centering" begin
        for T in (Float32, Float64)
            x = T[mod(7i, 13) + div(i, 6) for i in 1:70]
            state = OnlineDiagnostics(T; nchains=1, metrics)
            for i in eachindex(x)
                push!(state, x[i]; chain=1)
                i in (3,4,5,15,16,17,63,64,65) || continue
                expected = online_precision_oracle(x[1:i])
                report = diagnostics(state)
                @test report.mcse_mean[1][1] ≈ T(expected.mcse) rtol=64eps(T)
                @test report.ess_mean[1][1] ≈ T(expected.ess) rtol=64eps(T)
                @test report.iact_mean[1][1] ≈ T(expected.iact) rtol=64eps(T)
                @test report.precision[1][1] == (estimator=:dyadic_batchmeans,
                    batch_width=expected.width, completed_batches=expected.batches,
                    remainder=expected.remainder, n_draws=i)
            end
        end
        state = OnlineDiagnostics(Float64; nchains=1, metrics)
        append!(state, [0.0,0,2,2,100]; chain=1)
        @test only(mcse(state)) ≈ sqrt(4/5)
        @test only(ess(state)) ≈ 2451.5
        @test only(iact(state)) ≈ 4/1961.2
    end

    @testset "Compressed chronology and crossed levels" begin
        values, counts = Float32[3,7,-2,9,1], [3,67,2,61,9]
        x = values[vcat([fill(i,k) for (i,k) in enumerate(counts)]...)]
        compressed = OnlineDiagnostics(Float32; nchains=1, metrics)
        append!(compressed, values; chain=1, counts)
        chunks = OnlineDiagnostics(Float32; nchains=1, metrics)
        for range in (1:2,3:16,17:64,65:93,94:length(x))
            append!(chunks, x[range]; chain=1)
        end
        split = OnlineDiagnostics(Float32; nchains=1, metrics)
        for (v,k) in zip(values,counts)
            push!(split,v;chain=1,count=div(k,2))
            push!(split,v;chain=1,count=k-div(k,2))
        end
        expected = online_precision_oracle(values,counts)
        @test only(mcse(compressed)) ≈ Float32(expected.mcse) rtol=64eps(Float32)
        @test mcse(compressed) ≈ mcse(chunks) ≈ mcse(split)
        @test ess(compressed) ≈ ess(chunks) ≈ ess(split)
    end

    @testset "Native precision, large offsets and counts" begin
        for T in (Float32,Float64)
            offset = T(2)^T(precision(T)-1)
            values = offset .+ T[0,0,0,1,0,0,1,1,1,1,1,2,0,1,2,3,0]
            state = OnlineDiagnostics(T;nchains=1,metrics)
            append!(state,values;chain=1)
            expected = online_precision_oracle(values)
            @test only(mcse(state)) ≈ T(expected.mcse) rtol=64eps(T)
            @test only(ess(state)) ≈ T(expected.ess) rtol=64eps(T)
            extreme = OnlineDiagnostics(T;nchains=1,metrics)
            extremevalues = floatmax(T) .* T[-1,-1,1,1,-1,-1,1,1]
            append!(extreme,extremevalues;chain=1)
            oracle = online_precision_oracle(extremevalues)
            @test only(mcse(extreme)) ≈ T(oracle.mcse) rtol=64eps(T)
            @test only(ess(extreme)) ≈ T(oracle.ess) rtol=64eps(T)
        end
        values, counts = Float32[1e8,1,3,-2], [3,5*10^17,2^25+3,2^26+1]
        huge = OnlineDiagnostics(Float32;nchains=1,metrics)
        append!(huge,values;chain=1,counts)
        expected = online_precision_oracle(values,counts)
        @test diagnostics(huge).n_per_chain[1] == [sum(counts)]
        @test only(mcse(huge)) ≈ Float32(expected.mcse) rtol=128eps(Float32)
        @test only(ess(huge)) ≈ Float32(expected.ess) rtol=128eps(Float32)
    end

    @testset "Unavailable precision and fixed indicators" begin
        constant = OnlineDiagnostics(Float32;nchains=1,metrics=(:mcse_mean,:indicator),threshold=2)
        push!(constant,3f0;chain=1,count=16)
        @test diagnostics(constant).status[1].mcse_mean == [:constant]
        @test diagnostics(constant).indicator[1][1].status === :unobserved_event
        alternating = OnlineDiagnostics(Float64;nchains=1,metrics=(:mcse_mean,))
        append!(alternating,repeat([0.0,1],8);chain=1)
        @test diagnostics(alternating).status[1].mcse_mean == [:degenerate_estimator]
        @test isnan(only(mcse(alternating)))
        thresholds = Float32[0,2]
        state = OnlineDiagnostics(Float32;nchains=2,parameter_shape=(2,),
            metrics=(:indicator,),threshold=thresholds)
        thresholds .= 900
        x = Float32[-1 1; -1 1; -1 4; 1 4; 1 1; 2 3; 2 3; 2 1; -1 3]
        append!(state,x;chain=2)
        append!(state,reverse(x;dims=1);chain=1)
        report = diagnostics(state)
        for c in 1:2, p in 1:2
            events = Float32.((c==1 ? reverse(x[:,p]) : x[:,p]) .<= (p==1 ? 0 : 2))
            expected = online_precision_oracle(events)
            result = report.indicator[p][c]
            @test result.probability ≈ Float32(expected.mean)
            @test result.mcse_mean ≈ Float32(expected.mcse)
        end
    end

    @testset "Completed means exclude canceled within-batch scale" begin
        state = OnlineDiagnostics(Float32;nchains=1,metrics=(:mcse_mean,))
        for magnitude in (1f20,1f23)
            empty!(state)
            x = Float32[-magnitude,magnitude,0,0,-magnitude,magnitude,0,4,
                -magnitude,magnitude,0,8,-magnitude,magnitude,0,12]
            append!(state,x;chain=1)
            @test only(mcse(state)) ≈ sqrt(5f0/12)
            @test only(diagnostics(state)).status.mcse_mean == [:ok]
        end
    end

    @testset "Fixed-lag global centering and ring ordering" begin
        values, counts = Float32[2,-1,5,1,8], [2,3,19,1,7]
        x = values[vcat([fill(i,k) for (i,k) in enumerate(counts)]...)]
        state = OnlineDiagnostics(Float32;nchains=2,metrics=(:autocor,),max_lag=5)
        for (v,k) in zip(values,counts)
            push!(state,v;chain=1,count=k)
        end
        append!(state,reverse(x);chain=2)
        reference = Float32.(online_lag_oracle(x,5))
        @test autocor(state)[:,1] ≈ reference rtol=64eps(Float32)
        @test autocor(state)[:,2] ≈ reference rtol=64eps(Float32)
        @test diagnostics(state).autocor[1] ≈ hcat(reference,reference) rtol=64eps(Float32)
        @test autocor(state;lags=[5,1]) ≈ hcat(reference[[6,2]],reference[[6,2]])
        offset = OnlineDiagnostics(Float32;nchains=1,metrics=(:autocor,),max_lag=3)
        shifted = 1f8 .+ 8f0 .* x
        append!(offset,shifted;chain=1)
        @test autocor(offset)[:,1] ≈ Float32.(online_lag_oracle(shifted,3)) rtol=64eps(Float32)
        huge = OnlineDiagnostics(Float32;nchains=1,metrics=(:autocor,),max_lag=2)
        push!(huge,0f0;chain=1,count=2^26)
        push!(huge,1f0;chain=1,count=2^26)
        @test autocor(huge)[:,1] ≈ Float32[1,1-3/2^27,1-6/2^27]
    end

    @testset "Ordered-state lifecycle and coherent snapshots" begin
        allmetrics = (metrics...,:indicator,:autocor)
        state = OnlineDiagnostics(Float64;nchains=2,metrics=allmetrics,threshold=0,max_lag=3)
        x = [-2.0 3; -1 4; 1 -2; 3 -1; 2 1]
        append!(state,x)
        before = diagnostics(state)
        @test_throws Exception append!(state,[2.0 3; 4 NaN])
        source = OnlineDiagnostics(Float64;nchains=2,metrics=allmetrics,threshold=0,max_lag=3)
        append!(source,x)
        @test_throws Exception merge!(state,source)
        @test isequal(diagnostics(state),before)
        before.autocor[1][1,1] = 900
        @test diagnostics(state).autocor[1][1,1] == 1
        retained = diagnostics(state)
        empty!(state)
        @test diagnostics(state).n_draws == [0]
        @test all(isnan,mcse(state))
        append!(state,x)
        @test diagnostics(state).mcse_mean == retained.mcse_mean
        @test diagnostics(state).autocor == retained.autocor
        @test diagnostics(state).indicator == retained.indicator
        overflow = OnlineDiagnostics(Float32;nchains=1,metrics=(:mcse_mean,:autocor),max_lag=1)
        push!(overflow,1f0;chain=1,count=typemax(Int))
        previous = diagnostics(overflow)
        @test_throws Exception push!(overflow,2f0;chain=1)
        @test isequal(diagnostics(overflow),previous)

        concurrent = OnlineDiagnostics(Float64;nchains=2,metrics=allmetrics,threshold=0,max_lag=3)
        @sync for c in 1:2
            Threads.@spawn append!(concurrent,x[:,c];chain=c)
        end
        @test diagnostics(concurrent).mcse_mean == retained.mcse_mean
        @test diagnostics(concurrent).autocor == retained.autocor
        empty!(concurrent)
        snapshots = Channel{Any}(20)
        @sync begin
            for c in 1:2
                Threads.@spawn for _ in 1:20
                    append!(concurrent,x[:,c];chain=c)
                    yield()
                end
            end
            Threads.@spawn for _ in 1:20
                put!(snapshots,diagnostics(concurrent))
                yield()
            end
        end
        close(snapshots)
        @test all(r -> all(r.precision[1][c].n_draws == r.n_per_chain[1][c] for c in 1:2), snapshots)
    end
end
