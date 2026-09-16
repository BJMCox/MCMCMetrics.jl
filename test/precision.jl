@testset "Ordered repetitions and precision" begin
    x = Float32[0 5; 3 1; 1 4; 5 2; 2 0; 4 3; 0 4; 2 1; 5 3]
    w = [2, 0, 4, 1, 3, 2, 2, 1, 2]
    counts = hcat(w, w)
    dense = hcat([[x[i, c] for i in eachindex(w) for _ in 1:w[i]] for c in 1:2]...)
    for kind in (:mean, :bulk, :tail, :quantile, :interval)
        options = (; kind, prob=0.4, interval=(1, 3))
        @test isapprox(ess(x; counts, options...), ess(dense; options...); nans=true, rtol=5f-6)
    end
    @test autocor(x; counts, lags=[0, 1, 5, 12, 16]) ≈ autocor(dense; lags=[0, 1, 5, 12, 16])
    @test iact(x; counts) ≈ iact(dense)
    for estimand in (:mean, :variance, :sd, :quantile, :median)
        @test isapprox(mcse(x; counts, estimand), mcse(dense; estimand); nans=true, rtol=5f-6)
    end
    # A handful of runs can represent enormous logical lengths. A short lag
    # query still works, while a long Geyer window must respect its work budget.
    runs, repetitions = [0., 1., 0., 1.], fill(10^8, 4)
    @test autocor(runs; counts=repetitions, lags=[0, 1])[2] ≈ 1 - 7 / (4 * 10^8)
    @test_throws MCMCMetrics._WorkLimitError iact(runs; counts=repetitions, max_lag_work=64)
    @test_throws MCMCMetrics._WorkLimitError mcse(runs; counts=repetitions, estimator=:batchmeans, max_lag_work=64)
end

@testset "Batch MCSE" begin
    x = [1., 2, 4, 8, 3, 6, 9]
    # Means of the three retained pairs are 3/2, 6, 9/2. The final 9 is discarded.
    @test mcse(x; estimator=:batchmeans, batch_size=2)^2 ≈ 7 / 4
    @test mcse(x; estimator=:overlapping, batch_size=2)^2 ≈ 1144 / 735
    @test mcse(hcat(x, x .+ 100); estimator=:batchmeans, batch_size=2) ≈
        mcse(x; estimator=:batchmeans, batch_size=2) / sqrt(2)
    # The default rule floor(sqrt(7)) chooses the same size, without splitting.
    @test mcse(x; estimator=:batchmeans) ≈ mcse(x; estimator=:batchmeans, batch_size=2)
    compressed = [1., 4, 3, 9]
    counts = [2, 2, 2, 1]
    for estimator in (:batchmeans, :overlapping)
        @test mcse(compressed; counts, estimator, batch_size=2) ≈
            mcse([compressed[i] for i in eachindex(counts) for _ in 1:counts[i]]; estimator, batch_size=2)
    end
end

@testset "Batch degeneracy and native run arithmetic" begin
    alternating = [0., 1, 0, 1]
    for estimator in (:batchmeans, :overlapping)
        @test isnan(mcse(alternating; estimator, batch_size=2))
        @test isnan(mcse(hcat(alternating, [1., 2, 4, 8]); estimator, batch_size=2))
    end
    # The discarded singleton must not influence either centering or scaling.
    for T in (Float32, Float64)
        @test mcse(T[1, 2, 4, 8, 3, 6, 1e20]; estimator=:batchmeans, batch_size=2) ≈ T(sqrt(7 / 4))
    end
    @test isnan(mcse(Float32[0, 1]; counts=[10^12, 1], estimator=:batchmeans, batch_size=10^9))
    # Ten million means at -1 and ten million at +1 give this exact SE.
    @test mcse(Float32[-1, 1]; counts=[10^12, 10^12], estimator=:batchmeans,
        batch_size=100000) ≈ Float32(inv(sqrt(19_999_999)))

    values, counts, b = [-3., 2, 1, -2], [5, 1, 4, 3], 3
    dense = [values[i] for i in eachindex(counts) for _ in 1:counts[i]]
    means = [mean(dense[i:i+b-1]) for i in 1:b:(length(dense)-b+1)]
    @test mcse(values; counts, estimator=:batchmeans, batch_size=b)^2 ≈ var(means) / length(means)
    overlaps = [mean(dense[i:i+b-1]) for i in 1:(length(dense)-b+1)]
    oracle = b / ((length(dense)-b) * length(overlaps)) * sum(abs2, overlaps .- mean(dense))
    @test mcse(values; counts, estimator=:overlapping, batch_size=b)^2 ≈ oracle
end

@testset "Quantile and influence precision" begin
    x = [0., 1, 4, 2, 5, 1, 0, 3, 4, 2, 1, 5]
    # Invert the Beta CDF by bisection independently of beta_inc_inv, then
    # map through Statistics' type-7 quantile implementation.
    effective = ess(x; kind=:quantile, prob=0.5, split=false)
    a = effective / 2 + 1
    function probability_bound(target)
        lo, hi = 0., 1.
        for _ in 1:60
            mid = (lo + hi) / 2
            if first(SpecialFunctions.beta_inc(a, a, mid)) < target
                lo = mid
            else
                hi = mid
            end
        end
        (lo + hi) / 2
    end
    tail = SpecialFunctions.erfc(inv(sqrt(2))) / 2
    bounds = probability_bound.([tail, 1 - tail])
    expected = (quantile(x, bounds[2]) - quantile(x, bounds[1])) / 2
    @test mcse(x; estimand=:quantile, split=false) ≈ expected
    @test mcse(x; estimand=:median, split=false) ≈ expected
    tied = repeat([0., 0, 0, 1]; outer=40)
    @test isnan(mcse(tied; estimand=:quantile, prob=0.2, split=false))
    # Independent exact-rational covariance and IMS calculations on squared
    # centered draws provide these non-Gaussian influence references.
    @test mcse(x; estimand=:variance, split=false)^2 ≈ 253 / 1944
    @test mcse(x; estimand=:sd, split=false)^2 ≈ 23 / 2160
    scale = ldexp(1f0, 50)
    @test mcse(Float32.(x) .* scale; estimand=:variance, split=false) / scale / scale ≈
        Float32(mcse(x; estimand=:variance, split=false))
    @test mcse(Float32.(x) .* scale; estimand=:sd, split=false) / scale ≈
        Float32(mcse(x; estimand=:sd, split=false))
end

@testset "Sokal and diagnostic curves" begin
    x = [1., 2, 3, 4, 2, 1, 3, 2]
    # rho1=1/40 and rho2=-29/60: with c=1 the first accepted M is 2.
    @test iact(x; estimator=:sokal, window=1) ≈ 1 / 12
    @test isnan(ess(hcat(x, x .+ 100); kind=:mean, split=false, estimator=:sokal, window=100))
    counts = fill(2, length(x))
    curve = diagnostic_curve(x; prefixes=[16, 12], counts, kind=:mean, split=false)
    @test curve.prefixes == [16, 12]
    dense = repeat(x; inner=2)
    @test curve.values ≈ [ess(dense[1:n]; kind=:mean, split=false) for n in curve.prefixes]
    @test cost_normalize(curve.values, 2) ≈ curve.values / 2
end

using FFTW

@testset "FFTW precision extension" begin
    x = Float32.(reshape(sin.(1:160), 40, 4))
    @test autocor(x; autocov=:fft, lags=0:12) ≈ autocor(x; lags=0:12)
    for estimator in (:geyer, :sokal)
        @test ess(x; autocov=:fft, estimator, window=1) ≈ ess(x; estimator, window=1)
        @test mcse(x; autocov=:fft, estimator, window=1) ≈ mcse(x; estimator, window=1)
        @test iact(x; autocov=:fft, estimator, window=1) ≈ iact(x; estimator, window=1)
    end
end
