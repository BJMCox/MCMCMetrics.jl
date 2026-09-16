using MCMCMetrics, Random, Statistics, Test

# Stationary unit-variance AR(1). Its finite-length mean variance is analytic.
function stationary_ar1(rng, n, chains, rho)
    x = randn(rng, n, chains)
    noise = sqrt(1 - rho^2)
    for chain in axes(x, 2), i in 2:n
        x[i, chain] = rho * x[i - 1, chain] + noise * x[i, chain]
    end
    return x
end

@testset "Stationary process uncertainty" begin
    rng = Xoshiro(2811)
    n, chains, repetitions = 4096, 4, 64
    for rho in (-0.5, 0.0, 0.8)
        finite_tau = 1 + 2sum((1 - lag / n) * rho^lag for lag in 1:(n - 1))
        target_variance = finite_tau / (n * chains)
        means, errors, times = Float64[], Float64[], Float64[]
        for _ in 1:repetitions
            x = stationary_ar1(rng, n, chains, rho)
            push!(means, mean(x))
            push!(errors, mcse(x; split=false))
            push!(times, n * chains / ess(x; kind=:mean, split=false))
        end
        coverage = count(abs.(means) .<= 1.96 .* errors) / repetitions
        @info "AR(1) validation" rho finite_tau estimated_tau=mean(times) coverage
        @test mean(times) ≈ finite_tau rtol=0.10
        @test mean(abs2, errors) ≈ target_variance rtol=0.10
        # The variance-of-means estimate has relative SE sqrt(2/(repetitions-1)).
        @test var(means) ≈ target_variance rtol=4sqrt(2 / (repetitions - 1))
        @test coverage >= 0.95 - 4sqrt(0.95 * 0.05 / repetitions)
    end
end
