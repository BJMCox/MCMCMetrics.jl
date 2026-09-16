@testset "Geyer ESS and mean precision" begin
    # Rational arithmetic gives rho-pair zero 73669/102960. The next pair
    # is negative, so tau = 22189/51480 and ESS = 24/tau.
    antithetic = [0 3; 1 0; 4 1; 2 4; 5 2; 1 5; 0 1; 3 2; 4 0; 2 4; 1 3; 5 2]
    expected = 1235520 / 22189
    effective = ess(antithetic; kind=:mean, split=false)
    @test effective ≈ expected
    @test effective > length(antithetic)
    # Pooled raw sample variance is 1559/552.
    @test mcse(antithetic; split=false)^2 ≈ 34592651 / 682007040

    # Positive pairs are [68337,42685,38233,40949]/39424. IMS lowers
    # the final pair to 38233/39424, yielding tau = 749/88.
    nonmonotone = [1 2; 1 2; 1 0; 3 -1; 4 -2; 3 -1; 1 -3; 3 -1]
    @test ess(nonmonotone; kind=:mean, split=false) ≈ 1408 / 749

    # MCMCDiagnosticTools 0.3.19, kind=:basic and split_chains=1,
    # agrees here because its terminal-lag correction and ESS cap are inactive.
    reference = hcat([0,0,1,1,2,2,0,0,1,1,2,2], [2,2,0,0,1,1,2,2,0,0,1,1])
    @test ess(reference; kind=:mean, split=false) ≈ 2112 / 127

    shifted = (Int64(1) << 60) .+ antithetic
    @test ess(shifted; kind=:mean, split=false) ≈ expected
    @test mcse(shifted; split=false)^2 ≈ 34592651 / 682007040

    scale = ldexp(1f0, 100)
    large = Float32.(antithetic) .* scale
    estimate = ess(large; kind=:mean, split=false)
    error = mcse(large; split=false)
    @test estimate isa Float32 && isapprox(estimate, expected; rtol=2f-6)
    @test error isa Float32 && isapprox(error / scale, sqrt(34592651 / 682007040); rtol=2f-6)

    distant = Float32[0 1073741824; 1 1073741952; 4 1073742080; 2 1073742208;
        5 1073742336; 1 1073741952; 0 1073741824; 3 1073742080]
    # Exact rational arithmetic on the observed values gives these references.
    exact_ess = big"18446751151816789104" // big"17293829204827666291"
    exact_mcse2 = big"3323072539619640028587532551295938293" // big"11529219469885493190"
    @test ess(distant; kind=:mean, split=false) ≈ Float32(exact_ess)
    @test mcse(distant; split=false) ≈ Float32(sqrt(BigFloat(exact_mcse2)))

    subnormal = nextfloat(0.0)
    constant_chain = hcat(fill(subnormal, 8), [0.,1,4,2,5,1,0,3] .* subnormal)
    @test isnan(ess(constant_chain; kind=:mean, split=false))
    @test isnan(mcse(constant_chain; split=false))
end

@testset "Autocorrelation and per-chain IACT" begin
    chain = [1,2,3,4,2,1,3,2]
    @test autocor(chain; lags=0:3) ≈ [1, 1/40, -29/60, -9/40]
    @test iact(chain) ≈ 21 / 20
    @test iact(hcat(chain, reverse(chain))) ≈ [21/20, 21/20]
    tensor = cat(hcat(chain, reverse(chain)), hcat(chain .+ 5, reverse(chain) .+ 5); dims=3)
    parameter_first = PermutedDimsArray(tensor, (3, 1, 2))
    @test autocor(parameter_first; drawdim=2, chaindim=3, lags=0:3) ≈ autocor(tensor; lags=0:3)
    @test iact(parameter_first; drawdim=2, chaindim=3) ≈ iact(tensor)
    @test iact(chain; counts=fill(2, length(chain))) ≈ iact(repeat(chain; inner=2))
end

@testset "Rank and event ESS" begin
    x = [0 3; 1 0; 4 1; 2 4; 5 2; 1 5; 0 1; 3 2; 4 0; 2 4; 1 3; 5 2]
    @test ess(x) ≈ ess(x .^ 3)
    @test ess(x; kind=:quantile, prob=0.5) ≈ ess(x .<= 2; kind=:mean)
    @test ess((Int64(1) << 60) .+ x; kind=:quantile, prob=0.5) ≈ ess(x; kind=:quantile, prob=0.5)
    @test ess(x; kind=:interval, interval=(2, 4), split=false) ≈
        ess((2 .<= x) .& (x .<= 4); kind=:mean, split=false)
    tails = reshape(mod.(0:319, 40), 80, 4)
    @test ess(tails; kind=:tail) ≈ min(ess(tails .<= 1; kind=:mean), ess(tails .<= 37; kind=:mean))

    # The middle draw leaves the estimator before quantiles or ranks are formed.
    odd = vcat(x[1:6, :], [10^8 10^9], x[7:12, :])
    @test ess(odd) ≈ ess(x)
    @test ess(odd; kind=:quantile, prob=0.5) ≈ ess(x; kind=:quantile, prob=0.5)
    @test mcse(odd) > mcse(x)

    near_endpoint = Float32[0, 1, 0, 2, 0, 3, 0, 4]
    @test ess(near_endpoint; kind=:quantile, prob=prevfloat(1.0), split=false) ≈ Float32(224/27)
end
