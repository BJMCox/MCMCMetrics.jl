@testset "Single-chain checks" begin
    pattern = repeat([-1.0, -1, 1, 1], 2)
    x = [pattern; zeros(8); pattern .+ 1]
    g = geweke(x; first_fraction=1//3, last_fraction=1//3)
    # Window variance = 8/7; paired autocorrelations give IACT = 5/4.
    @test g.zscore ≈ -sqrt(14/5)
    @test g.pvalue ≈ SpecialFunctions.erfc(sqrt(7/5))
    @test g.status === :ok
    discarded = [pattern; fill(1e100,8); pattern .+ 1]
    @test geweke(discarded; first_fraction=1//3, last_fraction=1//3) == g
    @test geweke(x .* 1e200; first_fraction=1//3, last_fraction=1//3).zscore ≈ g.zscore
    opposite = [pattern .+ 15; zeros(8); pattern .- 15]
    @test geweke(opposite .* 1e307; first_fraction=1//3, last_fraction=1//3).zscore ≈
        geweke(opposite; first_fraction=1//3, last_fraction=1//3).zscore
    shaped = reshape(Float32.([x; x .+ 20]), 24, 2, 1)
    @test geweke(shaped; first_fraction=1//3, last_fraction=1//3).zscore ≈ fill(Float32(g.zscore),2,1)
    @test geweke(fill(3.0,40)).status === :constant

    h = heidelberger_welch(repeat([-1.0,-1,1,1],20) .+ 5)
    # Brownian-bridge area = 120, reference LRV = 14/13, length = 80.
    @test h.statistic ≈ 39/2240
    @test h.mean == 5 && h.stationarity && h.precision && h.burnin == 0
    @test h.halfwidth ≈ sqrt(2) * SpecialFunctions.erfcinv(0.05) * sqrt(41/3160)
    transient = heidelberger_welch([fill(100.0,40);repeat([-1.0,-1,1,1],20).+5])
    @test transient.burnin == 48 && transient.stationarity && transient.mean ≈ 5
    extreme = heidelberger_welch([fill(1e100,40);repeat([-1.0,-1,1,1],20).+5])
    @test extreme == transient
    scaled = heidelberger_welch((repeat([-1.0,-1,1,1],20) .+ 5) .* 1e200)
    @test scaled.statistic ≈ h.statistic && scaled.mean / 1e200 ≈ h.mean
    @test scaled.halfwidth / 1e200 ≈ h.halfwidth && scaled.status === :ok
    @test heidelberger_welch(ones(40)).status === :constant

    binary = repeat([0.0,0,0,1,0,1,1,1],100)
    r = raftery_lewis(binary; prob=0.5, atol=0.1, confidence=0.9)
    # All binary triples occur equally; transition probabilities are 1/2 and 199/399.
    @test (r.thinning,r.burnin,r.total,r.nmin) == (1,1,69,68)
    @test r.dependence_factor ≈ 69/68 && r.status === :ok
    @test raftery_lewis(binary[1:8]).status === :insufficient_draws
    upper = raftery_lewis(Float32.([1:49;100;50:99]); prob=1-1e-10, atol=.1, confidence=.9)
    lower = raftery_lewis(Float32.([2:50;1;51:100]); prob=1e-50, atol=.1, confidence=.9)
    @test upper.nmin == 1 && upper.total == upper.dependence_factor == 5 && upper.status === :ok
    @test lower.nmin == 1 && lower.total == lower.dependence_factor == 5 && lower.status === :ok
end
