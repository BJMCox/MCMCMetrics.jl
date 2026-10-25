using Test, Statistics
import Random
import MCMCMetrics, MCMCDiagnosticTools, PSIS, PosteriorStats

# MCMCDiagnosticTools 0.3.19. Even lengths avoid its rank-before-split
# convention for odd chains. The ESS fixture avoids its terminal-lag correction
# and finite-sample cap, which differ from our plain initial monotone sequence.
@testset "MCMCDiagnosticTools 0.3.19 reference" begin
    x = Float64[0 2; 1 0; 3 4; 2 3; 4 -1; -1 5; 5 1; 2 6]
    @test MCMCMetrics.rhat(x) ≈ MCMCDiagnosticTools.rhat(x)
    @test MCMCMetrics.rhat(x; kind=:basic, split=false) ≈
        MCMCDiagnosticTools.rhat(x; kind=:basic, split_chains=1)

    y = Float64[0 2; 0 2; 1 0; 1 0; 2 1; 2 1; 0 2; 0 2; 1 0; 1 0; 2 1; 2 1]
    @test MCMCMetrics.ess(y; kind=:mean, split=false) ≈
        MCMCDiagnosticTools.ess(y; kind=:basic, split_chains=1) ≈ 2112 / 127
    @test MCMCMetrics.ess(y; split=false) ≈
        MCMCDiagnosticTools.ess(y; kind=:bulk, split_chains=1)
    @test MCMCMetrics.mcse(y; split=false) ≈
        MCMCDiagnosticTools.mcse(y; kind=mean, split_chains=1)
end

@testset "MCMCDiagnosticTools 0.3.19 on AR(1) and stuck chains" begin
    # Deterministic noise keeps the fixture independent of RNG streams.
    x = zeros(1000, 4)
    for c in 1:4, i in 1:1000
        noise = mod(sin(12.9898i + 78.233c) * 43758.5453, 1) - 0.5
        x[i, c] = (i > 1 ? 0.9x[i-1, c] : 0.0) + noise
    end
    y = reshape(x, size(x)..., 1)
    @test MCMCMetrics.rhat(x) ≈ only(MCMCDiagnosticTools.rhat(y)) rtol=1e-14
    @test MCMCMetrics.rhat(x; kind=:basic) ≈ only(MCMCDiagnosticTools.rhat(y; kind=:basic)) rtol=1e-14
    @test MCMCMetrics.ess(x; kind=:tail) ≈ only(MCMCDiagnosticTools.ess(y; kind=:tail)) rtol=1e-12
    # MCMCDiagnosticTools adds a terminal even-lag correction to IMS.
    @test MCMCMetrics.ess(x) ≈ only(MCMCDiagnosticTools.ess(y)) rtol=5e-4
    @test MCMCMetrics.mcse(x) ≈ only(MCMCDiagnosticTools.mcse(y)) rtol=5e-4
    @test MCMCMetrics.mcse(x; kind=:sd) ≈ only(MCMCDiagnosticTools.mcse(y; kind=std)) rtol=5e-4

    stuck, half = copy(x), copy(x)
    stuck[:, 3] .= 5.0
    half[1:500, 1] .= 0.0
    for z in (stuck, half)
        @test MCMCMetrics.rhat(z) ≈ only(MCMCDiagnosticTools.rhat(reshape(z, size(z)..., 1))) rtol=1e-14
    end
end

@testset "Remaining pinned Julia references" begin
    binary = repeat([0.,0,0,1,0,1,1,1],100)
    ours = MCMCMetrics.raftery_lewis(binary;prob=.5,atol=.1,confidence=.9)
    reference = MCMCDiagnosticTools.rafterydiag(binary;q=.5,r=.1,s=.9)
    @test (ours.thinning,ours.burnin,ours.total,ours.nmin) ==
        (reference.thinning,reference.burnin,reference.total,reference.nmin)

    grouped = hcat(([-1.,1,-1,1] .+ c for c in (0,2,4,6))...)
    @test MCMCMetrics.rhat_nested(grouped,[1,1,2,2];kind=:basic,split=false) ≈
        MCMCDiagnosticTools.rhat_nested(grouped,[1,1,2,2];kind=:basic,split_chains=1)
    @test MCMCMetrics.rhat_nested(grouped,[1,1,2,2]) ≈
        MCMCDiagnosticTools.rhat_nested(grouped,[1,1,2,2])
    x = repeat([1. 0;-1 0;0 1;0 -1];inner=(2,1))
    # Distinct chain variances keep the reference's unrelated F confidence
    # interval away from its infinite-degrees-of-freedom quantile path.
    chains = cat(reshape(x,8,1,2),reshape(x .* [1.2 .8] .+ [1 .5],8,1,2);dims=2)
    @test MCMCMetrics.rhat_multivariate(chains;split=false) ≈
        sqrt(MCMCDiagnosticTools.gelmandiag_multivariate(chains).psrfmultivariate)

    states = [0 1;0 1;0 1;1 0;1 0;1 0;0 1;0 1]
    comparison = MCMCMetrics.categorical_weiss(states)
    independent = MCMCDiagnosticTools.weiss(states)
    @test [comparison.statistic,comparison.pvalue] ≈ [independent[1],independent[3]]
    # IID draws whose phi estimate is negative noise, which the reference clips.
    states = rand(Random.Xoshiro(3),1:3,1000,4)
    comparison = MCMCMetrics.categorical_weiss(states)
    independent = MCMCDiagnosticTools.weiss(states)
    @test comparison.phi == 0
    @test [comparison.statistic,comparison.pvalue] ≈ [independent[1],independent[3]]

    ratios = log.(collect(1.:200))
    fit, reference_fit = MCMCMetrics.psis(ratios;reff=1), PSIS.psis(ratios)
    @test fit.pareto_k ≈ reference_fit.pareto_shape
    @test fit.log_weights ≈ reference_fit.log_weights
    likelihood = reshape(-log.(collect(1.:800)),200,2,2)
    result = MCMCMetrics.loo(likelihood;reff=1)
    reference_result = PosteriorStats.loo(likelihood;reff=1)
    @test result.pointwise.elpd ≈ reference_result.pointwise.elpd
    @test [result.elpd,result.p_loo,result.se_elpd] ≈
        [reference_result.estimates.elpd,reference_result.estimates.p,reference_result.estimates.se_elpd]
end
