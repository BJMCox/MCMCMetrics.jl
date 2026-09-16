using Test, Statistics
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
