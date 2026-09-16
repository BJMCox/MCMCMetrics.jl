@testset "Importance weights and Pareto tails" begin
    logw = log.([1.0,2,3])
    d = importance_diagnostics(logw)
    @test d.ess_is ≈ 18/7 && d.max_weight ≈ 1/2
    @test d.entropy ≈ -(log(1/6)/6 + log(1/3)/3 + log(1/2)/2)
    @test importance_diagnostics(logw .+ 700).ess_is ≈ d.ess_is
    @test importance_diagnostics(Float32[0,-Inf,0]).ess_is == 2
    @test importance_diagnostics(zeros(Float32,4)).entropy ≈ log(4f0)
    @test importance_diagnostics(logw;counts=[2,1,1]).ess_is ≈ 49/15
    @test importance_mcse([1.0,2,5],logw;sampling=:iid)^2 ≈ 169/108
    @test importance_mcse([1.0,2,5,1e100],[logw;-Inf];sampling=:iid)^2 ≈ (169/162)*(4/3)

    ratios = log.(collect(1.0:200))
    original = copy(ratios)
    p = psis(ratios;reff=1)
    @test sum(exp,p.log_weights) ≈ 1 && ratios == original
    @test p.pareto_k ≈ -0.6360133060563312 rtol=1e-11 # pinned PSIS.jl 0.9.10
    @test psis(Float32.(ratios);reff=1).pareto_k ≈ Float32(p.pareto_k) rtol=2f-5
    shifted = psis(ratios .+ 600;reff=0.7)
    reference = psis(ratios;reff=0.7)
    @test shifted.log_weights ≈ reference.log_weights
    @test psis([0.0,-Inf,0,0];reff=1).status === :insufficient_tail
    @test psis(zeros(30);reff=1).ess_is ≈ 30
    @test psis(vcat(zeros(29),1.0);reff=1).status === :degenerate_tail
    shaped = reshape(repeat(ratios,4),100,2,2,2)
    @test psis(permutedims(shaped,(3,1,4,2));reff=ones(2,2),drawdim=2,chaindim=4).pareto_k ≈ fill(p.pareto_k,2,2)

    tail = pareto_tail(collect(1.0:200))
    @test tail.k_left ≈ tail.k_right
    @test pareto_tail(collect(1.0:200) .* 3 .+ 10).k ≈ tail.k
    @test pareto_tail(ones(30)).status == (left=:constant,right=:constant)
    heuristic = pareto_smoothed_minimum(0.5f0)
    @test heuristic.minimum == 100f0 && heuristic.estimator === :pareto_smoothed_expectation
end

@testset "Log offsets and dependent weighted precision" begin
    @test importance_diagnostics(Int64[0,1,2,3] .+ 2^60).ess_is ≈ importance_diagnostics([0.,1,2,3]).ess_is
    @test psis(collect(Int64,1:200) .+ 2^60;reff=1).log_weights ≈ psis(collect(1.:200);reff=1).log_weights
    values = hcat(sin.(1.:80),cos.(1.:80))
    logweights = values/4
    counts = fill(2,size(values))
    weights = exp.(logweights)
    average = sum(weights.*values)/sum(weights)
    influence = length(values).*weights.*(values.-average)/sum(weights)
    expected = mcse(influence;counts,split=false)
    @test importance_mcse(values,logweights;sampling=:mcmc,counts) ≈ expected
    @test importance_mcse(repeat(values;inner=(2,1)),repeat(logweights;inner=(2,1));sampling=:mcmc) ≈ expected
end
