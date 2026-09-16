@testset "Predictive scores" begin
    likelihood = reshape([1.0,2,3,4,2,2,4,4],4,1,2)
    logs = log.(likelihood)
    w = waic(logs)
    penalty = [var(log.(1.0:4)), log(2)^2/3]
    score = log.([5/2,3])-penalty
    @test w.pointwise.elpd ≈ score
    @test w.elpd ≈ sum(score) && w.p_waic ≈ sum(penalty)
    @test w.se_elpd ≈ abs(score[1]-score[2])
    @test waic(permutedims(logs,(3,1,2));drawdim=2,chaindim=3).elpd ≈ w.elpd
    @test waic(Float32.(logs)).elpd ≈ Float32(w.elpd)
    @test_throws ArgumentError waic(vec(logs)) # A joint log density has no observation axis.

    l = loo(logs;reff=1,sampling=:iid)
    @test l.pointwise.elpd ≈ log.([48/25,8/3]) # unsmoothed harmonic means for small tails
    @test isnan(l.mcse_elpd) && all(.!l.pointwise.reliable)
    repeated = repeat(reshape(-log.(1.0:200),200,1,1),1,1,2)
    full = loo(repeated;reff=1,sampling=:iid)
    @test full.mcse_elpd ≈ 2full.pointwise.mcse_elpd[1]
    @test full.se_elpd == 0 && full.mcse_elpd > 0
    auto = loo(repeated)
    @test auto.pointwise.reff[1] ≈ ess(exp.(repeated[:,1,1]);kind=:mean,split=false)/200

    comparison = compare_elpd([1.0,2,4],[0.0,1,1];observation_ids_a=1:3,observation_ids_b=1:3)
    @test comparison.elpd_difference == 5 && comparison.se_difference ≈ 2
    cdf = reshape([0.1,0.2,0.3,0.4,0.3,0.4,0.5,0.6],4,1,2)
    @test loo_pit(cdf,zeros(4,1,2)) ≈ [0.25,0.45]
    @test loo_pit(cdf,log.(likelihood)) ≈ [0.3,29/60]
end

@testset "Predictive log-scale boundaries" begin
    raw = reshape(Int64[0,1,2,3],4,1,1)
    shifted = raw .+ 2^60
    @test waic(shifted).p_waic ≈ 5/3
    @test loo(shifted;reff=1,sampling=:iid).p_loo ≈ loo(raw;reff=1,sampling=:iid).p_loo
    @test compare_elpd(UInt64[0],UInt64[1];observation_ids_a=[1],observation_ids_b=[1]).elpd_difference == -1
    @test compare_elpd([typemax(Int64)],[typemin(Int64)];observation_ids_a=[1],observation_ids_b=[1]).elpd_difference ≈ 2.0^64
    @test loo(reshape([-1e308,1e308],2,1,1);reff=1,sampling=:iid).elpd == -1e308
    @test loo(reshape(Float32[-3e38,3e38],2,1,1);reff=1,sampling=:iid).elpd == -3f38
    moderate = reshape(vcat(-log.(1.:31),fill(700.,169)),200,1,1)
    extreme = reshape(vcat(-log.(1.:31),fill(1e308,169)),200,1,1)
    baseline = loo(moderate;reff=1,sampling=:iid)
    candidate = loo(extreme;reff=1,sampling=:iid)
    @test candidate.mcse_elpd ≈ baseline.mcse_elpd
    @test candidate.p_loo ≈ baseline.p_loo
end
