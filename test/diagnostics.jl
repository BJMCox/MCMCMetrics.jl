@testset "Batch diagnostic status and shape" begin
    x = Float32[0 3; 1 0; 4 1; 2 4; 5 2; 1 5; 0 1; 3 2; 4 0; 2 4; 1 3; 5 2]
    result = only(diagnostics(x))
    @test result.index == CartesianIndex() && result.n_draws == 24 && result.n_chains == 2
    @test result.rhat ≈ rhat(x)
    @test result.ess_bulk ≈ ess(x)
    @test result.mcse_mean ≈ mcse(x)
    @test result.status == (rhat=:ok, ess_bulk=:ok, ess_tail=:constant, mcse_mean=:ok)
    @test isnan(result.ess_tail) && result.mcse_mean isa Float32

    single = only(diagnostics(view(x, :, 1)))
    @test isnan(single.rhat) && single.status.rhat === :insufficient_chains
    @test isfinite(single.mcse_mean) && single.status.mcse_mean === :ok

    short = only(diagnostics(x[1:6, :]))
    @test isnan(short.ess_bulk) && short.status.ess_bulk === :insufficient_draws
    constant = only(diagnostics(ones(12, 2)))
    @test all(isnan, (constant.rhat, constant.ess_bulk, constant.ess_tail, constant.mcse_mean))
    @test all(==(:constant), constant.status)
    failed = only(diagnostics(repeat([1., -1.], 4)))
    @test isnan(failed.mcse_mean) && failed.status.mcse_mean === :estimator_failed

    tensor = reshape(cat(x, x .+ 1, x .+ 2, x .+ 3; dims=3), 12, 2, 2, 2)
    @test diagnostics(tensor).index == vec(collect(CartesianIndices((2, 2))))
    ragged = [reshape(x[:, 1], :, 1), reshape(x[1:8, 2], :, 1)]
    unequal = only(diagnostics(ragged))
    @test all(==(:unequal_chain_lengths), unequal.status)
    @test iact(ragged) ≈ reshape([iact(x[:, 1]), iact(x[1:8, 2])], 2, 1)
end

@testset "Compressed aggregate keeps available metrics" begin
    runs = [0. 3; 1 0; 4 1; 2 4; 5 2; 1 5]
    counts = fill(3, size(runs))
    result = only(diagnostics(runs; counts))
    expanded = repeat(runs; inner=(3, 1))
    @test result.rhat ≈ rhat(expanded)
    @test result.n_draws == length(expanded) && result.status.rhat === :ok
    @test result.ess_bulk ≈ ess(expanded)
    @test result.mcse_mean ≈ mcse(expanded)
    limited = only(diagnostics(runs;counts,max_lag_work=1))
    @test limited.rhat ≈ result.rhat && limited.status.ess_bulk === :work_limit && limited.status.mcse_mean === :work_limit
end
