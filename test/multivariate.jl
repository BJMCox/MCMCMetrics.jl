@testset "Joint vector precision" begin
    stored = Float32[1 0;-1 0;0 1;0 -1]
    x = repeat(stored;inner=(2,1))
    expected = Float32[1/6 0;0 1/6]
    @test mc_cov(x;chaindim=nothing,batch_size=2) ≈ expected
    @test ess_multivariate(x;chaindim=nothing,batch_size=2) ≈ 24f0/7
    @test mc_cov(x;chaindim=nothing,batch_size=2,estimator=:overlapping) ≈ Float32[3/28 -1/84;-1/84 3/28]
    @test mc_cov(x;chaindim=nothing,batch_size=2,lugsail=(r=2,c=0.5)) ≈ Float32[11/42 0;0 11/42]
    transform = Float32[2 1;0 3]
    @test mc_cov(x*transform .+ 100;chaindim=nothing,batch_size=2) ≈ transform'*expected*transform
    @test ess_multivariate(x*transform .+ 100;chaindim=nothing,batch_size=2) ≈ 24f0/7
    @test mc_cov(stored;chaindim=nothing,counts=fill(2,4),batch_size=2) ≈ expected
    @test all(isnan,mc_cov(hcat(x[:,1],2x[:,1]);chaindim=nothing,batch_size=2))
    @test isnan(ess_multivariate(hcat(x[:,1],2x[:,1]);chaindim=nothing,batch_size=2))
    chains = cat(reshape(x,8,1,2),reshape(x .+ [1 0],8,1,2);dims=2)
    @test rhat_multivariate(chains;split=false) ≈ sqrt(35f0)/4
    @test rhat_multivariate(permutedims(chains,(3,1,2));drawdim=2,chaindim=3,split=false) ≈ sqrt(35f0)/4
    base = Int64[1 0;-1 0;0 1;0 -1]
    distant = cat(reshape(base,4,1,2),reshape(base .+ 10^18,4,1,2);dims=2)
    @test rhat_multivariate(distant;split=false) ≈ 1.5e18
end
