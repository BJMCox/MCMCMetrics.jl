@testset "Sampler metadata and ranks" begin
    energy, repeats = [1.0,3,2], [2,1,3]
    expanded = [1.0,1,3,2,2,2]
    @test bfmi(energy; counts=repeats) ≈ 30/17
    @test bfmi(expanded) ≈ bfmi(energy; counts=repeats)
    @test bfmi(Float32.(expanded) .* 8 .+ 32) ≈ Float32(30/17)
    @test bfmi([expanded expanded .+ 20]) ≈ fill(30/17,2)
    summary = sampler_diagnostics(energy=[energy energy], counts=hcat(repeats,repeats),
        divergent=Bool[0 1;1 0;0 0], acceptance=[0.1 0.2;0.3 0.4;0.7 0.8],
        tree_depth=[1 2;3 2;4 1], max_tree_depth=3, leapfrog_steps=[1 2;7 3;15 1])
    @test summary.divergences == [1,2]
    @test summary.acceptance_mean ≈ [2.6/6,3.2/6]
    @test summary.depth_saturations == [4,0]
    @test summary.leapfrog_total == [54,10]
    @test summary.bfmi ≈ fill(30/17,2)
    runs = [[1.0,NaN,3,2], [2.0,4,3]]
    multiplicities = [[2,0,1,3], [1,2,1]]
    nested = [reshape(v,:,1) for v in runs]
    literal = [[1.0,1,3,2,2,2], [2.0,4,4,3]]
    @test bfmi(nested;counts=multiplicities) ≈ bfmi.(literal)
    flags = [reshape(Bool[0,1,1,0],:,1), reshape(Bool[1,0,1],:,1)]
    depths = [reshape([1,-9,3,4],:,1), reshape([2,3,1],:,1)]
    nested_summary = sampler_diagnostics(energy=permutedims.(nested), divergent=permutedims.(flags),
        tree_depth=permutedims.(depths), leapfrog_steps=permutedims.(depths),
        max_tree_depth=3, counts=multiplicities, drawdim=2)
    @test nested_summary.bfmi ≈ bfmi.(literal)
    @test nested_summary.n_draws == [6,4] && nested_summary.divergences == [1,2]
    @test nested_summary.depth_saturations == [4,2] && nested_summary.depth_maximum == [4,3]
    @test nested_summary.leapfrog_total == [17,9]

    x, counts = [1.0 2;2 3], [2 1;1 2]
    hist = rank_histogram(x; counts, bins=2)
    @test hist.counts == [2 0;1 3]
    @test vec(sum(hist.frequency;dims=1)) ≈ ones(2)
    dense = [1.0 2;1 3;2 3]
    @test rank_histogram(dense;bins=2).counts == hist.counts
    @test rank_ecdf(x;counts,grid=[0,0.5,1]).ecdf == [0 0;1 1/3;1 1]
    @test rank_ecdf(dense;grid=[0,0.5,1]).ecdf == rank_ecdf(x;counts,grid=[0,0.5,1]).ecdf
    @test rank_histogram(permutedims(x);drawdim=2,chaindim=1,counts,bins=2).counts == hist.counts
    for (T, mass) in ((Float32,2^25), (Float64,2^55))
        huge = [mass,1,mass]
        @test rank_histogram(T[0,1,2];counts=huge,bins=4).counts == [mass,0,1,mass]
        @test rank_ecdf(T[0,1,2];counts=huge,grid=T[.25,.75]).ecdf ==
            T.([big(mass)//(2big(mass)+1), (big(mass)+1)//(2big(mass)+1)])
    end
end
