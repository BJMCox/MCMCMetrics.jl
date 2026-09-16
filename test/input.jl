@testset "Array layouts preserve parameters" begin
    base = [0.0 2.0; 1.0 0.0; 3.0 4.0; 2.0 3.0; 4.0 -1.0; -1.0 5.0; 5.0 1.0; 2.0 6.0]
    x = Array{Float64}(undef, 8, 2, 2, 2)
    for a in 1:2, b in 1:2
        x[:, :, a, b] = base .+ [0.0 a + 2b]
    end
    expected = [rhat(view(x, :, :, a, b)) for a in 1:2, b in 1:2]
    @test rhat(x) ≈ expected
    @test rhat(PermutedDimsArray(x, (3, 1, 4, 2)); drawdim=2, chaindim=4) ≈ expected
    @test rhat(view(x, :, :, :, 1)) ≈ expected[:, 1]

    chains = [reshape(view(x, :, c, :, :), 8, 4) for c in 1:2]
    @test rhat(chains) ≈ vec(expected)
    @test rhat([transpose(c) for c in chains]; drawdim=2) ≈ vec(expected)
    packed = cat(chains...; dims=3)
    @test rhat(ArraysOfArrays.sliced(packed, Val(2))) ≈ vec(expected)
end
