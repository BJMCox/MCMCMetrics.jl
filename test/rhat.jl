function reference_basic_rhat(x)
    n, m = size(x)
    means = vec(mean(x; dims=1))
    within = mean(var(x; dims=1, corrected=true))
    sqrt((n - 1) // n + var(means; corrected=true) / within)
end

function reference_modern_rhat(x)
    n = size(x, 1) ÷ 2
    retained = BigFloat.(hcat(x[1:n, :], x[end-n+1:end, :]))
    function rank_scores(y)
        # Direct midrank definition, independent of the compressed tie traversal.
        N = length(y)
        map(y) do v
            rank = count(w -> w < v, y) + (BigFloat(count(==(v), y)) + 1) / 2
            probability = (rank - BigFloat(3) / 8) / (N + BigFloat(1) / 4)
            lo, hi = BigFloat(-16), BigFloat(16)
            for _ in 1:precision(BigFloat)+8
                mid = (lo + hi) / 2
                cdf = SpecialFunctions.erfc(-mid / sqrt(BigFloat(2))) / 2
                cdf < probability ? (lo = mid) : (hi = mid)
            end
            (lo + hi) / 2
        end
    end
    max(reference_basic_rhat(rank_scores(retained)),
        reference_basic_rhat(rank_scores(abs.(retained .- median(vec(retained))))))
end

@testset "R-hat definitions and precision" begin
    x = [0.0 2.0; 1.0 0.0; 3.0 4.0; 2.0 3.0; 4.0 -1.0; -1.0 5.0; 5.0 1.0; 2.0 6.0]
    @test rhat(x; kind=:basic, split=false) ≈ Float64(reference_basic_rhat(BigFloat.(x))) rtol=2e-15
    @test rhat(x) ≈ Float64(reference_modern_rhat(x)) rtol=2e-15
    result32 = rhat(Float32.(x))
    @test result32 isa Float32 && isapprox(result32, Float32(reference_modern_rhat(x)); rtol=4eps(Float32))

    integers = Int64.(x) .+ Int64(2)^60
    @test rhat(integers) ≈ rhat(x) rtol=2e-15
    @test rhat(integers; kind=:basic, split=false) ≈ rhat(x; kind=:basic, split=false) rtol=2e-15
    local_values = Int64[0, 1, 4, 2, 5, 1, 0, 3]
    separated = hcat(local_values, local_values .+ (Int64(1) << 60))
    @test rhat(separated; kind=:basic, split=false) ≈ Float64(reference_basic_rhat(BigFloat.(separated))) rtol=3e-15

    scales = [-1.5 1.0; 0.5 0.75; 1.5 -1.25; 0.75 1.5; 1.0 1.25; 1.25 0.5; 1.75 1.75; 0.25 1.0]
    extreme = scales .* ldexp(1.0, 1023)
    @test rhat(extreme) ≈ rhat(scales) rtol=2e-15
    @test rhat(extreme; kind=:basic, split=false) ≈ rhat(scales; kind=:basic, split=false) rtol=2e-15
    @test rhat((x .+ 1) .* nextfloat(0.0)) ≈ rhat(x) rtol=2e-15
    ties = [3.0 0.0; 4.0 2.0; 2.0 5.0; 2.0 4.0; 2.0 4.0; 2.0 0.0; 3.0 0.0; 5.0 3.0]
    expected = reference_modern_rhat(ties)
    @test rhat(ties .* nextfloat(0.0)) ≈ Float64(expected) rtol=2e-15
    @test rhat(Float32.(ties) .* nextfloat(0.0f0)) ≈ Float32(expected) rtol=8eps(Float32)
    mixed = ties .* nextfloat(0.0)
    mixed[3, 2] = mixed[8, 1] = floatmax(Float64)
    @test rhat(mixed) ≈ Float64(reference_modern_rhat(mixed)) rtol=2e-15
end

@testset "Compressed draws preserve logical splits" begin
    values = [[0.0, NaN, 1.0, 3.0, -1.0, 4.0], [2.0, 0.0, 4.0, 1.0, 5.0]]
    counts = [[2, 0, 2, 1, 2, 2], [1, 2, 2, 2, 2]]
    nested = [reshape(v, :, 1) for v in values]
    expanded = hcat([reduce(vcat, [fill(v, n) for (v, n) in zip(vs, ns)]) for (vs, ns) in zip(values, counts)]...)
    @test only(rhat(nested; counts)) ≈ Float64(reference_modern_rhat(expanded)) rtol=3e-15
    @test only(rhat(nested; counts, kind=:basic, split=false)) ≈
        Float64(reference_basic_rhat(BigFloat.(expanded))) rtol=3e-15

    retained = expanded[[1:4; 6:9], :]
    @test rhat(expanded) ≈ rhat(retained) rtol=3e-15

    huge = Int(10)^17
    run_values = [0.0 1.0; 1.0 0.0; 3.0 4.0; 2.0 2.0]
    run_counts = [huge huge; 1 1; huge huge; 2 2]
    value = rhat(run_values; counts=run_counts, split=false)
    @test isfinite(value)
    @test rhat(Float32.(run_values); counts=run_counts, split=false) ≈ Float32(value) rtol=8eps(Float32)
end

@testset "Unavailable R-hat" begin
    @test isnan(rhat([1.0 2.0; 1.0 2.0; 1.0 3.0; 1.0 4.0]))
    @test isnan(rhat([0.0 2.0; 0.0 3.0; 1.0 4.0; 2.0 5.0]))
    @test isnan(rhat([0.0 1.0; 1.0 2.0; 2.0 3.0]))
end
