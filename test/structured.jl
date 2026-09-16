@testset "Grouped and categorical chains" begin
    x = hcat(([-1.,1,-1,1] .+ c for c in (0,2,4,6))...)
    @test rhat_nested(x,[1,1,2,2];kind=:basic,split=false) ≈ sqrt(17/5)
    @test rhat_nested(reshape([0.,2,4,6],1,4),[:a,:a,:b,:b];kind=:basic,split=false) ≈ sqrt(5)
    stored = x[1:2,:]
    @test rhat_nested(stored,[1,1,2,2];counts=fill(2,2,4),kind=:basic,split=false) ≈ sqrt(17/5)
    shaped = reshape(Float32.(x),4,4,1,1)
    @test only(rhat_nested(permutedims(shaped,(3,2,4,1)),[1,1,2,2];drawdim=4,chaindim=2,kind=:basic,split=false)) ≈ Float32(sqrt(17/5))
    states = [0 1; 0 1; 0 1; 1 0; 1 0; 1 0; 0 1; 0 1]
    summary = categorical_summary(states;levels=[0,1,2])
    @test summary.occupation == [5 3;3 5;0 0]
    @test summary.transitions[:,:,1] == [3 1 0;1 2 0;0 0 0]
    compressed = categorical_summary([0 1;1 0;0 1];counts=[3 3;3 3;2 2],levels=[0,1,2])
    @test compressed.transitions == summary.transitions
    @test compressed.probability == summary.probability
    comparison = categorical_weiss(states)
    @test comparison.statistic ≈ 25/87
    @test comparison.pvalue ≈ SpecialFunctions.erfc(sqrt(25/174))
    @test categorical_weiss(100 .- 7states) == comparison
    alternating = repeat([0 1;1 0],4)
    @test categorical_weiss(alternating).status === :outside_dar1
end

@testset "Movement and tempering" begin
    x = Float32[0 0;3 4;3 5]
    movement = squared_jump_distance(x;chaindim=nothing,counts=[2,1,3],cost=10)
    @test movement.mean_squared_jump ≈ 26f0/5
    @test movement.squared_jump_per_cost ≈ 26f0/10
    expanded = x[[1,1,2,3,3,3],:]
    @test squared_jump_distance(expanded;chaindim=nothing).mean_squared_jump ≈ movement.mean_squared_jump
    @test squared_jump_distance(x;chaindim=nothing,metric=Float32[4 0;0 1]).mean_squared_jump ≈ 53f0/2
    metric = reshape(Float32[1f-40],1,1)
    @test squared_jump_distance(Float32[-3f38,3f38];metric).mean_squared_jump ≈
        squared_jump_distance(Float32[-3f38,3f38]/1f20;metric=metric*1f20*1f20).mean_squared_jump
    paths = [3 1;1 2;2 3;3 2;2 1;1 2;1 3;3 2;1 3]
    summary = tempering_summary(paths;levels=[1,2,3])
    @test summary.replicas.round_trips == [2,1]
    @test summary.replicas.durations == [[4,3],[4]]
    @test summary.replicas.incomplete_trip == [false,true]
    @test vec(sum(summary.occupation;dims=1)) == [9,9]
    compressed = tempering_summary([1,3,1];levels=[1,2,3],counts=[3,2,4])
    @test only(compressed.replicas.durations) == [5]
end
