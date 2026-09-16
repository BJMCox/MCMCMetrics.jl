using LinearAlgebra

@testset "Online auxiliary metadata and lifecycle" begin
    counts = [2,0,3,1]
    x = [1. 2.; NaN NaN; 2. 4.; 0. 1.]
    metadata = (energy=[2.,NaN,4.,1.],divergent=[false,true,true,false],
        acceptance=[.5,NaN,.9,.1],tree_depth=[1,-1,3,2],leapfrog_steps=[2,-1,8,4],
        logweight=[0.,NaN,-1.,-Inf],loglikelihood=[-1. -2.;NaN NaN;-2. -1.;-3. -4.])
    metrics = (:bfmi,:divergence,:acceptance,:tree_depth,:leapfrog_steps,:importance,:waic,:jump_distance)
    metric = [2. .3;.3 1.]
    state = OnlineDiagnostics(Float64;nchains=1,parameter_shape=(2,),metrics,
        max_tree_depth=3,observation_shape=(2,),metric)
    append!(state,x;chain=1,counts,metadata)
    reference = sampler_diagnostics(;energy=metadata.energy,divergent=metadata.divergent,
        acceptance=metadata.acceptance,tree_depth=metadata.tree_depth,
        leapfrog_steps=metadata.leapfrog_steps,max_tree_depth=3,counts,chaindim=nothing)
    actual = sampler_diagnostics(state)
    for field in propertynames(reference)
        field === :bfmi_status ? (@test getproperty(actual,field) == getproperty(reference,field)) :
            (@test getproperty(actual,field) ≈ getproperty(reference,field))
    end
    @test bfmi(state) ≈ reference.bfmi
    w = importance_diagnostics(metadata.logweight;counts,chaindim=nothing)
    @test importance_diagnostics(state).ess_is ≈ w.ess_is
    @test importance_diagnostics(state).max_weight ≈ w.max_weight
    @test importance_diagnostics(state).entropy ≈ w.entropy
    predictive = waic(metadata.loglikelihood;counts,chaindim=nothing)
    @test waic(state).pointwise.elpd ≈ predictive.pointwise.elpd
    @test waic(state).p_waic ≈ predictive.p_waic
    @test waic(state).se_elpd ≈ predictive.se_elpd
    jump = squared_jump_distance(x;counts,chaindim=nothing,metric)
    @test only(squared_jump_distance(state).mean_squared_jump) ≈ jump.mean_squared_jump
    @test only(squared_jump_distance(state).n_transitions) == 5
    @test diagnostics(state).n_draws == [6,6]
    replay = OnlineDiagnostics(Float64;nchains=1,parameter_shape=(2,),metrics,
        max_tree_depth=3,observation_shape=(2,),metric)
    for i in axes(x,1)
        row = NamedTuple{keys(metadata)}(map(k -> k === :loglikelihood ? metadata[k][i,:] : metadata[k][i],keys(metadata)))
        for _ in 1:counts[i]
            push!(replay,x[i,:];chain=1,metadata=row)
        end
    end
    @test bfmi(replay) ≈ bfmi(state)
    @test waic(replay).pointwise.elpd ≈ waic(state).pointwise.elpd
    @test importance_diagnostics(replay).entropy ≈ importance_diagnostics(state).entropy
    @test squared_jump_distance(replay).mean_squared_jump ≈ squared_jump_distance(state).mean_squared_jump

    # A bad late field and accumulated work overflow must consume no earlier row.
    previous = sampler_diagnostics(state)
    bad = merge(metadata,(acceptance=[.5,NaN,.9,2.],))
    @test_throws ArgumentError append!(state,x;chain=1,counts,metadata=bad)
    @test sampler_diagnostics(state) == previous
    overflow = merge(metadata,(leapfrog_steps=[2,0,8,typemax(Int)],))
    @test_throws OverflowError append!(state,x;chain=1,counts,metadata=overflow)
    @test sampler_diagnostics(state) == previous
    @test importance_diagnostics(state).ess_is ≈ w.ess_is
    @test waic(state).pointwise.elpd ≈ predictive.pointwise.elpd
    @test_throws ArgumentError push!(state,x[1,:];chain=1)
    @test sampler_diagnostics(state) == previous
    saved = waic(state)
    saved.pointwise.elpd[1] = 123.
    @test waic(state).pointwise.elpd ≈ predictive.pointwise.elpd
    @test_throws ArgumentError merge!(state,deepcopy(state))
    empty!(state)
    @test only(sampler_diagnostics(state).n_draws) == 0
    @test importance_diagnostics(state).status == :insufficient_draws
    append!(state,x;chain=1,counts,metadata)
    @test sampler_diagnostics(state) == previous
end

@testset "Online auxiliary independent concurrent chains" begin
    state = OnlineDiagnostics(Float64;nchains=16,parameter_shape=(2,),metrics=(:bfmi,:jump_distance,:waic),observation_shape=(2,))
    Threads.@threads for c in 1:16
        for i in 1:33
            push!(state,[sin(i),cos(i)];chain=c,metadata=(energy=Float64(i),loglikelihood=[-i/10.,-i/20.]))
        end
    end
    @test bfmi(state) ≈ fill(bfmi(collect(1.:33.)),16)
    jump = squared_jump_distance(hcat(sin.(1:33),cos.(1:33));chaindim=nothing)
    @test squared_jump_distance(state).mean_squared_jump ≈ fill(jump.mean_squared_jump,16)
    ll = repeat(reshape(hcat(-collect(1:33)./10,-collect(1:33)./20),33,1,2),1,16,1)
    @test waic(state).pointwise.elpd ≈ waic(ll).pointwise.elpd
end

@testset "Online metadata layouts and actual pooling" begin
    chains = [[1.;2.;4.;;],[5.;6.;;]]
    logweights = [[0.,-1.,-2.],[-Inf,1.]]
    likelihood = [[-1. -2.;-2. -3.;-4. -1.],[-3. -2.;-1. -3.]]
    counts = [[2,1,3],[1,2]]
    state = OnlineDiagnostics(Float64;nchains=2,metrics=(:importance,:waic),observation_shape=(2,))
    # Nested parameter matrices have a one-element parameter axis.
    nested = OnlineDiagnostics(Float64;nchains=2,parameter_shape=(1,),metrics=(:importance,:waic),observation_shape=(2,))
    append!(nested,chains;counts,metadata=(logweight=logweights,loglikelihood=likelihood))
    for c in 1:2
        append!(state,vec(chains[c]);chain=c,counts=counts[c],metadata=(logweight=logweights[c],loglikelihood=likelihood[c]))
    end
    oracle_w = importance_diagnostics([reshape(w,:,1) for w in logweights];counts)
    @test importance_diagnostics(state).ess_is ≈ only(oracle_w.ess_is)
    @test importance_diagnostics(nested) == importance_diagnostics(state)
    oracle_waic = waic(likelihood;counts)
    @test waic(state).pointwise.elpd ≈ oracle_waic.pointwise.elpd
    @test waic(nested).pointwise.elpd == waic(state).pointwise.elpd
    # Metadata retains draw-by-chain axes when parameters use different axes.
    canonical = OnlineDiagnostics(Float32;nchains=2,metrics=(:acceptance,))
    append!(canonical,Float32[1 2 3;4 5 6];drawdim=2,chaindim=1,
        metadata=(acceptance=Float32[.1 .4;.2 .5;.3 .6],))
    @test sampler_diagnostics(canonical).acceptance_mean ≈ Float32[.2,.5]
end

@testset "Online auxiliary compressed numerical boundaries" begin
    state = OnlineDiagnostics(Float32;nchains=1,metrics=(:importance,:bfmi,:jump_distance))
    push!(state,0f0;chain=1,count=2^30,metadata=(energy=1f0,logweight=-Inf32))
    @test importance_diagnostics(state).status == :zero_mass
    push!(state,1f0;chain=1,count=2^30,metadata=(energy=2f0,logweight=0f0))
    @test importance_diagnostics(state).ess_is ≈ Float32(2^30)
    @test importance_diagnostics(state).entropy ≈ log(Float32(2^30))
    @test only(squared_jump_distance(state).mean_squared_jump) ≈ inv(Float32(2^31-1))
    @test only(bfmi(state)) ≈ Float32(4/2^31)
    extreme = OnlineDiagnostics(Float64;nchains=1,metrics=(:bfmi,:jump_distance),metric=fill(1e-310,1,1))
    push!(extreme,-1e308;chain=1,metadata=(energy=-1e308,))
    push!(extreme,1e308;chain=1,metadata=(energy=1e308,))
    @test only(bfmi(extreme)) ≈ 2.
    @test only(squared_jump_distance(extreme).mean_squared_jump) ≈ 4e306
    uniform = OnlineDiagnostics(Float32;nchains=1,metrics=(:importance,))
    push!(uniform,0f0;chain=1,count=2^25,metadata=(logweight=0f0,))
    for _ in 1:32
        push!(uniform,0f0;chain=1,metadata=(logweight=0f0,))
    end
    @test importance_diagnostics(uniform).ess_is == Float32(2^25+32)
end

function auxiliary_vector_oracle(x)
    n,p = size(x)
    width = 1 << ((8sizeof(Int)-1-leading_zeros(n))÷2)
    a = div(n,width)
    z = BigFloat.(x)
    means = [sum(z[(i-1)*width+1:i*width,j])/width for i in 1:a,j in 1:p]
    mean_cov = BigFloat(width)/n*cov(means)
    (;mean_cov,ess=det(mean_cov) > 0 ? exp((logdet(Symmetric(cov(z)))-logdet(Symmetric(mean_cov)))/p) : BigFloat(NaN))
end

@testset "Covariance reconstruction across coordinate scales" begin
    for T in (Float32,Float64)
        scales = T[nextfloat(zero(T)), T === Float32 ? 1f15 : 1e150]
        x = repeat(T[0 0;1 1;2 0;3 2];inner=(4,1)).*permutedims(scales)
        oracle = T.(auxiliary_vector_oracle(x).mean_cov)
        for order in ([1,2],[2,1])
            state = OnlineDiagnostics(T;nchains=1,parameter_shape=(2,),metrics=(:mc_cov,))
            append!(state,x[:,order];chain=1)
            online = only(mc_cov(state).mean_cov)
            batch = mc_cov(x[:,order];chaindim=nothing,batch_size=4)
            @test online[1,2] ≈ oracle[order[1],order[2]] rtol=32eps(T)
            @test batch[1,2] ≈ oracle[order[1],order[2]] rtol=32eps(T)
            @test issymmetric(online) && issymmetric(batch)
        end
    end
end

@testset "Online vector dyadic covariance" begin
    x = hcat([sin(i)+i/9 for i in 1:65],[cos(i/2)-i/20 for i in 1:65])
    state = OnlineDiagnostics(Float64;nchains=1,parameter_shape=(2,),metrics=(:mc_cov,:ess_multivariate))
    for n in (3,4,5,15,16,17,63,64,65)
        empty!(state)
        append!(state,x[1:n,:];chain=1)
        oracle = auxiliary_vector_oracle(x[1:n,:])
        if n ∉ (4,5)
            @test only(mc_cov(state).mean_cov) ≈ oracle.mean_cov
            @test only(ess_multivariate(state).ess_multivariate) ≈ oracle.ess
        end
    end
    original = only(mc_cov(state).mean_cov)
    original_ess = only(ess_multivariate(state).ess_multivariate)
    transform = [2. .3;-.7 1.]
    transformed = x*transform' .+ [1e5 -2e5]
    empty!(state); append!(state,transformed;chain=1)
    @test only(mc_cov(state).mean_cov) ≈ transform*original*transform' rtol=1e-9
    @test only(ess_multivariate(state).ess_multivariate) ≈ original_ess rtol=1e-9
    scaled = x .* [1e-80 1e80]
    empty!(state); append!(state,scaled;chain=1)
    @test only(ess_multivariate(state).ess_multivariate) ≈ original_ess
    held = mc_cov(state)
    held.mean_cov[1][1,1] = 100.
    @test only(mc_cov(state).mean_cov)[1,1] != 100.
    empty!(state); append!(state,hcat(x[:,1],x[:,1]);chain=1)
    @test only(mc_cov(state).status) == :singular_estimator
end

@testset "Online vector compressed chronology and native residuals" begin
    x = Float32[0 2;1 0;3 4;-2 1;4 -1]
    counts = [3,19,1,11,31]
    state = OnlineDiagnostics(Float32;nchains=1,parameter_shape=(2,),metrics=(:mc_cov,:ess_multivariate))
    append!(state,x;chain=1,counts)
    expanded = reduce(vcat,[repeat(x[i:i,:],counts[i],1) for i in axes(x,1)])
    oracle = auxiliary_vector_oracle(expanded)
    @test only(mc_cov(state).mean_cov) ≈ Float32.(oracle.mean_cov) rtol=1e-5
    @test only(ess_multivariate(state).ess_multivariate) ≈ Float32(oracle.ess) rtol=1e-5
    empty!(state)
    for i in axes(x,1)
        push!(state,x[i,:];chain=1,count=1)
        push!(state,x[i,:];chain=1,count=counts[i]-1)
    end
    @test only(mc_cov(state).mean_cov) ≈ Float32.(oracle.mean_cov) rtol=1e-5
    offset = Float32(1e8) .+ Float32(8).*expanded
    empty!(state); append!(state,offset;chain=1)
    shifted_oracle = auxiliary_vector_oracle(offset)
    @test only(mc_cov(state).mean_cov) ≈ Float32.(shifted_oracle.mean_cov) rtol=1e-5
    empty!(state); append!(state,x;chain=1,counts=[2^40,2^40,2^40,2^40,2^40])
    @test only(mc_cov(state).precision).n_draws == 5*2^40
    @test all(isfinite,only(mc_cov(state).mean_cov))
end
