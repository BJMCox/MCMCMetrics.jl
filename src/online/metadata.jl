const _ONLINE_PARAMETER_METRICS = (:mean, :variance, :rhat_basic, :mcse_mean,
    :ess_mean, :iact_mean, :indicator, :autocor)
const _ONLINE_AUXILIARY_METRICS = (:bfmi, :divergence, :acceptance, :tree_depth,
    :leapfrog_steps, :importance, :waic, :jump_distance, :mc_cov, :ess_multivariate)
const _ONLINE_METADATA_FIELDS = (bfmi=:energy, divergence=:divergent, acceptance=:acceptance,
    tree_depth=:tree_depth, leapfrog_steps=:leapfrog_steps, importance=:logweight, waic=:loglikelihood)

function _online_auxiliary(::Type{T}, metrics, moments; max_tree_depth, observation_shape, metric) where {T}
    np, nc = size(moments)
    if :tree_depth in metrics
        max_tree_depth isa Integer && 0 <= max_tree_depth <= typemax(Int) ||
            throw(ArgumentError("tree_depth requires a nonnegative Int max_tree_depth"))
    else
        isnothing(max_tree_depth) || throw(ArgumentError("max_tree_depth requires tree_depth"))
    end
    if :waic in metrics
        observation_shape isa Tuple && !isempty(observation_shape) &&
            all(d -> d isa Integer && d > 0, observation_shape) ||
            throw(ArgumentError("waic requires a fixed nonempty observation_shape"))
    else
        isnothing(observation_shape) || throw(ArgumentError("observation_shape requires waic"))
    end
    transform = if isnothing(metric)
        nothing
    else
        :jump_distance in metrics || throw(ArgumentError("metric requires jump_distance"))
        metric isa AbstractMatrix{T} && size(metric) == (np,np) &&
            all(isfinite,metric) && LinearAlgebra.issymmetric(metric) ||
            throw(ArgumentError("metric must be symmetric, finite, and use the state's type"))
        Matrix(LinearAlgebra.cholesky(LinearAlgebra.Symmetric(copy(metric))).U)
    end
    energy = :bfmi in metrics ? (moments=_online_moments(T,(1,nc)), previous=zeros(T,nc),
        scale=zeros(T,nc), root=zeros(T,nc)) : nothing
    divergence = :divergence in metrics ? zeros(Int,nc) : nothing
    acceptance = :acceptance in metrics ? _online_moments(T,(1,nc);second=false) : nothing
    depth = :tree_depth in metrics ? (limit=Int(max_tree_depth), saturated=zeros(Int,nc), maximum=zeros(Int,nc)) : nothing
    work = :leapfrog_steps in metrics ? zeros(Int,nc) : nothing
    weights = :importance in metrics ? _online_weights(T,1,nc) : nothing
    predictive = if :waic in metrics
        shape = map(Int, observation_shape)
        no = foldl(Base.checked_mul,shape;init=1)
        (shape=shape, moments=_online_moments(T,(no,nc)), weights=_online_weights(T,no,nc))
    else
        nothing
    end
    jumps = :jump_distance in metrics ? (previous=zeros(T,np,nc), root=zeros(T,nc),
        delta=zeros(T,np,nc), doubled=fill(false,np,nc), transform=transform) : nothing
    covariance = any(m -> m in (:mc_cov,:ess_multivariate),metrics) ? _online_covariance(T,moments) : nothing
    (;energy,divergence,acceptance,depth,work,weights,predictive,jumps,covariance)
end

@inline function _online_metadata_keys(state::OnlineDiagnostics{T,M}, metadata) where {T,M}
    metadata isa NamedTuple || throw(ArgumentError("metadata must be a NamedTuple"))
    expected = map(m -> _ONLINE_METADATA_FIELDS[m],filter(m -> haskey(_ONLINE_METADATA_FIELDS,m),M))
    length(keys(metadata)) == length(expected) && all(k -> k in expected,keys(metadata)) ||
        throw(ArgumentError("metadata must supply exactly the configured fields"))
    nothing
end

@inline function _online_metadata_validate(state::OnlineDiagnostics{T}, metadata, count) where {T}
    _online_metadata_keys(state,metadata)
    map(keys(metadata)) do name
        value = metadata[name]
        if name === :loglikelihood
            value isa AbstractArray{T} && size(value) == state.blocks.predictive.shape ||
                throw(ArgumentError("loglikelihood must have the fixed observation shape and state type"))
            count == 0 || all(isfinite,value) || throw(ArgumentError("loglikelihood must be finite"))
        elseif name === :divergent
            value isa Bool || throw(ArgumentError("divergent must be Bool"))
        elseif name in (:tree_depth,:leapfrog_steps)
            value isa Integer || throw(ArgumentError("depth and work must be integers"))
            count == 0 || 0 <= value <= typemax(Int) || throw(ArgumentError("depth and work must fit nonnegative Int"))
        else
            value isa T || throw(ArgumentError("floating metadata must use the state's type"))
            if count > 0
                (isfinite(value) || name === :logweight && value == -T(Inf)) ||
                    throw(ArgumentError("metadata must be finite, except zero-mass logweight"))
                name !== :acceptance || zero(T) <= value <= one(T) || throw(ArgumentError("acceptance must lie in [0,1]"))
            end
        end
    end
    nothing
end

# Metadata axes are independent of parameter draw axes. Normalize only views.
function _online_metadata_chunk(state::OnlineDiagnostics{T}, metadata, draws, single, nested) where {T}
    _online_metadata_keys(state,metadata)
    isempty(metadata) && return nothing
    nc = length(draws.chains)
    supplied = map(1:nc) do c
        map(keys(metadata)) do name
            value = metadata[name]
            tail = name === :loglikelihood ? state.blocks.predictive.shape : ()
            ni = size(draws.chains[c],1)
            if nested
                value isa AbstractVector{<:AbstractArray} && length(value) == nc ||
                    throw(DimensionMismatch("nested metadata requires one array per chain"))
                size(value[c]) == (ni,tail...) || throw(DimensionMismatch("metadata stored-draw axes differ"))
                return value[c]
            elseif single
                value isa AbstractArray && size(value) == (ni,tail...) ||
                    throw(DimensionMismatch("single-chain metadata must be stored draws by observations"))
                return value
            else
                value isa AbstractArray && size(value) == (ni,nc,tail...) ||
                    throw(DimensionMismatch("metadata must be stored draws by chains by observations"))
                return selectdim(value,2,c)
            end
        end |> values -> NamedTuple{keys(metadata)}(values)
    end
    for c in 1:nc
        for (name,values) in pairs(supplied[c])
            valid = name === :divergent ? eltype(values) <: Bool :
                name in (:tree_depth,:leapfrog_steps) ? eltype(values) <: Integer : eltype(values) === T
            valid || throw(ArgumentError("metadata array element types must match the configured fields"))
        end
        for i in axes(draws.chains[c],1)
            _online_metadata_validate(state,_online_metadata_row(supplied[c],i),_count(draws,c,i))
        end
    end
    supplied
end

function _online_metadata_row(metadata, i)
    NamedTuple{keys(metadata)}(map(keys(metadata)) do name
        name === :loglikelihood ? selectdim(metadata[name],1,i) : metadata[name][i]
    end)
end
_online_metadata_row(::Nothing,i) = (;)
_online_metadata_chain(::Nothing,c) = nothing
_online_metadata_chain(metadata,c) = metadata[c]

_online_work_check(::Nothing, metadata, c, k) = nothing
_online_work_check(work, metadata, c, k) = Base.checked_add(work[c],Base.checked_mul(Int(metadata.leapfrog_steps),k))
_online_work_check_chunk(::Nothing, metadata, draws, j, c) = nothing
function _online_work_check_chunk(work, metadata, draws, j, c)
    total = work[c]
    for i in axes(draws.chains[j],1)
        k = _count(draws,j,i)
        k == 0 && continue
        total = Base.checked_add(total,Base.checked_mul(Int(metadata.leapfrog_steps[i]),k))
    end
    total
end

function _online_energy!(block, x::T, c, n, count) where {T}
    if n > 0
        scale = _online_scale(block.previous[c],x,block.scale[c])
        if !iszero(scale)
            root = block.root[c]*(block.scale[c]/scale)
            block.root[c] = hypot(root,_online_center(x,block.previous[c],scale))
            block.scale[c] = scale
        end
    end
    block.previous[c] = x
    _online_add!(block.moments,1,c,x,count)
end

function _online_jumps!(block, draw, c, n)
    T = eltype(block.previous)
    if n > 0
        for (p,x) in enumerate(draw)
            delta = x-block.previous[p,c]
            block.doubled[p,c] = !isfinite(delta)
            block.delta[p,c] = isfinite(delta) ? delta : x/T(2)-block.previous[p,c]/T(2)
        end
        root = zero(T)
        for p in axes(block.previous,1)
            value = if isnothing(block.transform)
                block.delta[p,c]*(block.doubled[p,c] ? T(2) : one(T))
            else
                sum(j -> (block.transform[p,j]*block.delta[j,c])*
                    (block.doubled[j,c] ? T(2) : one(T)),axes(block.previous,1))
            end
            root = hypot(root,value)
        end
        block.root[c] = hypot(block.root[c],root)
    end
    for (p,x) in enumerate(draw)
        block.previous[p,c] = x
    end
end

@inline function _online_auxiliary_update!(b, draw, c, n, k, metadata)
    b.energy === nothing || _online_energy!(b.energy,metadata.energy,c,n,k)
    b.divergence === nothing || (b.divergence[c] += metadata.divergent ? k : 0)
    b.acceptance === nothing || _online_add!(b.acceptance,1,c,metadata.acceptance,k)
    if b.depth !== nothing
        b.depth.saturated[c] += metadata.tree_depth >= b.depth.limit ? k : 0
        b.depth.maximum[c] = max(b.depth.maximum[c],Int(metadata.tree_depth))
    end
    b.work === nothing || (b.work[c] += Int(metadata.leapfrog_steps)*k)
    b.weights === nothing || _online_weight!(b.weights,1,c,metadata.logweight,k)
    if b.predictive !== nothing
        for (p,x) in enumerate(metadata.loglikelihood)
            _online_add!(b.predictive.moments,p,c,x,k)
            _online_weight!(b.predictive.weights,p,c,x,k)
        end
    end
    b.jumps === nothing || _online_jumps!(b.jumps,draw,c,n)
    b.covariance === nothing || _online_covariance!(b.covariance,draw,c,n,k)
    nothing
end

function _online_auxiliary_clear!(b)
    if b.energy !== nothing
        _online_clear!(b.energy.moments)
        fill!(b.energy.previous,0); fill!(b.energy.scale,0); fill!(b.energy.root,0)
    end
    b.divergence === nothing || fill!(b.divergence,0)
    b.acceptance === nothing || _online_clear!(b.acceptance)
    if b.depth !== nothing
        fill!(b.depth.saturated,0); fill!(b.depth.maximum,0)
    end
    b.work === nothing || fill!(b.work,0)
    b.weights === nothing || _online_clear_weights!(b.weights)
    if b.predictive !== nothing
        _online_clear!(b.predictive.moments); _online_clear_weights!(b.predictive.weights)
    end
    if b.jumps !== nothing
        fill!(b.jumps.previous,0); fill!(b.jumps.root,0)
        fill!(b.jumps.delta,0); fill!(b.jumps.doubled,false)
    end
    b.covariance === nothing || _online_clear_covariance!(b.covariance)
end

# Copy only selected family state while holding all chain locks.
function _online_family_snapshot(state, names)
    _online_locked(state,eachindex(state.locks)) do
        (counts=copy(state.moments.n[1,:]),
            blocks=NamedTuple{names}(map(name -> deepcopy(state.blocks[name]),names)))
    end
end

function _online_bfmi(b, c, n, ::Type{T}) where {T}
    n < 2 && return (value=T(NaN),status=:insufficient_draws)
    m2 = b.moments.m2[1,c]+b.moments.m2_error[1,c]
    m2 <= zero(T) && return (value=T(NaN),status=:constant)
    ratio = b.scale[c]/b.moments.scale[1,c]
    value = abs2(b.root[c]*ratio)/m2
    if !isfinite(value)
        value = exp(T(2)*(log(b.root[c])+log(b.scale[c])-log(b.moments.scale[1,c]))-log(m2))
    end
    (value=value,status=isfinite(value) ? :ok : :nonfinite_result)
end

"""Return per-chain E-BFMI from configured full Hamiltonian-energy metadata."""
function bfmi(state::OnlineDiagnostics{T,M}) where {T,M}
    :bfmi in M || throw(ArgumentError("state does not track bfmi"))
    s = _online_family_snapshot(state,(:energy,))
    [_online_bfmi(s.blocks.energy,c,n,T).value for (c,n) in enumerate(s.counts)]
end

"""
    sampler_diagnostics(state::OnlineDiagnostics)

Owned per-chain summaries of configured energy, divergence, acceptance, depth,
and leapfrog work. Denominators use actual logical metadata counts. Counts repeat
each metadata value; per-proposal totals must not be attached to a repeated run.
"""
function sampler_diagnostics(state::OnlineDiagnostics{T,M}) where {T,M}
    any(m -> m in (:bfmi,:divergence,:acceptance,:tree_depth,:leapfrog_steps),M) ||
        throw(ArgumentError("state has no sampler metadata"))
    s = _online_family_snapshot(state,(:energy,:divergence,:acceptance,:depth,:work))
    b = s.blocks
    StructArray(map(enumerate(s.counts)) do (c,n)
        row = (chain=c,n_draws=n)
        if b.energy !== nothing
            result = _online_bfmi(b.energy,c,n,T)
            row = merge(row,(bfmi=result.value,bfmi_status=result.status))
        end
        b.divergence === nothing || (row = merge(row,(divergences=b.divergence[c],
            divergence_rate=n == 0 ? T(NaN) : T(b.divergence[c])/T(n))))
        if b.acceptance !== nothing
            a = b.acceptance
            row = merge(row,(acceptance_mean=n == 0 ? T(NaN) : muladd(a.scale[1,c],a.mean[1,c],a.origin[1,c]),))
        end
        b.depth === nothing || (row = merge(row,(depth_saturations=b.depth.saturated[c],
            depth_saturation_rate=n == 0 ? T(NaN) : T(b.depth.saturated[c])/T(n),depth_maximum=b.depth.maximum[c])))
        b.work === nothing || (row = merge(row,(leapfrog_total=b.work[c],leapfrog_mean=n == 0 ? T(NaN) : T(b.work[c])/T(n))))
        row
    end)
end

"""Owned per-chain jump summaries using the constructor-fixed SPD metric, if supplied."""
function squared_jump_distance(state::OnlineDiagnostics{T,M}) where {T,M}
    :jump_distance in M || throw(ArgumentError("state does not track jump_distance"))
    s = _online_family_snapshot(state,(:jumps,))
    StructArray(map(enumerate(s.counts)) do (c,n)
        transitions = max(0,n-1)
        root = s.blocks.jumps.root[c]
        mean = transitions == 0 ? T(NaN) : abs2(root/sqrt(T(transitions)))
        (mean_squared_jump=mean,total_squared_jump=abs2(root),n_transitions=transitions,
            status=transitions == 0 ? :insufficient_draws : isfinite(mean) ? :ok : :overflow)
    end)
end
