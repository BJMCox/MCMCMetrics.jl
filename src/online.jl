"""
    OnlineDiagnostics(T; nchains, parameter_shape=(), metrics=(:mean, :variance, :rhat_basic),
                      threshold=nothing, max_lag=nothing, max_tree_depth=nothing,
                      observation_shape=nothing, metric=nothing, names=nothing)

Accumulate independent chains in `Float32` or `Float64` without retaining draws.
Real observations convert to `T`, as in a `Vector{T}`; values that overflow `T`
are rejected as nonfinite. Choose `metrics` at construction. The
[Online diagnostics](@ref) guide lists each metric, its keywords, storage and
assumptions. Optional distinct `names`, one per parameter in linear order, add a
`name` column to `diagnostics`.

`push!` takes one parameter-shaped draw and an explicit `chain`. `append!` takes
batch array axes and integer repetition `counts`. With `chain`, its layout is
draws × parameter dimensions. Without `chain`, the input holds every chain.
Malformed chunks are rejected before any draw is consumed.

One lock protects each chain. Callers preserve each chain's draw order, keep
inputs unchanged during a call, and finish producers before `empty!`. Ordered
and auxiliary metrics reject `merge!`. For a live display, `progress_summary`
reads `(n_draws, max_rhat, min_ess)` without copying the state.
"""
struct OnlineDiagnostics{T,Metrics,N,S,B,K}
    shape::NTuple{N,Int}
    names::K
    moments::S
    blocks::B
    locks::Vector{ReentrantLock}
end

function OnlineDiagnostics(::Type{T}; nchains::Integer, parameter_shape::Tuple=(),
    metrics::Tuple=(:mean, :variance, :rhat_basic), threshold=nothing, max_lag=nothing,
    max_tree_depth=nothing, observation_shape=nothing, metric=nothing, names=nothing) where {T}
    T in (Float32, Float64) || throw(ArgumentError("online states support Float32 and Float64"))
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    all(d -> d isa Integer && d > 0, parameter_shape) ||
        throw(ArgumentError("parameter dimensions must be positive integers"))
    !isempty(metrics) && all(m -> m in (_ONLINE_PARAMETER_METRICS..., _ONLINE_AUXILIARY_METRICS...), metrics) &&
        allunique(metrics) || throw(ArgumentError("metrics must be distinct supported choices"))
    shape = map(Int, parameter_shape)
    nparameters = foldl(Base.checked_mul, shape; init=1)
    dims = (nparameters, Int(nchains))
    moments = _online_moments(T, dims; second=any(m -> m ∉ (:mean, :indicator), metrics))
    precision = any(m -> m in (:mcse_mean, :ess_mean, :iact_mean), metrics)
    levels = precision ? _online_levels(T, dims...) : nothing
    indicator = if :indicator in metrics
        threshold === nothing && throw(ArgumentError("indicator requires a fixed threshold"))
        threshold isa Real || (threshold isa AbstractArray{<:Real} && size(threshold) == shape) ||
            throw(ArgumentError("threshold must be scalar or parameter-shaped"))
        thresholds = threshold isa Real ? fill(T(threshold), nparameters) : vec(T.(copy(threshold)))
        all(isfinite, thresholds) || throw(ArgumentError("threshold must be finite in the state's type"))
        (threshold=thresholds, moments=_online_moments(T, dims), levels=_online_levels(T, dims...))
    else
        threshold === nothing || throw(ArgumentError("threshold requires the indicator metric"))
        nothing
    end
    lags = if :autocor in metrics
        max_lag isa Integer && 0 <= max_lag < typemax(Int) ||
            throw(ArgumentError("autocor requires an explicit nonnegative max_lag"))
        _OnlineLags(T, dims..., Int(max_lag))
    else
        max_lag === nothing || throw(ArgumentError("max_lag requires the autocor metric"))
        nothing
    end
    if names !== nothing
        names = vec(collect(names))
        length(names) == nparameters && allunique(names) ||
            throw(ArgumentError("names must hold one distinct name per parameter"))
    end
    auxiliary = _online_auxiliary(T, metrics, moments; max_tree_depth, observation_shape, metric)
    blocks = merge((precision=levels, indicator=indicator, lags=lags), auxiliary)
    return OnlineDiagnostics{T,metrics,length(shape),typeof(moments),typeof(blocks),typeof(names)}(
        shape, names, moments, blocks, [ReentrantLock() for _ in 1:nchains])
end

include("online/batches.jl")
include("online/buffers.jl")
include("online/metadata.jl")
include("online/weights.jl")
include("online/covariance.jl")

@inline function _online_update!(state::OnlineDiagnostics{T}, draw, c, count, metadata=(;)) where {T}
    count == 0 && return
    n = state.moments.n[1, c]
    np = size(state.moments, 1)
    blocks = state.blocks
    _online_auxiliary_update!(blocks, draw, c, n, count, metadata)
    blocks.precision === nothing || _online_prepare!(blocks.precision, T, np, c, n + count)
    if blocks.indicator !== nothing
        _online_prepare!(blocks.indicator.levels, T, np, c, n + count)
    end
    blocks.lags === nothing || _online_lags!(blocks.lags, state.moments, draw, c, n, count)
    for (p, x) in enumerate(draw)
        blocks.precision === nothing || _online_batches!(blocks.precision, state.moments, p, c, x, count)
        if blocks.indicator !== nothing
            indicator = blocks.indicator
            event = T(x <= indicator.threshold[p])
            _online_batches!(indicator.levels, indicator.moments, p, c, event, count)
            _online_add!(indicator.moments, p, c, event, count)
        end
        _online_add!(state.moments, p, c, x, count)
    end
end

@inline function _online_locked(f, state, chains)
    locked = 0
    try
        for chain in chains
            lock(state.locks[chain])
            locked += 1
        end
        return f()
    finally
        for i in locked:-1:1
            unlock(state.locks[chains[i]])
        end
    end
end

function _online_chain(state, chain)
    chain isa Integer && 1 <= chain <= length(state.locks) ||
        throw(ArgumentError("chain must identify a state chain"))
    return Int(chain)
end

function _online_count(count)
    count isa Integer && 0 <= count <= typemax(Int) ||
        throw(ArgumentError("counts must be nonnegative integers representable by Int"))
    return Int(count)
end

# Copy only inputs of another element type.
_online_convert(::Type{T}, x::AbstractArray{<:Real}) where {T} = eltype(x) === T ? x : T.(x)
_online_convert(::Type{T}, x::AbstractVector{<:AbstractMatrix}) where {T} = [_online_convert(T, c) for c in x]

# Keep offsets small before division. Opposite extreme values need scaled subtraction.
function _online_scale(origin, x, scale)
    delta = x - origin
    return isfinite(delta) ? max(scale, abs(delta)) : max(scale, abs(origin), abs(x))
end

function _online_center(x, origin, scale)
    iszero(scale) && return zero(scale)
    delta = x - origin
    return isfinite(delta) ? delta / scale : x / scale - origin / scale
end

# Preserve low bits when a large accumulated M2 receives many small increments.
function _online_sum(x, y)
    value = x + y
    part = value - x
    return value, (x - (value - part)) + (y - part)
end

# The count check belongs to the locked caller, before any parameter changes.
# Coordinates prevent both large-offset cancellation and overflowing raw M2.
function _online_add!(moments, p, c, origin::T, count, source_scale=zero(T),
    source_mean=zero(T), source_m2=zero(T), source_error=zero(T)) where {T}
    count == 0 && return
    oldcount = moments.n[p, c]
    newcount = oldcount + count
    if oldcount == 0
        moments.origin[p, c] = origin
        moments.scale[p, c] = source_scale
        moments.mean[p, c] = source_mean
        if hasproperty(moments, :m2)
            moments.m2[p, c] = source_m2
            moments.m2_error[p, c] = source_error
        end
    else
        anchor = count > oldcount ? origin : moments.origin[p, c]
        scale = _online_scale(moments.origin[p, c], origin, max(moments.scale[p, c], source_scale))
        oldfactor = iszero(scale) ? zero(T) : moments.scale[p, c] / scale
        sourcefactor = iszero(scale) ? zero(T) : source_scale / scale
        oldmean = _online_center(moments.origin[p, c], anchor, scale) + oldfactor * moments.mean[p, c]
        sourcemean = _online_center(origin, anchor, scale) + sourcefactor * source_mean
        delta = sourcemean - oldmean
        # Update from the larger group so a near-one weight cannot erase the smaller group.
        mean = count > oldcount ? sourcemean - (T(oldcount) / T(newcount)) * delta :
            oldmean + (T(count) / T(newcount)) * delta
        neworigin = muladd(scale, mean, anchor)
        # Move the anchor with the mean and retain its rounding residual in scaled coordinates.
        moments.mean[p, c] = mean - _online_center(neworigin, anchor, scale)
        moments.origin[p, c] = neworigin
        moments.scale[p, c] = scale
        if hasproperty(moments, :m2)
            weight = T(min(oldcount, count)) * (T(max(oldcount, count)) / T(newcount))
            value, error = _online_sum(oldfactor * (oldfactor * moments.m2[p, c]),
                sourcefactor * (sourcefactor * source_m2))
            value, addition_error = _online_sum(value, delta * (weight * delta))
            error += addition_error + oldfactor * (oldfactor * moments.m2_error[p, c]) +
                sourcefactor * (sourcefactor * source_error)
            moments.m2[p, c], moments.m2_error[p, c] = _online_sum(value, error)
        end
    end
    moments.n[p, c] = newcount
    return nothing
end

"""
    push!(state::OnlineDiagnostics, draw; chain, count=1, metadata=(;))

Consume one draw, repeated `count` times. A scalar state accepts a real scalar.
Parameter states accept real arrays with exactly `parameter_shape`. Values convert
to the state's type; a value that overflows it is nonfinite. Zero counts ignore
nonfinite values. Invalid input and count overflow leave the call unconsumed.

Configured auxiliary metrics require `metadata` fields `energy`, `divergent`,
`acceptance`, `tree_depth`, `leapfrog_steps`, `logweight`, or `loglikelihood`.
Supply exactly the configured fields. Floating metadata uses the state's type,
divergence is Bool, depth/work are nonnegative integers, and acceptance is in
[0,1]. Full Hamiltonian energy includes kinetic energy. Log weights allow -Inf;
pointwise log likelihoods must be finite and have `observation_shape`.
Counts repeat the draw and every metadata value; split runs when metadata differs.
"""
function Base.push!(state::OnlineDiagnostics{T}, draw; chain, count=1, metadata=(;)) where {T}
    c = _online_chain(state, chain)
    k = _online_count(count)
    if isempty(state.shape)
        draw isa Real || throw(ArgumentError("draw must be real"))
    else
        draw isa AbstractArray{<:Real} || throw(ArgumentError("draw must be a real array"))
        size(draw) == state.shape || throw(DimensionMismatch("draw has the wrong parameter shape"))
    end
    values = isempty(state.shape) ? T(draw) : _online_convert(T, draw)
    _online_metadata_validate(state, metadata, k)
    k == 0 && return state
    all(isfinite, values) || throw(ArgumentError("positive-count draws must be finite"))
    _online_locked(state, (c,)) do
        Base.checked_add(state.moments.n[1, c], k)
        _online_work_check(state.blocks.work, metadata, c, k)
        _online_update!(state, values, c, k, metadata)
    end
    return state
end

"""
    append!(state::OnlineDiagnostics, draws; chain=nothing, drawdim=1, chaindim,
            counts=nothing, metadata=(;))

Consume a complete chunk of real values, converted as in `push!`. With `chain`,
`chaindim` defaults to `nothing`.
Without `chain`, the batch defaults apply and every state chain must be present.
Validate the whole input and all count additions before updating any chain.

Metadata always has stored-draw × chain axes, independently of parameter draw
axes. With an addressed `chain`, omit the chain axis. Log likelihood metadata
adds the fixed observation axes after those axes. Nested parameter chains use
vectors of per-chain metadata arrays. Every positive-count draw supplies all
configured metadata; zero-count rows may contain nonfinite placeholders.
"""
function Base.append!(state::OnlineDiagnostics{T}, x; chain=nothing, drawdim=1,
    chaindim=chain === nothing ? _default_chaindim(x) : nothing, counts=nothing, metadata=(;)) where {T}
    x isa AbstractArray{<:Real} || x isa AbstractVector{<:AbstractMatrix{<:Real}} ||
        throw(ArgumentError("draws must be real"))
    draws = _draws(_online_convert(T, x); drawdim, chaindim, counts)
    draws.shape == state.shape || throw(DimensionMismatch("chunk has the wrong parameter shape"))
    chains = chain === nothing ? eachindex(state.locks) : (_online_chain(state, chain),)
    length(draws.chains) == length(chains) ||
        throw(DimensionMismatch("chunk must contain every addressed chain"))
    supplied = _online_metadata_chunk(state, metadata, draws, chain !== nothing,
        x isa AbstractVector{<:AbstractMatrix})
    _online_locked(state, chains) do
        for (j, c) in enumerate(chains)
            Base.checked_add(state.moments.n[1, c], draws.lengths[j])
            _online_work_check_chunk(state.blocks.work, _online_metadata_chain(supplied,j), draws, j, c)
        end
        for (j, c) in enumerate(chains)
            values = draws.chains[j]
            for i in axes(values, 1)
                k = draws.counts === nothing ? 1 : draws.counts[j][i]
                _online_update!(state, @view(values[i, :]), c, k, _online_metadata_row(_online_metadata_chain(supplied,j), i))
            end
        end
    end
    return state
end

function _online_snapshot(state; blocks=false)
    return _online_locked(state, eachindex(state.locks)) do
        moments = map(copy, StructArrays.components(state.moments))
        blocks ? (moments=moments, blocks=deepcopy(state.blocks)) : moments
    end
end

"""
    empty!(state::OnlineDiagnostics)

Reset all chains under their locks. Retain the scalar type, shape, selected
metrics, and allocated storage. Finish old producers before resetting a cycle.
"""
function Base.empty!(state::OnlineDiagnostics)
    _online_locked(state, eachindex(state.locks)) do
        _online_clear!(state.moments)
        state.blocks.precision === nothing || _online_clear!(state.blocks.precision)
        if state.blocks.indicator !== nothing
            _online_clear!(state.blocks.indicator.moments)
            _online_clear!(state.blocks.indicator.levels)
        end
        state.blocks.lags === nothing || _online_clear!(state.blocks.lags)
        _online_auxiliary_clear!(state.blocks)
    end
    return state
end

"""
    merge!(destination::OnlineDiagnostics, source::OnlineDiagnostics)

Merge disjoint chunks from matching chains. Type, shape, chain count, and metrics
must match. The caller guarantees the source and destination contain disjoint
observations from the same chain in each slot. This moment-only operation cannot
merge independent chains into a single chain.

Copy the source under its locks, then release those locks before locking the
destination. Reject self-merges and count overflow without changing either state.
"""
function Base.merge!(destination::OnlineDiagnostics{T,M}, source::OnlineDiagnostics{U,K}) where {T,M,U,K}
    all(isnothing, destination.blocks) && all(isnothing, source.blocks) ||
        throw(ArgumentError("merge! requires moment-only configurations"))
    destination === source && throw(ArgumentError("cannot merge a state with itself"))
    T === U && M === K && destination.shape == source.shape &&
        isequal(destination.names, source.names) &&
        length(destination.locks) == length(source.locks) ||
        throw(ArgumentError("online state configurations must match"))
    snapshot = _online_snapshot(source)
    _online_locked(destination, eachindex(destination.locks)) do
        for c in eachindex(destination.locks)
            Base.checked_add(destination.moments.n[1, c], snapshot.n[1, c])
        end
        for c in eachindex(destination.locks), p in axes(snapshot.n, 1)
            m2 = hasproperty(snapshot, :m2) ? snapshot.m2[p, c] : zero(T)
            error = hasproperty(snapshot, :m2_error) ? snapshot.m2_error[p, c] : zero(T)
            _online_add!(destination.moments, p, c, snapshot.origin[p, c], snapshot.n[p, c],
                snapshot.scale[p, c], snapshot.mean[p, c], m2, error)
        end
    end
    return destination
end

# `unequal` accepts different chain lengths, with each chain's own variance divisor
# and the shortest length in the (n-1)/n term.
function _online_rhat(snapshot, p, ::Type{T}, means=Vector{T}(undef, size(snapshot.n, 2)),
    variances=similar(means); unequal=false) where {T}
    counts = @view snapshot.n[p, :]
    length(counts) < 2 && return (T(NaN), :insufficient_chains)
    unequal || all(==(first(counts)), counts) || return (T(NaN), :unequal_chain_lengths)
    n = minimum(counts)
    n < 2 && return (T(NaN), :insufficient_draws)
    origin = snapshot.origin[p, 1]
    # A stuck chain is evidence of nonconvergence. Only a constant pool is undefined.
    all(c -> iszero(snapshot.m2[p, c]) && snapshot.origin[p, c] == origin, eachindex(counts)) &&
        return (T(NaN), :constant)
    scale = maximum(@view snapshot.scale[p, :])
    for c in eachindex(counts)
        scale = _online_scale(origin, snapshot.origin[p, c], scale)
    end
    for c in eachindex(counts)
        factor = snapshot.scale[p, c] / scale
        means[c] = _online_center(snapshot.origin[p, c], origin, scale) + factor * snapshot.mean[p, c]
        variances[c] = factor * (factor * (snapshot.m2[p, c] + snapshot.m2_error[p, c])) / T(counts[c] - 1)
    end
    result = _rhat_from_moments(means, variances, n)
    return (result, isnan(result) ? :estimator_failed : :ok)
end

"""
    diagnostics(state::OnlineDiagnostics)

Return an owned `StructArray` snapshot with one row per scalar parameter.
Selected `mean` and `variance` fields hold per-chain vectors. `rhat_basic` is the
unsplit classical R-hat. `status` has the selected metric names, with per-chain
statuses for moments and one status for R-hat. `index` identifies the parameter,
and `name` holds its configured name.

Configured `mcse_mean`, `ess_mean`, and `iact_mean` fields hold per-chain vectors.
`indicator` holds per-chain records with the frozen threshold, event probability,
precision and status. `precision` holds per-chain estimator identity
`:dyadic_batchmeans`, batch width, completed batches, remainder and draw count.
`autocor` and its status contain lag × chain matrices, for lags `0:max_lag`.
Zero marginal variance is `:constant`; constant indicators are
`:unobserved_event`; positive marginal variance with zero estimated LRV is
`:degenerate_estimator`. These precision estimates are unavailable (`NaN`).

`n_per_chain` holds logical chain counts. `n_draws` uses a checked `Int` sum.
An aggregate count overflow throws without changing the state. Changing a snapshot
or updating/resetting its source never changes the other object's data.
"""
function diagnostics(state::OnlineDiagnostics{T,Metrics}) where {T,Metrics}
    owned = _online_snapshot(state; blocks=true)
    snapshot, blocks = owned.moments, owned.blocks
    np, nc = size(snapshot.n)
    counts = ArraysOfArrays.VectorOfSimilarVectors(permutedims(snapshot.n))
    selected = filter(m -> m in _ONLINE_PARAMETER_METRICS, Metrics)
    fields = map(selected) do metric
        if metric in (:mcse_mean, :ess_mean, :iact_mean, :indicator, :autocor)
            return _online_metric(snapshot, blocks, metric, T)
        elseif metric === :rhat_basic
            values = [_online_rhat(snapshot, p, T) for p in 1:np]
            return (first.(values), last.(values))
        end
        values = Matrix{T}(undef, nc, np)
        statuses = Matrix{Symbol}(undef, nc, np)
        for p in 1:np, c in 1:nc
            n = snapshot.n[p, c]
            enough = metric === :mean ? n > 0 : n > 1
            value = if !enough
                T(NaN)
            elseif metric === :mean
                muladd(snapshot.scale[p, c], snapshot.mean[p, c], snapshot.origin[p, c])
            else
                scale = snapshot.scale[p, c]
                scale * (scale * ((snapshot.m2[p, c] + snapshot.m2_error[p, c]) / T(n - 1)))
            end
            values[c, p] = value
            statuses[c, p] = !enough ? :insufficient_draws : isfinite(value) ? :ok : :nonfinite_result
        end
        return (ArraysOfArrays.VectorOfSimilarVectors(values), ArraysOfArrays.VectorOfSimilarVectors(statuses))
    end
    results = NamedTuple{selected}(map(first, fields))
    statuses = NamedTuple{selected}(map(last, fields))
    if blocks.precision !== nothing || blocks.indicator !== nothing
        metadata = [[_online_batch_info(snapshot.n[p, c]) for c in 1:nc] for p in 1:np]
        results = merge(results, (precision=metadata,))
    end
    total = foldl(Base.checked_add, @view(snapshot.n[1, :]); init=0)
    index = (index=vec(collect(CartesianIndices(state.shape))),)
    state.names === nothing || (index = merge(index, (name=copy(state.names),)))
    return StructArray(merge(index, results,
        (n_draws=fill(total, np), n_per_chain=counts, n_chains=fill(nc, np),
         status=isempty(selected) ? fill((;), np) : StructArray(statuses))))
end

"""
    rhat(state::OnlineDiagnostics; kind=:basic, split=false)

Read unsplit basic R-hat for each parameter. The batch default `kind=:rank,
split=true` is unavailable online: ranks need every draw, and the split point
moves with each draw.
"""
function rhat(state::OnlineDiagnostics{T,Metrics}; kind=:basic, split=false) where {T,Metrics}
    kind === :basic && split === false ||
        throw(ArgumentError("online states support only kind=:basic with split=false"))
    :rhat_basic in Metrics || throw(ArgumentError("state does not track rhat_basic"))
    snapshot = _online_snapshot(state)
    values = [first(_online_rhat(snapshot, p, T)) for p in axes(snapshot.n, 1)]
    return _parameter_result(values, state.shape)
end

function _online_metric(moments, blocks, metric, ::Type{T}) where {T}
    np, nc = size(moments.n)
    if metric === :autocor
        columns = [[_online_autocor(moments, blocks.lags, p, c, T) for c in 1:nc] for p in 1:np]
        return (map(x -> reduce(hcat, first.(x)), columns), map(x -> reduce(hcat, last.(x)), columns))
    elseif metric === :indicator
        indicator = blocks.indicator
        records = [[begin
            estimate = _online_precision(indicator.moments, indicator.levels, p, c, T; indicator=true)
            m = indicator.moments
            probability = m.n[p, c] == 0 ? T(NaN) : muladd(m.scale[p, c], m.mean[p, c], m.origin[p, c])
            merge((threshold=indicator.threshold[p], probability=probability), estimate)
        end for c in 1:nc] for p in 1:np]
        return (records, map(x -> getproperty.(x, :status), records))
    end
    estimates = [_online_precision(moments, blocks.precision, p, c, T; metric) for c in 1:nc, p in 1:np]
    values = map(x -> getproperty(x, metric), estimates)
    statuses = map(x -> x.status, estimates)
    return (ArraysOfArrays.VectorOfSimilarVectors(values), ArraysOfArrays.VectorOfSimilarVectors(statuses))
end

function _online_precision_result(state::OnlineDiagnostics{T,Metrics}, metric, kind, estimator,
    pooled) where {T,Metrics}
    kind === :mean && estimator === :dyadic_batchmeans ||
        throw(ArgumentError("online precision requires kind=:mean and estimator=:dyadic_batchmeans"))
    metric in Metrics || throw(ArgumentError("state does not track $metric"))
    owned = _online_snapshot(state; blocks=true)
    moments, levels = owned.moments, owned.blocks.precision
    np, nc = size(moments.n)
    if pooled
        draws, batches = _online_moments(T, (1, 1)), _online_moments(T, (1, 1))
        values = [getproperty(_online_pooled_precision(moments, levels, p, T, draws, batches), metric)
            for p in 1:np]
        return _parameter_result(values, state.shape)
    end
    values = [getproperty(_online_precision(moments, levels, p, c, T), metric) for c in 1:nc, p in 1:np]
    return reshape(values, (nc, state.shape...))
end

"""
    mcse(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled=false)
    ess(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled=false)
    iact(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled=false)

Read configured raw-mean precision separately for each chain. The result shape
is chain × parameter axes, including the chain axis for one chain. Unavailable
estimates are NaN; `diagnostics(state)` also reports status and batch metadata.
The state must select the corresponding `:mcse_mean`, `:ess_mean`, or
`:iact_mean` metric. These estimates do not measure rank-based bulk precision.

`pooled=true` returns one value per parameter for the mean of all draws, from
replicated batch means: all complete batches center on their pooled mean, so
chains that disagree inflate the long-run variance and lower the ESS. Every chain
with draws uses the batch width of the shortest one. This assumes independent
chains with a common stationary law; it does not replace R-hat.
"""
mcse(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled::Bool=false) =
    _online_precision_result(state, :mcse_mean, kind, estimator, pooled)
ess(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled::Bool=false) =
    _online_precision_result(state, :ess_mean, kind, estimator, pooled)
iact(state::OnlineDiagnostics; kind=:mean, estimator=:dyadic_batchmeans, pooled::Bool=false) =
    _online_precision_result(state, :iact_mean, kind, estimator, pooled)

"""
    progress_summary(state::OnlineDiagnostics)

Return `(n_draws, max_rhat, min_ess)` for a live progress display. `max_rhat` is
the largest online basic R-hat and `min_ess` the smallest pooled mean ESS (see
`ess(state; pooled=true)`), each over parameters with a value. Running producers
almost never stop at equal chain lengths, so `max_rhat` also accepts unequal
lengths: each chain uses all its draws and its own variance divisor, and the
shortest length enters `(n-1)/n`. Chains that differ by d draws change the
between-chain term by a relative O(d/n) against equal lengths.
`rhat(state)` keeps the batch rule and needs equal lengths. A field is NaN when
its metric (`:rhat_basic`, or any of `:mcse_mean`, `:ess_mean`, `:iact_mean`) is
not configured or no parameter has a value. Holds all chain locks while it reads,
without copying the state.
"""
function progress_summary(state::OnlineDiagnostics{T,Metrics}) where {T,Metrics}
    levels = state.blocks.precision
    nc = length(state.locks)
    means, variances = Vector{T}(undef, nc), Vector{T}(undef, nc)
    draws, batches = _online_moments(T, (1, 1)), _online_moments(T, (1, 1))
    return _online_locked(state, eachindex(state.locks)) do
        moments = state.moments
        max_rhat = min_ess = T(NaN)
        for p in axes(moments.n, 1)
            if :rhat_basic in Metrics
                value = first(_online_rhat(moments, p, T, means, variances; unequal=true))
                isnan(value) || (max_rhat = isnan(max_rhat) ? value : max(max_rhat, value))
            end
            if levels !== nothing
                value = _online_pooled_precision(moments, levels, p, T, draws, batches).ess_mean
                isnan(value) || (min_ess = isnan(min_ess) ? value : min(min_ess, value))
            end
        end
        n_draws = foldl(Base.checked_add, @view(moments.n[1, :]); init=0)
        (n_draws=n_draws, max_rhat=max_rhat, min_ess=min_ess)
    end
end

"""
    autocor(state::OnlineDiagnostics; lags=nothing)

Read globally centered autocorrelations with shape lag × chain × parameter axes.
Default lags are `0:max_lag`, fixed at construction. Lags beyond a chain's current
length have NaN and `:insufficient_draws` status in `diagnostics`. Each lag uses
divisor n, matching batch `autocor`. The bounded lag cap does not estimate LRV.
"""
function autocor(state::OnlineDiagnostics{T,Metrics}; lags=nothing) where {T,Metrics}
    :autocor in Metrics || throw(ArgumentError("state does not track autocor"))
    owned = _online_snapshot(state; blocks=true)
    cap = size(owned.blocks.lags.ring, 2)
    selected = lags === nothing ? collect(0:cap) : collect(lags)
    all(k -> k isa Integer && 0 <= k <= cap, selected) ||
        throw(ArgumentError("requested lags must be within the configured max_lag"))
    np, nc = size(owned.moments.n)
    values = Array{T}(undef, length(selected), nc, np)
    for p in 1:np, c in 1:nc
        values[:, c, p] = first(_online_autocor(owned.moments, owned.blocks.lags, p, c, T))[selected .+ 1]
    end
    return reshape(values, (length(selected), nc, state.shape...))
end
