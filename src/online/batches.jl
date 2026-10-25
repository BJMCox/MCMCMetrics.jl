# Each chain owns its growing level vector; no shared array changes shape.
struct _OnlineBatchLevel{S,P}
    completed::S
    partial::P
end

function _online_moments(::Type{T}, dims; second=true) where {T}
    columns = (n=zeros(Int, dims), origin=zeros(T, dims), scale=zeros(T, dims), mean=zeros(T, dims))
    second && (columns = merge(columns, (m2=zeros(T, dims), m2_error=zeros(T, dims))))
    return StructArray(columns)
end

function _online_levels(::Type{T}, np, nc) where {T}
    level = _OnlineBatchLevel(_online_moments(T, (np, 1)), _online_moments(T, (np, 1); second=false))
    return [typeof(level)[] for _ in 1:nc]
end

# Level j holds width 2^j. Width 1 batches are the draws themselves.
_online_level_count(n) = n == 0 ? 0 : 8sizeof(Int) - 1 - leading_zeros(n)

function _online_prepare!(levels, ::Type{T}, np, c, n) where {T}
    while length(levels[c]) < _online_level_count(n)
        push!(levels[c], _OnlineBatchLevel(_online_moments(T, (np, 1)),
            _online_moments(T, (np, 1); second=false)))
    end
end

# Return the completed batch group for the next level.
@inline function _online_complete!(level, p)
    partial = level.partial
    group = (n=partial.n[p, 1], origin=partial.origin[p, 1], scale=partial.scale[p, 1],
        mean=partial.mean[p, 1])
    # Only the mean enters between-batch moments. Do not carry a huge scale
    # from excursions that canceled within the batch into its small residual.
    residual = group.scale*group.mean
    scale = abs(residual)
    mean = iszero(scale) ? zero(residual) : sign(residual)
    if iszero(scale) && !iszero(group.mean)
        scale, mean = group.scale, group.mean
    end
    _online_add!(level.completed, p, 1, group.origin, 1, scale, mean)
    partial.n[p, 1] = 0
    return group
end

# A level's partial holds only completed lower batches, so it changes only when
# the level below completes one. Input is an optional mixed lower batch `head`,
# then `run` lower batches of x. Return the same description for the next level.
@inline function _online_climb!(level, p, x, head, run, half)
    partial = level.partial
    completed = nothing
    if head !== nothing
        pending = partial.n[p, 1] > 0
        _online_add!(partial, p, 1, head.origin, head.n, head.scale, head.mean)
        pending && (completed = _online_complete!(level, p))
    end
    if run > 0 && partial.n[p, 1] > 0
        _online_add!(partial, p, 1, x, half)
        completed = _online_complete!(level, p)
        run -= 1
    end
    full, rest = divrem(run, 2)
    _online_add!(level.completed, p, 1, x, full)
    _online_add!(partial, p, 1, x, rest * half)
    return completed, full
end

# New widths must start with the prefix ending at that width, not the final run mean.
function _online_start!(level, moments, p, c, x, endpoint, width)
    n = moments.n[p, c]
    level.partial.n[p, 1] = n
    level.partial.origin[p, 1] = moments.origin[p, c]
    level.partial.scale[p, 1] = moments.scale[p, c]
    level.partial.mean[p, 1] = moments.mean[p, c]
    _online_add!(level.partial, p, 1, x, width - n)
    _online_complete!(level, p)
    full, rest = divrem(endpoint - width, width)
    _online_add!(level.completed, p, 1, x, full)
    half = width >> 1
    _online_add!(level.partial, p, 1, x, rest - rest % half)
end

function _online_batches!(levels, moments, p, c, x, count)
    count == 0 && return
    n = moments.n[p, c]
    endpoint = n + count
    started = _online_level_count(n)
    head, run = nothing, count
    for j in 1:started
        head, run = _online_climb!(levels[c][j], p, x, head, run, 1 << (j - 1))
        head === nothing && run == 0 && break
    end
    for j in started+1:_online_level_count(endpoint)
        _online_start!(levels[c][j], moments, p, c, x, endpoint, 1 << j)
    end
end

function _online_clear!(moments::StructArray)
    for column in StructArrays.components(moments)
        fill!(column, zero(eltype(column)))
    end
end

function _online_clear!(levels::Vector{<:Vector})
    for chain in levels, level in chain
        _online_clear!(level.completed)
        _online_clear!(level.partial)
    end
end

function _online_batch_info(n)
    exponent = n == 0 ? 0 : (8sizeof(Int) - 1 - leading_zeros(n)) ÷ 2
    width = 1 << exponent
    return (estimator=:dyadic_batchmeans, batch_width=width,
        completed_batches=div(n, width), remainder=rem(n, width), n_draws=n)
end

function _online_precision(moments, levels, p, c, ::Type{T}; indicator=false, metric=:all) where {T}
    n = moments.n[p, c]
    info = _online_batch_info(n)
    unavailable(status) = (mcse_mean=T(NaN), ess_mean=T(NaN), iact_mean=T(NaN), status=status)
    n < 2 && return unavailable(:insufficient_draws)
    drawm2 = moments.m2[p, c] + moments.m2_error[p, c]
    drawm2 <= zero(T) && return unavailable(indicator ? :unobserved_event : :constant)
    a = info.completed_batches
    a < 2 && return unavailable(:insufficient_batches)
    drawscale = moments.scale[p, c]
    batchm2, batchscale = if info.batch_width == 1
        drawm2, drawscale
    else
        level = levels[c][trailing_zeros(info.batch_width)].completed
        level.m2[p, 1] + level.m2_error[p, 1], level.scale[p, 1]
    end
    batchm2 <= zero(T) && return unavailable(:degenerate_estimator)
    drawvar = drawm2 / T(n - 1)
    batchvar = batchm2 / T(a - 1)
    # Form the MCSE directly: LRV itself need not fit in the user's type.
    mcse = batchscale * (sqrt(batchvar) * sqrt(T(info.batch_width) / T(n)))
    ratio = (drawscale / batchscale) * sqrt(drawvar / batchvar)
    ratio *= ratio
    ess = (T(n) / T(info.batch_width)) * ratio
    iact = T(info.batch_width) / ratio
    if !isfinite(ratio) || iszero(ratio)
        logratio = T(2) * (log(drawscale) - log(batchscale)) + log(drawvar) - log(batchvar)
        ess = exp(log(T(n)) - log(T(info.batch_width)) + logratio)
        iact = exp(log(T(info.batch_width)) - logratio)
    end
    selected = metric === :mcse_mean ? (mcse,) : metric === :ess_mean ? (ess,) :
        metric === :iact_mean ? (iact,) : (mcse,ess,iact)
    status = all(x -> isfinite(x) && x > zero(T), selected) ? :ok : :nonfinite_result
    return (mcse_mean=mcse, ess_mean=ess, iact_mean=iact, status=status)
end
