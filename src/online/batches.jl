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

function _online_prepare!(levels, ::Type{T}, np, c, n) where {T}
    needed = n == 0 ? 0 : 8sizeof(Int) - leading_zeros(n)
    while length(levels[c]) < needed
        push!(levels[c], _OnlineBatchLevel(_online_moments(T, (np, 1)),
            _online_moments(T, (np, 1); second=false)))
    end
end

function _online_complete!(level, p)
    partial = level.partial
    # Only the mean enters between-batch moments. Do not carry a huge scale
    # from excursions that canceled within the batch into its small residual.
    residual = partial.scale[p,1]*partial.mean[p,1]
    scale = abs(residual)
    mean = iszero(scale) ? zero(residual) : sign(residual)
    if iszero(scale) && !iszero(partial.mean[p,1])
        scale, mean = partial.scale[p,1], partial.mean[p,1]
    end
    _online_add!(level.completed, p, 1, partial.origin[p, 1], 1,
        scale, mean)
    partial.n[p, 1] = 0
end

# New widths must start with the prefix ending at that width, not the final run mean.
function _online_batches!(levels, moments, p, c, x, count)
    count == 0 && return
    n = moments.n[p, c]
    endpoint = n + count
    for (j, level) in enumerate(levels[c])
        width = 1 << (j - 1)
        width > endpoint && break
        left = count
        if n < width
            level.partial.n[p, 1] = moments.n[p, c]
            level.partial.origin[p, 1] = moments.origin[p, c]
            level.partial.scale[p, 1] = moments.scale[p, c]
            level.partial.mean[p, 1] = moments.mean[p, c]
        end
        partial = level.partial.n[p, 1]
        if partial > 0
            take = min(left, width - partial)
            _online_add!(level.partial, p, 1, x, take)
            left -= take
            partial + take == width && _online_complete!(level, p)
        end
        full, remainder = divrem(left, width)
        _online_add!(level.completed, p, 1, x, full)
        _online_add!(level.partial, p, 1, x, remainder)
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
    level = levels[c][trailing_zeros(info.batch_width) + 1].completed
    batchm2 = level.m2[p, 1] + level.m2_error[p, 1]
    batchm2 <= zero(T) && return unavailable(:degenerate_estimator)
    drawvar = drawm2 / T(n - 1)
    batchvar = batchm2 / T(a - 1)
    drawscale = moments.scale[p, c]
    batchscale = level.scale[p, 1]
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
