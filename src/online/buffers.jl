struct _OnlineLags{T,S}
    left::S
    right::S
    cross::Matrix{T}
    cross_error::Matrix{T}
    ring::Array{T,3}
    head::Vector{Int}
end

function _OnlineLags(::Type{T}, np, nc, cap) where {T}
    dims = (np, Base.checked_mul(nc, cap))
    return _OnlineLags(_online_moments(T, dims; second=false),
        _online_moments(T, dims; second=false), zeros(T, dims), zeros(T, dims),
        zeros(T, np, cap, nc), zeros(Int, nc))
end

# Merge a constant pair group in independently scaled left/right coordinates.
function _online_pair!(lags, p, slot, x::T, y::T, count) where {T}
    left, right = lags.left, lags.right
    n = left.n[p, slot]
    if n > 0
        sx = _online_scale(left.origin[p, slot], x, left.scale[p, slot])
        sy = _online_scale(right.origin[p, slot], y, right.scale[p, slot])
        fx = iszero(sx) ? zero(T) : left.scale[p, slot] / sx
        fy = iszero(sy) ? zero(T) : right.scale[p, slot] / sy
        dx = _online_center(x, left.origin[p, slot], sx) - fx * left.mean[p, slot]
        dy = _online_center(y, right.origin[p, slot], sy) - fy * right.mean[p, slot]
        weight = T(min(n, count)) * (T(max(n, count)) / T(n + count))
        value, error = _online_sum(fx * (fy * lags.cross[p, slot]), dx * (weight * dy))
        error += fx * (fy * lags.cross_error[p, slot])
        lags.cross[p, slot], lags.cross_error[p, slot] = _online_sum(value, error)
    end
    _online_add!(left, p, slot, x, count)
    _online_add!(right, p, slot, y, count)
end

function _online_lags!(lags, draw, c, n, count)
    cap = size(lags.ring, 2)
    cap == 0 && return
    nc = length(lags.head)
    # Only the first cap observations can pair with the preceding run.
    boundary = min(count, cap)
    head = lags.head[c]
    for i in 1:boundary
        for lag in 1:min(cap, n + i - 1)
            index = mod1(head - lag + 1, cap)
            slot = c + (lag - 1) * nc
            for (p, x) in enumerate(draw)
                _online_pair!(lags, p, slot, lags.ring[p, index, c], x, 1)
            end
        end
        head = mod1(head + 1, cap)
        for (p, x) in enumerate(draw)
            lags.ring[p, head, c] = x
        end
    end
    remaining = count - boundary
    if remaining > 0
        for lag in 1:cap, (p, x) in enumerate(draw)
            _online_pair!(lags, p, c + (lag - 1) * nc, x, x, remaining)
        end
        # Every ring value is now x; advance its logical cursor without expansion.
        head = mod1(head + rem(remaining, cap), cap)
    end
    lags.head[c] = head
end

function _online_clear!(lags::_OnlineLags)
    _online_clear!(lags.left)
    _online_clear!(lags.right)
    fill!(lags.cross, zero(eltype(lags.cross)))
    fill!(lags.cross_error, zero(eltype(lags.cross_error)))
    fill!(lags.ring, zero(eltype(lags.ring)))
    fill!(lags.head, 0)
end

function _online_autocor(moments, lags, p, c, lag, ::Type{T}) where {T}
    n = moments.n[p, c]
    n <= lag && return (T(NaN), :insufficient_draws)
    m2 = moments.m2[p, c] + moments.m2_error[p, c]
    m2 <= zero(T) && return (T(NaN), :constant)
    lag == 0 && return (one(T), :ok)
    slot = c + (lag - 1) * length(lags.head)
    scale, origin, mean = moments.scale[p, c], moments.origin[p, c], moments.mean[p, c]
    left, right = lags.left, lags.right
    fx, fy = left.scale[p, slot] / scale, right.scale[p, slot] / scale
    dx = _online_center(left.origin[p, slot], origin, scale) + fx * left.mean[p, slot] - mean
    dy = _online_center(right.origin[p, slot], origin, scale) + fy * right.mean[p, slot] - mean
    centered = fx * (fy * (lags.cross[p, slot] + lags.cross_error[p, slot]))
    value = (centered + dx * (T(n - lag) * dy)) / m2
    return (value, isfinite(value) ? :ok : :nonfinite_result)
end
