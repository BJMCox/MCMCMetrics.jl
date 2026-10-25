# Lag sums use u = (x - anchor) / scale for each parameter and chain. The anchor
# follows the chain mean and the scale is a power of two near the data scale, so
# fixed coordinates avoid cancellation and the compensated sums vectorize over
# parameters. Arrays are parameter × lag (or ring slot) × chain.
struct _OnlineLags{T}
    anchor::Matrix{T}
    scale::Matrix{T}
    total::Matrix{T}
    total_error::Matrix{T}
    cross::Array{T,3}
    cross_error::Array{T,3}
    prefix::Array{T,3}
    ring::Array{T,3}
    current::Matrix{T}
    head::Vector{Int}
    next::Vector{Int}
end

function _OnlineLags(::Type{T}, np, nc, cap) where {T}
    lagged() = zeros(T, np, cap, nc)
    return _OnlineLags(zeros(T, np, nc), ones(T, np, nc), zeros(T, np, nc), zeros(T, np, nc),
        lagged(), lagged(), lagged(), lagged(), zeros(T, np, nc), zeros(Int, nc), zeros(Int, nc))
end

# Bound |u| so that products times any Int count stay finite in Float32.
const _ONLINE_LAG_BOUND = 2.0^16

_online_slot(head, lag, cap) = mod1(head - lag + 1, cap)

# Move to u' = (u - shift) * factor. Rescale first, so that tiny old units do not
# underflow. Pair sums use Σx + Σy = 2Σu - prefix - recent.
function _online_lag_shift!(lags, p, c, n, shift::T, factor::T) where {T}
    cap = size(lags.ring, 2)
    shift *= factor
    total = (lags.total[p, c] + lags.total_error[p, c]) * factor
    recent = zero(T)
    for lag in 1:min(cap, n - 1)
        recent += lags.ring[p, _online_slot(lags.head[c], lag, cap), c] * factor
        sums = T(2) * total - lags.prefix[p, lag, c] * factor - recent
        cross = factor * (factor * (lags.cross[p, lag, c] + lags.cross_error[p, lag, c]))
        lags.cross[p, lag, c] = cross - shift * sums + T(n - lag) * shift * shift
        lags.cross_error[p, lag, c] = zero(T)
    end
    for lag in 1:cap
        lags.prefix[p, lag, c] = lags.prefix[p, lag, c] * factor - T(min(lag, n)) * shift
        lags.ring[p, lag, c] = lags.ring[p, lag, c] * factor - shift
    end
    lags.total[p, c] = total - T(n) * shift
    lags.total_error[p, c] = zero(T)
end

# Re-center on the running lag mean and adopt the chain moments' power-of-two scale.
function _online_lag_center!(lags, moments, p, c, n)
    T = eltype(lags.ring)
    anchor, scale = lags.anchor[p, c], lags.scale[p, c]
    reference = moments.scale[p, c]
    iszero(reference) && return
    target = exp2(T(exponent(reference)))
    moved = muladd(scale, (lags.total[p, c] + lags.total_error[p, c]) / T(n), anchor)
    # Shift by the anchor change that survives rounding.
    shift = isfinite(moved - anchor) ? (moved - anchor) / scale : zero(T)
    iszero(shift) && (moved = anchor)
    _online_lag_shift!(lags, p, c, n, shift, scale / target)
    lags.anchor[p, c], lags.scale[p, c] = moved, target
end

function _online_lag_coordinates!(lags, draw, c, n)
    T = eltype(lags.ring)
    for (p, x) in enumerate(draw)
        n == 0 && (lags.anchor[p, c] = x)
        u = _online_center(x, lags.anchor[p, c], lags.scale[p, c])
        if abs(u) > T(_ONLINE_LAG_BOUND)
            growth = exp2(T(exponent(u) + 1))
            _online_lag_shift!(lags, p, c, n, zero(T), inv(growth))
            lags.scale[p, c] *= growth
            u = _online_center(x, lags.anchor[p, c], lags.scale[p, c])
        end
        lags.current[p, c] = u
    end
end

# Compensated cross .+= ring .* current. One shared axis lets the compiler drop
# bounds checks, which the vectorized loop needs.
function _online_lag_add!(cross, error, ring, current)
    @simd for p in eachindex(cross, error, ring, current)
        r, u = ring[p], current[p]
        product = r * u
        value, rounding = _online_sum(cross[p], product)
        cross[p] = value
        error[p] += rounding + fma(r, u, -product)
    end
end

# Add `count` copies of the current coordinates after n draws.
function _online_lag_step!(lags, c, n, count)
    T = eltype(lags.ring)
    np, cap = size(lags.ring, 1), size(lags.ring, 2)
    k = T(count)
    if count == 1
        current = view(lags.current, :, c)
        for lag in 1:min(cap, n)
            slot = _online_slot(lags.head[c], lag, cap)
            _online_lag_add!(view(lags.cross, :, lag, c), view(lags.cross_error, :, lag, c),
                view(lags.ring, :, slot, c), current)
        end
        lags.head[c] = mod1(lags.head[c] + 1, cap)
        for p in 1:np
            u = lags.current[p, c]
            lags.ring[p, lags.head[c], c] = u
            for lag in n+1:cap
                lags.prefix[p, lag, c] += u
            end
        end
    else
        # Callers pass a run only after cap single steps filled the ring with it.
        for lag in 1:cap
            for p in 1:np
                u = lags.current[p, c]
                square = u * u
                product = k * square
                value, error = _online_sum(lags.cross[p, lag, c], product)
                lags.cross[p, lag, c] = value
                lags.cross_error[p, lag, c] += error + (fma(k, square, -product) + k * fma(u, u, -square))
            end
        end
        lags.head[c] = mod1(lags.head[c] + rem(count, cap), cap)
    end
    for p in 1:np
        u = lags.current[p, c]
        product = k * u
        value, error = _online_sum(lags.total[p, c], product)
        lags.total[p, c] = value
        lags.total_error[p, c] += error + fma(k, u, -product)
    end
end

function _online_lags!(lags, moments, draw, c, n, count)
    cap = size(lags.ring, 2)
    cap == 0 && return
    # Only the first cap observations can pair with the preceding run.
    boundary = min(count, cap)
    for m in n:n+boundary-1
        # Re-centering after every n/8 draws costs amortized O(cap / n) per draw.
        if m > 0 && m >= lags.next[c]
            for p in axes(lags.ring, 1)
                _online_lag_center!(lags, moments, p, c, m)
            end
            lags.next[c] = m + max(1, m >> 3)
        end
        _online_lag_coordinates!(lags, draw, c, m)
        _online_lag_step!(lags, c, m, 1)
    end
    count > boundary && _online_lag_step!(lags, c, n + boundary, count - boundary)
end

function _online_clear!(lags::_OnlineLags)
    for field in (lags.anchor, lags.total, lags.total_error, lags.cross, lags.cross_error,
        lags.prefix, lags.ring, lags.current)
        fill!(field, zero(eltype(field)))
    end
    fill!(lags.scale, one(eltype(lags.scale)))
    fill!(lags.head, 0)
    fill!(lags.next, 0)
end

# Globally centered lag products: Σ(x_{t-L} - μ)(x_t - μ) = S_xy - (n+L)d² + d(prefix + recent).
function _online_autocor(moments, lags, p, c, ::Type{T}) where {T}
    cap = size(lags.ring, 2)
    n = moments.n[p, c]
    m2 = moments.m2[p, c] + moments.m2_error[p, c]
    values, statuses = fill(T(NaN), cap + 1), fill(:insufficient_draws, cap + 1)
    n == 0 && return (values, statuses)
    if m2 <= zero(T)
        statuses[1:min(n, cap + 1)] .= :constant
        return (values, statuses)
    end
    values[1], statuses[1] = one(T), :ok
    d = (lags.total[p, c] + lags.total_error[p, c]) / T(n)
    ratio = lags.scale[p, c] / moments.scale[p, c]
    recent = zero(T)
    for lag in 1:min(cap, n - 1)
        recent += lags.ring[p, _online_slot(lags.head[c], lag, cap), c]
        centered = (lags.cross[p, lag, c] + lags.cross_error[p, lag, c]) - T(n + lag) * d * d +
            d * (lags.prefix[p, lag, c] + recent)
        value = ratio * (ratio * centered) / m2
        values[lag + 1], statuses[lag + 1] = value, isfinite(value) ? :ok : :nonfinite_result
    end
    return (values, statuses)
end
