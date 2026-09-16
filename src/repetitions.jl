struct _WorkLimitError <: Exception
    message::String
end

Base.showerror(io::IO, error::_WorkLimitError) = print(io, error.message)

function _check_window(window, max_lag_work)
    isfinite(window) && window > 0 || throw(ArgumentError("window must be finite and positive"))
    max_lag_work > 0 || throw(ArgumentError("max_lag_work must be positive"))
end

function _check_fft_counts(d, autocov)
    autocov === :fft && !isnothing(d.counts) &&
        throw(ArgumentError("FFT requires dense draws; compressed inputs are never expanded"))
end

function _single_chain_runs(d, p, c)
    indices = [i for i in axes(d.chains[c], 1) if _count(d, c, i) > 0]
    values = d.chains[c][indices, p]
    weights = [_count(d, c, i) for i in indices]
    (; values, weights, ranges=[1:length(values)], n=d.lengths[c])
end

function _weighted_mean(values::AbstractVector{T}, weights) where {T}
    total = foldl(Base.Checked.checked_add, weights; init=0)
    sum(i -> T(weights[i]) / T(total) * values[i], eachindex(values))
end

function _weighted_var(values::AbstractVector{T}, weights) where {T}
    total = foldl(Base.Checked.checked_add, weights; init=0)
    center = _weighted_mean(values, weights)
    sum(i -> T(weights[i]) / T(total - 1) * abs2(values[i] - center), eachindex(values))
end

# Intersect each run's [start,end) interval with the lag-shifted intervals.
# Both cursors advance monotonically, so one lag costs O(stored runs).
function _run_autocov(z::AbstractVector{T}, weights, ranges, n, lag) where {T}
    result = zero(T)
    for r in ranges
        i = j = first(r)
        istart = jstart = 0
        iend, jend = weights[i], weights[j]
        while i <= last(r) && j <= last(r)
            overlap = min(iend, jend - lag) - max(istart, jstart - lag)
            overlap > 0 && (result += T(overlap) / T(n) * z[i] * z[j])
            if iend <= jend - lag
                istart = iend
                i += 1
                i <= last(r) && (iend += weights[i])
            else
                jstart = jend
                j += 1
                j <= last(r) && (jend += weights[j])
            end
        end
    end
    result / T(length(ranges))
end

function _autocov_function(z, runs, autocov, max_lag_work)
    if autocov === :fft
        covariance = _fft_autocov(reshape(z, runs.n, length(runs.ranges)))
        return k -> covariance[k + 1]
    end
    work = Ref(0)
    return function (k)
        # Count a conservative two cursor advances per stored run and lag.
        cost = Base.Checked.checked_mul(2, length(z))
        cost <= max_lag_work - work[] || throw(_WorkLimitError(
            "direct lag work exceeds max_lag_work; increase the explicit budget or use dense FFT"))
        work[] += cost
        _run_autocov(z, runs.weights, runs.ranges, runs.n, k)
    end
end

function _fft_autocov end
_fft_autocov(z) = throw(ArgumentError("autocov=:fft requires loading FFTW"))

function _window_tau(rho, n, ::Type{T}, estimator, window) where {T}
    if estimator === :sokal
        tau = one(T)
        for lag in 1:n-1
            tau += T(2) * rho(lag)
            isfinite(tau) || return T(NaN)
            # Use tau = 1 + 2 sum(rho), and the first positive window M >= c*tau.
            tau > zero(T) && lag >= window * tau && return tau
        end
        return T(NaN)
    end
    previous = T(Inf)
    pairsum = zero(T)
    for lag in 0:2:n-2
        pair = (lag == 0 ? one(T) : rho(lag)) + rho(lag + 1)
        isfinite(pair) || return T(NaN)
        pair > zero(T) || break
        previous = min(previous, pair)
        pairsum += previous
    end
    tau = T(2) * pairsum - one(T)
    isfinite(tau) && tau > zero(T) ? tau : T(NaN)
end

function _run_ess_result(runs; estimator=:geyer, autocov=:direct, window=5,
    max_lag_work=100_000_000, T=_floattype(eltype(runs.values)))
    n, m = runs.n, length(runs.ranges)
    n >= 4 || return (T(NaN), :insufficient_draws)
    any(r -> all(i -> runs.values[i] == runs.values[first(r)], r), runs.ranges) &&
        return (T(NaN), :constant)
    origin, scale = _sample_origin_scale(runs.values, T)
    means, variances = Vector{T}(undef, m), Vector{T}(undef, m)
    z = Vector{T}(undef, length(runs.values))
    for (c, r) in enumerate(runs.ranges)
        chain = view(runs.values, r)
        chain_origin, _ = _sample_origin_scale(chain, T)
        centered = view(z, r)
        centered .= _centered_value.(chain, chain_origin, T) ./ scale
        weights = view(runs.weights, r)
        center = _weighted_mean(centered, weights)
        means[c] = _centered_value(chain_origin, origin, T) / scale + center
        variances[c] = _weighted_var(centered, weights)
        centered .-= center
    end
    W = Statistics.mean(variances)
    varplus = T(n - 1) / T(n) * W
    m > 1 && (varplus += Statistics.var(means))
    isfinite(varplus) && varplus > zero(T) || return (T(NaN), :estimator_failed)
    covariance = _autocov_function(z, runs, autocov, max_lag_work)
    rho = m == 1 ? k -> covariance(k) / varplus :
        k -> one(T) - (W - covariance(k)) / varplus
    tau = _window_tau(rho, n, T, estimator, window)
    isfinite(tau) ? (T(n) * T(m) / tau, :ok) : (T(NaN), :estimator_failed)
end

function _run_order_statistic(values, weights, index, order=sortperm(values))
    cumulative = 0
    for i in order
        cumulative += weights[i]
        cumulative >= index && return values[i]
    end
    last(values)
end

function _run_indicator(runs, prob, ::Type{T}) where {T}
    total = foldl(Base.Checked.checked_add, runs.weights; init=0)
    probability = Rational{BigInt}(prob)
    rank = 1 + Int(fld((total - 1) * numerator(probability), denominator(probability)))
    cutoff = _run_order_statistic(runs.values, runs.weights, rank)
    map(v -> T(v <= cutoff), runs.values)
end

function _transformed_run_ess(runs, ::Type{T}; kind, prob, interval, kwargs...) where {T}
    if kind === :tail
        lower = _transformed_run_ess(runs, T; kind=:quantile, prob=1//20, interval, kwargs...)
        upper = _transformed_run_ess(runs, T; kind=:quantile, prob=19//20, interval, kwargs...)
        lower[2] === :ok || return lower
        upper[2] === :ok || return upper
        return (min(lower[1], upper[1]), :ok)
    end
    values = kind === :bulk ? _rank_normalize(runs.values, runs.weights, T) :
        kind === :quantile ? _run_indicator(runs, prob, T) :
        kind === :interval ? map(v -> T(interval[1] <= v <= interval[2]), runs.values) : runs.values
    _run_ess_result(merge(runs, (; values)); T, kwargs...)
end
