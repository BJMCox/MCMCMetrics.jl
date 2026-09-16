function _precision_parameter_result(d, p; estimand=:mean, split=true,
    estimator=:geyer, autocov=:direct, prob=0.5, window=5, batch_size=nothing,
    max_lag_work=100_000_000)
    T = _floattype(d)
    if estimator in (:batchmeans, :overlapping)
        return _batch_mcse_result(d, p, estimator, batch_size, max_lag_work)
    end
    all(==(first(d.lengths)), d.lengths) || return (T(NaN), :unequal_chain_lengths)
    first(d.lengths) ÷ (split ? 2 : 1) >= 4 || return (T(NaN), :insufficient_draws)
    if estimand === :mean && isnothing(d.counts) && estimator === :geyer && autocov === :direct
        return _mcse_parameter_result(d, p; split)
    end
    full = _parameter_runs(d, p, false)
    runs = _parameter_runs(d, p, split)
    z, scale = _scaled_samples(full.values, T)
    if estimand in (:quantile, :median)
        probability = estimand === :median ? 1//2 : prob
        effective, status = _transformed_run_ess(runs, T; kind=:quantile,
            prob=probability, interval=nothing, estimator, autocov, window, max_lag_work)
        status === :ok || return (effective, status)
        a, b = effective * T(probability) + one(T), effective * T(1 - probability) + one(T)
        tail = SpecialFunctions.erfc(inv(sqrt(T(2)))) / T(2)
        lo = T(first(SpecialFunctions.beta_inc_inv(a, b, tail)))
        hi = T(first(SpecialFunctions.beta_inc_inv(a, b, one(T) - tail)))
        width = _run_quantile(z, full.weights, hi) - _run_quantile(z, full.weights, lo)
        width > zero(T) || return (T(NaN), :collapsed_interval)
        value = (width / T(2)) * scale
    elseif estimand === :mean
        effective, status = _run_ess_result(runs; estimator, autocov, window, max_lag_work, T)
        status === :ok || return (effective, status)
        value = sqrt(_weighted_var(z, full.weights) / effective) * scale
    else
        # Influence process for variance is (x - mu)^2 - sigma^2.
        # Its additive constant has no effect on ESS or variance.
        center = _weighted_mean(z, full.weights)
        squares = abs2.(z .- center)
        origin, _ = _sample_origin_scale(full.values, T)
        transformed = [abs2(_centered_value(v, origin, T) / scale - center) for v in runs.values]
        effective, status = _run_ess_result(merge(runs, (; values=transformed));
            estimator, autocov, window, max_lag_work, T)
        status === :ok || return (effective, status)
        error = sqrt(_weighted_var(squares, full.weights) / effective)
        value = estimand === :variance ? (error * scale) * scale :
            error / (T(2) * sqrt(_weighted_mean(squares, full.weights))) * scale
    end
    isfinite(value) ? (value, :ok) : (T(NaN), :estimator_failed)
end

function _run_quantile(values::AbstractVector{T}, weights, probability) where {T}
    total = foldl(Base.Checked.checked_add, weights; init=0)
    position = (total - 1) * Rational{BigInt}(probability)
    index = 1 + Int(fld(numerator(position), denominator(position)))
    fraction = T(position - (index - 1))
    order = sortperm(values)
    lo = _run_order_statistic(values, weights, index, order)
    hi = _run_order_statistic(values, weights, min(index + 1, total), order)
    (one(T) - fraction) * lo + fraction * hi
end

function _compensated_add(total, correction, value)
    adjusted = value - correction
    next = total + adjusted
    next, (next - total) - adjusted
end

function _stable_run_mean(values::AbstractVector{T}, weights, n) where {T}
    total = correction = zero(T)
    for i in eachindex(values, weights)
        total, correction = _compensated_add(total, correction, T(weights[i]) / T(n) * values[i])
    end
    total
end

# Clip before choosing numerical origin/scale: discarded values must have no
# effect on retained batch arithmetic. The inputs are owned by this caller.
function _clip_batch_runs!(runs, n)
    before = 0
    for i in eachindex(runs.weights)
        after = before + runs.weights[i]
        if after >= n
            runs.weights[i] = n - before
            resize!(runs.weights, i)
            resize!(runs.values, i)
            break
        end
        before = after
    end
    (; values=runs.values, weights=runs.weights, n)
end

"""
    _foreach_run_batch(f, values, weights, b; step=b, max_work=100_000_000)

Call `f(mean, multiplicity)` in window order for full width-b windows, starting
at offsets 0, step, ... . Values must already be centered/scaled and weights
must be positive integer run lengths. Caller clips discarded BM tails first.
Consecutive windows wholly inside a run are grouped; mixed windows integrate
their local overlaps with compensated arithmetic in eltype(values).

Storage is O(stored runs), never O(logical draws or windows). Both the number
of logical windows and the number of visited mixed-window run overlaps are
bounded by max_work; exhaustion throws _WorkLimitError before excess work.
"""
function _foreach_run_batch(f::F, values::AbstractVector{T}, weights, b;
    step=b, max_work=100_000_000) where {F,T}
    ends = cumsum(weights)
    last_start = last(ends) - b
    windows = last_start ÷ step + 1
    windows <= max_work || throw(_WorkLimitError("batch work exceeds max_lag_work"))
    start = 0
    i = 1
    work = 0
    while start <= last_start
        while ends[i] <= start
            i += 1
        end
        if start + b <= ends[i]
            multiplicity = (min(ends[i] - b, last_start) - start) ÷ step + 1
            f(values[i], multiplicity)
            start += multiplicity * step
            continue
        end
        total = correction = zero(T)
        j, position = i, start
        while position < start + b
            work < max_work || throw(_WorkLimitError("batch overlap work exceeds max_lag_work"))
            work += 1
            stop = min(ends[j], start + b)
            term = T(stop - position) / T(b) * values[j]
            total, correction = _compensated_add(total, correction, term)
            position = stop
            j += 1
        end
        f(total, 1)
        start += step
    end
    nothing
end

function _batch_mcse_result(d, p, estimator, batch_size, max_lag_work)
    T = _floattype(d)
    b = isnothing(batch_size) ? max(1, isqrt(minimum(d.lengths))) : batch_size
    b isa Integer && b >= 1 || throw(ArgumentError("batch_size must be a positive integer"))
    all(n -> n ÷ b >= 2, d.lengths) || return (T(NaN), :insufficient_draws)
    # Equal-length chains are not needed: combine independent chain mean
    # variances with weights proportional to their retained draw counts.
    lengths = [estimator === :batchmeans ? n ÷ b * b : n for n in d.lengths]
    retained = foldl(Base.Checked.checked_add, lengths; init=0)
    error = zero(T)
    work = 0
    for c in eachindex(d.chains)
        n = lengths[c]
        runs = _clip_batch_runs!(_single_chain_runs(d, p, c), n)
        origin, scale = _sample_origin_scale(runs.values, T)
        iszero(scale) && return (T(NaN), :constant)
        z = [_centered_value(v, origin, T) / scale for v in runs.values]
        a = n ÷ b
        batches = estimator === :batchmeans ? a : n - b + 1
        batches <= max_lag_work - work || throw(_WorkLimitError("batch work exceeds max_lag_work"))
        center = _stable_run_mean(z, runs.weights, n)
        sumsq, correction = Ref(zero(T)), Ref(zero(T))
        step = estimator === :batchmeans ? b : 1
        _foreach_run_batch(z, runs.weights, b; step, max_work=max_lag_work-work) do batchmean, multiplicity
            term = T(multiplicity) / T(batches) * abs2(batchmean - center)
            sumsq[], correction[] = _compensated_add(sumsq[], correction[], term)
        end
        work += batches
        longrun = estimator === :batchmeans ? T(b) * (T(a) / T(a - 1)) * sumsq[] :
            T(n) / T(n - b) * T(b) * sumsq[]
        isfinite(longrun) && longrun > zero(T) || return (T(NaN), :estimator_failed)
        chain_error = sqrt(longrun / T(n)) * scale
        error = hypot(error, T(n) / T(retained) * chain_error)
    end
    isfinite(error) && error > zero(T) ? (error, :ok) : (T(NaN), :estimator_failed)
end

"""
    diagnostic_curve(x; prefixes, diagnostic=ess, drawdim=1, chaindim, counts=nothing, kwargs...)

Evaluate a parameter diagnostic (`ess`, `mcse`, or `rhat`) at the requested
logical prefix lengths of every chain. Return `(prefixes, values)` with one
parameter-shaped value per requested prefix, in supplied order. Compressed
prefixes clip integer runs without expansion. No prefix lengths are invented.
"""
function diagnostic_curve(x; prefixes, diagnostic=ess, drawdim=1,
    chaindim=_default_chaindim(x), counts=nothing, kwargs...)
    d = _draws(x; drawdim, chaindim, counts)
    lengths = collect(prefixes)
    all(n -> n isa Integer && 1 <= n <= minimum(d.lengths), lengths) ||
        throw(ArgumentError("prefixes must be positive integers within every chain"))
    values = map(lengths) do n
        if isnothing(d.counts)
            chains = [view(chain, 1:n, :) for chain in d.chains]
            value = diagnostic(chains; kwargs...)
        else
            weights = map(eachindex(d.chains)) do c
                remaining = n
                map(d.counts[c]) do w
                    retained = min(w, remaining)
                    remaining -= retained
                    retained
                end
            end
            value = diagnostic(d.chains; counts=weights, kwargs...)
        end
        _parameter_result(vec(value), d.shape)
    end
    (; prefixes=lengths, values)
end

"""
    cost_normalize(effective, cost)

Divide ESS (a number or parameter array) by a caller-supplied finite positive
scalar cost, such as seconds or log-density evaluations. No timing is inferred.
"""
function cost_normalize(effective, cost::Real)
    isfinite(cost) && cost > 0 || throw(ArgumentError("cost must be finite and positive"))
    effective ./ cost
end
