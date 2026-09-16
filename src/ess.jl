"""
    ess(x; kind=:bulk, split=true, estimator=:geyer, autocov=:direct,
        prob=0.95, interval=nothing, drawdim=1, chaindim, counts=nothing)

Estimate effective sample size per parameter. Kinds are `:bulk` (normal scores
of pooled ranks), `:tail` (the smaller 5% and 95% quantile ESS), `:mean`
(untransformed draws), `:quantile` (the indicator below `prob`'s pooled
empirical quantile), and `:interval` (membership in the closed `interval`).
Quantile cutoffs use the supplied probability without narrowing its precision.
Indicators and diagnostic arithmetic retain the observation floating type.

By default, split each chain in half and discard its middle draw when odd.
Compute ranks and quantile thresholds after splitting. At least four retained
draws per split chain are required. A constant chain or transformed chain
returns a typed `NaN`. Integer repetition counts retain the exact run order,
including runs crossing split boundaries, without expanded storage.

The direct autocovariance divides by the chain length at every lag. The Geyer
estimator truncates at the first nonpositive pair of autocorrelations, then
reduces each retained pair to the minimum of all preceding pairs (IMS).
It uses every available complete pair, without a lag cap or ESS cap.
A terminal even-lag correction is not part of this plain IMS convention.
A nonpositive estimated integrated time returns `NaN`. Valid antithetic ESS
may exceed the draw count. The paired-sequence justification requires a
reversible chain and finite variance of the diagnosed process.

`estimator=:sokal` instead uses the first positive window sum
`tau(M) = 1 + 2sum(rho(1:M))` satisfying `M >= window*tau(M)` (`window=5`).
No such window returns `NaN`. This names the full-IACT convention; some
references call half this quantity the integrated autocorrelation time.

`autocov=:fft` requires loading FFTW and dense draws; it changes only the
autocovariance computation. Direct arithmetic remains the default. Compressed
and Sokal direct calculations limit cursor work to `max_lag_work=100_000_000`
per parameter and transform (two stored runs per lag is the charged upper
bound). Exceeding the budget throws a work-limit exception; increase it explicitly to permit more
work. Estimators never truncate silently or expand repetitions.

See Geyer (1992), doi:10.1214/ss/1177011137, and Vehtari et al. (2021),
doi:10.1214/20-BA1221. The latter supplies the multi-chain variance estimator.
"""
function ess(x; kind::Symbol=:bulk, split::Bool=true, estimator::Symbol=:geyer,
    autocov::Symbol=:direct, prob::Real=0.95, interval=nothing,
    window::Real=5, max_lag_work::Integer=100_000_000,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    _check_ess_options(kind, estimator, autocov, prob, interval)
    _check_window(window, max_lag_work)
    d = _draws(x; drawdim, chaindim, counts)
    _check_fft_counts(d, autocov)
    values = [_ess_parameter_result(d, p; kind, split, prob, interval,
        estimator, autocov, window, max_lag_work)[1]
        for p in 1:prod(d.shape)]
    _parameter_result(values, d.shape)
end

"""
    mcse(x; estimand=:mean, split=true, estimator=:geyer, autocov=:direct,
         prob=0.5, window=5, batch_size=nothing, kwargs...)

Estimate the Monte Carlo standard error of the mean as the pooled raw sample
standard deviation divided by the square root of raw mean ESS. All raw draws
contribute to the standard deviation, including middle draws discarded by ESS
splitting. The same finite-variance and reversibility assumptions as [`ess`](@ref)
apply. Integer counts are supported without expansion.

For `estimand=:variance` use the sample variance (divisor N-1) of the centered
squared influence process, divided by its own ESS; `:sd` divides that MCSE
by twice the pooled population SD. These require a finite fourth moment and
a CLT for that process.

For `:quantile` (`prob` strictly inside (0,1)) or `:median`, use indicator ESS
E to form Beta(E*p+1,E*(1-p)+1) bounds at normal -1 and +1 probabilities.
Map those bounds through pooled raw type-7 empirical quantiles and return
half the interval width. Collapsed tied intervals return `NaN`; this method
targets regular continuous quantiles, not discrete quantile uncertainty.

`estimator=:batchmeans` and `:overlapping` estimate mean MCSE from independent
chain batch means, without chain splitting (the `split` keyword is ignored).
The default batch size is `max(1, isqrt(shortest_logical_chain_length))`.
Each chain needs at least two full batches. Nonoverlapping batches discard
the incomplete final batch; overlapping batches use every full window and
the full-chain mean. Chain contributions are weighted by their retained
lengths. See Flegal and Jones (2010), doi:10.1214/09-AOS735.
Batch work limits both full windows and local run-overlap visits by
`max_lag_work`, without storing windows.
"""
function mcse(x; estimand::Symbol=:mean, split::Bool=true, estimator::Symbol=:geyer,
    autocov::Symbol=:direct, prob::Real=0.5, window::Real=5,
    batch_size=nothing, max_lag_work::Integer=100_000_000,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    estimand in (:mean, :variance, :sd, :quantile, :median) ||
        throw(ArgumentError("unsupported MCSE estimand"))
    batch = estimator in (:batchmeans, :overlapping)
    if batch
        estimand === :mean || throw(ArgumentError("batch estimators support mean MCSE"))
        autocov === :direct || throw(ArgumentError("batch estimators do not use FFT autocovariances"))
    else
        _check_geyer(estimator, autocov)
    end
    _check_window(window, max_lag_work)
    estimand in (:quantile, :median) && _check_ess_options(:quantile, :geyer, :direct, prob, nothing)
    d = _draws(x; drawdim, chaindim, counts)
    _check_fft_counts(d, autocov)
    values = [_precision_parameter_result(d, p; estimand, split, estimator,
        autocov, prob, window, batch_size, max_lag_work)[1] for p in 1:prod(d.shape)]
    _parameter_result(values, d.shape)
end

"""
    autocor(x; lags=nothing, autocov=:direct, max_lag_work=100_000_000,
            drawdim=1, chaindim, counts=nothing)

Return direct autocorrelations with shape lag × chain × parameter axes.
Omit the chain axis for a single-chain array input. Default lags are
`0:min(100, shortest_chain_length - 1)`. Each chain uses its own mean.
Autocovariances use divisor `n` at every lag and normalize by lag-zero
autocovariance. Constant chains return `NaN`, including at lag zero.
Integer counts use exact run overlaps. FFT requires dense input and FFTW.
Compressed direct work uses the cursor budget described by [`ess`](@ref),
separately for each chain and parameter. Dense direct work remains uncapped.
"""
function autocor(x; lags=nothing, autocov::Symbol=:direct,
    max_lag_work::Integer=100_000_000, drawdim=1,
    chaindim=_default_chaindim(x), counts=nothing)
    d = _draws(x; drawdim, chaindim, counts)
    _check_geyer(:geyer, autocov)
    _check_window(5, max_lag_work)
    _check_fft_counts(d, autocov)
    lagvalues = isnothing(lags) ? collect(0:min(100, minimum(d.lengths) - 1)) : collect(lags)
    all(k -> k isa Integer && 0 <= k < minimum(d.lengths), lagvalues) ||
        throw(ArgumentError("lags must be integers within each chain"))
    T = _floattype(d)
    values = Array{T}(undef, length(lagvalues), length(d.chains), prod(d.shape))
    isempty(lagvalues) && return reshape(values, (0, _chain_result_shape(x, d, chaindim)...))
    for p in 1:prod(d.shape), c in eachindex(d.chains)
        if isnothing(d.counts) && autocov === :direct
            z, _ = _scaled_samples(view(d.chains[c], :, p), T)
            z .-= Statistics.mean(z)
            v = _mean_autocov(z, 0)
            for (i, k) in enumerate(lagvalues)
                values[i, c, p] = v > zero(T) ? _mean_autocov(z, k) / v : T(NaN)
            end
            continue
        end
        runs = _single_chain_runs(d, p, c)
        z, _ = _scaled_samples(runs.values, T)
        z .-= _weighted_mean(z, runs.weights)
        covariance = _autocov_function(z, runs, autocov, max_lag_work)
        v = covariance(0)
        for (i, k) in enumerate(lagvalues)
            values[i, c, p] = v > zero(T) ? covariance(k) / v : T(NaN)
        end
    end
    reshape(values, (length(lagvalues), _chain_result_shape(x, d, chaindim)...))
end

"""
    iact(x; estimator=:geyer, autocov=:direct, drawdim=1, chaindim, counts=nothing)

Estimate integrated autocorrelation time separately for each unsplit chain
and parameter. Apply Geyer's initial positive and monotone sequence to
`autocovariance(lag) / autocovariance(0)`. Return a scalar for one scalar
chain. Otherwise return chain × parameter axes, omitting the chain axis for
a single-chain array input. The assumptions and failure rules of [`ess`](@ref)
apply. Combined multi-chain ESS uses a different variance estimator.
Integer counts use exact run overlaps. `estimator=:sokal`, `window=5`,
`autocov=:fft`, and `max_lag_work` follow [`ess`](@ref).
"""
function iact(x; estimator::Symbol=:geyer, autocov::Symbol=:direct,
    window::Real=5, max_lag_work::Integer=100_000_000,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    _check_geyer(estimator, autocov)
    d = _draws(x; drawdim, chaindim, counts)
    _check_window(window, max_lag_work)
    _check_fft_counts(d, autocov)
    T = _floattype(d)
    values = Matrix{T}(undef, length(d.chains), prod(d.shape))
    for p in 1:prod(d.shape), c in eachindex(d.chains)
        if isnothing(d.counts) && estimator === :geyer && autocov === :direct
            chain = view(d.chains[c], :, p)
            effective, _ = _geyer_ess_result(reshape(chain, :, 1))
            values[c, p] = T(length(chain)) / effective
            continue
        end
        runs = _single_chain_runs(d, p, c)
        effective, _ = _run_ess_result(runs; estimator, autocov, window, max_lag_work)
        values[c, p] = T(runs.n) / effective
    end
    _parameter_result(values, _chain_result_shape(x, d, chaindim))
end

_chain_result_shape(x, d, chaindim) = isnothing(chaindim) ? d.shape : (length(d.chains), d.shape...)
_chain_result_shape(::AbstractVector{<:AbstractMatrix}, d, chaindim) = (length(d.chains), d.shape...)

function _check_geyer(estimator, autocov)
    estimator in (:geyer, :sokal) || throw(ArgumentError("estimator must be :geyer or :sokal"))
    autocov in (:direct, :fft) || throw(ArgumentError("autocov must be :direct or :fft"))
end

function _check_ess_options(kind, estimator, autocov, prob, interval)
    _check_geyer(estimator, autocov)
    kind in (:bulk, :tail, :mean, :quantile, :interval) || throw(ArgumentError("unsupported ESS kind"))
    if kind === :quantile
        isfinite(prob) && 0 < prob < 1 || throw(ArgumentError("prob must lie strictly between zero and one"))
    elseif kind === :interval
        interval isa Tuple{Real,Real} && all(isfinite, interval) && interval[1] <= interval[2] ||
            throw(ArgumentError("interval must contain ordered finite bounds"))
    end
end

function _ess_samples(d, p; split::Bool)
    n = first(d.lengths)
    nout = split ? n ÷ 2 : n
    x = Matrix{eltype(first(d.chains))}(undef, nout, length(d.chains) * (split ? 2 : 1))
    for c in eachindex(d.chains)
        if split
            x[:, 2c - 1] .= view(d.chains[c], 1:nout, p)
            x[:, 2c] .= view(d.chains[c], (n - nout + 1):n, p)
        else
            x[:, c] .= view(d.chains[c], :, p)
        end
    end
    x
end

function _scaled_samples(x, ::Type{T}) where {T}
    origin, scale = _sample_origin_scale(x, T)
    z = map(v -> _centered_value(v, origin, T), x)
    iszero(scale) || (z ./= scale)
    z, scale
end

function _ess_parameter_result(d, p; kind=:bulk, split=true, prob=0.95,
    interval=nothing, samples=nothing, ranked=nothing,
    estimator=:geyer, autocov=:direct, window=5, max_lag_work=100_000_000)
    T = _floattype(d)
    all(==(first(d.lengths)), d.lengths) || return (T(NaN), :unequal_chain_lengths)
    first(d.lengths) ÷ (split ? 2 : 1) >= 4 || return (T(NaN), :insufficient_draws)
    if !isnothing(d.counts) || estimator !== :geyer || autocov !== :direct
        runs = _parameter_runs(d, p, split)
        return _transformed_run_ess(runs, T; kind, prob, interval, estimator,
            autocov, window, max_lag_work)
    end
    x = isnothing(samples) ? _ess_samples(d, p; split) : samples
    if kind === :bulk
        z = isnothing(ranked) ? reshape(_rank_normalize(vec(x), ones(Int, length(x))), size(x)) : ranked
    elseif kind === :quantile
        z = _quantile_indicator(x, prob, T)
    elseif kind === :tail
        lower = _geyer_ess_result(_quantile_indicator(x, 1//20, T))
        upper = _geyer_ess_result(_quantile_indicator(x, 19//20, T))
        lower[2] === :ok || return lower
        upper[2] === :ok || return upper
        return (min(lower[1], upper[1]), :ok)
    elseif kind === :interval
        z = map(v -> T(interval[1] <= v <= interval[2]), x)
    else
        z = x
    end
    _geyer_ess_result(z)
end

# Type-7 empirical quantiles interpolate adjacent order statistics. For the
# indicator x <= quantile, only the lower order statistic is needed. Comparing
# original observations avoids losing integer distinctions in a float threshold.
function _quantile_indicator(x, prob::Real, ::Type{T}) where {T}
    # The cutoff is an integer rank. Exact probability arithmetic prevents
    # multiplication or addition from rounding it to the next observation.
    probability = Rational{BigInt}(prob)
    index = 1 + Int(fld((length(x) - 1) * numerator(probability), denominator(probability)))
    cutoff = partialsort!(collect(vec(x)), index)
    map(v -> T(v <= cutoff), x)
end

function _mean_autocov(z::AbstractArray{T}, lag::Integer) where {T}
    n, m = size(z, 1), size(z, 2)
    result = zero(T)
    @inbounds for c in 1:m, i in 1:(n - lag)
        result += z[i, c] * z[i + lag, c]
    end
    result / T(n) / T(m)
end

function _geyer_ess_result(x::AbstractMatrix)
    T = _floattype(eltype(x))
    n, m = size(x)
    n >= 4 || return (T(NaN), :insufficient_draws)
    origin, scale = _sample_origin_scale(x, T)
    means, variances = Vector{T}(undef, m), Vector{T}(undef, m)
    z = Matrix{T}(undef, n, m)
    for c in 1:m
        chain = view(x, :, c)
        chain_origin, chain_scale = _sample_origin_scale(chain, T)
        iszero(chain_scale) && return (T(NaN), :constant)
        centered = view(z, :, c)
        # Center within each chain before scaling. A distant chain must not
        # round a small chain's distinct values to the same global residual.
        centered .= _centered_value.(chain, chain_origin, T) ./ scale
        local_mean = Statistics.mean(centered)
        variances[c] = Statistics.var(centered; mean=local_mean)
        means[c] = _centered_value(chain_origin, origin, T) / scale + local_mean
        centered .-= local_mean
    end
    W = Statistics.mean(variances)
    varplus = T(n - 1) / T(n) * W
    m > 1 && (varplus += Statistics.var(means))
    isfinite(varplus) && varplus > zero(T) || return (T(NaN), :estimator_failed)
    # For a single unsplit chain use the ordinary normalized autocovariance.
    rho = m == 1 ? k -> _mean_autocov(z, k) / varplus :
        k -> one(T) - (W - _mean_autocov(z, k)) / varplus
    tau = _window_tau(rho, n, T, :geyer, 5)
    isfinite(tau) && tau > zero(T) || return (T(NaN), :estimator_failed)
    (T(n) * T(m) / tau, :ok)
end

function _mcse_parameter_result(d, p; split=true, samples=nothing)
    !isnothing(d.counts) && return _precision_parameter_result(d, p; split)
    effective, status = _ess_parameter_result(d, p; kind=:mean, split, samples)
    status === :ok || return (effective, status)
    T = _floattype(d)
    pooled = isnothing(samples) || (split && isodd(first(d.lengths))) ?
        _ess_samples(d, p; split=false) : samples
    z, scale = _scaled_samples(pooled, T)
    value = (sqrt(Statistics.var(vec(z))) / sqrt(effective)) * scale
    isfinite(value) ? (value, :ok) : (T(NaN), :estimator_failed)
end
