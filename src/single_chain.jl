function _chain_records(f, x; drawdim, chaindim, counts)
    d = _draws(x; drawdim, chaindim, counts)
    isnothing(d.counts) || throw(ArgumentError("classical window diagnostics require uncompressed draws"))
    records = [f(view(chain, :, p), _floattype(d)) for chain in d.chains, p in 1:prod(d.shape)]
    shape = _chain_result_shape(x, d, chaindim)
    isempty(shape) ? only(records) : StructArray(reshape(records, shape))
end

"""
    geweke(x; first_fraction=0.1, last_fraction=0.5, estimator=:geyer, kwargs...)

Compare disjoint early and late windows of each chain using their separate
mean MCSEs. Return `zscore`, a two-sided asymptotic normal `pvalue`, window
lengths and `status`. Windows use floor(fraction * length) and each require
at least four draws. A non-significant result does not establish convergence.
The windows must be sufficiently
separated and the spectral estimator's finite-variance assumptions must hold.
Accept array axes; return chain × parameter records. Integer counts are not
supported for these window diagnostics. Additional keywords go to `mcse`.

Reference: Geweke (1992), Evaluating the accuracy of sampling-based approaches
to the calculation of posterior moments.
"""
function geweke(x; first_fraction=0.1, last_fraction=0.5, estimator=:geyer,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing, kwargs...)
    0 < first_fraction < 1 && 0 < last_fraction < 1 && first_fraction + last_fraction <= 1 ||
        throw(ArgumentError("early and late fractions must be positive and nonoverlapping"))
    _chain_records(x; drawdim, chaindim, counts) do chain, T
        n = length(chain)
        a, b = floor(Int, n * first_fraction), floor(Int, n * last_fraction)
        invalid = (zscore=T(NaN), pvalue=T(NaN), n_first=a, n_last=b,
            estimator=estimator, status=:insufficient_draws)
        min(a, b) >= 4 || return invalid
        early, late = view(chain, 1:a), view(chain, n-b+1:n)
        origin_a, scale_a = _sample_origin_scale(early, T)
        origin_b, scale_b = _sample_origin_scale(late, T)
        scale = max(scale_a, scale_b)
        iszero(scale) && origin_a == origin_b && return merge(invalid, (status=:constant,))
        min(scale_a, scale_b) > zero(T) || return merge(invalid, (status=:estimator_failed,))
        za = [_centered_value(v, origin_a, T) / scale_a for v in early]
        zb = [_centered_value(v, origin_b, T) / scale_b for v in late]
        error = hypot((scale_a / scale) * mcse(za; split=false, estimator, kwargs...),
            (scale_b / scale) * mcse(zb; split=false, estimator, kwargs...))
        error > zero(T) && isfinite(error) || return merge(invalid, (status=:estimator_failed,))
        difference = _centered_value(origin_a, origin_b, T)
        offset = isfinite(difference) ? difference / scale : T(origin_a) / scale - T(origin_b) / scale
        score = (offset + (scale_a / scale) * Statistics.mean(za) -
            (scale_b / scale) * Statistics.mean(zb)) / error
        (zscore=score, pvalue=SpecialFunctions.erfc(abs(score) / sqrt(T(2))),
            n_first=a, n_last=b, estimator=estimator, status=:ok)
    end
end

# Brownian-bridge integrated-square CDF (Csorgo and Faraway, 1996).
# Scaled Bessel K avoids the exponentially small factors underflowing twice.
function _cramer_cdf(q::T) where {T}
    q <= zero(T) && return zero(T)
    q >= T(10) && return one(T)
    total, coefficient = zero(T), one(T)
    for k in 0:63
        a = T(4k + 1)
        u = a * a / (T(16) * q)
        term = isfinite(u) ? coefficient * sqrt(a) * exp(-T(2) * u) *
            SpecialFunctions.besselkx(T(1) / T(4), u) : zero(T)
        total += term
        k > 0 && term <= eps(T) * total && break
        coefficient *= (T(k) + T(1) / T(2)) / T(k + 1)
    end
    clamp(total / (T(pi) * sqrt(q)), zero(T), one(T))
end

"""
    heidelberger_welch(x; alpha=0.05, rtol=0.1, estimator=:geyer, kwargs...)

Apply the Brownian-bridge stationarity test after discarding 0%, 10%, ...,
50% of each chain, stopping at the first non-rejection. Estimate its reference
long-run variance from the original last half. Report a two-sided
`1-alpha` mean interval half-width from the retained segment and its ratio
to the absolute mean. Near-zero means make relative precision unsuitable.

Return numeric records including `stationarity`, `precision`, `burnin`,
`statistic`, `pvalue`, `mean`, `halfwidth`, `estimator` and `status`.
Use at least 20 draws. The named spectral estimator replaces coda's AR fit;
the truncation grid includes exactly 50%. These finite-sample conventions
are explicit. Repeated checking does not yield a sequential error guarantee.
Reference: Heidelberger and Welch (1983), doi:10.1287/opre.31.6.1109.
"""
function heidelberger_welch(x; alpha=0.05, rtol=0.1, estimator=:geyer,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing, kwargs...)
    0 < alpha < 1 && isfinite(rtol) && rtol > 0 || throw(ArgumentError("invalid test probabilities or tolerance"))
    _chain_records(x; drawdim, chaindim, counts) do chain, T
        n = length(chain)
        invalid = (stationarity=false, precision=false, burnin=0, statistic=T(NaN),
            pvalue=T(NaN), mean=T(NaN), halfwidth=T(NaN), estimator=estimator,
            status=:insufficient_draws)
        n >= 20 || return invalid
        all(==(first(chain)), chain) && return merge(invalid, (status=:constant,))
        tail, tail_scale = _scaled_samples(view(chain, n÷2+1:n), T)
        lrv_root = sqrt(T(length(tail))) * mcse(tail; split=false, estimator, kwargs...)
        isfinite(lrv_root) && lrv_root > zero(T) || return merge(invalid, (status=:estimator_failed,))
        statistic, pvalue = T(NaN), T(NaN)
        for tenth in 0:5
            burnin = fld(n * tenth, 10)
            segment = view(chain, burnin+1:n)
            origin, scale = _sample_origin_scale(segment, T)
            iszero(scale) && return merge(invalid, (burnin, status=:estimator_failed))
            y = [_centered_value(v, origin, T) / scale for v in segment]
            mean = Statistics.mean(y)
            bridge, area = zero(T), zero(T)
            for value in y
                bridge += value - mean
                area += abs2(bridge)
            end
            # Rescale the square root; squaring physical scales can overflow.
            numerator, exponent = frexp(scale)
            denominator, tail_exponent = frexp(tail_scale)
            root = (sqrt(area) / T(length(y))) / lrv_root
            statistic = abs2(ldexp(root * (numerator / denominator), exponent - tail_exponent))
            pvalue = one(T) - _cramer_cdf(statistic)
            if pvalue > T(alpha)
                error = mcse(y; split=false, estimator, kwargs...)
                halfwidth = (sqrt(T(2)) * SpecialFunctions.erfcinv(T(alpha)) * error) * scale
                center = muladd(scale, mean, T(origin))
                status = isfinite(halfwidth) && isfinite(center) ? :ok : :estimator_failed
                return (stationarity=true, precision=status === :ok && halfwidth <= T(rtol) * abs(center),
                    burnin, statistic, pvalue, mean=center, halfwidth, estimator=estimator, status)
            end
        end
        merge(invalid, (burnin=n÷2, statistic, pvalue, status=:nonstationary))
    end
end

"""
    raftery_lewis(x; prob=0.95, atol=0.005, confidence=0.95, tolerance=0.001, kwargs...)

Estimate quantile-probability run length from a two-state Markov approximation.
Choose thinning by the first-versus-second-order likelihood-ratio BIC with
two extra parameters. Return `thinning`, `burnin`, `total`, `nmin`,
`dependence_factor` and `status` per chain and parameter. `atol` bounds
probability error, not error in the quantile's units. Pilot length must meet
the IID lower bound. Degenerate transitions produce an unavailable result.
This diagnoses the selected quantile and does not prescribe sampler thinning.
Reference: Raftery and Lewis (1992), How many iterations in the Gibbs sampler?
"""
function raftery_lewis(x; prob=0.95, atol=0.005, confidence=0.95, tolerance=0.001,
    drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    0 < prob < 1 && 0 < atol < 1 && 0 < confidence < 1 && 0 < tolerance < 1 ||
        throw(ArgumentError("probabilities and tolerances must lie between zero and one"))
    _chain_records(x; drawdim, chaindim, counts) do chain, T
        r, coverage, epsilon = T(atol), T(confidence), T(tolerance)
        all(v -> zero(T) < v < one(T), (r, coverage, epsilon)) ||
            throw(ArgumentError("tolerances and confidence must remain interior in the observation precision"))
        normal = sqrt(T(2)) * SpecialFunctions.erfinv(coverage)
        # Preserve keyword tail probabilities before narrowing; the IID bound
        # is strictly positive even if its exponential underflows.
        log_lower = T(log(prob)) + T(log(1-prob)) + T(2) * (log(normal) - log(r))
        lower = max(one(T), ceil(exp(log_lower)))
        nmin = isfinite(lower) && lower < T(typemax(Int)) ? Int(lower) : typemax(Int)
        invalid = (thinning=0, burnin=T(NaN), total=T(NaN), nmin,
            dependence_factor=T(NaN), status=:insufficient_draws)
        length(chain) >= max(4, nmin) || return invalid
        indicator = _quantile_indicator(chain, prob, T)
        all(==(first(indicator)), indicator) && return merge(invalid, (status=:constant,))
        for stride in 1:((length(chain)-1)÷3)
            y = view(indicator, 1:stride:length(chain))
            triples = zeros(Int, 2, 2, 2)
            for i in 1:length(y)-2
                triples[Int(y[i])+1, Int(y[i+1])+1, Int(y[i+2])+1] += 1
            end
            deviance = zero(T)
            for a in 1:2, b in 1:2, c in 1:2
                observed = triples[a, b, c]
                if observed > 0
                    expected = T(sum(@view triples[a,b,:])) *
                        (T(sum(@view triples[:,b,c])) / T(sum(@view triples[:,b,:])))
                    deviance += T(2observed) * log(T(observed) / expected)
                end
            end
            deviance < T(2) * log(T(length(y)-2)) || continue
            transitions = zeros(Int, 2, 2)
            for i in 1:length(y)-1
                transitions[Int(y[i])+1, Int(y[i+1])+1] += 1
            end
            a = T(transitions[1,2]) / T(sum(@view transitions[1,:]))
            b = T(transitions[2,1]) / T(sum(@view transitions[2,:]))
            isfinite(a + b) && a > zero(T) && b > zero(T) && a + b < T(2) ||
                return merge(invalid, (thinning=stride, status=:degenerate_transitions))
            decay = abs(one(T) - a - b)
            burn = iszero(decay) ? zero(T) : max(zero(T),
                ceil((log(epsilon) + log((a+b) / max(a,b))) / log(decay)))
            keep = ceil((T(2)-a-b) * a * b * abs2(normal / r) / (a+b)^3)
            total = T(stride) * (burn + keep)
            dependence_factor = total / T(nmin)
            valid = isfinite(total) && total > zero(T) && isfinite(dependence_factor)
            return (thinning=stride, burnin=T(stride) * burn, total, nmin,
                dependence_factor, status=valid ? :ok : :estimator_failed)
        end
        merge(invalid, (status=:markov_fit_failed,))
    end
end
