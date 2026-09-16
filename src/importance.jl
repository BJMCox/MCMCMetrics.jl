_record_result(rows, shape) = isempty(shape) ? only(rows) : StructArray(reshape(rows,shape))

function _weight_series(d, p)
    T = _floattype(d)
    entries = [(c,i) for (c,chain) in enumerate(d.chains) for i in axes(chain,1) if _count(d,c,i) > 0]
    isempty(entries) && throw(ArgumentError("a weight series must have positive logical length"))
    logweights = [d.chains[c][i,p] for (c,i) in entries]
    maximum_log = maximum(logweights)
    maximum_log > -T(Inf) || throw(ArgumentError("each weight series needs positive mass"))
    shifted = [_centered_value(v,maximum_log,T) for v in logweights]
    weights = exp.(shifted)
    repeats = [_count(d,c,i) for (c,i) in entries]
    total = sum(i -> T(repeats[i]) * weights[i], eachindex(weights))
    (; entries, logweights, shifted, weights, repeats, total)
end

"""
    importance_diagnostics(logweights; drawdim=1, chaindim, counts=nothing)

Return importance-weight concentration: `ess_is`, `max_weight`, `entropy`,
`n_draws` and `status`, pooled over draw and chain axes. Log weights may be
finite or -Inf (zero mass); each series must have positive mass. A common
log shift leaves results unchanged. Integer counts repeat weight observations.
Weight ESS is not temporal MCMC ESS and has no dependence correction here.
Return one record per remaining parameter, scalar or StructArray.
"""
function importance_diagnostics(logweights; drawdim=1, chaindim=_default_chaindim(logweights), counts=nothing)
    d = _draws(logweights; drawdim, chaindim, counts, allow_neginf=true)
    T = _floattype(d)
    n = foldl(Base.checked_add,d.lengths;init=0)
    rows = map(1:prod(d.shape)) do p
        s = _weight_series(d,p)
        squared = sum(i -> T(s.repeats[i])*abs2(s.weights[i]/s.total), eachindex(s.weights))
        entropy = -sum(eachindex(s.weights)) do i
            w = s.weights[i]/s.total
            iszero(w) ? zero(T) : T(s.repeats[i])*w*log(w)
        end
        (ess_is=inv(squared), max_weight=inv(s.total), entropy, n_draws=n, status=:ok)
    end
    _record_result(rows,d.shape)
end

"""
    importance_mcse(values, logweights; sampling, estimator=:geyer, kwargs...)

MCSE of a self-normalized weighted mean. Specify `sampling=:iid` for the
ratio delta method with the n/(n-1) finite-sample variance correction, or
`:mcmc` for MCSE of the centered weighted influence process. The latter
preserves each chain and passes `estimator` and further keywords to `mcse`.
A scalar relative-efficiency multiplier cannot replace that influence process.
Values and log weights have identical array axes and floating types. Integer
counts represent repeated ordered observations, not multiplicative weights.
Finite second moments of the weighted influence and a valid sampling law
are required. Pareto smoothing introduces bias not covered by this MCSE.
"""
function importance_mcse(values, logweights; sampling, estimator=:geyer,
    drawdim=1, chaindim=_default_chaindim(values), counts=nothing, kwargs...)
    sampling in (:iid,:mcmc) || throw(ArgumentError("sampling must be :iid or :mcmc"))
    d = _draws(values; drawdim, chaindim, counts)
    w = _draws(logweights; drawdim, chaindim, counts, allow_neginf=true)
    d.shape == w.shape && map(size,d.chains) == map(size,w.chains) ||
        throw(DimensionMismatch("values and weights must have matching axes"))
    T = _floattype(d)
    T === _floattype(w) || throw(ArgumentError("values and weights must use the same arithmetic type"))
    n = foldl(Base.checked_add,d.lengths;init=0)
    result = map(1:prod(d.shape)) do p
        s = _weight_series(w,p)
        n >= 2 || return T(NaN)
        observations = [d.chains[c][i,p] for (c,i) in s.entries]
        positive = findall(>(zero(T)),s.weights)
        origin, scale = _sample_origin_scale(view(observations,positive),T)
        iszero(scale) && return T(NaN)
        z = [iszero(s.weights[i]) ? zero(T) : _centered_value(observations[i],origin,T)/scale
            for i in eachindex(observations)]
        mean = sum(i -> (T(s.repeats[i])*s.weights[i]/s.total)*z[i],eachindex(z))
        influence = [s.weights[i]*(z[i]-mean) for i in eachindex(z)]
        if sampling === :iid
            variance = sum(i -> T(s.repeats[i])*abs2(influence[i]/s.total),eachindex(z))
            error = sqrt((T(n)/T(n-1))*variance)*scale
        else
            chains = [zeros(T,size(chain,1),1) for chain in d.chains]
            for (j,(c,i)) in enumerate(s.entries)
                chains[c][i,1] = (T(n)/s.total)*influence[j]
            end
            error = only(mcse(chains; counts=d.counts, split=false, estimator, kwargs...))*scale
        end
        isfinite(error) && error > zero(T) ? error : T(NaN)
    end
    _parameter_result(result,d.shape)
end
