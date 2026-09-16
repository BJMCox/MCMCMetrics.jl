function _predictive_draws(x; drawdim, chaindim, counts)
    d = _draws(x; drawdim, chaindim, counts)
    isempty(d.shape) && throw(ArgumentError("pointwise likelihoods require an explicit observation axis"))
    d
end

function _score_se(values::AbstractArray{T}) where {T}
    length(values) > 1 || return T(NaN)
    z, scale = _scaled_samples(values,T)
    sqrt(T(length(values))*Statistics.var(z))*scale
end

"""
    waic(loglikelihood; drawdim=1, chaindim, counts=nothing)

Compute WAIC from finite pointwise log likelihoods. Require explicit remaining
observation axes, even for one observation. Return total `elpd`, `waic`,
`p_waic`, across-observation `se_elpd`, and shaped `pointwise` records.
The penalty is the unbiased log-likelihood sample variance over all draws.
`se_elpd` assumes independent observation units; it is not Monte Carlo error.
Integer counts repeat likelihood draws without expansion. The caller owns
the likelihood factorization and observation identity.
Reference: Vehtari, Gelman and Gabry (2017), doi:10.1007/s11222-016-9696-4.
"""
function waic(loglikelihood; drawdim=1, chaindim=_default_chaindim(loglikelihood), counts=nothing)
    d = _predictive_draws(loglikelihood;drawdim,chaindim,counts)
    T = _floattype(d)
    n = foldl(Base.checked_add,d.lengths;init=0)
    rows = map(enumerate(vec(CartesianIndices(d.shape)))) do (p,index)
        s = _weight_series(d,p)
        n >= 2 || return (index,lppd=T(NaN),p_waic=T(NaN),elpd=T(NaN),status=:insufficient_draws)
        maximum_log = maximum(s.logweights)
        lppd = T(maximum_log) + log(s.total/T(n))
        z, scale = _scaled_samples(s.logweights,T)
        mean = sum(i -> (T(s.repeats[i])/T(n))*z[i],eachindex(z))
        v = sum(i -> (T(s.repeats[i])/T(n-1))*abs2(z[i]-mean),eachindex(z))
        penalty = scale*(scale*v)
        score = lppd-penalty
        (index,lppd,p_waic=penalty,elpd=score,status=isfinite(score) ? :ok : :nonfinite_result)
    end
    pointwise = StructArray(reshape(rows,d.shape))
    elpd = sum(pointwise.elpd)
    (; elpd,waic=-T(2)*elpd,p_waic=sum(pointwise.p_waic),
        se_elpd=_score_se(pointwise.elpd),pointwise)
end

function _influence_error(chains, sampling, estimator)
    T = eltype(first(chains))
    n = sum(size(c,1) for c in chains)
    if sampling === :iid
        n >= 2 || return T(NaN)
        pooled = reduce(vcat,chains)
        z, scale = _scaled_samples(pooled,T)
        return iszero(scale) ? T(NaN) : sqrt(Statistics.var(vec(z))/T(n))*scale
    end
    only(mcse(chains;split=false,estimator))
end

# Keep the likelihood origin separate from small log-weight adjustments. This
# preserves normalized masses even when adding log(9) to 1e308 rounds away.
function _loo_products(raw,fit,::Type{T}) where {T}
    low = minimum(raw)
    origins = [fit.smoothed_mask[i] ? raw[i] : low for i in eachindex(raw)]
    adjustments = [fit.smoothed_mask[i] ? fit.log_weights[i] : -fit.log_normalizer for i in eachindex(raw)]
    reference = firstindex(raw)
    for i in eachindex(raw)
        difference = _centered_value(origins[i],origins[reference],T)+(adjustments[i]-adjustments[reference])
        difference > zero(T) && (reference = i)
    end
    relative = [_centered_value(origins[i],origins[reference],T)+(adjustments[i]-adjustments[reference])
        for i in eachindex(raw)]
    normalization = LogExpFunctions.logsumexp(relative)
    (; origin=origins[reference],offset=adjustments[reference]+normalization,
        mass=exp.(relative .- normalization))
end

"""
    loo(loglikelihood; reff=nothing, sampling=:mcmc, estimator=:geyer,
        drawdim=1, chaindim, counts=nothing)

PSIS leave-one-out scores from finite pointwise log likelihoods with explicit
observation axes. Accept a scalar or observation-shaped likelihood `reff`.
If omitted, estimate raw likelihood ESS after a stable shift, preserving chain
identity. Constant likelihoods use reff=1 only for the weight-smoothing rule.

Return `elpd`, `p_loo`, across-observation `se_elpd`, posterior `mcse_elpd`,
shaped `pointwise` records, and normalized `log_weights` in the input layout.
MCSE uses the weighted log-mean influence process. The total includes its
cross-observation covariance. Specify `sampling=:iid` only for independent
draws. MCSE is unavailable for unreliable Pareto fits or degenerate influence
processes; it excludes smoothing bias. Observation SE assumes independent
observation units and differs from posterior MCSE. Counts are not supported.
Reference: Vehtari, Gelman and Gabry (2017), doi:10.1007/s11222-016-9696-4.
"""
function loo(loglikelihood; reff=nothing, sampling=:mcmc, estimator=:geyer,
    drawdim=1, chaindim=_default_chaindim(loglikelihood), counts=nothing)
    sampling in (:iid,:mcmc) || throw(ArgumentError("sampling must be :iid or :mcmc"))
    isnothing(counts) || throw(ArgumentError("PSIS-LOO requires uncompressed likelihood draws"))
    d = _predictive_draws(loglikelihood;drawdim,chaindim,counts)
    T = _floattype(d)
    n = foldl(Base.checked_add,d.lengths;init=0)
    weights = loglikelihood isa AbstractVector{<:AbstractMatrix} ?
        [fill(zero(T),size(c)) for c in loglikelihood] : fill(zero(T),size(loglikelihood))
    wd = _draws(weights;drawdim,chaindim)
    total_influence = [zeros(T,size(chain,1),1) for chain in d.chains]
    rows = map(enumerate(vec(CartesianIndices(d.shape)))) do (p,index)
        raw = [chain[i,p] for chain in d.chains for i in axes(chain,1)]
        low, high = extrema(raw)
        shifted = [_centered_value(v,high,T) for v in raw]
        chains = [[exp(_centered_value(v,high,T)) for v in view(chain,:,p)] for chain in d.chains]
        efficiency = if isnothing(reff)
            all(==(first(raw)),raw) ? one(T) :
                only(ess([reshape(c,:,1) for c in chains];kind=:mean,split=false))/T(n)
        else
            _series_efficiency(reff,d.shape,p,T)
        end
        lppd_shifted = LogExpFunctions.logsumexp(shifted)-log(T(n))
        lppd = T(high)+lppd_shifted
        if !(isfinite(efficiency) && efficiency > zero(T))
            for c in eachindex(wd.chains)
                wd.chains[c][:,p] .= T(NaN)
            end
            return (index,elpd=T(NaN),lppd,p_loo=T(NaN),mcse_elpd=T(NaN),
                pareto_k=T(NaN),reff=efficiency,tail_length=0,reliable=false,
                status=(psis=:efficiency_unavailable,mcse=:efficiency_unavailable))
        end
        fit = _psis_series([_centered_value(low,v,T) for v in raw],efficiency;return_smoothing=true)
        # Raw inverse likelihood times likelihood cancels exactly. Retain that
        # identity before centering can overflow or erase integer log offsets.
        products = _loo_products(raw,fit,T)
        elpd = T(products.origin)+products.offset
        penalty = _centered_value(high,products.origin,T)+(lppd_shifted-products.offset)
        # Two normalized masses avoid overflow in the raw likelihood ratio.
        influence = T(n) .* (products.mass .- exp.(fit.log_weights))
        offset = 0
        current = [zeros(T,size(chain,1),1) for chain in d.chains]
        for c in eachindex(d.chains)
            rows = offset+1:offset+size(d.chains[c],1)
            wd.chains[c][:,p] .= view(fit.log_weights,rows)
            current[c][:,1] .= view(influence,rows)
            total_influence[c] .+= current[c]
            offset = last(rows)
        end
        error = fit.reliable ? _influence_error(current,sampling,estimator) : T(NaN)
        status = !fit.reliable ? :unreliable_pareto : isfinite(error) ? :ok : :unavailable_precision
        (index,elpd,lppd,p_loo=penalty,mcse_elpd=error,
            pareto_k=fit.pareto_k,reff=efficiency,tail_length=fit.tail_length,reliable=fit.reliable,
            status=(psis=fit.status,mcse=status))
    end
    pointwise = StructArray(reshape(rows,d.shape))
    error = all(pointwise.reliable) ? _influence_error(total_influence,sampling,estimator) : T(NaN)
    (; elpd=sum(pointwise.elpd),p_loo=sum(pointwise.p_loo),se_elpd=_score_se(pointwise.elpd),
        mcse_elpd=error,pointwise,log_weights=weights)
end

"""
    compare_elpd(a, b; observation_ids_a, observation_ids_b)

Compare numeric pointwise ELPD arrays for exactly matched observation IDs in
the same order. Return the sum and across-observation standard error of paired
differences. This uncertainty concerns new independent observation units, not
posterior Monte Carlo error. The caller supplies the observation identity.
"""
function compare_elpd(a::AbstractArray,b::AbstractArray;observation_ids_a,observation_ids_b)
    size(a) == size(b) == size(observation_ids_a) == size(observation_ids_b) ||
        throw(DimensionMismatch("scores and observation IDs must have matching axes"))
    observation_ids_a == observation_ids_b && allunique(observation_ids_a) ||
        throw(ArgumentError("observation IDs must be unique and match in order"))
    T = _floattype(eltype(a))
    T === _floattype(eltype(b)) || throw(ArgumentError("score arithmetic types must match"))
    all(isfinite,a) && all(isfinite,b) || throw(ArgumentError("scores must be finite"))
    delta = map((x,y) -> _centered_value(x,y,T),a,b)
    (; elpd_difference=sum(delta),se_difference=_score_se(delta),pointwise=delta,n_observations=length(delta))
end

"""
    loo_pit(predictive_cdf, logweights; drawdim=1, chaindim, counts=nothing)

Integrate supplied conditional predictive CDF values at each observed outcome
using LOO importance weights. Require matching explicit observation axes and
CDF values in [0,1]. Normalize log weights stably. Uniform calibration applies
to continuous outcomes under the appropriate predictive model; discrete
outcomes need a separate randomized-PIT design. No CDF is inferred from a
joint density or from the posterior parameter draws.
"""
function loo_pit(predictive_cdf,logweights;drawdim=1,chaindim=_default_chaindim(predictive_cdf),counts=nothing)
    d = _predictive_draws(predictive_cdf;drawdim,chaindim,counts)
    w = _draws(logweights;drawdim,chaindim,counts,allow_neginf=true)
    d.shape == w.shape && map(size,d.chains) == map(size,w.chains) ||
        throw(DimensionMismatch("CDF values and log weights must have matching axes"))
    T = _floattype(d)
    T === _floattype(w) || throw(ArgumentError("CDF and weight arithmetic types must match"))
    result = map(1:prod(d.shape)) do p
        s = _weight_series(w,p)
        values = [d.chains[c][i,p] for (c,i) in s.entries]
        all(v -> 0 <= v <= 1,values) || throw(ArgumentError("CDF values must lie in [0,1]"))
        sum(i -> (T(s.repeats[i])*s.weights[i]/s.total)*T(values[i]),eachindex(values))
    end
    _parameter_result(result,d.shape)
end
