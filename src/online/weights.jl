function _online_weights(::Type{T}, np, nc) where {T}
    dims = (np,nc)
    (maximum=fill(-T(Inf),dims), total=zeros(T,dims), total_error=zeros(T,dims),
        squared=zeros(T,dims), squared_error=zeros(T,dims),
        weighted_log=zeros(T,dims), log_error=zeros(T,dims))
end

function _online_weight_combine!(b,p,c,maximum,total,squared,weighted_log)
    T = eltype(b.total)
    maximum == -T(Inf) && return
    oldmax = b.maximum[p,c]
    newmax = max(oldmax,maximum)
    oldshift, shift = oldmax-newmax, maximum-newmax
    factor, incoming = exp(oldshift), exp(shift)
    oldtotal = b.total[p,c]
    oldtotalerror = b.total_error[p,c]
    oldsquare = b.squared[p,c]
    oldlog = b.weighted_log[p,c]
    # A zero rescaling factor erases mass, including its logarithmic moment.
    priorlog = iszero(factor) ? zero(T) : factor*(oldlog+oldshift*oldtotal)
    nextlog = iszero(incoming) ? zero(T) : incoming*(weighted_log+shift*total)
    value, error = _online_sum(factor*oldtotal,incoming*total)
    b.total[p,c], b.total_error[p,c] = _online_sum(value,error+factor*b.total_error[p,c])
    value, error = _online_sum(factor*(factor*oldsquare),incoming*(incoming*squared))
    b.squared[p,c], b.squared_error[p,c] = _online_sum(value,error+factor*(factor*b.squared_error[p,c]))
    value, error = _online_sum(priorlog,nextlog)
    carry = iszero(factor) ? zero(T) : factor*(b.log_error[p,c]+oldshift*oldtotalerror)
    b.weighted_log[p,c], b.log_error[p,c] = _online_sum(value,error+carry)
    b.maximum[p,c] = newmax
    nothing
end

function _online_weight!(b,p,c,x::T,count) where {T}
    _online_weight_combine!(b,p,c,x,T(count),T(count),zero(T))
end

function _online_clear_weights!(b)
    fill!(b.maximum,-Inf)
    for name in (:total,:total_error,:squared,:squared_error,:weighted_log,:log_error)
        fill!(b[name],0)
    end
end

function _online_pool_weights(b)
    T = eltype(b.total)
    pooled = _online_weights(T,size(b.total,1),1)
    for c in axes(b.total,2), p in axes(b.total,1)
        _online_weight_combine!(pooled,p,1,b.maximum[p,c],b.total[p,c]+b.total_error[p,c],
            b.squared[p,c]+b.squared_error[p,c],b.weighted_log[p,c]+b.log_error[p,c])
    end
    pooled
end

"""
    importance_diagnostics(state::OnlineDiagnostics)

Pooled raw log-weight concentration: ESS, maximum normalized weight and entropy.
Negative infinity is zero mass; an empty or all-zero-mass prefix is unavailable.
This is weight concentration, with no temporal correction, PSIS or weighted MCSE.
"""
function importance_diagnostics(state::OnlineDiagnostics{T,M}) where {T,M}
    :importance in M || throw(ArgumentError("state does not track importance"))
    s = _online_family_snapshot(state,(:weights,))
    n = foldl(Base.checked_add,s.counts;init=0)
    b = _online_pool_weights(s.blocks.weights)
    total = b.total[1]+b.total_error[1]
    if iszero(total)
        return (ess_is=T(NaN),max_weight=T(NaN),entropy=T(NaN),n_draws=n,
            status=n == 0 ? :insufficient_draws : :zero_mass)
    end
    squared = b.squared[1]+b.squared_error[1]
    entropy = log(total)-(b.weighted_log[1]+b.log_error[1])/total
    (ess_is=total*(total/squared),max_weight=inv(total),entropy,n_draws=n,status=:ok)
end

"""
    waic(state::OnlineDiagnostics)

Pool configured finite pointwise log likelihoods by actual chain counts. Return
owned observation-shaped pointwise records and across-observation `se_elpd`.
The penalty uses unbiased pooled log-likelihood variance. Observation units must
be independent for the SE interpretation; this is not posterior Monte Carlo error.
"""
function waic(state::OnlineDiagnostics{T,M}) where {T,M}
    :waic in M || throw(ArgumentError("state does not track waic"))
    s = _online_family_snapshot(state,(:predictive,))
    b = s.blocks.predictive
    n = foldl(Base.checked_add,s.counts;init=0)
    pooled = _online_moments(T,(size(b.moments,1),1))
    for c in eachindex(s.counts), p in axes(pooled,1)
        m = b.moments
        _online_add!(pooled,p,1,m.origin[p,c],s.counts[c],m.scale[p,c],m.mean[p,c],m.m2[p,c],m.m2_error[p,c])
    end
    w = _online_pool_weights(b.weights)
    rows = map(enumerate(vec(CartesianIndices(b.shape)))) do (p,index)
        n >= 2 || return (index,lppd=T(NaN),p_waic=T(NaN),elpd=T(NaN),status=:insufficient_draws)
        lppd = w.maximum[p]+log((w.total[p]+w.total_error[p])/T(n))
        scale = pooled.scale[p]
        penalty = scale*(scale*((pooled.m2[p]+pooled.m2_error[p])/T(n-1)))
        score = lppd-penalty
        (index,lppd,p_waic=penalty,elpd=score,status=isfinite(score) ? :ok : :nonfinite_result)
    end
    pointwise = StructArray(reshape(rows,b.shape))
    elpd = sum(pointwise.elpd)
    (;elpd,waic=-T(2)*elpd,p_waic=sum(pointwise.p_waic),se_elpd=_score_se(pointwise.elpd),pointwise)
end
