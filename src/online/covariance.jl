# Scalar anchors/residuals are shared with the ordinary moment state. Cross
# moments use one scale per coordinate, so unit changes do not destroy pivots.
struct _OnlineCrossMoments{S,T}
    moments::S
    cross::Matrix{T}
    error::Matrix{T}
    factor::Vector{T}
    delta::Vector{T}
end

function _online_cross_moments(moments)
    T, np = eltype(moments.origin), size(moments,1)
    _OnlineCrossMoments(moments,zeros(T,np,np),zeros(T,np,np),zeros(T,np),zeros(T,np))
end

function _online_covariance(::Type{T}, moments) where {T}
    np, nc = size(moments)
    marginal = [_online_cross_moments(StructArray(map(a -> view(a,:,c:c),StructArrays.components(moments)))) for c in 1:nc]
    level = _OnlineBatchLevel(_online_cross_moments(_online_moments(T,(np,1))),
        _online_moments(T,(np,1);second=false))
    (marginal=marginal,levels=[typeof(level)[] for _ in 1:nc])
end

_online_vector_component(draw,p) = (draw[p],zero(draw[p]),zero(draw[p]))
function _online_vector_component(partial::StructArray,p)
    residual = partial.scale[p,1]*partial.mean[p,1]
    iszero(residual) && !iszero(partial.mean[p,1]) &&
        return (partial.origin[p,1],partial.scale[p,1],partial.mean[p,1])
    (partial.origin[p,1],abs(residual),sign(residual))
end

function _online_cross_add!(target, source, count)
    count == 0 && return
    m = target.moments
    T = eltype(m.origin)
    oldcount = m.n[1,1]
    oldcount == 0 && return
    weight = T(min(oldcount,count))*(T(max(oldcount,count))/T(oldcount+count))
    for p in axes(m,1)
        origin, source_scale, source_mean = _online_vector_component(source,p)
        scale = _online_scale(m.origin[p,1],origin,max(m.scale[p,1],source_scale))
        factor = iszero(scale) ? zero(T) : m.scale[p,1]/scale
        sourcefactor = iszero(scale) ? zero(T) : source_scale/scale
        target.factor[p] = factor
        target.delta[p] = _online_center(origin,m.origin[p,1],scale)+sourcefactor*source_mean-factor*m.mean[p,1]
    end
    for p in axes(m,1), q in 1:p-1
        fp, fq = target.factor[p], target.factor[q]
        value, error = _online_sum(fp*(fq*target.cross[p,q]),target.delta[p]*(weight*target.delta[q]))
        target.cross[p,q], target.error[p,q] = _online_sum(value,error+fp*(fq*target.error[p,q]))
    end
    nothing
end

function _online_vector_add!(target, source, count)
    count == 0 && return
    _online_cross_add!(target,source,count)
    for p in axes(target.moments,1)
        origin, scale, mean = _online_vector_component(source,p)
        _online_add!(target.moments,p,1,origin,count,scale,mean)
    end
end

function _online_covariance!(b, draw, c, n, count)
    marginal = b.marginal[c]
    moments = marginal.moments
    T, np = eltype(moments.origin), size(moments,1)
    endpoint = n+count
    needed = 8sizeof(Int)-leading_zeros(endpoint)
    while length(b.levels[c]) < needed
        push!(b.levels[c],_OnlineBatchLevel(_online_cross_moments(_online_moments(T,(np,1))),
            _online_moments(T,(np,1);second=false)))
    end
    for (j,level) in enumerate(b.levels[c])
        width = 1 << (j-1)
        width > endpoint && break
        left = count
        partial = level.partial
        if n < width
            for p in 1:np
                partial.n[p,1] = moments.n[p,1]
                partial.origin[p,1] = moments.origin[p,1]
                partial.scale[p,1] = moments.scale[p,1]
                partial.mean[p,1] = moments.mean[p,1]
            end
        end
        oldpartial = partial.n[1,1]
        if oldpartial > 0
            take = min(left,width-oldpartial)
            for p in 1:np
                _online_add!(partial,p,1,draw[p],take)
            end
            left -= take
            if oldpartial+take == width
                _online_vector_add!(level.completed,partial,1)
                fill!(partial.n,0)
            end
        end
        full, remainder = divrem(left,width)
        _online_vector_add!(level.completed,draw,full)
        for p in 1:np
            _online_add!(partial,p,1,draw[p],remainder)
        end
    end
    # The draw-major caller updates these shared scalar moments immediately after.
    _online_cross_add!(marginal,draw,count)
    nothing
end

function _online_clear_covariance!(b)
    for marginal in b.marginal
        fill!(marginal.cross,0); fill!(marginal.error,0)
        fill!(marginal.factor,0); fill!(marginal.delta,0)
    end
    for chain in b.levels, level in chain
        _online_clear!(level.completed.moments)
        fill!(level.completed.cross,0); fill!(level.completed.error,0)
        fill!(level.completed.factor,0); fill!(level.completed.delta,0)
        _online_clear!(level.partial)
    end
end

function _online_cross_covariance(b, count)
    T = eltype(b.cross)
    np = size(b.cross,1)
    [p == q ? (b.moments.m2[p,1]+b.moments.m2_error[p,1])/T(count-1) :
        (b.cross[max(p,q),min(p,q)]+b.error[max(p,q),min(p,q)])/T(count-1) for p in 1:np,q in 1:np]
end

# Correlation normalization makes the unresolved-pivot test independent of units.
function _online_cov_logdet(covariance::AbstractMatrix{T}) where {T}
    diagonal = LinearAlgebra.diag(covariance)
    all(x -> isfinite(x) && x > zero(T),diagonal) || return T(NaN)
    roots = sqrt.(diagonal)
    normalized = [covariance[p,q]/roots[p]/roots[q] for p in eachindex(roots),q in eachindex(roots)]
    determinant = _positive_logdet(normalized)
    determinant+sum(log,diagonal)
end

function _online_vector_precision(b,c,n,::Type{T}; metric=:mc_cov) where {T}
    info = _online_batch_info(n)
    np = size(b.marginal[c].cross,1)
    unavailable(status) = (mean_cov=fill(T(NaN),np,np),ess_multivariate=T(NaN),status=status,precision=info)
    n < 2 && return unavailable(:insufficient_draws)
    a = info.completed_batches
    a < 2 && return unavailable(:insufficient_batches)
    marginal = b.marginal[c]
    completed = b.levels[c][trailing_zeros(info.batch_width)+1].completed
    drawcov = _online_cross_covariance(marginal,n)
    batchcov = _online_cross_covariance(completed,a)
    drawdet, batchdet = _online_cov_logdet(drawcov), _online_cov_logdet(batchcov)
    isfinite(batchdet) || return unavailable(:singular_estimator)
    ms, bs = vec(marginal.moments.scale), vec(completed.moments.scale)
    all(>(zero(T)),bs) || return unavailable(:singular_estimator)
    factor = T(info.batch_width)/T(n)
    covariance = _rescale_covariance(batchcov,bs,factor)
    ess = isfinite(drawdet) && all(>(zero(T)),ms) ?
        exp((drawdet-batchdet+T(2)*sum(log.(ms).-log.(bs)))/T(np)-log(factor)) : T(NaN)
    finite = metric === :mc_cov ? all(isfinite,covariance) : isfinite(ess) && ess > zero(T)
    status = !isfinite(drawdet) ? :singular_marginal : !finite ? :nonfinite_result : :ok
    (mean_cov=covariance,ess_multivariate=ess,status,precision=info)
end

"""
    mc_cov(state::OnlineDiagnostics)

Owned per-chain records `(mean_cov, ess_multivariate, status, precision)` from
full vector dyadic batch means. Completed batches center on their own mean;
the covariance-of-mean denominator and marginal moments use every draw.
Storage is O(parameters² × chains × log(n)), explicitly opt-in. No chains are
pooled, and no ridge repairs singular or unresolved estimates. The consistency
conditions are those of scalar online dyadic precision.
An `:ok` status describes the scaled estimator and finite returned entries.
Native underflow can zero tiny entries in physical units; rescale coordinates
before a downstream factorization when their units span that range.
"""
function mc_cov(state::OnlineDiagnostics{T,M}) where {T,M}
    :mc_cov in M || throw(ArgumentError("state does not track mc_cov"))
    s = _online_family_snapshot(state,(:covariance,))
    StructArray([_online_vector_precision(s.blocks.covariance,c,n,T) for (c,n) in enumerate(s.counts)])
end

"""Owned per-chain multivariate ESS records using the configured full vector dyadic estimator."""
function ess_multivariate(state::OnlineDiagnostics{T,M}) where {T,M}
    :ess_multivariate in M || throw(ArgumentError("state does not track ess_multivariate"))
    s = _online_family_snapshot(state,(:covariance,))
    StructArray(map(enumerate(s.counts)) do (c,n)
        result = _online_vector_precision(s.blocks.covariance,c,n,T;metric=:ess_multivariate)
        (ess_multivariate=result.ess_multivariate,status=result.status,precision=result.precision)
    end)
end
