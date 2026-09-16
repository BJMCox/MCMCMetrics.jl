function _vector_runs(d,c,n)
    indices, weights = Int[], Int[]
    remaining = n
    for i in axes(d.chains[c],1)
        k = min(_count(d,c,i),remaining)
        if k > 0
            push!(indices,i)
            push!(weights,k)
            remaining -= k
        end
        remaining == 0 && break
    end
    (; values=d.chains[c][indices,:],weights,n)
end

function _weighted_covariance(z::AbstractMatrix{T}, weights) where {T}
    n = foldl(Base.checked_add,weights;init=0)
    p = size(z,2)
    means = [_stable_run_mean(view(z,:,j),weights,n) for j in 1:p]
    covariance = zeros(T,p,p)
    for j in 1:p, k in 1:j
        total = correction = zero(T)
        for i in axes(z,1)
            value = (T(weights[i])/T(n-1))*(z[i,j]-means[j])*(z[i,k]-means[k])
            total, correction = _compensated_add(total,correction,value)
        end
        covariance[j,k] = covariance[k,j] = total
    end
    (; means,covariance)
end

function _vector_batches(z::AbstractMatrix{T},weights,b,estimator,max_work) where {T}
    multiplicities = Int[]
    columns = map(axes(z,2)) do p
        means = T[]
        _foreach_run_batch(view(z,:,p),weights,b;
            step=estimator === :batchmeans ? b : 1,max_work) do mean,k
            push!(means,mean)
            p == 1 && push!(multiplicities,k)
        end
        means
    end
    # Run geometry, hence grouped window multiplicities, is shared by coordinates.
    (; means=reduce(hcat,columns),multiplicities)
end

function _batch_covariance(z::AbstractMatrix{T},weights,b,estimator,max_work) where {T}
    n = foldl(Base.checked_add,weights;init=0)
    batches = _vector_batches(z,weights,b,estimator,max_work)
    if estimator === :batchmeans
        return T(b)*_weighted_covariance(batches.means,batches.multiplicities).covariance
    end
    center = [_stable_run_mean(view(z,:,p),weights,n) for p in axes(z,2)]
    result = zeros(T,size(z,2),size(z,2))
    windows = n-b+1
    for j in axes(z,2), k in 1:j
        total = correction = zero(T)
        for i in axes(batches.means,1)
            term = (T(batches.multiplicities[i])/T(windows))*
                (batches.means[i,j]-center[j])*(batches.means[i,k]-center[k])
            total, correction = _compensated_add(total,correction,term)
        end
        result[j,k] = result[k,j] = T(n)/T(n-b)*T(b)*total
    end
    result
end

function _multivariate_precision(d;estimator=:batchmeans,batch_size=nothing,
    lugsail=nothing,max_lag_work=100_000_000)
    estimator in (:batchmeans,:overlapping) || throw(ArgumentError("choose :batchmeans or :overlapping"))
    max_lag_work > 0 || throw(ArgumentError("max_lag_work must be positive"))
    b = isnothing(batch_size) ? max(1,isqrt(minimum(d.lengths))) : batch_size
    b isa Integer && b > 0 || throw(ArgumentError("batch_size must be a positive integer"))
    T, p = _floattype(d), prod(d.shape)
    if !isnothing(lugsail)
        lugsail isa NamedTuple && keys(lugsail) == (:r,:c) || throw(ArgumentError("lugsail must be (r=..., c=...)"))
        isfinite(lugsail.r) && lugsail.r > 1 && 0 <= lugsail.c < 1 && T(lugsail.c) < one(T) ||
            throw(ArgumentError("lugsail requires r > 1 and 0 <= c < 1"))
        floor(Int,b/lugsail.r) >= 1 || throw(ArgumentError("lugsail's smaller batch must have positive width"))
    end
    invalid = (mean_cov=fill(T(NaN),p,p),marginal_cov=fill(T(NaN),p,p),scale=ones(T,p))
    all(n -> n÷b >= 2,d.lengths) || return invalid
    lengths = estimator === :batchmeans ? [n÷b*b for n in d.lengths] : d.lengths
    n = foldl(Base.checked_add,lengths;init=0)
    runs = [_vector_runs(d,c,lengths[c]) for c in eachindex(d.chains)]
    scales = [maximum(_sample_origin_scale(view(run.values,:,j),T)[2] for run in runs) for j in 1:p]
    scales .= ifelse.(iszero.(scales),one(T),scales)
    mean_cov, marginal_cov = zeros(T,p,p), zeros(T,p,p)
    for (c,run) in enumerate(runs)
        nc = lengths[c]
        z = Matrix{T}(undef,size(run.values))
        for j in 1:p
            origin, _ = _sample_origin_scale(view(run.values,:,j),T)
            z[:,j] .= [_centered_value(v,origin,T)/scales[j] for v in view(run.values,:,j)]
        end
        covariance = _batch_covariance(z,run.weights,b,estimator,max_lag_work)
        if !isnothing(lugsail)
            small = floor(Int,b/lugsail.r)
            count = estimator === :batchmeans ? nc÷small*small : nc
            sub = _vector_runs(d,c,count)
            # Use the same affine coordinates as the larger batches.
            small_z = z[1:size(sub.values,1),:]
            covariance = (covariance-T(lugsail.c)*_batch_covariance(
                small_z,sub.weights,small,estimator,max_lag_work))/(one(T)-T(lugsail.c))
        end
        mean_cov .+= (T(nc)/T(n))/T(n).*covariance
        marginal_cov .+= (T(nc-1)/T(n-length(runs))).*_weighted_covariance(z,run.weights).covariance
    end
    (; mean_cov,marginal_cov,scale=scales)
end

function _positive_logdet(covariance::AbstractMatrix{T}) where {T}
    all(isfinite,covariance) || return T(NaN)
    factor = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(covariance);check=false)
    LinearAlgebra.isposdef(factor) || return T(NaN)
    pivots = LinearAlgebra.diag(factor.L)
    # Rounded cancellation can leave a tiny positive pivot for a singular matrix.
    minimum(abs2,pivots) > T(length(pivots))*eps(T)*maximum(abs2,pivots) || return T(NaN)
    T(2)*sum(log,pivots)
end

# Keep exponents separate until the final rounding. A small coordinate scale
# can underflow before a large second scale restores a representable cross term.
function _rescale_covariance(covariance::AbstractMatrix{T}, scales, factor=one(T)) where {T}
    parts = frexp.(scales)
    fm, fe = frexp(factor)
    result = similar(covariance)
    for j in axes(result,2), i in 1:j
        m, e = frexp(covariance[i,j])
        im, ie = parts[i]
        jm, je = parts[j]
        value = ldexp(((m*im)*jm)*fm,e+ie+je+fe)
        result[i,j] = result[j,i] = value
    end
    result
end

"""
    mc_cov(x; estimator=:batchmeans, batch_size=nothing, lugsail=nothing, kwargs...)

Covariance of the pooled vector mean, flattening parameter axes in Cartesian
order. Use nonoverlapping batch means or `:overlapping` batch means. Default
width is floor(sqrt(shortest chain length)). Nonoverlapping batches discard
each incomplete remainder. Combine independent chains with weights proportional
to their retained lengths: sum(n_c * longrun_cov_c) / sum(n_c)^2.

Optional `lugsail=(r=3,c=0.5)` uses (Sigma_b-c*Sigma_floor(b/r))/(1-c).
Both estimates use the retained prefix; the smaller nonoverlapping estimate
may discard its own last partial batch. No ridge or eigenvalue repair is used.
Singular, indefinite or insufficient estimates return a NaN matrix. Treat a
Cholesky pivot squared at or below p*eps(T) times the largest as unresolved
at the input precision, rather than assigning a determinant to roundoff.
Native underflow can zero tiny entries after restoring physical units. Rescale
coordinates before a downstream factorization if their units span that range.

Accept `drawdim`, `chaindim`, integer `counts`, and `max_lag_work` (100_000_000).
Never expand repetitions. Store grouped batch means; overlapping-window storage
can grow with the explicit work budget. Stationarity, finite moments and a
suitable mixing/strong-approximation condition are required for consistency.
References: Vats et al., arXiv:1512.07713; Vats and Flegal, arXiv:1809.04541.
"""
function mc_cov(x;drawdim=1,chaindim=_default_chaindim(x),counts=nothing,kwargs...)
    d = _draws(x;drawdim,chaindim,counts)
    s = _multivariate_precision(d;kwargs...)
    isfinite(_positive_logdet(s.mean_cov)) || return fill(_floattype(d)(NaN),size(s.mean_cov))
    _rescale_covariance(s.mean_cov,s.scale)
end

"""
    ess_multivariate(x; kwargs...)

Joint mean ESS = exp((logdet(marginal_cov)-logdet(mean_cov))/nparameters).
Use the within-chain pooled sample covariance on the retained draws as the
common marginal covariance. This assumes chains target the same stationary
distribution; mean disagreement is assessed separately by R-hat. Estimator,
array and count keywords match `mc_cov`. Singular estimates return NaN.
The determinant ratio is invariant to invertible linear parameter transforms.
"""
function ess_multivariate(x;drawdim=1,chaindim=_default_chaindim(x),counts=nothing,kwargs...)
    d = _draws(x;drawdim,chaindim,counts)
    s = _multivariate_precision(d;kwargs...)
    exp((_positive_logdet(s.marginal_cov)-_positive_logdet(s.mean_cov))/_floattype(d)(prod(d.shape)))
end

"""
    rhat_multivariate(x; split=true, drawdim=1, chaindim, counts=nothing)

Square root of the Brooks–Gelman (1998) multivariate variance ratio:
sqrt((n-1)/n + (m+1)/m * lambda_max(W^-1 * cov(chain_means))).
Use m equal-length vector chains, each with n draws after optional splitting.
This is the largest scale ratio over linear projections, without rank transforms
or scalar degrees-of-freedom corrections. Flatten parameters in Cartesian order.
Return NaN for singular within-chain covariance. Preserve compressed runs.
Reference: doi:10.1080/10618600.1998.10474787, section 4.
"""
function rhat_multivariate(x;split=true,drawdim=1,chaindim=_default_chaindim(x),counts=nothing)
    d = _draws(x;drawdim,chaindim,counts)
    split isa Bool || throw(ArgumentError("split must be a Bool"))
    length(d.chains) >= 2 || throw(ArgumentError("at least two independent chains are required"))
    all(==(first(d.lengths)),d.lengths) || throw(ArgumentError("logical chain lengths must match"))
    T, p = _floattype(d), prod(d.shape)
    runs = [_parameter_runs(d,j,split) for j in 1:p]
    n, m = first(runs).n, length(first(runs).ranges)
    n >= 2 || return T(NaN)
    coordinates = [_sample_origin_scale(run.values,T) for run in runs]
    W, means = zeros(T,p,p), zeros(T,m,p)
    for (c,range) in enumerate(first(runs).ranges)
        z, offsets = zeros(T,length(range),p), zeros(T,p)
        for j in 1:p
            origin, scale = coordinates[j]
            iszero(scale) && return T(NaN)
            chain = view(runs[j].values,range)
            local_origin, _ = _sample_origin_scale(chain,T)
            z[:,j] .= [_centered_value(v,local_origin,T)/scale for v in chain]
            offsets[j] = _centered_value(local_origin,origin,T)/scale
        end
        s = _weighted_covariance(z,view(first(runs).weights,range))
        W .+= s.covariance/T(m)
        means[c,:] .= offsets .+ s.means
    end
    isfinite(_positive_logdet(W)) || return T(NaN)
    B = _weighted_covariance(means,ones(Int,m)).covariance
    L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(W)).L
    between = L \ (L \ B)'
    lambda = LinearAlgebra.eigmax(LinearAlgebra.Symmetric(between))
    sqrt(T(n-1)/T(n)+(T(m+1)/T(m))*lambda)
end
