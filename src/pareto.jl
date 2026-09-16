# Empirical-Bayes quadrature from Zhang and Stephens (2009), using the
# k convention and weak shrinkage in Vehtari et al. (2024), Appendices F/G.
# The quadrature formulas were checked against PSIS.jl (MIT): copyright
# (c) 2021 Seth Axen <seth.axen@gmail.com> and contributors. See LICENSE.md.
function _gpd_fit(excess::AbstractVector{T}; shrink=true) where {T}
    n = length(excess)
    failed = (k=T(NaN), scale=T(NaN), status=:fit_failed)
    n >= 5 || return merge(failed,(status=:insufficient_tail,))
    x = sort(excess)
    xmax = last(x)
    xmax > zero(T) && first(x) < xmax || return merge(failed,(status=:degenerate_tail,))
    quartile = x[max(fld(n+2,4),1)]
    quartile > zero(T) || return merge(failed,(status=:degenerate_tail,))
    # Fit on a unit scale so the empirical-prior grid remains numerically local.
    x ./= xmax
    q = quartile/xmax
    m = 30 + isqrt(n)
    theta = [(sqrt(T(m)/(T(j)-T(1)/T(2)))-one(T))/(T(3)*q)-one(T) for j in 1:m]
    likelihood = map(theta) do t
        isfinite(t) && t > -one(T) || return -T(Inf)
        k = sum(v -> log1p(t*v),x)/T(n)
        sigma = iszero(t) ? Statistics.mean(x) : k/t
        isfinite(k) && sigma > zero(T) ? -T(n)*(log(sigma)+k+one(T)) : -T(Inf)
    end
    normalization = LogExpFunctions.logsumexp(likelihood)
    isfinite(normalization) || return failed
    t = sum(j -> isfinite(likelihood[j]) ? theta[j]*exp(likelihood[j]-normalization) : zero(T),1:m)
    isfinite(t) && t > -one(T) || return failed
    k = sum(v -> log1p(t*v),x)/T(n)
    sigma = (iszero(t) ? Statistics.mean(x) : k/t)*xmax
    shrink && (k = (T(n)*k+T(5))/T(n+10))
    isfinite(k) && isfinite(sigma) && sigma > zero(T) ? (k,scale=sigma,status=:ok) : failed
end

function _gpd_isf(k::T, scale::T, survival::T) where {T}
    z = -log(survival)
    scale * (iszero(k) ? z : expm1(k*z)/k)
end

function _series_efficiency(reff, shape, p, ::Type{T}) where {T}
    if reff isa Real
        value = T(reff)
    else
        reff isa AbstractArray && size(reff) == shape || throw(DimensionMismatch("reff must match the remaining parameter axes"))
        value = T(reff[p])
    end
    isfinite(value) && value > zero(T) || throw(ArgumentError("reff must be finite and positive in the input precision"))
    value
end

_tail_length(n, reff::T) where {T} = min(cld(n,5), ceil(Int,min(T(n),T(3)*sqrt(T(n)/reff))))

function _psis_series(logratios::AbstractVector{T}, reff::T; return_smoothing=false) where {T}
    n = length(logratios)
    shifted = logratios .- maximum(logratios)
    smoothed_mask = return_smoothing ? falses(n) : nothing
    m = _tail_length(n,reff)
    k, status = T(NaN), :insufficient_tail
    if all(==(first(shifted)),shifted)
        status = :constant_weights
    elseif m >= 5 && m < n
        order = sortperm(shifted)
        cutoff = shifted[order[n-m]]
        threshold = exp(cutoff)
        tail = view(order,n-m+1:n)
        # exp(v)-exp(cutoff), retaining gaps when both are near one.
        excess = [exp(shifted[i]) * (-expm1(cutoff-shifted[i])) for i in tail]
        fit = all(isfinite,excess) ? _gpd_fit(excess) : (k=T(NaN),scale=T(NaN),status=:fit_failed)
        k, status = fit.k, fit.status
        if status === :ok
            smoothed = [_gpd_isf(k,fit.scale,(T(m-j)+T(1)/T(2))/T(m)) for j in 1:m]
            if all(v -> isfinite(v) && v >= zero(T),smoothed)
                for (j,i) in enumerate(tail)
                    value = min(log(threshold+smoothed[j]),zero(T))
                    isnothing(smoothed_mask) || (smoothed_mask[i] = true)
                    shifted[i] = value
                end
            else
                k, status = T(NaN), :fit_failed
            end
        end
    end
    log_normalizer = LogExpFunctions.logsumexp(shifted)
    shifted .-= log_normalizer
    threshold = n > 1 ? min(T(7)/T(10),one(T)-inv(log10(T(n)))) : -T(Inf)
    (; log_weights=shifted, smoothed_mask, log_normalizer, pareto_k=k, ess_is=reff/sum(v -> exp(T(2)*v),shifted),
        reff, tail_length=m, pareto_k_threshold=threshold,
        reliable=status === :constant_weights || (status === :ok && k < threshold), status)
end

"""
    psis(logratios; reff, drawdim=1, chaindim, counts=nothing)

Pareto-smooth the upper tail of importance ratios. Require an explicit relative
efficiency, scalar or parameter-shaped; `reff=1` states an IID assumption.
Finite log ratios and -Inf zero mass are allowed. Reject repetition counts:
smoothed order statistics cannot in general share a compressed run's value.

Return normalized `log_weights` in the input layout, `pareto_k`, `ess_is`,
`reff`, `tail_length`, fit `status`, `pareto_k_threshold` and `reliable`.
Keep the raw input unchanged. Tail length is min(ceil(n/5),ceil(3sqrt(n/reff))).
Use Zhang-Stephens empirical Bayes, weak k shrinkage toward 1/2, midpoint
quantiles, and truncation at the largest raw ratio. Small or degenerate tails
retain raw normalized weights with unavailable k and explicit status.
The reliability threshold uses nominal n, as in loo; it can be optimistic
for strongly dependent samples. Weight ESS is a heuristic, not estimand MCSE.
Reference: Vehtari et al. (2024), JMLR 25(72), https://jmlr.org/papers/v25/19-556.html.
"""
function psis(logratios; reff, drawdim=1, chaindim=_default_chaindim(logratios), counts=nothing)
    isnothing(counts) || throw(ArgumentError("PSIS requires uncompressed individual ratios"))
    d = _draws(logratios; drawdim, chaindim, allow_neginf=true)
    T = _floattype(d)
    output = logratios isa AbstractVector{<:AbstractMatrix} ?
        [similar(chain,T) for chain in logratios] : similar(logratios,T)
    # Build views of initialized output once; validation must never read undefined memory.
    if output isa AbstractVector{<:AbstractMatrix}
        foreach(a -> fill!(a,zero(T)),output)
    else
        fill!(output,zero(T))
    end
    out = _draws(output; drawdim, chaindim)
    rows = map(1:prod(d.shape)) do p
        s = _weight_series(d,p)
        result = _psis_series(s.shifted,_series_efficiency(reff,d.shape,p,T))
        for (j,(c,i)) in enumerate(s.entries)
            out.chains[c][i,p] = result.log_weights[j]
        end
        Base.structdiff(result,NamedTuple{(:log_weights,)})
    end
    fields = (:pareto_k,:ess_is,:reff,:tail_length,:pareto_k_threshold,:reliable,:status)
    metrics = NamedTuple{fields}(map(k -> _parameter_result(getproperty.(rows,k),d.shape),fields))
    merge((log_weights=output,),metrics)
end

"""
    pareto_tail(values; tail=:both, reff=1, tail_length=nothing, kwargs...)

Fit generalized Pareto tails to the actual estimand draws, separately from
importance log-ratio diagnostics. Return `k_left`, `k_right`, their selected
maximum `k`, tail size and fit statuses. Choose `:left`, `:right` or `:both`.
`reff=1` chooses the IID tail-size rule; dependent callers supply efficiency.
Values near k=1/2 or k=1 question variance or mean assumptions, respectively;
a finite tail fit cannot prove moment existence. Counts are not supported.
"""
function pareto_tail(values; tail=:both, reff=1, tail_length=nothing,
    drawdim=1, chaindim=_default_chaindim(values), counts=nothing)
    tail in (:both,:left,:right) || throw(ArgumentError("tail must be :both, :left or :right"))
    isnothing(counts) || throw(ArgumentError("estimand-tail fitting requires uncompressed draws"))
    d = _draws(values; drawdim, chaindim)
    T = _floattype(d)
    rows = map(1:prod(d.shape)) do p
        raw = [chain[i,p] for chain in d.chains for i in axes(chain,1)]
        n = length(raw)
        m = isnothing(tail_length) ? _tail_length(n,_series_efficiency(reff,d.shape,p,T)) : tail_length
        m isa Integer && 0 <= m < n || throw(ArgumentError("tail_length must be nonnegative and smaller than the draw count"))
        ordered = sort(raw)
        constant = first(ordered) == last(ordered)
        zright, _ = _scaled_samples(view(ordered,n-m:n),T)
        zleft, _ = _scaled_samples(view(ordered,1:m+1),T)
        fit_right = tail === :left ? (k=T(NaN),status=:not_requested) :
            constant ? (k=T(NaN),status=:constant) : _gpd_fit(zright[2:end] .- first(zright))
        fit_left = tail === :right ? (k=T(NaN),status=:not_requested) :
            constant ? (k=T(NaN),status=:constant) : _gpd_fit(last(zleft) .- reverse(zleft[1:end-1]))
        k = tail === :left ? fit_left.k : tail === :right ? fit_right.k : max(fit_left.k,fit_right.k)
        (k, k_left=fit_left.k,k_right=fit_right.k,tail_length=Int(m),
            status=(left=fit_left.status,right=fit_right.status))
    end
    _record_result(rows,d.shape)
end

"""
    pareto_smoothed_minimum(k)

Heuristic minimum effective sample size `10^(1/(1-max(0,k)))` for an explicitly
Pareto-smoothed expectation, from Vehtari et al. (2024). Return the qualified
estimator context with the value. k >= 1 gives Inf. This is neither a raw-mean
sample requirement nor an MCMC stopping guarantee; it ignores other bias.
"""
function pareto_smoothed_minimum(k::Real)
    T = _floattype(typeof(k))
    value = T(k)
    minimum = isnan(value) ? T(NaN) : value >= one(T) ? T(Inf) :
        T(10)^(inv(one(T)-max(zero(T),value)))
    (minimum, estimator=:pareto_smoothed_expectation, status=isnan(value) ? :unavailable : :heuristic)
end
