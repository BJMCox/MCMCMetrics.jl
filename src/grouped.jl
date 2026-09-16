function _nested_ratio(means::AbstractVector{T}, variances, groups) where {T}
    centers = [Statistics.mean(view(means,g)) for g in groups]
    within = Statistics.mean(groups) do g
        Statistics.mean(view(variances,g)) +
            (length(g) > 1 ? Statistics.var(view(means,g)) : zero(T))
    end
    within > zero(T) && isfinite(within) || return T(NaN)
    sqrt(one(T) + Statistics.var(centers)/within)
end

"""
    rhat_nested(x, superchain_ids; kind=:rank, split=true, drawdim=1, chaindim, counts=nothing)

Nested R-hat for equally sized groups of chains with equal logical lengths.
Chains within a superchain must start at the same position and use independent
randomness conditional on that position. IDs describe this design; they cannot
establish it. Require at least two superchains. Return one value per parameter.

Use Margossian et al. (2024), arXiv:2110.13017, equations 6–8. The variance
ratio is sqrt(1 + between_superchain / within_superchain), hence at least one.
`kind=:rank` takes the maximum rank/folded value. Split before ranking, dropping
odd middle draws, as in `rhat`. Set `split=false` for one-draw subchains;
these require at least two subchains per superchain. Counts remain compressed.
Zero within-superchain variation returns NaN. No convergence threshold is set.
"""
function rhat_nested(x, superchain_ids; kind=:rank, split=true, drawdim=1,
    chaindim=_default_chaindim(x), counts=nothing)
    kind in (:basic,:rank) || throw(ArgumentError("kind must be :basic or :rank"))
    split isa Bool || throw(ArgumentError("split must be a Bool"))
    d = _draws(x;drawdim,chaindim,counts)
    length(superchain_ids) == length(d.chains) || throw(DimensionMismatch("one superchain ID is required per chain"))
    ids = unique(superchain_ids)
    length(ids) >= 2 || throw(ArgumentError("at least two superchains are required"))
    original = [findall(==(id),superchain_ids) for id in ids]
    all(g -> length(g) == length(first(original)),original) || throw(ArgumentError("superchain sizes must match"))
    all(==(first(d.lengths)),d.lengths) || throw(ArgumentError("logical chain lengths must match"))
    groups = split ? [reduce(vcat,([2c-1,2c] for c in g)) for g in original] : original
    T = _floattype(d)
    result = map(1:prod(d.shape)) do p
        runs = _parameter_runs(d,p,split)
        n = runs.n
        n >= 1 && (n > 1 || length(first(groups)) > 1) || return T(NaN)
        all(==(first(runs.values)),runs.values) && return T(NaN)
        function evaluate(values, basic)
            moments = if n == 1
                z = basic ? first(_scaled_samples(values,T)) : values
                (means=z,variances=zeros(T,length(z)))
            elseif basic
                _basic_run_moments(values,runs.weights,runs.ranges,n,T)
            else
                _run_moments(values,runs.weights,runs.ranges,n)
            end
            _nested_ratio(moments.means,moments.variances,groups)
        end
        kind === :basic && return evaluate(runs.values,true)
        ranked = _rank_normalize(runs.values,runs.weights,T)
        folded = _rank_normalize(_folded_values(runs.values,runs.weights),runs.weights,T)
        max(evaluate(ranked,false),evaluate(folded,false))
    end
    _parameter_result(result,d.shape)
end
