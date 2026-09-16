"""
    rhat(x; kind=:rank, split=true, drawdim=1, chaindim, counts=nothing)

Compute R-hat for each parameter from at least two independent chains.
Arrays use draws × chains × parameter dimensions. A vector is one scalar chain.
Set `chaindim=nothing` for a single chain with parameter axes, or pass a vector
of stored-draw × parameter matrices for nested chains.

The default is the maximum of rank-normalized and folded rank-normalized
split R-hat (Vehtari et al., 2021, doi:10.1214/20-BA1221). `kind=:basic`
uses the classical variance ratio. `split=false` uses whole chains.
Splitting omits the middle draw of odd-length chains before ranking or folding.

`counts` contains exact nonnegative integer repetition counts, never importance
weights. Logical chain lengths must match. Computation uses stored runs without
expanding them. A constant original or split chain, including a constant folded
chain, makes the corresponding statistic unavailable (`NaN`).

Float32 and Float64 observations retain their arithmetic and result type.
Integer observations use their Julia floating counterpart, with exact integer
ordering before rank conversion. Return a scalar for no parameter axes,
otherwise an array with the parameter shape.
"""
function rhat(x; kind=:rank, split=true, drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    kind in (:rank, :basic) || throw(ArgumentError("R-hat kind must be :rank or :basic"))
    split isa Bool || throw(ArgumentError("split must be a Bool"))
    d = _draws(x; drawdim, chaindim, counts)
    length(d.chains) >= 2 || throw(ArgumentError("R-hat requires at least two independent chains"))
    all(==(first(d.lengths)), d.lengths) || throw(ArgumentError("R-hat requires equal logical chain lengths"))
    values = [_rhat_parameter_result(d, p; kind, split).value for p in 1:prod(d.shape)]
    _parameter_result(values, d.shape)
end

# The exact split clipping and nearer-tail rank transform follow BAT.jl's
# mcmc_convergence.jl at 8ea5b1495a069df988dd18975128032170ab6149 (MIT).
# Adapted portions copyright (c) 2017-2021 Oliver Schulz
# <oschulz@mpp.mpg.de> and contributors. See LICENSE.md for the MIT terms.
function _parameter_runs(d, p, split)
    n = first(d.lengths)
    width = split ? n ÷ 2 : n
    values = eltype(first(d.chains))[]
    weights = Int[]
    ranges = UnitRange{Int}[]
    for c in eachindex(d.chains), offset in (split ? (0, n - width) : (0,))
        first_index = length(values) + 1
        before = 0
        for i in axes(d.chains[c], 1)
            after = before + _count(d, c, i)
            w = max(0, min(after, offset + width) - max(before, offset))
            if w > 0
                push!(values, d.chains[c][i, p])
                push!(weights, w)
            end
            before = after
        end
        push!(ranges, first_index:length(values))
    end
    (; values, weights, ranges, n=width)
end

function _rank_normalize(values::AbstractVector, weights::AbstractVector{Int}, ::Type{T}=_floattype(eltype(values)), order=sortperm(values)) where {T}
    total = foldl(Base.Checked.checked_add, weights; init=0)
    scores = Vector{T}(undef, length(values))
    before = 0
    i = 1
    while i <= length(order)
        j = i + 1
        mass = weights[order[i]]
        while j <= length(order) && values[order[j]] == values[order[i]]
            mass = Base.Checked.checked_add(mass, weights[order[j]])
            j += 1
        end
        after = total - before - mass
        # Blom's tied rank probability, evaluated from the nearer tail so
        # large logical counts cannot round an upper-tail probability to one.
        probability = (T(min(before, after)) + T(mass) / T(2) + T(1) / T(8)) / (T(total) + T(1) / T(4))
        score = -sqrt(T(2)) * SpecialFunctions.erfcinv(T(2) * min(probability, T(1) / T(2)))
        before > after && (score = -score)
        for k in i:j-1
            scores[order[k]] = score
        end
        before += mass
        i = j
    end
    scores
end

function _folded_values(values, weights, order=sortperm(values))
    total = foldl(Base.Checked.checked_add, weights; init=0)
    low_rank, high_rank = total ÷ 2 + isodd(total), total ÷ 2 + 1
    cumulative = 0
    lo = first(values)
    hi = first(values)
    for i in order
        next = cumulative + weights[i]
        cumulative < low_rank <= next && (lo = values[i])
        if cumulative < high_rank <= next
            hi = values[i]
            break
        end
        cumulative = next
    end
    if eltype(values) <: Integer
        # Doubled integer distances preserve ties even beyond exact float ranks.
        center = big(lo) + big(hi)
        return [abs(2big(v) - center) for v in values]
    end
    T = eltype(values)
    # Folded distances are sorting keys only. Separate their exponent so tiny
    # median gaps and extreme outliers need no wider floating arithmetic.
    let lo=lo, hi=hi
        map(values) do v
            _, exponent = frexp(max(abs(v), abs(lo), abs(hi)))
            lower = ldexp(lo, -exponent)
            centered = ldexp(v, -exponent) - lower
            gap = ldexp(hi, -exponent) - lower
            mantissa, shift = frexp(abs(centered - gap / T(2)))
            iszero(mantissa) ? (typemin(Int), zero(T)) : (exponent + shift, mantissa)
        end
    end
end

function _run_moments(values, weights, ranges, n)
    T = eltype(values)
    means = [sum(i -> (T(weights[i]) / T(n)) * values[i], r) for r in ranges]
    variances = [sum(i -> (T(weights[i]) / T(n - 1)) * abs2(values[i] - means[c]), r)
        for (c, r) in enumerate(ranges)]
    (; means, variances)
end

function _basic_run_moments(values, weights, ranges, n, ::Type{T}) where {T}
    origin, scale = _sample_origin_scale(values, T)
    means, variances = Vector{T}(undef, length(ranges)), Vector{T}(undef, length(ranges))
    for (c, r) in enumerate(ranges)
        chain = view(values, r)
        chain_origin, _ = _sample_origin_scale(chain, T)
        centered = [_centered_value(v, chain_origin, T) / scale for v in chain]
        local_mean = sum(i -> T(weights[r[i]]) / T(n) * centered[i], eachindex(centered))
        variances[c] = sum(i -> T(weights[r[i]]) / T(n - 1) * abs2(centered[i] - local_mean), eachindex(centered))
        means[c] = _centered_value(chain_origin, origin, T) / scale + local_mean
    end
    (; means, variances)
end

function _rhat_from_moments(means::AbstractVector{T}, variances, n::Integer) where {T}
    n >= 2 && length(means) >= 2 || return T(NaN)
    all(>=(zero(T)), variances) || return T(NaN)
    within = sum(v -> v / T(length(variances)), variances)
    grand_mean = sum(m -> m / T(length(means)), means)
    between = sum(m -> abs2(m - grand_mean) / T(length(means) - 1), means)
    sqrt(T(n - 1) / T(n) + between / within)
end

function _rhat_parameter_result(d, p; kind=:rank, split=true, runs=nothing, ranked=nothing)
    T = _floattype(d)
    length(d.chains) >= 2 || return (value=T(NaN), status=:insufficient_chains)
    all(==(first(d.lengths)), d.lengths) || return (value=T(NaN), status=:unequal_chain_lengths)
    n = first(d.lengths)
    n >= (split ? 4 : 2) || return (value=T(NaN), status=:insufficient_draws)
    isnothing(runs) && (runs = _parameter_runs(d, p, split))
    if any(r -> all(i -> runs.values[i] == runs.values[first(r)], r), runs.ranges)
        return (value=T(NaN), status=:constant)
    end
    if kind === :basic
        moments = _basic_run_moments(runs.values, runs.weights, runs.ranges, runs.n, T)
        value = _rhat_from_moments(moments.means, moments.variances, runs.n)
    else
        order = sortperm(runs.values)
        isnothing(ranked) && (ranked = _rank_normalize(runs.values, runs.weights, T, order))
        folded = _rank_normalize(_folded_values(runs.values, runs.weights, order), runs.weights, T)
        if any(r -> all(i -> folded[i] == folded[first(r)], r), runs.ranges)
            return (value=T(NaN), status=:constant)
        end
        bulk = _run_moments(ranked, runs.weights, runs.ranges, runs.n)
        tail = _run_moments(folded, runs.weights, runs.ranges, runs.n)
        value = max(_rhat_from_moments(bulk.means, bulk.variances, runs.n),
            _rhat_from_moments(tail.means, tail.variances, runs.n))
    end
    (value=value, status=isnan(value) ? :estimator_failed : :ok)
end
