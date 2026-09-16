"""
    categorical_summary(x; levels=nothing, drawdim=1, chaindim, counts=nothing)

Occupation counts and ordered transition counts for one integer-valued state
per draw. Return `levels`, state × chain `occupation` and `probability`, and
from-state × to-state × chain `transitions`. Optional unique integer `levels`
include known unvisited states. Otherwise sort the observed labels. Counts
repeat states, including their self transitions, without expanding the draws.
Separate chains never contribute transitions between one another.
"""
function categorical_summary(x; levels=nothing, drawdim=1,
    chaindim=_default_chaindim(x), counts=nothing)
    d = _scalar_chain_draws(x;drawdim,chaindim,counts)
    all(c -> eltype(c) <: Integer,d.chains) || throw(ArgumentError("categorical labels must be integers"))
    observed = [chain[i,1] for (c,chain) in enumerate(d.chains) for i in axes(chain,1) if _count(d,c,i) > 0]
    labels = isnothing(levels) ? sort!(unique(observed)) : collect(levels)
    !isempty(labels) && all(v -> v isa Integer,labels) && allunique(labels) ||
        throw(ArgumentError("levels must be a nonempty set of distinct integer labels"))
    lookup = Dict(label => i for (i,label) in enumerate(labels))
    all(v -> haskey(lookup,v),observed) || throw(ArgumentError("an observed state is absent from levels"))
    k, m = length(labels), length(d.chains)
    occupation, transitions = zeros(Int,k,m), zeros(Int,k,k,m)
    for (c,chain) in enumerate(d.chains)
        previous = 0
        for i in axes(chain,1)
            w = _count(d,c,i)
            w == 0 && continue
            current = lookup[chain[i,1]]
            occupation[current,c] += w
            transitions[current,current,c] += w-1
            previous > 0 && (transitions[previous,current,c] += 1)
            previous = current
        end
    end
    T = _floattype(d)
    probability = [d.lengths[c] > 0 ? T(occupation[i,c])/T(d.lengths[c]) : T(NaN)
        for i in 1:k, c in 1:m]
    (; levels=labels,occupation,probability,transitions,n_draws=copy(d.lengths))
end

"""
    categorical_weiss(x; drawdim=1, chaindim, counts=nothing)

Weiss's dependence-adjusted Pearson test of equal state frequencies, as in
Deonovic and Smith (2017), arXiv:1706.04919, section 2.1.2. Require independent,
equal-length categorical chains and a common stationary DAR(1) dependence
parameter. This model repeats the last state with probability phi and otherwise
draws from the common categorical distribution. General Markov chains need
not satisfy that model. Sparse frequencies also weaken the chi-square limit.

Return `statistic`, `pvalue`, `dof`, fitted `phi`, `correction` and `status`.
An estimated phi outside [0,1) returns unavailable, without clipping it into
the model. A large p-value does not establish convergence. Labels have no
numeric interpretation. Integer repetitions preserve transitions exactly.
"""
function categorical_weiss(x;drawdim=1,chaindim=_default_chaindim(x),counts=nothing)
    s = categorical_summary(x;drawdim,chaindim,counts)
    T = eltype(s.probability)
    n, m = first(s.n_draws), length(s.n_draws)
    all(==(n),s.n_draws) || throw(ArgumentError("Weiss requires equal chain lengths"))
    invalid = (statistic=T(NaN),pvalue=T(NaN),dof=(length(s.levels)-1)*(m-1),
        phi=T(NaN),correction=T(NaN),status=:insufficient_draws)
    n >= 2 && m >= 2 || return invalid
    pooled = vec(Statistics.mean(s.probability;dims=2))
    same = sum(c -> sum(i -> T(s.transitions[i,i,c])/T(n-1),eachindex(pooled))/T(m),1:m)
    denominator = one(T)-sum(abs2,pooled)
    denominator > zero(T) || return merge(invalid,(status=:constant,))
    phi = one(T)+inv(T(n))-(one(T)-same)/denominator
    zero(T) <= phi < one(T) || return merge(invalid,(phi,status=:outside_dar1,))
    correction = (one(T)+phi)/(one(T)-phi)
    pearson = sum(T(n)*abs2(s.probability[i,c]-pooled[i])/pooled[i]
        for c in 1:m for i in eachindex(pooled))
    statistic = pearson/correction
    pvalue = last(SpecialFunctions.gamma_inc(T(invalid.dof)/T(2),statistic/T(2)))
    (; statistic,pvalue,dof=invalid.dof,phi,correction,status=:ok)
end
