"""
    squared_jump_distance(x; metric=nothing, cost=nothing, drawdim=1, chaindim, counts=nothing)

Summarize squared distances between consecutive vector draws in each chain.
Flatten parameter axes in Cartesian order. An optional symmetric positive
definite matrix defines distance delta' * metric * delta. Otherwise use
Euclidean distance. Counts add repeated states and their zero-length moves.

Return per-chain `mean_squared_jump`, `total_squared_jump`, `n_transitions`
and `status`. An optional positive total cost, scalar for a single chain or
one value per chain, adds `squared_jump_per_cost`. The caller defines cost
units. Zero movement is a valid result; fewer than two draws is unavailable.
Movement and acceptance cannot establish convergence.
"""
function squared_jump_distance(x;metric=nothing,cost=nothing,drawdim=1,
    chaindim=_default_chaindim(x),counts=nothing)
    d = _draws(x;drawdim,chaindim,counts)
    T, p = _floattype(d), prod(d.shape)
    transform = if isnothing(metric)
        nothing
    else
        size(metric) == (p,p) && LinearAlgebra.issymmetric(metric) && all(isfinite,metric) ||
            throw(ArgumentError("metric must be a finite symmetric parameter covariance inverse"))
        LinearAlgebra.cholesky(LinearAlgebra.Symmetric(T.(metric))).U
    end
    costs = isnothing(cost) ? nothing : cost isa Real ? [cost] : collect(cost)
    if !isnothing(costs)
        length(costs) == length(d.chains) && all(v -> isfinite(T(v)) && T(v) > zero(T),costs) ||
            throw(ArgumentError("supply one finite positive total cost per chain"))
    end
    rows = map(eachindex(d.chains)) do c
        chain, previous, root = d.chains[c], 0, zero(T)
        delta = zeros(T,p)
        doubled = falses(p)
        for i in axes(chain,1)
            _count(d,c,i) == 0 && continue
            if previous > 0
                for j in eachindex(delta)
                    delta[j] = _centered_value(chain[i,j],chain[previous,j],T)
                    doubled[j] = !isfinite(delta[j])
                    doubled[j] && (delta[j] = T(chain[i,j])/T(2)-T(chain[previous,j])/T(2))
                end
                # Apply a small metric before restoring an overflowing difference.
                mapped = isnothing(transform) ? delta .* ifelse.(doubled,T(2),one(T)) :
                    [sum(j -> (transform[k,j]*delta[j])*(doubled[j] ? T(2) : one(T)),1:p) for k in 1:p]
                distance = LinearAlgebra.norm(mapped)
                root = hypot(root,distance)
            end
            previous = i
        end
        transitions = max(0,d.lengths[c]-1)
        mean = transitions > 0 ? abs2(root/sqrt(T(transitions))) : T(NaN)
        result = (mean_squared_jump=mean,total_squared_jump=abs2(root),n_transitions=transitions,
            status=transitions == 0 ? :insufficient_draws : isfinite(mean) ? :ok : :overflow)
        isnothing(costs) ? result : merge(result,(squared_jump_per_cost=abs2(root/sqrt(T(costs[c]))),))
    end
    isnothing(chaindim) && !(x isa AbstractVector{<:AbstractMatrix}) ? only(rows) : StructArray(rows)
end

"""
    tempering_summary(level_trace; levels, drawdim=1, chaindim, counts=nothing)

Summarize replica traces over caller-ordered integer temperature `levels`.
The second array axis identifies replicas, not independent posterior chains.
Return replica occupations and complete first-level → last-level → first-level
round trips. Each duration counts logical transitions from the first cold
visit (or previous completed return), including waiting at endpoints. Ignore
an initial hot visit and an unfinished final trip. Counts are exact run lengths.
No replica-exchange acceptance rate or convergence claim is inferred.
"""
function tempering_summary(level_trace;levels,drawdim=1,
    chaindim=_default_chaindim(level_trace),counts=nothing)
    length(levels) >= 2 || throw(ArgumentError("at least two ordered temperature levels are required"))
    s = categorical_summary(level_trace;levels,drawdim,chaindim,counts)
    d = _scalar_chain_draws(level_trace;drawdim,chaindim,counts)
    cold, hot = first(s.levels), last(s.levels)
    rows = map(eachindex(d.chains)) do c
        started, visited_hot, start, time = false, false, 0, 0
        durations = Int[]
        for i in axes(d.chains[c],1)
            w = _count(d,c,i)
            w == 0 && continue
            level = d.chains[c][i,1]
            if level == cold
                if !started
                    started, start = true, time
                elseif visited_hot
                    push!(durations,time-start)
                    start, visited_hot = time, false
                end
            elseif level == hot && started
                visited_hot = true
            end
            time += w
        end
        (round_trips=length(durations),durations,incomplete_trip=visited_hot)
    end
    (; levels=s.levels,occupation=s.occupation,probability=s.probability,
        n_draws=s.n_draws,replicas=StructArray(rows))
end
