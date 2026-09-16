function _pooled_rankdata(d, p)
    entries = [(value=chain[i,p], count=_count(d,c,i), chain=c)
        for (c,chain) in enumerate(d.chains) for i in axes(chain,1) if _count(d,c,i) > 0]
    order = sortperm(entries; by=e -> e.value)
    total = foldl(Base.checked_add, d.lengths; init=0)
    ranks = Vector{Rational{widen(Int)}}(undef, length(entries))
    before, i = 0, 1
    while i <= length(order)
        j, mass = i+1, entries[order[i]].count
        while j <= length(order) && entries[order[j]].value == entries[order[i]].value
            mass = Base.checked_add(mass, entries[order[j]].count)
            j += 1
        end
        # Widen before doubling; Base compares rationals with floats exactly.
        rank = (2widen(before) + mass) // (2widen(total))
        for k in i:j-1
            ranks[order[k]] = rank
        end
        before += mass
        i = j
    end
    entries, ranks
end

"""
    rank_histogram(x; bins=10, drawdim=1, chaindim, counts=nothing)

Return pooled midrank bin `counts`, per-chain `frequency`, and probability
`edges`. `bins` is a positive number of equal bins or strictly increasing
edges from 0 to 1. Ties share their exact pooled midrank. Counts include
integer repetitions, with no expansion. The last bin includes its right edge.
Arrays have bin × chain × parameter axes; single-chain inputs omit that axis.
This returns plot data without a uniformity test for dependent observations.
"""
function rank_histogram(x; bins=10, drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    d = _draws(x; drawdim, chaindim, counts)
    T = _floattype(d)
    if bins isa Integer
        bins > 0 || throw(ArgumentError("bins must be positive"))
        edges = collect(range(zero(T), one(T); length=bins+1))
    else
        edges = T.(collect(bins))
    end
    length(edges) >= 2 && first(edges) == 0 && last(edges) == 1 && all(>(0), diff(edges)) ||
        throw(ArgumentError("bin edges must increase strictly from zero to one"))
    result = zeros(Int, length(edges)-1, length(d.chains), prod(d.shape))
    for p in 1:prod(d.shape)
        entries, ranks = _pooled_rankdata(d,p)
        for (entry, rank) in zip(entries, ranks)
            bin = min(searchsortedlast(edges,rank), length(edges)-1)
            result[bin,entry.chain,p] += entry.count
        end
    end
    frequency = [d.lengths[c] > 0 ? T(result[b,c,p])/T(d.lengths[c]) : T(NaN)
        for b in axes(result,1), c in axes(result,2), p in axes(result,3)]
    shape = (length(edges)-1, _chain_result_shape(x,d,chaindim)...)
    (; edges, counts=reshape(result,shape), frequency=reshape(frequency,shape))
end

"""
    rank_ecdf(x; grid=0:0.05:1, drawdim=1, chaindim, counts=nothing)

Return each chain's empirical CDF of pooled midranks at an explicit probability
grid. Result fields are `grid` and `ecdf`, with grid × chain × parameter axes.
Ties and repetitions follow [`rank_histogram`](@ref). No IID confidence band
is attached to dependent MCMC ranks.
"""
function rank_ecdf(x; grid=0:0.05:1, drawdim=1, chaindim=_default_chaindim(x), counts=nothing)
    d = _draws(x; drawdim, chaindim, counts)
    T = _floattype(d)
    points = T.(collect(grid))
    all(t -> isfinite(t) && 0 <= t <= 1, points) && issorted(points) ||
        throw(ArgumentError("rank grid must be sorted within [0,1]"))
    result = zeros(Int, length(points), length(d.chains), prod(d.shape))
    for p in 1:prod(d.shape)
        entries, ranks = _pooled_rankdata(d,p)
        for (entry, rank) in zip(entries,ranks), j in searchsortedfirst(points,rank):length(points)
            result[j,entry.chain,p] += entry.count
        end
    end
    values = [d.lengths[c] > 0 ? T(result[j,c,p])/T(d.lengths[c]) : T(NaN)
        for j in axes(result,1), c in axes(result,2), p in axes(result,3)]
    (; grid=points, ecdf=reshape(values, (length(points), _chain_result_shape(x,d,chaindim)...)))
end
