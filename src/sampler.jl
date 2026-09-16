function _scalar_chain_draws(x; drawdim, chaindim, counts)
    d = _draws(x; drawdim, chaindim, counts)
    prod(d.shape) == 1 || throw(ArgumentError("sampler metadata must contain one scalar per draw and chain"))
    d
end

function _bfmi_chain(d, c)
    T = _floattype(d)
    n = d.lengths[c]
    n >= 2 || return T(NaN)
    retained = [i for i in axes(d.chains[c], 1) if _count(d, c, i) > 0]
    values = view(d.chains[c], retained, 1)
    z, scale = _scaled_samples(values, T)
    iszero(scale) && return T(NaN)
    mean = sum(i -> (T(_count(d, c, retained[i])) / T(n)) * z[i], eachindex(z))
    variance = sum(i -> (T(_count(d, c, retained[i])) / T(n)) * abs2(z[i] - mean), eachindex(z))
    jumps = sum(i -> abs2(z[i] - z[i-1]), 2:length(z); init=zero(T))
    variance > zero(T) ? (jumps / T(n)) / variance : T(NaN)
end

"""
    bfmi(energy; drawdim=1, chaindim, counts=nothing)

Estimate E-BFMI per chain as mean squared successive energy difference divided
by the unbiased energy variance. Input must contain full Hamiltonian energy,
including kinetic energy. A log-posterior column is insufficient. Integer
repetitions include zero-movement transitions without expanding the chain.
Constant energies return NaN. No diagnostic threshold is imposed.
Reference: Betancourt (2016), arXiv:1604.00695.
"""
function bfmi(energy; drawdim=1, chaindim=_default_chaindim(energy), counts=nothing)
    d = _scalar_chain_draws(energy; drawdim, chaindim, counts)
    values = [_bfmi_chain(d, c) for c in eachindex(d.chains)]
    isempty(_chain_result_shape(energy, d, chaindim)) ? only(values) : values
end

"""
    sampler_diagnostics(; energy=nothing, divergent=nothing, acceptance=nothing,
        tree_depth=nothing, max_tree_depth=nothing, leapfrog_steps=nothing,
        drawdim=1, chaindim, counts=nothing)

Summarize supplied sampler metadata into a StructArray with one row per chain.
Report only supplied fields: E-BFMI, divergence count/rate, mean acceptance,
tree-depth saturation count/rate and maximum, or leapfrog work totals/means.
Metadata must have matching draw/chain axes. Divergence flags are Bool,
acceptance lies in [0,1], and depth/work counts are nonnegative integers.
`max_tree_depth` is required with tree depths. Repetitions repeat every supplied
metadata value; do not attach per-proposal metadata to a compressed run unless
that value really applies to every repeated draw. No thresholds are inferred.
"""
function sampler_diagnostics(; energy=nothing, divergent=nothing, acceptance=nothing,
    tree_depth=nothing, max_tree_depth=nothing, leapfrog_steps=nothing,
    drawdim=1, chaindim=:auto, counts=nothing)
    supplied = (; energy, divergent, acceptance, tree_depth, leapfrog_steps)
    names = Tuple(k for k in keys(supplied) if !isnothing(supplied[k]))
    isempty(names) && throw(ArgumentError("at least one metadata array is required"))
    axes_chain = chaindim === :auto ? _default_chaindim(supplied[first(names)]) : chaindim
    data = map(names) do name
        x = supplied[name]
        d = _scalar_chain_draws(x; drawdim, chaindim=axes_chain, counts)
        if name === :divergent
            all(c -> eltype(c) <: Bool, d.chains) || throw(ArgumentError("divergence metadata must be Bool"))
        elseif name in (:tree_depth, :leapfrog_steps)
            all(c -> eltype(c) <: Integer, d.chains) || throw(ArgumentError("depth and work metadata must be integer counts"))
        end
        for c in eachindex(d.chains), i in axes(d.chains[c], 1)
            _count(d, c, i) == 0 && continue
            value = d.chains[c][i,1]
            (name !== :acceptance || 0 <= value <= 1) || throw(ArgumentError("acceptance must lie in [0,1]"))
            (name ∉ (:tree_depth, :leapfrog_steps) || value >= 0) || throw(ArgumentError("depth and work must be nonnegative"))
        end
        d
    end
    baseline = first(data)
    all(d -> d.lengths == baseline.lengths && map(size, d.chains) == map(size, baseline.chains), data) ||
        throw(DimensionMismatch("metadata must describe matching draws and chains"))
    if :tree_depth in names
        max_tree_depth isa Integer && max_tree_depth >= 0 || throw(ArgumentError("a nonnegative max_tree_depth is required"))
    end
    StructArray(map(eachindex(baseline.chains)) do c
        n = baseline.lengths[c]
        fields = map(names, data) do name, d
            T = _floattype(d)
            kept = [i for i in axes(d.chains[c],1) if _count(d,c,i) > 0]
            values = view(d.chains[c], :, 1)
            if name === :energy
                value = _bfmi_chain(d,c)
                return (bfmi=value, bfmi_status=n < 2 ? :insufficient_draws : isnan(value) ? :constant : :ok)
            elseif name === :divergent
                total = sum(i -> values[i] ? _count(d,c,i) : 0, kept; init=0)
                return (divergences=total, divergence_rate=n == 0 ? T(NaN) : T(total)/T(n))
            elseif name === :acceptance
                mean = n == 0 ? T(NaN) : sum(i -> (T(_count(d,c,i))/T(n))*T(values[i]), kept; init=zero(T))
                return (acceptance_mean=mean,)
            elseif name === :tree_depth
                total = sum(i -> values[i] >= max_tree_depth ? _count(d,c,i) : 0, kept; init=0)
                return (depth_saturations=total, depth_saturation_rate=n == 0 ? T(NaN) : T(total)/T(n),
                    depth_maximum=isempty(kept) ? 0 : maximum(i -> values[i], kept))
            else
                total = foldl(Base.checked_add,
                    (Base.checked_mul(Int(values[i]), _count(d,c,i)) for i in kept); init=0)
                return (leapfrog_total=total, leapfrog_mean=n == 0 ? T(NaN) : T(total)/T(n))
            end
        end
        merge((chain=c, n_draws=n), fields...)
    end)
end
