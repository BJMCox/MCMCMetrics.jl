struct _Draws{T,C,W,S}
    chains::C
    counts::W
    shape::S
    lengths::Vector{Int}
end

_floattype(::_Draws{T}) where {T} = T
_floattype(::Type{T}) where {T<:Union{Float32,Float64}} = T
_floattype(::Type{T}) where {T<:Integer} = _floattype(typeof(float(zero(T))))
_floattype(::Type) = throw(ArgumentError("observations must use Float32, Float64, or integers with a supported floating counterpart"))

_default_chaindim(x::AbstractArray) = ndims(x) == 1 ? nothing : 2
_default_chaindim(::AbstractVector{<:AbstractMatrix}) = nothing

_parameter_result(values, shape::Tuple) = isempty(shape) ? only(values) : reshape(values, shape)
_count(::_Draws{T,C,Nothing}, c, i) where {T,C} = 1
_count(d::_Draws, c, i) = d.counts[c][i]

function _checked_counts(counts, chains)
    isnothing(counts) && return nothing
    length(counts) == length(chains) || throw(DimensionMismatch("one count vector is required per chain"))
    map(counts, chains) do weights, chain
        weights isa AbstractVector{<:Integer} || throw(ArgumentError("repetition counts must be integer vectors"))
        length(weights) == size(chain, 1) || throw(DimensionMismatch("counts must match stored draws"))
        result = Vector{Int}(undef, length(weights))
        for (i, w) in enumerate(weights)
            0 <= w <= typemax(Int) || throw(ArgumentError("repetition counts must be nonnegative and fit Int"))
            result[i] = Int(w)
        end
        result
    end
end

function _validated_draws(chains, counts, shape; allow_neginf=false)
    isempty(chains) && throw(ArgumentError("at least one chain is required"))
    S = eltype(first(chains))
    T = _floattype(S)
    all(c -> eltype(c) == S, chains) || throw(ArgumentError("all chains must use the same observation type"))
    weights = _checked_counts(counts, chains)
    lengths = Vector{Int}(undef, length(chains))
    for c in eachindex(chains)
        chain = chains[c]
        size(chain, 1) > 0 || throw(ArgumentError("the draw axis must not be empty"))
        lengths[c] = isnothing(weights) ? size(chain, 1) : foldl(Base.Checked.checked_add, weights[c]; init=0)
        for p in axes(chain, 2), i in axes(chain, 1)
            if (isnothing(weights) || weights[c][i] > 0) &&
                !(isfinite(chain[i, p]) || (allow_neginf && chain[i,p] == -Inf))
                throw(ArgumentError("positive-count observations must be finite"))
            end
        end
    end
    _Draws{T,typeof(chains),typeof(weights),typeof(shape)}(chains, weights, shape, lengths)
end

function _draws(x::AbstractArray; drawdim=1, chaindim=_default_chaindim(x), counts=nothing, allow_neginf=false)
    Base.require_one_based_indexing(x)
    N = ndims(x)
    drawdim isa Integer && 1 <= drawdim <= N || throw(ArgumentError("invalid draw dimension"))
    isnothing(chaindim) || (chaindim isa Integer && 1 <= chaindim <= N && chaindim != drawdim) ||
        throw(ArgumentError("draw and chain dimensions must be distinct valid dimensions"))
    parameters = Tuple(i for i in 1:N if i != drawdim && i != chaindim)
    shape = map(i -> size(x, i), parameters)
    all(>(0), shape) || throw(ArgumentError("parameter axes must not be empty"))
    if isnothing(chaindim)
        ordered = PermutedDimsArray(x, (drawdim, parameters...))
        chains = [reshape(ordered, size(x, drawdim), prod(shape))]
        weights = isnothing(counts) ? nothing : [counts]
    else
        ordered = PermutedDimsArray(x, (drawdim, parameters..., chaindim))
        chains = [reshape(selectdim(ordered, N, c), size(x, drawdim), prod(shape)) for c in axes(x, chaindim)]
        if !isnothing(counts)
            counts isa AbstractMatrix{<:Integer} || throw(ArgumentError("counts must be a stored-draw by chain integer matrix"))
            size(counts) == (size(x, drawdim), size(x, chaindim)) || throw(DimensionMismatch("counts must match the draw and chain axes"))
        end
        weights = isnothing(counts) ? nothing : collect(eachcol(counts))
    end
    _validated_draws(chains, weights, shape; allow_neginf)
end

function _draws(x::AbstractVector{<:AbstractMatrix}; drawdim=1, chaindim=nothing, counts=nothing, allow_neginf=false)
    Base.require_one_based_indexing(x)
    isempty(x) && throw(ArgumentError("at least one chain is required"))
    drawdim in (1, 2) || throw(ArgumentError("nested matrices require drawdim=1 or drawdim=2"))
    isnothing(chaindim) || throw(ArgumentError("the outer vector already identifies independent chains"))
    foreach(Base.require_one_based_indexing, x)
    chains = drawdim == 1 ? x : [PermutedDimsArray(c, (2, 1)) for c in x]
    p = size(first(chains), 2)
    p > 0 || throw(ArgumentError("the parameter axis must not be empty"))
    all(c -> size(c, 2) == p, chains) || throw(DimensionMismatch("chain parameter shapes must match"))
    _validated_draws(chains, counts, (p,); allow_neginf)
end

# Subtract before conversion so integer offsets do not erase small differences.
_centered_value(x::Bool, origin::Bool, ::Type{T}) where {T} = T(x) - T(origin)
function _centered_value(x::Integer, origin::Integer, ::Type{T}) where {T}
    x >= origin ? T(widen(x) - widen(origin)) : -T(widen(origin) - widen(x))
end
_centered_value(x, origin, ::Type{T}) where {T} = T(x) - T(origin)

function _sample_origin_scale(x, ::Type{T}) where {T}
    lo, hi = extrema(x)
    origin = lo == hi || eltype(x) <: Integer ? lo : T(lo) / T(2) + T(hi) / T(2)
    scale = max(abs(_centered_value(lo, origin, T)), abs(_centered_value(hi, origin, T)))
    origin, scale
end
