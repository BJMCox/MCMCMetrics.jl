struct _Draws{T,C,W,S}
    chains::C
    counts::W
    shape::S
    lengths::Vector{Int}
    finite::BitVector
end

_floattype(::_Draws{T}) where {T} = T
_floattype(::Type{T}) where {T<:Union{Float32,Float64}} = T
_floattype(::Type{T}) where {T<:Integer} = _floattype(typeof(float(zero(T))))
_floattype(::Type{Missing}) = throw(ArgumentError("observations must not contain missing values; drop or impute them first"))
_floattype(::Type{Union{Missing,T}}) where {T<:Real} = _floattype(Missing)
_floattype(::Type) = throw(ArgumentError("observations must use Float32, Float64, or integers with a supported floating counterpart"))

_default_chaindim(x::AbstractArray) = ndims(x) == 1 ? nothing : 2
_default_chaindim(::AbstractVector{<:AbstractMatrix}) = nothing

_parameter_result(values, ::Tuple{}) = only(values)
_parameter_result(values, shape::Tuple) = reshape(values, shape)
# A typed result vector keeps the return type inferable for any chain view type.
function _parameter_values(f::F, d::_Draws{T}) where {F,T}
    values = Vector{T}(undef, prod(d.shape))
    _foreach_parameter(p -> values[p] = f(p), length(values))
    values
end

# One task per parameter. Each result depends only on its own parameter, so
# results do not depend on the thread count.
function _foreach_parameter(f::F, P) where {F}
    tasks = [Threads.@spawn f(p) for p in 1:P]
    failure = nothing
    for task in tasks
        try
            wait(task)
        catch error
            isnothing(failure) && (failure = error)
        end
    end
    # Throw the lowest failing parameter's own exception, as a serial loop would.
    failure isa TaskFailedException && throw(failure.task.exception)
    isnothing(failure) || throw(failure)
    nothing
end

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

# Pass the observation type S from the input type so T and the result infer.
# `nonfinite=true` marks a parameter instead of throwing, so one bad parameter
# leaves the others available.
function _validated_draws(chains, counts, shape, ::Type{S}; allow_neginf=false, nonfinite=false) where {S}
    isempty(chains) && throw(ArgumentError("at least one chain is required"))
    T = _floattype(S)
    all(c -> eltype(c) == S, chains) || throw(ArgumentError("all chains must use the same observation type"))
    weights = _checked_counts(counts, chains)
    lengths = Vector{Int}(undef, length(chains))
    finite = trues(prod(shape))
    for c in eachindex(chains)
        chain = chains[c]
        size(chain, 1) > 0 || throw(ArgumentError("the draw axis must not be empty"))
        lengths[c] = isnothing(weights) ? size(chain, 1) : foldl(Base.Checked.checked_add, weights[c]; init=0)
        for p in axes(chain, 2), i in axes(chain, 1)
            if (isnothing(weights) || weights[c][i] > 0) &&
                !(isfinite(chain[i, p]) || (allow_neginf && chain[i,p] == -Inf))
                nonfinite || throw(ArgumentError("positive-count observations must be finite"))
                finite[p] = false
            end
        end
    end
    _Draws{T,typeof(chains),typeof(weights),typeof(shape)}(chains, weights, shape, lengths, finite)
end

# The tuple length follows from N and the type of chaindim, so shapes infer.
_parameter_dims(::Val{N}, drawdim, ::Nothing) where {N} = ntuple(i -> i < drawdim ? i : i + 1, Val(N - 1))
function _parameter_dims(::Val{N}, drawdim, chaindim) where {N}
    lo, hi = minmax(drawdim, chaindim)
    ntuple(i -> i < lo ? i : i + 1 < hi ? i + 1 : i + 2, Val(N - 2))
end

function _draws(x::AbstractArray{<:Any,N}; drawdim=1, chaindim=_default_chaindim(x), counts=nothing, allow_neginf=false, nonfinite=false) where {N}
    Base.require_one_based_indexing(x)
    drawdim isa Integer && 1 <= drawdim <= N || throw(ArgumentError("invalid draw dimension"))
    isnothing(chaindim) || (chaindim isa Integer && 1 <= chaindim <= N && chaindim != drawdim) ||
        throw(ArgumentError("draw and chain dimensions must be distinct valid dimensions"))
    parameters = _parameter_dims(Val(N), drawdim, chaindim)
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
    # The chain view type depends on runtime dimensions; keep T and the shape inferable.
    _validated_draws(chains, weights, shape, eltype(x); allow_neginf, nonfinite)::_Draws{_floattype(eltype(x)),<:Any,<:Any,typeof(shape)}
end

function _draws(x::AbstractVector{<:AbstractMatrix}; drawdim=1, chaindim=nothing, counts=nothing, allow_neginf=false, nonfinite=false)
    Base.require_one_based_indexing(x)
    isempty(x) && throw(ArgumentError("at least one chain is required"))
    drawdim in (1, 2) || throw(ArgumentError("nested matrices require drawdim=1 or drawdim=2"))
    isnothing(chaindim) || throw(ArgumentError("the outer vector already identifies independent chains"))
    foreach(Base.require_one_based_indexing, x)
    chains = drawdim == 1 ? x : [PermutedDimsArray(c, (2, 1)) for c in x]
    p = size(first(chains), 2)
    p > 0 || throw(ArgumentError("the parameter axis must not be empty"))
    all(c -> size(c, 2) == p, chains) || throw(DimensionMismatch("chain parameter shapes must match"))
    _validated_draws(chains, counts, (p,), eltype(first(x)); allow_neginf, nonfinite)
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
