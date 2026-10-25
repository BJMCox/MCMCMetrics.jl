# These diagnostics operate independently on each named parameter. Joint,
# predictive, and sampler-metadata diagnostics need explicit numeric inputs.
const _PARAMETER_DIAGNOSTICS = (
    :rhat, :ess, :mcse, :iact, :autocor, :diagnostics, :diagnostic_curve,
    :geweke, :heidelberger_welch, :raftery_lewis, :rank_histogram, :rank_ecdf,
)

"""
    ParameterResults{K} <: AbstractDict{K,Any}

Per-parameter results of a chain-container method, keyed by parameter and
iterated in the container's parameter order.
"""
struct ParameterResults{K} <: AbstractDict{K,Any}
    keys::Vector{K}
    # Values keep their own scalar types; a shared type would promote Float32.
    values::Vector{Any}
    index::Dict{K,Int}
end

Base.length(r::ParameterResults) = length(r.keys)
Base.iterate(r::ParameterResults, i=1) =
    i > length(r) ? nothing : (r.keys[i] => r.values[i], i + 1)
Base.haskey(r::ParameterResults, key) = haskey(r.index, key)
Base.get(r::ParameterResults, key, default) =
    (i = get(r.index, key, 0); i == 0 ? default : r.values[i])

function _parameterwise(getdraws, f, parameters; kwargs...)
    any(k -> haskey(kwargs, k), (:drawdim, :chaindim)) &&
        throw(ArgumentError("chain containers determine their draw and chain dimensions"))
    keys = collect(parameters)
    allunique(keys) || throw(ArgumentError("parameters must be distinct"))
    values = Any[f(_observed_array(getdraws(key)); drawdim=1, chaindim=2, kwargs...)
        for key in keys]
    index = Dict{eltype(keys),Int}(k => i for (i, k) in enumerate(keys))
    _parameter_results(f, ParameterResults(keys, values, index))
end

_parameter_results(f, results) = results
_parameter_results(::typeof(diagnostics), results) =
    StructArray(merge((parameter=key,), row) for (key, table) in results for row in table)

function _observed_array(x)
    T = Base.nonmissingtype(eltype(x))
    T === eltype(x) && return x
    any(ismissing, x) && throw(ArgumentError("diagnostics require observed draws without missing values"))
    Array{T}(x)
end
