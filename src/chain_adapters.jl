# These diagnostics operate independently on each named parameter. Joint,
# predictive, and sampler-metadata diagnostics need explicit numeric inputs.
const _PARAMETER_DIAGNOSTICS = (
    :rhat, :ess, :mcse, :iact, :autocor, :diagnostics, :diagnostic_curve,
    :geweke, :heidelberger_welch, :raftery_lewis, :rank_histogram, :rank_ecdf,
)

function _parameterwise(getdraws, f, parameters; kwargs...)
    any(k -> haskey(kwargs, k), (:drawdim, :chaindim)) &&
        throw(ArgumentError("chain containers determine their draw and chain dimensions"))
    keys = collect(parameters)
    allunique(keys) || throw(ArgumentError("parameters must be distinct"))
    # Values keep their own scalar types; a shared type would promote Float32.
    results = OrderedDict{eltype(keys),Any}(
        key => f(_observed_array(getdraws(key)); drawdim=1, chaindim=2, kwargs...)
        for key in keys)
    _parameter_results(f, results)
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
