# These diagnostics operate independently on each named parameter. Joint,
# predictive, and sampler-metadata diagnostics need explicit numeric inputs.
const _PARAMETER_DIAGNOSTICS = (
    :rhat, :ess, :mcse, :iact, :autocor, :diagnostics, :diagnostic_curve,
    :geweke, :heidelberger_welch, :raftery_lewis, :rank_histogram, :rank_ecdf,
)

function _parameterwise(getdraws, f, parameters; kwargs...)
    any(k -> haskey(kwargs, k), (:drawdim, :chaindim)) &&
        throw(ArgumentError("chain containers determine their draw and chain dimensions"))
    # Values can have different scalar types and shapes. Inferring a shared
    # dictionary value type would promote Float32 results to Float64.
    Dict{eltype(parameters),Any}(
        key => f(_observed_array(getdraws(key)); drawdim=1, chaindim=2, kwargs...)
        for key in parameters)
end

function _observed_array(x)
    T = Base.nonmissingtype(eltype(x))
    T === eltype(x) && return x
    any(ismissing, x) && throw(ArgumentError("diagnostics require observed draws without missing values"))
    Array{T}(x)
end
