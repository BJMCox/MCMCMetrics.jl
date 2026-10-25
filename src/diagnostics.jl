"""
    diagnostics(x; drawdim=1, chaindim, counts=nothing, max_lag_work=nothing)

Return a `StructArray` with one row per scalar parameter. Columns are `index`,
`rhat`, `ess_bulk`, `ess_tail`, `mcse_mean`, `n_draws`, `n_chains`, and `status`.
Each `index` is a `CartesianIndex` into the parameter shape. `n_draws` is the
total logical count before splitting. `status` holds one symbol per metric.

Use modern split rank/folded R-hat, split bulk/tail ESS, and mean MCSE with
direct Geyer autocovariances. A single chain leaves R-hat unavailable while
retaining its supported precision diagnostics. Compressed inputs use exact
run overlaps without expansion. Their `max_lag_work` budget (see [`ess`](@ref))
yields `:work_limit` for an exhausted precision metric, preserving other results.
A parameter with a non-finite positive-count draw gets `NaN` metrics with
status `:nonfinite`; the other parameters stay available.
"""
function diagnostics(x; drawdim=1, chaindim=_default_chaindim(x), counts=nothing,
    max_lag_work=nothing)
    d = _draws(x; drawdim, chaindim, counts, nonfinite=true)
    _check_window(5, max_lag_work)
    budget = _lag_budget(d, max_lag_work)
    T, P = _floattype(d), prod(d.shape)
    n_draws = foldl(Base.Checked.checked_add, d.lengths; init=0)
    names = (:rhat, :ess_bulk, :ess_tail, :mcse_mean)
    values = NamedTuple{names}(ntuple(_ -> Vector{T}(undef, P), length(names)))
    status = NamedTuple{names}(ntuple(_ -> Vector{Symbol}(undef, P), length(names)))
    for p in 1:P
        for (j, (value, s)) in enumerate(_diagnostic_metrics(d, p, budget))
            values[j][p], status[j][p] = value, s
        end
    end
    StructArrays.StructArray((; index=vec(collect(CartesianIndices(d.shape))), values...,
        n_draws=fill(n_draws, P), n_chains=fill(length(d.chains), P),
        status=StructArrays.StructArray(status)))
end

function _diagnostic_metrics(d, p, max_lag_work)
    T = _floattype(d)
    d.finite[p] || return ntuple(_ -> (T(NaN), :nonfinite), 4)
    if isnothing(d.counts) && all(==(first(d.lengths)), d.lengths)
        runs = _parameter_runs(d, p, true)
        rankvalues = _rank_normalize(runs.values, runs.weights, T)
        samples = reshape(runs.values, runs.n, 2length(d.chains))
        ranked = reshape(rankvalues, size(samples))
        return (_rhat_parameter_result(d, p; runs, ranked=rankvalues),
            _ess_parameter_result(d, p; kind=:bulk, samples, ranked),
            _ess_parameter_result(d, p; kind=:tail, samples),
            _mcse_parameter_result(d, p; samples))
    end
    (_rhat_parameter_result(d, p),
        _budgeted_result(() -> _ess_parameter_result(d, p; kind=:bulk, max_lag_work), T),
        _budgeted_result(() -> _ess_parameter_result(d, p; kind=:tail, max_lag_work), T),
        _budgeted_result(() -> _precision_parameter_result(d, p; max_lag_work), T))
end

function _budgeted_result(f, ::Type{T}) where {T}
    try
        f()
    catch error
        error isa _WorkLimitError || rethrow()
        (T(NaN), :work_limit)
    end
end
