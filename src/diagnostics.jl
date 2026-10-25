"""
    diagnostics(x; drawdim=1, chaindim, counts=nothing, max_lag_work=nothing)

Return a `StructArray` with one row per scalar parameter. Columns are `index`,
`mean`, `sd`, `q5`, `q95`, `rhat`, `ess_bulk`, `ess_tail`, `mcse_mean`,
`mcse_sd`, `n_draws`, `n_chains`, and `status`. Each `index` is a
`CartesianIndex` into the parameter shape. `n_draws` is the total logical count
before splitting. `status` holds one symbol per metric from `rhat` onwards.

As in posterior's `summarise_draws`, `mean`, `sd` (divisor N-1) and the type-7
quantiles `q5` and `q95` pool all logical draws. `mcse_sd` is
`mcse(x; kind=:sd)`.

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
    summaries = (:mean, :sd, :q5, :q95)
    names = (:rhat, :ess_bulk, :ess_tail, :mcse_mean, :mcse_sd)
    summary = NamedTuple{summaries}(ntuple(_ -> Vector{T}(undef, P), length(summaries)))
    values = NamedTuple{names}(ntuple(_ -> Vector{T}(undef, P), length(names)))
    status = NamedTuple{names}(ntuple(_ -> Vector{Symbol}(undef, P), length(names)))
    _foreach_parameter(P) do p
        for (j, value) in enumerate(_summary_values(d, p))
            summary[j][p] = value
        end
        for (j, (value, s)) in enumerate(_diagnostic_metrics(d, p, budget))
            values[j][p], status[j][p] = value, s
        end
    end
    index = vec(collect(CartesianIndices(d.shape)))
    StructArrays.StructArray((; index, summary..., values...,
        n_draws=fill(n_draws, P), n_chains=fill(length(d.chains), P),
        status=StructArrays.StructArray(status)))
end

function _diagnostic_metrics(d, p, max_lag_work)
    T = _floattype(d)
    d.finite[p] || return ntuple(_ -> (T(NaN), :nonfinite), 5)
    sd = _budgeted_result(() -> _precision_parameter_result(d, p; kind=:sd, max_lag_work), T)
    if isnothing(d.counts) && all(==(first(d.lengths)), d.lengths)
        runs = _parameter_runs(d, p, true)
        rankvalues = _rank_normalize(runs.values, runs.weights, T)
        samples = reshape(runs.values, runs.n, 2length(d.chains))
        ranked = reshape(rankvalues, size(samples))
        return (_rhat_parameter_result(d, p; runs, ranked=rankvalues),
            _ess_parameter_result(d, p; kind=:bulk, samples, ranked),
            _ess_parameter_result(d, p; kind=:tail, samples),
            _mcse_parameter_result(d, p; samples), sd)
    end
    (_rhat_parameter_result(d, p),
        _budgeted_result(() -> _ess_parameter_result(d, p; kind=:bulk, max_lag_work), T),
        _budgeted_result(() -> _ess_parameter_result(d, p; kind=:tail, max_lag_work), T),
        _budgeted_result(() -> _precision_parameter_result(d, p; max_lag_work), T), sd)
end

function _summary_values(d, p)
    T = _floattype(d)
    d.finite[p] || return ntuple(_ -> T(NaN), 4)
    entries = [(c, i) for c in eachindex(d.chains) for i in axes(d.chains[c], 1) if _count(d, c, i) > 0]
    values = [d.chains[c][i, p] for (c, i) in entries]
    weights = [_count(d, c, i) for (c, i) in entries]
    origin, _ = _sample_origin_scale(values, T)
    z, scale = _scaled_samples(values, T)
    location(v) = T(origin) + v * scale
    (location(_weighted_mean(z, weights)), sqrt(_weighted_var(z, weights)) * scale,
        location(_run_quantile(z, weights, 1//20)), location(_run_quantile(z, weights, 19//20)))
end

function _budgeted_result(f, ::Type{T}) where {T}
    try
        f()
    catch error
        error isa _WorkLimitError || rethrow()
        (T(NaN), :work_limit)
    end
end
