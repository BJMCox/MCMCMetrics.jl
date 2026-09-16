"""
    diagnostics(x; drawdim=1, chaindim, counts=nothing, max_lag_work=100_000_000)

Return a `StructArray` with one row per scalar parameter. Columns are `index`,
`rhat`, `ess_bulk`, `ess_tail`, `mcse_mean`, `n_draws`, `n_chains`, and `status`.
Each `index` is a `CartesianIndex` into the parameter shape. `n_draws` is the
total logical count before splitting. `status` holds one symbol per metric.

Use modern split rank/folded R-hat, split bulk/tail ESS, and mean MCSE with
direct Geyer autocovariances. A single chain leaves R-hat unavailable while
retaining its supported precision diagnostics. Compressed inputs use exact
run overlaps without expansion. Their explicit `max_lag_work` budget yields
`:work_limit` for an exhausted precision metric, preserving other results.
"""
function diagnostics(x; drawdim=1, chaindim=_default_chaindim(x), counts=nothing,
    max_lag_work=100_000_000)
    d = _draws(x; drawdim, chaindim, counts)
    max_lag_work > 0 || throw(ArgumentError("max_lag_work must be positive"))
    T = _floattype(d)
    n_draws = foldl(Base.Checked.checked_add, d.lengths; init=0)
    rows = map(enumerate(vec(CartesianIndices(d.shape)))) do (p, index)
        if isnothing(d.counts) && all(==(first(d.lengths)), d.lengths)
            runs = _parameter_runs(d, p, true)
            rankvalues = _rank_normalize(runs.values, runs.weights, T)
            r, rstatus = _rhat_parameter_result(d, p; runs, ranked=rankvalues)
            samples = reshape(runs.values, runs.n, 2length(d.chains))
            ranked = reshape(rankvalues, size(samples))
            bulk, bstatus = _ess_parameter_result(d, p; kind=:bulk, samples, ranked)
            tail, tstatus = _ess_parameter_result(d, p; kind=:tail, samples)
            error, mstatus = _mcse_parameter_result(d, p; samples)
        else
            r, rstatus = _rhat_parameter_result(d, p)
            bulk, bstatus = _budgeted_result(T) do
                _ess_parameter_result(d,p;kind=:bulk,max_lag_work)
            end
            tail, tstatus = _budgeted_result(T) do
                _ess_parameter_result(d,p;kind=:tail,max_lag_work)
            end
            error, mstatus = _budgeted_result(T) do
                _precision_parameter_result(d,p;max_lag_work)
            end
        end
        (; index, rhat=r, ess_bulk=bulk, ess_tail=tail, mcse_mean=error,
            n_draws, n_chains=length(d.chains),
            status=(rhat=rstatus, ess_bulk=bstatus, ess_tail=tstatus, mcse_mean=mstatus))
    end
    StructArrays.StructArray(rows)
end

function _budgeted_result(f, ::Type{T}) where {T}
    try
        f()
    catch error
        error isa _WorkLimitError || rethrow()
        (T(NaN), :work_limit)
    end
end
