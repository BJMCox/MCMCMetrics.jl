# Sampler adapters

MCMCMetrics has no BAT or sampler dependency. Keep sampler ownership, warmup,
cycle selection, parameter names and termination policy in the adapter.

## Optional chain packages

Load **FlexiChains**, **MCMCChains**, or **InferenceObjects** alongside MCMCMetrics
to enable its adapter. These weak dependencies are optional for array inputs
and online accumulators. Use qualified calls such as `MCMCMetrics.rhat` to avoid
conflicts with diagnostics exported by other packages.

| Container | Default selection | Layout handling |
|:--|:--|:--|
| `FlexiChains.FlexiChain` | Parameters, excluding extras | Public indexing stacks each parameter into draw × chain × component axes |
| `MCMCChains.Chains` | The `:parameters` section | Views retain separate chains in its draw × parameter × chain storage |
| `InferenceObjects.InferenceData` | The `posterior` group | Named `draw` and `chain` axes determine the layout |
| `InferenceObjects.Dataset` | All variables in the supplied dataset | Named axes determine the layout, regardless of physical axis order |

These examples share 1,000 synthetic independent draws per chain for one scalar parameter:

```@example adapters
using MCMCMetrics, Random
x = randn(Xoshiro(42), Float32, 1_000, 4);  # draw × chain
nothing # hide
```

### FlexiChains

```@example adapters
import FlexiChains
samples = map(value -> Dict(FlexiChains.Parameter(:alpha) => value), x)
fc = FlexiChains.FlexiChain{Symbol}(size(x)..., samples)
MCMCMetrics.rhat(fc)
```

Only parameters enter diagnostics. An extra with the same name cannot replace a
parameter. Keys remain intact, including the `VarName` keys used by Turing.

### MCMCChains

MCMCChains stores **draw × parameter × chain** arrays:

```@example adapters
import MCMCChains
mc = MCMCChains.Chains(reshape(x, size(x, 1), 1, size(x, 2)), [:alpha])
tail_ess = MCMCMetrics.ess(mc; parameters=(:alpha,), kind=:tail)
@assert isfinite(tail_ess[:alpha]) # hide
tail_ess # hide
```

The default selection is the `:parameters` section. Component names such as
`Symbol("beta[1]")` remain separate scalar keys. Chain boundaries stay intact.

With short chains, a tail indicator can be constant in a split chain, yielding
`NaN` tail ESS. Inspect `MCMCMetrics.diagnostics(mc)[:alpha].status` for the cause.

### InferenceObjects

Each variable can have its own component shape and numeric type:

```@example adapters
import InferenceObjects
posterior = InferenceObjects.namedtuple_to_dataset((
    alpha=x, beta=cat(Float64.(x), -Float64.(x); dims=3)))
data = InferenceObjects.InferenceData(; posterior)
report = MCMCMetrics.diagnostics(data)
report[:beta].index
```

InferenceObjects requires separate `draw` and `chain` axes. A combined `sample`
axis does not identify independent chains. `warmup_posterior` and `sample_stats`
are not included when passing an InferenceData object.

### Results and selection

Every adapter returns a dictionary keyed by the original parameter key. Each
value is the corresponding array method's result: scalar metrics for scalar
parameters, or arrays with the parameter's component shape. `diagnostics` returns
one StructArray per parameter, with component indices and per-metric statuses.
Each variable retains its numeric precision, including in mixed Float32/Float64 inputs.

Result arrays use positional indices and preserve component and chain order.
Coordinate labels stay in the source container. Use `parameters=(key1, key2)`
to select variables, and the container's indexing to select draws and chains.
Container methods determine their axes and reject `drawdim` and `chaindim` overrides.

Supported methods are `rhat`, `ess`, `mcse`, `iact`, `autocor`, `diagnostics`,
`diagnostic_curve`, `geweke`, `heidelberger_welch`, `raftery_lewis`,
`rank_histogram`, and `rank_ecdf`. Their estimator keywords still apply.
Explicit integer `counts` must match the selected draw and chain order.
Sampler weights never become repetition counts automatically.

Selected variables need supported numeric values and fixed component shapes.
Missing observations cause an error. Missing-capable storage containing only
numbers retains their precision. No draws or variables are silently dropped.

These methods run batch diagnostics. Online updates, joint multivariate checks,
predictive scores, and sampler metadata use explicit numeric inputs.

## Batch checkpoints

1. Select retained production draws from one sampling cycle.
2. Keep independent chains in distinct slots.
3. Expose flat numeric parameter matrices without inventing a new chain axis.
4. Supply exact integer run counts for compressed rejected states.
5. Map numeric results back to parameter names and sampler-owned policy.

BAT-style nested storage can expose one parameter × stored-draw matrix per
chain with `ArraysOfArrays.flatview`. Pass that vector of matrices with
`drawdim=2`. StructArrays can retain positions, counts and sampler IDs in
separate columns; only positions and checked counts enter ordinary diagnostics.
The package accepts matrix views and retains the documented parameter order.

If the sampler stores repetition counts as floating values, validate that every
positive value is an exactly represented, nonnegative integer fitting Int before
conversion. Arbitrary importance weights are not repetition counts.
Zero-count placeholder values do not contribute observations.

Use `diagnostics(chains; drawdim=2, counts)` for a typed per-parameter report.
Inspect each metric's status. `:work_limit`, `:unequal_chain_lengths`, a constant
transform or failed LRV must not satisfy a convergence or precision criterion.
Apply a convergence threshold only to the intended R-hat variant.

## Online checkpoints

Allocate one `OnlineDiagnostics` for a cycle, fixing type, parameter shape,
chain slots and requested metrics. Append only new retained observations.
Preserve chronology within each chain; separate chains may update concurrently.
Snapshots own their results and acquire all addressed locks coherently.

Finish all producers before `empty!` starts another cycle. Do not merge or reuse
an ordered accumulator across unrelated chains, cycles or adaptation phases.
Moment-only `merge!` has a separate disjoint-chunk contract.

Online raw-mean precision uses dyadic batch means. It cannot supply modern
rank/folded R-hat or bulk ESS. A BAT policy that requires those metrics can keep
its retained chains for batch checkpoints while using online moments/precision
between checkpoints. MCMCMetrics sets no automatic checkpoint schedule or
termination threshold. Repeated checking needs sampler-owned stopping policy.

## Metadata boundaries

Energy, divergences, acceptance probabilities, tree depths, work, log weights and
pointwise log likelihoods are explicit numeric inputs. The adapter knows which
transition produced each field. A repetition count repeats the whole supplied
observation and metadata, not just its parameter position. If metadata differs
inside a repeated-position run, split the run or retain the raw transition data.
Do not repeat a per-run aggregate work total once per represented transition.

Predictive diagnostics require a stable observation identity and likelihood
factorization. Nested R-hat requires actual grouped initialization. Temperature
replicas must not silently become independent posterior-chain IDs.

The package supplies batch and online numerical seams. BAT integration and
its convergence-test type hierarchy remain BAT-side work.
