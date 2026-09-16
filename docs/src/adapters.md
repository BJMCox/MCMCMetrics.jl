# Sampler adapters

MCMCMetrics has no BAT or sampler dependency. Keep sampler ownership, warmup,
cycle selection, parameter names and termination policy in the adapter.

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
