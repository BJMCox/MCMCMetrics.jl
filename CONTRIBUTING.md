# Development

Run package tests from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Tests check public behavior against independent mathematical references.
Run with `--threads=4` to exercise concurrent online updates on multiple threads.

Each chain adapter has an independent test project. Run one from the repository root:

```sh
julia --project=test/optional/flexichains -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=test/optional/flexichains test/optional/flexichains/runtests.jl
```

Replace `flexichains` with `mcmcchains` or `inferenceobjects` for the other adapters.
Core tests do not install these chain packages.

Run pinned external reference checks in their own environment:

```sh
julia --project=test/reference -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=test/reference test/reference/compare.jl
```

Run replicated IID and AR(1) uncertainty checks separately:

```sh
julia --project=test/reference test/slow/processes.jl
julia --project=test/reference test/slow/extended.jl
```

Run benchmarks in their own environment:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=benchmark --threads=4 benchmark/diagnostics.jl
```

Build the docs and run doctests:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Generated HTML is in `docs/build`. Serve that directory with a local HTTP server
to view it.

## CI

The workflow checks Julia 1.10 and current stable Julia, one and four threads,
Linux/macOS/Windows, and strict documentation builds. Each chain adapter runs
separately on Julia 1.10 and current stable Julia on Linux. The workflow retains
documentation and coverage reports as build artifacts.

The Linux stable-Julia core and adapter jobs upload coverage for `src` and `ext` to
[Codecov](https://app.codecov.io/gh/BJMCox/MCMCMetrics.jl) through GitHub OIDC.
The upload needs no stored Codecov token. Upload errors fail the job.

GitHub Pages uses Actions as its publishing source. After the test and docs jobs
pass on `main`, the deployment job publishes
[the documentation](https://bjmcox.github.io/MCMCMetrics.jl/).

## Dependencies

Keep the runtime dependency set small. Keep documentation, reference packages,
and benchmark tools in their own environments. Audit the transitive closure
when a runtime dependency changes.

## Releases

Update the changelog and package version before a release. Verify CI on the exact
release commit and obtain maintainer approval before tagging or registration.
