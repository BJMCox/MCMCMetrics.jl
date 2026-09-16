using Documenter
using MCMCMetrics

DocMeta.setdocmeta!(MCMCMetrics, :DocTestSetup, :(using MCMCMetrics); recursive=true)

makedocs(
    root = @__DIR__,
    sitename = "MCMCMetrics.jl",
    repo = Documenter.Remotes.GitHub("BJMCox", "MCMCMetrics.jl"),
    modules = [MCMCMetrics],
    checkdocs = :exports,
    doctest = true,
    warnonly = false,
    format = Documenter.HTML(
        prettyurls = true,
        edit_link = "main",
        canonical = "https://bjmcox.github.io/MCMCMetrics.jl/",
    ),
    pages = [
        "Getting started" => "index.md",
        "Guides" => [
            "Array inputs and results" => "arrays.md",
            "Diagnostic families" => "families.md",
            "Online diagnostics" => "online.md",
            "Sampler adapters" => "adapters.md",
        ],
        "API reference" => "api.md",
    ],
)
