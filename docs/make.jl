using Tempest
using Documenter

DocMeta.setdocmeta!(Tempest, :DocTestSetup, :(using Tempest); recursive = true)

makedocs(;
    modules = [Tempest],
    authors = "AntonOresten <antonoresten@proton.me> and contributors",
    sitename = "Tempest.jl",
    format = Documenter.HTML(;
        canonical = "https://jool-space.github.io/Tempest.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Guide" => [
            "Usage" => "usage.md",
            "Training state" => "state.md",
            "Semantic verbs" => "verbs.md",
            "Implementation reverse rules" => "reverse-rules.md",
            "Scalar derivative rules" => "scalar-rules.md",
        ],
        "Compiler" => [
            "Architecture" => "architecture.md",
            "Implementations" => "implementations.md",
            "Graph regions" => "graph-regions.md",
            "Proposed regions" => "proposed-regions.md",
            "Layouts" => "layouts.md",
            "Source guide" => "internals.md",
        ],
        "API" => ["Exported" => "api.md", "Implementation API" => "api-implementations.md"],
        "History" => "history.md",
        "Reports (historical)" => [
            "Outside review" => "outside-review.md",
            "Spelling invariance audit" => "spelling-invariance.md",
            "Decision log" => "review-decisions.md",
            "Transformer training step" => "transformer-step.md",
            "Reduction domains and schedules" => "reduction-domains.md",
            "Many-array updates" => "many-array-updates.md",
            "cuTile issues" => "cutile-issues.md",
        ],
    ],
)

if !("--local" in ARGS)
    deploydocs(; repo = "github.com/jool-space/Tempest.jl", devbranch = "main")
end
