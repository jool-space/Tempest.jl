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
            "Many-array updates" => "many-array-updates.md",
            "Semantic verbs" => "verbs.md",
            "Implementation reverse rules" => "reverse-rules.md",
            "Scalar derivative rules" => "scalar-rules.md",
        ],
        "Compiler" => [
            "Implementations" => "implementations.md",
            "Graph regions" => "graph-regions.md",
            "Proposed regions" => "proposed-regions.md",
            "Layouts" => "layouts.md",
            "Source guide" => "internals.md",
            "Direction" => "compiler-direction.md",
            "Interface direction" => "interface-direction.md",
            "Spelling invariance" => "spelling-invariance.md",
        ],
        "API" => "api.md",
        "Reports" => [
            "Decision log" => "review-decisions.md",
            "Outside review" => "outside-review.md",
            "Transformer training step" => "transformer-step.md",
            "Reduction domains and schedules" => "reduction-domains.md",
            "cuTile issues" => "cutile-issues.md",
        ],
    ],
)

if !("--local" in ARGS)
    deploydocs(; repo = "github.com/jool-space/Tempest.jl", devbranch = "main")
end
