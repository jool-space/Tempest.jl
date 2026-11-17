using Tempest
using Documenter

DocMeta.setdocmeta!(Tempest, :DocTestSetup, :(using Tempest; using Tempest.Compiler: trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs); recursive = true)

makedocs(;
    modules = [Tempest],
    authors = "AntonOresten <antonoresten@proton.me> and contributors",
    sitename = "Tempest.jl",
    format = Documenter.HTML(;
        canonical = "https://docs.jool.space/Tempest.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Tempest" => "api.md",
        "Tempest.Compiler" => [
            "Beneath compile" => "compiler.md",
            "Adding a target" => "targets.md",
            "Building kernels" => "kernels.md",
            "IR" => "ir.md",
            "Implement" => "implement.md",
        ],
    ],
    # Internal docstrings are for reading the source; the manual covers the public names,
    # and a public name missing from it is a warning in the log, not a failed build.
    checkdocs = :public,
    warnonly = [:missing_docs],
)

deploydocs(;
    repo="github.com/jool-space/Tempest.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Tempest.jl",
)
