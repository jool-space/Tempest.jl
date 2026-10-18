using Tempest
using Documenter

DocMeta.setdocmeta!(Tempest, :DocTestSetup, :(using Tempest); recursive = true)

makedocs(;
    modules = [Tempest],
    authors = "AntonOresten <antonoresten@proton.me> and contributors",
    sitename = "Tempest.jl",
    format = Documenter.HTML(;
        canonical = "https://docs.jool.space/Tempest.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md", "API" => "api.md"],
    # Internal docstrings are for reading the source; the manual covers the exports.
    checkdocs = :exports,
)

deploydocs(;
    repo="github.com/jool-space/Tempest.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Tempest.jl",
)
