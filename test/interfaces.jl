# Extensions are written against Tempest's public names, as packages of their
# own would be: what they implement is `Tempest.Implementations`, and what they
# read is `Tempest.IR` and the plan names public in `Tempest.Compiler`. An
# extension that still reaches past them is a broken test until it is rewritten.
const ROOT = joinpath(@__DIR__, "..")
const EXTENSIONS = sort!(map(n -> replace(n, r"\.jl$" => ""), readdir(joinpath(ROOT, "ext"))))
const AGAINST_INTERFACES = ["cuBLASExt"]

# The source files of an extension: ext/Name.jl, or every file in ext/Name/.
function sources(name)
    file, dir = joinpath(ROOT, "ext", name * ".jl"), joinpath(ROOT, "ext", name)
    isfile(file) ? [file] : sort!(filter(endswith(".jl"), readdir(dir; join = true)))
end

# `a.b.c` as `[:a, :b, :c]`, or nothing for any other expression.
function dotted(e)
    e isa Symbol && return [e]
    e isa Expr && e.head === :. || return nothing
    # `using A.B: c` spells paths with symbols, code with quoted ones.
    all(a -> a isa Symbol, e.args) && return collect(Symbol, e.args)
    length(e.args) == 2 && e.args[2] isa QuoteNode || return nothing
    front = dotted(e.args[1])
    front === nothing ? nothing : push!(front, e.args[2].value)
end

# Each name along `path` from module `m` that is not public where it is
# defined, written in full; the walk stops at the first name not a module.
function private_steps!(found, m::Module, path)
    for name in path
        Base.ispublic(m, name) || push!(found, string(m, ".", name))
        isdefined(m, name) && (x = getglobal(m, name)) isa Module || return
        m = x
    end
end

"The names of Tempest's modules the files use that are not public."
function private_names(files)
    found, aliases = Set{String}(), Dict{Symbol, Module}(:Tempest => Tempest)
    module_of(path) = foldl((m, n) -> getglobal(m, n), path[2:end]; init = aliases[path[1]])
    function walk(e)
        e isa Expr || return
        if e.head in (:using, :import)
            for a in e.args
                if a isa Expr && a.head === :(:)
                    path = dotted(a.args[1])
                    haskey(aliases, path[1]) || continue
                    private_steps!(found, aliases[path[1]], path[2:end])
                    for n in a.args[2:end]
                        name = n.head === :as ? last(dotted(n.args[1])) : last(dotted(n))
                        private_steps!(found, module_of(path), [name])
                    end
                elseif a isa Expr && a.head === :as
                    path = dotted(a.args[1])
                    haskey(aliases, path[1]) || continue
                    private_steps!(found, aliases[path[1]], path[2:end])
                    aliases[a.args[2]] = module_of(path)
                else
                    path = dotted(a)
                    haskey(aliases, path[1]) && private_steps!(found, aliases[path[1]], path[2:end])
                end
            end
        elseif e.head === :. && (path = dotted(e)) !== nothing && haskey(aliases, path[1])
            private_steps!(found, aliases[path[1]], path[2:end])
        else
            foreach(walk, e.args)
        end
    end
    foreach(f -> walk(Meta.parseall(read(f, String); filename = f)), files)
    sort!(collect(found))
end

@testset "public names are defined" begin
    for m in (Tempest, T.IR, T.Compiler, T.Implementations, T.Operations)
        @test filter(n -> !isdefined(m, n), names(m)) == Symbol[]
    end
end

@testset "$name uses public names only" for name in EXTENSIONS
    if name in AGAINST_INTERFACES
        @test private_names(sources(name)) == String[]
    else
        @test_broken private_names(sources(name)) == String[]
    end
end

# The check sees through aliases, and through `import` and `using` lists.
@testset "what the check finds" begin
    source(code) = (f = tempname() * ".jl"; write(f, code); [f])
    @test private_names(source("Tempest.Compiler.Plan; Tempest.Compiler.lower")) == ["Tempest.Compiler.lower"]
    @test private_names(source("import Tempest.Compiler as C\nC.lower(p)")) == ["Tempest.Compiler.lower"]
    @test private_names(source("using Tempest.Compiler: Plan, lower")) == ["Tempest.Compiler.lower"]
    @test private_names(source("import Tempest.Implementations: claims, REGISTERED")) == ["Tempest.Implementations.REGISTERED"]
    @test private_names(source("using Tempest: CUDATarget; using Tempest.Operations")) == String[]
end
