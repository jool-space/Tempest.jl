# Content keys, the tuning record and the precompile workload: keys name
# content and are the same in every session, and the record lives in
# Tempest's scratch space unless the environment names a directory. That a
# kernel compiles once per key and loads from disk after is in cuda/cache.jl.
include("../fixtures/programs.jl")

# Code outside any package, identified by its code.
module Elsewhere
scale(x) = 2x
end

closure(s) = x -> s * x

@testset "content keys" begin
    # Generated names count up through a session; keys number them instead.
    a, b = gensym(:t), gensym(:t)
    @test C.content_key(:(for $a in 1:2; f($a); end)) == C.content_key(:(for $b in 1:2; f($b); end))
    @test C.content_key(:(x + 1)) != C.content_key(:(x + 2))
    @test C.content_key(1.0f0) != C.content_key(1.0)
    # A generated function is its source.
    f, g, h = map(n -> C.generated(:(function (x) x + $n end)), (1, 1, 2))
    @test C.content_key(f) == C.content_key(g) != C.content_key(h)
    @test_throws ErrorException C.content_key(Core.eval(C.Generated, :(function (x) x end)))
    # Package code is identified by its package; other code by its code, and
    # a closure by what it captures too.
    @test occursin("Tempest = sources", C.canonical_text(T.logsumexp))
    @test occursin("Base = julia", C.canonical_text(sin))
    before = C.content_key(Elsewhere.scale)
    Core.eval(Elsewhere, :(scale(x) = 3x))
    @test C.content_key(Elsewhere.scale) != before
    @test C.content_key(closure(1.0f0)) != C.content_key(closure(2.0f0))
    # A checked key names Tempest's own code without its identity, which an
    # edit anywhere in Tempest changes; the kernel is checked by what it uses.
    @test occursin("Tempest = checked by dependencies", C.canonical_text(T.logsumexp; checked = true))
    @test occursin("Base = julia", C.canonical_text(T.logsumexp, sin; checked = true))
    # Before Julia 1.12 there are no edges to check by, and the key is whole.
    @test (C.content_key(T.logsumexp; checked = true) == C.content_key(T.logsumexp)) == !C.EDGES
end

@testset "what a kernel depends on of Tempest's code" begin
    # The methods of Tempest's that inference reached, from the edges it keeps:
    # here a fold's observe, through a function outside Tempest that calls it.
    observed(s) = IR.fold_observe(IR.SumFold(), s)
    observed((1.0f0,))
    mi = only(methods(observed)).specializations
    mi = mi isa Core.MethodInstance ? mi : first(filter(!isnothing, collect(mi)))
    bytes = C.dependencies(mi.cache, m -> isdefined(m, :cache) ? m.cache : nothing)
    if !C.EDGES
        @test isempty(bytes)
    else
        records = C.deserialize(IOBuffer(bytes))
        @test any(r -> r[1] === :method && r[2] <: Tuple{typeof(IR.fold_observe), Vararg}, records)
        @test all(r -> r[1] !== :method || C.own_code(which(r[2]).module), records)
        @test C.current(bytes)
        # Changed code, a record that cannot be read back, and none.
        kind, sig, _ = first(filter(r -> r[1] === :method, records))
        io = IOBuffer(); C.serialize(io, [(kind, sig, "other code")])
        @test !C.current(take!(io))
        @test !C.current(UInt8[1, 2, 3])
    end
    @test C.current(UInt8[])
end

@testset "where the tuning record lives" begin
    # In Tempest's scratch space, unless the environment names a directory.
    scratch(depot) = joinpath(depot, "scratchspaces", string(Base.PkgId(Tempest).uuid), "tuning")
    withenv("TEMPEST_CACHE" => nothing) do
        @test C.tuning_directory() == scratch(first(DEPOT_PATH)) && isdir(C.tuning_directory())
    end
    elsewhere = mktempdir()
    withenv("TEMPEST_CACHE" => elsewhere) do
        @test C.tuning_directory() == joinpath(elsewhere, "tuning")
        @test dirname(C.record_path(T.CPUTarget())) == joinpath(elsewhere, "tuning")
    end
    # The path is found when Tempest loads, not baked in when it precompiles:
    # a process with another first depot keeps its record there.
    depot = mktempdir()
    script = "using Tempest; print(Tempest.Compiler.tuning_directory())"
    path = withenv("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ';' : ':'), "TEMPEST_CACHE" => nothing) do
        readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`)
    end
    @test path == scratch(depot)
end

# What Tempest compiles while it precompiles, which development sessions skip.
@testset "the precompile workload" begin
    @test T.workload() === nothing
end
