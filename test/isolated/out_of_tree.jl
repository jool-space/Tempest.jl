# Run in a fresh process by extensions/loading.jl: a package outside Tempest
# (test/packages/MyLib, a path dependency of the test project) registers its
# implementation from __init__, and `Tempest.activate` makes it a default
# candidate of later executables only.
using Tempest, Test
using MyLib

g(x, rows) = Tempest.take_rows(x, rows)
x = reshape(sin.(1.0:35.0), 5, 7)
rows = [3, 1, 5, 2, 2, 4, 1]
expected = reshape([x[r, n] for (n, r) in enumerate(rows)], 1, 7)
prep(; kwargs...) = prepare_forward(g, x, rows; kwargs...)

@testset "An implementation package outside Tempest" begin
    before = prep()
    @test :mylib ∉ before.candidates
    @test only(before.providers) === :cpu

    Tempest.activate(MyLib)
    ex = prep()
    explanation = sprint(explain, ex)
    @test occursin("Implementation candidates: mylib (registry) → cpu (target) → LinearAlgebra (target)", explanation)
    @test occursin("call take_rows [mylib]", explanation)
    @test execute!(ex) ≈ expected
    # Activation does not change an executable prepared before it.
    @test :mylib ∉ before.candidates
    @test execute!(before) ≈ expected

    # Keywords construct the instance; a scope overrides the registry.
    Tempest.activate(MyLib; name = :mine)
    @test only(prep().providers) === :mine
    scoped = Tempest.activate(prep, MyLib.MyKernels(name = :scoped))
    @test occursin("scoped (scoped)", sprint(explain, scoped))
    @test only(Tempest.deactivate(prep, MyLib).providers) === :cpu
    @test only(prep().providers) === :mine
    # An explicit candidate precedes the registry and is not tried twice.
    explicit = prep(; implementations = (MyLib.MyKernels(name = :explicit),))
    @test explicit.candidates == [:explicit, :cpu, :LinearAlgebra]
    @test explicit.candidate_sources == [:explicit, :target, :target]

    Tempest.deactivate(MyLib)
    @test only(prep().providers) === :cpu
end
