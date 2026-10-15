# Run in a fresh process by extensions/loading.jl: a package outside Tempest
# (test/packages/MyLib, a path dependency of the test project) registers its
# implementation from __init__, and `Tempest.activate` makes it a default
# candidate of later executables only.
using Tempest, Test
using MyLib

f(x) = Tempest.logsumexp(x; dims = 1)
x = reshape(sin.(1.0:35.0), 5, 7)
expected = log.(sum(exp.(x); dims = 1))

@testset "An implementation package outside Tempest" begin
    before = prepare_forward(f, x)
    @test :mylib ∉ before.candidates
    @test only(before.providers) === :cpu

    Tempest.activate(MyLib)
    ex = prepare_forward(f, x)
    explanation = sprint(explain, ex)
    @test occursin("Implementation candidates: mylib (registry) → cpu (target) → LinearAlgebra (target)", explanation)
    @test occursin("call logsumexp [mylib]", explanation)
    @test execute!(ex) ≈ expected
    # Activation does not change an executable prepared before it.
    @test :mylib ∉ before.candidates
    @test execute!(before) ≈ expected

    # Keywords construct the instance; a scope overrides the registry.
    Tempest.activate(MyLib; name = :mine)
    @test only(prepare_forward(f, x).providers) === :mine
    scoped = Tempest.activate(() -> prepare_forward(f, x), MyLib.MyKernels(name = :scoped))
    @test occursin("scoped (scoped)", sprint(explain, scoped))
    @test only(Tempest.deactivate(() -> prepare_forward(f, x), MyLib).providers) === :cpu
    @test only(prepare_forward(f, x).providers) === :mine
    # An explicit candidate precedes the registry and is not tried twice.
    explicit = prepare_forward(f, x; implementations = (MyLib.MyKernels(name = :explicit),))
    @test explicit.candidates == [:explicit, :cpu, :LinearAlgebra]
    @test explicit.candidate_sources == [:explicit, :target, :target]

    Tempest.deactivate(MyLib)
    @test only(prepare_forward(f, x).providers) === :cpu
end
