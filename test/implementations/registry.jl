# The implementation registry on CPU: ranks, names, scopes and explicit
# candidates decide the order; executables keep the candidates they resolved.
# Modules registered here are removed again, since workers run several tests.
module Zeta end
module Alpha end
module Beta end

struct Ranked <: Implementation
    name::Symbol
end
Ranked(; name = :ranked) = Ranked(name)
Tempest.implementation_name(r::Ranked) = r.name

struct Late <: Implementation end
Tempest.implementation_name(::Late) = :late

names(target = CPUTarget()) = map(Tempest.implementation_name, default_implementations(target))

@testset "Implementation registry" begin
    try
        # Equal ranks fall back to module names; registration order is irrelevant.
        Tempest.register!(Zeta, Ranked; target = CPUTarget, rank = 25)
        Tempest.register!(Alpha, () -> Ranked(:alpha); target = CPUTarget, rank = 25)
        Tempest.register!(Beta, Late; target = CPUTarget, rank = 10, active = false)
        @test names() == (:cpu, :alpha, :ranked, :LinearAlgebra)
        # Only implementations registered for the target take part.
        @test :alpha ∉ names(Tempest.CUDATarget())

        x = reshape(Float64.(1:6), 2, 3)
        before = prepare_forward(x -> sin.(x), x)
        @test before.candidate_sources == [:target, :registry, :registry, :target]
        Tempest.activate(Beta)
        @test names() == (:late, :cpu, :alpha, :ranked, :LinearAlgebra)
        after = prepare_forward(x -> sin.(x), x)
        @test first(after.candidates) === :late
        @test occursin("late (registry) → cpu (target)", sprint(explain, after))
        @test :late ∉ before.candidates

        # Scopes nest on the registry and end with their function.
        Tempest.deactivate(Beta, Zeta) do
            @test names() == (:cpu, :alpha, :LinearAlgebra)
            Tempest.activate(Zeta) do
                ex = prepare_forward(x -> sin.(x), x)
                @test ex.candidates == [:cpu, :alpha, :ranked, :LinearAlgebra]
                @test ex.candidate_sources == [:target, :registry, :scoped, :target]
            end
        end
        @test names() == (:late, :cpu, :alpha, :ranked, :LinearAlgebra)

        # Keywords, or an instance of a registered type, replace the active one.
        Tempest.activate(Zeta; name = :keyword)
        @test :keyword in names()
        Tempest.activate(Ranked(:other))
        @test names() == (:late, :cpu, :alpha, :other, :LinearAlgebra)
        Tempest.deactivate(Beta)
        @test :late ∉ names()

        # Unregistered modules and types are explained.
        @test_throws "has not registered a Tempest implementation" Tempest.activate(Base)
        @test_throws "cannot be deactivated" Tempest.deactivate(Base)
        @test_throws "no registered module provides" Tempest.activate(CPUKernels())
        @test_throws ArgumentError Tempest.register!(Alpha, Ranked; rank = 0, target = Int)
    finally
        foreach(m -> delete!(Tempest.REGISTRY, m), (Zeta, Alpha, Beta))
    end
    @test default_implementations(CPUTarget()) == (CPUKernels(), LinearAlgebraGemm())
end
