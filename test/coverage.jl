# The coverage suite: compositions nobody wrote code for plan and run, and each
# plan has the structure recorded for it (fixtures/compositions.jl): its kernels,
# the bytes it stores, and no stored value larger than a bound. Plans are
# chosen by estimate, so the checks are deterministic and do not depend on the
# device's speed. An expectation not met yet is `@test_broken`, under a
# testset that names what closes the gap.
# Plans are by estimate: setup.jl keeps measurements out of the user's tuning
# record, so none is recorded to override an estimate.
include("fixtures/compositions.jl")

"The largest difference from the interpreter's results, relative to their largest magnitude."
function relative_error(ex, reference)
    got = filter(!isnothing, C.leaves(outputs(execute!(ex))))
    maximum(map((a, b) -> maximum(abs.(Array(a) .- b)) / max(maximum(abs.(b)), eps(Float32)), got, reference))
end

function structure(ex)
    s = C.structure(ex.plan)
    largest = maximum(v -> C.value_bytes(ex.plan.graph.values[v]), s.stored; init = 0)
    (kernels = s.kernels, bytes = s.bytes, largest)
end

# Kernels and bytes are recorded exactly; the largest value is a bound.
holds(field, got, expected) = field === :largest ? got <= expected : got == expected

function check_structure(got, expected)
    for (field, e) in pairs(expected)
        e isa Gap && continue
        @test holds(field, got[field], e)
    end
    for closer in unique(e.closer for e in expected if e isa Gap)
        @testset "gap: $closer" begin
            for (field, e) in pairs(expected)
                e isa Gap && e.closer == closer && @test_broken holds(field, got[field], e.value)
            end
        end
    end
end

const TOLERANCE = 1.0e-3

if CUDA_FUNCTIONAL
    references = Dict(c.name => filter(!isnothing, C.leaves(interpret(c.program, c.args...))) for c in COMPOSITIONS)
    # The streaming plan: every fold that can be recomputed where it is read
    # is. The structure recorded for a composition is this plan's. The cover
    # chooses by cost, so its plan may store more: it is correct, and never
    # worse by its own estimate than the streaming plan.
    streaming(c) = C.recomputable_folds(C.rewrite(C.lower(c.program)))
    prepared(c, target; recompute = ()) = prepare(c.program, map(CuArray, c.args)...; target, measure = false, recompute)
    function estimated(ex)
        d = C.Decider(ex.target, ex.plan.graph; measure = false)
        sum(g -> C.decide(d, ex.plan, g).choice.estimate, filter(g -> C.chosen(ex.plan, g), ex.plan.groups); init = 0.0)
    end

    @testset "coverage" begin
        @testset "every implementation" begin
            @testset "$(c.name)" for c in COMPOSITIONS
                ex = prepared(c, T.CUDATarget(); recompute = streaming(c))
                @test relative_error(ex, references[c.name]) <= TOLERANCE
                check_structure(structure(ex), c.expected)
                chosen = prepared(c, T.CUDATarget())
                @test relative_error(chosen, references[c.name]) <= TOLERANCE
                @test estimated(chosen) <= estimated(ex) * (1 + 1.0e-6)
            end
        end
        # Removing an implementation changes how fast a program runs, never
        # whether it runs: without cuBLAS (the generator and the floor), and
        # without the generator (cuBLAS and the floor).
        @testset "an implementation removed" begin
            @testset "$(c.name)" for c in COMPOSITIONS
                ex = prepared(c, T.CUDATarget(cuTile); recompute = streaming(c))
                @test relative_error(ex, references[c.name]) <= TOLERANCE
                check_structure(structure(ex), c.generated)
                ex = prepared(c, T.CUDATarget(cuBLAS))
                @test !any(k -> occursin("cuTile", string(k)), ex.kernels)
                @test relative_error(ex, references[c.name]) <= TOLERANCE
            end
        end
    end
else
    @info "coverage: no functional GPU, so the suite's CUDA cases are skipped"
end
