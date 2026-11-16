# A traced number meets Base's numbers as they meet each other: in Base's
# arithmetic and comparisons, either side, outside a broadcast and inside one,
# the program computes what Julia does, of the type Julia gives. `>`, `>=`
# and `!=` are Base's, from `<`, `<=` and `==`.
const PLAIN = (true, 2, Float16(0.5), 0.5f0, 0.5, big"0.5", 1 // 2, π)

# Whether the traced program computes `f(x)`, its value and its type.
function computes(f, x)
    y = interpret(trace(f, x), x)
    r = f(x)
    r isa AbstractArray ? eltype(y) == eltype(r) && y == r : typeof(y) == typeof(r) && isequal(y, r)
end

@testset "a traced number meets Base's numbers" begin
    x = Float32[0.25, 0.5]
    @testset "$f" for f in (+, -, *, /, max, min, atan, rem, mod, div, fld, cld, <, <=, >, >=, ==, !=)
        for v in PLAIN
            v === π && f in (==, !=) && continue
            @test computes(x -> f(sum(x), v), x)
            @test computes(x -> f(v, sum(x)), x)
            @test computes(x -> f.(x, v), x)
            @test computes(x -> f.(v, x), x)
        end
    end
    @test computes(x -> muladd(sum(x), 2, 0.5f0), x)
    @test computes(x -> muladd(2, sum(x), 0.5), x)

    # Tempest states each mixed method by hand, a traced number against any
    # `Real`, and Base's own mixed methods against a narrower type meet them
    # ambiguously or promote to a plain type the traced side never becomes.
    # Promotion to a traced type closes them all, once a plain number can be
    # made traced: a constant on no tape yet, or on the tape being traced.
    @testset "gap: Base's mixed methods against a float, a rational or π; closed by promotion to a traced type" begin
        @test_broken computes(x -> isless(sum(x), 0.5f0), x)
        @test_broken computes(x -> isless(0.5, sum(x)), x)
        @test_broken computes(x -> sum(x) == π, x)
        @test_broken computes(x -> sum(x) != π, x)
        @test_broken computes(x -> sum(x)^(1 // 2), x)
        @test_broken computes(x -> copysign(1.0, sum(x)), x)
        @test_broken computes(x -> flipsign(2, sum(x)), x)
        @test_broken computes(x -> sum(map(float, promote(sum(x), 0.5f0))), x)
        @test_broken computes(x -> fma(0.5f0, sum(x), 0.5f0), x)
    end
end
