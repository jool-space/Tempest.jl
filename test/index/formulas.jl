# Index formulas: simplified divisions and remainders agree with `fld` and
# `mod` everywhere on their domain, and a small term below a common factor
# drops out.
@testset "index formulas" begin
    d = IR.Domain((4, 16, 3))
    e = IR.add(IR.var(1), IR.mul(IR.var(2), 4, d), d) # i1 + 4·i2, with i1 < 4
    @test IR.fdiv(e, 32, d) == IR.fdiv(IR.var(2), 8, d)
    @test IR.fmod(e, 32, d) == IR.add(IR.var(1), IR.mul(IR.fmod(IR.var(2), 8, d), 4, d), d)
    points = Iterators.product(0:3, 0:15, 0:2)
    for a in -3:6, b in (0, 2, 4, 8, 12), c in (0, 1, 6), k0 in -5:7, k in (2, 3, 4, 6, 8, 12, 32)
        e = IR.add(IR.add(IR.mul(IR.var(1), a, d), IR.mul(IR.var(2), b, d), d), IR.add(IR.mul(IR.var(3), c, d), k0, d), d)
        q, r = IR.fdiv(e, k, d), IR.fmod(e, k, d)
        @test all(x -> IR.evaluate_index(q, x) == fld(IR.evaluate_index(e, x), k) &&
                       IR.evaluate_index(r, x) == mod(IR.evaluate_index(e, x), k), points)
        # The two recombine to the expression they came from.
        @test IR.add(IR.mul(q, k, d), r, d) == e
    end
    # Digits of a number recombine into it, also when nested divisions have
    # composed: (i mod 32) + 32·((i ÷ 32) mod 2) + 64·(i ÷ 64) is i.
    d = IR.Domain((128,))
    i = IR.var(1)
    digits = [IR.fmod(i, 32, d), IR.mul(IR.fmod(IR.fdiv(i, 32, d), 2, d), 32, d), IR.mul(IR.fdiv(i, 64, d), 64, d)]
    @test foldl((a, b) -> IR.add(a, b, d), digits) == i
    @test IR.is_affine(foldl((a, b) -> IR.add(a, b, d), reverse(digits)))
end
