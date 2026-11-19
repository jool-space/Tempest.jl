# A term reaches a subterm along many paths: a chain of maps, each reading
# the last twice, reaches its first along a number of paths doubling with
# each map, and a loop's gradient reaches its steps' values so. Each walk
# over terms visits a subterm once, so that its visits grow with the
# chain's length and not with its paths: before, the addresses of a loop's
# gradient of 64 steps were walked along 10^39 paths, and its preparation
# did not end.
import cuTile
const X = Base.get_extension(Tempest, :cuTileExt)

# A leaf that counts the visits of the walks that reach it.
struct Visited <: IR.Term
    visits::Base.RefValue{Int}
end
IR.variables(::Visited) = UInt64(0)
visit!(t::Visited) = (t.visits[] += 1; t)
C.at_the_step_parts(t::Visited, at, from, to, known) = visit!(t)
C.loaded_bytes_parts(t::Visited, graph, known) = (visit!(t); 0.0)
X.renamed_parts(t::Visited, σ, d, known) = visit!(t)
X.masked_value_of(c::X.Context, t::Visited, mask, d::IR.Domain, known) = (visit!(t); :finite)

# `t` read twice by each of `n` maps after it.
doubled(t, n) = foldl((x, _) -> IR.Apply(-, (x, x), Float32), 1:n; init = t)

@testset "a walk over terms visits a subterm once, whatever the paths to it" begin
    n = 40
    x = single(data(64, 32))
    p = trace(x -> foldl((y, _) -> y .* y, 1:n; init = x), x)
    ex = prepare(p, x; target = T.CUDATarget(T.cuTileGenerator(); device = 0), build = false)
    v, d = only(collect(zip(C.group_views(ex), ex.decisions)))
    # The kernel computes the whole chain, its one load reached along 2^40
    # paths: its addresses are the load's and the store's.
    @test count(t -> t isa IR.Apply, IR.subterms(values(v.terms))) == n
    @test length(X.view_addresses(v)) == 2
    c = X.Context(v, d.choice.claim.schedule)
    domain = IR.Domain((64, 32))
    leaf = Visited(Ref(0))
    walks = [
        "a load moved to the step" => () -> C.at_the_step(doubled(leaf, n), Dict(1 => 1), IR.constant(0), IR.var(1)),
        "the bytes a floor kernel loads" => () -> C.loaded_bytes(doubled(leaf, n), nothing, Dict{IR.Term, Float64}()),
        "a term with its positions exchanged" => () -> X.renamed(doubled(leaf, n), (IR.var(2), IR.var(1)), domain),
        "a term where a mask fails" => () -> X.masked_value(c, doubled(leaf, n), [IR.var(1)], domain),
    ]
    @testset "$name" for (name, walk) in walks
        leaf.visits[] = 0
        walk()
        @test leaf.visits[] == 1
    end
end
