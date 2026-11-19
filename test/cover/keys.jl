# The keys of the cover's proposals (see `Tempest.Compiler.ProposalKey`).

# A proposal's key written out whole: the text of the groups around it, its
# edit and its values' shapes, its values numbered together in the order the
# text first mentions them. Proposals have one key exactly where they have
# one such text.
function whole_key(p, a, q)
    around = Set{Int}()
    foreach(v -> union!(around, C.held(p, a, v)), C.fixes(q))
    io = IOBuffer()
    foreach(j -> print(io, "| ", C.numbered_text(p, p.groups[j])), sort!(collect(around); by = j -> a.places[j]))
    C.key_text(io, p, q)
    print(io, " || ", join(map(v -> "%$v", C.fixes(q)), " "))
    numbers = Dict{Int, Int}()
    text = C.numbered_values(String(take!(io)), v -> get!(numbers, v, length(numbers) + 1))
    string(text, "; ", C.value_shapes(p, numbers))
end

# The proposals, by their positions, in classes of one key.
function classes(keys)
    by = Dict{Any, Vector{Int}}()
    foreach(((n, k),) -> push!(get!(() -> Int[], by, k), n), enumerate(keys))
    sort!(collect(values(by)))
end

# The gradient of two loops stepping by two, unrolled, has many proposals
# around the same groups, a loop's steps among them: each key holds its
# place by number, and the text of each place is written once. Written out
# whole, the keys of a 64-step loop's proposals were 3.6 GB; here the
# proposals' edits together are shorter than the plan's groups' text, where
# their keys written whole are 87 times as long.
@testset "proposals around the same groups share their place's text" begin
    σ(x) = 1 / (1 + exp(-x))
    function f(xs, a, W, U)
        s = a
        for t in 1:2:15
            z = σ.(U * xs[:, :, t])
            s = 0.5 .* s .+ 0.5 .* tanh.(z)
        end
        for t in 1:2:15
            z = (U * xs[:, :, t]) .* (U * xs[:, :, t])
            y = transpose(W) * s
            s = σ.(y) .* s .+ (1 .- σ.(y)) .* tanh.(z)
        end
        sum(abs2, s)
    end
    g = (ys...) -> T.withgradient(f, ys...)
    args = (data(3, 2, 16), data(4, 2; scale = 0.5), data(4, 4; scale = 0.5), data(4, 3; scale = 0.5))
    ready = C.setup(trace(g, args...); target = T.CPUTarget(), build = false)
    C.in_tables(ready.tables) do
        d = ready.decider
        base = C.Planned(d.graph, C.Inputs(d)).plan
        proposed = [q for (q, _) in C.proposals(d, base)]
        a = C.Around(base, C.inputs(d, C.Proposal[]).fixed)
        places = C.Places()
        keys = [C.proposal_key(base, a, q, places) for q in proposed]
        @test classes(keys) == classes([whole_key(base, a, q) for q in proposed])
        @test 4 * length(places.numbers) < length(proposed)
        @test sum(k -> sizeof(k.edit), keys) < sum(g -> sizeof(C.numbered_text(base, g)), base.groups)
    end
end
