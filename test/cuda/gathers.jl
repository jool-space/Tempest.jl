# Embeddings on the device: gathers by runtime indices inside the kernels that
# read them, folds and broadcasts over part of the axes a schedule collapses,
# and a gather's adjoint, a scatter that sums by key, on cuTile and the floor.

# A layer norm in Float32, as cuda/folds.jl has it.
function layernorm32(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
end

# Gathers by runtime indices, element by element inside the kernels that read
# them and never stored: an embedding lookup with pointwise work and a sum; a
# lookup with a layer norm, where the embedding plus a bias, read three times,
# joins the kernel that reads it (a proposal of the planner's own); a long sum
# over gathered columns; and a product with a gathered factor.
tokens = Int32.(mod1.(37 .* (1:240), 1000))
const GATHERS = [
    ("embedding, pointwise work and a sum", (E, t) -> sum(abs2, tanh.(E[:, t]); dims = 1),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3)), 1),
    ("embedding and layer norm", (E, t, b, w) -> layernorm32(E[:, t] .+ b, w),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64; scale = 0.3), data(64; scale = 0.1)), 1),
    ("long sum over gathered columns", (E, t) -> sum(E[:, t]; dims = 2), (data(64, 1000; scale = 0.01), tokens), 1),
    ("product with a gathered factor", (W, E, t) -> W * E[:, t], (data(96, 64; scale = 0.02), data(64, 1000; scale = 0.01), tokens), 1),
]

@testset "gathers on cuTile" begin
    @testset "$name" for (name, f, args, kernels) in GATHERS
        args = map(single, args)
        p = trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
        @test length(ex.kernels) == kernels && all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-3)
    end
end

# Folds and broadcasts over part of the axes a schedule collapses into one
# tile axis: a sum over the middle axis of an array, spelled directly and
# through `permutedims`, and a long one that splits; a positional embedding
# broadcast over the batch; a table broadcast over heads and batch, as rotary
# positions read it. The gradient of the embedding and layer norm sums over
# the batch alone, and its scatter sums by key in the same kernel.
const PARTIAL_COLLAPSE = [
    ("sum over the middle axis", x -> sum(x; dims = 2), (data(8, 64, 6),), false),
    ("sum over the middle axis, permuted", x -> sum(permutedims(x, (2, 1, 3)); dims = 1), (data(64, 8, 6),), false),
    ("long sum over the middle axis", x -> sum(abs2, x; dims = 2), (data(40, 3000, 7),), false),
    ("positional embedding", (E, t, P) -> E[:, t] .+ P, (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64, 80)), false),
    ("a table broadcast over heads and batch", (x, c) -> x .* c, (data(32, 4, 64, 16), data(32, 1, 64)), false),
    ("positional embedding and layer norm, training", (E, t, P, w) -> sum(abs2, layernorm32(E[:, t] .+ P, w)),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64, 80; scale = 0.2), data(64; scale = 0.1)), true),
]

@testset "folds over part of a collapsed axis on cuTile" begin
    @testset "$name" for (name, f, args, training) in PARTIAL_COLLAPSE
        args = map(single, args)
        p = training ? differentiate(trace(f, args...)) : trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
        @test !any(k -> k isa C.FloorKernels, ex.kernels)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-4)
    end
end

# A gather's adjoint scatters on the device, a sum by key in the kernel that
# computes its elements, on cuTile and on the floor alike: each element adds
# atomically, or on cuTile each box of a row's elements, so the sums at an
# index selected more than once come in no fixed order and differ from the
# interpreter's by rounding, which is the order every kernel is held to. With
# rows narrower than a tile and fewer tokens than one, repeated indices, two
# index arrays, and ranges and integers beside an index array.
const SCATTERS = [
    ("embedding", (E, t, g) -> sum(E[:, t] .* g), (data(64, 109), Int32.(mod1.(37 .* (1:1024) .^ 2, 109)), data(64, 1024; scale = 0.37))),
    ("embedding of short rows", (E, t, g) -> sum(E[:, t] .* g), (data(4, 5), Int32[2, 5, 1, 2, 3, 5, 5], data(4, 7))),
    ("repeated indices in two arrays", (x, i, j) -> sum(abs2, x[i, j]), (data(8, 6), [3, 1, 3, 8, 3], Int32[6, 6, 1])),
    ("a strided range and an integer beside index arrays", x -> sum(abs2, x[2:2:8, [1, 4, 4]]) + sum(abs2, x[3, [2, 2]]), (data(8, 6),)),
    ("a reversed range", x -> sum(abs2, x[8:-3:1, [2, 2, 5]]), (data(8, 6),)),
]

@testset "scatters on the device" begin
    @testset "$name, $where" for (name, f, args) in SCATTERS, (where, target) in (("cuTile", T.CUDATarget(cuTile)), ("the floor", T.CUDATarget(:full, true, ())))
        args = map(single, args)
        p = differentiate(trace(f, args...))
        ex = prepare(p, map(CuArray, args)...; target)
        scatters = [k for (k, g) in enumerate(ex.plan.groups) if any(r -> ex.plan.graph.values[r].definition isa C.Keyed, g.reductions)]
        @test !isempty(scatters) && all(k -> (ex.kernels[k] isa C.FloorKernels) == (where == "the floor"), scatters)
        # An embedding's rows add on cuTile as boxes, a TMA reduction each.
        if where == "cuTile" && name == "embedding"
            ext = Base.get_extension(Tempest, :cuTileExt)
            @test all(k -> any(t -> occursin("atomic_store_add", string(ext.source(t))), ex.kernels[k].kernels), scatters)
        end
        reference = C.leaves(interpret(p, args...))[2:2]
        @test agree(C.leaves(outputs(execute!(ex)))[2:2], reference; rtol = 1.0e-5)
        # A scatter runs on the device, so the program can be captured.
        capture!(ex)
        @test agree(C.leaves(outputs(replay!(ex)))[2:2], reference; rtol = 1.0e-5)
    end
end
