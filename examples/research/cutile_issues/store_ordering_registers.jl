# tileiras register allocation depends on whether a store is token-ordered
# after an unrelated load. The kernel loads a column of `x`, reduces it, and
# stores a scaled column into `y`. With separate array parameters, cuTile 1.0
# puts `x` and `y` in different alias classes, so the store's input token does
# not include the load's result token. tileiras then allocates 255 registers.
# A tuple parameter puts both arrays in one alias class; the only change in the
# Tile IR is a `join_tokens` of the load's token into the store's, and the same
# kernel uses about 25 registers and runs several times faster.
# Run: julia --project=examples examples/research/cutile_issues/store_ordering_registers.jl
using CUDACore
import cuTile as ct

@inline function normalize_column(y, x, rows::Int32)
    column = ct.bid(1)
    v = ct.load(x, (Int32(1), column), (1024, 1))
    s = sum(v .* v; dims = 1) ./ Float32(rows)
    ct.store(y, (Int32(1), column), v .* (1f0 ./ sqrt.(s .+ 1f-5)))
    return
end

separate(y, x, rows) = normalize_column(y, x, rows)
tupled(arrays, rows) = normalize_column(arrays..., rows)

x = CuArray(randn(Float32, 1024, 1024))
y = similar(x)
cx, cy = ct.cuTileconvert(x), ct.cuTileconvert(y)
rows = Int32(1024)
kernels = (
    separate = (CUDACore.@cuda(backend=ct, launch=false, separate(y, x, rows)), (cy, cx, rows)),
    tuple = (CUDACore.@cuda(backend=ct, launch=false, tupled((y, x), rows)), ((cy, cx), rows)),
)
expected = Array(x) ./ sqrt.(sum(abs2, Array(x); dims = 1) ./ 1024 .+ 1f-5)
for (name, (kernel, args)) in pairs(kernels)
    kernel(args...; blocks = 1024)
    CUDACore.synchronize()
    Array(y) ≈ expected || error("wrong result for $name")
    elapsed = CUDACore.@elapsed for _ in 1:50
        kernel(args...; blocks = 1024)
    end
    attributes = CUDACore.attributes(kernel.fun)
    println(
        rpad(name, 9),
        round(1e6 * elapsed / 50; digits = 1), " μs, ",
        attributes[CUDACore.FUNC_ATTRIBUTE_NUM_REGS], " registers",
    )
end
if get(ENV, "SHOW_TILE_IR", "false") == "true"
    for (f, T) in ((separate, Tuple{typeof(cy), typeof(cx), Int32}), (tupled, Tuple{typeof((cy, cx)), Int32}))
        ct.code_tiled(stdout, f, T)
    end
end
