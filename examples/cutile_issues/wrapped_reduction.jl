# Standalone reproducer for the cuTile loop/argument issue encountered while
# developing reduction fusion. No Tempest dependency and no expected-failure
# assertion: rerun after compiler updates to revisit the streaming schedule.
using CUDACore
import cuTile as ct

load_column_tile(x, block, column, tile) =
    ct.load(x, (block, column), (tile, 1); padding_mode = ct.PaddingMode.Zero)
load_column_tile(x::Tuple, block, column, tile) = load_column_tile(only(x), block, column, tile)
load_column_tile(x::ct.Extruded, block, column, tile) =
    ct._eval_bc(x, (block, column), (tile, 1))

function two_pass_norm(out, x, rows::Int32, tile::Int)
    column = ct.bid(1)
    acc = zeros(Float32, (tile, 1))
    for block = Int32(1):cld(rows, Int32(tile))
        value = load_column_tile(x, block, column, tile)
        acc = acc .+ value .* value
    end
    total = sum(acc)
    for block = Int32(1):cld(rows, Int32(tile))
        value = load_column_tile(x, block, column, tile)
        ct.store(out, (block, column), value ./ total)
    end
    return
end

function main()
    CUDACore.functional() || error("this probe requires a functional CUDA device")
    # Full tiles isolate argument lowering from reduction padding.
    host = reshape(sin.(Float32.(1:(512 * 7))), 512, 7)
    x = CuArray(host)
    out = similar(x)
    expected = host ./ sum(abs2, host; dims = 1)
    println("cuTile version: ", Base.pkgversion(ct))
    for (name, input) in
        (("plain", x), ("tuple", (x,)), ("broadcast wrapper", ct.preprocess(x, Val(2))))
        try
            CUDACore.@cuda backend=ct blocks=7 two_pass_norm(
                out,
                input,
                Int32(512),
                ct.Constant(256),
            )
            @assert Array(out) ≈ expected
            println(name, ": passed")
        catch err
            println(name, ": ", first(split(sprint(showerror, err), '\n')))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
