# Which scalar functions the generator lowers over tiles: found by compiling a
# trial kernel that applies the function once, through the kernel cache, so
# that a function cuTile lowers compiles once per machine and loads after, and
# one it cannot fails once and fails from the cache after (`store_failure`).
# Asked as the generator claims (`expressible`): a group that applies a
# function cuTile has no lowering for, like Base's `exp10`, which branches on a
# comparison of its argument, is not claimed, rather than claimed and failing
# to build.

# What the generator lowers itself rather than as a scalar function over
# tiles: a fold's element of several components, a product's weight, and
# conversions.
own(f) = f === tuple || f isa Weight || f isa Cast

# The length of a trial's tiles.
const TRIAL = 16

# Answers by function, the kinds of its arguments and its result's type.
const LOWERED = Dict{Any, Bool}()
const LOWERED_LOCK = ReentrantLock()

Implement.lowers(::cuTileGenerator, ::CUDATarget, t, g) = lowered(t, g)

"""
Whether the generator lowers `t`, an `Apply` in `g`. True for what it lowers
itself, for a function of literals, which it applies on the host, for
arguments or results it has no tiles for, which `expressible` rejects, and
while a preparation compiles nothing (`compiling`).
"""
function lowered(t::Apply, g)
    (own(t.f) || all(a -> a isa Lit, t.args)) && return true
    kinds = map(a -> a isa Lit ? (:literal, placeholder(a.value)) : (:tile, term_eltype(a, g)), t.args)
    all(((kind, x),) -> kind === :literal || x <: TILE_TYPES, kinds) && t.eltype <: TILE_TYPES || return true
    Implement.compiling() || return true
    @lock LOWERED_LOCK get!(() -> trial(t.f, kinds, t.eltype), LOWERED, (t.f, kinds, t.eltype))
end

# A literal argument enters the trial as a two of its type, so that a
# function is tried once per type rather than per value; a function that
# lowers some literals and not others fails at build (see `build_group`).
placeholder(x) = x isa Number && !(x isa Bool) ? oftype(x, 2) : x

# Bools are stored as bytes (`kernel_eltype`); a trial's come from a
# comparison and leave converted.
trial_storage(T) = T === Bool ? Int32 : T

function trial(f, kinds, R)
    buffers = Any[CuArray{trial_storage(R)}(undef, TRIAL)]
    params, body, args = Symbol[], Expr[], Any[]
    for (i, (kind, x)) in enumerate(kinds)
        kind === :literal && (push!(args, x); continue)
        a, y = Symbol(:arg, i), Symbol(:t, i)
        push!(buffers, CuArray{trial_storage(x)}(undef, TRIAL))
        push!(params, a)
        load = :($(ct.load)($a; index = Int32(1), shape = ($TRIAL,)))
        push!(body, :($y = $(x === Bool ? :($load .!= Int32(0)) : load)))
        push!(args, y)
    end
    y = :($f.($(args...)))
    R === Bool && (y = :(convert($(ct.Tile){Int32}, $y)))
    source = :(function (out, $(params...))
        $(body...)
        $(ct.store)(out; index = Int32(1), tile = $y)
        return
    end)
    arrays = map(b -> ct.TileArray(reinterpret(Ptr{eltype(b)}, pointer(b)), (Int32(TRIAL),), (Int32(1),)), buffers)
    try
        GC.@preserve buffers tile_function(Tempest.Compiler.generated(source), arrays)
        true
    catch err
        err isa InterruptException && rethrow()
        @debug "cuTile does not lower $f over $(kinds)" exception = err
        false
    end
end
