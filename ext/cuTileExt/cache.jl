# Tile kernels through Tempest's kernel cache, and their launches.
#
# A kernel compiles once per content key (see `Tempest.content_key`): on a hit
# its CUBIN loads without inference or code generation. A launch passes the
# parameters cuTile's code generation gives the kernel, packed once at build
# time, since a prepared program's buffers never move.

# Every kernel's entry point, so a cached CUBIN needs no other record.
const ENTRY = "tempest_tile"

"""
    tile_function(f, args; occupancy) -> CuFunction

The compiled kernel of `f` for the converted arguments `args`, from the cache
or compiled through cuTile and kept.
"""
function tile_function(f, args; occupancy = nothing)
    tt = Tuple{map(Core.Typeof, args)...}
    groups = ct.alias_groups(args...)
    arch = ct.device_sm_arch()
    key = Tempest.content_key("cuTile kernel", f, tt, occupancy, groups, arch,
        ct.tileiras_version(), ct.bytecode_version(), ENTRY)
    found = Tempest.load_kernel(key)
    entry, cubin = found !== nothing ? found : Base.invokelatest(f, tt) do f, tt
        argtypes, constants = ct.unwrap_argtypes(f, tt)
        job = ct.invoke_frozen(ct.launch_job, f, argtypes, constants; sm_arch = arch, occupancy,
            name = ENTRY, alias_groups = groups)
        Tempest.store_kernel(key, ENTRY, ct.invoke_frozen(ct.compile_or_lookup, job).cubin)
    end
    CuFunction(CuModule(cubin), entry)
end

# The parameters of a value, in the order of cuTile's code generation: an
# array's pointer, sizes and strides; nothing for a ghost; a primitive as is;
# any other struct field by field.
function flatten!(out, x)
    T = typeof(x)
    if x isa ct.TileArray
        push!(out, x.ptr, x.sizes..., x.strides...)
    elseif ct.is_ghost_type(T)
    elseif isprimitivetype(T)
        push!(out, x)
    else
        foreach(i -> flatten!(out, getfield(x, i)), 1:fieldcount(T))
    end
    out
end

"A launch of a compiled tile kernel on fixed arguments."
struct TileLaunch
    fun::CuFunction
    boxes::Vector{Any}
    pointers::Vector{Ptr{Cvoid}}
    seed::Base.RefValue{UInt32}
    grid::NTuple{3, Int}
end

function TileLaunch(fun::CuFunction, f, args, grid)
    any(>(ct.MAX_GRID_DIM), grid) && error("grid $grid exceeds Tile IR's 24-bit limit")
    # The kernel function's captured values come first, the seed of cuTile's
    # kernel state last.
    values = flatten!(Any[], f)
    foreach(a -> flatten!(values, a), args)
    boxes = Any[Base.RefValue{typeof(v)}(v) for v in values]
    seed = Base.RefValue{UInt32}(0)
    push!(boxes, seed)
    pointers = Ptr{Cvoid}[pointer_from_objref(b) for b in boxes]
    TileLaunch(fun, boxes, pointers, seed, ntuple(i -> get(grid, i, 1), 3))
end

# One thread per block lets the driver take the thread count from the CUBIN.
function (l::TileLaunch)()
    l.seed[] = rand(UInt32)
    GC.@preserve l begin
        CUDACore.cuLaunchKernel(l.fun, l.grid..., 1, 1, 1, 0, CUDACore.stream(), l.pointers, C_NULL)
    end
    nothing
end

"Compile (or load) `f` for `args` and return its launch on `grid`."
tile_launch(f, args, grid; occupancy = nothing) = TileLaunch(tile_function(f, args; occupancy), f, args, grid)
