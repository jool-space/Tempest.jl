# Tile kernels through Tempest's kernel cache, and their launches.
#
# A kernel compiles once per content key (see `Tempest.Compiler.content_key`): on a hit
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
function tile_function(@nospecialize(f), @nospecialize(args); occupancy = nothing)
    tt = Tuple{map(Core.Typeof, args)...}
    groups = alias_groups(args)
    # Every kernel's function is a type of its own: the key is untyped, and
    # what follows takes the function and its types without specializing, so
    # that no Julia code compiles per kernel.
    key = Any[CUDACore.context(), f, tt, occupancy, groups]
    asked = Tempest.Compiler.session()
    fun = get(FUNCTIONS, key, nothing)
    fun === nothing || return fun
    get!(FUNCTIONS, key, compiled_function(f, tt, occupancy, groups), asked)
end

function compiled_function(@nospecialize(f), @nospecialize(tt), @nospecialize(occupancy), @nospecialize(groups))
    entry, cubin = compiled_binary(f, tt, occupancy, groups, ct.device_sm_arch())
    CuFunction(CuModule(cubin), entry)
end

# The CUBIN of `f` for `tt` on architecture `arch`, which needs no device. A
# kernel is keyed without Tempest's own code and stored with what its
# compilation used (`Tempest.Compiler.dependencies`), so that it loads only
# while that holds, and an edit to Tempest compiles again only the kernels
# whose code it changed.
function compiled_binary(@nospecialize(f), @nospecialize(tt), @nospecialize(occupancy), @nospecialize(groups), arch)
    parts = Any["cuTile kernel", f, tt, occupancy, groups, arch, ct.tileiras_version(), ct.bytecode_version(), ENTRY]
    key = Tempest.Compiler.content_key(parts...; checked = true)
    Tempest.Compiler.cached_kernel(Compilation(parts, key), key)
end

# A kernel's compilation, as the cache runs it on a miss: one type for every
# kernel, where a closure over the kernel's function and argument types would
# be a type of each kernel's, compiled again for it.
struct Compilation
    parts::Vector{Any}
    key::String
end

function (c::Compilation)()
    _, f, tt, occupancy, groups, arch = c.parts
    Base.invokelatest(f, tt) do f, tt
        job = tile_job(f, tt, occupancy, groups, arch)
        cubin = try
            ct.invoke_frozen(ct.compile_or_lookup, job).cubin
        catch err
            err isa Union{ct.CodegenErrors, ct.IRError} && keep_failure(c.key, err, job)
            rethrow()
        end
        Tempest.Compiler.store_kernel(c.key, ENTRY, cubin; dependencies = job_dependencies(job))
    end
end

# A kernel cuTile cannot express fails the same way next time, while what its
# inference used holds; one whose inference failed is not kept.
function keep_failure(key, err, job)
    dependencies = try
        job_dependencies(job)
    catch
        nothing
    end
    Tempest.Compiler.store_failure(key, sprint(showerror, err); dependencies)
end

# cuTile's compilation of `f` for `tt`, which its code runs in cuTile's own world.
function tile_job(@nospecialize(f), @nospecialize(tt), @nospecialize(occupancy), @nospecialize(groups), arch)
    argtypes, constants = ct.unwrap_argtypes(f, tt)
    ct.invoke_frozen(ct.launch_job, f, argtypes, constants, (; sm_arch = arch, occupancy, name = ENTRY, alias_groups = groups))
end

# What a job's compilation used: the dependencies of the kernel's generic
# inference, whose edges cover any constant-seeded one's, its calls dispatched
# in cuTile's method table over Julia's.
function job_dependencies(job)
    cache = ct.inference_cache(job)
    code = ct.invoke_frozen(ct.infer, cache, job.source)
    Tempest.Compiler.dependencies(code, mi -> get(cache, mi, nothing), ct.cuTileMethodTable)
end

# A kernel's function, loaded once per context for the kernel, its argument
# types, its occupancy and the aliasing among its arguments: a body launches
# the same kernel at every execution, and identical groups share theirs. The
# kernel is a function Tempest generated, defined once and never again, from
# a key that names the code it calls (`EMITTED`'s) or for one trial: the
# function stands for that code.
const FUNCTIONS = Tempest.Compiler.SessionTable{Vector{Any}, CuFunction}()

# The groups of arrays among `args` whose memory overlaps, as cuTile's
# `alias_groups` finds them, without compiling a generated function for each
# kernel's argument types.
function alias_groups(@nospecialize(args))
    leaves = ct.array_leaves(map(Core.Typeof, args))
    ranges = map(((n, path),) -> memory_range(field_at(args[n - 1], path)), leaves)
    labels = collect(eachindex(ranges))
    changed = true
    while changed
        changed = false
        for i in eachindex(ranges), j in (i + 1):lastindex(ranges)
            if labels[i] != labels[j] && ct.overlaps(ranges[i], ranges[j])
                labels[i] = labels[j] = min(labels[i], labels[j])
                changed = true
            end
        end
    end
    ct.labels_to_groups(Tuple(map(l -> count(==(l), labels) > 1 ? l : 0, labels)), leaves)
end

# The field of `x` that `path` names, field within field. A kernel's
# arguments are cuTile's arrays, whose types hold what each array's spec
# says of it (its alignment, which strides divide what): they are read by
# their fields, with code compiled once whatever the spec, where Base's
# `foldl`, `getproperty` and the collection of them into a vector compiled
# for each spec a program's kernels take.
function field_at(@nospecialize(x), path)
    for name in path
        x = getfield(x, name)
    end
    x
end

# The bytes an array can reach, `[lo, hi)`; empty for an empty array.
function memory_range(@nospecialize(a))
    ptr, sizes, strides = getfield(a, :ptr), getfield(a, :sizes), getfield(a, :strides)
    bytes = sizeof(eltype(typeof(ptr)))
    lo = hi = reinterpret(Int, ptr)
    for d in 1:nfields(sizes)
        n = Int(getfield(sizes, d))
        n == 0 && return (0, 0)
        extent = (n - 1) * Int(getfield(strides, d)) * bytes
        extent < 0 ? (lo += extent) : (hi += extent)
    end
    (lo, hi + bytes)
end

# The parameters of a value, in the order of cuTile's code generation: an
# array's pointer, sizes and strides; nothing for a ghost; a primitive as is;
# any other struct field by field.
function flatten!(out, @nospecialize(x))
    T = typeof(x)
    if x isa ct.TileArray
        push!(out, getfield(x, :ptr))
        sizes, strides = getfield(x, :sizes), getfield(x, :strides)
        for d in 1:nfields(sizes)
            push!(out, getfield(sizes, d))
        end
        for d in 1:nfields(strides)
            push!(out, getfield(strides, d))
        end
    elseif ct.is_ghost_type(T)
    elseif isprimitivetype(T)
        push!(out, x)
    else
        for i in 1:fieldcount(T)
            flatten!(out, getfield(x, i))
        end
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

# A grid is a tuple of a kernel's rank, read by its elements, so that a launch
# is made by code compiled once whatever the rank.
function TileLaunch(fun::CuFunction, @nospecialize(f), @nospecialize(args), @nospecialize(grid::Tuple))
    sides = Int[getfield(grid, k) for k in 1:nfields(grid)]
    any(>(ct.MAX_GRID_DIM), sides) && error("grid $grid exceeds Tile IR's 24-bit limit")
    any(>(GRID_SIDE), sides[2:end]) && error("grid $grid exceeds the driver's $GRID_SIDE blocks along the second and third axes")
    # The kernel function's captured values come first, the seed of cuTile's
    # kernel state last. The seed is drawn once, not at each launch as
    # cuTile's own launches draw it: Tempest's kernels draw none of cuTile's
    # random numbers, their randomness being the program's, from its seeds,
    # and a launch whose seed differed each time would differ in each graph
    # recorded of it, which `retarget!` reads as a change it cannot make.
    values = flatten!(Any[], f)
    for a in args
        flatten!(values, a)
    end
    boxes = Any[Base.RefValue{typeof(v)}(v) for v in values]
    seed = Base.RefValue{UInt32}(rand(CUDACore.launch_rng(), UInt32))
    push!(boxes, seed)
    pointers = Ptr{Cvoid}[pointer_from_objref(b) for b in boxes]
    TileLaunch(fun, boxes, pointers, seed, (get(sides, 1, 1), get(sides, 2, 1), get(sides, 3, 1)))
end

# One thread per block lets the driver take the thread count from the CUBIN.
function (l::TileLaunch)()
    GC.@preserve l begin
        CUDACore.cuLaunchKernel(l.fun, l.grid..., 1, 1, 1, 0, CUDACore.stream(), l.pointers, C_NULL)
    end
    nothing
end

"Compile (or load) `f` for `args` and return its launch on `grid`."
tile_launch(@nospecialize(f), @nospecialize(args), @nospecialize(grid); occupancy = nothing) = TileLaunch(tile_function(f, args; occupancy), f, args, grid)
