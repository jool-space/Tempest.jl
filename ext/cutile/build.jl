# The launches of a schedule, and building them on device buffers.

coordinates(c::Context, t, kb) = code(c.s) == 1 ? (t, kb) : (kb, t)

function store_statistic!(k, st, c, r, value, kb, workspaces)
    spec = statistic_spec(c, r, statistic_buffer(c, r, workspaces))
    spans = statistic_spans(c)
    ia, ib = coordinates(c, :(Int32(1)), kb)
    push!(st, store_expr(argument!(k, spec), spans, tile(c)..., ia, ib, value))
end

function load_statistics!(k, st, c, rs, names, kb, workspaces)
    spans = statistic_spans(c)
    ia, ib = code(c.s) == 3 ? (:(Int32(1)), :(Int32(1))) : coordinates(c, :(Int32(1)), kb)
    for r in rs
        spec = statistic_spec(c, r, statistic_buffer(c, r, workspaces))
        name = gensym(:stat)
        push!(st, :($name = $(load_expr(argument!(k, spec), spans, tile(c)..., ia, ib))))
        names[Stat(r)] = name
    end
end

function emit_statistic_outputs!(k, st, c, names, kb, workspaces)
    ms = statistic_outputs(c)
    isempty(ms) && return
    ia, ib = code(c.s) == 3 ? (:(Int32(1)), :(Int32(1))) : coordinates(c, :(Int32(1)), kb)
    emit_tiles!(k, st, c, [c.g.terms[m] for m in ms], names, ia, ib)
    store_values!(k, st, c, ms, names, ia, ib)
end

function pointwise_kernel(c::Context)
    (ta, tb), (a, b) = tile(c), extent(c.s)
    k = TileKernel("pointwise", (cld(a, ta), cld(b, tb)), c.s.occupancy)
    push!(k.body, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
    names = emit_tiles!(k, k.body, c, [c.g.terms[m] for m in c.g.outputs], Dict{Term, Any}(), :ia, :ib)
    store_outputs!(k, k.body, c, c.g.outputs, names, :ia, :ib)
    [k]
end

# 1-based block id to 1-based tiles of two kept axes, grouping the first's tiles
# so that neighbouring blocks share operand tiles in L2.
function swizzle(m, n, tm, tn, group, bid)
    blocks_m = cld(m, tm)
    blocks_n = cld(n, tn)
    in_group = group * blocks_n
    b0 = bid - Int32(1)
    first_m = fld(b0, in_group) * group
    size_m = min(blocks_m - first_m, group)
    (first_m + rem(b0, size_m) + Int32(1), fld(rem(b0, in_group), size_m) + Int32(1))
end

# The block index of each kept tile axis, from the grid: the first two kept
# axes through the swizzle when the schedule asks for one, the rest one grid
# dimension each.
function block_indices!(st, c::Context)
    s = c.s
    kept = [i for i in eachindex(s.tiling) if !s.tiling[i].reduced]
    blocks(i) = cld(s.tiling[i].extent, s.tiling[i].tile)
    indices = Vector{Any}(fill(:(Int32(1)), length(s.tiling)))
    grid = Int[]
    rest = kept
    if s.swizzle > 0 && length(kept) >= 2
        i, j = kept[1], kept[2]
        ti, tj = s.tiling[i], s.tiling[j]
        push!(st, :((block_i, block_j) = $swizzle(Int32($(ti.extent)), Int32($(tj.extent)), Int32($(ti.tile)),
            Int32($(tj.tile)), Int32($(s.swizzle)), $(ct.bid)(1))))
        indices[i], indices[j] = :block_i, :block_j
        push!(grid, blocks(i) * blocks(j))
        rest = kept[3:end]
    end
    for i in rest
        name = Symbol(:block_, i)
        push!(st, :($name = $(ct.bid)($(length(grid) + 1))))
        indices[i] = name
        push!(grid, blocks(i))
    end
    length(grid) <= 3 || error("a grid has at most three dimensions")
    indices, Tuple(grid)
end

# Reductions along one swept tile axis, each tile of the kept axes in one block.
# Resident when one tile spans the swept axis, streamed otherwise. A sum of a
# bilinear product accumulates with `muladd`; every other fold elementwise.
function unsplit_kernel(c::Context, workspaces)
    k = TileKernel(string(c.s.sketch), (), c.s.occupancy)
    st = k.body
    indices, grid = block_indices!(st, c)
    k.grid = grid
    p = only(swept(c.s))
    at(t) = (ix = copy(indices); ix[p] = t; Tuple(ix))
    n = sweeps(c.s)
    resident = n == 1
    one32 = :(Int32(1))
    names = Dict{Term, Any}()
    statistics = Dict{Term, Any}()
    valid1 = resident ? valid_mask!(st, c, one32) : nothing
    for phase in reduction_phases(c.g)
        products = Dict(r => bilinear(c, r) for r in phase)
        accumulators = Dict{Int, Vector{Symbol}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            if products[r] !== nothing
                (a, b, i, j) = products[r]
                accumulators[r] = [Symbol(:acc_, r)]
                push!(st, :($(only(accumulators[r])) = fill(0.0f0, ($(tile(c)[i]), $(tile(c)[j])))))
            elseif !resident
                accumulators[r] = state_symbols(r, f)
                foreach((s, init) -> push!(st, :($s = $init)), accumulators[r], init_state(f, A, tile(c)))
            end
        end
        body = resident ? st : Any[]
        t = resident ? one32 : gensym(:t)
        valid = resident ? valid1 : valid_mask!(body, c, t)
        scope = resident ? names : merge(Dict{Term, Any}(), statistics)
        roots = [products[r] === nothing ? c.g.terms[r] : product_factor(c, products[r][q]) for r in phase for q in (1, 2)]
        emit_tiles!(k, body, c, unique(roots), scope, at(t)...)
        states = Dict{Int, Vector{Any}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            if products[r] !== nothing
                accumulate_product!(body, c, scope, products[r], only(accumulators[r]), t)
            elseif resident
                x = as_accumulator(st, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                states[r] = direct_state!(st, f, A, x, valid1, p)
            else
                x = as_accumulator(body, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                accumulate_elements!(body, f, A, accumulators[r], x, valid)
            end
        end
        resident || push!(st, :(for $t in Int32(1):Int32($n)
            $(body...)
        end))
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = products[r] !== nothing ? product_state(c, products[r], only(accumulators[r])) :
                resident ? states[r] : reduce_state!(st, f, A, accumulators[r], p)
            stat = observe!(st, f, A, c.graph.values[r].eltype, state)
            names[Stat(r)] = stat
            statistics[Stat(r)] = stat
        end
        stored = [r for r in phase if c.stored[r]]
        store_values!(k, st, c, stored, names, at(one32)...)
    end
    scope = resident ? names : merge(Dict{Term, Any}(), statistics)
    ms = statistic_outputs(c)
    emit_tiles!(k, st, c, [c.g.terms[m] for m in ms], scope, at(one32)...)
    store_values!(k, st, c, ms, scope, at(one32)...)
    outputs = full_outputs(c)
    if !isempty(outputs)
        if resident
            emit_tiles!(k, st, c, [c.g.terms[m] for m in outputs], names, at(one32)...)
            store_values!(k, st, c, outputs, names, at(one32)...)
        else
            t = gensym(:t)
            body = Any[]
            scope = merge(Dict{Term, Any}(), statistics)
            emit_tiles!(k, body, c, [c.g.terms[m] for m in outputs], scope, at(t)...)
            store_values!(k, body, c, outputs, scope, at(t)...)
            push!(st, :(for $t in Int32(1):Int32($n)
                $(body...)
            end))
        end
    end
    [k]
end

partial_key(r, i) = (:partials, r, i)

# Reductions whose reduced extent is shared among blocks: per phase, a launch
# of partial states and a launch that merges them; then one for the outputs.
function split_kernels(c::Context, workspaces)
    kernels = TileKernel[]
    full = code(c.s) == 3
    (ta, tb), (a, b) = tile(c), extent(c.s)
    blocks = full ? (cld(a, ta), cld(b, tb)) : (kept_tiles(c.s), c.s.splits)
    parts = full ? prod(blocks) : c.s.splits
    earlier = Int[]
    phases = reduction_phases(c.g)
    for (p, phase) in enumerate(phases)
        k = TileKernel("partial states", blocks, c.s.occupancy)
        st = k.body
        names = Dict{Term, Any}()
        if full
            push!(st, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
            load_statistics!(k, st, c, earlier, names, :ib, workspaces)
            valid = full_mask!(st, c, :ia, :ib)
            emit_tiles!(k, st, c, [c.g.terms[r] for r in phase], names, :ia, :ib)
        else
            push!(st, :(kb = $(ct.bid)(1)), :(part = $(ct.bid)(2)))
            load_statistics!(k, st, c, earlier, names, :kb, workspaces)
        end
        accumulators = Dict{Int, Vector{Symbol}}()
        direct = Dict{Int, Vector{Symbol}}()
        if full
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(st, names[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                direct[r] = direct_state!(st, f, A, x, valid, (1, 2))
            end
        else
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                accumulators[r] = state_symbols(r, f)
                foreach((s, init) -> push!(st, :($s = $init)), accumulators[r], init_state(f, A, tile(c)))
            end
            chunk = cld(sweeps(c.s), c.s.splits)
            t = gensym(:t)
            body = Any[]
            valid = valid_mask!(body, c, t)
            scope = copy(names)
            emit_tiles!(k, body, c, [c.g.terms[r] for r in phase], scope, coordinates(c, t, :kb)...)
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(body, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                accumulate_elements!(body, f, A, accumulators[r], x, valid)
            end
            push!(st, :(for $t in ((part - Int32(1)) * Int32($chunk) + Int32(1)):min(part * Int32($chunk), Int32($(sweeps(c.s))))
                $(body...)
            end))
        end
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = full ? direct[r] : reduce_state!(st, f, A, accumulators[r], reduced_dims(c))
            for (i, s) in enumerate(state)
                key = partial_key(r, i)
                if full
                    workspaces[key] = Workspace(key, A, parts)
                    push!(st, store_expr(argument!(k, ArgSpec(key, 0, (parts, 1), (1, 1))), (true, false), 1, 1,
                        :(ia + (ib - Int32(1)) * Int32($(blocks[1]))), :(Int32(1)), s))
                elseif code(c.s) == 1
                    workspaces[key] = Workspace(key, A, parts * b)
                    push!(st, store_expr(argument!(k, ArgSpec(key, 0, (parts, b), (1, parts))), (true, true), 1, tb, :part, :kb, s))
                else
                    workspaces[key] = Workspace(key, A, a * parts)
                    push!(st, store_expr(argument!(k, ArgSpec(key, 0, (a, parts), (1, a))), (true, true), ta, 1, :kb, :part, s))
                end
            end
        end
        push!(kernels, k)
        push!(kernels, merge_kernel(c, phase, earlier, parts, workspaces, p == length(phases)))
        append!(earlier, phase)
    end
    outputs = full_outputs(c)
    if !isempty(outputs)
        grid = full ? blocks : code(c.s) == 1 ? (sweeps(c.s), kept_tiles(c.s)) : (kept_tiles(c.s), sweeps(c.s))
        k = TileKernel("outputs", grid, nothing)
        st = k.body
        push!(st, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
        names = Dict{Term, Any}()
        load_statistics!(k, st, c, c.g.reductions, names, code(c.s) == 1 ? :ib : :ia, workspaces)
        emit_tiles!(k, st, c, [c.g.terms[m] for m in outputs], names, :ia, :ib)
        store_outputs!(k, st, c, outputs, names, :ia, :ib)
        push!(kernels, k)
    end
    kernels
end

# Merge the partial states of one phase into its statistics, per kept tile.
function merge_kernel(c::Context, phase, earlier, parts, workspaces, last)
    full = code(c.s) == 3
    (ta, tb), (a, b) = tile(c), extent(c.s)
    k = TileKernel("merge", full ? (1,) : (kept_tiles(c.s),), nothing)
    st = k.body
    push!(st, :(kb = $(ct.bid)(1)))
    width = full ? min(FULL_TILE_ELEMENTS, nextpow(2, parts)) : nextpow(2, parts)
    shape = full ? (width, 1) : code(c.s) == 1 ? (width, tb) : (ta, width)
    for r in phase
        f, A = fold_of(c, r), accumulator(c, r)
        acc = state_symbols(r, f)
        foreach((s, init) -> push!(st, :($s = $init)), acc, init_state(f, A, shape))
        t = gensym(:t)
        body = Any[]
        valid = gensym(:valid)
        if code(c.s) == 2
            push!(body, :($valid = reshape($(ct.arange)($width), (1, $width)) .<= Int32($parts) - ($t - Int32(1)) * Int32($width)))
        else
            push!(body, :($valid = reshape($(ct.arange)($width), ($width, 1)) .<= Int32($parts) - ($t - Int32(1)) * Int32($width)))
        end
        loaded = map(1:state_width(f)) do i
            key = partial_key(r, i)
            name = gensym(:part)
            spec, spans, index = if full
                ArgSpec(key, 0, (parts, 1), (1, 1)), (true, false), (t, :(Int32(1)))
            elseif code(c.s) == 1
                ArgSpec(key, 0, (parts, b), (1, parts)), (true, true), (t, :kb)
            else
                ArgSpec(key, 0, (a, parts), (1, a)), (true, true), (:kb, t)
            end
            push!(body, :($name = $(load_expr(argument!(k, spec), spans, shape[1], shape[2], index...))))
            name
        end
        accumulate_states!(body, f, A, acc, loaded, valid)
        push!(st, :(for $t in Int32(1):Int32($(cld(parts, width)))
            $(body...)
        end))
        state = reduce_state!(st, f, A, acc, full ? (1, 2) : code(c.s))
        stat = observe!(st, f, A, c.graph.values[r].eltype, state)
        spec = statistic_spec(c, r, statistic_buffer(c, r, workspaces))
        ia, ib = full ? (:(Int32(1)), :(Int32(1))) : coordinates(c, :(Int32(1)), :kb)
        push!(st, store_expr(argument!(k, spec), statistic_spans(c), ta, tb, ia, ib, stat))
    end
    if last
        names = Dict{Term, Any}()
        load_statistics!(k, st, c, [earlier; phase], names, :kb, workspaces)
        emit_statistic_outputs!(k, st, c, names, :kb, workspaces)
    end
    k
end

function schedule_kernels(c::Context, workspaces)
    c.s.sketch === :pointwise && return pointwise_kernel(c)
    c.s.sketch in (:resident, :streamed, :product) && return unsplit_kernel(c, workspaces)
    split_kernels(c, workspaces)
end

function tile_array(buffer::CuArray{T}, spec::ArgSpec) where {T}
    ptr = reinterpret(Ptr{T}, pointer(buffer)) + spec.offset * sizeof(T)
    ct.TileArray(ptr, Int32.(spec.sizes), Int32.(spec.strides))
end

struct BuiltTiles
    schedule::Any
    kernels::Vector{TileKernel}
end

function Base.show(io::IO, b::BuiltTiles)
    print(io, "cuTile ", b.schedule, ": ", join((string(k.name, " grid ", k.grid) for k in b.kernels), "; "))
end

function Tempest.build_claim(::Tempest.CuTileGenerator, claim::Claim, target, p::Plan, g::Group, buffers)
    c = Context(p.graph, p.stored, g, claim.schedule)
    workspaces = Dict{Any, Workspace}()
    launch_kernels(schedule_kernels(c, workspaces), workspaces, buffers, claim.schedule)
end

function launch_kernels(kernels, workspaces, buffers, schedule)
    scratch = Dict(key => CuArray{w.eltype}(undef, w.length) for (key, w) in workspaces)
    launches = map(kernels) do k
        f = Core.eval(Tempest.Generated, source(k))
        arrays = map(spec -> tile_array(spec.buffer isa Int ? buffers[spec.buffer] : scratch[spec.buffer], spec), k.args)
        occupancy = k.occupancy
        compiled = Base.invokelatest(f, arrays, occupancy) do f, arrays, occupancy
            occupancy === nothing ? CUDACore.@cuda(backend = ct, launch = false, f(arrays...)) :
            CUDACore.@cuda(backend = ct, launch = false, occupancy = occupancy, f(arrays...))
        end
        grid = k.grid
        () -> compiled(arrays...; blocks = grid)
    end
    keep = (scratch, buffers)
    launch = () -> (foreach(l -> l(), launches); keep; nothing)
    (launch, BuiltTiles(schedule, kernels))
end
