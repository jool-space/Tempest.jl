# The launches of a schedule, and building them on device buffers.

coordinates(c::Context, t, kb) = c.s.reduced == 1 ? (t, kb) : (kb, t)

function store_statistic!(k, st, c, r, value, kb, workspaces)
    spec = statistic_spec(c, r, statistic_buffer(c, r, workspaces))
    spans = statistic_spans(c)
    ia, ib = coordinates(c, :(Int32(1)), kb)
    push!(st, store_expr(argument!(k, spec), spans, tile(c)..., ia, ib, value))
end

function load_statistics!(k, st, c, rs, names, kb, workspaces)
    spans = statistic_spans(c)
    ia, ib = c.s.reduced == 3 ? (:(Int32(1)), :(Int32(1))) : coordinates(c, :(Int32(1)), kb)
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
    ia, ib = c.s.reduced == 3 ? (:(Int32(1)), :(Int32(1))) : coordinates(c, :(Int32(1)), kb)
    emit_tiles!(k, st, c, [c.g.terms[m] for m in ms], names, ia, ib)
    for m in ms
        v = output_view(c, m)
        push!(st, store_expr(argument!(k, value_spec(c, v)), v.spans, tile(c)..., ia, ib, names[c.g.terms[m]]))
    end
end

function pointwise_kernel(c::Context)
    (ta, tb), (a, b) = tile(c), c.s.extent
    k = TileKernel("pointwise", (cld(a, ta), cld(b, tb)), c.s.occupancy)
    push!(k.body, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
    names = emit_tiles!(k, k.body, c, [c.g.terms[m] for m in c.g.outputs], Dict{Term, Any}(), :ia, :ib)
    store_outputs!(k, k.body, c, c.g.outputs, names, :ia, :ib)
    [k]
end

# Partial states. A fold a block does not complete leaves its partial state per
# slot in a workspace laid out as (the extent of the collapsed axis the fold
# keeps, slots), or as a column of slots when it keeps none; `merge_kernel`
# completes it. A split fold's slots are the blocks sharing its kept tile;
# another fold's are the blocks or tiles of the schedule. (A column merges
# about 1 us faster than a row of slots on GB10: (1, 1024) tiles lost to
# (1024, 1) on every full reduction.)
partial_key(r, i) = (:partials, r, i)

function partial_spec!(c::Context, r, i, slots, workspaces)
    _, extent = fold_kept(c, r)
    key = partial_key(r, i)
    workspaces[key] = Workspace(key, accumulator(c, r), extent * slots)
    extent == 1 ? ArgSpec(key, 0, (slots, 1), (1, 1)) : ArgSpec(key, 0, (extent, slots), (1, extent))
end

function partial_shape(c::Context, r)
    q = first(fold_kept(c, r))
    q == 0 ? (1, 1) : q == 1 ? (tile(c)[1], 1) : (1, tile(c)[2])
end

# Store fold `r`'s partial state at tile `along` of the axis it keeps, in `slot`.
function store_partial!(k, st, c, r, state, along, slot, slots, workspaces)
    q = first(fold_kept(c, r))
    ta, tb = tile(c)
    for (i, s) in enumerate(state)
        sym = argument!(k, partial_spec!(c, r, i, slots, workspaces))
        if q == 0
            push!(st, store_expr(sym, (true, false), 1, 1, slot, :(Int32(1)), s))
        elseif q == 1
            push!(st, store_expr(sym, (true, true), ta, 1, along, slot, s))
        else
            push!(st, store_expr(sym, (true, true), tb, 1, along, slot, :(reshape($s, ($tb, 1)))))
        end
    end
end

# The partial state of each fold in `rs` over the tile at (ia, ib).
function tile_states!(k, st, c, rs, names, ia, ib)
    states = Dict{Int, Vector{Symbol}}()
    isempty(rs) && return states
    emit_tiles!(k, st, c, [c.g.terms[r] for r in rs], names, ia, ib)
    valid = full_mask!(st, c, ia, ib)
    for r in rs
        f, A = fold_of(c, r), accumulator(c, r)
        x = as_accumulator(st, names[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
        states[r] = direct_state!(st, f, A, x, valid, fold_dims(c, r))
    end
    states
end

# Complete folds from their partial states: one launch per collapsed axis kept
# and slot count, each block merging the slots of a tile of the kept extent.
function merge_kernel(c::Context, rs, slots, workspaces)
    _, extent = fold_kept(c, first(rs))
    column = extent == 1
    tm = min(nextpow(2, extent), 128)
    width = min(nextpow(2, slots), column ? FULL_TILE_ELEMENTS : max(1, 2048 ÷ tm))
    shape = column ? (width, 1) : (tm, width)
    k = TileKernel("merge", (cld(extent, tm),), nothing)
    st = k.body
    push!(st, :(i = $(ct.bid)(1)))
    for r in rs
        f, A = fold_of(c, r), accumulator(c, r)
        acc = state_symbols(r, f)
        foreach((s, init) -> push!(st, :($s = $init)), acc, init_state(f, A, shape))
        t = gensym(:t)
        loop = Any[]
        valid = gensym(:valid)
        lanes = column ? (width, 1) : (1, width)
        push!(loop, :($valid = reshape($(ct.arange)($width), $lanes) .<= Int32($slots) - ($t - Int32(1)) * Int32($width)))
        loaded = map(1:state_width(f)) do j
            name = gensym(:part)
            spec = partial_spec!(c, r, j, slots, workspaces)
            index = column ? (t, :(Int32(1))) : (:i, t)
            push!(loop, :($name = $(load_expr(argument!(k, spec), (true, true), shape..., index...))))
            name
        end
        accumulate_states!(loop, f, A, acc, loaded, valid)
        push!(st, :(for $t in Int32(1):Int32($(cld(slots, width)))
            $(loop...)
        end))
        state = reduce_state!(st, f, A, acc, column ? (1, 2) : 2)
        stat = observe!(st, f, A, c.graph.values[r].eltype, state)
        spec = ArgSpec(statistic_buffer(c, r, workspaces), 0, (extent, 1), (1, 1))
        push!(st, store_expr(argument!(k, spec), (true, false), tm, 1, :i, :(Int32(1)), stat))
    end
    k
end

# One merge per kept axis and slot count among `rs`.
function merge_kernels(c::Context, rs, slots, workspaces)
    groups = Dict{Tuple{Int, Int}, Vector{Int}}()
    for r in rs
        push!(get!(groups, (first(fold_kept(c, r)), slots[r]), Int[]), r)
    end
    [merge_kernel(c, groups[key], key[2], workspaces) for key in sort!(collect(keys(groups)))]
end

# Folds along one axis, each kept tile in one block, or several kept tiles one
# after another in each block when the schedule's chunk asks for it. Folds
# over the swept axis complete in the block; others leave partial states.
function unsplit_kernel(c::Context, workspaces, slots)
    chunk = c.s.chunk
    blocks = cld(kept_tiles(c.s), chunk)
    k = TileKernel(string(c.s.sketch), (blocks,), c.s.occupancy)
    st = k.body
    n = sweeps(c.s)
    resident = n == 1
    partial = partial_folds(c)
    # Partial states held in registers across the block's tiles: all of them
    # when resident, those reducing both axes when streamed. A streamed fold
    # keeping the swept axis stores each sweep's partial state instead.
    held(r) = resident || first(fold_kept(c, r)) == 0
    held_states = Dict{Int, Vector{Symbol}}()
    for r in partial
        held(r) || continue
        f, A = fold_of(c, r), accumulator(c, r)
        held_states[r] = [Symbol(:partial_, r, :_, i) for i in 1:state_width(f)]
        foreach((s, init) -> push!(st, :($s = $init)), held_states[r], init_state(f, A, partial_shape(c, r)))
        slots[r] = blocks
    end
    valid1 = resident ? valid_mask!(st, c, :(Int32(1))) : nothing
    body = chunk == 1 ? st : Any[]
    push!(body, chunk == 1 ? :(kb = $(ct.bid)(1)) : :(kb = ($(ct.bid)(1) - Int32(1)) * Int32($chunk) + j))
    names = Dict{Term, Any}()
    statistics = Dict{Term, Any}()
    ia1, ib1 = coordinates(c, :(Int32(1)), :kb)
    for phase in reduction_phases(c)
        if resident
            emit_tiles!(k, body, c, [c.g.terms[r] for r in phase], names, ia1, ib1)
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(body, names[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                state = direct_state!(body, f, A, x, valid1, c.s.reduced)
                stat = observe!(body, f, A, c.graph.values[r].eltype, state)
                names[Stat(r)] = stat
                c.stored[r] && store_statistic!(k, body, c, r, stat, :kb, workspaces)
            end
            continue
        end
        accumulators = Dict{Int, Vector{Symbol}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            accumulators[r] = state_symbols(r, f)
            foreach((s, init) -> push!(body, :($s = $init)), accumulators[r], init_state(f, A, tile(c)))
        end
        loop = Any[]
        t = gensym(:t)
        valid = valid_mask!(loop, c, t)
        scope = merge(Dict{Term, Any}(), statistics)
        emit_tiles!(k, loop, c, [c.g.terms[r] for r in phase], scope, coordinates(c, t, :kb)...)
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            T = Tempest.term_eltype(c.g.terms[r], c.graph)
            x = as_accumulator(loop, scope[c.g.terms[r]], A, T, tile(c))
            accumulate_elements!(loop, f, A, accumulators[r], x, valid)
        end
        push!(body, :(for $t in Int32(1):Int32($n)
            $(loop...)
        end))
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = reduce_state!(body, f, A, accumulators[r], c.s.reduced)
            stat = observe!(body, f, A, c.graph.values[r].eltype, state)
            names[Stat(r)] = stat
            statistics[Stat(r)] = stat
            c.stored[r] && store_statistic!(k, body, c, r, stat, :kb, workspaces)
        end
    end
    emit_statistic_outputs!(k, body, c, resident ? names : merge(Dict{Term, Any}(), statistics), :kb, workspaces)
    outputs = full_outputs(c)
    if resident
        if !isempty(outputs)
            emit_tiles!(k, body, c, [c.g.terms[m] for m in outputs], names, ia1, ib1)
            store_outputs!(k, body, c, outputs, names, ia1, ib1)
        end
        for (r, state) in tile_states!(k, body, c, partial, names, ia1, ib1)
            combine_states!(body, fold_of(c, r), accumulator(c, r), held_states[r], state)
        end
    elseif !isempty(outputs) || !isempty(partial)
        t = gensym(:t)
        loop = Any[]
        scope = merge(Dict{Term, Any}(), statistics)
        ia, ib = coordinates(c, t, :kb)
        emit_tiles!(k, loop, c, [c.g.terms[m] for m in outputs], scope, ia, ib)
        store_outputs!(k, loop, c, outputs, scope, ia, ib)
        for (r, state) in tile_states!(k, loop, c, partial, scope, ia, ib)
            if held(r)
                combine_states!(loop, fold_of(c, r), accumulator(c, r), held_states[r], state)
            else
                slots[r] = kept_tiles(c.s)
                store_partial!(k, loop, c, r, state, t, :kb, slots[r], workspaces)
            end
        end
        push!(body, :(for $t in Int32(1):Int32($n)
            $(loop...)
        end))
    end
    chunk == 1 || push!(st, :(for j in Int32(1):Int32($chunk)
        $(body...)
    end))
    if any(held, partial)
        push!(st, :(slot = $(ct.bid)(1)))
        for r in partial
            held(r) && store_partial!(k, st, c, r, held_states[r], :(Int32(1)), :slot, slots[r], workspaces)
        end
    end
    [k]
end

# Folds whose swept extent is shared among blocks: per phase, a launch of
# partial states and a merge; then the outputs, whose blocks each cover one
# tile and leave partial states of the remaining folds.
function split_kernels(c::Context, workspaces, slots)
    kernels = TileKernel[]
    full = c.s.reduced == 3
    (ta, tb), (a, b) = tile(c), c.s.extent
    blocks = full ? (cld(a, ta), cld(b, tb)) : (kept_tiles(c.s), c.s.splits)
    parts = full ? prod(blocks) : c.s.splits
    earlier = Int[]
    for phase in reduction_phases(c)
        k = TileKernel("partial states", blocks, c.s.occupancy)
        st = k.body
        names = Dict{Term, Any}()
        states = Dict{Int, Vector{Symbol}}()
        if full
            push!(st, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
            load_statistics!(k, st, c, earlier, names, :ib, workspaces)
            states = tile_states!(k, st, c, phase, names, :ia, :ib)
        else
            push!(st, :(kb = $(ct.bid)(1)), :(part = $(ct.bid)(2)))
            load_statistics!(k, st, c, earlier, names, :kb, workspaces)
            accumulators = Dict{Int, Vector{Symbol}}()
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                accumulators[r] = state_symbols(r, f)
                foreach((s, init) -> push!(st, :($s = $init)), accumulators[r], init_state(f, A, tile(c)))
            end
            chunk = cld(sweeps(c.s), c.s.splits)
            t = gensym(:t)
            loop = Any[]
            valid = valid_mask!(loop, c, t)
            scope = copy(names)
            emit_tiles!(k, loop, c, [c.g.terms[r] for r in phase], scope, coordinates(c, t, :kb)...)
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(loop, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                accumulate_elements!(loop, f, A, accumulators[r], x, valid)
            end
            push!(st, :(for $t in ((part - Int32(1)) * Int32($chunk) + Int32(1)):min(part * Int32($chunk), Int32($(sweeps(c.s))))
                $(loop...)
            end))
            for r in phase
                states[r] = reduce_state!(st, fold_of(c, r), accumulator(c, r), accumulators[r], c.s.reduced)
            end
        end
        for r in phase
            slots[r] = parts
            slot = full ? :(ia + (ib - Int32(1)) * Int32($(blocks[1]))) : :part
            store_partial!(k, st, c, r, states[r], full ? :(Int32(1)) : :kb, slot, parts, workspaces)
        end
        push!(kernels, k)
        append!(kernels, merge_kernels(c, phase, slots, workspaces))
        append!(earlier, phase)
    end
    if !isempty(statistic_outputs(c))
        k = TileKernel("outputs at the folds' positions", full ? (1,) : (kept_tiles(c.s),), nothing)
        push!(k.body, :(kb = $(ct.bid)(1)))
        names = Dict{Term, Any}()
        load_statistics!(k, k.body, c, earlier, names, :kb, workspaces)
        emit_statistic_outputs!(k, k.body, c, names, :kb, workspaces)
        push!(kernels, k)
    end
    outputs = full_outputs(c)
    partial = partial_folds(c)
    if !isempty(outputs) || !isempty(partial)
        grid = full ? blocks : c.s.reduced == 1 ? (sweeps(c.s), kept_tiles(c.s)) : (kept_tiles(c.s), sweeps(c.s))
        k = TileKernel("outputs", grid, nothing)
        st = k.body
        push!(st, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
        names = Dict{Term, Any}()
        load_statistics!(k, st, c, earlier, names, c.s.reduced == 1 ? :ib : :ia, workspaces)
        emit_tiles!(k, st, c, [c.g.terms[m] for m in outputs], names, :ia, :ib)
        store_outputs!(k, st, c, outputs, names, :ia, :ib)
        for (r, state) in tile_states!(k, st, c, partial, names, :ia, :ib)
            q = first(fold_kept(c, r))
            if q == 0
                slots[r] = prod(grid)
                store_partial!(k, st, c, r, state, :(Int32(1)), :(ia + (ib - Int32(1)) * Int32($(grid[1]))), slots[r], workspaces)
            elseif q == 1
                slots[r] = grid[2]
                store_partial!(k, st, c, r, state, :ia, :ib, slots[r], workspaces)
            else
                slots[r] = grid[1]
                store_partial!(k, st, c, r, state, :ib, :ia, slots[r], workspaces)
            end
        end
        push!(kernels, k)
    end
    kernels
end

function schedule_kernels(c::Context, workspaces)
    c.s.sketch === :pointwise && return pointwise_kernel(c)
    slots = Dict{Int, Int}()
    kernels = c.s.sketch in (:resident, :streamed) ? unsplit_kernel(c, workspaces, slots) :
        split_kernels(c, workspaces, slots)
    [kernels; merge_kernels(c, partial_folds(c), slots, workspaces)]
end

function tile_array(buffer::CuArray{T}, spec::ArgSpec) where {T}
    ptr = reinterpret(Ptr{T}, pointer(buffer)) + spec.offset * sizeof(T)
    ct.TileArray(ptr, Int32.(spec.sizes), Int32.(spec.strides))
end

struct BuiltTiles
    schedule::TileSchedule
    kernels::Vector{TileKernel}
end

function Base.show(io::IO, b::BuiltTiles)
    print(io, "cuTile ", b.schedule, ": ", join((string(k.name, " grid ", k.grid) for k in b.kernels), "; "))
end

function Tempest.build_claim(::Tempest.CuTileGenerator, claim::Claim, p::Plan, g::Group, buffers)
    c = Context(p.graph, p.stored, g, claim.schedule, claim.axes)
    workspaces = Dict{Any, Workspace}()
    kernels = schedule_kernels(c, workspaces)
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
    (launch, BuiltTiles(claim.schedule, kernels))
end
