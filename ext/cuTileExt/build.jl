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

# Reductions along one axis, each kept tile in one block.
function unsplit_kernel(c::Context, workspaces)
    k = TileKernel(string(c.s.sketch), (kept_tiles(c.s),), c.s.occupancy)
    st = k.body
    push!(st, :(kb = $(ct.bid)(1)))
    n = sweeps(c.s)
    resident = n == 1
    names = Dict{Term, Any}()
    statistics = Dict{Term, Any}()
    ia1, ib1 = coordinates(c, :(Int32(1)), :kb)
    valid1 = resident ? valid_mask!(st, c, :(Int32(1))) : nothing
    for phase in reduction_phases(c.g)
        if resident
            emit_tiles!(k, st, c, [c.g.terms[r] for r in phase], names, ia1, ib1)
            for r in phase
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(st, names[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                state = direct_state!(st, f, A, x, valid1, c.s.reduced)
                stat = observe!(st, f, A, c.graph.values[r].eltype, state)
                names[Stat(r)] = stat
                c.stored[r] && store_statistic!(k, st, c, r, stat, :kb, workspaces)
            end
            continue
        end
        accumulators = Dict{Int, Vector{Symbol}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            accumulators[r] = state_symbols(r, f)
            foreach((s, init) -> push!(st, :($s = $init)), accumulators[r], init_state(f, A, tile(c)))
        end
        body = resident ? st : Any[]
        t = resident ? :(Int32(1)) : gensym(:t)
        valid = resident ? valid1 : valid_mask!(body, c, t)
        scope = resident ? names : merge(Dict{Term, Any}(), statistics)
        emit_tiles!(k, body, c, [c.g.terms[r] for r in phase], scope, coordinates(c, t, :kb)...)
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            T = Tempest.term_eltype(c.g.terms[r], c.graph)
            x = as_accumulator(body, scope[c.g.terms[r]], A, T, tile(c))
            accumulate_elements!(body, f, A, accumulators[r], x, valid)
        end
        resident || push!(st, :(for $t in Int32(1):Int32($n)
            $(body...)
        end))
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = reduce_state!(st, f, A, accumulators[r], c.s.reduced)
            stat = observe!(st, f, A, c.graph.values[r].eltype, state)
            names[Stat(r)] = stat
            statistics[Stat(r)] = stat
            c.stored[r] && store_statistic!(k, st, c, r, stat, :kb, workspaces)
        end
    end
    emit_statistic_outputs!(k, st, c, resident ? names : merge(Dict{Term, Any}(), statistics), :kb, workspaces)
    outputs = full_outputs(c)
    if !isempty(outputs)
        if resident
            emit_tiles!(k, st, c, [c.g.terms[m] for m in outputs], names, ia1, ib1)
            store_outputs!(k, st, c, outputs, names, ia1, ib1)
        else
            t = gensym(:t)
            body = Any[]
            scope = merge(Dict{Term, Any}(), statistics)
            emit_tiles!(k, body, c, [c.g.terms[m] for m in outputs], scope, coordinates(c, t, :kb)...)
            store_outputs!(k, body, c, outputs, scope, coordinates(c, t, :kb)...)
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
    full = c.s.reduced == 3
    (ta, tb), (a, b) = tile(c), c.s.extent
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
                elseif c.s.reduced == 1
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
        grid = full ? blocks : c.s.reduced == 1 ? (sweeps(c.s), kept_tiles(c.s)) : (kept_tiles(c.s), sweeps(c.s))
        k = TileKernel("outputs", grid, nothing)
        st = k.body
        push!(st, :(ia = $(ct.bid)(1)), :(ib = $(ct.bid)(2)))
        names = Dict{Term, Any}()
        load_statistics!(k, st, c, c.g.reductions, names, c.s.reduced == 1 ? :ib : :ia, workspaces)
        emit_tiles!(k, st, c, [c.g.terms[m] for m in outputs], names, :ia, :ib)
        store_outputs!(k, st, c, outputs, names, :ia, :ib)
        push!(kernels, k)
    end
    kernels
end

# Merge the partial states of one phase into its statistics, per kept tile.
function merge_kernel(c::Context, phase, earlier, parts, workspaces, last)
    full = c.s.reduced == 3
    (ta, tb), (a, b) = tile(c), c.s.extent
    k = TileKernel("merge", full ? (1,) : (kept_tiles(c.s),), nothing)
    st = k.body
    push!(st, :(kb = $(ct.bid)(1)))
    width = full ? min(FULL_TILE_ELEMENTS, nextpow(2, parts)) : nextpow(2, parts)
    shape = full ? (width, 1) : c.s.reduced == 1 ? (width, tb) : (ta, width)
    for r in phase
        f, A = fold_of(c, r), accumulator(c, r)
        acc = state_symbols(r, f)
        foreach((s, init) -> push!(st, :($s = $init)), acc, init_state(f, A, shape))
        t = gensym(:t)
        body = Any[]
        valid = gensym(:valid)
        if c.s.reduced == 2
            push!(body, :($valid = reshape($(ct.arange)($width), (1, $width)) .<= Int32($parts) - ($t - Int32(1)) * Int32($width)))
        else
            push!(body, :($valid = reshape($(ct.arange)($width), ($width, 1)) .<= Int32($parts) - ($t - Int32(1)) * Int32($width)))
        end
        loaded = map(1:state_width(f)) do i
            key = partial_key(r, i)
            name = gensym(:part)
            spec, spans, index = if full
                ArgSpec(key, 0, (parts, 1), (1, 1)), (true, false), (t, :(Int32(1)))
            elseif c.s.reduced == 1
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
        state = reduce_state!(st, f, A, acc, full ? (1, 2) : c.s.reduced)
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
    c.s.sketch in (:resident, :streamed) && return unsplit_kernel(c, workspaces)
    split_kernels(c, workspaces)
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

function Tempest.build_claim(::Tempest.cuTileGenerator, claim::Claim, p::Plan, g::Group, buffers)
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
