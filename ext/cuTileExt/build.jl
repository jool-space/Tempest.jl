# The launches of a schedule, and building them on device buffers.

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
# dimension each, those beyond the grid's `limit` of dimensions sharing its
# last. A kept axis whose blocks cover several tiles (its `chunk`) indexes them
# in a loop: `chunks` lists (axis, loop variable, chunk), and `bases` the block
# index of each kept axis before chunking.
function block_indices!(st, c::Context; limit = 3)
    s = c.s
    kept = [i for i in eachindex(s.tiling) if !s.tiling[i].reduced]
    blocks(i) = cld(cld(s.tiling[i].extent, s.tiling[i].tile), s.tiling[i].chunk)
    indices = Vector{Any}(fill(:(Int32(1)), length(s.tiling)))
    grid = Int[]
    chunks = Tuple{Int, Symbol, Int}[]
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
    own = length(grid) + length(rest) <= limit ? length(rest) : max(0, limit - length(grid) - 1)
    for i in rest[1:own]
        name = Symbol(:block_, i)
        push!(st, :($name = $(ct.bid)($(length(grid) + 1))))
        indices[i] = name
        push!(grid, blocks(i))
    end
    shared = rest[(own + 1):end]
    if !isempty(shared)
        push!(st, :(remaining = $(ct.bid)($(length(grid) + 1)) - Int32(1)))
        for i in shared
            name = Symbol(:block_, i)
            push!(st, :($name = rem(remaining, Int32($(blocks(i)))) + Int32(1)), :(remaining = div(remaining, Int32($(blocks(i))))))
            indices[i] = name
        end
        push!(grid, prod(blocks, shared))
    end
    bases = copy(indices)
    for i in kept
        chunk = s.tiling[i].chunk
        chunk > 1 || continue
        j = Symbol(:within_, i)
        push!(chunks, (i, j, chunk))
        indices[i] = :(($(bases[i]) - Int32(1)) * Int32($chunk) + $j)
    end
    indices, isempty(grid) ? (1,) : Tuple(grid), chunks, bases
end

block_counts(c::Context, axes) = [cld(cld(c.s.tiling[i].extent, c.s.tiling[i].tile), c.s.tiling[i].chunk) for i in axes]
tile_counts(c::Context, axes) = [cld(c.s.tiling[i].extent, c.s.tiling[i].tile) for i in axes]

# Every tile on its own block, from a linear grid: one index per tile axis.
function tile_indices!(st, c::Context)
    counts = tile_counts(c, eachindex(c.s.tiling))
    push!(st, :(rest = $(ct.bid)(1) - Int32(1)))
    indices = map(eachindex(counts)) do i
        name = Symbol(:tile_, i)
        push!(st, :($name = rem(rest, Int32($(counts[i]))) + Int32(1)))
        push!(st, :(rest = div(rest, Int32($(counts[i])))))
        name
    end
    indices, (prod(counts; init = 1),)
end

# Folds along one swept tile axis, each tile of the kept axes in one block (or
# several in turn). Resident when one tile spans the swept axis, streamed
# otherwise. A sum of a bilinear product accumulates with `muladd`; every other
# fold elementwise. Folds over other axes leave partial states: held across
# the kernel's loops when they reduce what the loops run over, stored per
# sweep otherwise.
function unsplit_kernel(c::Context, workspaces, slots)
    k = TileKernel(string(c.s.sketch), (), c.s.occupancy)
    st = k.body
    indices, grid, chunks, bases = block_indices!(st, c)
    k.grid = grid
    p = only(swept(c.s))
    at(t) = (ix = copy(indices); ix[p] = t; Tuple(ix))
    n = sweeps(c.s)
    resident = n == 1
    one32 = :(Int32(1))
    partial = partial_folds(c)
    held(r) = resident || p in fold_tiles(c, r)
    # Slots of a fold's partial states: the blocks along the kept axes it reduces.
    kept = [i for i in eachindex(c.s.tiling) if !c.s.tiling[i].reduced]
    slot_of(r) = slot_expr(bases, [i for i in kept if i in fold_tiles(c, r)], block_counts(c, [i for i in kept if i in fold_tiles(c, r)]))
    held_states = Dict{Int, Vector{Symbol}}()
    for r in partial
        held(r) || continue
        f, A = fold_of(c, r), accumulator(c, r)
        held_states[r] = [Symbol(:partial_, r, :_, i) for i in 1:state_width(f)]
        foreach((x, init) -> push!(st, :($x = $init)), held_states[r], init_state(f, A, fold_tile_shape(c, r)))
    end
    body = isempty(chunks) ? st : Any[]
    names = Dict{Term, Any}()
    statistics = Dict{Term, Any}()
    valid1 = resident ? valid_mask!(body, c, one32) : nothing
    for phase in reduction_phases(c)
        products = Dict(r => bilinear(c, r) for r in phase)
        accumulators = Dict{Int, Vector{Symbol}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            if products[r] !== nothing
                (a, b, i, j) = products[r]
                accumulators[r] = [Symbol(:acc_, r)]
                push!(body, :($(only(accumulators[r])) = fill(0.0f0, ($(tile(c)[i]), $(tile(c)[j])))))
            elseif !resident
                accumulators[r] = state_symbols(r, f)
                foreach((x, init) -> push!(body, :($x = $init)), accumulators[r], init_state(f, A, tile(c)))
            end
        end
        loop = resident ? body : Any[]
        t = resident ? one32 : gensym(:t)
        valid = resident ? valid1 : valid_mask!(loop, c, t)
        scope = resident ? names : merge(Dict{Term, Any}(), statistics)
        roots = [products[r] === nothing ? c.g.terms[r] : product_factor(c, products[r][q]) for r in phase for q in (1, 2)]
        online = Dict{Int, Any}()
        if !resident
            for r in phase
                o = online_softmax(c, r)
                o === nothing || (online[r] = o)
            end
            hoist_folds!(k, body, c, unique(roots), scope, at(one32), Set(o[3] for o in values(online)))
            for (r, o) in online
                online[r] = online_step!(k, body, loop, c, o, scope, valid, at(t), products[r], only(accumulators[r]))
            end
        end
        emit_tiles!(k, loop, c, unique(roots), scope, at(t)...)
        states = Dict{Int, Vector{Any}}()
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            if products[r] !== nothing
                accumulate_product!(loop, c, scope, products[r], only(accumulators[r]), t)
            elseif resident
                x = as_accumulator(loop, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                states[r] = direct_state!(loop, f, A, x, valid1, p)
            else
                x = as_accumulator(loop, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
                accumulate_elements!(loop, f, A, accumulators[r], x, valid)
            end
        end
        resident || push!(body, :(for $t in Int32(1):Int32($n)
            $(loop...)
        end))
        for (r, total) in online
            acc = only(accumulators[r])
            push!(body, :($acc = $acc ./ $(products[r] === nothing ? total : matrix_tile(c, total, products[r][3], products[r][4]))))
        end
        isempty(online) || (k.name *= ", online softmax")
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = products[r] !== nothing ? product_state(c, products[r], only(accumulators[r])) :
                resident ? states[r] : reduce_state!(body, f, A, accumulators[r], p)
            stat = observe!(body, f, A, c.graph.values[r].eltype, state)
            names[Stat(r)] = stat
            statistics[Stat(r)] = stat
        end
        stored = [r for r in phase if c.stored[r]]
        store_values!(k, body, c, stored, names, at(one32)...)
    end
    scope = resident ? names : merge(Dict{Term, Any}(), statistics)
    ms = statistic_outputs(c)
    emit_tiles!(k, body, c, [c.g.terms[m] for m in ms], scope, at(one32)...)
    store_values!(k, body, c, ms, scope, at(one32)...)
    outputs = full_outputs(c)
    if resident
        emit_tiles!(k, body, c, [c.g.terms[m] for m in outputs], names, at(one32)...)
        store_values!(k, body, c, outputs, names, at(one32)...)
        for (r, state) in tile_states!(k, body, c, partial, names, at(one32))
            combine_states!(body, fold_of(c, r), accumulator(c, r), held_states[r], state)
        end
    elseif !isempty(outputs) || !isempty(partial)
        t = gensym(:t)
        loop = Any[]
        scope = merge(Dict{Term, Any}(), statistics)
        emit_tiles!(k, loop, c, [c.g.terms[m] for m in outputs], scope, at(t)...)
        store_values!(k, loop, c, outputs, scope, at(t)...)
        for (r, state) in tile_states!(k, loop, c, partial, scope, at(t))
            if held(r)
                combine_states!(loop, fold_of(c, r), accumulator(c, r), held_states[r], state)
            else
                slot, slots[r] = slot_of(r)
                store_partial!(k, loop, c, r, state, at(t), slot, slots[r], workspaces)
            end
        end
        push!(body, :(for $t in Int32(1):Int32($n)
            $(loop...)
        end))
    end
    for (i, j, chunk) in reverse(chunks)
        body = Any[:(for $j in Int32(1):Int32($chunk)
            $(body...)
        end)]
    end
    isempty(chunks) || append!(st, body)
    for r in partial
        held(r) || continue
        slot, slots[r] = slot_of(r)
        store_partial!(k, st, c, r, held_states[r], bases, slot, slots[r], workspaces)
    end
    [k]
end

# Recomputed folds that do not move along the swept axis are computed once,
# before the sweep.
function hoist_folds!(k, st, c::Context, roots, scope, indices, exclude)
    axes = c.s.tiling[only(swept(c.s))].axes
    for t in Tempest.subterms(roots; known = t -> haskey(scope, t))
        t isa Tempest.Fold && !(t in exclude) || continue
        any(q -> Tempest.mentions(t, q), axes) && continue
        emit_tiles!(k, st, c, [t], scope, indices...)
    end
end

# The online softmax: a sum whose term weighs by `exp(y - lse)`, where `lse` is
# a recomputed logsumexp of `y` over the swept axis, accumulates in the sweep
# that computes `lse`: each step raises the running maximum, rescales the sum
# and the running sum of exponentials by how much it rose, and weighs by
# `exp(y - maximum)`; the sum is divided by the sum of exponentials at the
# end. Returns the weight, `y` and `lse`, or nothing.
function online_softmax(c::Context, r)
    c.s.online && fold_of(c, r) isa Tempest.SumFold || return nothing
    axes = c.s.tiling[only(swept(c.s))].axes
    length(axes) == 1 || return nothing
    K = only(axes)
    roots = (c.g.terms[r],)
    for w in Tempest.subterms(roots)
        w isa Apply && w.f === exp && length(w.args) == 1 || continue
        d = only(w.args)
        d isa Apply && d.f === (-) && length(d.args) == 2 && d.args[2] isa Tempest.Fold || continue
        y, F = d.args
        def = c.graph.values[F.value].definition
        def.fold isa Tempest.LogSumExpFold && F.extents == (c.g.domain[K],) || continue
        Tempest.mentions(F, K) && continue
        index = ntuple(q -> q in def.reduced ? Tempest.var(K) : F.index[q], length(def.iteration))
        Tempest.operand_term(c.graph, c.stored, c.g, def.operand, index, Tempest.Domain(c.g.domain)) == y || continue
        # Nothing else reads the logsumexp, whose value is only known at the end.
        all(s -> !(s isa Apply) || s == d || !(F in s.args), Tempest.subterms(roots)) || continue
        return (w, y, F)
    end
    nothing
end

# One step of the online softmax for weight `w`: the running maximum and sum
# start before the sweep, `w` becomes `exp(y - maximum)`, and the accumulator
# of the sum it weighs is rescaled. Returns the running sum.
function online_step!(k, before, loop, c::Context, (w, y, F), scope, valid, indices, product, acc)
    p = only(swept(c.s))
    A = Tempest.fold_types(Tempest.LogSumExpFold(), Tempest.term_eltype(y, c.graph)).accumulator
    shape = Base.setindex(spanned_shape(c, y), 1, p)
    m, l = gensym(:max), gensym(:sum)
    push!(before, :($m = fill($(A(-Inf)), $shape)), :($l = fill($(zero(A)), $shape)))
    emit_tiles!(k, loop, c, [y], scope, indices...)
    x = as_accumulator(loop, scope[y], A, Tempest.term_eltype(y, c.graph), spanned_shape(c, y))
    top, sh, rise, weight = gensym(:top), gensym(:shift), gensym(:rise), gensym(:weight)
    push!(loop, :($top = max.($m, maximum(ifelse.($valid, $x, $(A(-Inf))); dims = $p))))
    push!(loop, :($sh = $SHIFT.($top)))
    push!(loop, :($rise = exp.($m .- $sh)))
    push!(loop, :($weight = exp.($x .- $sh)))
    push!(loop, :($l = $l .* $rise .+ sum(ifelse.($valid, $weight, $(zero(A))); dims = $p)))
    push!(loop, :($m = $top))
    push!(loop, :($acc = $acc .* $(product === nothing ? rise : matrix_tile(c, rise, product[3], product[4]))))
    scope[w] = weight
    l
end

# Folds whose swept axis is shared among blocks: per phase, a launch of
# partial states (each block summing a chunk of the swept axis; a product's
# sums too, which is split-K) and a merge. A schedule whose folds reduce every
# tile axis has each tile on its own block instead. Then the outputs, and the
# partial states of folds over other axes, one tile per block.
function split_kernels(c::Context, workspaces, slots)
    kernels = TileKernel[]
    full = length(swept(c.s)) == length(c.s.tiling)
    earlier = Int[]
    for phase in reduction_phases(c)
        if full
            k = TileKernel("partial states", (), c.s.occupancy)
            indices, k.grid = tile_indices!(k.body, c)
            names = Dict{Term, Any}()
            load_folds!(k, k.body, c, earlier, names, indices, workspaces)
            for (r, state) in tile_states!(k, k.body, c, phase, names, indices)
                slot, slots[r] = slot_expr(indices, eachindex(c.s.tiling), tile_counts(c, eachindex(c.s.tiling)))
                store_partial!(k, k.body, c, r, state, indices, slot, slots[r], workspaces)
            end
            push!(kernels, k)
        else
            push!(kernels, split_phase_kernel(c, phase, earlier, workspaces, slots))
        end
        append!(kernels, merge_kernels(c, phase, slots, workspaces))
        append!(earlier, phase)
    end
    ms = statistic_outputs(c)
    if !isempty(ms)
        k = TileKernel("outputs at the folds' positions", (), nothing)
        indices, k.grid, _, _ = block_indices!(k.body, c)
        names = Dict{Term, Any}()
        load_folds!(k, k.body, c, earlier, names, indices, workspaces)
        emit_tiles!(k, k.body, c, [c.g.terms[m] for m in ms], names, indices...)
        store_values!(k, k.body, c, ms, names, indices...)
        push!(kernels, k)
    end
    outputs = full_outputs(c)
    partial = partial_folds(c)
    if !isempty(outputs) || !isempty(partial)
        k = TileKernel("outputs", (), nothing)
        indices, k.grid = tile_indices!(k.body, c)
        names = Dict{Term, Any}()
        load_folds!(k, k.body, c, earlier, names, indices, workspaces)
        emit_tiles!(k, k.body, c, [c.g.terms[m] for m in outputs], names, indices...)
        store_values!(k, k.body, c, outputs, names, indices...)
        for (r, state) in tile_states!(k, k.body, c, partial, names, indices)
            axes = fold_tiles(c, r)
            slot, slots[r] = slot_expr(indices, axes, tile_counts(c, axes))
            store_partial!(k, k.body, c, r, state, indices, slot, slots[r], workspaces)
        end
        push!(kernels, k)
    end
    kernels
end

# One phase of a split schedule: blocks over the kept tiles and the splits,
# each summing its chunk of the swept axis into partial states.
function split_phase_kernel(c::Context, phase, earlier, workspaces, slots)
    k = TileKernel("partial states", (), c.s.occupancy)
    st = k.body
    indices, grid, _, _ = block_indices!(st, c; limit = 2)
    push!(st, :(part = $(ct.bid)($(length(grid) + 1))))
    k.grid = (grid..., c.s.splits)
    p = only(swept(c.s))
    at(t) = (ix = copy(indices); ix[p] = t; Tuple(ix))
    names = Dict{Term, Any}()
    load_folds!(k, st, c, earlier, names, at(:(Int32(1))), workspaces)
    products = Dict(r => bilinear(c, r) for r in phase)
    accumulators = Dict{Int, Vector{Symbol}}()
    for r in phase
        f, A = fold_of(c, r), accumulator(c, r)
        if products[r] !== nothing
            (a, b, i, j) = products[r]
            accumulators[r] = [Symbol(:acc_, r)]
            push!(st, :($(only(accumulators[r])) = fill(0.0f0, ($(tile(c)[i]), $(tile(c)[j])))))
        else
            accumulators[r] = state_symbols(r, f)
            foreach((x, init) -> push!(st, :($x = $init)), accumulators[r], init_state(f, A, tile(c)))
        end
    end
    chunk = cld(sweeps(c.s), c.s.splits)
    t = gensym(:t)
    loop = Any[]
    valid = valid_mask!(loop, c, t)
    scope = copy(names)
    roots = [products[r] === nothing ? c.g.terms[r] : product_factor(c, products[r][q]) for r in phase for q in (1, 2)]
    emit_tiles!(k, loop, c, unique(roots), scope, at(t)...)
    for r in phase
        f, A = fold_of(c, r), accumulator(c, r)
        if products[r] !== nothing
            accumulate_product!(loop, c, scope, products[r], only(accumulators[r]), t)
        else
            x = as_accumulator(loop, scope[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
            accumulate_elements!(loop, f, A, accumulators[r], x, valid)
        end
    end
    push!(st, :(for $t in ((part - Int32(1)) * Int32($chunk) + Int32(1)):min(part * Int32($chunk), Int32($(sweeps(c.s))))
        $(loop...)
    end))
    for r in phase
        f, A = fold_of(c, r), accumulator(c, r)
        state = products[r] !== nothing ? product_state(c, products[r], only(accumulators[r])) :
            reduce_state!(st, f, A, accumulators[r], p)
        slots[r] = c.s.splits
        store_partial!(k, st, c, r, state, at(:(Int32(1))), :part, c.s.splits, workspaces)
    end
    k
end

# Complete folds from their partial states: each block merges every slot of a
# tile of the kept extents, all slots in one step where they fit.
function merge_kernel(c::Context, rs, slots, workspaces)
    kept = kept_tiles_of(c, first(rs))
    extents = [c.s.tiling[q].extent for q in kept]
    width = min(nextpow(2, slots), FULL_TILE_ELEMENTS)
    budget = max(1, 2048 ÷ width)
    tiles_m = map(extents) do e
        m = min(nextpow(2, e), budget)
        budget = max(1, budget ÷ m)
        m
    end
    counts = cld.(extents, tiles_m)
    k = TileKernel("merge", (prod(counts; init = 1),), nothing)
    st = k.body
    push!(st, :(rest = $(ct.bid)(1) - Int32(1)))
    idx = map(eachindex(counts)) do q
        name = Symbol(:merge_, q)
        push!(st, :($name = rem(rest, Int32($(counts[q]))) + Int32(1)), :(rest = div(rest, Int32($(counts[q])))))
        name
    end
    shape = (tiles_m..., width)
    lanes = (map(_ -> 1, tiles_m)..., width)
    for r in rs
        f, A = fold_of(c, r), accumulator(c, r)
        acc = state_symbols(r, f)
        foreach((x, init) -> push!(st, :($x = $init)), acc, init_state(f, A, shape))
        t = gensym(:t)
        loop = Any[]
        valid = gensym(:valid)
        push!(loop, :($valid = reshape($(ct.arange)($width), $lanes) .<= Int32($slots) - ($t - Int32(1)) * Int32($width)))
        loaded = map(1:state_width(f)) do j
            name = gensym(:part)
            sym = argument!(k, partial_spec!(c, r, j, slots, workspaces))
            push!(loop, :($name = $(ct.load)($sym; index = ($(idx...), $t), shape = $shape, padding_mode = $(ct.PaddingMode.Zero))))
            name
        end
        accumulate_states!(loop, f, A, acc, loaded, valid)
        push!(st, :(for $t in Int32(1):Int32($(cld(slots, width)))
            $(loop...)
        end))
        state = reduce_state!(st, f, A, acc, length(shape))
        stat = observe!(st, f, A, c.graph.values[r].eltype, state)
        v, spec = fold_spec!(c, r, workspaces)
        full_tiles = ntuple(i -> i in kept ? tiles_m[findfirst(==(i), kept)] : 1, length(c.s.tiling))
        full_index = ntuple(i -> i in kept ? idx[findfirst(==(i), kept)] : :(Int32(1)), length(c.s.tiling))
        push!(st, store_nd(argument!(k, spec), v, full_tiles, full_index, :(reshape($stat, $full_tiles))))
    end
    k
end

# One merge per set of kept tile axes and slot count among `rs`.
function merge_kernels(c::Context, rs, slots, workspaces)
    groups = Dict{Tuple{Vector{Int}, Int}, Vector{Int}}()
    for r in rs
        push!(get!(groups, (kept_tiles_of(c, r), slots[r]), Int[]), r)
    end
    [merge_kernel(c, groups[key], key[2], workspaces) for key in sort!(collect(keys(groups)))]
end

function schedule_kernels(c::Context, workspaces)
    c.s.sketch === :pointwise && return pointwise_kernel(c)
    slots = Dict{Int, Int}()
    kernels = c.s.sketch in (:resident, :streamed) || (c.s.sketch === :product && c.s.splits == 1) ?
        unsplit_kernel(c, workspaces, slots) : split_kernels(c, workspaces, slots)
    [kernels; merge_kernels(c, partial_folds(c), slots, workspaces)]
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

function Tempest.build_claim(::Tempest.cuTileGenerator, claim::Claim, target, p::Plan, g::Group, buffers)
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
