# The launches of a schedule, and building them on device buffers.

function pointwise_kernel(c::Context)
    k = TileKernel("pointwise", (), c.s.occupancy)
    indices, k.grid, _, _ = block_indices!(k.body, c)
    names = emit_tiles!(k, k.body, c, [c.g.terms[m] for m in c.g.outputs], Dict{Term, Any}(), indices...)
    store_outputs!(k, k.body, c, c.g.outputs, names, indices...)
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
    linear_indices!(st, counts, :tile_), (prod(counts; init = 1),)
end

# Indices of `counts` tiles from the linear block id, named `prefix` and the
# axis; an axis of one tile has index one.
function linear_indices!(st, counts, prefix)
    many = findall(>(1), counts)
    isempty(many) || push!(st, :(rest = $(ct.bid)(1) - Int32(1)))
    map(eachindex(counts)) do i
        counts[i] == 1 && return :(Int32(1))
        name = Symbol(prefix, i)
        push!(st, :($name = rem(rest, Int32($(counts[i]))) + Int32(1)))
        i == last(many) || push!(st, :(rest = div(rest, Int32($(counts[i])))))
        name
    end
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
    p = sweep_axis(c.s)
    at(t) = step_indices(c.s, indices, t)
    n = sweeps(c.s)
    resident = n == 1
    one32 = :(Int32(1))
    partial = partial_folds(c)
    held(r) = resident || all(q -> q in fold_tiles(c, r), swept(c.s))
    # Slots of a fold's partial states: the blocks along the kept axes it reduces.
    kept = [i for i in eachindex(c.s.tiling) if !c.s.tiling[i].reduced]
    slot_of(r) = slot_expr(bases, [i for i in kept if i in fold_tiles(c, r)], block_counts(c, [i for i in kept if i in fold_tiles(c, r)]))
    held_states = Dict{Int, Vector{Symbol}}()
    for r in partial
        held(r) || continue
        f, A = fold_of(c, r), accumulator(c, r)
        held_states[r] = [Symbol(:partial_, r, :_, i) for i in eachindex(fold_init(f, A))]
        init_state!(st, held_states[r], f, A, fold_tile_shape(c, r))
    end
    body = isempty(chunks) ? st : Any[]
    names = Dict{Term, Any}()
    statistics = Dict{Term, Any}()
    # Lanes past the swept extent are masked for folds summed elementwise; a
    # product's operands pad with zeros instead.
    elementwise(phase) = any(r -> bilinear(c, r) === nothing, phase)
    valid1 = resident && any(elementwise, reduction_phases(c)) ? valid_mask!(body, c, at(one32)[p]) : nothing
    for phase in reduction_phases(c)
        sums = Dict(r => fold_sums(c, r) for r in phase)
        # Folds of a resident tile that do not sum fold it directly.
        accumulators = Dict(r => fold_accumulators!(body, c, r, sums[r]) for r in phase if !resident || sums[r] !== nothing)
        loop = resident ? body : Any[]
        t = resident ? one32 : gensym(:t)
        valid = resident ? valid1 : elementwise(phase) ? valid_mask!(loop, c, at(t)[p]) : nothing
        scope = resident ? names : merge(Dict{Term, Any}(), statistics)
        roots = phase_roots(c, phase, sums)
        resident || hoist_folds!(k, body, c, roots, scope, at(one32))
        emit_tiles!(k, loop, c, roots, scope, at(t)...)
        states = Dict{Int, Vector{Any}}()
        for r in phase
            if haskey(accumulators, r)
                fold_step!(k, loop, c, r, sums[r], accumulators[r], scope, valid, at(t))
            else
                f, A = fold_of(c, r), accumulator(c, r)
                x = as_accumulator(loop, scope[c.g.terms[r]], A, term_eltype(c.g.terms[r], c.g), tile(c))
                states[r] = direct_state!(loop, f, A, x, valid1, p)
            end
        end
        if !resident
            # Steps a mask rules out for every fold of the phase are skipped.
            live = live_steps(c, phase, p, indices, n)
            lo, hi = live === nothing ? (:(Int32(1)), :(Int32($n))) : live
            live === nothing || (k.name *= ", masked steps skipped")
            push!(body, :(for $t in $lo:$hi
                $(loop...)
            end))
        end
        any(r -> fold_of(c, r) isa Relative{Exponential}, phase) && (k.name *= ", online softmax")
        for r in phase
            f, A = fold_of(c, r), accumulator(c, r)
            state = haskey(states, r) ? states[r] : fold_state!(body, c, r, sums[r], accumulators[r])
            stat = observe!(body, f, A, c.g.values[r].eltype, state)
            names[Stat(r)] = stat
            statistics[Stat(r)] = stat
        end
        stored = [r for r in phase if c.g.values[r].stored]
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
            combine!(body, fold_of(c, r), accumulator(c, r), held_states[r], state)
        end
    elseif !isempty(outputs) || !isempty(partial)
        t = gensym(:t)
        loop = Any[]
        scope = merge(Dict{Term, Any}(), statistics)
        emit_tiles!(k, loop, c, [c.g.terms[m] for m in outputs], scope, at(t)...)
        store_values!(k, loop, c, outputs, scope, at(t)...)
        for (r, state) in tile_states!(k, loop, c, partial, scope, at(t))
            if held(r)
                combine!(loop, fold_of(c, r), accumulator(c, r), held_states[r], state)
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
function hoist_folds!(k, st, c::Context, roots, scope, indices)
    axes = reduce(vcat, (c.s.tiling[q].axes for q in swept(c.s)))
    for t in subterms(roots; known = t -> haskey(scope, t))
        t isa Recomputed || continue
        any(q -> mentions(t, q), axes) && continue
        emit_tiles!(k, st, c, [t], scope, indices...)
    end
end

# Folds that accumulate as sums of terms along tile axis `p`: a sum of a
# bilinear product, and an exponential fold of `(y, g...)`, whose sums are of each
# `g` times the weight `exp(y - m)` against the running maximum `m` of `y`
# (`Weight` stands for it), in the type the fold accumulates. Each sum is a
# term and its product, or nothing when the term adds elementwise. Nothing for
# other folds, which accumulate elementwise in their state.
struct Weight end

# The parts of an exponential fold's element: `y`, and each `g`; one `g` of one for
# an element that is a number.
function shifted_parts(c::Context, body)
    body isa Apply && body.f === tuple && return (first(body.args), Base.tail(body.args))
    (body, (Lit(one(term_eltype(body, c.g))),))
end

# The type an exponential fold's maximum and sums accumulate in.
shifted_type(c::Context, f, body) = fieldtype(fold_types(f, term_eltype(body, c.g)).state, 1)

function sweep_sums(c::Context, f, body, p)
    if f isa Relative{Exponential}
        y, gs = shifted_parts(c, body)
        W = shifted_type(c, f, body)
        w = Apply(Weight(), (y,), W)
        return map(gs) do g
            g isa Lit && isone(g.value) && return (w, bilinear_over(c, SumFold(), w, p))
            term_eltype(g, c.g) === W || (g = Apply(Cast{W}(), (g,), W))
            term = Apply(*, (g, w), W)
            (term, bilinear_over(c, SumFold(), term, p))
        end
    end
    product = bilinear_over(c, f, body, p)
    product === nothing ? nothing : [(body, product)]
end

# The terms a step computes before its sums add: the products' factors but
# the weights, which the step computes, and an exponential fold's `y`; or the term
# of a fold that accumulates elementwise.
function step_roots(c::Context, f, body, sums)
    sums === nothing && return Term[body]
    factors = Term[product_factor(c, x) for (_, product) in sums if product !== nothing for x in product[1:2] if !(x isa Apply && x.f isa Weight)]
    f isa Relative{Exponential} ? [first(shifted_parts(c, body)); factors] : factors
end

# Accumulators of sums, named from `prefix`: an exponential fold's running maximum,
# over the lanes `y` spans but the swept ones, then each sum's, a product's
# matrix or the lanes its term spans.
function sum_accumulators!(st, c::Context, f, body, sums, p, prefix)
    starts = Any[product === nothing ? :(fill($(zero(term.eltype)), $(Base.setindex(spanned_shape(c, term), 1, p)))) :
        :(fill(0.0f0, ($(tile(c)[product[3]]), $(tile(c)[product[4]])))) for (term, product) in sums]
    if f isa Relative{Exponential}
        y = first(shifted_parts(c, body))
        pushfirst!(starts, :(fill($(shifted_type(c, f, body)(-Inf)), $(Base.setindex(spanned_shape(c, y), 1, p)))))
    end
    acc = [Symbol(prefix, :_, i) for i in eachindex(starts)]
    foreach((a, x) -> push!(st, :($a = $x)), acc, starts)
    acc
end

# One step of the sums. An exponential fold's running maximum rises to the step's
# first, and its sums scale by how much it rose; lanes past the swept extent
# weigh nothing. Then each sum adds its terms.
function sum_step!(k, st, c::Context, f, body, sums, acc, scope, valid, indices, p)
    rise = nothing
    if f isa Relative{Exponential}
        y = first(shifted_parts(c, body))
        T = shifted_type(c, f, body)
        x = as_accumulator(st, scope[y], T, term_eltype(y, c.g), spanned_shape(c, y))
        m = first(acc)
        masked, top, shift, rise, w = gensym(:y), gensym(:top), gensym(:shift), gensym(:rise), gensym(:weight)
        push!(st, :($masked = ifelse.($valid, $x, $(T(-Inf)))))
        push!(st, :($top = max.($m, maximum($masked; dims = $p))))
        push!(st, :($shift = $(lse_shift).($top)))
        push!(st, :($rise = exp.($m .- $shift)))
        push!(st, :($w = exp.($masked .- $shift)))
        push!(st, :($m = $top))
        scope[Apply(Weight(), (y,), T)] = w
    end
    for ((term, product), a) in zip(sums, acc[(end - length(sums) + 1):end])
        if product === nothing
            emit_tiles!(k, st, c, [term], scope, indices...)
            push!(st, :($a = $a .* $rise .+ sum(ifelse.($valid, $(scope[term]), $(zero(term.eltype))); dims = $p)))
        else
            rise === nothing || push!(st, :($a = $a .* $(matrix_tile(c, rise, product[3], product[4]))))
            accumulate_product!(st, c, scope, product, a, indices[p], p)
        end
    end
end

# The state at the end of the sweep: an exponential fold's running maximum, and the
# sums.
sum_state(c::Context, sums, acc) = Any[acc[1:(end - length(sums))];
    [product === nothing ? a : from_matrix(c, a, product[3], product[4]) for ((_, product), a) in zip(sums, acc[(end - length(sums) + 1):end])]]

# A group's fold in a sweep along the schedule's swept axis: its sums, its
# accumulators, a step, and its state at the end; for folds that do not sum,
# their state's.
fold_sums(c::Context, r) = sweep_sums(c, fold_of(c, r), c.g.terms[r], sweep_axis(c.s))

# A fold's state is named by its place among the group's reductions, not by its
# value: value numbers differ between groups that are alike, and a name that
# held one would give each such group a kernel of its own.
function fold_accumulators!(st, c::Context, r, sums)
    f, A, place = fold_of(c, r), accumulator(c, r), findfirst(==(r), c.g.reductions)
    sums === nothing || return sum_accumulators!(st, c, f, c.g.terms[r], sums, sweep_axis(c.s), Symbol(:acc_, place))
    acc = state_symbols(place, f, A)
    init_state!(st, acc, f, A, tile(c))
    acc
end

function fold_step!(k, st, c::Context, r, sums, acc, scope, valid, indices)
    f, A, term = fold_of(c, r), accumulator(c, r), c.g.terms[r]
    sums === nothing || return sum_step!(k, st, c, f, term, sums, acc, scope, valid, indices, sweep_axis(c.s))
    x = as_accumulator(st, scope[term], A, term_eltype(term, c.g), tile(c))
    accumulate_elements!(st, f, A, acc, x, valid)
end

fold_state!(st, c::Context, r, sums, acc) =
    sums === nothing ? reduce_state!(st, fold_of(c, r), accumulator(c, r), acc, sweep_axis(c.s)) : sum_state(c, sums, acc)

phase_roots(c::Context, phase, sums) = unique(Term[x for r in phase for x in step_roots(c, fold_of(c, r), c.g.terms[r], sums[r])])

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
    p = sweep_axis(c.s)
    at(t) = step_indices(c.s, indices, t)
    names = Dict{Term, Any}()
    load_folds!(k, st, c, earlier, names, at(:(Int32(1))), workspaces)
    sums = Dict(r => fold_sums(c, r) for r in phase)
    accumulators = Dict(r => fold_accumulators!(st, c, r, sums[r]) for r in phase)
    chunk = cld(sweeps(c.s), c.s.splits)
    t = gensym(:t)
    loop = Any[]
    valid = any(r -> bilinear(c, r) === nothing, phase) ? valid_mask!(loop, c, at(t)[p]) : nothing
    scope = copy(names)
    emit_tiles!(k, loop, c, phase_roots(c, phase, sums), scope, at(t)...)
    foreach(r -> fold_step!(k, loop, c, r, sums[r], accumulators[r], scope, valid, at(t)), phase)
    live = live_steps(c, phase, p, indices, sweeps(c.s))
    lo, hi = live === nothing ? (:(Int32(1)), :(Int32($(sweeps(c.s))))) : live
    live === nothing || (k.name *= ", masked steps skipped")
    push!(st, :(for $t in max($lo, (part - Int32(1)) * Int32($chunk) + Int32(1)):min($hi, part * Int32($chunk), Int32($(sweeps(c.s))))
        $(loop...)
    end))
    for r in phase
        state = fold_state!(st, c, r, sums[r], accumulators[r])
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
    idx = linear_indices!(st, counts, :merge_)
    shape = (tiles_m..., width)
    lanes = (map(_ -> 1, tiles_m)..., width)
    for r in rs
        f, A = fold_of(c, r), accumulator(c, r)
        acc = state_symbols(findfirst(==(r), c.g.reductions), f, A)
        init_state!(st, acc, f, A, shape)
        t = gensym(:t)
        loop = Any[]
        valid = gensym(:valid)
        push!(loop, :($valid = reshape($(ct.arange)($width), $lanes) .<= Int32($slots) - ($t - Int32(1)) * Int32($width)))
        loaded = map(eachindex(acc)) do j
            name = gensym(:part)
            sym = argument!(k, partial_spec!(c, r, j, slots, workspaces))
            push!(loop, :($name = $(ct.load)($sym; index = ($(idx...), $t), shape = $shape, padding_mode = $(ct.PaddingMode.Zero))))
            name
        end
        combine!(loop, f, A, acc, loaded, valid)
        push!(st, :(for $t in Int32(1):Int32($(cld(slots, width)))
            $(loop...)
        end))
        state = reduce_state!(st, f, A, acc, length(shape))
        stat = observe!(st, f, A, c.g.values[r].eltype, state)
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

function build_claim(::Tempest.Compiler.cuTileGenerator, claim::Claim, target, g::GroupView, buffers)
    c = Context(g, claim.schedule)
    workspaces = Dict{Any, Workspace}()
    launch_kernels(schedule_kernels(c, workspaces), workspaces, buffers, claim.schedule, target)
end

function launch_kernels(kernels, workspaces, buffers, schedule, target)
    scratch = Dict(key => CuArray{w.eltype}(undef, w.length) for (key, w) in workspaces)
    launches = Tempest.Compiler.concurrently(kernels, target) do k
        f = Tempest.Compiler.generated(source(k))
        arrays = map(spec -> tile_array(spec.buffer isa Int ? buffers[spec.buffer] : scratch[spec.buffer], spec), k.args)
        tile_launch(f, arrays, k.grid; k.occupancy)
    end
    keep = (scratch, buffers)
    launch = () -> (foreach(l -> l(), launches); keep; nothing)
    (launch, BuiltTiles(schedule, kernels))
end
