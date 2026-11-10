# A value masked by formulas over its positions: `ifelse(cond, x, literal)`
# with `cond` a conjunction of comparisons between index formulas, a
# position and an integer the program reads among them, or a map of such a
# value that is a known number where the mask fails, as the probabilities
# exp(s - lse) of masked scores s are zero. The condition is one too, false
# where it fails, so that a choice by it is known there wherever the
# condition is computed, stored once for every layer that reads it say. An
# implementation that sweeps such a value skips the tiles the mask rules out
# entirely (the cuTile generator's `live_steps`), as FlexAttention's block
# mask does, which it derives from the formulas rather than by evaluating the
# mask on tiles. The dual walk, from the readers back, finds where a value
# is observed at all (`observed`): scores stored bare and masked where they
# are read are observed only where the mask holds.

"""
    masked(graph, v) -> (mask, value) or nothing

The formulas over `v`'s axes that mask it, each at most zero where the mask
holds, and the number `v` is where the mask fails; nothing when `v` is no
such value.
"""
masked(graph::IndexGraph, v) = @lock graph.masks get!(() -> masked_of(graph, v), graph.masks[].of, v)

"""
    mask_reads(graph, v) -> Vector{Int}

The inputs the formulas of `v`'s mask, and of where it is observed, read, as
a length the device holds: a kernel that reads or stores `v` may read them to
bound its sweep (see `masked` and `observed`), and needs run after no other
kernel to.
"""
function mask_reads(graph::IndexGraph, v)
    mask = masked(graph, v)
    formulas = [mask === nothing ? Affine[] : first(mask); something(observed(graph, v), Affine[])]
    [u for e in formulas for u in indirect_values(e) if graph.values[u].definition isa Union{Input, ConstantArray}]
end

"""
    observed(graph, v) -> Vector{Affine} or nothing

Formulas over `v`'s axes, each at most zero where the program's results may
depend on what `v` holds: elsewhere every map reading `v` is a choice that
takes its other branch, or is observed nowhere itself. Nothing when they may
depend on all of `v`. A kernel storing `v` need not compute it elsewhere: the
scores of a decoding step, stored bare and masked by the length where they
are read, are observed up to the length, and the product into them need not
sweep the features of the keys past it.
"""
function observed(graph::IndexGraph, v)
    @lock graph.masks begin
        found = graph.masks[].observed
        isempty(found) && append!(found, observed_values(graph))
        found[v]
    end
end

# Each value's `observed`, from the last value to the first, after its
# readers. A read by a gather, by the results, by anything but a map, or
# through an index that places no variable of the reader on an axis of the
# value (see `read_observes`), observes all of it. A recurrence's state reads
# the value it is carried from, later in the graph, by none of its operands.
function observed_values(graph::IndexGraph)
    n = length(graph.values)
    users = [Tuple{Int, Int}[] for _ in 1:n]
    for (u, val) in enumerate(graph.values)
        d = val.definition
        for (k, o) in enumerate(operands(d)), r in read_values(o)
            push!(users[r], (u, o isa Read && o.value == r ? k : 0))
        end
        d isa Carried && foreach(r -> push!(users[r], (u, 0)), read_values(d.from))
    end
    for o in graph.results, r in read_values(o)
        push!(users[r], (0, 0))
    end
    found = Vector{Any}(nothing, n)
    for v in n:-1:1
        isempty(users[v]) && continue
        common = nothing
        for (u, k) in users[v]
            f = k == 0 ? nothing : read_observes(graph, found, u, k)
            common = f === nothing ? Affine[] : common === nothing ? f : intersect(common, f)
            isempty(common) && break
        end
        found[v] = isempty(common) ? nothing : common
    end
    found
end

# The positions of its `k`th operand that map `u` observes, as formulas over
# the operand's axes: where `u` is observed, and for the branch a choice
# takes where its condition holds, where it holds. A formula that mentions a
# variable of `u` the read places on no axis of the operand, as a broadcast
# does, is dropped, which observes more. Nothing when there are none, or the
# read's index is no permutation of variables of `u`.
function read_observes(graph::IndexGraph, found, u, k)
    def = graph.values[u].definition
    def isa Pointwise || return nothing
    dims = graph.values[u].dims
    formulas = Affine[something(found[u], Affine[])...]
    if def.f === ifelse && k == 2
        w = walked(graph, u, Tuple(var(j) for j in eachindex(dims)), Domain(dims))
        w.condition === nothing || append!(formulas, w.condition)
    end
    isempty(formulas) && return nothing
    o = def.operands[k]
    back = Any[constant(0) for _ in dims]
    placed = falses(length(dims))
    for (a, e) in enumerate(o.index)
        isempty(e.terms) && continue
        length(e.terms) == 1 && e.constant == 0 || return nothing
        (atom, coefficient) = only(e.terms)
        atom isa Var && coefficient == 1 && !placed[atom.n] || return nothing
        back[atom.n], placed[atom.n] = var(a), true
    end
    d = Domain(graph.values[o.value].dims)
    mapped = [substitute(e, Tuple(back), d) for e in formulas if all(j -> placed[j] || !mentions(e, j), eachindex(dims))]
    isempty(mapped) ? nothing : unique(mapped)
end

function masked_of(graph::IndexGraph, v)
    d = Domain(graph.values[v].dims)
    index = Tuple(var(k) for k in 1:length(graph.values[v].dims))
    for m in value_masks(graph, v, index, d, false)
        x = masked_value(graph, v, index, m, d, false)
        x isa Number && return (m, x)
    end
    nothing
end

# The mask a value read at `index` is a condition for, when its definition
# conjoins comparisons of index formulas: those of its comparisons that are,
# when others are not, since the conjunction fails wherever one of them does.
function mask_of(graph::IndexGraph, v, index, d::Domain, depth = 0)
    depth < 32 || return nothing
    def = graph.values[v].definition
    def isa Pointwise || return nothing
    through(e) = substitute(e, index, d)
    if def.f === (&)
        parts = filter(!isnothing, map(o -> o isa Read ? mask_of(graph, o.value, map(through, o.index), d, depth + 1) : nothing, def.operands))
        return isempty(parts) ? nothing : reduce(vcat, parts)
    end
    def.f in (<=, <, >=, >) && length(def.operands) == 2 && any(o -> o isa IndexOperand, def.operands) || return nothing
    a, b = map(o -> compared(graph, o, through, d), def.operands)
    a === nothing || b === nothing ? nothing : comparison_mask(def.f, a, b, d)
end

# A side of a comparison with a position, as a formula: a position, an
# integer read from a value (see `held_formula`), or an integer, as a window's
# width is.
compared(graph, o::IndexOperand, through, d) = through(o.expr)
compared(graph, o::Literal{<:Integer}, through, d) = constant(Int(o.value))
compared(graph, o::Read, through, d) = held_formula(o.value, map(through, o.index), graph.values[o.value].eltype, d)
compared(graph, o, through, d) = nothing

# The masks of the `ifelse` definitions `v`, read at `index`, is computed
# from, through the pointwise values it reads, in the order a walk from `v`
# meets them. Each value at each index is walked once in a graph (`found`):
# values read by several, as a derivative's are, are a graph, not a tree, and
# a value read at its own positions, as each step of a loop reads the step
# before, asks what that value is asked itself. The walk follows one read at
# positions shifted from the reader's, as a slice of masked scores is read
# (`shifted` once it has), and no second: a stencil reads the step before at
# shifted positions, and following every shift meets each value of the steps
# before at a cone of positions that grows with the steps.
function value_masks(graph::IndexGraph, v, index, d::Domain, shifted)
    found = graph.masks[].found
    key = (v, index, d, shifted)
    haskey(found, key) && return found[key]
    found[key] = masks_of(graph, v, index, d, shifted)
end

function masks_of(graph::IndexGraph, v, index, d::Domain, shifted)
    masks = Vector{Affine}[]
    def = graph.values[v].definition
    def isa Pointwise || return masks
    # A condition of comparisons masks itself.
    own = mask_of(graph, v, index, d)
    own === nothing || push!(masks, own)
    w = walked(graph, v, index, d)
    w.condition !== nothing && def.operands[3] isa Literal && push!(masks, w.condition)
    for (o, at) in zip(def.operands, w.indices)
        o isa Read || continue
        shift = shifts(o.index)
        shifted && shift && continue
        union!(masks, value_masks(graph, o.value, at, d, shifted || shift))
    end
    masks
end

# What a walk meets at a map read at `index`: the mask its `ifelse`'s
# condition is, when it is one, and the index each operand is read at, the
# same whatever mask is asked of it (`walks`).
function walked(graph::IndexGraph, v, index, d::Domain)
    walks = graph.masks[].walks
    key = (v, index, d)
    haskey(walks, key) && return walks[key]
    walks[key] = walk_of(graph, v, index, d)
end

function walk_of(graph::IndexGraph, v, index, d::Domain)
    def = graph.values[v].definition
    through(e) = substitute(e, index, d)
    indices = map(o -> o isa Read ? map(through, o.index) : nothing, def.operands)
    condition = def.f === ifelse && length(def.operands) == 3 && def.operands[1] isa Read ? mask_of(graph, def.operands[1].value, indices[1], d) : nothing
    (; condition, indices)
end

# Whether an index reads at positions shifted from the reader's: a formula of
# it adds a constant to the reader's variables, there or under a quotient, a
# remainder or a gather, as a slice, a stencil's neighbours and a circular
# shift do. A permutation, a broadcast and a reshape shift nothing, and reads
# that shift nothing meet a value at few positions however many steps they
# cross.
shifts(index) = any(shifts, index)
shifts(e::Affine) = !isempty(e.terms) && e.constant != 0 || any(t -> shifts(t.first), e.terms)
shifts(a::Var) = false
shifts(a::Union{Div, Mod}) = shifts(a.e)
shifts(a::Indirect) = shifts(a.index)

# What `v`, read at `index`, is where `mask` fails, when the mask's literal
# decides it: a number, `false` for the mask's own condition; `:finite` for
# an input, a fold, an index, and for what sums, scales, exponentiates or
# converts those; nothing when unknown, as a map read through a second shift
# is (see `value_masks`). Each value at each index is asked once a mask in a
# graph (`known`).
function masked_value(graph::IndexGraph, v, index, mask, d::Domain, shifted)
    known = graph.masks[].known
    key = (v, index, mask, d, shifted)
    haskey(known, key) && return known[key]
    known[key] = masked_value_of(graph, v, index, mask, d, shifted)
end

function masked_value_of(graph::IndexGraph, v, index, mask, d::Domain, shifted)
    val = graph.values[v]
    def = val.definition
    def isa Pointwise || return :finite
    w = walked(graph, v, index, d)
    function read_value(o, at)
        shift = shifts(o.index)
        shifted && shift && graph.values[o.value].definition isa Pointwise && return nothing
        masked_value(graph, o.value, at, mask, d, shifted || shift)
    end
    operand(o, at) = o isa Literal ? o.value : o isa IndexOperand ? :finite : o isa Read ? read_value(o, at) : nothing
    mask_of(graph, v, index, d) == mask && return false
    w.condition == mask && return operand(def.operands[3], w.indices[3])
    masked_result(def.f, val.eltype, map(operand, def.operands, w.indices))
end
