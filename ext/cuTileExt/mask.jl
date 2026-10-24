# The steps of a sweep a mask rules out. A fold whose body is a mask over its
# positions, `ifelse(cond, x, identity)` with the fold's identity where the
# condition fails, sees nothing at a step whose whole tile fails it: the step
# may be skipped. When each comparison of the condition is between two
# formulas over the kept axes and the swept one, the steps that may pass it
# are an interval of the sweep, bounded from the block's indices by interval
# arithmetic on the formulas, as FlexAttention's block mask skips the key
# blocks a causal mask rules out, found here from the mask's formula rather
# than by evaluating it on block corners. A formula may read an integer the
# block shares, a sequence's length say, which the bound loads once, as
# FlashAttention's decode bounds its keys by `cache_seqlens`. The mask may sit
# in the fold's own term or in a value it loads, computed and stored by an
# earlier kernel, as attention's masked scores are for its backward pass: a
# stored value's definition is read through to the formula.

# The mask a term is a condition for, when it conjoins comparisons of index
# formulas (see `Tempest.Compiler.masked`).
function mask_of(c::Context, t, d::Domain)
    t isa Apply || return nothing
    if t.f === (&)
        parts = map(a -> mask_of(c, a, d), t.args)
        return any(isnothing, parts) ? nothing : reduce(vcat, parts)
    end
    t.f in (<=, <, >=, >) && length(t.args) == 2 && any(a -> a isa Ix, t.args) || return nothing
    a, b = map(a -> compared(c, a, d), t.args)
    a === nothing || b === nothing ? nothing : IR.comparison_mask(t.f, a, b, d)
end

# A side of a comparison with a position, as a formula: a position, or an
# integer loaded from a value (see `Tempest.IR.held_formula`).
compared(c::Context, t::Ix, d) = t.expr
compared(c::Context, t::Load, d) = IR.held_formula(t.value, t.index, c.g.values[t.value].eltype, d)
compared(c::Context, t, d) = nothing

# A loaded value's mask, over the kernel's axes: the view's, through the
# load's formula.
function load_mask(c::Context, t::Load, d::Domain)
    m = c.g.values[t.value].masked
    m === nothing ? nothing : (map(e -> IR.substitute(e, t.index, d), m[1]), m[2])
end

# The condition a fold's body is masked by: an `ifelse` in it, or a value it
# loads that is masked, with a literal where the condition fails, under which
# the body is the fold's identity. An exponential fold's maximum and weights
# see nothing of a position at minus infinity; a sum nothing of a zero, which
# a product with a zero factor is.
function mask_condition(c::Context, r)
    f, body = fold_of(c, r), c.g.terms[r]
    f isa Relative{Exponential} && ((body, _) = shifted_parts(c, body))
    identity = f isa Relative{Exponential} ? v -> v == -Inf : f isa SumFold ? iszero :
        f isa MaxFold ? v -> v == typemin(typeof(v)) : f isa MinFold ? v -> v == typemax(typeof(v)) : return nothing
    d = Domain(c.g.domain)
    found = Vector{Affine}[]
    for t in subterms((body,))
        if t isa Apply && t.f === ifelse && length(t.args) == 3 && t.args[3] isa Lit
            m = mask_of(c, t.args[1], d)
            m === nothing || push!(found, m)
        elseif t isa Load
            m = load_mask(c, t, d)
            m === nothing || push!(found, m[1])
        end
    end
    for mask in unique(found)
        v = masked_value(c, body, mask, d)
        v isa Number && identity(v) && return mask
    end
    nothing
end

# What a term is where `mask` fails, when the mask's literal decides it: a
# number; `:finite` for an index, a statistic, a fold or an unmasked load;
# nothing when unknown.
function masked_value(c::Context, t, mask, d::Domain)
    t isa Lit && return t.value
    t isa Union{Ix, Stat, Recomputed} && return :finite
    if t isa Load
        m = load_mask(c, t, d)
        return m !== nothing && m[1] == mask ? m[2] : :finite
    end
    t isa Apply || return nothing
    t.f === ifelse && length(t.args) == 3 && mask_of(c, t.args[1], d) == mask && return masked_value(c, t.args[3], mask, d)
    IR.masked_result(t.f, t.eltype, map(a -> masked_value(c, a, mask, d), t.args))
end

"""
    live_steps(k, c, phase, p, indices, n) -> (lo, hi) or nothing

Int32 expressions over the block's indices bounding the steps of the sweep
along axis `p`, of `n` steps, at which some fold of `phase` sees a position
its mask does not rule out: the union over the folds of the steps each is
live at, each the intersection of what its mask's comparisons allow. Nothing
when a fold of the phase has no mask the generator can bound. An integer a
formula reads is an argument of kernel `k`.
"""
function live_steps(k, c::Context, phase, p, indices, n)
    # `TEMPEST_SKIP_MASKED=0` runs every step, for telling a wrong answer's cause.
    get(ENV, "TEMPEST_SKIP_MASKED", "1") == "0" && return nothing
    los, his = Any[], Any[]
    for r in phase
        mask = mask_condition(c, r)
        mask === nothing && return nothing
        lo, hi = Any[:(Int32(1))], Any[:(Int32($n))]
        for e in mask
            bounds = step_bounds(k, c, e, p, indices)
            bounds === nothing && return nothing
            α, β = bounds
            # Live where α·(t − 1) + β ≤ 0.
            if α > 0
                push!(hi, :(Int32(1) + fld(-$β, Int32($α))))
            elseif α < 0
                push!(lo, :(Int32(1) + cld($β, Int32($(-α)))))
            else
                push!(hi, :(ifelse($β <= Int32(0), Int32($n), Int32(0))))
            end
        end
        push!(los, :(max($(lo...)))); push!(his, :(min($(hi...))))
    end
    (:(min($(los...))), :(max($(his...))))
end

"""
    live_fraction(c, r) -> Float64

The share of the pairs of a block and a step of the sweep that fold `r`
takes, each tile's lowest of its mask's formulas bounding it as `live_steps`
does: what an estimate prices of a sweep whose masked steps are skipped. One
when the fold has no such mask, its tile is resident, a variable shares its
tile axis or lies on another swept one, or a formula reads an integer the
device holds, which a kernel is timed at the far end of (`held_integers`).
"""
function live_fraction(c::Context, r)
    sweeps(c.s) > 1 || return 1.0
    mask = mask_condition(c, r)
    mask === nothing && return 1.0
    p = sweep_axis(c.s)
    axes = Int[]
    for e in mask, (atom, _) in e.terms
        atom isa Var || return 1.0
        i = findfirst(ax -> atom.n in ax.axes, c.s.tiling)
        i === nothing && continue
        ax = c.s.tiling[i]
        ax.axes == [atom.n] && (i == p || !ax.reduced) || return 1.0
        i in axes || push!(axes, i)
    end
    counts = [cld(c.s.tiling[i].extent, c.s.tiling[i].tile) for i in axes]
    total = prod(counts; init = 1)
    total <= 1 << 20 || return 1.0
    # The lowest of formula `e` over the tiles at `at`, one per axis of `axes`.
    function lowest(e, at)
        x = e.constant
        for (atom, coefficient) in e.terms
            i = findfirst(ax -> atom.n in ax.axes, c.s.tiling)
            i === nothing && continue
            tile = c.s.tiling[i].tile
            x += coefficient * tile * (at[findfirst(==(i), axes)] - 1) + min(coefficient, 0) * (tile - 1)
        end
        x
    end
    count(at -> all(e -> lowest(e, at) <= 0, mask), Iterators.product((1:n for n in counts)...)) / total
end

"""
    ruled_out(k, c, stored, indices) -> Expr or nothing

An Int32 expression over the block's indices, above zero when each value of
`stored` is its mask's literal across the block's whole tile, so that the
block's sweep may be skipped: a decoding step's scores past the length the
device holds are minus infinity whatever the keys there, and the sweep over
the features reads every key. A tile is ruled out when some formula of a
value's mask is above zero at its lowest over the tile (`step_bounds`).
Only sums are skipped: a sum over nothing is zero, a finite statistic, as a
mask's literal takes a fold to be (`Tempest.Compiler.masked`), where an
exponential fold over nothing is not. Nothing when a stored value has no
such mask, or the kernel keeps partial states.
"""
function ruled_out(k, c::Context, stored, indices)
    isempty(partial_folds(c)) && isempty(keyed_folds(c)) || return nothing
    all(r -> fold_of(c, r) isa SumFold, c.g.reductions) || return nothing
    get(ENV, "TEMPEST_SKIP_MASKED", "1") == "0" && return nothing
    p = sweep_axis(c.s)
    d = Domain(c.g.domain)
    # The least over the values of each one's highest formula at its lowest
    # over the tile: the tile is ruled out where it is above zero.
    highest = Any[]
    for m in stored
        mask, place = c.g.values[m].masked, c.g.placement[m]
        (mask === nothing || place === nothing || isempty(first(mask))) && return nothing
        lowest = Any[]
        for e in first(mask)
            bounds = step_bounds(k, c, IR.substitute(e, place, d), p, indices)
            # A formula that moves along the sweep is not one tile's.
            (bounds === nothing || first(bounds) != 0) && return nothing
            push!(lowest, last(bounds))
        end
        push!(highest, reduce((a, b) -> :(max($a, $b)), lowest))
    end
    isempty(highest) ? nothing : reduce((a, b) -> :(min($a, $b)), highest)
end

# The lowest value of formula `e` over a tile at step `t` of the sweep along
# `p`, as `α·(t − 1) + β`: α a number, β an Int32 expression over the
# block's indices. Nothing when a variable of `e` is not alone on its tile
# axis, or lies on another swept axis, or an integer it reads differs between
# the block's lanes.
function step_bounds(k, c::Context, e::Affine, p, indices)
    α = 0
    β = Any[:(Int32($(e.constant)))]
    for (atom, coefficient) in e.terms
        if atom isa Indirect
            held = block_read(k, c, atom, indices)
            held === nothing && return nothing
            push!(β, :(Int32($coefficient) * $held))
            continue
        end
        atom isa Var || return nothing
        i = findfirst(ax -> atom.n in ax.axes, c.s.tiling)
        i === nothing && continue
        ax = c.s.tiling[i]
        ax.axes == [atom.n] || return nothing
        if i == p
            α += coefficient * ax.tile
        elseif ax.reduced
            return nothing
        else
            push!(β, :(Int32($(coefficient * ax.tile)) * ($(indices[i]) - Int32(1))))
        end
        # The lowest over the tile's lanes, 0 to tile − 1.
        coefficient < 0 && push!(β, :(Int32($(coefficient * (ax.tile - 1)))))
    end
    (α, :(+($(β...))))
end

# An integer read at an index the grid binds, as an Int32 the block loads
# once: every variable of the index on a kept tile axis of one lane, so that
# neither the lanes nor the sweep move it. Nothing otherwise, when the
# integer is not stored, or when it may not fit an Int32.
function block_read(k, c::Context, a::Indirect, indices)
    c.g.values[a.value].stored || return nothing
    T = c.g.values[a.value].eltype
    typemin(Int32) <= typemin(T) && typemax(T) <= typemax(Int32) || return nothing
    for e in a.index, (atom, _) in e.terms
        atom isa Var || return nothing
        i = findfirst(ax -> atom.n in ax.axes, c.s.tiling)
        i === nothing && continue
        c.s.tiling[i].tile == 1 && !c.s.tiling[i].reduced || return nothing
    end
    flat = argument!(k, ArgSpec(a.value, 0, (storage_length(c.g.values[a.value]),), (1,)))
    at = block_scalar(c, address(c.g.values[a.value], a.index, Domain(c.g.domain)), indices)
    :(Int32($flat[Int32(1) + $at]))
end
