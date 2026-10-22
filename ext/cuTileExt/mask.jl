# The steps of a sweep a mask rules out. A fold whose body is a mask over its
# positions, `ifelse(cond, x, identity)` with the fold's identity where the
# condition fails, sees nothing at a step whose whole tile fails it: the step
# may be skipped. When each comparison of the condition is between two
# formulas over the kept axes and the swept one, the steps that may pass it
# are an interval of the sweep, bounded from the block's indices by interval
# arithmetic on the formulas, as FlexAttention's block mask skips the key
# blocks a causal mask rules out, found here from the mask's formula rather
# than by evaluating it on block corners. The mask may sit in the fold's own
# term or in a value it loads, computed and stored by an earlier kernel, as
# attention's masked scores are for its backward pass: a stored value's
# definition is read through to the formula.

const IR = Tempest.IR

# The mask a term is a condition for, when it conjoins comparisons of index
# formulas (see `Tempest.Compiler.masked`).
function mask_of(t, d::Domain)
    t isa Apply || return nothing
    if t.f === (&)
        parts = map(a -> mask_of(a, d), t.args)
        return any(isnothing, parts) ? nothing : reduce(vcat, parts)
    end
    t.f in (<=, <, >=, >) && length(t.args) == 2 && all(a -> a isa Ix, t.args) || return nothing
    IR.comparison_mask(t.f, t.args[1].expr, t.args[2].expr, d)
end

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
            m = mask_of(t.args[1], d)
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
    t.f === ifelse && length(t.args) == 3 && mask_of(t.args[1], d) == mask && return masked_value(c, t.args[3], mask, d)
    IR.masked_result(t.f, t.eltype, map(a -> masked_value(c, a, mask, d), t.args))
end

"""
    live_steps(c, phase, p, indices, n) -> (lo, hi) or nothing

Int32 expressions over the block's indices bounding the steps of the sweep
along axis `p`, of `n` steps, at which some fold of `phase` sees a position
its mask does not rule out: the union over the folds of the steps each is
live at, each the intersection of what its mask's comparisons allow. Nothing
when a fold of the phase has no mask the generator can bound.
"""
function live_steps(c::Context, phase, p, indices, n)
    # `TEMPEST_SKIP_MASKED=0` runs every step, for telling a wrong answer's cause.
    get(ENV, "TEMPEST_SKIP_MASKED", "1") == "0" && return nothing
    los, his = Any[], Any[]
    for r in phase
        mask = mask_condition(c, r)
        mask === nothing && return nothing
        lo, hi = Any[:(Int32(1))], Any[:(Int32($n))]
        for e in mask
            bounds = step_bounds(c, e, p, indices)
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

# The lowest value of formula `e` over a tile at step `t` of the sweep along
# `p`, as `α·(t − 1) + β`: α a number, β an Int32 expression over the
# block's indices. Nothing when a variable of `e` is not alone on its tile
# axis, or lies on another swept axis.
function step_bounds(c::Context, e::Affine, p, indices)
    α = 0
    β = Any[:(Int32($(e.constant)))]
    for (atom, coefficient) in e.terms
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
