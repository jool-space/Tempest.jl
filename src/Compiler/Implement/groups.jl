# What an implementation sees of a group of the plan: the values the group
# computes, each a term over its domain, and every value it computes or reads,
# as data. The plan stays the compiler's: a view holds what claims, their
# estimates and their builds read, and nothing an implementation could change.

public GroupView, ValueView, members, product_axes

"""
What a view holds of a value: its element type and dims, whether it is a
number rather than an array (stored as one element), and whether it is stored;
for a fold, the fold and the axes of its domain it reduces, and whether it is
a fold by key (`keyed`), which reduces every axis and combines each element
at its placement, after `into` where it is given: the contents it starts
from, a term over its own domain; for a scan its fold, no axes, and `scan`:
whether it runs from the last element down (`reverse`) and leaves each
element out of its own prefix (`exclusive`), nothing for any other value; and
for a value masked by formulas over its positions, `masked`: the formulas,
each at most zero where the mask holds, and the number the value is where it
fails (see `Tempest.Compiler.IR.comparison_mask` and
`Tempest.Compiler.IR.masked_result`); and for a recurrence's state,
`carried`: the axis it steps along and the value it is carried from, a load
over its domain read one step back (see `Tempest.Compiler.Carried`).
"""
struct ValueView
    eltype::DataType
    dims::Tuple{Vararg{Int}}
    scalar::Bool
    stored::Bool
    fold::Union{Nothing, Fold}
    reduced::Tuple{Vararg{Int}}
    keyed::Bool
    into::Union{Nothing, Term}
    scan::Union{Nothing, @NamedTuple{reverse::Bool, exclusive::Bool}}
    masked::Union{Nothing, Tuple{Vector{Affine}, Any}}
    carried::Union{Nothing, @NamedTuple{axis::Int, from::Term}}
end

"""
    GroupView

A group of the plan as implementations see it. A group computes its members
over one domain: `reductions`, its folds, and `outputs`, its other values, each
defined by its term in `terms`. A `:recurrence` group's outputs are a
recurrence's state, its term where the state starts, and the value it is
carried from, which each step computes from the state there. A `:scan` group's
`reductions` are scans along its `primary_axes` in one direction, which its
members read at the position its sweep has reached. Folds over `primary_axes`, the
axes its first fold not by key reduces, can be read within the group, and `phase` counts the folds each member waits for. `values` describes every
value the group computes or reads, `placement` where it keeps each member (its
index over the domain, or nothing), and `bytes` what it reads from storage and
writes. `reference(target, buffers)` builds the group's reference kernels, the
floor's, which every claim agrees with.
"""
struct GroupView
    kind::Symbol
    domain::Tuple{Vararg{Int}}
    reductions::Vector{Int}
    outputs::Vector{Int}
    terms::Dict{Int, Term}
    phase::Dict{Int, Int}
    primary_axes::Tuple{Vararg{Int}}
    values::Dict{Int, ValueView}
    placement::Dict{Int, Union{Nothing, Tuple{Vararg{Affine}}}}
    bytes::Float64
    reference::Any
end

"A group of no members over `domain`, reading `values`: where a recomputed fold's body runs."
GroupView(kind::Symbol, domain, values) = GroupView(kind, domain, Int[], Int[], Dict{Int, Term}(), Dict{Int, Int}(),
    (), values, Dict{Int, Union{Nothing, Tuple{Vararg{Affine}}}}(), 0.0, nothing)

"The values a group computes, its folds first."
members(g::GroupView) = [g.reductions; g.outputs]

# Products: groups whose folds sum a product of two factors. Which of the
# domain's axes each factor reads splits the kept axes into those only the
# first reads (M), those only the second reads (N) and those both read (a
# batch, B); the summed axes are K. Implementations of products read this
# split, and each factor as a strided matrix over it.

# The domain's axes a term reads, also through the folds it recomputes, or
# nothing when it reads one of the group's folds.
function axes_read(t::Term, n)
    axes = Set{Int}()
    for s in subterms((t,); nested = true)
        s isa Stat && return nothing
        exprs = s isa Load ? s.index : s isa Ix ? (s.expr,) : ()
        for e in exprs, k in 1:n
            mentions(e, k) && push!(axes, k)
        end
    end
    axes
end

"""
    product_axes(group) -> (M, N, K, B) or nothing
    product_axes(group, r) -> (M, N, K, B) or nothing

The axes of a group whose folds each sum a product of two factors over the
group's primary axes `K`: those only the first factor reads (`M`), those only
the second reads (`N`), and those both read and keep (`B`). Nothing when a fold
is not such a product, or the folds split the axes differently; a fold of
weights alone, a log-sum-exp beside attention's weighted sum, has no factors
and rides the products' sweep. Given a fold `r`, the axes of that fold's
product alone.
"""
function product_axes(g::GroupView)
    found = nothing
    for r in g.reductions
        weights_alone(g, r) && continue
        groups = product_axes(g, r)
        groups === nothing && return nothing
        found === nothing || found == groups || found == (groups[2], groups[1], groups[3], groups[4]) || return nothing
        found === nothing && (found = groups)
    end
    found
end

"Whether fold `r` is an exponential fold of a number: of the weights `exp(y - m)` alone."
weights_alone(g::GroupView, r) = g.values[r].fold isa Relative{Exponential} && !g.values[r].keyed &&
    !(g.terms[r] isa Apply && g.terms[r].f === tuple)

function product_axes(g::GroupView, r)
    n = length(g.domain)
    K = sort!(collect(g.primary_axes))
    # Axes of extent one, kept or summed, are read by neither factor and need
    # no tile axis.
    kept = [k for k in 1:n if !(k in K) && g.domain[k] > 1]
    g.values[r].keyed && return nothing
    factors = product_factors(g.values[r].fold, g.terms[r])
    factors === nothing && return nothing
    ax, ay = axes_read(factors[1], n), axes_read(factors[2], n)
    (ax === nothing || ay === nothing) && return nothing
    all(k -> g.domain[k] == 1 || k in ax && k in ay, K) || return nothing
    M = [k for k in kept if k in ax && !(k in ay)]
    N = [k for k in kept if k in ay && !(k in ax)]
    B = [k for k in kept if k in ax && k in ay]
    length(M) + length(N) + length(B) == length(kept) && !isempty(M) && !isempty(N) || return nothing
    (M, N, K, B)
end
