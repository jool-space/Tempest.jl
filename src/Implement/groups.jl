# What an implementation sees of a group of the plan: the values the group
# computes, each a term over its domain, and every value it computes or reads,
# as data. The plan stays the compiler's: a view holds what claims, their
# estimates and their builds read, and nothing an implementation could change.

public GroupView, ValueView, members, product_axes

"""
What a view holds of a value: its element type and dims, whether it is a
number rather than an array (stored as one element), and whether it is stored;
for a fold, the fold and the axes of its domain it reduces; and for a value
masked by formulas over its positions, `masked`: the formulas, each at most
zero where the mask holds, and the number the value is where it fails (see
`Tempest.IR.comparison_mask` and `Tempest.IR.masked_result`).
"""
struct ValueView
    eltype::DataType
    dims::Tuple{Vararg{Int}}
    scalar::Bool
    stored::Bool
    fold::Union{Nothing, Fold}
    reduced::Tuple{Vararg{Int}}
    masked::Union{Nothing, Tuple{Vector{Affine}, Any}}
end

"""
    GroupView

A group of the plan as implementations see it. A group computes its members
over one domain: `reductions`, its folds, and `outputs`, its other values, each
defined by its term in `terms`. An `:opaque` group computes one value that the
index level does not model, such as a scatter. Folds over `primary_axes`, the
axes its first fold reduces, can be read within the group, and `phase` counts the folds each member waits for. `values` describes every
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

The axes of a group whose folds each sum a product of two factors over the
group's primary axes `K`: those only the first factor reads (`M`), those only
the second reads (`N`), and those both read and keep (`B`). Nothing when a fold
is not such a product, or the folds split the axes differently.
"""
function product_axes(g::GroupView)
    n = length(g.domain)
    K = sort!(collect(g.primary_axes))
    # Axes of extent one, kept or summed, are read by neither factor and need
    # no tile axis.
    kept = [k for k in 1:n if !(k in K) && g.domain[k] > 1]
    found = nothing
    for r in g.reductions
        factors = product_factors(g.values[r].fold, g.terms[r])
        factors === nothing && return nothing
        ax, ay = axes_read(factors[1], n), axes_read(factors[2], n)
        (ax === nothing || ay === nothing) && return nothing
        all(k -> g.domain[k] == 1 || k in ax && k in ay, K) || return nothing
        M = [k for k in kept if k in ax && !(k in ay)]
        N = [k for k in kept if k in ay && !(k in ax)]
        B = [k for k in kept if k in ax && k in ay]
        length(M) + length(N) + length(B) == length(kept) && !isempty(M) && !isempty(N) || return nothing
        groups = (M, N, K, B)
        found === nothing || found == groups || found == (N, M, K, B) || return nothing
        found === nothing && (found = groups)
    end
    found
end
