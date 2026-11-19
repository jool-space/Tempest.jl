# What an implementation sees of a group of the plan: the values the group
# computes, each a term over its domain, and every value it computes or reads,
# as data. The plan stays the compiler's: a view holds what claims, their
# estimates and their builds read, and nothing an implementation could change.

public GroupView, ValueView, members, own_domain, product_axes, structure, Structure, numbering, code

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
`Tempest.Compiler.IR.masked_result`); `observed`: formulas over its
positions, each at most zero where the program's results may depend on what
it holds, nothing when they may depend on all of it (see
`Tempest.Compiler.observed`); and for a recurrence's state,
`carried`: the axis it steps along, the value it is carried from, a load
over its domain read one step back, and whether it steps from the axis's
last position to its first, reading one step on (`reverse`; see
`Tempest.Compiler.Carried`).
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
    observed::Union{Nothing, Vector{Affine}}
    carried::Union{Nothing, @NamedTuple{axis::Int, from::Term, reverse::Bool}}
    # Compiled once, whatever the rank of the value and the type of its terms.
    Base.@nospecializeinfer function ValueView(eltype, @nospecialize(dims), scalar, stored, @nospecialize(fold), @nospecialize(reduced), keyed,
            @nospecialize(into), @nospecialize(scan), @nospecialize(masked), @nospecialize(observed), @nospecialize(carried))
        new(eltype, dims, scalar, stored, fold, reduced, keyed, into, scan, masked, observed, carried)
    end
end

"""
    Structure

A group in its own numbering (see `structure`): equal to another only where
the two are one group but for the numbers of their values. Its functions,
folds and element types are objects, which a function redefined stays: what
they are as code is `code(structure)`, which a table kept longer than one
preparation names too (see `Tempest.Compiler.definitions`).
"""
struct Structure
    parts::Vector{Any}
    hash::UInt
    code::Vector{Any}
end

# Its parts in arrays rather than tuples, a group's many members of many
# kinds: a tuple of them would be a type of its own, whose hash and equality
# Julia would compile for each group.
Structure(parts::Vector{Any}, code::Vector{Any}) = Structure(parts, hash(parts), code)

"""
    code(s::Structure) -> Vector{Any}

The code the parts of `s` name, each once: the functions its terms apply, its
folds and its values' element types. Structures equal name the same.
"""
code(s::Structure) = s.code
Base.:(==)(a::Structure, b::Structure) = a.hash == b.hash && isequal(a.parts, b.parts)
Base.hash(s::Structure, h::UInt) = hash(s.hash, h)

# What a value keeps once found, which tasks reading it side by side may each
# find and store: the same answer.
mutable struct Found{T}
    @atomic value::Union{Nothing, T}
end

"""
    GroupView

A group of the plan as implementations see it. A group computes its members
over one domain: `reductions`, its folds, and `outputs`, its other values, each
defined by its term in `terms`; a fold may cover fewer of the domain's
positions (see `own_domain`). A `:recurrence` group's outputs are a
recurrence's state, its term where the state starts; the values of its cycle
it stores, in order; and last the value the state is carried from, the state
one step on, stored only where something else reads it. Each step computes
them in order, reading what it stored, over the axes their placements
mention (see `recurrence_axes`). A `:scan` group's
`reductions` are scans along its `primary_axes` in one direction, which its
members read at the position its sweep has reached. Folds over `primary_axes`, the
axes its first fold not by key reduces, can be read within the group, and `phase` counts the folds each member waits for. `values` describes every
value the group computes or reads, `placement` where it keeps each member (its
index over the domain, or nothing), and `bytes` what it reads from storage and
writes. `reference(target, buffers)` builds the group's reference kernels, the
floor's, which every claim agrees with. Its `structure` is the rest in its own
numbering.
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
    # Its `structure` and `numbering`, found on first need.
    structure::Found{Tuple{Structure, Vector{Int}}}
    # Compiled once, whatever the rank of the group's domain.
    Base.@nospecializeinfer function GroupView(kind, @nospecialize(domain), reductions, outputs, terms, phase, @nospecialize(primary_axes),
            values, placement, bytes, @nospecialize(reference))
        new(kind, domain, reductions, outputs, terms, phase, primary_axes, values, placement, bytes, reference, Found{Tuple{Structure, Vector{Int}}}(nothing))
    end
end

"""
    structure(group) -> Structure

All the view holds but `reference`, each value numbered by where it first
appears: its members first, in order, then the values their placements and
terms read, then those the values' own views name. Groups of one structure
are one group but for their values' numbers, as a model's layers are: what
an implementation answers of one (`claims`, `variants`, `estimate`) it
answers of the others, where its claims name no value, and it may keep its
answers by it (see `Tempest.Compiler.SessionTable`). Found once a view.
"""
structure(g::GroupView) = first(structure_numbering(g))

"""
    numbering(group) -> Vector{Int}

The values of the group by their number in its `structure`: what an
implementation that keeps what it made for one group by its structure, its
kernels say, reads to give another group of the structure its own values.
"""
numbering(g::GroupView) = last(structure_numbering(g))

function structure_numbering(g::GroupView)
    found = g.structure
    s = @atomic found.value
    s === nothing || return s
    s = structure_of(g)
    @atomic found.value = s
    s
end

function structure_of(g::GroupView)
    numbers, order = Dict{Int, Int}(), Int[]
    number(v) = get!(() -> (push!(order, v); length(order)), numbers, v)
    renamed = Dict{Term, Term}()
    term(t) = t === nothing ? nothing : revalued(t, number, renamed)
    formulas(es) = es === nothing ? nothing : Affine[revalued(e, number) for e in es]
    foreach(number, members(g))
    ms = Any[Any[get(g.phase, m, nothing), formulas(get(g.placement, m, nothing)), term(get(g.terms, m, nothing))] for m in members(g)]
    function value_parts(x::ValueView)
        masked = x.masked === nothing ? nothing : Any[formulas(first(x.masked)), last(x.masked)]
        carried = x.carried === nothing ? nothing : Any[x.carried.axis, term(x.carried.from), x.carried.reverse]
        Any[x.eltype, x.dims, x.scalar, x.stored, x.fold, x.reduced, x.keyed, term(x.into), x.scan, masked, formulas(x.observed), carried]
    end
    # Each value's view, in its number's order, which a view may extend; then
    # those nothing names, in the plan's order.
    vs, rest, k, i = Any[], sort!(collect(keys(g.values))), 0, 1
    while true
        if k == length(order)
            while i <= length(rest) && haskey(numbers, rest[i])
                i += 1
            end
            i > length(rest) && break
            number(rest[i])
        end
        k += 1
        x = get(g.values, order[k], nothing)
        push!(vs, x === nothing ? nothing : value_parts(x))
    end
    code = Base.IdSet{Any}(t.f for t in values(renamed) if t isa Apply)
    for x in values(g.values)
        push!(code, x.eltype)
        x.fold === nothing || push!(code, x.fold)
    end
    (Structure(Any[g.kind, g.domain, g.primary_axes, g.bytes, length(g.reductions), ms, vs], collect(code)), order)
end

"A group of no members over `domain`, reading `values`: where a recomputed fold's body runs."
Base.@nospecializeinfer GroupView(kind::Symbol, @nospecialize(domain), values) = GroupView(kind, domain, Int[], Int[], Dict{Int, Term}(), Dict{Int, Int}(),
    (), values, Dict{Int, Union{Nothing, Tuple{Vararg{Affine}}}}(), 0.0, nothing)

"The values a group computes, its folds first."
members(g::GroupView) = [g.reductions; g.outputs]

"""
    own_domain(group, m) -> extents

The positions member `m` of a group covers: the group's domain, or, for a fold
over fewer positions along axes it keeps, its own within the domain, as MLA's
dV, over the values' 64 features, in the group of dK and dQ, over the 96 of
the queries and keys. Such a fold's term is a term over the domain, but only
at its own positions does it read within what it reads, and only there is it
kept; nothing in the group reads it.
"""
function own_domain(g::GroupView, m)
    v = g.values[m]
    m in g.reductions && !v.keyed || return g.domain
    kept = [k for k in eachindex(g.domain) if !(k in v.reduced)]
    length(kept) == length(v.dims) || return g.domain
    extents = collect(g.domain)
    extents[kept] .= v.dims
    Tuple(extents)
end

"""
    recurrence_axes(g::GroupView) -> (; axis, parallel, inner)

A `:recurrence` group's axes: the state's axis it steps along; those of the
state's along which no term reads what the group computes at another
position, its items' (`parallel`); and the rest, which each step computes
whole (`inner`).
"""
function recurrence_axes(g::GroupView)
    state = g.values[first(g.outputs)]
    recurrence_axes(state.carried.axis, length(state.dims), g.domain, g.outputs, g.terms, g.placement)
end

function recurrence_axes(axis, n, domain, outputs, terms, places)
    across = Set{Int}()
    for m in outputs, t in subterms((terms[m],); nested = true)
        t isa Load && t.value in outputs || continue
        for (e, p) in zip(t.index, places[t.value])
            e == p || foreach(k -> mentions(p, k) && push!(across, k), eachindex(domain))
        end
    end
    axis in across && error("a recurrence reads its state at another step")
    parallel = [k for k in 1:n if k != axis && !(k in across)]
    (; axis, parallel, inner = [k for k in eachindex(domain) if k != axis && !(k in parallel)])
end

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
