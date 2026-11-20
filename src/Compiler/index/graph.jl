# The index level: every value is a definition over its own domain, and every
# read of another value goes through an index formula over the reader's domain.
# Layout operations do not exist here; they were absorbed into formulas.

abstract type Operand end

"Read `value` at `index`, one expression per axis of `value`, over the reader's domain."
struct Read <: Operand
    value::Int
    index::Tuple{Vararg{Affine}}
    # Compiled once, whatever the rank of the index, as the definitions'
    # constructors are, whatever the ranks of their tuples.
    Read(value, @nospecialize(index)) = new(value, index::Tuple{Vararg{Affine}})
end

"A constant, broadcast over the reader's domain."
struct Literal{T} <: Operand
    value::T
end

"An integer computed from the reader's indices, converted to `T`: positions."
struct IndexOperand{T} <: Operand
    expr::Affine
end

IndexOperand(expr, T::Type) = IndexOperand{T}(expr)

abstract type Definition end

"The `argument`th input of the program."
struct Input <: Definition
    argument::Int
end

"A constant array the program needs at run time, such as the indices of a gather."
struct ConstantArray{A} <: Definition
    data::A
end

# A definition's function or fold is a field without a type parameter, as
# `Apply`'s is: definitions are data to plan, and a parameter would compile
# every method that takes one again for each function and fold a program
# applies.

"""
`f` applied to the operands at each index of the value's domain. The operands of
a commutative `f` are kept in one order, so that spelling does not change the value.
"""
struct Pointwise <: Definition
    f::Any
    operands::Tuple{Vararg{Operand}}
    Base.@nospecializeinfer function Pointwise(@nospecialize(f), @nospecialize(operands::Tuple{Vararg{Operand}}))
        commutative(f) && (operands = Tuple(sort(collect(Operand, operands); by = operand_key, lt = parts_isless)))
        new(f, operands)
    end
end

# A total order on operands by content: reads, then positions, then constants.
operand_key(o::Read) = (0, o.value, map_elements(affine_key, o.index))
operand_key(o::IndexOperand) = (1, affine_key(o.expr))
operand_key(o::Literal) = (2, string(nameof(typeof(o.value))), o.value)

"""
A fold of `operand` over the `reduced` axes of `iteration`. The operand is read
over the iteration domain; the value's own domain is the iteration's other axes.
"""
struct Reduction <: Definition
    fold::Fold
    operand::Operand
    iteration::Tuple{Vararg{Int}}
    reduced::Tuple{Vararg{Int}}
    Reduction(fold, @nospecialize(operand), @nospecialize(iteration), @nospecialize(reduced)) =
        new(fold, operand, int_tuple(iteration), int_tuple(reduced))
end

"""
A fold of `operand` along the `axes` of the value's domain that keeps them:
at each position, the fold of the operand at the positions up to it along
those axes, in their linear order (the first fastest), the other axes fixed;
from the last position down with `reverse`, and without the position itself
with `exclusive`. The operand is read over the value's domain. Its elements
combine in order, the earlier state first, so its fold need not commute.
"""
struct Scan <: Definition
    fold::Fold
    operand::Operand
    axes::Tuple{Vararg{Int}}
    reverse::Bool
    exclusive::Bool
    Scan(fold, @nospecialize(operand), @nospecialize(axes), reverse, exclusive) = new(fold, operand, int_tuple(axes), reverse, exclusive)
end

"""
A fold by key, a scatter or a write: each element of `operand`, read over
`iteration`, combined into the value at `key`, its position as formulas over
the iteration, one per axis of the value. The elements a position takes
combine in any order, as a fold's do, after `into`, an operand over the
value's own domain, where it is given: the contents a write starts from. A
position no element reaches holds `into` there, or the fold's initial state.
The value holds the states, in the accumulator of the operand's elements: the
fold's state is one number, observed as itself.
"""
struct Keyed <: Definition
    fold::Fold
    operand::Operand
    iteration::Tuple{Vararg{Int}}
    key::Tuple{Vararg{Affine}}
    into::Union{Nothing, Operand}
    Keyed(fold, @nospecialize(operand), @nospecialize(iteration), @nospecialize(key), @nospecialize(into)) =
        new(fold, operand, int_tuple(iteration), key::Tuple{Vararg{Affine}}, into)
end

Keyed(fold, operand, iteration, key) = Keyed(fold, operand, iteration, key, nothing)

"""
The state of a recurrence along `axis` of the value's domain: at the axis's
first position `init`, read over the value's domain, and at each other the
operand `from` read one position back; where `reverse`, at its last position
`init` and at each other `from` one position on, as a recurrence's
derivative steps through its steps. `from` reads a value computed from
this one, later in the graph: the one read that does not follow the graph's
order, so the two values and those between are computed position by position
along the axis together (see `recurrence_regions`). `operands` leaves it out.
"""
struct Carried <: Definition
    init::Operand
    from::Operand
    axis::Int
    reverse::Bool
end

Carried(init, from, axis) = Carried(init, from, axis, false)

"""
`operand`, whose value lives on another device than what it reads: a
crossing between devices, a copy (see `with_devices`) of one span of what it
reads in storage (see `storage_span`), all of it or a range of its last axis
that is all its readers read. It is the one read that joins two devices:
every other joins values on one.
"""
struct Crossed <: Definition
    operand::Read
end

"""
A value: its definition, the extents of its domain (none of them one; see
`wide_axes`), its element type, and whether it is a number rather than an array.
"""
struct Value
    definition::Definition
    dims::Tuple{Vararg{Int}}
    eltype::DataType
    scalar::Bool
    Value(@nospecialize(definition), @nospecialize(dims), eltype, scalar) = new(definition, int_tuple(dims), eltype, scalar)
end

domain(v::Value) = Domain(v.dims)
iteration_domain(v::Value) = v.definition isa Union{Reduction, Keyed} ? Domain(v.definition.iteration) : domain(v)

function dense_address(dims, index, d::Domain)
    total = constant(0)
    stride = 1
    for (e, n) in zip(index, dims)
        total = add(total, scale(e, stride), d)
        stride *= n
    end
    total
end

"Whether value `v` read at `index` over a reader of shape `dims` is its storage as it is."
reads_storage(v::Value, index, dims) = prod(v.dims; init = 1) == prod(dims; init = 1) && storage_span(v, index, dims) == 0

"""
    storage_span(v, index, dims) -> Int or nothing

Where value `v` read at `index` over a reader of shape `dims` is one span of
its storage, the reader's elements in its own order: the elements of `v`
before the span; nothing where the read is not one.
"""
function storage_span(v::Value, index, dims)
    d = Domain(dims)
    offset = add(dense_address(v.dims, index, d), scale(dense_address(dims, identity_index(dims), d), -1), d)
    isconstant(offset) ? offset.constant : nothing
end

# What `masked` finds of a graph's values: each value's mask (`of`), and what
# the walks under it find of a value read at an index, which the walks from
# many values share (see plan/masks.jl): the indices it reads at and its
# condition's mask (`walks`), the masks it is computed from (`found`), and
# what it is where a mask fails (`known`); and where each value is observed
# (`observed`), found for every value at once, empty until asked.
struct Masks
    of::Dict{Int, Any}
    walks::Dict{Any, Any}
    found::Dict{Any, Vector{Vector{Affine}}}
    known::Dict{Any, Any}
    observed::Vector{Any}
end

Masks() = Masks(Dict{Int, Any}(), Dict{Any, Any}(), Dict{Any, Vector{Vector{Affine}}}(), Dict{Any, Any}(), Any[])

"""
    IndexGraph

A program at the index level: values in topological order, the values that are
arguments, and each result as a read over the result's own shape (results may be
layouts of values, or constants). `rewrites` notes the values that rewrites of
the graph made, and what each rewrote. `backward` marks the values of the
program's backward, where a reverse pass recorded it. `devices` gives the
device each value lives on, by label, the first device the arguments are on
being 1 (see `with_devices`): all on 1 until the graph is placed. `masks` holds what
`masked` finds (`Masks`), once per graph: every view of a group asks it of
the values the group reads.
"""
struct IndexGraph
    values::Vector{Value}
    arguments::Vector{Int}
    results::Vector{Operand}
    result_dims::Vector{Tuple{Vararg{Int}}}
    result_scalar::Vector{Bool}
    result_eltype::Vector{DataType}
    rewrites::Vector{Pair{Int, String}}
    backward::BitVector
    devices::Vector{Int}
    masks::Base.Lockable{Masks, ReentrantLock}
end

IndexGraph(values, arguments, results, dims, scalar, eltype, rewrites = Pair{Int, String}[], backward = falses(length(values)),
        devices = ones(Int, length(values))) =
    IndexGraph(values, arguments, results, dims, scalar, eltype, rewrites, backward, devices, Base.Lockable(Masks()))

# The element type of what an operand reads, among a graph's values.
eltype_of(values::Vector{Value}, o::Read) = values[o.value].eltype
eltype_of(values::Vector{Value}, o::Literal) = typeof(o.value)
eltype_of(values::Vector{Value}, ::IndexOperand{T}) where {T} = T
eltype_of(graph::IndexGraph, o::Operand) = eltype_of(graph.values, o)

# The operands a definition reads, in a vector, which every loop over it and
# every function it is passed to compiles once for, where the tuple a
# pointwise value holds was compiled for each mix of operand kinds a program
# has.
operands(d::Definition) = Operand[]
function operands(d::Pointwise)
    out = Vector{Operand}(undef, length(d.operands))
    for k in eachindex(out)
        out[k] = d.operands[k]
    end
    out
end
operands(d::Union{Reduction, Scan, Crossed}) = Operand[d.operand]
operands(d::Carried) = Operand[d.init]
# A fold by key reads its key's positions, as a gather reads its indices, and
# the contents it starts from.
function operands(d::Keyed)
    out = Operand[d.operand]
    for e in d.key
        isempty(indirect_values(e)) || push!(out, IndexOperand{Int}(e))
    end
    d.into === nothing || push!(out, d.into)
    out
end

function reads(d::Definition)
    out = Read[]
    for o in operands(d)
        o isa Read && push!(out, o)
    end
    out
end

# The domain each operand of value `v` is read over, as `operands` lists them:
# a fold's operand, and a fold by key's with its key's positions, over the
# iteration, and every other over the value's own domain.
operand_domains(v::Value) = operand_domains(v.definition, v)
operand_domains(d::Definition, v::Value) = Domain[domain(v) for _ in operands(d)]
operand_domains(d::Reduction, v::Value) = Domain[Domain(d.iteration)]
function operand_domains(d::Keyed, v::Value)
    over = Domain(d.iteration)
    out = Domain[over]
    for e in d.key
        isempty(indirect_values(e)) || push!(out, over)
    end
    d.into === nothing || push!(out, domain(v))
    out
end

# The values an operand reads: its own, and those its formulas gather from.
gathered_values(o::Read) = indirect_values(o.index)
gathered_values(o::IndexOperand) = indirect_values(o.expr)
gathered_values(o::Literal) = Int[]
read_values(o::Read) = pushfirst!(gathered_values(o), o.value)
read_values(o::Operand) = gathered_values(o)

# The spans of values computed together, position by position along their
# recurrence's axis: from each carried state to the last value it is carried
# from, overlapping spans merged.
function recurrence_regions(graph::IndexGraph)
    spans = UnitRange{Int}[]
    for (id, v) in enumerate(graph.values)
        d = v.definition
        d isa Carried && d.from isa Read && d.from.value > id || continue
        span = id:d.from.value
        if !isempty(spans) && first(span) <= last(last(spans))
            spans[end] = first(last(spans)):max(last(last(spans)), last(span))
        else
            push!(spans, span)
        end
    end
    spans
end

function readers(graph::IndexGraph)
    users = [Int[] for _ in graph.values]
    for (id, v) in enumerate(graph.values)
        for r in reads(v.definition)
            push!(users[r.value], id)
        end
    end
    users
end

describe(d::Input) = "input $(d.argument)"
describe(d::ConstantArray) = "constant array $(size(d.data))"
describe(d::Pointwise) = "pointwise($(d.f))"
describe(d::Reduction) = "reduce($(fold_name(d.fold)), reduce $(d.reduced) of $(d.iteration))"
describe(d::Scan) = "scan($(scan_summary(d.fold, d.axes, d.reverse, d.exclusive)))"
describe(d::Crossed) = string("crossed from ", sprint(show_operand, d.operand))
describe(d::Carried) = string("carried along ", d.axis, " from ", sprint(show_operand, d.from), d.reverse ? " one on" : " one back")
describe(d::Keyed) = "reduce($(fold_name(d.fold)) by key [$(join(map(string, d.key), ", "))] of $(d.iteration)" *
    (d.into === nothing ? ")" : string(" into ", sprint(show_operand, d.into), ")"))

show_operand(io, o::Read) = print(io, "%", o.value, "[", join(map(string, o.index), ", "), "]")
show_operand(io, o::Literal) = print(io, repr(o.value))
show_operand(io, o::IndexOperand{T}) where {T} = print(io, T, "(", o.expr, ")")

function Base.show(io::IO, ::MIME"text/plain", g::IndexGraph)
    println(io, "IndexGraph with $(length(g.values)) values")
    for (id, v) in enumerate(g.values)
        print(io, "  %", id, " = ", describe(v.definition))
        ops = operands(v.definition)
        if !isempty(ops)
            print(io, " ")
            for (k, o) in enumerate(ops)
                k > 1 && print(io, ", ")
                show_operand(io, o)
            end
        end
        print(io, " :: ", v.eltype, v.scalar ? " scalar" : string(v.dims))
        println(io, all(==(1), g.devices) ? "" : string(" on device ", g.devices[id]))
    end
    for (k, r) in enumerate(g.results)
        print(io, "  result ", k, " = ")
        show_operand(io, r)
        println(io)
    end
end
