# The program level: what the user wrote, as a typed graph with concrete shapes.
# Nodes are appended to a tape during tracing; every node's inputs precede it.

abstract type Operation end

"An input of the program."
struct Argument <: Operation end

"A constant: a number filling the node's shape, or an array of it."
struct Constant{T} <: Operation
    value::T
end

"`f` applied elementwise, with the inputs broadcast to the node's shape as Julia broadcasts."
struct Map{F} <: Operation
    f::F
end

"A fold of the input over `axes`; the reduced axes stay, with extent one."
struct Reduce{F} <: Operation
    fold::F
    axes::Tuple{Vararg{Int}}
end

"""
A product summed over shared axes, as an einsum: `a` and `b` label the axes of
the two inputs and `out` those of the result. Labels in both inputs but not in
the result are summed over.
"""
struct Contract <: Operation
    a::Tuple{Vararg{Int}}
    b::Tuple{Vararg{Int}}
    out::Tuple{Vararg{Int}}
end

"""
The one-based index of each element as `Int32`: its linear index for `axis`
zero, else its index along that axis.
"""
struct Positions <: Operation
    axis::Int
end

Positions() = Positions(0)

"The input in another shape with the same elements in the same linear order."
struct Reshape <: Operation end

"The input's axes reordered: axis `k` of the result is axis `perm[k]` of the input."
struct Permute <: Operation
    perm::Tuple{Vararg{Int}}
end

"""
Part of the input, its first input. Each entry of `indices` selects along one
axis of the input: an integer drops the axis, a range keeps it, and an
`IndexInput` replaces it by the axes of an array of integers (a gather).
"""
struct Select{I <: Tuple} <: Operation
    indices::I
end

"""
The adjoint of a selection: an array of the shape `dims` of zeros, with each
element of the first input added at the position it was selected from.
"""
struct Scatter{I <: Tuple} <: Operation
    indices::I
end

"""
An entry of a selection's or a scatter's indices: the node's `input`th input,
an array of one-based integers, whose axes take the place of the axis it
indexes. An index out of range is an error in the interpreters, and undefined
in a prepared program, as under `@inbounds`.
"""
struct IndexInput
    input::Int
end

struct Node
    op::Operation
    inputs::Tuple{Vararg{Int}}
    dims::Tuple{Vararg{Int}}
    eltype::DataType
    # A Julia number rather than an array, as `sum(x)` returns one.
    scalar::Bool
end

mutable struct Tape
    nodes::Vector{Node}
    # While calls are expanded (see `expand_calls`): whether a call of an
    # operation expands, and what each expansion recorded.
    expands::Any
    expansions::Vector{Any}
end

Tape(nodes = Node[]) = Tape(nodes, Returns(false), [])

function record!(tape::Tape, op, inputs, dims, T; scalar = false)
    isconcretetype(T) || throw(ArgumentError("element type $T of a $(op_name(op)) is not concrete"))
    push!(tape.nodes, Node(op, Tuple(inputs), Tuple(dims), T, scalar))
    length(tape.nodes)
end

"""
    Program

A traced program: its nodes, the nodes that are its inputs and results, and
the structures they came in, as nested tuples, named tuples and `Fields` of
leaf indices.
"""
struct Program
    nodes::Vector{Node}
    arguments::Vector{Int}
    results::Vector{Int}
    input_structure::Any
    output_structure::Any
end

op_name(op::Operation) = lowercase(string(nameof(typeof(op))))
op_detail(op::Operation) = ""
op_detail(op::Constant) = op.value isa AbstractArray ? " array" : " " * repr(op.value)
op_detail(op::Map) = " " * string(op.f)
op_detail(op::Reduce) = " " * string(fold_name(op.fold)) * " over " * string(op.axes)
op_detail(op::Contract) = " " * string(op.a) * "·" * string(op.b) * "→" * string(op.out)
op_detail(op::Permute) = " " * string(op.perm)
op_detail(op::Positions) = op.axis == 0 ? "" : " along axis $(op.axis)"
op_detail(op::Union{Select, Scatter}) = " " * join(map(index_summary, op.indices), ", ")

index_summary(i::Integer) = string(i)
index_summary(i::AbstractRange) = string(i)
index_summary(i::IndexInput) = "by input " * string(i.input)

function show_nodes(io::IO, nodes, results)
    for (id, n) in enumerate(nodes)
        inputs = join((" %$i" for i in n.inputs), ",")
        shape = n.scalar ? "" : string(n.dims)
        mark = id in results ? "  [result]" : ""
        println(io, "  %", id, " = ", op_name(n.op), op_detail(n.op), inputs, " :: ", n.eltype, shape, mark)
    end
end

# `n` of something, as people count it.
counted(n, word) = n == 1 ? "1 $word" : "$n $(word)s"

node_type(n::Node) = n.scalar ? string(n.eltype) : string(n.eltype, n.dims)

# The program's inputs and results by type and shape, or by number when many.
function flow(p::Program)
    side(ids, word) = isempty(ids) ? "()" : length(ids) > 3 ? counted(length(ids), word) :
        join(map(id -> node_type(p.nodes[id]), ids), ", ")
    string(side(p.arguments, "input"), " → ", side(p.results, "result"))
end

Base.show(io::IO, p::Program) = print(io, "Program(", flow(p), "; ", counted(length(p.nodes), "node"), ")")

function Base.show(io::IO, ::MIME"text/plain", p::Program)
    println(io, "Program with ", counted(length(p.nodes), "node"), ": ", flow(p))
    show_nodes(io, p.nodes, p.results)
end

# Nested inputs and outputs: tuples, named tuples and immutable structs, down to
# leaves. A program's structures record a struct as `Fields`, since the leaf
# indices they hold could not be its fields.

"A struct in a structure: its type, and its fields."
struct Fields{T}
    fields::NamedTuple
end

# Whether `x` is a struct to look inside. Mutable structs are leaves, as
# closures are: their identity is theirs, and a program does not write them.
# So are Tempest's own, such as the nodes in a call's structure.
function nested(x)
    T = typeof(x)
    !(x isa Union{AbstractArray, Number, Function, Type, Module, AbstractString, Symbol, AbstractDict, Tuple, NamedTuple}) &&
        isstructtype(T) && !ismutabletype(T) && fieldcount(T) > 0 && Base.moduleroot(parentmodule(T)) !== Base.moduleroot(@__MODULE__)
end

fields(x) = NamedTuple{fieldnames(typeof(x))}(ntuple(k -> getfield(x, k), fieldcount(typeof(x))))

# A struct of the kind of `T` with these fields, its type parameters found anew.
function construct(T, fields::NamedTuple)
    try
        Base.typename(T).wrapper(values(fields)...)
    catch err
        throw(ArgumentError("Tempest builds a $(nameof(T)) from new fields to trace through it, and its constructor " *
            "failed; fields holding arrays or numbers need types that are parameters of the struct\n" * sprint(showerror, err)))
    end
end

leaves(x::Union{Tuple, NamedTuple}) = reduce(vcat, map(leaves, values(x)); init = Any[])
leaves(x::Fields) = leaves(x.fields)
leaves(x) = nested(x) ? leaves(fields(x)) : Any[x]

# `g` on each element of a tuple or named tuple, its names kept.
each(g, x::Tuple) = map(g, x)
each(g, x::NamedTuple) = NamedTuple{keys(x)}(map(g, values(x)))

# `x` with `f` applied to each leaf, structs built anew from their fields.
rebuild(f, x::Union{Tuple, NamedTuple}) = each(y -> rebuild(f, y), x)
rebuild(f, x::Fields{T}) where {T} = construct(T, rebuild(f, x.fields))
rebuild(f, x) = nested(x) ? construct(typeof(x), rebuild(f, fields(x))) : f(x)

# The structure of `x`, `f` applied to each leaf: as `rebuild`, with structs as `Fields`.
structure_of(f, x::Union{Tuple, NamedTuple}) = each(y -> structure_of(f, y), x)
structure_of(f, x::Fields{T}) where {T} = Fields{T}(structure_of(f, x.fields))
structure_of(f, x) = nested(x) ? Fields{typeof(x)}(structure_of(f, fields(x))) : f(x)

# As `structure_of`, with each struct a named tuple of its fields: the structure
# of a gradient, which has no value for a struct's other fields.
tangent(f, x::Union{Tuple, NamedTuple}) = each(y -> tangent(f, y), x)
tangent(f, x::Fields) = tangent(f, x.fields)
tangent(f, x) = nested(x) ? tangent(f, fields(x)) : f(x)

# Each of the program's arguments among `inputs`, given in the structure it was traced with.
function argument_leaves(p, inputs)
    positions = leaves(p.input_structure)
    xs = leaves(inputs)
    length(positions) == length(xs) || throw(ArgumentError("the program takes $(length(positions)) leaves, got $(length(xs))"))
    out = Vector{Any}(nothing, length(p.arguments))
    foreach((k, x) -> k === nothing || (out[k] = x), positions, xs)
    out
end

function live(nodes, results)
    keep = falses(length(nodes))
    stack = collect(results)
    while !isempty(stack)
        id = pop!(stack)
        keep[id] && continue
        keep[id] = true
        append!(stack, nodes[id].inputs)
    end
    keep
end

"""
    prune(program) -> Program

The program without the nodes its results do not read. Arguments stay, and a
call keeps all its results when any of them is read.
"""
function prune(p::Program)
    keep = live(p.nodes, p.results)
    foreach(a -> keep[a] = true, p.arguments)
    for (id, n) in enumerate(p.nodes)
        n.op isa Call && keep[id] && foreach(r -> keep[r] = true, results_of(p.nodes, n.op.instance))
    end
    all(keep) && return p
    new = zeros(Int, length(p.nodes))
    nodes = Node[]
    for (id, n) in enumerate(p.nodes)
        keep[id] || continue
        op = n.op
        if op isa Call
            op = Call(op.f, op.kwargs, op.output, op.instance == id ? length(nodes) + 1 : new[op.instance], op.structure)
        end
        push!(nodes, Node(op, map(i -> new[i], n.inputs), n.dims, n.eltype, n.scalar))
        new[id] = length(nodes)
    end
    Program(nodes, map(i -> new[i], p.arguments), map(i -> new[i], p.results), p.input_structure, p.output_structure)
end
