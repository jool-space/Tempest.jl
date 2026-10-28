# The program level: what the user wrote, as a typed graph with concrete shapes.
# Nodes are appended to a tape during tracing; every node's inputs precede it.

abstract type Operation end

"An input of the program."
struct Argument <: Operation end

"A stand-in for a recurrence's state or slice while its step is traced; no program keeps one (see `recurrence`)."
struct Placeholder <: Operation end

"A constant: a number filling the node's shape, an array of it, or a range of integers."
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
A fold of the input along `axes` that keeps them, as Base's `accumulate`:
each element is the fold of the elements up to it in the axes' linear order,
the first fastest; from the last down with `reverse`, and without the element
itself with `exclusive`.
"""
struct Accumulate{F} <: Operation
    fold::F
    axes::Tuple{Vararg{Int}}
    reverse::Bool
    exclusive::Bool
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

"The input in another shape with the same elements in the same linear order."
struct Reshape <: Operation end

"The input's axes reordered: axis `k` of the result is axis `perm[k]` of the input."
struct Permute <: Operation
    perm::Tuple{Vararg{Int}}
end

"""
Part of the input, its first input. Each entry of `indices` selects along one
axis of the input: an integer drops the axis, a range keeps it, and an
`IndexInput` replaces it by the axes of an array of integers (a gather). The
indices are data, not a type parameter: every method on a selection would be
compiled once per shape of selection a program makes.
"""
struct Select <: Operation
    indices::Tuple
end

"""
The adjoint of a selection: an array of the shape `dims` of zeros, with each
element of the first input added at the position it was selected from, the
elements at one position summed as a sum fold sums them.
"""
struct Scatter <: Operation
    indices::Tuple
end

"""
A write: the first input with the elements of the second at the positions
`indices` select, a selection's indices whose index arrays are inputs from the
third on. Where several elements land at one position, one of them does,
unspecified which; the interpreter places them in order, the last landing,
as Base does.
"""
struct Update <: Operation
    indices::Tuple
end

"""
An entry of a selection's, a scatter's or a write's indices: the node's
`input`th input, an array of one-based integers, whose axes take the place of
the axis it indexes. An index out of range is an error in the interpreters,
and undefined in a prepared program, as under `@inbounds`.
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
    # The nodes a reverse pass appended: the backward of the program.
    backward::BitSet
    # Of each recurrence's step being traced, the last node before it: a step
    # writes no array made before it (see `recurrence`).
    steps::Vector{Int}
end

Tape() = Tape(Node[], BitSet(), Int[])
Tape(nodes::Vector{Node}) = Tape(nodes, BitSet(), Int[])
Tape(nodes::Vector{Node}, backward::BitSet) = Tape(nodes, backward, Int[])

function record!(tape::Tape, @nospecialize(op), @nospecialize(inputs), @nospecialize(dims), @nospecialize(T); scalar = false)
    isconcretetype(T) || throw(ArgumentError("element type $T of a $(op_name(op)) is not concrete"))
    push!(tape.nodes, Node(op, Tuple(inputs), Tuple(dims), T, scalar))
    length(tape.nodes)
end

"""
    Program

A traced program: its nodes, the nodes that are its inputs and results, the
structures they came in, as nested tuples, named tuples and `Fields` of leaf
indices, and the nodes that are its backward, where a reverse pass recorded
them.
"""
struct Program
    nodes::Vector{Node}
    arguments::Vector{Int}
    results::Vector{Int}
    input_structure::Any
    output_structure::Any
    backward::BitSet
end

Program(nodes, arguments, results, input_structure, output_structure) =
    Program(nodes, arguments, results, input_structure, output_structure, BitSet())

"""
A recurrence's step: a program whose arguments are the state's arrays, a slice
of each scanned array and the values the step captures, in that order, as the
inputs of the recurrence's nodes are; `steps` steps, the state's `k`th array
carried by the step's `carried[k]`th result, the `k`th scanned array sliced
along its axis `sliced[k]`.
"""
struct LoopBody
    program::Program
    steps::Int
    carried::Vector{Int}
    sliced::Vector{Int}
end

"""
A recurrence, as Base's `accumulate` over slices of arrays runs one: `body`
run once per step, each step's result the next step's state. The node is the
body's `result`th result at every step, stacked along a last axis, so a step's
result is its slice there (see `LoopBody`).
"""
struct Loop <: Operation
    body::LoopBody
    result::Int
end

op_name(op::Operation) = lowercase(string(nameof(typeof(op))))
op_detail(op::Operation) = ""
op_detail(op::Constant) = op.value isa AbstractRange ? " " * repr(op.value) : op.value isa AbstractArray ? " array" : " " * repr(op.value)
op_detail(op::Map) = " " * string(op.f)
op_detail(op::Reduce) = " " * string(fold_name(op.fold)) * " over " * string(op.axes)
op_detail(op::Accumulate) = " " * scan_summary(op.fold, op.axes, op.reverse, op.exclusive)
op_detail(op::Loop) = string(" of ", op.body.steps, " steps, result ", op.result)

# A scan in words: its fold, its axes, and its direction where it is not the plain one.
scan_summary(fold, axes, reverse, exclusive) =
    string(fold_name(fold), " along ", axes, reverse ? ", reversed" : "", exclusive ? ", exclusive" : "")
op_detail(op::Contract) = " " * string(op.a) * "·" * string(op.b) * "→" * string(op.out)
op_detail(op::Permute) = " " * string(op.perm)
op_detail(op::Union{Select, Scatter, Update}) = " " * join(map(index_summary, op.indices), ", ")

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
# So are Tempest's own.
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

The program without the nodes its results do not read. Arguments stay.
"""
function prune(p::Program)
    keep = live(p.nodes, p.results)
    foreach(a -> keep[a] = true, p.arguments)
    all(keep) && return p
    new = zeros(Int, length(p.nodes))
    nodes = Node[]
    for (id, n) in enumerate(p.nodes)
        keep[id] || continue
        push!(nodes, Node(n.op, map(i -> new[i], n.inputs), n.dims, n.eltype, n.scalar))
        new[id] = length(nodes)
    end
    Program(nodes, map(i -> new[i], p.arguments), map(i -> new[i], p.results), p.input_structure, p.output_structure,
        BitSet(new[i] for i in p.backward if keep[i]))
end
