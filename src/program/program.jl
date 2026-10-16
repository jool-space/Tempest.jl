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
Part of the input. Each entry of `indices` selects along one axis of the input:
an integer drops the axis, a range or a vector of integers keeps it.
"""
struct Select{I <: Tuple} <: Operation
    indices::I
end

"""
The adjoint of a selection: an array of the shape `dims` of zeros, with each
input element added at the position it was selected from.
"""
struct Scatter{I <: Tuple} <: Operation
    indices::I
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
end

Tape() = Tape(Node[])

function record!(tape::Tape, op, inputs, dims, T; scalar = false)
    isconcretetype(T) || throw(ArgumentError("element type $T of a $(op_name(op)) is not concrete"))
    push!(tape.nodes, Node(op, Tuple(inputs), Tuple(dims), T, scalar))
    length(tape.nodes)
end

"""
    Program

A traced program: its nodes, the nodes that are its inputs and results, and
the structures they came in, as nested tuples and named tuples of leaf indices.
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
index_summary(i::AbstractVector) = "[" * string(length(i)) * " indices]"

function show_nodes(io::IO, nodes, results)
    for (id, n) in enumerate(nodes)
        inputs = join((" %$i" for i in n.inputs), ",")
        shape = n.scalar ? "" : string(n.dims)
        mark = id in results ? "  [result]" : ""
        println(io, "  %", id, " = ", op_name(n.op), op_detail(n.op), inputs, " :: ", n.eltype, shape, mark)
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::Program)
    println(io, "Program with $(length(p.nodes)) nodes")
    show_nodes(io, p.nodes, p.results)
end

# Nested inputs and outputs: tuples and named tuples of leaves.
leaves(x::Union{Tuple, NamedTuple}) = reduce(vcat, map(leaves, values(x)); init = Any[])
leaves(x) = Any[x]

function rebuild(f, structure::Union{Tuple, NamedTuple})
    rebuilt = map(s -> rebuild(f, s), values(structure))
    structure isa NamedTuple ? NamedTuple{keys(structure)}(rebuilt) : rebuilt
end
rebuild(f, leaf) = f(leaf)

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
