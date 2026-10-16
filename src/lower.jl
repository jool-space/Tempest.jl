# From the program level (Tempest's traced graph) to the index level. Layout
# operations become formulas; calls are replaced by their references first.

# How a program-level node is read: a value through a formula over the node's
# own domain, a constant, or an integer computed from the index (positions).
abstract type Access end

struct ValueAccess <: Access
    value::Int
    index::Tuple{Vararg{Affine}}
end

struct LiteralAccess{T} <: Access
    value::T
end

struct IndexAccess <: Access
    expr::Affine
    eltype::DataType
end

identity_index(dims) = ntuple(var, length(dims))

# Read `access`, a node of shape `dims`, from a reader whose domain is `reader`,
# with Julia's broadcasting: axes of extent one, and missing trailing axes,
# are read at index zero.
function operand(access::ValueAccess, dims, reader::Domain)
    arguments = ntuple(k -> dims[k] == 1 ? constant(0) : var(k), length(dims))
    Read(access.value, map(e -> substitute(e, arguments, reader), access.index))
end

operand(access::LiteralAccess, dims, ::Domain) = Literal(access.value)

function operand(access::IndexAccess, dims, reader::Domain)
    arguments = ntuple(k -> dims[k] == 1 ? constant(0) : var(k), length(dims))
    IndexOperand(substitute(access.expr, arguments, reader), access.eltype)
end

# The program-level index of a layout node's input, for each index of the node,
# as expressions over the node's domain.
function linear_index(dims, d::Domain)
    linear = constant(0)
    stride = 1
    for (k, n) in enumerate(dims)
        linear = add(linear, scale(var(k), stride), d)
        stride *= n
    end
    linear
end

function delinearize(linear, dims, d::Domain)
    stride = 1
    map(dims) do n
        e = fmod(fdiv(linear, stride, d), n, d)
        stride *= n
        e
    end
end

layout_index(::Tempest.ReshapeOp, dims, parentdims, d) =
    delinearize(linear_index(dims, d), parentdims, d)

function layout_index(::Tempest.TransposeOp, dims, parentdims, d)
    length(parentdims) == 1 && return (var(2),)
    (var(2), var(1))
end

function layout_index(op::Tempest.PermuteOp, dims, parentdims, d)
    index = Vector{Affine}(undef, length(parentdims))
    for (k, p) in enumerate(op.permutation)
        index[p] = var(k)
    end
    Tuple(index)
end

function layout_index(op::Union{Tempest.ViewOp, Tempest.SliceOp}, dims, parentdims, d, gather)
    s = op.selection
    axis = 0
    map(s.indices) do idx
        if idx isa Integer
            constant(idx - 1)
        elseif idx isa Tempest.IndexList
            axis += 1
            list = idx.indices
            lo, hi = extrema(list)
            add(normalize(0, (Indirect(gather(list), (var(axis),), lo, hi) => 1,), d), -1, d)
        else
            axis += 1
            add(scale(var(axis), step(idx)), first(idx) - 1, d)
        end
    end
end

compose(access::ValueAccess, index, d) =
    ValueAccess(access.value, map(e -> substitute(e, index, d), access.index))
compose(access::LiteralAccess, index, d) = access
compose(access::IndexAccess, index, d) = IndexAccess(substitute(access.expr, index, d), access.eltype)

"""
    lower(program) -> IndexGraph

Lower a traced, possibly differentiated, Tempest program to the index level.
Calls are first replaced by their references.
"""
function lower(program::Tempest.ExecutionProgram)
    program = Tempest.resolve_semantics(program, Tempest.CPUTarget(), (), IdDict{Any, Vector{String}}(), String[])
    nodes = program.nodes
    live = Tempest.live_nodes(nodes, program.results)
    values = Value[]
    access = Vector{Access}(undef, length(nodes))
    arguments = Int[]
    gathers = IdDict{Any, Int}()
    function gather(list)
        get!(gathers, list) do
            push!(values, Value(ConstantArray(collect(list)), (length(list),), eltype(list), false))
            length(values)
        end
    end
    function push_value!(definition, node)
        push!(values, Value(definition, node.dims, node.dtype, node.scalar))
        ValueAccess(length(values), identity_index(node.dims))
    end
    argument_ids = program.primal.arguments
    for (id, node) in enumerate(nodes)
        if node.op isa Tempest.Argument
            k = findfirst(==(id), argument_ids)
            access[id] = push_value!(Input(k), node)
            push!(arguments, length(values))
            continue
        end
        live[id] || continue
        d = Domain(node.dims)
        op = node.op
        inputs = map(i -> access[i], node.inputs)
        input_dims = map(i -> nodes[i].dims, node.inputs)
        access[id] = if op isa Tempest.Constant
            LiteralAccess(op.value)
        elseif op isa Tempest.Positions
            IndexAccess(add(linear_index(node.dims, d), 1, d), Int32)
        elseif op isa Tempest.Pointwise
            push_value!(Map(op.f, map((a, dims) -> operand(a, dims, d), inputs, input_dims)), node)
        elseif op isa Tempest.FoldReduction
            parent = only(input_dims)
            reduced = op.axes isa Colon ? Tuple(1:length(parent)) : op.axes
            iteration = Domain(parent)
            push_value!(Fold(op.fold, operand(only(inputs), parent, iteration), parent, reduced), node)
        elseif op isa Union{Tempest.ViewOp, Tempest.SliceOp}
            compose(only(inputs), layout_index(op, node.dims, only(input_dims), d, gather), d)
        elseif op isa Tempest.LayoutOperation
            compose(only(inputs), layout_index(op, node.dims, only(input_dims), d), d)
        else
            # Products, scatters and anything not modelled yet: operands are read
            # densely in their program shapes.
            reads = map(inputs, input_dims) do a, dims
                a isa ValueAccess || error("opaque operation $(op) on a constant or positions operand")
                operand(a, dims, Domain(dims))
            end
            shapes = map(i -> (nodes[i].dims, nodes[i].scalar), node.inputs)
            push_value!(Opaque(node, reads, shapes), node)
        end
    end
    results = Operand[]
    for r in program.results
        node = nodes[r]
        push!(results, operand(access[r], node.dims, Domain(node.dims)))
    end
    IndexGraph(
        values,
        arguments,
        results,
        [nodes[r].dims for r in program.results],
        [nodes[r].scalar for r in program.results],
        [nodes[r].dtype for r in program.results],
    )
end

lower(program::Tempest.TensorProgram) = lower(Tempest.ForwardProgram(program))
