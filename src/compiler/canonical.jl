# Exact aliases in the prepared program.
#
# A chain of layout operations whose composed index map is the identity onto an
# ancestor of the same shape denotes that ancestor: a singleton axis dropped and
# restored (`dropdims(s; dims = 1)'`), a vector transposed and flattened
# (`vec(w')'`), a selection of every element (`x[:, :]`), a permutation undone.
# Tracing keeps those nodes, because Base distinguishes the spellings: the
# transpose of a transposed vector is the vector, while the transpose of a 1 × n
# matrix is n × 1. The prepared program does not need them, and they cost:
# region growth, proposals and fusion roots stop at layout operations, so an
# alias that moves nothing splits a region and materializes its operand. Here
# consumers are pointed at the ancestor and the alias is left dead.

# The nearest ancestor that `id` is an exact alias of, or `id` itself.
function exact_alias(nodes, id)
    node = nodes[id]
    is_layout(node) || return id
    chain = [id]
    while is_layout(nodes[last(chain)])
        push!(chain, only(nodes[last(chain)].inputs))
    end
    for k = 2:length(chain)
        ancestor = nodes[chain[k]]
        ancestor.dims == node.dims && ancestor.scalar == node.scalar || continue
        mapping = dense_map(ancestor.dims)
        parentdims = ancestor.dims
        for j = (k - 1):-1:1
            mapping = transform_layout(nodes[chain[j]], parentdims, mapping)
            parentdims = nodes[chain[j]].dims
        end
        linear_identity(mapping) && return exact_alias(nodes, chain[k])
    end
    id
end

"""
    collapse_aliases(program) -> program

Point every consumer of an exact alias (a layout operation whose composed index
map is the identity onto an ancestor of the same shape and kind) at that
ancestor, results included. Node IDs are unchanged; the aliases become dead.
"""
function collapse_aliases(program::ExecutionProgram)
    nodes = program.nodes
    live = live_nodes(nodes, program.results)
    alias = Dict{Int, Int}()
    for (id, node) in enumerate(nodes)
        live[id] && is_layout(node) || continue
        root = exact_alias(nodes, id)
        root == id || (alias[id] = root)
    end
    isempty(alias) && return program
    lookup(id) = get(alias, id, id)
    rewritten = map(nodes) do node
        Node(node.op, map(lookup, node.inputs), node.dims, node.dtype, node.scalar)
    end
    remap_program(program, rewritten, lookup)
end

# Reshapes sink through pointwise work to its sources.
#
# `reshape(f.(a, b), dims)` and `f.(reshape(a, dims), reshape(b, dims))` compute
# the same elements in the same linear order whenever every array operand has
# the shape of the result. The second form keeps the reshape out of the
# pointwise chain: it lands on a source (an argument, a product, a reduction),
# where the planner reads it as a free alias, or meets a reshape coming the
# other way and cancels. `dropdims(s; dims = 1)` followed by vector arithmetic
# and a transpose back, or `sum(abs2, vec(h))`, then plan as their
# shape-preserving spellings. A pointwise value with other users, or one that
# is a result, is left where it is: sinking would compute it twice, and
# duplicating a chain that backward also reads split more regions than it
# joined when it was tried. The transpose of a vector is a reshape here. Node
# IDs change; arguments keep theirs, since they come first.
is_reshape(nodes, id) =
    nodes[id].op isa ReshapeOp ||
    (nodes[id].op isa TransposeOp && length(nodes[only(nodes[id].inputs)].dims) == 1)

function sink_reshapes(program::ExecutionProgram)
    nodes = program.nodes
    live = live_nodes(nodes, program.results)
    users = zeros(Int, length(nodes))
    for (id, node) in enumerate(nodes)
        live[id] || continue
        foreach(input -> users[input] += 1, unique(node.inputs))
    end
    results = Set(program.results)
    # Whether the pointwise `node` can take a reshape: every operand is a value
    # of its shape, or a scalar that broadcasts unchanged.
    function sinkable(node)
        all(node.inputs) do input
            operand = nodes[input]
            operand.dims == node.dims || operand.scalar || isempty(operand.dims)
        end
    end
    function sinks(id)
        node = nodes[id]
        node.op isa Pointwise && !node.scalar && sinkable(node) && users[id] == 1 && id ∉ results
    end
    any(id -> live[id] && is_reshape(nodes, id) && sinks(only(nodes[id].inputs)), eachindex(nodes)) ||
        return program

    out = Node[]
    mapped = Dict{Int, Int}()
    emit!(node) = (push!(out, node); length(out))
    # A call's results are node IDs, so a renumbered call is a new call object
    # with the same origin, shared by its result nodes, which stay consecutive.
    calls = IdDict{SemanticCall, SemanticCall}()
    function renumbered(op::CallResult)
        call = op.call
        get!(calls, call) do
            ids = call_ids(call)
            base = length(out)
            SemanticCall(call.f, call.kwargs, call.specs,
                         tree_map(id -> base + findfirst(==(id), ids), call.results), call.origin)
        end |> CallResult
    end
    renumbered(op) = op
    # The new ID of `id` presented with `dims`: itself when the shape agrees, a
    # reshape of its source when it is one, the sunk pointwise operation, a
    # constant of the new shape, or an explicit reshape.
    function reshaped(id, dims, scalar)
        node = nodes[id]
        node.dims == dims && node.scalar == scalar && return mapped[id]
        is_reshape(nodes, id) && return reshaped(only(node.inputs), dims, scalar)
        if !scalar && sinks(id)
            operands = map(node.inputs) do input
                nodes[input].dims == node.dims && !nodes[input].scalar ?
                reshaped(input, dims, false) : mapped[input]
            end
            return emit!(Node(node.op, operands, dims, node.dtype, false))
        end
        if node.op isa Constant && !scalar
            return emit!(Node(node.op, (), dims, node.dtype, false))
        end
        emit!(Node(ReshapeOp(), (mapped[id],), dims, node.dtype, scalar))
    end
    for (id, node) in enumerate(nodes)
        if live[id] && is_reshape(nodes, id)
            mapped[id] = reshaped(only(node.inputs), node.dims, node.scalar)
        else
            op = renumbered(node.op)
            mapped[id] = emit!(Node(op, map(i -> mapped[i], node.inputs), node.dims, node.dtype, node.scalar))
        end
    end
    remap_program(program, out, id -> mapped[id])
end

# The program preparation plans: semantic calls resolved, reshapes sunk to
# their sources, exact aliases removed.
canonical_program(program, target, candidates, excluded, notes) =
    collapse_aliases(sink_reshapes(resolve_semantics(program, target, candidates, excluded, notes)))
