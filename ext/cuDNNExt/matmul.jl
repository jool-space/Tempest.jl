# Matrix products with their epilogues, proposed as regions.
#
# cuDNN fuses a matrix product with the pointwise work that consumes it and
# with a sum after it: a bias, an activation, an activation's derivative and a
# bias gradient. CuDNNKernels proposes each product with that epilogue, grown
# along the product's consumers, and offers every prefix as an alternative.
# Tempest selects among the proposals and keeps everything else, including
# rejected proposals, in its default partition.
#
# Probed with cuDNN 9.26/9.27 on GB10 (sm_121), DEFAULT_MATH and deterministic
# engines:
#
# - Float16 and BFloat16 products with bias, tanh, ReLU and GELU epilogues
#   build, with the product's value also written out, and with a transposed
#   operand. Float32 builds too, but its engines round the operands to TF32
#   without saying so in their numerical notes: their error matched cuBLAS's
#   TF32 mode. Float32 therefore needs CUDATarget(matmul_precision = :tf32).
# - A sum builds only after a product: alone or after pointwise work it has no
#   engine. With half-precision operands the sum's output must be Float32; a
#   Float16 or BFloat16 sum output has no engine.
# - Half-precision intermediate tensors build, so each epilogue value is
#   declared in its Tempest type and rounds where Tempest rounds it.
# - Every fused engine is runtime-compiled; the cuBLASLt engine serves a lone
#   product only. On single layers the fused engines beat cuBLAS plus Tempest's
#   epilogue kernels while the right operand (k × n) fit in about three
#   quarters of the 24 MiB L2, and ran at 0.4-0.9 times their speed beyond it
#   (examples/benchmarks/matmul_epilogue.jl with LAYERS). An inner dimension
#   of 8192 was also slower. `profitable` declines those products.

const MAX_EPILOGUE_MEMBERS = 32
const MAX_INNER_DIMENSION = 4096
# Off only to measure the declined shapes (matmul_epilogue.jl with LAYERS).
const PROFITABILITY = Ref(true)

## Pointwise modes
#
# From Stiletto's broadcast tracer (Stiletto src/language/broadcast.jl, commit
# 46e304c), without its comparisons and logical operations, whose Bool results
# Tempest's graph regions do not carry.

unary_mode(@nospecialize f) = nothing
unary_mode(::typeof(exp)) = cuDNN.CUDNN_POINTWISE_EXP
unary_mode(::typeof(log)) = cuDNN.CUDNN_POINTWISE_LOG
unary_mode(::typeof(sqrt)) = cuDNN.CUDNN_POINTWISE_SQRT
unary_mode(::typeof(tanh)) = cuDNN.CUDNN_POINTWISE_TANH_FWD
unary_mode(::typeof(abs)) = cuDNN.CUDNN_POINTWISE_ABS
unary_mode(::typeof(cos)) = cuDNN.CUDNN_POINTWISE_COS
unary_mode(::typeof(sin)) = cuDNN.CUDNN_POINTWISE_SIN
unary_mode(::typeof(tan)) = cuDNN.CUDNN_POINTWISE_TAN
unary_mode(::typeof(inv)) = cuDNN.CUDNN_POINTWISE_RECIPROCAL
unary_mode(::typeof(-)) = cuDNN.CUDNN_POINTWISE_NEG
unary_mode(::typeof(identity)) = cuDNN.CUDNN_POINTWISE_IDENTITY
# A conversion is an identity whose output tensor has the new type.
unary_mode(::Tempest.Cast) = cuDNN.CUDNN_POINTWISE_IDENTITY

binary_mode(@nospecialize f) = nothing
binary_mode(::typeof(+)) = cuDNN.CUDNN_POINTWISE_ADD
binary_mode(::typeof(*)) = cuDNN.CUDNN_POINTWISE_MUL
binary_mode(::typeof(-)) = cuDNN.CUDNN_POINTWISE_SUB
binary_mode(::typeof(/)) = cuDNN.CUDNN_POINTWISE_DIV
binary_mode(::typeof(max)) = cuDNN.CUDNN_POINTWISE_MAX
binary_mode(::typeof(min)) = cuDNN.CUDNN_POINTWISE_MIN

# Squares are products of a value with itself.
squares(f) = f === abs2 || f isa Tempest.LiteralPower{2}

function pointwise_supported(f, arity)
    arity == 1 && return unary_mode(f) !== nothing || squares(f)
    arity == 2 && return binary_mode(f) !== nothing
    # Wider sums and products round left to right, as Julia's do.
    f === (+) || f === (*)
end

# Julia's max and min return NaN when either operand is NaN.
nan_propagation(f) =
    f === max || f === min ? cuDNN.CUDNN_PROPAGATE_NAN : cuDNN.CUDNN_NOT_PROPAGATE_NAN

const EPILOGUE_TYPES = (HALF_TYPES..., Float32)

function epilogue_value(node)
    node.dtype in EPILOGUE_TYPES || return false
    op = node.op
    op isa Pointwise && return pointwise_supported(op.f, length(node.inputs))
    op isa FoldReduction && return op.fold isa SumFold
    false
end

## Proposals

# Float32 products run on TF32 tensor cores, which the target must permit.
product_types(target) = target.matmul_precision === :tf32 ? EPILOGUE_TYPES : HALF_TYPES

function Tempest.propose_regions(::CuDNNKernels, request::ProposalRequest)
    request.target isa CUDATarget || return RegionProposal[]
    nodes = request.program.nodes
    types = product_types(request.target)
    proposals = RegionProposal[]
    for (anchor, node) in enumerate(nodes)
        request.live[anchor] && node.op isa MatMul || continue
        node.dtype in types && all(id -> nodes[id].dtype === node.dtype, node.inputs) || continue
        append!(proposals, epilogue_proposals(request, anchor))
    end
    proposals
end

# Grow from the product along its consumers in value order, offering every
# prefix. A sum ends its branch: nothing after a sum built.
function epilogue_proposals(request, anchor)
    nodes = request.program.nodes
    # Values that depend on the product; an operand among them that stays
    # outside the region would make a cycle. Every member that depends on
    # anything in the region depends on the product.
    downstream = falses(length(nodes))
    downstream[anchor] = true
    for id in (anchor + 1):length(nodes)
        downstream[id] = any(input -> downstream[input], nodes[id].inputs)
    end
    members = Set([anchor])
    feeding = Set([anchor])
    proposals = RegionProposal[]
    while length(members) < MAX_EPILOGUE_MEMBERS
        frontier = sort!(unique!([user for id in feeding for user in request.users[id] if user ∉ members]))
        added = nothing
        for id in frontier
            added = epilogue_addition(request, members, feeding, downstream, id)
            added === nothing || break
        end
        added === nothing && break
        union!(members, added)
        nodes[first(added)].op isa FoldReduction || union!(feeding, added)
        push!(proposals, RegionProposal(members; label = epilogue_label(nodes, anchor, members)))
    end
    proposals
end

# The values `id` brings into the region: itself, then the pointwise producers
# of its other operands that nothing outside the region reads. Return nothing
# when `id` cannot join.
function epilogue_addition(request, members, feeding, downstream, id)
    nodes = request.program.nodes
    node = nodes[id]
    epilogue_value(node) || return nothing
    if node.op isa FoldReduction
        return only(node.inputs) in feeding ? [id] : nothing
    end
    any(in(feeding), node.inputs) || return nothing
    added = [id]
    function producers!(input, depth)
        (input in members || input in added || nodes[input].op isa Constant) && return true
        downstream[input] && return false
        private =
            depth > 0 &&
            nodes[input].op isa Pointwise &&
            epilogue_value(nodes[input]) &&
            input ∉ request.program.results &&
            all(user -> user in members || user in added, request.users[input])
        # Any other operand is an input of the region.
        private || return true
        push!(added, input)
        all(operand -> producers!(operand, depth - 1), nodes[input].inputs)
    end
    for input in node.inputs
        input in feeding && continue
        # A sum's value cannot feed later work.
        input in members && return nothing
        producers!(input, 4) || return nothing
    end
    added
end

function epilogue_name(node)
    op = node.op
    op isa FoldReduction && return op.axes isa Colon ? "sum" : "sum dims=$(op.axes)"
    f = op.f
    f isa Tempest.Cast ? "$(node.dtype)" : f isa Tempest.LiteralPower ? "^$(typeof(f).parameters[1])" : string(f)
end

function epilogue_label(nodes, anchor, members)
    names = [epilogue_name(nodes[id]) for id in sort!(collect(members)) if id != anchor && !(nodes[id].op isa Constant)]
    "matmul → " * join(names, ", ")
end

## Support

function proposal_operand_type(region)
    nodes = region.program.nodes
    products = filter(id -> nodes[id].op isa MatMul, region.members)
    isempty(products) && return Unsupported("cuDNN proposes regions around a matrix product")
    nodes[first(products)].dtype
end

function support(::CuDNNKernels, request::ProposedRequest)
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    region = request.region
    nodes = region.program.nodes
    T = proposal_operand_type(region)
    T isa Unsupported && return T
    if T === Float32 && !(T in product_types(request.target))
        return Unsupported(
            "cuDNN's Float32 matrix products round operands to TF32; " *
            "CUDATarget(matmul_precision = :tf32) permits it",
        )
    end
    T in EPILOGUE_TYPES || return Unsupported("cuDNN matrix products take Float16, BFloat16 or Float32, not $T")
    for id in region.members
        node = nodes[id]
        any(iszero, node.dims) && return Unsupported("cuDNN regions need nonempty arrays")
        node.op isa Union{MatMul, Constant} && continue
        epilogue_value(node) || return Unsupported("cuDNN has no pointwise mode for %$id")
        if node.op isa FoldReduction && any(user -> user in region.members, region_users(region, id))
            return Unsupported("cuDNN has no engine for work after a sum")
        end
    end
    for id in region.members
        nodes[id].op isa MatMul || continue
        verdict = profitable(nodes, id)
        verdict isa Unsupported && return verdict
    end
    all(id -> nodes[id].dtype in EPILOGUE_TYPES, region.inputs) ||
        return Unsupported("cuDNN region inputs must be Float16, BFloat16 or Float32")
    # Construction needs buffers; the engine check does not. Checking it here
    # lets a nested proposal take over without replanning. Operands that are
    # transposes are presented as transposes; construction rechecks the actual
    # presentation.
    presentations = map(region.inputs) do id
        nodes[id].op isa Tempest.TransposeOp && Tempest.native_region_input(region, id) ? :transpose : :dense
    end
    graph = region_graph(region, presentations)
    graph isa Unsupported ? graph : Supported()
end

region_users(region, id) =
    [user for (user, node) in enumerate(region.program.nodes) if id in node.inputs]

# A measured policy for the fused engines, not a capability limit.
function profitable(nodes, product)
    PROFITABILITY[] || return Supported()
    left, right = nodes[product].inputs
    inner, columns = nodes[right].dims
    inner <= MAX_INNER_DIMENSION || return Unsupported(
        "cuDNN's fused engines were slower than cuBLAS for an inner dimension above $MAX_INNER_DIMENSION",
    )
    bytes = inner * columns * sizeof(nodes[right].dtype)
    limit = 3 * CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE) ÷ 4
    bytes <= limit || return Unsupported(
        "cuDNN's fused engines were slower than cuBLAS once the right operand " *
        "($(bytes >> 20) MiB) exceeded three quarters of L2 ($(limit >> 20) MiB)",
    )
    Supported()
end

## Graphs

# Rank lifting from Stiletto's emission (Stiletto src/compiler/emission.jl,
# commit 46e304c): every tensor is declared at one rank, at least 3 as the
# matmul engines require, by trailing singleton dimensions. For column-major
# storage that is free, and it is exactly Julia's broadcasting of a bias vector.
lift_dims(dims, rank) = Int64[collect(Int64, dims); ones(Int64, rank - length(dims))]

function lift_strides(dims, strides, rank)
    s = collect(Int64, strides)
    [s; fill(Int64(max(prod(dims; init = 1), 1)), rank - length(s))]
end

input_strides(dims, presentation) =
    presentation === :transpose ? (dims[2], 1) : Tuple(dense_strides(dims))

# Half-precision sums write Float32: no engine wrote a Float16 or BFloat16 sum.
sum_output_type(node) = node.op isa FoldReduction && node.dtype in HALF_TYPES ? Float32 : node.dtype

"""
    region_graph(region, presentations) -> Graph | Unsupported

The built cuDNN graph for a proposed region whose inputs have the given
presentations (`:dense` or `:transpose`). Values are numbered in Tempest's
standard form and named by that number, so equivalent regions share one cached
execution plan. Constants are by-value scalars bound at launch, so regions that
differ only in a constant's value share it too.
"""
function region_graph(region, presentations)
    nodes = region.program.nodes
    (; local_id, operands) = standard_numbering(region)
    ordered = sort!(collect(keys(local_id)); by = id -> local_id[id])
    rank = max(3, maximum(id -> length(nodes[id].dims), ordered))
    presentation = Dict(zip(region.inputs, presentations))
    describe(id) = id in region.inputs ? (:input, nodes[id].dims, nodes[id].dtype, presentation[id]) :
        (nodes[id].op isa Constant ? :constant : nodes[id].op, nodes[id].dims, nodes[id].dtype,
         Tuple(local_id[i] for i in operands(id)), id in region.outputs)
    key = (:tempest_proposed_region, rank, map(describe, ordered)...)
    try
        cuDNN.cached_graph(key) do
            graph = emit_region(region, local_id, operands, ordered, rank, presentation)
            cuDNN.build!(graph; math_mode = CUDACore.DEFAULT_MATH, deterministic = true)
            graph
        end
    catch err
        cuDNN.graph_unsupported(err) || rethrow()
        Unsupported("cuDNN has no engine for $(region.label) with $(proposal_operand_type(region)) operands")
    end
end

value_name(local_id, id) = "v$(local_id[id])"

function emit_region(region, local_id, operands, ordered, rank, presentation)
    nodes = region.program.nodes
    T = proposal_operand_type(region)
    graph = Graph(; io_dtype = T, intermediate_dtype = Float32, compute_dtype = Float32)
    tensors = Dict{Int, cuDNN.Tensor}()
    for id in ordered
        node = nodes[id]
        name = value_name(local_id, id)
        dims = lift_dims(node.dims, rank)
        if id in region.inputs
            strides = lift_strides(node.dims, input_strides(node.dims, presentation[id]), rank)
            tensors[id] = tensor!(graph; dims, strides, dtype = node.dtype, name)
            continue
        end
        if node.op isa Constant
            tensors[id] = scalar!(graph, Float32; rank, name)
            continue
        end
        xs = [tensors[i] for i in operands(id)]
        output = id in region.outputs
        y = if output
            tensor!(graph; dims, dtype = sum_output_type(node), name, output = true)
        else
            tensor!(graph; dims, dtype = node.dtype, virtual = true, name)
        end
        emit_value!(graph, node, xs, y, name)
        tensors[id] = y
    end
    graph
end

function emit_value!(graph, node, xs, y, name)
    op = node.op
    if op isa MatMul
        cuDNN.matmul!(graph, xs[1], xs[2]; c = y, name)
    elseif op isa FoldReduction
        x = only(xs)
        dims = op.axes isa Colon ? Tuple(1:length(x.dims)) : op.axes
        cuDNN.reduction!(graph, cuDNN.CUDNN_REDUCE_TENSOR_ADD, x; y, dims, deterministic = true, name)
    elseif squares(op.f)
        cuDNN.pointwise!(graph, cuDNN.CUDNN_POINTWISE_MUL, only(xs), only(xs); y, name)
    elseif length(xs) == 1
        cuDNN.pointwise!(graph, unary_mode(op.f), only(xs); y, name)
    else
        # Fold wider sums and products left to right through intermediates of
        # the node's type, as Julia rounds them.
        x = xs[1]
        mode = binary_mode(op.f)
        for (i, operand) in enumerate(xs[2:end])
            last = i == length(xs) - 1
            target = last ? y : tensor!(graph; dims = y.dims, dtype = node.dtype, virtual = true, name = "$name.$i")
            cuDNN.pointwise!(graph, mode, x, operand; y = target, nan_propagation = nan_propagation(op.f), name = "$name.$i")
            x = target
        end
    end
    y
end

## Construction

presentation(array) = array isa Transpose ? :transpose : :dense
dense_buffer(array) = array isa Transpose ? parent(array) : array

function build_implementation(::CuDNNKernels, request::ProposedRequest, runtime, outputs, inputs)
    region = request.region
    nodes = region.program.nodes
    graph = region_graph(region, map(presentation, inputs))
    graph isa Unsupported && return graph
    (; local_id) = standard_numbering(region)
    # Engines assume their declared 16-byte alignment.
    for array in (map(dense_buffer, inputs)..., outputs...)
        UInt(pointer(array)) % 16 == 0 || return Unsupported("cuDNN needs 16-byte aligned buffers")
    end
    widened = [i for (i, id) in enumerate(region.outputs) if sum_output_type(nodes[id]) !== nodes[id].dtype]
    specs = (
        WorkspaceSpec(UInt8, graph.workspace_size),
        (WorkspaceSpec(Float32, length(outputs[i])) for i in widened)...,
    )
    T = proposal_operand_type(region)
    operands = T === Float32 ? "Float32 operands rounded to TF32" : "$(nameof(T)) operands"
    summary = "cuDNN $(region.label): $operands, Float32 accumulation" *
              (isempty(widened) ? "" : ", sums written in Float32 and rounded to $(nameof(T))") *
              "; engine workspace $(graph.workspace_size) bytes"
    ImplementationPlan(specs) do buffers
        values = Dict{String, Any}()
        for (id, array) in zip(region.inputs, inputs)
            values[value_name(local_id, id)] = dense_buffer(array)
        end
        for id in region.members
            op = nodes[id].op
            op isa Constant && (values[value_name(local_id, id)] = Float32(op.value))
        end
        after = Any[]
        for (i, (id, array)) in enumerate(zip(region.outputs, outputs))
            slot = findfirst(==(i), widened)
            if slot === nothing
                values[value_name(local_id, id)] = array
            else
                wide = workspace_array(buffers[1 + slot], size(array))
                values[value_name(local_id, id)] = wide
                push!(after, conversion(array, wide))
            end
        end
        launch = bound_launch(runtime, graph, values, buffers[1], (), Tuple(after), summary)
        BuiltImplementation(launch, (outputs, inputs, buffers), 0)
    end
end
