# Research only: export a bounded Tempest graph to NVIDIA TensorIR v0.1.4.
# No Python tracing, backend registration, compilation, or execution happens here.
# Target: https://github.com/NVIDIA/tensor-ir/tree/v0.1.4
# julia --project=. examples/research/tensor_ir_export.jl [directory | --test]
# tensor_ir-opt -layout-propagation-pipeline exported.mlir -o lowered.mlir
# Lowering success does not establish that TileIR assembles or executes.
module TensorIRExport

using Tempest

const D = Tempest

"""A graph outside this exporter's bounded semantic contract, not a compiler failure."""
struct UnsupportedExport <: Exception
    reason::String
end
Base.showerror(io::IO, error::UnsupportedExport) =
    print(io, "TensorIR research export: ", error.reason)
reject(message) = throw(UnsupportedExport(message))

# Experiment-owned operations can opt in without making this exporter depend on
# their tracing modules. The fixture module installs its maximum rule on demand.
reduction_description(op) = nothing
reduction_description(op::D.SumReduction) = (mode = "add", axes = op.axes)

tensor_type(dims) = "tensor<$(join(dims, "x"))xf32>"
value_name(id) = "%v$id"

function tensor_spec(dims)
    strides = ntuple(i -> prod(dims[1:(i - 1)]), length(dims))
    "$(tensor_type(dims)) {nv_tensor_ir.stride = \"($(join(strides, ",")))\"}"
end

function literal(value)
    valid = value isa Float32 ||
            (value isa Integer && isfinite(Float32(value)) && Float32(value) == value)
    valid || reject("only Float32 literals and exactly representable integer literals are accepted")
    isfinite(value) || reject("nonfinite literals are outside this experiment")
    # Float64's printed decimal exactly identifies the original Float32 value.
    repr(Float64(value))
end

function check_node(node, id)
    if node.op isa D.Constant
        node.scalar || reject("array constants at %$id are unsupported")
        literal(node.op.value)
        return
    end
    node.dtype === Float32 || reject("%$id uses mixed precision or a non-Float32 type")
    !node.scalar && !isempty(node.dims) || reject("runtime scalars and rank-zero arrays are unsupported")
    all(>(0), node.dims) || reject("empty dimensions are unsupported")
    node.op isa Union{D.Argument, D.Pointwise, D.ReshapeOp} ||
        reduction_description(node.op) !== nothing ||
        reject("$(typeof(node.op)) at %$id requires an unsupported operation or layout mapping")
end

# TensorIR broadcasts equal-rank tensors. Julia aligns leading axes, so a vector
# must become (H, 1), never (1, H). Cache each presentation without duplicating
# the producer: a value's identity and a consumer's broadcast are distinct.
function broadcast_operand!(io, presentations, nodes, id, dims)
    node = nodes[id]
    node.dims == dims && !node.scalar && return value_name(id)
    key = (id, dims)
    haskey(presentations, key) && return presentations[key]
    name = "%b$(length(presentations) + 1)"
    source = value_name(id)
    if node.op isa D.Constant
        println(io, "    $name = nv_tensor_ir.splat $source : $(tensor_type(dims))")
    else
        length(node.dims) <= length(dims) || reject("broadcast would remove an axis")
        padded = (node.dims..., ntuple(_ -> 1, length(dims) - length(node.dims))...)
        all(a == 1 || a == b for (a, b) in zip(padded, dims)) ||
            reject("incompatible broadcast from $(node.dims) to $dims")
        if padded != node.dims
            shaped = "$(name)_shape"
            println(io, "    $shaped = reshape $source : $(tensor_type(node.dims)) -> $(tensor_type(padded))")
            source = shaped
        end
        if padded == dims
            presentations[key] = source
            return source
        end
        println(io, "    $name = broadcast $source : $(tensor_type(padded)) -> $(tensor_type(dims))")
    end
    presentations[key] = name
end

function emit_pointwise!(io, output, f, operands, dims)
    type = tensor_type(dims)
    arity = length(operands)
    if f === (+) || f === (*)
        arity >= 2 || reject("$f needs at least two operands")
        operation = f === (+) ? "add" : "mul"
        left = first(operands)
        for i in 2:arity
            result = i == arity ? output : "$(output)_fold$i"
            println(io, "    $result = $operation $left, $(operands[i]) : $type")
            left = result
        end
        return
    elseif f === abs2 || f isa Union{D.LiteralPower{2}, D.LiteralPower{3}}
        arity == 1 || reject("literal power requires one operand")
        input = only(operands)
        cube = f isa D.LiteralPower{3}
        square = cube ? "$(output)_square" : output
        println(io, "    $square = mul $input, $input : $type")
        cube && println(io, "    $output = mul $square, $input : $type")
        return
    elseif f isa D.Cast{Float32} || f === identity
        arity == 1 || reject("identity conversion requires one operand")
        # All operands have already passed Float32 validation. A same-shape
        # reshape is an identity and folds away in TensorIR canonicalization.
        println(io, "    $output = reshape $(only(operands)) : $type -> $type")
        return
    end
    binary = ((-, "sub"), (/, "div"))
    unary = ((exp, "exp"), (sqrt, "sqrt"), (inv, "reciprocal"),
             (tanh, "tanh_fwd"), (sin, "sin"))
    if f === (-) && arity == 1
        println(io, "    $output = neg $(only(operands)) : $type")
        return
    end
    for (function_value, operation) in (binary..., unary...)
        f === function_value || continue
        expected = any(pair -> pair[1] === f, binary) ? 2 : 1
        arity == expected || reject("$f with $arity operands is unsupported")
        println(io, "    $output = $operation $(join(operands, ", ")) : $type")
        return
    end
    reject("scalar function $f has no TensorIR mapping in this experiment")
end

function emit_reshape!(io, output, source, input_dims, output_dims)
    # TensorIR reshape traverses the LAST dimension fastest; Julia traverses
    # the FIRST. Removing/inserting only singleton axes agrees in both orders,
    # including (H, 1) -> (H,) for a reduced weight gradient. General reshapes
    # need an explicit index transformation, even with column-major I/O strides.
    filter(!=(1), input_dims) == filter(!=(1), output_dims) ||
        reject("reshape may only insert or remove singleton axes")
    println(io, "    $output = reshape $source : $(tensor_type(input_dims)) -> $(tensor_type(output_dims))")
end

"""
    export_mlir(program; name = "tempest_region")

Export dense positive-rank Float32 array graphs, built-in pointwise functions,
sum, and explicitly registered experimental reductions to NVIDIA TensorIR MLIR.
Leading-axis Julia broadcasts become explicit reshape/broadcast operations. Inputs and flattened
output leaves have column-major strides; the output tree remains in a comment.
TensorProgram records shapes, not concrete input strides: callers must bind dense
column-major buffers. Reshape may only insert/remove singleton axes. Other layouts,
opaque calls, scalar outputs, repeated output ids, arbitrary functions, and mixed
precision raise `UnsupportedExport`. Integer literals require exact conversion.
This proves neither target compiler support nor numerical/runtime correctness.
"""
function export_mlir(program::D.TensorProgram; name = "tempest_region")
    occursin(r"^[A-Za-z_][A-Za-z_0-9]*$", name) || reject("invalid graph name")
    nodes = program.nodes
    results = Int[D.tree_leaves(program.output)...]
    length(unique(results)) == length(results) || reject("repeated output ids require an explicit output alias map")
    active = D.live_nodes(nodes, results)
    for (id, node) in enumerate(nodes)
        (active[id] || id in program.arguments) && check_node(node, id)
    end
    any(id -> nodes[id].scalar, results) && reject("scalar outputs are unsupported")
    io = IOBuffer()
    println(io, "// Output tree (Tempest value ids): ", repr(program.output))
    println(io, "module {")
    inputs = ["$(value_name(id)): $(tensor_spec(nodes[id].dims))" for id in program.arguments]
    outputs = [tensor_spec(nodes[id].dims) for id in results]
    println(io, "  nv_tensor_ir.graph @$name(\n    ", join(inputs, ",\n    "))
    println(io, "  ) -> (", join(outputs, ", "), ") {")
    presentations = Dict{Tuple{Int, Tuple}, String}()
    for (id, node) in enumerate(nodes)
        active[id] || continue
        op = node.op
        output = value_name(id)
        if op isa D.Argument
            continue
        elseif op isa D.Constant
            println(io, "    $output = nv_tensor_ir.constant $(literal(op.value)) : f32")
        elseif op isa D.Pointwise
            operands = [broadcast_operand!(io, presentations, nodes, i, node.dims) for i in node.inputs]
            emit_pointwise!(io, output, op.f, operands, node.dims)
        elseif op isa D.ReshapeOp
            source = only(node.inputs)
            emit_reshape!(io, output, value_name(source), nodes[source].dims, node.dims)
        else
            source = only(node.inputs)
            reduction = reduction_description(op)
            axes = reduction.axes isa Colon ? Tuple(1:length(nodes[source].dims)) : reduction.axes
            node.dims == ntuple(i -> i in axes ? 1 : nodes[source].dims[i], length(nodes[source].dims)) ||
                reject("reduction must retain singleton axes")
            mode = reduction.mode
            dimensions = join((axis - 1 for axis in axes), ", ")
            println(io, "    $output = reduce($(value_name(source))) ",
                    "<dimensions = [$dimensions], reduction_mode = <$mode>>")
            println(io, "      : $(tensor_type(nodes[source].dims)) -> $(tensor_type(node.dims))")
        end
    end
    println(io, "    results ", join(value_name.(results), ", "), " : ",
            join((tensor_type(nodes[id].dims) for id in results), ", "))
    println(io, "  }\n}")
    String(take!(io))
end

# Loading the exporter alone imports neither Test nor ReductionScopes. These
# convenience entry points preserve the original experiment/CLI interface.
function fixtures()
    owner = parentmodule(@__MODULE__)
    if !isdefined(owner, :TensorIRExportTests)
        Base.include(owner, joinpath(@__DIR__, "tensor_ir_export_tests.jl"))
    end
    Base.invokelatest(getfield, owner, :TensorIRExportTests)
end
fixture_function(name) = Base.invokelatest(getfield, fixtures(), name)
examples(; kwargs...) = Base.invokelatest(fixture_function(:examples); kwargs...)
check_exporter() = Base.invokelatest(fixture_function(:check_exporter))

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        print(Base.invokelatest(export_mlir, examples().softmax; name = "softmax"))
    elseif only(ARGS) == "--test"
        check_exporter()
    else
        directory = only(ARGS)
        mkpath(directory)
        for (name, program) in pairs(examples())
            source = Base.invokelatest(export_mlir, program; name = string(name))
            write(joinpath(directory, "$name.mlir"), source)
        end
    end
end

end # module
