# The CPU provider executes the same scalar expression trees as generated GPU
# kernels. These loops are also useful for checking scheduling independently of
# the allocating graph interpreter in ir/interpret.jl.
scalar_eval(x::Literal, arrays, i) = x.value
scalar_eval(x::Position, arrays, i) = position_value(x, i)

@inline function scalar_eval(x::Read{I}, arrays, i) where {I}
    if x.dims == x.shape && linear_identity(x.mapping)
        return arrays[I][i]
    end
    logical = broadcast_index(i, x.dims, x.shape)
    arrays[I][map_index(x.mapping, logical)]
end

@inline scalar_eval(x::ScalarCall, arrays, i) = x.f(scalar_args(x.args, arrays, i)...)

@inline scalar_args(::Tuple{}, arrays, i) = ()
@inline scalar_args(args::Tuple, arrays, i) =
    (scalar_eval(first(args), arrays, i), scalar_args(Base.tail(args), arrays, i)...)

function cpu_pointwise!(out, expression, arrays, count)
    for i = 1:count
        out[i] = scalar_eval(expression, arrays, i)
    end
    nothing
end

function cpu_sum!(out, ::Type{A}, expression, arrays, count) where {A}
    total = zero(A)
    for i = 1:count
        total += scalar_eval(expression, arrays, i)
    end
    out[1] = total
    nothing
end

function cpu_axis_sum!(out, ::Type{A}, expression, arrays, layout) where {A}
    for o in eachindex(out)
        total = zero(A)
        for r = 1:layout.count
            total += scalar_eval(expression, arrays, reduction_index(layout, o, r))
        end
        out[o] = total
    end
    nothing
end

# Other folds: a left fold in index order, observed once per output.
function cpu_fold!(out, fold, init, expression, arrays, count)
    state = init
    for i = 1:count
        state = fold_combine(fold, state, fold_lift(fold, scalar_eval(expression, arrays, i)))
    end
    out[1] = fold_observe(fold, state)
    nothing
end

function cpu_axis_fold!(out, fold, init, expression, arrays, layout)
    for o in eachindex(out)
        state = init
        for r = 1:layout.count
            element = scalar_eval(expression, arrays, reduction_index(layout, o, r))
            state = fold_combine(fold, state, fold_lift(fold, element))
        end
        out[o] = fold_observe(fold, state)
    end
    nothing
end

function cpu_scatter!(out, e, arrays, count, selection)
    fill!(out, zero(eltype(out)))
    for i = 1:count
        out[map_index(selection, i)] += scalar_eval(e, arrays, i)
    end
    nothing
end

function support(::CPUKernels, request::RegionRequest)
    if !(request.target isa CPUTarget)
        return Unsupported("requires a CPU target")
    end
    if request.step.region isa
       Union{Elementwise, FullReduction, AxisReductionRegion, ScatterRegion}
        Supported()
    else
        Unsupported("CPU scalar kernels do not handle $(region_name(request.step.region))")
    end
end

function support(::LinearAlgebraGemm, request::RegionRequest)
    if !(request.target isa CPUTarget)
        return Unsupported("requires a CPU target")
    end
    request.step.region isa Union{GemmRegion, BatchedGemmRegion} ? Supported() :
    Unsupported("requires a matrix product")
end

function build_implementation(::LinearAlgebraGemm, request::RegionRequest, runtime, out, arrays)
    region = request.step.region
    launch = if region isa BatchedGemmRegion
        () -> (batched_mul!(out, arrays..., region.transpose_a, region.transpose_b); nothing)
    else
        () -> (mul!(out, arrays...); nothing)
    end
    BuiltImplementation(launch, (out, arrays), 0)
end

function build_implementation(
    ::CPUKernels,
    request::RegionRequest{<:ScalarRegion},
    runtime,
    out,
    arrays,
)
    launch = cpu_launch(request.step.region, out, arrays)
    BuiltImplementation(launch, (out, arrays), 0)
end

cpu_launch(r::Elementwise, out, arrays) =
    () -> cpu_pointwise!(out, r.expression, arrays, r.count)

cpu_launch(r::FullReduction{<:ScalarExpression, SumFold}, out, arrays) =
    () -> cpu_sum!(out, sum_accumulator(r), r.expression, arrays, r.count)

cpu_launch(r::AxisReductionRegion{<:ScalarExpression, <:AxisReduction, SumFold}, out, arrays) =
    () -> cpu_axis_sum!(out, sum_accumulator(r), r.expression, arrays, r.layout)

cpu_launch(r::FullReduction, out, arrays) =
    () -> cpu_fold!(out, r.fold, r.init, r.expression, arrays, r.count)

cpu_launch(r::AxisReductionRegion, out, arrays) =
    () -> cpu_axis_fold!(out, r.fold, r.init, r.expression, arrays, r.layout)

cpu_launch(r::ScatterRegion, out, arrays) =
    () -> cpu_scatter!(out, r.expression, arrays, r.count, r.selection)

function support(impl::CPUKernels, request::InPlaceRequest)
    request.region.step.region isa Elementwise ||
        return Unsupported("in-place execution requires elementwise work")
    support(impl, request.region)
end

# Built-in verbs on CPU, through their references.
function support(::CPUKernels, request::CallRequest)
    request.target isa CPUTarget || return Unsupported("requires a CPU target")
    request.call.f in INDEX_VERBS ||
        request.call.f in (attention_forward, attention_backward) ||
        return Unsupported("no CPU kernel for $(request.call.f)")
    Supported()
end

# The references are the CPU kernels: the index verbs loop in index order.
function build_implementation(::CPUKernels, request::CallRequest, runtime, out, inputs)
    f, kwargs = request.call.f, request.call.kwargs
    launch = () -> begin
        result = reference(f, inputs...; kwargs...)
        foreach(copyto!, tree_leaves(out), tree_leaves(result))
        nothing
    end
    BuiltImplementation(launch, (out, inputs), 0)
end
