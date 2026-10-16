# Prepared programs: a plan with fixed buffers and a launch per kernel.

abstract type Target end

"Run on the CPU, through the same generated floor code CUDA runs."
struct CPUTarget <: Target end

target_for(x::AbstractArray) = CPUTarget()
target_for(x) = nothing

function default_target(inputs)
    for leaf in leaves(inputs)
        t = target_for(leaf)
        t === nothing || return t
    end
    CPUTarget()
end

# Generated kernels live here.
module Generated end

"""
    Executable

A program prepared for a target: its plan, one buffer per stored value, and a
launch per kernel. Inputs are copied into owned buffers by `bind!`; `outputs`
returns the result buffers, which the next execution overwrites.
"""
struct Executable{T <: Target}
    program::Program
    plan::Plan
    target::T
    buffers::Vector{Any}
    launches::Vector{Any}
    kernels::Vector{Any}
end

storage_length(v::Value) = v.scalar || isempty(v.dims) ? 1 : prod(v.dims)

"""
    prepare(program, inputs...; target) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default.
"""
function prepare(program::Program, inputs...; target = default_target(inputs))
    p = plan(lower(program))
    graph = p.graph
    buffers = Vector{Any}(nothing, length(graph.values))
    for (id, v) in enumerate(graph.values)
        d = v.definition
        if d isa ConstantArray
            buffers[id] = upload(target, vec(collect(d.data)))
        elseif p.stored[id]
            buffers[id] = allocate(target, v.eltype, storage_length(v))
        end
    end
    built = map(g -> build(target, p, g, buffers), p.groups)
    ex = Executable(program, p, target, buffers, Any[first(b) for b in built], Any[last(b) for b in built])
    bind!(ex, inputs...)
    ex
end

"Copy `inputs` into the executable's input buffers."
function bind!(ex::Executable, inputs...)
    arguments = leaves(inputs)
    for (id, v) in enumerate(ex.plan.graph.values)
        d = v.definition
        d isa Input || continue
        x = arguments[d.argument]
        copyto!(ex.buffers[id], x isa Number ? [x] : vec(x))
    end
    ex
end

"Run every kernel in order."
function execute!(ex::Executable)
    foreach(launch -> launch(), ex.launches)
    outputs(ex)
end

"The results, in the program's output structure, as arrays over the buffers."
function outputs(ex::Executable)
    graph = ex.plan.graph
    rebuild(ex.program.output_structure) do k
        r = graph.results[k]
        reshape(ex.buffers[r.value], graph.result_scalar[k] ? () : graph.result_dims[k])
    end
end

# The CPU target.
allocate(::CPUTarget, T, n) = Vector{T}(undef, n)
upload(::CPUTarget, x) = x

kernel_slots(p, g) = sort!(collect(union(group_reads(p.graph, g), [v for v in members(g) if p.stored[v]])))

function build(::CPUTarget, p::Plan, g::Group, buffers)
    g.kind === :opaque && return opaque_launch(p, g, buffers)
    k = scalar_kernel(p.graph, p.stored, g, kernel_slots(p, g))
    f = Core.eval(Generated, :(function (buffers, n)
        for lin in 0:(n - 1)
            $(k.body)
        end
        nothing
    end))
    args = Tuple(buffers[v] for v in k.slots)
    (() -> Base.invokelatest(f, args, k.items), k)
end

# Opaque operations run their reference on dense host copies of their operands.
function opaque_launch(p, g, buffers)
    v = only(g.outputs)
    graph = p.graph
    function launch()
        arrays = Vector{Any}(undef, length(graph.values))
        for (id, b) in enumerate(buffers)
            b === nothing && continue
            val = graph.values[id]
            arrays[id] = val.scalar ? Array(b)[1] : reshape(Array(b), val.dims)
        end
        y = evaluate_value(graph.values[v].definition, graph.values[v], arrays, graph, ())
        copyto!(buffers[v], y isa Number ? [y] : vec(y))
    end
    (launch, :opaque)
end
