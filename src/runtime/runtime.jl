# Prepared programs: a plan with fixed buffers and a launch per kernel.

abstract type Target end

"Run on the CPU, through the same generated floor code CUDA runs."
struct CPUTarget <: Target end

"""
    CUDATarget(packages...; precision = :full)

Run on an NVIDIA GPU; needs CUDACore loaded. `packages` names the packages whose
implementations the target may use, as in `CUDATarget(cuTile)`; by default it
uses every implementation available when a program is prepared. The floor is
always used. `precision` is the numerics policy for products: `:full` keeps
Float32 products in Float32, `:tf32` allows TF32 tensor cores for them, and
`:tf32x3` splits each Float32 factor into two TF32 parts and sums three
tensor-core products (close to Float32 accuracy). Half-precision products
always accumulate in Float32.
"""
struct CUDATarget <: Target
    precision::Symbol
    packages::Union{Nothing, Tuple{Vararg{Module}}}
end

function CUDATarget(packages::Module...; precision::Symbol = :full)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, isempty(packages) ? nothing : packages)
end

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
An implementation realizes kernels. `claims(implementation, target, plan, group)`
returns the ways it can realize a group (empty when it cannot), and
`build_claim(implementation, claim, target, plan, group, buffers)` returns a launch and
a description. A target's floor realizes any group.
"""
abstract type Implementation end

"cuTile's generator; active when cuTile is loaded."
struct cuTileGenerator <: Implementation end

claims(::Implementation, target, plan, group) = ()
function build_claim end

"""
    claims_call(implementation, target, f, kwargs, inputs) -> Bool

Whether `implementation` realizes calls of the verb `f` with these keyword
arguments on `target`; `inputs` holds the `(dims, eltype)` of each argument.
A claimed call is one opaque group whose kernel computes all its results; an
unclaimed call expands to its reference.
"""
claims_call(::Implementation, target, f, kwargs, inputs) = false

"""
    epilogue_results(implementation, f, kwargs) -> results

The results of a claimed call over whose output tiles the implementation's
kernel applies an epilogue: pointwise work after the result, absorbed by the
planner when its reads align with the result's tile.
"""
epilogue_results(::Implementation, f, kwargs) = ()

"""
    build_call(implementation, plan, group, buffers, target) -> (launch, description)

Build the kernels of a claimed call: the group's opaque results, each written
to its buffer or, where `group.epilogues` names a value, that value computed
from the result's tile by the group's terms.
"""
function build_call end

# Implementations by the package that provides them, ordered by its name.
const REGISTERED = Pair{Module, Vector{Implementation}}[]

"""
    register!(package::Module, implementations...)

Make `implementations` available under `package`, replacing what it registered
before; extensions call this when their packages load.
"""
function register!(package::Module, implementations::Implementation...)
    filter!(p -> p.first !== package, REGISTERED)
    isempty(implementations) || push!(REGISTERED, package => Implementation[implementations...])
    sort!(REGISTERED; by = p -> string(nameof(p.first)))
    nothing
end

"The packages a target names, or nothing for every registered one."
packages(::Target) = nothing
packages(target::CUDATarget) = target.packages

"The packages whose implementations a target uses, in order."
function used_packages(target)
    named = packages(target)
    named === nothing && return Module[p.first for p in REGISTERED]
    for m in named
        any(p -> p.first === m, REGISTERED) || throw(ArgumentError("$m provides no implementations; is its extension loaded?"))
    end
    Module[p.first for p in REGISTERED if p.first in named]
end

"The implementations a target uses, in order: by package name, then as registered."
implementations(target) = Implementation[i for (m, is) in REGISTERED if m in used_packages(target) for i in is]

call_inputs(nodes, n) = map(i -> (nodes[i].dims, nodes[i].eltype), n.inputs)

function call_claimant(target, nodes, n)
    k = findfirst(i -> claims_call(i, target, n.op.f, n.op.kwargs, call_inputs(nodes, n)), implementations(target))
    k === nothing ? nothing : implementations(target)[k]
end

function takes_epilogue(target, nodes, n)
    c = call_claimant(target, nodes, n)
    c !== nothing && n.op.output in epilogue_results(c, n.op.f, n.op.kwargs)
end

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
    packages::Vector{Module}
    buffers::Vector{Any}
    launches::Vector{Any}
    kernels::Vector{Any}
    decisions::Vector{Any}
    accepted::Vector{Pair{Vector{Int}, Implementation}}
end

storage_length(v::Value) = v.scalar || isempty(v.dims) ? 1 : prod(v.dims)

"""
    prepare(program, inputs...; target, measure = true) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default. Close calls between
claims are timed when `measure` allows and the target can, and kept in the
tuning record.
"""
function prepare(program::Program, inputs...; target = default_target(inputs), measure::Bool = true)
    graph = lower(program;
        keep_call = (n, nodes) -> call_claimant(target, nodes, n) !== nothing,
        epilogue = (n, nodes) -> takes_epilogue(target, nodes, n))
    d = Decider(target, graph; measure)
    p, accepted = cover(d)
    ds = decisions(d, p)
    buffers = d.buffers
    n = length(p.graph.values)
    length(buffers) < n && append!(buffers, fill(nothing, n - length(buffers)))
    for (id, v) in enumerate(p.graph.values)
        def = v.definition
        if def isa ConstantArray
            buffers[id] === nothing && (buffers[id] = upload(target, vec(collect(def.data))))
        elseif p.stored[id]
            buffers[id] === nothing && (buffers[id] = allocate(target, v.eltype, storage_length(v)))
        else
            buffers[id] = nothing # stored only in a plan not taken
        end
    end
    built = map((g, x) -> build_group(d, p, g, x, buffers), p.groups, ds)
    ex = Executable(program, p, target, used_packages(target), buffers, Any[first(b) for b in built],
        Any[last(b) for b in built], Any[ds...], accepted)
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

function explain(io::IO, ex::Executable)
    explain(io, ex.plan)
    ex.target isa CUDATarget && println(io, "products in ", ex.target.precision === :tf32 ? "TF32 where Float32" :
        ex.target.precision === :tf32x3 ? "three TF32 products where Float32" : "full precision",
        "; half-precision products accumulate in Float32")
    for (set, i) in ex.accepted
        println(io, "candidate ", join(map(v -> "%$v", set), " "), " from ", nameof(typeof(i)), " replaces the generator's grouping")
    end
    println(io, "implementations from ", isempty(ex.packages) ? "no package" : join(map(string ∘ nameof, ex.packages), ", "),
        ", and the floor:")
    for (k, d) in enumerate(ex.kernels)
        x = ex.decisions[k]
        println(io, "  kernel ", k, ": ", d, x === nothing ? "" : string(" (", x, ")"))
    end
end

explain(ex::Executable) = explain(stdout, ex)

"The results, in the program's output structure, as arrays over the buffers."
function outputs(ex::Executable)
    graph = ex.plan.graph
    rebuild(ex.program.output_structure) do k
        k === nothing && return nothing
        r = graph.results[k]
        reshape(ex.buffers[r.value], graph.result_scalar[k] ? () : graph.result_dims[k])
    end
end

# The CPU target.
allocate(::CPUTarget, T, n) = Vector{T}(undef, n)
upload(::CPUTarget, x) = x

kernel_slots(p, g) = sort!(collect(union(group_reads(p.graph, g), [v for v in members(g) if p.stored[v]])))

"The floor's kernels for one group, in launch order."
struct FloorKernels
    kernels::Vector{ScalarKernel}
end

Base.show(io::IO, f::FloorKernels) = print(io, "floor, ", join((string(k.items, " items") for k in f.kernels), " then "))

function build_floor(target::CPUTarget, p::Plan, g::Group, buffers)
    ks = scalar_kernels(p.graph, p.stored, g, kernel_slots(p, g))
    launches = map(ks) do k
        f = Core.eval(Generated, :(function (buffers, n)
            for lin in 0:(n - 1)
                $(k.body)
            end
            nothing
        end))
        args = Tuple(buffers[v] for v in k.slots)
        () -> Base.invokelatest(f, args, k.items)
    end
    (() -> foreach(l -> l(), launches), FloorKernels(ks))
end

# A claimed call is built by its claimant; other opaque work runs through
# `opaque_launch`.
function build_opaque(target, p::Plan, g::Group, buffers)
    n = p.graph.values[first(g.outputs)].definition.node
    n.op isa Call || return opaque_launch(p, g, buffers)
    # The claimant is found again from the call itself.
    d = p.graph.values[first(g.outputs)].definition
    inputs = map((o, (dims, _)) -> (dims, p.graph.values[o.value].eltype), d.operands, d.shapes)
    k = findfirst(i -> claims_call(i, target, n.op.f, n.op.kwargs, inputs), implementations(target))
    k === nothing && error("no implementation builds the claimed call of $(n.op.f)")
    build_call(implementations(target)[k], p, g, buffers, target)
end

# Opaque work and regions run their references, and epilogues, on dense host
# copies.
function opaque_launch(p, g, buffers)
    graph = p.graph
    function launch()
        arrays = Vector{Any}(undef, length(graph.values))
        for (id, b) in enumerate(buffers)
            b === nothing && continue
            val = graph.values[id]
            arrays[id] = val.scalar ? Array(b)[1] : reshape(Array(b), val.dims)
        end
        run_group!(g, arrays, graph)
        for v in g.outputs
            p.stored[v] || continue
            y = arrays[v]
            copyto!(buffers[v], y isa Number ? [y] : vec(y))
        end
    end
    (launch, :opaque)
end
