# Prepared programs: a plan with fixed buffers and a launch per kernel.

public CPUTarget, CUDATarget

abstract type Target end

"Run on the CPU, through the same generated floor code CUDA runs."
struct CPUTarget <: Target end

Base.show(io::IO, ::CPUTarget) = print(io, "CPUTarget()")

"""
    CUDATarget(packages...; precision = :full, online = true)

Run on an NVIDIA GPU; needs CUDACore loaded. `packages` names the packages whose
implementations the target may use, as in `CUDATarget(cuTile)`; by default it
uses every implementation available when a program is prepared. The floor is
always used. `precision` is the numerics policy for products: `:full` keeps
Float32 products in Float32, `:tf32` allows TF32 tensor cores for them, and
`:tf32x3` splits each Float32 factor into two TF32 parts and sums three
tensor-core products (close to Float32 accuracy). Half-precision products
always accumulate in Float32. `online` permits the online softmax (see
`rewrite`): a sum weighted by a softmax accumulates in one pass with the
softmax's maximum, rescaled as it grows, which changes rounding.
"""
struct CUDATarget <: Target
    precision::Symbol
    online::Bool
    packages::Union{Nothing, Tuple{Vararg{Module}}}
end

function CUDATarget(packages::Module...; precision::Symbol = :full, online::Bool = true)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, online, isempty(packages) ? nothing : packages)
end

Base.show(io::IO, t::CUDATarget) = print(io, "CUDATarget(", t.packages === nothing ? "" : join(map(string ∘ nameof, t.packages), ", ") * "; ",
    "precision = :", t.precision, t.online ? "" : ", online = false", ")")

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

"cuTile's generator; active when cuTile is loaded."
struct cuTileGenerator <: Implementation end

in_place(::cuTileGenerator, claim) = true

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

"""
    Executable

A program prepared for a target: its plan, its storage, and a launch per
kernel. Each stored value has a buffer: inputs, constants and results their
own, other values a view of their slot, and a result that replaces an argument
the argument's. `bind!` copies inputs into the buffers the executable owns,
`execute!` and `replay!` run the program over them, and `outputs` is where
the results are, overwritten by each run.
"""
mutable struct Executable{T <: Target}
    program::Program
    plan::Plan
    target::T
    packages::Vector{Module}
    storage::Storage
    slots::Vector{Any}
    buffers::Vector{Any}
    # Per argument of the program: its buffer, and the caller's array when it
    # is bound by reference.
    arguments::Vector{Any}
    referenced::Vector{Any}
    # Per result: an array over its buffer, or over the argument's it replaces.
    results::Vector{Any}
    # The results in the program's output structure, as `outputs` gives them.
    outputs::Any
    launches::Vector{Any}
    kernels::Vector{Any}
    decisions::Vector{Any}
    considered::Vector{Any}
    # Copies of staged results into the arguments they replace, after the kernels.
    copies::Vector{Any}
    # The graph `execute!` replays, once captured (see `capture!`).
    captured::Any
end


"""
    prepare(program, inputs...; target, measure = :close, replaces = (), by_reference = (), recompute = ()) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default. Close calls between
claims are timed when `measure` allows and the target can, and kept in the
tuning record. `measure = :none` times no claim but still uses the times in
the record; `:close`, the default, times the options whose estimates are
within the close-call factor of the best; `:all` times every option of every
group and both sides of every comparison. `true` and `false` mean `:close` and
`:none`.

`replaces` declares state, as pairs `result => argument` of positions among the
program's results and its arguments, alike in structure, shapes and types:
`(2 => 1, 3 => 2)` for a step `(params, state, batch) -> (loss, params, state)`.
After each execution the argument holds the result, as if bound again, so the
next execution reads it where it is. A result is written into the argument's
storage when the kernel that writes it reads the argument only at the address
it writes, and the kernels that read the argument otherwise can run first;
else it is written apart and copied there after the last kernel. `explain`
says which.

Inputs are copied into storage the executable owns, except the arrays at the
positions in `by_reference`, which must be of the target's array type: they
are the executable's storage. The caller keeps them alive and may write them
between executions, and a result that replaces one writes into it.

`recompute` pins folds to be recomputed where they are read rather than
stored, by their numbers among the values of the program's index graph, as
`explain` prints them: what the cover would otherwise decide by cost (see
`Tempest.Compiler.recomputable_folds`).

A training loop, where the step returns the loss and the new parameters and
optimizer state:

```julia
ex = prepare(trace(step, params, state, batch), params, state, batch; replaces = (2 => 1, 3 => 2))
capture!(ex)
loss = first(outputs(ex))       # where each step leaves the loss
for batch in batches
    bind!(ex, 3 => batch)       # only the batch is new
    replay!(ex)                 # parameters and state are updated where they are
    @show loss[]
end
```

For a function called on new arguments each time, `compile` does the binding
and copying.
"""
function prepare(program::Program, @nospecialize(inputs...); target = default_target(inputs), measure = :close,
        replaces = (), by_reference = (), recompute = ())
    graph = lower(program)
    permits_rewrites(target) && (graph = rewrite(graph))
    d = Decider(target, graph; measure, recompute)
    p, considered = cover(d)
    ds = decisions(d, p)
    pairs = replacements(program, replaces)
    values = p.graph.values
    input = Dict(v.definition.argument => id for (id, v) in enumerate(values) if v.definition isa Input)
    result_value(k) = p.graph.results[k].value
    # A result that is an argument's storage would change as another result replaces it.
    for k in eachindex(p.graph.results), (r, a) in pairs
        k != r && result_value(k) == input[a] && throw(ArgumentError("result $k is argument $a, which result $r replaces"))
    end
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || in_place(o.implementation, o.claim), x.options)
    p, order, s = storage(p, [result_value(k) => input[a] for (k, a) in pairs]; in_place = k -> writes_in_place(ds[k]))
    ds = ds[order]
    slots = Any[allocate(target, UInt8, bytes) for bytes in s.sizes]
    referenced = Vector{Any}(nothing, length(program.arguments))
    xs = argument_leaves(program, inputs)
    for j in by_reference, k in leaves(program.input_structure[j])
        k === nothing || xs[k] isa Number || (referenced[k] = xs[k])
    end
    buffers = Vector{Any}(nothing, length(values))
    for (id, v) in enumerate(values)
        def = v.definition
        if def isa ConstantArray
            # Uploaded already if a claim was measured.
            buffers[id] = get(d.buffers, id, nothing)
            buffers[id] === nothing && (buffers[id] = upload(target, vec(collect(def.data))))
        elseif def isa Input && referenced[def.argument] !== nothing
            x = referenced[def.argument]
            bindable(target, x) && eltype(x) === v.eltype && length(x) == storage_length(v) ||
                throw(ArgumentError("input $(def.argument) is bound by reference, but $(typeof(x)) is not storage on $target"))
            buffers[id] = vec(x)
        elseif s.slots[id] > 0
            buffers[id] = slot_view(slots[s.slots[id]], v.eltype, storage_length(v))
        elseif p.stored[id]
            buffers[id] = allocate(target, v.eltype, storage_length(v))
        end
    end
    foreach(((r, a),) -> buffers[r] = buffers[a], s.donated)
    arguments = Any[get(buffers, get(input, k, 0), nothing) for k in eachindex(program.arguments)]
    replaced = Dict(pairs)
    results = Any[reshape(haskey(replaced, k) ? arguments[replaced[k]] : buffers[result_value(k)],
        p.graph.result_scalar[k] ? () : p.graph.result_dims[k]) for k in eachindex(p.graph.results)]
    built = concurrently(((g, x),) -> build_group(d, p, g, x, buffers), collect(zip(p.groups, ds)), target)
    copies = Any[(() -> copyto!(buffers[a], buffers[r])) for (r, a) in s.staged]
    outs = rebuild(k -> k === nothing ? nothing : results[k], program.output_structure)
    ex = Executable(program, p, target, used_packages(target), s, slots, buffers, arguments, referenced, results, outs,
        Any[first(b) for b in built], Any[last(b) for b in built], Any[ds...], Any[considered...], copies, nothing)
    bind!(ex, inputs...)
    ex
end

"A value of `n` elements of type `T` in the first bytes of a slot."
slot_view(slot, T, n) = reinterpret(T, view(slot, 1:(n * sizeof(T))))

"Whether array `x` can be storage on `target`, bound by reference."
bindable(::Target, x) = false
bindable(::CPUTarget, x) = x isa Array

# The results that replace arguments: pairs of a result's position among the
# program's results and the argument's among its arguments, leaf by leaf.
function replacements(program, replaces)
    pairs = Pair{Int, Int}[]
    for (k, j) in (replaces isa Pair ? (replaces,) : replaces)
        out, in = program.output_structure[k], program.input_structure[j]
        structure_of(Returns(nothing), out) == structure_of(Returns(nothing), in) ||
            throw(ArgumentError("result $k cannot replace argument $j, whose structure differs"))
        for (r, a) in zip(leaves(out), leaves(in))
            (r === nothing) == (a === nothing) ||
                throw(ArgumentError("result $k cannot replace argument $j: one has an array or a number where the other has none"))
            r === nothing && continue
            x, y = program.nodes[program.results[r]], program.nodes[program.arguments[a]]
            (x.dims, x.eltype, x.scalar) == (y.dims, y.eltype, y.scalar) ||
                throw(ArgumentError("result $k cannot replace argument $j: $(x.eltype)$(x.dims) in place of $(y.eltype)$(y.dims)"))
            push!(pairs, r => a)
        end
    end
    allunique(first.(pairs)) && allunique(last.(pairs)) ||
        throw(ArgumentError("each result replaces one argument, and each argument is replaced once"))
    pairs
end

"""
    Tempest.Compiler.footprint(ex) -> (; bytes, apart)

The bytes of a prepared program's buffers (inputs, constants, results and
slots), and what they would be with each value stored apart instead of in
slots.
"""
function footprint(ex::Executable)
    graph, s = ex.plan.graph, ex.storage
    own = sum(v -> s.slots[v] == 0 && ex.buffers[v] !== nothing ? value_bytes(graph.values[v]) : 0, eachindex(graph.values); init = 0)
    own -= sum(((r, a),) -> value_bytes(graph.values[r]), s.donated; init = 0)
    apart = sum(v -> s.slots[v] > 0 ? value_bytes(graph.values[v]) : 0, eachindex(graph.values); init = 0)
    (bytes = own + sum(s.sizes; init = 0), apart = own + apart)
end

"""
    task_state(target) -> function

A function that, called first in a new task, gives the task the state on
`target` of the task that asked for it (for CUDA, its context and stream).
"""
task_state(target) = () -> nothing

"""
    concurrently(f, xs, target) -> results

`f` of each of `xs`, in tasks that take turns on the calling thread, each in
the calling task's state on `target`. Groups and kernels are built this way:
while one task waits for a compilation running `elsewhere`, the others start
theirs. Device objects are made on the calling thread, so at exit they are
finalized after what the caller made from them later (a module unloaded before
a captured graph that launches it crashes the driver).
"""
function concurrently(f, xs, target)
    enter = task_state(target)
    map(fetch_result, [@async((enter(); f(x))) for x in xs])
end

"""
    elsewhere(f) -> f()

Run `f` on another thread, if there is one, and wait for it while the calling
thread's other tasks run. For compilation, which makes no device objects.
"""
elsewhere(f) = fetch_result(Threads.@spawn f())

function fetch_result(t::Task)
    try
        fetch(t)
    catch err
        err isa TaskFailedException ? throw(err.task.exception) : rethrow()
    end
end

"""
    bind!(ex, inputs...)
    bind!(ex, position => input, ...)

Copy inputs into the executable's input buffers: all of them, in the
structure `prepare` took them, or those at the positions given, as a new
batch while parameters stay where the last execution left them. An array
bound by reference and bound again is not copied.

```jldoctest
julia> ex = prepare(trace(x -> 2 .* x, [1.0, 2.0]), [1.0, 2.0]);

julia> bind!(ex, [3.0, 4.0]);

julia> outputs(execute!(ex))
2-element Vector{Float64}:
 6.0
 8.0
```
"""
function bind!(ex::Executable, @nospecialize(inputs...))
    bind_leaves!(ex, leaves(ex.program.input_structure), leaves(inputs))
end

function bind!(ex::Executable, bindings::Pair{<:Integer}...)
    foreach(((j, x),) -> bind_leaves!(ex, leaves(ex.program.input_structure[j]), leaves(x)), bindings)
    ex
end

function bind_leaves!(ex, positions, xs)
    length(positions) == length(xs) || throw(ArgumentError("expected $(length(positions)) inputs, got $(length(xs))"))
    for (k, x) in zip(positions, xs)
        k === nothing || x === ex.referenced[k] || copyto!(ex.arguments[k], x isa Number ? [x] : vec(x))
    end
    ex
end

"""
    execute!(ex) -> ex

Run the program on the inputs bound last, launching every kernel in order,
then copy staged results into the arguments they replace. The results are in
`outputs(ex)`. Whether or not the program is captured, this launches kernel by
kernel; `replay!` runs the capture.
"""
function execute!(ex::Executable)
    run_once(ex)
    ex
end

# What an execution launches, and what a capture records.
function run_once(ex::Executable)
    foreach(launch -> launch(), ex.launches)
    foreach(copy -> copy(), ex.copies)
end

"""
    capture!(ex; repetitions = 1) -> ex

Record what `execute!` launches as one graph, for `replay!` to run, each
replay `repetitions` executions of the program (for timing). Recording runs
nothing. What varies between replays lives in device buffers: inputs bound by
copy or by reference, scalars and seeds among them, are read where they are,
so they need no new capture. Capturing again replaces the graph. On a target
without graphs, a replay launches the kernels.

```julia
ex = capture!(prepare(p, CuArray(x)))
replay!(ex)                     # runs the graph
bind!(ex, CuArray(y))           # copied into the buffers the graph reads
replay!(ex)                     # the same graph, on the new values
```
"""
function capture!(ex::Executable; repetitions::Integer = 1)
    ex.captured === nothing || release(ex.target, ex.captured)
    ex.captured = nothing
    ex.captured = capture(ex.target, () -> foreach(_ -> run_once(ex), 1:repetitions), (ex.launches, ex.copies))
    ex
end

"""
    replay!(ex) -> ex

Run what `capture!` recorded, on the inputs bound last: what `execute!` does,
`repetitions` times over, as one launch of a graph on a target with graphs.
The results are in `outputs(ex)`.
"""
function replay!(ex::Executable)
    ex.captured === nothing && throw(ArgumentError("nothing to replay: capture! records the program first"))
    replay(ex.target, ex.captured)
    ex
end

"""
    capture(target, run, keep) -> graph

`run`, which launches work on `target`, captured as a graph for `replay(target,
graph)` to launch again. The graph keeps `keep`, what it launches, alive until
`release(target, graph)` or the collector destroys it. A target without graphs
replays `run` itself.
"""
capture(::Target, run, keep) = run
replay(::Target, run) = run()
release(::Target, graph) = nothing

function explain(io::IO, ex::Executable)
    explain(io, ex.plan)
    ex.target isa CUDATarget && println(io, "products in ", ex.target.precision === :tf32 ? "TF32 where Float32" :
        ex.target.precision === :tf32x3 ? "three TF32 products where Float32" : "full precision",
        "; half-precision products accumulate in Float32",
        ex.target.online ? "; softmax online where a sum is weighted by it" : "; no online softmax")
    foreach(c -> explain(io, ex.plan, c), ex.considered)
    explain(io, ex.plan, ex.storage)
    println(io, "implementations from ", isempty(ex.packages) ? "no package" : join(map(string ∘ nameof, ex.packages), ", "),
        ", and the floor:")
    for (k, d) in enumerate(ex.kernels)
        x = ex.decisions[k]
        println(io, "  kernel ", k, ": ", d, x === nothing ? "" : string(" (", x, ")"))
    end
end

explain(ex::Executable) = explain(stdout, ex)

# What runs kernel `k`: an implementation or the floor.
function runner(ex::Executable, k)
    d = ex.decisions[k]
    d === nothing || d.choice.implementation === nothing ? "floor" : label(d.choice.implementation)
end

function Base.show(io::IO, ex::Executable)
    print(io, "Executable(", flow(ex.program), "; ", ex.target, ", ", counted(length(ex.kernels), "kernel"), ")")
end

function Base.show(io::IO, ::MIME"text/plain", ex::Executable)
    println(io, "Executable on ", ex.target, ": ", flow(ex.program))
    runners = map(k -> runner(ex, k), eachindex(ex.kernels))
    counts = sort!([(count(==(r), runners), r) for r in unique(runners)]; rev = true)
    println(io, "  ", counted(length(runners), "kernel"), ": ", join(map(((n, r),) -> "$n $r", counts), ", "))
    ex.captured === nothing || println(io, "  captured: replay! runs the recording")
    print(io, "  explain(ex) shows what each kernel computes and stores, and what runs it")
end

"""
    outputs(ex) -> results

Where the executable leaves its results, in the program's output structure:
arrays over its buffers, the same arrays every time, which each `execute!` or
`replay!` fills again. Copy what you keep. A number is an array of no
dimensions here, since it lives in a buffer too; `compile` returns numbers.
"""
outputs(ex::Executable) = ex.outputs

# The CPU target.
allocate(::CPUTarget, T, n) = Vector{T}(undef, n)
upload(::CPUTarget, x) = x

kernel_slots(p, g) = sort!(collect(union(group_reads(p.graph, g), [v for v in members(g) if p.stored[v]])))

"The floor's kernels for one group, in launch order."
struct FloorKernels
    kernels::Vector{ScalarKernel}
end

Base.show(io::IO, f::FloorKernels) = print(io, "floor, ", join((string(k.items, " items") for k in f.kernels), " then "))

"""
A floor kernel's launch on the CPU. Each kernel is a new function, so its
launch holds the function and arguments untyped: a closure over them would be
a new type too, and every method it passes through would compile again for it.
"""
struct HostLaunch
    f::Any
    args::Tuple
    items::Int
end

(l::HostLaunch)() = Base.invokelatest(l.f, l.args, l.items)

function build_floor(target::CPUTarget, p::Plan, g::Group, buffers)
    ks = scalar_kernels(p.graph, p.stored, g, kernel_slots(p, g))
    launches = map(ks) do k
        f = generated(:(function (buffers, n)
            for lin in 0:(n - 1)
                $(k.body)
            end
            nothing
        end))
        HostLaunch(f, Tuple(buffers[v] for v in k.slots), k.items)
    end
    (() -> foreach(l -> l(), launches), FloorKernels(ks))
end
