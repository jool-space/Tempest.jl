# Prepared programs: a plan with fixed buffers and a launch per kernel.

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
    # Whether its kernels were built; a plan alone can be explained, not run.
    built::Bool
end


"""
    prepare(program, inputs...; target, measure = :close, replaces = (), by_reference = (), recompute = (), build = true) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default. Close calls between
claims are timed when `measure` allows and the target can, and kept in the
tuning record. `measure = :none` times no claim but still uses the times in
the record; `:close`, the default, times the options whose estimates are
within the close-call factor of the best; `:all` times every option of every
group and both sides of every comparison. `true` and `false` mean `:close` and
`:none`.

A claim that fails to build is no option once it has: the program is planned
again without it, each map of the group it claimed proposed to be stored
apart, so that a scalar function a kernel package cannot lower runs on the
floor and the rest of the group on the package. `explain` names the claims
that failed.

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
`Tempest.Compiler.recomputable_folds`). `partition` decides which of the
forward's maps the backward recomputes rather than reads, by a minimum cut
of what saving them costs (see `Tempest.Compiler.partition`); `false` saves
every map the backward reads, as a trial's comparison.

`build = false` plans and decides but builds and allocates nothing: the
executable can be explained, not run, on a machine without the target's
device. No claim is timed, and claims are priced on nominal rates.

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
        replaces = (), by_reference = (), recompute = (), partition::Bool = true, build::Bool = true)
    # The terms an earlier preparation interned are of its program.
    IR.forget_terms!()
    forget_matches!()
    graph = lower(program)
    permits_rewrites(target) && (graph = rewrite(graph; rounded = rounded_backward(target)))
    pairs = replacements(program, replaces)
    input = Dict(v.definition.argument => id for (id, v) in enumerate(graph.values) if v.definition isa Input)
    pinned = Set{Int}(recompute)
    # A value reading an argument a result replaces is not recomputed:
    # recomputed in the kernel that writes the result, it would read the
    # argument where the result is written, and the result would be copied
    # there instead (see `writes_over`). It reads it directly, or through
    # maps, which may be recomputed with it.
    replaced = Set(input[a] for (_, a) in pairs)
    function reads_replaced(v)
        d = graph.values[v].definition
        any(operands(d)) do o
            any(read_values(o)) do r
                r in replaced || graph.values[r].definition isa Pointwise && reads_replaced(r)
            end
        end
    end
    kept = Set(v for (v, val) in enumerate(graph.values) if val.definition isa Reduction && !(v in pinned) && reads_replaced(v))
    if partition
        # The forward's maps the backward recomputes rather than reads (see
        # `partition`).
        saved, recomputed = Compiler.partition(graph)
        filter!(!reads_replaced, recomputed)
        foreach(v -> push!(graph.rewrites, v => "recomputed in the backward where it is read: saving it would cost more than its operands"), recomputed)
        union!(pinned, recomputed)
    end
    d = Decider(target, graph; measure = build ? measure : :dry, recompute = pinned, keep = kept)
    # Planned again while the last plan's builds failed, a few times at most;
    # then what fails runs on the next option, the floor last (see
    # `build_group`). The last plan's buffers are let go first. Without
    # `build`, implementations compile nothing as they claim (`compiling`).
    ex = Base.ScopedValues.with(Implement.COMPILING => build) do
        ex = planned(program, d, inputs, pairs, input, by_reference, build)
        for _ in 2:PLANS
            any(x -> x !== nothing && x.how === :instead, ex.decisions) || break
            ex = nothing
            ex = planned(program, forget_decisions!(d), inputs, pairs, input, by_reference, build)
        end
        ex
    end
    ex.built && bind!(ex, inputs...)
    ex
end

# The plans a preparation makes at most, each after builds that failed.
const PLANS = 4

# The program planned by decider `d` and, with `build`, built on buffers of
# its own, its inputs not yet bound: `prepare` once its graph is lowered.
function planned(program, d, inputs, pairs, input, by_reference, build)
    target = d.target
    p, considered = cover(d)
    search!(d, p)
    ds = decisions(d, p)
    values = p.graph.values
    result_value(k) = p.graph.results[k].value
    # A result that is an argument's storage would change as another result replaces it.
    for k in eachindex(p.graph.results), (r, a) in pairs
        k != r && result_value(k) == input[a] && throw(ArgumentError("result $k is argument $a, which result $r replaces"))
    end
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options)
    p, order, s = storage(p, [result_value(k) => input[a] for (k, a) in pairs]; in_place = k -> writes_in_place(ds[k]))
    ds = ds[order]
    if !build
        none(n) = Vector{Any}(nothing, n)
        return Executable(program, p, target, s, Any[], none(length(values)),
            none(length(program.arguments)), none(length(program.arguments)), none(length(p.graph.results)), nothing,
            Any[], none(length(p.groups)), Any[ds...], Any[considered...], Any[], nothing, false)
    end
    slots = Any[allocate(target, UInt8, bytes) for bytes in s.sizes]
    referenced = Vector{Any}(nothing, length(program.arguments))
    xs = argument_leaves(program, inputs)
    for j in by_reference, k in leaves(program.input_structure[j])
        k === nothing || xs[k] isa Number || (referenced[k] = xs[k])
    end
    buffers = Vector{Any}(nothing, length(values))
    # Results in an argument's storage, and folds by key over the contents they
    # start from, take that storage, in the order values are computed.
    taken = sort!([s.donated; s.overwrites]; by = first)
    for (id, v) in enumerate(values)
        any(t -> first(t) == id, taken) && continue
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
    foreach(((r, a),) -> buffers[r] = buffers[a], taken)
    arguments = Any[get(buffers, get(input, k, 0), nothing) for k in eachindex(program.arguments)]
    replaced = Dict(pairs)
    results = Any[reshape(haskey(replaced, k) ? arguments[replaced[k]] : buffers[result_value(k)],
        p.graph.result_scalar[k] ? () : p.graph.result_dims[k]) for k in eachindex(p.graph.results)]
    built = concurrently(((g, x),) -> build_group(d, p, g, x, buffers), collect(zip(p.groups, ds)), target)
    copies = Any[(() -> copyto!(buffers[a], buffers[r])) for (r, a) in s.staged]
    outs = rebuild(k -> k === nothing ? nothing : results[k], program.output_structure)
    Executable(program, p, target, s, slots, buffers, arguments, referenced, results, outs,
        Any[b[1] for b in built], Any[b[2] for b in built], Any[b[3] for b in built], Any[considered...], copies, nothing, true)
end

runnable(ex::Executable) = ex.built || throw(ArgumentError("the program was prepared with build = false: it can be explained, not run"))

"A value of `n` elements of type `T` in the first bytes of a slot."
slot_view(slot, T, n) = reinterpret(T, view(slot, 1:(n * sizeof(T))))

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
    own -= sum(v -> s.slots[v] == 0 ? value_bytes(graph.values[v]) : 0, unique(first.([s.donated; s.overwrites])); init = 0)
    apart = sum(v -> s.slots[v] > 0 ? value_bytes(graph.values[v]) : 0, eachindex(graph.values); init = 0)
    (bytes = own + sum(s.sizes; init = 0), apart = own + apart)
end

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
elsewhere(@nospecialize(f)) = fetch_result(Threads.@spawn f())

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
    runnable(ex)
    bind_leaves!(ex, leaves(ex.program.input_structure), leaves(inputs))
end

function bind!(ex::Executable, bindings::Pair{<:Integer}...)
    runnable(ex)
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
    runnable(ex)
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
    runnable(ex)
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

function explain(io::IO, ex::Executable)
    explain(io, ex.plan)
    ex.target isa CUDATarget && println(io, "products in ", ex.target.precision === :tf32 ? "TF32 where Float32" :
        ex.target.precision === :tf32x3 ? "three TF32 products where Float32" : "full precision",
        "; half-precision products accumulate in Float32",
        ex.target.online ? "; softmax online where a sum is weighted by it" : "; no online softmax")
    foreach(c -> explain(io, ex.plan, c), ex.considered)
    explain(io, ex.plan, ex.storage)
    is = implementations(ex.target)
    println(io, isempty(is) ? "the floor alone:" : string("implementations ", join(map(repr, is), ", "), " and the floor:"))
    for (k, d) in enumerate(ex.kernels)
        x = ex.decisions[k]
        println(io, "  kernel ", k, ": ", ex.built ? d : "not built", x === nothing ? "" : string(" (", x, ")"))
        x === nothing || foreach(f -> println(io, "    a claim of ", f.implementation, " failed to build: ", f.reason), x.failed)
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
    ex.built || println(io, "  not built: explain(ex) shows the plan, and nothing runs")
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

kernel_slots(p, g) = sort!(collect(union(group_reads(p.graph, g), [v for v in members(g) if p.stored[v]])))

"The floor's kernels for one group, in launch order."
struct FloorKernels
    kernels::Vector{ScalarKernel}
end

function Base.show(io::IO, f::FloorKernels)
    described(k) = string(k.items, " items", k.split === nothing ? "" : string(" split ", k.split.parts, "×", k.split.lanes, " lanes",
        k.split.side > 1 ? string(", ", k.split.side, " side by side") : ""))
    print(io, "floor, ", join(map(described, f.kernels), " then "))
end

"The launches of a group's floor kernels, in order, and the workspaces they share, which live as long as the launches."
struct FloorRun
    launches::Vector{Any}
    workspaces::Vector{Any}
end

(r::FloorRun)() = foreach(l -> l(), r.launches)

# Whether fold by key `r` is in the storage of the contents it starts from:
# `storage` put it over them.
starts_in_place(graph, r, buffers) = graph.values[r].definition.into isa Read && buffers[graph.values[r].definition.into.value] === buffers[r]

# The floor for a group on any target: its scalar kernels, each launched as
# the target launches one (see `floor_launch`), their folds split as it asks
# (see `floor_split`). A fold by key in the storage of the contents it starts
# from starts from them there.
function build_floor(target, p::Plan, g::Group, buffers)
    held = filter(r -> keyed(p.graph, r) && starts_in_place(p.graph, r, buffers), g.reductions)
    ks = scalar_kernels(p.graph, p.stored, g, kernel_slots(p, g); held, split = (items, positions, adjacent) -> floor_split(target, items, positions, adjacent))
    spaces = Dict{Int, Any}()
    space(w) = get!(() -> allocate(target, w.state, w.length), spaces, w.fold)
    launches = Any[floor_launch(target, k, (Tuple(buffers[v] for v in k.slots)..., map(space, k.workspaces)...)) for k in ks]
    (FloorRun(launches, collect(values(spaces))), FloorKernels(ks))
end
