# Prepared programs: a plan with fixed buffers and a launch per kernel.

"""
    Executable

A program prepared for a target: its plan, its storage, and a launch per
kernel. Each stored value has a buffer: inputs, constants and results their
own, other values a view of their slot, a result that replaces an argument
the argument's, and a result that is an input as it was before the program
wrote it a buffer of its own, which the input is copied into before the
kernels run. `bind!` copies inputs into the buffers the executable owns, and
`point!` points the kernels at an array the program only reads where the
caller has it, when they can; `execute!` and `replay!` run the program over
them, and `outputs` is where the results are, overwritten by each run.
"""
# An argument's value, and the kernels that read it.
const Readers = @NamedTuple{value::Int, kernels::Vector{Int}}

# A value moved, as `retarget!` takes it: its old and new addresses and its bytes.
const Move = @NamedTuple{value::Int, from::UInt, to::UInt, bytes::UInt}

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
    # The inputs the program writes that results are as they were before: per
    # input, its argument, its buffer and the buffer those results are over,
    # which it is copied into ahead of the kernels.
    saves::Vector{Any}
    # Copies of staged results into the arguments they replace, after the kernels.
    copies::Vector{Any}
    # The graph `execute!` replays, once captured (see `capture!`).
    captured::Any
    # Per argument whose launches can read the caller's array where it is
    # (see `point!`): its value, and the kernels that read it; nothing for one
    # copied. And per such argument, the array they read: its buffer, or the
    # caller's array they were pointed at last, and its address. And per
    # argument, the address of the array it was bound to last, as a number:
    # holding the array would keep a caller's last arrays alive, as a
    # training step's last weights. And per argument, whether it is pointed
    # only once the caller brings the same array twice in a row: once its
    # graph had to be recorded again.
    pointable::Vector{Union{Nothing, Readers}}
    pointed::Vector{Any}
    at::Vector{UInt}
    seen::Vector{UInt}
    settles::Vector{Bool}
    # The executions of the program a replay runs (see `capture!`).
    repetitions::Int
    # Whether its kernels were built; a plan alone can be explained, not run.
    built::Bool
end


"""
    prepare(program, inputs...; target, measure = :close, replaces = (), by_reference = (), recompute = (), ride = (), build = true, verify) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default. Close calls between
claims are timed when `measure` allows and the target can, and kept in the
tuning record. `measure = :none` times no claim but still uses the times in
the record; `:close`, the default, times the options whose estimates are
within the close-call factor of the best; `:all` times every option of every
group and both sides of every comparison. `true` and `false` mean `:close` and
`:none`. The environment's `TEMPEST_MEASURE` (`none`, `close` or `all`) sets
the default, as the tests set it to `none`: on a device other jobs share, a
close call timed is decided by chance.

A claim that fails to build is no option once it has: the program is planned
again without it, each map of the group it claimed proposed to be stored
apart, so that a scalar function a kernel package cannot lower runs on the
floor and the rest of the group on the package. `explain` names the claims
that failed.

`replaces` declares state, as pairs `result => argument` of positions among the
program's results and its arguments, alike in structure, shapes and types:
`(2 => 1, 3 => 2)` for a step `(params, state, batch) -> (loss, params, state)`.
After each execution the argument holds the result, as if bound again, so the
next execution reads it where it is. An input the program writes is replaced
so by what it writes; a result that is the input as it was before the write
is copied apart before the kernels run. A result is written into the argument's
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
`Tempest.Compiler.recomputable_folds`). `ride` pins sums over a
recurrence's steps to be added as the steps run, so numbered: what the cover
would otherwise decide by cost (see `Tempest.Compiler.riding_sums`).
`partition` decides which of the
forward's maps the backward recomputes rather than reads, by a minimum cut
of what saving them costs (see `Tempest.Compiler.partition`); `false` saves
every map the backward reads, as a trial's comparison.

`build = false` plans and decides but builds and allocates nothing: the
executable can be explained, not run, on a machine without the target's
device. No claim is timed, and claims are priced on nominal rates.

`verify = true` checks every plan the preparation forms, the cover's trials
among them, and the storage of the one it runs, throwing where one breaks an
invariant the kernels rely on (see `Tempest.Compiler.verify`). It is the
default where the environment sets `TEMPEST_VERIFY=true`, as the tests do.

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
function prepare(program::Program, @nospecialize(inputs...); target = default_target(inputs), measure = measuring(),
        replaces = (), by_reference = (), recompute = (), ride = (), partition::Bool = true, build::Bool = true, verify::Bool = verifying())
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
    # maps, which may be recomputed with it. Each value's answer is kept:
    # maps share operands, and walked afresh from every reader, a chain of
    # them is walked once per path through it.
    replaced = Set(input[a] for (_, a) in pairs)
    answers = Dict{Int, Bool}()
    function reads_replaced(v)
        haskey(answers, v) && return answers[v]
        d = graph.values[v].definition
        answers[v] = any(operands(d)) do o
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
    d = Decider(target, graph; measure = build ? measure : :dry, recompute = pinned, keep = kept, ride)
    # Planned again while the last plan's builds failed, a few times at most;
    # then what fails runs on the next option, the floor last (see
    # `build_group`). The last plan's buffers are let go first. Without
    # `build`, implementations compile nothing as they claim (`compiling`).
    ex = Base.ScopedValues.with(Implement.COMPILING => build, VERIFYING => verify) do
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
function planned(program, d, @nospecialize(inputs), pairs, input, by_reference, build)
    target = d.target
    p, considered = cover(d)
    search!(d, p)
    ds = decisions(d, p)
    values = p.graph.values
    result_value(k) = p.graph.results[k].value
    # A result that is an argument's storage would change as another result
    # replaces it. One that is an input as it was before the program wrote it,
    # as `old` in `x -> (old = copy(x); x .= 0.9f0 .* x; old)`, is saved: the
    # input is copied into a buffer of its own before the kernels run.
    saved = Dict{Int, Int}()
    for k in eachindex(p.graph.results), (r, a) in pairs
        k != r && result_value(k) == input[a] || continue
        (r => a) in program.writes || throw(ArgumentError("result $k is argument $a, which result $r replaces"))
        saved[k] = input[a]
    end
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options)
    p, order, s = storage(p, [result_value(k) => input[a] for (k, a) in pairs]; in_place = k -> writes_in_place(ds[k]))
    ds = ds[order]
    VERIFYING[] && verify(p, s; in_place = k -> writes_in_place(ds[k]))
    if !build
        none(n) = Vector{Any}(nothing, n)
        return Executable(program, p, target, s, Any[], none(length(values)),
            none(length(program.arguments)), none(length(program.arguments)), none(length(p.graph.results)), nothing,
            Any[], none(length(p.groups)), Any[ds...], Any[considered...], Any[], Any[], nothing,
            Vector{Union{Nothing, Readers}}(nothing, length(program.arguments)), none(length(program.arguments)),
            zeros(UInt, length(program.arguments)), zeros(UInt, length(program.arguments)), fill(false, length(program.arguments)), 1, false)
    end
    slots = Any[allocate(target, UInt8, bytes) for bytes in s.sizes]
    referenced = Vector{Any}(nothing, length(program.arguments))
    xs = argument_leaves(program, inputs)
    for j in by_reference, k in leaves(program.input_structure[j])
        k === nothing || (referenced[k] = xs[k])
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
    # One buffer per input saved, however many results are it.
    befores = Dict(v => allocate(target, values[v].eltype, storage_length(values[v])) for v in unique(Base.values(saved)))
    saves = Any[(argument = values[v].definition.argument, input = buffers[v], before = b) for (v, b) in befores]
    replaced = Dict(pairs)
    stored(k) = haskey(replaced, k) ? arguments[replaced[k]] : haskey(saved, k) ? befores[saved[k]] : buffers[result_value(k)]
    results = Any[reshape(stored(k), p.graph.result_scalar[k] ? () : p.graph.result_dims[k]) for k in eachindex(p.graph.results)]
    built = concurrently(((g, x),) -> build_group(d, p, g, x, buffers), collect(zip(p.groups, ds)), target)
    copies = Any[(() -> copyto!(buffers[a], buffers[r])) for (r, a) in s.staged]
    outs = rebuild(k -> k isa Int ? results[k] : constant_leaf(k), program.output_structure)
    launches = Any[b[1] for b in built]
    # An argument the program only reads, from its own buffer, which no
    # result is and no other value shares, can be read where the caller has
    # it, when every launch can be pointed elsewhere.
    pointable = Vector{Union{Nothing, Readers}}(nothing, length(program.arguments))
    if all(repoints, launches)
        written = Set(input[a] for (_, a) in pairs)
        shared = Set(last.(taken))
        returned = Set(result_value(k) for k in eachindex(p.graph.results))
        for (a, v) in input
            (v in written || v in shared || v in returned || referenced[a] !== nothing) && continue
            readers = [k for (k, g) in enumerate(p.groups) if v in group_reads(p.graph, g)]
            pointable[a] = (value = v, kernels = readers)
        end
    end
    pointed = Any[x === nothing ? nothing : arguments[a] for (a, x) in enumerate(pointable)]
    at = UInt[x === nothing ? 0 : reinterpret(UInt, pointer(x)) for x in pointed]
    Executable(program, p, target, s, slots, buffers, arguments, referenced, results, outs,
        launches, Any[b[2] for b in built], Any[b[3] for b in built], Any[considered...], saves, copies, nothing, pointable, pointed, at, zeros(UInt, length(program.arguments)),
        fill(false, length(program.arguments)), 1, true)
end

runnable(ex::Executable) = ex.built || throw(ArgumentError("the program was prepared with build = false: it can be explained, not run"))

"A value of `n` elements of type `T` in the first bytes of a slot."
slot_view(slot, T, n) = reinterpret(T, view(slot, 1:(n * sizeof(T))))

# The results that replace arguments: pairs of a result's position among the
# program's results and the argument's among its arguments, leaf by leaf. The
# program's writes into its inputs are among them.
function replacements(program, replaces)
    pairs = copy(program.writes)
    for (k, j) in (replaces isa Pair ? (replaces,) : replaces)
        out, in = program.output_structure[k], program.input_structure[j]
        structure_of(Returns(nothing), out) == structure_of(Returns(nothing), in) ||
            throw(ArgumentError("result $k cannot replace argument $j, whose structure differs"))
        for (r, a) in zip(leaves(out), leaves(in))
            (r isa Int) == (a isa Int) ||
                throw(ArgumentError("result $k cannot replace argument $j: one has an array or a number where the other has none"))
            r isa Int || continue
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
    own += sum(save -> sizeof(save.before), ex.saves; init = 0)
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
    moves = nothing
    for (k, x) in zip(positions, xs)
        k === nothing && continue
        x === ex.referenced[k] && continue
        moves = point_at!(ex, k, ex.arguments[k], moves)
        copyto!(ex.arguments[k], vec(x))
    end
    retarget!(ex, moves)
end

"""
    point!(ex, inputs...) -> ex

Bind inputs as `compile`'s call does: an argument the program only reads is
read where the caller has it, when every launch can be pointed elsewhere
(`repoints`) and the target can read the array where it is (`pointable`);
it is copied into its buffer otherwise, as `bind!` copies. An array where
the launches read already costs nothing. The launches of one elsewhere are
pointed at it, and a captured graph is changed where it records their
addresses (`retarget!`), or recorded again where the target cannot change
it; recording again costs more than copying an array that moves every call,
as a training step's weights do, so an argument whose graph was recorded
again is read where it is only once the caller brings the same array twice
in a row. The executable holds the arrays it reads until it is bound again.
"""
function point!(ex::Executable, @nospecialize(inputs...))
    runnable(ex)
    positions, xs = leaves(ex.program.input_structure), leaves(inputs)
    length(positions) == length(xs) || throw(ArgumentError("expected $(length(positions)) inputs, got $(length(xs))"))
    moves = nothing
    for (k, x) in zip(positions, xs)
        k === nothing || (moves = point_leaf!(ex, k, x, moves))
    end
    retarget!(ex, moves)
end

# Bind input `x` to argument `k`, as `point!` binds each: one call a leaf,
# specialized on the leaf's type, a model's hundreds of arrays each checked
# in a few hundred nanoseconds.
function point_leaf!(ex::Executable, k::Int, x, moves)
    if ex.pointable[k] !== nothing && pointable(ex.target, x)
        here = reinterpret(UInt, pointer(x))
        again = here == ex.seen[k]
        ex.seen[k] = here
        here == ex.at[k] && return moves
        (again || !ex.settles[k]) && return point_at!(ex, k, x, moves)
    end
    moves = point_at!(ex, k, ex.arguments[k], moves)
    x === ex.referenced[k] || copyto!(ex.arguments[k], vec(x))
    moves
end

# Point argument `k` at array `x`, its buffer or the caller's array, adding
# to `moves` (made on the first) what moved for each launch that reads it:
# the launch, the argument, the move as `retarget!` takes it, and the
# storage the launch reads from now on, which `retarget!` points it at.
function point_at!(ex::Executable, k, x, moves)
    readers = ex.pointable[k]
    readers === nothing && return moves
    to = reinterpret(UInt, pointer(x))
    to == ex.at[k] && return moves
    buffer = vec(x)
    move = (value = readers.value, from = ex.at[k], to, bytes = UInt(sizeof(x)))
    moves = something(moves, Tuple{Int, Int, Move, Any}[])
    foreach(i -> push!(moves, (i, k, move, buffer)), readers.kernels)
    ex.pointed[k] = x
    ex.at[k] = to
    moves
end

# Point the launches at their new storage, and change the captured graph
# where it records their addresses, launch by launch, or record it again
# where the target cannot; an argument whose graph was recorded again waits
# from then on to be brought twice in a row.
function retarget!(ex::Executable, moves)
    moves === nothing && return ex
    sort!(moves; by = first)
    changed = true
    start = 1
    while start <= length(moves)
        launch = moves[start][1]
        stop = start
        while stop < length(moves) && moves[stop + 1][1] == launch
            stop += 1
        end
        mine = view(moves, start:stop)
        # The launch is pointed at the new storage once, by the target or after it.
        pointed = Ref(false)
        repoint() = pointed[] || (foreach(m -> repoint!(ex.launches[launch], m[3].value, m[4]), mine); pointed[] = true)
        if ex.captured === nothing
            repoint()
        elseif !retarget!(ex.target, ex.captured, collect(launch_steps(ex, launch)), [m[3] for m in mine], repoint)
            foreach(m -> ex.settles[m[2]] = true, mine)
            changed = false
        end
        repoint()
        start = stop + 1
    end
    changed || recapture!(ex)
    ex
end

# Record the captured graph again, after its launches were pointed elsewhere.
function recapture!(ex::Executable)
    ex.captured === nothing && return ex
    ex.captured = recapture(ex.target, ex.captured, steps(ex), (ex.saves, ex.launches, ex.copies))
    ex
end

# What a graph records, step by step: in each of the executions a replay
# runs, the inputs saved, each launch, and the staged copies.
function steps(ex::Executable)
    once = Any[(() -> copyto!(save.before, save.input) for save in ex.saves)..., ex.launches..., ex.copies...]
    reduce(vcat, fill(once, ex.repetitions))
end

# The steps of launch `i`, one in each execution.
launch_steps(ex::Executable, i) =
    (length(ex.saves) + i) .+ (length(ex.saves) + length(ex.launches) + length(ex.copies)) .* (0:(ex.repetitions - 1))

"""
    execute!(ex) -> ex

Run the program on the inputs bound last: copy the inputs it writes into
the results that are them as they were before, launch every kernel in order,
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
    foreach(save -> copyto!(save.before, save.input), ex.saves)
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
    ex.repetitions = repetitions
    ex.captured = capture(ex.target, steps(ex), (ex.saves, ex.launches, ex.copies))
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
    foreach(save -> println(io, "  input ", save.argument, " copied apart before the kernels, as results return it before the program writes it"), ex.saves)
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

# The buffers a group's kernels take, in the order the group first mentions
# their values (`appearance`), as `canonical` numbers them: groups alike take
# theirs in one order, and generate one kernel, however the values are numbered.
function kernel_slots(p, g)
    slots = union(group_reads(p.graph, g), [v for v in members(g) if p.stored[v]])
    order = Dict(v => k for (k, v) in enumerate(appearance(p, g)))
    sort!(collect(slots); by = v -> (get(order, v, typemax(Int)), v))
end

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

repoints(r::FloorRun) = all(repoints, r.launches)
repoint!(r::FloorRun, value, buffer) = (foreach(l -> repoint!(l, value, buffer), r.launches); r)

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
