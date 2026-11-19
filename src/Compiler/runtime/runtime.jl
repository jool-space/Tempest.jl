# Prepared programs: a plan with fixed buffers and a launch per kernel.

"""
    Executable

A program prepared for a target: its plan, its storage, and a launch per
kernel. Each stored value has storage: constants and results a buffer of
their own, other values a view of their slot, a result that replaces an
argument the argument's, and a result that is an input as it was before the
program wrote it a buffer of its own, which the input is copied into before
the kernels run. An argument's storage is the array its binding gives it:
`point!` makes it the caller's array where the kernels can read and write it
there, and `bind!` copies the caller's array into a buffer the executable
owns, made when a binding first copies into it. `execute!` and `replay!` run
the program over them, and copy what it wrote into an argument back into the
caller's array where that is not its storage, and `outputs` is where the
results are, overwritten by each run.
"""
# Where an argument's storage is used, for `point!` to move it: each step of
# an execution (see `Step`) that reads or writes it, by its place among
# them, with the value the step names it by, the argument's own or a value
# written over it, a result in place of the argument or a fold by key; and
# the results over it.
const Uses = @NamedTuple{steps::Vector{Tuple{Int, Int}}, results::Vector{Int}}

# A value moved, as `retarget!` takes it: its old and new addresses and its bytes.
const Move = @NamedTuple{value::Int, from::UInt, to::UInt, bytes::UInt}

# What a binding moved, as `retarget!` takes it: per step that uses an
# argument whose storage moved, the step, the argument, the move of the
# value the step names the storage by, and the storage the step uses from
# now on. Of one type whatever moved, so that the first binding, which
# moves nothing in a call or nothing yet recorded, compiles what a later
# one that moves a recorded step runs.
const Moves = Vector{Tuple{Int, Int, Move, Any}}

"""
A step of an execution that copies one buffer into another: an input saved
apart before the kernels, a staged result into the argument it replaces
after them, or on the CPU a crossing (see `crossing`). It copies as many
elements as `to` holds, from `from` after the first `offset`: all of it but
for a crossing of a span. It repoints as a launch does (see `repoint!`), by
the values it copies to and from, 0 for a buffer of no value, and keeps its
offset, which is where in the value's storage the span starts, whatever
buffer holds it.
"""
mutable struct Copy
    to::Any
    from::Any
    const values::Pair{Int, Int}
    const offset::Int
end

Copy(to, from, values) = Copy(to, from, values, 0)

(c::Copy)() = (copyto!(c.to, 1, c.from, c.offset + 1, length(c.to)); nothing)

repoints(::Copy) = true

function repoint!(c::Copy, value, buffer)
    first(c.values) == value && (c.to = buffer)
    last(c.values) == value && (c.from = buffer)
    c
end

mutable struct Executable{T <: Target}
    program::Program
    plan::Plan
    target::T
    storage::Storage
    slots::Vector{Any}
    # Per value, its storage, but for a value over an argument's: that is
    # the argument's, its buffer or the caller's array (`pointed`).
    buffers::Vector{Any}
    # Per argument of the program: the buffer a binding copies it into, made
    # when one first does (`own!`), or the array bound by reference; and the
    # caller's array when it is bound by reference.
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
    # input, its argument, the buffer those results are over, and the `Copy`
    # of the input into it ahead of the kernels.
    saves::Vector{Any}
    # The `Copy` of each staged result into the argument it replaces, after the kernels.
    copies::Vector{Copy}
    # The steps of an execution, in order, each on its device: the inputs
    # saved, each launch, and the staged copies (see `walk`).
    steps::Vector{Step}
    # What each step reads and writes, as storage (see `step_touches`): what
    # orders the steps between streams (see `stream_schedule`).
    touches::Vector{Touches}
    # How an execution runs them, each on its stream (see `walk`).
    schedule::Schedule
    # What `replay!` runs, once captured (see `capture!`).
    captured::Union{Nothing, Captured}
    # The device each of the plan's labels stands for (see `Placement`), and
    # each argument's and each result's, where a binding finds, binds and
    # writes back the one and returns the other (see `on_each_device`).
    devices::Vector{Union{Nothing, Int}}
    argument_devices::Vector{Union{Nothing, Int}}
    result_devices::Vector{Union{Nothing, Int}}
    # Per device of `devices`, the stream that used the executable last
    # there, nothing before its first use (see `taken!`).
    streams::Vector{Any}
    # Per argument whose storage can be the caller's array (see `point!`):
    # where the storage is used; nothing for one copied. And per such
    # argument, its storage: its buffer, or the caller's array it was
    # pointed at last, as the first binding found it or a later one, and
    # its address. And per
    # argument, the address of the array it was bound to last, as a number:
    # holding the array would keep a caller's last arrays alive, as a
    # training step's last weights. And per argument, whether it is pointed
    # only once the caller brings the same array twice in a row: once its
    # graph had to be recorded again. And per argument the program writes,
    # the caller's array `point!` bound it to where that is not its storage,
    # which each run copies what the program wrote back into, else nothing.
    pointable::Vector{Union{Nothing, Uses}}
    pointed::Vector{Any}
    at::Vector{UInt}
    seen::Vector{UInt}
    settles::Vector{Bool}
    back::Vector{Any}
    # The executions of the program a replay runs (see `capture!`).
    repetitions::Int
    # Whether its kernels were built; a plan alone can be explained, not run.
    built::Bool
    # The decisions it was built on (see `Reached`).
    reached::Any
end


"""
    prepare(program, inputs...; target, measure = :none, replaces = (), by_reference = (), point = false, recompute = (), ride = (), partition = true, build = true, verify) -> Executable

Plan `program` for the shapes and types of `inputs` and build its kernels on
`target`, chosen from the inputs' array types by default.

- `measure`: which claims are timed, where the target can, and kept in the
  tuning record: `:none`, the default, none, though the times in the record
  are still used; `:close`, the options whose estimates are within the
  close-call factor of the best; `:all`, every option of every group and
  both sides of every comparison. `true` and `false` mean `:close` and
  `:none`, and the environment's `TEMPEST_MEASURE` overrides the default.
- `replaces`: state, as pairs `result => argument` of positions among the
  program's results and its arguments, alike in structure, shapes and types:
  after each execution the argument holds the result.
- `by_reference`: the positions of the arguments whose arrays, of the
  target's array type, are the executable's storage, where `bind!` copies
  the others into storage it owns.
- `point`: bind `inputs` as `point!` does, the caller's arrays the storage,
  as `compile` prepares.
- `recompute`, `ride`: folds pinned to be recomputed where they are read,
  and values pinned to be computed as a recurrence's steps run, by their
  numbers as `explain` prints them: what the cover would otherwise decide by
  cost.
- `partition`: whether a minimum cut decides which of the forward's maps the
  backward recomputes rather than reads; `false` saves every map the
  backward reads.
- `build`: `false` plans and decides but builds and allocates nothing: the
  executable can be explained, not run.
- `verify`: check every plan the preparation forms and the storage of the
  one it runs (see `Tempest.Compiler.verify`); the default where the
  environment sets `TEMPEST_VERIFY=true`.

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

A function that writes its arguments needs none of this: `compile` writes
them where the caller has them, for any arrays a call brings.

The manual's page [Preparing a program](@ref) says how decisions are kept
and replayed, what a claim that fails to build leaves, and each keyword in
full.
"""
function prepare(program::Program, @nospecialize(inputs...); target = default_target(inputs), measure = measuring(),
        replaces = (), by_reference = (), point::Bool = false, recompute = (), ride = (), partition::Bool = true, build::Bool = true,
        verify::Bool = verifying())
    refuse_devices(target, inputs, input_paths)
    s = setup(program; target, placement = argument_devices(target, program, inputs), measure, replaces, recompute, ride, partition, build)
    prepared(s, decide!(s; verify), inputs...; by_reference, point, verify)
end

"""
The tables a preparation fills as it plans, its own, so that two at once, on
two tasks, share none and neither empties the other's: the terms it interns
(`IR.INTERNED`) and the library's matches of its groups (`MATCHES`).
"""
struct Tables
    interned::IR.Interned
    matches::Dict{Key, Vector{Binding}}
end

Tables() = Tables(IR.Interned(), Dict{Key, Vector{Binding}}())

# `f()` with `t` as the tables of the preparation running.
in_tables(f, t::Tables) = Base.ScopedValues.with(f, IR.INTERNED => t.interned, MATCHES => t.matches)

"""
A preparation set up, before it decides: the program, its decider as
`prepare` makes it, on the program's index graph lowered and rewritten with
the folds pinned, kept and riding, and placed, the device each label of its
plan stands for its decider's (see `Placement`); the results paired with the
arguments they replace, the value of each argument, whether it builds, and
its tables.
`decide!` reaches its decisions and `prepared` builds them.
"""
struct Setup{D}
    program::Program
    decider::D # a `Decider`, defined with the cover
    pairs::Vector{Pair{Int, Int}}
    input::Dict{Int, Int}
    build::Bool
    tables::Tables
end

# `f()` as preparation `s` runs it: within its tables, building as it does
# and verifying as asked, on its target's device where it builds. One that
# builds nothing touches no device, as while a package precompiles without one.
function running(f, s::Setup, verify)
    scoped() = Base.ScopedValues.with(f, Implement.COMPILING => s.build, VERIFYING => verify)
    in_tables(s.tables) do
        s.build ? on_device(scoped, s.decider.target) : scoped()
    end
end

"""
    setup(program; target, placement, measure, replaces, recompute, ride, partition, build) -> Setup

`program` set up as `prepare` sets it up, the keywords as `prepare` takes
them, `placement` where its arguments are (see `Placement`), all at home
by default: lowered, rewritten, placed
(see `with_devices`) and partitioned, with its decider, and with tables
of its own (see `Tables`), which `decide!`, `given` and `prepared` fill.
"""
function setup(program::Program; target, placement = Placement(target, program), measure = measuring(), replaces = (),
        recompute = (), ride = (), partition::Bool = true, build::Bool = true)
    tables = Tables()
    target = at_home(target, first(placement.devices))
    in_tables(tables) do
        # Each phase marked (see `phase`).
        lowered = phase(() -> lower(program), :lower)
        rewritten = permits_rewrites(target) ? phase(() -> rewrite(lowered; rounded = rounded_backward(target)), :rewrite) : lowered
        pairs = replacements(program, replaces)
        # A result that replaces an argument is written where the argument
        # lives, and a gradient of one returned there.
        pins = [k => placement.labels[a] for (k, a) in [pairs; program.gradients]]
        graph = phase(() -> with_devices(rewritten, placement.labels; pins), :place)
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
            saved, recomputed = phase(() -> Compiler.partition(graph), :partition)
            filter!(!reads_replaced, recomputed)
            foreach(v -> push!(graph.rewrites, v => "recomputed in the backward where it is read: saving it would cost more than its operands"), recomputed)
            union!(pinned, recomputed)
        end
        d = phase(() -> Decider(target, graph; measure = build ? measure : :dry, recompute = pinned, keep = kept, ride, devices = placement.devices), :decider)
        Setup(program, d, collect(Pair{Int, Int}, pairs), input, build, tables)
    end
end

"""
    decide!(s; verify, timer) -> Decided

The decisions of the preparation set up as `s` and the plan they form,
reached as `prepare` reaches them (see `reach!`): replayed where decisions
kept for it hold, else by the cover, settled. `timer`, `(target, launch) ->
seconds`, times each claim and calibration reference measured: `timed` by
default, one timing at a time in the process, on the task deciding; a
search behind a compiled program's first result has the program's calls
time them (see `Search`). The preparation keeps it, deciding again where
claims fail to build.
"""
function decide!(s::Setup; verify::Bool = verifying(), timer = s.decider.timer)
    s.decider.timer = timer
    running(() -> reach!(s.decider; settle = settling()), s, verify)
end

"""
    given(s, r) -> Decided

Decisions `r`, reached for the preparation set up as `s` by a search
elsewhere in this process, and the plan they form: replayed without what
they rest on held against the files (see `given!`).
"""
given(s::Setup, r) = in_tables(() -> given!(s.decider, r), s.tables)

"""
    prepared(s, decided, inputs...; by_reference, point, verify) -> Executable

The preparation set up as `s` built on decisions `decided` (see `decide!`
and `given`) and bound to `inputs`, as `prepare` builds and binds it. Where claims
fail to build, it decides again, knowing they failed, and builds again. The
decisions it builds on are kept for a preparation of the same program to
replay, unless they were replayed.
"""
function prepared(s::Setup, decided, @nospecialize(inputs...); by_reference = (), point::Bool = false, verify::Bool = verifying())
    d = s.decider
    # Decided and built again while the last plan's builds failed, a few
    # times at most; then what fails runs on the next option, the floor last
    # (see `build_group`). The last plan's buffers are let go first. Without
    # `build`, implementations compile nothing as they claim (`compiling`).
    ex = running(s, verify) do
        ex = phase(() -> planned(s, decided, inputs, by_reference, point), :planned)
        for _ in 2:PLANS
            any(x -> x !== nothing && x.how === :instead, ex.decisions) || break
            ex = nothing
            forget_decisions!(d)
            ex = phase(() -> planned(s, decide!(s; verify), inputs, by_reference, point), :planned)
        end
        # Decisions the cover reached are kept, for a preparation of the
        # same program to replay (see `reach!`); those replayed are kept already.
        r = ex.reached
        r.key === nothing || r.settled || store_reached(settled(r, d))
        ex
    end
    ex.built && phase(() -> point ? point!(ex, inputs...) : bind!(ex, inputs...), :bind)
    ex
end

# The plans a preparation builds at most, each after builds that failed.
const PLANS = 4

# The program of preparation `prep` planned on `decided` and, where it
# builds, built on the storage the binding of `inputs` it makes first gives
# them (see `prepare`'s `point`), before they are bound.
function planned(prep::Setup, decided, @nospecialize(inputs), by_reference, point::Bool)
    program, d, pairs, input, build = prep.program, prep.decider, prep.pairs, prep.input, prep.build
    target = d.target
    p, considered, ds, reached = decided.plan, decided.considered, decided.decisions, decided.reached
    # The sections of `planned` (see `lap!`).
    clock = mark()
    values = p.graph.values
    result_value(k) = p.graph.results[k].value
    # A result that is an argument's storage would change as another result
    # replaces it. One that is an input as it was before the program wrote it,
    # as `old` in `x -> (old = copy(x); x .= 0.9f0 .* x; old)`, is saved: the
    # input is copied into a buffer of its own before the kernels run.
    # Each result is looked up among the pairs by its value, the pairs kept by
    # the input each replaces: compared with every pair, a step that updates
    # each of a stack's parameters compared its results and its pairs, each
    # as many as the stack has layers.
    saved = Dict{Int, Int}()
    replacing = Dict{Int, Vector{Pair{Int, Int}}}()
    foreach(((r, a),) -> push!(get!(() -> Pair{Int, Int}[], replacing, input[a]), r => a), pairs)
    for k in eachindex(p.graph.results), (r, a) in get(replacing, result_value(k), ())
        k == r && continue
        (r => a) in program.writes || throw(ArgumentError("result $k is argument $a, which result $r replaces"))
        saved[k] = input[a]
    end
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options)
    p, order, s = storage(p, [result_value(k) => input[a] for (k, a) in pairs]; in_place = k -> writes_in_place(ds[k]), seconds = k -> estimated(ds[k]))
    clock = lap!(:storage, clock)
    ds = ds[order]
    VERIFYING[] && verify(p, s; in_place = k -> writes_in_place(ds[k]))
    clock = lap!(:verify, clock)
    n = length(program.arguments)
    # Each value on the device its label stands for, each argument on its
    # value's, an argument the program does not read at home, and each
    # result on its value's. What is made for a value, a slot, a buffer or
    # a group's launches, is made with the task on its device.
    devices = d.devices
    device(v) = devices[p.graph.devices[v]]
    on(f, v) = on_device(f, target, device(v))
    argument_devices = Union{Nothing, Int}[haskey(input, k) ? device(input[k]) : devices[1] for k in 1:n]
    result_devices = Union{Nothing, Int}[device(result_value(k)) for k in eachindex(p.graph.results)]
    if !build
        none(n) = Vector{Any}(nothing, n)
        return Executable(program, p, target, s, Any[], none(length(values)),
            none(n), none(n), none(length(p.graph.results)), nothing,
            Any[], none(length(p.groups)), Any[ds...], Any[considered...], Any[], Copy[], Step[], Touches[], Schedule(target, Step[], Touches[]), nothing,
            devices, argument_devices, result_devices, none(length(devices)), Vector{Union{Nothing, Uses}}(nothing, n), none(n),
            zeros(UInt, n), zeros(UInt, n), fill(false, n), none(n), 1, false, reached)
    end
    slots = Any[on_device(() -> allocate(target, UInt8, bytes), target, devices[s.devices[k]]) for (k, bytes) in enumerate(s.sizes)]
    referenced = Vector{Any}(nothing, n)
    xs = argument_leaves(program, inputs)
    for j in by_reference, k in leaves(program.input_structure[j])
        k === nothing || (referenced[k] = xs[k])
    end
    # Results in an argument's storage, and folds by key over the contents they
    # start from, take that storage, in the order values are computed: each
    # value's storage is its root's, and a value over an argument's storage
    # is that argument's.
    taken = sort!([s.donated; s.overwrites]; by = first)
    root = collect(eachindex(values))
    foreach(((r, a),) -> root[r] = root[a], taken)
    over(v) = (def = values[root[v]].definition) isa Input ? def.argument : 0
    # Each argument's storage, where its launches are built, as the first
    # binding gives it (see `prepare`'s `point`): the array bound by
    # reference; the caller's array where `point!` binds them, and the
    # kernels can read it where it is; else a buffer of the executable's own,
    # which that binding copies into. `pointed` keeps it as the caller has
    # it, and the launches read it as a vector.
    arguments = Vector{Any}(nothing, n)
    pointed = Vector{Any}(nothing, n)
    own(k) = (v = values[input[k]]; on(() -> allocate(target, v.eltype, storage_length(v)), input[k]))
    for (k, id) in input
        x, v = xs[k], values[id]
        if referenced[k] !== nothing
            bindable(target, x) && eltype(x) === v.eltype && length(x) == storage_length(v) ||
                throw(ArgumentError("input $k is bound by reference, but $(typeof(x)) is not storage on $target"))
            arguments[k] = pointed[k] = vec(x)
        elseif point && on(() -> pointable(target, x, address_of(x)), id) && eltype(x) === v.eltype && length(x) == storage_length(v)
            pointed[k] = x
        else
            arguments[k] = pointed[k] = own(k)
        end
    end
    buffers = Vector{Any}(nothing, length(values))
    for (id, v) in enumerate(values)
        root[id] == id || continue
        def = v.definition
        if def isa ConstantArray
            # Uploaded already if a claim was measured: the decider times
            # at home. A constant read on several devices is a value on each.
            buffers[id] = p.graph.devices[id] == 1 ? get(d.buffers, id, nothing) : nothing
            buffers[id] === nothing && (buffers[id] = on(() -> upload(target, vec(collect(def.data))), id))
        elseif s.slots[id] > 0
            buffers[id] = slot_view(slots[s.slots[id]], v.eltype, storage_length(v))
        elseif p.stored[id] && !(def isa Input)
            buffers[id] = on(() -> allocate(target, v.eltype, storage_length(v)), id)
        end
    end
    # The values over an argument's storage take it, the others their root's.
    function share!(buffers)
        for v in eachindex(values)
            a = over(v)
            a == 0 ? root[v] == v || (buffers[v] = buffers[root[v]]) : (buffers[v] = vec(pointed[a]))
        end
        buffers
    end
    share!(buffers)
    # One buffer per input saved, however many results are it.
    befores = Dict(v => on(() -> allocate(target, values[v].eltype, storage_length(values[v])), v) for v in unique(Base.values(saved)))
    replaced = Dict(pairs)
    # What its launches share and nothing else uses (see `Resources`), per
    # device, made as the first that needs it is built; the launches hold it.
    # Each group is built on its device.
    resources = [Resources(target) for _ in devices]
    built_on(g, x) = on_device(() -> build_group(d, p, g, x, buffers, resources[group_device(p, g)]), target, devices[group_device(p, g)])
    build() = concurrently(((g, x),) -> built_on(g, x), collect(zip(p.groups, ds)), target)
    built = build()
    # An argument's storage, read or written, can be the caller's array when
    # every launch can be pointed elsewhere, unless the caller's array is the
    # storage already, bound by reference. Where a launch cannot be, an
    # argument built on the caller's array has a buffer of its own instead,
    # its storage for good, and the launches are built again on it.
    movable = all(b -> repoints(b[1]), built)
    if !movable && any(k -> arguments[k] === nothing && pointed[k] !== nothing, 1:n)
        foreach(k -> arguments[k] === nothing && pointed[k] !== nothing && (arguments[k] = pointed[k] = own(k)), 1:n)
        built = (share!(buffers); build())
        movable = all(b -> repoints(b[1]), built)
    end
    clock = lap!(:build, clock)
    launches = Any[b[1] for b in built]
    saves = Any[(argument = values[v].definition.argument, before = b, copy = Copy(b, buffers[v], 0 => v)) for (v, b) in befores]
    copies = Copy[Copy(buffers[a], buffers[r], a => r) for (r, a) in s.staged]
    # Each step on its device: an input saved on the input's, a launch on its
    # group's, reading from the device its group reads from, another for a
    # crossing (see `source_device`), and a staged result on its argument's.
    steps = Step[Step(save.copy, device(last(save.copy.values))) for save in saves]
    append!(steps, Step(l, devices[group_device(p, g)], devices[source_device(p, g)]) for (l, g) in zip(launches, p.groups))
    append!(steps, Step(c, device(first(c.values))) for c in copies)
    # The storage each value lives in: its slot, else the argument's it is
    # over, else its root's own buffer.
    storage_of(v) = s.slots[v] > 0 ? (:slot, s.slots[v]) : over(v) > 0 ? (:argument, over(v)) : (:value, root[v])
    touches = step_touches(p, saves, copies, storage_of)
    schedule = Schedule(target, steps, touches)
    stored(k) = haskey(replaced, k) ? vec(pointed[replaced[k]]) : haskey(saved, k) ? befores[saved[k]] : buffers[result_value(k)]
    results = Any[reshape(stored(k), p.graph.result_scalar[k] ? () : p.graph.result_dims[k]) for k in eachindex(p.graph.results)]
    # Where each pointable argument's storage is used: the steps that use
    # it, by the values they name it by, in the order of `steps`, and the
    # results over it.
    used = Vector{Union{Nothing, Uses}}(nothing, n)
    if movable
        foreach(a -> referenced[a] === nothing && (used[a] = (steps = Tuple{Int, Int}[], results = Int[])), keys(input))
        named = [[[last(save.copy.values)] for save in saves]; [kernel_slots(p, g) for g in p.groups]; [[first(c.values)] for c in copies]]
        for (i, vs) in enumerate(named), v in vs
            a = over(v)
            a == 0 || used[a] === nothing || push!(used[a].steps, (i, v))
        end
        for k in eachindex(results)
            a = haskey(replaced, k) ? replaced[k] : haskey(saved, k) ? 0 : over(result_value(k))
            a == 0 || used[a] === nothing || push!(used[a].results, k)
        end
    end
    # The executable reaches an argument's storage through `pointed` alone
    # (see `point!`), and holds no caller's array once bound to another. A
    # launch may hold `buffers` itself, as a recurrence's steps do.
    kept = Any[over(v) == 0 ? buffers[v] : nothing for v in eachindex(values)]
    foreach(a -> used[a] === nothing && (pointed[a] = nothing), 1:n)
    # An array's address is taken on its device, where CUDA's arrays let a task take it.
    at = UInt[x === nothing ? 0 : on(() -> address_of(x), input[k]) for (k, x) in enumerate(pointed)]
    lap!(:executable, clock)
    Executable(program, p, target, s, slots, kept, arguments, referenced, results, outputs_over(program, results),
        launches, Any[b[2] for b in built], Any[b[3] for b in built], Any[considered...], saves, copies, steps, touches, schedule, nothing,
        devices, argument_devices, result_devices, Vector{Any}(nothing, length(devices)), used, pointed, at, zeros(UInt, n), fill(false, n), Vector{Any}(nothing, n), 1, true, reached)
end

# The address of array `x`'s first element, as a number, as `pointer` gives
# it; 0 for an array without one.
address_of(x) = x isa StridedArray ? reinterpret(UInt, pointer(x)) : UInt(0)

"""
    taken!(ex) -> ex

`ex` taken for a use by the calling task: what every use of an executable
begins with, a binding (`bind!`, `point!`, and the one `prepare` makes), an
execution, a capture and a replay, and so each call of a compiled function.

An executable is used by one stream at a time on each of its devices. Its
kernels read and write its storage by address, in launches the arrays'
package does not see, so nothing orders what a task launches after what
another task's stream still has to run there, a binding's copy into an
argument's buffer or an execution's kernels: a program prepared, and so
bound, on one task and executed on another read its input before the copy
had run. So on each device, a task whose stream (see `task_stream`) is not
the one that used the executable last waits for that one (`quiesce`), and
is the one that used it last from then on; a task on the same stream
compares the two and goes on. What a call leaves on the lanes of its
crossings its own streams wait for as it ends (see `walk`), and so the
stream that waits for them waits for the lanes. The arrays the executable
shares with its caller are taken one by one, before each run's launches
(see `take_shared!`).

Uses in turn, from any tasks, each find what the uses before left. Two
tasks that use one executable at the same moment are not ordered by this:
they take turns themselves, as a compiled function's calls do under its
lock. A program prepared with `build = false` has nothing to use, and
throws.
"""
function taken!(ex::Executable)
    ex.built || throw(ArgumentError("the program was prepared with build = false: it can be explained, not run"))
    for k in eachindex(ex.devices)
        on_device(() -> take_stream!(ex, k), ex.target, ex.devices[k])
    end
    ex
end

# With the task on device `k` of those of `ex`: the task's stream there is
# the one that used `ex` last, once the one that did has run its work.
function take_stream!(ex::Executable, k)
    now = task_stream(ex.target)
    last = ex.streams[k]
    now === last && return nothing
    last === nothing || quiesce(ex.target, last)
    ex.streams[k] = now
    nothing
end

# Take, each on its device, the arrays `ex` shares with its caller, before
# a run's launches read and write them (see `taken`): its results, which
# `outputs` hands the caller, and the arguments bound by reference, the
# caller's own arrays. The arrays a call's arguments are pointed at are
# taken as the call finds them (see `located!`).
function take_shared!(ex::Executable)
    on_each_device(k -> taken(ex.target, ex.results[k]), k -> ex.result_devices[k], ex.target, eachindex(ex.results))
    all(isnothing, ex.referenced) ||
        on_each_device(a -> ex.referenced[a] === nothing || taken(ex.target, ex.referenced[a]), a -> ex.argument_devices[a], ex.target, eachindex(ex.referenced))
    ex
end

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

The bytes of the buffers a prepared program holds of its own (the inputs it
copies into, constants, results and slots), and what they would be with
each value stored apart instead of in slots. An argument read where the
caller has it, or bound by reference, holds none.
"""
function footprint(ex::Executable)
    graph, s = ex.plan.graph, ex.storage
    taken = Set(first.([s.donated; s.overwrites]))
    own = sum(v -> s.slots[v] == 0 && !(v in taken) && ex.buffers[v] !== nothing ? value_bytes(graph.values[v]) : 0, eachindex(graph.values); init = 0)
    own += sum(k -> ex.referenced[k] === nothing && ex.arguments[k] !== nothing ? sizeof(ex.arguments[k]) : 0, eachindex(ex.arguments); init = 0)
    own += sum(save -> sizeof(save.before), ex.saves; init = 0)
    apart = sum(v -> s.slots[v] > 0 ? value_bytes(graph.values[v]) : 0, eachindex(graph.values); init = 0)
    (bytes = own + sum(s.sizes; init = 0), apart = own + apart)
end

"""
    concurrently(f, xs, target) -> results

`f` of each of `xs`, in tasks that take turns on the calling thread, each in
the calling task's state on `target`. Groups and kernels are built this way:
while one task waits for its kernel, compiling on a thread of its own (see
`cached_kernel`), the others ask for theirs. Device objects are made on the
calling thread, so at exit they are finalized after what the caller made from
them later (a module unloaded before a captured graph that launches it
crashes the driver). It returns once every task has ended (see `ended_all`).
"""
function concurrently(f, xs, target)
    enter = task_state(target)
    ended_all([@async((enter(); f(x))) for x in xs])
end

"""
    side_by_side(f, xs, target) -> results

`f` of each of `xs` on the threads there are, each task in the calling task's
state on `target`, the results in the order of `xs`. For work that is of its
arguments alone and whose results the caller keeps: what implementations
offer for a group, and what they estimate of a claim. What device objects
such work makes are kept in tables of the session, as a lowering trial's
module and an attention library's graph are, and a captured graph that
launches one is destroyed at exit before any finalizer runs. It returns once
every task has ended (see `ended_all`).
"""
function side_by_side(f, xs, target)
    enter = task_state(target)
    ended_all([Threads.@spawn((enter(); f(x))) for x in xs])
end

# The results of `tasks`, in order, once every one has ended: what the first
# that failed threw is thrown only then. Thrown at once, the tasks after it
# would run on unwaited, a group's other kernels compiling beside the timings
# its stage takes once its builds return.
function ended_all(tasks)
    foreach(t -> try wait(t) catch end, tasks)
    map(fetch_result, tasks)
end

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
    taken!(ex)
    refuse_devices(ex.target, inputs, input_paths)
    bind_leaves!(ex, leaves(ex.program.input_structure), leaves(inputs), () -> input_paths(inputs))
end

function bind!(ex::Executable, bindings::Pair{<:Integer}...)
    taken!(ex)
    refuse_devices(ex.target, map(last, bindings), _ -> binding_paths(bindings))
    positions, xs = Any[], Any[]
    for (j, x) in bindings
        here, given = leaves(ex.program.input_structure[j]), leaves(x)
        length(here) == length(given) || throw(ArgumentError("expected $(length(here)) inputs, got $(length(given))"))
        append!(positions, here)
        append!(xs, given)
    end
    bind_leaves!(ex, positions, xs, () -> binding_paths(bindings))
end

# Copy each of `xs` into the buffer of the argument at the same place among
# `positions`, each on its argument's device; `paths()` names them, where
# one is on another device than the program was prepared with it on.
function bind_leaves!(ex, positions, xs, paths)
    length(positions) == length(xs) || throw(ArgumentError("expected $(length(positions)) inputs, got $(length(xs))"))
    given = [j for j in eachindex(positions) if positions[j] !== nothing]
    misplaced = [positions[j] for j in given if !read_on(ex.argument_devices[positions[j]], xs[j])]
    isempty(misplaced) || refuse_placed(ex, positions, xs, misplaced, paths())
    moves = Moves()
    function bind_one(j)
        k, x = positions[j], xs[j]
        ex.back[k] = nothing
        x === ex.referenced[k] && return
        own = own!(ex, k)
        point_at!(ex, k, own, moves)
        upload!(ex.target, own, vec(x))
    end
    on_each_device(bind_one, j -> ex.argument_devices[positions[j]], ex.target, given)
    retarget!(ex, moves)
end

"""
    on_each_device(f, device, target, items)

`f(item)` for each of `items` with the calling task on its device,
`device(item)`: the binding's counterpart to `walk`, which finds, binds and
writes back a call's arguments and returns its results each on its own
device, entering each device once, in the order their items first come,
and each one's items in order. A call on one device enters it once.
"""
function on_each_device(f, device, target::Target, items)
    isempty(items) && return nothing
    home = device(first(items))
    if all(x -> device(x) == home, items)
        on_device(() -> foreach(f, items), target, home)
        return nothing
    end
    entered = Union{Nothing, Int}[]
    for x in items
        d = device(x)
        d in entered && continue
        push!(entered, d)
        on_device(() -> foreach(y -> device(y) == d && f(y), items), target, d)
    end
    nothing
end

# Throw for the arguments `misplaced`, on other devices than the program was
# prepared with them on, naming each by its path among `paths`, as
# `positions`, the arguments of the leaves `xs`, place them.
function refuse_placed(ex, positions, xs, misplaced, paths)
    held = [string(paths[j], " is on ", device_named(device_of(xs[j])), ", where the program was prepared with it on ",
        device_named(ex.argument_devices[k])) for (j, k) in enumerate(positions) if k !== nothing && k in misplaced]
    throw(ArgumentError(string("arrays are on other devices than the program was prepared with them on: ", join(held, "; "),
        ". Bring each to its device, or prepare the program for where they are")))
end

device_named(d) = d === nothing ? "the host" : string("device ", d)

"""
    point!(ex, inputs...) -> ex

Bind inputs as `compile`'s call does: an argument's storage is the caller's
array, read and written where it is, when every launch can be pointed
elsewhere (`repoints`) and the target can read and write the array where it
is (`pointable`); the array is copied otherwise, as `bind!` copies, into a
buffer of the executable's own made the first time one is, and each
`execute!` or `replay!` after copies what the program wrote there back into
it. An argument bound by reference keeps its
array. An array the storage is already costs nothing. The steps that use the
storage of one elsewhere, the kernels that read or write it and the copies
of saved inputs and staged results, are pointed at it, the results over it
are over it, and a captured graph is changed where it records their
addresses (`retarget!`), or recorded again where the target cannot change
it; recording again costs more than copying an array that moves every call,
as a training step's weights do, so an argument whose graph was recorded
again is the caller's array only once the caller brings the same array twice
in a row. The executable holds the arrays it reads and writes until it is
bound again.
"""
point!(ex::Executable, @nospecialize(inputs...)) = point_each!(ex, bound(ex, inputs))

"""
A call's arrays, one per argument of a program, each with where it lives:
the address of its first element, as `pointer` gives it, 0 for an array
without one, and the addresses its elements span (see `extent`). One walk of
the call's leaves finds the arrays, `compile`'s call checking each against
its signature as it goes, and notes those on another device than the plan
put their argument on, `devices` (see `device_of`); the addresses are asked
after, of each array on its device (see `located!`): on a machine whose
devices read each other's memory, asking on another would open its memory
to that device's kernels. Binding the call (`point!`) and finding memory it
shares with an argument the program writes read them.
"""
struct Passed
    arrays::Vector{Any}
    at::Vector{UInt}
    extents::Vector{Union{Nothing, UnitRange{UInt}}}
    devices::Vector{Union{Nothing, Int}}
    elsewhere::Vector{Int}
end

# The arrays of a call of `ex`, none found yet.
Passed(ex::Executable) = (n = length(ex.argument_devices);
    Passed(Vector{Any}(nothing, n), zeros(UInt, n), Vector{Union{Nothing, UnitRange{UInt}}}(nothing, n), ex.argument_devices, Int[]))

# The arrays among `inputs`, given in the structure the program of `ex` was
# traced with, bound to its arguments, their addresses found. Arrays on
# other devices than the plan's throw.
Base.@nospecializeinfer function bound(ex::Executable, @nospecialize(inputs))
    p = ex.program
    positions, xs = leaves(p.input_structure), leaves(inputs)
    length(positions) == length(xs) || throw(ArgumentError("expected $(length(positions)) inputs, got $(length(xs))"))
    b = Passed(ex)
    for (k, x) in zip(positions, xs)
        k === nothing || place!(b, k, x)
    end
    isempty(b.elsewhere) || refuse_elsewhere(ex, b, positions, inputs, input_paths)
    located!(b, ex, inputs, input_paths, "the program")
end

# Array `x` as argument `k` of `b`, noted where it is on another device
# than the plan put the argument on.
function place!(b::Passed, k::Int, @nospecialize(x))
    b.arrays[k] = x
    read_on(b.devices[k], x) || push!(b.elsewhere, k)
    b
end

# Throw for the arrays of `b` on other devices than the plan's, among
# `inputs`, the arguments of whose leaves are `positions`: on a target that
# plans on one device, as a program over several, not planned there (see
# `refuse_devices`); else naming each and the device it belongs on.
function refuse_elsewhere(ex, b::Passed, positions, @nospecialize(inputs), paths)
    refuse_devices(ex.target, inputs, paths)
    refuse_placed(ex, positions, leaves(inputs), b.elsewhere, paths(inputs))
end

# The addresses of the arrays of `b`, each asked on its argument's device;
# arrays that share memory with one the program writes throw, named by
# their paths among `inputs` and the program by `writer`. Every binding of
# a call's arrays, `point!`'s and `compile`'s call, finds them here, and
# an array asked its address is taken by it for the launches that follow,
# as a result is (see `taken`).
function located!(b::Passed, ex::Executable, @nospecialize(inputs), paths, writer)
    on_each_device(k -> locate!(b, k, b.arrays[k]), k -> b.devices[k], ex.target, eachindex(b.arrays))
    isempty(ex.program.writes) || refuse_shared(ex, b, inputs, paths, writer)
    b
end

# A call whose arrays share memory with one the program writes throws: the
# program reads each array as one of its own, which the function itself,
# run on the same arrays, would not, as `f!(x, x)` writing its first
# argument and reading its second reads what it wrote; and the steps of an
# execution are ordered by the storage each touches, an argument's its own,
# so that a crossing reading one array and a kernel writing another over
# the same memory would not be ordered. Which arrays may
# share memory is Base's `mightalias`, as broadcasting's, asked only of two
# that the addresses their elements span (see `extent`) do not show apart:
# the arrays sorted by where they start find in one pass every two whose
# spans meet, none in a call of arrays each its own allocation, and two
# rows of one matrix, which lie between each other and share no element,
# are asked and pass. An array whose addresses are not known is asked
# beside every other. `b` holds the call's arrays.
function refuse_shared(ex::Executable, b::Passed, @nospecialize(inputs), paths, writer)
    xs = b.arrays
    written = falses(length(xs))
    foreach(((_, a),) -> written[a] = true, ex.program.writes)
    refuse(k, a) = refuse_sharing(ex, paths(inputs), k, a, writer)
    function ask(k, j)
        (written[k] || written[j]) && Base.mightalias(xs[k], xs[j]) || return
        written[j] ? refuse(k, j) : refuse(j, k)
    end
    spans = Tuple{UInt, UInt, Int}[]
    unknown = Int[]
    for (k, r) in enumerate(b.extents)
        r === nothing ? push!(unknown, k) : isempty(r) || push!(spans, (first(r), last(r), k))
    end
    sort!(spans)
    # The spans begun and not ended, each by where it ends.
    open = Tuple{UInt, Int}[]
    for (start, stop, k) in spans
        filter!(((ends, _),) -> ends >= start, open)
        foreach(((_, j),) -> ask(k, j), open)
        push!(open, (stop, k))
    end
    for u in unknown, k in eachindex(xs)
        k == u || ask(u, k)
    end
end

# Argument `k` shares memory with argument `a`, which `writer` writes: each
# named by its path among `paths`, as the program's inputs place them.
function refuse_sharing(ex, paths, k, a, writer)
    positions = leaves(ex.program.input_structure)
    path(j) = paths[findfirst(==(j), positions)]
    throw(ArgumentError(string(path(k), " shares memory with ", path(a), ", which ", writer,
        " writes: the program reads its arrays apart, so pass arrays that share none")))
end

# Where array `x`, argument `k` of `b`, lives: one call a leaf, which asks
# once. Compiled once whatever the leaf's type, as the binding's other walks
# of the leaves are: a dispatch of a few tens of nanoseconds a leaf, where a
# body for each type of array compiled again for each program.
function locate!(b::Passed, k::Int, @nospecialize(x))
    if x isa StridedArray
        at = reinterpret(UInt, pointer(x))
        b.at[k], b.extents[k] = at, extent(x, at)
    else
        b.extents[k] = extent(x)
    end
    nothing
end

# Whether a program that put an argument on `device` reads array `x` there:
# an array on the host, which is copied where it cannot be the storage, or
# on that device; a program on the host's memory alone reads a device's
# arrays by copying them too.
read_on(device, @nospecialize(x)) = (d = device_of(x)) === nothing || device === nothing || d == device

# Throw where the arrays among `inputs` and `target` are on more than one
# device, beside the host (see `device_of`), naming each array by its path,
# as `paths(inputs)` gives them in the order of `leaves`: a program over
# several devices is not planned yet. Asked before anything reads the arrays.
function refuse_devices(target, @nospecialize(inputs), paths)
    plans_across_devices(target) && return
    xs = leaves(inputs)
    devices = Union{Nothing, Int}[device_of(x) for x in xs]
    placed = unique(filter(!isnothing, [device_of(target); devices]))
    length(placed) <= 1 && return
    names = paths(inputs)
    held = [string(names[k], " on device ", d) for (k, d) in enumerate(devices) if d !== nothing]
    on_target = device_of(target) === nothing ? "" : string(", and the target on device ", device_of(target))
    throw(ArgumentError(string("the arrays are on more than one device: ", join(held, ", "), on_target,
        "; a program over several devices is not planned yet, so bring its arrays to one")))
end

# The path of each leaf of `inputs`, as a preparation takes them, in the
# order `leaves` gives them.
Base.@nospecializeinfer function input_paths(@nospecialize(inputs::Tuple))
    out = String[]
    for k in 1:nfields(inputs)
        leaf_paths!(out, getfield(inputs, k), string("input ", k))
    end
    out
end

# The path of each leaf of the inputs `bindings` give by position, as `bind!` takes them.
Base.@nospecializeinfer function binding_paths(@nospecialize(bindings::Tuple))
    out = String[]
    for k in 1:nfields(bindings)
        binding = getfield(bindings, k)
        leaf_paths!(out, getfield(binding, 2), string("input ", getfield(binding, 1)))
    end
    out
end

# The addresses the elements of array `x` span, its first byte to its last,
# along its strides from `at`, its first element's: none for an array with
# no elements, or that holds no memory, as a range; those of the array it
# wraps where it has no strides, as a transpose of one; `nothing` where
# neither is known.
function extent(x::StridedArray, at = reinterpret(UInt, pointer(x)))
    isempty(x) && return UInt(1):UInt(0)
    start = stop = at
    for (n, s) in zip(size(x), strides(x))
        reach = (n - 1) * s * Base.elsize(x)
        reach < 0 ? (start -= UInt(-reach)) : (stop += UInt(reach))
    end
    start:(stop + UInt(Base.elsize(x)) - UInt(1))
end

function extent(x)
    isbits(x) && return UInt(1):UInt(0)
    p = parent(x)
    p === x ? nothing : extent(p)
end

# Bind the arrays of `b`, as `point!` binds a call's inputs.
function point_each!(ex::Executable, b::Passed)
    taken!(ex)
    xs = b.arrays
    length(xs) == length(ex.arguments) || throw(ArgumentError("expected $(length(ex.arguments)) arrays, got $(length(xs))"))
    moves = Moves()
    on_each_device(k -> point_leaf!(ex, k, xs[k], b.at[k], moves), k -> ex.argument_devices[k], ex.target, eachindex(xs))
    for (_, a) in ex.program.writes
        x = xs[a]
        ex.back[a] = x === ex.pointed[a] || x === ex.referenced[a] ? nothing : x
    end
    retarget!(ex, moves)
end

# Bind input `x`, whose first element is at `at`, to argument `k`, as
# `point!` binds each: one call a leaf, a model's hundreds of arrays each
# checked in a few hundred nanoseconds, by one body whatever the leaf's type
# (see `locate!`).
function point_leaf!(ex::Executable, k::Int, @nospecialize(x), at::UInt, moves::Moves)
    if ex.pointable[k] !== nothing && pointable(ex.target, x, at)
        again = at == ex.seen[k]
        ex.seen[k] = at
        at == ex.at[k] && return
        (again || !ex.settles[k]) && return point_at!(ex, k, x, moves)
    end
    x === ex.referenced[k] && return
    own = own!(ex, k)
    point_at!(ex, k, own, moves)
    upload!(ex.target, own, vec(x))
    nothing
end

# The buffer of the executable's own that a binding copies argument `k` into,
# made when a binding first does: an argument read and written where the
# caller has it needs none (see `prepare`'s `point`).
function own!(ex::Executable, k)
    b = ex.arguments[k]
    b === nothing || return b
    values = ex.plan.graph.values
    v = values[findfirst(v -> v.definition isa Input && v.definition.argument == k, values)]
    ex.arguments[k] = allocate(ex.target, v.eltype, storage_length(v))
end

# After a run, what the program wrote into an argument bound to an array
# that is not its storage (see `point!`), an array not of the target's array
# type, or not as aligned as its kernels assume, or one waiting to be brought
# twice in a row, is copied into that array, from the storage it wrote, on
# the argument's device.
function write_back!(ex::Executable)
    back(a) = (x = ex.back[a]; x === nothing || download!(ex.target, vec(x), ex.arguments[a]))
    on_each_device(back ∘ last, w -> ex.argument_devices[last(w)], ex.target, ex.program.writes)
    ex
end

# Make array `x`, its buffer or the caller's array, argument `k`'s storage,
# adding to `moves` what moved for each step that uses it (see `Moves`),
# the storage the step uses from now on being what `retarget!` points it
# at. The results over the storage are over `x`.
function point_at!(ex::Executable, k, x, moves::Moves)
    uses = ex.pointable[k]
    uses === nothing && return
    to = address_of(x)
    to == ex.at[k] && return
    buffer = vec(x)
    for (i, v) in uses.steps
        push!(moves, (i, k, (value = v, from = ex.at[k], to, bytes = UInt(sizeof(x))), buffer))
    end
    if !isempty(uses.results)
        foreach(r -> ex.results[r] = reshape(buffer, size(ex.results[r])), uses.results)
        ex.outputs = outputs_over(ex.program, ex.results)
    end
    ex.pointed[k] = x
    ex.at[k] = to
    nothing
end

# The results in the program's output structure, as `outputs` gives them.
outputs_over(program, results) = rebuild(k -> k isa Int ? results[k] : constant_leaf(k), program.output_structure)

# Point the steps at their new storage, and change the recording where it
# records their addresses, step by step, each in its segment, or record the
# segment again where the target cannot change it; a step a recording holds
# in several segments, an execution over several devices repeated, is
# recorded again in each. An argument whose segment was recorded again
# waits from then on to be brought twice in a row.
function retarget!(ex::Executable, moves::Moves)
    isempty(moves) && return ex
    sort!(moves; by = first)
    r = ex.captured
    stale = Set{Int}()
    start = 1
    while start <= length(moves)
        i = moves[start][1]
        stop = start
        while stop < length(moves) && moves[stop + 1][1] == i
            stop += 1
        end
        mine = view(moves, start:stop)
        # The step is pointed at the new storage once, by the target or after
        # it, with the task on the step's device, where its launches take
        # their buffers' addresses.
        pointed = Ref(false)
        point_step() = foreach(m -> repoint!(ex.steps[i].run, m[3].value, m[4]), mine)
        repoint() = pointed[] || (on_device(point_step, ex.target, ex.steps[i].device); pointed[] = true)
        if r !== nothing
            at = recorded_steps(ex, i)
            held = unique(first(r.at[j]) for j in at)
            segment = r.segments[first(held)]
            changed = length(held) == 1 && on_stream(ex.target, segment.stream) do
                retarget!(ex.target, segment.graph, [last(r.at[j]) for j in at], [m[3] for m in mine], repoint)
            end
            if !changed
                foreach(m -> ex.settles[m[2]] = true, mine)
                union!(stale, held)
            end
        end
        repoint()
        start = stop + 1
    end
    foreach(k -> recapture_segment!(ex.target, r, r.segments[k], ex.steps), sort!(collect(stale)))
    ex
end

# What a graph records, step by step: each of the executions a replay runs.
recorded(ex::Executable) = repeat(ex.steps, ex.repetitions)

# Where step `i` of an execution is among what a graph records, once in each execution.
recorded_steps(ex::Executable, i) = i .+ length(ex.steps) .* (0:(ex.repetitions - 1))

"""
    execute!(ex) -> ex

Run the program on the inputs bound last: copy the inputs it writes into
the results that are them as they were before, launch every kernel in order,
then copy staged results into the arguments they replace, and what it wrote
into an argument into the caller's array `point!` bound it to, where that is
not its storage. The results are in `outputs(ex)`. Whether or not the
program is captured, this launches kernel by kernel; `replay!` runs the
capture.
"""
function execute!(ex::Executable)
    taken!(ex)
    take_shared!(ex)
    walk(ex.target, ex.schedule)
    write_back!(ex)
end

"""
    capture!(ex; repetitions = 1) -> ex

Record what `execute!` launches, for `replay!` to run, each replay
`repetitions` executions of the program (for timing): as one graph on one
device, and over several as a graph per run of the schedule of the
executions, the runs of a device's steps and of a lane's crossings (see
`Captured`). Captured runs
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
    taken!(ex)
    release!(ex)
    ex.repetitions = repetitions
    ex.captured = record(ex.target, Schedule(ex.target, recorded(ex), repeat(ex.touches, repetitions)), ex.steps)
    ex
end

# Destroy what `capture!` recorded of `ex`, once nothing replays it.
function release!(ex::Executable)
    ex.captured === nothing || release_recording(ex.target, ex.captured)
    ex.captured = nothing
    ex
end

"""
    replay!(ex) -> ex

Run what `capture!` recorded, on the inputs bound last: what `execute!` does,
`repetitions` times over, as one launch of a graph on a target with graphs,
then copy what it wrote into an argument into the caller's array `point!`
bound it to, where that is not its storage. The results are in `outputs(ex)`.
"""
function replay!(ex::Executable)
    taken!(ex)
    ex.captured === nothing && throw(ArgumentError("nothing to replay: capture! records the program first"))
    take_shared!(ex)
    replay_recording(ex.target, ex.captured)
    write_back!(ex)
end

function explain(io::IO, ex::Executable)
    explain(io, ex.plan)
    length(ex.devices) > 1 && println(io, "devices by label: ", join([string(l, " is ", device_named(d)) for (l, d) in enumerate(ex.devices)], ", "))
    ex.target isa CUDATarget && println(io, "products in ", ex.target.precision === :tf32 ? "TF32 where Float32" :
        ex.target.precision === :tf32x3 ? "three TF32 products where Float32" : "full precision",
        "; half-precision products accumulate in Float32",
        ex.target.online ? "; softmax online where a sum is weighted by it" : "; no online softmax")
    foreach(c -> explain(io, ex.plan, c), ex.considered)
    explain(io, ex.plan, ex.storage)
    foreach(save -> println(io, "  input ", save.argument, " copied apart before the kernels, as results return it before the program writes it"), ex.saves)
    for a in (ex.built ? unique(last.(ex.program.writes)) : ())
        println(io, "  input ", a, ex.referenced[a] !== nothing ? " is the array bound by reference" :
            ex.pointable[a] !== nothing ? " is the caller's array where the target can write it there, else copied in and back" :
            " copied in and back: a kernel cannot be pointed at the caller's array")
    end
    is = implementations(ex.target)
    println(io, isempty(is) ? "the floor alone:" : string("implementations ", join(map(repr, is), ", "), " and the floor:"))
    for (k, d) in enumerate(ex.kernels)
        x = ex.decisions[k]
        println(io, "  kernel ", k, ": ", ex.built ? d : "not built", x === nothing ? "" : string(" (", x, ")"))
        x === nothing || foreach(f -> println(io, "    ", f.implementation == label(nothing) ? "the floor" : string("a claim of ", f.implementation), " failed to build: ", f.reason), x.failed)
    end
end

explain(ex::Executable) = explain(stdout, ex)

# What runs kernel `k`: an implementation or the floor.
function runner(ex::Executable, k)
    d = ex.decisions[k]
    label(d === nothing ? nothing : d.choice.implementation)
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
`replay!` fills again. Copy what you keep: a read or a copy of one, on any
task, waits for the run that filled it, and the next run for the copy (see
`taken!`). A number is an array of no dimensions here, since it lives in a
buffer too; `compile` returns numbers.
"""
outputs(ex::Executable) = ex.outputs

# The buffers a group's kernels take, in the order the group first mentions
# their values (`appearance`), as `canonical` numbers them: groups alike take
# theirs in one order, and generate one kernel, however the values are numbered.
function kernel_slots(p, g, stored = p.stored)
    slots = union(group_reads(p.graph, g), [v for v in members(g) if stored[v]])
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
starts_in_place(graph, r, buffers) = graph.values[r].definition.into isa Read && same_storage(buffers[graph.values[r].definition.into.value], buffers[r])

"""
    same_storage(a, b)

Whether buffers `a` and `b` are one storage: the same elements at the same
address, whichever array holds them. Values over one argument's storage each
hold a vector of the caller's array, made apiece (see `share!`), so that two
of them are one storage and never one object.
"""
same_storage(a, b) = a === b || a !== nothing && b !== nothing && typeof(a) == typeof(b) && length(a) == length(b) && pointer(a) == pointer(b)

# The floor for a group on any target: its scalar kernels, each launched as
# the target launches one (see `floor_launch`), their folds split as it asks
# (see `floor_split`). A fold by key in the storage of the contents it starts
# from starts from them there.
function build_floor(target, p::Plan, g::Group, buffers)
    held = filter(r -> keyed(p.graph, r) && starts_in_place(p.graph, r, buffers), g.reductions)
    stored = floor_stored(p, g)
    ks = scalar_kernels(p.graph, stored, g, kernel_slots(p, g, stored); held, split = (items, positions, adjacent) -> floor_split(target, items, positions, adjacent))
    spaces = Dict{Int, Any}()
    space(w) = get!(() -> allocate(target, w.state, w.length), spaces, w.fold)
    # What the floor keeps that the plan does not store, in storage of its own.
    own = Dict{Int, Any}()
    buffer(v) = buffers[v] !== nothing || p.stored[v] ? buffers[v] :
        get!(() -> allocate(target, p.graph.values[v].eltype, storage_length(p.graph.values[v])), own, v)
    launches = Any[floor_launch(target, k, (Tuple(buffer(v) for v in k.slots)..., map(space, k.workspaces)...)) for k in ks]
    (FloorRun(launches, [collect(values(spaces)); collect(values(own))]), FloorKernels(ks))
end

"""
    floor_run(kind, target, p, g, buffers) -> (run, description)

What runs group `g` of plan `p` on `buffers` where no implementation's
claim does, by its kind: the floor's kernels (see `build_floor`), and for a
crossing the target's copy between its two devices (see `crossing`), which
no kernel of the floor's makes.
"""
floor_run(::GroupKind, target, p::Plan, g::Group, buffers) = build_floor(target, p, g, buffers)

function floor_run(::CrossingKind, target, p::Plan, g::Group, buffers)
    c = only(g.outputs)
    from = crossed_from(p.graph, c)
    (crossing(target, buffers[c], buffers[from], c => from, crossed_offset(p.graph, c)), CrossingCopy(value_bytes(p.graph.values[c])))
end

"A crossing as the target runs it: a copy of `bytes` between two devices."
struct CrossingCopy
    bytes::Int
end

Base.show(io::IO, c::CrossingCopy) = print(io, "the target's copy of ", Base.format_bytes(c.bytes), " between devices")

# What a group's floor kernels store: what the plan stores, and for a group
# whose members read its values where a step stored them (see `stepwise`),
# a recurrence's, every member, its states among them, which its kernel
# reads from storage at the next step where a claim's would hold them.
function floor_stored(p::Plan, g::Group)
    stepwise(g.kind) || return p.stored
    stored = copy(p.stored)
    foreach(m -> stored[m] = true, g.outputs)
    stored
end
