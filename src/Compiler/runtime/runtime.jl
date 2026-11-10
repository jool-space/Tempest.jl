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
# an execution (see `execution`) that reads or writes it, by its place among
# them, with the value the step names it by, the argument's own or a value
# written over it, a result in place of the argument or a fold by key; and
# the results over it.
const Uses = @NamedTuple{steps::Vector{Tuple{Int, Int}}, results::Vector{Int}}

# A value moved, as `retarget!` takes it: its old and new addresses and its bytes.
const Move = @NamedTuple{value::Int, from::UInt, to::UInt, bytes::UInt}

"""
A step of an execution that copies one buffer into another: an input saved
apart before the kernels, or a staged result into the argument it replaces
after them. It repoints as a launch does (see `repoint!`), by the values it
copies to and from, 0 for a buffer of no value.
"""
mutable struct Copy
    to::Any
    from::Any
    const values::Pair{Int, Int}
end

(c::Copy)() = (copyto!(c.to, c.from); nothing)

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
    # The graph `execute!` replays, once captured (see `capture!`).
    captured::Any
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
    prepare(program, inputs...; target, measure = :close, replaces = (), by_reference = (), point = false, recompute = (), ride = (), build = true, verify) -> Executable

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

What the cover decided, the proposals it took and the claim that runs each
kernel, is kept beside the tuning record (see `Tempest.Compiler.reach!`),
and a preparation of the same program, on the same target and device, with
the record and the rates file holding what those decisions read, replays it:
it plans the program once, and plans no trial, prices and measures nothing.
Anything else a decision reads that differs, the program's extents, element
types or captured constants, the target's numerics, Tempest's own code or a
time recorded since, makes them the cover's to reach again. Decisions are
replayed only where the cover would reach them again on the record as it is:
a preparation that measured as it decided covers again on what it measured,
until a cover measures nothing, three covers at most, and keeps the last
one's; with the environment's `TEMPEST_SETTLE=false` it keeps the first's,
which the next preparation reaches again. With `TEMPEST_REPLAY=false` a
preparation neither replays decisions nor keeps its own: the cover reaches
them every time. A fresh `TEMPEST_CACHE` holds no decisions, no record and
no kernels.

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

`bind!` copies inputs into storage the executable owns, except the arrays at
the positions in `by_reference`, which must be of the target's array type:
they are the executable's storage. The caller keeps them alive and may write
them between executions, and a result that replaces one writes into it.
`point!` makes the caller's arrays the storage at each binding instead, as
`compile` does. The preparation binds `inputs` as `bind!` does, or as
`point!` does with `point = true`, as `compile` prepares: the kernels are
built on the caller's arrays where they can read them there, and the
executable makes no storage of its own for those arguments until a binding
copies into it, holding the arrays as `point!` does, until it is bound to
others.

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
Where it sets `TEMPEST_CROSSCHECK=true`, as the tests do too, decisions a
preparation replays are checked against those its cover reaches, which it
reaches as well, at a cover's cost.

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
"""
function prepare(program::Program, @nospecialize(inputs...); target = default_target(inputs), measure = measuring(),
        replaces = (), by_reference = (), point::Bool = false, recompute = (), ride = (), partition::Bool = true, build::Bool = true,
        verify::Bool = verifying())
    # The terms an earlier preparation interned are of its program.
    IR.forget_terms!()
    forget_matches!()
    s = setup(program; target, measure, replaces, recompute, ride, partition, build)
    prepared(s, decide!(s; verify), inputs...; by_reference, point, verify)
end

"""
A preparation set up, before it decides: the program, its decider as
`prepare` makes it, on the program's index graph lowered and rewritten with
the folds pinned, kept and riding, the results paired with the arguments
they replace, the value of each argument, and whether it builds. `decide!`
reaches its decisions and `prepared` builds them.
"""
struct Setup{D}
    program::Program
    decider::D # a `Decider`, defined with the cover
    pairs::Vector{Pair{Int, Int}}
    input::Dict{Int, Int}
    build::Bool
end

"""
    setup(program; target, measure, replaces, recompute, ride, partition, build) -> Setup

`program` set up as `prepare` sets it up, the keywords as `prepare` takes
them: lowered, rewritten and partitioned, with its decider. It forgets
nothing an earlier preparation interned or matched, as `prepare` does first.
"""
function setup(program::Program; target, measure = measuring(), replaces = (), recompute = (), ride = (), partition::Bool = true,
        build::Bool = true)
    # Each phase marked (see `phase`).
    lowered = phase(() -> lower(program), :lower)
    graph = permits_rewrites(target) ? phase(() -> rewrite(lowered; rounded = rounded_backward(target)), :rewrite) : lowered
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
        saved, recomputed = phase(() -> Compiler.partition(graph), :partition)
        filter!(!reads_replaced, recomputed)
        foreach(v -> push!(graph.rewrites, v => "recomputed in the backward where it is read: saving it would cost more than its operands"), recomputed)
        union!(pinned, recomputed)
    end
    d = phase(() -> Decider(target, graph; measure = build ? measure : :dry, recompute = pinned, keep = kept, ride), :decider)
    Setup(program, d, collect(Pair{Int, Int}, pairs), input, build)
end

"""
    decide!(s; verify) -> Decided

The decisions of the preparation set up as `s` and the plan they form,
reached as `prepare` reaches them (see `reach!`): replayed where decisions
kept for it hold, else by the cover, settled.
"""
decide!(s::Setup; verify::Bool = verifying()) =
    Base.ScopedValues.with(Implement.COMPILING => s.build, VERIFYING => verify) do
        reach!(s.decider; settle = settling())
    end

"""
    given(s, r) -> Decided

Decisions `r`, reached for the preparation set up as `s` by a search
elsewhere in this process, and the plan they form: replayed without what
they rest on held against the files (see `given!`).
"""
given(s::Setup, r) = given!(s.decider, r)

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
    ex = Base.ScopedValues.with(Implement.COMPILING => s.build, VERIFYING => verify) do
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
    p, order, s = storage(p, [result_value(k) => input[a] for (k, a) in pairs]; in_place = k -> writes_in_place(ds[k]))
    clock = lap!(:storage, clock)
    ds = ds[order]
    VERIFYING[] && verify(p, s; in_place = k -> writes_in_place(ds[k]))
    clock = lap!(:verify, clock)
    if !build
        none(n) = Vector{Any}(nothing, n)
        return Executable(program, p, target, s, Any[], none(length(values)),
            none(length(program.arguments)), none(length(program.arguments)), none(length(p.graph.results)), nothing,
            Any[], none(length(p.groups)), Any[ds...], Any[considered...], Any[], Copy[], nothing,
            Vector{Union{Nothing, Uses}}(nothing, length(program.arguments)), none(length(program.arguments)),
            zeros(UInt, length(program.arguments)), zeros(UInt, length(program.arguments)), fill(false, length(program.arguments)),
            none(length(program.arguments)), 1, false, reached)
    end
    slots = Any[allocate(target, UInt8, bytes) for bytes in s.sizes]
    n = length(program.arguments)
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
    own(k) = (v = values[input[k]]; allocate(target, v.eltype, storage_length(v)))
    for (k, id) in input
        x, v = xs[k], values[id]
        if referenced[k] !== nothing
            bindable(target, x) && eltype(x) === v.eltype && length(x) == storage_length(v) ||
                throw(ArgumentError("input $k is bound by reference, but $(typeof(x)) is not storage on $target"))
            arguments[k] = pointed[k] = vec(x)
        elseif point && pointable(target, x, address_of(x)) && eltype(x) === v.eltype && length(x) == storage_length(v)
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
            # Uploaded already if a claim was measured.
            buffers[id] = get(d.buffers, id, nothing)
            buffers[id] === nothing && (buffers[id] = upload(target, vec(collect(def.data))))
        elseif s.slots[id] > 0
            buffers[id] = slot_view(slots[s.slots[id]], v.eltype, storage_length(v))
        elseif p.stored[id] && !(def isa Input)
            buffers[id] = allocate(target, v.eltype, storage_length(v))
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
    befores = Dict(v => allocate(target, values[v].eltype, storage_length(values[v])) for v in unique(Base.values(saved)))
    replaced = Dict(pairs)
    # What its launches share and nothing else uses (see `Resources`), made
    # as the first that needs it is built; the launches hold it.
    resources = Resources(target)
    build() = concurrently(((g, x),) -> build_group(d, p, g, x, buffers, resources), collect(zip(p.groups, ds)), target)
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
    stored(k) = haskey(replaced, k) ? vec(pointed[replaced[k]]) : haskey(saved, k) ? befores[saved[k]] : buffers[result_value(k)]
    results = Any[reshape(stored(k), p.graph.result_scalar[k] ? () : p.graph.result_dims[k]) for k in eachindex(p.graph.results)]
    # Where each pointable argument's storage is used: the steps that use
    # it, by the values they name it by, and the results over it.
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
    at = UInt[x === nothing ? 0 : address_of(x) for x in pointed]
    lap!(:executable, clock)
    Executable(program, p, target, s, slots, kept, arguments, referenced, results, outputs_over(program, results),
        launches, Any[b[2] for b in built], Any[b[3] for b in built], Any[considered...], saves, copies, nothing, used, pointed, at, zeros(UInt, n),
        fill(false, n), Vector{Any}(nothing, n), 1, true, reached)
end

# The address of array `x`'s first element, as a number, as `pointer` gives
# it; 0 for an array without one.
address_of(x) = x isa StridedArray ? reinterpret(UInt, pointer(x)) : UInt(0)

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
        ex.back[k] = nothing
        x === ex.referenced[k] && continue
        own = own!(ex, k)
        moves = point_at!(ex, k, own, moves)
        upload!(ex.target, own, vec(x))
    end
    retarget!(ex, moves)
end

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
point!(ex::Executable, @nospecialize(inputs...)) = point_each!(ex, bound(ex.program, inputs))

"""
A call's arrays, one per argument of a program, each with where it lives:
the address of its first element, as `pointer` gives it, 0 for an array
without one, and the addresses its elements span (see `extent`). One walk of
the call's leaves finds them, `compile`'s call checking each against its
signature as it goes, and binding the call (`point!`) and finding memory it
shares with an argument the program writes read them.
"""
struct Passed
    arrays::Vector{Any}
    at::Vector{UInt}
    extents::Vector{Union{Nothing, UnitRange{UInt}}}
end

Passed(n::Int) = Passed(Vector{Any}(nothing, n), zeros(UInt, n), Vector{Union{Nothing, UnitRange{UInt}}}(nothing, n))

# The arrays among `inputs`, given in the structure program `p` was traced
# with, bound to its arguments.
Base.@nospecializeinfer function bound(p::Program, @nospecialize(inputs))
    positions, xs = leaves(p.input_structure), leaves(inputs)
    length(positions) == length(xs) || throw(ArgumentError("expected $(length(positions)) inputs, got $(length(xs))"))
    b = Passed(length(p.arguments))
    for (k, x) in zip(positions, xs)
        k === nothing || place!(b, k, x)
    end
    b
end

# Array `x` as argument `k` of `b`: one call a leaf, specialized on the
# leaf's type, which asks where it lives once.
function place!(b::Passed, k::Int, x)
    b.arrays[k] = x
    if x isa StridedArray
        at = reinterpret(UInt, pointer(x))
        b.at[k], b.extents[k] = at, extent(x, at)
    else
        b.extents[k] = extent(x)
    end
    b
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
    runnable(ex)
    xs = b.arrays
    length(xs) == length(ex.arguments) || throw(ArgumentError("expected $(length(ex.arguments)) arrays, got $(length(xs))"))
    moves = nothing
    for k in eachindex(xs)
        moves = point_leaf!(ex, k, xs[k], b.at[k], moves)
    end
    for (_, a) in ex.program.writes
        x = xs[a]
        ex.back[a] = x === ex.pointed[a] || x === ex.referenced[a] ? nothing : x
    end
    retarget!(ex, moves)
end

# Bind input `x`, whose first element is at `at`, to argument `k`, as
# `point!` binds each: one call a leaf, specialized on the leaf's type, a
# model's hundreds of arrays each checked in a few hundred nanoseconds.
function point_leaf!(ex::Executable, k::Int, x, at::UInt, moves)
    if ex.pointable[k] !== nothing && pointable(ex.target, x, at)
        again = at == ex.seen[k]
        ex.seen[k] = at
        at == ex.at[k] && return moves
        (again || !ex.settles[k]) && return point_at!(ex, k, x, moves)
    end
    x === ex.referenced[k] && return moves
    own = own!(ex, k)
    moves = point_at!(ex, k, own, moves)
    upload!(ex.target, own, vec(x))
    moves
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
# twice in a row, is copied into that array, from the storage it wrote.
function write_back!(ex::Executable)
    for (_, a) in ex.program.writes
        x = ex.back[a]
        x === nothing || download!(ex.target, vec(x), ex.arguments[a])
    end
    ex
end

# Make array `x`, its buffer or the caller's array, argument `k`'s storage,
# adding to `moves` (made on the first) what moved for each step that uses
# it: the step, the argument, the move of the value it names the storage
# by, as `retarget!` takes it, and the storage the step uses from now on,
# which `retarget!` points it at. The results over the storage are over `x`.
function point_at!(ex::Executable, k, x, moves)
    uses = ex.pointable[k]
    uses === nothing && return moves
    to = address_of(x)
    to == ex.at[k] && return moves
    buffer = vec(x)
    moves = something(moves, Tuple{Int, Int, Move, Any}[])
    for (i, v) in uses.steps
        push!(moves, (i, k, (value = v, from = ex.at[k], to, bytes = UInt(sizeof(x))), buffer))
    end
    if !isempty(uses.results)
        foreach(r -> ex.results[r] = reshape(buffer, size(ex.results[r])), uses.results)
        ex.outputs = outputs_over(ex.program, ex.results)
    end
    ex.pointed[k] = x
    ex.at[k] = to
    moves
end

# The results in the program's output structure, as `outputs` gives them.
outputs_over(program, results) = rebuild(k -> k isa Int ? results[k] : constant_leaf(k), program.output_structure)

# Point the steps at their new storage, and change the captured graph where
# it records their addresses, step by step, or record it again where the
# target cannot; an argument whose graph was recorded again waits from then
# on to be brought twice in a row.
function retarget!(ex::Executable, moves)
    moves === nothing && return ex
    sort!(moves; by = first)
    changed = true
    start = 1
    while start <= length(moves)
        i = moves[start][1]
        stop = start
        while stop < length(moves) && moves[stop + 1][1] == i
            stop += 1
        end
        mine = view(moves, start:stop)
        # The step is pointed at the new storage once, by the target or after it.
        pointed = Ref(false)
        repoint() = pointed[] || (foreach(m -> repoint!(execution_step(ex, i), m[3].value, m[4]), mine); pointed[] = true)
        if ex.captured === nothing
            repoint()
        elseif !retarget!(ex.target, ex.captured, collect(execution_steps(ex, i)), [m[3] for m in mine], repoint)
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

# The steps of an execution, in order: the inputs saved, each launch, and
# the staged copies.
execution(ex::Executable) = Any[(save.copy for save in ex.saves)..., ex.launches..., ex.copies...]

# Step `i` of an execution.
function execution_step(ex::Executable, i)
    s, l = length(ex.saves), length(ex.launches)
    i <= s ? ex.saves[i].copy : i <= s + l ? ex.launches[i - s] : ex.copies[i - s - l]
end

# What a graph records, step by step: each of the executions a replay runs.
steps(ex::Executable) = reduce(vcat, fill(execution(ex), ex.repetitions))

# Where step `i` of an execution is among what a graph records, once in each execution.
execution_steps(ex::Executable, i) = i .+ (length(ex.saves) + length(ex.launches) + length(ex.copies)) .* (0:(ex.repetitions - 1))

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
    runnable(ex)
    run_once(ex)
    write_back!(ex)
end

# What an execution launches, and what a capture records.
function run_once(ex::Executable)
    foreach(save -> save.copy(), ex.saves)
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
`repetitions` times over, as one launch of a graph on a target with graphs,
then copy what it wrote into an argument into the caller's array `point!`
bound it to, where that is not its storage. The results are in `outputs(ex)`.
"""
function replay!(ex::Executable)
    ex.captured === nothing && throw(ArgumentError("nothing to replay: capture! records the program first"))
    replay(ex.target, ex.captured)
    write_back!(ex)
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
