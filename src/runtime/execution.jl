mutable struct Prepared{R <: AbstractRuntime}
    storage::StoragePlan
    runtime::R
    inputs::Vector{Any}
    buffers::Vector{Any} # owns every slot; views retain the same storage
    values::Dict{Int, Any}
    scratch::Vector{Any} # shared workspace slots, distinct from value storage
    # Heterogeneous launch objects are intentional; scalar kernels specialize
    # their expression trees separately from this host dispatch loop.
    calls::Vector{Any}
    providers::Vector{Symbol}
    result::Any
    selection_notes::Vector{Vector{String}}
    semantic_notes::Vector{String}
    candidates::Vector{Symbol}
    candidate_notes::Vector{Vector{String}} # implementation_notes, resolved at preparation
    workspace_slots::Vector{Vector{Int}}
    state::StateBindings
    groups::Vector{Vector{Int}}
    proposals::Vector{ProposalRecord}
end

"""Copy new input contents into the executable's stable buffers (outside capture)."""
Base.@nospecializeinfer function bind!(ex::Prepared, args...)
    @nospecialize args
    check_runtime(ex.runtime)
    flat_args = validate(ex.storage.program.primal, args)
    for (dest, src) in zip(ex.inputs, flat_args)
        src isa Number ? fill!(dest, src) : copyto!(dest, src)
    end
    ex
end

"""
Return borrowed `(loss_buffer, gradient_buffers...)` for training, or the original
result structure for forward-only execution. Scalar leaves are 0D buffers.
Treat results as read-only: they can alias each other or owned input storage.
Copy results that need to survive rebinding/replay or be mutated independently.
"""
outputs(ex::Prepared) = ex.result

"""Execute the prepared schedule without capture; inputs and output addresses stay fixed."""
function execute!(ex::Prepared)
    check_runtime(ex.runtime)
    for call in ex.calls
        call()
    end
    commit_state!(ex.state)
    outputs(ex)
end

"""
Warm and capture the schedule, retaining the graph and its owned buffers.
Donated state is saved outside capture and restored afterward, including on
failure. Other output buffers may contain warmup results. A replay advances state
once; capture and recapture leave its values unchanged.
"""
function capture!(ex::Prepared)
    check_runtime(ex.runtime)
    saved = map(copy, ex.state.buffers)
    try
        capture_runtime!(ex.runtime, ex)
    finally
        for (buffer, snapshot) in zip(ex.state.buffers, saved)
            copyto!(buffer, snapshot)
        end
    end
    ex
end

"""Replay a captured executable and return its borrowed output buffers."""
function replay!(ex::Prepared)
    check_runtime(ex.runtime)
    replay_runtime!(ex.runtime)
    outputs(ex)
end

# Optional diagnostics for a built callable; this does not participate in
# implementation selection or execution.
implementation_summary(::Any) = nothing
implementation_summary(built::BuiltImplementation) = implementation_summary(built.launch)

"""
    explain([io], executable)
    explain([io], storage_plan)

Print scheduled regions, dependencies, storage slots, and alias materializations.
For an executable, also report its runtime, chosen implementations, rejection
reasons, and owned input/workspace bytes. This is a plan report, not a profiler.
"""
function explain(io::IO, ex::Prepared)
    explain_runtime(io, ex.runtime)
    foreach(note -> println(io, "  semantic ", note), ex.semantic_notes)
    explain_proposals(io, ex.proposals, ex.storage.steps)
    explain(io, ex.storage; providers = ex.providers)
    scratchbytes = sum(a -> sizeof(eltype(a)) * length(a), ex.scratch; init = 0)
    inputbytes = sum(a -> sizeof(eltype(a)) * length(a), ex.inputs; init = 0)
    println(
        io,
        "Owned input copies: ",
        inputbytes,
        " bytes; shared implementation workspace: ",
        scratchbytes,
        " bytes",
    )
    println(io, "Implementation candidates: ", join(ex.candidates, " → "))
    for (name, notes) in zip(ex.candidates, ex.candidate_notes), note in notes
        println(io, "  ", name, " ", note)
    end
    for (index, slots) in enumerate(ex.workspace_slots)
        if !isempty(slots)
            println(io, "  region $index uses workspace slots $slots")
        end
    end
    for (i, notes) in enumerate(ex.selection_notes), note in notes
        println(io, "  region ", i, " rejected ", note)
    end
    implementationbytes =
        sum(c -> c isa BuiltImplementation ? c.workspace_bytes : 0, ex.calls; init = 0)
    println(io, "Owned implementation workspace: ", implementationbytes, " bytes")
    println(io, "Providers used: ", join(unique(ex.providers), ", "))
    explain_state(io, ex.state)
    println(
        io,
        "Scheduled calls: ",
        length(ex.calls),
        " for ",
        length(ex.storage.steps),
        " regions",
    )
    for (index, call) in enumerate(ex.calls)
        summary = implementation_summary(call)
        summary === nothing || println(io, "  call $index: $summary")
    end
    for (launch, indices) in enumerate(ex.groups)
        length(indices) > 1 && println(
            io,
            "  launch ",
            launch,
            " groups regions ",
            first(indices),
            ":",
            last(indices),
        )
    end
end

explain(ex::Prepared) = explain(stdout, ex)

"""
    bind_argument!(executable, index, value)

Copy one positional argument, including its tuple/NamedTuple leaves, into owned
buffers. Use this to replace a batch without resetting evolving parameter and
optimizer state. The complete argument is validated before any buffer changes.
"""
function bind_argument!(ex::Prepared, index::Integer, argument)
    check_runtime(ex.runtime)
    program = ex.storage.program.primal
    if index isa Bool || !(1 <= index <= length(program.input_structure))
        throw(ArgumentError("argument index is out of range"))
    end
    structure = program.input_structure[index]
    same_structure(structure, argument) ||
        throw(ArgumentError("argument structure differs from trace"))
    ids = tree_leaves(structure)
    leaves = tree_leaves(argument)
    for (id, leaf) in zip(ids, leaves)
        validate_leaf(program.nodes[id], leaf)
    end
    destinations = Dict(zip(program.arguments, ex.inputs))
    for (id, leaf) in zip(ids, leaves)
        destination = destinations[id]
        leaf isa Number ? fill!(destination, leaf) : copyto!(destination, leaf)
    end
    ex
end

# Wall-clock seconds for each call, run in order. Runtimes with asynchronous
# devices override this with device timing.
function time_calls(::AbstractRuntime, calls)
    map(calls) do call
        started = time_ns()
        call()
        (time_ns() - started) / 1e9
    end
end

function call_description(ex::Prepared, launch)
    indices = ex.groups[launch]
    step = ex.storage.steps[first(indices)]
    provider = ex.providers[first(indices)]
    name = step.region isa OpaqueCall ? "call $(step.region.call.f)" : string(region_name(step.region))
    dims = ex.storage.program.nodes[step.output].dims
    text = "$name [$provider] $(join(dims, "×"))"
    length(indices) > 1 && (text *= " ($(length(indices)) grouped regions)")
    summary = implementation_summary(ex.calls[launch])
    summary === nothing ? text : "$text; $summary"
end

"""
    Tempest.profile([io], executable; samples = 20, top = 20)

Execute the schedule `samples` times without capture, timing every call on the
device, and print the calls ranked by median time with their share of the
total. Each sample advances donated state once, like `execute!`. Returns a
vector of `(call, seconds, description)` in schedule order.
"""
function profile(io::IO, ex::Prepared; samples::Integer = 20, top::Integer = 20)
    check_runtime(ex.runtime)
    execute!(ex)
    runs = map(1:samples) do _
        times = time_calls(ex.runtime, ex.calls)
        commit_state!(ex.state)
        times
    end
    medians = [sort!([run[i] for run in runs])[cld(samples, 2)] for i in eachindex(ex.calls)]
    total = sum(medians)
    rows = [(call = i, seconds = medians[i], description = call_description(ex, i)) for i in eachindex(medians)]
    println(io, "Profile: ", length(rows), " calls, ", round(1e3 * total; digits = 3),
            " ms per execution (sum of per-call medians over ", samples, " samples)")
    for row in first(sort(rows; by = row -> -row.seconds), top)
        println(io, lpad(round(1e6 * row.seconds; digits = 1), 9), " μs ",
                lpad(round(100 * row.seconds / total; digits = 1), 5), "%  call ",
                row.call, ": ", row.description)
    end
    rows
end

profile(ex::Prepared; kwargs...) = profile(stdout, ex; kwargs...)
