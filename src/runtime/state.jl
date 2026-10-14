# State is functional in the graph. Proven elementwise updates may write directly
# into inputs; implementations explicitly opt in. Other updates commit afterward.
# Sources that alias state are snapshotted together before any destination changes.
struct StateCopy{D, S}
    destination::D
    source::S
end

(copy::StateCopy)() = copyto!(copy.destination, copy.source)

struct StateBindings
    arguments::Tuple
    snapshots::Vector{StateCopy}
    commits::Vector{StateCopy}
    buffers::Vector{Any} # all donated buffers, including direct writes, for capture restoration
end

StateBindings() = StateBindings((), StateCopy[], StateCopy[], Any[])

function validate_donation(program, args, donate)
    donate isa Tuple || throw(ArgumentError("donate must be a tuple of argument indices"))
    isempty(donate) && return
    program isa ForwardProgram ||
        throw(ArgumentError("donation requires a forward/update program"))
    structure = program.structure
    structure isa Union{Tuple, NamedTuple} ||
        throw(ArgumentError("donation requires tuple or NamedTuple outputs"))
    for index in donate
        if !(
            index isa Integer &&
            !(index isa Bool) &&
            1 <= index <= min(length(args), length(structure))
        )
            throw(ArgumentError("donated indices must name both an input and an output group"))
        end
        input = program.primal.input_structure[index]
        output = structure[index]
        same_structure(input, output) ||
            throw(ArgumentError("donated input/output structures differ"))
        for (a, b) in zip(tree_leaves(input), tree_leaves(output))
            before, after = program.nodes[a], program.nodes[b]
            if (before.dtype, before.dims, before.scalar) !=
               (after.dtype, after.dims, after.scalar)
                throw(ArgumentError("donated input/output signatures differ"))
            end
        end
    end
    length(unique(donate)) == length(donate) ||
        throw(ArgumentError("donated indices must be unique"))
    leaves = tree_leaves(args)
    ids = program.primal.arguments
    donated_ids =
        Set(id for index in donate for id in tree_leaves(program.primal.input_structure[index]))
    for (i, a) in enumerate(leaves), (j, b) in enumerate(leaves)
        if i != j &&
           ids[i] in donated_ids &&
           a isa AbstractArray &&
           b isa AbstractArray &&
           Base.mightalias(a, b)
            throw(ArgumentError("donated array leaves must not alias other input leaves"))
        end
    end
    nothing
end

function prepare_state!(ex, donate)
    isempty(donate) && return ex
    program = ex.storage.program
    destinations = Dict(zip(program.primal.arguments, ex.inputs))
    state_arrays = [
        destinations[id] for index in donate for
        id in tree_leaves(program.primal.input_structure[index])
    ]
    snapshots = StateCopy[]
    commits = StateCopy[]
    result_values = Dict{Int, Any}()
    for id in unique(program.results)
        source = resolve_value!(ex.values, ex.storage, id)
        root = first(ex.storage.layouts[id])
        if !haskey(ex.storage.donations, root) &&
           any(state -> Base.mightalias(source, state), state_arrays)
            snapshot = allocate(ex.runtime, eltype(source), size(source))
            push!(snapshots, StateCopy(snapshot, source))
            result_values[id] = snapshot
        else
            result_values[id] = source
        end
    end
    result = result_structure(program, id -> result_values[id])
    for index in donate
        input_ids = program.primal.input_structure[index]
        for (id, source) in zip(tree_leaves(input_ids), tree_leaves(result[index]))
            source === destinations[id] || push!(commits, StateCopy(destinations[id], source))
        end
    end
    # Substitute destinations by output position, not value id: a single logical
    # value may be returned in several groups with different state ownership.
    updated = map(1:length(result)) do index
        if index in donate
            tree_map(id -> destinations[id], program.primal.input_structure[index])
        else
            result[index]
        end
    end
    ex.result =
        result isa NamedTuple ? NamedTuple{keys(result)}(Tuple(updated)) : Tuple(updated)
    ex.state = StateBindings(donate, snapshots, commits, Any[state_arrays...])
    ex
end

function commit_state!(state::StateBindings)
    foreach(copy -> copy(), state.snapshots)
    foreach(copy -> copy(), state.commits)
    nothing
end

function explain_state(io, state::StateBindings)
    isempty(state.arguments) && return
    println(
        io,
        "State donation: output groups ",
        state.arguments,
        " → owned input groups ",
        state.arguments,
    )
    println(
        io,
        "  ",
        length(state.commits),
        " staged commits after all regions; ",
        length(state.buffers) - length(state.commits),
        " direct state writes",
    )
    bytes = sum(
        copy -> sizeof(eltype(copy.destination)) * length(copy.destination),
        state.snapshots;
        init = 0,
    )
    println(io, "  simultaneous-update snapshots: ", bytes, " bytes")
end

struct RejectedDonation
    input::Int
    reasons::Vector{String}
end

# A read of the overwritten input must address exactly the output lane. Other
# inputs may broadcast or gather: only reads of this destination can race writes.
same_lane_read(::Literal, position, count) = true

function same_lane_read(read::Read{I}, position, count) where {I}
    I != position ||
        (linear_identity(read.mapping) && read.dims == read.shape && prod(read.shape) == count)
end

same_lane_read(call::ScalarCall, position, count) =
    all(arg -> same_lane_read(arg, position, count), call.args)

function donation_candidates(program, steps, layouts, lastuse, donate, blocked)
    candidates = Dict{Int, Int}()
    isempty(donate) && return candidates
    pairs = Pair{Int, Int}[]
    for index in donate
        append!(
            pairs,
            (
                out => input for (out, input) in zip(
                    tree_leaves(program.structure[index]),
                    tree_leaves(program.primal.input_structure[index]),
                )
            ),
        )
    end
    output_counts = Dict{Int, Int}()
    for (output, _) in pairs
        output_counts[output] = get(output_counts, output, 0) + 1
    end
    writers = Dict(step.output => (time, step) for (time, step) in enumerate(steps))
    for (output, input) in pairs
        input in blocked && continue
        output_counts[output] == 1 || continue
        haskey(writers, output) || continue
        time, step = writers[output]
        step.region isa Elementwise || continue
        first(layouts[output]) == output || continue
        get(lastuse, input, 0) <= time || continue
        safe = all(enumerate(step.dependencies)) do (position, dependency)
            dependency != input ||
                same_lane_read(step.region.expression, position, step.region.count)
        end
        if safe
            candidates[output] = input
        end
    end
    candidates
end
