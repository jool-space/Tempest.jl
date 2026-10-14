# Implementation selection for a storage plan: request construction, candidate
# choice with rejection reporting, and grouping of independent elementwise
# regions into one launch. Group legality is decided here after physical
# storage assignment; the policy of trying candidates in order is `select_schedule`.
"""
A group of independent elementwise requests. Outputs and inputs are vectors with
one entry per request. Each member retains its own shape, expression, and alias
contract. Rejecting a group lets preparation try the original individual regions.
"""
struct GroupedRequest <: ImplementationRequest
    requests::Vector{ImplementationRequest}
end

region_request(request::RegionRequest) = request
region_request(request::InPlaceRequest) = request.region

function support(impl::CPUKernels, request::GroupedRequest)
    for member in request.requests
        region_request(member).step.region isa Elementwise ||
            return Unsupported("groups require elementwise regions")
        verdict = checked_support(impl, member)
        verdict isa Unsupported && return verdict
    end
    Supported()
end

function build_implementation(::CPUKernels, request::GroupedRequest, runtime, outputs, inputs)
    regions = [region_request(member).step.region for member in request.requests]
    launch = () -> begin
        for (region, out, arrays) in zip(regions, outputs, inputs)
            cpu_pointwise!(out, region.expression, arrays, region.count)
        end
        nothing
    end
    BuiltImplementation(launch, (outputs, inputs), 0)
end

# Check physical hazards as well as graph edges. Storage reuse can make two
# logically independent regions unsafe to execute together.
function independent_bindings(a, b)
    !Base.mightalias(a.output, b.output) &&
        !any(input -> Base.mightalias(a.output, input), b.inputs) &&
        !any(input -> Base.mightalias(b.output, input), a.inputs)
end

function consecutive_group(storage, bindings, start; limit = 128)
    first_step = storage.steps[start]
    first_step.region isa Elementwise || return [start]
    indices = [start]
    for index = (start + 1):min(length(storage.steps), start + limit - 1)
        step = storage.steps[index]
        step.region isa typeof(first_step.region) || break
        all(previous -> independent_bindings(bindings[previous], bindings[index]), indices) ||
            break
        push!(indices, index)
    end
    indices
end

function schedule_request(target, program, storage, step, binding)
    request = implementation_request(target, program, step, binding.inputs)
    if haskey(storage.donations, step.output)
        aliases = Tuple(
            i for i in eachindex(binding.inputs) if
            Base.mightalias(binding.output, binding.inputs[i])
        )
        return InPlaceRequest(request, aliases)
    end
    request
end

function select_schedule(implementations, target, program, storage, runtime, bindings; grouped)
    notes = [String[] for _ in storage.steps]
    requests = [
        schedule_request(target, program, storage, step, binding) for
        (step, binding) in zip(storage.steps, bindings)
    ]
    chosen = Any[]
    groups = Vector{Int}[]
    # Selected regions that fail to build are collected, so one replan excludes
    # them all; other rejections return at once.
    rejected = RejectedGraphRegion[]
    index = 1
    while index <= length(storage.steps)
        indices = grouped ? consecutive_group(storage, bindings, index) : [index]
        selected = nothing
        if length(indices) > 1
            request = GroupedRequest(ImplementationRequest[requests[i] for i in indices])
            outputs = [bindings[i].output for i in indices]
            inputs = [bindings[i].inputs for i in indices]
            selected = choose_implementation(
                implementations,
                request,
                runtime,
                outputs,
                inputs,
                notes[index],
            )
        end
        if selected === nothing
            indices = [index]
            binding = bindings[index]
            region = storage.steps[index].region
            # A proposed region is built only by the implementation that proposed it.
            selected = choose_implementation(
                region isa ProposedRegion ? (region.implementation,) : implementations,
                requests[index],
                runtime,
                binding.output,
                binding.inputs,
                notes[index],
            )
        end
        if selected === nothing
            step = storage.steps[index]
            if step.region isa SelectedRegion
                region = step.region
                push!(rejected, RejectedGraphRegion(region.members, notes[index], region isa ProposedRegion))
                index += 1
                continue
            elseif haskey(storage.donations, step.output)
                return RejectedDonation(storage.donations[step.output], notes[index])
            elseif step.region isa OpaqueCall
                return RejectedCall(step.region.call, notes[index])
            end
            throw(
                ArgumentError(
                    "no supported implementation for region $index ($(region_name(step.region))): $(join(notes[index], "; "))",
                ),
            )
        end
        push!(chosen, selected)
        push!(groups, indices)
        index = last(indices) + 1
    end
    isempty(rejected) || return rejected
    (; chosen, groups, notes)
end
