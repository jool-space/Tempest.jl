# Run with: julia --project=. examples/research/reduction_scopes.jl
# Research only: phase analysis and CPU sweeps of real graphs, not a GPU scheduler.
# The accepted family is one rectangular domain, identity indexing, and reductions
# over one common axis set. Views, reshapes, contractions, and opaque calls reject.
module ReductionScopes

using Tempest
using Random
using Test

const D = Tempest

# This operation belongs to the experiment. It does not extend Base.maximum,
# production tracing, automatic differentiation, or any production backend.
struct ResearchMaximum <: D.Operation
    axes::Tuple{Vararg{Int}}
end

reduce_max(x; dims) = maximum(x; dims)
function reduce_max(x::D.TracedArray; dims)
    axes = dims isa Integer ? (dims,) : Tuple(dims)
    all(i -> 1 <= i <= ndims(x), axes) || reject("invalid reduction axes")
    axes = Tuple(sort!(unique!(collect(axes))))
    shape = ntuple(i -> i in axes ? 1 : size(x, i), ndims(x))
    D.emit(x.nodes, ResearchMaximum(axes), (x.id,), shape, eltype(x); scalar = false)
end
D.evaluate(op::ResearchMaximum, node, xs) = maximum(only(xs); dims = op.axes)

is_reduction(op) = op isa Union{D.SumReduction, ResearchMaximum}
reduction_axes(op, rank) = op.axes isa Colon ? Tuple(1:rank) : op.axes

struct ScopeAnalysis
    program::D.TensorProgram
    domain::Tuple
    axes::Tuple
    active::BitVector
    phases::Vector{Int}
    last_use::Vector{Int}
    reductions::Vector{Int}
end

reject(reason) = throw(ArgumentError(reason))

# Julia broadcasting aligns leading axes. Singleton axes require no new domain.
function fits_domain(shape, domain)
    length(shape) <= length(domain) &&
        all(i -> shape[i] == 1 || shape[i] == domain[i], eachindex(shape))
end

function analyze(program)
    nodes = program.nodes
    results = Int[D.tree_leaves(program.output)...]
    active = D.live_nodes(nodes, results)
    reductions = findall(i -> active[i] && is_reduction(nodes[i].op), eachindex(nodes))
    isempty(reductions) && reject("the candidate has no reduction")
    first_reduction = nodes[first(reductions)]
    domain = nodes[only(first_reduction.inputs)].dims
    axes = reduction_axes(first_reduction.op, length(domain))
    isempty(domain) && reject("the reference domain must have array axes")
    for (id, node) in enumerate(nodes)
        if active[id] && node.op isa D.LayoutOperation
            reject("layout operation at %$id needs coordinate mapping; equal shapes are insufficient")
        end
    end
    all(i -> fits_domain(nodes[i].dims, domain), results) ||
        reject("an escaping output introduces an axis outside the reference domain")

    phases = zeros(Int, length(nodes))
    for (id, node) in enumerate(nodes)
        active[id] || continue
        fits_domain(node.dims, domain) || reject("%$id is outside the reference domain")
        input_phase = maximum((phases[i] for i in node.inputs); init = 0)
        if is_reduction(node.op)
            source = nodes[only(node.inputs)]
            source.dims == domain || reject("%$id reduces a different iteration domain")
            reduction_axes(node.op, length(domain)) == axes ||
                reject("%$id reduces incompatible axes")
            phases[id] = input_phase + 1
        elseif node.op isa Union{D.Argument, D.Constant, D.Pointwise}
            phases[id] = input_phase
        else
            reject("$(typeof(node.op)) is outside this experiment's operation set")
        end
    end

    # A reduction consumes its input before its completion frontier. A residual
    # use after that frontier requires retaining or reconstructing the value.
    last_use = copy(phases)
    for (id, node) in enumerate(nodes)
        active[id] || continue
        use_phase = phases[id] - Int(is_reduction(node.op))
        for input in node.inputs
            last_use[input] = max(last_use[input], use_phase)
        end
    end
    final_phase = maximum(phases)
    for id in results
        last_use[id] = final_phase
    end
    ScopeAnalysis(program, domain, axes, active, phases, last_use, reductions)
end

function live_across(analysis, frontier)
    findall(eachindex(analysis.phases)) do id
        analysis.active[id] && analysis.phases[id] < frontier <= analysis.last_use[id]
    end
end

# This is a recipe, not a recomputation decision or a cost model. Stop at a
# completed reduction: reconstructing pointwise work must not rerun a reduction.
# Unknown scalar functions are conservatively excluded because tracing alone
# does not establish purity. Inputs must still be retained or reloadable.
function recompute_inputs(program, id)
    node = program.nodes[id]
    node.op isa Union{D.Argument, D.Constant} && return [id]
    is_reduction(node.op) && return [id]
    node.op isa D.Pointwise || return nothing
    pure_scalar(node.op.f) || return nothing
    leaves = Int[]
    for input in node.inputs
        sources = recompute_inputs(program, input)
        isnothing(sources) && return nothing
        append!(leaves, sources)
    end
    sort!(unique!(leaves))
end

pure_scalar(f) = f in (+, -, *, /, sin, cos, exp, log, sqrt, inv, abs2, tanh) ||
    f isa D.LiteralPower

reduction_identity(node) =
    node.op isa ResearchMaximum ? typemin(node.dtype) : zero(node.dtype)

logical_index(shape, coordinate) =
    ntuple(i -> shape[i] == 1 ? 1 : coordinate[i], length(shape))

function joined_coordinate(rank, bound_axes, bound, free_axes, free)
    coordinate = ones(Int, rank)
    for (axis, index) in zip(bound_axes, Tuple(bound))
        coordinate[axis] = index
    end
    for (axis, index) in zip(free_axes, Tuple(free))
        coordinate[axis] = index
    end
    Tuple(coordinate)
end

# An executable schedule witness: one sweep per reduction phase, then one for
# outputs. Reduction results survive between sweeps; pointwise values only live
# in a lane cache and are recomputed on later sweeps. No full-size intermediate
# is retained unless it is an escaping output. This deliberately chooses more
# recomputation, not necessarily the best placement or a numerically identical
# summation tree. There are no padded lanes or GPU ownership assumptions here.
function execute_sweeps(analysis, args...)
    program = analysis.program
    nodes = program.nodes
    for (id, node) in enumerate(nodes)
        if analysis.active[id] && node.op isa D.Pointwise && !pure_scalar(node.op.f)
            reject("%$id has no purity contract for recomputation")
        end
    end
    inputs = Dict(zip(program.arguments, D.validate(program, args)))
    result_ids = unique(Int[D.tree_leaves(program.output)...])
    buffers = Dict(id => Array{nodes[id].dtype}(undef, nodes[id].dims) for id in result_ids)
    phase_count = maximum(analysis.phases)
    groups = [
        filter(id -> analysis.phases[id] == phase, analysis.reductions)
        for phase in 1:phase_count
    ]
    visits = zeros(Int, phase_count + 1)
    evaluations = zeros(Int, length(nodes))
    bound_axes = analysis.axes
    free_axes = Tuple(i for i in eachindex(analysis.domain) if !(i in bound_axes))
    bound_domain = Tuple(analysis.domain[i] for i in bound_axes)
    free_domain = Tuple(analysis.domain[i] for i in free_axes)

    function lane_value(id, coordinate, cache, completed)
        haskey(cache, id) && return cache[id]
        node = nodes[id]
        value = if node.op isa D.Argument
            input = inputs[id]
            input isa Number ? input : input[logical_index(node.dims, coordinate)...]
        elseif node.op isa D.Constant
            node.op.value
        elseif is_reduction(node.op)
            completed[id]
        else
            operands = map(i -> lane_value(i, coordinate, cache, completed), node.inputs)
            convert(node.dtype, node.op.f(operands...))
        end
        evaluations[id] += 1
        cache[id] = value
    end

    for free in CartesianIndices(free_domain)
        completed = Dict{Int, Any}()
        for (phase, reductions) in enumerate(groups)
            accumulators = Dict(id => reduction_identity(nodes[id]) for id in reductions)
            for bound in CartesianIndices(bound_domain)
                coordinate = joined_coordinate(
                    length(analysis.domain), bound_axes, bound, free_axes, free,
                )
                cache = Dict{Int, Any}()
                visits[phase] += 1
                for id in reductions
                    node = nodes[id]
                    value = lane_value(only(node.inputs), coordinate, cache, completed)
                    combine = node.op isa ResearchMaximum ? max : +
                    accumulators[id] = convert(node.dtype, combine(accumulators[id], value))
                end
            end
            merge!(completed, accumulators)
        end
        for bound in CartesianIndices(bound_domain)
            coordinate = joined_coordinate(
                length(analysis.domain), bound_axes, bound, free_axes, free,
            )
            cache = Dict{Int, Any}()
            visits[end] += 1
            for id in result_ids
                index = logical_index(nodes[id].dims, coordinate)
                buffers[id][index...] = lane_value(id, coordinate, cache, completed)
            end
        end
    end
    value = D.tree_map(program.output) do id
        nodes[id].scalar ? buffers[id][] : buffers[id]
    end
    (; value, visits, evaluations)
end

function report(name, analysis)
    println("\n$name: domain=$(analysis.domain), reduce=$(analysis.axes)")
    println("  Logical liveness only; input/constant leaves can remain reloadable.")
    nodes = analysis.program.nodes
    for phase in 0:maximum(analysis.phases)
        ids = findall(i -> analysis.active[i] && analysis.phases[i] == phase, eachindex(nodes))
        labels = ["%$id:$(D.operation_name(nodes[id].op))" for id in ids]
        println("  phase $phase: ", join(labels, ", "))
        phase == 0 && continue
        live = live_across(analysis, phase)
        println("  live across frontier $phase: ", live)
        for id in live
            nodes[id].op isa D.Pointwise || continue
            recipe = recompute_inputs(analysis.program, id)
            println("    %$id recompute inputs: ", isnothing(recipe) ? "unknown purity" : recipe)
        end
    end
end

function rmsnorm(x, residual, weight)
    h = x .+ residual
    rstd = inv.(sqrt.(sum(abs2, h; dims = 1) ./ size(x, 1) .+ 1f-5))
    (y = h .* rstd .* weight, residual = h, rstd = rstd)
end

function layernorm(x, weight)
    centered = x .- sum(x; dims = 1) ./ size(x, 1)
    variance = sum(abs2, centered; dims = 1) ./ size(x, 1)
    centered ./ sqrt.(variance .+ 1f-5) .* weight
end

function rms_input_backward(x, weight, dy)
    rstd = inv.(sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1f-5))
    g = dy .* weight
    rstd .* (g .- x .* rstd .* rstd .* sum(g .* x; dims = 1) ./ size(x, 1))
end

function composed(x)
    a = sum(sin.(x); dims = 1)
    b = sum(abs2, x; dims = 1)
    h = tanh.(x .+ a ./ sqrt.(b .+ 1f0))
    c = sum(h; dims = 1)
    (y = h .- c ./ size(x, 1), shared = h, a = a)
end

function softmax(x)
    z = exp.(x .- reduce_max(x; dims = 1))
    z ./ sum(z; dims = 1)
end

function check_case(name, f, args...)
    program = trace(f, args...)
    analysis = analyze(program)
    expected = D.tree_leaves(f(args...))
    actual = D.interpret(D.ForwardProgram(program), args...)
    @test length(actual) == length(expected)
    @test all(isapprox(a, b; rtol = 2f-5, atol = 2f-6) for (a, b) in zip(actual, expected))
    execution = execute_sweeps(analysis, args...)
    @test D.same_structure(execution.value, f(args...))
    @test all(
        isapprox(a, b; rtol = 2f-5, atol = 2f-6)
        for (a, b) in zip(D.tree_leaves(execution.value), expected)
    )
    @test execution.visits == fill(prod(analysis.domain), maximum(analysis.phases) + 1)
    report(name, analysis)
    println("  CPU sweep visits (last sweep writes outputs): ", execution.visits)
    analysis
end

function run_probe()
    rng = MersenneTwister(23)
    x = randn(rng, Float32, 259, 7) # Deliberately not a power of two.
    residual = randn(rng, Float32, size(x))
    weight = randn(rng, Float32, size(x, 1))
    @testset "Common reduction scopes (research only)" begin
        rms = check_case(
            "RMSNorm with escaping intermediates", rmsnorm, x, residual, weight,
        )
        @test rms.phases[rms.reductions] == [1]
        h = rms.program.output.residual
        @test h in live_across(rms, 1)
        @test rms.last_use[h] == maximum(rms.phases)

        layer = check_case("Centered LayerNorm", layernorm, x, weight)
        @test layer.phases[layer.reductions] == [1, 2]
        backward = check_case(
            "Explicit RMSNorm input backward", rms_input_backward, x, weight, residual,
        )
        @test backward.phases[backward.reductions] == [1, 1]
        small_x = Float64.(x[1:7, 1:3])
        small_w = Float64.(weight[1:7])
        dy = Float64.(residual[1:7, 1:3])
        direction = randn(rng, size(small_x))
        loss(z) = sum(rmsnorm(z, zero(z), small_w).y .* dy)
        step = 1e-5
        numerical = (loss(small_x + step * direction) -
                     loss(small_x - step * direction)) / (2step)
        analytical = sum(rms_input_backward(small_x, small_w, dy) .* direction)
        @test isapprox(analytical, numerical; rtol = 1e-7)
        arbitrary = check_case("Unrecognized composed formula", composed, x)
        @test arbitrary.phases[arbitrary.reductions] == [1, 1, 2]
        @test arbitrary.program.output.shared in live_across(arbitrary, 2)
        shared_recipe = recompute_inputs(arbitrary.program, arbitrary.program.output.shared)
        @test all(id -> id in shared_recipe, arbitrary.reductions[1:2])
        @test !(arbitrary.reductions[3] in shared_recipe)

        stable = check_case("Stable softmax (owned maximum operation)", softmax, 100f0 .* x)
        @test stable.phases[stable.reductions] == [1, 2]
        execution = execute_sweeps(stable, 100f0 .* x)
        exponential = only(findall(stable.program.nodes) do node
            node.op isa D.Pointwise && node.op.f === exp
        end)
        @test execution.evaluations[exponential] == 2length(x)
        @test all(isapprox.(sum(softmax(100f0 .* x); dims = 1), 1f0; atol = 2f-6))

        multi_axis(x) = x .- sum(x; dims = (1, 3)) ./ (size(x, 1) * size(x, 3))
        multiple = check_case("Two bound axes", multi_axis, randn(rng, Float32, 9, 4, 3))
        @test multiple.axes == (1, 3)

        unknown(x) = sum((t -> t * t).(x); dims = 1)
        unproven = analyze(trace(unknown, x))
        @test_throws r"purity contract" execute_sweeps(unproven, x)

        orthogonal(x) = (sum(x; dims = 1), sum(x; dims = 2))
        @test_throws r"incompatible axes" analyze(trace(orthogonal, x))
        transposed(x) = x .- sum(transpose(x); dims = 1)
        square = x[1:7, :]
        @test_throws r"coordinate mapping" analyze(trace(transposed, square))
        newaxis(x, b) = sum(x; dims = 1) .+ b
        expanded = trace(newaxis, x, zeros(Float32, 1, 1, 3))
        @test_throws r"escaping output" analyze(expanded)
        reshaped(x) = sum(reshape(x, 7, 259); dims = 1)
        @test_throws r"coordinate mapping" analyze(trace(reshaped, x))

        # Full normalization backward also reduces over columns for dweight.
        # This probe accepts its input-gradient part, not the entire graph.
        function full_backward(x, w, dy)
            rstd = inv.(sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1f-5))
            dx = rms_input_backward(x, w, dy)
            dw = sum(dy .* x .* rstd; dims = 2)
            (dx, dw)
        end
        complete = trace(full_backward, x, weight, residual)
        @test_throws r"incompatible axes" analyze(complete)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_probe()
end

end # module
