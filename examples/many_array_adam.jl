# Run: julia --project=examples examples/many_array_adam.jl [array_count]
# This measures the optimizer slice with supplied gradients, independently of a model.
using Tempest
using CUDACore
import cuBLAS

function many_array_update(params, moments, gradients)
    beta1, beta2 = 0.9f0, 0.999f0
    step = moments.step + 1.0f0
    first = Tuple(
        map(
            (m, g) -> beta1 .* m .+ (1.0f0 - beta1) .* g,
            collect(moments.first),
            collect(gradients),
        ),
    )
    second = Tuple(
        map(
            (v, g) -> beta2 .* v .+ (1.0f0 - beta2) .* abs2.(g),
            collect(moments.second),
            collect(gradients),
        ),
    )
    next_params = map(collect(params), collect(first), collect(second)) do p, m, v
        p .-
        0.001f0 .* (m ./ (1.0f0 - beta1 ^ step)) ./
        (sqrt.(v ./ (1.0f0 - beta2 ^ step)) .+ 1.0f-8)
    end
    Tuple(next_params), (first = first, second = second, step = step)
end

function many_array_problem(count)
    sizes = (17, 63, 127, 513, 1025)
    params = ntuple(i -> fill(Float32(i) / count, sizes[mod1(i, length(sizes))]), count)
    gradients = Tuple(map(p -> fill(0.2f0, size(p)), collect(params)))
    moments = (
        first = Tuple(map(zero, collect(params))),
        second = Tuple(map(zero, collect(params))),
        step = 0.0f0,
    )
    params, moments, gradients
end

function benchmark_many_arrays(count; options...)
    args = many_array_problem(count)
    trace_seconds = @elapsed program = trace(many_array_update, args...)
    println("Traced $count arrays in $trace_seconds seconds")
    flush(stdout)
    preparation_seconds = @elapsed ex = prepare_forward(
        program,
        args...;
        target = CUDATarget(kernels = :cuda),
        donate = (1, 2),
        options...,
    )
    println("Prepared in $preparation_seconds seconds")
    flush(stdout)
    capture!(ex)
    expected = args[1:2]
    for _ = 1:3
        expected = many_array_update(expected..., args[3])
        actual = replay!(ex)
        for (a, b) in zip(Tempest.tree_leaves(actual), Tempest.tree_leaves(expected))
            observed = b isa Number ? only(Array(a)) : Array(a)
            @assert isapprox(observed, b; rtol = 2.0f-4, atol = 2.0f-6)
        end
    end
    function sample()
        bind!(ex, args...)
        CUDACore.synchronize()
        elapsed = @elapsed begin
            for _ = 1:100
                replay!(ex)
            end
            CUDACore.synchronize()
        end
        elapsed / 100
    end
    sample()
    seconds = minimum(sample() for _ = 1:5)
    bytes = sum(sizeof, ex.buffers; init = 0)
    input_bytes = sum(sizeof, ex.inputs; init = 0)
    implementation_bytes = sum(call -> call.workspace_bytes, ex.calls; init = 0)
    println((;
        arrays = count,
        trace_seconds,
        preparation_seconds,
        value_bytes = bytes,
        input_bytes,
        implementation_bytes,
        regions = length(ex.storage.steps),
        launches = length(ex.calls),
        commits = length(ex.state.commits),
        replay_us = 1e6 * seconds,
    ))
    ex
end

if abspath(PROGRAM_FILE) == @__FILE__
    CUDACore.functional() || error("CUDA is unavailable")
    benchmark_many_arrays(isempty(ARGS) ? 100 : parse(Int, only(ARGS)))
end
