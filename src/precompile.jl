# Work Tempest does while it precompiles, so that its own code is compiled once
# rather than in every session: small programs of each kind traced,
# differentiated, lowered, planned, explained and prepared for the CPU.
# Development sessions skip it, so a test runs `workload` too.

using PrecompileTools: @compile_workload

"Trace, differentiate, interpret, prepare, run and explain small programs of each kind on the CPU."
function workload()
    sample(dims...) = Float32.(reshape(sin.(1:prod(dims)), dims))
    gelu(x) = x / (1 + exp(-1.702f0 * x))
    rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
    function layernorm(x, w)
        m = sum(x; dims = 1) ./ size(x, 1)
        (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
    end
    x, w, b, v = sample(8, 6), sample(4, 8), sample(4), sample(8)
    q, k = sample(4, 4, 8, 2), sample(4, 2, 8, 2)
    losses = [
        ((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (w, x, b)),
        ((x, v) -> sum(abs2, rmsnorm(x, v)), (x, v)),
        ((x, v) -> sum(abs2, layernorm(x, v)), (x, v)),
        (x -> sum(exp.(x .- logsumexp(x; dims = 1)) .* x), (x,)),
        (x -> sum(exp.(x .- logsumexp(x; dims = 2))) + sum(maximum(x; dims = 1)), (x,)),
        ((q, k) -> sum(abs2, attention(q, k, k; causal = true) .+ q), (q, k)),
        ((x, s) -> sum(abs2, dropout(permutedims(x), s; rate = 0.1)), (x, Int32[1])),
        (x -> sum(abs2, Float16.(reshape(x, 6, 8))), (x,)),
    ]
    for (f, args) in losses
        p = trace(f, args...)
        for program in (p, differentiate(p))
            interpret(program, args...)
            ex = prepare(program, args...)
            execute!(ex)
            explain(devnull, ex)
        end
    end
    nothing
end

@compile_workload workload()
