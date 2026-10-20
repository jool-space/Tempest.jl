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
    # Causal attention over (features, keys or queries, batch), written out.
    function attention(q, k, v)
        s = contract(k, q, (1, 2, 3), (1, 4, 3), (2, 4, 3))
        s = ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32)
        contract(v, exp.(s .- logsumexp(s; dims = 1)), (1, 2, 3), (2, 4, 3), (1, 4, 3))
    end
    # A mask from a counter hash of a seed and the position: a scalar function
    # of integers, with no derivative.
    keep(seed, i) = (h = (UInt32(seed % UInt32) * 0x9e3779b9) ⊻ (UInt32(i % UInt32) * 0xc2b2ae35); h ⊻= h >> 16; h *= 0x7feb352d; (h >> 8) % 10 != 0)
    dropout(x, seed) = x .* ifelse.(keep.(reshape(seed[1:1], 1, 1), reshape(1:length(x), size(x))), 1.25f0, 0.0f0)
    x, w, b, v = sample(8, 6), sample(4, 8), sample(4), sample(8)
    q, k = sample(4, 8, 2), sample(4, 8, 2)
    losses = [
        ((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (w, x, b)),
        ((x, v) -> sum(abs2, rmsnorm(x, v)), (x, v)),
        ((x, v) -> sum(abs2, layernorm(x, v)), (x, v)),
        (x -> sum(exp.(x .- logsumexp(x; dims = 1)) .* x), (x,)),
        (x -> sum(exp.(x .- logsumexp(x; dims = 2))) + sum(maximum(x; dims = 1)), (x,)),
        ((q, k) -> sum(abs2, attention(q, k, k) .+ q), (q, k)),
        ((x, s) -> sum(abs2, dropout(permutedims(x), s)), (x, Int32[1])),
        (x -> sum(abs2, Float16.(reshape(x, 6, 8))), (x,)),
        (x -> (y = copy(x); y[2:3, :] .= 0; sum(abs2, vcat(y, x))), (x,)),
    ]
    for (f, args) in losses
        p = Compiler.trace(f, args...)
        for program in (p, Compiler.differentiate(p))
            Compiler.interpret(program, args...)
            ex = Compiler.prepare(program, args...)
            Compiler.execute!(ex)
            explain(devnull, ex)
        end
    end
    nothing
end

@compile_workload workload()
