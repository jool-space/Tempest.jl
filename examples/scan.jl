# A linear recurrence as a parallel scan
#
# `h[t] = a[t] * h[t-1] + b[t]` along time for every channel: the state update
# of linear RNNs and state-space models. Written as Base's `accumulate` over
# (factor, offset) pairs, Tempest runs it in sequence, as Base promises: the
# combine is the user's function and nothing says it associates. Named as a
# `GenericFold` with the same combine, the claim that it associates is made,
# and the scan runs in parallel on cuTile. The reference is a plain loop in
# Float64 on the CPU.
#
#   julia --project=test examples/scan.jl

using Tempest, CUDACore, cuTile, cuBLAS, Printf

# Two steps of the recurrence, the earlier first, are one step: h ↦ a2 (a1 h + b1) + b2.
chain((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2)

# As Base spells it, from h = 0.
in_sequence(a, b) = last.(accumulate(chain, tuple.(a, b); dims = 1, init = (one(eltype(a)), zero(eltype(b)))))

# The same combine, claimed to associate: init is its identity, observe reads h.
const RECURRENCE = Tempest.GenericFold(; init = E -> (one(fieldtype(E, 1)), zero(fieldtype(E, 2))),
    lift = identity, combine = chain, observe = last)
in_parallel(a, b) = accumulate(RECURRENCE, tuple.(a, b); dims = 1)

# The reference: the loop the recurrence says.
function looped(a, b)
    h = similar(b)
    for c in axes(b, 2)
        s = zero(eltype(b))
        for t in axes(b, 1)
            h[t, c] = s = a[t, c] * s + b[t, c]
        end
    end
    h
end

# 4096 steps of time, 1024 channels; factors near 1 so the state remembers.
steps, channels = 4096, 1024
a = 0.99 .+ 0.01 .* sin.(reshape(1:steps * channels, steps, channels))
b = sin.(0.7 .* reshape(1:steps * channels, steps, channels))
reference = looped(a, b)
args = map(x -> CuArray(Float32.(x)), (a, b))

function run(name, f)
    c = compile(f, args...)
    h = c(args...)
    ms = minimum(1:10) do _
        t = time_ns()
        c(args...)
        CUDACore.synchronize()
        (time_ns() - t) / 1.0e6
    end
    err = maximum(abs, Array(h) .- reference) / maximum(abs, reference)
    @printf("%-12s relative difference from Float64 %.1e, %.2f ms\n", name, err, ms)
    c
end

run("in sequence", in_sequence)
c = run("in parallel", in_parallel)
println()
explain(c)
