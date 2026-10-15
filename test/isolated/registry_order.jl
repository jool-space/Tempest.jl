# Run in fresh processes by extensions/loading.jl, once per load order given in
# ARGS: the candidate list depends on the registered ranks, not on the order
# in which extensions were loaded. Prints the candidates of a CUDA executable.
using Tempest
for name in ARGS
    @eval import $(Symbol(name))
end
Tempest.activate(cuDNN)
x = CUDACore.CuArray(Float32[0.2, 0.3, 0.4])
ex = prepare(x -> sum(abs2, x), x)
println(join(("$name ($source)" for (name, source) in zip(ex.candidates, ex.candidate_sources)), " → "))
