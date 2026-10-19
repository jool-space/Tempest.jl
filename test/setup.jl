# Evaluated in every test's module, after the packages runtests.jl loads: what
# all tests share. Programs that several tests run are in fixtures/programs.jl.
using LinearAlgebra, BFloat16s
import Tempest as T
using Tempest.Compiler
import Tempest.Compiler as C
import Tempest.IR as IR
import Tempest.Operations as Ops

const CUDA_FUNCTIONAL = CUDACore.functional()

# Measurements the tests take stay out of the user's tuning record.
ENV["TEMPEST_CACHE"] = mktempdir()
CUDA_FUNCTIONAL && haskey(ENV, "TEMPEST_RATES") && (mkpath(C.tuning_directory()); cp(ENV["TEMPEST_RATES"], C.rates_path(T.CUDATarget())))

# Reproducible arrays, in Float64 so that finite differences are accurate.
data(dims...; scale = 1.0) = 2.0 .* sin.(scale .* reshape(Float64.(1:prod(dims)), dims...))

# In Float32, which tiles take.
single(x) = x isa AbstractArray{<:AbstractFloat} ? Float32.(x) : x isa AbstractFloat ? Float32(x) : x

# Results agree with a reference wherever they live, a number standing for an
# array of no dimensions.
on_host(x) = x isa AbstractArray ? (ndims(x) == 0 ? Array(x)[] : Array(x)) : x
agree(a, b; rtol = 1.0e-10) = all(map((x, y) -> isapprox(on_host(x), on_host(y); rtol, atol = rtol * 1.0e-2), a, b))

# What `f` throws, in words, or "" when it throws nothing.
message(f) = try f(); "" catch err; sprint(showerror, err) end
