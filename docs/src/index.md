# Tempest.jl

Tempest compiles Julia array programs into fused GPU kernels. A whole training
step, with the loss, its gradient and the optimizer, is traced once from plain
Julia, planned into kernels, and replayed as one CUDA graph. Kernel packages
such as cuTile and cuBLAS compete to run each part, and whatever no package
claims still runs.

```julia
using Tempest, CUDACore, cuTile, cuBLAS

struct Dense{W, B}
    w::W
    b::B
end
(d::Dense)(x) = tanh.(d.w * x .+ d.b)
loss(m, x) = sum(abs2, m(x))

m = Dense(CuArray(randn(Float32, 64, 32)), CuArray(randn(Float32, 64)))
x = CuArray(randn(Float32, 32, 16))

grad = compile((m, x) -> Tempest.gradient(loss, m, x), m, x)
l, dm, dx = grad(m, x)   # dm is a named tuple (w = ..., b = ...)
explain(grad)            # each kernel: what it computes and stores, and what runs it
```

Without a GPU, the same code runs on the CPU with plain arrays. The
[API](api.md) page lists what Tempest exports.
