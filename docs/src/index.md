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

Without a GPU, the same code runs on the CPU with plain arrays.

## What holds

Tempest owns coverage and kernel packages own speed. A program written with
Base's arrays and the primitives on the [API](api.md) page runs with no
kernel package loaded, on the kernels Tempest generates for itself, so
loading a package or removing one changes how fast a program runs and never
whether it runs. Each kind of change has one place: a new function is array
code in the package that owns it, a new kernel is a change to a kernel
package, and a better optimization is a change to Tempest.

What else holds is stated by the tests, not here. A property not yet met is
a broken test under a name that says what would close it, and `TRIALS.md`
in the repository is the docket of what the code is being measured against.

## Where to go next

The [API](api.md) page documents the functions Tempest exports for compiling,
differentiating and explaining a program, the primitives programs can call,
and the interface an implementation package writes to.
