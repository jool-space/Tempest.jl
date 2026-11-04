<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/tempest-banner-dark.avif">
    <img alt="Tempest.jl" src="docs/src/assets/tempest-banner-light.avif" width="90%">
  </picture>
</p>

> [!NOTE]  
> This package is in active development.

Tempest is a Julia tensor program compiler that aims to be the equivalent of `torch.compile`. It can reuse memory, fuse operations, pick the best of many implementations for an operation, and eliminate launch overhead with CUDA graphs.
```julia
using Tempest, CUDA

loss(w, b, x) = sum(abs2, tanh.(w' * x .+ b))

w = CUDA.randn(32, 64)
b = CUDA.randn(64)
x = CUDA.randn(32, 16)

grad = compile((w, b, x) -> Tempest.withgradient(loss, w, b, x), w, b, x)
l, (dw, db, dx) = grad(w, b, x)
```
