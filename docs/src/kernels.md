# Writing a kernel

A kernel package implements an operation on a target. Tempest asks it for a
kernel at each call of the operation, prices that kernel against the
operation's reference fused into its neighbours and against every other
package's kernel, and runs the winner. The package never sees the plan.

> This page describes the interface, and *Not built* at the end says what is
> missing from it. The older protocol, a struct subtyping `Implementation`
> with `claims_call`, `epilogue_results` and `build_call` methods, remains for
> a kernel that states products at a rate.

## The operation

An operation belongs to the library that defines it, not to any kernel. It
states what is computed, in array code. Its backward keeps a name of its own,
so that a kernel can claim it too. A value the backward needs is a result of
the forward: there is no checkpoint mechanism, and a result nobody reads is
not stored.

```julia
module Norms
using Tempest

function rmsnorm_forward(x, w; eps, offset)
    rstd = 1 ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps)
    (y = x .* rstd .* (w .+ offset), rstd = rstd)
end
@operation rmsnorm_forward(x, w; eps, offset)

function rmsnorm_backward(x, w, rstd, dy, drstd; eps, offset)
    M = size(x, 1)
    dd = sum(dy .* (w .+ offset) .* x; dims = 1) ./ M
    dx = rstd .* (dy .* (w .+ offset) .- rstd .^ 2 .* dd .* x) .- drstd .* rstd .^ 3 .* x ./ M
    dw = vec(sum(dy .* x .* rstd; dims = 2))
    (dx = dx, dw = dw)
end
@operation rmsnorm_backward(x, w, rstd, dy, drstd; eps, offset)

Tempest.has_pullback(::typeof(rmsnorm_forward)) = true
Tempest.verb_pullback(::typeof(rmsnorm_forward), (x, w), y, dy; eps, offset) =
    Tuple(rmsnorm_backward(x, w, y.rstd, dy.y, dy.rstd; eps, offset))

rmsnorm(x, w; eps = 1f-6, offset = 0f0) = rmsnorm_forward(x, w; eps, offset).y
end
```

Without any kernel package this is complete. The forward is one generated
kernel and so is the backward, and `rmsnorm` runs eagerly on plain arrays.

## The kernel package

```julia
module TileNorms

using Tempest, cuTile
using Tempest: CUDATarget, Spec, Kernel, bytes, register!
using Norms: rmsnorm_forward, rmsnorm_backward

include("tiles.jl") # the tile programs: rms_norm_fwd, rms_norm_bwd_dx_partial_dw, rms_norm_bwd_dw

function forward(::CUDATarget, x::Spec{Float32, 2}, w::Spec{Float32, 1}; eps, offset)
    M, N = size(x)
    # Two reads of x and one write of y: more than the one pass Tempest assumes.
    Kernel(; cost = 3bytes(x), params = (TILE_M = (128, 256, 512),)) do (y, rstd), (x, w), (; TILE_M)
        launch(rms_norm_fwd, N, x, w, y, rstd, Float32(offset), Float32(eps), Constant(TILE_M))
    end
end

function backward(::CUDATarget, x::Spec{Float32, 2}, w::Spec{Float32, 1}, rstd::Spec{Float32, 2},
        dy::Spec{Float32, 2}, drstd::Spec{Float32, 2}; eps, offset)
    M, N = size(x)
    groups = 64
    Kernel(; cost = [4bytes(x), 4M * groups],
            scratch = (Spec(Float32, M, groups), Spec(Int, groups))) do (dx, dw), (x, w, rstd, dy, drstd), (partial, locks)
        launch(rms_norm_bwd_dx_partial_dw, N, dx, dy, fill!(partial, 0), x, w, vec(rstd), vec(drstd),
            fill!(locks, 0), Constant(Float32(offset)), Constant(groups), Constant(256))
        launch(rms_norm_bwd_dw, cld(M, 64), partial, dw, Constant(64), Constant(64))
    end
end

function __init__()
    register!(TileNorms, rmsnorm_forward => forward, rmsnorm_backward => backward)
end

end
```

**Two moments.** A method runs when Tempest prices a call, before any buffer
exists: its arguments are `Spec`s. The kernel's body runs when the program
runs: its arguments are the target's arrays, reshaped to the program's shapes,
host `Vector`s on the CPU and `CuArray`s on CUDA. The body receives the
results first, then the arguments, then scratch, parameters and the forms it
absorbs where the kernel declared them. The arguments shadow the specs of the
same names on purpose, so the seam is visible where it is.

**`Spec{T, N}`** is an element type and dimensions: `size`, `eltype` and
`bytes`, nothing else. There is no layout in it. A kernel reads its arguments
densely in their program shapes and writes its results the same way, and
Tempest copies where a value is not stored that way.

**`Kernel`** carries what Tempest needs to choose and run it:

- `cost`: bytes moved, a number for one launch or a vector with a number for
  each. Leave it out for one launch that moves the bytes its group reads and
  writes, which Tempest computes from the plan: that counts what an absorbed
  term reads and leaves out a result nobody stores. It is an estimate: it
  ranks kernels and says which are close, and close calls are measured (see
  below). A kernel package states no time; the device's rates are Tempest's.
  Until they are, a `Work` in place of a number states flops at a rate, for
  a kernel whose products the bytes do not rank.
- `scratch`: specs of the workspace the kernel needs. Tempest allocates them
  when it builds the kernel, and they are the kernel's for the program's life.
- `params`: named candidate values. Tempest passes the first candidate of
  each to the body.
- `epilogue`: per result, the forms of work after it that the kernel absorbs
  (see *Forms*).
- the body. It runs on the target's stream once per execution, and neither
  synchronizes nor allocates. A result nobody reads arrives as `nothing` and
  may be left unwritten; the kernel above does not store `rstd` at inference.

**Declining.** A method that does not apply is no method. A method may also
return `nothing`, for conditions the types do not express, such as a size the
kernel cannot tile.

**Variants.** Three things that are easy to confuse:

- Another element type or rank is another method of the same function.
- Another algorithm is another function, registered under the same operation:
  `rmsnorm_forward => forward_resident, rmsnorm_forward => forward_chunked`.
  Each is priced on its own and `explain` names which won.
- The knobs of one algorithm are its `params`. A function may also return a
  vector of kernels, which Tempest treats the same way.

## Forms

Work after a kernel's result is where fusion with a hand-written kernel
happens, and it comes in a few forms a kernel can name per result:

- a scalar scale, `y * s` with `s` a scalar;
- an add, `y + c` with `c` a value of the result's shape read at the
  result's own index, which the kernel may accumulate into in place;
- a pointwise function of the result, applied to each output tile before it
  is stored.

A kernel that names a form on a result receives the matching terms when the
program has them: the scalar, the array to add, or the function to apply, and
the planner stores the result after that work rather than before it. A term
the kernel has no form for stays outside, in a kernel of its own or in the
next reader's. The terms arrive in the body's last group, `after`, in the
structure of `epilogue`: for a pointwise function, a function to apply to the
result, which is `identity` where the program has no work after that result.
On the CPU it takes the result's array and returns the array to store; on CUDA
it is a tile function of the output tile and its index. The body stores the
result after it, in the array it was given.

The same forms are what a library takes as coefficients and epilogues:
cuBLAS's alpha and beta are the scale and the add, and its fused bias and GELU
are pointwise functions. A gradient accumulated across uses of a weight is an
add on the backward kernel that produces one contribution.

## What Tempest does with it

At each call of `rmsnorm_forward` in a program, Tempest has these options:
every kernel each registered function returns for the target and the call's
specs and keywords; the reference expanded and fused into its neighbours, as
the generator compiles it; and the floor. It prices each by its cost through
the device's rates. The rates are Tempest's: bandwidth, launch overhead and
the product rate at each precision, measured on the device the first time it
is used and kept in the tuning record. Options within a factor of 1.5 of the
best are close calls: at `compile` with `measure = true`, the default, they
are built and timed as the minimum of several alternating runs, the
parameters of each kernel enumerated within a budget, and the times go into
the tuning record, keyed by package, function, parameters, shapes, element
types, keywords and device. A later process reads the record and measures
nothing. `explain` prints the winner by package and function, with its
parameters and whether it was estimated or measured. A kernel whose build
fails is withdrawn and the next option runs. The reference is always an
option, so removing the package changes how fast the program runs, never
whether it runs.

## Not built

- Products in cost. A kernel states bytes, and a `Work` may state flops at the
  rate the kernel reaches. The hand-written attention keeps `claims_call`,
  `epilogue_results` and `build_call` until Tempest counts products and
  measures rates.
- Measured rates. Today the product rates are two hand-entered tables in the
  cuBLAS and cuTile extensions, with other devices scaled from one. A rate
  that is wrong by more than the close-call factor decides without a
  measurement.
- Parameter search. Measurement and the record exist; the enumeration and
  its budget do not. A kernel runs with the first candidate of each
  parameter, and `explain` says so. The generator returns one schedule per
  group.
- Forms. Only `pointwise` exists. Its term is a host function on the CPU and a
  cuTile tile function on CUDA, which no body takes yet, so a CUDA kernel in
  another language cannot take one. The scale and the add are not recognized,
  and cuBLAS takes no epilogue.
- Layouts. A spec carries no layout, so a kernel cannot accept a strided
  argument, and a permuted argument is copied before the call even where the
  kernel could have read it in place. The spec should carry the layout, dense
  by default, with a kernel accepting or declining on it; and where the
  argument is another kernel's result, Tempest should write it in the layout
  the consumer accepts rather than copy.
- Prologues. A kernel cannot take work before its arguments. The generated
  path can.
- Written-out patterns. A package recognizes a computation spelled in
  primitives with `candidates` on an `Implementation` registered beside its
  pairs.
- Group claims. The generator and cuBLAS's recognition of products claim
  groups of index-level values through a `GroupView`, and keep the
  `Implementation` struct. This page is for kernels of named operations.
