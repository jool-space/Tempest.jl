# The CUDA target

[`CUDATarget`](@ref Tempest.CUDATarget) runs a program on NVIDIA GPUs; it
needs CUDACore loaded. What it is made with says where the program runs,
what builds its kernels, and how its sums and products may round.

## Devices

A program runs where its arrays are, whatever device the calling task is on,
and a target is at home (see `Placement`) on the first device an argument is
on, or, with every array on the host, on the calling task's device.

`device`, as CUDA numbers them from 0, names the home instead: where arrays
on the host go, and where claims are timed.

## Implementations

The target uses the floor and the `implementations` given, as in
`CUDATarget(cuTileGenerator())`, ties between their estimates going to the
first; by default Tempest's own whose packages are loaded when it is made,
cuBLAS's, cuDNN's and cuTile's (see
[`cuBLASProducts`](@ref Tempest.cuBLASProducts),
[`cuDNNAttention`](@ref Tempest.cuDNNAttention) and
[`cuTileGenerator`](@ref Tempest.cuTileGenerator)).

Another package's implementation takes part only when it is given, and with
none at all, `CUDATarget(precision, online, ())`, the floor runs every
kernel.

## Precision of products

`precision` is the numerics policy for products:

| `precision` | Float32 products |
|:--|:--|
| `:full`, the default | kept in Float32 |
| `:tf32` | may use TF32 tensor cores |
| `:tf32x3` | each Float32 factor split into two TF32 parts, three tensor-core products summed (close to Float32 accuracy) |

Half-precision products always accumulate in Float32, and a product of a
half-precision value, read through its conversion to Float32, and a Float32
one is a half-precision product under any `precision`, the Float32 factor
rounded to the half type: a BFloat16 attention's backward rounds its
probabilities and their gradient so, as FlashAttention and cuDNN do, where
the CPU multiplies in Float32 as the program does. The target trades that
precision for speed as `cu` does, which makes arrays Float32.

## The online softmax

`online` permits the online softmax (see `rewrite`): a sum weighted by a
softmax accumulates in one pass with the softmax's maximum, rescaled as it
grows, which changes rounding.

## A backward that reads rounded sums

`rounded_backward` lets a backward read a forward's sum as the program
rounds it to half precision where the sums of the derivative are exchanged
(see `exchanged_sums`): the softmax backward's Σ dO·o reads the output o as
a half-precision forward stores it, as FlashAttention and cuDNN read theirs,
rather than in Float32, and the forward stores the rounded output only.

With it false the backward reads the sum as the program computes it: in
BFloat16 at 64 features, 8 heads, 1024 positions, the gradients' largest
error is 6.2e-3 of their largest element rather than 6.8e-3 for a loss of
the output's squares, where Σ dO·o nearly cancels the sums it is taken from,
and 5.5e-3 either way when the output's cotangent comes from elsewhere.

## Deterministic sums

`deterministic` keeps a sum's rounding from depending on the order its
blocks run in. Without it a sweep may compute a sum in the sweep's
orientation and add it into its value atomically, as FlashAttention-2's
backward adds the queries' gradient, which then differs from run to run in
its last bits: attention's training step in Float32 at 1024 positions, 64
features, 8 heads, causal and TF32 takes 437 μs, and 702 deterministic.

A sum by key, as an embedding's gradient is, accumulates atomically either
way.
