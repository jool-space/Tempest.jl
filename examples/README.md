# Examples

Programs written in plain Julia, compiled by Tempest and checked against a
reference in Float64 on the CPU. Each prints how far its results are from the
reference, `explain`'s account of the kernels (what each computes, stores and
recomputes, and which package runs it), and one line of kernels, memory and
time. Run them from the repository's root with the test environment, on a
GPU:

```
julia --project=test examples/attention.jl
```

- `attention.jl`: causal attention's training step, with NNlib's
  `batched_mul` and `softmax`: the gradient through a masked softmax, and what
  the plan stores for the backward against what FlashAttention-2 keeps.
- `mlp.jl`: four dense layers on structs, trained by gradient descent: the
  loss, its gradient and the update as one compiled step, called once per step.
- `scan.jl`: a linear recurrence `h = a h′ + b` along time, as Base's
  `accumulate` (in sequence, as Base promises) and as a `GenericFold`, which
  claims its combine associates and runs as a parallel scan.

A GPT's training step, with Adam and grouped-query attention, is
`test/bench/gpt.jl`.
