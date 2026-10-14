# Tempest: working notes before the first executable

These notes capture a conversation, not a specification. They describe our current understanding, including guesses that implementation should overturn. They are not acceptance criteria, a roadmap that must be completed in order, or authority over working code and measurements.

**For future contributors and coding agents:** do not turn the headings below into commandments. Do not preserve an abstraction merely because it appears here. When evidence changes the design, change the design and these notes. Small experiments that expose a bad assumption are progress. Ask about consequential changes in direction; ordinary implementation details do not need to be elevated into doctrine.

There is no settled public API, package decomposition, IR, differentiation strategy, or precision policy yet. Code fragments are sketches.

## What we are trying to achieve

Clean Julia model code with competitive CUDA training performance and deliberate memory use. Not an eager layer library with a graph-capture wrapper, and not a promise to compile arbitrary Julia into one GPU kernel.

The motivating experience is that good individual kernels are not enough. Composition can introduce unnecessary materialization, long-lived backward scratch, gradient temporaries, representation conversions, and launch overhead. We want to express models mathematically while making those execution decisions somewhere inspectable.

A possible description:

> Compile Julia tensor code into prepared training programs, combining vendor implementations and generated kernels with explicit precision and storage planning.

“Competitive” needs matched measurements: model and shapes, hardware, numerical policy, forward and backward, optimizer work, memory, and steady-state step time. Compilation and preparation costs should be reported separately. CUDA capture cannot compensate for poor kernels or excess memory traffic. Neither Julia nor this project gets a performance exemption because its model code is attractive.

The project was initially called Conductor, but that name was already registered, and then Deductor, which read as too negative. Tempest fits the jool-space gas-giant theme: a storm on a gas giant is a stable structure that keeps its shape inside turbulent flow, as a prepared schedule does inside the model code it was derived from. The *tempus*/*tempo* root fits a program flowing through its schedule, and much of the work is deciding where the temporary buffers go. It need not implement every kernel it schedules.

## The existing pieces are assets, not obligations

The surrounding repositories already provide substantial machinery:

- **Stiletto:** traces Julia array expressions and lowers them to cuDNN graphs. Its fixed pointwise vocabulary reflects cuDNN's supported operations. Its existing compiler should remain useful even if Tempest experiments with a different frontend.
- **Tylo:** cuTile implementations including normalization, attention, and backward kernels, with saved-state and scratch concepts. Its standalone AD adapters are useful, but their allocation lifetimes need not dictate planned training execution.
- **Vallmo:** concrete model code and specialized CUDA/cuTile kernels. Its separation of weight data and computational verbs is attractive; inference lifetime patterns do not automatically extend to training.
- **CUDA and PTX:** kernel compilation/runtime and low-level instruction access. PTX is an implementation tool, not necessarily a separate model-level backend.
- **Microscaling:** real composite array representations, with logical element type separate from physical elements and scales, explicit block geometry, and swizzled scale storage. Stiletto already interprets whole-array stores into blockscaled destinations as quantization, potentially fused with the producer.
- **Pol:** storage descriptions and materialization, arena/frame lifetimes, and vocabulary for outputs, scratch, and checkpoints. This is useful substrate, not yet a whole-program lifetime planner.
- **cuBLASLt and cuDNN:** strong implementations that we should exploit, not replace as a matter of principle.

`Pol.Allocating`, `Shadowed`, Tylo's `Duplicated`, and related wrappers are provisional conveniences. No compatibility requirement follows from their appearance in the conversation. In particular, a kernel's overwrite destination is not interchangeable with an AD system's derivative-storage semantics.

The intended work is the missing connection between mathematical composition and these implementations. It should not duplicate physical representations already expressed by Microscaling or allocator machinery already expressed by Pol merely to make everything look uniform.

## A lesson from Vall

Vall concentrated on choosing an implementation for an already concrete call: registration, feasibility, cache keys, measurement buckets, and tuning. Those mechanisms can be useful, but the call boundary had already fixed many important decisions, including output allocation and the extent of the operation being optimized.

The harder questions are whether operations should share a kernel, whether an intermediate should exist at all, which representation a consumer needs, what backward retains, and when storage dies. A sophisticated local selector cannot recover choices hidden by its input abstraction.

Keep the useful ideas—reference implementations, build/apply separation, inspectable plans, provider-owned implementations, explicit workspace—but resist rebuilding the global selection framework before there is a real training computation to organize.

Also avoid documenting desired properties as established guarantees. A mutating function can allocate internally. A reference path can be unsupported or unsuitable for capture. Plan validity can depend on context and alignment, not just shape. Tuning mutating operations can alter optimizer, RNG, gradient, or scaling state. These are contracts to establish through implementation and tests, not slogans.

## The distinction that currently seems important

A **tensor program** describes values and computation before committing every value to storage. A **prepared execution** describes concrete implementations, representations, buffers, dependencies, and execution state.

That distinction need not imply two elaborate class hierarchies. Nor does it require a universal graph IR before the first test. It does suggest keeping logical value identity separate from a particular buffer early enough to make useful decisions.

A plausible flow is:

```text
Julia tensor code
    → tensor computation and derivatives
    → saved/recomputed work and implementation regions
    → physical representations and kernel schedule
    → storage/workspace plan
    → ordinary launches or captured replay
```

This is explanatory, not a mandatory sequence of compiler passes. Differentiation, implementation choice, checkpointing, and precision interact. A selected attention implementation, for example, may determine the auxiliary state its backward needs. Experiments should tell us which decisions can be separated and which must be revisited together.

The prepared program might own selected library plans, kernels, stable storage, and optional graph executables. It should eventually explain what it chose. A global cache may help preparation, but should not be the only place the executable's effective decisions exist.

## Fusion should permit boundaries, not forbid them

One semantic operation may require several launches. Several semantic operations may become one launch. A kernel corpus should support both.

For an expression such as:

```julia
y = custom_activation.(A * B .+ bias)
z = sum(y; dims = 1)
```

possible implementations include a supported cuDNN region, a strong vendor GEMM followed by a generated activation/reduction kernel, or a specialized fused GEMM. The custom GEMM is not automatically the best choice. Avoiding a materialization may not compensate for reduced GEMM efficiency.

Reduction fusion depends on tile ownership, reduction axes, and inter-block communication. It is not just appending a scalar expression to a store.

A cuTile generation path is attractive for generic device-compilable scalar expressions and selected reductions. The frontend should preserve broadcast expressions rather than eagerly materializing them. Recognizable expressions can still lower to cuDNN's vocabulary; other supported callables can specialize and inline in generated kernels. GPU compilability alone does not establish differentiability.

Stiletto as a regional backend is valuable. Extending it with cuTile is also an experiment worth considering. But “another Stiletto backend” does not by itself supply partitioning, differentiation, or lifetime planning. We should reuse machinery where it fits, without pretending those missing decisions are merely backend emission.

## Differentiation and kernel contracts

A small tensor-level reverse pass with explicit primitive derivative recipes looks like a tractable beginning. This is not yet a decision against existing AD tools. Scalar AD, ChainRules integration, and other frontend strategies should be evaluated on actual needs.

Differentiate logical operations rather than allocator bookkeeping or the internals of vendor libraries. Preserve specialized operations such as attention long enough to use efficient backward implementations.

A useful implementation description may include:

- supported shapes, layouts, representations, and numerical modes;
- outputs and optional saved auxiliary results;
- inputs or outputs that backward must retain;
- workspace, including initialization and algorithm-dependent sizes;
- reads, writes, aliasing permissions, and overwrite/accumulate capabilities;
- a matching backward implementation or a decomposition.

These are questions the executor must eventually answer, not a required interface with seven methods. The forward and backward of a selected implementation may share an opaque saved-state format. Different providers need not standardize their internal attention statistics.

Standalone `rrule`s can remain interoperability adapters. They need not be the optimized training executor's representation. For example, allocating backward scratch during forward and retaining it in every pullback makes integration convenient but can keep many mutually reusable buffers alive. The planner should be able to run the same kernels with different lifetimes.

Gradient accumulation deserves explicit treatment. An overwrite-only kernel can be a good primitive, but a GEMM capable of accumulating directly into parameter-gradient storage should not be forced through a temporary plus an addition. Shared/tied parameters must retain one logical identity even when they have multiple compute representations.

## Memory is about last use, not just allocation speed

An arena does not discover when an activation dies. A pullback retaining an array wrapper does not prevent its arena bytes from being reclaimed. A gradient returned from a retracted scratch frame is invalid even if its wrapper remains reachable.

Likely lifetime categories include persistent training state, saved forward values, transient intermediates, per-operation workspace, gradient accumulators, and stable input/output buffers. The categories may change; the lifetime distinctions cannot simply be ignored.

Pol's frames are a useful implementation mechanism. LIFO need not become the semantic lifetime contract for residual branches, shared values, or later multi-stream work. A single-stream schedule with last-use buffer reuse is a reasonable experiment. Reuse must respect aliases, device execution order, and any persistent references held by library plans or graph executions.

Checkpointing is a save/recompute decision, not merely allocating a buffer called `checkpoints`. It needs access to dependencies and, for stochastic work, reproducible randomness.

## Mixed precision and MXFP8

Precision should be explicit per role or operation, not inferred globally from `eltype`.

A trainable parameter can have an optimizer/master representation, one or more compute representations, and a gradient accumulator. FP32 master weights and moments, BF16 compute, and FP32 parameter-gradient accumulation are one starting policy, not a universal requirement.

Microscaling already represents composite physical storage. The new question is who selects and schedules those representations, not how to invent another quantized-array wrapper.

For `Y = XW`, forward, input gradient, and weight gradient are different GEMMs with potentially different layout and block-scaling needs. Transposing an already block-quantized matrix preserves its existing groups; it does not generally produce a newly quantized representation grouped along the other axis. Backend requirements may therefore require multiple orientations, with associated storage or recomputation costs.

Treat quantized forward/backward GEMMs as explicit numerical implementations of a logical training operation. Do not expect differentiating packing and scale computation to accidentally define the desired training algorithm. Higher-precision backward, selected low-precision gradients, and sensitive layers kept in higher precision should remain possible.

Stores into Microscaling destinations provide a concrete conversion mechanism. The planner can eventually choose where to fuse those stores, when to refresh quantized weights after updates, and which representations to retain. Correctness includes numerical training behavior, not just matching the storage format.

## Capture and shapes

CUDA graph capture is a promising production execution target, not the foundation of AD or memory management. Ordinary launches should be able to use the same prepared kernels and buffers. Uncaptured does not have to mean allocation-heavy eager execution.

Preparation must provision workspace, compile and tune, and warm relevant paths before capture. Warmup must not silently advance real optimizer or RNG state. Repeated replay must be tested; one launch is insufficient.

Captured pointers and ordinary scalar kernel arguments are fixed unless explicitly updated. Changing a Julia variable or rebuilding an array wrapper does not update the captured launch. Device-resident changing state is useful where implementations support it. RNG advancement, checkpoint replay, learning rates, counters, loss scaling, and quantization state need deliberate semantics.

Symbolic tensor shapes, dynamic kernel dimensions, and dynamic CUDA graph execution are distinct. The same grid size is not enough to make an old capture correct for a new shape; captured strides and dimensions also matter. CUDA offers update mechanisms subject to restrictions, but patching arbitrary captured vendor internals is not a sensible initial dependency.

Concrete shapes and one stream are a good first scope. Later, a few shape buckets with fixed capacity and device-side valid lengths may be sufficient for many workloads. Padding must be excluded correctly from loss, reductions, and backward; a capacity-sized GEMM still does capacity-sized work. Sharing arena storage across concurrent executions requires explicit ownership or ordering.

## The experiment to start with

A deliberately small numerical example:

```julia
f(x, w) = sum(abs2, tanh.(w' * x))
```

Try to produce its forward and backward using vendor GEMMs, generated cuTile pointwise/reduction work, explicit saved values, and planned storage. Validate against a straightforward reference. Run uncaptured first, then capture the same execution.

Replacing `tanh` with a custom scalar expression is a useful next test, with an explicit derivative rule if necessary. It probes whether this is composition rather than a hardcoded menu of named activations.

A residual normalization/gated-MLP block can then test real kernel integration, saved-state lifetimes, accumulation, and region lowering through Stiletto. MXFP8 destinations test whether physical representation choices can change without rewriting the model. Attention tests paired implementations with backend-specific auxiliary state.

This order is tentative. If the tiny example hides a decisive issue, use a better example. Do not grow an abstraction solely to satisfy the example's spelling.

Potentially useful first evidence:

- outputs and gradients agree with a reference under stated tolerances;
- the plan exposes kernel boundaries and saved values;
- backward scratch can be shared rather than retained per layer;
- planned reuse does not change numerical results;
- repeated captured execution agrees with uncaptured execution;
- timing and peak storage are reported without conflating host allocations, device outputs, and library workspace.

Full model/optimizer capture, symbolic shapes, general mutation, a universal tuner, distributed execution, and arbitrary Julia AD need not block that evidence. They are future questions, not commitments or permanent exclusions.

## How to keep this project honest

Prefer a working vertical slice to a framework with fake providers. Let a second real workload reveal what is genuinely shared. Do not mistake extensive prose for established invariants, or turn exploratory names into compatibility burdens.

Performance policy should be visible and reproducible. Julia's promise here is not the absence of hardware-specific decisions; it is that those decisions can be represented, inspected, and changed in Julia rather than existing only as obscure environment flags. A faster result obtained with different precision or omitted work is a different comparison.

The question to keep asking is:

> Can we change fusion boundaries, precision, saved-state strategy, and storage reuse without rewriting the model—and can we explain the resulting execution?

If the answer improves through a smaller or different design than these notes suggest, take that design.
