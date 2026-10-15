# History

What was decided, when, and where the evidence is. Newest first. The linked
reports are dated records; their plans are superseded by
[Architecture](architecture.md#Open-directions).

## October 2026

- **16 October: guard rails and a cost check** (`guard-rails` branch).
  Statistics' `mean`, `var` and `std` work on traced arrays as sums. A GPU
  benchmark test (`test/benchmarks/guard_rails.jl`) asserts speed against the
  CUDA kernels instead of plan shapes. It found that full reductions still lost
  to the CUDA kernels after the occupancy hint (256-value tiles; now 1024), that
  the second-axis schedule left 16 blocks on 2048 rows and stepped 8 columns at
  a time (now at least 64 blocks and 32 columns), that cuTile's scalar full sums
  completed serially (a Float16 sum 3.7 times slower), and that a lone full sum
  took three launches. Graph regions and estimating proposals are now kept only
  when a bytes-and-launches estimate does not say they lose
  ([Architecture](architecture.md#Selection-by-estimate)).
- **15 October: outside review** ([report](outside-review.md)). Coverage was
  being treated as profitability: cuTile full reductions ran 4–7 times slower
  than the kernels they displaced. Fixed the occupancy hint of partial-sum
  kernels, added exact canonical forms (permutations compose, reshapes sink,
  identity layout chains collapse) and rank-N graph domains.
- **15 October: spelling invariance** ([audit](spelling-invariance.md)).
  `logsumexp` became a fold, `batched_mul` a primitive product, `dropout` and
  `causal_fill` pointwise over positions; proposals may claim calls; Base's
  pure functions are recomputable. The rule for primitives and calls dates
  from here.
- **15 October: implementation registry, device rates, layout copies**
  ([log](review-decisions.md#An-implementation-registry;-device-rates;-layout-copies-—-15-October-2026)).
  Default candidates come from a registry ordered by rank; cost models read
  device-derived rates.
- **15 October: Float32 accumulation for half-precision folds**
  ([log](review-decisions.md#Float32-accumulation-for-half-precision-folds-—-15-October-2026)).
- **15 October: why cuDNN's fused products lose**
  ([log](review-decisions.md#Why-cuDNN's-fused-products-lose,-and-the-rule-that-replaced-4096-—-15-October-2026)):
  tile order beyond L2 and product throughput; column chunks and an estimate
  replaced a size threshold.
- **15 October: occupancy hints remeasured**
  ([log](review-decisions.md#Occupancy-hints-remeasured-—-15-October-2026)).
- **14 October: implementation-proposed regions**
  ([design](proposed-regions.md)), with cuDNN products and epilogues as the
  first consumer; cuDNN attention ([implementations](implementations.md#cuDNN)).
- **14 October: pruning.** Research code moved to the tag `research-2026-09`.

## September 2026

- **23 September: workload pass** ([transformer step](transformer-step.md)).
  A GPT training step as one captured graph, compared with Reactant/XLA;
  generated CUDA kernels stopped spilling index metadata; packed and split
  reduction schedules ([reduction domains](reduction-domains.md)).
- **22 September: consolidation and rename** from Deductor to Tempest; the
  older whole-column reduction schedule was removed after measurement.
- **21–23 September: compiler direction.** A TensorIR experiment compiled a
  residual RMSNorm step; the conclusion was to keep Tempest's graph as the
  source of meaning, select regions on it before storage, and let providers
  supply schedules (the cuTile graph provider). The principles are in
  [Architecture](architecture.md); the experiments in the
  [detailed log](review-decisions.md#Compiler-direction:-experiments-and-integration-—-21–23-September-2026).
- **20 September: kernel composition.** Implementation reverse rules let a
  kernel library replace a forward/backward pair with private saved values
  ([reverse rules](reverse-rules.md)).
- **7 September: functional state.** `with_update`, donation in place or
  staged, and grouping of independent elementwise updates
  ([many-array updates](many-array-updates.md)).
- **6 September: first review** of the prototype: typed operations and
  regions, one selection protocol, scalar rules from ChainRules
  ([scalar rules](scalar-rules.md)).
