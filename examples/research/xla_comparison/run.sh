#!/bin/bash
# Traces the GPT step under Reactant/XLA (nanoGPT's train.jl) and Tempest with
# Nsight Systems, one run at a time. Log clocks alongside:
#   nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,utilization.gpu \
#     --format=csv,noheader -lms 500 > $OUT/clocks.log
# then run analyze.py, categories.py and clocks_for.py (trace, steps, skipped
# steps) on each .sqlite from $OUT (clocks_for.py reads clocks.log there).
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NANOGPT=${NANOGPT:-$HOME/dev/jool/nanoGPT}
TEXT=${TEXT:-$HOME/dev/jool/Reactant/docs/src}
S=${OUT:-$PWD/xla_comparison_out}
mkdir -p "$S"
NSYS="nsys profile -t cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true --export=sqlite"
GPT2X="NANOGPT_HIDDEN=768 NANOGPT_LAYERS=12 NANOGPT_HEADS=12 NANOGPT_KV_HEADS=12 NANOGPT_FFN=3072 NANOGPT_T=256 NANOGPT_B=16"
GPT2T="TEMPEST_GPT_HIDDEN=768 TEMPEST_GPT_LAYERS=12 TEMPEST_GPT_HEADS=12 TEMPEST_GPT_KV_HEADS=12 TEMPEST_GPT_FFN=3072 TEMPEST_GPT_T=256 TEMPEST_GPT_B=16"
DRIVER=$ROOT/examples/research/xla_comparison/tempest_driver.jl

xla() { (cd "$NANOGPT/gpuenv" && env JULIA_PKG_OFFLINE=true NANOGPT_FRONTEND=vanilla NANOGPT_TEXT=$TEXT "$@"); }
tempest() { (cd "$ROOT" && env TEMPEST_GPT_TEXT=$TEXT "$@"); }

xla NANOGPT_PRECISION=highest NANOGPT_STEPS=60 $NSYS -o "$S/xla_small_full" julia --project=. ../train.jl > "$S/xla_small_full.out" 2>&1
tempest TEMPEST_GPT_PRECISION=full TEMPEST_GPT_STEPS=60 $NSYS -o "$S/tempest_small_full" julia --project=examples "$DRIVER" > "$S/tempest_small_full.out" 2>&1
xla NANOGPT_STEPS=16 $GPT2X $NSYS -o "$S/xla_gpt2" julia --project=. ../train.jl > "$S/xla_gpt2.out" 2>&1
tempest COMPOSED=1 TEMPEST_GPT_PRECISION=tf32 TEMPEST_GPT_STEPS=16 $GPT2T $NSYS -o "$S/tempest_gpt2_composed" julia --project=examples "$DRIVER" > "$S/tempest_gpt2_composed.out" 2>&1
tempest TEMPEST_GPT_PRECISION=tf32 TEMPEST_GPT_STEPS=16 $GPT2T $NSYS -o "$S/tempest_gpt2_fused" julia --project=examples "$DRIVER" > "$S/tempest_gpt2_fused.out" 2>&1
