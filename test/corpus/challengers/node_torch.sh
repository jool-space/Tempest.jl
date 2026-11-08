#!/usr/bin/env bash
# torch's neural ODE training step by its own script, PathMNISTReactant.jl's
# benchmarks/pytorch/bench_node.py (at 7684532), once per compile mode, TF32,
# at the corpus's batch for the shape. Each run prints one JSON line.
#
#   bash node_torch.sh small|full
#
# TEMPEST_CORPUS_TORCH is the python of a venv with torch
# (~/dev/jool/challengers/node-pytorch), TEMPEST_CORPUS_NODE_TORCH the
# directory of bench_node.py and its common.py, and
# TEMPEST_CORPUS_NODE_TORCH_MODES the modes (reduce-overhead max-autotune).
python=${TEMPEST_CORPUS_TORCH:-$HOME/dev/jool/challengers/node-pytorch/bin/python}
scripts=${TEMPEST_CORPUS_NODE_TORCH:-$HOME/dev/jool/challengers/pathmnist-7684532/benchmarks/pytorch}
batch=$([ "$1" = full ] && echo 1024 || echo 128)
cd "$scripts" || exit 1
# Each run compiles from nothing, as Tempest's cold side does: in caches of
# its own, removed after.
cold() {
    local caches
    caches=$(mktemp -d)
    TORCHINDUCTOR_CACHE_DIR=$caches/inductor TRITON_CACHE_DIR=$caches/triton "$@"
    local status=$?
    rm -rf "$caches"
    return $status
}
# A mode that fails prints a line saying so, with the first error it gave.
errors=$(mktemp)
for mode in ${TEMPEST_CORPUS_NODE_TORCH_MODES:-reduce-overhead max-autotune}; do
    if ! cold "$python" bench_node.py --compile "$mode" --precision tf32 --batch-size "$batch" 2> "$errors"; then
        why=$(grep -m 1 -E "Error|error" "$errors" | tr -d '"\\' | cut -c1-160)
        echo "{\"compile\": \"$mode\", \"failed\": \"$why\"}"
    fi
    cat "$errors" >&2
done
rm -f "$errors"
