#!/usr/bin/env bash
# torch's LSTM training step by its script, torch_lstm.py: cuDNN's nn.LSTM,
# and Flux's cell as a plain loop eager and under torch.compile, each also
# replayed as a captured CUDA graph, TF32, at batch 64 (small) or 1024
# (full); the loss and the gradient of every weight, no optimizer step.
#
#   bash lstm_torch.sh small|full
#
# TEMPEST_CORPUS_TORCH is the python of a venv with torch
# (~/dev/jool/challengers/node-pytorch), TEMPEST_CORPUS_LSTM_TORCH the script
# (~/dev/jool/challengers/lstm/torch_lstm.py).
python=${TEMPEST_CORPUS_TORCH:-$HOME/dev/jool/challengers/node-pytorch/bin/python}
script=${TEMPEST_CORPUS_LSTM_TORCH:-$HOME/dev/jool/challengers/lstm/torch_lstm.py}
batch=$([ "$1" = full ] && echo 1024 || echo 64)
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
cold "$python" "$script" "$batch" tf32
