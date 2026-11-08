#!/usr/bin/env bash
# The corpus's GPT training step in torch (gpt_torch.py), at the shape of
# test/corpus/gpt.jl: small, the bench's character model of Tempest's
# docs/src, its vocabulary counted from them as the bench counts it; full,
# GPT-2 124M's geometry at batch TEMPEST_CORPUS_GPT_BATCH (4).
#
#   bash gpt_torch.sh small|full
#
# TEMPEST_CORPUS_TORCH is the python of a venv with torch
# (~/dev/jool/challengers/node-pytorch).
python=${TEMPEST_CORPUS_TORCH:-$HOME/dev/jool/challengers/node-pytorch/bin/python}
here=$(dirname "$0")
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
if [ "$1" = full ]; then
    shape="--vocab 50257 --hidden 768 --heads 12 --kv-heads 12 --ffn 3072 --layers 12 --T 1024 --batch ${TEMPEST_CORPUS_GPT_BATCH:-4}"
else
    vocab=$(find "$here/../../../docs/src" -name '*.md' -print0 |
        xargs -0 "$python" -I -c 'import sys; print(len(set("\n".join(open(f, encoding="utf-8").read() for f in sys.argv[1:]))))')
    shape="--vocab $vocab --hidden 128 --heads 4 --kv-heads 2 --ffn 384 --layers 4 --T 64 --batch 16"
fi
cold "$python" "$here/gpt_torch.py" $shape
