#!/usr/bin/env bash
# The heat equation's gradient in CUDA.jl by hand (heat_cuda.jl), in the test
# project, at 100 (small) or 1000 (full) steps.
#
#   bash heat_cuda.sh small|full
here=$(dirname "$0")
exec julia --startup-file=no --project="$here/../.." "$here/heat_cuda.jl" "$1"
