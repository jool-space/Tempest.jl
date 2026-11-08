#!/usr/bin/env bash
# Reactant's neural ODE training step by its script, bench_node_port.jl
# (PathMNISTReactant.jl's benchmarks/julia/bench_node.jl at 7684532 ported
# off ReactantNitro): RK4 unrolled at trace, Enzyme's adjoint, RAdam, batch
# 128, at XLA's default precision. It prints a JSON line for its first run,
# which compiles, and one for its second.
#
#   bash node_reactant.sh
#
# TEMPEST_CORPUS_NODE_REACTANT is the Julia project holding the script
# (~/dev/jool/challengers/node-reactant). XLA does not preallocate the
# memory the GPU shares with the machine.
project=${TEMPEST_CORPUS_NODE_REACTANT:-$HOME/dev/jool/challengers/node-reactant}
XLA_REACTANT_GPU_PREALLOCATE=false exec julia --startup-file=no --project="$project" "$project/bench_node_port.jl" default
