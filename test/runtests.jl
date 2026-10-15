# Each file below test/ runs as its own test, in parallel worker processes:
# `julia --project=test test/runtests.jl [--jobs=N] [--list] [name prefixes...]`.
# setup.jl is evaluated in every test's module. fixtures/ holds helpers that a
# CPU test shares with its GPU counterpart, isolated/ scripts that run in
# fresh processes, and packages/ a package outside Tempest; none runs as a test.
using ParallelTestRunner
using Tempest
import CUDACore

const CUDA_FUNCTIONAL = CUDACore.functional()
const CUTILE_ENABLED = CUDA_FUNCTIONAL && get(ENV, "TEMPEST_TEST_CUTILE", "true") == "true"
const CUDNN_FUNCTIONAL = CUDA_FUNCTIONAL && (@eval(import cuDNN); cuDNN.functional())
CUDA_FUNCTIONAL || @warn "CUDA unavailable; GPU tests skipped"
CUDA_FUNCTIONAL && !CUDNN_FUNCTIONAL && @warn "cuDNN unavailable; cuDNN tests skipped"

testsuite = find_tests(@__DIR__)
filter!(testsuite) do (name, _)
    name == "setup" && return false
    startswith(name, "fixtures/") && return false
    startswith(name, "isolated/") && return false
    startswith(name, "packages/") && return false
    name == "cuda/cudnn" && return CUDNN_FUNCTIONAL
    startswith(name, "cuda/") && return CUDA_FUNCTIONAL
    startswith(name, "cutile/") && return CUTILE_ENABLED
    true
end

# Every worker loads the same packages, so no test sees an extension that an
# earlier test on its worker happened to load.
const init_code = quote
    using Tempest, Test
    const CUDA_FUNCTIONAL = $CUDA_FUNCTIONAL
    const CUTILE_ENABLED = $CUTILE_ENABLED
    $(CUDA_FUNCTIONAL ? :(using CUDACore; import cuBLAS, cuDNN) : nothing)
    $(CUTILE_ENABLED ? :(import cuTile) : nothing)
    include($(joinpath(@__DIR__, "setup.jl")))
end

# ChainRules adds scalar rules for Base functions; keep it off the shared workers.
const OWN_WORKER = ("extensions/chainrules", "cuda/chainrules")
test_worker(name) = name in OWN_WORKER ? addworker() : nothing

# Each worker holds a CUDA context and its own compiled kernels. The GPU's
# free memory says little on a unified-memory device, so cap the default.
args = parse_args(ARGS)
if CUDA_FUNCTIONAL && args.jobs === nothing && !haskey(ENV, "PTR_NUM_JOBS")
    jobs = min(ParallelTestRunner.default_njobs(), 12)
    args = ParallelTestRunner.ParsedArgs(
        Some(jobs), args.verbose, args.quickfail, args.list, args.custom, args.positionals,
    )
end

runtests(Tempest, args; testsuite, init_code, test_worker)
