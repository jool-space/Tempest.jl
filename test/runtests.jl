# Each file in test/ runs as its own test, in parallel worker processes:
# `julia --project=test test/runtests.jl [--jobs=N] [--list] [name prefixes...]`,
# as in `julia --project=test test/runtests.jl plan` for one. setup.jl is
# evaluated in every test's module; fixtures/ holds programs that several tests
# include, and bench/ the scripts that time Tempest; neither runs as a test.
# Tests on CUDA skip themselves when no GPU is functional.
using ParallelTestRunner
using Tempest
import CUDACore

const CUDA_FUNCTIONAL = CUDACore.functional()
CUDA_FUNCTIONAL || @warn "CUDA unavailable; GPU tests skipped"

testsuite = find_tests(@__DIR__)
filter!(((name, _),) -> name != "setup" && !startswith(name, "fixtures/") && !startswith(name, "bench/"), testsuite)

# The device's rates are measured here, once and alone: workers share the GPU
# and would measure badly. Each copies the file into its own tuning directory.
if CUDA_FUNCTIONAL
    import cuBLAS, cuTile
    ENV["TEMPEST_CACHE"] = mktempdir()
    foreach(precision -> Tempest.Compiler.calibrate!(Tempest.CUDATarget(; precision)), (:full, :tf32, :tf32x3))
    ENV["TEMPEST_RATES"] = Tempest.Compiler.rates_path(Tempest.CUDATarget())
end

# Every worker loads the same packages, so no test sees an extension that an
# earlier test on its worker happened to load.
const init_code = quote
    using Tempest, Test, NNlib, CUDACore
    import Statistics, cuBLAS, cuTile
    include($(joinpath(@__DIR__, "setup.jl")))
end

# Each worker holds a CUDA context and its own compiled kernels. The GPU's
# free memory says little on a unified-memory device, so cap the default.
args = parse_args(ARGS)
if CUDA_FUNCTIONAL && args.jobs === nothing && !haskey(ENV, "PTR_NUM_JOBS")
    jobs = min(ParallelTestRunner.default_njobs(), 12)
    args = ParallelTestRunner.ParsedArgs(
        Some(jobs), args.verbose, args.quickfail, args.list, args.custom, args.positionals,
    )
end

runtests(Tempest, args; testsuite, init_code)
