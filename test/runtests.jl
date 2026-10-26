# Each file under test/ runs as its own test, in parallel worker processes:
# `julia --project=test test/runtests.jl [--jobs=N] [--list] [name prefixes...]`,
# as in `julia --project=test test/runtests.jl plan/` for the planner's tests.
# A file covers one subject, in a directory named for the part of src/Compiler
# it exercises: program/, index/, plan/, cover/, codegen/ and runtime/.
# extensions/ holds NNlib's and Einops's functions on traced arrays, and cuda/
# what runs only on CUDA; decisions.jl, interfaces.jl and show.jl span the
# parts. setup.jl is evaluated in every test's module; fixtures/ holds
# programs that several tests include, and bench/ the scripts that time
# Tempest; neither runs as a test.
using ParallelTestRunner
using Tempest
import CUDACore

# `functional(true)` throws the reason CUDA is not functional. On CI, whose
# runners have a GPU, that fails the run; elsewhere the warning names it.
const CUDA_FUNCTIONAL = try
    CUDACore.functional(true)
catch err
    haskey(ENV, "CI") && rethrow()
    @warn "CUDA unavailable; GPU tests skipped" reason = sprint(showerror, err)
    false
end

testsuite = find_tests(@__DIR__)
filter!(((name, _),) -> name != "setup" && !startswith(name, "fixtures/") && !startswith(name, "bench/"), testsuite)
# Without CUDA, cuda/ is dropped whole, and the tests elsewhere skip their
# cases on CUDA.
CUDA_FUNCTIONAL || filter!(((name, _),) -> !startswith(name, "cuda/"), testsuite)

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
    using Tempest, Test, NNlib, CUDACore, Einops
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

# A worker on CUDA holds about 3.9 GB from the start, past the runner's
# default of 3.8 GB at which a worker is replaced, so every file would get a
# new one; under coverage a new worker compiles much of Tempest again. A worker
# is replaced past 8 GB here, or JULIA_TEST_MAXRSS_MB.
max_worker_rss = parse(Int, get(ENV, "JULIA_TEST_MAXRSS_MB", "8192")) * 2^20

runtests(Tempest, args; testsuite, init_code, max_worker_rss)
