# Each file under test/ runs as its own test, in parallel worker processes:
# `julia --project=test test/runtests.jl [--jobs=N] [--list] [name prefixes...]`,
# as in `julia --project=test test/runtests.jl plan/` for the planner's tests.
# A file covers one subject, in a directory named for the part of src/Compiler
# it exercises: program/, index/, plan/, cover/, codegen/ and runtime/.
# extensions/ holds NNlib's, Einops's and SpecialFunctions' functions on traced
# arrays and numbers; cuda/ what runs only on CUDA; decisions.jl,
# interfaces.jl and show.jl span the parts. setup.jl is evaluated in every test's module; fixtures/ holds
# programs that several tests include, bench/ the scripts that time Tempest,
# fuzz/ the randomized program checker (fuzz/run.jl) with the faults it
# found open, and corpus/ whole programs run against their challengers
# (corpus/run.jl); none runs as a test.
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
filter!(((name, _),) -> name != "setup" && !startswith(name, "fixtures/") && !startswith(name, "bench/") && !startswith(name, "fuzz/") &&
    !startswith(name, "corpus/"), testsuite)
# Without CUDA, cuda/ is dropped whole, and the tests elsewhere skip their
# cases on CUDA.
CUDA_FUNCTIONAL || filter!(((name, _),) -> !startswith(name, "cuda/"), testsuite)

# The device's rates the plans are priced on. Where the tests carry them for
# the device (fixtures/rates), those: calibrated anew, an estimate close to
# another's falls either way from run to run, and in CI the other job shares
# the device, so the plans the tests check would differ by chance. Otherwise,
# or with TEMPEST_CALIBRATE set, they are measured here, once and alone:
# workers share the GPU and would measure badly. Each worker copies the file
# into its own tuning directory. A fixture is the file this writes, copied.
if CUDA_FUNCTIONAL
    import cuBLAS, cuTile, cuDNN
    ENV["TEMPEST_CACHE"] = mktempdir()
    fixed = joinpath(@__DIR__, "fixtures", "rates", basename(Tempest.Compiler.rates_path(Tempest.CUDATarget())))
    if isfile(fixed) && !haskey(ENV, "TEMPEST_CALIBRATE")
        ENV["TEMPEST_RATES"] = fixed
    else
        foreach(precision -> Tempest.Compiler.calibrate!(Tempest.CUDATarget(; precision)), (:full, :tf32, :tf32x3))
        ENV["TEMPEST_RATES"] = Tempest.Compiler.rates_path(Tempest.CUDATarget())
    end
end

# Every worker loads the same packages, so no test sees an extension that an
# earlier test on its worker happened to load.
const init_code = quote
    using Tempest, Test, NNlib, CUDACore, Einops
    import ChainRulesCore, SpecialFunctions, Statistics, cuBLAS, cuDNN, cuTile
    include($(joinpath(@__DIR__, "setup.jl")))
end

# Each worker holds a CUDA context and its own compiled kernels, about 4.4 GB
# under coverage. The runner's default divides free memory by 2 GB, but under a
# memory limit free memory is the limit less the file cache earlier jobs left,
# which gave CI's 20 GiB runners 1 to 3 jobs; here the limit and the CPUs decide.
args = parse_args(ARGS)
if CUDA_FUNCTIONAL && args.jobs === nothing && !haskey(ENV, "PTR_NUM_JOBS")
    cpus = @static isdefined(Sys, :EFFECTIVE_CPU_THREADS) ? Sys.EFFECTIVE_CPU_THREADS : Sys.CPU_THREADS
    jobs = clamp(min(cpus, Int(Sys.total_memory() ÷ (5 * 2^30))), 1, 12)
    args = ParallelTestRunner.ParsedArgs(
        Some(jobs), args.verbose, args.quickfail, args.list, args.custom, args.positionals,
    )
end

# A worker on CUDA holds 1.6 GB loaded and about 3.9 GB after one file of
# kernels, past the runner's default of 3.8 GB at which a worker is replaced, so
# every file would get a new one; under coverage a new worker compiles much of
# Tempest again. A worker is replaced past 8 GB here, or JULIA_TEST_MAXRSS_MB.
max_worker_rss = parse(Int, get(ENV, "JULIA_TEST_MAXRSS_MB", "8192")) * 2^20

runtests(Tempest, args; testsuite, init_code, max_worker_rss)
