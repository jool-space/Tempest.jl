# Kernels stored by one process and looked up by another whose code differs
# from it where neither a key's source nor a package's identity names it.
# Outside this pair, since each needs another precompile of a package: an
# edit to Tempest itself, whose kernels must still load, and a developed
# package's file edited while loaded (reproducers in the challengers'
# night-2026-11-19/review3: `round2/run3.sh` and `devpkg.sh`).
# cuda/cache.jl runs this twice on one store, `VARIANT` 1 then 2, each in a
# process of its own, as a script edited between two runs is. A kernel of
# code changed must not be loaded, and a kernel of code unchanged must be.
using Tempest, Test, CUDACore
import cuBLAS, cuTile
include(joinpath(@__DIR__, "..", "setup.jl"))
const VARIANT = parse(Int, ENV["VARIANT"])
factor = Float32(VARIANT + 1)

# A script's method of a package's function, called with a closure.
struct Gain end
@eval Base.map(f, ::Gain) = f($factor)
through_closure(x::Float32) = map(g -> g * x, Gain())

# A constant of an installed package's module, defined after its image was
# loaded, read by a method of that module no image holds.
Core.eval(LinearAlgebra, :(const CHANGED_SLOPE = $factor))
Core.eval(LinearAlgebra, :(changed_slope(x::Float32) = CHANGED_SLOPE * x))

# A constant of a package loaded from its image, read by a method the image
# holds, redefined.
push!(LOAD_PATH, joinpath(@__DIR__, "packages"))
const ImageConstant = Base.require(@__MODULE__, :ImageConstant)
VARIANT == 2 && Core.eval(ImageConstant, :(const SLOPE = 3.0f0))

# Mutable state of that package a kernel's code reads, set once at startup,
# as a script configuring a package does, or the package's own `__init__`.
VARIANT == 2 && (ImageConstant.CONFIG.slope = 3.0f0)

# A constant of an enumeration, which selects the arithmetic.
@enum Mode fast slow
@eval const MODE = $(VARIANT == 1 ? fast : slow)
picked(x::Float32) = MODE == fast ? 2.0f0 * x : 3.0f0 * x

# A generated function, its generator's code changed.
@eval @generated generated_scale(x::Float32) = :($$factor * x)

# A script's scalar function that makes a closure, the same in both: its
# kernels load.
twice_by_closure(x::Float32) = sum(ntuple(i -> x * i, 2))

# A fold of closures, the same in both: its kernels load.
sums = T.GenericFold(init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],))

x = single(data(256, 64)) ./ 4
changed = [
    ("a script's method called with a closure", x -> sum(through_closure.(x) .* x; dims = 1), x),
    ("a package's constant defined since its image", x -> sum(LinearAlgebra.changed_slope.(x) .* x; dims = 1), x),
    ("an enumeration's constant", x -> sum(picked.(x) .* x; dims = 1), x),
    ("a constant an image held, redefined", x -> sum(ImageConstant.act.(x) .* x; dims = 1), x),
    ("a generated function", x -> sum(generated_scale.(x) .* x; dims = 1), x),
]
unchanged = [("a fold of closures", x -> reduce(sums, x; dims = 1), data(80, 112)),
    ("a script's function that makes a closure", x -> sum(twice_by_closure.(x) .* x; dims = 1), x)]
@testset "$name, variant $VARIANT" for (name, f, a) in [changed; unchanged]
    p = trace(f, a)
    before = C.cache_counts()
    ex = prepare(p, CuArray(a); target = T.CUDATarget())
    got = only(map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))))
    after = C.cache_counts()
    @test isapprox(got, only(filter(!isnothing, C.leaves(interpret(p, a)))); rtol = 1.0e-4)
    if VARIANT == 2 && C.EDGES
        if (name, f, a) in unchanged
            @test after.loaded > before.loaded && after.compiled == before.compiled
        else
            @test after.compiled > before.compiled
        end
    end
end

# The stored kernel read the field as the first process had it, and the image
# vouches for the binding, which setting a field leaves as it was: the second
# process is handed it. Closed by one of two rules, the user's to choose: a
# kernel's code may not read mutable state (refused, as a constant array and a
# variable already are), or the state is known by its content.
@testset "waiting on a rule for mutable state a kernel's code reads, set at startup, variant $VARIANT" begin
    p = trace(x -> sum(ImageConstant.configured.(x) .* x; dims = 1), x)
    ex = prepare(p, CuArray(x); target = T.CUDATarget())
    got = only(map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))))
    right = isapprox(got, only(filter(!isnothing, C.leaves(interpret(p, x)))); rtol = 1.0e-4)
    if VARIANT == 2 && C.EDGES
        @test_broken right
    else
        @test right
    end
end
