# The session's unseen programs for a commit: twelve cases of the fuzzer's
# generator, each from a seed drawn from the commit's hash, so that no commit
# knows its own. Three of each kind: a loop as it is, a loop's gradient, a
# case of another family as it is, and its gradient. From each seed drawn, the
# first seed after it whose case is of the kind and runs untraced in Float64
# at both shapes is taken (unseen.jl). Each is written to OUT as a program
# file of its own, `unseen-K-SEED.jl`, and their paths to OUT/programs.txt.
#
#     julia --project=test/corpus/session select.jl COMMIT OUT
using SHA, Random
const COMMIT, OUT = ARGS[1], ARGS[2]
const SESSION = @__DIR__

const KINDS = [(loop = true, gradient = false), (loop = true, gradient = true), (loop = false, gradient = false),
    (loop = false, gradient = true)]

drawn(k) = parse(Int, bytes2hex(sha256("$COMMIT $k"))[1:12]; base = 16) % 999_000_000 + 1

file_text(seed, gradient) = """
    # An unseen program of the session (select.jl).
    const SEED = $seed
    const GRADIENT = $gradient
    include($(repr(joinpath(SESSION, "unseen.jl"))))
    """

# Whether the case of `seed` is of the kind and runs untraced at both shapes.
function fits(path, kind)
    m = Module(:Fits)
    Core.eval(m, :(include(path) = Base.include($m, path)))
    Base.include(m, path)
    c = Base.invokelatest(m.Gen.generate, m.SEED; big = true)
    (c.family === :loop) == kind.loop || return false
    try
        for shape in (:small, :doubled)
            r = Base.invokelatest(m.untraced, shape)
            all(x -> all(isfinite, x), Base.invokelatest(Tempest_leaves, r)) || return false
        end
        true
    catch e
        e isa InterruptException && rethrow()
        false
    end
end

import Tempest
Tempest_leaves(r) = Tempest.Compiler.leaves(r)

mkpath(OUT)
paths = String[]
for k in 1:12
    kind = KINDS[(k - 1) ÷ 3 + 1]
    seed = drawn(k)
    while true
        path = joinpath(OUT, "unseen-$k-$seed.jl")
        write(path, file_text(seed, kind.gradient))
        fits(path, kind) && break
        rm(path)
        seed += 1
    end
    push!(paths, joinpath(OUT, "unseen-$k-$seed.jl"))
    println("unseen $k: seed $seed, ", kind.loop ? "a loop" : "another family", kind.gradient ? ", its gradient" : "")
end
write(joinpath(OUT, "programs.txt"), join(paths, '\n') * '\n')
