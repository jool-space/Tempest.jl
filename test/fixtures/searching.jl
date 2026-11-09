# A compiled program's search as the tests drive it: a small LSTM's training
# step, the other plan a search gives it on the CPU, and a target whose
# timings take a millisecond. runtime/compile.jl and fixtures/at_once.jl
# include it.

# A small LSTM whose training step writes its parameters and Adam's moments
# in place, for a search behind the first result to give another plan.
σ(x) = 1 / (1 + exp(-x))
function lstm_cell(p, (h, c), x)
    i, f = σ.(p.Wi * x .+ p.Ui * h .+ p.bi), σ.(p.Wf * x .+ p.Uf * h .+ p.bf)
    g, o = tanh.(p.Wg * x .+ p.Ug * h .+ p.bg), σ.(p.Wo * x .+ p.Uo * h .+ p.bo)
    c = f .* c .+ i .* g
    (o .* tanh.(c), c)
end
lstm(p, xs, h0) = Tuple(map(first, accumulate((s, x) -> lstm_cell(p, s, x), eachslice(xs; dims = 2); init = (h0, h0))))
lstm_weights() = (; (Symbol(:W, k) => data(3, 2; scale = 0.3) for k in "ifgo")..., (Symbol(:U, k) => data(3, 3; scale = 0.3) for k in "ifgo")...,
    (Symbol(:b, k) => data(3; scale = 0.1) for k in "ifgo")...)
function adam!(p, m, v, xs, h0)
    l, (g,) = Tempest.withgradient(q -> sum(h -> sum(abs2, h), lstm(q, xs, h0)), p)
    for (w, mw, vw, d) in zip(values(p), values(m), values(v), values(g))
        mw .= 0.9 .* mw .+ 0.1 .* d
        vw .= 0.999 .* vw .+ 0.001 .* d .^ 2
        w .-= 0.01 .* mw ./ (sqrt.(vw) .+ 1.0e-8)
    end
    l
end

# The plan the search behind a first result gives on the CPU, which times
# nothing: the program prepared again with every fold that can be recomputed
# where it is read recomputed, which is another plan of it, its arguments
# bound as `compile` binds them.
function other_plan(c, args...)
    program = c.executable.program
    folds = C.recomputable_folds(C.rewrite(C.lower(program); rounded = C.rounded_backward(T.CPUTarget())))
    timer -> prepare(program, c.f, args...; recompute = folds, point = true)
end

# `c` searching behind its calls as `work(timer)` does.
searching(c, work) = C.Searching(c, C.Search(work))

# A target whose timings take a millisecond, noting how many ran at once.
struct Clock <: C.Target end
const TIMING_NOW, TIMING_MOST = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
function C.time_launch(::Clock, launch)
    Threads.atomic_max!(TIMING_MOST, Threads.atomic_add!(TIMING_NOW, 1) + 1)
    sleep(0.005)
    launch()
    Threads.atomic_sub!(TIMING_NOW, 1)
    1.0e-3
end
