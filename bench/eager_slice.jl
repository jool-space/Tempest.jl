# The eager slice: Julia array code that runs eagerly, traced and checked
# against eager execution. Each case runs on CPU arrays as it is, then is
# traced, interpreted, and prepared and run on the CPU; it counts when every
# result agrees with eager execution. A case that does not is classified by
# the first obstacle it meets: it branches on array values (or takes a shape
# from them), mutates, loops over scalars, draws from a global random number
# generator, misses a method on traced arrays, or something else. NNlib comes
# first, then Base, Statistics and LinearAlgebra. The slice should only grow.
#   julia --project=dev bench/eager_slice.jl
using Tempest, NNlib, Statistics, LinearAlgebra, Random
import Tempest as T

sample(dims...; scale = 1.0f0) = 2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...))

x, y = sample(8, 6; scale = 0.7f0), sample(8, 6; scale = 0.3f0)
u = sample(8; scale = 0.5f0)
a, b, c = sample(4, 5, 3; scale = 0.4f0), sample(5, 6, 3; scale = 0.2f0), sample(4, 6, 3; scale = 0.6f0)
image = sample(8, 8, 3, 2; scale = 0.2f0)              # width, height, channels, batch
weights = sample(3, 3, 3, 4; scale = 0.4f0)
volume = sample(4, 4, 4, 2, 1; scale = 0.3f0)
columns = [3, 1, 6, 1]
scale3, bias3 = sample(3; scale = 0.8f0), sample(3; scale = 0.1f0)
q, k, v = sample(8, 5, 2; scale = 0.3f0), sample(8, 7, 2; scale = 0.2f0), sample(8, 7, 2; scale = 0.1f0)
dense = DenseConvDims(image, weights)
outgrad = sample(6, 6, 4, 2; scale = 0.5f0)

const CASES = Tuple{String, String, Function, Tuple}[]
case!(group, name, f, args...) = push!(CASES, (group, name, f, args))

## NNlib.

for name in NNlib.ACTIVATIONS
    f = getfield(NNlib, name)
    case!("NNlib", "$name.(x)", x -> f.(x), x)
end
case!("NNlib", "softmax(x)", x -> softmax(x), x)
case!("NNlib", "softmax(x; dims = 2)", x -> softmax(x; dims = 2), x)
case!("NNlib", "logsoftmax(x)", x -> logsoftmax(x), x)
case!("NNlib", "logsumexp(x; dims = 1)", x -> NNlib.logsumexp(x; dims = 1), x)
case!("NNlib", "logsumexp(x)", x -> NNlib.logsumexp(x), x)
case!("NNlib", "batched_mul(a, b)", batched_mul, a, b)
case!("NNlib", "a ⊠ b", (a, b) -> a ⊠ b, a, b)
case!("NNlib", "batched_mul(batched_transpose(a), c)", (a, c) -> batched_mul(batched_transpose(a), c), a, c)
case!("NNlib", "batched_mul(a, batched_adjoint(c))", (a, c) -> batched_mul(batched_adjoint(c), a), a, c)
case!("NNlib", "batched_vec(a, x)", (a, x) -> batched_vec(a, x[1:5, 1:3]), a, x)
case!("NNlib", "dropout(x, 0.5)", x -> dropout(x, 0.5f0), x)
case!("NNlib", "dropout(Xoshiro(1), x, 0.5)", x -> dropout(Xoshiro(1), x, 0.5f0), x)
case!("NNlib", "conv(image, weights)", conv, image, weights)
case!("NNlib", "conv with padding and stride", (i, f) -> conv(i, f; pad = 1, stride = 2), image, weights)
case!("NNlib", "depthwiseconv(image, weights)", depthwiseconv, image, sample(3, 3, 2, 3; scale = 0.4f0))
case!("NNlib", "∇conv_data(dy, weights)", (dy, f) -> ∇conv_data(dy, f, dense), outgrad, weights)
case!("NNlib", "∇conv_filter(image, dy)", (i, dy) -> ∇conv_filter(i, dy, dense), image, outgrad)
case!("NNlib", "maxpool(image, (2, 2))", i -> maxpool(i, (2, 2)), image)
case!("NNlib", "meanpool(image, (2, 2))", i -> meanpool(i, (2, 2)), image)
case!("NNlib", "lpnormpool(image, 2, (2, 2))", i -> lpnormpool(i, 2, (2, 2)), image)
case!("NNlib", "upsample_nearest(image, (2, 2))", i -> upsample_nearest(i, (2, 2)), image)
case!("NNlib", "upsample_bilinear(image, (2, 2))", i -> upsample_bilinear(i, (2, 2)), image)
case!("NNlib", "upsample_linear(a, 2)", a -> upsample_linear(a, 2), a)
case!("NNlib", "upsample_trilinear(volume, (2, 2, 2))", v -> upsample_trilinear(v, (2, 2, 2)), volume)
case!("NNlib", "pixel_shuffle(image, 2)", i -> pixel_shuffle(i[:, :, 1:4, :], 2), cat(image, image; dims = 3))
case!("NNlib", "pad_zeros(image, 1)", i -> pad_zeros(i, 1), image)
case!("NNlib", "pad_constant(image, 1, 0.5)", i -> pad_constant(i, 1, 0.5f0), image)
case!("NNlib", "pad_repeat(image, 1)", i -> pad_repeat(i, 1), image)
case!("NNlib", "pad_reflect(image, 1)", i -> pad_reflect(i, 1), image)
case!("NNlib", "pad_symmetric(image, 1)", i -> pad_symmetric(i, 1), image)
case!("NNlib", "pad_circular(image, 1)", i -> pad_circular(i, 1), image)
case!("NNlib", "gather(x, columns)", NNlib.gather, x, columns)
case!("NNlib", "scatter(+, x, columns)", (x, i) -> NNlib.scatter(+, x[:, 1:4], i), x, columns)
case!("NNlib", "normalise(x; dims = 1)", x -> NNlib.normalise(x; dims = 1), x)
case!("NNlib", "layernorm(u, u, x)", (u, x) -> NNlib.layernorm(u, u, x), u, x)
case!("NNlib", "rmsnorm(u, x)", (u, x) -> NNlib.rmsnorm(u, x), u, x)
case!("NNlib", "groupnorm(s, b, image, 3)", (s, b, i) -> NNlib.groupnorm(s, b, i, 3), scale3, bias3, image)
case!("NNlib", "instancenorm(s, b, image)", (s, b, i) -> NNlib.instancenorm(s, b, i), scale3, bias3, image)
case!("NNlib", "batchnorm(s, b, image)", (s, b, i) -> NNlib.batchnorm(s, b, i), scale3, bias3, image)
case!("NNlib", "glu(x, 1)", x -> NNlib.glu(x, 1), x)
case!("NNlib", "make_causal_mask(x)", make_causal_mask, x)
case!("NNlib", "dot_product_attention(q, k, v)", (q, k, v) -> dot_product_attention(q, k, v; nheads = 2), q, k, v)
case!("NNlib", "unfold(image, size(weights))", (i, f) -> NNlib.unfold(i, size(f)), image, weights)
case!("NNlib", "imrotate(image, 0.3)", i -> imrotate(i, 0.3), image)

## Base, Statistics and LinearAlgebra.

case!("Base", "sum(x)", sum, x)
case!("Base", "sum(x; dims = 1)", x -> sum(x; dims = 1), x)
case!("Base", "sum(abs2, x; dims = 2)", x -> sum(abs2, x; dims = 2), x)
case!("Base", "prod(x; dims = 1)", x -> prod(x; dims = 1), x)
case!("Base", "maximum(x)", maximum, x)
case!("Base", "maximum(x; dims = 2)", x -> maximum(x; dims = 2), x)
case!("Base", "maximum(abs, x)", x -> maximum(abs, x), x)
case!("Base", "minimum(x; dims = 1)", x -> minimum(x; dims = 1), x)
case!("Base", "extrema(x)", extrema, x)
case!("Base", "extrema(x; dims = 1)", x -> extrema(x; dims = 1), x)
case!("Base", "argmax(x)", argmax, x)
case!("Base", "argmax(x; dims = 1)", x -> argmax(x; dims = 1), x)
case!("Base", "findmax(x)", findmax, x)
case!("Base", "findmax(x; dims = 2)", x -> findmax(x; dims = 2), x)
case!("Base", "mean(x)", mean, x)
case!("Base", "mean(x; dims = 1)", x -> mean(x; dims = 1), x)
case!("Base", "var(x)", var, x)
case!("Base", "var(x; dims = 1)", x -> var(x; dims = 1), x)
case!("Base", "std(x; dims = 2)", x -> std(x; dims = 2), x)
case!("Base", "median(u)", median, u)
case!("Base", "cumsum(x; dims = 1)", x -> cumsum(x; dims = 1), x)
case!("Base", "cumprod(u)", cumprod, u)
case!("Base", "clamp.(x, -0.5, 0.5)", x -> clamp.(x, -0.5f0, 0.5f0), x)
case!("Base", "abs2.(x)", x -> abs2.(x), x)
case!("Base", "round.(x)", x -> round.(x), x)
case!("Base", "isfinite.(x)", x -> isfinite.(x), x)
case!("Base", "ifelse.(x .> 0, x, 0)", x -> ifelse.(x .> 0, x, 0.0f0), x)
case!("Base", "norm(x)", norm, x)
case!("Base", "norm(x, 1)", x -> norm(x, 1), x)
case!("Base", "norm(x, Inf)", x -> norm(x, Inf), x)
case!("Base", "dot(x, y)", dot, x, y)
case!("Base", "normalize(u)", normalize, u)
case!("Base", "x * y'", (x, y) -> x * y', x, y)
case!("Base", "x' * u", (x, u) -> x' * u, x, u)
case!("Base", "tr(x' * y)", (x, y) -> tr(x' * y), x, y)
case!("Base", "diag(x)", diag, x)
case!("Base", "triu(x)", triu, x)
case!("Base", "transpose(x)", transpose, x)
case!("Base", "permutedims(x, (2, 1))", x -> permutedims(x, (2, 1)), x)
case!("Base", "permutedims(image, (3, 1, 2, 4))", i -> permutedims(i, (3, 1, 2, 4)), image)
case!("Base", "reshape(x, 6, 8)", x -> reshape(x, 6, 8), x)
case!("Base", "vec(x)", vec, x)
case!("Base", "dropdims(sum(x; dims = 1); dims = 1)", x -> dropdims(sum(x; dims = 1); dims = 1), x)
case!("Base", "view(x, 2:4, :)", x -> view(x, 2:4, :), x)
case!("Base", "x[:, [1, 3]]", x -> x[:, [1, 3]], x)
case!("Base", "x[x .> 0]", x -> x[x .> 0], x)
case!("Base", "selectdim(x, 2, 3)", x -> selectdim(x, 2, 3), x)
case!("Base", "eachcol(x) summed", x -> sum(eachcol(x)), x)
case!("Base", "cat(x, y; dims = 3)", (x, y) -> cat(x, y; dims = 3), x, y)
case!("Base", "vcat(x, y)", vcat, x, y)
case!("Base", "hcat(x, y)", hcat, x, y)
case!("Base", "repeat(x, 2, 1)", x -> repeat(x, 2, 1), x)
case!("Base", "repeat(x; inner = (1, 2))", x -> repeat(x; inner = (1, 2)), x)
case!("Base", "reverse(x; dims = 1)", x -> reverse(x; dims = 1), x)
case!("Base", "circshift(x, (1, 0))", x -> circshift(x, (1, 0)), x)
case!("Base", "sort(u)", sort, u)
case!("Base", "sort(x; dims = 1)", x -> sort(x; dims = 1), x)
case!("Base", "map(abs, x)", x -> map(abs, x), x)
case!("Base", "map(+, x, y)", (x, y) -> map(+, x, y), x, y)
case!("Base", "mapreduce(abs2, +, x)", x -> mapreduce(abs2, +, x), x)
case!("Base", "mapreduce(abs2, +, x; dims = 1)", x -> mapreduce(abs2, +, x; dims = 1), x)
case!("Base", "reduce(+, x)", x -> reduce(+, x), x)
case!("Base", "reduce(max, x; dims = 1)", x -> reduce(max, x; dims = 1), x)
case!("Base", "accumulate(+, x; dims = 1)", x -> accumulate(+, x; dims = 1), x)
case!("Base", "count(>(0), x)", x -> count(>(0), x), x)
case!("Base", "any(>(1), x)", x -> any(>(1), x), x)
case!("Base", "zero(x) .+ x", x -> zero(x) .+ x, x)

## Running and comparing.

scalar(r) = r isa AbstractArray && ndims(r) == 0 ? r[] : r

function agree(eager, traced)
    es, ts = T.leaves(eager), map(scalar, T.leaves(traced))
    length(es) == length(ts) || return false
    all(zip(es, ts)) do (e, t)
        e isa Number && t isa Number && return isapprox(e, t; rtol = 1.0e-4, atol = 1.0e-5)
        e isa AbstractArray && t isa AbstractArray && size(e) == size(t) || return false
        all(isapprox.(e, t; rtol = 1.0e-4, atol = 1.0e-5))
    end
end

traced_type(@nospecialize(S)) = S isa DataType && (S <: Union{T.TracedArray, T.Traced} || any(traced_type, S.parameters))
traced_value(@nospecialize(z)) = traced_type(typeof(z)) || (z isa Type && traced_type(z))

const RNG = (:rand, :randn, :rand!, :randn!, :randexp, :default_rng, :shuffle, :randperm)
const MUTATING = (:setindex!, :copyto!, :fill!, :materialize!, :map!, :broadcast!, :mul!, :accumulate!, :sort!)

"""
    classify(err, frames) -> (why, detail)

Why a case does not trace, from its error and where it was thrown: the first
obstacle met, which may hide others.
"""
function classify(err, frames)
    # An error in a task, as NNlib's threaded loops throw them.
    while err isa Union{CompositeException, TaskFailedException}
        err isa CompositeException && (err = first(err.exceptions); continue)
        frames = stacktrace(err.task.backtrace)
        err = err.task.exception
    end
    # The frames inside the traced code, not those of the harness around it.
    outer = findfirst(f -> f.func === :trace && parentmodule(f) === Tempest || f.func === :check && parentmodule(f) === Main, frames)
    names = Set(f.func for f in frames[1:something(outer, length(frames) + 1) - 1])
    message = sprint(showerror, err)
    detail = first(replace(first(split(message, '\n')), r"\s+" => " "), 110)
    why = if err isa TypeError && err.got isa T.Traced && err.expected === Bool ||
             err isa MethodError && err.f === Bool && any(traced_value, err.args)
        "branches on array values"
    elseif :LogicalIndex in names || err isa MethodError && any(a -> a isa Tuple && any(traced_value, a), err.args)
        # A shape that depends on array values.
        "branches on array values"
    elseif any(in(names), RNG)
        "a global RNG"
    elseif :iterate in names || err isa BoundsError && occursin("CartesianIndex", message) ||
           occursin("invalid index: Traced", message) || occursin(r"results are arrays and numbers, got \w+\{Any\}", message)
        "scalar loops"
    elseif err isa Base.CanonicalIndexError || err isa TypeError && err.got isa T.Traced ||
           err isa MethodError && nameof(err.f) in MUTATING
        "mutation"
    elseif err isa MethodError && any(traced_value, err.args) || occursin("unsupported index", message) ||
           occursin("needs `init`", message)
        "a missing method on traced arrays"
    else
        "other"
    end
    why, detail
end

"""
    check(f, args) -> (agrees, why, detail)

Run `f` eagerly, then traced, interpreted, and prepared and run on the CPU.
`agrees` is nothing when `f` does not run eagerly.
"""
function check(f, args)
    eager = try
        f(args...)
    catch err
        return nothing, "", sprint(showerror, err)
    end
    # Code that draws from a global generator differs between runs.
    isequal(eager, f(args...)) || return false, "a global RNG", "eager results differ between runs"
    stage = "trace"
    try
        p = trace(f, args...)
        stage = "interpret"
        agree(eager, interpret(p, args...)) || return false, "other", "interpreted results differ from eager ones"
        stage = "prepare"
        agree(eager, execute!(prepare(p, args...))) || return false, "other", "prepared results differ from eager ones"
        true, "", ""
    catch err
        why, detail = classify(err, stacktrace(catch_backtrace()))
        false, why, stage == "trace" ? detail : string(stage, ": ", detail)
    end
end

results = map(CASES) do (group, name, f, args)
    agrees, why, detail = check(f, args)
    (; group, name, agrees, why, detail)
end
# The slice is code that runs eagerly.
for r in results
    r.agrees === nothing && println("not in the slice, since it does not run eagerly: ", r.name, ": ", first(split(r.detail, '\n')))
end
results = filter(r -> r.agrees !== nothing, results)

n, m = count(r -> r.agrees, results), length(results)
println("eager slice: ", n, " of ", m, " trace and agree")
for group in unique(r.group for r in results)
    rs = filter(r -> r.group == group, results)
    println("  ", group, ": ", count(r -> r.agrees, rs), " of ", length(rs))
end
println("failures, by the first obstacle met:")
whys = [r.why for r in results if !r.agrees]
for why in sort!(unique(whys); by = w -> -count(==(w), whys))
    println("  ", rpad(why, 36), count(==(why), whys))
end
println()
println("| group | case | traces and agrees | why not | error |")
println("| --- | --- | --- | --- | --- |")
for r in results
    println("| ", r.group, " | `", r.name, "` | ", r.agrees ? "yes" : "no", " | ", r.why, " | ", replace(r.detail, "|" => "\\|"), " |")
end
