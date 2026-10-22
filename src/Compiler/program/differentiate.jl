# Reverse-mode differentiation at the program level. Derivative rules are
# ordinary Julia on handles: each records its nodes into the same tape, so the
# backward graph is program code, planned together with the forward.

public gradient, scalar_partials

const Handle = Union{Traced, TracedArray}

h(tape, id) = Traced(tape, id)
lit(x::Traced, v) = convert(node(x).eltype, v)

"""
    scalar_partials(f, g, y, xs...) -> tuple

The contributions of `g`, the cotangent of `y = f(xs...)`, to each argument:
one handle per argument, or `nothing` where the argument has no derivative.
Arguments are elementwise handles; broadcasting is undone by the caller. Define
a method for a function Tempest records whole (one that branches on its
argument, say) to make it differentiable.
"""
function scalar_partials(f, g, y, xs...)
    throw(ArgumentError("no derivative rule for $f; define Tempest.scalar_partials(::typeof($f), g, y, xs...)"))
end

const NONDIFFERENTIABLE = (
    sign, floor, ceil, round, trunc, isnan, isinf, isfinite, signbit, !, <, <=, >, >=, ==, !=,
    isless, &, |, xor, rem, mod, div, fld, cld,
)
for f in NONDIFFERENTIABLE
    @eval scalar_partials(::typeof($f), g, y, xs...) = map(_ -> nothing, xs)
end

scalar_partials(::typeof(identity), g, y, x) = (g,)
scalar_partials(::Cast, g, y, x) = (g,)
scalar_partials(::typeof(+), g, y, a, b) = (g, g)
scalar_partials(::typeof(-), g, y, a, b) = (g, -g)
scalar_partials(::typeof(-), g, y, a) = (-g,)
scalar_partials(::typeof(*), g, y, a, b) = (g * b, g * a)
scalar_partials(::typeof(/), g, y, a, b) = (g / b, -(g * y) / b)
scalar_partials(::typeof(muladd), g, y, a, b, c) = (g * b, g * a, g)
scalar_partials(::typeof(^), g, y, a, b::Traced) =
    (g * b * a^(b - 1), node(b).eltype <: Integer ? nothing : g * y * log(a))
scalar_partials(::typeof(^), g, y, a, b::Real) = (g * b * a^(b - 1), nothing)
scalar_partials(::typeof(exp), g, y, x) = (g * y,)
scalar_partials(::typeof(exp2), g, y, x) = (g * y * lit(y, log(2)),)
scalar_partials(::typeof(exp10), g, y, x) = (g * y * lit(y, log(10)),)
scalar_partials(::typeof(expm1), g, y, x) = (g * (y + 1),)
scalar_partials(::typeof(log), g, y, x) = (g / x,)
scalar_partials(::typeof(log2), g, y, x) = (g / (x * lit(y, log(2))),)
scalar_partials(::typeof(log10), g, y, x) = (g / (x * lit(y, log(10))),)
scalar_partials(::typeof(log1p), g, y, x) = (g / (x + 1),)
scalar_partials(::typeof(sqrt), g, y, x) = (g / (2 * y),)
scalar_partials(::typeof(cbrt), g, y, x) = (g / (3 * y * y),)
scalar_partials(::typeof(abs), g, y, x) = (g * sign(x),)
scalar_partials(::typeof(abs2), g, y, x) = (g * (2 * x),)
scalar_partials(::typeof(inv), g, y, x) = (-(g * y * y),)
scalar_partials(::typeof(sin), g, y, x) = (g * cos(x),)
scalar_partials(::typeof(cos), g, y, x) = (-(g * sin(x)),)
scalar_partials(::typeof(tan), g, y, x) = (g * (1 + y * y),)
scalar_partials(::typeof(sinh), g, y, x) = (g * cosh(x),)
scalar_partials(::typeof(cosh), g, y, x) = (g * sinh(x),)
scalar_partials(::typeof(tanh), g, y, x) = (g * (1 - y * y),)
scalar_partials(::typeof(asin), g, y, x) = (g / sqrt(1 - x * x),)
scalar_partials(::typeof(acos), g, y, x) = (-(g / sqrt(1 - x * x)),)
scalar_partials(::typeof(atan), g, y, x) = (g / (1 + x * x),)
scalar_partials(::typeof(asinh), g, y, x) = (g / sqrt(x * x + 1),)
scalar_partials(::typeof(acosh), g, y, x) = (g / sqrt(x * x - 1),)
scalar_partials(::typeof(atanh), g, y, x) = (g / (1 - x * x),)
scalar_partials(::typeof(atan), g, y, a, b) = (g * b / (a * a + b * b), -(g * a) / (a * a + b * b))
scalar_partials(::typeof(max), g, y, a, b) = (ifelse(a >= b, g, zero(g)), ifelse(a >= b, zero(g), g))
scalar_partials(::typeof(min), g, y, a, b) = (ifelse(a <= b, g, zero(g)), ifelse(a <= b, zero(g), g))
scalar_partials(::typeof(ifelse), g, y, c, a, b) = (nothing, ifelse(c, g, zero(g)), ifelse(c, zero(g), g))
scalar_partials(::typeof(copysign), g, y, a, b) = (ifelse(signbit(a) == signbit(b), g, -g), nothing)
scalar_partials(::typeof(flipsign), g, y, a, b) = (ifelse(signbit(b), -g, g), nothing)

# Constants among a map's inputs are plain numbers to its rule.
function rule_argument(tape, id)
    op = tape.nodes[id].op
    op isa Constant && op.value isa Number ? op.value : h(tape, id)
end

function pullback(op::Map, n, id, g, tape)
    xs = map(i -> rule_argument(tape, i), n.inputs)
    contributions = scalar_partials(op.f, g, h(tape, id), xs...)
    map(c -> c isa Traced ? c : nothing, contributions)
end

reduce_handle(tape, x, axes; scalar) =
    h(tape, record!(tape, Reduce(SumFold(), axes), (x.id,), scalar ? () : ntuple(d -> d in axes ? 1 : node(x).dims[d], length(node(x).dims)), node(x).eltype; scalar))

function expand(tape, g, dims; scalar = false)
    n = node(g)
    n.dims == dims && n.scalar == scalar && return g
    # A constant expands to a constant, which later rules can see through.
    n.op isa Constant && n.op.value isa Number && return h(tape, record!(tape, Constant(n.op.value), (), dims, n.eltype; scalar))
    h(tape, record!(tape, Map(identity), (g.id,), dims, n.eltype; scalar))
end

function pullback(op::Reduce, n, id, g, tape)
    x = h(tape, only(n.inputs))
    xn = node(x)
    contribution = fold_pullback(op.fold, x, h(tape, id), g, op.axes, n.scalar, tape)
    (expand(tape, contribution, xn.dims; scalar = xn.scalar),)
end

fold_pullback(::SumFold, x, y, g, axes, full, tape) = g
function fold_pullback(f::Relative, x, y, g, axes, full, tape)
    f.observe == LOGSUMEXP || throw(ArgumentError("no derivative rule for a relative fold observed by $(f.observe)"))
    g * exp(x - y)
end

# Tied extrema share the cotangent equally.
function fold_pullback(::Union{MaxFold, MinFold}, x, y, g, axes, full, tape)
    ties = cast(node(x).eltype, x == y)
    ties * (g / reduce_handle(tape, ties, axes; scalar = full))
end

# A product's partial by one element is the product of the others, computed
# without dividing by zeros: with no zero in the slice it is the product over
# the element; with one zero it is the product of the rest at the zero.
function fold_pullback(::ProdFold, x, y, g, axes, full, tape)
    T = node(x).eltype
    is_zero = cast(T, x == zero(T))
    count = reduce_handle(tape, is_zero, axes; scalar = full)
    factors = x + is_zero
    rest = h(tape, record!(tape, Reduce(ProdFold(), axes), (factors.id,), node(y).dims, T; scalar = full))
    g * rest * (cast(T, count == 0) / factors + cast(T, count == 1) * is_zero)
end

function fold_pullback(f::GenericFold, x, y, g, axes, full, tape)
    f.pullback === nothing && throw(ArgumentError("fold $f has no derivative rule; pass `pullback` to GenericFold"))
    f.pullback(x, y, g, axes)
end

function pullback(op::Contract, n, id, g, tape)
    a, b = map(i -> TracedArray(tape, i), n.inputs)
    gg = TracedArray(tape, g.id)
    da = contract(gg, b, op.out, op.b, op.a)
    db = contract(a, gg, op.a, op.out, op.b)
    (h(tape, da.id), h(tape, db.id))
end

pullback(::Reshape, n, id, g, tape) =
    (h(tape, record!(tape, Reshape(), (g.id,), tape.nodes[only(n.inputs)].dims, node(g).eltype; scalar = tape.nodes[only(n.inputs)].scalar)),)

pullback(op::Permute, n, id, g, tape) =
    (h(tape, record!(tape, Permute(Tuple(invperm(op.perm))), (g.id,), tape.nodes[only(n.inputs)].dims, node(g).eltype)),)

# A selection and a scatter are each other's adjoints; their index arrays get
# no cotangent.
function pullback(op::Select, n, id, g, tape)
    inverse = inverse_selection(op, n, g, tape)
    inverse === nothing || return (h(tape, inverse), map(_ -> nothing, Base.tail(n.inputs))...)
    adjoint_selection(Scatter(op.indices), n, g, tape)
end
pullback(op::Scatter, n, id, g, tape) = adjoint_selection(Select(op.indices), n, g, tape)

# A selection of whole axes, each by a constant permutation or as it is, has
# the selection by the inverse permutations as its adjoint: a gather, which
# fuses, where a scatter would not. Nothing for any other selection.
function inverse_selection(op, n, g, tape)
    x = tape.nodes[first(n.inputs)]
    length(op.indices) == length(x.dims) && n.dims == x.dims || return nothing
    inputs = [g.id]
    indices = map(enumerate(op.indices)) do (axis, i)
        i isa AbstractRange && i == 1:x.dims[axis] && return i
        i isa IndexInput || return nothing
        c = tape.nodes[n.inputs[i.input]].op
        c isa Constant && c.value isa AbstractVector && length(c.value) == x.dims[axis] && isperm(c.value) || return nothing
        push!(inputs, constant!(tape, invperm(c.value)))
        IndexInput(length(inputs))
    end
    any(isnothing, indices) && return nothing
    record!(tape, Select(Tuple(indices)), inputs, x.dims, node(g).eltype)
end

function adjoint_selection(op, n, g, tape)
    x, indices... = n.inputs
    xn = tape.nodes[x]
    (h(tape, record!(tape, op, (g.id, indices...), xn.dims, node(g).eltype; scalar = xn.scalar)), map(_ -> nothing, indices)...)
end

pullback(::Union{Constant, Argument}, n, id, g, tape) = map(_ -> nothing, n.inputs)

# Undo broadcasting: sum a contribution over the axes its input was broadcast
# along, and give it the input's shape and element type.
function unbroadcast(tape, c::Traced, input)
    target = tape.nodes[input]
    cn = node(c)
    if cn.dims != target.dims || (cn.scalar != target.scalar)
        if target.scalar || isempty(target.dims)
            if !isempty(cn.dims)
                c = reduce_handle(tape, c, Tuple(1:length(cn.dims)); scalar = true)
            end
            if !target.scalar
                c = h(tape, record!(tape, Reshape(), (c.id,), (), node(c).eltype))
            end
        else
            padded = ntuple(k -> get(target.dims, k, 1), length(cn.dims))
            axes = Tuple(k for k in 1:length(cn.dims) if padded[k] == 1 && cn.dims[k] != 1)
            isempty(axes) || (c = reduce_handle(tape, c, axes; scalar = false))
            node(c).dims == target.dims || (c = h(tape, record!(tape, Reshape(), (c.id,), target.dims, node(c).eltype)))
        end
    end
    node(c).eltype === target.eltype ? c : cast(target.eltype, c)
end

## Shifts a loss does not depend on.
#
# The log-sum-exp written out, `m + log(Σ exp(x - m))`, is the same for every
# `m` constant along the sum, the maximum or any other, and so is a softmax
# written with one, `exp(x - m) / Σ exp(x - m)`. So a value read only as such
# a shift has no derivative there, though the rules would compute one, and
# the maximum's own derivative after it: contributions that sum to zero. What
# a loss does as `m` moves by `c` is found from the nodes between them, by the
# exponential law of relative folds (see `Relative`): `exp` turns a shift into
# a factor and `log` a factor into a shift; shifts add, factors multiply; a
# sum along axes `c` is constant along keeps a factor, a maximum or a
# log-sum-exp a shift. Where the loss stays, `m` gets nothing from the nodes
# between, as JAX's `logsumexp` stops the gradient of its maximum, without a
# function written to stop it.
#
# The derivatives of the nodes between are the same at every shift too, so
# their rules read them at the normal shift, `m + log(s)` for `s` the sum of
# `exp(y - m)`, where `s` is one. Then the rules read `y` and the log-sum-exp
# alone, as a named log-sum-exp's rule does, and nothing reads the maximum and
# the sum apart: the rewrites make them one fold (see `rewrite`).

"How a value moves as `m` moves by `c`: by `k c`, or by the factor `exp(k c)`."
struct Shift
    k::Int
end
struct Factor
    k::Int
end
const STILL = Shift(0)

factor(k) = iszero(k) ? STILL : Factor(k)
shift_of(r::Shift) = r.k
shift_of(r) = nothing
factor_of(r::Factor) = r.k
factor_of(r::Shift) = iszero(r.k) ? 0 : nothing
factor_of(r) = nothing

# Under each map, how its value moves as its inputs move, or nothing for a
# move the laws do not say.
function moved(op::Map, rs, mdims)
    f = op.f
    # Rounding to an integer is not a shift: its derivative is zero.
    (f === identity || f isa Cast{<:AbstractFloat}) && return only(rs)
    if f === (+) || f === (-)
        ks = map(shift_of, rs)
        any(isnothing, ks) && return nothing
        return Shift(f(ks...))
    elseif f === (*) || f === (/)
        ks = map(factor_of, rs)
        any(isnothing, ks) && return nothing
        return factor(f === (*) ? +(ks...) : -(ks...))
    elseif f === exp
        k = shift_of(only(rs))
        return k === nothing ? nothing : factor(k)
    elseif f === log
        k = factor_of(only(rs))
        return k === nothing ? nothing : Shift(k)
    end
    nothing
end

# A fold moves as its operand does only along axes `c` is constant along.
function moved(op::Reduce, rs, mdims)
    all(a -> get(mdims, a, 1) == 1, op.axes) || return nothing
    r = only(rs)
    op.fold isa SumFold && return factor_of(r) === nothing ? nothing : r
    op.fold isa Union{MaxFold, MinFold} && return shift_of(r) === nothing ? nothing : r
    op.fold == Relative(Exponential(), LOGSUMEXP) && return shift_of(r) === nothing ? nothing : r
    nothing
end

moved(op, rs, mdims) = nothing

# How the nodes between `m` and the loss move as `m` moves, through the
# readers `readers` lists for each node, when the loss stays; nothing when it
# does not. A move the laws do not say reaches the loss, since every node that
# moves is read on the way to it.
function moves_from(tape, loss, m, readers)
    mdims = tape.nodes[m].dims
    moves = Dict{Int, Union{Shift, Factor}}(m => Shift(1))
    pending = BitSet(readers[m])
    while !isempty(pending)
        id = popfirst!(pending)
        n = tape.nodes[id]
        r = moved(n.op, map(i -> get(moves, i, STILL), n.inputs), mdims)
        r === nothing && return nothing
        moves[id] = r
        r === STILL || union!(pending, readers[id])
    end
    get(moves, loss, STILL) === STILL ? moves : nothing
end

# The sum of `exp(y - m)` among the nodes `moves` lists, shaped as `m` is, for
# a `y` that stays: the sum the normal shift makes one. Nothing without one.
function shifted_sum(tape, m, moves)
    shape = (tape.nodes[m].dims, tape.nodes[m].scalar)
    for id in sort!(collect(keys(moves)))
        n = tape.nodes[id]
        n.op isa Reduce && n.op.fold isa SumFold && (n.dims, n.scalar) == shape || continue
        e = tape.nodes[only(n.inputs)]
        e.op isa Map && e.op.f === exp || continue
        d = tape.nodes[only(e.inputs)]
        d.op isa Map && d.op.f === (-) && length(d.inputs) == 2 && d.inputs[2] == m &&
            get(moves, d.inputs[1], STILL) === STILL && return id
    end
    nothing
end

# The nodes `moves` lists recorded anew at the normal shift: each node's id,
# and the id it is read at. Nothing without a sum to normalize.
function normal_shift!(tape, m, moves)
    s = shifted_sum(tape, m, moves)
    s === nothing && return nothing
    # A number filling the node's shape, as the node's value.
    filling(n, x) = record!(tape, Constant(convert(n.eltype, x)), (), n.dims, n.eltype; scalar = n.scalar)
    at = Dict(m => (h(tape, m) + log(h(tape, s))).id, s => filling(tape.nodes[s], 1))
    for id in sort!(collect(keys(moves)))
        haskey(at, id) && continue
        n = tape.nodes[id]
        inputs = map(i -> get(at, i, i), n.inputs)
        at[id] = if n.op isa Map
            # Constants fold, and identities simplify, as when traced: the
            # sum's log is zero, and a quotient by it the dividend.
            xs = map(i -> rule_argument(tape, i), inputs)
            any(x -> x isa Traced, xs) ? record_map(n.op.f, xs...).id : filling(n, n.op.f(xs...))
        else
            record!(tape, n.op, inputs, n.dims, n.eltype; scalar = n.scalar)
        end
    end
    at
end

# The shifts among `range` the loss does not depend on, and the nodes their
# rules read anew, at the normal shift. A node between two shifts is read as
# it is.
function cancelled_shifts!(tape, loss, range, keep)
    readers = [Int[] for _ in tape.nodes]
    for id in range
        keep[id] || continue
        foreach(i -> push!(readers[i], id), tape.nodes[id].inputs)
    end
    shifts = BitSet()
    at = Dict{Int, Int}()
    for m in range
        keep[m] && m != loss && tape.nodes[m].eltype <: AbstractFloat || continue
        moves = moves_from(tape, loss, m, readers)
        moves === nothing && continue
        push!(shifts, m)
        any(id -> haskey(at, id), keys(moves)) && continue
        normal = normal_shift!(tape, m, moves)
        normal === nothing || merge!(at, normal)
    end
    shifts, at
end

# The reverse pass from the scalar node `loss` over the nodes `range` of the
# tape, recorded into the tape: the cotangent of each node it reaches. Nodes
# before `range` are constants to it.
function reverse_pass!(tape, loss, range)
    first_appended = length(tape.nodes) + 1
    cotangents = reverse_pass_nodes!(tape, loss, range)
    union!(tape.backward, first_appended:length(tape.nodes))
    cotangents
end

function reverse_pass_nodes!(tape, loss, range)
    cotangents = Dict{Int, Traced}()
    cotangents[loss] = h(tape, constant!(tape, one(tape.nodes[loss].eltype)))
    keep = live(tape.nodes, [loss])
    shifts, at = cancelled_shifts!(tape, loss, range, keep)
    for id in reverse(range)
        n = tape.nodes[id]
        (keep[id] && haskey(cotangents, id)) || continue
        # The rule reads the node and its inputs where they are read anew.
        read = Node(n.op, map(i -> get(at, i, i), n.inputs), n.dims, n.eltype, n.scalar)
        contributions = pullback(n.op, read, get(at, id, id), cotangents[id], tape)
        for (input, c) in zip(n.inputs, contributions)
            (c === nothing || input in shifts) && continue
            # A constant operand has no cotangent.
            tape.nodes[input].op isa Constant && continue
            tape.nodes[input].eltype <: AbstractFloat || continue
            c = unbroadcast(tape, c, input)
            cotangents[input] = haskey(cotangents, input) ? cotangents[input] + c : c
        end
    end
    cotangents
end

# The cotangent of node `id`: zero where the pass did not reach it, and
# nothing unless its elements are floating-point numbers.
function cotangent!(tape, cotangents, id)
    a = tape.nodes[id]
    a.eltype <: AbstractFloat || return nothing
    get(cotangents, id) do
        z = h(tape, constant!(tape, zero(a.eltype)))
        a.scalar ? z : h(tape, record!(tape, Map(identity), (z.id,), a.dims, a.eltype))
    end
end

"""
    differentiate(program) -> Program

The program extended by its reverse pass. `program` must have one scalar
result, the loss. The new program returns the loss and, in the structure of the
inputs, the gradient with respect to every floating-point input (`nothing` for
other inputs), with each struct a named tuple of its fields'.

```jldoctest
julia> p = differentiate(trace(x -> sum(abs2, x), [1.0, 2.0, 3.0]));

julia> interpret(p, [1.0, 2.0, 3.0])
(14.0, [2.0, 4.0, 6.0])
```
"""
function differentiate(p::Program)
    length(p.results) == 1 && p.nodes[only(p.results)].scalar ||
        throw(ArgumentError("differentiate needs a program with one scalar result"))
    tape = Tape(copy(p.nodes), copy(p.backward))
    loss = only(p.results)
    cotangents = reverse_pass!(tape, loss, eachindex(p.nodes))
    results = [loss]
    function result(position)
        position === nothing && return nothing
        g = cotangent!(tape, cotangents, p.arguments[position])
        g === nothing && return nothing
        push!(results, g.id)
        length(results)
    end
    gradients = tangent(result, p.input_structure)
    # Rules may build what nothing reads, such as a constant operand's share.
    prune(Program(tape.nodes, p.arguments, results, p.input_structure, (1, gradients...), tape.backward))
end

"""
    Tempest.gradient(f, args...) -> (value, gradients...)

The value of `f(args...)`, a number, and its gradient with respect to each
argument, in that argument's structure: an array or a number for each
floating-point array or number, and `nothing` for anything else. A struct's
gradient is a named tuple of its fields'. On Julia
arrays this is `interpret(differentiate(trace(f, args...)), args...)`.

In traced code it differentiates `f` where it is called, so a training step
is a function like any other, traced and prepared as one program:

    function step(params, state, batch)
        loss, grads = Tempest.gradient(p -> loss_of(p, batch), params)
        new_params, new_state = adam(params, state, grads)
        (loss, new_params, new_state)
    end

What `f` closes over is a constant to the gradient.
"""
function gradient(f, args...)
    k = findfirst(x -> x isa Union{Traced, TracedArray}, leaves(args))
    k === nothing && return interpret(differentiate(trace(f, args...)), args...)
    tape = leaves(args)[k].tape
    # Each floating-point argument enters as a node of its own, whose
    # cotangent the pass collects.
    function stand_in(x)
        x isa Union{Traced, TracedArray} && x.tape !== tape &&
            throw(ArgumentError("the arguments of a gradient must be traced by one program"))
        x isa Union{AbstractFloat, AbstractArray{<:AbstractFloat}} || return x
        id = x isa Union{Traced, TracedArray} ? x.id : constant!(tape, x)
        n = tape.nodes[id]
        s = record!(tape, Map(identity), (id,), n.dims, n.eltype; scalar = n.scalar)
        # The function never writes its arguments, as a program its inputs.
        n.scalar ? Traced(tape, s) : TracedArray(tape, s; input = true)
    end
    stand_ins = rebuild(stand_in, args)
    start = length(tape.nodes) + 1
    value = f(stand_ins...)
    value isa Number && !(value isa Traced) && (value = Traced(tape, constant!(tape, value)))
    value isa Traced && value.tape === tape && node(value).scalar ||
        throw(ArgumentError("a gradient needs a function that returns a number, got $(typeof(value))"))
    cotangents = reverse_pass!(tape, value.id, start:length(tape.nodes))
    gradients = tangent(stand_ins) do x
        x isa Union{Traced, TracedArray} || return nothing
        g = cotangent!(tape, cotangents, x.id)
        g === nothing ? nothing : node(g).scalar ? Traced(tape, g.id) : TracedArray(tape, g.id)
    end
    (value, gradients...)
end
