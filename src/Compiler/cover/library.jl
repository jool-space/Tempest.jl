# Libraries (see `Implement.Library`): a template is traced and planned as a
# program of its own, and a group of a plan computes it when its members are
# the template's group's, term for term, with each of the template's axes one
# or more of the group's and each of its arrays a strided view of a value the
# group reads or stores.
#
# A match is found in two passes. The first traces the template with axes of
# small distinct extents and finds which of the group's values and axes each
# of the template's arrays and axes is, literals aside, since a literal may
# depend on an extent (a mean's divisor, a scale of one over a root). The
# second traces it at the extents found and must give the same binding with
# every literal equal: what is claimed is what was compared, at its shape.

# Odd, so that no axis's reciprocal is exact: one over a power of two is a
# product, one over any other extent a quotient (see `map_definition`).
const PROBE_EXTENTS = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)

# A template's argument as tracing sees it: a size and an element type.
struct Shape{T, N} <: AbstractArray{T, N}
    dims::NTuple{N, Int}
end

Base.size(s::Shape) = s.dims
Base.getindex(::Shape, i...) = throw(ArgumentError("a template's argument has no elements"))

# The axis names of a template, each once, in the order its inputs name them.
axis_names(t::Template) = Tuple(unique(n for axes in values(t.inputs) for n in axes))

# A template traced with inputs of element types `types` and axes of
# `extents`, a named tuple, rewritten where the target permits (`rewrites`),
# and planned with every fold recomputed where it can be: its one group's
# view, and the values that are its inputs and its outputs, in order; nothing
# when it is not one group that stores its outputs.
function template_group(t::Template, rewrites::Bool, types, extents)
    args = map((axes, T) -> Shape{T, length(axes)}(map(a -> extents[a], axes)), Tuple(values(t.inputs)), types)
    graph = lower(trace(t.f, args...))
    rewrites && (graph = rewrite(graph))
    p = plan(graph; recompute = recomputable_folds(graph))
    length(p.groups) == 1 || return nothing
    g = group_view(p, only(p.groups))
    inputs = zeros(Int, length(t.inputs))
    for (v, val) in enumerate(p.graph.values)
        val.definition isa Input && (inputs[val.definition.argument] = v)
    end
    # Each result reads a member over the result's shape, the axes of extent
    # one left out and the others in order.
    outputs = Int[]
    for o in p.graph.results
        o isa Read && o.value in members(g) || return nothing
        vars = [e.terms[1].first.n for e in o.index if !isconstant(e) && length(e.terms) == 1 && e.terms[1].second == 1 && e.terms[1].first isa Var]
        all(e -> isconstant(e) ? e.constant == 0 : length(e.terms) == 1, o.index) && issorted(vars; lt = <=) &&
            length(vars) == count(>(1), p.graph.values[o.value].dims) || return nothing
        push!(outputs, o.value)
    end
    length(outputs) == length(t.outputs) && allunique(outputs) && all(>(0), inputs) || return nothing
    (; group = g, inputs, outputs)
end

const TEMPLATE_GROUPS = Dict{Any, Any}()
const PROBE_GROUPS = Dict{Tuple{UInt, Bool}, Any}()
const MATCHES = Dict{Any, Union{Nothing, Binding}}()
const LIBRARY_LOCK = ReentrantLock()

# Matches are keyed by a group's terms, which one preparation shares: equal
# terms of another are other objects, compared whole.
forget_matches!() = @lock LIBRARY_LOCK empty!(MATCHES)

# The template traced in Float32 at the probe's extents: what tells groups
# apart before any is matched.
function probe_group(t::Template, rewrites)
    key = (objectid(t), rewrites)
    found = @lock LIBRARY_LOCK get(PROBE_GROUPS, key, missing)
    found === missing || return found
    probe = NamedTuple{axis_names(t)}(PROBE_EXTENTS[1:length(axis_names(t))])
    tg = cached_template_group(t, rewrites, ntuple(_ -> Float32, length(t.inputs)), probe)
    @lock LIBRARY_LOCK PROBE_GROUPS[key] = tg
end

function cached_template_group(t::Template, rewrites, types, extents)
    key = (objectid(t), rewrites, types, extents)
    found = @lock LIBRARY_LOCK get(TEMPLATE_GROUPS, key, missing)
    found === missing || return found
    tg = template_group(t, rewrites, types, extents)
    @lock LIBRARY_LOCK TEMPLATE_GROUPS[key] = tg
end

# What the terms of a template's group (`a`) and a group (`b`) unify to: the
# value of `b` each value of `a` is, and each load's and stored member's index
# on both sides, for the axes to be found from.
mutable struct Unifier
    a::GroupView
    b::GroupView
    exact::Bool
    values::Dict{Int, Int}
    reads::Vector{Tuple{Int, Tuple{Vararg{Affine}}, Int, Tuple{Vararg{Affine}}}}
end

Unifier(a, b, exact) = Unifier(a, b, exact, Dict{Int, Int}(), [])

snapshot(u::Unifier) = (copy(u.values), length(u.reads))
restore!(u::Unifier, (values, n)) = (u.values = values; resize!(u.reads, n); u)

function same_value!(u::Unifier, x, y)
    haskey(u.values, x) && return u.values[x] == y
    any(==(y), values(u.values)) && return false
    u.values[x] = y
    true
end

const COMMUTATIVE = (+, *, max, min)

# Unification is a search: a commutative function's operands unify in either
# order, and members with any members of their kind, and a choice that unifies
# may still fail later, when the axes are found. Each step takes what to do
# once it holds, `then`, and is true as soon as some choice makes `then` true,
# restoring what it bound otherwise. Each step wraps the next in a closure,
# nested as deep as the terms: specialized on it, each depth would compile
# anew.

function unify!(@nospecialize(then), u::Unifier, a::Term, b::Term)
    s = snapshot(u)
    found = if !u.exact && (x = scaled(a)) !== nothing && (y = scaled(b)) !== nothing
        # Literals aside, a quotient by one is a product by another.
        a.eltype === b.eltype && unify!(then, u, x, y)
    elseif a isa Apply
        b isa Apply && isequal(a.f, b.f) && a.eltype === b.eltype && length(a.args) == length(b.args) &&
            (unify_all!(then, u, a.args, b.args) ||
             length(a.args) == 2 && any(f -> f === a.f, COMMUTATIVE) && (restore!(u, s); unify_all!(then, u, a.args, reverse(b.args))))
    elseif a isa Load
        b isa Load && same_value!(u, a.value, b.value) && (push!(u.reads, (a.value, a.index, b.value, b.index)); then())
    elseif a isa Stat
        b isa Stat && same_value!(u, a.value, b.value) && then()
    elseif a isa Lit
        b isa Lit && typeof(a.value) === typeof(b.value) && (!u.exact || isequal(a.value, b.value)) && then()
    else
        # Index formulas and recomputed folds are not matched yet.
        false
    end
    found || restore!(u, s)
    found
end

# Folds alike, the numbers their observes compute with aside unless `exact`,
# as a literal is: a mean's divisor moves into its fold's observe.
same_fold(a, b, exact) = exact ? isequal(a, b) : isequal(numbers_aside(a), numbers_aside(b))
numbers_aside(f::Relative) = Relative(f.law, numbers_aside(f.observe))
numbers_aside(f::GenericFold) = GenericFold(f.init, f.lift, f.combine, numbers_aside(f.observe), f.pullback)
numbers_aside(o::Composed) = Composed(o.f === (/) && any(a -> a isa Number, o.args) ? (*) : o.f,
    map(a -> a isa Number ? typeof(a) : numbers_aside(a), o.args))
numbers_aside(x) = x

# The term a term scales by a literal, as a product or a quotient; or nothing.
function scaled(t::Term)
    t isa Apply && length(t.args) == 2 || return nothing
    x, y = t.args
    t.f === (/) && y isa Lit && return x
    t.f === (*) && y isa Lit && return x
    t.f === (*) && x isa Lit && return y
    nothing
end

function unify_all!(@nospecialize(then), u::Unifier, as, bs, i = 1)
    i > length(as) && return then()
    unify!(() -> unify_all!(then, u, as, bs, i + 1), u, as[i], bs[i])
end

# The members of the template's group, from the `i`th on, each unified with
# a member of the group not `used`.
function unify_members!(@nospecialize(then), u::Unifier, ms, ns, used, i = 1)
    i > length(ms) && return then()
    for (j, n) in enumerate(ns)
        used[j] && continue
        used[j] = true
        unify_member!(() -> unify_members!(then, u, ms, ns, used, i + 1), u, ms[i], n) && return true
        used[j] = false
    end
    false
end

function unify_member!(@nospecialize(then), u::Unifier, m, n)
    x, y = u.a.values[m], u.b.values[n]
    (m in u.a.reductions) == (n in u.b.reductions) && x.eltype === y.eltype && x.stored == y.stored && x.scalar == y.scalar &&
        same_fold(x.fold, y.fold, u.exact) && x.keyed == y.keyed && x.into === nothing && y.into === nothing && x.scan == y.scan &&
        haskey(u.a.terms, m) == haskey(u.b.terms, n) || return false
    s = snapshot(u)
    same_value!(u, m, n) || return false
    x.stored && push!(u.reads, (m, something(u.a.placement[m], ()), n, something(u.b.placement[n], ())))
    found = haskey(u.a.terms, m) ? unify!(then, u, u.a.terms[m], u.b.terms[n]) : then()
    found || restore!(u, s)
    found
end

# For each wide axis of a group's domain, what reads or reduces it: the arrays
# (as `name`s them) whose addresses move along it, and the members (as `member`
# names them) whose folds reduce it. Nothing when an address is not affine.
function axis_uses(g::GroupView, reads, name, member)
    d = Domain(g.domain)
    uses = [(Set{Symbol}(), Set{Int}()) for _ in g.domain]
    for (v, index) in reads
        e = address(g.values[v], index, d)
        is_affine(e) || return nothing
        for k in eachindex(g.domain)
            g.domain[k] > 1 && coefficient(e, k) != 0 && push!(uses[k][1], name(v))
        end
    end
    for m in members(g), k in g.values[m].reduced
        push!(uses[k][2], member(m))
    end
    uses
end

# A template's array's axes as the template's axes it reads along them, at
# `index`: each a variable, or a constant along an axis of extent one (zero);
# nothing for any other index.
function template_axes(index, dims)
    axes = map(index, dims) do e, n
        n == 1 && return 0
        e.constant == 0 && length(e.terms) == 1 && e.terms[1].second == 1 && e.terms[1].first isa Var || return nothing
        e.terms[1].first.n
    end
    any(isnothing, axes) || !allunique(filter(!iszero, collect(axes))) ? nothing : axes
end

# The binding of template `t`, the `k`th of its library's, whose group `tg`
# was unified with group `g` in `u`. Each axis of the template's stands for
# the group's axes that the same arrays read and the same folds reduce, in
# the order the first array to read them strides along them, which every
# array must chain over. Nothing when an axis of the group is none of the
# template's, an array is not one strided view, or two of the template's
# axes are used alike.
function binding(t::Template, k, tg, g::GroupView, u::Unifier)
    a = tg.group
    names = Dict{Int, Symbol}()
    foreach((v, n) -> names[v] = n, tg.inputs, keys(t.inputs))
    foreach((v, n) -> names[v] = n, tg.outputs, t.outputs)
    image = Dict(y => x for (x, y) in u.values)
    treads = unique((x, i) for (x, i, _, _) in u.reads)
    greads = unique((y, j) for (_, _, y, j) in u.reads)
    allunique(first.(treads)) && allunique(first.(greads)) || return nothing
    tuses = axis_uses(a, treads, v -> names[v], identity)
    guses = axis_uses(g, greads, v -> names[image[v]], m -> image[m])
    (tuses === nothing || guses === nothing) && return nothing
    wide(dom) = [j for j in eachindex(dom) if dom[j] > 1]
    tw = wide(a.domain)
    allunique(tuses[tw]) || return nothing
    stands = Dict(j => Int[] for j in tw)
    for kg in wide(g.domain)
        j = findfirst(j -> tuses[j] == guses[kg], tw)
        j === nothing && return nothing
        push!(stands[tw[j]], kg)
    end
    d = Domain(g.domain)
    index = Dict(greads)
    views = [(names[x], template_axes(i, a.values[x].dims), u.values[x]) for (x, i) in treads]
    any(v -> v[2] === nothing, views) && return nothing
    addresses = Dict(y => address(g.values[y], index[y], d) for (_, _, y) in views)
    all(is_affine, values(addresses)) || return nothing
    ordered = Set{Int}()
    for (_, axes, y) in views, j in axes
        j == 0 || j in ordered || (push!(ordered, j); sort!(stands[j]; by = kg -> abs(coefficient(addresses[y], kg))))
    end
    extent(j) = j == 0 ? 1 : prod(kg -> g.domain[kg], stands[j]; init = 1)
    by_name = Dict{Symbol, Int}()
    found = Dict{Symbol, Strided}()
    for (n, axes, y) in views
        # Traced at the extents found, an input's axes of extent one are gone,
        # and with them which name each axis has.
        haskey(t.inputs, n) && length(axes) == length(t.inputs[n]) && for (name, j) in zip(t.inputs[n], axes)
            j == 0 || get!(by_name, name, j) == j || return nothing
        end
        e = addresses[y]
        strides = map(j -> j == 0 || isempty(stands[j]) ? 0 : group_stride(e, stands[j], g.domain), axes)
        any(isnothing, strides) && return nothing
        found[n] = Strided(g.values[y].eltype, e.constant, map(extent, axes), strides, nothing, y)
    end
    order = (keys(t.inputs)..., t.outputs...)
    all(n -> haskey(found, n), order) || return nothing
    names_t = axis_names(t)
    extents = map(name -> haskey(by_name, name) ? extent(by_name[name]) : 1, names_t)
    Binding(k, NamedTuple{names_t}(extents), NamedTuple{order}(Tuple(found[n] for n in order)))
end

# Whether two groups may unify: of one kind, with as many folds and other
# members, which apply the same functions to as many loads.
function alike(a::GroupView, b::GroupView)
    shapes(x) = sort!([term_shape(x.terms[m]) for m in members(x) if haskey(x.terms, m)])
    a.kind === b.kind && length(a.reductions) == length(b.reductions) && length(a.outputs) == length(b.outputs) && shapes(a) == shapes(b)
end

# The binding of a template's group, `tg` as `template_group` gives it, to a
# group of the plan, unified with literals compared (`exact`) or not; nothing
# when they differ.
function unify_group(t::Template, k, tg, g::GroupView, exact)
    tg !== nothing && alike(tg.group, g) || return nothing
    a = tg.group
    u = Unifier(a, g, exact)
    found = Ref{Union{Nothing, Binding}}(nothing)
    unify_members!(u, members(a), members(g), falses(length(members(g)))) do
        all(v -> haskey(u.values, v), tg.inputs) || return false
        found[] = binding(t, k, tg, g, u)
        found[] !== nothing
    end
    found[]
end

"""
    match_template(template, k, target, group) -> Binding or nothing

The binding of `group`, a `GroupView`, to `template`, the `k`th of its
library's: the template, traced at the extents and element types the binding
gives its axes and arrays, computes what the group does, term for term. Each
array has every axis its template names, of extent one where no axis of the
group stands for it.
"""
function match_template(t::Template, k, target, g::GroupView)
    rewrites = permits_rewrites(target)
    probe = NamedTuple{axis_names(t)}(PROBE_EXTENTS[1:length(axis_names(t))])
    # Told apart from the template traced in Float32 before it is traced at
    # the group's types, which are numbers.
    shape = probe_group(t, rewrites)
    shape === nothing || alike(shape.group, g) || return nothing
    ms = members(g)
    reads = unique(l.value for m in ms if haskey(g.terms, m) for l in loads(g.terms[m]) if !(l.value in ms))
    types = unique(g.values[v].eltype for v in reads)
    !isempty(types) && all(T -> T <: Number, types) || return nothing
    for ts in Iterators.product(ntuple(_ -> types, length(t.inputs))...)
        b = unify_group(t, k, cached_template_group(t, rewrites, ts, probe), g, false)
        b === nothing && continue
        Tuple(b.arrays[n].eltype for n in keys(t.inputs)) == ts || continue
        exact = unify_group(t, k, cached_template_group(t, rewrites, ts, b.extents), g, true)
        exact !== nothing && same_binding(exact, b) && return b
    end
    nothing
end

# Bindings alike over their axes wider than one: the template traced at the
# extents found has none of the others.
view_key(s::Strided) = (s.eltype, s.offset, [(n, d) for (n, d) in zip(s.size, s.strides) if n != 1], s.value)
same_binding(a::Binding, b::Binding) =
    a.template == b.template && keys(a.arrays) == keys(b.arrays) && map(view_key, a.arrays) == map(view_key, b.arrays)

## The implementation a library is.

# A group's binding to each of the library's templates it computes, those
# the library supports, the first template's first; or, for a group of
# several folds each of which computes a template alone, alike but for where
# their arrays start, their bindings, when the library supports them as one.
function Implement.claims(l::Library, target, g)
    found = Any[b for b in bindings(l, target, g) if supports(l, target, b)]
    isempty(found) || return found
    bs = instances(l, target, g)
    bs === nothing ? found : Any[bs]
end

# What makes instances alike: all but where their arrays start.
layout_key(b::Binding) = (b.template, b.extents, map(s -> (s.eltype, s.size, s.strides), b.arrays))

function instances(l::Library, target, g)
    length(members(g)) > 1 && isempty(g.outputs) || return nothing
    bs = Binding[]
    for m in members(g)
        b = bindings(l, target, member_view(g, m))
        isempty(b) && return nothing
        push!(bs, first(b))
    end
    allequal(map(layout_key, bs)) && supports(l, target, bs) ? bs : nothing
end

# Fold `m` of a group as a group of its own over the same domain.
member_view(g, m) = GroupView(g.kind, g.domain, [m], Int[], Dict(m => g.terms[m]), Dict(m => get(g.phase, m, 1)),
    g.primary_axes, g.values, Dict(m => g.placement[m]), g.bytes, g.reference)

# A group's binding to each of the library's templates it computes. Matched
# once per group's terms and arrays: alike layers are one match. A group of
# another kind, or with other numbers of folds and members than the template
# traced in Float32, is none of its, which the cover asks of every group it
# prices.
function bindings(l::Library, target, g)
    found = Binding[]
    for (k, t) in enumerate(templates(l, target))
        shape = probe_group(t, permits_rewrites(target))
        shape === nothing || shape.group.kind === g.kind && length(shape.group.reductions) == length(g.reductions) &&
            length(shape.group.outputs) == length(g.outputs) || continue
        key = (objectid(t), k, permits_rewrites(target), g.kind, g.domain, Tuple(members(g)),
            Tuple(get(g.terms, m, nothing) for m in members(g)), Tuple(g.placement[m] for m in members(g)),
            Tuple(sort!([(v, x.eltype, x.dims, x.stored) for (v, x) in g.values])))
        b = @lock LIBRARY_LOCK get(MATCHES, key, missing)
        if b === missing
            b = match_template(t, k, target, g)
            @lock LIBRARY_LOCK MATCHES[key] = b
        end
        b === nothing || push!(found, b)
    end
    found
end

# A library's launch moves the group's bytes and does the products its folds
# sum, in the type of their factors.
function Implement.estimate(::Library, b, target, g)
    flops, factor = 0.0, Float32
    for r in g.reductions
        fs = product_factors(g.values[r].fold, g.terms[r])
        fs === nothing && continue
        flops += 2.0 * prod(g.domain; init = 1)
        x = first(fs)
        factor = x isa Apply && x.f isa Cast ? term_eltype(only(x.args), g) : term_eltype(x, g)
    end
    [Work(g.bytes; flops, factor)]
end

# A claim is a binding, or instances' bindings; a library that defines its
# own of these methods need not say which.
Implement.record_name(l::Library, claim) = binding_name(l, claim)
binding_name(l, b::Binding) = string(nameof(typeof(l)), " ", b.template, " ",
    join((string(n, "=", e) for (n, e) in pairs(b.extents)), " "), " ",
    join((string(n, " ", s.eltype, " +", s.offset, " ", s.strides) for (n, s) in pairs(b.arrays)), ", "))
binding_name(l, bs::Vector{Binding}) = string(binding_name(l, first(bs)), " ×", length(bs))

Implement.build_claim(l::Library, claim, target, g, buffers) = build(l, target, with_buffers(claim, buffers))
with_buffers(b::Binding, buffers) =
    Binding(b.template, b.extents, map(s -> Strided(s.eltype, s.offset, s.size, s.strides, buffers[s.value], s.value), b.arrays))
with_buffers(bs::Vector{Binding}, buffers) = [with_buffers(b, buffers) for b in bs]

# A library proposes, for each template that is one fold, each fold that
# shares a group with other values, or that a group recomputes, and whose
# term applies what the template's does to as many loads: the cover plans
# it as a group of its own, which the library then claims or not. Of a group
# it does not claim whole, it proposes the most folds that are instances of
# one template alike, when it supports them as one.
function Implement.candidates(l::Library, target, p::Plan)
    shapes = Any[]
    for t in templates(l, target)
        tg = probe_group(t, permits_rewrites(target))
        (tg === nothing || members(tg.group) != tg.group.reductions || length(tg.group.reductions) != 1) && continue
        r = only(tg.group.reductions)
        push!(shapes, (tg.group.values[r].fold, term_shape(tg.group.terms[r])))
    end
    isempty(shapes) && return Vector{Int}[]
    fits(fold, t) = (fold, term_shape(t)) in shapes
    found = Vector{Int}[]
    for g in group_views(p)
        g.kind === :reduction || continue
        for r in g.reductions
            length(members(g)) > 1 && fits(g.values[r].fold, g.terms[r]) && push!(found, [r])
        end
        roots = [g.terms[m] for m in members(g) if haskey(g.terms, m)]
        for t in subterms(roots; nested = true)
            t isa Recomputed && fits(g.values[t.value].fold, t.body) && !([t.value] in found) && push!(found, [t.value])
        end
        length(g.reductions) > 1 && isempty(Implement.claims(l, target, g)) || continue
        classes = Dict{Any, Vector{Pair{Int, Binding}}}()
        for r in g.reductions
            b = bindings(l, target, member_view(g, r))
            isempty(b) || push!(get!(() -> Pair{Int, Binding}[], classes, layout_key(first(b))), r => first(b))
        end
        isempty(classes) && continue
        class = argmax(length, values(classes))
        length(class) > 1 && supports(l, target, last.(class)) && push!(found, sort(first.(class)))
    end
    found
end

# What a fold's term applies, conversions aside and a quotient a product, and
# how many loads it reads.
function term_shape(t::Term)
    fs = sort!([objectid(s.f === (/) ? (*) : s.f) for s in subterms((t,)) if s isa Apply && !(s.f isa Cast)])
    (fs, count(s -> s isa Load, subterms((t,))))
end
