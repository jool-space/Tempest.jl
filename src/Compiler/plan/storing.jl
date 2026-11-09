# What is stored, decided before kernels form: the results, what a gather or
# a fold by key reads from storage, the states of recurrences, and each map
# and fold that is not recomputed where it is read. The rules are applied in
# order, a later one deciding again what an earlier one did.

# What `plan` decides before it forms kernels: the graph with its results
# stored, what is stored, what runs, and the folds recomputed where read.
# The values in `store` are stored as those a gather reads are.
function storing(graph::IndexGraph; recompute = (), store = (), ride = ())
    recomputed = Set{Int}(recompute)
    graph = with_stored_results(graph)
    values = graph.values
    n = length(values)
    # What the results read, directly or through others: nothing else runs.
    # A state read keeps what it is carried from, after it, to a fixpoint: a
    # state whose next value only another state's reads, as crossing states'
    # are, was left with no kernel.
    live = falses(n)
    for o in graph.results, r in read_values(o)
        live[r] = true
    end
    while true
        for id in n:-1:1
            live[id] || continue
            for o in operands(values[id].definition), r in read_values(o)
                live[r] = true
            end
        end
        grown = false
        for (id, v) in enumerate(values)
            d = v.definition
            live[id] && d isa Carried && d.from isa Read && !live[d.from.value] && (live[d.from.value] = grown = true)
        end
        grown || break
    end
    # Each sum riding a recurrence's steps, by the recurrence's state, and
    # the values between it and the cycle, which its steps compute.
    cycles = recurrences(graph)
    riders = Dict{Int, Vector{Int}}()
    between = Dict{Int, Set{Int}}()
    for r in ride
        k = findfirst(((cs, _, cycle),) -> live[first(cs)] && riding(graph, live, cycle, r) !== nothing, cycles)
        k === nothing && throw(NotAGroup([r]))
        cs, _, cycle = cycles[k]
        c = first(cs)
        push!(get!(() -> Int[], riders, c), r)
        union!(get!(() -> Set{Int}(), between, c), riding(graph, live, cycle, r))
    end
    foreach(b -> union!(recomputed, b), Base.values(between))
    # Reads of a value, one per running reader and formula: a reader that
    # reads it several times at one index computes it once.
    counts = zeros(Int, n)
    for (id, v) in enumerate(values), o in unique(operands(v.definition))
        live[id] && o isa Read && (counts[o.value] += 1)
    end
    results = Set(o.value for o in graph.results if o isa Read)
    for r in results
        counts[r] += 1
    end
    # A gather reads its indices from storage, and a fold by key the contents
    # it starts from.
    materialized = falses(n)
    for id in 1:n
        live[id] || continue
        d = values[id].definition
        for o in operands(d)
            foreach(r -> materialized[r] = true, gathered_values(o))
        end
        d isa Keyed && d.into !== nothing && foreach(r -> materialized[r] = true, read_values(d.into))
    end
    foreach(v -> materialized[v] = true, store)
    # A recurrence a sum rides runs a step at a time (see `build_stepped`),
    # which starts each state from storage: a start recomputed in the
    # recurrence's kernel, as `tanh.(a)` read once, left the steps nothing to
    # start from.
    for (cs, _, _) in cycles, c in cs
        haskey(riders, first(cs)) || continue
        init = values[c].definition.init
        init isa Read && (materialized[init.value] = true)
    end
    # Maps of recomputed folds are recomputed with them: a fold recomputed
    # where it is read has no storage, so a map of it is a map of its term. A
    # recomputed map's readers may still be stored; they recompute it inside.
    after = falses(n)
    for id in 1:n
        d = values[id].definition
        d isa Pointwise || continue
        after[id] = any(o -> o isa Read && (o.value in recomputed && values[o.value].definition isa Reduction || after[o.value]), d.operands)
    end
    # Work nothing reads is not stored.
    stored = map(eachindex(values)) do id
        d = values[id].definition
        id in results || materialized[id] || !(id in recomputed || after[id]) || return false
        d isa Union{Pointwise, Reduction, Scan, Keyed} && !live[id] && return false
        d isa Pointwise ? (counts[id] != 1 || id in results || materialized[id] || fold_slice(values, d)) : true
    end
    # A sum another sum reads, adding a fold, is stored: the sums of a value
    # used many times, as its gradient accumulates over each use. Recomputed,
    # the last one's reader would load every term, a kernel growing with the
    # uses; stored, each is written beside the fold it adds.
    users = readers(graph)
    for id in 1:n
        live[id] && !stored[id] && !(id in recomputed || after[id]) && accumulates(values, id) || continue
        any(r -> live[r] && accumulates(values, r), users[id]) && (stored[id] = true)
    end
    # A recurrence's state is stored, and the values of its cycle something
    # outside it reads, the value the state is carried from among them: its
    # group stores them at each step. The rest it recomputes at each step,
    # the value it is carried from becoming the state one step on (see
    # `recurrence_group`); so does a map of what is stored, which a reader
    # outside recomputes too.
    if !isempty(cycles)
        users = readers(graph)
        for (cs, fs, cycle) in cycles
            foreach(c -> stored[c] = true, cs)
            # A sum riding the steps, and what it computes there, are read
            # within them.
            c = first(cs)
            within = union(Set(cycle), get(riders, c, Int[]), get(between, c, Set{Int}()))
            for w in cycle
                w in cs && continue
                d = values[w].definition
                # A value a state is carried from that another of the step
                # reads, as an LSTM's hidden state reads its cell's, is
                # stored where the step reads it.
                stored[w] = w in results || materialized[w] || any(u -> live[u] && !(u in within), users[w]) ||
                    w in fs && any(u -> live[u] && u in cycle && !(u in cs), users[w])
                stored[w] && !(w in fs) && d isa Pointwise && all(o -> !(o isa Read) || stored[o.value], d.operands) && (stored[w] = false)
            end
        end
    end
    # A pointwise value that reads nothing stored is cheaper to recompute.
    for id in eachindex(values)
        d = values[id].definition
        d isa Pointwise && stored[id] && !(id in results) && !materialized[id] || continue
        all(o -> !(o isa Read) || !stored[o.value] && values[o.value].definition isa Pointwise, d.operands) || continue
        all(o -> o isa Literal || o isa IndexOperand, d.operands) && (stored[id] = false)
    end
    # A stored half-precision value widened to Float32 is widened where it is
    # read, from the half value: a reader loads half the bytes, and a product
    # takes the half value on the half tensor cores (see `widened`), where it
    # would take a stored Float32 copy at Float32's precision. BFloat16
    # attention with its output rounded, as cuDNN computes it, stored its
    # output's gradient so for the product into the values' gradient: the
    # step took 515 μs at TF32 and 900 at full precision, 412 and 430 now.
    for id in eachindex(values)
        d = values[id].definition
        stored[id] && widened(graph, d) && !(id in results) && !materialized[id] || continue
        o = only(d.operands)
        o isa Read && stored[o.value] && (stored[id] = false)
    end
    foreach(b -> foreach(u -> stored[u] = false, b), Base.values(between))
    (; graph, stored, live, results, materialized, riders)
end

# Whether value `v` sums two operands, a fold among them: a step of an accumulation.
function accumulates(values, v)
    d = values[v].definition
    d isa Pointwise && d.f === (+) && length(d.operands) == 2 && any(o -> o isa Read && values[o.value].definition isa Reduction, d.operands)
end

# Whether `d` widens a half-precision value to Float32, as `Float32.(k)` does
# before a product: a product takes the half value as it is, its products
# exact in Float32, and the tensor cores accumulate them in Float32.
widened(graph, d) = d isa Pointwise && d.f isa Cast{Float32} && length(d.operands) == 1 &&
    eltype_of(graph, only(d.operands)) <: HalfFloat

# Whether `d` is a slice of a fold: the fold read at constants along some of
# its axes. It is stored, so that what reads it reads the slice and not the
# whole fold, which the fold's group need not write (see `slice_placement`).
function fold_slice(values, d::Pointwise)
    d.f === identity && length(d.operands) == 1 && only(d.operands) isa Read || return false
    o = only(d.operands)
    values[o.value].definition isa Reduction && any(e -> isempty(e.terms), o.index)
end

# Whether stored pointwise value `id` is read only within one group over its
# domain, at the domain's index: by the group's members, or by values the
# group recomputes where it reads them, which read it so in turn.
function recomputable(p::Plan, users, id, live, results, materialized)
    v = p.graph.values[id]
    v.definition isa Pointwise && p.stored[id] && live[id] && !(id in results) && !materialized[id] || return false
    groups = Set{Int}()
    pending, seen = [id], Set{Int}()
    while !isempty(pending)
        w = pop!(pending)
        dims = p.graph.values[w].dims
        for u in users[w]
            live[u] && !(u in seen) || continue
            push!(seen, u)
            all(operands(p.graph.values[u].definition)) do o
                !(w in read_values(o)) || o isa Read && o.value == w && o.index == identity_index(dims)
            end || return false
            p.owner[u] == 0 && !p.stored[u] ? push!(pending, u) : push!(groups, p.owner[u])
        end
    end
    length(groups) == 1 || return false
    k = only(groups)
    k != 0 && k != p.owner[id] && p.groups[k].kind in (:reduction, :scan, :pointwise) && p.groups[k].domain == v.dims
end

# Every result becomes a value read whole, so results are buffers: a result
# that is a layout of a value, or a constant, gets a value of its own. A fold
# or pointwise value that only one result reads, through a permutation of its
# axes, is stored in the result's order instead of copied: its domain is
# permuted, and its id stays.
function with_stored_results(graph::IndexGraph)
    values = copy(graph.values)
    results = copy(graph.results)
    users = readers(graph)
    for (k, r) in enumerate(results)
        r isa Read || continue
        v = values[r.value]
        p = permutation(r.index)
        (p === nothing || issorted(p)) && continue
        v.definition isa Union{Pointwise, Reduction} && !v.scalar || continue
        isempty(users[r.value]) && count(o -> o isa Read && o.value == r.value, results) == 1 || continue
        values[r.value] = permuted(v, p)
        results[k] = Read(r.value, identity_index(values[r.value].dims))
    end
    for (k, r) in enumerate(results)
        dims = graph.result_dims[k]
        # A value read whole, or a reshape of it, presents its storage as it is.
        r isa Read && values[r.value].scalar == graph.result_scalar[k] && reads_storage(values[r.value], r.index, dims) && continue
        push!(values, Value(narrow(Pointwise(identity, (r,)), dims), map(k -> dims[k], wide_axes(dims)),
            graph.result_eltype[k], graph.result_scalar[k]))
        results[k] = Read(length(values), wide_index(dims))
    end
    IndexGraph(values, graph.arguments, results, graph.result_dims, graph.result_scalar, graph.result_eltype, graph.rewrites,
        vcat(graph.backward, falses(length(values) - length(graph.backward))))
end

# The axes an index reads a value at, when it is a permutation of the reader's
# variables.
function permutation(index)
    p = Int[]
    for e in index
        e.constant == 0 && length(e.terms) == 1 && e.terms[1].first isa Var && e.terms[1].second == 1 || return nothing
        push!(p, e.terms[1].first.n)
    end
    sort(p) == 1:length(p) ? p : nothing
end

# Value `v` stored with its axes permuted: `v′[x] = v[x[p[1]], …, x[p[n]]]`. A
# fold permutes the kept axes of its iteration; its reduced axes stay.
function permuted(v::Value, p)
    dims = Tuple(v.dims[findfirst(==(j), p)] for j in eachindex(p))
    d = v.definition
    if d isa Pointwise
        σ = Tuple(var(p[j]) for j in eachindex(p))
        return Value(Pointwise(d.f, map(o -> through(o, σ, Domain(dims)), d.operands)), dims, v.eltype, v.scalar)
    end
    kept = Tuple(k for k in eachindex(d.iteration) if !(k in d.reduced))
    iteration = collect(d.iteration)
    σ = Vector{Affine}(map(var, 1:length(d.iteration)))
    for (j, k) in enumerate(kept)
        iteration[kept[p[j]]] = d.iteration[k]
        σ[k] = var(kept[p[j]])
    end
    iteration = Tuple(iteration)
    operand = through(d.operand, Tuple(σ), Domain(iteration))
    Value(Reduction(d.fold, operand, iteration, d.reduced), dims, v.eltype, v.scalar)
end
