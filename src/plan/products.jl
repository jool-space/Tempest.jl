# Products at the index level: groups whose folds sum a product of two
# factors. Which of the domain's axes each factor reads splits the kept axes
# into those only the first reads (M), those only the second reads (N) and
# those both read (a batch, B); the summed axes are K. Implementations of
# products read this split, and each factor as a strided matrix over it.

# The domain's axes a term reads, also through the folds it recomputes, or
# nothing when it reads one of the group's folds.
function axes_read(t::Term, n)
    axes = Set{Int}()
    for s in subterms((t,); nested = true)
        s isa Stat && return nothing
        exprs = s isa Load ? s.index : s isa Ix ? (s.expr,) : ()
        for e in exprs, k in 1:n
            mentions(e, k) && push!(axes, k)
        end
    end
    axes
end

"""
    product_axes(graph, g) -> (M, N, K, B) or nothing

The axes of a group whose folds each sum a product of two factors over the
group's primary axes `K`: those only the first factor reads (`M`), those only
the second reads (`N`), and those both read and keep (`B`). Nothing when a fold
is not such a product, or the folds split the axes differently.
"""
function product_axes(graph, g::Group)
    n = length(g.domain)
    K = sort!(collect(primary_axes(graph, g)))
    # Axes of extent one are read by neither factor and need no tile axis.
    kept = [k for k in 1:n if !(k in K) && g.domain[k] > 1]
    found = nothing
    for r in g.reductions
        t = g.terms[r]
        t isa Apply && t.f === (*) && length(t.args) == 2 || return nothing
        ax, ay = axes_read(t.args[1], n), axes_read(t.args[2], n)
        (ax === nothing || ay === nothing) && return nothing
        all(k -> k in ax && k in ay, K) || return nothing
        M = [k for k in kept if k in ax && !(k in ay)]
        N = [k for k in kept if k in ay && !(k in ax)]
        B = [k for k in kept if k in ax && k in ay]
        length(M) + length(N) + length(B) == length(kept) && !isempty(M) && !isempty(N) || return nothing
        groups = (M, N, K, B)
        found === nothing || found == groups || found == (N, M, K, B) || return nothing
        found === nothing && (found = groups)
    end
    found
end

"""
    group_stride(e, axes, domain) -> stride or nothing

The stride of address `e` along `axes` collapsed into one axis (the first
fastest), or nothing when its coefficients do not chain over them.
"""
function group_stride(e::Affine, axes, domain)
    stride = nothing
    p = 1
    for k in axes
        n = domain[k]
        c = coefficient(e, k)
        c === nothing && return nothing
        if n > 1
            c % p == 0 || return nothing
            s = c ÷ p
            stride === nothing ? (stride = s) : (stride == s || return nothing)
        end
        p *= n
    end
    something(stride, 0)
end
