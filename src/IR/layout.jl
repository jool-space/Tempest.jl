# Layouts: a value is stored densely, its first axis fastest, and an index into
# it is a formula over a domain, so its position in storage is one too. A value
# here is anything with `dims` and `scalar`: a number is stored as one element.

public storage_length, address, aligned_axes, group_stride, wide_axes

"The elements value `v` stores."
storage_length(v) = v.scalar || isempty(v.dims) ? 1 : prod(v.dims)

"The axes of `dims` wider than one."
wide_axes(dims) = Tuple(k for k in eachindex(dims) if dims[k] != 1)

function dense_strides(dims)
    s = 1
    map(dims) do n
        stride = s
        s *= n
        stride
    end
end

"""
    address(v, index, domain) -> Affine

The position of `index`, a formula over `domain` per axis of value `v`, in the
dense storage of `v`.
"""
function address(v, index, d::Domain)
    (v.scalar || isempty(v.dims)) && return constant(0)
    total = constant(0)
    for (e, s) in zip(index, dense_strides(v.dims))
        total = add(total, scale(e, s), d)
    end
    total
end

"""
    aligned_axes(dims, index, domain) -> axes or nothing

The axes of `domain` along which an index reads a value of `dims`, one per axis
of the value, when each axis of the value is read at one variable of the domain
of the same extent, in the domain's order: the value is the domain's tile with
the other axes broadcast.
"""
function aligned_axes(dims, index, D)
    axes = Int[]
    for (k, e) in enumerate(index)
        e.constant == 0 && length(e.terms) == 1 && e.terms[1].first isa Var && e.terms[1].second == 1 || return nothing
        n = e.terms[1].first.n
        D[n] == dims[k] && (isempty(axes) || n > last(axes)) || return nothing
        push!(axes, n)
    end
    axes
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
