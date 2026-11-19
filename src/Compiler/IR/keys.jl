# Keys of the tables the compiler fills with a program's parts. A tuple of
# them has a type that follows the program: the arity of a node, the rank of
# a domain, the function a map applies, the number of a group's members. A
# table keyed by such tuples compiles its lookups, and the hashing and
# comparing of its keys, again for each shape the program has: the library's
# matches alone compiled 24 lookups in a warm session of the LSTM, 2.2 s.
# A `Key` hashes its parts once and compares them element by element in
# methods compiled once, whatever their shape, so a table keyed by it
# compiles its lookups once.

public map_elements, all_elements, any_element

"""
    Key(parts)

`parts`, a value or a tuple of them nested to any depth, as a key that hashes
and compares as the parts do, element by element, with code compiled once for
every shape of parts.
"""
struct Key
    parts::Any
    hash::UInt
    Base.@nospecializeinfer Key(@nospecialize(parts)) = new(parts, parts_hash(parts, 0x6b43a9b5f2e1d087))
end

Base.hash(k::Key, h::UInt) = hash(k.hash, h)
Base.isequal(a::Key, b::Key) = a.hash == b.hash && parts_isequal(a.parts, b.parts)
Base.:(==)(a::Key, b::Key) = isequal(a, b)

# A tuple element by element; anything else as Base hashes and compares it.
Base.@nospecializeinfer function parts_hash(@nospecialize(x), h::UInt)
    x isa Tuple || return hash(x, h)::UInt
    h = hash(nfields(x), h)
    for i in 1:nfields(x)
        h = parts_hash(getfield(x, i), h)
    end
    h
end

Base.@nospecializeinfer function parts_isequal(@nospecialize(a), @nospecialize(b))
    a isa Tuple || return isequal(a, b)::Bool
    b isa Tuple && nfields(a) == nfields(b) || return false
    for i in 1:nfields(a)
        parts_isequal(getfield(a, i), getfield(b, i)) || return false
    end
    true
end

"""
    parts_cmp(a, b) -> Int

`cmp` of two values or tuples of them nested to any depth, as Base orders
tuples, element by element and the shorter first where one begins the
other, by code compiled once for every shape: the keys a program's parts
are ordered by, indices and operands, are tuples whose shape follows the
program, and Base's `cmp` and `isless` on them compile for each.
"""
Base.@nospecializeinfer function parts_cmp(@nospecialize(a), @nospecialize(b))
    a isa Tuple && b isa Tuple || return cmp(a, b)::Int
    n, m = nfields(a), nfields(b)
    for i in 1:min(n, m)
        c = parts_cmp(getfield(a, i), getfield(b, i))
        c == 0 || return c
    end
    cmp(n, m)
end

"Whether `a` comes before `b` in the order `parts_cmp` gives."
parts_isless(@nospecialize(a), @nospecialize(b)) = parts_cmp(a, b) < 0

"""
    map_elements(f, t::Tuple) -> Tuple
    map_elements(f, a::Tuple, b::Tuple) -> Tuple

`map(f, t)`, by code compiled once for every type of `t`: the walkers of a
program's nested arguments meet tuples as long as the user makes them, and
the planner indices, shapes and operands of a program's every rank and mix
of kinds. Of two tuples, element by element, as many as the shorter has.
"""
Base.@nospecializeinfer function map_elements(f, @nospecialize(t::Tuple))
    out = Vector{Any}(undef, nfields(t))
    for i in eachindex(out)
        out[i] = f(getfield(t, i))
    end
    Tuple(out)
end

Base.@nospecializeinfer function map_elements(f, @nospecialize(a::Tuple), @nospecialize(b::Tuple))
    out = Vector{Any}(undef, min(nfields(a), nfields(b)))
    for i in eachindex(out)
        out[i] = f(getfield(a, i), getfield(b, i))
    end
    Tuple(out)
end

"""
    all_elements(p, t::Tuple) -> Bool

`all(p, t)`, by code compiled once for every type of `t`, as `map_elements`
maps.
"""
Base.@nospecializeinfer function all_elements(p, @nospecialize(t::Tuple))
    for i in 1:nfields(t)
        p(getfield(t, i))::Bool || return false
    end
    true
end

"""
    any_element(p, t::Tuple) -> Bool

`any(p, t)`, by code compiled once for every type of `t`, as `map_elements`
maps.
"""
Base.@nospecializeinfer function any_element(p, @nospecialize(t::Tuple))
    for i in 1:nfields(t)
        p(getfield(t, i))::Bool && return true
    end
    false
end
