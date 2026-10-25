# Masks: a value masked by formulas over its positions, `ifelse(cond, x,
# literal)` with `cond` a conjunction of comparisons between index formulas.
# A position may be compared with an integer the program reads, a length on
# the device say, which the formula holds as a gather. The planner finds
# masks on the index graph and implementations on a kernel's terms, so the
# algebra both walk with is here.

public comparison_mask, held_formula, masked_result

"""
    comparison_mask(f, a, b, domain) -> Vector{Affine}

The mask that the comparison `f(a, b)` of index formulas states: formulas,
each at most zero where it holds. `a <= b` is `a - b <= 0`, `a < b` is
`a - b + 1 <= 0`.
"""
function comparison_mask(f, a::Affine, b::Affine, d::Domain)
    f in (<=, <) || ((a, b) = (b, a))
    e = add(a, scale(b, -1), d)
    [f in (<, >) ? add(e, 1, d) : e]
end

"""
    held_formula(value, index, T, domain) -> Affine or nothing

The integer of type `T` that `value` holds at `index`, as a formula: a gather
known to lie in the type's range, so that a position compared with it is a
mask. Nothing for a type wider than 32 bits, whose range the arithmetic of
formulas, in `Int`, could overflow.
"""
function held_formula(value, index, ::Type{T}, d::Domain) where {T}
    T <: Integer && T !== Bool && sizeof(T) <= 4 || return nothing
    normalize(0, (Indirect(value, Tuple(index), Int(typemin(T)), Int(typemax(T))) => 1,), d)
end

"""
    masked_result(f, T, args)

What `f` gives, as a `T`, where a mask fails, from its arguments there, each
a number, `:finite` or nothing (unknown): a number when the numbers decide
it, as a zero factor or an infinite summand does, in exact arithmetic;
`:finite` for sums, products and the functions here of finite arguments,
which an overflow could make infinite, so that a program whose unmasked
positions overflow computes a different nothing here.
"""
function masked_result(f, T, args)
    any(isnothing, args) && return nothing
    all(a -> a isa Number, args) && return f isa Cast ? convert(T, only(args)) : f === ifelse ? nothing : f(args...)
    known = [a for a in args if a isa Number]
    if f === (*)
        any(iszero, known) && return zero(T)
        return all(isfinite, known) ? :finite : nothing
    elseif f === (+)
        infinite = filter(isinf, known)
        return isempty(infinite) ? :finite : length(infinite) == 1 ? convert(T, only(infinite)) : nothing
    elseif f === (-) && length(args) == 2
        a, b = args
        a isa Number && isinf(a) && return b isa Number && isinf(b) ? nothing : convert(T, a)
        b isa Number && isinf(b) && return convert(T, -b)
        return :finite
    end
    f in (exp, tanh, abs, abs2, max, min) || f isa Cast ? (all(a -> a === :finite || isfinite(a), args) ? :finite : nothing) : nothing
end
