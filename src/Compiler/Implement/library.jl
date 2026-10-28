# Libraries: implementations that realize a group by calling something that
# computes a function Tempest can already compute. A library names that
# function, its template, and Tempest finds the groups that compute it: it
# traces the template at the group's extents and element types and compares
# the two at the index level, so that a match is never a guess. The library
# sees none of the group's terms, only where each of the template's arrays
# lives: a strided view of a buffer.

public Library, Template, Strided, Binding, templates, supports, build

"""
A library realizes the groups that compute one of its `templates` on a
target: Tempest matches them and hands `supports` and `build` a `Binding` of
the template's arrays. `claims`, `estimate`, `build_claim`, `record_name`
and `candidates` are Tempest's for a library.
"""
abstract type Library <: Implementation end

"""
    Template(f; inputs, outputs)

A function a library computes, `f`, of arrays named and with named axes in
`inputs`, as `(A = (:m, :k), B = (:k, :n))`, returning the arrays named in
`outputs`, one name for each, in order. Tempest traces `f` with arrays of the
axes' extents; an axis no axis of a group matches has extent one. An axis
stands for the group's axes that the same arrays read, the same folds sum
and the same index formulas (a causal mask's) move with. Axes alike in all
three are interchangeable, as attention's query positions and the query
heads that share a key head are: the library is offered a binding for each
way, those whose arrays are laid out as `inputs` names their axes first.
"""
struct Template
    f::Any
    inputs::NamedTuple
    outputs::Tuple{Vararg{Symbol}}
    function Template(f; inputs::NamedTuple, outputs)
        all(a -> a isa Tuple{Vararg{Symbol}} && allunique(a), values(inputs)) ||
            throw(ArgumentError("each input names its axes, each once"))
        isempty(intersect(keys(inputs), outputs)) || throw(ArgumentError("an input and an output share a name"))
        new(f, inputs, Tuple(outputs))
    end
end

Base.show(io::IO, t::Template) = print(io, "Template(", t.f, ", ", join(keys(t.inputs), ", "), " → ", join(t.outputs, ", "), ")")

"""
An array of a binding: elements of `eltype` at `offset` and the given
`strides`, in elements, along axes of extents `size`. `buffer` is the storage
it is a view of when the library builds, and nothing when it is asked whether
it supports the binding.
"""
struct Strided
    eltype::DataType
    offset::Int
    size::Tuple{Vararg{Int}}
    strides::Tuple{Vararg{Int}}
    buffer::Any
    value::Int
end

Base.pointer(s::Strided) = pointer(s.buffer) + s.offset * sizeof(s.eltype)

"""
A group matched to the `template`th of a library's `templates`: the
extent of each of the template's axes in `extents`, and each of its inputs
and outputs as a `Strided` view in `arrays`, by name.
"""
struct Binding
    template::Int
    extents::NamedTuple
    arrays::NamedTuple
end

"""
    templates(library, target) -> templates

The functions `library` computes on `target`, a tuple of `Template`s; none by
default, so that a library computes nothing on a target it does not name.
"""
templates(::Library, target) = ()

"""
    supports(library, target, binding) -> Bool
    supports(library, target, bindings::Vector{Binding}) -> Bool

Whether `library` computes `binding` on `target`: the element types, extents
and strides it takes; true by default. Given `bindings`, instances of one
template that a group computes side by side, alike but for where their arrays
start, as products of chunks of one value are: whether it computes them all
in one launch; false by default.
"""
supports(::Library, target, ::Binding) = true
supports(::Library, target, ::Vector{Binding}) = false

"""
    build(library, target, binding) -> (launch, description)
    build(library, target, bindings::Vector{Binding}) -> (launch, description)

What `build_claim` is for a library: `launch()` computes the template into
the binding's outputs from its inputs, each `Strided` with its `buffer`, on
the task's stream, or every instance of `bindings` it supports, and
`description` says what it runs.
"""
function build end
