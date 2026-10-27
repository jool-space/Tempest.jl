# The kernel cache: compiled binaries kept across sessions, keyed by the content
# of what was compiled, so that preparing a program in a new process loads
# binaries instead of inferring and compiling kernels again.
#
# A key is a hash of the kernel's code and everything that decides its binary:
# the generated source, written out with every function and type it names; the
# argument types; the compiler options; the toolchain; and the identity of each
# package those names come from (its installed version, or a hash of its sources
# when it is developed). Code outside any package is identified by its lowered
# code. The binary's parameters follow from the argument types, so a cached
# kernel is its entry name and its binary.
#
# Tempest's own code may be left out of a key (`content_key`'s `checked`), so
# that an edit to Tempest does not make every kernel compile again: the kernel
# is stored with what its compilation used of Tempest's code instead, found
# from the edges inference keeps (`dependencies`), and loads only while that
# is unchanged (`current`). A kernel uses a handful of Tempest's methods, the
# folds' functions and the generator's helpers, of the hundreds it has. A
# kernel that failed to compile has no inference to read that from: its failure
# is kept under the whole key (`cached_failure`).
#
# The binaries are kept in Julia's object cache, through CompilerCaching, with
# cuTile's: the runtime's store from Julia 1.14, a store of CompilerCaching's
# before. The store is shared and bounded (`JULIA_OBJCACHE_CAPACITY`, by default
# 512 MiB), removes the entries used least recently, and may lose a write, so a
# kernel can compile again; `JULIA_OBJCACHE=0` turns it off.

using SHA
using Serialization
using CompilerCaching: ObjCache

public content_key, cached_kernel, store_kernel, store_failure, cached_failure, uncached_kernel, generated, dependencies

"Kernels loaded from the cache and kernels compiled (or failed to), in this session."
const CACHE_COUNTS = (loaded = Threads.Atomic{Int}(0), compiled = Threads.Atomic{Int}(0))

cache_counts() = (loaded = CACHE_COUNTS.loaded[], compiled = CACHE_COUNTS.compiled[])

# Tempest's kernels in the object cache, and the version of an entry's layout:
# the length of the entry name, the name, the length of what the kernel depends
# on of Tempest's code, that, and the binary; or a length of zero and the
# compiler's message for a kernel that failed to compile.
const KERNELS = "Tempest/kernel"
const KERNEL_SCHEMA = 2

store_key(key) = ObjCache.keyhash(KERNEL_SCHEMA, key)

"""
    load_kernel(key) -> (entry, binary) or nothing

The cached kernel under `key`, if there is one and what it depends on of
Tempest's code is unchanged (`current`). A kernel that failed to compile fails
again, with the compiler's message.
"""
function load_kernel(key)
    bytes = ObjCache.get(KERNELS, store_key(key))
    bytes === nothing && return nothing
    fail_again(bytes)
    n = Int(only(reinterpret(UInt32, bytes[1:4])))
    d = Int(only(reinterpret(UInt32, bytes[(5 + n):(8 + n)])))
    current(bytes[(9 + n):(8 + n + d)]) || return nothing
    Threads.atomic_add!(CACHE_COUNTS.loaded, 1)
    (String(bytes[5:(4 + n)]), bytes[(9 + n + d):end])
end

"""
    store_kernel(key, entry, binary; dependencies) -> (entry, binary)

Count a compiled kernel and keep it under `key`, with what it depends on of
Tempest's code (see `dependencies`), which a key that leaves that code out
needs. A failure to write only loses the entry.
"""
function store_kernel(key, entry::String, binary::Vector{UInt8}; dependencies = UInt8[])
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    put_entry(key, [reinterpret(UInt8, [UInt32(ncodeunits(entry))]); codeunits(entry);
        reinterpret(UInt8, [UInt32(length(dependencies))]); dependencies; binary])
    (entry, binary)
end

"Count a kernel that failed to compile and keep the compiler's message under `key`."
function store_failure(key, message::String)
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    put_entry(key, [reinterpret(UInt8, [UInt32(0)]); codeunits(message)])
end

"""
    cached_failure(key)

Fail again, with the compiler's message, if the kernel under `key` failed to
compile before (`store_failure`); otherwise do nothing. A kernel keyed without
Tempest's code checks its failures under the whole key.
"""
function cached_failure(key)
    bytes = ObjCache.get(KERNELS, store_key(key))
    bytes === nothing || fail_again(bytes)
    nothing
end

function fail_again(bytes)
    if only(reinterpret(UInt32, bytes[1:4])) == 0
        Threads.atomic_add!(CACHE_COUNTS.loaded, 1)
        error("this kernel failed to compile before:\n", String(bytes[5:end]))
    end
end

"""
    uncached_kernel(entry, binary) -> (entry, binary)

Count a kernel compiled in this session that the cache cannot keep, one that
refers to values of the session, and return it as `cached_kernel` expects.
"""
uncached_kernel(entry, binary) = (Threads.atomic_add!(CACHE_COUNTS.compiled, 1); (entry, binary))

function put_entry(key, bytes)
    ObjCache.put!(KERNELS, store_key(key), bytes)
    nothing
end

# Within a session a key is resolved once. Programs stack identical layers, whose
# groups build their kernels at the same time and so have the same key in flight
# together: the first caller loads or compiles it, and the others wait for that
# task, or get its result after, instead of loading or compiling again.
const RESOLVED_KERNELS = Dict{String, Task}()
const RESOLVED_LOCK = ReentrantLock()

"""
    cached_kernel(compile, key) -> (entry, binary)

The kernel under `key`: from the object cache, or compiled by `compile()`, which
returns `(entry, binary)` and stores it, when the cache has none. This happens
once per key in a session, so that callers in flight together wait for the
first. The compilation runs `elsewhere`, and a failure fails every caller.
"""
function cached_kernel(@nospecialize(compile), key)
    task = @lock RESOLVED_LOCK get!(RESOLVED_KERNELS, key) do
        @async begin
            found = load_kernel(key)
            found === nothing ? elsewhere(compile) : found
        end
    end
    fetch_result(task)
end

# Generated kernels live here.
module Generated end

# Generated functions, by type, with the source that defined them: in a key a
# generated function is its source, since its name differs between sessions.
const GENERATED_SOURCES = IdDict{Any, Expr}()
const GENERATED_LOCK = ReentrantLock()

"""
    generated(source::Expr) -> function

Define the function `source`, an anonymous function expression, in the
module `Generated`, keeping its source: in a `content_key` a generated
function is its source, since its name differs between sessions.
"""
function generated(source::Expr)
    f = Core.eval(Generated, source)
    @lock GENERATED_LOCK GENERATED_SOURCES[typeof(f)] = source
    f
end

generated_source(T) = @lock GENERATED_LOCK get(GENERATED_SOURCES, T, nothing)

"""
    content_key(parts...; checked = false) -> String

A hash of `parts` that is the same in every session for the same content: of
their `canonical_text`. With `checked`, Tempest's own code is named without its
identity, for a kernel stored with what it depends on of that code
(`dependencies`), from Julia 1.12, whose inference keeps the edges to find it.
"""
content_key(@nospecialize(parts...); checked = false) = bytes2hex(sha256(canonical_text(parts...; checked = checked && EDGES)))

"The content of `parts` written out, with the identities of the packages they name."
function canonical_text(@nospecialize(parts...); checked = false)
    c = Canonical(IOBuffer(), Set{Module}(), Base.IdSet{Any}(), Dict{Symbol, Int}())
    print(c.io, "julia ", VERSION, " ", Base.GIT_VERSION_INFO.commit, " options ", compiler_options(), "\n")
    for p in parts
        canonical(c, p)
        print(c.io, "\n")
    end
    for m in sort!(collect(c.roots); by = string)
        print(c.io, m, " = ", checked && own_code(m) ? "checked by dependencies" : package_identity(m), "\n")
    end
    String(take!(c.io))
end

# The command-line options that change generated code.
function compiler_options()
    o = Base.JLOptions()
    (o.opt_level, o.check_bounds, o.fast_math, o.debug_level, o.can_inline)
end

struct Canonical
    io::IOBuffer
    roots::Set{Module}
    seen::Base.IdSet{Any}
    gensyms::Dict{Symbol, Int}
end

function canonical(c::Canonical, e::Expr)
    print(c.io, "(", e.head)
    for a in e.args
        a isa LineNumberNode && continue
        print(c.io, " ")
        canonical(c, a)
    end
    print(c.io, ")")
end

# Generated names count up through a session, so they are numbered in order of
# appearance instead.
function canonical(c::Canonical, s::Symbol)
    name = string(s)
    startswith(name, "##") || return print(c.io, ":", name)
    print(c.io, ":#", get!(() -> length(c.gensyms) + 1, c.gensyms, s))
end

canonical(c::Canonical, ::LineNumberNode) = nothing
canonical(c::Canonical, q::QuoteNode) = (print(c.io, "(quote "); canonical(c, q.value); print(c.io, ")"))
canonical(c::Canonical, @nospecialize(x::Union{Number, Char, AbstractString, Nothing, Missing})) = print(c.io, typeof(x), "(", repr(x), ")")
canonical(c::Canonical, m::Module) = (push!(c.roots, Base.moduleroot(m)); print(c.io, "module ", join(fullname(m), ".")))

function canonical(c::Canonical, g::GlobalRef)
    canonical(c, g.mod)
    print(c.io, ".", g.name)
    # Code outside any package is identified by what its names refer to.
    if !identified(g.mod) && isdefined(g.mod, g.name) && !(g in c.seen)
        push!(c.seen, g)
        print(c.io, " = ")
        canonical(c, getglobal(g.mod, g.name))
    end
end

function canonical(c::Canonical, @nospecialize(x::Array))
    print(c.io, "array ")
    canonical(c, typeof(x))
    print(c.io, size(x))
    if isbitstype(eltype(x))
        print(c.io, " ", bytes2hex(sha256(reinterpret(UInt8, vec(x)))))
    else
        foreach(y -> (print(c.io, " "); canonical(c, y)), x)
    end
end

function canonical(c::Canonical, @nospecialize(T::Type))
    print(c.io, "type ")
    canonical_type(c, T)
end

# Any other value: its type, then its fields.
function canonical(c::Canonical, @nospecialize(x))
    T = typeof(x)
    canonical_type(c, T)
    Base.issingletontype(T) && return
    if ismutable(x)
        x in c.seen && return print(c.io, "(seen)")
        push!(c.seen, x)
    end
    print(c.io, "(")
    for i in 1:nfields(x)
        i > 1 && print(c.io, ", ")
        isdefined(x, i) ? canonical(c, getfield(x, i)) : print(c.io, "#undef")
    end
    print(c.io, ")")
end

# A term's fields but its hash, which is the session's.
function canonical(c::Canonical, t::Union{Apply, Recomputed})
    canonical_type(c, typeof(t))
    print(c.io, "(")
    for i in 1:(nfields(t) - 1)
        i > 1 && print(c.io, ", ")
        canonical(c, getfield(t, i))
    end
    print(c.io, ")")
end

function canonical_type(c::Canonical, @nospecialize(T::DataType))
    source = generated_source(T)
    if source !== nothing
        print(c.io, "generated ")
        return canonical(c, source)
    end
    m = T.name.module
    m === Generated && error("$T was generated without `Tempest.Compiler.generated`, so it cannot be part of a cache key")
    canonical(c, m)
    print(c.io, ".", T.name.name)
    if !isempty(T.parameters)
        print(c.io, "{")
        for (i, p) in enumerate(T.parameters)
            i > 1 && print(c.io, ", ")
            p isa Union{Type, TypeVar} ? canonical_type(c, p) : canonical(c, p)
        end
        print(c.io, "}")
    end
    # A type outside any package is identified by its fields and by the code of
    # its methods, when it is called.
    identified(m) || T in c.seen || canonical_definition(c, T)
end

function canonical_type(c::Canonical, @nospecialize(T::Union))
    print(c.io, "Union{")
    foreach(S -> (canonical_type(c, S); print(c.io, ", ")), Base.uniontypes(T))
    print(c.io, "}")
end

function canonical_type(c::Canonical, @nospecialize(T::UnionAll))
    print(c.io, "where ", T.var.name, " <: ")
    canonical_type(c, T.var.ub)
    print(c.io, " ")
    canonical_type(c, T.body)
end

canonical_type(c::Canonical, v::TypeVar) = print(c.io, "var ", v.name)
canonical_type(c::Canonical, ::Core.TypeofBottom) = print(c.io, "Union{}")

function canonical_definition(c::Canonical, @nospecialize(T::DataType))
    push!(c.seen, T)
    print(c.io, " defined as [")
    if isstructtype(T)
        for i in 1:fieldcount(T)
            print(c.io, fieldname(T, i), "::")
            canonical_type(c, fieldtype(T, i))
            print(c.io, " ")
        end
    end
    matches = Base._methods_by_ftype(Tuple{T, Vararg{Any}}, -1, Base.get_world_counter())
    for match in something(matches, ())
        method = match.method
        print(c.io, " method ")
        canonical_type(c, method.sig)
        code = try
            Base.uncompressed_ast(method).code
        catch
            Any[string(method.file, ":", method.line)]
        end
        foreach(s -> (print(c.io, "; "); canonical(c, s)), code)
    end
    print(c.io, "]")
end

# Whether a module's code is identified by its package: Core, Base, the
# standard library, and installed or developed packages.
identified(m::Module) = (r = Base.moduleroot(m); r === Core || r === Base || pkgdir(r) !== nothing)

const IDENTITIES = Dict{Module, String}()
const IDENTITIES_LOCK = ReentrantLock()

"""
    package_identity(m) -> String

What identifies the code of the package `m`: the Julia build for Core, Base and
the standard library, the installation path (which names the version's content)
for an installed package, and a hash of its sources for a developed one: the
.jl files under src/ and ext/, not the .cov files a run with coverage writes
beside them as its processes exit.
"""
package_identity(m::Module) = @lock IDENTITIES_LOCK get!(() -> find_identity(m), IDENTITIES, m)

function find_identity(m::Module)
    julia = string("julia ", VERSION, " ", Base.GIT_VERSION_INFO.commit)
    (m === Core || m === Base) && return julia
    dir = pkgdir(m)
    dir === nothing && return "unidentified"
    startswith(dir, Sys.STDLIB) && return julia
    any(depot -> startswith(dir, joinpath(depot, "packages")), DEPOT_PATH) && return dir
    ctx = SHA2_256_CTX()
    for sub in ("src", "ext"), (root, _, files) in walkdir(joinpath(dir, sub))
        for file in sort(filter(endswith(".jl"), files))
            path = joinpath(root, file)
            update!(ctx, codeunits(relpath(path, dir)))
            update!(ctx, read(path))
        end
    end
    string("sources ", bytes2hex(digest!(ctx)))
end

## What a kernel depends on of Tempest's own code

# Tempest and its extensions, whose code a checked key leaves out; not the
# kernels Tempest generates, whose source the key holds.
own_code(m::Module) = m !== Generated && (r = Base.moduleroot(m); r === ROOT || pkgdir(r) == pkgdir(ROOT))

const ROOT = Base.moduleroot(@__MODULE__)

# Inference keeps a code instance's edges from Julia 1.12. Before, a kernel has
# nothing to be checked by, and its key keeps Tempest's identity.
const EDGES = hasfield(Core.CodeInstance, :edges)

"""
    dependencies(code, lookup) -> Vector{UInt8}

What a kernel compiled from the code instance `code` depends on of Tempest's
own code, written out: each method of Tempest or its extensions its inference
reached, with its lowered code; each such global it read, with its value; and
the fields of each such type its methods were specialized on. They are found
through the edges inference keeps, `lookup(mi)` giving the code instance of a
method instance an edge names, as the compiler that inferred them caches it.
"""
function dependencies(code, lookup)
    EDGES || return UInt8[]
    records = Set{Tuple}()
    seen = Base.IdSet{Any}()
    function visit(x)
        x in seen && return
        push!(seen, x)
        if x isa Core.CodeInstance
            visit(x.def)
            foreach(visit, x.edges)
        elseif x isa Core.MethodInstance
            x.def isa Method && visit(x.def)
            own_types!(records, x.specTypes)
            c = lookup(x)
            c === nothing || visit(c)
        elseif x isa Method
            own_code(x.module) && push!(records, (:method, x.sig, method_text(x)))
        elseif x isa Core.Binding
            g = x.globalref
            own_code(g.mod) && push!(records, (:global, g, global_text(g)))
        end
    end
    visit(code)
    io = IOBuffer()
    serialize(io, sort!(collect(records); by = string))
    take!(io)
end

# The types of Tempest's own code among `T`'s parameters, with their fields.
function own_types!(records, @nospecialize(T))
    T = Base.unwrap_unionall(T)
    if T isa Union
        foreach(S -> own_types!(records, S), Base.uniontypes(T))
    elseif T isa DataType
        own_code(T.name.module) && isstructtype(T) && push!(records, (:type, T.name.wrapper, type_text(T.name.wrapper)))
        foreach(p -> p isa Type && own_types!(records, p), T.parameters)
    end
end

const CHECKED = Dict{Tuple, Bool}()
const CHECKED_LOCK = ReentrantLock()

"""
    current(bytes) -> Bool

Whether what a kernel depends on of Tempest's code, as `dependencies` wrote it,
is unchanged in this session: each method is still the one its signature
dispatches to, with the same lowered code; each global has the same value; each
type the same fields. What cannot be read back, as a type gone, is changed.
"""
function current(bytes)
    isempty(bytes) && return true
    records = try
        deserialize(IOBuffer(bytes))
    catch
        return false
    end
    @lock CHECKED_LOCK all(r -> get!(() -> unchanged(r), CHECKED, r), records)
end

function unchanged(r)
    kind, x, text = r
    try
        kind === :method && return method_text(which(x)) == text
        kind === :global && return global_text(x) == text
        kind === :type && return type_text(x) == text
    catch
    end
    false
end

function method_text(m::Method)
    c = Canonical(IOBuffer(), Set{Module}(), Base.IdSet{Any}(), Dict{Symbol, Int}())
    canonical_type(c, m.sig)
    code = try
        Base.uncompressed_ast(m).code
    catch
        Any[string(m.file, ":", m.line)]
    end
    foreach(s -> (print(c.io, "; "); canonical(c, s)), code)
    bytes2hex(sha256(take!(c.io)))
end

function global_text(g::GlobalRef)
    isdefined(g.mod, g.name) || return "undefined"
    c = Canonical(IOBuffer(), Set{Module}(), Base.IdSet{Any}(), Dict{Symbol, Int}())
    canonical(c, getglobal(g.mod, g.name))
    bytes2hex(sha256(take!(c.io)))
end

function type_text(@nospecialize(T))
    S = Base.unwrap_unionall(T)
    string(fieldnames(S), " ", S.types)
end
