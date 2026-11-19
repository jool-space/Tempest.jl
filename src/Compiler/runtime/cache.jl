# The kernel cache: compiled binaries kept across sessions, keyed by the content
# of what was compiled, so that preparing a program in a new process loads
# binaries instead of inferring and compiling kernels again.
#
# A key is a hash of the kernel's code and everything that decides its binary:
# the generated source, written out with every function and type it names; the
# argument types; the compiler options; the toolchain; and the identity of each
# package those names come from, the build of the image it was loaded from.
# Code no image holds, a script's, or a package's loaded from its files, is
# written by its code: its lowered code, with the values of the constants it
# reads, and a value by its content. The binary's parameters follow from the
# argument types, so a cached kernel is its entry name and its binary.
#
# A key names the code a kernel is compiled from only as far as its source
# names it: what the compilation reached beyond, by inference, the key cannot
# name, since naming it would be inferring the kernel, which a load is there
# to save. A method a script writes for a package's function, for a type of
# its own, a method more specific than the one a call matched, added since, or
# one deleted, change the kernel and not its key. So a kernel is stored with
# what its compilation used, read from the edges inference keeps
# (`dependencies`): each call's signature with the methods it matched in the
# method table it was dispatched in, each `invoke`, each global inference
# kept an edge to and each type, with what identifies each then; and it loads
# only while each of them still holds in this process (`holds`), as Julia
# checks a package image's code against the session that loads it. Each is
# stored once, under its digest, and checked once a session, as kernels share
# most of what they reach. A kernel whose dependencies cannot all be written
# so that another process reads back the same, or whose inference cannot be
# read, is not stored.
#
# A method is identified by the build of the image that holds it, and
# otherwise by its code as loaded (`method_identity`). Tempest's own code is
# identified by its code wherever it was loaded from, and left out of a key
# (`content_key`'s `checked`), so that an edit to Tempest compiles again only
# the kernels whose code it changed. A kernel that failed to compile is kept
# the same way, with what its inference used, and fails again only while that
# holds. Before Julia 1.12 inference keeps no edges, there is nothing to check
# a kernel by, and nothing is loaded from the store: kernels compile in every
# process, once a session.
#
# The binaries are kept in Julia's object cache, through CompilerCaching, with
# cuTile's: the runtime's store from Julia 1.14, a store of CompilerCaching's
# before. The store is shared and bounded (`JULIA_OBJCACHE_CAPACITY`, by default
# 512 MiB), removes the entries used least recently, and may lose a write, so a
# kernel can compile again; `JULIA_OBJCACHE=0` turns it off. A kernel whose
# dependencies were lost with it compiles again. Entries of an earlier layout
# are under other keys (`KERNEL_SCHEMA`), never read, and removed as the
# store fills.

using SHA
using Serialization
using CompilerCaching: ObjCache

public content_key, cached_kernel, store_kernel, store_failure, uncached_kernel, generated, dependencies, definitions

"Kernels loaded from the cache and kernels compiled (or failed to), in this session."
const CACHE_COUNTS = (loaded = Threads.Atomic{Int}(0), compiled = Threads.Atomic{Int}(0))

cache_counts() = (loaded = CACHE_COUNTS.loaded[], compiled = CACHE_COUNTS.compiled[])

# Tempest's kernels in the object cache, what they depend on, each under its
# digest, and the version of an entry's layout: the number of a kernel's
# dependencies, their digests, the length of the entry name, the name, and
# the binary; or a length of zero and the compiler's message for a kernel that
# failed to compile.
const KERNELS = "Tempest/kernel"
const DEPENDENCIES = "Tempest/dependency"
const KERNEL_SCHEMA = 4

store_key(key) = ObjCache.keyhash(KERNEL_SCHEMA, key)

"""
    load_kernel(key) -> (entry, binary) or nothing

The cached kernel under `key`, if there is one and what its compilation used
still holds (`holds`). A kernel that failed to compile fails again, with the
compiler's message.
"""
function load_kernel(key)
    EDGES || return nothing
    bytes = ObjCache.get(KERNELS, store_key(key))
    bytes === nothing && return nothing
    d = Int(only(reinterpret(UInt32, bytes[1:4])))
    held(reinterpret(UInt128, bytes[5:(4 + 16d)])) || return nothing
    at = 5 + 16d
    n = Int(only(reinterpret(UInt32, bytes[at:(at + 3)])))
    Threads.atomic_add!(CACHE_COUNTS.loaded, 1)
    n == 0 && error("this kernel failed to compile before:\n", String(bytes[(at + 4):end]))
    (String(bytes[(at + 4):(at + 3 + n)]), bytes[(at + 4 + n):end])
end

"""
    store_kernel(key, entry, binary; dependencies) -> (entry, binary)

Count a compiled kernel and keep it under `key`, with what its compilation
used (see `dependencies`); not where that could not be read (`nothing`).
A failure to write only loses the entry.
"""
function store_kernel(key, entry::String, binary::Vector{UInt8}; dependencies)
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    put_entry(key, dependencies, [reinterpret(UInt8, [UInt32(ncodeunits(entry))]); codeunits(entry); binary])
    (entry, binary)
end

"""
    store_failure(key, message; dependencies)

Count a kernel that failed to compile and keep the compiler's message under
`key`, with what its inference used, so that it fails again while that holds.
"""
function store_failure(key, message::String; dependencies)
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    put_entry(key, dependencies, [reinterpret(UInt8, [UInt32(0)]); codeunits(message)])
end

# An entry whose dependencies could not be read, or not all written so that
# another process reads them back the same (`written`), is not kept, nor any
# before Julia 1.12, which nothing would load.
put_entry(key, ::Nothing, rest) = nothing

function put_entry(key, dependencies, rest)
    EDGES || return nothing
    each = map(written, dependencies)
    any(isnothing, each) && return nothing
    digests = UInt128[kept(bytes) for bytes in each]
    ObjCache.put!(KERNELS, store_key(key), [reinterpret(UInt8, [UInt32(length(digests))]); reinterpret(UInt8, digests); rest])
    nothing
end

"""
    uncached_kernel(entry, binary) -> (entry, binary)

Count a kernel compiled in this session that the cache cannot keep, one that
refers to values of the session, and return it as `cached_kernel` expects.
"""
uncached_kernel(entry, binary) = (Threads.atomic_add!(CACHE_COUNTS.compiled, 1); (entry, binary))

# Within a session a key is resolved once. Programs stack identical layers, whose
# groups build their kernels at the same time and so have the same key in flight
# together: the first caller loads or compiles it, and the others wait for that
# task, or get its result after, instead of loading or compiling again. The
# task is made under the table's lock, as `get!` on a session table would not:
# a kernel compiled twice is more than waste.
const RESOLVED_KERNELS = SessionTable{String, Task}()

"""
    cached_kernel(compile, key) -> (entry, binary)

The kernel under `key`: from the object cache, or compiled by `compile()`, which
returns `(entry, binary)` and stores it, when the cache has none. This happens
once per key in a session, so that callers in flight together wait for the
first, and a failure fails every caller. The kernel is loaded or compiled on
a thread from the moment it is asked for, as neither makes a device object:
the tasks that build a stage's options side by side (`concurrently`) take
turns on one thread, and a compilation they ran there would begin only once
the last of them had asked for its kernel, as the GPT step's first stage of
147 kernels asked for its last 2.8 s in.
"""
function cached_kernel(@nospecialize(compile), key)
    observe(:kernel, key)
    task = @lock RESOLVED_KERNELS.lock get!(answers(RESOLVED_KERNELS), key) do
        Threads.@spawn resolved_kernel(compile, key)
    end
    fetch_result(task)
end

# The kernel under `key`, loaded, or compiled by `compile()` where the cache has
# none. Compiled once, whatever compiles kernels, and ahead of a session (see
# CUDAExt): the many tasks that first ask for kernels at once would each
# compile it.
Base.@nospecializeinfer function resolved_kernel(@nospecialize(compile), key::String)
    found = load_kernel(key)
    found === nothing ? phase(compile, :compile) : found
end

# Generated kernels live here.
module Generated end

# Generated functions, by type, with the source that defined them: in a key a
# generated function is its source, since its name differs between sessions.
const GENERATED_SOURCES = IdDict{Any, Expr}()
const GENERATED_LOCK = ReentrantLock()

"""
    generated(source::Expr, into = Generated) -> function

Define the function `source`, an anonymous function expression, in module
`into`, keeping its source: in a `content_key` a generated function is its
source, since its name differs between sessions. A package's precompile
workload defines its kernels in a module of its own, since `Generated` is
closed to it then.
"""
function generated(source::Expr, into::Module = Generated)
    f = own_definition(into, source)
    @lock GENERATED_LOCK GENERATED_SOURCES[typeof(f)] = source
    f
end

generated_source(T) = @lock GENERATED_LOCK get(GENERATED_SOURCES, T, nothing)

"""
    content_key(parts...; checked = false) -> String

A hash of `parts` that is the same in every session for the same content: of
their `canonical_text`. With `checked`, Tempest's own code is named without its
identity, for a kernel stored with what its compilation used of that code
(`dependencies`).
"""
content_key(@nospecialize(parts...); checked = false) = bytes2hex(sha256(canonical_text(parts...; checked)))

"The content of `parts` written out, with the identities of the packages they name."
function canonical_text(@nospecialize(parts...); checked = false)
    c = Canonical()
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

Canonical() = Canonical(IOBuffer(), Set{Module}(), Base.IdSet{Any}(), Dict{Symbol, Int}())

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
# A float by its bits, which its printing does not tell apart for NaNs.
canonical(c::Canonical, x::Base.IEEEFloat) = print(c.io, typeof(x), "(", bytes2hex(reinterpret(UInt8, [x])), ")")
# A simple vector, as a type's parameters are held, by its elements.
canonical(c::Canonical, x::Core.SimpleVector) = (print(c.io, "svec("); foreach(y -> (canonical(c, y); print(c.io, ", ")), x); print(c.io, ")"))
canonical(c::Canonical, m::Module) = (push!(c.roots, Base.moduleroot(m)); print(c.io, "module ", join(fullname(m), ".")))

function canonical(c::Canonical, g::GlobalRef)
    canonical(c, g.mod)
    print(c.io, ".", g.name)
    # Code no image holds is identified by what its names refer to; and so is
    # a constant no image vouches for, which inference may fold into code
    # compiled from it.
    if isdefined(g.mod, g.name) && !(g in c.seen) && (!identified(g.mod) || unvouched_constant(g))
        push!(c.seen, g)
        print(c.io, " = ")
        canonical(c, getglobal(g.mod, g.name))
    end
end

# A constant whose value its package's identity does not name: Tempest's own,
# whose identity a checked key leaves out, or one whose binding no image holds
# as it is now, defined since the image was loaded, a redefinition making a
# new partition of the binding. Its value is written by its content, as any
# value is, through its structure to its leaves; a mutable one's too, a
# table, which inference folds into code where a reader of it is declared
# foldable (`Base.@assume_effects`).
unvouched_constant(g::GlobalRef) = isconst(g.mod, g.name) && (own_code(g.mod) || !defined_in_image(convert(Core.Binding, g)))

defined_in_image(b::Core.Binding) = isdefined(b, :partitions) && in_image(b.partitions)

# An array or a memory by its elements: a dictionary, whose fields hold its
# keys and values in memories, by its contents.
function canonical(c::Canonical, @nospecialize(x::Union{Array, Memory}))
    print(c.io, "array ")
    canonical(c, typeof(x))
    print(c.io, size(x))
    if isbitstype(eltype(x))
        print(c.io, " ", bytes2hex(sha256(reinterpret(UInt8, collect(vec(x))))))
    else
        for i in eachindex(x)
            print(c.io, " ")
            isassigned(x, i) ? canonical(c, x[i]) : print(c.io, "#undef")
        end
    end
end

function canonical(c::Canonical, @nospecialize(T::Type))
    print(c.io, "type ")
    canonical_type(c, T)
end

# Any other value: its type, then its fields; a primitive value, as an enum's,
# by its bytes; a pointer, which means nothing in another process, by its
# address in this session, so that no other session is handed what was
# derived from it.
function canonical(c::Canonical, @nospecialize(x))
    T = typeof(x)
    canonical_type(c, T)
    if isprimitivetype(T)
        print(c.io, "(", bytes2hex(reinterpret(UInt8, [x])))
        x isa Union{Ptr, Core.LLVMPtr} && print(c.io, " in session ", session())
        return print(c.io, ")")
    end
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
    print(c.io, " defined as [", ismutabletype(T) ? "mutable " : "", "<: ")
    canonical_type(c, supertype(T))
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
        foreach(s -> (print(c.io, "; "); canonical(c, s)), lowered_code(method))
    end
    print(c.io, "]")
end

# Whether a module's code is identified by its package: Core, Base, and a
# package loaded from an image, whose build names the code it holds, installed
# or developed; and Tempest, whose own code a checked key leaves out, and
# whose image its precompile workload is building. A package loaded from its
# files is written by its code, as a script is. Asked of every module a
# kernel's source names.
identified(m::Module) = get!(() -> find_identified(Base.moduleroot(m)), IDENTIFIED, Base.moduleroot(m))
find_identified(r::Module) = r === Core || r === Base || r === ROOT || (pkgdir(r) !== nothing && (in_image(r) || own_root(r)))

const IDENTIFIED = SessionTable{Module, Bool}()

"""
    definitions(code) -> Vector{String}

What a session table's key holds of `code`, functions, folds and types an
answer was derived from, beside the objects themselves (see `SessionTable`):
the content key of each no package identifies (see `content_key`), which
writes it by its definition, as a store's key does, so that a function
redefined in the session, the same object, is another key. A package's code
is named by the object alone, since no code changes within a session
(`session`).
Sorted, so that code named in any order is named alike.
"""
definitions(code) = sort!(unique!(String[content_key(x) for x in code if !of_packages(x)]))

# Whether all the code `x` is, its type, its type's parameters and what its
# fields hold, is of packages. A parameter is a type or a value, as a function
# in `Val{f}`, whose code is as much the type's as a type's would be. An array
# is its type: its elements are data. A mutable value is walked once.
function of_packages(@nospecialize(x), seen = Base.IdSet{Any}())
    x isa Type && return type_of_packages(x)
    x isa Module && return identified(x)
    type_of_packages(typeof(x)) || return false
    (isbits(x) || x isa AbstractArray || x in seen) && return true
    ismutable(x) && push!(seen, x)
    all(i -> !isdefined(x, i) || of_packages(getfield(x, i), seen), 1:nfields(x))
end

function type_of_packages(@nospecialize(T))
    T = Base.unwrap_unionall(T)
    T isa Union && return all(type_of_packages, Base.uniontypes(T))
    T isa DataType || return true
    identified(T.name.module) && all(of_packages, T.parameters)
end

const IDENTITIES = SessionTable{Module, String}()

"""
    package_identity(m) -> String

What identifies the code of the package `m` as this process loaded it: the
build of the image it was loaded from, which Julia rebuilds when the
package's files, its dependencies or its compile-time preferences change.
A package loaded from its files has none, and its code is written out where
a key names it (`identified`).
"""
package_identity(m::Module) = get!(() -> find_identity(m), IDENTITIES, m)

find_identity(m::Module) = identified(m) ? string("image ", Base.module_build_id(m)) : "unidentified"

## What a kernel's compilation used

# Tempest and its extensions, whose code a checked key leaves out; not the
# kernels Tempest generates, whose source the key holds. Asked of every
# module a key names, so the answer is kept per module: `pkgdir` takes the
# loading lock, on which kernels compiled side by side waited.
own_code(m::Module) = m !== Generated && (r = Base.moduleroot(m); r === ROOT || own_root(r))

const OWN = SessionTable{Module, Bool}()

own_root(r::Module) = get!(() -> pkgdir(r) == pkgdir(ROOT), OWN, r)

const ROOT = Base.moduleroot(@__MODULE__)

# Inference keeps a code instance's edges from Julia 1.12. Before, a kernel has
# nothing to be checked by, and none is loaded from the store.
const EDGES = hasfield(Core.CodeInstance, :edges)

"""
What a kernel's compilation used, and what it found then, as `dependencies`
reads it: a call (`:call`) of a signature, and the methods it matched in the
method table the compilation dispatched in (`table`: the global holding an
overlay table, or `nothing` for Julia's own); an `invoke` (`:invoke`) of a
signature, and the method it named; a global inference kept an edge to
(`:global`) and what it held, where the constants a method reads of its own
module, which inference keeps no edge to, are written with the method's code
(`method_identity`); or a type (`:type`) a method was specialized on, its
fields, mutability and supertype. What it
found is written by what identifies it (`method_identity`, `global_identity`,
`type_text`).
"""
struct Dependency
    kind::Symbol
    table::Union{Nothing, GlobalRef}
    named::Any
    found::String
end

"""
    dependencies(code, lookup, table) -> Vector{Dependency} or nothing

What a kernel compiled from the code instance `code` used, as inference's
edges record it: each call its inference made, from the kernel down, with the
methods it matched in `table`, the method table the compilation dispatched
in; each `invoke`; each global its edges name or a method it reached names
in its code; and each type its methods were specialized on that no package's
image holds. Nothing where a method's code to read its globals from is not
kept: the kernel is not stored. `lookup(mi)` gives the code instance of a
method instance an edge names, as the compiler that inferred them caches it.
"""
function dependencies(code, lookup, table::Union{Nothing, Core.MethodTable})
    EDGES || return Dependency[]
    at = table_name(table)
    found = Set{Dependency}()
    seen = Base.IdSet{Any}()
    unread = Ref(false)
    keep!(kind, named, text) = push!(found, Dependency(kind, at, named, text))
    function visit(x)
        x in seen && return
        push!(seen, x)
        if x isa Core.CodeInstance
            visit(instance(x))
            read_edges(x.edges) do kind, named, callees
                foreach(visit, callees)
                if kind === :global
                    keep!(kind, named.globalref, global_identity(named))
                else
                    keep!(kind, named, identities(callee_method(c) for c in callees))
                end
            end
        elseif x isa Core.MethodInstance
            x.def isa Method && visit(x.def)
            specialized_types!(keep!, x.specTypes)
            c = lookup(x)
            c === nothing || visit(c)
        elseif x isa Method
            read = globals_read(x)
            read === nothing && return (unread[] = true)
            foreach(g -> keep!(:global, g, global_identity(convert(Core.Binding, g))), read)
        end
    end
    visit(code)
    unread[] ? nothing : collect(found)
end

"""
    globals_read(m) -> Vector{GlobalRef} or nothing

The globals method `m`'s code names, to which inference keeps no edge where
they are of the method's own module: Julia finds them in a precompiled
method's source as it checks it. Nothing where its code is not kept, and a
kernel that reached it is not stored. Kept per method.
"""
globals_read(m::Method) = get!(() -> find_globals(m), GLOBALS_READ, m)

const GLOBALS_READ = SessionTable{Method, Union{Nothing, Vector{GlobalRef}}}()

function find_globals(m::Method)
    isdefined(m, :generator) || isdefined(m, :source) || return nothing
    found = GlobalRef[]
    globals_named!(found, lowered_code(m))
    unique!(found)
end

function globals_named!(found, @nospecialize(x))
    if x isa GlobalRef
        push!(found, x)
    elseif x isa Expr
        foreach(a -> globals_named!(found, a), x.args)
    elseif x isa AbstractVector
        foreach(a -> globals_named!(found, a), x)
    end
    found
end

# The calls, invokes and globals among a code instance's edges, as inference
# writes them and Julia's `verify_method` reads them: a method instance or a
# code instance alone is a call of its signature that matched its method; a
# count, a signature and that many callees, a call that matched those; a
# binding, a global; a signature and a callee, an `invoke` of the callee, or
# nothing to check where the callee is a method table.
function read_edges(f, edges)
    j = 1
    while j <= length(edges)
        e = edges[j]
        if e isa Union{Core.CodeInstance, Core.MethodInstance}
            f(:call, instance(e).specTypes, Any[e])
            j += 1
        elseif e isa Int
            f(:call, edges[j + 1], Any[edges[k] for k in (j + 2):(j + 1 + abs(e))])
            j += 2 + abs(e)
        elseif e isa Core.Binding
            f(:global, e, Any[])
            j += 1
        else
            edges[j + 1] isa Core.MethodTable || f(:invoke, e, Any[edges[j + 1]])
            j += 2
        end
    end
end

# The method instance of a code instance, through the override of its ABI
# where it has one.
instance(x) = x isa Core.MethodInstance ? x : instance(x.def)

callee_method(x) = x isa Method ? x : instance(x).def

identities(methods) = join(sort!([method_identity(m) for m in methods]), "\n")

# The types among `T`'s parameters that no package's image holds, with their
# fields: a struct a script defined, or Tempest's own.
function specialized_types!(keep!, @nospecialize(T))
    T = Base.unwrap_unionall(T)
    if T isa Union
        foreach(S -> specialized_types!(keep!, S), Base.uniontypes(T))
    elseif T isa DataType
        isstructtype(T) && !by_package(T.name, T.name.module) && keep!(:type, T.name.wrapper, type_text(T.name.wrapper))
        foreach(p -> p isa Type && specialized_types!(keep!, p), T.parameters)
    end
end

# A method table by the global that holds it; Julia's own as `nothing`.
table_name(::Nothing) = nothing
table_name(t::Core.MethodTable) = t.module === Core && t.name === :methodtable ? nothing : GlobalRef(t.module, t.name)

# The view of a method table that inference dispatches through in `world`.
method_view(::Nothing, world) = Core.Compiler.InternalMethodTable(world)
method_view(g::GlobalRef, world) = isdefined(g.mod, g.name) ? Core.Compiler.OverlayMethodTable(world, getglobal(g.mod, g.name)) : nothing

# A dependency as another process reads it back. A type serialization cannot
# name, an anonymous function of `Main`, which it writes out whole and reads
# back as a type of its own, or a function Tempest generated, whose name is
# its session's, is written by where it is defined and what it is: its global
# and its definition, or its source. Read back, it is that global or that
# source's function in the reading session, where its definition is the same,
# and the dependency does not hold where there is none. A type neither names
# makes the kernel unstorable (`nothing`).
# The module and the name, not a `GlobalRef`, which serialization writes as
# the anonymous function it names, whole.
struct Anonymous
    mod::Module
    name::Symbol
    parameters::Vector{Any}
    definition::String
end

# A generated function by its source's content, which values spliced into it
# take part in: Tempest's own code it names checked by the dependencies, as a
# kernel's key leaves it out, so that a rebuild of Tempest leaves it readable.
struct GeneratedType
    source::String
end

struct Applied
    wrapper::Any
    parameters::Vector{Any}
end

struct Unportable <: Exception end

function written(d::Dependency)
    named = try
        portable(d.named)
    catch err
        err isa Unportable || rethrow()
        return nothing
    end
    io = IOBuffer()
    serialize(io, (d.kind, d.table, named, d.found))
    take!(io)
end

function portable(@nospecialize(x))
    x isa Type || return x
    T = Base.unwrap_unionall(x)
    if T isa Union
        parts = Any[portable(S) for S in Base.uniontypes(T)]
        return all(((S, P),) -> S === P, zip(Base.uniontypes(T), parts)) ? x : Applied(Union, parts)
    end
    T isa DataType || return x
    source = generated_source(T)
    source === nothing || return x isa UnionAll ? throw(Unportable()) : GeneratedType(content_key(source; checked = true))
    T.name.module === Generated && throw(Unportable())
    parameters = Any[p isa Type ? portable(p) : p isa TypeVar ? p : portable_value(p) for p in T.parameters]
    if Serialization.should_send_whole_type(nothing, T)
        x === T.name.wrapper && return Anonymous(T.name.module, T.name.name, Any[], definition_text(x))
        x isa UnionAll && throw(Unportable())
        return Anonymous(T.name.module, T.name.name, parameters, definition_text(T))
    end
    all(((p, q),) -> p === q, zip(T.parameters, parameters)) && return x
    x isa UnionAll && throw(Unportable())
    Applied(T.name.wrapper, parameters)
end

# A value among a type's parameters, as a function in `Val{f}`: its type
# portable where it is one serialization cannot name.
portable_value(@nospecialize(v)) = Base.issingletontype(typeof(v)) && portable(typeof(v)) !== typeof(v) ? throw(Unportable()) : v

function restored(@nospecialize(x))
    x isa Applied && return x.wrapper === Union ? Union{map(restored, x.parameters)...} : x.wrapper{map(restored, x.parameters)...}
    if x isa GeneratedType
        found = generated_type(x.source)
        found === nothing && throw(Unportable())
        return found
    end
    if x isa Anonymous
        isdefined(x.mod, x.name) || throw(Unportable())
        wrapper = getglobal(x.mod, x.name)
        T = isempty(x.parameters) ? wrapper : wrapper{map(restored, x.parameters)...}
        definition_text(T) == x.definition || throw(Unportable())
        return T
    end
    x
end

# The function Tempest generated in this session from a source of the content
# `key`, if any: each generated function's source written once a session.
function generated_type(key::String)
    found = get(GENERATED_BY_CONTENT, key, nothing)
    found === nothing || return found
    asked = session()
    for (T, source) in @lock GENERATED_LOCK collect(GENERATED_SOURCES)
        get(INDEXED, objectid(T), false) && continue
        get!(GENERATED_BY_CONTENT, content_key(source; checked = true), T, asked)
        get!(INDEXED, objectid(T), true, asked)
    end
    get(GENERATED_BY_CONTENT, key, nothing)
end

const GENERATED_BY_CONTENT = SessionTable{String, Any}()
# By identity: a dictionary of types compares them by subtyping.
const INDEXED = SessionTable{UInt64, Bool}()

definition_text(@nospecialize(T)) = (c = Canonical(); canonical_type(c, T); bytes2hex(sha256(take!(c.io))))

# Whether `x`, a method, a binding's partition or a type's name of module `m`,
# is what its package's identity names: held by the image of a package other
# than Tempest, whose build names its code. Not a script's code, nor a method
# a script evaluated into the package's module since, nor a constant defined
# there since, which no image holds.
by_package(@nospecialize(x), m::Module) = !own_code(m) && identified(m) && in_image(x)

in_image(@nospecialize(x)) = ccall(:jl_object_in_image, Cint, (Any,), x) != 0

"""
    method_identity(m) -> String

What identifies the code of method `m` in any session: its package's identity
and its signature, where its package's image holds it; otherwise its code as
loaded (`method_text`): its lowered code, the values of the constants it reads
that no image's identity names, and a generated function's generator's code.
Kept per method, a method's code being fixed within a session.
"""
method_identity(m::Method) = get!(() -> find_method_identity(m), METHOD_IDENTITIES, m)

const METHOD_IDENTITIES = SessionTable{Method, String}()

function find_method_identity(m::Method)
    by_package(m, m.module) || return method_text(m)
    c = Canonical()
    print(c.io, "package ", package_identity(Base.moduleroot(m.module)), " ", m.module, ".", m.name, " ")
    canonical_type(c, m.sig)
    String(take!(c.io))
end

# What identifies a global: its package's, where the package's image holds its
# binding as it is now; the value it holds where it is constant, and the type
# it is declared to hold where not, which is what code compiled from it can
# depend on.
function global_identity(b::Core.Binding)
    owner = imported(b)
    owner === b || return string("imports ", owner.globalref, " ", global_identity(owner))
    g = b.globalref
    isdefined(b, :partitions) && by_package(b.partitions, g.mod) &&
        return string("package ", package_identity(Base.moduleroot(g.mod)), " ", g)
    isconst(g.mod, g.name) ? global_text(g) : string("variable of ", Core.get_binding_type(g.mod, g.name))
end

# The binding a name is bound to by `using` or `import`: a name imported is
# known by what it imports, whose image is its own module's, not the importing
# module's (as Tempest's generated code imports Base's functions).
function imported(b::Core.Binding)
    world = Base.get_world_counter()
    while true
        partition = Base.lookup_binding_partition(world, b)
        Base.is_some_imported(Base.binding_kind(partition)) || return b
        to = Base.partition_restriction(partition)
        if !(to isa Core.Binding)
            g = b.globalref
            owner = Base.binding_module(g.mod, g.name)
            owner === g.mod && return b
            to = convert(Core.Binding, GlobalRef(owner, g.name))
        end
        b = to
    end
end

"""
    holds(d::Dependency) -> Bool

Whether what a kernel's compilation used is as it was, in this session: a
call matches the same methods, an `invoke` names the same method, a global
and a type are identified alike. What cannot be read back, as a type gone,
has changed.
"""
function holds(d::Dependency)
    world = Base.get_world_counter()
    d.kind === :global && return global_identity(convert(Core.Binding, d.named)) == d.found
    d.kind === :type && return type_text(d.named) == d.found
    view = method_view(d.table, world)
    view === nothing && return false
    d.kind === :call ? matched(d.named, view) == d.found : invoked(d.named, view) == d.found
end

function matched(@nospecialize(sig), view)
    found = Core.Compiler.findall(sig, view)
    found === nothing ? "no methods" : identities(m.method for m in found)
end

function invoked(@nospecialize(sig), view)
    match = first(Core.Compiler.findsup(sig, view))
    match === nothing ? "no method" : method_identity(match.method)
end

# Each dependency is kept once under its digest, which a kernel's entry lists,
# and checked once a session: kernels share most of what they reach. Those
# this session wrote to the store are not written again.
const CHECKED = SessionTable{UInt128, Bool}()
const STORED = SessionTable{UInt128, Bool}()

function kept(bytes::Vector{UInt8})
    digest = first(reinterpret(UInt128, sha256(bytes)[1:16]))
    asked = session()
    if get(STORED, digest, false) !== true
        ObjCache.put!(DEPENDENCIES, ObjCache.keyhash(KERNEL_SCHEMA, digest), bytes)
        get!(STORED, digest, true, asked)
    end
    digest
end

# Whether every dependency under `digests` holds: those the session checked
# read under one lock, a kernel listing hundreds, and the others checked.
function held(digests::AbstractVector{UInt128})
    unchecked = UInt128[]
    @lock CHECKED.lock begin
        known = answers(CHECKED)
        for digest in digests
            holding = get(known, digest, nothing)
            holding === false && return false
            holding === nothing && push!(unchecked, digest)
        end
    end
    all(held, unchecked)
end

# Whether the dependency under `digest` holds: false where the store lost it.
function held(digest::UInt128)
    asked = session()
    known = get(CHECKED, digest, nothing)
    known === nothing || return known
    bytes = ObjCache.get(DEPENDENCIES, ObjCache.keyhash(KERNEL_SCHEMA, digest))
    get!(CHECKED, digest, bytes !== nothing && read_holds(bytes), asked)
end

# A dependency that cannot be read back in this session, as one naming a type
# gone or an anonymous function defined otherwise, or bytes the store holds
# damaged, does not hold. Checking one read back raises what it raises.
function read_holds(bytes)
    d = try
        kind, table, named, found = deserialize(IOBuffer(bytes))
        Dependency(kind, table, restored(named), found)
    catch
        return false
    end
    holds(d)
end

function method_text(m::Method)
    c = Canonical()
    canonical_type(c, m.sig)
    foreach(s -> (print(c.io, "; "); canonical(c, s)), lowered_code(m))
    bytes2hex(sha256(take!(c.io)))
end

# A method's lowered code; a generated function's, its generator's. A method
# whose code is not kept is its session's alone, so that no other session is
# handed what was derived from it.
function lowered_code(m::Method)
    if isdefined(m, :generator)
        m.generator isa Core.GeneratedFunctionStub || return Any["generated in session $(session())"]
        return Any[:generated; [s for g in methods(m.generator.gen) for s in lowered_code(g)]]
    end
    isdefined(m, :source) || return Any["without code in session $(session())"]
    Base.uncompressed_ast(m).code
end

function global_text(g::GlobalRef)
    isdefined(g.mod, g.name) || return "undefined"
    c = Canonical()
    canonical(c, getglobal(g.mod, g.name))
    bytes2hex(sha256(take!(c.io)))
end

function type_text(@nospecialize(T))
    S = Base.unwrap_unionall(T)
    string(ismutabletype(S) ? "mutable " : "", supertype(S), " ", fieldnames(S), " ", S.types)
end
