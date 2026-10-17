# The kernel cache: compiled binaries on disk, keyed by the content of what was
# compiled, so that preparing a program in a new process loads binaries instead
# of inferring and compiling kernels again.
#
# A key is a hash of the kernel's code and everything that decides its binary:
# the generated source, written out with every function and type it names; the
# argument types; the compiler options; the toolchain; and the identity of each
# package those names come from (its installed version, or a hash of its sources
# when it is developed). Code outside any package is identified by its lowered
# code. The binary's parameters follow from the argument types, so a cached
# kernel is its entry name and its binary.

using SHA
using Scratch: @get_scratch!

# Tempest's caches live in its scratch spaces in the depot: `kernels` and
# `tuning`, the tuning record. Their paths are found when Tempest loads, so
# none is part of its precompiled image; while Tempest precompiles there are
# none, and its workload uses no cache unless an environment variable names one.
const SCRATCH = (kernels = Ref{Union{Nothing, String}}(nothing), tuning = Ref{Union{Nothing, String}}(nothing))

function __init__()
    SCRATCH.kernels[] = @get_scratch!("kernels")
    SCRATCH.tuning[] = @get_scratch!("tuning")
end

"""
    kernel_cache() -> directory or nothing

Where compiled kernels are kept: `ENV["TEMPEST_KERNEL_CACHE"]` when set, otherwise
Tempest's `kernels` scratch space. Empty or `"0"` turns the cache off.
"""
function kernel_cache()
    dir = get(ENV, "TEMPEST_KERNEL_CACHE", nothing)
    dir === nothing && return SCRATCH.kernels[]
    dir in ("", "0") ? nothing : dir
end

"Kernels loaded from the cache and kernels compiled (or failed to), in this session."
const CACHE_COUNTS = (loaded = Threads.Atomic{Int}(0), compiled = Threads.Atomic{Int}(0))

cache_counts() = (loaded = CACHE_COUNTS.loaded[], compiled = CACHE_COUNTS.compiled[])

entry_path(dir, key) = joinpath(dir, key[1:2], key)

"""
    load_kernel(key) -> (entry, binary) or nothing

The cached kernel under `key`, if there is one. A kernel that failed to compile
fails again, with the compiler's message.
"""
function load_kernel(key)
    dir = kernel_cache()
    dir === nothing && return nothing
    path = entry_path(dir, key)
    isfile(path) || return nothing
    bytes = try
        read(path)
    catch err
        @debug "unreadable kernel cache entry" path exception = err
        return nothing
    end
    n = Int(only(reinterpret(UInt32, bytes[1:4])))
    Threads.atomic_add!(CACHE_COUNTS.loaded, 1)
    n == 0 && error("this kernel failed to compile before:\n", String(bytes[5:end]))
    (String(bytes[5:(4 + n)]), bytes[(5 + n):end])
end

"""
    store_kernel(key, entry, binary) -> (entry, binary)

Count a compiled kernel and keep it under `key`. Writing is atomic, so processes
preparing at once never read a partial entry; a failure to write only loses the
entry.
"""
function store_kernel(key, entry::String, binary::Vector{UInt8})
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    write_entry(key, entry, binary)
    (entry, binary)
end

"Count a kernel that failed to compile and keep the compiler's message."
function store_failure(key, message::String)
    Threads.atomic_add!(CACHE_COUNTS.compiled, 1)
    write_entry(key, "", Vector{UInt8}(message))
end

function write_entry(key, entry, binary)
    dir = kernel_cache()
    dir === nothing && return
    path = entry_path(dir, key)
    try
        mkpath(dirname(path))
        temporary = string(path, ".", getpid(), ".", rand(UInt32), ".partial")
        write(temporary, reinterpret(UInt8, [UInt32(ncodeunits(entry))]), entry, binary)
        mv(temporary, path; force = true)
    catch err
        @debug "could not write a kernel cache entry" path exception = err
    end
end

# Generated functions, by type, with the source that defined them: in a key a
# generated function is its source, since its name differs between sessions.
const GENERATED_SOURCES = IdDict{Any, Expr}()
const GENERATED_LOCK = ReentrantLock()

"Define the function `source` in `Generated`, keeping its source for cache keys."
function generated(source::Expr)
    f = Core.eval(Generated, source)
    @lock GENERATED_LOCK GENERATED_SOURCES[typeof(f)] = source
    f
end

generated_source(T) = @lock GENERATED_LOCK get(GENERATED_SOURCES, T, nothing)

"""
    content_key(parts...) -> String

A hash of `parts` that is the same in every session for the same content: of
their `canonical_text`.
"""
content_key(parts...) = bytes2hex(sha256(canonical_text(parts...)))

"The content of `parts` written out, with the identities of the packages they name."
function canonical_text(parts...)
    c = Canonical(IOBuffer(), Set{Module}(), Base.IdSet{Any}(), Dict{Symbol, Int}())
    print(c.io, "julia ", VERSION, " ", Base.GIT_VERSION_INFO.commit, " options ", compiler_options(), "\n")
    for p in parts
        canonical(c, p)
        print(c.io, "\n")
    end
    for m in sort!(collect(c.roots); by = string)
        print(c.io, m, " = ", package_identity(m), "\n")
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

function canonical_type(c::Canonical, @nospecialize(T::DataType))
    source = generated_source(T)
    if source !== nothing
        print(c.io, "generated ")
        return canonical(c, source)
    end
    m = T.name.module
    m === Generated && error("$T was generated without `Tempest.generated`, so it cannot be part of a cache key")
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
for an installed package, and a hash of its sources for a developed one.
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
        for file in sort(files)
            path = joinpath(root, file)
            update!(ctx, codeunits(relpath(path, dir)))
            update!(ctx, read(path))
        end
    end
    string("sources ", bytes2hex(digest!(ctx)))
end
