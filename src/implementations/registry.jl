# The implementations a target uses by default. A package that provides an
# Implementation registers it under its module from `__init__`, with a rank;
# the target's baseline kernels belong to the target and are not registered.
# Candidates are ordered by rank, then by name, never by load order, and each
# `prepare` resolves them once.

mutable struct Registration
    mod::Module
    constructor::Any
    target::Type
    rank::Int
    instance::Union{Nothing, Implementation} # the active instance; nothing while inactive
end

const REGISTRY = Dict{Module, Registration}()
const REGISTRY_LOCK = ReentrantLock()

# Changes a scope makes to the registry: module => instance, or nothing when
# deactivated. A nested scope adds its changes to the enclosing scope's.
const SCOPED_IMPLEMENTATIONS =
    ScopedValue{Dict{Module, Union{Nothing, Implementation}}}(Dict{Module, Union{Nothing, Implementation}}())

"""
    Tempest.register!(mod::Module, constructor; target = CUDATarget, rank, active = true)

Register `constructor` as the implementation `mod` provides for `target`.
Call it from `__init__`, of a package or of an extension, not at top level:
mutation of another module's state during precompilation does not persist.
`constructor(; kwargs...)` builds the instance that `Tempest.activate(mod;
kwargs...)` enables; with `active = true` its default instance is enabled now.
Candidates are ordered by `rank`, lowest first, then by module name.
Registering a module again replaces its registration.
"""
function register!(mod::Module, constructor; target::Type = CUDATarget, rank::Integer, active::Bool = true)
    target <: ExecutionTarget || throw(ArgumentError("target must be an ExecutionTarget type"))
    instance = active ? construct(mod, constructor) : nothing
    lock(REGISTRY_LOCK) do
        REGISTRY[mod] = Registration(mod, constructor, target, Int(rank), instance)
    end
    nothing
end

function construct(mod, constructor; kwargs...)
    instance = constructor(; kwargs...)
    instance isa Implementation || throw(
        ArgumentError("the constructor registered for $mod returned a $(typeof(instance)), not an Implementation"),
    )
    instance
end

function registration(mod::Module)
    found = lock(() -> get(REGISTRY, mod, nothing), REGISTRY_LOCK)
    found === nothing && throw(ArgumentError(unregistered_message(mod)))
    found
end

# The registration whose constructor is the instance's type.
function registration(instance::Implementation)
    found = lock(REGISTRY_LOCK) do
        for r in values(REGISTRY)
            r.constructor isa Type && instance isa r.constructor && return r
        end
        nothing
    end
    found === nothing && throw(
        ArgumentError(
            "no registered module provides $(typeof(instance)); pass it to " *
            "prepare(...; implementations), or register its module with Tempest.register!",
        ),
    )
    found
end

# For a trigger of one of Tempest's extensions, name the packages still missing.
function unregistered_message(mod)
    name = string(nameof(mod))
    project = Base.parsed_toml(joinpath(pkgdir(@__MODULE__), "Project.toml"))
    for (extension, triggers) in get(project, "extensions", Dict{String, Any}())
        triggers = triggers isa String ? [triggers] : triggers
        name in triggers || continue
        absent = filter(t -> !package_loaded(t), triggers)
        isempty(absent) && continue
        return "$name is loaded but Tempest's $extension isn't; load $(join(absent, ", ", " and "))"
    end
    baseline = "Tempest's baseline kernels belong to their target and cannot be deactivated"
    "$name has not registered a Tempest implementation. $baseline; " *
    "a package registers its own with Tempest.register! from its __init__"
end

package_loaded(name) = any(m -> string(nameof(m)) == name, values(Base.loaded_modules))

"""
    Tempest.activate(mod::Module; kwargs...)
    Tempest.activate(implementation::Implementation)
    Tempest.activate(f, mod_or_implementation; kwargs...)

Enable the implementation `mod` registered, constructed with `kwargs`, or the
given instance of a registered implementation, as a default candidate of the
executables prepared afterwards. For example `Tempest.activate(cuDNN;
float32_cast = Core.BFloat16)`. Existing executables keep their candidates.
With a function `f`, the change holds only while `f()` runs, in its task and
the tasks it starts, and `f()`'s value is returned. A candidate passed to
`prepare(...; implementations)` precedes a scope's, which overrides the
registry's.
"""
function activate(mod::Module; kwargs...)
    r = registration(mod)
    instance = construct(mod, r.constructor; kwargs...)
    lock(() -> (r.instance = instance), REGISTRY_LOCK)
    instance
end

function activate(instance::Implementation)
    r = registration(instance)
    lock(() -> (r.instance = instance), REGISTRY_LOCK)
    instance
end

function activate(f::Function, mod::Module; kwargs...)
    r = registration(mod)
    with_registry_changes(f, mod => construct(mod, r.constructor; kwargs...))
end

activate(f::Function, instance::Implementation) =
    with_registry_changes(f, registration(instance).mod => instance)

"""
    Tempest.deactivate(mods::Module...)
    Tempest.deactivate(f, mods::Module...)

Remove the implementations these modules registered from the default
candidates, for example `Tempest.deactivate(cuTile)` to use CUDA's generated
kernels everywhere. A target's baseline kernels cannot be deactivated. With a
function `f`, only while `f()` runs.
"""
function deactivate(mods::Module...)
    registrations = map(registration, mods)
    lock(REGISTRY_LOCK) do
        foreach(r -> r.instance = nothing, registrations)
    end
    nothing
end

function deactivate(f::Function, mods::Module...)
    foreach(registration, mods)
    with_registry_changes(f, (mod => nothing for mod in mods)...)
end

function with_registry_changes(f, changes...)
    scoped = merge(SCOPED_IMPLEMENTATIONS[], Dict{Module, Union{Nothing, Implementation}}(changes))
    with(f, SCOPED_IMPLEMENTATIONS => scoped)
end

"""
    baseline_implementations(target) -> Tuple of (implementation, rank) pairs

The kernels that are part of a target: always candidates, ordered among the
registered implementations by rank. Unknown targets have none.
"""
baseline_implementations(::ExecutionTarget) = ()
baseline_implementations(::CPUTarget) = ((CPUKernels(), 20), (LinearAlgebraGemm(), 30))

# (implementation, source) pairs, by rank and then by name. A source is
# :target for baseline kernels, :scoped for a scope's change, :registry otherwise.
function default_candidates(target::ExecutionTarget)
    scoped = SCOPED_IMPLEMENTATIONS[]
    entries = Tuple{Int, String, Implementation, Symbol}[]
    for (instance, rank) in baseline_implementations(target)
        push!(entries, (rank, string(implementation_name(instance)), instance, :target))
    end
    lock(REGISTRY_LOCK) do
        for r in values(REGISTRY)
            target isa r.target || continue
            instance, source = haskey(scoped, r.mod) ? (scoped[r.mod], :scoped) : (r.instance, :registry)
            instance === nothing || push!(entries, (r.rank, string(nameof(r.mod)), instance, source))
        end
    end
    sort!(entries; by = entry -> (entry[1], entry[2]))
    [(entry[3], entry[4]) for entry in entries]
end

"""
    default_implementations(target) -> Tuple

The candidates appended after the user's implementations during preparation:
the target's baseline kernels and the active registered implementations (see
`Tempest.activate`), ordered by rank and then by name. Each executable
resolves this list once.
"""
default_implementations(target::ExecutionTarget) = Tuple(first.(default_candidates(target)))
