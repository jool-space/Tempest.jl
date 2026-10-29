# Work the generator does while it precompiles, without a GPU: preparing
# small programs for CUDA without building them, which plans, claims,
# estimates and decides; then emitting their kernels and naming them for the
# kernel cache, and taking the chosen ones through cuTile's front end to Tile
# IR, for an architecture the machine need not have. cuTile's compiler is
# compiled so here, with Tempest's methods defined, which void cuTile's own
# compiled code: without it a session's first kernels spent 13 s compiling
# Julia. It takes every chosen kernel, since cuTile compiles its emitter again
# for each intrinsic signature and tile type, which one kernel of each kind
# leaves to the session (1.6 s of attention's). Not tileiras, which compiles
# no Julia and would keep its CUBINs in the object cache. Loading and
# launching need a device, so they compile in the session.

using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

# A tile array over no memory, enough to name a kernel: at its offset in a
# buffer at `base`, as `tile_array` places one.
function unbacked_tile_array(T, spec, base)
    S = kernel_eltype(T)
    ct.TileArray(Ptr{S}(base) + spec.offset * sizeof(S), Int32.(spec.sizes), Int32.(spec.strides))
end

# Kernel `k`'s arguments over no memory, for group `g`. Each buffer lies a
# terabyte from the others, so that arrays overlap only where they share one,
# as in a session: the kernel compiles for the aliasing a session's has.
function unbacked_arguments(g, k, workspaces)
    bases = Dict{Any, Int}()
    map(k.args) do spec
        T = spec.buffer isa Int ? g.values[spec.buffer].eltype : workspaces[spec.buffer].eltype
        unbacked_tile_array(T, spec, get!(() -> (length(bases) + 1) << 40, bases, spec.buffer))
    end
end

# The kernels the workload compiles are defined here, since `Generated` is
# closed to it, for an architecture tileiras compiles for without a device.
module Workload end
const WORKLOAD_ARCH = v"12.1"

# An extension has no preferences of its own, since no project names it: its
# workload follows Tempest's, so `[Tempest] precompile_workload = false` turns
# every workload off.
if !isdefined(PrecompileTools, :workload_enabled) || PrecompileTools.workload_enabled(Tempest)
    @setup_workload begin
        sample(dims...) = Float32.(reshape(sin.(1:prod(dims)), dims))
        gelu(x) = x / (1 + exp(-1.702f0 * x))
        rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
        # Causal attention over (features, keys or queries, heads, batch), written out.
        function attention(q, k, v)
            s = Tempest.Compiler.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 3, 4))
            s = ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32)
            Tempest.Compiler.contract(v, exp.(s .- Tempest.logsumexp(s; dims = 1)), (1, 2, 3, 4), (2, 5, 3, 4), (1, 5, 3, 4))
        end
        # Grouped queries: two query heads per key head, the queries over five axes.
        function grouped(q, k, v)
            f, h, n, b = size(q)
            q5 = reshape(q, f, h ÷ size(k, 2), size(k, 2), n, b)
            s = Tempest.Compiler.contract(k, q5, (1, 2, 3, 4), (1, 5, 2, 6, 4), (3, 6, 5, 2, 4))
            s = ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32)
            Tempest.Compiler.contract(v, exp.(s .- Tempest.logsumexp(s; dims = 1)), (1, 2, 3, 4), (3, 6, 5, 2, 4), (1, 6, 5, 2, 4))
        end
        x, w, b, v, y = sample(64, 48), sample(32, 64), sample(32), sample(64), sample(48, 4096)
        tall, long, h = sample(8192, 16), sample(16, 8192), Float16.(sample(256, 64))
        q, k = sample(16, 32, 2, 2), sample(16, 32, 2, 2)
        losses = [
            ((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (w, x, b)),
            ((x, v) -> sum(abs2, rmsnorm(x, v)), (x, v)),
            (x -> sum(exp.(x .- Tempest.logsumexp(x; dims = 1)) .* x), (x,)),
            (y -> sum(exp.(y .- Tempest.logsumexp(y; dims = 2))) + sum(maximum(y; dims = 1)), (y,)),
            ((t, l) -> sum(abs2, sum(t; dims = 1)) + sum(abs2, sum(l; dims = 2)) + sum(abs2, l * t), (tall, long)),
            (h -> sum(abs2, sum(abs2, h; dims = 1)) + sum(abs2, h), (h,)),
            ((q, k) -> sum(abs2, attention(q, k, k) .+ q), (q, k)),
            ((q, k) -> sum(abs2, grouped(q, k, k)), (sample(16, 4, 32, 2), sample(16, 2, 32, 2))),
        ]
        @compile_workload begin
            for (f, args) in losses, target in (CUDATarget(cuTileGenerator()), CUDATarget(cuTileGenerator(); precision = :tf32))
                program = Tempest.Compiler.trace((xs...) -> Tempest.gradient(f, xs...), args...)
                ex = Tempest.Compiler.prepare(program, args...; target, build = false)
                explain(devnull, ex)
                for g in Tempest.Compiler.group_views(ex), claim in claims(cuTileGenerator(), target, g)
                    workspaces = Dict{Any, Workspace}()
                    for k in schedule_kernels(Context(g, claim.schedule), workspaces)
                        arrays = unbacked_arguments(g, k, workspaces)
                        Tempest.Compiler.content_key("cuTile kernel", source(k), Tuple{map(Core.Typeof, arrays)...}, alias_groups(arrays); checked = true)
                        foreach(a -> flatten!(Any[], a), arrays)
                    end
                end
                # cuTile reads the version of the bytecode it emits from tileiras.
                target.precision === :full && ct.tileiras_available() || continue
                for (g, d) in zip(Tempest.Compiler.group_views(ex), ex.decisions)
                    d !== nothing && d.choice.implementation isa cuTileGenerator || continue
                    workspaces = Dict{Any, Workspace}()
                    for k in schedule_kernels(Context(g, d.choice.claim.schedule), workspaces)
                        arrays = unbacked_arguments(g, k, workspaces)
                        f = Tempest.Compiler.generated(source(k), Workload)
                        Base.invokelatest(f, Tuple{map(Core.Typeof, arrays)...}) do f, tt
                            job = tile_job(f, tt, k.occupancy, alias_groups(arrays), WORKLOAD_ARCH)
                            ct.invoke_frozen(ct.emit_tile, job)
                            job_dependencies(job)
                        end
                    end
                end
            end
        end
    end
    # What runs a compilation on a cache miss, which the workload leaves out
    # since it would write the cache: 0.4 s of a session's first kernel.
    precompile(Tuple{Compilation})
end
