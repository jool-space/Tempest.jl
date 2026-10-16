# Work the generator does while it precompiles, without a GPU: claiming the
# groups of small programs, emitting their kernels and naming them for the
# kernel cache. Loading and launching need a device, so they compile in the
# session.

using PrecompileTools: @setup_workload, @compile_workload

# A tile array over no memory, enough to name a kernel.
unbacked_tile_array(T, spec) = ct.TileArray(Ptr{T}(256), Int32.(spec.sizes), Int32.(spec.strides))

@setup_workload begin
    sample(dims...) = Float32.(reshape(sin.(1:prod(dims)), dims))
    gelu(x) = x / (1 + exp(-1.702f0 * x))
    rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
    x, w, b, v, y = sample(64, 48), sample(32, 64), sample(32), sample(64), sample(48, 4096)
    losses = [
        ((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (w, x, b)),
        ((x, v) -> sum(abs2, rmsnorm(x, v)), (x, v)),
        (x -> sum(exp.(x .- Tempest.logsumexp(x; dims = 1)) .* x), (x,)),
        (y -> sum(exp.(y .- Tempest.logsumexp(y; dims = 2))) + sum(maximum(y; dims = 1)), (y,)),
    ]
    @compile_workload begin
        for (f, args) in losses, target in (CUDATarget(), CUDATarget(precision = :tf32))
            p = Tempest.plan(Tempest.lower(differentiate(trace(f, args...))))
            for g in p.groups, claim in Tempest.claims(Tempest.cuTileGenerator(), target, p, g)
                workspaces = Dict{Any, Workspace}()
                for k in schedule_kernels(Context(p.graph, p.stored, g, claim.schedule), workspaces)
                    arrays = map(k.args) do spec
                        T = spec.buffer isa Int ? p.graph.values[spec.buffer].eltype : workspaces[spec.buffer].eltype
                        unbacked_tile_array(T, spec)
                    end
                    Tempest.content_key("cuTile kernel", source(k), Tuple{map(Core.Typeof, arrays)...}, alias_groups(arrays))
                    foreach(a -> flatten!(Any[], a), arrays)
                end
            end
        end
    end
end
