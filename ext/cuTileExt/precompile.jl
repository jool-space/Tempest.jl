# Work the generator does while it precompiles, without a GPU: preparing
# small programs for CUDA without building them, which plans, claims,
# estimates and decides; then emitting their kernels and naming them for the
# kernel cache. Loading and launching need a device, so they compile in the
# session.

using PrecompileTools: @setup_workload, @compile_workload

# A tile array over no memory, enough to name a kernel.
unbacked_tile_array(T, spec) = ct.TileArray(Ptr{T}(256), Int32.(spec.sizes), Int32.(spec.strides))

@setup_workload begin
    sample(dims...) = Float32.(reshape(sin.(1:prod(dims)), dims))
    gelu(x) = x / (1 + exp(-1.702f0 * x))
    rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
    # Causal attention over (features, keys or queries, heads, batch), written out.
    function attention(q, k, v)
        s = Tempest.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 3, 4))
        s = ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32)
        Tempest.contract(v, exp.(s .- Tempest.logsumexp(s; dims = 1)), (1, 2, 3, 4), (2, 5, 3, 4), (1, 5, 3, 4))
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
    ]
    @compile_workload begin
        # Registered for this process alone; the package registers it when it loads.
        register!(ct, cuTileGenerator())
        for (f, args) in losses, target in (CUDATarget(), CUDATarget(precision = :tf32))
            program = Tempest.Compiler.differentiate(Tempest.Compiler.trace(f, args...))
            ex = Tempest.Compiler.prepare(program, args...; target, build = false)
            explain(devnull, ex)
            p = ex.plan
            for g in map(g -> Tempest.Compiler.group_view(p, g), p.groups), claim in claims(cuTileGenerator(), target, g)
                workspaces = Dict{Any, Workspace}()
                for k in schedule_kernels(Context(g, claim.schedule), workspaces)
                    arrays = map(k.args) do spec
                        T = spec.buffer isa Int ? g.values[spec.buffer].eltype : workspaces[spec.buffer].eltype
                        unbacked_tile_array(T, spec)
                    end
                    Tempest.Compiler.content_key("cuTile kernel", source(k), Tuple{map(Core.Typeof, arrays)...}, alias_groups(arrays))
                    foreach(a -> flatten!(Any[], a), arrays)
                end
            end
        end
        register!(ct)
    end
end
