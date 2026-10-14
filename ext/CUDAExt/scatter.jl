# A unique selection can scatter directly after zeroing the parent. Repeated
# indices use source-order segments, so gradients are deterministic on replay.
function scatter_unique_kernel(out, e, arrays, count, selection)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= count
        @inbounds out[device_index(selection, (i - 1) % UInt32)] = device_eval(e, arrays, i)
    end
    return
end

function scatter_segments_kernel(out, e, arrays, offsets, sources)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= length(out)
        total = zero(eltype(out))
        @inbounds for k = (offsets[i] + 1):offsets[i + 1]
            total += device_eval(e, arrays, sources[k])
        end
        @inbounds out[i] = total
    end
    return
end

function scatter_call(r, out, e, arrays, count, selection)
    if selection.unique || count == 0
        selection = gpu_mapping(r, selection)
        return function ()
            fill!(out, zero(eltype(out)))
            if count == 0
                return nothing
            end
            CUDACore.@cuda threads=TILE_SIZE blocks=cld(count, TILE_SIZE) scatter_unique_kernel(
                out,
                e,
                arrays,
                count,
                selection,
            )
            nothing
        end
    end
    offsets, sources = map(CuArray, scatter_schedule(selection, length(out)))
    append!(r.layout_buffers, (offsets, sources))
    return function ()
        if isempty(out)
            return nothing
        end
        CUDACore.@cuda threads=TILE_SIZE blocks=cld(length(out), TILE_SIZE) scatter_segments_kernel(
            out,
            e,
            arrays,
            offsets,
            sources,
        )
        nothing
    end
end
