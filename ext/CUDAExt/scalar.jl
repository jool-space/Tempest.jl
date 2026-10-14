function cuda_pointwise_kernel(out, expression, arrays, count)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= count
        @inbounds out[i] = device_eval(expression, arrays, i)
    end
    return
end

# Sums a block's lanes in shared memory, in a fixed order, into lane 1.
@inline function block_sum(scratch, lane, value, width)
    @inbounds scratch[lane] = value
    CUDACore.sync_threads()
    offset = width ÷ 2
    while offset > 0
        if lane <= offset
            @inbounds scratch[lane] += scratch[lane + offset]
        end
        CUDACore.sync_threads()
        offset ÷= 2
    end
    @inbounds scratch[1]
end

# Each block sums a fixed, strided subset of the elements. The block count is
# chosen at build time, so the summation order is the same on every replay.
function cuda_reduction_kernel(work, expression, arrays, count)
    lane = CUDACore.threadIdx().x
    i = (CUDACore.blockIdx().x - 1) * TILE_SIZE + lane
    step = CUDACore.gridDim().x * TILE_SIZE
    total = zero(eltype(work))
    while i <= count
        total += device_eval(expression, arrays, i)
        i += step
    end
    scratch = CUDACore.CuStaticSharedArray(eltype(work), TILE_SIZE)
    total = block_sum(scratch, lane, total, TILE_SIZE)
    if lane == 1
        @inbounds work[CUDACore.blockIdx().x] = total
    end
    return
end

function cuda_partials_kernel(out, work, count)
    lane = CUDACore.threadIdx().x
    total = zero(eltype(out))
    i = lane
    while i <= count
        @inbounds total += work[i]
        i += TILE_SIZE
    end
    scratch = CUDACore.CuStaticSharedArray(eltype(out), TILE_SIZE)
    total = block_sum(scratch, lane, total, TILE_SIZE)
    if lane == 1
        @inbounds out[1] = total
    end
    return
end

function cuda_pointwise!(out, expression, arrays, count)
    if count == 0
        return nothing
    end
    CUDACore.@cuda threads=TILE_SIZE blocks=cld(count, TILE_SIZE) cuda_pointwise_kernel(
        out,
        expression,
        arrays,
        count,
    )
    nothing
end

# Enough blocks to fill the device several times; each lane then sums a few
# elements before the block reduction.
reduction_blocks(count) = min(cld(count, TILE_SIZE), 1024)

function cuda_sum!(out, expression, arrays, count, work)
    if count == 0
        fill!(out, zero(eltype(out)))
        return nothing
    end
    blocks = reduction_blocks(count)
    CUDACore.@cuda threads=TILE_SIZE blocks=blocks cuda_reduction_kernel(
        work,
        expression,
        arrays,
        count,
    )
    # Other sums may have provisioned a longer buffer; never consume its tail.
    CUDACore.@cuda threads=TILE_SIZE blocks=1 cuda_partials_kernel(out, work, blocks)
    nothing
end
