# Deterministic axis sums. When the fastest-varying iteration dimension is
# reduced, one block per output reads contiguous elements across its lanes.
# Otherwise lanes cover consecutive outputs, so each warp reads contiguous
# storage, and a second block dimension splits the reduction. Both sum in an
# order fixed at build time.
function cuda_contiguous_sum_kernel(out, expression, arrays, count)
    lane = CUDACore.threadIdx().x
    width = CUDACore.blockDim().x
    o = CUDACore.blockIdx().x
    total = zero(eltype(out))
    r = lane
    while r <= count
        total += device_eval(expression.expression, arrays, 0, coordinates(expression, o, r))
        r += width
    end
    scratch = CUDACore.CuStaticSharedArray(eltype(out), TILE_SIZE)
    total = block_sum(scratch, lane, total, width)
    if lane == 1
        @inbounds out[o] = total
    end
    return
end

# Short contiguous columns: one warp per output, eight outputs per block, and a
# fixed butterfly order across lanes. One 256-thread block per 256-element
# column of attention probabilities reached 65 GB/s.
function cuda_warp_sum_kernel(out, expression, arrays, count)
    lane = (CUDACore.threadIdx().x - 1) % 32 + 1
    o = (CUDACore.blockIdx().x - 1) * SLICE_WARPS + (CUDACore.threadIdx().x - 1) ÷ 32 + 1
    o <= length(out) || return
    total = zero(eltype(out))
    r = lane
    while r <= count
        total += device_eval(expression.expression, arrays, 0, coordinates(expression, o, r))
        r += 32
    end
    total = warp_reduce(+, total)
    if lane == 1
        @inbounds out[o] = total
    end
    return
end

const WARP_SUM_LIMIT = 1024

const OUTPUT_LANES = 32
const REDUCTION_LANES = 32

function cuda_strided_sum_kernel(out, expression, arrays, count)
    x = CUDACore.threadIdx().x
    y = CUDACore.threadIdx().y
    lanes = CUDACore.blockDim().y
    o = (CUDACore.blockIdx().x - 1) * OUTPUT_LANES + x
    total = zero(eltype(out))
    if o <= length(out)
        r = y
        while r <= count
            total += device_eval(expression.expression, arrays, 0, coordinates(expression, o, r))
            r += lanes
        end
    end
    scratch = CUDACore.CuStaticSharedArray(eltype(out), (OUTPUT_LANES, REDUCTION_LANES))
    @inbounds scratch[x, y] = total
    CUDACore.sync_threads()
    offset = lanes ÷ 2
    while offset > 0
        if y <= offset
            @inbounds scratch[x, y] += scratch[x, y + offset]
        end
        CUDACore.sync_threads()
        offset ÷= 2
    end
    if y == 1 && o <= length(out)
        @inbounds out[o] = scratch[x, 1]
    end
    return
end

# A power of two between `low` and `high`, no larger than needed for `count`.
lane_count(count, low, high) = clamp(nextpow(2, max(count, 1)), low, high)

function axis_sum_call(out, expression, arrays, layout, multiprocessors)
    isempty(out) && return () -> nothing
    count = layout.count
    if contiguous_reduction(layout) && count <= WARP_SUM_LIMIT
        kernel = CUDACore.@cuda launch=false cuda_warp_sum_kernel(out, expression, arrays, count)
        blocks = cld(length(out), SLICE_WARPS)
        return () -> begin
            kernel(out, expression, arrays, count; threads = 32SLICE_WARPS, blocks)
            nothing
        end
    end
    if contiguous_reduction(layout)
        threads = lane_count(count, 32, TILE_SIZE)
        kernel = CUDACore.@cuda launch=false cuda_contiguous_sum_kernel(
            out,
            expression,
            arrays,
            count,
        )
        return () -> begin
            kernel(out, expression, arrays, count; threads, blocks = length(out))
            nothing
        end
    end
    blocks = cld(length(out), OUTPUT_LANES)
    # Split the reduction further when there are too few output blocks to fill
    # the device.
    lanes = blocks >= 2multiprocessors ? 8 : REDUCTION_LANES
    lanes = min(lanes, lane_count(count, 1, REDUCTION_LANES))
    kernel = CUDACore.@cuda launch=false cuda_strided_sum_kernel(out, expression, arrays, count)
    () -> begin
        kernel(out, expression, arrays, count; threads = (OUTPUT_LANES, lanes), blocks)
        nothing
    end
end

# Folds other than sums. Each lane folds a fixed, strided subset sequentially,
# then lanes combine in a fixed butterfly order, as for the sum kernels: results
# are deterministic and need no atomics. State components shuffle separately,
# so any tuple of bits values works.
@inline shuffle_state(state, offset) =
    map(value -> CUDACore.shfl_xor_sync(0xffffffff, value, offset), state)

@inline function warp_fold(fold, state)
    offset = 16
    while offset > 0
        state = Tempest.fold_combine(fold, state, shuffle_state(state, offset))
        offset ÷= 2
    end
    state
end

@inline fold_element(fold, state, element) =
    Tempest.fold_combine(fold, state, Tempest.fold_lift(fold, element))

# One warp per output; eight outputs per block. Lanes read consecutive reduction
# positions, which are contiguous when the first reduced axis is the fastest.
function cuda_axis_fold_kernel(out, fold, init, expression, arrays, count)
    lane = (CUDACore.threadIdx().x - 1) % 32 + 1
    o = (CUDACore.blockIdx().x - 1) * SLICE_WARPS + (CUDACore.threadIdx().x - 1) ÷ 32 + 1
    o <= length(out) || return
    state = init
    r = lane
    while r <= count
        element = device_eval(expression.expression, arrays, 0, coordinates(expression, o, r))
        state = fold_element(fold, state, element)
        r += 32
    end
    state = warp_fold(fold, state)
    if lane == 1
        @inbounds out[o] = Tempest.fold_observe(fold, state)
    end
    return
end

# A block's lanes fold a strided subset, warps combine in a butterfly, and the
# first lane combines the warps' states in order.
@inline function block_fold(fold, state, shared)
    lane = (CUDACore.threadIdx().x - 1) % 32 + 1
    warp = (CUDACore.threadIdx().x - 1) ÷ 32 + 1
    state = warp_fold(fold, state)
    if lane == 1
        @inbounds shared[warp] = state
    end
    CUDACore.sync_threads()
    if CUDACore.threadIdx().x == 1
        state = @inbounds shared[1]
        for w = 2:(CUDACore.blockDim().x ÷ 32)
            state = Tempest.fold_combine(fold, state, @inbounds shared[w])
        end
    end
    state
end

function cuda_fold_partials_kernel(work, fold, init, expression, arrays, count)
    i = (CUDACore.blockIdx().x - 1) * TILE_SIZE + CUDACore.threadIdx().x
    step = CUDACore.gridDim().x * TILE_SIZE
    state = init
    while i <= count
        state = fold_element(fold, state, device_eval(expression, arrays, i))
        i += step
    end
    shared = CUDACore.CuStaticSharedArray(typeof(init), TILE_SIZE ÷ 32)
    state = block_fold(fold, state, shared)
    if CUDACore.threadIdx().x == 1
        @inbounds work[CUDACore.blockIdx().x] = state
    end
    return
end

function cuda_fold_finish_kernel(out, fold, init, work, count)
    i = CUDACore.threadIdx().x
    state = init
    while i <= count
        state = Tempest.fold_combine(fold, state, @inbounds work[i])
        i += TILE_SIZE
    end
    shared = CUDACore.CuStaticSharedArray(typeof(init), TILE_SIZE ÷ 32)
    state = block_fold(fold, state, shared)
    if CUDACore.threadIdx().x == 1
        @inbounds out[1] = Tempest.fold_observe(fold, state)
    end
    return
end

function fold_call(out, region::AxisReductionRegion, expression, arrays)
    isempty(out) && return () -> nothing
    fold, init, count = region.fold, region.init, region.layout.count
    kernel = CUDACore.@cuda launch=false cuda_axis_fold_kernel(out, fold, init, expression, arrays, count)
    blocks = cld(length(out), SLICE_WARPS)
    () -> begin
        kernel(out, fold, init, expression, arrays, count; threads = 32SLICE_WARPS, blocks)
        nothing
    end
end

# Two launches: per-block partial states into workspace, then one block
# combines them. The block count is fixed at build time, so the combination
# order is the same on every replay.
function fold_call(out, region::FullReduction, expression, arrays, work)
    fold, init, count = region.fold, region.init, region.count
    # An empty fold still launches one block, which writes the observed identity.
    blocks = reduction_blocks(max(count, 1))
    partial = CUDACore.@cuda launch=false cuda_fold_partials_kernel(
        work, fold, init, expression, arrays, count,
    )
    finish = CUDACore.@cuda launch=false cuda_fold_finish_kernel(out, fold, init, work, blocks)
    () -> begin
        partial(work, fold, init, expression, arrays, count; threads = TILE_SIZE, blocks)
        finish(out, fold, init, work, blocks; threads = TILE_SIZE, blocks = 1)
        nothing
    end
end
