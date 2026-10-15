# Built-in verbs: runtime-index verbs, dropout and the causal fill. They write
# into planned output storage and capture like other calls.
function support(::CUDAKernels, request::CallRequest)
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    if index_support(request.call.f) || request.call.f in (Tempest.dropout, Tempest.causal_fill)
        all(s -> isbitstype(s.dtype), request.inputs) ||
            return Unsupported("index verbs need bits element types")
        return Supported()
    end
    Unsupported("no CUDA kernel for $(request.call.f)")
end

# Warps per block of the one-warp-per-slice reduction kernels.
const SLICE_WARPS = 8

@inline function warp_reduce(op, value)
    offset = 16
    while offset > 0
        value = op(value, CUDACore.shfl_xor_sync(0xffffffff, value, offset))
        offset ÷= 2
    end
    value
end

function dropout_kernel(out, x, seed, rate, stream)
    i = (CUDACore.blockIdx().x - 1) * INDEX_THREADS + CUDACore.threadIdx().x
    i <= length(out) || return
    @inbounds out[i] = Tempest.dropout_value(x[i], Tempest.hash_uniform(seed[1], stream, i), rate)
    return
end

function dropout_call(out, x, seed, rate, stream)
    kernel = CUDACore.@cuda launch=false dropout_kernel(out, x, seed, rate, stream)
    blocks = cld(length(out), INDEX_THREADS)
    () -> (isempty(out) || kernel(out, x, seed, rate, stream; threads = INDEX_THREADS, blocks); nothing)
end

function causal_fill_kernel(out, x, rows, columns, value)
    i = (CUDACore.blockIdx().x - 1) * INDEX_THREADS + CUDACore.threadIdx().x
    i <= length(out) || return
    n = (i - 1) % rows + 1
    m = ((i - 1) ÷ rows) % columns + 1
    @inbounds out[i] = n > m ? eltype(out)(value) : x[i]
    return
end

function build_implementation(::CUDAKernels, request::CallRequest, runtime, out, inputs)
    if request.call.f === Tempest.causal_fill
        x = only(inputs)
        rows, columns, value = size(x, 1), size(x, 2), request.call.kwargs.value
        kernel = CUDACore.@cuda launch=false causal_fill_kernel(out, x, rows, columns, value)
        blocks = cld(length(out), INDEX_THREADS)
        launch = () -> (isempty(out) || kernel(out, x, rows, columns, value; threads = INDEX_THREADS, blocks); nothing)
        return BuiltImplementation(launch, (out, inputs), 0)
    end
    if request.call.f === Tempest.dropout
        (; rate, stream) = request.call.kwargs
        x, seed = inputs
        return BuiltImplementation(dropout_call(out, x, seed, rate, stream), (out, inputs), 0)
    end
    BuiltImplementation(index_call(request.call.f, out, inputs), (out, inputs), 0)
end

# Runtime-index verbs. Out-of-range indices read as NaN on the device rather
# than faulting; the CPU implementation throws.
const INDEX_THREADS = 256

index_support(f) = f in (
    Tempest.gather_columns,
    Tempest.scatter_columns,
    Tempest.take_rows,
    Tempest.put_rows,
)

function gather_columns_kernel(out, table, indices, rows, columns)
    i = (CUDACore.blockIdx().x - 1) * INDEX_THREADS + CUDACore.threadIdx().x
    i <= length(out) || return
    row = (i - 1) % rows + 1
    column = @inbounds indices[(i - 1) ÷ rows + 1]
    @inbounds out[i] = 1 <= column <= columns ? table[row + (column - 1) * rows] :
        eltype(out)(NaN)
    return
end

function take_rows_kernel(out, x, rows, count)
    n = (CUDACore.blockIdx().x - 1) * INDEX_THREADS + CUDACore.threadIdx().x
    n <= length(out) || return
    row = @inbounds rows[n]
    @inbounds out[n] = 1 <= row <= count ? x[row + (n - 1) * count] : eltype(out)(NaN)
    return
end

function put_rows_kernel(out, values, rows, count)
    i = (CUDACore.blockIdx().x - 1) * INDEX_THREADS + CUDACore.threadIdx().x
    i <= length(out) || return
    row = (i - 1) % count + 1
    n = (i - 1) ÷ count + 1
    @inbounds out[i] = rows[n] == row ? values[n] : zero(eltype(out))
    return
end

# One block per (table column, row tile). The block scans the indices in chunks
# of INDEX_THREADS, compacts the matching positions in increasing order with
# warp ballots, and adds those value columns in that order: repeated indices
# sum deterministically, and the scan costs one step per chunk.
function scatter_columns_kernel(out, values, indices, rows)
    thread = CUDACore.threadIdx().x
    column = CUDACore.blockIdx().x
    row = (CUDACore.blockIdx().y - 1) * INDEX_THREADS + thread
    lane = (thread - 1) % 32
    warp = (thread - 1) ÷ 32 + 1
    matches = CUDACore.CuStaticSharedArray(Int32, INDEX_THREADS)
    offsets = CUDACore.CuStaticSharedArray(Int32, INDEX_THREADS ÷ 32 + 1)
    count = length(indices)
    total = zero(eltype(out))
    base = 0
    while base < count
        n = base + thread
        flag = n <= count && @inbounds(indices[n]) == column
        ballot = CUDACore.vote_ballot_sync(0xffffffff, flag)
        if lane == 0
            @inbounds offsets[warp + 1] = count_ones(ballot)
        end
        CUDACore.sync_threads()
        if thread == 1
            @inbounds offsets[1] = 0
            for w = 2:(INDEX_THREADS ÷ 32 + 1)
                @inbounds offsets[w] += offsets[w - 1]
            end
        end
        CUDACore.sync_threads()
        if flag
            before = count_ones(ballot & ((UInt32(1) << lane) - UInt32(1)))
            @inbounds matches[offsets[warp] + before + 1] = n
        end
        CUDACore.sync_threads()
        found = @inbounds offsets[INDEX_THREADS ÷ 32 + 1]
        if row <= rows
            for k = 1:found
                @inbounds total += values[row + (matches[k] - 1) * rows]
            end
        end
        CUDACore.sync_threads()
        base += INDEX_THREADS
    end
    if row <= rows
        @inbounds out[row + (column - 1) * rows] = total
    end
    return
end

function index_call(f, out, inputs)
    if f === Tempest.gather_columns
        table, indices = inputs
        rows, columns = size(table)
        kernel = CUDACore.@cuda launch=false gather_columns_kernel(out, table, indices, rows, columns)
        blocks = cld(length(out), INDEX_THREADS)
        return () -> (isempty(out) || kernel(out, table, indices, rows, columns; threads = INDEX_THREADS, blocks); nothing)
    elseif f === Tempest.take_rows
        x, rows = inputs
        count = size(x, 1)
        kernel = CUDACore.@cuda launch=false take_rows_kernel(out, x, rows, count)
        blocks = cld(length(out), INDEX_THREADS)
        return () -> (isempty(out) || kernel(out, x, rows, count; threads = INDEX_THREADS, blocks); nothing)
    elseif f === Tempest.put_rows
        values, rows = inputs
        count = size(out, 1)
        kernel = CUDACore.@cuda launch=false put_rows_kernel(out, values, rows, count)
        blocks = cld(length(out), INDEX_THREADS)
        return () -> (isempty(out) || kernel(out, values, rows, count; threads = INDEX_THREADS, blocks); nothing)
    end
    values, indices = inputs
    rows = size(values, 1)
    kernel = CUDACore.@cuda launch=false scatter_columns_kernel(out, values, indices, rows)
    blocks = (size(out, 2), cld(rows, INDEX_THREADS))
    () -> (isempty(out) || kernel(out, values, indices, rows; threads = INDEX_THREADS, blocks); nothing)
end
