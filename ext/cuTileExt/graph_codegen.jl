# The emitted code uses two-dimensional tiles in logical array coordinates.
# Singleton reads retain singleton tile axes and broadcast through Julia calls.
function graph_binding_table(bindings)
    Dict(
        binding.parameters[1] => (i, binding) for (i, binding) in enumerate(bindings.parameters)
    )
end

# Generation-time view of a fold node: its fold type, identity state and
# whether it has a resident lowering. Sums keep their original expressions.
# Other folds map their scalar functions over tile lanes, one state component
# at a time, and pass a zero-size combiner to cuTile's tuple `mapreduce`.
function fold_binding(bindings, id)
    _, binding = bindings[id]
    F = binding.parameters[2].parameters[1]
    (; F, init = binding.parameters[6], resident = binding.parameters[7], sum = F === SumFold)
end

fold_accumulators(id, fold) =
    fold.sum ? [Symbol(:accumulator_, id)] : [Symbol(:accumulator_, id, :_, k) for k in eachindex(fold.init)]

function fold_initial(fold, accumulators, extent)
    fold.sum && return [:($(only(accumulators)) = zeros(Float32, $extent))]
    [:($a = fill($(fold.init[k]), $extent)) for (k, a) in enumerate(accumulators)]
end

reduced_extent(extent, dims) = ntuple(i -> i in dims ? 1 : extent[i], length(extent))

# A scalar callable over same-extent lanes. Operands are flattened first, as for
# pointwise callables: cuTile 1.0 can reverse multi-axis shapes otherwise. A
# result independent of its lanes, such as the `one(x)` of a logsumexp lift,
# comes back as a scalar and is broadcast again.
function lane_map(callable, operands, extent)
    count = prod(extent)
    flattened = map(operands) do operand
        :(reshape(ct.broadcast_to(ct.Tile($operand), $extent), ($count,)))
    end
    :(reshape(ct.broadcast_to(ct.Tile($callable.($(flattened...))), ($count,)), $extent))
end

# The lifted state of a tile of contributions. f(0) need not be zero or finite:
# padded lanes take the identity state after evaluating the contribution.
function fold_masked_state!(statements, fold, contribution, mask, extent)
    fold.sum && return [:(ifelse.($mask, $contribution, 0.0f0))]
    map(eachindex(fold.init)) do k
        name = gensym(:state)
        lifted = lane_map(:(Tempest.FoldLift{$k, $(fold.F)}()), [contribution], extent)
        push!(statements, :($name = ifelse.($mask, $lifted, $(fold.init[k]))))
        name
    end
end

# Combine states into the accumulators. Every component reads the old
# accumulators, so the new ones are bound only after all are computed.
function fold_merge!(statements, fold, accumulators, states, extent)
    width = length(accumulators)
    merged = map(1:width) do k
        name = gensym(:merged)
        merge = :(Tempest.FoldMerge{$k, $width, $(fold.F)}())
        push!(statements, :($name = $(lane_map(merge, [accumulators; states], extent))))
        name
    end
    for (accumulator, name) in zip(accumulators, merged)
        push!(statements, :($accumulator = $name))
    end
end

function fold_accumulate!(statements, fold, accumulators, contribution, mask, extent)
    state = fold_masked_state!(statements, fold, contribution, mask, extent)
    if fold.sum
        accumulator = only(accumulators)
        push!(statements, :($accumulator = $accumulator .+ $(only(state))))
    else
        fold_merge!(statements, fold, accumulators, state, extent)
    end
end

# The unobserved state reduced over `dims`: a tuple of tiles with singleton
# reduced axes. cuTile's reduction body receives scalar state tuples.
function fold_reduced(fold, states, dims)
    if length(states) == 1
        :((reduce(
            Tempest.FoldOperator{$(fold.F)}(),
            $(only(states));
            dims = $dims,
            init = $(only(fold.init)),
        ),))
    else
        :(mapreduce(
            identity,
            Tempest.FoldCombiner{$(fold.F)}(),
            $(states...);
            dims = $dims,
            init = $(fold.init),
        ))
    end
end

fold_observed(fold, reduced, extent) = lane_map(
    :(Tempest.FoldObserve{$(fold.F)}()),
    [:($reduced[$k]) for k in eachindex(fold.init)],
    extent,
)

# The completed value over `dims` of states with the given extent.
function fold_complete!(statements, fold, states, dims, extent)
    fold.sum && return :(sum($(only(states)); dims = $dims))
    reduced = gensym(:reduced)
    push!(statements, :($reduced = $(fold_reduced(fold, states, dims))))
    fold_observed(fold, reduced, reduced_extent(extent, dims))
end

# A fold with a resident lowering reduces a slice held in registers directly.
function fold_resident!(statements, fold, contribution, mask, dims, extent)
    values = :(ct.broadcast_to(ct.Tile($contribution), $extent))
    :(Tempest.fold_resident($(fold.F).instance, $values, $mask, $dims))
end

# Partials columns of each fold in a phase, in group order.
function fold_columns(bindings, group)
    columns = UnitRange{Int}[]
    next = 1
    for id in group
        width = fold_binding(bindings, id).sum ? 1 : length(fold_binding(bindings, id).init)
        push!(columns, next:(next + width - 1))
        next += width
    end
    columns
end

function emit_graph_value!(
    statements,
    cache,
    bindings,
    arguments,
    id,
    tile;
    statistics = (),
    statistic_axis = 0,
    allow_tma = nothing,
)
    haskey(cache, id) && return cache[id]
    position, binding = bindings[id]
    _, operation, dependencies, shape, dtype = binding.parameters
    if operation <: FoldReduction && isempty(statistics)
        return Symbol(:completed_, id)
    end
    name = gensym(:value)
    expression = if operation <: Argument
        input = findfirst(==(id), arguments)
        rows, columns = matrix_shape(shape)
        row_index = rows == 1 ? :(Int32(1)) : :block_row
        column_index = columns == 1 ? :(Int32(1)) : :block_column
        extent = (rows == 1 ? 1 : tile[1], columns == 1 ? 1 : tile[2])
        array = Symbol(:input_, input)
        :(ct.load(
            $array,
            ($row_index, $column_index),
            $extent;
            padding_mode = ct.PaddingMode.Zero,
            allow_tma = $allow_tma,
        ))
    elseif operation <: Constant
        :(program.bindings[$position].op.value)
    elseif operation <: Pointwise
        operands = map(dependencies) do dependency
            emit_graph_value!(
                statements,
                cache,
                bindings,
                arguments,
                dependency,
                tile;
                statistics,
                statistic_axis,
                allow_tma,
            )
        end
        extent = ntuple(i -> matrix_shape(shape)[i] == 1 ? 1 : tile[i], 2)
        # cuTile 1.0 scalar-call broadcasting reverses some multi-axis tile
        # shapes when the callable contains scalar constants. Expand Julia's
        # singleton axes first, then use its working vector broadcast path.
        flattened = map(operands) do operand
            :(reshape(ct.broadcast_to(ct.Tile($operand), $extent), ($(prod(extent)),)))
        end
        :(reshape($dtype.(program.bindings[$position].op.f.($(flattened...))), $extent))
    elseif operation <: FoldReduction
        slot = findfirst(==(id), statistics)
        if statistic_axis == 0
            :(reshape(ct.load(statistics_buffer, (Int32($slot),), (1,)), (1, 1)))
        elseif statistic_axis == 1
            :(reshape(ct.load(statistics_buffer, (block_column, Int32($slot)), ($(tile[2]), 1)), (1, $(tile[2]))))
        else
            :(ct.load(statistics_buffer, (block_row, Int32($slot)), ($(tile[1]), 1)))
        end
    else
        error("unchecked operation in cuTile graph schedule: $operation")
    end
    push!(statements, :($name = $expression))
    cache[id] = name
    name
end

function graph_valid_lanes(tile)
    rows, columns = tile
    quote
        remaining_rows = rows - (block_row - Int32(1)) * Int32($rows)
        remaining_columns = columns - (block_column - Int32(1)) * Int32($columns)
        valid_rows = reshape(ct.arange($rows), ($rows, 1)) .<= remaining_rows
        valid_columns = reshape(ct.arange($columns), (1, $columns)) .<= remaining_columns
        valid = valid_rows .& valid_columns
    end
end

function graph_reduction_loop(body, axis, tile)
    extent = axis == 1 ? :rows : :columns
    block = axis == 1 ? :block_row : :block_column
    quote
        for $block = Int32(1):cld($extent, Int32($(tile[axis])))
            $(body...)
        end
    end
end

function emit_graph_sweep!(statements, bindings, arguments, group, axis, tile)
    body = Expr[graph_valid_lanes(tile)]
    cache = Dict{Int, Symbol}()
    for id in group
        fold = fold_binding(bindings, id)
        accumulators = fold_accumulators(id, fold)
        append!(statements, fold_initial(fold, accumulators, tile))
        source = only(bindings[id][2].parameters[3])
        contribution = emit_graph_value!(body, cache, bindings, arguments, source, tile)
        fold_accumulate!(body, fold, accumulators, contribution, :valid, tile)
    end
    push!(statements, graph_reduction_loop(body, axis, tile))
    for id in group
        fold = fold_binding(bindings, id)
        completed = Symbol(:completed_, id)
        value = fold_complete!(statements, fold, fold_accumulators(id, fold), axis, tile)
        push!(statements, :($completed = $value))
    end
end

function emit_graph_output!(
    body,
    cache,
    bindings,
    arguments,
    id,
    position,
    tile;
    statistics = (),
    statistic_axis = 0,
)
    shape = matrix_shape(bindings[id][2].parameters[4])
    value = emit_graph_value!(body, cache, bindings, arguments, id, tile; statistics, statistic_axis)
    destination = Symbol(:output_, position)
    extent = (shape[1] == 1 ? 1 : tile[1], shape[2] == 1 ? 1 : tile[2])
    # Broadcast-invariant outputs have exactly one writer even when the launch
    # grid covers many tiles along an axis they do not carry.
    condition = :($(shape[1] != 1) || block_row == Int32(1))
    condition = :($condition && ($(shape[2] != 1) || block_column == Int32(1)))
    push!(
        body,
        quote
            if $condition
                ct.store(
                    $destination,
                    (block_row, block_column),
                    ct.broadcast_to(ct.Tile($value), $extent),
                )
            end
        end,
    )
end

function emit_axis_outputs!(statements, bindings, arguments, outputs, axis, tile)
    invariant_body, element_body = Expr[], Expr[]
    invariant_cache, element_cache = Dict{Int, Symbol}(), Dict{Int, Symbol}()
    for (position, id) in enumerate(outputs)
        invariant = matrix_shape(bindings[id][2].parameters[4])[axis] == 1
        body = invariant ? invariant_body : element_body
        cache = invariant ? invariant_cache : element_cache
        emit_graph_output!(body, cache, bindings, arguments, id, position, tile)
    end
    block = axis == 1 ? :block_row : :block_column
    push!(statements, :($block = Int32(1)))
    append!(statements, invariant_body)
    isempty(element_body) || push!(statements, graph_reduction_loop(element_body, axis, tile))
end

function graph_array_bindings(outputs, inputs; offset = 0)
    statements = Expr[]
    for position in eachindex(outputs)
        name = Symbol(:output_, position)
        push!(statements, :($name = arrays[$(offset + position)]))
    end
    for position in eachindex(inputs)
        name = Symbol(:input_, position)
        push!(statements, :($name = arrays[$(offset + length(outputs) + position)]))
    end
    statements
end

# A resident column is one tile. Values are one-dimensional: element values
# span the tile, column values are length-one tiles, and completed sums are
# scalars, so pointwise calls broadcast directly without materialized expansion.
function emit_column_value!(statements, cache, bindings, arguments, id, tile)
    haskey(cache, id) && return cache[id]
    position, binding = bindings[id]
    _, operation, dependencies, shape, dtype = binding.parameters
    operation <: FoldReduction && return Symbol(:completed_, id)
    name = gensym(:value)
    expression = if operation <: Argument
        input = findfirst(==(id), arguments)
        rows = matrix_shape(shape)[1]
        extent = rows == 1 ? 1 : tile
        column_index = matrix_shape(shape)[2] == 1 ? :(Int32(1)) : :block_column
        array = Symbol(:input_, input)
        :(reshape(ct.load($array, (Int32(1), $column_index), ($extent, 1)), ($extent,)))
    elseif operation <: Constant
        :(program.bindings[$position].op.value)
    elseif operation <: Pointwise
        operands = map(dependencies) do dependency
            emit_column_value!(statements, cache, bindings, arguments, dependency, tile)
        end
        :($dtype.(program.bindings[$position].op.f.($(operands...))))
    else
        error("unchecked operation in resident cuTile column schedule: $operation")
    end
    push!(statements, :($name = $expression))
    cache[id] = name
    name
end

# Every value is emitted once into a shared cache, so later reduction phases
# and the outputs reuse it instead of reloading or recomputing.
function emit_resident_column!(statements, bindings, arguments, groups, outputs, tile)
    push!(statements, :(valid = ct.arange($tile) .<= rows))
    cache = Dict{Int, Symbol}()
    for group in groups, id in group
        source = only(bindings[id][2].parameters[3])
        contribution = emit_column_value!(statements, cache, bindings, arguments, source, tile)
        completed = Symbol(:completed_, id)
        fold = fold_binding(bindings, id)
        # Sums complete to a scalar; other folds keep a length-one tile.
        value = if fold.sum
            :(sum(ifelse.(valid, $contribution, 0.0f0)))
        elseif fold.resident
            fold_resident!(statements, fold, contribution, :valid, 1, (tile,))
        else
            state = fold_masked_state!(statements, fold, contribution, :valid, (tile,))
            fold_complete!(statements, fold, state, 1, (tile,))
        end
        push!(statements, :($completed = $value))
    end
    # Compute every escaping value before the first store. Interleaving stores
    # with computation changed cuTile's register allocation substantially.
    values = map(id -> emit_column_value!(statements, cache, bindings, arguments, id, tile), outputs)
    for (position, (id, value)) in enumerate(zip(outputs, values))
        destination = Symbol(:output_, position)
        stored = if matrix_shape(bindings[id][2].parameters[4])[1] == 1
            :(reshape(ct.Tile($value), (1, 1)))
        else
            :(reshape(ct.broadcast_to(ct.Tile($value), ($tile,)), ($tile, 1)))
        end
        push!(statements, :(ct.store($destination, (Int32(1), block_column), $stored)))
    end
end

# Several resident columns per block: two-dimensional values with one shared
# cache, column statistics as (1, columns) tiles, and every escaping value
# computed before the first store, as for a single resident column.
function emit_resident_tile!(statements, bindings, arguments, groups, outputs, tile)
    push!(statements, :(block_row = Int32(1)))
    push!(statements, graph_valid_lanes(tile))
    cache = Dict{Int, Symbol}()
    for group in groups, id in group
        source = only(bindings[id][2].parameters[3])
        contribution = emit_graph_value!(statements, cache, bindings, arguments, source, tile)
        completed = Symbol(:completed_, id)
        fold = fold_binding(bindings, id)
        value = if fold.resident
            fold_resident!(statements, fold, contribution, :valid, 1, tile)
        else
            state = fold_masked_state!(statements, fold, contribution, :valid, tile)
            fold_complete!(statements, fold, state, 1, tile)
        end
        push!(statements, :($completed = $value))
    end
    for id in outputs
        emit_graph_value!(statements, cache, bindings, arguments, id, tile)
    end
    for (position, id) in enumerate(outputs)
        emit_graph_output!(statements, cache, bindings, arguments, id, position, tile)
    end
end

@inline @generated function graph_axis_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    rows::Int32,
    columns::Int32,
    phase,
    arrays::Vararg{Any, Count},
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Count}
    bindings = graph_binding_table(Bindings)
    statements = graph_array_bindings(Outputs, Inputs)
    axis = only(Axes)
    fixed = axis == 1 ? :block_column : :block_row
    push!(statements, :($fixed = ct.bid(1)))
    if Resident && Tile[2] == 1
        emit_resident_column!(statements, bindings, Inputs, Groups, Outputs, Tile[1])
    elseif Resident
        emit_resident_tile!(statements, bindings, Inputs, Groups, Outputs, Tile)
    else
        for group in Groups
            emit_graph_sweep!(statements, bindings, Inputs, group, axis, Tile)
        end
        emit_axis_outputs!(statements, bindings, Inputs, Outputs, axis, Tile)
    end
    quote
        $(statements...)
        return
    end
end

@inline @generated function graph_partial_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    rows::Int32,
    columns::Int32,
    ::Val{Phase},
    arrays::Vararg{Any, Count},
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Phase, Count}
    bindings = graph_binding_table(Bindings)
    statistics = Tuple(id for group in Groups for id in group)
    statements = graph_array_bindings((), Inputs; offset = 2)
    cache = Dict{Int, Symbol}()
    columns = fold_columns(bindings, Groups[Phase])
    for (position, id) in enumerate(Groups[Phase])
        source = only(bindings[id][2].parameters[3])
        contribution =
            emit_graph_value!(statements, cache, bindings, Inputs, source, Tile; statistics)
        fold = fold_binding(bindings, id)
        if fold.sum
            push!(
                statements,
                quote
                    value = sum(ifelse.(valid, $contribution, 0.0f0))
                    ct.store(
                        partials,
                        (linear_block, Int32($(only(columns[position])))),
                        reshape(ct.Tile(value), (1, 1)),
                    )
                end,
            )
            continue
        end
        # Each block stores its partial state, one column per component.
        state = fold_masked_state!(statements, fold, contribution, :valid, Tile)
        reduced = gensym(:reduced)
        push!(statements, :($reduced = $(fold_reduced(fold, state, (1, 2)))))
        for (k, column) in enumerate(columns[position])
            push!(statements, :(ct.store(partials, (linear_block, Int32($column)), $reduced[$k])))
        end
    end
    quote
        partials, statistics_buffer = arrays[1], arrays[2]
        block_row, block_column = ct.bid(1), ct.bid(2)
        linear_block = block_row + (block_column - Int32(1)) * cld(rows, Int32($(Tile[1])))
        $(graph_valid_lanes(Tile))
        $(statements...)
        return
    end
end

@inline @generated function graph_output_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    rows::Int32,
    columns::Int32,
    phase,
    arrays::Vararg{Any, Count},
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Count}
    bindings = graph_binding_table(Bindings)
    statistics = Tuple(id for group in Groups for id in group)
    statements = graph_array_bindings(Outputs, Inputs; offset = 1)
    cache = Dict{Int, Symbol}()
    statistic_axis = Splits > 1 ? only(Axes) : 0
    for (position, id) in enumerate(Outputs)
        emit_graph_output!(
            statements,
            cache,
            bindings,
            Inputs,
            id,
            position,
            Tile;
            statistics,
            statistic_axis,
        )
    end
    quote
        statistics_buffer = arrays[1]
        block_row, block_column = ct.bid(1), ct.bid(2)
        $(statements...)
        return
    end
end

function finish_graph_reductions(statistics, partials, count::Int32, offset::Int32)
    accumulator = zeros(Float32, 256)
    for block = Int32(1):cld(count, Int32(256))
        value = reshape(
            ct.load(partials, (block, ct.bid(1)), (256, 1); padding_mode = ct.PaddingMode.Zero),
            (256,),
        )
        accumulator = accumulator .+ value
    end
    statistics[offset + ct.bid(1)] = sum(accumulator)
    return
end

# Completes one phase containing folds other than sums: block `position`
# combines its statistic's partial states and stores the observed value.
@inline @generated function graph_finish_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    count::Int32,
    offset::Int32,
    ::Val{Phase},
    statistics,
    partials,
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Phase}
    bindings = graph_binding_table(Bindings)
    columns = fold_columns(bindings, Groups[Phase])
    branches = map(enumerate(Groups[Phase])) do (position, id)
        # Sums sharing a phase with other folds take the general path here.
        fold = (; fold_binding(bindings, id)..., sum = false)
        accumulators = [gensym(:accumulator) for _ in columns[position]]
        initial = fold_initial(fold, accumulators, 256)
        body = Expr[:(valid = (block - Int32(1)) * Int32(256) .+ ct.arange(256) .<= count)]
        loaded = map(columns[position]) do column
            name = gensym(:partial)
            push!(
                body,
                :($name = reshape(
                    ct.load(partials, (block, Int32($column)), (256, 1); padding_mode = ct.PaddingMode.Zero),
                    (256,),
                )),
            )
            name
        end
        # Loaded partials are already lifted states: mask and combine them.
        masked = map(enumerate(loaded)) do (k, value)
            name = gensym(:masked)
            push!(body, :($name = ifelse.(valid, $value, $(fold.init[k]))))
            name
        end
        fold_merge!(body, fold, accumulators, masked, (256,))
        completion = Expr[]
        complete = fold_complete!(completion, fold, accumulators, 1, (256,))
        quote
            if ct.bid(1) == Int32($position)
                $(initial...)
                for block = Int32(1):cld(count, Int32(256))
                    $(body...)
                end
                $(completion...)
                ct.store(statistics, (offset + Int32($position),), $complete)
            end
        end
    end
    quote
        $(branches...)
        return
    end
end

# One block sums a chunk of the reduced axis for one kept tile. Statistics of
# earlier phases are per kept index; the partial sums of this phase go to
# column (position - 1) * Splits + split of the partials matrix.
@inline @generated function graph_split_partial_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    rows::Int32,
    columns::Int32,
    ::Val{Phase},
    arrays::Vararg{Any, Count},
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Phase, Count}
    bindings = graph_binding_table(Bindings)
    axis = only(Axes)
    statistics = Tuple(id for group in Groups for id in group)
    kept_block, reduced_block = axis == 1 ? (:block_column, :block_row) : (:block_row, :block_column)
    extent = axis == 1 ? :rows : :columns
    kept_tile = Tile[3 - axis]
    body = Expr[graph_valid_lanes(Tile)]
    cache = Dict{Int, Symbol}()
    setup, finish = Expr[], Expr[]
    columns = fold_columns(bindings, Groups[Phase])
    for (position, id) in enumerate(Groups[Phase])
        fold = fold_binding(bindings, id)
        accumulators = fold_accumulators(id, fold)
        append!(setup, fold_initial(fold, accumulators, Tile))
        source = only(bindings[id][2].parameters[3])
        contribution = emit_graph_value!(
            body,
            cache,
            bindings,
            Inputs,
            source,
            Tile;
            statistics,
            statistic_axis = axis,
            # tileiras stages evenly divisible sweeps through TMA and shared
            # memory; for these short chunks that was slower (8192 x 32: 36%
            # of copy bandwidth with TMA, 45% without). Resident tiles keep it.
            allow_tma = false,
        )
        fold_accumulate!(body, fold, accumulators, contribution, :valid, Tile)
        if fold.sum
            push!(
                finish,
                :(ct.store(
                    partials,
                    ($kept_block, Int32($((only(columns[position]) - 1) * Splits)) + split),
                    reshape(sum($(only(accumulators)); dims = $axis), ($kept_tile, 1)),
                )),
            )
            continue
        end
        # The chunk's state, reduced over the swept axis but not yet observed.
        reduced = gensym(:reduced)
        push!(finish, :($reduced = $(fold_reduced(fold, accumulators, axis))))
        for (k, column) in enumerate(columns[position])
            push!(
                finish,
                :(ct.store(
                    partials,
                    ($kept_block, Int32($((column - 1) * Splits)) + split),
                    reshape($reduced[$k], ($kept_tile, 1)),
                )),
            )
        end
    end
    quote
        partials, statistics_buffer = arrays[1], arrays[2]
        $(graph_array_bindings((), Inputs; offset = 2)...)
        $kept_block = ct.bid(1)
        split = ct.bid(2)
        chunk = cld(cld($extent, Int32($(Tile[axis]))), Int32($Splits))
        $(setup...)
        for $reduced_block = ((split - Int32(1)) * chunk + Int32(1)):min(split * chunk, cld($extent, Int32($(Tile[axis]))))
            $(body...)
        end
        $(finish...)
        return
    end
end

# Completes one phase: sums each statistic's Splits partial columns.
function finish_split_reductions(
    statistics,
    partials,
    offset::Int32,
    ::Val{Rows},
    ::Val{Splits},
) where {Rows, Splits}
    block, position = ct.bid(1), ct.bid(2)
    partial = ct.load(partials, (block, position), (Rows, Splits))
    ct.store(statistics, (block, offset + position), sum(partial; dims = 2))
    return
end

# Completes one split phase containing folds other than sums: block column
# `position` combines its statistic's Splits partial states per kept index.
@inline @generated function graph_split_finish_kernel(
    program::GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits},
    offset::Int32,
    ::Val{Phase},
    ::Val{Rows},
    statistics,
    partials,
) where {Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits, Phase, Rows}
    bindings = graph_binding_table(Bindings)
    columns = fold_columns(bindings, Groups[Phase])
    branches = map(enumerate(Groups[Phase])) do (position, id)
        fold = (; fold_binding(bindings, id)..., sum = false)
        statements = Expr[]
        loaded = map(columns[position]) do column
            name = gensym(:partial)
            push!(statements, :($name = ct.load(partials, (block, Int32($column)), ($Rows, $Splits))))
            name
        end
        complete = fold_complete!(statements, fold, loaded, 2, (Rows, Splits))
        quote
            if position == Int32($position)
                $(statements...)
                ct.store(statistics, (block, offset + position), $complete)
            end
        end
    end
    quote
        block, position = ct.bid(1), ct.bid(2)
        $(branches...)
        return
    end
end
