# Kernel entry points generated from a GraphSweepProgram: one per phase kind
# (axis sweep, partial reductions, outputs, finishing and split reductions).
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
