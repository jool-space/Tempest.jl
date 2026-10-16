# What a plan decided, in words: each kernel, what it computes and reads.

function describe_value(graph, v)
    val = graph.values[v]
    string("%", v, " ", val.eltype, val.scalar ? "" : string(val.dims))
end

function explain(io::IO, p::Plan)
    graph = p.graph
    stored = count(id -> p.stored[id] && !(graph.values[id].definition isa Union{Input, ConstantArray}), eachindex(graph.values))
    recomputed = count(id -> !p.stored[id], eachindex(graph.values))
    println(io, length(p.groups), " kernels; ", stored, " values stored, ", recomputed, " recomputed where read")
    for (k, g) in enumerate(p.groups)
        reads = sort!(collect(group_reads(graph, g)))
        print(io, "kernel ", k, ": ", g.kind, " over ", g.domain)
        isempty(g.reduced) || print(io, ", reducing axes ", g.reduced)
        println(io, isempty(reads) ? "" : string(", reads ", join(map(r -> "%$r", reads), " ")))
        for r in sort(g.reductions; by = r -> (g.phase[r], r))
            println(io, "  phase ", g.phase[r], ": ", describe_value(graph, r), " = ",
                fold_name(graph.values[r].definition.fold), " of ", g.terms[r])
        end
        for m in g.outputs
            println(io, "  writes ", describe_value(graph, m), haskey(g.terms, m) ? string(" = ", g.terms[m]) : "")
        end
        for r in g.cross
            d = graph.values[r].definition
            println(io, "  across axes ", d.reduced, ": ", describe_value(graph, r), " = ",
                fold_name(d.fold), " of ", g.terms[r])
        end
    end
end

explain(p::Plan) = explain(stdout, p)
