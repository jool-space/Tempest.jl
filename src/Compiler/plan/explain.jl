# What a plan decided, in words: each kernel, what it computes and reads.

public explain

function describe_value(graph, v)
    val = graph.values[v]
    string("%", v, " ", val.eltype, val.scalar ? "" : string(val.dims))
end

"""
    Tempest.Compiler.structure(plan) -> (; kernels, stored, bytes)

What tests compare between plans: the number of kernels, the values written
to storage (inputs and constants apart), and their bytes.
"""
function structure(p::Plan)
    graph = p.graph
    stored = [id for id in eachindex(graph.values) if p.stored[id] && !(graph.values[id].definition isa Union{Input, ConstantArray})]
    (kernels = length(p.groups), stored, bytes = sum(v -> value_bytes(graph.values[v]), stored; init = 0))
end

"""
    explain([io], c::Compiled)
    explain([io], ex::Executable)
    explain([io], plan)

Print what Tempest decided for a program. For each kernel: what it computes,
what it reads, and which values it stores or leaves to be recomputed where
they are read. For a prepared program, also the precision of its products,
the close calls measured between implementations, how values share storage,
and which implementation runs each kernel, at what estimated or measured time.

It describes how Tempest keeps its promises today; what it prints may change.
"""
function explain end

# The plan's size, in a line: kernels, and values stored and recomputed.
function plan_summary(p::Plan)
    s = structure(p)
    recomputed = count(id -> !p.stored[id], eachindex(p.graph.values))
    string(counted(s.kernels, "kernel"), "; ", counted(length(s.stored), "value"), " stored (",
        Base.format_bytes(s.bytes), "), ", recomputed, " recomputed where read")
end

Base.show(io::IO, p::Plan) = print(io, "Plan(", counted(length(p.groups), "kernel"), ")")

function Base.show(io::IO, ::MIME"text/plain", p::Plan)
    println(io, "Plan: ", plan_summary(p))
    print(io, "  explain(plan) shows each kernel")
end

function explain(io::IO, p::Plan)
    graph = p.graph
    println(io, plan_summary(p))
    for (v, note) in graph.rewrites
        println(io, "rewritten: ", describe_value(graph, v), ", ", note)
    end
    for (k, g) in enumerate(p.groups)
        reads = sort!(collect(group_reads(graph, g)))
        print(io, "kernel ", k, ": ", g.kind, " over ", g.domain)
        println(io, isempty(reads) ? "" : string(", reads ", join(map(r -> "%$r", reads), " ")))
        for r in sort(g.reductions; by = r -> (g.phase[r], r))
            println(io, "  phase ", g.phase[r], ": ", describe_value(graph, r), " = ",
                fold_name(graph.values[r].definition.fold), " over axes ", fold_axes(graph, r), " of ", g.terms[r])
        end
        for m in g.outputs
            p.stored[m] || continue
            println(io, "  writes ", describe_value(graph, m), haskey(g.terms, m) ? string(" = ", g.terms[m]) : "")
        end
    end
end

explain(p::Plan) = explain(stdout, p)
