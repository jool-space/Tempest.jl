# The compiler's code is compiled once, whatever the program: only the tracer,
# which runs the user's function, and the runtime, which runs kernels on the
# user's arrays, meet the program's types. The keys and orders that make it
# so agree with Base's, and a program of other numbers of inputs, nodes and
# operands than one prepared before it, with the same functions, ranks and
# extents, compiles none of the compiler's methods again.

@testset "structural keys compare as Base" begin
    # A key is equal to another, and hashes alike, as its parts are.
    parts = Any[(1, (2, 3)), (1, (2, 3)), (1, (2, 3), ()), (1, 2, 3), ((1, 2), 3), (:a, (Int, (1,))), (:a, (Int, (1,)))]
    for a in parts, b in parts
        @test isequal(IR.Key(a), IR.Key(b)) == isequal(a, b)
        isequal(a, b) && @test hash(IR.Key(a)) == hash(IR.Key(b))
    end
end

@testset "definitions carry no type parameter" begin
    # So the planner's methods that take one compile once, whatever function
    # or fold a program applies.
    @test all(isconcretetype, (C.Pointwise, C.Reduction, C.Scan, C.Keyed))
end

@testset "records built by one body convert their integers as the default constructors did" begin
    # A tuple of other integers becomes one of `Int`s, and one of `Int`s is
    # kept as it is: a domain, a group's, a value's dims and the axes each
    # fold of a program's gradient iterates, reduces, scans and keys over.
    narrow(t) = Int32.(t)
    @test IR.Domain(narrow((3, 4))) === IR.Domain((3, 4))
    @test C.Group(C.PointwiseKind(), narrow((2, 3))).domain === (2, 3)
    x, i = ones(Float32, 3, 4), [1, 3, 1]
    graph = C.lower(differentiate(trace((x, i) -> sum(abs2, cumsum(x; dims = 2)[i, :]), x, i)))
    @test all(d -> any(v -> v.definition isa d, graph.values), (C.Reduction, C.Scan, C.Keyed))
    for v in graph.values
        @test C.Value(v.definition, narrow(v.dims), v.eltype, v.scalar) === v
        d = v.definition
        d isa C.Reduction && @test C.Reduction(d.fold, d.operand, narrow(d.iteration), narrow(d.reduced)) === d
        d isa C.Scan && @test C.Scan(d.fold, d.operand, narrow(d.axes), d.reverse, d.exclusive) === d
        d isa C.Keyed && @test C.Keyed(d.fold, d.operand, narrow(d.iteration), d.key, d.into) === d
    end
end

# From Julia 1.12 every method's instances are in one table.
if isdefined(Core, :methodtable)
    @testset "a second program compiles none of the compiler's methods again" begin
        function instances()
            s = Set{Core.MethodInstance}()
            Base.visit(Core.methodtable) do m
                for mi in Base.specializations(m)
                    push!(s, mi)
                end
            end
            s
        end
        # The tracer meets the user's function and the runtime the user's arrays.
        own(m::Method) = (f = string(m.file); occursin("/src/Compiler/", f) && !occursin("/program/tracing.jl", f) && !occursin("/runtime/", f))
        a(x, y) = sum(abs2, sum(tanh.(x .* y) .+ x; dims = 2))
        # Of more inputs than any program the precompile workload prepares, so
        # that no instance for its numbers is in Tempest's image; and of the
        # first's extents, by which the planner chooses: a fold stored where
        # the first's is recomputed is planned by code the first never ran.
        b(x, y, z, w) = sum(abs2, sum(tanh.(x .* y .* z .* w) .+ x .+ y .+ z; dims = 2))
        function run(f, args...)
            p = trace(f, args...)
            for q in (p, differentiate(p))
                execute!(prepare(q, args...))
            end
        end
        run(a, data(5, 6), data(5, 6; scale = 2.0))
        before = instances()
        compiled = Set(mi.def for mi in before if mi.def isa Method && own(mi.def))
        run(b, data(5, 6), data(5, 6; scale = 2.0), data(5, 6; scale = 3.0), data(5, 6; scale = 4.0))
        again = [mi for mi in setdiff(instances(), before) if mi.def in compiled]
        @test isempty(again) || (foreach(mi -> println(stderr, "compiled again: ", mi), again); false)
    end
end
