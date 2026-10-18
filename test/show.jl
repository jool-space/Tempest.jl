# What the REPL shows: programs as their nodes, prepared programs and plans in
# a few lines that point to `explain`, and each in one line inside containers.
plain(x) = sprint(show, MIME"text/plain"(), x)

@testset "displays" begin
    p = trace(x -> sum(abs2, x), data(3))
    @test sprint(show, p) == "Program(Float64(3,) → Float64; 3 nodes)"
    @test startswith(plain(p), "Program with 3 nodes: Float64(3,) → Float64\n  %1 = argument")
    many = trace((a, b, c, d) -> (sum(a .* b .* c .* d), a), data(2), data(2), data(2), data(2))
    @test occursin("4 inputs → Float64, Float64(2,)", sprint(show, many))

    ex = prepare(trace(x -> 2 .* x, data(2)), data(2))
    @test sprint(show, ex) == "Executable(Float64(2,) → Float64(2,); CPUTarget(), 1 kernel)"
    @test plain(ex) == """
        Executable on CPUTarget(): Float64(2,) → Float64(2,)
          1 kernel: 1 floor
          explain(ex) shows what each kernel computes and stores, and what runs it"""
    @test occursin("captured: replay! runs the recording", plain(capture!(ex)))
    @test plain(ex.plan) == "Plan: 1 kernel; 1 value stored (16 bytes), 0 recomputed where read\n  explain(plan) shows each kernel"
    @test sprint(show, [ex.plan]) == "Tempest.Plan[Plan(1 kernel)]"
end
