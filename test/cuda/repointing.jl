# A captured program's first call on arrays at other addresses than those it
# was bound to, the first in its process: the code that points its steps and
# its recording at them is compiled by the program's first binding, of the
# types every binding passes, so that the call that first moves an array
# compiles none of it (the 4B's second call of a chunk spent 0.9 s compiling
# it where the call ran in 0.06).
@testset "the first move of a recorded program's arrays compiles no retargeting" begin
    script = """
        using Tempest, CUDACore, cuTile
        const C = Tempest.Compiler
        const X = Base.get_extension(Tempest, :CUDAExt)
        instances(f) = sum(m -> length(collect(Base.specializations(m))), methods(f))
        entries = (C.retarget!, X.learn!, X.patch!, X.record_alone!, C.point_at!)
        f!(s, x) = (s .= 0.5f0 .* s .+ x; sum(s))
        s, x = CUDACore.zeros(Float32, 8), CUDACore.ones(Float32, 8)
        c = compile(f!, s, x)
        c(s, x)
        before = map(instances, entries)
        c(CUDACore.zeros(Float32, 8), x)
        print(sum(map(instances, entries) .- before))
        """
    @test readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`) == "0"
end
