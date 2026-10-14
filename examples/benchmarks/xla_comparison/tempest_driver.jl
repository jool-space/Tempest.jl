# The GPT example for run.sh; COMPOSED=1 turns off the fused attention provider.
using Tempest
import cuTile
include(joinpath(@__DIR__, "..", "..", "gpt.jl"))
if get(ENV, "COMPOSED", "0") == "1"
    ext = Base.get_extension(Tempest, :cuTileExt)
    @eval ext support(::CuTileKernels, request::Tempest.CallRequest) = Tempest.Unsupported("disabled for profiling")
    println("fused attention disabled")
end
GPTExample.main()
