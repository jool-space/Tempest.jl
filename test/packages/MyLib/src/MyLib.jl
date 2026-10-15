# An implementation package outside Tempest, using only Tempest's public names:
# a CPU logsumexp over the first axis, registered from __init__ and inactive
# until `Tempest.activate(MyLib)`. test/isolated/out_of_tree.jl loads it.
module MyLib

using Tempest

struct MyKernels <: Tempest.Implementation
    name::Symbol
end

MyKernels(; name = :mylib) = MyKernels(name)

Tempest.implementation_name(k::MyKernels) = k.name

function Tempest.support(::MyKernels, request::Tempest.CallRequest)
    request.call.f === Tempest.logsumexp || return Tempest.Unsupported("MyLib implements logsumexp only")
    request.target isa CPUTarget || return Tempest.Unsupported("MyLib runs on the CPU")
    request.call.kwargs.dims == (1,) || return Tempest.Unsupported("MyLib reduces the first axis")
    request.inputs[1].dtype <: AbstractFloat || return Tempest.Unsupported("MyLib needs floating-point values")
    Tempest.Supported()
end

function Tempest.build_implementation(::MyKernels, ::Tempest.CallRequest, runtime, out, inputs)
    x = only(inputs)
    columns = reshape(x, size(x, 1), :)
    results = vec(out)
    launch = () -> begin
        for j in axes(columns, 2)
            m = maximum(view(columns, :, j))
            results[j] = m + log(sum(v -> exp(v - m), view(columns, :, j)))
        end
        nothing
    end
    Tempest.BuiltImplementation(launch, (out, inputs), 0)
end

__init__() = Tempest.register!(MyLib, MyKernels; target = CPUTarget, rank = 5, active = false)

end
