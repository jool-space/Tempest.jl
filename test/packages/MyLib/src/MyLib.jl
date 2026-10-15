# An implementation package outside Tempest, using only Tempest's public names:
# a CPU `take_rows`, registered from __init__ and inactive until
# `Tempest.activate(MyLib)`. test/isolated/out_of_tree.jl loads it.
module MyLib

using Tempest

struct MyKernels <: Tempest.Implementation
    name::Symbol
end

MyKernels(; name = :mylib) = MyKernels(name)

Tempest.implementation_name(k::MyKernels) = k.name

function Tempest.support(::MyKernels, request::Tempest.CallRequest)
    request.call.f === Tempest.take_rows || return Tempest.Unsupported("MyLib implements take_rows only")
    request.target isa CPUTarget || return Tempest.Unsupported("MyLib runs on the CPU")
    request.inputs[1].dtype <: AbstractFloat || return Tempest.Unsupported("MyLib needs floating-point values")
    Tempest.Supported()
end

function Tempest.build_implementation(::MyKernels, ::Tempest.CallRequest, runtime, out, inputs)
    x, rows = inputs
    columns = reshape(x, size(x, 1), :)
    results = vec(out)
    launch = () -> begin
        for (n, r) in enumerate(rows)
            results[n] = columns[r, n]
        end
        nothing
    end
    Tempest.BuiltImplementation(launch, (out, inputs), 0)
end

__init__() = Tempest.register!(MyLib, MyKernels; target = CPUTarget, rank = 5, active = false)

end
