# Tempest's own implementations of the kernel packages it has extensions for:
# values a target holds, whose methods each extension defines when its
# package loads. Until then one cannot be made.

public cuBLASProducts, cuDNNAttention, cuTileGenerator

"""
    cuTileGenerator()

Tempest's groups as cuTile kernels: the generator, which claims every group
it can tile and lower, and searches its tiles. Needs cuTile loaded.
"""
struct cuTileGenerator <: Implementation
    cuTileGenerator() = needs(new(), :cuTile)
end

"""
    cuBLASProducts()

Products as cuBLAS's batched ones, of Float32 factors or half ones widened to
Float32, and products alike but for where their operands start as one call.
Needs cuBLAS loaded.
"""
struct cuBLASProducts <: Library
    cuBLASProducts() = needs(new(), :cuBLAS)
end

"""
    cuDNNAttention()

Attention of half-precision queries, keys and values as cuDNN's SDPA engines
compute it, with and without a causal mask. Needs cuDNN loaded.
"""
struct cuDNNAttention <: Library
    cuDNNAttention() = needs(new(), :cuDNN)
end

const PACKAGED = Union{cuTileGenerator, cuBLASProducts, cuDNNAttention}

Base.show(io::IO, i::PACKAGED) = print(io, nameof(typeof(i)), "()")

package_of(::cuTileGenerator) = :cuTile
package_of(::cuBLASProducts) = :cuBLAS
package_of(::cuDNNAttention) = :cuDNN

loaded(::Type{<:PACKAGED}) = false

needs(i, package) = Base.invokelatest(loaded, typeof(i)) ? i : throw(ArgumentError("$(nameof(typeof(i)))() needs $package loaded"))

# In the order of their packages' names, as ties between estimates go to the
# implementation a target holds first.
const PACKAGES = (cuBLASProducts, cuDNNAttention, cuTileGenerator)

"Tempest's own implementations whose packages are loaded: those a `CUDATarget` holds by default."
loaded_implementations() = Tuple(T() for T in PACKAGES if Base.invokelatest(loaded, T))
