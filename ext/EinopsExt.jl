# Einops spells a product with labels; Tempest's product is the same thing
# with the labels numbered. A contraction with a traced factor records one
# `contract`, and the plan decides how it runs.
module EinopsExt

using Einops: Einops
using Tempest: Tempest
using Tempest.Compiler: TracedArray

# Labels numbered in order of first appearance.
function numbered(ixs, iout)
    labels = unique((ixs[1]..., ixs[2]...))
    number(ls) = map(l -> findfirst(==(l), labels), ls)
    number(ixs[1]), number(ixs[2]), number(iout)
end

for (A, B) in ((TracedArray, TracedArray), (TracedArray, AbstractArray), (AbstractArray, TracedArray))
    @eval Einops.contract(ixs::NTuple{2, Tuple}, iout::Tuple, x::$A, y::$B; kws...) =
        Tempest.Compiler.contract(x, y, numbered(ixs, iout)...)
end

end
