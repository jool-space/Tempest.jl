"""
TempestSeed: the new core described in DESIGN.md, built beside Tempest. It
takes Tempest's traced programs as its program level and plans them through
the index level and schedules.
"""
module TempestSeed

import Tempest

include("index.jl")
include("graph.jl")
include("lower.jl")
include("interpret.jl")

end
