# Adding a target

A target is where a program runs. A package that adds one defines the names
below for its type, as CUDAExt does for `CUDATarget`; the CPU's definitions
are Tempest's own.

```@docs
Tempest.Compiler.Target
Tempest.Compiler.target_for
Tempest.Compiler.device_of
Tempest.Compiler.default_device
Tempest.Compiler.at_home
Tempest.Compiler.on_device
Tempest.Compiler.implementations
Tempest.Compiler.numerics
Tempest.Compiler.permits_rewrites
Tempest.Compiler.rounded_backward
Tempest.Compiler.deterministic
Tempest.Compiler.product_class
Tempest.Compiler.allocate
Tempest.Compiler.upload
Tempest.Compiler.upload!
Tempest.Compiler.download!
Tempest.Compiler.bindable
Tempest.Compiler.pointable
Tempest.Compiler.task_state
Tempest.Compiler.floor_launch
Tempest.Compiler.ScalarKernel
Tempest.Compiler.accumulate!
Tempest.Compiler.floor_split
Tempest.Compiler.Split
Tempest.Compiler.merge_lanes
Tempest.Compiler.Step
Tempest.Compiler.crosses
Tempest.Compiler.taken
Tempest.Compiler.crossing
Tempest.Compiler.stream
Tempest.Compiler.lane
Tempest.Compiler.link
Tempest.Compiler.on_stream
Tempest.Compiler.task_stream
Tempest.Compiler.event
Tempest.Compiler.record!
Tempest.Compiler.wait!
Tempest.Compiler.Scheduled
Tempest.Compiler.capture
Tempest.Compiler.replay
Tempest.Compiler.release
Tempest.Compiler.recapture
Tempest.Compiler.retarget!
Tempest.Compiler.device_name
Tempest.Compiler.toolchain
Tempest.Compiler.measurable
Tempest.Compiler.time_launch
Tempest.Compiler.time_crossing
Tempest.Compiler.quiesce
Tempest.Compiler.device_rates
```
