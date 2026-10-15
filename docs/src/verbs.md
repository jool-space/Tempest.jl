# Semantic verbs and custom kernels

The public verb describes results. Its implementation writes into supplied
storage. Tempest can retain the verb as an opaque call or expand its reference
before scalar fusion and buffer planning. Neither Pol.Allocating nor a model-level
implementation keyword is required.

## Defining a verb

```julia
using Tempest: TensorSpec, reference, record_call

pair_wave(x) = reference(pair_wave, x)
pair_wave(x::Tempest.TracedArray) = record_call(pair_wave, x)

Tempest.output_spec(::typeof(pair_wave), x) = (; s = TensorSpec(x), c = TensorSpec(x))

Tempest.reference(::typeof(pair_wave), x) = (; s = sin.(x), c = cos.(x))

Tempest.verb_pullback(::typeof(pair_wave), inputs, y, dy) = (dy.s .* y.c .- dy.c .* y.s,)
```

`record_call` uses output metadata without running either the reference or a
kernel. Results may be a single tensor/scalar or nested tuples/named tuples.
`TensorSpec(T, dims; scalar=false)` describes an array;
`TensorSpec(T, (); scalar=true)` describes a symbolic number. All outputs must
have floating element types. Array shape and scalar-versus-array kind are checked
against the reference when it is expanded.

The pullback receives positional inputs as a tuple and output/cotangent trees
with the declared result structure. It returns one cotangent per input, or
`nothing` for no contribution. Unused outputs receive symbolic zero cotangents.
Their derivatives are not guessed. A pullback can call a second semantic verb
representing a hand-written backward kernel, as the runnable example does.

Positional symbolic inputs are runtime data. Ordinary floating positional
constants are also accepted. Keywords are immutable trace-time configuration;
put replay-changing scalars in positional inputs. Captured mutable keyword
objects, input mutation, and explicit output/input aliases are unsupported.

A normal array method can run the reference eagerly, as above, or use its own
kernel wrapper. Only the symbolic method needs to call `record_call`. There is
no automatic interception of arbitrary CUDA launches and no requirement to remove
`::CuArray` from the actual launcher.

## Registering an implementation

```julia
using Tempest: Implementation, CallRequest, Supported, Unsupported

struct MyWave <: Implementation end

function Tempest.support(::MyWave, r::CallRequest)
    if !(r.call.f === pair_wave)
        return Unsupported("different operation")
    end
    if !(r.target isa CUDATarget)
        return Unsupported("CUDA only")
    end
    if !(all(s -> s.dtype === Float32, r.inputs))
        return Unsupported("Float32 only")
    end
    Supported()
end

# build_implementation(::MyWave, r::CallRequest, runtime, out, inputs)
# returns BuiltImplementation(launch, owners, workspace_bytes)
```

The request carries the exact function identity, static keywords, output
specification, logical input metadata, and target. Multiple implementations may
support the same function. Pass their instances in preference order to
`prepare(...; implementations=(MyWave(), ...))`.

Construction receives the full output tree, with each leaf replaced by a physical
buffer, and a tuple of input buffers. Scalar leaves use zero-dimensional buffers;
read them on the device when they must change during replay. This first contract
provides dense inputs, materializing mapped views and transposes as needed.
Stride/layout negotiation is subsequent work.

The launcher must write all declared outputs, read only declared inputs, and
retain its resource owners. Workspace and kernel compilation belong in
construction, outside capture. The compiler provides distinct output destinations
for a call; unused sibling outputs still get storage because an opaque launcher
may write them. No automatic reference verification or alias/effect inference is
implied by registration.

The complete example in `examples/semantic_kernels.jl` defines a two-output CUDA
forward kernel, a separate backward kernel, their support/build methods, and a
training loss. It compares with the reference and captures/replays changed inputs.
Run it with:

```sh
julia --project=examples examples/semantic_kernels.jl
```

## Selection, decomposition, and fusion

Before final planning, Tempest checks candidate support for semantic calls.
Eligible calls remain opaque. Calls with no eligible implementation expand by
tracing their reference on isolated symbolic inputs and splicing that graph into
the surrounding computation. References can compose other semantic verbs;
recursive reference expansion is rejected.

If construction later reports `Unsupported(reason)`, the next eligible candidate
is tried. If all reject, preparation expands that call's reference and replans.
Unexpected construction errors propagate. An operation without a supported
implementation or reference reports an error. Forward-only opaque execution does
not require a reference or derivative, though a reference is valuable for tests.

Expanded pointwise operations can fuse with their neighbors under the ordinary
region policy. Selected opaque calls are explicit boundaries: Tempest cannot fuse
inside them or reinterpret their internals. An implementation can claim a call
together with the work around it by [proposing a region](proposed-regions.md)
that contains it, as a library claims a product with its epilogue. A supported
call is not compared with its expanded reference by cost: an implementation
should decline a call it would realize worse than the reference fused with its
neighbours. A verb whose meaning is a fold, a product or a pointwise function
should lower to that primitive instead of recording a call; see
[spelling invariance](spelling-invariance.md).

Forward/backward kernels can share saved data through explicit outputs and inputs.
In this stage those values belong to the declared semantic result tree; alternative
implementations of the same call must agree on that tree. Private,
implementation-specific saved-state contracts and joint forward/backward selection
remain subsequent work.

## General forward-only programs

`trace` now accepts a symbolic value or nested tuple/named tuple of values.
`prepare_forward(f, examples...; ...)` compiles that result without differentiating
it. `prepare_forward(trace(f, examples...), examples...)` accepts a saved trace.

```julia
ex = prepare_forward(x -> (; y = sin.(x), total = sum(x)), randn(Float32, 32))
y = execute!(ex)
y.y       # borrowed array
y.total[] # symbolic-number results are borrowed 0D buffers
```

Input passthroughs, repeated output references, views, empty selections, and
zero-dimensional arrays are supported. Returned references can alias each other;
all are read-only and overwritten by later execution/rebinding. Copy values you
need to retain. CUDA capture/replay supports the same output structure.

`prepare` remains the scalar-loss training convenience. `differentiate` requires
a single symbolic-number result and an explicit `verb_pullback` for each semantic
call contributing to that result. Automatic AD of a verb's reference is not yet
provided; write the decomposition directly or supply its pullback recipe.

## Implementation-specific saved values

A provider can now select a complete forward/backward recipe with private saved
outputs. See [Implementation-specific reverse rules](reverse-rules.md) for the
`ReverseRule` contract, lifetime handling, and paired fallback. The public verb
still returns only its mathematical result.
