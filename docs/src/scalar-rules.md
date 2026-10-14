# Reusing scalar derivative rules

Tempest uses its built-in graph recipes first, then user-defined
`scalar_partials`. Loading ChainRules enables an optional fallback for scalar
functions with a symbolically traceable `rrule`:

```julia
using Tempest, ChainRules

loss(x) = sum(log1p.(x) .+ expm1.(x))
x = Float32[0.2, 0.4, 0.7]
ex = prepare(loss, x)
value, gradient = execute!(ex)
```

ChainRulesCore is a weak dependency. `using Tempest` does not load it or the
ChainRules rule collection. Loading ChainRulesCore alone enables the adapter
for rules supplied by another package; loading ChainRules supplies its Base and
standard-library rules. No derivative library becomes a mandatory dependency.

## What the adapter does

The adapter calls the public `rrule` API on fresh symbolic scalar arguments,
seeds its pullback with one, and forces deferred cotangents. It then lifts the
resulting scalar arithmetic over the original array inputs. Broadcast gradients
still reduce to each input's shape, including singleton axes and zero-dimensional
arrays. The rule's forward output is bound to the existing graph output so that
saved-output derivatives can reuse its storage.

This produces ordinary pointwise nodes before fusion and implementation
selection. Vendor candidates inspect those operations in the usual way. A rule
being traceable does not mean that cuDNN supports its arithmetic; fallback and
rejection remain implementation decisions. The cuTile adapter currently routes
`log1p`, `expm1`, `asin`, `acos`, `sign`, and unary `atan` to CUDA because their Base
implementations have no verified cuTile lowering. These rejections appear in
`explain`; numerical formulas are not substituted to make a candidate fit.

## Deliberate limits

- The rule must accept Tempest's symbolic `Real` scalars. A rule restricted to
  `Float64` or `CuArray` does not become applicable to them.
- Rules may use the supported scalar primitives. Data-dependent comparisons and
  branches still fail during tracing. Upstream `hypot`, `min`, and `max` rules
  currently reach this boundary.
- `ZeroTangent` becomes a zero partial. `NoTangent` for a data input is rejected:
  nondifferentiable inputs require a deliberate policy. Semantic verbs already
  let their pullback return `nothing` for such inputs.
- Scalar results and real partials are supported. Structured tangents, captured
  function tangents, arbitrary mutation, and arbitrary Julia AD are outside this
  adapter's contract. `ProjectTo` works when its projection is symbolically
  expressible; this is not a claim about every projector.
- Unexpected exceptions inside a rule propagate. They do not trigger a different
  derivative implementation.

Existing Tempest recipes remain first, including its saved-output, factored
`tanh` derivative. Loading a rule package does not silently replace those
numerical choices. A custom `scalar_partials` rule also takes priority.

## Experiment and decision

The September 2026 experiment used ChainRules 1.73.0, ChainRulesCore 1.26.1,
and DiffRules 1.16.0. Initially,
several failures came from missing scalar frontend primitives, including
`sincos`, `expm1`, and `log1p`. Adding those forward primitives made the associated
rules traceable without writing new derivative formulas. Symbolic predicates
remain a separate limitation.

The numerical tests cover shared derivatives, singleton broadcasting,
zero-dimensional inputs, rebinding, storage reuse, projection, thunks, constant
outputs, and explicit-rule precedence. CUDA tests compare captured replay against
CPU finite differences with changed inputs.

DiffRules was also evaluated through its public expression API. It provides
useful scalar coverage, but choosing a table also chooses mathematical policy:
its `abs` derivative is one at zero, while ChainRules returns zero there. Some
DiffRules expressions also branch on input values, so it is not used as a
second runtime fallback.

We use public `rrule` rather than ChainRulesCore's `derivatives_given_output`,
which its source explicitly labels an experimental implementation detail.
The upstream contracts are documented in the
[ChainRules introduction](https://juliadiff.org/ChainRulesCore.jl/stable/) and
[DiffRules API](https://juliadiff.org/DiffRules.jl/stable/).
