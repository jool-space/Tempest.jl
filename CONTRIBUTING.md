# Working on Tempest

Write code for the person who will debug and change it next. The compiler is an
experiment, but its source is part of the deliverable, not disposable scaffolding.
See [the source guide](docs/src/internals.md) for the current stages and invariants.

Prefer direct Julia and small functions with a clear responsibility. Use names
that explain the role of graph metadata, storage, and implementation decisions.
Mathematical names and short dispatch methods are useful; multi-statement
one-liners and dense chains of side effects usually are not.

Separate mathematical operations from their realization. Express decomposable
built-in derivatives as ordinary graph operations. Backend adapters should not
need private knowledge of AD helper names. Add abstractions when they clarify a
real shared responsibility, and remove them when a simpler representation works.

Keep errors meaningful. An unsupported operation is a deliberate result of a
capability check. A bug inside an extension must retain its exception and context.
Avoid catch-all fallback and silent defaults for unknown operation kinds.

Comments should explain invariants, ownership, numerical choices, and constraints
that cannot be read directly from the code. Update the source guide when pass
boundaries change. Current limitations belong in documentation rather than being
hidden behind optimistic interface names.

Use `.JuliaFormatter.toml` for spacing and wrapping. Review formatter output:
formatting does not choose useful names, function boundaries, or abstractions.
Do not compress a clear expression simply because it fits on one line.

Run the relevant existing tests. Add regressions for changed contracts and failure
modes; avoid tests that merely restate a helper's implementation. Numerical
correctness, alias lifetimes, and repeated GPU replay need independent checks.
Passing tests establish behavior, not maintainability or performance. Report
performance improvements only with measurements.
