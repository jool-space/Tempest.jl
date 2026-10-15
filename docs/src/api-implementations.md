```@meta
CurrentModule = Tempest
```

# Implementation API

Public names that are not exported: the implementation protocol, the
registry, region estimates, semantic verbs and folds. Refer to them as
`Tempest.support` or bring them in with `using Tempest: support`. The
[implementations guide](implementations.md) explains how they fit together.

```@autodocs
Modules = [Tempest]
Filter = t -> !Base.isexported(Tempest, nameof(t))
```
