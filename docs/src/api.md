```@meta
CurrentModule = Tempest
```

# API reference

The exported names, for model authors. Names that are public but not exported,
for implementation and verb authors, are on the
[implementation API](api-implementations.md) page.

```@index
Pages = ["api.md", "api-implementations.md"]
```

```@autodocs
Modules = [Tempest]
Filter = t -> Base.isexported(Tempest, nameof(t))
```
