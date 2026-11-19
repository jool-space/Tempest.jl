# Tempest.Compiler.IR

```@docs
Tempest.Compiler.IR
```

## Index formulas

```@docs
Tempest.Compiler.IR.Domain
Tempest.Compiler.IR.Var
Tempest.Compiler.IR.Affine
Tempest.Compiler.IR.Div
Tempest.Compiler.IR.Mod
Tempest.Compiler.IR.Indirect
Tempest.Compiler.IR.is_affine
Tempest.Compiler.IR.indirect_values
Tempest.Compiler.IR.substitute
Tempest.Compiler.IR.revalued
Tempest.Compiler.IR.fdiv
Tempest.Compiler.IR.fmod
```

## Layout

```@docs
Tempest.Compiler.IR.storage_length
Tempest.Compiler.IR.address
Tempest.Compiler.IR.group_stride
```

## Folds

```@docs
Tempest.Compiler.IR.Relative
Tempest.Compiler.IR.Exponential
Tempest.Compiler.IR.Square
Tempest.Compiler.IR.Statistic
Tempest.Compiler.IR.Sum
Tempest.Compiler.IR.Composed
Tempest.Compiler.IR.LOGSUMEXP
Tempest.Compiler.IR.InSequence
Tempest.Compiler.IR.Overwrite
Tempest.Compiler.IR.fold_reduce
Tempest.Compiler.IR.product_factors
```

## Terms

```@docs
Tempest.Compiler.IR.Load
Tempest.Compiler.IR.Ix
Tempest.Compiler.IR.Stat
Tempest.Compiler.IR.Recomputed
Tempest.Compiler.IR.Cast
Tempest.Compiler.IR.mentions
Tempest.Compiler.IR.mentions_within
Tempest.Compiler.IR.reached_elements
Tempest.Compiler.IR.sector_elements
Tempest.Compiler.IR.SECTOR_BYTES
```

## Masks

```@docs
Tempest.Compiler.IR.comparison_mask
Tempest.Compiler.IR.held_formula
Tempest.Compiler.IR.masked_result
```
