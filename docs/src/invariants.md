# Plan invariants

[`verify`](@ref Tempest.Compiler.verify) checks the invariants the kernels of
a plan rely on, and throws an [`InvalidPlan`](@ref Tempest.Compiler.InvalidPlan)
naming the value and the kernel at the first that fails. The exception
carries the invariant's number here.

[`prepare`](@ref Tempest.Compiler.prepare) checks every plan it forms, and
its storage, with `verify = true`, the default when the environment sets
`TEMPEST_VERIFY=true`, as the tests do.

## 1. Owners and order

Every value a kernel stores is a member of that kernel's group and of no
other; every value a kernel or a result reads is an input, a constant, or
stored by a kernel earlier in the order.

## 2. Storage

With `storage`, simulating the order:

- no slot is given to a value while a kernel that uses the value it holds
  does not run before the value's, by the order the program gives the
  kernels (see `plan_order`: on one stream earlier, or before it through the
  values it reads), but where a pair the storage records allows it, the
  kernel the last to use it;
- each donated, staged, overwritten and reused pair holds by the rule that
  allows it, tested again, the other kernels reading what it writes over
  running before its kernel by that order;
- and each lifetime covers its value's kernel and every reader.

On one device, whose kernels are one stream, a value's users run before a
kernel where they are earlier.

`in_place(k)` says whether kernel `k`'s realization reads each element a set
of its outputs needs before it writes any of them, as `storage` was told.

## 3. Recurrences

A recurrence's group has every state of its cycle first among its members,
each carried from a different value it computes, and stores the values of
the cycle no other kernel stores; each member outside the cycle rides its
steps along them or at their last, and each fold of it is a sum that rides
them, through values nothing stores.

## 4. Terms

Each member's term loads values stored apart, its own only in a
recurrence's; each member has a place within the domain; a fold read as a
`Stat` is a primary fold of the group complete in an earlier phase; and a
fold a term recomputes is not a member of the group.

## 5. Devices

Every read joins values on one device but a crossing's, which reads a value
on another device than its own, one span of its storage (see
`storage_span`); every member of a kernel, and every value it reads, is on
its device, but a crossing's kernel's, whose one member is a crossing,
reading its source (both stored, by 1); with `storage`, a slot holds values
of one device, and a result that replaces an argument is on that argument's
device.
