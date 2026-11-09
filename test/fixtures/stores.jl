# What a run keeps on disk, in one directory of the run's own: Tempest's
# tuning directory (TEMPEST_CACHE: the record of what it measured, the
# device's rates, the decisions it kept to replay) and the store of compiled
# kernels (Julia's object cache, JULIA_OBJCACHE_PATH), so that a run neither
# reads what an earlier one left nor leaves anything for a later one. The test
# runner gives the suite one before its workers start; a script calls
# `run_stores!` before it prepares anything. A process opens its object cache
# once: at its first kernel before Julia 1.14, as it starts from 1.14, where a
# script is started with the settings `run_stores` gives instead.

"""
    run_stores(dir) -> settings

The settings that keep a run's stores in `dir`: TEMPEST_CACHE, and the object
cache in it, for a process started with them.
"""
run_stores(dir) = ["TEMPEST_CACHE" => dir, "JULIA_OBJCACHE_PATH" => joinpath(dir, "objcache")]

"""
    run_stores!(dir = mktempdir()) -> dir

This process's stores kept in `dir`, by default a new directory removed when
the process exits, however it exits short of being killed. A directory given
is kept, for a later run to read what this one left on purpose.
"""
function run_stores!(dir = mktempdir())
    mkpath(dir)
    foreach(((k, v),) -> ENV[k] = v, run_stores(dir))
    dir
end
