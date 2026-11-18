#!/usr/bin/env bash
# Makes the depot a session starts from: this environment's dependencies,
# each at the manifest's version, downloaded and precompiled, and nothing of
# Tempest's, whose precompiling the session times.
#
#   bash deps.sh DEPOT
#
# DEPOT must not exist. Its environment `deps` is this one without Tempest,
# what Tempest depends on held as dependencies of its own, the paths of
# sources made absolute.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
depot=$1
[ -e "$depot" ] && { echo "$depot exists"; exit 1; }
mkdir -p "$depot/deps"
cp "$here/Project.toml" "$here/Manifest.toml" "$depot/deps/"
export JULIA_DEPOT_PATH=$depot
julia=${TEMPEST_SESSION_JULIA:-julia}
"$julia" --startup-file=no --project="$depot/deps" -e '
    using Pkg, TOML
    here, env = ARGS[1], dirname(Base.active_project())
    absolute(p) = isabspath(p) ? p : normpath(joinpath(here, p))
    project = TOML.parsefile(joinpath(env, "Project.toml"))
    manifest = TOML.parsefile(joinpath(env, "Manifest.toml"))
    for (name, source) in project["sources"]
        haskey(source, "path") && (source["path"] = absolute(source["path"]))
    end
    for (name, entries) in manifest["deps"], e in entries
        haskey(e, "path") && (e["path"] = absolute(e["path"]))
    end
    tempest = only(manifest["deps"]["Tempest"])
    for name in [tempest["deps"]; get(tempest, "weakdeps", String[])]
        haskey(manifest["deps"], name) && (project["deps"][name] = only(manifest["deps"][name])["uuid"])
    end
    open(io -> TOML.print(io, project), joinpath(env, "Project.toml"), "w")
    open(io -> TOML.print(io, manifest), joinpath(env, "Manifest.toml"), "w")
    Pkg.rm("Tempest")
    Pkg.instantiate()
    Pkg.precompile()
    Pkg.status()
' "$here"
