#!/usr/bin/env bash
# One session of the preparation goal (README.md): a home directory made
# empty, holding the depot of precompiled dependencies deps.sh made, Tempest
# precompiled and loaded in it, timed, then each program run three times
# (run.jl's scenarios first, kept and doubled) and once after the corpus's
# other programs (second), rows in OUT/corpus.tsv, what each process compiled
# in OUT/traces.
#
#   bash session.sh OUT PROGRAM...
#
# PROGRAM is a corpus name or a program file's path, as run.jl takes it.
# TEMPEST_SESSION_DEPS names the depot deps.sh made; TEMPEST_SESSION_JULIA the
# julia binary (the one on the path otherwise); TEMPEST_SESSION_REFERENCES the
# directory the untraced outputs are kept in across sessions (OUT/references
# otherwise); TEMPEST_CORPUS_QWEN the 4B's checkpoint; JULIA_NUM_THREADS the
# threads (Julia's default otherwise); TEMPEST_SESSION_SCENARIOS the
# scenarios (all four otherwise). Nothing else of the caller's environment
# reaches the processes but the GPU lock and the devices seen.
set -u
here=$(cd "$(dirname "$0")" && pwd)
tree=$(cd "$here/../../.." && pwd)
out=$1
shift
deps=${TEMPEST_SESSION_DEPS:?the depot deps.sh made}
julia=${TEMPEST_SESSION_JULIA:-$(julia --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')}
home=$out/home
[ -e "$home" ] && { echo "$home exists: a session starts from an empty home"; exit 1; }
mkdir -p "$home/.julia" "$out/logs"
environment=(env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$home" JULIA_DEPOT_PATH="$home/.julia:$deps"
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" TEMPEST_CORPUS="$out" TEMPEST_CORPUS_STORES="$home/stores"
    TEMPEST_CORPUS_REFERENCES="${TEMPEST_SESSION_REFERENCES:-$out/references}"
    TEMPEST_CORPUS_QWEN="${TEMPEST_CORPUS_QWEN:-$HOME/models/Qwen/Qwen3.5-4B}" TEMPEST_CORPUS_TIMEOUT="${TEMPEST_CORPUS_TIMEOUT:-3600}"
    CUDA_CACHE_PATH="$home/stores/nv")
[ -n "${JULIA_NUM_THREADS:-}" ] && environment+=(JULIA_NUM_THREADS="$JULIA_NUM_THREADS")
[ -n "${TEMPEST_GPU_LOCK:-}" ] && environment+=(TEMPEST_GPU_LOCK="$TEMPEST_GPU_LOCK")
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && environment+=(CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
now() { date +%s.%N; }
commit=$(git -C "$tree" describe --always --dirty --abbrev=7)
echo "session at $commit on $(hostname), threads ${JULIA_NUM_THREADS:-default}, $(date '+%F %T')" | tee -a "$out/session.log"
# Precompiling Tempest, its extensions and what depends on it, then loading
# what a session's programs load.
t0=$(now)
"${environment[@]}" "$julia" --startup-file=no --project="$here" -e 'using Pkg; Pkg.precompile()' > "$out/logs/precompile.log" 2>&1 ||
    { echo "precompiling failed; see $out/logs/precompile.log"; exit 1; }
t1=$(now)
"${environment[@]}" "$julia" --startup-file=no --project="$here" -e 'using Tempest, CUDACore, cuTile, cuBLAS, cuDNN, Vallmo' > "$out/logs/load.log" 2>&1 ||
    { echo "loading failed; see $out/logs/load.log"; exit 1; }
t2=$(now)
printf "precompile_s\t%.1f\tload_s\t%.1f\tcommit\t%s\thost\t%s\tthreads\t%s\n" "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" \
    "$commit" "$(hostname)" "${JULIA_NUM_THREADS:-default}" | tee -a "$out/session.tsv"
status=0
for program in "$@"; do
    echo "== $program $(date '+%T')" | tee -a "$out/session.log"
    "${environment[@]}" "$julia" --startup-file=no --project="$here" "$tree/test/corpus/run.jl" "$program" --scenarios "${TEMPEST_SESSION_SCENARIOS:-first,kept,doubled,second}" \
        --runs 1 >> "$out/session.log" 2>&1 || { echo "$program failed; see $out/session.log"; status=1; }
done
# What the processes wrote in the home outside the stores and the depot's
# compiled code: a cache outside the stores would be a hole in the session.
find "$home" -type f -not -path "$home/stores/*" -not -path "$home/.julia/compiled/*" -not -path "$home/.julia/logs/*" > "$out/outside-stores.txt"
echo "files outside the stores: $(wc -l < "$out/outside-stores.txt")" | tee -a "$out/session.log"
echo "== done $(date '+%T')" | tee -a "$out/session.log"
exit $status
