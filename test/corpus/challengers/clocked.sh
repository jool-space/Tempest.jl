#!/usr/bin/env bash
# Runs a corpus job and prints, after it, the median SM clock over the moments
# the GPU was busy while it ran, sampled four times a second, as
# `corpus clock MHz` (or `unread`).
#
#   bash clocked.sh SECONDS COMMAND...
#
# The command is stopped after SECONDS, exiting 124 as `timeout` does.
seconds=$1
shift
samples=$(mktemp)
(while true; do
    nvidia-smi --query-gpu=clocks.sm,utilization.gpu --format=csv,noheader,nounits >> "$samples" 2> /dev/null
    sleep 0.25
done) &
sampler=$!
timeout "$seconds" "$@"
status=$?
kill "$sampler"
awk -F', *' '$2 > 0 { print $1 }' "$samples" | sort -n |
    awk '{ c[NR] = $1 } END { print "corpus clock", NR ? c[int((NR + 1) / 2)] : "unread" }'
rm -f "$samples"
exit "$status"
