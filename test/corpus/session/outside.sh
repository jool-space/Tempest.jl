#!/usr/bin/env bash
# Runs a session's job, telling it the moment it was started, taken here,
# outside it, just before it starts: TEMPEST_CORPUS_STARTED, in seconds since
# the epoch. It is stopped after SECONDS, exiting 124 as `timeout` does.
#
#   bash outside.sh SECONDS COMMAND...
seconds=$1
shift
export TEMPEST_CORPUS_STARTED=$(date +%s.%N)
exec timeout "$seconds" "$@"
