#!/bin/sh
# Load-independent cost of replaying a dump: hardware cycles and instructions retired per
# replay iteration (single GPU thread), from the difference between a 1- and an N-iteration run.
#   cost.sh <gpu_replay binary> <dumpdir> [N]
bin="$1"; dump="$2"; n="${3:-6}"
out="${TMPDIR:-/tmp}/cost_out.png"
measure() {
    GPU_THREADS=1 /usr/bin/time -l "$bin" "$dump" "$out" -r "$1" $EXTRA 2>&1 |
        awk '/instructions retired/{i=$1} /cycles elapsed/{c=$1} END{print c, i}'
}
set -- $(measure 1) $(measure "$n")
awk -v c1="$1" -v i1="$2" -v cn="$3" -v in_="$4" -v n="$n" 'BEGIN{
    printf "per iteration: %.1f Mcycles, %.1f Minstructions\n", (cn-c1)/(n-1)/1e6, (in_-i1)/(n-1)/1e6 }'
