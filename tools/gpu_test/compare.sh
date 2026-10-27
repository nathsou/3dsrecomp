#!/bin/sh
# Replays the given dumps with build/gpu_replay and compares every written PNG with the files of
# the same name in a reference directory (produced by an older build with the same command).
#   compare.sh <refdir> <outdir> <dump>...
set -e
ref="$1"; out="$2"; shift 2
here="$(dirname "$0")"
replay="${REPLAY:-$here/build/gpu_replay}"
mkdir -p "$out"
status=0
for d in "$@"; do
    name="$(basename "$d")"
    "$replay" "$d" "$out/$name.png" -t 2>/dev/null >/dev/null
done
for f in "$ref"/*.png; do
    b="$(basename "$f")"
    if cmp -s "$f" "$out/$b"; then echo "same  $b"; else echo "DIFF  $b"; status=1; fi
done
exit $status
