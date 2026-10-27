#!/bin/sh
# Compile the fragment shaders written by `<game> --emit-shaders <dir> ...` into one Metal library:
# tools/build_metallib.sh <dir> <out.metallib>. Only shaders that changed are recompiled (in parallel).
set -e
D=$1; OUT=$2
if ! xcrun -f metal >/dev/null 2>&1; then
  echo "build_metallib: no Metal compiler (xcrun metal), shaders will be compiled at run time"
  exit 0
fi
cd "$D"
# same semantics as the run-time compiler with MTLMathModeSafe
for f in *.metal; do [ "${f%.metal}.air" -nt "$f" ] || echo "$f"; done |
  xargs -P "$(sysctl -n hw.ncpu)" -n 16 sh -c 'for f; do xcrun metal -c -w -fno-fast-math -o "${f%.metal}.air" "$f" || exit 255; done' sh
NEWEST=$(ls -t *.air | head -1)
if [ ! -f "$OUT" ] || [ "$NEWEST" -nt "$OUT" ] || [ "$(ls *.air | wc -l)" != "$(cat .count 2>/dev/null)" ]; then
  xcrun metallib -o "$OUT" *.air
  ls *.air | wc -l > .count
fi
touch "$OUT"
