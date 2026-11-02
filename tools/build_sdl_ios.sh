#!/bin/sh
# Build SDL3 as static libraries for iOS devices and the iOS simulator (Homebrew's SDL3 is macOS only):
# tools/build_sdl_ios.sh <SDL3 source folder> (a release from https://github.com/libsdl-org/SDL, 3.4 or later).
# The installs go to external/sdl-ios/{iphoneos,iphonesimulator}, where the iOS build looks for them.
set -e
SRC=$1
[ -f "$SRC/include/SDL3/SDL.h" ] || { echo "usage: $0 <SDL3 source folder>"; exit 1; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
for SDK in iphoneos iphonesimulator; do
  B=$ROOT/external/sdl-build/$SDK
  cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=$SDK -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 -DCMAKE_BUILD_TYPE=Release -DSDL_SHARED=OFF -DSDL_STATIC=ON \
    -DSDL_TEST_LIBRARY=OFF -DSDL_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$ROOT/external/sdl-ios/$SDK" >/dev/null
  cmake --build "$B" >/dev/null
  cmake --install "$B" >/dev/null
  echo "SDL3 for $SDK: external/sdl-ios/$SDK"
done
