#!/bin/sh
# Make a macOS app bundle:
#   tools/bundle_macos.sh <name> <gamedir> [App Name]
#     one game: the executable and the game files it needs (exheader.bin, code.bin, romfs.bin if any) in
#     Contents/Resources/game
#   tools/bundle_macos.sh --games <name>... [--name "App Name"]
#     a multi-game build (cmake -DGAMES=... or -DGAMES=all): the `recomp` executable with its game list, each game's
#     files in Contents/Resources/games/<name>/ (from data/<name>) and its code in Contents/Frameworks/<name>.dylib
# Homebrew reading sdmc:/ finds it in ~/Library/Application Support/3dsrecomp/sdmc.
set -e
B=${BUILD:-build}  # BUILD: the cmake build folder
GAMES=
if [ "$1" = --games ]; then
  shift
  N=recomp; APP="3DS Games"
  while [ $# -gt 0 ]; do
    if [ "$1" = --name ]; then APP=$2; shift 2; else GAMES="$GAMES $1"; shift; fi
  done
  [ -n "$GAMES" ] && [ -f "$B/$N" ] || { echo "usage: BUILD=<folder> $0 --games <name>... [--name \"App Name\"]"; exit 1; }
  D=data/$(echo $GAMES | cut -d' ' -f1)  # (the icon's game)
else
  N=$1; D=$2; APP=$3
  [ -n "$N" ] && [ -f "$D/info.json" ] || { echo "usage: $0 <name> <gamedir> [App Name]"; exit 1; }
fi
# the app's name: the game's own title (its SMDH's English short title), else "<name> app" (never <build>/<name>,
# the executable itself)
if [ -z "$APP" ]; then
  APP=$(python3 -c "
import sys
try:
    d = open('$D/exefs/icon.bin', 'rb').read()
    t = d[8 + 0x200:8 + 0x280].decode('utf-16-le').split(chr(0))[0].strip()
    print(''.join(c for c in t if c not in '/\\\\:'))
except Exception:
    pass" 2>/dev/null)
  [ -n "$APP" ] && [ "$APP" != "$N" ] || APP="$N app"
fi
OUT="$B/$APP.app"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
cp "$B/$N" "$OUT/Contents/MacOS/$N"
# a game's files: copy_game <data dir> <destination> <its precompiled shaders>
copy_game() {
  mkdir -p "$2"
  cp "$1/exheader.bin" "$1/code.bin" "$1/info.json" "$2/"
  [ -f "$1/romfs.bin" ] && cp "$1/romfs.bin" "$2/"
  [ -f "$1/exefs/icon.bin" ] && mkdir -p "$2/exefs" && for x in icon banner logo; do [ -f "$1/exefs/$x.bin" ] && cp "$1/exefs/$x.bin" "$2/exefs/"; done
  tid=$(python3 -c "import json;print(json.load(open('$1/info.json'))['program_id'])")
  # shipped pipeline and vertex program lists, the profile
  for f in pipelines.bin profile.txt vsprograms.bin; do [ -f "games/$tid/$f" ] && cp -L "games/$tid/$f" "$2/"; done
  [ -f "$3" ] && cp "$3" "$4"
  return 0
}
if [ -n "$GAMES" ]; then
  mkdir -p "$OUT/Contents/Frameworks"
  for g in $GAMES; do
    [ -f "$B/$g.dylib" ] || { echo "no $B/$g.dylib (is $g in the build's GAMES?)"; exit 1; }
    copy_game "data/$g" "$OUT/Contents/Resources/games/$g" "$B/shaders_$g.metallib" "$OUT/Contents/Resources/games/$g/shaders.metallib"
    cp "$B/$g.dylib" "$OUT/Contents/Frameworks/"
    codesign --force -s - "$OUT/Contents/Frameworks/$g.dylib" >/dev/null 2>&1 || true
  done
  [ -d system_archives ] && cp -r system_archives "$OUT/Contents/Resources/games/"
else
  copy_game "$D" "$OUT/Contents/Resources/game" "$B/shaders.metallib" "$OUT/Contents/Resources/shaders.metallib"
  [ -d system_archives ] && cp -r system_archives "$OUT/Contents/Resources/game/"
fi
# app icon from the game's SMDH
if [ -f "$D/exefs/icon.bin" ] && command -v magick >/dev/null; then
  T=$(mktemp -d)
  python3 tools/smdh_icon.py "$D/exefs/icon.bin" "$T/icon.ppm"
  mkdir -p "$T/icon.iconset"
  for s in 16 32 64 128 256 512; do
    magick "$T/icon.ppm" -filter Mitchell -resize ${s}x${s} "$T/icon.iconset/icon_${s}x${s}.png"
    magick "$T/icon.ppm" -filter Mitchell -resize $((s*2))x$((s*2)) "$T/icon.iconset/icon_${s}x${s}@2x.png"
  done
  iconutil -c icns "$T/icon.iconset" -o "$OUT/Contents/Resources/icon.icns" && ICON='<key>CFBundleIconFile</key><string>icon</string>'
  rm -rf "$T"
fi
cat > "$OUT/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>$N</string>
  <key>CFBundleIdentifier</key><string>local.3dsrecomp.$N</string>
  <key>CFBundleName</key><string>$APP</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>NSHighResolutionCapable</key><true/>
  $ICON
  <key>LSMinimumSystemVersion</key><string>13.0</string>
</dict></plist>
PLIST
# SDL3 from Homebrew is linked dynamically; copy it in and point the binary at it
SDL=$(otool -L "$OUT/Contents/MacOS/$N" | awk '/libSDL3/ {print $1}')
if [ -n "$SDL" ]; then
  mkdir -p "$OUT/Contents/Frameworks"
  cp "$SDL" "$OUT/Contents/Frameworks/"
  install_name_tool -change "$SDL" "@executable_path/../Frameworks/$(basename $SDL)" "$OUT/Contents/MacOS/$N"
  codesign --force -s - "$OUT/Contents/Frameworks/$(basename $SDL)" >/dev/null 2>&1 || true
fi
codesign --force -s - "$OUT" >/dev/null 2>&1 || true
echo "built $OUT"
