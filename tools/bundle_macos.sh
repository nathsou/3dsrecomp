#!/bin/sh
# Make a macOS app bundle: tools/bundle_macos.sh <name> <gamedir> [App Name]
# The bundle contains the executable and the game files it needs (exheader.bin, code.bin, romfs.bin if any).
# Homebrew reading sdmc:/ finds it in ~/Library/Application Support/3dsrecomp/sdmc.
set -e
N=$1; D=$2; APP=${3:-$N}; B=${BUILD:-build}  # BUILD: the cmake build folder
[ -n "$N" ] && [ -f "$D/info.json" ] || { echo "usage: $0 <name> <gamedir> [App Name]"; exit 1; }
OUT="$B/$APP.app"
rm -rf "$OUT"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources/game"
cp "$B/$N" "$OUT/Contents/MacOS/$N"
cp "$D/exheader.bin" "$D/code.bin" "$D/info.json" "$OUT/Contents/Resources/game/"
[ -f "$D/romfs.bin" ] && cp "$D/romfs.bin" "$OUT/Contents/Resources/game/"
[ -f "$D/exefs/icon.bin" ] && mkdir -p "$OUT/Contents/Resources/game/exefs" && for x in icon banner logo; do [ -f "$D/exefs/$x.bin" ] && cp "$D/exefs/$x.bin" "$OUT/Contents/Resources/game/exefs/"; done
[ -d system_archives ] && cp -r system_archives "$OUT/Contents/Resources/game/"
# shipped pipeline and vertex program lists and their precompiled shaders
TID=$(python3 -c "import json;print(json.load(open('$D/info.json'))['program_id'])")
[ -f "games/$TID/pipelines.bin" ] && cp "games/$TID/pipelines.bin" "$OUT/Contents/Resources/game/"
[ -f "games/$TID/vsprograms.bin" ] && cp "games/$TID/vsprograms.bin" "$OUT/Contents/Resources/game/"
[ -f "$B/shaders.metallib" ] && cp "$B/shaders.metallib" "$OUT/Contents/Resources/"
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
