#!/bin/sh
# Make a Linux app folder:
#   tools/bundle_linux.sh <name> <gamedir> [App Name]
#     one game: <build>/<App Name>/ holds the executable (<name>), the game files it needs (resources/game/:
#     exheader.bin, code.bin, romfs.bin, the shipped pipeline lists and profile, system archives)
#   tools/bundle_linux.sh --games <name>... [--name "App Name"]
#     a multi-game build (cmake -DGAMES=... or -DGAMES=all): the `recomp` executable with its game list, each game's
#     files in resources/games/<name>/ (from data/<name>) and its code in lib/<name>.so
# with an icon from the (first) game's SMDH and a .desktop launcher for that folder (install it with
# `cp "<build>/<App Name>/<executable>.desktop" ~/.local/share/applications/`).
# Saves go to $XDG_DATA_HOME/3dsrecomp (~/.local/share/3dsrecomp). The libraries it links (SDL3, Vulkan, shaderc,
# FreeType, HarfBuzz, Fontconfig, libcurl, FFmpeg) are the system's. With appimagetool on the PATH, an AppImage of the
# folder is made too (<build>/<App Name>.AppImage).
set -e
B=${BUILD:-build}  # BUILD: the cmake build folder
GAMES=
if [ "$1" = --games ]; then
  shift
  N=recomp; APP="3DS Games"
  while [ $# -gt 0 ]; do
    if [ "$1" = --name ]; then APP=$2; shift 2; else GAMES="$GAMES $1"; shift; fi
  done
  [ -n "$GAMES" ] && [ -x "$B/$N" ] || { echo "usage: BUILD=<folder> $0 --games <name>... [--name \"App Name\"]"; exit 1; }
  D=data/$(echo $GAMES | cut -d' ' -f1)  # (the icon's game)
else
  N=$1; D=$2; APP=$3
  [ -n "$N" ] && [ -f "$D/info.json" ] && [ -x "$B/$N" ] || { echo "usage: $0 <name> <gamedir> [App Name] (BUILD=<build folder>)"; exit 1; }
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
OUT="$B/$APP"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$B/$N" "$OUT/$N"
# a game's files: copy_game <data dir> <destination>
copy_game() {
  mkdir -p "$2"
  cp "$1/exheader.bin" "$1/code.bin" "$1/info.json" "$2/"
  [ -f "$1/romfs.bin" ] && cp "$1/romfs.bin" "$2/"
  if [ -f "$1/exefs/icon.bin" ]; then
    mkdir -p "$2/exefs"
    for x in icon banner logo; do [ -f "$1/exefs/$x.bin" ] && cp "$1/exefs/$x.bin" "$2/exefs/"; done
  fi
  tid=$(python3 -c "import json;print(json.load(open('$1/info.json'))['program_id'])")
  for f in pipelines.bin profile.txt vsprograms.bin; do [ -f "games/$tid/$f" ] && cp -L "games/$tid/$f" "$2/"; done
  return 0
}
if [ -n "$GAMES" ]; then
  mkdir -p "$OUT/lib"
  for g in $GAMES; do
    [ -f "$B/$g.so" ] || { echo "no $B/$g.so (is $g in the build's GAMES?)"; exit 1; }
    copy_game "data/$g" "$OUT/resources/games/$g"
    cp "$B/$g.so" "$OUT/lib/"
  done
  [ -d system_archives ] && cp -r system_archives "$OUT/resources/games/"  # (shared: looked for next to each game's folder)
else
  copy_game "$D" "$OUT/resources/game"
  [ -d system_archives ] && cp -r system_archives "$OUT/resources/game/"
fi
# icon from the game's SMDH
ICON=""
if [ -f "$D/exefs/icon.bin" ] && command -v magick >/dev/null; then
  T=$(mktemp -d)
  python3 tools/smdh_icon.py "$D/exefs/icon.bin" "$T/icon.ppm"
  magick "$T/icon.ppm" -filter Mitchell -resize 256x256 "$OUT/$N.png"
  rm -rf "$T"
  ICON="$N"
fi
ABS=$(cd "$OUT" && pwd)
cat > "$OUT/$N.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$APP
Exec="$ABS/$N"
Path=$ABS
Icon=${ICON:+$ABS/$N.png}
Categories=Game;
Terminal=false
DESKTOP
chmod +x "$OUT/$N.desktop"
echo "built $OUT"
# an AppImage (appimagetool: https://github.com/AppImage/appimagetool)
if command -v appimagetool >/dev/null; then
  A="$B/$APP.AppDir"
  rm -rf "$A"
  cp -r "$OUT" "$A"
  printf '#!/bin/sh\nexec "$(dirname "$(readlink -f "$0")")/%s" "$@"\n' "$N" > "$A/AppRun"
  chmod +x "$A/AppRun"
  sed -e "s|^Exec=.*|Exec=$N|" -e "/^Path=/d" -e "s|^Icon=.*|Icon=$N|" "$OUT/$N.desktop" > "$A/$N.desktop"
  [ -n "$ICON" ] || magick -size 256x256 xc:gray "$A/$N.png" 2>/dev/null || true
  ARCH=$(uname -m) appimagetool "$A" "$B/$APP.AppImage" >/dev/null && echo "built $B/$APP.AppImage"
  rm -rf "$A"
fi
