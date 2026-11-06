#!/bin/sh
# Make a Linux app folder: tools/bundle_linux.sh <name> <gamedir> [App Name]
# <build>/<App Name>/ holds the executable (<name>), the game files it needs (resources/game/: exheader.bin, code.bin,
# romfs.bin, the shipped pipeline lists and profile, system archives), an icon from the game's SMDH and a .desktop
# launcher for that folder (install it with `cp "<build>/<App Name>/<name>.desktop" ~/.local/share/applications/`).
# Saves go to $XDG_DATA_HOME/3dsrecomp (~/.local/share/3dsrecomp). The libraries it links (SDL3, Vulkan, shaderc,
# FreeType, HarfBuzz, Fontconfig, libcurl, FFmpeg) are the system's. With appimagetool on the PATH, an AppImage of the
# folder is made too (<build>/<App Name>.AppImage).
set -e
N=$1; D=$2; APP=${3:-$N}; B=${BUILD:-build}  # BUILD: the cmake build folder
[ -n "$N" ] && [ -f "$D/info.json" ] && [ -x "$B/$N" ] || { echo "usage: $0 <name> <gamedir> [App Name] (BUILD=<build folder>)"; exit 1; }
OUT="$B/$APP"
rm -rf "$OUT"
mkdir -p "$OUT/resources/game"
cp "$B/$N" "$OUT/$N"
G="$OUT/resources/game"
cp "$D/exheader.bin" "$D/code.bin" "$D/info.json" "$G/"
[ -f "$D/romfs.bin" ] && cp "$D/romfs.bin" "$G/"
if [ -f "$D/exefs/icon.bin" ]; then
  mkdir -p "$G/exefs"
  for x in icon banner logo; do [ -f "$D/exefs/$x.bin" ] && cp "$D/exefs/$x.bin" "$G/exefs/"; done
fi
[ -d system_archives ] && cp -r system_archives "$G/"
TID=$(python3 -c "import json;print(json.load(open('$D/info.json'))['program_id'])")
for f in pipelines.bin profile.txt vsprograms.bin; do [ -f "games/$TID/$f" ] && cp "games/$TID/$f" "$G/"; done
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
