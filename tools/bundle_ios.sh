#!/bin/sh
# Make an iPhone / iPad app bundle:
#   BUILD=<ios build folder> tools/bundle_ios.sh <name> <gamedir> [App Name]
#     one game: the executable and the game files (as tools/bundle_macos.sh, in game/ next to the executable)
#   BUILD=<ios build folder> tools/bundle_ios.sh --games <name>... [--name "App Name"]
#     a multi-game build (cmake -DGAMES=...): the `recomp` executable with its game list, each game's files in
#     games/<name>/ (from data/<name>) and its code in Frameworks/<name>.dylib
# For the simulator it is signed ad hoc: xcrun simctl install <device> "<build>/<App Name>.app".
# For a device, IDENTITY (a signing identity, e.g. "Apple Development: ...") and PROFILE (a provisioning profile
# for BUNDLE_ID) sign it with the profile's entitlements; install it with Xcode's Devices window or
# `xcrun devicectl device install app`. A profile with the Extended Virtual Addressing capability (paid accounts)
# makes sure the 4 GiB the games reserve for their address space is granted.
set -e
B=${BUILD:?BUILD: the iOS cmake build folder}
if [ "$1" = --games ]; then
  shift
  EXE=recomp; APP="3DS Games"; GAMES=
  while [ $# -gt 0 ]; do
    if [ "$1" = --name ]; then APP=$2; shift 2; else GAMES="$GAMES $1"; shift; fi
  done
  [ -n "$GAMES" ] && [ -f "$B/$EXE" ] || { echo "usage: BUILD=<folder> $0 --games <name>... [--name \"App Name\"]"; exit 1; }
else
  EXE=$1; D=$2; APP=${3:-$1}
  [ -n "$EXE" ] && [ -f "$D/info.json" ] && [ -f "$B/$EXE" ] || { echo "usage: BUILD=<folder> $0 <name> <gamedir> [App Name]"; exit 1; }
fi
SYSROOT=$(sed -n 's/^CMAKE_OSX_SYSROOT:[A-Z]*=//p' "$B/CMakeCache.txt")
case "$SYSROOT" in *[Ss]imulator*) PLATFORM=iPhoneSimulator ;; *) PLATFORM=iPhoneOS ;; esac
BUNDLE_ID=${BUNDLE_ID:-local.3dsrecomp.$EXE}
OUT="$B/$APP.app"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$B/$EXE" "$OUT/$EXE"
T=$(mktemp -d)

# a game's files: copy_game <data dir> <destination> <its precompiled shaders>
copy_game() {
  mkdir -p "$2"
  cp "$1/exheader.bin" "$1/code.bin" "$1/info.json" "$2/"
  [ -f "$1/romfs.bin" ] && cp "$1/romfs.bin" "$2/"
  [ -f "$1/exefs/icon.bin" ] && mkdir -p "$2/exefs" && for x in icon banner logo; do [ -f "$1/exefs/$x.bin" ] && cp "$1/exefs/$x.bin" "$2/exefs/"; done
  local tid=$(python3 -c "import json;print(json.load(open('$1/info.json'))['program_id'])")
  for f in pipelines.bin profile.txt vsprograms.bin; do [ -f "games/$tid/$f" ] && cp "games/$tid/$f" "$2/"; done
  [ -f "$3" ] && cp "$3" "$2/shaders.metallib"
  return 0
}
ICON_SRC=
if [ -n "$GAMES" ]; then
  mkdir -p "$OUT/Frameworks"
  for g in $GAMES; do
    [ -f "$B/$g.dylib" ] || { echo "no $B/$g.dylib (is $g in the build's GAMES?)"; exit 1; }
    copy_game "data/$g" "$OUT/games/$g" "$B/shaders_$g.metallib"
    cp "$B/$g.dylib" "$OUT/Frameworks/"
    [ -f "data/$g/exefs/icon.bin" ] && python3 tools/smdh_icon.py "data/$g/exefs/icon.bin" "$T/$g.ppm" && ICON_SRC="$ICON_SRC $T/$g.ppm"
  done
  [ -d system_archives ] && cp -r system_archives "$OUT/games/"  # (shared: the runtime looks next to each game's folder)
else
  copy_game "$D" "$OUT/game" "$B/shaders.metallib"
  [ -d system_archives ] && cp -r system_archives "$OUT/game/"
  [ -f "$D/exefs/icon.bin" ] && python3 tools/smdh_icon.py "$D/exefs/icon.bin" "$T/icon.ppm" && ICON_SRC="$T/icon.ppm"
fi
# home screen icon: the game's own, or the first four games' in a square
if [ -n "$ICON_SRC" ] && command -v magick >/dev/null; then
  set -- $ICON_SRC
  if [ $# -ge 4 ]; then
    magick \( "$1" "$2" +append \) \( "$3" "$4" +append \) -append "$T/app.ppm"
  else
    cp "$1" "$T/app.ppm"
  fi
  magick "$T/app.ppm" -filter Mitchell -resize 120x120 "$OUT/AppIcon60x60@2x.png"
  magick "$T/app.ppm" -filter Mitchell -resize 180x180 "$OUT/AppIcon60x60@3x.png"
  magick "$T/app.ppm" -filter Mitchell -resize 152x152 "$OUT/AppIcon76x76@2x~ipad.png"
  magick "$T/app.ppm" -filter Mitchell -resize 167x167 "$OUT/AppIcon83.5x83.5@2x~ipad.png"
  ICONS='<key>CFBundleIcons</key><dict><key>CFBundlePrimaryIcon</key><dict><key>CFBundleIconFiles</key><array><string>AppIcon60x60</string></array></dict></dict>
  <key>CFBundleIcons~ipad</key><dict><key>CFBundlePrimaryIcon</key><dict><key>CFBundleIconFiles</key><array><string>AppIcon60x60</string><string>AppIcon76x76</string><string>AppIcon83.5x83.5</string></array></dict></dict>'
fi
ORIENT='<string>UIInterfaceOrientationPortrait</string><string>UIInterfaceOrientationLandscapeLeft</string><string>UIInterfaceOrientationLandscapeRight</string>'
cat > "$OUT/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleExecutable</key><string>$EXE</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleName</key><string>$APP</string>
  <key>CFBundleDisplayName</key><string>$APP</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleSupportedPlatforms</key><array><string>$PLATFORM</string></array>
  <key>MinimumOSVersion</key><string>16.0</string>
  <key>LSRequiresIPhoneOS</key><true/>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
  <key>UILaunchScreen</key><dict/>
  <key>UIRequiresFullScreen</key><true/>
  <key>UIStatusBarHidden</key><true/>
  <key>UIApplicationSupportsIndirectInputEvents</key><true/>
  <key>UISupportedInterfaceOrientations</key><array>$ORIENT</array>
  <key>UISupportedInterfaceOrientations~ipad</key><array>$ORIENT<string>UIInterfaceOrientationPortraitUpsideDown</string></array>
  <key>GCSupportsControllerUserInteraction</key><true/>
  <key>GCSupportedGameControllers</key><array><dict><key>ProfileName</key><string>ExtendedGamepad</string></dict></array>
  $ICONS
</dict></plist>
PLIST
printf 'APPL????' > "$OUT/PkgInfo"
if [ "$PLATFORM" = iPhoneOS ] && [ -n "$IDENTITY" ] && [ -n "$PROFILE" ]; then
  cp "$PROFILE" "$OUT/embedded.mobileprovision"
  security cms -D -i "$PROFILE" > "$T/profile.plist"
  plutil -extract Entitlements xml1 -o "$T/entitlements.plist" "$T/profile.plist"
  # as Xcode does: the profile's wildcard ("TEAM.*", e.g. the keychain group) becomes this app's id
  python3 - "$T/entitlements.plist" "$BUNDLE_ID" <<'PY'
import plistlib, sys
p, bid = sys.argv[1], sys.argv[2]
e = plistlib.load(open(p, 'rb'))
team = e['com.apple.developer.team-identifier']
fix = lambda v: [fix(x) for x in v] if isinstance(v, list) else v.replace(team + '.*', team + '.' + bid) if isinstance(v, str) else v
plistlib.dump({k: fix(v) for k, v in e.items()}, open(p, 'wb'))
PY
  for m in "$OUT"/Frameworks/*.dylib; do [ -f "$m" ] && codesign --force --sign "$IDENTITY" --timestamp=none "$m"; done
  codesign --force --sign "$IDENTITY" --entitlements "$T/entitlements.plist" --timestamp=none "$OUT"
elif [ "$PLATFORM" = iPhoneOS ]; then
  echo "not signed for a device: set IDENTITY and PROFILE (see the top of $0)"
else
  for m in "$OUT"/Frameworks/*.dylib; do [ -f "$m" ] && codesign --force --sign - --timestamp=none "$m"; done
  codesign --force --sign - --timestamp=none "$OUT"
fi
rm -rf "$T"
echo "built $OUT"
