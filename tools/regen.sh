#!/bin/sh
# Regenerate the C for a game: tools/regen.sh <gamedir> <name>
set -e
D=$1; N=$2
TID=$(python3 -c "import json;print(json.load(open('$D/info.json'))['program_id'])")
# fold in code that a run found missing (runs from a build or from the app bundle)
mkdir -p games/$TID; touch games/$TID/functions.txt
for M in save/missing_functions.txt "$HOME/Library/Application Support/3dsrecomp/missing_functions.txt"; do
  [ -f "$M" ] && sort -u "$M" >> games/$TID/functions.txt && rm "$M"
done
sort -u -o games/$TID/functions.txt games/$TID/functions.txt
PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python  # the recompiler needs only the standard library
$PY -m tools.recomp.gen $D generated/$N --extra games/$TID/functions.txt
