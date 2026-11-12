// A free camera on the right stick for games of Ocarina of Time 3D's engine whose profile describes it ("camera" in
// runtime/profile.cpp), as Majora's Mask 3D has on the C-stick: the stick turns the camera around Link, and the
// game's own camera takes over again when it is needed (Z-targeting, first person, cutscenes...).
#pragma once

namespace freecam {

bool configured();
// once per frame (UI thread), after profile::update(): the right stick (-1..1, up positive); true while the free
// camera applies (the stick is then the camera's)
bool update(float rsx, float rsy);

}  // namespace freecam
