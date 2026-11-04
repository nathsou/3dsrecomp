// Aiming with the right stick and the mouse for games whose profile describes their aim ("aim" in
// runtime/profile.cpp): the game's own stick aiming is switched on in its pad state and fed the right stick, and the
// mouse's motion is added to the aim delta of the frame being computed, both on the game's thread as it makes the
// calls the profile names. Nothing goes through the touch screen.
#pragma once

namespace aim {

bool configured();
// once per frame (UI thread), after profile::update(): the right stick (-1..1, up positive) and the mouse's motion
// since the last frame (top-screen pixels); true while aiming applies (the mouse is then captured)
bool update(float rsx, float rsy, float mouse_dx, float mouse_dy);
bool active();

}  // namespace aim
