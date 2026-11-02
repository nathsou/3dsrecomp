// (iOS) Keeps SDL's view filling its window at the screen's scale (runtime/uiview.mm). Held sideways, the keyboard
// showing or hiding (a game's own keyboard asking for text) left it a portrait rectangle in the landscape window: the
// picture in a corner and the touch controls off screen until the app restarted. SDL lays the view out for the
// keyboard from the deprecated statusBarOrientation, which reads "unknown" in an app with scenes, and so takes the
// window for a portrait one.
#pragma once
#include <SDL3/SDL.h>

namespace uiview {
// once a frame, on the main thread: puts the view of the renderer's window back when it no longer fills it (and
// logs what was off); false when there was nothing to do (also on other systems)
bool keep_filling(SDL_Renderer *renderer);
}  // namespace uiview
