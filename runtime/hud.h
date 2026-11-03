// The game profile's HUD (runtime/profile.h), drawn over the top screen with the game's own icons.
#pragma once
#include <SDL3/SDL.h>

namespace hud {
void draw(SDL_Renderer *r, const SDL_FRect &top);  // top: where the top screen is, in render coordinates
}
