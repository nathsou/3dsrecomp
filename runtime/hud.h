// The game profile's HUD (runtime/profile.h), drawn over the top screen with the game's own icons.
#pragma once
#include <SDL3/SDL.h>

#include <string>
#include <vector>

#include "rt.h"

namespace hud {
void draw(SDL_Renderer *r, const SDL_FRect &top);  // top: where the top screen is, in render coordinates
// an image from the game's files (a BCLIM, BCFNT or .bctex, in a NARC/SARC/DARC archive when inner is given) as
// top-down RGBA rows
bool load_image(const std::string &file, const std::string &inner, std::vector<u32> &px, int &w, int &h);
}
