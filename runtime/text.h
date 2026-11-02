// Text rasterised with Core Text (the system font) into SDL textures, for overlays drawn in window pixels
// (runtime/text.mm).
#pragma once
#include <SDL3/SDL.h>

#include <string>

namespace text {
struct Texture {
    SDL_Texture *tex = nullptr;  // premultiplied alpha
    int w = 0, h = 0;
};
Texture render(SDL_Renderer *r, const std::string &s, float px, SDL_Color c, bool bold);  // px: the font size
float width(const std::string &s, float px, bool bold);  // how wide render() would make it
}  // namespace text
