// Text rasterised in the system font into SDL textures, for overlays drawn in window pixels: Core Text on Apple
// (runtime/text.mm), FreeType and HarfBuzz with the fonts Fontconfig picks elsewhere (runtime/text_freetype.cpp).
#pragma once
#include <SDL3/SDL.h>

#include <string>

namespace text {
struct Texture {
    SDL_Texture *tex = nullptr;  // premultiplied alpha
    int w = 0, h = 0;
};
Texture render(SDL_Renderer *r, const std::string &s, float px, SDL_Color c, bool bold);  // px: the font size
// white bold text with a dark outline (22% of px wide, centred on the glyphs' edges) and pad pixels of margin all
// round (the extra HUD's numbers)
Texture render_outlined(SDL_Renderer *r, const std::string &s, float px, int pad);
float width(const std::string &s, float px, bool bold);  // how wide render() would make it
}  // namespace text
