// Playing a two-screen game on one screen (runtime/onescreen.mm), for games whose profile says "onescreen": in
// the single-screen layouts the bottom screen never shows as such; its menus come up as a panel (frontend.cpp)
// and the rest is reached through overlays drawn over the top screen, with controls in the manner of a
// single-screen console: a hub of the bottom screen's buttons, its map enlarged, a ring of the tools in the
// pockets, and a pointer for touching the panel. The profile binds host inputs to them ("bind", "hub", "map",
// "tools" in runtime/profile.cpp).
#pragma once
#include <SDL3/SDL.h>

#include "rt.h"

namespace onescreen {

// host inputs, by position on a gamepad (the keyboard's keys stand for them as in the controls table)
enum : u32 {
    H_A = 1 << 0, H_B = 1 << 1, H_X = 1 << 2, H_Y = 1 << 3, H_L = 1 << 4, H_R = 1 << 5, H_ZL = 1 << 6,
    H_ZR = 1 << 7, H_START = 1 << 8, H_BACK = 1 << 9, H_LS = 1 << 10, H_RS = 1 << 11, H_DUP = 1 << 12,
    H_DDOWN = 1 << 13, H_DLEFT = 1 << 14, H_DRIGHT = 1 << 15, H_RSUP = 1 << 16, H_RSDOWN = 1 << 17,
    H_RSLEFT = 1 << 18, H_RSRIGHT = 1 << 19,
};
u32 host_mask(const char *names);  // "ZL+A" -> host bits (scripted input)

// each frame, before input(): whether one-screen play is on, and where the panel is (w 0: none), in render
// coordinates, with the top screen's place
void set_state(bool active, const SDL_FRect &top, const SDL_FRect &panel);
bool active();
bool overlay_open();  // the hub, the map or the tool ring is up (it takes the game's input)
bool busy();          // an overlay is playing inputs on the game (the panel stays hidden)

// host inputs (and the right stick, -1..1) -> the game's: binds, overlays, the pointer; buttons/cx/cy hold the
// default translation on entry (unbound inputs play their own 3DS button) and may be replaced
void input(u32 host, float rsx, float rsy, u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty);
// a click or tap at a render-coordinate point (true: an overlay used it)
bool click(float x, float y);

// a new bottom-screen picture (RGBA, w x h at the internal scale): the hub's icons are kept from it
void bottom_image(const u32 *img, int w, int h);

// over the top screen (after the panel): the overlays, the pointer
void draw(SDL_Renderer *r, SDL_Texture *bottom, float px);  // px: window pixels per render unit

// a filled rounded rectangle, textured with t (whole) if given
void rounded_rect(SDL_Renderer *r, SDL_Texture *t, const SDL_FRect &d, float rad, SDL_FColor c);

}  // namespace onescreen
