// On-screen controls for touch screens (iPhone, iPad; RECOMP_TOUCH=1 tries them on a Mac, the mouse being a
// finger): a circle pad, A B X Y, L R, Start, Select, a menu button and, when one screen shows at a time, a button
// that swaps the screens. They go in the room the screens leave; fingers elsewhere touch the bottom screen.
// Drawn with SDL's renderer in window pixels, labels with Core Text (runtime/touch.mm).
#pragma once
#include <SDL3/SDL.h>

#include <cstdint>

namespace touch {
bool enabled();
void init(SDL_Window *window, SDL_Renderer *renderer);
// the part of the window (in points) left for the screens
SDL_FRect screen_area();
// where a window point lands on the bottom screen, in its pixels (set by the frontend); false if not on it
void set_bottom_screen_mapper(bool (*map)(float x, float y, int &tx, int &ty));
// the corner screen of the single-screen layouts (set by the frontend): what a window point is on (0 nothing,
// 1 its picture of the bottom screen, 2 what moves it: its grip or a picture of the top screen, 3 its resize
// handle), and a gesture on it (mode 1 move, 2 resize from the handle, 3 pinch; phase 0 begin, 1 move, 2 end;
// the second point: a pinch's other finger)
void set_pip_hooks(int (*at)(float x, float y), void (*gesture)(int mode, int phase, float x0, float y0, float x1, float y1));
bool handle_event(const SDL_Event &ev);  // finger events: true if used
void reset();                            // lift every finger (the menu opened)
void gamepad(bool in_use);               // a gamepad plays: hide the controls (until the next touch)
void set_swap_shown(bool on);            // the swap button (single-screen layouts)
bool take_menu();                        // the menu button was tapped (once)
bool take_swap();                        // the swap button was tapped (once)
// adds what the fingers hold: 3DS button bits (frontend.cpp's), the circle pad, the bottom screen's touch
void poll(uint32_t &buttons, int16_t &cx, int16_t &cy, bool &touched, uint16_t &tx, uint16_t &ty);
void draw();  // after the screens, before the menu
}  // namespace touch
