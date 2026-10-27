// The in-game menu (Esc, or the gamepad's Home button): saves, clock, graphics, speed and audio settings.
// It pauses the game while open and is drawn over the last frame (runtime/menu.mm).
#pragma once
#include <SDL3/SDL.h>

namespace menu {
void init(SDL_Window *window, SDL_Renderer *renderer);
bool is_open();
void toggle();
bool handle_event(const SDL_Event &ev);  // true if the menu used the event (the game does not see it)
void update();                           // once per host frame: gamepad stick navigation
void draw();                             // after the game screens, before SDL_RenderPresent
void set_fps(float fps);                 // the game's frame rate, for the counter
}  // namespace menu

// frontend.cpp: what the menu changes
void frontend_set_layout(int layout);
int frontend_layout_count();
const char *frontend_layout_name(int layout);
void frontend_set_fullscreen(bool on);
void frontend_set_split(bool on);  // separate windows, one per screen
void frontend_set_scale_mode(int mode);  // 0 auto, 1-4
int frontend_current_scale();            // the internal resolution in use (0 without Metal)
[[noreturn]] void frontend_restart();
[[noreturn]] void frontend_quit();
