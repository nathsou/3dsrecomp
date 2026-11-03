// Stacks in the pockets, as in Animal Crossing: New Horizons, for games whose profile has "stack" lines (an option
// in the menu, off by default): items of the profile's categories pile up in one pocket. The game knows only the
// pocket's one item; the extra ones are kept here, for each owner, and written next to the save (savedata/
// recomp_stacks.txt) whenever the game writes it, so they stay in step with what it saved.
#pragma once
#include <SDL3/SDL.h>

#include <string>

namespace stacks {

void init();  // after the profile is loaded
// the game wrote this file in its save (any thread): the stacks as they are now go next to it
void save_written(const std::string &path);
// over the bottom screen (d: where it is drawn, px: window pixels per render unit): the stacks' counts on their
// pockets while the pockets are open
void draw(SDL_Renderer *r, const SDL_FRect &d, float px, float alpha);

}  // namespace stacks
