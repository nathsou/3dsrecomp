// Per-game profiles (games/<title id>/profile.txt): which screen matters in which part of a game, and a HUD
// drawn over the top screen from values read in guest memory. The file format is described in profile.cpp.
#pragma once
#include <string>
#include <vector>

#include "rt.h"

namespace profile {

enum Focus { FOCUS_NONE, FOCUS_TOP, FOCUS_BOTTOM, FOCUS_BOTH };

// a HUD element, in top-screen pixels (negative x / y: from the right / bottom edge)
struct Element {
    enum Kind { COUNTER, PIPS, PICK, ICON } kind = ICON;
    int x = 0, y = 0;
    std::string icon, icon_off;            // COUNTER / PIPS / ICON: icon names
    int value = -1;                        // index of the value shown
    int digits = 0;                        // COUNTER: zero-padded width (0: as is)
    int count = 0;                         // PIPS: how many
    std::vector<std::pair<s64, std::string>> picks;  // PICK: value -> icon name
    int cond = -1;                         // shown when this condition holds (-1: always)
};

struct Icon {
    std::string name, file, inner;  // RomFS file, and the texture inside it (SARC archives, Yaz0 or not)
    int rx = 0, ry = 0, rw = 0, rh = 0;  // part of the texture (w 0: all of it)
    float scale = 1;                // drawn size relative to the texture's pixels
};

struct Button {
    int tx = 0, ty = 0;  // bottom-screen pixel the button taps
    int cond = -1;
};

bool load();              // games/<title id>/profile.txt (or RECOMP_PROFILE); false if the game has none
bool loaded();
bool has_focus_rules();
bool has_hud();
void update();            // once per frame (UI thread): evaluates every value and condition
Focus focus();            // the first focus rule that holds (FOCUS_NONE: none does)
bool hud_visible();
const std::vector<Element> &elements();
const std::vector<Icon> &icons();
bool cond(int i);         // a condition's state this frame
bool value(int i, s64 &v);  // a value this frame (false: its pointer chain was null)
const Button *button();   // the HUD action button, if it applies now
bool keyboard_active();   // the game's own keyboard is up ("keyboard when ...")
bool key(const std::string &k, int &x, int &y);  // where a character or named key is on the bottom screen

}  // namespace profile
