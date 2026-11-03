// Per-game profiles (games/<title id>/profile.txt): which screen matters in which part of a game, and a HUD
// drawn over the top screen from values read in guest memory. The file format is described in profile.cpp.
#pragma once
#include <string>
#include <vector>

#include "rt.h"

namespace profile {

enum Focus { FOCUS_NONE, FOCUS_TOP, FOCUS_BOTTOM, FOCUS_BOTH, FOCUS_PANEL };

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

// bits in a save file that say a session ended without saving ("unsaved"), in a block led by its CRC-32C
struct UnsavedMark {
    std::string file;  // in the save data
    u32 offset;
    u8 bits;
    u32 block, size;   // the CRC's offset, and how many bytes after it it covers
};

struct Button {
    int tx = 0, ty = 0;  // bottom-screen pixel the button taps
    int cond = -1;
};

// one-screen controls (runtime/onescreen.mm): host inputs bound to 3DS buttons or overlays, the hub's entries,
// the map's part of the bottom screen, the tool ring's sources
struct Bind {
    std::string host;    // a gamepad input by position: a b x y l r zl zr start back ls rs dup ddown dleft dright
                         // rs-up rs-down rs-left rs-right
    std::string target;  // hub map tools tool-next tool-prev tool-away pointer none, or 3DS buttons (A+B, DUP, ...)
    u32 buttons = 0;     // the 3DS buttons, when it is buttons
};
struct HubEntry {
    std::string label;
    enum Action { TAP, PRESS, MAP, TOOLS } action = TAP;
    int x = 0, y = 0;       // TAP: bottom-screen pixel
    u32 buttons = 0;        // PRESS
    int ix = 0, iy = 0, iw = 0, ih = 0;  // its icon: a part of the bottom screen (w 0: none)
    std::string glyph;      // or a character drawn as its icon
    int cond = -1;          // offered when this holds
};
struct MapArea {
    int x = 0, y = 0, w = 0, h = 0;  // bottom-screen pixels
    int cond = -1;
};
struct Tools {
    int pockets = -1, held = -1;  // values: the first pocket (items 4 bytes apart), the item in hand
    u32 empty = 0x7ffe;           // the empty item
    std::vector<std::pair<int, int>> slots;  // where each pocket is on the bottom screen while the pockets are open
    int portrait_x = 0, portrait_y = 0;      // where an item is dropped to hold it
    u32 open = 0;                 // the 3DS buttons that open the pockets
    int open_cond = -1;           // the pockets are open
    int ready_cond = -1;          // the tool ring can be used (the player is free to move)
    std::string items;            // item records (RomFS): a category byte per item, from "first"
    u32 record = 0, first = 0, category_off = 0, kind_lo = 0, kind_hi = 0;
    int icon_size = 28;           // items' icons: squares this big around the pockets, kept while they show
    std::string names;            // item names (RomFS MSBT, or several in a umsbt: the first)
    u32 hold_fn = 0, hold_on = 0;  // the game's own "hold this item" (fn(&item, 0, 0)), called when its thread
                                   // dispatches to hold_on; else the pockets are worked through out of sight
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
bool value_address(int i, u32 &a);  // where it was read this frame (guest memory values)
const Button *button();   // the HUD action button, if it applies now
bool keyboard_active();   // the game's own keyboard is up ("keyboard when ...")
bool key(const std::string &k, int &x, int &y);  // where a character or named key is on the bottom screen
const std::vector<UnsavedMark> &unsaved_marks();
void set_runtime(const std::string &flag, s64 v);  // a "value <name> = runtime <flag>" (read at the next update)
bool onescreen();         // "onescreen": no corner screen in the single-screen layouts
const std::vector<Bind> &binds();
const std::vector<HubEntry> &hub();
const std::vector<MapArea> &map_areas();
const Tools &tools();
u32 button_mask(const std::string &names);  // "A+B", "DUP", ... as 3DS button bits (0: none known)

}  // namespace profile
