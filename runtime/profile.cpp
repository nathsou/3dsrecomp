// Per-game profiles. A profile is a text file, one directive per line ('#' starts a comment):
//
//   symbol <name> = <start> [<step>]...
//       A number found in the game rather than written down, so that one profile serves every region and version of
//       a game. Wherever a number goes, a symbol's name may stand instead (defined on an earlier line). Start:
//         vtable <class>   the address a class's objects hold as their vtable pointer, from the game's RTTI
//                          (classes outside namespaces with one primary vtable, CRO modules' too)
//         string "<text>"  the address of that NUL-terminated text in the program's read-only data (its first
//                          occurrence that starts a string): with "user", the functions using it, for games
//                          built without RTTI (a RomFS path or a message the code near a value refers to)
//         code [thumb] "<hex bytes>"
//                          the first place in the program's code holding those bytes ("??" matches any byte;
//                          "thumb": Thumb code, bit 0 set): a function found by its own instructions
//         <symbol> | <n>   an earlier symbol, or a number
//       Steps, each on the address so far:
//         slot <n>         the function in its n-th word (a vtable's n-th virtual function)
//         call <k>         the k-th function the function there calls (bl / blx, in order, from 0)
//         global <k>       the k-th distinct address in the program's data or bss that the function there loads from
//                          its literal pool (its static variables, in order of first use)
//         user <k>         the k-th function (in address order) whose literal pool holds the address (for a
//                          vtable: the class's constructors, destructors and factories)
//         + <n>, - <n>     an offset
//         > <n>            the word there, plus n (a pointer in the program's initialised data)
//       A function is read from its start up to its first literal pool. A symbol that cannot be found is an error
//       (the directives using it are left out), and the log lists every symbol's value.
//   region <JPN|USA|EUR|AUS|CHN|KOR|TWN>[,...] <directive>
//       The directive only for those regions (the game's own, from its icon's region lockout): e.g. keyboards
//       whose keys differ.
//   value <name> = <type> <address> [> <offset>]...
//       A number in guest memory. Types: u8 u16 u32 s8 s16 s32 f32 (f32 is truncated to an integer). Each
//       "> off" follows a pointer: the address so far is read as a u32 and off is added (a null or unmapped
//       pointer makes the value absent, and every condition using it false). Numbers: decimal or 0x hex.
//   value <name> = runtime <flag>
//       A number the runtime sets: "single" is 1 while one screen is shown (the single-screen layouts),
//       "bottom_dark" the share (percent) of the bottom screen's picture that is dark and opaque.
//   set <value> = <number> [when <condition>]
//       Writes the number to the value's place in guest memory every frame while the condition holds (data
//       changes only: e.g. a layout pane's visible flag).
//   focus top|bottom|both|panel [after <frames>] [when <condition>]
//       Which screen matters (first rule that holds wins; none: the user's own layout). "both": the top
//       screen with the bottom one shown beside it (picture in picture). "panel": in the single-screen layouts,
//       the bottom screen over the top one, as a card over the dimmed picture (or floating on its own
//       transparency when its background is hidden, see "set"), nothing when it shows nothing. "after": the
//       condition must have held that many frames in a row (a page whose first frames are not to be shown).
//   bind <host input> <target>
//       One-screen controls (in the single-screen layouts with "onescreen"): a gamepad input by position (a b x y
//       l r zl zr start back ls rs dup ddown dleft dright rs-up rs-down rs-left rs-right; the keyboard's keys
//       stand for them as in the controls table) plays 3DS buttons (A+B, DUP, CUP, ...) or opens an overlay:
//       hub, map, tools (the ring), tool-next / tool-prev (hold the next or previous tool in the pockets), tool-away
//       (put the held tool away); "pointer" makes the right stick move a pointer over the bottom screen's panel, and the
//       input touch where it points; "none" does nothing. Unbound inputs play their own 3DS button.
//   hub "<label>" tap <x> <y> | press <buttons> | map | tools [icon <x> <y> <w> <h> | glyph "<c>"] [when <cond>]
//       An entry of the hub (a phone-like launcher): tap the bottom screen there, press 3DS buttons, or open the
//       map or the tool ring. Its icon is that part of the bottom screen, kept from the last time it showed, or a
//       character (the map's entry shows the map, the tools' entry the tool in hand).
//   map <x> <y> <w> <h> [when <condition>]
//       The map overlay: that part of the bottom screen, live and enlarged (first area whose condition holds).
//   tools pockets <value> held <value> [empty <item>]
//   tools items <romfs file> record <bytes> first <item> category <offset> kinds <lo> <hi>
//   tools icon <size>
//   tools textures <darc> <prefix> field <offset> plus <n> [<item>=<number>]...
//       The items' icons from the game's own textures: in the RomFS darc, <prefix><number>.bclim (32x32 ETC1A4),
//       <number> being the u16 at <offset> in the item's record plus <n>, or the one given for an item. (Before
//       the pockets show: icons kept from them only once they have.)
//   tools names <romfs msbt/umsbt>
//   tools slots <x> <y> ...
//   tools equip <portrait x> <portrait y> open <buttons> when-open <value> <op> <n> ready <value> <op> <n>
//       The tool ring: the pockets (16 items, 4 bytes apart, from the first one's value) whose category is a
//       tool kind, with their names from the game's own files and their icons as the pockets show them (squares
//       of that size around each pocket, kept while the pockets are open); choosing one opens the pockets out of
//       sight, taps it and picks the first choice ("Hold"); putting the held one away taps the portrait.
//   tools hold <function> on <address>
//       Tools are held with the game's own request instead: <function>(&item, 0, 0) (true: the change of tools
//       starts; the empty item puts the held one away), called on the game's thread as it dispatches to <address>
//       (an indirect call, e.g. the player's update through its vtable). The pocket gets the tool that was in hand
//       once the new one is.
//   stack kinds <category> ... max <n>
//   stack owner <value> ...
//   stack on <address>
//   stack free <offset> <state> ...
//   stack spread <state> ... keep <n>
//   stack shown when <cond>
//   stack underfoot <offset>
//       Stacks in the pockets, as in Animal Crossing: New Horizons (an option in the menu): items of those
//       categories (as "tools items" reads them) pile up to n in one pocket, the extra ones kept by the runtime
//       for each owner (the values tell players and towns apart) and written next to the save whenever the game
//       writes it. The game's thread tidies the pockets as it dispatches to <address>, while the player is free
//       and the pockets are closed ("tools equip ... when-open ... ready"): a second pocket with the same item joins the first one's stack, and a stack's
//       pocket that empties (the item dropped, sold, given, planted) gets the next one. With "stack free", the
//       player is free while the byte at <offset> in the object called there (its first argument: the player) is
//       one of the states (standing, walking: indoors too, where the ready condition may not tell). With
//       "stack spread", while that byte is one of these states (a conversation: a shop's counter, a villager), the
//       stacks' extra items are laid out in free pockets as it starts, n of them left free for what the
//       conversation may give, so the game's own screens can take several at once (all of a stack sold in one
//       sale); they gather back into their stacks once the player is free. "stack shown" draws the counts while
//       its condition holds too, for screens other than the pockets that show them in the same places. "stack
//       underfoot" names where the player keeps the item it stands on (the 2 bytes at <offset>): with the pockets
//       full, a stack of that item with room gives up its pocket while the player stands there, so the game picks
//       the item up into it and it joins the stack (it goes back once the player steps off).
//   aim delta <return address> <x offset> <y offset> [per s<n>] [active r<n> <offset>] on <address> ...
//   aim stick <pixels per second>
//   aim when <condition>
//       Aiming with the right stick and the mouse (runtime/aim.cpp), while the condition holds: as the game's thread
//       dispatches to one of the "on" addresses returning to the return address (a call made while the frame's aim
//       delta sits on the stack, as two floats at sp + the offsets), the mouse's motion since the last frame and the
//       right stick (at most "stick" pixels per second, default 300) are added to it. "per s<n>": the delta is a
//       rate the game multiplies by the frame time it holds in that VFP register (the mouse's motion is divided by
//       it). "active": the byte at that register + offset (the game's own "aim input this frame") is set while the
//       stick is held or the mouse moved lately. The mouse is captured meanwhile.
//   clearbottom
//       Render targets are cleared to transparent after each display transfer (before they are drawn into again):
//       for games whose bottom screen's pass reuses the top screen's target under a background, which "set" hides.
//   onescreen
//       In the single-screen layouts the bottom screen never shows in a corner: the profile's panel and overlays
//       (hub, map, tools) stand in for it.
//   hud [when <condition>]
//       When the HUD is drawn over the top screen.
//   icon <name> <romfs path>[:<file inside a NARC/SARC/DARC archive>] [rect <x> <y> <w> <h>] [scale <s>]
//       An image from the game's own files: a BCLIM texture, a BCFNT font's glyph sheets stacked top to bottom, or
//       an MTXT/CTPK .bctex texture
//       (archives may be Yaz0-compressed).
//   counter <icon> <value> at <x> <y> [digits <n>] [when <condition>]
//       The icon, then "x" and the value.
//   pips <icon on> <icon off> <value> <count> at <x> <y> [when <condition>]
//       count icons in a row, "on" where bit i of the value is set.
//   pick <value> at <x> <y> <n>=<icon> ... [when <condition>]
//       The icon listed for the value's current number (nothing for others).
//   show <icon> at <x> <y> [when <condition>]
//   button tap <x> <y> [when <condition>]
//       What the HUD action button (keyboard E, gamepad left stick click) does: tap the bottom screen there.
//   keyboard [when <condition>]
//       When the game's own keyboard is on the bottom screen: the Mac's keyboard then types by tapping its keys.
//   key <characters | space | backspace | shift | enter | escape> at <x> <y> [step <dx>]
//       Where the keys are, in bottom-screen pixels (320x240): one named key, or a row of characters <dx> apart.
//       A letter typed in upper case taps "shift" first; Return taps "enter", Esc "escape" (closing the keyboard).
//   unsaved <save file> <offset> <bits> crc32c <block> <size>
//       Bits the game sets in its save when a session starts and clears when it saves, which a session that ends
//       without a save leaves set (Animal Crossing scolds the player for it). With the menu's "Forgive unsaved
//       quits" on they are cleared at launch, and the CRC-32C at <block> (of the <size> bytes after it) updated;
//       nothing is changed unless that CRC is right to begin with.
//
// Positions are top-screen pixels (400x240); a negative x or y counts from the right or bottom edge.
// Conditions: <value> <op> <number> [and|or ...], with op one of == != < <= > >= & (any bit set) !& (none set);
// "and" binds tighter than "or".
// RECOMP_PROFILE=<file> loads another profile; RECOMP_PROFILE_TRACE=<n> logs every value and the focus every
// n frames (for writing profiles).
#include "profile.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include "gpu_iface.h"
#include "kernel.h"
#include "loader.h"

// the game's classes (tools/recomp/rtti.py), for symbols
#ifdef RECOMP_MODULES  // several games in one app: the loaded game's (set_game_classes)
static const GuestClass *game_classes;
static u32 game_class_count;
void set_game_classes(const GuestClass *c, u32 count) {
    game_classes = c;
    game_class_count = count;
}
#else
extern "C" {
extern const GuestClass guest_classes[];
extern const u32 guest_class_count;
__attribute__((weak)) const GuestClass guest_classes[1] = {};
__attribute__((weak)) const u32 guest_class_count = 0;
}
static const GuestClass *const game_classes = guest_classes;
static const u32 &game_class_count = guest_class_count;
#endif

namespace profile {
namespace {

struct Value {
    std::string name;
    char type = 'I';  // 'b' u8, 'h' u16, 'I' u32, 'c' s8, 's' s16, 'i' s32, 'f' f32; 'r' a runtime flag
    std::string flag;  // 'r': its name
    u32 addr = 0;
    u32 at = 0;        // where it was read (the end of the pointer chain), when ok
    std::vector<u32> offs;
    bool ok = false;
    s64 v = 0;
};
struct Term { int value; std::string op; s64 n; bool or_next = false; };  // or_next: "or" follows
struct Cond { std::vector<Term> terms; bool state = false; };
struct SetRule { int value; s64 n; int cond; };
struct FocusRule { Focus f; int cond; int after = 0; int held = 0; };  // (after: frames the condition must hold)

bool is_loaded;
std::vector<Value> values;
std::vector<Cond> conds;
std::vector<FocusRule> focus_rules;
int hud_cond = -2;  // -2: no HUD, -1: always
std::vector<Element> elems;
std::vector<Icon> icon_list;
std::vector<Button> buttons;
int keyboard_cond = -2;  // -2: no keyboard
std::map<std::string, std::pair<int, int>> keys;  // a character (UTF-8) or a named key -> bottom-screen pixel
std::vector<UnsavedMark> marks;
std::vector<SetRule> set_rules;
std::vector<Bind> bind_list;
std::vector<HubEntry> hub_list;
std::vector<MapArea> map_list;
Tools tool_cfg;
Stacks stack_cfg;
Aim aim_cfg;
std::map<std::string, s64> runtime_flags;
bool one_screen;
Focus cur_focus = FOCUS_NONE;
int trace_every;
u64 frames;

}  // namespace
u32 button_mask(const std::string &names) {
    static const std::pair<const char *, u32> btns[] = {
        {"A", 1 << 0}, {"B", 1 << 1}, {"SELECT", 1 << 2}, {"START", 1 << 3}, {"DRIGHT", 1 << 4}, {"DLEFT", 1 << 5},
        {"DUP", 1 << 6}, {"DDOWN", 1 << 7}, {"R", 1 << 8}, {"L", 1 << 9}, {"X", 1 << 10}, {"Y", 1 << 11},
        {"ZL", 1 << 14}, {"ZR", 1 << 15}, {"CRIGHT", 1u << 28}, {"CLEFT", 1u << 29}, {"CUP", 1u << 30},
        {"CDOWN", 1u << 31}};
    u32 m = 0;
    for (size_t p = 0; p <= names.size();) {
        size_t e = names.find('+', p);
        if (e == std::string::npos) e = names.size();
        const std::string part = names.substr(p, e - p);
        bool known = false;
        for (auto &[n, bit] : btns)
            if (part == n) m |= bit, known = true;
        if (!known) return 0;
        p = e + 1;
    }
    return m;
}
namespace {

std::map<std::string, s64> symbols;

bool num(const std::string &t, s64 &out) {
    if (t.empty()) return false;
    if (auto it = symbols.find(t); it != symbols.end()) {
        out = it->second;
        return true;
    }
    char *e;
    out = strtoll(t.c_str(), &e, 0);
    return *e == 0;
}

// ---- symbols: addresses found through the game's RTTI and its code

const GuestClass *find_class(const std::string &name) {
    for (u32 i = 0; i < game_class_count; i++)
        if (name == game_classes[i].name) return &game_classes[i];
    return nullptr;
}

bool program_global(u32 a) { return a >= g_game.data_addr && a < g_game.data_addr + g_game.data_size + g_game.bss_size; }

// The function at fn (Thumb with bit 0) from its start up to its first literal pool: the program globals it loads
// from the pool (distinct, in order of first use) and the functions it calls (bl / blx, in order).
void scan_function(u32 fn, std::vector<u32> &globals, std::vector<u32> &calls) {
    auto global = [&](u32 lit) {
        if (!kernel::guest_readable(lit, 4)) return;
        const u32 v = r32(lit);
        if (program_global(v) && std::find(globals.begin(), globals.end(), v) == globals.end()) globals.push_back(v);
    };
    u32 pool = ~0u;
    if (fn & 1) {
        for (u32 pc = fn & ~1u, n = 0; n < 4096 && pc < pool && kernel::guest_readable(pc, 4); n++, pc += 2) {
            const u16 h = r16(pc);
            if ((h & 0xF800) == 0x4800) {  // ldr rX, [pc, #imm]
                const u32 lit = ((pc + 4) & ~3u) + (h & 0xFF) * 4;
                pool = std::min(pool, lit);
                global(lit);
            } else if ((h & 0xF800) == 0xF000) {  // bl / blx: two halves
                const u16 h2 = r16(pc + 2);
                s32 off = (s32)(((h & 0x7FF) << 12) | ((h2 & 0x7FF) << 1));
                if (off & 0x400000) off -= 0x800000;
                if ((h2 & 0xF800) == 0xF800) calls.push_back((pc + 4 + off) | 1);
                else if ((h2 & 0xF800) == 0xE800) calls.push_back((pc + 4 + off) & ~3u);
                pc += 2;
            }
        }
        return;
    }
    for (u32 pc = fn, n = 0; n < 4096 && pc < pool && kernel::guest_readable(pc, 4); n++, pc += 4) {
        const u32 w = r32(pc), cond = w >> 28;
        if (cond != 0xF && (w & 0x0F7F0000) == 0x051F0000) {  // ldr rX, [pc, #+-imm]
            const u32 lit = pc + 8 + ((w & (1u << 23)) ? (w & 0xFFF) : -(w & 0xFFF));
            if (lit > pc) pool = std::min(pool, lit);
            global(lit);
        } else if ((w & 0x0E000000) == 0x0A000000 && (cond == 0xF || (w & 0x01000000))) {  // bl, blx #imm
            s32 off = (s32)(w & 0xFFFFFF);
            if (off & 0x800000) off -= 0x1000000;
            const u32 t = pc + 8 + off * 4;
            calls.push_back(cond == 0xF ? (t | ((w >> 23) & 2) | 1) : t);
        }
    }
}

// the functions whose literal pool holds v, in address order
std::vector<u32> users_of(u32 v) {
    std::vector<u32> fs;
    for (u32 a = g_game.text_addr; a + 4 <= g_game.text_addr + g_game.text_size; a += 4)
        if (r32(a) == v)
            if (const u32 f = function_start(a); f && std::find(fs.begin(), fs.end(), f) == fs.end()) fs.push_back(f);
    std::sort(fs.begin(), fs.end());
    return fs;
}

// the address of a NUL-terminated text in the program's read-only data (the first occurrence that starts a string:
// at the segment's start or after a NUL), 0 if absent
u32 find_string(const std::string &text) {
    const u32 a0 = g_game.ro_addr, n = g_game.ro_size;
    if (text.empty() || !n || !kernel::guest_readable(a0, n)) return 0;
    const u8 *p = (const u8 *)gptr(a0);
    for (u32 i = 0; i + text.size() < n; i++)
        if ((i == 0 || p[i - 1] == 0) && p[i + text.size()] == 0 && memcmp(p + i, text.data(), text.size()) == 0)
            return a0 + i;
    return 0;
}

// the first address in the program's code holding a byte pattern ("e5 9f 00 04 ?? ..."), 0 if absent or malformed
u32 find_code(const std::string &hex) {
    std::vector<int> pat;
    std::istringstream ss(hex);
    for (std::string b; ss >> b;) {
        for (size_t k = 0; k < b.size(); k += 2) {
            const std::string by = b.substr(k, 2);
            if (by == "??") pat.push_back(-1);
            else if (by.size() == 2 && isxdigit((u8)by[0]) && isxdigit((u8)by[1])) pat.push_back((int)strtol(by.c_str(), nullptr, 16));
            else return 0;
        }
    }
    const u32 a0 = g_game.text_addr, n = g_game.text_size;
    if (pat.empty() || !n || !kernel::guest_readable(a0, n)) return 0;
    const u8 *p = (const u8 *)gptr(a0);
    for (u32 i = 0; i + pat.size() <= n; i += 2) {
        size_t k = 0;
        while (k < pat.size() && (pat[k] < 0 || p[i + k] == pat[k])) k++;
        if (k == pat.size()) return a0 + i;
    }
    return 0;
}

// "<start> [<step>]..." from t[i..]
bool resolve(const std::vector<std::string> &t, size_t i, s64 &out, std::string &err) {
    if (i >= t.size()) {
        err = "expected 'vtable <class>', 'string \"<text>\"', 'code \"<bytes>\"', a symbol or a number";
        return false;
    }
    u32 a;
    if (t[i] == "vtable") {
        const GuestClass *c = i + 1 < t.size() ? find_class(t[i + 1]) : nullptr;
        if (!c) {
            err = "no class " + (i + 1 < t.size() ? t[i + 1] : std::string("named")) + " in the game's RTTI";
            return false;
        }
        a = c->vptr;
        i += 2;
    } else if (t[i] == "code") {
        const bool thumb = i + 1 < t.size() && t[i + 1] == "thumb";
        const std::string q = i + 1 + thumb < t.size() ? t[i + 1 + thumb] : "";
        if (q.size() < 2 || q.front() != '"' || q.back() != '"') {
            err = "expected code [thumb] \"<hex bytes>\"";
            return false;
        }
        a = find_code(q.substr(1, q.size() - 2));
        if (!a) {
            err = "no code " + q + " in the program";
            return false;
        }
        a |= thumb ? 1 : 0;
        i += 2 + thumb;
    } else if (t[i] == "string") {
        const std::string q = i + 1 < t.size() ? t[i + 1] : "";
        if (q.size() < 2 || q.front() != '"' || q.back() != '"') {
            err = "expected string \"<text>\"";
            return false;
        }
        a = find_string(q.substr(1, q.size() - 2));
        if (!a) {
            err = "no string " + q + " in the program";
            return false;
        }
        i += 2;
    } else {
        s64 n;
        if (!num(t[i], n)) {
            err = "unknown symbol " + t[i];
            return false;
        }
        a = (u32)n;
        i++;
    }
    char at[16];
    for (; i < t.size(); i += 2) {
        s64 n;
        if (i + 1 >= t.size() || !num(t[i + 1], n)) {
            err = "expected a number after " + t[i];
            return false;
        }
        const std::string &s = t[i];
        snprintf(at, sizeof at, "%08x", a);
        if (s == "+") a += (u32)n;
        else if (s == "-") a -= (u32)n;
        else if (s == "slot" || s == ">") {
            const u32 w = s == "slot" ? a + 4 * (u32)n : a;
            if (!kernel::guest_readable(w, 4)) {
                err = "nothing at " + std::string(at);
                return false;
            }
            a = r32(w) + (s == ">" ? (u32)n : 0);
        } else if (s == "call" || s == "global" || s == "user") {
            std::vector<u32> globals, calls, users;
            if (s == "user") users = users_of(a);
            else scan_function(a, globals, calls);
            const auto &list = s == "call" ? calls : s == "global" ? globals : users;
            if ((u64)n >= list.size()) {
                err = std::string(at) + " has " + std::to_string(list.size()) + " " + s + (list.size() == 1 ? "" : "s") +
                      ", not " + std::to_string(n + 1);
                return false;
            }
            a = list[n];
        } else {
            err = "unknown step " + s;
            return false;
        }
    }
    out = a;
    return true;
}

int value_index(const std::string &n) {
    for (size_t i = 0; i < values.size(); i++)
        if (values[i].name == n) return (int)i;
    return -1;
}

// "<value> <op> <n> [and ...]" from toks[i..]; returns the condition's index, or -2 on error
int parse_cond(const std::vector<std::string> &t, size_t i, std::string &err) {
    Cond c;
    while (i < t.size()) {
        if (i + 2 >= t.size()) {
            err = "incomplete condition";
            return -2;
        }
        Term term;
        term.value = value_index(t[i]);
        term.op = t[i + 1];
        if (term.value < 0) {
            err = "unknown value " + t[i];
            return -2;
        }
        static const char *ops[] = {"==", "!=", "<", "<=", ">", ">=", "&", "!&"};
        bool known = false;
        for (const char *o : ops) known |= term.op == o;
        if (!known || !num(t[i + 2], term.n)) {
            err = "bad comparison " + term.op + " " + t[i + 2];
            return -2;
        }
        c.terms.push_back(term);
        i += 3;
        if (i < t.size()) {
            if (t[i] != "and" && t[i] != "or") {
                err = "expected 'and' or 'or', got " + t[i];
                return -2;
            }
            c.terms.back().or_next = t[i] == "or";
            i++;
        }
    }
    conds.push_back(c);
    return (int)conds.size() - 1;
}

// the "when ..." tail of a directive (if any) from toks[i..]: -1 when there is none
int when(const std::vector<std::string> &t, size_t i, std::string &err) {
    if (i >= t.size()) return -1;
    if (t[i] != "when") {
        err = "unexpected " + t[i];
        return -2;
    }
    return parse_cond(t, i + 1, err);
}

bool parse_line(const std::string &line, std::string &err) {
    std::vector<std::string> t;
    {
        std::istringstream ss(line.substr(0, line.find('#')));
        for (std::string w; ss >> w;) {
            if (w.size() > 1 && w[0] == '"') {  // a quoted label, spaces included
                while (w.back() != '"' || w.size() == 1) {
                    std::string more;
                    if (!(ss >> more)) break;
                    w += " " + more;
                }
            }
            t.push_back(w);
        }
    }
    if (t.empty()) return true;
    if (t[0] == "region") {  // "region EUR,AUS <directive>": only for those regions
        static const char *names[] = {"JPN", "USA", "EUR", "AUS", "CHN", "KOR", "TWN"};
        if (t.size() < 3) {
            err = "expected 'region <names> <directive>'";
            return false;
        }
        bool mine = false;
        std::istringstream rs(t[1]);
        for (std::string r; std::getline(rs, r, ',');) {
            bool known = false;
            for (int i = 0; i < 7; i++)
                if (r == names[i]) known = true, mine |= g_game.region == i;
            if (!known) {
                err = "unknown region " + r;
                return false;
            }
        }
        if (!mine) return true;
        t.erase(t.begin(), t.begin() + 2);
    }
    const std::string &d = t[0];
    if (d == "symbol") {
        s64 v;
        if (t.size() < 4 || t[2] != "=") {
            err = "expected 'symbol <name> = <start> [<step>]...'";
            return false;
        }
        if (!resolve(t, 3, v, err)) return false;
        symbols[t[1]] = v;
        LOG("[profile] symbol %s = %08llx\n", t[1].c_str(), (unsigned long long)v);
        return true;
    }
    auto at = [&](size_t i, int &x, int &y) {  // "at <x> <y>" at t[i]
        s64 a, b;
        if (i + 2 >= t.size() || t[i] != "at" || !num(t[i + 1], a) || !num(t[i + 2], b)) {
            err = "expected 'at <x> <y>'";
            return false;
        }
        x = (int)a;
        y = (int)b;
        return true;
    };
    if (d == "value") {
        if (t.size() < 5 || t[2] != "=") {
            err = "expected 'value <name> = <type> <address> [> <offset>]...'";
            return false;
        }
        Value v;
        v.name = t[1];
        if (t[3] == "runtime") {
            v.type = 'r';
            v.flag = t[4];
            values.push_back(v);
            return true;
        }
        static const std::pair<const char *, char> types[] = {{"u8", 'b'}, {"u16", 'h'}, {"u32", 'I'}, {"s8", 'c'},
                                                              {"s16", 's'}, {"s32", 'i'}, {"f32", 'f'}};
        v.type = 0;
        for (auto &[n, c] : types)
            if (t[3] == n) v.type = c;
        s64 a;
        if (!v.type || !num(t[4], a)) {
            err = v.type ? "unknown address or symbol " + t[4] : "bad type " + t[3];
            return false;
        }
        v.addr = (u32)a;
        for (size_t i = 5; i < t.size(); i += 2) {
            s64 o;
            if (t[i] != ">" || i + 1 >= t.size() || !num(t[i + 1], o)) {
                err = "expected '> <offset>'";
                return false;
            }
            v.offs.push_back((u32)o);
        }
        values.push_back(v);
        return true;
    }
    if (d == "focus") {
        static const std::pair<const char *, Focus> kinds[] = {
            {"top", FOCUS_TOP}, {"bottom", FOCUS_BOTTOM}, {"both", FOCUS_BOTH}, {"panel", FOCUS_PANEL}};
        Focus f = FOCUS_NONE;
        for (auto &[n, k] : kinds)
            if (t.size() >= 2 && t[1] == n) f = k;
        if (f == FOCUS_NONE) {
            err = "expected 'focus top|bottom|both|panel'";
            return false;
        }
        size_t i = 2;
        s64 after = 0;
        if (i < t.size() && t[i] == "after") {
            if (i + 1 >= t.size() || !num(t[i + 1], after) || after < 0) {
                err = "expected 'after <frames>'";
                return false;
            }
            i += 2;
        }
        const int c = when(t, i, err);
        if (c == -2) return false;
        focus_rules.push_back({f, c, (int)after});
        return true;
    }
    if (d == "set") {
        s64 n;
        int v;
        if (t.size() < 4 || (v = value_index(t[1])) < 0 || t[2] != "=" || !num(t[3], n) || values[v].type == 'r') {
            err = "expected 'set <value> = <number>' (a value in guest memory)";
            return false;
        }
        const int c = when(t, 4, err);
        if (c == -2) return false;
        set_rules.push_back({v, n, c});
        return true;
    }
    if (d == "onescreen") {
        one_screen = true;
        return true;
    }

    if (d == "bind") {
        if (t.size() != 3) {
            err = "expected 'bind <host input> <target>'";
            return false;
        }
        Bind b{t[1], t[2], 0};
        static const char *overlays[] = {"hub", "map", "tools", "tool-next", "tool-prev", "tool-away", "pointer",
                                         "none"};
        if (std::find(std::begin(overlays), std::end(overlays), t[2]) == std::end(overlays) &&
            !(b.buttons = button_mask(t[2]))) {
            err = "unknown target " + t[2];
            return false;
        }
        bind_list.push_back(b);
        return true;
    }
    if (d == "hub") {
        if (t.size() < 3 || t[1].size() < 2 || t[1][0] != '"') {
            err = "expected 'hub \"<label>\" <action>'";
            return false;
        }
        HubEntry h;
        h.label = t[1].substr(1, t[1].size() - 2);
        size_t i = 3;
        s64 a, b, c2, e;
        if (t[2] == "tap" && t.size() >= 5 && num(t[3], a) && num(t[4], b)) {
            h.action = HubEntry::TAP;
            h.x = (int)a, h.y = (int)b;
            i = 5;
        } else if (t[2] == "press" && t.size() >= 4 && (h.buttons = button_mask(t[3]))) {
            h.action = HubEntry::PRESS;
            i = 4;
        } else if (t[2] == "map") {
            h.action = HubEntry::MAP;
        } else if (t[2] == "tools") {
            h.action = HubEntry::TOOLS;
        } else {
            err = "expected tap <x> <y>, press <buttons>, map or tools";
            return false;
        }
        if (i + 4 < t.size() && t[i] == "icon" && num(t[i + 1], a) && num(t[i + 2], b) && num(t[i + 3], c2) &&
            num(t[i + 4], e)) {
            h.ix = (int)a, h.iy = (int)b, h.iw = (int)c2, h.ih = (int)e;
            i += 5;
        } else if (i + 1 < t.size() && t[i] == "glyph" && t[i + 1].size() > 2 && t[i + 1][0] == '"') {
            h.glyph = t[i + 1].substr(1, t[i + 1].size() - 2);
            i += 2;
        }
        if ((h.cond = when(t, i, err)) == -2) return false;
        hub_list.push_back(h);
        return true;
    }
    if (d == "map") {
        s64 x, y, w, h;
        if (t.size() < 5 || !num(t[1], x) || !num(t[2], y) || !num(t[3], w) || !num(t[4], h)) {
            err = "expected 'map <x> <y> <w> <h>'";
            return false;
        }
        MapArea m{(int)x, (int)y, (int)w, (int)h, -1};
        if ((m.cond = when(t, 5, err)) == -2) return false;
        map_list.push_back(m);
        return true;
    }
    if (d == "tools" && t.size() >= 2) {
        Tools &c = tool_cfg;
        auto val = [&](size_t i, int &out) {
            if (i >= t.size() || (out = value_index(t[i])) < 0) {
                err = "unknown value " + (i < t.size() ? t[i] : std::string());
                return false;
            }
            return true;
        };
        auto n = [&](size_t i, u32 &out) {
            s64 v;
            if (i >= t.size() || !num(t[i], v)) {
                err = "expected a number";
                return false;
            }
            out = (u32)v;
            return true;
        };
        if (t[1] == "pockets") {
            if (t.size() < 5 || t[3] != "held" || !val(2, c.pockets) || !val(4, c.held)) {
                if (err.empty()) err = "expected 'tools pockets <value> held <value> [empty <item>]'";
                return false;
            }
            if (t.size() >= 7 && t[5] == "empty") return n(6, c.empty);
            return true;
        }
        if (t[1] == "items") {
            if (t.size() != 12 || t[3] != "record" || t[5] != "first" || t[7] != "category" || t[9] != "kinds") {
                err = "expected 'tools items <file> record <n> first <item> category <off> kinds <lo> <hi>'";
                return false;
            }
            c.items = t[2];
            return n(4, c.record) && n(6, c.first) && n(8, c.category_off) && n(10, c.kind_lo) && n(11, c.kind_hi) &&
                   c.record > 0;
        }
        if (t[1] == "icon" && t.size() == 3) {
            u32 v;
            if (!n(2, v) || v < 4 || v > 64) return false;
            c.icon_size = (int)v;
            return true;
        }
        if (t[1] == "textures") {
            if (t.size() < 8 || t[4] != "field" || t[6] != "plus") {
                err = "expected 'tools textures <darc> <prefix> field <offset> plus <n> [<item>=<number>]...'";
                return false;
            }
            c.textures = t[2];
            c.texture_prefix = t[3];
            s64 add;
            if (!n(5, c.texture_field) || !num(t[7], add)) return false;
            c.texture_add = (int)add;
            for (size_t i = 8; i < t.size(); i++) {
                const size_t eq = t[i].find('=');
                s64 item, number;
                if (eq == std::string::npos || !num(t[i].substr(0, eq), item) || !num(t[i].substr(eq + 1), number)) {
                    err = "expected <item>=<number>, got " + t[i];
                    return false;
                }
                c.texture_of[(u32)item] = (int)number;
            }
            return true;
        }
        if (t[1] == "names" && t.size() == 3) {
            c.names = t[2];
            return true;
        }
        if (t[1] == "slots") {
            for (size_t i = 2; i + 1 < t.size(); i += 2) {
                u32 x, y;
                if (!n(i, x) || !n(i + 1, y)) return false;
                c.slots.push_back({(int)x, (int)y});
            }
            return true;
        }
        if (t[1] == "hold") {
            if (t.size() != 5 || t[3] != "on") {
                err = "expected 'tools hold <function> on <address>'";
                return false;
            }
            return n(2, c.hold_fn) && n(4, c.hold_on);
        }
        if (t[1] == "equip") {
            u32 x, y;
            size_t wo = std::find(t.begin(), t.end(), "when-open") - t.begin();
            size_t rd = std::find(t.begin(), t.end(), "ready") - t.begin();
            if (t.size() < 6 || !n(2, x) || !n(3, y) || t[4] != "open" || !(c.open = button_mask(t[5])) ||
                wo != 6 || rd >= t.size() || rd < wo) {
                if (err.empty()) err = "expected 'tools equip <x> <y> open <buttons> when-open <cond> ready <cond>'";
                return false;
            }
            c.portrait_x = (int)x, c.portrait_y = (int)y;
            std::vector<std::string> a(t.begin() + wo + 1, t.begin() + rd), b(t.begin() + rd + 1, t.end());
            return (c.open_cond = parse_cond(a, 0, err)) != -2 && (c.ready_cond = parse_cond(b, 0, err)) != -2;
        }
        err = "unknown tools directive " + t[1];
        return false;
    }
    if (d == "aim" && t.size() >= 2) {
        Aim &c = aim_cfg;
        s64 a = 0, b = 0, v = 0;
        if (t[1] == "delta") {  // aim delta <return> <x off> <y off> [per s<n>] [active r<n> <off>] on <address> ...
            const char *usage = "expected 'aim delta <return address> <x offset> <y offset> [per s<n>] [active r<n> <offset>] "
                                "on <address> ...'";
            if (t.size() < 7 || !num(t[2], a) || !num(t[3], b) || !num(t[4], v)) {
                err = usage;
                return false;
            }
            c.delta_ret = (u32)a, c.delta_x = (u32)b, c.delta_y = (u32)v;
            size_t i = 5;
            auto reg = [&](const std::string &w, char kind, s64 &n) { return w.size() > 1 && w[0] == kind && num(w.substr(1), n); };
            for (;;) {
                if (i + 1 < t.size() && t[i] == "per" && reg(t[i + 1], 's', a) && a < 32) {
                    c.delta_per = (int)a;
                    i += 2;
                } else if (i + 2 < t.size() && t[i] == "active" && reg(t[i + 1], 'r', a) && a < 13 && num(t[i + 2], b)) {
                    c.active_reg = (int)a, c.active_off = (u32)b;
                    i += 3;
                } else {
                    break;
                }
            }
            if (i + 1 >= t.size() || t[i] != "on") {
                err = usage;
                return false;
            }
            for (i++; i < t.size(); i++) {
                if (!num(t[i], a)) {
                    err = "expected an address";
                    return false;
                }
                c.delta_on.push_back((u32)a);
            }
            return true;
        }
        if (t[1] == "stick" && t.size() == 3 && num(t[2], a) && a > 0) {
            c.stick_speed = (float)a;
            return true;
        }
        if (t[1] == "when") return (c.cond = parse_cond(t, 2, err)) != -2;
        err = "unknown aim directive " + t[1];
        return false;
    }
    if (d == "stack" && t.size() >= 2) {
        Stacks &c = stack_cfg;
        if (t[1] == "owner") {
            for (size_t i = 2; i < t.size(); i++) {
                const int v = value_index(t[i]);
                if (v < 0) {
                    err = "unknown value " + t[i];
                    return false;
                }
                c.owner.push_back(v);
            }
            return !c.owner.empty();
        }
        s64 a;
        if (t[1] == "on") {
            if (t.size() != 3 || !num(t[2], a)) {
                err = "expected 'stack on <address>'";
                return false;
            }
            c.on = (u32)a;
            return true;
        }
        if (t[1] == "free") {
            if (t.size() < 4 || !num(t[2], a)) {
                err = "expected 'stack free <offset> <state> ...'";
                return false;
            }
            c.free_off = (u32)a;
            for (size_t i = 3; i < t.size(); i++) {
                s64 v;
                if (!num(t[i], v)) {
                    err = "expected a number";
                    return false;
                }
                c.free_states.push_back((u8)v);
            }
            return true;
        }
        if (t[1] == "spread") {
            const size_t k = std::find(t.begin(), t.end(), "keep") - t.begin();
            s64 n;
            if (k < 3 || k + 2 != t.size() || !num(t[k + 1], n) || n < 0) {
                err = "expected 'stack spread <state> ... keep <n>'";
                return false;
            }
            for (size_t i = 2; i < k; i++) {
                if (!num(t[i], a)) {
                    err = "expected a number";
                    return false;
                }
                c.spread_states.push_back((u8)a);
            }
            c.spread_keep = (int)n;
            return true;
        }
        if (t[1] == "underfoot") {
            if (t.size() != 3 || !num(t[2], a)) {
                err = "expected 'stack underfoot <offset>'";
                return false;
            }
            c.underfoot_off = (u32)a;
            return true;
        }
        if (t[1] == "shown") {
            c.shown_cond = when(t, 2, err);
            if (c.shown_cond < 0 && err.empty()) err = "expected 'stack shown when <cond>'";
            return c.shown_cond >= 0;
        }
        if (t[1] == "kinds") {
            const size_t m = std::find(t.begin(), t.end(), "max") - t.begin();
            s64 mx;
            if (m < 3 || m + 2 != t.size() || !num(t[m + 1], mx) || mx < 2) {
                err = "expected 'stack kinds <category> ... max <n>'";
                return false;
            }
            for (size_t i = 2; i < m; i++) {
                if (!num(t[i], a)) {
                    err = "expected a number";
                    return false;
                }
                c.kinds.push_back({(u32)a, (int)mx});
            }
            return true;
        }
        err = "unknown stack directive " + t[1];
        return false;
    }
    if (d == "clearbottom") {
        gpu::set_clear_after_top_transfer(true);
        return true;
    }
    if (d == "hud") {
        hud_cond = when(t, 1, err);
        return hud_cond != -2;
    }
    if (d == "icon") {
        if (t.size() < 3) {
            err = "expected 'icon <name> <romfs path>[:<inner>]'";
            return false;
        }
        Icon ic;
        ic.name = t[1];
        const size_t colon = t[2].find(':');
        ic.file = t[2].substr(0, colon);
        if (colon != std::string::npos) ic.inner = t[2].substr(colon + 1);
        for (size_t i = 3; i < t.size();) {
            s64 a, b, c2, e;
            if (t[i] == "rect" && i + 4 < t.size() && num(t[i + 1], a) && num(t[i + 2], b) && num(t[i + 3], c2) &&
                num(t[i + 4], e)) {
                ic.rx = (int)a, ic.ry = (int)b, ic.rw = (int)c2, ic.rh = (int)e;
                i += 5;
            } else if (t[i] == "scale" && i + 1 < t.size()) {
                ic.scale = strtof(t[i + 1].c_str(), nullptr);
                i += 2;
            } else {
                err = "unexpected " + t[i];
                return false;
            }
        }
        icon_list.push_back(ic);
        return true;
    }
    if (d == "counter" || d == "show") {
        Element e;
        e.kind = d == "counter" ? Element::COUNTER : Element::ICON;
        size_t i = 2;
        if (t.size() < 2) {
            err = "missing icon";
            return false;
        }
        e.icon = t[1];
        if (e.kind == Element::COUNTER) {
            if (t.size() < 3 || (e.value = value_index(t[2])) < 0) {
                err = "unknown value";
                return false;
            }
            i = 3;
        }
        if (!at(i, e.x, e.y)) return false;
        i += 3;
        if (i + 1 < t.size() && t[i] == "digits") {
            e.digits = atoi(t[i + 1].c_str());
            i += 2;
        }
        if ((e.cond = when(t, i, err)) == -2) return false;
        elems.push_back(e);
        return true;
    }
    if (d == "pips") {
        Element e;
        e.kind = Element::PIPS;
        if (t.size() < 5 || (e.value = value_index(t[3])) < 0) {
            err = "expected 'pips <on> <off> <value> <count> at <x> <y>'";
            return false;
        }
        e.icon = t[1];
        e.icon_off = t[2];
        e.count = atoi(t[4].c_str());
        if (!at(5, e.x, e.y)) return false;
        if ((e.cond = when(t, 8, err)) == -2) return false;
        elems.push_back(e);
        return true;
    }
    if (d == "pick") {
        Element e;
        e.kind = Element::PICK;
        if (t.size() < 2 || (e.value = value_index(t[1])) < 0) {
            err = "unknown value";
            return false;
        }
        if (!at(2, e.x, e.y)) return false;
        size_t i = 5;
        for (; i < t.size() && t[i] != "when"; i++) {
            const size_t eq = t[i].find('=');
            s64 n;
            if (eq == std::string::npos || !num(t[i].substr(0, eq), n)) {
                err = "expected <n>=<icon>, got " + t[i];
                return false;
            }
            e.picks.push_back({n, t[i].substr(eq + 1)});
        }
        if ((e.cond = when(t, i, err)) == -2) return false;
        elems.push_back(e);
        return true;
    }
    if (d == "button") {
        Button b;
        s64 x, y;
        if (t.size() < 4 || t[1] != "tap" || !num(t[2], x) || !num(t[3], y)) {
            err = "expected 'button tap <x> <y>'";
            return false;
        }
        b.tx = (int)x;
        b.ty = (int)y;
        if ((b.cond = when(t, 4, err)) == -2) return false;
        buttons.push_back(b);
        return true;
    }
    if (d == "keyboard") {
        keyboard_cond = when(t, 1, err);
        return keyboard_cond != -2;
    }
    if (d == "key") {
        s64 x, y, step = 0;
        if (t.size() < 5 || t[2] != "at" || !num(t[3], x) || !num(t[4], y) ||
            (t.size() > 5 && (t.size() != 7 || t[5] != "step" || !num(t[6], step)))) {
            err = "expected 'key <characters> at <x> <y> [step <dx>]'";
            return false;
        }
        static const char *named[] = {"space", "backspace", "shift", "enter", "escape"};
        if (std::find(std::begin(named), std::end(named), t[1]) != std::end(named)) {
            keys[t[1]] = {(int)x, (int)y};
            return true;
        }
        int n = 0;
        for (size_t i = 0; i < t[1].size(); n++) {  // one key per UTF-8 character
            size_t j = i + 1;
            while (j < t[1].size() && ((u8)t[1][j] & 0xC0) == 0x80) j++;
            keys[t[1].substr(i, j - i)] = {(int)(x + n * step), (int)y};
            i = j;
        }
        return true;
    }
    if (d == "unsaved") {
        s64 off, bits, block, size;
        if (t.size() != 7 || !num(t[2], off) || !num(t[3], bits) || t[4] != "crc32c" || !num(t[5], block) ||
            !num(t[6], size) || off < block + 4 || off >= block + 4 + size || bits <= 0 || bits > 0xff) {
            err = "expected 'unsaved <save file> <offset> <bits> crc32c <block> <size>' (the offset in the block)";
            return false;
        }
        marks.push_back({t[1], (u32)off, (u8)bits, (u32)block, (u32)size});
        return true;
    }
    err = "unknown directive " + d;
    return false;
}

void read_value(Value &v) {
    v.ok = false;
    if (v.type == 'r') {
        auto it = runtime_flags.find(v.flag);
        v.v = it == runtime_flags.end() ? 0 : it->second;
        v.ok = true;
        return;
    }
    u32 a = v.addr;
    for (u32 off : v.offs) {
        if (!kernel::guest_readable(a, 4)) return;
        const u32 p = r32(a);
        if (!p) return;
        a = p + off;
    }
    const u32 size = v.type == 'b' || v.type == 'c' ? 1 : v.type == 'h' || v.type == 's' ? 2 : 4;
    if (!kernel::guest_readable(a, size)) return;
    v.at = a;
    switch (v.type) {
    case 'b': v.v = r8(a); break;
    case 'c': v.v = (s8)r8(a); break;
    case 'h': v.v = r16(a); break;
    case 's': v.v = (s16)r16(a); break;
    case 'I': v.v = r32(a); break;
    case 'i': v.v = (s32)r32(a); break;
    case 'f': {
        const u32 u = r32(a);
        float f;
        memcpy(&f, &u, 4);
        if (!std::isfinite(f)) return;
        v.v = (s64)f;
        break;
    }
    }
    v.ok = true;
}

bool eval(const Term &t) {
    const Value &v = values[t.value];
    if (!v.ok) return false;
    if (t.op == "==") return v.v == t.n;
    if (t.op == "!=") return v.v != t.n;
    if (t.op == "<") return v.v < t.n;
    if (t.op == "<=") return v.v <= t.n;
    if (t.op == ">") return v.v > t.n;
    if (t.op == ">=") return v.v >= t.n;
    if (t.op == "&") return (v.v & t.n) != 0;
    return (v.v & t.n) == 0;  // "!&"
}

}  // namespace

bool load() {
    std::string path;
    if (const char *p = getenv("RECOMP_PROFILE")) path = p;
    else {
        char tid[20];
        snprintf(tid, sizeof tid, "%016llx", (unsigned long long)g_game.program_id);
        for (const auto &p : {g_game.dir + "/profile.txt", std::string(RECOMP_SOURCE_DIR) + "/games/" + tid + "/profile.txt"})
            if (FILE *f = fopen(p.c_str(), "r")) {
                fclose(f);
                path = p;
                break;
            }
    }
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) {
        LOG("[profile] cannot read %s\n", path.c_str());
        return false;
    }
    std::string line, err;
    for (int n = 1; std::getline(in, line); n++)
        if (!parse_line(line, err)) {
            LOG("[profile] %s:%d: %s\n", path.c_str(), n, err.c_str());
            err.clear();
        }
    if (const char *t = getenv("RECOMP_PROFILE_TRACE")) trace_every = atoi(t);
    is_loaded = true;
    LOG("[profile] %s: %zu values, %zu focus rules, %s, %zu icons, %zu buttons\n", path.c_str(), values.size(),
        focus_rules.size(), hud_cond == -2 ? "no HUD" : "HUD", icon_list.size(), buttons.size());
    return true;
}

bool loaded() { return is_loaded; }
bool has_focus_rules() { return !focus_rules.empty(); }
const std::vector<UnsavedMark> &unsaved_marks() { return marks; }
bool has_hud() { return hud_cond != -2 && !elems.empty(); }

void update() {
    if (!is_loaded) return;
    for (auto &v : values) read_value(v);
    for (auto &c : conds) {  // a sum of products: "and" groups joined by "or"
        bool any = false, group = true;
        for (auto &t : c.terms) {
            group = group && eval(t);
            if (t.or_next) {
                any = any || group;
                group = true;
            }
        }
        c.state = any || group;
    }
    for (auto &r : set_rules) {
        Value &v = values[r.value];
        if (!v.ok || v.v == r.n || !cond(r.cond)) continue;
        switch (v.type) {
        case 'b': case 'c': w8(v.at, (u8)r.n); break;
        case 'h': case 's': w16(v.at, (u16)r.n); break;
        case 'f': {
            const float f = (float)r.n;
            u32 u;
            memcpy(&u, &f, 4);
            w32(v.at, u);
            break;
        }
        default: w32(v.at, (u32)r.n); break;
        }
    }
    cur_focus = FOCUS_NONE;
    for (auto &r : focus_rules) {
        r.held = r.cond < 0 || conds[r.cond].state ? r.held + 1 : 0;
        if (cur_focus == FOCUS_NONE && r.held > r.after) cur_focus = r.f;
    }
    frames++;
    if (trace_every > 0 && frames % trace_every == 0) {
        std::string s;
        for (auto &v : values) {
            char b[64];
            if (v.ok) snprintf(b, sizeof b, " %s=%lld", v.name.c_str(), (long long)v.v);
            else snprintf(b, sizeof b, " %s=-", v.name.c_str());
            s += b;
        }
        static const char *fn[] = {"-", "top", "bottom", "both", "panel"};
        LOG("[profile] focus %s hud %d:%s\n", fn[cur_focus], hud_visible(), s.c_str());
    }
}

Focus focus() { return cur_focus; }
bool hud_visible() { return hud_cond != -2 && (hud_cond == -1 || conds[hud_cond].state); }
const std::vector<Element> &elements() { return elems; }
const std::vector<Icon> &icons() { return icon_list; }
bool cond(int i) { return i < 0 || (i < (int)conds.size() && conds[i].state); }
bool value(int i, s64 &v) {
    if (i < 0 || i >= (int)values.size() || !values[i].ok) return false;
    v = values[i].v;
    return true;
}
void set_runtime(const std::string &flag, s64 v) { runtime_flags[flag] = v; }
const std::vector<Bind> &binds() { return bind_list; }
const std::vector<HubEntry> &hub() { return hub_list; }
const std::vector<MapArea> &map_areas() { return map_list; }
const Tools &tools() { return tool_cfg; }
const Stacks &stacks() { return stack_cfg; }
const Aim &aim() { return aim_cfg; }

bool value_now(int i, s64 &out, u32 *at) {
    if (i < 0 || i >= (int)values.size()) return false;
    Value v = values[i];
    if (v.type != 'r') read_value(v);
    if (!v.ok) return false;
    out = v.v;
    if (at) *at = v.at;
    return true;
}

bool cond_now(int i) {
    if (i < 0) return true;
    if (i >= (int)conds.size()) return false;
    bool any = false, group = true;
    for (const Term &t : conds[i].terms) {
        s64 v;
        bool r = false;
        if (value_now(t.value, v)) {
            if (t.op == "==") r = v == t.n;
            else if (t.op == "!=") r = v != t.n;
            else if (t.op == "<") r = v < t.n;
            else if (t.op == "<=") r = v <= t.n;
            else if (t.op == ">") r = v > t.n;
            else if (t.op == ">=") r = v >= t.n;
        }
        group = group && r;
        if (t.or_next) {
            any = any || group;
            group = true;
        }
    }
    return any || group;
}
bool onescreen() { return one_screen; }
bool value_address(int i, u32 &a) {
    if (i < 0 || i >= (int)values.size() || !values[i].ok || values[i].type == 'r') return false;
    a = values[i].at;
    return true;
}
bool keyboard_active() { return keyboard_cond != -2 && (keyboard_cond == -1 || conds[keyboard_cond].state); }
bool key(const std::string &k, int &x, int &y) {
    auto it = keys.find(k);
    if (it == keys.end()) return false;
    x = it->second.first;
    y = it->second.second;
    return true;
}
const Button *button() {
    for (auto &b : buttons)
        if (cond(b.cond)) return &b;
    return nullptr;
}

}  // namespace profile
