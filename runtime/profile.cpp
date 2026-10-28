// Per-game profiles. A profile is a text file, one directive per line ('#' starts a comment):
//
//   value <name> = <type> <address> [> <offset>]...
//       A number in guest memory. Types: u8 u16 u32 s8 s16 s32 f32 (f32 is truncated to an integer). Each
//       "> off" follows a pointer: the address so far is read as a u32 and off is added (a null or unmapped
//       pointer makes the value absent, and every condition using it false). Numbers: decimal or 0x hex.
//   focus top|bottom|both [when <condition>]
//       Which screen matters (first rule that holds wins; none: the user's own layout). "both": the top
//       screen with the bottom one shown beside it (picture in picture).
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
//
// Positions are top-screen pixels (400x240); a negative x or y counts from the right or bottom edge.
// Conditions: <value> <op> <number> [and ...], with op one of == != < <= > >= & (any bit set) !& (none set).
// RECOMP_PROFILE=<file> loads another profile; RECOMP_PROFILE_TRACE=<n> logs every value and the focus every
// n frames (for writing profiles).
#include "profile.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include "kernel.h"
#include "loader.h"

namespace profile {
namespace {

struct Value {
    std::string name;
    char type = 'I';  // 'b' u8, 'h' u16, 'I' u32, 'c' s8, 's' s16, 'i' s32, 'f' f32
    u32 addr = 0;
    std::vector<u32> offs;
    bool ok = false;
    s64 v = 0;
};
struct Term { int value; std::string op; s64 n; };
struct Cond { std::vector<Term> terms; bool state = false; };
struct FocusRule { Focus f; int cond; };

bool is_loaded;
std::vector<Value> values;
std::vector<Cond> conds;
std::vector<FocusRule> focus_rules;
int hud_cond = -2;  // -2: no HUD, -1: always
std::vector<Element> elems;
std::vector<Icon> icon_list;
std::vector<Button> buttons;
Focus cur_focus = FOCUS_NONE;
int trace_every;
u64 frames;

bool num(const std::string &t, s64 &out) {
    if (t.empty()) return false;
    char *e;
    out = strtoll(t.c_str(), &e, 0);
    return *e == 0;
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
            if (t[i] != "and") {
                err = "expected 'and', got " + t[i];
                return -2;
            }
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
        for (std::string w; ss >> w;) t.push_back(w);
    }
    if (t.empty()) return true;
    const std::string &d = t[0];
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
        static const std::pair<const char *, char> types[] = {{"u8", 'b'}, {"u16", 'h'}, {"u32", 'I'}, {"s8", 'c'},
                                                              {"s16", 's'}, {"s32", 'i'}, {"f32", 'f'}};
        v.type = 0;
        for (auto &[n, c] : types)
            if (t[3] == n) v.type = c;
        s64 a;
        if (!v.type || !num(t[4], a)) {
            err = "bad type or address";
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
        if (t.size() < 2 || (t[1] != "top" && t[1] != "bottom" && t[1] != "both")) {
            err = "expected 'focus top|bottom|both'";
            return false;
        }
        const int c = when(t, 2, err);
        if (c == -2) return false;
        focus_rules.push_back({t[1] == "top" ? FOCUS_TOP : t[1] == "bottom" ? FOCUS_BOTTOM : FOCUS_BOTH, c});
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
    err = "unknown directive " + d;
    return false;
}

void read_value(Value &v) {
    v.ok = false;
    u32 a = v.addr;
    for (u32 off : v.offs) {
        if (!kernel::guest_readable(a, 4)) return;
        const u32 p = r32(a);
        if (!p) return;
        a = p + off;
    }
    const u32 size = v.type == 'b' || v.type == 'c' ? 1 : v.type == 'h' || v.type == 's' ? 2 : 4;
    if (!kernel::guest_readable(a, size)) return;
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
bool has_hud() { return hud_cond != -2 && !elems.empty(); }

void update() {
    if (!is_loaded) return;
    for (auto &v : values) read_value(v);
    for (auto &c : conds) {
        c.state = true;
        for (auto &t : c.terms) c.state = c.state && eval(t);
    }
    cur_focus = FOCUS_NONE;
    for (auto &r : focus_rules)
        if (r.cond < 0 || conds[r.cond].state) {
            cur_focus = r.f;
            break;
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
        static const char *fn[] = {"-", "top", "bottom", "both"};
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
const Button *button() {
    for (auto &b : buttons)
        if (cond(b.cond)) return &b;
    return nullptr;
}

}  // namespace profile
