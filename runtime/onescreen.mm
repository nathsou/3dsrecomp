// One-screen play (onescreen.h): binds, the hub, the map, the tool ring and the pointer, drawn over the top screen.
// Everything acts on the game through its own inputs: taps on the bottom screen where its buttons are, presses,
// and its pockets' own menus, so the game's state never changes behind its back.
#include "onescreen.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "profile.h"
#include "settings.h"
#include "text.h"

namespace kernel {
std::vector<u8> romfs_file(const std::string &path);  // (kernel.h clashes with MacTypes.h)
}

namespace onescreen {
namespace {

constexpr int kSlots = 16;             // pockets
constexpr int kRepeatDelay = 14;       // frames before a held direction repeats, then every kRepeatEvery
constexpr int kRepeatEvery = 5;
constexpr int kStick = 80;             // circle pad value that counts as a direction
constexpr int kToastFrames = 100;
constexpr int kHintFrames = 600;       // the controls hint after one-screen play starts
constexpr float kPointerSpeed = 4.5f;  // bottom-screen pixels per frame at full tilt

bool on;
SDL_FRect top_r, panel_r;
enum Overlay { NONE, HUB, MAP, TOOLS } overlay = NONE;
int hub_sel, tool_sel;
u32 prev_host;
int repeat_frames;
u32 repeat_dir;
bool hold_dir;  // directions are ignored until released (the one that opened an overlay)
std::string toast;
int toast_left, hint_left = -1;

// ---- the pointer (bottom-screen pixels), shown while the right stick moves it
float ptr_x = 160, ptr_y = 120;
int ptr_idle = 1000;

// ---- scripted inputs the overlays play on the game: presses, waits for a condition, touches
struct Step {
    u32 buttons = 0;
    int frames = 1;
    int wait_cond = -2;   // >= -1: wait (up to frames) until this condition holds (-1: always)
    bool wait_not = false;
    int if_cond = -2;     // >= -1: skipped unless this condition holds when the step begins
    bool or_abort = false;  // a wait that times out ends the whole sequence (the game did not follow)
    bool touch = false;
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // a touch from (x0, y0) to (x1, y1) over the step
};
std::vector<Step> macro;
int step_frame;

// ---- the hub's icons: parts of the bottom screen, kept from the last picture where their entry was offered
struct Icon {
    std::vector<u32> px;
    int w = 0, h = 0;
    bool dirty = false;
    SDL_Texture *tex = nullptr;
};
std::vector<Icon> hub_icons;
u64 frames_seen;

// ---- the tool ring: item records (categories) and names from the game's files
struct Items {
    bool loaded = false;
    std::vector<u8> records;
    std::vector<std::string> names;
} items;
// items' icons, kept from the pockets while they show (by item number)
std::map<u32, Icon> item_icons;

struct Tool {
    int slot;  // pocket, or -1: the hands (putting the held tool away)
    u32 item;
};

void say(const std::string &s) {
    toast = s;
    toast_left = kToastFrames;
}

// names in an MSBT (the first one of a umsbt), control tags dropped
std::vector<std::string> msbt_names(const std::vector<u8> &f) {
    std::vector<std::string> out;
    auto le32 = [&](size_t o) { return o + 4 <= f.size() ? (u32)(f[o] | f[o + 1] << 8 | f[o + 2] << 16 | (u32)f[o + 3] << 24) : 0u; };
    size_t base = 0;
    if (f.size() >= 8 && memcmp(f.data(), "MsgStdBn", 8)) base = le32(0);  // a umsbt: offsets and sizes first
    if (base + 0x20 > f.size() || memcmp(f.data() + base, "MsgStdBn", 8)) return out;
    const u32 nsec = f[base + 0xe] | f[base + 0xf] << 8;
    size_t p = base + 0x20;
    for (u32 i = 0; i < nsec && p + 16 <= f.size(); i++) {
        const u32 size = le32(p + 4);
        if (!memcmp(f.data() + p, "TXT2", 4)) {
            const size_t t = p + 16;
            const u32 n = le32(t);
            for (u32 k = 0; k < n; k++) {
                const size_t a = t + le32(t + 4 + 4 * k), b = k + 1 < n ? t + le32(t + 8 + 4 * k) : t + size;
                std::string s;
                for (size_t q = a; q + 1 < b && q + 1 < f.size(); q += 2) {
                    const u32 c = f[q] | f[q + 1] << 8;
                    if (c == 0) break;
                    if (c == 0x0e) {  // a tag: group, type, size, then size bytes of parameters
                        if (q + 7 >= f.size()) break;
                        q += 6 + (f[q + 6] | f[q + 7] << 8);
                        continue;
                    }
                    if (c < 0x80) s += (char)c;
                    else if (c < 0x800) s += (char)(0xc0 | c >> 6), s += (char)(0x80 | (c & 63));
                    else s += (char)(0xe0 | c >> 12), s += (char)(0x80 | (c >> 6 & 63)), s += (char)(0x80 | (c & 63));
                }
                out.push_back(s);
            }
            break;
        }
        p = (p + 16 + size + 15) & ~(size_t)15;
    }
    return out;
}

void load_items() {
    if (items.loaded) return;
    items.loaded = true;
    const profile::Tools &c = profile::tools();
    if (!c.items.empty()) items.records = kernel::romfs_file(c.items);
    if (!c.names.empty()) items.names = msbt_names(kernel::romfs_file(c.names));
    LOG("[onescreen] %zu item records, %zu names\n", c.record ? items.records.size() / c.record : 0, items.names.size());
}

// an item's record field (-1: none)
int item_field(u32 item, u32 off, int bytes) {
    const profile::Tools &c = profile::tools();
    const u32 id = item & 0xffff;
    if (!c.record || id < c.first) return -1;
    const size_t o = (size_t)(id - c.first) * c.record + off;
    if (o + bytes > items.records.size() || off + bytes > c.record) return -1;
    return bytes == 1 ? items.records[o] : items.records[o] | items.records[o + 1] << 8;
}

bool is_tool(u32 item) {
    const profile::Tools &c = profile::tools();
    const int k = item_field(item, c.category_off, 1);
    return (item & 0xffff) != c.empty && k >= (int)c.kind_lo && k <= (int)c.kind_hi;
}

std::string item_name(u32 item) {
    const u32 id = item & 0xffff, first = profile::tools().first;
    return id >= first && id - first < items.names.size() ? items.names[id - first] : "";
}

bool read_pockets(u32 out[kSlots], u32 &held) {
    const profile::Tools &c = profile::tools();
    u32 a;
    s64 h;
    if (!profile::value_address(c.pockets, a) || !profile::value(c.held, h)) return false;
    for (int i = 0; i < kSlots; i++) out[i] = r32(a + 4 * i);
    held = (u32)h;
    return true;
}

// the tools in the pockets, then the hands when a tool is held (to put it away)
std::vector<Tool> tool_list(u32 &held) {
    std::vector<Tool> v;
    u32 p[kSlots];
    if (!read_pockets(p, held)) return v;
    for (int i = 0; i < kSlots; i++)
        if (is_tool(p[i])) v.push_back({i, p[i]});
    if (is_tool(held)) v.push_back({-1, held});
    return v;
}

void tap(std::vector<Step> &m, int x, int y) {
    Step s;
    s.touch = true;
    s.x0 = s.x1 = (float)x, s.y0 = s.y1 = (float)y;
    s.frames = 6;
    m.push_back(s);
    m.push_back({0, 6});
}

void macro_open_pockets(std::vector<Step> &m) {
    const profile::Tools &c = profile::tools();
    m.push_back({c.open, 6});
    Step w;
    w.wait_cond = c.open_cond;
    w.frames = 120;
    w.or_abort = true;
    m.push_back(w);
    m.push_back({0, 26});  // (the pockets slide in and take touches)
}

void macro_close_pockets(std::vector<Step> &m) {
    const profile::Tools &c = profile::tools();
    m.push_back({0, 6});
    for (int i = 0; i < 4; i++) {  // B until they close (it is not taken while they are busy)
        Step b{1u << 1, 6};
        b.if_cond = c.open_cond;
        m.push_back(b);
        Step w;
        w.wait_cond = c.open_cond;
        w.wait_not = true;
        w.frames = 40;
        w.if_cond = c.open_cond;
        m.push_back(w);
    }
}


// holds a tool: taps its pocket and picks the menu's first choice ("Hold"), or puts the held one away: taps the
// portrait and picks its first choice ("Remove Equipment"); the pockets stay out of sight meanwhile
void use_tool(const Tool &t) {
    const profile::Tools &c = profile::tools();
    if (!profile::cond(c.ready_cond)) {
        say("Not now");
        return;
    }
    if ((int)c.slots.size() < kSlots) return;
    std::vector<Step> m;
    macro_open_pockets(m);
    if (t.slot >= 0) tap(m, c.slots[t.slot].first, c.slots[t.slot].second);
    else tap(m, c.portrait_x, c.portrait_y);
    m.push_back({0, 16});       // (the choices open)
    m.push_back({1u << 0, 6});  // A: the first choice
    m.push_back({0, 14});
    macro_close_pockets(m);
    macro = m;
    step_frame = 0;
    load_items();
    say(t.slot >= 0 ? item_name(t.item) : "Hands free");
}

// the next or previous tool after the one in hand (quick switching), steps ahead (presses made while one was
// being taken out count)
int pending_steps;
void cycle_tool(int steps) {
    const int dir = steps > 0 ? 1 : -1;
    load_items();
    u32 held;
    std::vector<Tool> v = tool_list(held);
    std::vector<Tool> pockets;
    for (auto &t : v)
        if (t.slot >= 0) pockets.push_back(t);
    if (pockets.empty()) {
        say("No tools in your pockets");
        return;
    }
    // the pockets in order: the tool after (or before) the last one taken out, whose slot the held one takes
    static int last = -1;
    const Tool *pick = nullptr;
    for (int n = 0; n < std::abs(steps); n++) {
        pick = nullptr;
        for (auto &t : pockets)
            if (dir > 0 ? t.slot > last && (!pick || t.slot < pick->slot) : t.slot < last && (!pick || t.slot > pick->slot))
                pick = &t;
        if (!pick) pick = dir > 0 ? &pockets.front() : &pockets.back();
        last = pick->slot;
    }
    use_tool(*pick);
}

std::vector<int> offered_hub() {
    std::vector<int> v;
    const auto &h = profile::hub();
    for (size_t i = 0; i < h.size(); i++)
        if (profile::cond(h[i].cond)) v.push_back((int)i);
    return v;
}

void open_overlay(Overlay o);

void activate_hub(int i) {
    const profile::HubEntry &e = profile::hub()[i];
    overlay = NONE;
    std::vector<Step> m;
    switch (e.action) {
    case profile::HubEntry::TAP: tap(m, e.x, e.y); break;
    case profile::HubEntry::PRESS: m.push_back({e.buttons, 6}); break;
    case profile::HubEntry::MAP: open_overlay(MAP); return;
    case profile::HubEntry::TOOLS: open_overlay(TOOLS); return;
    }
    macro = m;
    step_frame = 0;
}

const profile::MapArea *map_area() {
    for (auto &a : profile::map_areas())
        if (profile::cond(a.cond)) return &a;
    return nullptr;
}

void open_overlay(Overlay o) {
    if (o == MAP && !map_area()) {
        say("No map here");
        return;
    }
    if (o == TOOLS) {
        load_items();
        u32 held;
        std::vector<Tool> v = tool_list(held);
        if (v.empty()) {
            say("No tools in your pockets");
            return;
        }
        if (!profile::cond(profile::tools().ready_cond)) {
            say("Not now");
            return;
        }
        tool_sel = 0;
        hold_dir = true;  // (the press that opened it does not move it)
    }
    if (o == HUB) {
        if (offered_hub().empty()) {
            say("Not now");
            return;
        }
        hub_sel = 0;
        hold_dir = true;
    }
    overlay = o;
}

// one step of a direction this frame (pressed, or held and repeating): bit of H_DUP.. or 0
u32 direction(u32 host, s16 cx, s16 cy) {
    u32 d = host & (H_DUP | H_DDOWN | H_DLEFT | H_DRIGHT);
    if (cy > kStick) d |= H_DUP;
    if (cy < -kStick) d |= H_DDOWN;
    if (cx < -kStick) d |= H_DLEFT;
    if (cx > kStick) d |= H_DRIGHT;
    if (!d) {
        repeat_dir = 0;
        hold_dir = false;
        return 0;
    }
    if (hold_dir) return 0;
    if (d != repeat_dir) {
        repeat_dir = d;
        repeat_frames = 0;
        return d;
    }
    repeat_frames++;
    return repeat_frames >= kRepeatDelay && (repeat_frames - kRepeatDelay) % kRepeatEvery == 0 ? d : 0;
}

// the 3DS button an unbound host input plays
u32 default_button(u32 h) {
    switch (h) {
    case H_A: return 1 << 0;
    case H_B: return 1 << 1;
    case H_BACK: return 1 << 2;
    case H_START: return 1 << 3;
    case H_DRIGHT: return 1 << 4;
    case H_DLEFT: return 1 << 5;
    case H_DUP: return 1 << 6;
    case H_DDOWN: return 1 << 7;
    case H_R: return 1 << 8;
    case H_L: return 1 << 9;
    case H_X: return 1 << 10;
    case H_Y: return 1 << 11;
    case H_ZL: return 1 << 14;
    case H_ZR: return 1 << 15;
    default: return 0;
    }
}

struct HostName {
    const char *name;
    u32 bit;
};
const HostName kHostNames[] = {
    {"a", H_A}, {"b", H_B}, {"x", H_X}, {"y", H_Y}, {"l", H_L}, {"r", H_R}, {"zl", H_ZL}, {"zr", H_ZR},
    {"start", H_START}, {"back", H_BACK}, {"ls", H_LS}, {"rs", H_RS}, {"dup", H_DUP}, {"ddown", H_DDOWN},
    {"dleft", H_DLEFT}, {"dright", H_DRIGHT}, {"rs-up", H_RSUP}, {"rs-down", H_RSDOWN}, {"rs-left", H_RSLEFT},
    {"rs-right", H_RSRIGHT},
};

u32 host_bit(const std::string &n) {
    for (auto &h : kHostNames)
        if (n == h.name) return h.bit;
    return 0;
}

// the bound target of a host input ("" when unbound)
const profile::Bind *bound(u32 bit) {
    for (auto &b : profile::binds())
        if (host_bit(b.host) == bit) return &b;
    return nullptr;
}

// ---- drawing
std::map<std::string, text::Texture> text_cache;

const text::Texture &label(SDL_Renderer *r, const std::string &s, float size, SDL_Color c, bool bold) {
    char key[64];
    snprintf(key, sizeof key, "|%.1f|%02x%02x%02x%02x|%d", size, c.r, c.g, c.b, c.a, bold);
    auto it = text_cache.find(s + key);
    if (it != text_cache.end()) return it->second;
    if (text_cache.size() > 300) {
        for (auto &[k, t] : text_cache) SDL_DestroyTexture(t.tex);
        text_cache.clear();
    }
    return text_cache[s + key] = text::render(r, s, size, c, bold);
}

// text at (x, y) in render units (align: 0 left, 1 centre, 2 right), size in top-screen pixels
void put_text(SDL_Renderer *r, const std::string &s, float x, float y, float size, SDL_Color c, bool bold, int align,
              float px) {
    const float k = top_r.w / 400;
    const text::Texture &t = label(r, s, size * k * px, c, bold);
    if (!t.tex) return;
    const float w = t.w / px, h = t.h / px;
    const SDL_FRect d{x - (align == 1 ? w / 2 : align == 2 ? w : 0), y, w, h};
    SDL_RenderTexture(r, t.tex, nullptr, &d);
}

SDL_Texture *icon_tex(SDL_Renderer *r, Icon &ic) {
    if (ic.dirty) {
        if (!ic.tex || ic.tex->w != ic.w || ic.tex->h != ic.h) {
            if (ic.tex) SDL_DestroyTexture(ic.tex);
            ic.tex = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, ic.w, ic.h);
            SDL_SetTextureBlendMode(ic.tex, SDL_BLENDMODE_BLEND);
            SDL_SetTextureScaleMode(ic.tex, SDL_SCALEMODE_LINEAR);
        }
        SDL_UpdateTexture(ic.tex, nullptr, ic.px.data(), ic.w * 4);
        ic.dirty = false;
    }
    return ic.tex;
}

SDL_Texture *item_icon(SDL_Renderer *r, u32 item) {
    auto it = item_icons.find(item & 0xffff);
    return it == item_icons.end() ? nullptr : icon_tex(r, it->second);
}

void pill(SDL_Renderer *r, const std::string &s, float cx, float y, float px, float a) {
    const float k = top_r.w / 400;
    const float w = text::width(s, 11 * k * px, true) / px + 16 * k, h = 20 * k;
    rounded_rect(r, nullptr, {cx - w / 2, y, w, h}, h / 2, {0.12f, 0.11f, 0.1f, 0.82f * a});
    put_text(r, s, cx, y + 3 * k, 11, {255, 255, 255, (Uint8)(255 * a)}, true, 1, px);
}

std::string host_label(u32 bit) {
    switch (bit) {
    case H_ZL: return "ZL";
    case H_ZR: return "ZR";
    case H_L: return "L";
    case H_R: return "R";
    case H_BACK: return "\xe2\x88\x92";  // minus
    case H_START: return "+";
    case H_DUP: return "\xe2\x86\x91";   // arrows
    case H_DDOWN: return "\xe2\x86\x93";
    case H_DLEFT: return "\xe2\x86\x90";
    case H_DRIGHT: return "\xe2\x86\x92";
    default: return "";
    }
}

std::string clock_text() {
    const time_t t = time(nullptr) + (time_t)g_settings.clock_offset;
    struct tm tm;
    localtime_r(&t, &tm);
    char b[32];
    strftime(b, sizeof b, "%l:%M %p", &tm);
    std::string s = b;
    return s[0] == ' ' ? s.substr(1) : s;
}

// whether an overlay target can be used now (else its input plays its own 3DS button)
bool available(const std::string &target) {
    if (panel_r.w) return false;  // a menu is up
    if (target == "hub") return !offered_hub().empty();
    if (target == "map") return map_area() != nullptr;
    if (target.rfind("tool", 0) == 0) return profile::cond(profile::tools().ready_cond);
    return true;
}

}  // namespace

u32 host_mask(const char *names) {
    u32 m = 0;
    std::string s = names;
    for (char &c : s) c = (char)tolower(c);
    for (size_t p = 0; p <= s.size();) {
        size_t e = s.find('+', p);
        if (e == std::string::npos) e = s.size();
        m |= host_bit(s.substr(p, e - p));
        p = e + 1;
    }
    return m;
}

void set_state(bool active, const SDL_FRect &top, const SDL_FRect &panel) {
    if (active && !on && hint_left < 0) hint_left = kHintFrames;
    if (!active) overlay = NONE;
    on = active;
    top_r = top;
    panel_r = panel;
}

bool active() { return on; }
void toggle_hub() {
    if (overlay == HUB) overlay = NONE;
    else if (overlay == NONE && available("hub")) open_overlay(HUB);
    else if (overlay != NONE) overlay = NONE;
    else say("Not now");
}
bool busy() { return on && !macro.empty(); }
bool overlay_open() { return on && overlay != NONE; }

void input(u32 host, float rsx, float rsy, u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty) {
    const u32 pressed = host & ~prev_host;
    prev_host = host;
    frames_seen++;
    if (toast_left > 0) toast_left--;
    const bool field = profile::tools().ready_cond >= 0 ? profile::cond(profile::tools().ready_cond) : !offered_hub().empty();
    if (hint_left > 0 && overlay == NONE && !panel_r.w && field) hint_left--;  // (in the field)
    if (!on) {
        macro.clear();
        return;
    }
    // a scripted sequence plays alone (B or the start of another input does not stop it: it is short)
    while (!macro.empty() && step_frame == 0 && macro.front().if_cond >= -1 && !profile::cond(macro.front().if_cond))
        macro.erase(macro.begin());
    static int pending_wait;
    if (macro.empty() && pending_steps) {  // tool switches asked for meanwhile, once the field is back
        if (profile::cond(profile::tools().ready_cond) && !panel_r.w) {
            const int n = pending_steps;
            pending_steps = 0;
            cycle_tool(n);
        } else if (++pending_wait > 180) {
            pending_steps = 0;
        }
    } else {
        pending_wait = 0;
    }
    if (!macro.empty()) {
        for (const HostName &h : kHostNames)  // (switching on while one is being taken out: queued)
            if (pressed & h.bit)
                if (const profile::Bind *b = bound(h.bit)) {
                    if (b->target == "tool-next") pending_steps++;
                    if (b->target == "tool-prev") pending_steps--;
                }
        Step &s = macro.front();
        buttons = s.buttons;
        cx = cy = 0;
        touch = false;
        if (s.touch) {
            const float t = s.frames > 1 ? (float)step_frame / (s.frames - 1) : 1;
            touch = true;
            tx = (u16)std::clamp((int)std::lround(s.x0 + (s.x1 - s.x0) * t), 0, 319);
            ty = (u16)std::clamp((int)std::lround(s.y0 + (s.y1 - s.y0) * t), 0, 239);
        }
        bool done = ++step_frame >= s.frames;
        const bool met = s.wait_cond >= -1 && profile::cond(s.wait_cond) != s.wait_not;
        if (done && s.or_abort && !met) {
            macro.clear();
            step_frame = 0;
            say("Not now");
            return;
        }
        if (done || met) {
            macro.erase(macro.begin());
            step_frame = 0;
        }
        return;
    }
    // the pointer: the right stick over the panel, a bound input touches where it points
    const bool panel = panel_r.w > 0 && overlay == NONE;
    if (panel && (std::fabs(rsx) > 0.2f || std::fabs(rsy) > 0.2f)) {
        auto curve = [](float v) { return (v < 0 ? -1 : 1) * std::max(0.0f, std::fabs(v) - 0.2f) / 0.8f; };
        ptr_x = std::clamp(ptr_x + curve(rsx) * std::fabs(curve(rsx)) * kPointerSpeed * 2, 0.0f, 319.0f);
        ptr_y = std::clamp(ptr_y - curve(rsy) * std::fabs(curve(rsy)) * kPointerSpeed * 2, 0.0f, 239.0f);  // (up: -y)
        ptr_idle = 0;
    } else {
        ptr_idle++;
    }
    // binds: unbound inputs keep their default button; bound ones play their target, or open an overlay where it
    // is available (the player free in the field: no menu, no dialog), else keep their own button too (the d-pad
    // moves through a dialog's choices)
    for (const HostName &h : kHostNames) {
        const profile::Bind *b = bound(h.bit);
        if (!b) continue;
        const bool stick = h.bit & (H_RSUP | H_RSDOWN | H_RSLEFT | H_RSRIGHT);
        if (stick && panel) {  // (the right stick moves the pointer over a panel)
            buttons &= ~default_button(h.bit);
            continue;
        }
        if (b->buttons) {
            buttons &= ~default_button(h.bit);
            if (host & h.bit) {
                buttons |= b->buttons & 0x0fffffff;
                if (b->buttons & 1u << 30) cy = 155;
                if (b->buttons & 1u << 31) cy = -155;
                if (b->buttons & 1u << 29) cx = -155;
                if (b->buttons & 1u << 28) cx = 155;
            }
            continue;
        }
        if (b->target == "pointer") {
            if (!panel) continue;
            buttons &= ~default_button(h.bit);
            if (host & h.bit) {
                touch = true;
                tx = (u16)ptr_x;
                ty = (u16)ptr_y;
                ptr_idle = 0;
            }
            continue;
        }
        if (overlay != NONE) {  // its own input closes it
            buttons &= ~default_button(h.bit);
            const bool mine = (overlay == HUB && b->target == "hub") || (overlay == MAP && b->target == "map");
            if (mine && (pressed & h.bit)) {
                overlay = NONE;
                prev_host |= host;  // (not seen as pressed again)
                buttons = 0;
                return;
            }
            continue;
        }
        if (!available(b->target)) {
            // (the hub and the map have no button of their own on the game's side: say why nothing happens)
            if ((pressed & h.bit) && !panel_r.w && (b->target == "hub" || b->target == "map")) say("Not now");
            continue;
        }
        buttons &= ~default_button(h.bit);
        if (!(pressed & h.bit)) continue;
        if (b->target == "hub") open_overlay(HUB);
        else if (b->target == "map") open_overlay(MAP);
        else if (b->target == "tools") open_overlay(TOOLS);
        else if (b->target == "tool-next") cycle_tool(1);
        else if (b->target == "tool-prev") cycle_tool(-1);
        else if (b->target == "tool-away") {
            load_items();
            u32 held;
            std::vector<Tool> v = tool_list(held);
            if (!v.empty() && v.back().slot < 0) use_tool(v.back());
        }
    }
    if (overlay == NONE) return;
    // an overlay takes the game's input
    const s16 lx = cx, ly = cy;
    buttons = 0;
    cx = cy = 0;
    touch = false;
    const u32 d = direction(host, lx, ly);
    switch (overlay) {
    case HUB: {
        std::vector<int> v = offered_hub();
        if (v.empty()) {
            overlay = NONE;
            break;
        }
        constexpr int kCols = 3;
        const int n = (int)v.size();
        hub_sel = std::clamp(hub_sel, 0, n - 1);
        if (d & H_DRIGHT) hub_sel = std::min(n - 1, hub_sel + 1);
        if (d & H_DLEFT) hub_sel = std::max(0, hub_sel - 1);
        if (d & H_DDOWN && hub_sel + kCols < n) hub_sel += kCols;
        if (d & H_DUP && hub_sel - kCols >= 0) hub_sel -= kCols;
        if (pressed & H_A) activate_hub(v[hub_sel]);
        else if (pressed & H_B) overlay = NONE;
        break;
    }
    case MAP:
        if (pressed & (H_A | H_B)) overlay = NONE;
        break;
    case TOOLS: {
        u32 held;
        std::vector<Tool> v = tool_list(held);
        if (v.empty()) {
            overlay = NONE;
            break;
        }
        const int n = (int)v.size();
        if (d & (H_DRIGHT | H_DDOWN)) tool_sel = (tool_sel + 1) % n;
        if (d & (H_DLEFT | H_DUP)) tool_sel = (tool_sel + n - 1) % n;
        tool_sel = std::clamp(tool_sel, 0, n - 1);
        if (pressed & H_A) {
            overlay = NONE;
            use_tool(v[tool_sel]);
        } else if (pressed & H_B) {
            overlay = NONE;
        }
        break;
    }
    case NONE: break;
    }
}

bool click(float x, float y) {
    if (!on || overlay == NONE) return false;
    const float k = top_r.w / 400;
    if (overlay == HUB) {
        std::vector<int> v = offered_hub();
        constexpr int kCols = 3;
        const int rows = ((int)v.size() + kCols - 1) / kCols;
        const float pw = 168 * k, ph = (62 + rows * 54) * k, px0 = top_r.x + 16 * k, py0 = top_r.y + (top_r.h - ph) / 2;
        for (size_t i = 0; i < v.size(); i++) {
            const float cxp = px0 + 14 * k + (i % kCols) * 50 * k, cyp = py0 + 40 * k + (i / kCols) * 54 * k;
            if (x >= cxp && x < cxp + 46 * k && y >= cyp && y < cyp + 50 * k) {
                activate_hub(v[i]);
                return true;
            }
        }
        if (x < px0 || x > px0 + pw || y < py0 || y > py0 + ph) overlay = NONE;
        return true;
    }
    overlay = NONE;
    return true;
}

// a part of a picture (bottom-screen pixels) into an icon; false if it is outside
bool crop(Icon &ic, const u32 *img, int w, int h, int bx, int by, int bw, int bh) {
    const float s = w / 320.0f;
    const int x0 = (int)(bx * s), y0 = (int)(by * s), iw = (int)(bw * s), ih = (int)(bh * s);
    if (x0 < 0 || y0 < 0 || iw <= 0 || ih <= 0 || x0 + iw > w || y0 + ih > h) return false;
    ic.px.resize((size_t)iw * ih);
    for (int y = 0; y < ih; y++) memcpy(&ic.px[(size_t)y * iw], &img[(size_t)(y0 + y) * w + x0], iw * 4);
    ic.w = iw;
    ic.h = ih;
    ic.dirty = true;
    return true;
}

void bottom_image(const u32 *img, int w, int h) {
    // the items' icons from the pockets while they show (they have settled once open a while)
    const profile::Tools &c = profile::tools();
    static int open_frames;
    open_frames = profile::cond(c.open_cond) && c.open_cond >= 0 ? open_frames + 1 : 0;
    // (an icon is kept when two looks 10 frames apart agree: not while the pockets slide in or out)
    static Icon last[kSlots];
    static u32 last_item[kSlots];
    if (open_frames > 30 && open_frames % 10 == 0 && (int)c.slots.size() >= kSlots) {
        u32 p[kSlots], held;
        if (read_pockets(p, held))
            for (int i = 0; i < kSlots; i++) {
                Icon now;
                const u32 item = p[i] & 0xffff;
                if (item == c.empty || !crop(now, img, w, h, c.slots[i].first - c.icon_size / 2,
                                             c.slots[i].second - c.icon_size / 2, c.icon_size, c.icon_size))
                    continue;
                if (last_item[i] == item && last[i].px == now.px) item_icons[item] = now;
                last[i] = std::move(now);
                last_item[i] = item;
            }
    }
    if (!open_frames)
        for (u32 &i : last_item) i = 0;
    // keep the hub's icons from a picture where their entry is offered (once a second is enough)
    const auto &hub = profile::hub();
    if (hub_icons.size() != hub.size()) hub_icons.assign(hub.size(), {});
    if (frames_seen % 30) return;
    for (size_t i = 0; i < hub.size(); i++) {
        const profile::HubEntry &e = hub[i];
        if (!e.iw || !profile::cond(e.cond)) continue;
        crop(hub_icons[i], img, w, h, e.ix, e.iy, e.iw, e.ih);
    }
}

void rounded_rect(SDL_Renderer *r, SDL_Texture *t, const SDL_FRect &d, float rad, SDL_FColor c) {
    constexpr int kSeg = 6;  // segments per corner
    rad = std::min(rad, std::min(d.w, d.h) / 2);
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    auto vert = [&](float x, float y) {
        SDL_Vertex q;
        q.position = {x, y};
        q.color = c;
        q.tex_coord = {(x - d.x) / d.w, (y - d.y) / d.h};
        v.push_back(q);
    };
    vert(d.x + d.w / 2, d.y + d.h / 2);
    const float ox[4] = {d.x + d.w - rad, d.x + rad, d.x + rad, d.x + d.w - rad};
    const float oy[4] = {d.y + rad, d.y + rad, d.y + d.h - rad, d.y + d.h - rad};
    for (int q = 0; q < 4; q++)
        for (int i = 0; i <= kSeg; i++) {
            const float a = (float)M_PI / 2 * (q + (float)i / kSeg);
            vert(ox[q] + std::cos(a) * rad, oy[q] - std::sin(a) * rad);
        }
    const int n = (int)v.size() - 1;
    for (int i = 1; i <= n; i++) {
        idx.push_back(0);
        idx.push_back(i);
        idx.push_back(i % n + 1);
    }
    SDL_RenderGeometry(r, t, v.data(), (int)v.size(), idx.data(), (int)idx.size());
}

void draw(SDL_Renderer *r, SDL_Texture *bottom, float px) {
    if (!on) return;
    const float k = top_r.w / 400;
    const SDL_Color ink{70, 52, 34, 255}, white{255, 255, 255, 255};
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    auto dim = [&](float a) {
        SDL_SetRenderDrawColorFloat(r, 0, 0, 0, a);
        SDL_RenderFillRect(r, &top_r);
    };
    switch (overlay) {
    case MAP: {
        const profile::MapArea *a = map_area();
        if (!a || !bottom) break;
        dim(0.55f);
        const float s = bottom->w / 320.0f;
        const SDL_FRect src{a->x * s, a->y * s, a->w * s, a->h * s};
        const float h = top_r.h * 0.84f, w = h * a->w / a->h;
        const SDL_FRect d{top_r.x + (top_r.w - w) / 2, top_r.y + (top_r.h - h) / 2 + 6 * k, w, h};
        rounded_rect(r, nullptr, {d.x - 4 * k, d.y - 4 * k, d.w + 8 * k, d.h + 8 * k}, 10 * k, {1, 0.98f, 0.92f, 1});
        SDL_SetTextureBlendMode(bottom, SDL_BLENDMODE_BLEND);
        SDL_SetTextureScaleMode(bottom, SDL_SCALEMODE_LINEAR);
        SDL_RenderTexture(r, bottom, &src, &d);
        SDL_SetTextureBlendMode(bottom, SDL_BLENDMODE_NONE);
        put_text(r, "Map", d.x, top_r.y + 2 * k, 13, white, true, 0, px);
        put_text(r, "B  Close", d.x + d.w, top_r.y + 4 * k, 10, white, false, 2, px);
        break;
    }
    case HUB: {
        dim(0.35f);
        std::vector<int> v = offered_hub();
        constexpr int kCols = 3;
        const int rows = ((int)v.size() + kCols - 1) / kCols;
        const float pw = 168 * k, ph = (62 + rows * 54) * k, x0 = top_r.x + 16 * k, y0 = top_r.y + (top_r.h - ph) / 2;
        rounded_rect(r, nullptr, {x0 - 2 * k, y0 + 2 * k, pw + 4 * k, ph + 4 * k}, 20 * k, {0, 0, 0, 0.25f});
        rounded_rect(r, nullptr, {x0, y0, pw, ph}, 18 * k, {0.99f, 0.97f, 0.91f, 0.98f});
        put_text(r, clock_text(), x0 + pw / 2, y0 + 8 * k, 14, ink, true, 1, px);
        for (size_t i = 0; i < v.size(); i++) {
            const profile::HubEntry &e = profile::hub()[v[i]];
            const float cx0 = x0 + 14 * k + (i % kCols) * 50 * k, cy0 = y0 + 40 * k + (i / kCols) * 54 * k;
            const bool sel = (int)i == hub_sel;
            if (sel) rounded_rect(r, nullptr, {cx0 - 2 * k, cy0 - 2 * k, 50 * k, 54 * k}, 10 * k, {0.36f, 0.76f, 0.69f, 0.9f});
            rounded_rect(r, nullptr, {cx0 + 5 * k, cy0 + 2 * k, 36 * k, 32 * k}, 9 * k,
                         sel ? SDL_FColor{1, 1, 1, 1} : SDL_FColor{0.93f, 0.89f, 0.8f, 1});
            Icon *ic = (size_t)v[i] < hub_icons.size() ? &hub_icons[v[i]] : nullptr;
            if (ic && !ic->px.empty()) icon_tex(r, *ic);
            const profile::MapArea *ma = e.action == profile::HubEntry::MAP ? map_area() : nullptr;
            SDL_Texture *tool_icon = nullptr;
            if (e.action == profile::HubEntry::TOOLS) {  // the tool in hand, or the first in the pockets
                load_items();
                u32 held;
                std::vector<Tool> tl = tool_list(held);
                if (!tl.empty()) tool_icon = item_icon(r, tl.back().slot < 0 ? held : tl.front().item);
            }
            if (ma && bottom) {  // the map itself
                const float sc = bottom->w / 320.0f, ih = 28 * k, iw = ih * ma->w / ma->h;
                const SDL_FRect src{ma->x * sc, ma->y * sc, ma->w * sc, ma->h * sc};
                const SDL_FRect d{cx0 + 23 * k - iw / 2, cy0 + 4 * k, iw, ih};
                SDL_SetTextureBlendMode(bottom, SDL_BLENDMODE_BLEND);
                SDL_RenderTexture(r, bottom, &src, &d);
                SDL_SetTextureBlendMode(bottom, SDL_BLENDMODE_NONE);
            } else if (tool_icon) {
                const SDL_FRect d{cx0 + 9 * k, cy0 + 2 * k, 28 * k, 28 * k};
                SDL_RenderTexture(r, tool_icon, nullptr, &d);
            } else if (ic && ic->tex) {
                const float ih = 26 * k, iw = std::min(34 * k, ih * ic->w / std::max(1, ic->h));
                const SDL_FRect d{cx0 + 23 * k - iw / 2, cy0 + 5 * k, iw, ih};
                SDL_RenderTexture(r, ic->tex, nullptr, &d);
            } else if (!e.glyph.empty()) {
                put_text(r, e.glyph, cx0 + 23 * k, cy0 + 6 * k, 17, ink, false, 1, px);
            } else {
                put_text(r, e.label.substr(0, 1), cx0 + 23 * k, cy0 + 7 * k, 16, ink, true, 1, px);
            }
            put_text(r, e.label, cx0 + 23 * k, cy0 + 36 * k, 8.5f, sel ? white : ink, true, 1, px);
        }
        put_text(r, "A  Open     B  Close", x0 + pw / 2, y0 + ph - 17 * k, 9, ink, false, 1, px);
        break;
    }
    case TOOLS: {
        u32 held;
        std::vector<Tool> v = tool_list(held);
        if (v.empty()) break;
        dim(0.3f);
        const int n = (int)v.size();
        const float cx0 = top_r.x + top_r.w / 2, cy0 = top_r.y + top_r.h / 2 - 6 * k, rad = n > 1 ? 70 * k : 0;
        for (int i = 0; i < n; i++) {
            const float a = (float)M_PI * 2 * i / n - (float)M_PI / 2;
            const bool sel = i == tool_sel;
            const float s = (sel ? 44 : 34) * k, x = cx0 + std::cos(a) * rad, y = cy0 + std::sin(a) * rad;
            rounded_rect(r, nullptr, {x - s / 2 - 2 * k, y - s / 2 - 2 * k, s + 4 * k, s + 4 * k}, s / 2 + 2 * k,
                         sel ? SDL_FColor{0.36f, 0.76f, 0.69f, 1} : SDL_FColor{0, 0, 0, 0.25f});
            rounded_rect(r, nullptr, {x - s / 2, y - s / 2, s, s}, s / 2, {1, 0.98f, 0.93f, 1});
            if (v[i].slot < 0) {
                put_text(r, "\xe2\x9c\x8b", x, y - 9 * k, 14, ink, false, 1, px);  // a raised hand: put away
            } else if (SDL_Texture *t = item_icon(r, v[i].item)) {
                const SDL_FRect d{x - s * 0.45f, y - s * 0.45f, s * 0.9f, s * 0.9f};
                SDL_RenderTexture(r, t, nullptr, &d);
            } else {  // (not seen in the pockets yet)
                put_text(r, item_name(v[i].item).substr(0, 1), x, y - 10 * k, 16, ink, true, 1, px);
            }
        }
        const Tool &t = v[std::clamp(tool_sel, 0, n - 1)];
        const std::string name = t.slot < 0 ? "Put away" : item_name(t.item);
        pill(r, name, cx0, n > 1 ? cy0 - 10 * k : cy0 + 30 * k, px, 1);  // in the ring, or under a lone tool
        put_text(r, "A  Hold     B  Close", cx0, top_r.y + top_r.h - 20 * k, 10, white, false, 1, px);
        break;
    }
    case NONE: break;
    }
    // the pointer over the panel
    if (panel_r.w && ptr_idle < 120 && overlay == NONE) {
        const float a = ptr_idle < 90 ? 1 : (120 - ptr_idle) / 30.0f;
        const float x = panel_r.x + ptr_x * panel_r.w / 320, y = panel_r.y + ptr_y * panel_r.h / 240;
        const bool down = (prev_host & H_ZR) != 0;  // (pressed: smaller, as a finger touching)
        const float o = down ? 8 : 10;
        rounded_rect(r, nullptr, {x - o * k, y - o * k, 2 * o * k, 2 * o * k}, o * k, {0, 0, 0, 0.35f * a});
        rounded_rect(r, nullptr, {x - (o - 2) * k, y - (o - 2) * k, 2 * (o - 2) * k, 2 * (o - 2) * k}, (o - 2) * k,
                     {1, 1, 1, 0.95f * a});
        rounded_rect(r, nullptr, {x - 3.5f * k, y - 3.5f * k, 7 * k, 7 * k}, 3.5f * k, {0.36f, 0.76f, 0.69f, a});
    }
    if (toast_left > 0) {
        const float a = std::min(1.0f, toast_left / 15.0f);
        pill(r, toast, top_r.x + top_r.w / 2, top_r.y + top_r.h - 30 * k, px, a);
    }
    // what the controls do, for a while after one-screen play starts
    const bool field = profile::tools().ready_cond >= 0 ? profile::cond(profile::tools().ready_cond) : !offered_hub().empty();
    if (hint_left > 0 && overlay == NONE && !panel_r.w && field) {
        std::string s;
        auto add = [&](const char *target, const char *what) {
            for (auto &b : profile::binds())
                if (b.target == target) {
                    const std::string l = host_label(host_bit(b.host));
                    if (l.empty()) continue;
                    if (!s.empty()) s += "    ";
                    s += l + " " + what;
                    return;
                }
        };
        add("hub", "Hub");
        add("map", "Map");
        add("tools", "Tools");
        if (!s.empty()) pill(r, s, top_r.x + top_r.w / 2, top_r.y + 6 * k, px, std::min(1.0f, hint_left / 30.0f));
    }
}

}  // namespace onescreen
