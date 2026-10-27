// The in-game menu. Pages are lists of rows rebuilt every frame from the current state (an immediate-mode
// layout), drawn with SDL's renderer in window pixels; text is rasterised with Core Text (the system font)
// into cached textures. Keyboard, gamepad and mouse drive it.
#include "menu.h"

#import <AppKit/AppKit.h>
#import <CoreText/CoreText.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "audio_iface.h"
#include "loader.h"
#include "settings.h"

namespace kernel {
std::string save_root();  // (kernel.h clashes with MacTypes.h)
}

namespace menu {
namespace {

SDL_Window *win;
SDL_Renderer *ren;
bool open_ = false;
float fps = 0;

// ---- text
struct Text {
    SDL_Texture *tex;
    int w, h;
};
std::map<std::string, Text> text_cache;

const Text &text(const std::string &s, float px, SDL_Color c, bool bold = false) {
    char key[64];
    snprintf(key, sizeof key, "|%.1f|%02x%02x%02x%02x|%d", px, c.r, c.g, c.b, c.a, bold);
    auto it = text_cache.find(s + key);
    if (it != text_cache.end()) return it->second;
    if (text_cache.size() > 400) {
        for (auto &[k, t] : text_cache) SDL_DestroyTexture(t.tex);
        text_cache.clear();
    }
    CTFontRef font = CTFontCreateUIFontForLanguage(bold ? kCTFontUIFontEmphasizedSystem : kCTFontUIFontSystem, px, nullptr);
    CGColorRef col = CGColorCreateSRGB(c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0);
    CFStringRef str = CFStringCreateWithCString(nullptr, s.c_str(), kCFStringEncodingUTF8);
    const void *keys[] = {kCTFontAttributeName, kCTForegroundColorAttributeName};
    const void *vals[] = {font, col};
    CFDictionaryRef attrs = CFDictionaryCreate(nullptr, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks,
                                               &kCFTypeDictionaryValueCallBacks);
    CFAttributedStringRef as = CFAttributedStringCreate(nullptr, str ? str : CFSTR("?"), attrs);
    CTLineRef line = CTLineCreateWithAttributedString(as);
    CGFloat ascent, descent, leading;
    double width = CTLineGetTypographicBounds(line, &ascent, &descent, &leading);
    int w = std::max(1, (int)std::ceil(width) + 2), h = std::max(1, (int)std::ceil(ascent + descent) + 2);
    std::vector<uint32_t> px_(w * h, 0);
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGContextRef ctx = CGBitmapContextCreate(px_.data(), w, h, 8, w * 4, cs,
                                             (uint32_t)kCGImageAlphaPremultipliedLast | (uint32_t)kCGBitmapByteOrder32Big);
    CGContextSetTextPosition(ctx, 1, std::ceil(descent) + 1);
    CTLineDraw(line, ctx);
    CGContextFlush(ctx);
    SDL_Texture *t = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STATIC, w, h);
    SDL_UpdateTexture(t, nullptr, px_.data(), w * 4);
    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_BLEND_PREMULTIPLIED);
    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CFRelease(line);
    CFRelease(as);
    CFRelease(attrs);
    if (str) CFRelease(str);
    CGColorRelease(col);
    CFRelease(font);
    return text_cache[s + key] = {t, w, h};
}

// ---- pages
struct Row {
    std::string label, value, note;
    std::function<void(int)> change;  // value rows: left/right (-1/+1)
    std::function<void()> activate;   // action rows
    bool header = false;              // section title, not selectable
};
enum Page { MAIN, SAVES, BACKUP, CLOCK, GRAPHICS, SPEED, CONFIRM };
struct Level {
    Page page;
    int sel = 0;
};
std::vector<Level> stack;
std::string confirm_text;
std::function<void()> confirm_yes;
SaveBackup picked;
std::string toast;
Uint64 toast_until;
int msaa_at_launch, aniso_at_launch;  // the values the GPU backend started with

void say(const std::string &s) {
    toast = s;
    toast_until = SDL_GetTicks() + 3000;
}
void push(Page p) { stack.push_back({p, 0}); }
void pop() {
    if (stack.size() > 1) stack.pop_back();
    else toggle();
}
void confirm(const std::string &q, std::function<void()> yes) {
    confirm_text = q;
    confirm_yes = std::move(yes);
    push(CONFIRM);
}
template <class T> void cycle(T &v, const std::vector<T> &opts, int dir) {
    auto it = std::find(opts.begin(), opts.end(), v);
    int i = it == opts.end() ? 0 : (int)(it - opts.begin());
    v = opts[(i + dir + (int)opts.size()) % opts.size()];
}
std::string onoff(bool b) { return b ? "On" : "Off"; }
bool restart_pending() { return g_settings.msaa != msaa_at_launch || g_settings.aniso != aniso_at_launch; }

std::string clock_text(const char *fmt) {
    time_t t = time(nullptr) + (time_t)g_settings.clock_offset;
    char b[96];
    strftime(b, sizeof b, fmt, localtime(&t));
    return b;
}
// moves one field of the 3DS date (tm_year, tm_mon, ...) and keeps the rest
void shift_clock(int tm::*field, int dir) {
    time_t now = time(nullptr), t = now + (time_t)g_settings.clock_offset;
    tm lt;
    localtime_r(&t, &lt);
    lt.*field += dir;
    lt.tm_isdst = -1;
    g_settings.clock_offset = (s64)(mktime(&lt) - now);
}

std::vector<Row> rows(Page p) {
    std::vector<Row> r;
    Settings &s = g_settings;
    switch (p) {
    case MAIN:
        r.push_back({"Resume", "", "", nullptr, [] { toggle(); }});
        r.push_back({"Saves", "", "", nullptr, [] { push(SAVES); }});
        r.push_back({"Clock", clock_text("%d %b %Y %H:%M"), "", nullptr, [] { push(CLOCK); }});
        r.push_back({"Graphics", "", "", nullptr, [] { push(GRAPHICS); }});
        r.push_back({"Speed & audio", "", "", nullptr, [] { push(SPEED); }});
        if (restart_pending()) r.push_back({"Restart to apply the graphics changes", "", "", nullptr, [] { frontend_restart(); }});
        r.push_back({"Restart game", "", "", nullptr, [] {
                         confirm("Restart the game? Progress since its last save is lost.", [] { frontend_restart(); });
                     }});
        r.push_back({"Quit", "", "", nullptr, [] {
                         confirm("Quit? Progress since the game's last save is lost.", [] { frontend_quit(); });
                     }});
        break;
    case SAVES: {
        r.push_back({"Current save", save_summary(), "The game saves by itself; this is the data it keeps.", nullptr, nullptr, true});
        r.push_back({"Back up now", "", "", nullptr, [] { say(save_backup_now() ? "Backed up" : "Nothing to back up"); }});
        r.push_back({"Back up at each launch", onoff(s.auto_backup), "Keeps the last 5 automatic copies.",
                     [&s](int) { s.auto_backup = !s.auto_backup; }});
        r.push_back({"Show in Finder", "", "", nullptr, [] {
                         NSString *p = [NSString stringWithUTF8String:kernel::save_root().c_str()];
                         [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:p]];
                     }});
        if (save_exists())
            r.push_back({"Delete save…", "", "", nullptr, [] {
                             confirm("Delete the current save and restart the game? A copy is kept in the backups.", [] {
                                 if (save_delete_current()) frontend_restart();
                                 say("Could not delete the save");
                             });
                         }});
        auto bs = save_backups();
        r.push_back({bs.empty() ? "No backups yet" : "Backups", "", "", nullptr, nullptr, true});
        for (auto &b : bs)
            r.push_back({b.name, std::to_string((b.bytes + 1023) / 1024) + " KB", "", nullptr, [b] {
                             picked = b;
                             push(BACKUP);
                         }});
        break;
    }
    case BACKUP:
        r.push_back({"Backup " + picked.name, std::to_string((picked.bytes + 1023) / 1024) + " KB", "", nullptr, nullptr, true});
        r.push_back({"Restore this backup…", "", "", nullptr, [] {
                         confirm("Replace the current save with this backup and restart the game? The current save is "
                                 "backed up first.",
                                 [] {
                                     if (save_restore(picked)) frontend_restart();
                                     say("Could not restore the backup");
                                 });
                     }});
        r.push_back({"Delete this backup…", "", "", nullptr, [] {
                         confirm("Delete this backup? This cannot be undone.", [] {
                             say(save_delete_backup(picked) ? "Backup deleted" : "Could not delete it");
                             stack.pop_back();  // (the confirmation is already closed) leave the backup page
                         });
                     }});
        break;
    case CLOCK: {
        const s64 o = s.clock_offset;
        r.push_back({"3DS clock", clock_text("%a %d %b %Y, %H:%M:%S"),
                     o == 0 ? "Follows the Mac's clock." : "Set apart from the Mac's clock; it keeps running from there.",
                     nullptr, nullptr, true});
        r.push_back({"Year", clock_text("%Y"), "", [](int d) { shift_clock(&tm::tm_year, d); }});
        r.push_back({"Month", clock_text("%B"), "", [](int d) { shift_clock(&tm::tm_mon, d); }});
        r.push_back({"Day", clock_text("%d"), "", [](int d) { shift_clock(&tm::tm_mday, d); }});
        r.push_back({"Hour", clock_text("%H"), "", [](int d) { shift_clock(&tm::tm_hour, d); }});
        r.push_back({"Minute", clock_text("%M"), "", [](int d) { shift_clock(&tm::tm_min, d); }});
        r.push_back({"Use the Mac's clock", "", "Games read the clock when they need it; some notice when it moves.",
                     nullptr, [&s] { s.clock_offset = 0; }});
        break;
    }
    case GRAPHICS: {
        const int cur = frontend_current_scale();
        if (!cur) r.push_back({"The software renderer ignores these settings", "", "", nullptr, nullptr, true});
        r.push_back({"Internal resolution", s.scale ? std::to_string(s.scale) + "×" : "Auto (" + std::to_string(cur) + "×)",
                     "Auto renders at the window's size.", [&s](int d) {
                         cycle(s.scale, {0, 1, 2, 3, 4}, d);
                         frontend_set_scale_mode(s.scale);
                     }});
        r.push_back({"Anti-aliasing", s.msaa > 1 ? std::to_string(s.msaa) + "× MSAA" : "Off",
                     s.msaa != msaa_at_launch ? "Applies after a restart." : "", [&s](int d) { cycle(s.msaa, {1, 2, 4, 8}, d); }});
        r.push_back({"Texture filtering", s.aniso > 1 ? std::to_string(s.aniso) + "× anisotropic" : "Standard",
                     s.aniso != aniso_at_launch ? "Applies after a restart." : "",
                     [&s](int d) { cycle(s.aniso, {1, 2, 4, 8, 16}, d); }});
        r.push_back({"Separate windows", s.split ? "One per screen" : "Off",
                     "Put each screen on its own display; F11 makes the window you are in fullscreen.", [&s](int) {
                         s.split = !s.split;
                         frontend_set_split(s.split);
                     }});
        r.push_back({"Screen layout", frontend_layout_name(s.layout),
                     s.split ? "Used when the screens share one window." : "F2 cycles the layouts while playing.", [&s](int d) {
                         const int n = frontend_layout_count();
                         s.layout = (s.layout + d + n) % n;
                         frontend_set_layout(s.layout);
                     }});
        r.push_back({"Fullscreen", onoff(s.fullscreen), "", [&s](int) {
                         s.fullscreen = !s.fullscreen;
                         frontend_set_fullscreen(s.fullscreen);
                     }});
        if (restart_pending()) r.push_back({"Restart now to apply", "", "", nullptr, [] { frontend_restart(); }});
        break;
    }
    case SPEED: {
        char f[32];
        snprintf(f, sizeof f, "%.0f fps", fps);
        r.push_back({"Game frame rate", f, "As the game draws it: most 3DS games run at 30 or 60 fps.", nullptr, nullptr, true});
        r.push_back({"Game speed", std::to_string(s.speed) + "%",
                     s.speed == 100 ? "The 3DS's own pace (60 screen refreshes a second)."
                                    : "Everything runs at this pace, music included.",
                     [&s](int d) { cycle(s.speed, {50, 75, 100, 150, 200, 300}, d); }});
        r.push_back({"Frame rate counter", onoff(s.show_fps), "", [&s](int) { s.show_fps = !s.show_fps; }});
        r.push_back({"Volume", s.volume ? std::to_string(s.volume) + "%" : "Muted", "", [&s](int d) {
                         s.volume = std::clamp(s.volume + d * 10, 0, 100);
                         audio::set_volume(s.volume / 100.0f);
                     }});
        break;
    }
    case CONFIRM:
        r.push_back({confirm_text, "", "", nullptr, nullptr, true});
        r.push_back({"Cancel", "", "", nullptr, [] { pop(); }});
        r.push_back({"Confirm", "", "", nullptr, [] {
                         auto yes = confirm_yes;
                         stack.pop_back();
                         yes();
                     }});
        break;
    }
    return r;
}

const char *title(Page p) {
    switch (p) {
    case MAIN: return "Paused";
    case SAVES: return "Saves";
    case BACKUP: return "Backup";
    case CLOCK: return "Clock";
    case GRAPHICS: return "Graphics";
    case SPEED: return "Speed & audio";
    case CONFIRM: return "Are you sure?";
    }
    return "";
}

// ---- layout (window pixels), shared by draw() and mouse hit-testing
struct Layout {
    float k, x, y, w, row_h, top;
};
Layout layout() {
    int ow = 0, oh = 0;
    SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
    Layout l;
    l.k = std::clamp(std::min(ow / 640.0f, oh / 560.0f), 0.5f, 4.0f);
    l.w = std::min(ow - 24 * l.k, 560 * l.k);
    l.x = (ow - l.w) / 2;
    l.row_h = 40 * l.k;
    l.top = std::max(16 * l.k, oh * 0.08f);
    l.y = l.top + 64 * l.k;
    return l;
}
std::vector<float> row_y;  // top of each row from the last draw (for the mouse)

bool selectable(const Row &r) { return !r.header && (r.change || r.activate); }

void move(int dir) {
    auto r = rows(stack.back().page);
    int &sel = stack.back().sel;
    for (int n = 0; n < (int)r.size(); n++) {
        sel = (sel + dir + (int)r.size()) % (int)r.size();
        if (selectable(r[sel])) return;
    }
}
void fix_sel() {
    auto r = rows(stack.back().page);
    int &sel = stack.back().sel;
    sel = std::clamp(sel, 0, std::max(0, (int)r.size() - 1));
    if (!r.empty() && !selectable(r[sel])) move(1);
}
void act(int dir) {  // dir 0: activate, else change
    auto r = rows(stack.back().page);
    int sel = stack.back().sel;
    if (sel < 0 || sel >= (int)r.size()) return;
    if (dir && r[sel].change) r[sel].change(dir);
    else if (!dir && r[sel].activate) r[sel].activate();
    else if (!dir && r[sel].change) r[sel].change(1);
    if (!stack.empty()) fix_sel();
    settings_save();
}

void fill(float x, float y, float w, float h, SDL_Color c) {
    SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, c.a);
    SDL_FRect rc{x, y, w, h};
    SDL_RenderFillRect(ren, &rc);
}
void put(const Text &t, float x, float y) {
    SDL_FRect d{std::round(x), std::round(y), (float)t.w, (float)t.h};
    SDL_RenderTexture(ren, t.tex, nullptr, &d);
}

}  // namespace

void init(SDL_Window *w, SDL_Renderer *r) {
    win = w;
    ren = r;
    msaa_at_launch = g_settings.msaa;
    aniso_at_launch = g_settings.aniso;
}

bool is_open() { return open_; }

void toggle() {
    open_ = !open_;
    stack.clear();
    if (open_) {
        push(MAIN);
        fix_sel();
    }
    settings_save();
}

void set_fps(float f) { fps = f; }

bool handle_event(const SDL_Event &ev) {
    const bool key = ev.type == SDL_EVENT_KEY_DOWN, pad = ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
    if ((key && ev.key.scancode == SDL_SCANCODE_ESCAPE) || (pad && ev.gbutton.button == SDL_GAMEPAD_BUTTON_GUIDE)) {
        if (open_) pop();
        else toggle();
        return true;
    }
    if (!open_) return false;
    if (key) {
        switch (ev.key.scancode) {
        case SDL_SCANCODE_UP: move(-1); break;
        case SDL_SCANCODE_DOWN: move(1); break;
        case SDL_SCANCODE_LEFT: act(-1); break;
        case SDL_SCANCODE_RIGHT: act(1); break;
        case SDL_SCANCODE_RETURN:
        case SDL_SCANCODE_SPACE:
        case SDL_SCANCODE_X: act(0); break;
        case SDL_SCANCODE_BACKSPACE:
        case SDL_SCANCODE_Z: pop(); break;
        default: return false;  // F-keys etc. keep working
        }
        return true;
    }
    if (pad) {
        switch (ev.gbutton.button) {
        case SDL_GAMEPAD_BUTTON_DPAD_UP: move(-1); break;
        case SDL_GAMEPAD_BUTTON_DPAD_DOWN: move(1); break;
        case SDL_GAMEPAD_BUTTON_DPAD_LEFT: act(-1); break;
        case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: act(1); break;
        case SDL_GAMEPAD_BUTTON_EAST: act(0); break;  // the 3DS's A position
        case SDL_GAMEPAD_BUTTON_SOUTH:                // B
        case SDL_GAMEPAD_BUTTON_START: pop(); break;
        default: break;
        }
        return true;
    }
    if ((ev.type == SDL_EVENT_MOUSE_MOTION && ev.motion.windowID != SDL_GetWindowID(win)) ||
        (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.windowID != SDL_GetWindowID(win)))
        return true;  // the bottom screen's window, when the screens have separate windows
    if (ev.type == SDL_EVENT_MOUSE_MOTION || ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        const float d = SDL_GetWindowPixelDensity(win);
        const float mx = (ev.type == SDL_EVENT_MOUSE_MOTION ? ev.motion.x : ev.button.x) * d;
        const float my = (ev.type == SDL_EVENT_MOUSE_MOTION ? ev.motion.y : ev.button.y) * d;
        Layout l = layout();
        auto r = rows(stack.back().page);
        for (int i = 0; i < (int)row_y.size() && i < (int)r.size(); i++)
            if (selectable(r[i]) && mx >= l.x && mx < l.x + l.w && my >= row_y[i] && my < row_y[i] + l.row_h) {
                stack.back().sel = i;
                if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN)
                    act(r[i].change && ev.button.button == SDL_BUTTON_RIGHT ? -1 : r[i].change ? 1 : 0);
            }
        return true;
    }
    return ev.type == SDL_EVENT_KEY_UP || ev.type == SDL_EVENT_GAMEPAD_BUTTON_UP || ev.type == SDL_EVENT_MOUSE_BUTTON_UP;
}

void update() {
    if (!open_) return;
    // the left stick moves like the d-pad, with a repeat
    static Uint64 next;
    SDL_Gamepad *g = nullptr;
    int n = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    if (ids && n) g = SDL_GetGamepadFromID(ids[0]);
    SDL_free(ids);
    if (!g) return;
    const int ay = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY), ax = SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX);
    const Uint64 now = SDL_GetTicks();
    if (std::abs(ay) < 20000 && std::abs(ax) < 20000) {
        next = 0;
        return;
    }
    if (now < next) return;
    next = now + (next ? 120 : 350);
    if (std::abs(ay) >= std::abs(ax)) move(ay > 0 ? 1 : -1);
    else act(ax > 0 ? 1 : -1);
}

void draw() {
    const Settings &s = g_settings;
    int lw, lh;
    SDL_RendererLogicalPresentation mode;
    SDL_GetRenderLogicalPresentation(ren, &lw, &lh, &mode);
    SDL_SetRenderLogicalPresentation(ren, 0, 0, SDL_LOGICAL_PRESENTATION_DISABLED);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    Layout l = layout();
    const SDL_Color white{255, 255, 255, 255}, grey{170, 174, 182, 255}, accent{255, 205, 64, 255};

    if (s.show_fps) {
        char f[32];
        snprintf(f, sizeof f, "%.0f fps", fps);
        const Text &t = text(f, 13 * l.k, white, true);
        fill(8 * l.k, 8 * l.k, t.w + 12 * l.k, t.h + 6 * l.k, {0, 0, 0, 150});
        put(t, 14 * l.k, 11 * l.k);
    }
    if (open_) {
        int ow, oh;
        SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
        fill(0, 0, ow, oh, {10, 12, 18, 228});
        const Page p = stack.back().page;
        auto r = rows(p);
        put(text(title(p), 28 * l.k, white, true), l.x, l.top);
        if (stack.size() > 1) {
            const Text &t = text(g_game.name, 14 * l.k, grey);
            put(t, l.x + l.w - t.w, l.top + 12 * l.k);
        }
        row_y.assign(r.size(), 0);
        float y = l.y;
        const float max_y = oh - 70 * l.k;
        // keep the selection visible on long pages (the backup list)
        float sel_y = l.y;
        for (int i = 0; i < stack.back().sel && i < (int)r.size(); i++) sel_y += r[i].note.empty() ? l.row_h : l.row_h * 1.45f;
        if (sel_y + l.row_h * 2 > max_y) y -= sel_y + l.row_h * 2 - max_y;
        for (int i = 0; i < (int)r.size(); i++) {
            const Row &row = r[i];
            const float h = row.note.empty() ? l.row_h : l.row_h * 1.45f;
            row_y[i] = y;
            if (y + h > l.y - 4 && y < max_y) {
                const bool sel = i == stack.back().sel && selectable(row);
                if (sel) fill(l.x - 10 * l.k, y, l.w + 20 * l.k, h - 4 * l.k, {255, 255, 255, 34});
                if (sel) fill(l.x - 10 * l.k, y, 4 * l.k, h - 4 * l.k, accent);
                const Text &lt = text(row.label, (row.header ? 15 : 19) * l.k, row.header ? grey : white, row.header);
                put(lt, l.x, y + (l.row_h - lt.h) / 2 - 2 * l.k);
                if (!row.value.empty()) {
                    std::string v = row.change && sel ? "‹  " + row.value + "  ›" : row.value;
                    const Text &vt = text(v, 17 * l.k, row.change ? accent : grey, false);
                    put(vt, l.x + l.w - vt.w, y + (l.row_h - vt.h) / 2 - 2 * l.k);
                }
                if (!row.note.empty()) put(text(row.note, 13 * l.k, grey), l.x, y + l.row_h - 10 * l.k);
            }
            y += h;
        }
        const char *hint = "↑↓ choose   ←→ change   Return select   Esc back";
        const Text &ht = text(hint, 13 * l.k, grey);
        put(ht, (ow - ht.w) / 2, oh - 34 * l.k);
        if (SDL_GetTicks() < toast_until) {
            const Text &tt = text(toast, 16 * l.k, white, true);
            fill((ow - tt.w) / 2 - 14 * l.k, oh - 76 * l.k, tt.w + 28 * l.k, tt.h + 12 * l.k, {40, 120, 70, 230});
            put(tt, (ow - tt.w) / 2, oh - 70 * l.k);
        }
    }
    SDL_SetRenderLogicalPresentation(ren, lw, lh, mode);
}

}  // namespace menu
