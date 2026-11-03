// The in-game menu. Pages are lists of rows rebuilt every frame from the current state (an immediate-mode
// layout), drawn with SDL's renderer in window pixels; text is rasterised with Core Text (the system font)
// into cached textures. Keyboard, gamepad, mouse and fingers drive it: a finger drags the page to scroll it and
// a tap acts when the finger lifts.
#include "menu.h"

#include <TargetConditionals.h>
#if TARGET_OS_IOS
#import <UIKit/UIKit.h>
#else
#import <AppKit/AppKit.h>
#endif
#import <CoreText/CoreText.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "audio_iface.h"
#include "cloud.h"
#include "gpu_iface.h"
#include "loader.h"
#include "profile.h"
#include "settings.h"
#include "swkbd.h"
#include "text.h"
#include "touch.h"

namespace kernel {
std::string save_root();  // (kernel.h clashes with MacTypes.h)
}

namespace menu {
namespace {

SDL_Window *win;
SDL_Renderer *ren;
bool open_ = false;
constexpr const char *kDevice = TARGET_OS_IOS ? "device" : "Mac";  // whose clock the 3DS clock follows
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
    float scroll = 0;    // window pixels scrolled down the page
    bool follow = true;  // scroll to keep the selection in view (it moved by keys); a page the mouse or a finger
                         // points at stays put (the rows must not slide under it)
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
std::string when(time_t t) {
    char b[64];
    strftime(b, sizeof b, "%a %d %b, %H:%M", localtime(&t));
    return b;
}
bool cloud_restart;  // a save picked from the cloud: restored (and the game restarted) once it's downloaded

// Cloud saves (runtime/cloud.mm): the bucket, each device's save of this game
void cloud_rows(std::vector<Row> &r) {
    CloudView v = cloud_view();
    if (cloud_restart && !v.fetched.empty()) {  // (restored as any backup is, then the game restarts)
        cloud_restart = false;
        SaveBackup b{v.fetched.substr(v.fetched.rfind('/') + 1), v.fetched, 0};
        if (save_restore(b)) frontend_restart();
        say("Could not use the downloaded save");
    }
    if (cloud_restart && !v.busy && !v.error.empty()) cloud_restart = false;
    r.push_back({"Cloud saves", "", "", nullptr, nullptr, true});
    if (!v.configured) {
        r.push_back({"Connect…", "", "Copy your bucket's line first - s3://<key ID>:<application key>@<endpoint>/<bucket> - "
                                    "then select this. Every game on this " + std::string(kDevice) + " uses it.",
                     nullptr, [] {
#if TARGET_OS_IOS
                         NSString *str = UIPasteboard.generalPasteboard.string;
#else
                         NSString *str = [NSPasteboard.generalPasteboard stringForType:NSPasteboardTypeString];
#endif
                         std::string why;
                         if (cloud_connect(str ? str.UTF8String : "", &why)) say("Connected");
                         else say("Can't connect: " + why);
                     }});
        return;
    }
    static Uint64 asked;  // (listed again when the page is shown, at most every 2 minutes)
    if (!v.busy && (!asked || SDL_GetTicks() - asked > 120000)) {
        asked = SDL_GetTicks();
        if (!v.listed || time(nullptr) - v.listed > 60) cloud_refresh();
    }
    std::string line = v.error.empty() ? v.status : v.error;
    if (v.progress >= 0) line += " " + std::to_string((int)(v.progress * 100)) + "%";
    r.push_back({"Sync this device", onoff(v.sync), v.where, [v](int) { cloud_set_sync(!v.sync); }});
    r.push_back({"Status", line.empty() ? (v.busy ? "Working…" : "") : line, "", nullptr, nullptr, true});
    r.push_back({"Upload now", "", "This device is \"" + v.device + "\" (cloud.ini in the saves folder).", nullptr, [] {
                     cloud_upload_now();
                 }});
    r.push_back({"Check again", "", "", nullptr, [] { cloud_refresh(); }});
    for (auto &s : v.slots) {
        std::string note = s.current ? "The save here now." : !s.same_game ? "Made with other game files (another version): not used."
                                                                           : "Select to use this save instead.";
        std::function<void()> use;
        if (!s.current && s.same_game)
            use = [s] {
                confirm("Use the save from " + s.device + " (" + when(s.saved) + ")? It replaces the one here, which is "
                        "backed up first, and the game restarts.",
                        [s] {
                            cloud_fetch(s.key);
                            cloud_restart = true;
                        });
            };
        r.push_back({s.device + (s.mine ? " (this device)" : ""), when(s.saved), note, nullptr, use});
    }
    if (v.slots.empty() && v.listed) r.push_back({"No device has uploaded a save of this game yet", "", "", nullptr, nullptr, true});
    r.push_back({"Disconnect…", "", "", nullptr, [] {
                     confirm("Stop syncing on this device and forget the key? The saves in the bucket stay.", [] { cloud_disconnect(); });
                 }});
}
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
        if (!profile::stacks().kinds.empty())
            r.push_back({"Stack items", onoff(s.stack_items),
                         "Several of the same item share one pocket, as in New Horizons. Off: stacks go back to "
                         "free pockets as room allows.",
                         [&s](int) { s.stack_items = !s.stack_items; }});
        if (restart_pending()) r.push_back({"Restart to apply the graphics changes", "", "", nullptr, [] { frontend_restart(); }});
        r.push_back({"Restart game", "", "", nullptr, [] {
                         confirm("Restart the game? Progress since its last save is lost.", [] { frontend_restart(); });
                     }});
#ifdef RECOMP_MODULES
        r.push_back({"Choose another game", "", "", nullptr, [] {
                         confirm("Leave this game? Progress since its last save is lost.", [] { frontend_choose_game(); });
                     }});
#endif
#if !TARGET_OS_IOS  // (iOS apps are left from the app switcher)
        r.push_back({"Quit", "", "", nullptr, [] {
                         confirm("Quit? Progress since the game's last save is lost.", [] { frontend_quit(); });
                     }});
#endif
        break;
    case SAVES: {
        r.push_back({"Current save", save_summary(), "The game saves by itself; this is the data it keeps.", nullptr, nullptr, true});
        r.push_back({"Back up now", "", "", nullptr, [] { say(save_backup_now() ? "Backed up" : "Nothing to back up"); }});
        r.push_back({"Back up at each launch", onoff(s.auto_backup), "Keeps the last 5 automatic copies.",
                     [&s](int) { s.auto_backup = !s.auto_backup; }});
        if (!profile::unsaved_marks().empty())
            r.push_back({"Forgive unsaved quits", onoff(s.forgive_unsaved),
                         "After a crash or closing without saving, the game goes on from what it last wrote without "
                         "scolding you (Resetti). Applies at the next launch.",
                         [&s](int) { s.forgive_unsaved = !s.forgive_unsaved; }});
#if !TARGET_OS_IOS
        r.push_back({"Show in Finder", "", "", nullptr, [] {
                         NSString *p = [NSString stringWithUTF8String:kernel::save_root().c_str()];
                         [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:p]];
                     }});
#endif
        cloud_rows(r);
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
                     o == 0 ? std::string("Follows the ") + kDevice + "'s clock."
                            : std::string("Set apart from the ") + kDevice + "'s clock; it keeps running from there.",
                     nullptr, nullptr, true});
        r.push_back({"Year", clock_text("%Y"), "", [](int d) { shift_clock(&tm::tm_year, d); }});
        r.push_back({"Month", clock_text("%B"), "", [](int d) { shift_clock(&tm::tm_mon, d); }});
        r.push_back({"Day", clock_text("%d"), "", [](int d) { shift_clock(&tm::tm_mday, d); }});
        r.push_back({"Hour", clock_text("%H"), "", [](int d) { shift_clock(&tm::tm_hour, d); }});
        r.push_back({"Minute", clock_text("%M"), "", [](int d) { shift_clock(&tm::tm_min, d); }});
        r.push_back({std::string("Use the ") + kDevice + "'s clock", "", "Games read the clock when they need it; some notice when it moves.",
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
        if (gpu::texture_filter_count() > 1)
            r.push_back({"Upscale textures", gpu::texture_filter_name(s.tex_filter),
                         "Sharpens the game's small textures: Sharp for crisp cartoon edges, Smooth for softer art.",
                         [&s](int d) {
                             const int n = gpu::texture_filter_count();
                             s.tex_filter = (s.tex_filter + d + n) % n;
                             gpu::set_texture_filter(s.tex_filter);
                         }});
#if !TARGET_OS_IOS
        r.push_back({"Separate windows", s.split ? "One per screen" : "Off",
                     "Put each screen on its own display; F11 makes the window you are in fullscreen.", [&s](int) {
                         s.split = !s.split;
                         frontend_set_split(s.split);
                     }});
#endif
        r.push_back({"Screen layout", frontend_layout_name(s.layout),
                     s.split            ? "Used when the screens share one window."
                     : touch::enabled() ? "Touch controls go in the room the screens leave."
                                        : "F2 cycles the layouts while playing.",
                     [&s](int d) {
                         const int n = frontend_layout_count();
                         s.layout = (s.layout + d + n) % n;
                         frontend_set_layout(s.layout);
                     }});
        {
            static const std::vector<std::string> pip_names = {"Off", "When it changes", "Always"},
                                                   corners = {"Top right", "Top left", "Bottom right", "Bottom left"},
                                                   hud_names = {"Off", "When the bottom screen is hidden", "Always"};
            r.push_back({"Other screen in a corner", pip_names[s.pip],
                         touch::enabled() ? "In the single-screen layouts. The swap button swaps the screens; tap the corner one to touch it."
                                          : "In the single-screen layouts. Tab swaps the screens; click the corner to touch it.",
                         [&s](int d) { cycle(s.pip, {0, 1, 2}, d); }});
            if (s.pip) {
                r.push_back({"Position", s.pip_x >= 0 ? "Where you dragged it" : corners[s.pip_corner],
                             touch::enabled() ? "Drag the tab next to it to move it; drag its handle or pinch it to resize it."
                                              : "Drag the tab next to it to move it; drag its handle or scroll over it to resize it.",
                             [&s](int d) {
                                 if (s.pip_x >= 0) s.pip_x = s.pip_y = -1;  // back to a corner
                                 else cycle(s.pip_corner, {0, 1, 2, 3}, d);
                             }});
                r.push_back({"Size", std::to_string(s.pip_size) + "%", "Of the window's height.",
                             [&s](int d) { s.pip_size = std::clamp(s.pip_size + d * 5, 15, 80); }});
            }
            r.push_back({"Automatic screen", profile::has_focus_rules() ? onoff(s.auto_focus) : "Not for this game",
                         profile::has_focus_rules() ? "In the single-screen layouts, shows the screen the game needs "
                                                      "(such as the bottom one on menus that use it)."
                                                    : "",
                         profile::has_focus_rules() ? std::function<void(int)>([&s](int) { s.auto_focus = !s.auto_focus; })
                                                    : nullptr});
            if (profile::has_hud())
                r.push_back({"Extra HUD", hud_names[s.hud],
                             "What the bottom screen shows during play, over the top screen. E (or a left stick click) "
                             "uses the stored item.",
                             [&s](int d) { cycle(s.hud, {0, 1, 2}, d); }});
        }
#if !TARGET_OS_IOS
        r.push_back({"Fullscreen", onoff(s.fullscreen), "", [&s](int) {
                         s.fullscreen = !s.fullscreen;
                         frontend_set_fullscreen(s.fullscreen);
                     }});
#endif
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

// ---- layout (window pixels), shared by draw() and hit-testing; inside the safe area (clear of a phone's
// camera cutout and home indicator)
struct Layout {
    float k, x, y, w, row_h, top, bottom;  // the rows go from y to bottom
};
Layout layout() {
    int ow = 0, oh = 0;
    SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
    const float d = SDL_GetWindowPixelDensity(win);
    SDL_Rect sa;
    if (!SDL_GetWindowSafeArea(win, &sa) || sa.w <= 0) sa = {0, 0, (int)(ow / d), (int)(oh / d)};
    const float sx = sa.x * d, sy = sa.y * d, sw = sa.w * d, sh = sa.h * d;
    Layout l;
    l.k = std::clamp(std::min(sw / 640.0f, sh / 560.0f), 0.5f, 4.0f);
    if (touch::enabled()) l.k = std::max(l.k * 1.2f, d);  // (rows at least 44 points high: a fingertip)
    l.w = std::min(sw - 24 * l.k, 560 * l.k);
    l.x = sx + (sw - l.w) / 2;
    l.row_h = (touch::enabled() ? 44 : 40) * l.k;
    if (touch::enabled()) {  // (a phone held sideways is short: little room around the rows)
        l.top = sy + 12 * l.k;
        l.y = l.top + 52 * l.k;
        l.bottom = sy + sh - 30 * l.k;
    } else {
        l.top = sy + std::max(16 * l.k, sh * 0.08f);
        l.y = l.top + 64 * l.k;
        l.bottom = sy + sh - 70 * l.k;
    }
    return l;
}

// a text in lines no wider than w, broken between words
const std::vector<std::string> &wrapped(const std::string &s, float px, bool bold, float w) {
    static std::map<std::string, std::vector<std::string>> cache;
    char key[48];
    snprintf(key, sizeof key, "|%.1f|%d|%.0f", px, bold, w);
    auto it = cache.find(s + key);
    if (it != cache.end()) return it->second;
    if (cache.size() > 400) cache.clear();
    std::vector<std::string> lines;
    std::string line;
    for (size_t i = 0; i <= s.size();) {
        size_t j = s.find(' ', i);
        if (j == std::string::npos) j = s.size();
        const std::string word = s.substr(i, j - i), next = line.empty() ? word : line + " " + word;
        if (!line.empty() && text::width(next, px, bold) > w) {
            lines.push_back(line);
            line = word;
        } else {
            line = next;
        }
        i = j + 1;
    }
    lines.push_back(line);
    return cache[s + key] = lines;
}

// how a row is laid out: its label, value and note, the value beside the label or under it when there is no
// room, the note in as many lines as it takes
struct RowBox {
    std::vector<std::string> label, note;
    std::string value;
    float label_px, h;
    bool value_below = false;
};
RowBox measure(const Row &row, const Layout &l, bool sel) {
    RowBox b;
    b.label_px = (row.header ? 15 : 19) * l.k;
    if (!row.value.empty()) b.value = row.change && (sel || touch::enabled()) ? "‹  " + row.value + "  ›" : row.value;
    // (the room is kept for the arrows the selection shows, so choosing a row moves none of them)
    const float vw = row.value.empty() ? 0 : text::width(row.change ? "‹  " + row.value + "  ›" : row.value, 17 * l.k, false) + 16 * l.k;
    if (vw && text::width(row.label, b.label_px, row.header) > l.w - vw) b.value_below = true;
    b.label = wrapped(row.label, b.label_px, row.header, b.value_below || !vw ? l.w : l.w - vw);
    if (!row.note.empty()) b.note = wrapped(row.note, 13 * l.k, false, l.w);
    const float content = b.label.size() * b.label_px * 1.3f + (b.value_below ? 17 * l.k * 1.35f : 0) +
                          b.note.size() * 13 * l.k * 1.3f;
    b.h = std::max(l.row_h, content + 14 * l.k);
    return b;
}

// each row's place from the last draw (for the mouse and fingers): its top and height, and where its value is
struct RowHit {
    float y, h, vx0, vx1;
};
std::vector<RowHit> row_hits;
float max_scroll = 0;

bool selectable(const Row &r) { return !r.header && (r.change || r.activate); }

void move(int dir) {
    auto r = rows(stack.back().page);
    stack.back().follow = true;
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


// ---- software keyboard (swkbd.h): a text field over the game while it waits for typed text
swkbd::Request kb;
bool kb_open = false;
std::string kb_text, kb_edit;  // the text, and the input method's composition (not typed yet)
SDL_FRect kb_ok_rc, kb_cancel_rc;

std::vector<uint32_t> code_points(const std::string &s) {
    std::vector<uint32_t> o;
    for (size_t i = 0; i < s.size();) {
        const unsigned char b = s[i];
        const int n = b < 0x80 ? 0 : b >= 0xF0 ? 3 : b >= 0xE0 ? 2 : b >= 0xC0 ? 1 : -1;
        if (n < 0 || i + n >= s.size() + (n ? 0 : 1)) {
            i++;
            continue;
        }
        uint32_t c = n ? b & (0x3F >> n) : b;
        for (int k = 1; k <= n; k++) c = c << 6 | (s[i + k] & 0x3F);
        o.push_back(c);
        i += n + 1;
    }
    return o;
}
size_t utf16_len(const std::string &s) {
    size_t n = 0;
    for (uint32_t c : code_points(s)) n += c >= 0x10000 ? 2 : 1;
    return n;
}

void kb_insert(const std::string &in) {
    for (uint32_t c : code_points(in)) {
        if (c == '\r') c = '\n';
        if (c == '\n' && !kb.multiline) c = ' ';
        if (c < 0x20 && c != '\n') continue;
        if (kb.numpad && (c < '0' || c > '9')) continue;
        if ((kb.no_at && c == '@') || (kb.no_percent && c == '%') || (kb.no_backslash && c == '\\')) continue;
        if (kb.digits_limited && c >= '0' && c <= '9') {
            int digits = 0;
            for (char d : kb_text) digits += d >= '0' && d <= '9';
            if (digits >= kb.max_digits) continue;
        }
        const size_t units = c >= 0x10000 ? 2 : 1;
        if (kb.max_len && utf16_len(kb_text) + units > (size_t)kb.max_len) break;
        char u[5] = {};
        if (c < 0x80) u[0] = (char)c;
        else if (c < 0x800) u[0] = (char)(0xC0 | c >> 6), u[1] = (char)(0x80 | (c & 0x3F));
        else if (c < 0x10000) u[0] = (char)(0xE0 | c >> 12), u[1] = (char)(0x80 | (c >> 6 & 0x3F)), u[2] = (char)(0x80 | (c & 0x3F));
        else u[0] = (char)(0xF0 | c >> 18), u[1] = (char)(0x80 | (c >> 12 & 0x3F)), u[2] = (char)(0x80 | (c >> 6 & 0x3F)), u[3] = (char)(0x80 | (c & 0x3F));
        kb_text += u;
    }
}
void kb_backspace() {
    while (!kb_text.empty() && (kb_text.back() & 0xC0) == 0x80) kb_text.pop_back();
    if (!kb_text.empty()) kb_text.pop_back();
}
bool kb_valid() {
    const bool blank = kb_text.find_first_not_of(" \n") == std::string::npos;
    switch (kb.valid) {
    case 1: return !kb_text.empty();
    case 2: return !blank;
    case 3: return kb_text.empty() || !blank;
    case 4: return !kb.max_len || utf16_len(kb_text) == (size_t)kb.max_len;
    default: return true;
    }
}
void kb_close(bool ok) {
    if (ok && !kb_valid()) return;
    if (!ok && !kb.can_cancel) return;
    swkbd::finish(kb_text, ok);
    kb_open = false;
    kb_edit.clear();
    SDL_StopTextInput(win);
}

bool kb_event(const SDL_Event &ev) {
    switch (ev.type) {
    case SDL_EVENT_TEXT_INPUT:
        kb_insert(ev.text.text);
        kb_edit.clear();
        return true;
    case SDL_EVENT_TEXT_EDITING:
        kb_edit = ev.edit.text ? ev.edit.text : "";
        return true;
    case SDL_EVENT_KEY_DOWN: {
        const SDL_Scancode k = ev.key.scancode;
        const bool cmd = ev.key.mod & SDL_KMOD_GUI;
        if (!kb_edit.empty()) return true;  // the input method has the keys
        if (k == SDL_SCANCODE_RETURN || k == SDL_SCANCODE_KP_ENTER) {
            if (kb.multiline && (ev.key.mod & SDL_KMOD_SHIFT)) kb_insert("\n");
            else kb_close(true);
        } else if (k == SDL_SCANCODE_ESCAPE) {
            kb_close(false);
        } else if (k == SDL_SCANCODE_BACKSPACE) {
            if (cmd) kb_text.clear();
            else kb_backspace();
        } else if (cmd && k == SDL_SCANCODE_V) {
            if (char *c = SDL_GetClipboardText()) {
                kb_insert(c);
                SDL_free(c);
            }
        } else if (k >= SDL_SCANCODE_F1 && k <= SDL_SCANCODE_F12) {
            return false;  // diagnostics keep working
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
        if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_EAST) kb_close(true);  // the 3DS's A position
        if (ev.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) kb_close(false);
        return true;
    case SDL_EVENT_MOUSE_BUTTON_DOWN: {
        const float d = SDL_GetWindowPixelDensity(win);
        const SDL_FPoint p{ev.button.x * d, ev.button.y * d};
        if (SDL_PointInRectFloat(&p, &kb_ok_rc)) kb_close(true);
        else if (kb.can_cancel && SDL_PointInRectFloat(&p, &kb_cancel_rc)) kb_close(false);
        return true;
    }
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_GAMEPAD_BUTTON_UP:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_MOTION:
        return true;
    default:
        return false;
    }
}

void kb_draw() {
    int ow, oh;
    SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
    const float k = std::clamp(std::min(ow / 640.0f, oh / 560.0f), 0.5f, 4.0f);
    const SDL_Color white{255, 255, 255, 255}, grey{170, 174, 182, 255}, accent{255, 205, 64, 255};
    fill(0, 0, ow, oh, {0, 0, 0, 110});
    const float w = std::min(ow - 24 * k, 520 * k), x = (ow - w) / 2, pad = 18 * k;
    const float field_h = 44 * k, h = pad * 2 + 24 * k + field_h + 14 * k + 34 * k;
    const float y = std::max(8 * k, (oh - h) * 0.4f);
    fill(x, y, w, h, {22, 25, 34, 240});
    fill(x, y, w, 3 * k, accent);
    float cy = y + pad;
    const std::string hint = !kb.hint.empty() ? kb.hint : kb.numpad ? "Enter a number" : "Enter text";
    put(text(hint, 15 * k, grey), x + pad, cy);
    if (kb.max_len) {
        const Text &c = text(std::to_string(utf16_len(kb_text)) + " / " + std::to_string(kb.max_len), 13 * k, grey);
        put(c, x + w - pad - c.w, cy + 2 * k);
    }
    cy += 24 * k;
    // the field, scrolled to keep the end of the text (where the caret is) in view
    const SDL_FRect field{x + pad, cy, w - 2 * pad, field_h};
    fill(field.x, field.y, field.w, field.h, {255, 255, 255, 22});
    fill(field.x, field.y + field.h - 2 * k, field.w, 2 * k, accent);
    std::string shown;
    if (kb.password) {
        for (size_t i = 0, n = code_points(kb_text).size(); i < n; i++) shown += "\u2022";
    } else {
        for (char c : kb_text) shown += c == '\n' ? std::string(" \u21b5 ") : std::string(1, c);
    }
    const float tx0 = field.x + 10 * k, avail = field.w - 20 * k;
    const Text *t = shown.empty() ? nullptr : &text(shown, 22 * k, white);
    const Text *e = kb_edit.empty() ? nullptr : &text(kb_edit, 22 * k, accent);
    const float tw = (t ? t->w : 0) + (e ? e->w : 0);
    const float tx = tx0 - std::max(0.0f, tw + 4 * k - avail);
    const SDL_Rect clip{(int)field.x, (int)field.y, (int)field.w, (int)field.h};
    SDL_SetRenderClipRect(ren, &clip);
    const float th = 22 * k * 1.3f, ty = field.y + (field.h - th) / 2;
    if (t) put(*t, tx, ty);
    if (e) {
        put(*e, tx + (t ? t->w : 0), ty);
        fill(tx + (t ? t->w : 0), ty + e->h - 2 * k, e->w, 1 * k, accent);
    }
    if (SDL_GetTicks() / 530 % 2 == 0 || !kb_edit.empty()) fill(tx + tw + 1 * k, ty + 3 * k, 2 * k, th - 6 * k, white);
    SDL_SetRenderClipRect(ren, nullptr);
    // the input method's candidate window goes under the field (window coordinates)
    const float d = SDL_GetWindowPixelDensity(win);
    const SDL_Rect area{(int)(field.x / d), (int)(field.y / d), (int)(field.w / d), (int)(field.h / d)};
    SDL_SetTextInputArea(win, &area, (int)((std::min(tx + tw, field.x + field.w) - field.x) / d));
    cy += field_h + 14 * k;
    // buttons: cancel on the left, confirm on the right, as on the 3DS
    auto button = [&](const std::string &key, const std::string &label, float bx, bool right, bool enabled, SDL_FRect &rc) {
        const Text &kt = text(key, 13 * k, grey), &lt = text(label, 17 * k, enabled ? (right ? accent : white) : grey, right);
        const float bw = kt.w + lt.w + 30 * k, bh = 34 * k;
        rc = {right ? bx - bw : bx, cy, bw, bh};
        fill(rc.x, rc.y, rc.w, rc.h, {255, 255, 255, (Uint8)(enabled ? 26 : 10)});
        put(kt, rc.x + 10 * k, rc.y + (bh - kt.h) / 2);
        put(lt, rc.x + 20 * k + kt.w, rc.y + (bh - lt.h) / 2);
    };
    if (kb.can_cancel) button("Esc", kb.cancel.empty() ? "Cancel" : kb.cancel, x + pad, false, true, kb_cancel_rc);
    else kb_cancel_rc = {};
    button("Return", kb.ok.empty() ? "OK" : kb.ok, x + w - pad, true, kb_valid(), kb_ok_rc);
}

}  // namespace

void init(SDL_Window *w, SDL_Renderer *r) {
    win = w;
    ren = r;
    msaa_at_launch = g_settings.msaa;
    aniso_at_launch = g_settings.aniso;
}

bool is_open() { return open_; }

namespace {

// the row at (x, y), window pixels (-1: none)
int row_at(float x, float y) {
    const Layout l = layout();
    if (y < l.y || y >= l.bottom || x < l.x - 10 * l.k || x >= l.x + l.w + 10 * l.k) return -1;
    auto r = rows(stack.back().page);
    for (int i = 0; i < (int)row_hits.size() && i < (int)r.size(); i++)
        if (selectable(r[i]) && y >= row_hits[i].y && y < row_hits[i].y + row_hits[i].h) return i;
    return -1;
}

// a click or tap: the title line goes back (on touch screens, which have no Esc), a row is chosen or its value
// changed; the left half of a value ("‹") steps it back, as the right mouse button does
void click(float x, float y, bool right_button) {
    const Layout l = layout();
    if (y < l.y - 8 * l.k) {
        if (touch::enabled()) pop();
        return;
    }
    const int i = row_at(x, y);
    if (i < 0) return;
    stack.back().sel = i;
    stack.back().follow = false;
    auto r = rows(stack.back().page);
    const RowHit &h = row_hits[i];
    act(!r[i].change ? 0 : right_button || (x >= h.vx0 - 24 * l.k && x < (h.vx0 + h.vx1) / 2) ? -1 : 1);
}

// fingers on the menu: each one that moves a few points drags the page, and one that lifts without having moved
// taps where it went down, on the row it went down on. Every finger counts on its own, so a thumb resting on the
// screen while the phone is held blocks nothing.
struct MenuFinger {
    float x, y, last_y;
    int row;       // the row it went down on (-1: none)
    int sel_before;  // the selection before it lit that row (back when the finger turns out to drag)
    bool dragged = false;
};
std::map<SDL_FingerID, MenuFinger> fingers;

void finger_event(const SDL_Event &ev) {
    int ow = 0, oh = 0;
    SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
    const float x = ev.tfinger.x * ow, y = ev.tfinger.y * oh;
    const SDL_FingerID id = ev.tfinger.fingerID;
    if (ev.type == SDL_EVENT_FINGER_DOWN) {
        const int i = row_at(x, y);
        fingers[id] = {x, y, y, i, stack.back().sel};
        if (i >= 0) {  // (lit while the finger is on it)
            stack.back().sel = i;
            stack.back().follow = false;
        }
        return;
    }
    auto it = fingers.find(id);
    if (it == fingers.end()) return;
    MenuFinger &f = it->second;
    if (ev.type == SDL_EVENT_FINGER_MOTION) {
        if (!f.dragged && std::hypot(x - f.x, y - f.y) > 10 * layout().k) {
            f.dragged = true;
            if (f.row >= 0 && stack.back().sel == f.row) stack.back().sel = f.sel_before;
        }
        if (f.dragged) {
            stack.back().follow = false;
            stack.back().scroll = std::clamp(stack.back().scroll - (y - f.last_y), 0.0f, max_scroll);
        }
        f.last_y = y;
        return;
    }
    const MenuFinger up = f;
    fingers.erase(it);
    // (another finger may have scrolled the page meanwhile: a row that is no longer there is not tapped)
    if (ev.type == SDL_EVENT_FINGER_UP && !up.dragged && (up.row < 0 || row_at(up.x, up.y) == up.row))
        click(up.x, up.y, false);
}

}  // namespace

void toggle() {
    open_ = !open_;
    stack.clear();
    fingers.clear();
    if (open_) {
        push(MAIN);
        fix_sel();
        cloud_flush();  // (a pause: the save uploaded now if it changed)
    }
    settings_save();
}

void set_fps(float f) { fps = f; }

bool keyboard_open() { return kb_open; }

bool handle_event(const SDL_Event &ev) {
    if (kb_open) return kb_event(ev);
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
    if (ev.type == SDL_EVENT_FINGER_DOWN || ev.type == SDL_EVENT_FINGER_MOTION || ev.type == SDL_EVENT_FINGER_UP ||
        ev.type == SDL_EVENT_FINGER_CANCELED) {
        if (ev.tfinger.windowID == SDL_GetWindowID(win)) finger_event(ev);
        return true;
    }
    // the mouse events iOS makes from fingers are left out (the finger events above are used), and on a Mac
    // trying the touch controls, where the mouse makes the finger events, the mouse's own
    const bool mouse_finger = !TARGET_OS_IOS && touch::enabled();
    if ((ev.type == SDL_EVENT_MOUSE_MOTION && (ev.motion.which == SDL_TOUCH_MOUSEID || mouse_finger)) ||
        ((ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN || ev.type == SDL_EVENT_MOUSE_BUTTON_UP) &&
         (ev.button.which == SDL_TOUCH_MOUSEID || mouse_finger)))
        return true;
    if ((ev.type == SDL_EVENT_MOUSE_MOTION && ev.motion.windowID != SDL_GetWindowID(win)) ||
        (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.windowID != SDL_GetWindowID(win)))
        return true;  // the bottom screen's window, when the screens have separate windows
    if (ev.type == SDL_EVENT_MOUSE_WHEEL) {
        stack.back().follow = false;
        stack.back().scroll = std::clamp(stack.back().scroll - ev.wheel.y * layout().row_h, 0.0f, max_scroll);
        return true;
    }
    if (ev.type == SDL_EVENT_MOUSE_MOTION) {
        const float d = SDL_GetWindowPixelDensity(win);
        if (const int i = row_at(ev.motion.x * d, ev.motion.y * d); i >= 0 && i != stack.back().sel) {
            stack.back().sel = i;
            stack.back().follow = false;
        }
        return true;
    }
    if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
        const float d = SDL_GetWindowPixelDensity(win);
        click(ev.button.x * d, ev.button.y * d, ev.button.button == SDL_BUTTON_RIGHT);
        return true;
    }
    return ev.type == SDL_EVENT_KEY_UP || ev.type == SDL_EVENT_GAMEPAD_BUTTON_UP || ev.type == SDL_EVENT_MOUSE_BUTTON_UP;
}

void update() {
    swkbd::Request r;
    if (swkbd::pending(&r) && (!kb_open || r.serial != kb.serial)) {
        kb = r;
        kb_open = true;
        kb_text.clear();
        kb_edit.clear();
        kb_insert(r.initial);
        if (open_) toggle();  // the menu would pause the game
        SDL_StartTextInput(win);
    }
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
    if (kb_open) kb_draw();
    // a reminder when the game's own keyboard comes up (profile "keyboard"): the Mac's keyboard types into it
    static Uint64 typing_since;
    if (!frontend_typing()) typing_since = 0;
    else if (!typing_since) typing_since = SDL_GetTicks();
    if (typing_since && SDL_GetTicks() - typing_since < 5000 && !kb_open && !open_) {
        const Text &t = text("Type with your keyboard   ·   Return confirms", 14 * l.k, white, true);
        int ow, oh;
        SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
        fill((ow - t.w) / 2 - 14 * l.k, 12 * l.k, t.w + 28 * l.k, t.h + 12 * l.k, {20, 22, 30, 215});
        put(t, (ow - t.w) / 2, 18 * l.k);
    }
    if (open_) {
        int ow, oh;
        SDL_GetCurrentRenderOutputSize(ren, &ow, &oh);
        fill(0, 0, ow, oh, {10, 12, 18, 228});
        const Page p = stack.back().page;
        auto r = rows(p);
        const Text &tt = text(title(p), 28 * l.k, white, true);
        put(tt, l.x, l.top);
        if (touch::enabled()) {  // (the whole title line is the button)
            const Text &t = text(stack.size() > 1 ? "‹ Back" : "Close", 19 * l.k, accent, true);
            put(t, l.x + l.w - t.w, l.top + (tt.h - t.h) / 2);
        } else if (stack.size() > 1) {
            const Text &t = text(g_game.name, 14 * l.k, grey);
            put(t, l.x + l.w - t.w, l.top + 12 * l.k);
        }
        // the rows, scrolled when the page is longer than the room it has
        Level &lv = stack.back();
        std::vector<RowBox> boxes;
        std::vector<float> tops;
        float total = 0;
        for (int i = 0; i < (int)r.size(); i++) {
            boxes.push_back(measure(r[i], l, i == lv.sel && selectable(r[i])));
            tops.push_back(total);
            total += boxes.back().h;
        }
        const float view = l.bottom - l.y;
        max_scroll = std::max(0.0f, total - view);
        if (lv.follow && lv.sel < (int)r.size()) {  // the selection in view, with a little of the rows around it
            const float a = tops[lv.sel] - l.row_h / 2, b = tops[lv.sel] + boxes[lv.sel].h + l.row_h / 2;
            if (a < lv.scroll) lv.scroll = a;
            if (b > lv.scroll + view) lv.scroll = b - view;
        }
        lv.scroll = std::clamp(lv.scroll, 0.0f, max_scroll);
        row_hits.assign(r.size(), {});
        const SDL_Rect clip{0, (int)(l.y - 4 * l.k), ow, (int)(view + 4 * l.k)};
        SDL_SetRenderClipRect(ren, &clip);
        for (int i = 0; i < (int)r.size(); i++) {
            const Row &row = r[i];
            const RowBox &b = boxes[i];
            const float y = l.y + tops[i] - lv.scroll;
            row_hits[i] = {y, b.h, l.x + l.w, l.x + l.w};
            if (y + b.h < l.y - 4 * l.k || y > l.bottom) continue;
            const bool sel = i == lv.sel && selectable(row);
            if (sel) fill(l.x - 10 * l.k, y, l.w + 20 * l.k, b.h - 4 * l.k, {255, 255, 255, 34});
            if (sel) fill(l.x - 10 * l.k, y, 4 * l.k, b.h - 4 * l.k, accent);
            const float lh = b.label_px * 1.3f;
            const float content = b.label.size() * lh + (b.value_below ? 17 * l.k * 1.35f : 0) + b.note.size() * 13 * l.k * 1.3f;
            float cy = y + (b.h - 4 * l.k - content) / 2;
            for (const auto &line : b.label) {
                const Text &lt = text(line, b.label_px, row.header ? grey : white, row.header);
                put(lt, l.x, cy + (lh - lt.h) / 2);
                if (!b.value.empty() && !b.value_below && &line == &b.label.front()) {
                    const Text &vt = text(b.value, 17 * l.k, row.change ? accent : grey, false);
                    put(vt, l.x + l.w - vt.w, cy + (lh - vt.h) / 2);
                    row_hits[i].vx0 = l.x + l.w - vt.w;
                }
                cy += lh;
            }
            if (!b.value.empty() && b.value_below) {
                const Text &vt = text(b.value, 17 * l.k, row.change ? accent : grey, false);
                put(vt, l.x + l.w - vt.w, cy);
                row_hits[i].vx0 = l.x + l.w - vt.w;
                cy += 17 * l.k * 1.35f;
            }
            for (const auto &line : b.note) {
                put(text(line, 13 * l.k, grey), l.x, cy);
                cy += 13 * l.k * 1.3f;
            }
        }
        SDL_SetRenderClipRect(ren, nullptr);
        if (max_scroll > 0) {  // where the page is scrolled to, at the right edge
            const float bh = std::max(view * view / total, 24 * l.k), by = l.y + (view - bh) * lv.scroll / max_scroll;
            fill(std::min(l.x + l.w + 16 * l.k, ow - 6 * l.k), by, 3 * l.k, bh, {255, 255, 255, 70});
        }
        const char *hint = touch::enabled() ? "Drag to scroll   ·   tap ‹ or › to change a value"
                                            : "↑↓ choose   ←→ change   Return select   Esc back";
        const Text &ht = text(hint, 13 * l.k, grey);
        const float hint_y = touch::enabled() ? l.bottom + (30 * l.k - ht.h) / 2 : l.bottom + 70 * l.k - 34 * l.k;
        put(ht, (ow - ht.w) / 2, hint_y);
        if (SDL_GetTicks() < toast_until) {
            const Text &tt = text(toast, 16 * l.k, white, true);
            const float ty = l.bottom - tt.h - 24 * l.k;
            fill((ow - tt.w) / 2 - 14 * l.k, ty - 6 * l.k, tt.w + 28 * l.k, tt.h + 12 * l.k, {40, 120, 70, 230});
            put(tt, (ow - tt.w) / 2, ty);
        }
    }
    SDL_SetRenderLogicalPresentation(ren, lw, lh, mode);
}

}  // namespace menu
