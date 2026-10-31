// The host side: an SDL3 window showing both screens, keyboard/gamepad input, VBlank pacing, and the GPU
// (GX commands run on this thread). Audio frames are ticked from their own thread.
#include "frontend.h"

#include <SDL3/SDL.h>
#include <pthread/qos.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio_iface.h"
#include "gpu_iface.h"
#include "kernel.h"
#include "loader.h"
#include "hud.h"
#include "profile.h"
#include "menu.h"
#include "services/services.h"
#include "settings.h"

#include <mach-o/dyld.h>

using namespace std::chrono;

void watch_report();
static SDL_Window *window;
static SDL_Renderer *renderer;
static SDL_Texture *tex_top, *tex_bottom;
static std::mutex wake_m;
static std::condition_variable wake_cv;
static bool wake_flag;
static SDL_Gamepad *pad;
static int scale = 2;
static u32 shot_frame = 0, shot_from = 0;  // RECOMP_SHOT every n frames, from RECOMP_SHOT_FROM
static u64 frame_count = 0;
static u64 game_frames = 0;  // new top-screen images (the game's frame rate)
static std::atomic<bool> paused;  // the menu is open: no VBlank, no audio frames, no input
// Screen layouts (F2 or the menu): the logical canvas and where each screen goes on it (w 0: not shown)
struct ScreenLayout {
    const char *name;
    int w, h;
    SDL_FRect top, bottom;
};
static const ScreenLayout layouts[] = {
    {"Stacked", 400, 480, {0, 0, 400, 240}, {40, 240, 320, 240}},
    {"Top screen only", 400, 240, {0, 0, 400, 240}, {}},
    {"Side by side", 720, 240, {0, 0, 400, 240}, {400, 0, 320, 240}},
    {"Large top, small bottom", 560, 240, {0, 0, 400, 240}, {400, 120, 160, 120}},
    {"Bottom screen only", 320, 240, {}, {0, 0, 320, 240}},
    {"Stacked, bottom first", 400, 480, {0, 240, 400, 240}, {40, 0, 320, 240}},
};
constexpr int kLayouts = sizeof layouts / sizeof layouts[0];
static int layout;
// Tab / right-stick click: the same layout with the other screen (single-screen and stacked layouts)
static int swapped_layout(int l) {
    switch (l) {
    case 1: return 4;
    case 4: return 1;
    case 0: return 5;
    case 5: return 0;
    default: return l;
    }
}
// separate windows (the menu's Graphics page): the bottom screen in its own window, so each screen can go on
// its own display; the main window then shows the top screen only
static SDL_Window *window2;
static SDL_Renderer *renderer2;
static int vsync = 0;  // presents wait for the display's refresh (no tearing) at normal speed; off when sped up
static SDL_Texture *tex_bottom2, *hi_bottom2;
static int hb2w, hb2h;
static u64 shown_ver[2];  // display image versions already uploaded (per screen)
static int shown_layout() { return window2 ? 1 : layout; }
// Metal internal resolution: 0 auto (from the window size), 1-4 fixed; F3 cycles; -1 without Metal
static int scale_mode = -1;
constexpr int kMaxAutoScale = 4;

static void apply_layout() {
    const ScreenLayout &l = layouts[shown_layout()];
    SDL_SetRenderLogicalPresentation(renderer, l.w, l.h, SDL_LOGICAL_PRESENTATION_LETTERBOX);
}
// reshape a normal window from one layout to another, keeping the screens their size on the display
static void reshape(int from, int to) {
    if (from == to || (SDL_GetWindowFlags(window) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_MAXIMIZED))) return;
    int ww, wh;
    SDL_GetWindowSize(window, &ww, &wh);
    float k = std::min((float)ww / layouts[from].w, (float)wh / layouts[from].h);
    SDL_Rect usable;
    if (SDL_GetDisplayUsableBounds(SDL_GetDisplayForWindow(window), &usable))
        k = std::min({k, usable.w * 0.95f / layouts[to].w, usable.h * 0.95f / layouts[to].h});
    SDL_SetWindowSize(window, (int)(layouts[to].w * k), (int)(layouts[to].h * k));
}
// window positions are remembered (per game) so a two-display setup comes back as it was
static void restore_geometry(SDL_Window *w, const int *g) {
    if (g[2] <= 0 || g[3] <= 0) return;
    SDL_Point centre{g[0] + g[2] / 2, g[1] + g[3] / 2};
    if (!SDL_GetDisplayForPoint(&centre)) return;  // that display is gone
    SDL_SetWindowSize(w, g[2], g[3]);
    SDL_SetWindowPosition(w, g[0], g[1]);
}
static void store_geometry(SDL_Window *w) {
    if (!w || (SDL_GetWindowFlags(w) & SDL_WINDOW_FULLSCREEN)) return;
    int *g = w == window ? g_settings.geom : g_settings.geom2;
    SDL_GetWindowPosition(w, &g[0], &g[1]);
    SDL_GetWindowSize(w, &g[2], &g[3]);
}
int frontend_layout_count() { return kLayouts; }
const char *frontend_layout_name(int l) { return layouts[std::clamp(l, 0, kLayouts - 1)].name; }

static std::mutex gpu_m;
static std::condition_variable gpu_cv;
static bool gpu_wake;

namespace kernel {
void frontend_wake() {
    std::lock_guard<std::mutex> lk(gpu_m);
    gpu_wake = true;
    gpu_cv.notify_one();
}
}  // namespace kernel

void frontend_init() {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) FATAL("SDL_Init: %s\n", SDL_GetError());
    if (const char *s = getenv("RECOMP_SCALE")) scale = atoi(s);
    layout = std::clamp(g_settings.layout, 0, kLayouts - 1);
    window = SDL_CreateWindow(g_game.name.c_str(), layouts[layout].w * scale, layouts[layout].h * scale, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) FATAL("SDL_CreateWindow: %s\n", SDL_GetError());
    restore_geometry(window, g_settings.geom);
    renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) FATAL("SDL_CreateRenderer: %s\n", SDL_GetError());
    SDL_SetRenderVSync(renderer, 0);
    apply_layout();
    if (g_settings.fullscreen) SDL_SetWindowFullscreen(window, true);
    tex_top = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 400, 240);
    tex_bottom = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 320, 240);
    SDL_SetTextureScaleMode(tex_top, SDL_SCALEMODE_LINEAR);
    SDL_SetTextureScaleMode(tex_bottom, SDL_SCALEMODE_LINEAR);
    if (const char *s = getenv("RECOMP_SHOT")) shot_frame = (u32)atoi(s);
    if (const char *s = getenv("RECOMP_SHOT_FROM")) shot_from = (u32)atoi(s);
    // RECOMP_SPEED=<percent>: game speed for this run (tests: e.g. 400 plays intros four times faster)
    if (const char *s = getenv("RECOMP_SPEED")) g_settings.speed = std::clamp(atoi(s), 25, 2000);
    gpu::init();
    audio::init();
    if (gpu::scale()) scale_mode = gpu::scale_is_auto() ? 0 : gpu::scale();
    if (scale_mode == 0 && g_settings.scale) frontend_set_scale_mode(g_settings.scale);
    if (g_settings.tex_filter && !getenv("GPU_TEX_FILTER")) gpu::set_texture_filter(g_settings.tex_filter);
    audio::set_volume(g_settings.volume / 100.0f);
    menu::init(window, renderer);
    if (g_settings.split) frontend_set_split(true);
    profile::load();
}

// ---- what the in-game menu changes (runtime/menu.mm)
void frontend_set_layout(int l) {
    const int from = shown_layout();
    layout = std::clamp(l, 0, kLayouts - 1);
    apply_layout();
    reshape(from, shown_layout());
}
void frontend_set_split(bool on) {
    if (on == (window2 != nullptr)) return;
    const int from = shown_layout();
    if (on) {
        int x, y, ww, wh;
        SDL_GetWindowPosition(window, &x, &y);
        SDL_GetWindowSize(window, &ww, &wh);
        const float k = std::min((float)ww / layouts[from].w, (float)wh / layouts[from].h);
        window2 = SDL_CreateWindow((g_game.name + " (bottom screen)").c_str(), (int)(320 * k), (int)(240 * k),
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (!window2) {
            LOG("frontend: cannot open the second window: %s\n", SDL_GetError());
            return;
        }
        renderer2 = SDL_CreateRenderer(window2, nullptr);
        SDL_SetRenderVSync(renderer2, vsync);
        SDL_SetRenderLogicalPresentation(renderer2, 320, 240, SDL_LOGICAL_PRESENTATION_LETTERBOX);
        tex_bottom2 = SDL_CreateTexture(renderer2, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, 320, 240);
        SDL_SetWindowPosition(window2, x + ww + 16, y);  // beside the main window, unless a saved place exists
        restore_geometry(window2, g_settings.geom2);
        if (g_settings.fullscreen2) SDL_SetWindowFullscreen(window2, true);
    } else {
        store_geometry(window2);
        SDL_DestroyRenderer(renderer2);  // (and its textures)
        SDL_DestroyWindow(window2);
        window2 = nullptr;
        renderer2 = nullptr;
        tex_bottom2 = hi_bottom2 = nullptr;
    }
    shown_ver[0] = shown_ver[1] = 0;
    apply_layout();
    reshape(from, shown_layout());
    SDL_RaiseWindow(window);
}
void frontend_set_fullscreen(bool on) { SDL_SetWindowFullscreen(window, on); }
void frontend_set_scale_mode(int mode) {
    if (scale_mode < 0) return;
    scale_mode = mode;
    if (mode) gpu::set_scale(mode);
}
int frontend_current_scale() { return gpu::scale(); }
extern char **g_argv;
void frontend_restart() {
    store_geometry(window);
    store_geometry(window2);
    settings_save();
    LOG("frontend: restarting\n");
    unsetenv("RECOMP_INPUT");  // a test script runs once
    char exe[4096];
    uint32_t len = sizeof exe;
    if (_NSGetExecutablePath(exe, &len) == 0) execv(exe, g_argv);
    FATAL("cannot restart: %s\n", strerror(errno));
}
void frontend_quit() {
    store_geometry(window);
    store_geometry(window2);
    settings_save();
    _exit(0);
}

// window pixels per 3DS pixel
static float present_factor() {
    SDL_FRect r;
    if (!SDL_GetRenderLogicalPresentationRect(renderer, &r) || r.w <= 0) return 1;
    return r.w / layouts[shown_layout()].w;
}

// auto scale: the smallest internal scale that covers the window's pixels, once the size has settled, lowered
// while the GPU is saturated (a heavy game at a high scale would otherwise drop its own frames)
static void update_auto_scale() {
    static int pending, stable, load_cap = kMaxAutoScale, hot, cool;
    static u64 frames_before;
    if (scale_mode != 0) return;
    if (frame_count % 60 == 0 && !paused) {
        const float b = gpu::busy();
        const int s = gpu::scale();
        // the game is dropping frames: neither its full rate nor a steady half rate (many games run at 30 fps)
        const u64 fps = game_frames - frames_before;
        const bool dropping = fps < 57 && (fps < 28 || fps > 31);
        // with the GPU busy too (the GPU thread waits on the GPU, so ~50% already means it is the bottleneck)
        hot = dropping && b > 0.5f ? hot + 1 : 0;
        // raise it again only when the next scale should fit comfortably (the cost grows with the pixel count)
        cool = s && b * (s + 1) * (s + 1) / (s * s) < 0.45f ? cool + 1 : 0;
        if (hot >= 3 && s > 1) {
            load_cap = s - 1;
            hot = 0;
            LOG("frontend: %llu fps with the GPU %.0f%% busy at %dx, automatic resolution lowered to %dx\n",
                (unsigned long long)fps, b * 100, s, load_cap);
        } else if (cool >= 20 && load_cap < kMaxAutoScale) {
            load_cap++;
            cool = 0;
        }
    }
    if (frame_count % 60 == 0) frames_before = game_frames;
    const int want = std::min(load_cap, std::clamp((int)std::ceil(present_factor() - 0.05f), 1, kMaxAutoScale));
    if (want == gpu::scale()) {
        stable = 0;
        return;
    }
    stable = want == pending ? stable + 1 : 0;
    pending = want;
    if (stable >= 30) gpu::set_scale(want);
}

static void cycle_scale() {
    if (scale_mode < 0) return;
    scale_mode = (scale_mode + 1) % 5;
    if (scale_mode) gpu::set_scale(scale_mode);
    LOG("frontend: internal resolution %s\n", scale_mode ? std::to_string(scale_mode).append("x").c_str() : "auto");
}

// nearest at integer magnifications, sharp bilinear at other ones, linear when shrinking
static void set_filter(SDL_Texture *t, float out_w, int tex_w) {
    const float k = out_w / tex_w;
    const SDL_ScaleMode m = k >= 1 && std::fabs(k - std::round(k)) < 0.01f ? SDL_SCALEMODE_NEAREST
                            : k > 1                                       ? SDL_SCALEMODE_PIXELART
                                                                          : SDL_SCALEMODE_LINEAR;
    SDL_ScaleMode cur;
    if (!SDL_GetTextureScaleMode(t, &cur) || cur != m) SDL_SetTextureScaleMode(t, m);
}

// Converts a guest framebuffer (stored rotated: one 240-pixel column per line, bottom to top) to RGBA.
static void convert_fb(const kernel::FramebufferInfo &f, int width, u32 *out) {
    const int height = 240;
    if (!f.valid || !f.addr_left) {
        memset(out, 0, (size_t)width * height * 4);
        return;
    }
    u32 fmt = f.format & 7;
    int bpp = fmt == 0 ? 4 : fmt == 1 ? 3 : 2;
    u32 stride = f.stride ? f.stride : 240 * bpp;
    const u8 *base = gptr(f.addr_left);
    for (int x = 0; x < width; x++) {
        const u8 *col = base + (size_t)x * stride;
        for (int y = 0; y < height; y++) {
            const u8 *p = col + (size_t)(height - 1 - y) * bpp;
            u32 r, g, b;
            switch (fmt) {
            case 0: r = p[3]; g = p[2]; b = p[1]; break;
            case 1: r = p[2]; g = p[1]; b = p[0]; break;
            case 2: {
                u16 v = p[0] | (p[1] << 8);
                r = ((v >> 11) & 31) * 255 / 31; g = ((v >> 5) & 63) * 255 / 63; b = (v & 31) * 255 / 31;
                break;
            }
            case 3: {
                u16 v = p[0] | (p[1] << 8);
                r = ((v >> 11) & 31) * 255 / 31; g = ((v >> 6) & 31) * 255 / 31; b = ((v >> 1) & 31) * 255 / 31;
                break;
            }
            default: {
                u16 v = p[0] | (p[1] << 8);
                r = ((v >> 12) & 15) * 17; g = ((v >> 8) & 15) * 17; b = ((v >> 4) & 15) * 17;
                break;
            }
            }
            out[y * width + x] = r | (g << 8) | (b << 16) | 0xFF000000u;
        }
    }
}

static void write_ppm(const char *path, const u32 *px, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        u8 c[3] = {(u8)px[i], (u8)(px[i] >> 8), (u8)(px[i] >> 16)};
        fwrite(c, 1, 3, f);
    }
    fclose(f);
}

// Uploads the screen image: the GPU backend's high-resolution one when it has it, else the guest framebuffer.
// (the high-resolution image is uploaded only when it is a new one)
static SDL_Texture *upload_screen(SDL_Renderer *r, int screen, SDL_Texture *&hires, int &hw, int &hh,
                                  SDL_Texture *native, u32 *buf) {
    u64 *shown = shown_ver;
    kernel::FramebufferInfo f = kernel::gsp_framebuffer(screen);
    int width = screen ? 320 : 400;
    const u32 fb = f.valid ? mem::vaddr_to_paddr(f.addr_left) : 0;
    const u64 ver = !fb ? 0 : gpu::use_display_image(fb, f.stride, screen ? 320 : 400, hires ? shown[screen] : 0, [&](const u32 *img, int w, int h) {
        if (!hires || w != hw || h != hh) {
            if (hires) SDL_DestroyTexture(hires);
            hires = SDL_CreateTexture(r, SDL_PIXELFORMAT_ABGR8888, SDL_TEXTUREACCESS_STREAMING, w, h);
            hw = w;
            hh = h;
        }
        SDL_UpdateTexture(hires, nullptr, img, w * 4);
    });
    static u32 last_fb;
    if (screen == 0 && (ver ? ver != shown[0] : f.addr_left != last_fb)) game_frames++;  // a new image: the game's fps
    if (screen == 0) last_fb = f.addr_left;
    shown[screen] = ver;
    if (ver) return hires;
    convert_fb(f, width, buf);
    SDL_UpdateTexture(native, nullptr, buf, width * 4);
    return native;
}

// RECOMP_WINSHOT="f1,f2,...": the whole window (menu included) as <prefix>_<frame>.ppm, for tests
static void write_window_shot(SDL_Renderer *r, const char *prefix) {
    static const char *winshot = getenv("RECOMP_WINSHOT");
    if (!winshot || (std::string(",") + winshot + ",").find("," + std::to_string(frame_count + 1) + ",") == std::string::npos)
        return;
    SDL_Surface *s = SDL_RenderReadPixels(r, nullptr);
    if (!s) return;
    SDL_Surface *c = SDL_ConvertSurface(s, SDL_PIXELFORMAT_ABGR8888);
    char name[64];
    snprintf(name, sizeof name, "%s_%06llu.ppm", prefix, (unsigned long long)frame_count + 1);
    std::vector<u32> px((size_t)c->w * c->h);
    for (int y = 0; y < c->h; y++) memcpy(&px[(size_t)y * c->w], (u8 *)c->pixels + (size_t)y * c->pitch, c->w * 4);
    write_ppm(name, px.data(), c->w, c->h);
    SDL_DestroySurface(c);
    SDL_DestroySurface(s);
}

// ---- where the screens go. In the single-screen layouts the focused screen fills the window and the other one
// can show in a corner (picture in picture): always, or for a few seconds whenever its picture changes a lot. With
// automatic focus, the game's profile picks the focused screen (Tab flips it until the profile's choice changes).
struct Placement {
    SDL_FRect top{}, bottom{};  // w 0: not shown
    int pip = -1;               // the screen in the corner (-1: none)
    float alpha = 1;            // the corner screen's opacity
};
static Placement placement;
static int manual_flip;                                  // Tab while the profile picks the screen
static profile::Focus flip_for = profile::FOCUS_NONE;   // the profile's choice when Tab was pressed
static Uint64 pip_active_until;
static float pip_alpha;
static u64 hud_tap_until;  // the HUD button's tap on the bottom screen lasts until this frame
// moving and resizing the corner screen with the mouse: a grip tab (move) and a handle (resize) appear next to it
// while the pointer is over it; the scroll wheel resizes it too
static SDL_FRect pip_rect, pip_grip, pip_handle;  // this frame's, in canvas units (w 0: none)
static int pip_drag;                               // 0 none, 1 moving, 2 resizing
static float drag_dx, drag_dy, drag_ax, drag_ay;   // grab offset / the corner that stays put while resizing
static bool pip_hover;
static int hud_tap_x, hud_tap_y;
// typing into a game's own keyboard (profile "keyboard" / "key"): characters become taps on its keys
static bool typing;          // the game's keyboard is up: the Mac's keyboard types into it
static std::deque<std::pair<int, int>> key_taps;
static u64 key_tap_start;    // frame the first queued tap began
static void type_key(const std::string &k) {
    int x, y;
    if (!profile::key(k, x, y)) return;
    if (key_taps.empty()) key_tap_start = frame_count;
    key_taps.push_back({x, y});
}
static void type_text(const std::string &s) {
    for (size_t i = 0; i < s.size();) {
        size_t j = i + 1;
        while (j < s.size() && ((u8)s[j] & 0xC0) == 0x80) j++;
        const std::string c = s.substr(i, j - i);
        i = j;
        int x, y;
        if (c == " ") type_key("space");
        else if (profile::key(c, x, y)) type_key(c);
        else if (c.size() == 1 && isupper((u8)c[0]) && profile::key(std::string(1, (char)tolower(c[0])), x, y)) {
            type_key("shift");
            type_key(std::string(1, (char)tolower(c[0])));
        }
    }
}

static bool single_layout() { return !window2 && (layout == 1 || layout == 4); }
static profile::Focus auto_focus() { return g_settings.auto_focus ? profile::focus() : profile::FOCUS_NONE; }

// 0 top, 1 bottom
static int focused_screen() {
    const profile::Focus f = auto_focus();
    if (f != flip_for) {
        manual_flip = 0;
        flip_for = f;
    }
    int s = layout == 4 ? 1 : 0;
    if (f == profile::FOCUS_TOP || f == profile::FOCUS_BOTH) s = 0;
    if (f == profile::FOCUS_BOTTOM) s = 1;
    return s ^ manual_flip;
}

// whether a screen's picture changed a lot since the last frame (40x30 cells of luminance)
static bool screen_changed(int screen) {
    static float prev[40 * 30];
    static int prev_screen = -1;
    static u32 buf[400 * 240];
    const int w = screen ? 320 : 400;
    convert_fb(kernel::gsp_framebuffer(screen), w, buf);
    float cur[40 * 30] = {};
    for (int y = 0; y < 240; y += 2)
        for (int x = 0; x < w; x += 2) {
            const u32 p = buf[y * w + x];
            cur[(y * 30 / 240) * 40 + x * 40 / w] += (float)((p & 255) * 2 + (p >> 8 & 255) * 5 + (p >> 16 & 255)) / 8;
        }
    const float cell = (w / 40.0f / 2) * (240 / 30.0f / 2);
    int changed = 0;
    for (int i = 0; i < 40 * 30; i++) {
        cur[i] /= cell;
        changed += std::fabs(cur[i] - prev[i]) > 10;
    }
    const bool first = prev_screen != screen;
    memcpy(prev, cur, sizeof prev);
    prev_screen = screen;
    return !first && changed > 40 * 30 * 8 / 100;
}

static void place_screens() {
    const ScreenLayout &l = layouts[shown_layout()];
    placement = {l.top, l.bottom, -1, 1};
    if (!single_layout()) return;
    const int main = focused_screen(), other = main ^ 1;
    auto fit = [&](int screen) {
        const float sw = screen ? 320 : 400, k = std::min(l.w / sw, l.h / 240.0f);
        return SDL_FRect{(l.w - sw * k) / 2, (l.h - 240 * k) / 2, sw * k, 240 * k};
    };
    (main ? placement.bottom : placement.top) = fit(main);
    (other ? placement.bottom : placement.top) = {};
    // the corner screen
    const Uint64 now = SDL_GetTicks();
    if (g_settings.pip == 1 && screen_changed(other)) pip_active_until = now + 3000;
    const float ph = std::min(l.h * g_settings.pip_size / 100.0f, l.w * 240.0f / (other ? 320 : 400)),
                pw = ph * (other ? 320 : 400) / 240.0f, m = l.h * 0.025f;
    const int c = g_settings.pip_corner;
    SDL_FRect r{c == 1 || c == 3 ? m : l.w - pw - m, c >= 2 ? l.h - ph - m : m, pw, ph};
    if (g_settings.pip_x >= 0) {
        r.x = std::clamp(g_settings.pip_x * l.w / 1000.0f, 0.0f, l.w - pw);
        r.y = std::clamp(g_settings.pip_y * l.h / 1000.0f, 0.0f, l.h - ph);
    }
    // the grip tab and the resize handle go on the side facing the middle of the window
    const bool below = r.y + r.h / 2 < l.h / 2, left = r.x + r.w / 2 > l.w / 2;
    const float gh = std::max(l.h * 0.05f, 6.0f), gw = std::min(r.w * 0.4f, gh * 5);
    pip_rect = r;
    pip_grip = {r.x + (r.w - gw) / 2, below ? r.y + r.h : r.y - gh, gw, gh};
    pip_handle = {left ? r.x - gh : r.x + r.w, below ? r.y + r.h : r.y - gh, gh, gh};
    float mx, my, lx = -1, ly = -1;
    if (SDL_GetMouseFocus() == window) {
        SDL_GetMouseState(&mx, &my);
        SDL_RenderCoordinatesFromWindow(renderer, mx, my, &lx, &ly);
    }
    auto in = [&](const SDL_FRect &q) { return lx >= q.x && lx < q.x + q.w && ly >= q.y && ly < q.y + q.h; };
    pip_hover = pip_drag || (pip_alpha > 0.5f && (in(r) || in(pip_grip) || in(pip_handle)));
    const bool want = g_settings.pip == 2 || auto_focus() == profile::FOCUS_BOTH || pip_hover ||
                      (g_settings.pip == 1 && now < pip_active_until);
    pip_alpha = std::clamp(pip_alpha + (want ? 0.12f : -0.08f), 0.0f, 1.0f);
    if (pip_alpha <= 0) {
        pip_rect = pip_grip = pip_handle = {};
        return;
    }
    (other ? placement.bottom : placement.top) = r;
    placement.pip = other;
    placement.alpha = pip_alpha;
}

// the mouse moving or resizing the corner screen; true if the event was used
static bool pip_mouse(const SDL_Event &ev) {
    if (window2 || placement.pip < 0) {
        pip_drag = 0;
        return false;
    }
    const ScreenLayout &l = layouts[shown_layout()];
    float lx = 0, ly = 0;
    auto at = [&](float x, float y) { SDL_RenderCoordinatesFromWindow(renderer, x, y, &lx, &ly); };
    auto in = [&](const SDL_FRect &q) { return q.w && lx >= q.x && lx < q.x + q.w && ly >= q.y && ly < q.y + q.h; };
    const float aspect = pip_rect.h > 0 ? pip_rect.w / pip_rect.h : 4.0f / 3;
    auto store = [&](float x, float y, float h) {
        const float w = h * aspect;
        x = std::clamp(x, 0.0f, l.w - w);
        y = std::clamp(y, 0.0f, l.h - h);
        g_settings.pip_x = (int)std::lround(x * 1000 / l.w);
        g_settings.pip_y = (int)std::lround(y * 1000 / l.h);
        g_settings.pip_size = std::clamp((int)std::lround(h * 100 / l.h), 15, 80);
    };
    switch (ev.type) {
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (ev.button.button != SDL_BUTTON_LEFT || SDL_GetWindowFromID(ev.button.windowID) != window) return false;
        at(ev.button.x, ev.button.y);
        if (in(pip_grip) || (placement.pip == 0 && in(pip_rect))) {  // (the top screen takes no touches)
            pip_drag = 1;
            drag_dx = lx - pip_rect.x;
            drag_dy = ly - pip_rect.y;
            return true;
        }
        if (in(pip_handle)) {
            pip_drag = 2;  // the opposite corner stays put
            drag_ax = pip_handle.x < pip_rect.x ? pip_rect.x + pip_rect.w : pip_rect.x;
            drag_ay = pip_handle.y < pip_rect.y ? pip_rect.y + pip_rect.h : pip_rect.y;
            return true;
        }
        return false;
    case SDL_EVENT_MOUSE_MOTION:
        if (!pip_drag) return false;
        at(ev.motion.x, ev.motion.y);
        if (pip_drag == 1) {
            store(lx - drag_dx, ly - drag_dy, pip_rect.h);
        } else {
            const float h = std::clamp(std::max(std::fabs(ly - drag_ay), std::fabs(lx - drag_ax) / aspect), l.h * 0.15f, l.h * 0.8f);
            store(lx < drag_ax ? drag_ax - h * aspect : drag_ax, ly < drag_ay ? drag_ay - h : drag_ay, h);
        }
        return true;
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (!pip_drag || ev.button.button != SDL_BUTTON_LEFT) return false;
        pip_drag = 0;
        settings_save();
        return true;
    case SDL_EVENT_MOUSE_WHEEL: {  // resize around the middle
        at(ev.wheel.mouse_x, ev.wheel.mouse_y);
        if (!(in(pip_rect) || in(pip_grip) || in(pip_handle)) || ev.wheel.y == 0) return false;
        const float h = std::clamp(pip_rect.h * (1 + 0.06f * ev.wheel.y), l.h * 0.15f, l.h * 0.8f);
        store(pip_rect.x + (pip_rect.w - h * aspect) / 2, pip_rect.y + (pip_rect.h - h) / 2, h);
        settings_save();
        return true;
    }
    }
    return false;
}

// a screenshot is due at frame n (RECOMP_SHOT / RECOMP_WINSHOT)
static bool shot_due(u64 n) {
    static const char *winshot = getenv("RECOMP_WINSHOT");
    if (shot_frame && n >= shot_from && n % shot_frame == 0) return true;
    return winshot && (std::string(",") + winshot + ",").find("," + std::to_string(n) + ",") != std::string::npos;
}

static void present() {
    static u32 top[400 * 240], bottom[320 * 240];
    // faster than the 3DS: draw at most every 12 ms (a display shows no more), except the frames being captured
    static auto last_drawn = steady_clock::now();
    const auto t = steady_clock::now();
    if (g_settings.speed > 100 && t - last_drawn < milliseconds(12) && !shot_due(frame_count + 1)) {
        frame_count++;
        g_frame = frame_count;
        return;
    }
    last_drawn = t;
    static SDL_Texture *hi_top, *hi_bottom;
    static int htw, hth, hbw, hbh;
    SDL_Texture *t_top = upload_screen(renderer, 0, hi_top, htw, hth, tex_top, top);
    SDL_Texture *t_bottom = window2 ? upload_screen(renderer2, 1, hi_bottom2, hb2w, hb2h, tex_bottom2, bottom)
                                    : upload_screen(renderer, 1, hi_bottom, hbw, hbh, tex_bottom, bottom);
    if (window2) {
        SDL_SetRenderDrawColor(renderer2, 0, 0, 0, 255);
        SDL_RenderClear(renderer2);
        SDL_FRect r, full{0, 0, 320, 240};
        SDL_GetRenderLogicalPresentationRect(renderer2, &r);
        set_filter(t_bottom, r.w > 0 ? r.w : 320, t_bottom->w);
        SDL_RenderTexture(renderer2, t_bottom, nullptr, &full);
        write_window_shot(renderer2, "window2");
        SDL_RenderPresent(renderer2);
    }
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    place_screens();
    const Placement &pl = placement;
    const float f = present_factor();
    auto draw = [&](int screen) {
        const SDL_FRect &d = screen ? pl.bottom : pl.top;
        SDL_Texture *t = screen ? t_bottom : t_top;
        if (!d.w || !t) return;
        const bool corner = pl.pip == screen;
        if (corner) {  // a thin dark frame
            const float b = std::max(1.0f, d.h * 0.012f);
            const SDL_FRect fr{d.x - b, d.y - b, d.w + 2 * b, d.h + 2 * b};
            SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
            SDL_SetRenderDrawColorFloat(renderer, 0, 0, 0, 0.7f * pl.alpha);
            SDL_RenderFillRect(renderer, &fr);
            SDL_SetTextureAlphaModFloat(t, pl.alpha);
        }
        set_filter(t, d.w * f, t->w);
        SDL_RenderTexture(renderer, t, nullptr, &d);
        if (corner) SDL_SetTextureAlphaModFloat(t, 1);
    };
    // the corner screen goes over the main one
    draw(pl.pip == 0 ? 1 : 0);
    draw(pl.pip == 0 ? 0 : 1);
    if (pl.pip >= 0 && pip_hover) {  // grip tab (three bars) and resize handle (a diagonal)
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColorFloat(renderer, 0.1f, 0.1f, 0.12f, 0.85f * pl.alpha);
        SDL_RenderFillRect(renderer, &pip_grip);
        SDL_RenderFillRect(renderer, &pip_handle);
        SDL_SetRenderDrawColorFloat(renderer, 1, 1, 1, 0.8f * pl.alpha);
        const SDL_FRect &g = pip_grip, &h = pip_handle;
        for (int i = 1; i <= 3; i++) {
            const SDL_FRect bar{g.x + g.w * 0.3f, g.y + g.h * i / 4 - g.h * 0.05f, g.w * 0.4f, std::max(g.h * 0.1f, 1.0f)};
            SDL_RenderFillRect(renderer, &bar);
        }
        for (float t = 0.25f; t <= 0.75f; t += 0.02f) {
            const SDL_FRect dot{h.x + h.w * t - 0.5f, h.y + h.h * (1 - t) - 0.5f, std::max(h.w * 0.1f, 1.0f),
                                std::max(h.h * 0.1f, 1.0f)};
            SDL_RenderFillRect(renderer, &dot);
        }
    }
    // the HUD over the top screen: the bottom screen's information when it is out of sight (or always)
    const bool bottom_seen = window2 || (pl.bottom.w && (pl.pip != 1 || pl.alpha > 0.5f));
    if (pl.top.w && pl.pip != 0 && (g_settings.hud == 2 || (g_settings.hud == 1 && !bottom_seen))) hud::draw(renderer, pl.top);
    menu::draw();
    write_window_shot(renderer, "window");
    SDL_RenderPresent(renderer);
    frame_count++;
    g_frame = frame_count;
    if (shot_frame && frame_count >= shot_from && frame_count % shot_frame == 0) {
        convert_fb(kernel::gsp_framebuffer(0), 400, top);
        convert_fb(kernel::gsp_framebuffer(1), 320, bottom);
        char name[64];
        snprintf(name, sizeof name, "shot_top_%06llu.ppm", (unsigned long long)frame_count);
        write_ppm(name, top, 400, 240);
        snprintf(name, sizeof name, "shot_bottom_%06llu.ppm", (unsigned long long)frame_count);
        write_ppm(name, bottom, 320, 240);
        const u32 *img;
        int w, h;
        kernel::FramebufferInfo f = kernel::gsp_framebuffer(0);
        if (f.valid && gpu::get_display_image_for(mem::vaddr_to_paddr(f.addr_left), f.stride, 400, &img, &w, &h)) {
            snprintf(name, sizeof name, "shot_top_hires_%06llu.ppm", (unsigned long long)frame_count);
            write_ppm(name, img, w, h);
        }
    }
}

// 3DS buttons
enum : u32 {
    BTN_A = 1 << 0, BTN_B = 1 << 1, BTN_SELECT = 1 << 2, BTN_START = 1 << 3, BTN_RIGHT = 1 << 4, BTN_LEFT = 1 << 5,
    BTN_UP = 1 << 6, BTN_DOWN = 1 << 7, BTN_R = 1 << 8, BTN_L = 1 << 9, BTN_X = 1 << 10, BTN_Y = 1 << 11,
    BTN_ZL = 1 << 14, BTN_ZR = 1 << 15,
};

// scripted input for testing: RECOMP_INPUT="frame:BUTTONS:duration,..." (BUTTONS like A+START; T@x/y touches
// the bottom screen at pixel x, y)
struct ScriptEntry { u64 frame, dur; u32 buttons; int cx, cy, tx = -1, ty = -1; };
// host keys for testing the menu: "frame:KEY_<SDL key name>" (KEY_Escape, KEY_Down, KEY_Return...), typed text
// ("frame:TEXT_<text>", for the software keyboard, or taps on the game's own keyboard from the profile's keys), and
// the mouse in layout coordinates: "frame:MOUSE_DOWN@x/y"
// (also MOUSE_MOVE, MOUSE_UP) and "frame:MOUSE_WHEEL@x/y/dy"
struct ScriptKey {
    u64 frame;
    SDL_Scancode key;
    SDL_EventType mouse = (SDL_EventType)0;
    float x = 0, y = 0, wheel = 0;
    std::string text;
};
static std::vector<ScriptKey> script_keys;
static std::vector<ScriptEntry> script;
static void parse_script() {
    const char *s = getenv("RECOMP_INPUT");
    if (!s) return;
    std::string str = s;
    size_t pos = 0;
    while (pos < str.size()) {
        size_t e = str.find(',', pos);
        std::string item = str.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
        pos = e == std::string::npos ? str.size() : e + 1;
        unsigned long long f = 0, d = 1;
        char names[128] = {};
        if (sscanf(item.c_str(), "%llu:%127[^:]:%llu", &f, names, &d) < 2) continue;
        u32 b = 0;
        std::string n = names;
        if (n.rfind("KEY_", 0) == 0) {
            script_keys.push_back({f, SDL_GetScancodeFromName(n.c_str() + 4)});
            continue;
        }
        if (n.rfind("TEXT_", 0) == 0) {
            script_keys.push_back({f, SDL_SCANCODE_UNKNOWN, (SDL_EventType)0, 0, 0, 0, n.substr(5)});
            continue;
        }
        if (n.rfind("MOUSE_", 0) == 0) {
            ScriptKey k{f, SDL_SCANCODE_UNKNOWN};
            const size_t at = n.find('@');
            const std::string kind = n.substr(6, at == std::string::npos ? std::string::npos : at - 6);
            k.mouse = kind == "DOWN" ? SDL_EVENT_MOUSE_BUTTON_DOWN : kind == "UP" ? SDL_EVENT_MOUSE_BUTTON_UP
                    : kind == "WHEEL"  ? SDL_EVENT_MOUSE_WHEEL        : SDL_EVENT_MOUSE_MOTION;
            if (at != std::string::npos) sscanf(n.c_str() + at + 1, "%f/%f/%f", &k.x, &k.y, &k.wheel);
            script_keys.push_back(k);
            continue;
        }
        auto has = [&](const char *k) { return n.find(k) != std::string::npos; };
        if (has("START")) b |= BTN_START;
        if (has("SELECT")) b |= BTN_SELECT;
        if (n == "A" || has("A+") || has("+A")) b |= BTN_A;
        if (n == "B" || has("B+") || has("+B")) b |= BTN_B;
        if (has("X")) b |= BTN_X;
        if (has("Y")) b |= BTN_Y;
        if (has("DRIGHT")) b |= BTN_RIGHT;
        if (has("DLEFT")) b |= BTN_LEFT;
        if (has("DUP")) b |= BTN_UP;
        if (has("DDOWN")) b |= BTN_DOWN;
        if (has("ZL")) b |= BTN_ZL;
        if (has("ZR")) b |= BTN_ZR;
        if (has("+L") || n == "L" || has("L+")) b |= BTN_L;
        if (has("+R") || n == "R" || has("R+")) b |= BTN_R;
        int cx = 0, cy = 0;
        if (has("CRIGHT")) cx = 155;
        if (has("CNUDGE")) cx = 12;
        if (has("CLEFT")) cx = -155;
        if (has("CUP")) cy = 155;
        if (has("CDOWN")) cy = -155;
        int tx = -1, ty = -1;
        if (size_t t = n.find("T@"); t != std::string::npos) sscanf(n.c_str() + t + 2, "%d/%d", &tx, &ty);
        script.push_back({f, d, b, cx, cy, tx, ty});
    }
}

static void poll_input(u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty) {
    const bool *k = SDL_GetKeyboardState(nullptr);
    static const bool none[SDL_SCANCODE_COUNT] = {};
    const bool *kb = typing ? none : k;  // keys type while the game's keyboard is up (the arrows still move)
    buttons = 0;
    if (kb[SDL_SCANCODE_X]) buttons |= BTN_A;
    if (kb[SDL_SCANCODE_Z]) buttons |= BTN_B;
    if (kb[SDL_SCANCODE_S]) buttons |= BTN_X;
    if (kb[SDL_SCANCODE_A]) buttons |= BTN_Y;
    if (kb[SDL_SCANCODE_Q]) buttons |= BTN_L;
    if (kb[SDL_SCANCODE_W]) buttons |= BTN_R;
    if (kb[SDL_SCANCODE_1]) buttons |= BTN_ZL;
    if (kb[SDL_SCANCODE_2]) buttons |= BTN_ZR;
    if (kb[SDL_SCANCODE_RETURN]) buttons |= BTN_START;
    if (kb[SDL_SCANCODE_BACKSPACE] || k[SDL_SCANCODE_RSHIFT]) buttons |= BTN_SELECT;
    if (kb[SDL_SCANCODE_I]) buttons |= BTN_UP;
    if (kb[SDL_SCANCODE_K]) buttons |= BTN_DOWN;
    if (kb[SDL_SCANCODE_J]) buttons |= BTN_LEFT;
    if (kb[SDL_SCANCODE_L]) buttons |= BTN_RIGHT;
    int x = 0, y = 0;
    if (k[SDL_SCANCODE_LEFT]) x -= 1;
    if (k[SDL_SCANCODE_RIGHT]) x += 1;
    if (k[SDL_SCANCODE_UP]) y += 1;
    if (k[SDL_SCANCODE_DOWN]) y -= 1;
    cx = (s16)(x * 155);
    cy = (s16)(y * 155);
    if (pad) {
        auto btn = [&](SDL_GamepadButton b) { return SDL_GetGamepadButton(pad, b); };
        if (btn(SDL_GAMEPAD_BUTTON_EAST)) buttons |= BTN_A;
        if (btn(SDL_GAMEPAD_BUTTON_SOUTH)) buttons |= BTN_B;
        if (btn(SDL_GAMEPAD_BUTTON_NORTH)) buttons |= BTN_X;
        if (btn(SDL_GAMEPAD_BUTTON_WEST)) buttons |= BTN_Y;
        if (btn(SDL_GAMEPAD_BUTTON_LEFT_SHOULDER)) buttons |= BTN_L;
        if (btn(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER)) buttons |= BTN_R;
        if (btn(SDL_GAMEPAD_BUTTON_START)) buttons |= BTN_START;
        if (btn(SDL_GAMEPAD_BUTTON_BACK)) buttons |= BTN_SELECT;
        if (btn(SDL_GAMEPAD_BUTTON_DPAD_UP)) buttons |= BTN_UP;
        if (btn(SDL_GAMEPAD_BUTTON_DPAD_DOWN)) buttons |= BTN_DOWN;
        if (btn(SDL_GAMEPAD_BUTTON_DPAD_LEFT)) buttons |= BTN_LEFT;
        if (btn(SDL_GAMEPAD_BUTTON_DPAD_RIGHT)) buttons |= BTN_RIGHT;
        if (SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16000) buttons |= BTN_ZL;
        if (SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16000) buttons |= BTN_ZR;
        int ax = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTX), ay = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFTY);
        if (abs(ax) > 4000 || abs(ay) > 4000) {
            cx = (s16)(ax * 156 / 32767);
            cy = (s16)(-ay * 156 / 32767);
        }
    }
    for (auto &e : script)
        if (frame_count >= e.frame && frame_count < e.frame + e.dur) {
            buttons |= e.buttons;
            if (e.cx) cx = (s16)e.cx;
            if (e.cy) cy = (s16)e.cy;
        }
    // touch: mouse on the bottom screen
    float mx, my;
    SDL_MouseButtonFlags mb = SDL_GetMouseState(&mx, &my);
    touch = false;
    for (auto &e : script)
        if (e.tx >= 0 && frame_count >= e.frame && frame_count < e.frame + e.dur) {
            touch = true;
            tx = (u16)std::clamp(e.tx, 0, 319);
            ty = (u16)std::clamp(e.ty, 0, 239);
        }
    if (!touch && !key_taps.empty()) {  // typed keys: each tap held 6 frames, then 6 released (the game runs at 30 fps)
        const u64 t = frame_count - key_tap_start;
        if (t < 6) {
            touch = true;
            tx = (u16)std::clamp(key_taps.front().first, 0, 319);
            ty = (u16)std::clamp(key_taps.front().second, 0, 239);
        } else if (t >= 12) {
            key_taps.pop_front();
            key_tap_start = frame_count;
        }
    }
    if (!touch && frame_count < hud_tap_until) {  // the HUD button (profile: "button tap x y")
        touch = true;
        tx = (u16)std::clamp(hud_tap_x, 0, 319);
        ty = (u16)std::clamp(hud_tap_y, 0, 239);
    }
    if (!touch && (mb & SDL_BUTTON_LMASK) && window2 && SDL_GetMouseFocus() == window2) {
        float lx, ly;  // the bottom screen's own window
        SDL_RenderCoordinatesFromWindow(renderer2, mx, my, &lx, &ly);
        if (lx >= 0 && lx < 320 && ly >= 0 && ly < 240) {
            touch = true;
            tx = (u16)lx;
            ty = (u16)ly;
        }
    } else if (!touch && !pip_drag && (mb & SDL_BUTTON_LMASK) && SDL_GetMouseFocus() == window) {
        float lx, ly;
        SDL_RenderCoordinatesFromWindow(renderer, mx, my, &lx, &ly);
        const SDL_FRect &b = placement.bottom;  // touch: the mouse on the bottom screen, wherever it is
        if (b.w && (placement.pip != 1 || placement.alpha > 0.5f) && lx >= b.x && lx < b.x + b.w && ly >= b.y && ly < b.y + b.h) {
            touch = true;
            tx = (u16)std::clamp((int)((lx - b.x) * 320 / b.w), 0, 319);
            ty = (u16)std::clamp((int)((ly - b.y) * 240 / b.h), 0, 239);
        }
    }
}

// GX commands run on their own thread so a slow frame never delays VBlank or presentation

static void gpu_thread() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(gpu_m);
            gpu_cv.wait_for(lk, milliseconds(2), [] { return gpu_wake; });
            gpu_wake = false;
        }
        kernel::gsp_process_gx();
    }
}

// The HID module refreshes the pad shared memory at 234 Hz like the real one; the main loop polls the host
// devices at VBlank and publishes a snapshot.
struct InputSnapshot { u32 buttons; s16 cx, cy; bool touch; u16 tx, ty; };
static std::mutex input_m;
static InputSnapshot input_now{};

static void hid_thread() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const auto period = nanoseconds(1000000000ll / 234);
    auto next = steady_clock::now();
    for (;;) {
        next += period;
        std::this_thread::sleep_until(next);
        InputSnapshot s;
        {
            std::lock_guard<std::mutex> lk(input_m);
            s = input_now;
        }
        kernel::hid_update(s.buttons, s.cx, s.cy, s.touch, s.tx, s.ty);
    }
}

static void audio_thread() {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    auto next = steady_clock::now();
    for (;;) {
        next += nanoseconds(1000000000ll * 160 / 32728 * 100 / g_settings.speed);
        std::this_thread::sleep_until(next);
        if (paused) {
            next = steady_clock::now();
            continue;
        }
        kernel::dsp_tick();
    }
}

void watch_init();
void watch_report();

void frontend_run() {
    parse_script();
    watch_init();
    std::thread(audio_thread).detach();
    std::thread(gpu_thread).detach();
    std::thread(hid_thread).detach();
    auto frame = nanoseconds(16715000);  // 59.83 Hz (at 100% game speed)
    auto next_vblank = steady_clock::now() + frame;
    auto next_time = steady_clock::now();
    u64 max_frames = 0;
    if (const char *s = getenv("RECOMP_FRAMES")) max_frames = strtoull(s, nullptr, 10);
    for (;;) {
        auto now = steady_clock::now();
        if (now < next_vblank) {
            std::this_thread::sleep_until(next_vblank);
            continue;
        }
        // faster than the 3DS: the next VBlank waits until the game has finished its frame (for up to one 3DS
        // frame), so it never falls behind and plays the same at any speed
        if (g_settings.speed > 100 && !paused && now < next_vblank + nanoseconds(16715000) &&
            (kernel::guest_busy() || kernel::gsp_gx_pending())) {
            std::this_thread::sleep_for(microseconds(100));
            continue;
        }
        // host events once per frame (pumping the macOS event loop is not free)
        for (auto &k : script_keys)
            if (k.frame == frame_count && k.mouse) {
                float wx, wy;
                SDL_RenderCoordinatesToWindow(renderer, k.x, k.y, &wx, &wy);
                SDL_Event me{};
                me.type = k.mouse;
                if (k.mouse == SDL_EVENT_MOUSE_WHEEL) {
                    me.wheel.windowID = SDL_GetWindowID(window);
                    me.wheel.y = k.wheel;
                    me.wheel.mouse_x = wx;
                    me.wheel.mouse_y = wy;
                } else if (k.mouse == SDL_EVENT_MOUSE_MOTION) {
                    me.motion.windowID = SDL_GetWindowID(window);
                    me.motion.x = wx;
                    me.motion.y = wy;
                } else {
                    me.button.windowID = SDL_GetWindowID(window);
                    me.button.button = SDL_BUTTON_LEFT;
                    me.button.down = k.mouse == SDL_EVENT_MOUSE_BUTTON_DOWN;
                    me.button.x = wx;
                    me.button.y = wy;
                }
                SDL_PushEvent(&me);
            } else if (k.frame == frame_count && !k.text.empty() && !typing && !menu::keyboard_open()) {
                type_text(k.text);  // a test knows the game's own keyboard is up, with or without a "keyboard" rule
            } else if (k.frame == frame_count && !k.text.empty()) {
                SDL_Event te{};
                te.type = SDL_EVENT_TEXT_INPUT;
                te.text.windowID = SDL_GetWindowID(window);
                te.text.text = k.text.c_str();
                SDL_PushEvent(&te);
            } else if (k.frame == frame_count) {
                SDL_Event ke{};
                ke.type = SDL_EVENT_KEY_DOWN;
                ke.key.scancode = k.key;
                ke.key.down = true;
                SDL_PushEvent(&ke);
            }
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) frontend_quit();
            if (ev.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED) {
                if (window2 && ev.window.windowID == SDL_GetWindowID(window2)) {  // back to one window
                    g_settings.split = false;
                    frontend_set_split(false);
                    settings_save();
                } else {
                    frontend_quit();
                }
                continue;
            }
            if (ev.type == SDL_EVENT_WINDOW_MOVED || ev.type == SDL_EVENT_WINDOW_RESIZED)
                store_geometry(SDL_GetWindowFromID(ev.window.windowID));
            if (menu::handle_event(ev)) continue;
            if (typing && ev.type == SDL_EVENT_TEXT_INPUT) {
                type_text(ev.text.text);
                continue;
            }
            if (typing && ev.type == SDL_EVENT_KEY_DOWN) {
                const SDL_Scancode sc = ev.key.scancode;
                if (sc == SDL_SCANCODE_BACKSPACE) type_key("backspace");
                if (sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER) type_key("enter");
                if ((sc < SDL_SCANCODE_F1 && sc != SDL_SCANCODE_TAB) || (sc >= SDL_SCANCODE_KP_DIVIDE && sc <= SDL_SCANCODE_KP_PERIOD))
                    continue;  // a character (its text comes as SDL_EVENT_TEXT_INPUT), not a shortcut
            }
            if (ev.type == SDL_EVENT_GAMEPAD_ADDED && !pad) pad = SDL_OpenGamepad(ev.gdevice.which);
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_F11) {  // the window being used
                const bool second = window2 && SDL_GetKeyboardFocus() == window2;
                SDL_Window *w = second ? window2 : window;
                bool &fs = second ? g_settings.fullscreen2 : g_settings.fullscreen;
                fs = !(SDL_GetWindowFlags(w) & SDL_WINDOW_FULLSCREEN);
                SDL_SetWindowFullscreen(w, fs);
                settings_save();
            }
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_F2) {
                g_settings.layout = (layout + 1) % kLayouts;
                frontend_set_layout(g_settings.layout);
                settings_save();
            }
            if (!window2 && ((ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_TAB && !ev.key.repeat) ||
                             (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && ev.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK))) {
                if (single_layout() && auto_focus() != profile::FOCUS_NONE) {
                    manual_flip ^= 1;  // the profile picks the screen: look at the other one for now
                    pip_alpha = 0;
                } else if (const int l = swapped_layout(layout); l != layout) {
                    g_settings.layout = l;
                    frontend_set_layout(l);
                    settings_save();
                }
            }
            if ((ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_E && !ev.key.repeat) ||
                (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && ev.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK))
                if (const profile::Button *b = profile::button()) {  // the HUD button: a tap on the bottom screen
                    hud_tap_x = b->tx;
                    hud_tap_y = b->ty;
                    hud_tap_until = frame_count + 6;
                }
            if (pip_mouse(ev)) continue;
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_F3) {
                cycle_scale();
                g_settings.scale = scale_mode > 0 ? scale_mode : 0;
                settings_save();
            }
            if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_F12) {  // diagnostics into log.txt
                LOG("[diag] F12: %llu cmdlists, %llu display transfers so far\n",
                    (unsigned long long)kernel::gx_stats.cmdlists, (unsigned long long)kernel::gx_stats.transfers);
                kernel::dump_threads();
            }
        }
        update_auto_scale();
        menu::update();
        paused = menu::is_open();
        frame = nanoseconds(16715000ll * 100 / g_settings.speed);
        kernel::set_clock_speed((u32)g_settings.speed);
        if (const int v = g_settings.speed <= 100 && !getenv("RECOMP_NOVSYNC"); v != vsync) {
            vsync = v;
            SDL_SetRenderVSync(renderer, vsync);
            if (renderer2) SDL_SetRenderVSync(renderer2, vsync);
        }
        now = steady_clock::now();
        {
            u32 b;
            s16 cx, cy;
            bool touch;
            u16 tx = 0, ty = 0;
            poll_input(b, cx, cy, touch, tx, ty);
            if (paused || menu::keyboard_open()) b = 0, cx = cy = 0, touch = false;
            {
                std::lock_guard<std::mutex> lk(input_m);
                input_now = {b, cx, cy, touch, tx, ty};
            }
            if (!paused) kernel::gsp_vblank();  // the game waits for VBlank, so it stops while the menu is open
            profile::update();
            if (profile::keyboard_active() != typing) {
                typing = !typing;
                key_taps.clear();
                if (typing) SDL_StartTextInput(window);
                else if (!menu::keyboard_open()) SDL_StopTextInput(window);
            }
            present();
            static const char *peek = getenv("RECOMP_PEEK");
            if (peek) {
                static u32 last = 0xFFFFFFFF;
                u32 a = (u32)strtoul(peek, nullptr, 16);
                u32 v = r32(a);
                if (v != last) LOG("[peek] %08x = %08x\n", a, v);
                last = v;
            }
            // RECOMP_MEMDUMP="f1,f2,...": guest memory snapshots (mem_<frame>_<address>.bin) for finding game state
            static const char *memdump = getenv("RECOMP_MEMDUMP");
            if (memdump && (std::string(",") + memdump + ",").find("," + std::to_string(frame_count) + ",") != std::string::npos)
                kernel::dump_memory(("mem_" + std::to_string(frame_count)).c_str());
            if (max_frames && frame_count >= max_frames) {
                if (getenv("RECOMP_THREADS")) kernel::dump_threads();
                watch_report();
                if (getenv("RECOMP_DUMP_DSP")) {
                    FILE *f = fopen("dsp_ram.bin", "wb");
                    fwrite(gptr(mem::DSP_BASE), 1, mem::DSP_SIZE, f);
                    fclose(f);
                }
                _exit(0);
            }
            next_vblank += frame;
            if (now > next_vblank + frame * 4) next_vblank = now + frame;
            if (now >= next_time) {
                update_shared_page_time();
                next_time = now + seconds(1);
                static u64 last_game_frames;
                menu::set_fps((float)(game_frames - last_game_frames));
                last_game_frames = game_frames;
                static u64 last_frames, last_lists, last_ns, last_xfers;
                static int sec;
                // Watchdog: a game that stops submitting GPU work for 5 s is probably stuck; record why.
                static u64 wd_lists;
                static int wd_quiet;
                if (kernel::gx_stats.cmdlists != wd_lists) {
                    wd_lists = kernel::gx_stats.cmdlists;
                    wd_quiet = 0;
                } else if (++wd_quiet == 5 && frame_count > 600) {
                    LOG("[diag] no GPU work for 5 s; threads:\n");
                    kernel::dump_threads();
                }
                static bool stats = getenv("RECOMP_STATS") != nullptr;
                ++sec;
                if (stats ? sec % 2 == 0 : sec % 30 == 0) {
                    const double span = stats ? 2 : 30;
                    u64 lists = kernel::gx_stats.cmdlists, ns = kernel::gx_stats.gpu_ns;
                    u64 xf = kernel::gx_stats.transfers;
                    LOG("[stats] %llu vblanks/%.0fs, %llu cmdlists, %llu display transfers, gpu %.1f ms/s\n",
                        (unsigned long long)(frame_count - last_frames), span, (unsigned long long)(lists - last_lists),
                        (unsigned long long)(xf - last_xfers), (ns - last_ns) / (span * 1e6));
                    last_xfers = xf;
                    last_frames = frame_count;
                    last_lists = lists;
                    last_ns = ns;
                }
            }
        }
    }
}
