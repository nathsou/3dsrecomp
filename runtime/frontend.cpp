// The host side: an SDL3 window showing both screens, keyboard/gamepad input, VBlank pacing, and the GPU
// (GX commands run on this thread). Audio frames are ticked from their own thread.
#include "frontend.h"

#include "cloud.h"

#include <SDL3/SDL.h>
#include "platform.h"
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
#include "onescreen.h"
#include "profile.h"
#include "menu.h"
#include "services/services.h"
#include "settings.h"
#include "stacks.h"
#include "touch.h"
#include "uiview.h"

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
    SDL_FRect area{};  // single-screen: where the screen fits (w 0: the whole canvas)
};
static const ScreenLayout kBaseLayouts[] = {
    {"Stacked", 400, 480, {0, 0, 400, 240}, {40, 240, 320, 240}},
    {"Top screen only", 400, 240, {0, 0, 400, 240}, {}},
    {"Side by side", 720, 240, {0, 0, 400, 240}, {400, 0, 320, 240}},
    {"Large top, small bottom", 560, 240, {0, 0, 400, 240}, {400, 120, 160, 120}},
    {"Bottom screen only", 320, 240, {}, {0, 0, 320, 240}},
    {"Stacked, bottom first", 400, 480, {0, 240, 400, 240}, {40, 0, 320, 240}},
    // with touch controls only: reshaped to the window by update_touch_layout()
    {"Fitted around the touch controls", 400, 480, {0, 0, 400, 240}, {40, 240, 320, 240}},
};
constexpr int kLayouts = sizeof kBaseLayouts / sizeof kBaseLayouts[0];
// as shown: with touch controls, each one reshaped to the window by update_touch_layout()
static ScreenLayout layouts[kLayouts] = {kBaseLayouts[0], kBaseLayouts[1], kBaseLayouts[2], kBaseLayouts[3],
                                         kBaseLayouts[4], kBaseLayouts[5], kBaseLayouts[6]};
constexpr int kTouchLayout = kLayouts - 1;
static int layout;
static bool touch_single;  // the touch layout shows one screen at a time (a wide window)
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
// the bottom screen's picture: how much of it the game drew (alpha of its render target, sampled on a grid);
// with its background hidden by the profile ("set"), what is left floats over the top screen
static float bottom_cover = 1, bottom_opaque = 1;  // fractions of samples with alpha > 1/8, > 15/16
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
    if (from == to || touch::enabled() || (SDL_GetWindowFlags(window) & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_MAXIMIZED)))
        return;  // (the touch layouts are window-shaped)
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
int frontend_layout_count() { return touch::enabled() ? kLayouts : kLayouts - 1; }
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

static bool bottom_at(float wx, float wy, int &tx, int &ty);
static int pip_finger_at(float wx, float wy);
static void pip_finger_gesture(int mode, int phase, float x0, float y0, float x1, float y1);
static std::atomic<bool> in_background;  // (iOS) the app is not on screen: nothing runs or draws

#if TARGET_OS_IOS
// Leaving the screen: the menu opens, which pauses the game and saves the settings, so the player comes back to a
// paused game; nothing is drawn meanwhile (iOS refuses GPU work from the background). SDL calls this on the main
// thread while the main loop polls events.
static bool SDLCALL app_events(void *, SDL_Event *ev) {
    if (ev->type == SDL_EVENT_WILL_ENTER_BACKGROUND) {
        if (!menu::is_open() && !menu::keyboard_open()) menu::toggle();
        paused = true;
        in_background = true;
    }
    if (ev->type == SDL_EVENT_DID_ENTER_FOREGROUND) in_background = false;
    return true;
}
#endif

void frontend_init() {
    if (!TARGET_OS_IOS && touch::enabled()) SDL_SetHint(SDL_HINT_MOUSE_TOUCH_EVENTS, "1");  // the mouse is a finger
#if TARGET_OS_IOS
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight Portrait PortraitUpsideDown");
    SDL_SetHint(SDL_HINT_IOS_HIDE_HOME_INDICATOR, "2");  // dimmed; a swipe from the bottom edge needs two goes
#endif
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) FATAL("SDL_Init: %s\n", SDL_GetError());
#if TARGET_OS_IOS
    SDL_AddEventWatch(app_events, nullptr);
#endif
    {  // a gamepad already connected (its "added" event may have gone to the game list of a multi-game app)
        int n = 0;
        SDL_JoystickID *ids = SDL_GetGamepads(&n);
        if (n > 0) pad = SDL_OpenGamepad(ids[0]);
        SDL_free(ids);
    }
    if (const char *s = getenv("RECOMP_SCALE")) scale = atoi(s);
    layout = std::clamp(g_settings.layout, 0, frontend_layout_count() - 1);
    if (getenv("RECOMP_TOUCH")) layout = kTouchLayout;  // (trying the touch controls on a Mac)
    window = SDL_CreateWindow(g_game.name.c_str(), layouts[layout].w * scale, layouts[layout].h * scale,
                              SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY | (TARGET_OS_IOS ? SDL_WINDOW_FULLSCREEN : 0));
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
    touch::init(window, renderer);
    touch::set_bottom_screen_mapper(bottom_at);
    touch::set_pip_hooks(pip_finger_at, pip_finger_gesture);
    if (g_settings.split) frontend_set_split(true);
    profile::load();
    stacks::init();
}

// ---- what the in-game menu changes (runtime/menu.mm)
static void update_touch_layout();
void frontend_set_layout(int l) {
    const int from = shown_layout();
    layout = std::clamp(l, 0, frontend_layout_count() - 1);
    apply_layout();
    update_touch_layout();
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
extern std::string g_start_game;  // main.cpp: (multi-game app) the game playing
// starts the app again (an iOS app cannot: it closes, saying so, and the player opens it again)
[[noreturn]] static void relaunch(const char *ios_message) {
    store_geometry(window);
    store_geometry(window2);
    settings_save();
    cloud_quit();  // (the save, if it changed, uploaded first)
#if TARGET_OS_IOS
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, g_game.title.c_str(), ios_message, window);
    _exit(0);
#endif
    LOG("frontend: restarting\n");
    unsetenv("RECOMP_INPUT");  // a test script runs once
    if (const std::string exe = platform::exe_path(); !exe.empty()) execv(exe.c_str(), g_argv);
    FATAL("cannot restart: %s\n", strerror(errno));
}
void frontend_restart() {
    if (!g_start_game.empty()) setenv("RECOMP_START_GAME", g_start_game.c_str(), 1);  // the same game, not the list
    relaunch("Open the app again to restart the game.");
}
void frontend_choose_game() {
    unsetenv("RECOMP_START_GAME");
    relaunch("Open the app again to choose another game.");
}
void frontend_quit() {
    store_geometry(window);
    store_geometry(window2);
    settings_save();
    cloud_quit();
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

// The bottom screen's picture, sampled on a grid: how much the game drew (alpha: with its background hidden by the
// profile, the rest is transparent). In one-screen play black is keyed out (a screen left black shows nothing, a
// dialog's choices on it float): by the GPU in its display images (gpu::set_key_black_bottom; cheap, so the UI
// thread keeps its timing), here for the guest framebuffer's 320x240 picture. Returns the picture to show.
// In one-screen play a picture that suddenly covers the whole screen for a few frames is a page's transition (Animal
// Crossing wipes its bottom screen with its menus' pattern as they open and close): it is not shown, the last
// picture stays (nullptr: keep the texture as it is). One that stays is a page of its own.
static const u32 *bottom_stats(const u32 *img, int w, int h) {
    int cover = 0, opaque = 0, dark = 0, shade = 0, n = 0;
    for (int y = h / 60; y < h; y += h / 30)
        for (int x = w / 80; x < w; x += w / 40, n++) {
            const u32 p = img[(size_t)y * w + x], a = p >> 24;
            const u32 m = std::max({p & 255, p >> 8 & 255, p >> 16 & 255});
            cover += a > 32;
            opaque += a > 240;
            dark += a > 32 && m < 12;
            shade += a > 240 && m < 100;
        }
    // the share of dark opaque samples (percent), for profiles: "value <name> = runtime bottom_dark" (a dialog's
    // list of choices is a dark box on a light page)
    profile::set_runtime("bottom_dark", n ? shade * 100 / n : 0);
    constexpr int kWipeFrames = 14;  // (Animal Crossing's last 8)
    static int full_run;             // pictures in a row covering the whole screen
    static bool seen_partial;        // one that did not since one-screen play began
    if (n && cover >= n * 98 / 100) {
        full_run++;
    } else {
        full_run = 0;
        seen_partial = true;
    }
    if (onescreen::active() && seen_partial && full_run > 0 && full_run < kWipeFrames) return nullptr;
    static std::vector<u32> keyed;
    if (onescreen::active() && n && dark > n * 6 / 10 && w <= 320) {  // (display images: keyed by the GPU)
        keyed.assign(img, img + (size_t)w * h);
        for (u32 &p : keyed) {
            const int m = (int)std::max({p & 255, p >> 8 & 255, p >> 16 & 255});
            const u32 a = (u32)std::clamp((m - 3) * 64, 0, 255) * (p >> 24) / 255;  // (as the GPU's: only near black)
            p = (p & 0xffffff) | a << 24;
        }
        img = keyed.data();
        cover -= dark;
        opaque = 0;
    }
    bottom_cover = n ? (float)cover / n : 1;
    bottom_opaque = n ? (float)opaque / n : 1;
    if (onescreen::active()) onescreen::bottom_image(img, w, h);
    return img;
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
            SDL_SetTextureBlendMode(hires, SDL_BLENDMODE_NONE);  // (the image's alpha is the render target's)
            hw = w;
            hh = h;
        }
        if (screen == 1) img = bottom_stats(img, w, h);
        if (img) SDL_UpdateTexture(hires, nullptr, img, w * 4);
    });
    static u32 last_fb;
    if (screen == 0 && (ver ? ver != shown[0] : f.addr_left != last_fb)) game_frames++;  // a new image: the game's fps
    if (screen == 0) last_fb = f.addr_left;
    shown[screen] = ver;
    if (ver) return hires;
    convert_fb(f, width, buf);
    const u32 *pic = screen == 1 ? bottom_stats(buf, width, 240) : buf;  // (opaque: no alpha in guest memory)
    if (pic) SDL_UpdateTexture(native, nullptr, pic, width * 4);
    return native;
}

// RECOMP_WINSHOT="f1,f2,...": the whole window (menu included) as <prefix>_<frame>.ppm, for tests
std::atomic<int> g_winshot_request;  // RECOMP_LIVE's WINSHOT: the window at the next present
static void write_window_shot(SDL_Renderer *r, const char *prefix) {
    static const char *winshot = getenv("RECOMP_WINSHOT");
    const bool asked = r == renderer && g_winshot_request.exchange(0);
    if (!asked && (!winshot || (std::string(",") + winshot + ",").find("," + std::to_string(frame_count + 1) + ",") == std::string::npos))
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
    bool panel = false;         // the bottom screen is a panel over the top one (profile "focus panel")
    bool card = false;          // ... drawn as a card (it is opaque), else floating on its own alpha
};
static float panel_anim;  // 0..1: the panel coming in / going out
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
static std::mutex live_text_m;
static std::string live_text;  // RECOMP_LIVE's TEXT, typed at the next frame
void frontend_live_text(const std::string &s) {
    std::lock_guard<std::mutex> lk(live_text_m);
    live_text += s;
}
static std::vector<std::string> live_keys;  // RECOMP_LIVE's KEY, pressed at the next frame
void frontend_live_key(const std::string &name) {
    std::lock_guard<std::mutex> lk(live_text_m);
    live_keys.push_back(name);
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

bool frontend_typing() { return typing; }
static bool single_layout() { return !window2 && (layout == 1 || layout == 4 || (layout == kTouchLayout && touch_single)); }

// With touch controls every layout is a canvas of the window's shape with the screens in the part the controls
// leave. The touch layout puts them one above the other unless that makes them much smaller than one screen alone
// (a phone held sideways), in which case one shows at a time like the single-screen layouts; the others keep their
// own arrangement, as large as it fits.
static void update_touch_layout() {
    if (!touch::enabled()) return;
    touch::set_swap_shown(single_layout());
    if (window2) return;
    int ww, wh;
    SDL_GetWindowSize(window, &ww, &wh);
    const SDL_FRect a = touch::screen_area();
    if (ww <= 0 || wh <= 0 || a.w <= 0 || a.h <= 0) return;
    ScreenLayout l = kBaseLayouts[layout];
    bool single = touch_single;
    float k;
    if (layout == kTouchLayout) {
        const float k2 = std::min(a.w / 400, a.h / 480), k1 = std::min(a.w / 400, a.h / 240);
        single = k2 < 0.66f * k1;
        k = single ? k1 : k2;
        l.h = single ? 240 : 480;  // (the arrangement's size, for now)
    } else {
        k = std::min(a.w / l.w, a.h / l.h);
    }
    // the arrangement in the middle of the room it has
    const float bx = (a.x + (a.w - l.w * k) / 2) / k, by = (a.y + (a.h - l.h * k) / 2) / k;
    auto moved = [&](SDL_FRect r) { return r.w ? SDL_FRect{r.x + bx, r.y + by, r.w, r.h} : r; };
    l.top = moved(l.top);
    l.bottom = moved(l.bottom);
    if (layout == kTouchLayout) l.area = single ? SDL_FRect{a.x / k, a.y / k, a.w / k, a.h / k} : SDL_FRect{};
    else if (layout == 1 || layout == 4) l.area = {bx, by, (float)l.w, (float)l.h};  // (the corner screen stays in it)
    l.w = (int)std::lround(ww / k);
    l.h = (int)std::lround(wh / k);
    const ScreenLayout &o = layouts[layout];
    if (l.w == o.w && l.h == o.h && SDL_RectsEqualFloat(&l.top, &o.top) && SDL_RectsEqualFloat(&l.bottom, &o.bottom) &&
        SDL_RectsEqualFloat(&l.area, &o.area) && single == touch_single)
        return;
    layouts[layout] = l;
    touch_single = single;
    apply_layout();
}
static profile::Focus auto_focus() { return g_settings.auto_focus ? profile::focus() : profile::FOCUS_NONE; }

// 0 top, 1 bottom
static int focused_screen() {
    const profile::Focus f = auto_focus();
    if (f != flip_for) {
        manual_flip = 0;
        flip_for = f;
    }
    int s = layout == 4 ? 1 : 0;
    if (f == profile::FOCUS_TOP || f == profile::FOCUS_BOTH || f == profile::FOCUS_PANEL) s = 0;
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

// where the single-screen layouts fit their screen and put the corner screen (canvas units)
static SDL_FRect screen_area(const ScreenLayout &l) { return l.area.w ? l.area : SDL_FRect{0, 0, (float)l.w, (float)l.h}; }

static void place_screens() {
    const ScreenLayout &l = layouts[shown_layout()];
    placement = {l.top, l.bottom, -1, 1};
    profile::set_runtime("single", single_layout());
    static const bool debug = getenv("RECOMP_PROFILE_DEBUG");  // (a "debug" runtime flag for writing profiles)
    profile::set_runtime("debug", debug);
    if (!single_layout()) {
        panel_anim = 0;
        return;
    }
    const SDL_FRect area = screen_area(l);
    auto fit = [&](int screen) {
        const float sw = screen ? 320 : 400, k = std::min(area.w / sw, area.h / 240.0f);
        return SDL_FRect{area.x + (area.w - sw * k) / 2, area.y + (area.h - 240 * k) / 2, sw * k, 240 * k};
    };
    // the profile's panel: the bottom screen over the top one, when it shows something
    // (wanted a few frames in a row first: the picture of a page that is still coming up is not shown)
    constexpr int kPanelDelay = 4;
    static int wanted;
    wanted = auto_focus() == profile::FOCUS_PANEL && !manual_flip && bottom_cover > 0.01f && !onescreen::busy()
                 ? wanted + 1
                 : 0;
    const bool want_panel = wanted >= kPanelDelay || (panel_anim > 0 && wanted > 0);
    panel_anim = std::clamp(panel_anim + (want_panel ? 0.2f : -0.25f), 0.0f, 1.0f);
    if (auto_focus() == profile::FOCUS_PANEL || (profile::onescreen() && !manual_flip)) {
        placement.top = fit(0);
        placement.bottom = {};
        pip_rect = pip_grip = pip_handle = {};
        if (panel_anim <= 0) return;
        const SDL_FRect t = placement.top;
        const float k = t.h / 240;  // render units per top-screen pixel
        placement.card = bottom_opaque > 0.9f;
        const float e = 1 - (1 - panel_anim) * (1 - panel_anim);  // ease out
        // a card a bit smaller than the screen's height; floating content at its own size, where it would be
        // on the 3DS (centred)
        const float ph = placement.card ? 240 * k * 0.92f : 240 * k, pw = ph * 320 / 240;
        placement.bottom = {t.x + (t.w - pw) / 2, t.y + (t.h - ph) / 2 + (1 - e) * 14 * k, pw, ph};
        placement.panel = true;
        placement.alpha = e;
        return;
    }
    panel_anim = 0;
    const int main = focused_screen(), other = main ^ 1;
    (main ? placement.bottom : placement.top) = fit(main);
    (other ? placement.bottom : placement.top) = {};
    // the corner screen
    const Uint64 now = SDL_GetTicks();
    if (g_settings.pip == 1 && screen_changed(other)) pip_active_until = now + 3000;
    const float ph = std::min(area.h * g_settings.pip_size / 100.0f, area.w * 240.0f / (other ? 320 : 400)),
                pw = ph * (other ? 320 : 400) / 240.0f, m = area.h * 0.025f;
    const int c = g_settings.pip_corner;
    SDL_FRect r{area.x + (c == 1 || c == 3 ? m : area.w - pw - m), area.y + (c >= 2 ? area.h - ph - m : m), pw, ph};
    if (g_settings.pip_x >= 0) {
        r.x = area.x + std::clamp(g_settings.pip_x * area.w / 1000.0f, 0.0f, area.w - pw);
        r.y = area.y + std::clamp(g_settings.pip_y * area.h / 1000.0f, 0.0f, area.h - ph);
    }
    // the grip tab and the resize handle go on the side facing the middle of the window
    const bool below = r.y + r.h / 2 < l.h / 2, left = r.x + r.w / 2 > l.w / 2;
    float gh = std::max(l.h * 0.05f, 6.0f);
    if (touch::enabled())  // a fingertip: 34 points
        gh = std::max(gh, 34 * SDL_GetWindowPixelDensity(window) / std::max(present_factor(), 0.01f));
    const float gw = std::min(r.w * 0.4f, gh * 5);
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

// the corner screen moved (its top left) or resized (its height), kept in the area; stored relative to the area
static void store_pip(float x, float y, float h) {
    const SDL_FRect a = screen_area(layouts[shown_layout()]);
    const float aspect = pip_rect.h > 0 ? pip_rect.w / pip_rect.h : 4.0f / 3;
    h = std::min(std::clamp(h, a.h * 0.15f, a.h * 0.8f), a.w / aspect);
    const float w = h * aspect;
    g_settings.pip_x = (int)std::lround(std::clamp(x - a.x, 0.0f, a.w - w) * 1000 / a.w);
    g_settings.pip_y = (int)std::lround(std::clamp(y - a.y, 0.0f, a.h - h) * 1000 / a.h);
    g_settings.pip_size = std::clamp((int)std::lround(h * 100 / a.h), 15, 80);
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
            store_pip(lx - drag_dx, ly - drag_dy, pip_rect.h);
        } else {
            const float h = std::clamp(std::max(std::fabs(ly - drag_ay), std::fabs(lx - drag_ax) / aspect), l.h * 0.15f, l.h * 0.8f);
            store_pip(lx < drag_ax ? drag_ax - h * aspect : drag_ax, ly < drag_ay ? drag_ay - h : drag_ay, h);
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
        store_pip(pip_rect.x + (pip_rect.w - h * aspect) / 2, pip_rect.y + (pip_rect.h - h) / 2, h);
        settings_save();
        return true;
    }
    }
    return false;
}

// Fingers on the corner screen (touch.h): what a window point is on (0 nothing, 1 its picture of the bottom
// screen, 2 what moves it, 3 its resize handle) ...
static int pip_finger_at(float wx, float wy) {
    if (window2 || placement.pip < 0 || placement.alpha <= 0.5f) return 0;
    float lx, ly;
    SDL_RenderCoordinatesFromWindow(renderer, wx, wy, &lx, &ly);
    auto in = [&](const SDL_FRect &q) { return q.w && lx >= q.x && lx < q.x + q.w && ly >= q.y && ly < q.y + q.h; };
    if (in(pip_grip)) return 2;
    if (in(pip_handle)) return 3;
    if (in(pip_rect)) return placement.pip == 0 ? 2 : 1;  // (the top screen takes no touches: its picture moves it)
    return 0;
}
// ... and moving it (mode 1), resizing it from its handle (2) or pinching it (3: the point between the two fingers
// stays put in it while it scales with their distance)
static void pip_finger_gesture(int mode, int phase, float x0, float y0, float x1, float y1) {
    static float grab_x, grab_y, dist0, h0, fx, fy;
    float ax, ay, bx, by;
    SDL_RenderCoordinatesFromWindow(renderer, x0, y0, &ax, &ay);
    SDL_RenderCoordinatesFromWindow(renderer, x1, y1, &bx, &by);
    const float aspect = pip_rect.h > 0 ? pip_rect.w / pip_rect.h : 4.0f / 3;
    if (phase == 2) {
        pip_drag = 0;
        settings_save();
        return;
    }
    if (phase == 0) {
        pip_drag = mode;
        grab_x = ax - pip_rect.x;
        grab_y = ay - pip_rect.y;
        drag_ax = pip_handle.x < pip_rect.x ? pip_rect.x + pip_rect.w : pip_rect.x;  // the corner a resize keeps
        drag_ay = pip_handle.y < pip_rect.y ? pip_rect.y + pip_rect.h : pip_rect.y;
        dist0 = std::max(std::hypot(bx - ax, by - ay), 1.0f);
        h0 = pip_rect.h;
        fx = ((ax + bx) / 2 - pip_rect.x) / pip_rect.w;
        fy = ((ay + by) / 2 - pip_rect.y) / pip_rect.h;
        return;
    }
    if (mode == 1) {
        store_pip(ax - grab_x, ay - grab_y, pip_rect.h);
    } else if (mode == 2) {
        const float h = std::max(std::fabs(ay - drag_ay), std::fabs(ax - drag_ax) / aspect);
        store_pip(ax < drag_ax ? drag_ax - h * aspect : drag_ax, ay < drag_ay ? drag_ay - h : drag_ay, h);
    } else {
        const float h = h0 * std::hypot(bx - ax, by - ay) / dist0;
        store_pip((ax + bx) / 2 - fx * h * aspect, (ay + by) / 2 - fy * h, h);
    }
}

// where a point of the main window lands on the bottom screen (when it shows), in its pixels
static bool bottom_at(float wx, float wy, int &tx, int &ty) {
    float lx, ly;
    SDL_RenderCoordinatesFromWindow(renderer, wx, wy, &lx, &ly);
    const SDL_FRect &b = placement.bottom;
    if (!b.w || ((placement.pip == 1 || placement.panel) && placement.alpha <= 0.5f) || lx < b.x || lx >= b.x + b.w || ly < b.y || ly >= b.y + b.h)
        return false;
    tx = std::clamp((int)((lx - b.x) * 320 / b.w), 0, 319);
    ty = std::clamp((int)((ly - b.y) * 240 / b.h), 0, 239);
    return true;
}

// The profile's panel: the dimmed top screen with the bottom screen over it, as a card with rounded corners and a
// soft shadow when it is opaque, else floating on its own alpha (its background hidden by the profile).
static void rounded_rect(SDL_Renderer *r, SDL_Texture *t, const SDL_FRect &d, float rad, SDL_FColor c) {
    constexpr int kSeg = 6;  // segments per corner
    std::vector<SDL_Vertex> v;
    std::vector<int> idx;
    const float cx = d.x + d.w / 2, cy = d.y + d.h / 2;
    auto vert = [&](float x, float y) {
        SDL_Vertex q;
        q.position = {x, y};
        q.color = c;
        q.tex_coord = {(x - d.x) / d.w, (y - d.y) / d.h};
        v.push_back(q);
    };
    vert(cx, cy);
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

static void draw_panel(SDL_Renderer *r, SDL_Texture *t, const SDL_FRect &top, const SDL_FRect &d, bool card,
                       float a, float f) {
    if (!t || !d.w) return;
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    const float k = top.h / 240;
    // the world dims behind it: fully for a card, as much as floating content covers (a menu more than a
    // speech bubble)
    SDL_SetRenderDrawColorFloat(r, 0, 0, 0, (card ? 0.38f : 0.45f * std::min(1.0f, bottom_cover * 1.6f)) * a);
    SDL_RenderFillRect(r, &top);
    if (card) {
        for (int i = 3; i >= 1; i--) {  // a soft shadow
            const float g = i * 2.5f * k;
            rounded_rect(r, nullptr, {d.x - g, d.y - g + 3 * k, d.w + 2 * g, d.h + 2 * g}, 12 * k + g,
                         {0, 0, 0, 0.12f * a});
        }
        rounded_rect(r, nullptr, {d.x - 1.5f * k, d.y - 1.5f * k, d.w + 3 * k, d.h + 3 * k}, 13.5f * k,
                     {1, 1, 1, 0.9f * a});
    }
    set_filter(t, d.w * f, t->w);
    // (a card is opaque: its pixels' alpha says nothing once it is in, but fades it in and out)
    SDL_SetTextureBlendMode(t, card && a >= 1 ? SDL_BLENDMODE_NONE : SDL_BLENDMODE_BLEND);
    SDL_SetTextureAlphaModFloat(t, a);
    if (card) rounded_rect(r, t, d, 12 * k, {1, 1, 1, 1});
    else SDL_RenderTexture(r, t, nullptr, &d);
    SDL_SetTextureAlphaModFloat(t, 1);
    SDL_SetTextureBlendMode(t, SDL_BLENDMODE_NONE);
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
        stacks::draw(renderer2, full, r.w > 0 ? r.w / 320 : 1, 1);
        write_window_shot(renderer2, "window2");
        SDL_RenderPresent(renderer2);
    }
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    place_screens();
    const Placement &pl = placement;
    const float f = present_factor();
    gpu::set_key_black_bottom(onescreen::active());
    onescreen::set_state(single_layout() && profile::onescreen() && !window2, pl.top,
                         pl.panel && pl.alpha > 0.5f ? pl.bottom : SDL_FRect{});
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
    if (pl.panel) {
        draw(0);
        draw_panel(renderer, t_bottom, pl.top, pl.bottom, pl.card, pl.alpha, f);
        stacks::draw(renderer, pl.bottom, f, pl.alpha);
    } else {
        // the corner screen goes over the main one
        draw(pl.pip == 0 ? 1 : 0);
        if (pl.pip == 0) stacks::draw(renderer, pl.bottom, f, pl.alpha);
        draw(pl.pip == 0 ? 0 : 1);
        if (pl.pip != 0) stacks::draw(renderer, pl.bottom, f, pl.pip == 1 ? pl.alpha : 1);
    }
    if (pl.pip >= 0 && (pip_hover || (touch::enabled() && pl.alpha > 0.5f))) {  // grip tab (three bars), resize handle (a diagonal)
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
    onescreen::draw(renderer, t_bottom, f);
    if (touch::enabled() && !menu::is_open() && !menu::keyboard_open()) touch::draw();
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
        f = kernel::gsp_framebuffer(1);
        if (getenv("RECOMP_SHOT_ALPHA") && f.valid &&
            gpu::get_display_image_for(mem::vaddr_to_paddr(f.addr_left), f.stride, 320, &img, &w, &h)) {
            std::vector<u32> a(img, img + w * h);
            for (u32 &p : a) p = (p >> 24) * 0x010101u;
            snprintf(name, sizeof name, "shot_bottom_alpha_%06llu.ppm", (unsigned long long)frame_count);
            write_ppm(name, a.data(), w, h);
        }
    }
}

// scripted input for testing: RECOMP_INPUT="frame:BUTTONS:duration,..." (BUTTONS like A+START; T@x/y touches
// the bottom screen at pixel x, y)
struct ScriptEntry { u64 frame, dur; u32 buttons; int cx, cy, tx = -1, ty = -1; };
// host keys for testing the menu: "frame:KEY_<SDL key name>" (KEY_Escape, KEY_Down, KEY_Return...), typed text
// ("frame:TEXT_<text>", for the software keyboard, or taps on the game's own keyboard from the profile's keys), the mouse in layout coordinates: "frame:MOUSE_DOWN@x/y"
// (also MOUSE_MOVE, MOUSE_UP) and "frame:MOUSE_WHEEL@x/y/dy", and fingers in window points for the touch controls:
// "frame:TOUCH_DOWN@x/y/finger" (also TOUCH_MOVE, TOUCH_UP; finger 0 when left out)
struct ScriptKey {
    u64 frame;
    SDL_Scancode key;
    SDL_EventType mouse = (SDL_EventType)0;
    float x = 0, y = 0, wheel = 0;  // (wheel: the finger for TOUCH_)
    std::string text;
    SDL_EventType finger = (SDL_EventType)0;
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
        if (n.rfind("TOUCH_", 0) == 0) {
            ScriptKey k{f, SDL_SCANCODE_UNKNOWN};
            const size_t at = n.find('@');
            const std::string kind = n.substr(6, at == std::string::npos ? std::string::npos : at - 6);
            k.finger = kind == "DOWN" ? SDL_EVENT_FINGER_DOWN : kind == "UP" ? SDL_EVENT_FINGER_UP : SDL_EVENT_FINGER_MOTION;
            if (at != std::string::npos) sscanf(n.c_str() + at + 1, "%f/%f/%f", &k.x, &k.y, &k.wheel);
            script_keys.push_back(k);
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

// One-screen play: the host's inputs by position (onescreen.h), the keyboard's keys standing for the gamepad's as
// in the controls table, and the right stick (-1..1, up positive); RECOMP_LIVE's PAD commands add to them
std::atomic<u32> g_live_host;
static u32 host_input(float &rsx, float &rsy) {
    using namespace onescreen;
    const bool *k = SDL_GetKeyboardState(nullptr);
    static const bool none[SDL_SCANCODE_COUNT] = {};
    const bool *kb = typing ? none : k;
    u32 h = g_live_host;
    const std::pair<SDL_Scancode, u32> keys[] = {
        {SDL_SCANCODE_X, H_A}, {SDL_SCANCODE_Z, H_B}, {SDL_SCANCODE_S, H_X}, {SDL_SCANCODE_A, H_Y},
        {SDL_SCANCODE_Q, H_L}, {SDL_SCANCODE_W, H_R}, {SDL_SCANCODE_1, H_ZL}, {SDL_SCANCODE_2, H_ZR},
        {SDL_SCANCODE_RETURN, H_START}, {SDL_SCANCODE_BACKSPACE, H_BACK}, {SDL_SCANCODE_I, H_DUP},
        {SDL_SCANCODE_K, H_DDOWN}, {SDL_SCANCODE_J, H_DLEFT}, {SDL_SCANCODE_L, H_DRIGHT}};
    for (auto &[sc, bit] : keys)
        if (kb[sc]) h |= bit;
    rsx = rsy = 0;
    if (h & H_RSUP) rsy = 1;
    if (h & H_RSDOWN) rsy = -1;
    if (h & H_RSLEFT) rsx = -1;
    if (h & H_RSRIGHT) rsx = 1;
    if (!pad) return h;
    const std::pair<SDL_GamepadButton, u32> btns[] = {
        {SDL_GAMEPAD_BUTTON_EAST, H_A}, {SDL_GAMEPAD_BUTTON_SOUTH, H_B}, {SDL_GAMEPAD_BUTTON_NORTH, H_X},
        {SDL_GAMEPAD_BUTTON_WEST, H_Y}, {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, H_L},
        {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, H_R}, {SDL_GAMEPAD_BUTTON_START, H_START},
        {SDL_GAMEPAD_BUTTON_BACK, H_BACK}, {SDL_GAMEPAD_BUTTON_DPAD_UP, H_DUP},
        {SDL_GAMEPAD_BUTTON_DPAD_DOWN, H_DDOWN}, {SDL_GAMEPAD_BUTTON_DPAD_LEFT, H_DLEFT},
        {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, H_DRIGHT}};
    for (auto &[b, bit] : btns)
        if (SDL_GetGamepadButton(pad, b)) h |= bit;
    if (SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 16000) h |= H_ZL;
    if (SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 16000) h |= H_ZR;
    const float x = SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f,
                y = -SDL_GetGamepadAxis(pad, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f;
    if (std::fabs(x) > std::fabs(rsx)) rsx = x;
    if (std::fabs(y) > std::fabs(rsy)) rsy = y;
    if (rsy > 0.5f) h |= H_RSUP;
    if (rsy < -0.5f) h |= H_RSDOWN;
    if (rsx < -0.5f) h |= H_RSLEFT;
    if (rsx > 0.5f) h |= H_RSRIGHT;
    return h;
}

static void poll_input_devices(u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty);
// sx, sy: the right stick as the second circle pad (Circle Pad Pro), unless one-screen play uses it
static void poll_input(u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty, s16 &sx, s16 &sy) {
    poll_input_devices(buttons, cx, cy, touch, tx, ty);
    float rsx, rsy;
    const u32 host = host_input(rsx, rsy);
    constexpr float CSTICK_RADIUS = 0x7FF;
    const bool own = onescreen::active();
    sx = own ? 0 : (s16)(rsx * CSTICK_RADIUS);
    sy = own ? 0 : (s16)(rsy * CSTICK_RADIUS);
    if (const u32 live = g_live_host; live) {  // (scripted host input: its default buttons too)
        static const u32 defaults[][2] = {
            {onescreen::H_A, BTN_A}, {onescreen::H_B, BTN_B}, {onescreen::H_X, BTN_X}, {onescreen::H_Y, BTN_Y},
            {onescreen::H_L, BTN_L}, {onescreen::H_R, BTN_R}, {onescreen::H_ZL, BTN_ZL}, {onescreen::H_ZR, BTN_ZR},
            {onescreen::H_START, BTN_START}, {onescreen::H_BACK, BTN_SELECT}, {onescreen::H_DUP, BTN_UP},
            {onescreen::H_DDOWN, BTN_DOWN}, {onescreen::H_DLEFT, BTN_LEFT}, {onescreen::H_DRIGHT, BTN_RIGHT}};
        for (auto &d : defaults)
            if (live & d[0]) buttons |= d[1];
    }
    onescreen::input(host, rsx, rsy, buttons, cx, cy, touch, tx, ty);
}

static void poll_input_devices(u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty) {
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
    if (!touch && !key_taps.empty()) {  // typed keys: each tap held 8 frames, then 8 released (the game reads the touch
        const u64 t = frame_count - key_tap_start;  // screen at 30 fps, less when busy: shorter taps were lost now and then)
        if (t < 8) {
            touch = true;
            tx = (u16)std::clamp(key_taps.front().first, 0, 319);
            ty = (u16)std::clamp(key_taps.front().second, 0, 239);
        } else if (t >= 16) {
            key_taps.pop_front();
            key_tap_start = frame_count;
        }
    }
    if (!touch && frame_count < hud_tap_until) {  // the HUD button (profile: "button tap x y")
        touch = true;
        tx = (u16)std::clamp(hud_tap_x, 0, 319);
        ty = (u16)std::clamp(hud_tap_y, 0, 239);
    }
    if (touch::enabled()) {  // fingers: the on-screen controls and the bottom screen (not the mouse)
        touch::poll(buttons, cx, cy, touch, tx, ty);
        return;
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
        int x, y;  // touch: the mouse on the bottom screen, wherever it is
        if (bottom_at(mx, my, x, y)) {
            touch = true;
            tx = (u16)x;
            ty = (u16)y;
        }
    }
}

// Tab, a right stick click or the swap button: the other screen (for now, when the game's profile picks it)
static void swap_screens() {
    if (window2) return;
    if (onescreen::active()) {  // one-screen play: the bottom screen's buttons are in the hub
        onescreen::toggle_hub();
        return;
    }
    if (single_layout() && (auto_focus() != profile::FOCUS_NONE || swapped_layout(layout) == layout)) {
        manual_flip ^= 1;  // the profile picks the screen, or no layout has the other one: look at it for now
        pip_alpha = 0;
    } else if (const int l = swapped_layout(layout); l != layout) {
        g_settings.layout = l;
        frontend_set_layout(l);
        settings_save();
    }
}

// GX commands run on their own thread so a slow frame never delays VBlank or presentation

static void gpu_thread() {
    platform::set_thread_interactive();
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
struct InputSnapshot { u32 buttons; s16 cx, cy; bool touch; u16 tx, ty; s16 sx, sy; };
static std::mutex input_m;
static InputSnapshot input_now{};

static void hid_thread() {
    platform::set_thread_interactive();
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
        kernel::hid_update(s.buttons, s.cx, s.cy, s.touch, s.tx, s.ty, s.sx, s.sy);
    }
}

static void audio_thread() {
    platform::set_thread_interactive();
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
        {
            std::lock_guard<std::mutex> lk(live_text_m);
            if (!live_text.empty()) type_text(live_text);  // (RECOMP_LIVE's TEXT: the game's own keyboard)
            live_text.clear();
            for (auto &n : live_keys) {
                SDL_Event ke{};
                ke.type = SDL_EVENT_KEY_DOWN;
                ke.key.scancode = SDL_GetScancodeFromName(n.c_str());
                ke.key.down = true;
                SDL_PushEvent(&ke);
            }
            live_keys.clear();
        }
        for (auto &k : script_keys)
            if (k.frame == frame_count && k.finger) {
                int ww, wh;
                SDL_GetWindowSize(window, &ww, &wh);
                SDL_Event fe{};
                fe.type = k.finger;
                fe.tfinger.touchID = 1;
                fe.tfinger.fingerID = (SDL_FingerID)k.wheel + 1;
                fe.tfinger.x = k.x / ww;
                fe.tfinger.y = k.y / wh;
                fe.tfinger.windowID = SDL_GetWindowID(window);
                SDL_PushEvent(&fe);
            } else if (k.frame == frame_count && k.mouse) {
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
            int ex, ey;  // Esc while the game's own keyboard is up: its close key, when the profile names one
            if (typing && ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_ESCAPE && !menu::is_open() &&
                profile::key("escape", ex, ey)) {
                type_key("escape");
                continue;
            }
            if (menu::handle_event(ev)) continue;
            if (touch::enabled()) {  // fingers; the mouse events iOS makes from them were only for the menu
                if (!menu::is_open() && !menu::keyboard_open() && touch::handle_event(ev)) continue;
                if (((ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN || ev.type == SDL_EVENT_MOUSE_BUTTON_UP) && ev.button.which == SDL_TOUCH_MOUSEID) ||
                    (ev.type == SDL_EVENT_MOUSE_MOTION && ev.motion.which == SDL_TOUCH_MOUSEID))
                    continue;
                if (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN || (ev.type == SDL_EVENT_GAMEPAD_AXIS_MOTION && std::abs(ev.gaxis.value) > 16000))
                    touch::gamepad(true);
            }
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
            if (ev.type == SDL_EVENT_GAMEPAD_REMOVED && pad && SDL_GetGamepadID(pad) == ev.gdevice.which) {
                SDL_CloseGamepad(pad);  // another one still connected takes over; else back to the touch controls
                int n = 0;
                SDL_JoystickID *ids = SDL_GetGamepads(&n);
                pad = n > 0 ? SDL_OpenGamepad(ids[0]) : nullptr;
                SDL_free(ids);
                if (!pad) touch::gamepad(false);
            }
            // Select + Start: the menu, for gamepads whose Home button the system keeps (iOS)
            if (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && pad && !menu::is_open() &&
                ((ev.gbutton.button == SDL_GAMEPAD_BUTTON_START && SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_BACK)) ||
                 (ev.gbutton.button == SDL_GAMEPAD_BUTTON_BACK && SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_START)))) {
                menu::toggle();
                continue;
            }
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
            if ((ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_TAB && !ev.key.repeat) ||
                (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && ev.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK))
                swap_screens();
            if ((ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_E && !ev.key.repeat) ||
                (ev.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && ev.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK))
                if (const profile::Button *b = profile::button()) {  // the HUD button: a tap on the bottom screen
                    hud_tap_x = b->tx;
                    hud_tap_y = b->ty;
                    hud_tap_until = frame_count + 6;
                }
            if (!touch::enabled() && pip_mouse(ev)) continue;
            if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.button == SDL_BUTTON_LEFT &&
                onescreen::overlay_open() && SDL_GetWindowFromID(ev.button.windowID) == window) {
                float lx, ly;  // a click on the one-screen overlays (hub entries; elsewhere closes them)
                SDL_RenderCoordinatesFromWindow(renderer, ev.button.x, ev.button.y, &lx, &ly);
                if (onescreen::click(lx, ly)) continue;
            }
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
        if (touch::take_menu()) menu::toggle();
        if (touch::take_swap()) swap_screens();
        update_auto_scale();
        menu::update();
        if (menu::is_open() && !paused) touch::reset();  // (fingers lifted while it is open are not seen)
        paused = menu::is_open();
        if (in_background) {  // (iOS) until the app is back on screen
            std::this_thread::sleep_for(milliseconds(50));
            next_vblank = steady_clock::now();
            continue;
        }
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
            s16 sx, sy;
            poll_input(b, cx, cy, touch, tx, ty, sx, sy);
            if (paused || menu::keyboard_open()) b = 0, cx = cy = sx = sy = 0, touch = false;
            {
                std::lock_guard<std::mutex> lk(input_m);
                input_now = {b, cx, cy, touch, tx, ty, sx, sy};
            }
            if (!paused) kernel::gsp_vblank();  // the game waits for VBlank, so it stops while the menu is open
            profile::update();
            uiview::keep_filling(renderer);
            update_touch_layout();
            // (on iOS only with a hardware keyboard: the game's own keyboard is tapped on the touch screen, and the
            // system's on-screen one would cover the controls)
            if (const bool want = profile::keyboard_active() && (!TARGET_OS_IOS || SDL_HasKeyboard()); want != typing) {
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
                    static u64 last_games;
                    LOG("[stats] %llu vblanks/%.0fs, %llu game frames, %llu cmdlists, %llu display transfers, gpu %.1f ms/s\n",
                        (unsigned long long)(frame_count - last_frames), span, (unsigned long long)(game_frames - last_games),
                        (unsigned long long)(lists - last_lists), (unsigned long long)(xf - last_xfers),
                        (ns - last_ns) / (span * 1e6));
                    last_games = game_frames;
                    last_xfers = xf;
                    last_frames = frame_count;
                    last_lists = lists;
                    last_ns = ns;
                }
            }
        }
    }
}
