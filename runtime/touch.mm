// On-screen controls for touch screens (touch.h). Their places follow the window's shape: under the screens when
// the window is taller than wide, in a column on each side otherwise. Sizes are in points (a button is about a
// fingertip wide), so they stay the same physical size on a phone and on a tablet.
#include "touch.h"

#include <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "frontend.h"
#include "text.h"

namespace touch {
namespace {

SDL_Window *win;
SDL_Renderer *ren;
bool (*bottom_at)(float, float, int &, int &);
int (*pip_at)(float, float);
void (*pip_gesture)(int, int, float, float, float, float);

enum Kind { NONE, PAD, BUTTON, MENU, SWAP, SCREEN, PIP_MOVE, PIP_SIZE, PIP_PINCH };
struct Control {
    Kind kind;
    uint32_t bit;       // BUTTON: the 3DS button
    const char *label;  // BUTTON
    SDL_FRect r;        // in window points
    bool round;         // a circle (else a pill)
};
std::vector<Control> controls;
float unit;                  // a face button's diameter, in points
float pad_x, pad_y, pad_r;   // the circle pad
SDL_FRect area;              // what the screens get
int laid_w, laid_h;          // the window size the controls were laid out for
SDL_Rect laid_safe;
bool laid_swap, swap_shown, hidden, menu_tapped, swap_tapped;

struct Finger {
    Kind kind = NONE;
    int control = -1;   // the control it went down on
    float x = 0, y = 0;  // window points
    uint32_t bits = 0;   // BUTTON: the buttons under it
    int tx = 0, ty = 0;  // SCREEN: bottom-screen pixels
    bool on_pip = false;  // SCREEN: went down on the corner screen's picture (a second finger there pinches it)
};
std::map<SDL_FingerID, Finger> fingers;

// the corner screen's gesture from where its fingers are now
void pip_update(int phase) {
    const Finger *f[2] = {};
    int n = 0, mode = 0;
    for (const auto &[id, g] : fingers)
        if (g.kind == PIP_MOVE || g.kind == PIP_SIZE || g.kind == PIP_PINCH) {
            if (n < 2) f[n++] = &g;
            mode = g.kind == PIP_MOVE ? 1 : g.kind == PIP_SIZE ? 2 : 3;
        }
    if (!n || !pip_gesture) return;
    pip_gesture(mode, phase, f[0]->x, f[0]->y, f[n - 1]->x, f[n - 1]->y);
}

// two fingers on the corner screen: a pinch (the ones moving it, resizing it or touching its picture stop)
void start_pinch() {
    pip_update(2);  // (the one-finger gesture ends where it is)
    for (auto &[id, g] : fingers)
        if (g.kind == PIP_MOVE || g.kind == PIP_SIZE || (g.kind == SCREEN && g.on_pip)) g.kind = PIP_PINCH;
    pip_update(0);
}

// how far a point is outside a control (negative inside)
float distance(const Control &c, float x, float y) {
    const float cx = c.r.x + c.r.w / 2, cy = c.r.y + c.r.h / 2;
    if (c.round) return std::hypot(x - cx, y - cy) - c.r.w / 2;
    const float rad = c.r.h / 2;  // a pill: a segment's distance minus the radius
    const float px = std::clamp(x, c.r.x + rad, c.r.x + c.r.w - rad);
    return std::hypot(x - px, y - cy) - rad;
}

void lay_out() {
    int W = 0, H = 0;
    SDL_GetWindowSize(win, &W, &H);
    SDL_Rect safe;
    if (!SDL_GetWindowSafeArea(win, &safe) || safe.w <= 0) safe = {0, 0, W, H};
    if (W == laid_w && H == laid_h && SDL_RectsEqual(&safe, &laid_safe) && swap_shown == laid_swap) return;
    laid_w = W;
    laid_h = H;
    laid_safe = safe;
    laid_swap = swap_shown;
    const float u = unit = std::clamp(std::min(W, H) * 0.11f, 38.0f, 64.0f);
    const float sx = safe.x, sy = safe.y, sw = safe.w, sh = safe.h;
    controls.clear();
    auto circle = [&](Kind k, uint32_t bit, const char *label, float x, float y, float d) {
        controls.push_back({k, bit, label, {x - d / 2, y - d / 2, d, d}, true});
    };
    auto pill = [&](uint32_t bit, const char *label, float x, float y, float w, float h) {
        controls.push_back({BUTTON, bit, label, {x - w / 2, y - h / 2, w, h}, false});
    };
    float lx, rx, cy, top, bottom;  // the pad's and A B X Y's middles; the shoulder and Start / Select rows
    if (H > W) {  // the controls under the screens (which take the full width when they can)
        const float hc = std::clamp(sh - sw * 1.2f, 5.6f * u, sh * 0.5f), y0 = sy + sh - hc;
        area = {sx, sy, sw, sh - hc};
        lx = sx + std::max(2.0f * u, sw * 0.22f);
        rx = sx + sw - (lx - sx);
        cy = y0 + hc * 0.52f;
        top = y0 + 0.55f * u;
        bottom = sy + sh - 0.45f * u;
        const float mid = sx + sw / 2;
        circle(MENU, 0, nullptr, swap_shown ? mid - 0.6f * u : mid, top, 0.8f * u);
        if (swap_shown) circle(SWAP, 0, nullptr, mid + 0.6f * u, top, 0.8f * u);
        pill(BTN_SELECT, "SELECT", mid - 1.0f * u, bottom, 1.6f * u, 0.6f * u);
        pill(BTN_START, "START", mid + 1.0f * u, bottom, 1.6f * u, 0.6f * u);
    } else {  // a column each side
        lx = std::max(sx + 1.3f * u, 1.6f * u);
        rx = std::min(sx + sw - 1.3f * u, W - 1.6f * u);
        area = {lx + 1.3f * u, sy, rx - lx - 2.6f * u, sh};
        cy = sy + sh * 0.6f;
        top = sy + 0.6f * u;
        bottom = sy + sh - 0.45f * u;
        circle(MENU, 0, nullptr, lx, sy + 1.8f * u, 0.8f * u);
        if (swap_shown) circle(SWAP, 0, nullptr, rx, sy + 1.8f * u, 0.8f * u);
        pill(BTN_SELECT, "SELECT", lx, bottom, 1.6f * u, 0.6f * u);
        pill(BTN_START, "START", rx, bottom, 1.6f * u, 0.6f * u);
    }
    pill(BTN_L, "L", lx, top, 2.2f * u, 0.85f * u);
    pill(BTN_R, "R", rx, top, 2.2f * u, 0.85f * u);
    pad_x = lx;
    pad_y = cy;
    pad_r = 1.25f * u;
    circle(PAD, 0, nullptr, lx, cy, 2 * pad_r);
    circle(BUTTON, BTN_A, "A", rx + u, cy, u);
    circle(BUTTON, BTN_B, "B", rx, cy + u, u);
    circle(BUTTON, BTN_X, "X", rx, cy - u, u);
    circle(BUTTON, BTN_Y, "Y", rx - u, cy, u);
}

// the control a finger going down at a point takes (-1: none), allowing a little slack around each
int hit(float x, float y) {
    int best = -1;
    float best_d = 0;
    for (int i = 0; i < (int)controls.size(); i++) {
        const float d = distance(controls[i], x, y), slack = controls[i].kind == PAD ? 0.5f * unit : 0.2f * unit;
        if (d < slack && (best < 0 || d < best_d)) {
            best = i;
            best_d = d;
        }
    }
    return best;
}

// the buttons under a finger that went down on a button: a round one can press its neighbour too when the
// finger is between them (A+B, ...); a pill only its own
uint32_t buttons_under(const Finger &f) {
    const Control &c = controls[f.control];
    if (!c.round) return distance(c, f.x, f.y) < 0.4f * unit ? c.bit : 0;
    uint32_t bits = 0;
    for (const Control &o : controls)
        if (o.kind == BUTTON && o.round && distance(o, f.x, f.y) < 0.25f * unit) bits |= o.bit;
    return bits;
}

// the circle pad's position for a finger, -1..1 on each axis (y down)
void pad_vector(const Finger &f, float &vx, float &vy) {
    vx = (f.x - pad_x) / (pad_r * 0.8f);
    vy = (f.y - pad_y) / (pad_r * 0.8f);
    const float len = std::hypot(vx, vy);
    if (len > 1) {
        vx /= len;
        vy /= len;
    }
}

// ---- drawing (window pixels)
std::map<std::string, text::Texture> labels;

const text::Texture &label(const char *s, float px) {
    char key[96];
    snprintf(key, sizeof key, "%s|%.0f", s, px);
    auto it = labels.find(key);
    if (it != labels.end()) return it->second;
    return labels[key] = text::render(ren, s, px, {255, 255, 255, 255}, true);
}

// the outline of a rounded rectangle (a circle when the radius is half its size), inset by `in`
std::vector<SDL_FPoint> outline(SDL_FRect r, float rad, float in) {
    r = {r.x + in, r.y + in, r.w - 2 * in, r.h - 2 * in};
    rad = std::max(0.0f, rad - in);
    const float cxs[4] = {r.x + r.w - rad, r.x + rad, r.x + rad, r.x + r.w - rad};
    const float cys[4] = {r.y + rad, r.y + rad, r.y + r.h - rad, r.y + r.h - rad};
    constexpr int kSteps = 10;  // per quarter circle
    std::vector<SDL_FPoint> p;
    for (int q = 0; q < 4; q++)
        for (int i = 0; i <= kSteps; i++) {
            const float a = (float)M_PI / 2 * (q + (float)i / kSteps);
            p.push_back({cxs[q] + rad * std::cos(a), cys[q] - rad * std::sin(a)});
        }
    return p;
}

void fill(const SDL_FRect &r, float rad, SDL_FColor c) {
    const std::vector<SDL_FPoint> p = outline(r, rad, 0);
    std::vector<SDL_Vertex> v{{{r.x + r.w / 2, r.y + r.h / 2}, c, {}}};
    for (const auto &q : p) v.push_back({q, c, {}});
    std::vector<int> idx;
    for (int i = 0; i < (int)p.size(); i++) idx.insert(idx.end(), {0, 1 + i, 1 + (i + 1) % (int)p.size()});
    SDL_RenderGeometry(ren, nullptr, v.data(), (int)v.size(), idx.data(), (int)idx.size());
}

void stroke(const SDL_FRect &r, float rad, float t, SDL_FColor c) {
    const std::vector<SDL_FPoint> o = outline(r, rad, 0), in = outline(r, rad, t);
    std::vector<SDL_Vertex> v;
    for (size_t i = 0; i < o.size(); i++) {
        v.push_back({o[i], c, {}});
        v.push_back({in[i], c, {}});
    }
    std::vector<int> idx;
    const int n = (int)o.size();
    for (int i = 0; i < n; i++) {
        const int a = 2 * i, b = 2 * ((i + 1) % n);
        idx.insert(idx.end(), {a, a + 1, b, b, a + 1, b + 1});
    }
    SDL_RenderGeometry(ren, nullptr, v.data(), (int)v.size(), idx.data(), (int)idx.size());
}

}  // namespace

bool enabled() {
    static const bool on = TARGET_OS_IOS || getenv("RECOMP_TOUCH");
    return on;
}

void init(SDL_Window *window, SDL_Renderer *renderer) {
    win = window;
    ren = renderer;
}

SDL_FRect screen_area() {
    lay_out();
    return area;
}

void set_bottom_screen_mapper(bool (*map)(float, float, int &, int &)) { bottom_at = map; }
void set_pip_hooks(int (*at)(float, float), void (*gesture)(int, int, float, float, float, float)) {
    pip_at = at;
    pip_gesture = gesture;
}
void gamepad(bool in_use) { hidden = in_use; }
void set_swap_shown(bool on) { swap_shown = on; }
void reset() {
    pip_update(2);
    fingers.clear();
}

bool take_menu() {
    const bool t = menu_tapped;
    menu_tapped = false;
    return t;
}

bool take_swap() {
    const bool t = swap_tapped;
    swap_tapped = false;
    return t;
}

bool handle_event(const SDL_Event &ev) {
    if (ev.type != SDL_EVENT_FINGER_DOWN && ev.type != SDL_EVENT_FINGER_MOTION && ev.type != SDL_EVENT_FINGER_UP &&
        ev.type != SDL_EVENT_FINGER_CANCELED)
        return false;
    lay_out();
    const float x = ev.tfinger.x * laid_w, y = ev.tfinger.y * laid_h;
    const SDL_FingerID id = ev.tfinger.fingerID;
    if (ev.type == SDL_EVENT_FINGER_DOWN) {
        hidden = false;
        Finger f;
        f.x = x;
        f.y = y;
        f.control = hit(x, y);
        const int pip = f.control < 0 && pip_at ? pip_at(x, y) : 0;
        const bool pip_busy = std::any_of(fingers.begin(), fingers.end(), [](const auto &g) {
            return g.second.kind == PIP_MOVE || g.second.kind == PIP_SIZE || g.second.kind == PIP_PINCH;
        });
        const bool second_on_pip = pip == 1 && std::any_of(fingers.begin(), fingers.end(), [](const auto &g) {
            return g.second.kind == SCREEN && g.second.on_pip;
        });
        if (f.control >= 0) {
            f.kind = controls[f.control].kind;
            if (f.kind == BUTTON) f.bits = buttons_under(f);
        } else if (pip_busy || second_on_pip) {  // another finger on the corner screen: a pinch
            f.kind = PIP_PINCH;
            fingers[id] = f;
            start_pinch();
            return true;
        } else if (pip == 2 || pip == 3) {
            f.kind = pip == 2 ? PIP_MOVE : PIP_SIZE;
            fingers[id] = f;
            pip_update(0);
            return true;
        } else if (bottom_at && bottom_at(x, y, f.tx, f.ty)) {
            f.kind = SCREEN;
            f.on_pip = pip == 1;
        }
        fingers[id] = f;
        return true;
    }
    auto it = fingers.find(id);
    if (it == fingers.end()) return true;
    Finger &f = it->second;
    f.x = x;
    f.y = y;
    if (ev.type == SDL_EVENT_FINGER_MOTION) {
        if (f.kind == PIP_MOVE || f.kind == PIP_SIZE || f.kind == PIP_PINCH) pip_update(1);
        if (f.kind == BUTTON) f.bits = buttons_under(f);
        if (f.kind == SCREEN) {
            int tx, ty;
            if (bottom_at && bottom_at(x, y, tx, ty)) {  // (off the screen: where it left)
                f.tx = tx;
                f.ty = ty;
            }
        }
        return true;
    }
    // lifted: the menu and swap buttons act when the finger comes up on them
    if (ev.type == SDL_EVENT_FINGER_UP && (f.kind == MENU || f.kind == SWAP) &&
        distance(controls[f.control], x, y) < 0.4f * unit)
        (f.kind == MENU ? menu_tapped : swap_tapped) = true;
    if (f.kind == PIP_MOVE || f.kind == PIP_SIZE || f.kind == PIP_PINCH) {
        pip_update(2);  // the gesture ends; a pinch's other finger does nothing more until it lifts
        for (auto &[gid, g] : fingers)
            if (g.kind == PIP_PINCH) g.kind = NONE;
    }
    fingers.erase(id);
    return true;
}

void poll(uint32_t &buttons, int16_t &cx, int16_t &cy, bool &touched, uint16_t &tx, uint16_t &ty) {
    for (const auto &[id, f] : fingers) {
        if (f.kind == BUTTON) buttons |= f.bits;
        if (f.kind == PAD) {
            float vx, vy;
            pad_vector(f, vx, vy);
            if (std::hypot(vx, vy) > 0.12f) {  // a small dead zone around the middle
                cx = (int16_t)std::lround(vx * 155);
                cy = (int16_t)std::lround(-vy * 155);
            }
        }
        if (f.kind == SCREEN && !touched) {
            touched = true;
            tx = (uint16_t)std::clamp(f.tx, 0, 319);
            ty = (uint16_t)std::clamp(f.ty, 0, 239);
        }
    }
}

void draw() {
    if (hidden) return;
    lay_out();
    int lw, lh;
    SDL_RendererLogicalPresentation mode;
    SDL_GetRenderLogicalPresentation(ren, &lw, &lh, &mode);
    SDL_SetRenderLogicalPresentation(ren, 0, 0, SDL_LOGICAL_PRESENTATION_DISABLED);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
    const float d = SDL_GetWindowPixelDensity(win), u = unit * d, line = std::max(1.0f, 0.045f * u);
    uint32_t held = 0;
    const Finger *pad_finger = nullptr;
    for (const auto &[id, f] : fingers) {
        if (f.kind == BUTTON) held |= f.bits;
        if (f.kind == PAD) pad_finger = &f;
    }
    const SDL_FColor edge{1, 1, 1, 0.42f}, face{1, 1, 1, 0.1f}, lit{1, 1, 1, 0.42f};
    for (const Control &c : controls) {
        const SDL_FRect r{c.r.x * d, c.r.y * d, c.r.w * d, c.r.h * d};
        const float rad = c.round ? r.w / 2 : r.h / 2;
        if (c.kind == PAD) {  // the base, and the knob where the finger holds it
            fill(r, rad, {1, 1, 1, 0.06f});
            stroke(r, rad, line, edge);
            float vx = 0, vy = 0;
            if (pad_finger) pad_vector(*pad_finger, vx, vy);
            const float kr = 0.55f * u, travel = r.w / 2 - kr;
            const SDL_FRect k{r.x + r.w / 2 + vx * travel - kr, r.y + r.h / 2 + vy * travel - kr, 2 * kr, 2 * kr};
            fill(k, kr, {1, 1, 1, pad_finger ? 0.45f : 0.22f});
            stroke(k, kr, line, edge);
            continue;
        }
        const bool on = (c.kind == BUTTON && (held & c.bit)) ||
                        std::any_of(fingers.begin(), fingers.end(), [&](const auto &f) {
                            return f.second.kind == c.kind && (c.kind == MENU || c.kind == SWAP);
                        });
        fill(r, rad, on ? lit : face);
        stroke(r, rad, line, edge);
        const float mx = r.x + r.w / 2, my = r.y + r.h / 2;
        if (c.kind == MENU) {  // three bars
            for (int i = -1; i <= 1; i++) {
                const SDL_FRect bar{mx - r.w * 0.24f, my + i * r.h * 0.17f - line, r.w * 0.48f, 2 * line};
                fill(bar, line, {1, 1, 1, 0.8f});
            }
        } else if (c.kind == SWAP) {  // two screens, one over the other
            const float w = r.w * 0.42f, h = r.h * 0.26f;
            stroke({mx - w / 2, my - h - r.h * 0.03f, w, h}, line, line, {1, 1, 1, 0.8f});
            stroke({mx - w * 0.4f, my + r.h * 0.03f, w * 0.8f, h}, line, line, {1, 1, 1, 0.8f});
        } else if (c.label) {
            const float px = !c.round ? (c.label[1] ? 0.26f : 0.42f) * u : 0.46f * u;
            const text::Texture &t = label(c.label, std::round(px));
            SDL_SetTextureAlphaModFloat(t.tex, 0.85f);
            const SDL_FRect at{std::round(mx - t.w / 2.0f), std::round(my - t.h / 2.0f), (float)t.w, (float)t.h};
            SDL_RenderTexture(ren, t.tex, nullptr, &at);
        }
    }
    SDL_SetRenderLogicalPresentation(ren, lw, lh, mode);
}

}  // namespace touch
