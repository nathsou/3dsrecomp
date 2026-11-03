// Aiming without the touch screen (aim.h). The profile names a call the game makes while the frame's aim delta
// (the reticle's and the camera's motion, from the stylus or the game's own button aiming) sits on the stack: the
// right stick and the mouse's motion are added to it there, so the game's own reticle limits and camera turning
// apply, and the game's "aim input this frame" flag is set meanwhile (without it, the game recenters the reticle as
// it does when the stylus is lifted).
#include "aim.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

#include "profile.h"
#include "rt.h"
#include "settings.h"

namespace aim {
namespace {

constexpr float kDeadzone = 0.15f;
constexpr double kMouseHold = 1.5;  // seconds the aim stays where the mouse left it (input flag kept)

std::atomic<bool> on;
std::mutex m;
float stick_x, stick_y;    // the right stick, past the dead zone and curved (-1..1, up positive)
float mouse_x, mouse_y;    // mouse motion not yet handed to the game (top-screen pixels)
bool engaged;              // the stick is held or the mouse moved lately: the game's input flag is set
bool hooked;

void put_float(u32 a, float f) {
    u32 w;
    memcpy(&w, &f, 4);
    w32(a, w);
}
float get_float(u32 a) {
    const u32 w = r32(a);
    float f;
    memcpy(&f, &w, 4);
    return f;
}

// a call made while the frame's aim delta is on the stack
void delta_hook(CPU *c) {
    const profile::Aim &a = profile::aim();
    if (!on || c->r[14] != a.delta_ret) return;
    float mx, my, sx, sy;
    bool held;
    {
        std::lock_guard<std::mutex> g(m);
        mx = mouse_x, my = mouse_y, sx = stick_x, sy = stick_y, held = engaged;
        mouse_x = mouse_y = 0;
    }
    if (held && a.active_reg >= 0) w8(c->r[a.active_reg] + a.active_off, 1);
    if (mx == 0 && my == 0 && sx == 0 && sy == 0) return;
    if (a.delta_per >= 0) {  // a rate the game multiplies by its frame time (in that register): pixels per frame
        const float dt = c->s[a.delta_per];
        if (dt > 1e-4f) mx /= dt, my /= dt;
    }
    const u32 sp = c->r[13];
    const float speed = a.stick_speed * g_settings.aim_speed / 100;
    put_float(sp + a.delta_x, get_float(sp + a.delta_x) + mx + sx * speed);
    put_float(sp + a.delta_y, get_float(sp + a.delta_y) + my - sy * speed);  // (screen y grows downwards)
}

}  // namespace

bool configured() { return profile::aim().cond != -2; }
bool active() { return on; }

bool update(float rsx, float rsy, float mouse_dx, float mouse_dy) {
    const profile::Aim &a = profile::aim();
    if (a.cond == -2) return false;
    if (!hooked) {
        hooked = true;
        for (u32 at : a.delta_on) rt_add_dispatch_hook(at, delta_hook);
    }
    const bool now = profile::cond(a.cond);
    auto curve = [](float v) {
        if (std::fabs(v) < kDeadzone) return 0.0f;
        const float t = (std::fabs(v) - kDeadzone) / (1 - kDeadzone);
        return std::copysign(t * t, v);  // (fine aiming near the centre, full speed at the rim)
    };
    using clock = std::chrono::steady_clock;
    static clock::time_point last_mouse;
    if (mouse_dx != 0 || mouse_dy != 0) last_mouse = clock::now();
    const bool mouse_recent = std::chrono::duration<double>(clock::now() - last_mouse).count() < kMouseHold;
    {
        std::lock_guard<std::mutex> g(m);
        stick_x = curve(rsx), stick_y = curve(rsy);
        if (now) {
            mouse_x += mouse_dx, mouse_y += mouse_dy;
        } else {
            mouse_x = mouse_y = 0;
        }
        engaged = now && (stick_x != 0 || stick_y != 0 || mouse_recent);
    }
    on = now;
    return now;
}

}  // namespace aim
