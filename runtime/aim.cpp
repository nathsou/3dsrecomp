// Aiming without the touch screen (aim.h). The profile names two of the game's calls:
// - "aim pad": the copy of the pad state the game makes for its player each frame; r1 is the state copied, where
//   the control mode byte is set to the game's stick-aiming mode and the right stick written as its aim stick;
// - "aim delta": a call made while the frame's aim delta (reticle and camera motion, from the stylus and the stick)
//   sits on the stack; the mouse's motion is added to it there, so the game's own limits and camera turning apply.
#include "aim.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

#include "profile.h"
#include "rt.h"

namespace aim {
namespace {

constexpr float kDeadzone = 0.15f;

std::atomic<bool> on;
std::mutex m;
float stick_x, stick_y;    // the right stick, past the dead zone (-1..1, up positive)
float mouse_x, mouse_y;    // mouse motion not yet handed to the game (top-screen pixels)
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

// the game copying its pad state (r1)
void pad_hook(CPU *c) {
    if (!on) return;
    const profile::Aim &a = profile::aim();
    float x, y;
    {
        std::lock_guard<std::mutex> g(m);
        x = stick_x, y = stick_y;
    }
    const u32 pad = c->r[1];
    w8(pad + a.mode_off, a.mode);
    put_float(pad + a.stick_x, x);
    put_float(pad + a.stick_y, -y);  // (the game's sticks: up is negative)
}

// a call made while the frame's aim delta is on the stack
void delta_hook(CPU *c) {
    const profile::Aim &a = profile::aim();
    if (!on || c->r[14] != a.delta_ret) return;
    float x, y;
    {
        std::lock_guard<std::mutex> g(m);
        x = mouse_x, y = mouse_y;
        mouse_x = mouse_y = 0;
    }
    if (x == 0 && y == 0) return;
    const u32 sp = c->r[13];
    if (a.delta_per >= 0) {  // a rate the game multiplies by its frame time (in that register): pixels per frame
        const float dt = c->s[a.delta_per];
        if (dt > 1e-4f) x /= dt, y /= dt;
    }
    put_float(sp + a.delta_x, get_float(sp + a.delta_x) + x);
    put_float(sp + a.delta_y, get_float(sp + a.delta_y) + y);
}

}  // namespace

bool configured() { return profile::aim().cond != -2; }
bool active() { return on; }

bool update(float rsx, float rsy, float mouse_dx, float mouse_dy) {
    const profile::Aim &a = profile::aim();
    if (a.cond == -2) return false;
    if (!hooked) {
        hooked = true;
        if (a.pad_on) rt_add_dispatch_hook(a.pad_on, pad_hook);
        for (u32 at : a.delta_on) rt_add_dispatch_hook(at, delta_hook);
    }
    const bool now = profile::cond(a.cond);
    auto dz = [](float v) { return std::fabs(v) < kDeadzone ? 0.0f : (v - std::copysign(kDeadzone, v)) / (1 - kDeadzone); };
    {
        std::lock_guard<std::mutex> g(m);
        stick_x = dz(rsx), stick_y = dz(rsy);
        if (now) {
            mouse_x += mouse_dx, mouse_y += mouse_dy;
        } else {
            mouse_x = mouse_y = 0;
        }
    }
    on = now;
    return now;
}

}  // namespace aim
