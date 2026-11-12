// The free camera (freecam.h). The game's camera update (Camera_Update in the OoT decompilation's names) is called
// through the dispatcher, where it is hooked: it runs as usual, then, once the right stick has moved, the camera is
// put where the stick turned it, around the point the game's camera looks at (Link's head, which the game keeps
// following), at the usual distance and out of walls (the game's own collision check), and the game is told the
// camera's direction (Link runs relative to it). The game's camera goes on from there each frame, so it takes over
// smoothly whenever the free camera stops applying. The structures are the decompilation's, as Ocarina of Time 3D
// lays them out.
#include "freecam.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>

#include "profile.h"
#include "rt.h"
#include "settings.h"

namespace freecam {
namespace {

// Camera
constexpr u32 CAM_AT = 0x80, CAM_EYE = 0x8c, CAM_EYE_NEXT = 0xa4;  // Vec3f
constexpr u32 CAM_GLOBAL_CTX = 0xd4, CAM_PLAYER = 0xd8;
constexpr u32 CAM_INPUT_DIR = 0x17c, CAM_DIR = 0x182;  // Vec3s: pitch, yaw, roll
constexpr u32 CAM_STATUS = 0x188, CAM_SETTING = 0x18a;
constexpr s16 CAM_STAT_ACTIVE = 7;
// GlobalContext
constexpr u32 GCTX_SCENE = 0x104, GCTX_VIEW_EYE = 0x1b8, GCTX_MAIN_CAMERA = 0x364;
// Player
constexpr u32 PLAYER_STATE1 = 0x1710, PLAYER_STATE2 = 0x1714;
// the game's camera is needed: lock-on and parallel (Z) targeting, getting an item, first person, riding Epona, a
// cutscene (state 1); crawling through a passage (state 2)
constexpr u32 STATE1_OWN_CAMERA = 0x20938230, STATE2_OWN_CAMERA = 0x00040000;
// camera settings with a fixed or scripted camera (shops, the market's back alley and balcony, Bombchu Bowling,
// Gerudo Fortress's cells, fixed and rotating rooms), and the castle's hedge maze (a scene of fixed cameras)
constexpr s16 kOwnSettings[] = {0x14, 0x15, 0x19, 0x1a, 0x1b, 0x1d, 0x23, 0x40, 0x46};
constexpr s16 SCENE_HEDGE_MAZE = 0x45;
// SaveContext
constexpr u32 SAVE_MASTER_QUEST = 0xe;  // (Master Quest's world is mirrored)
// CamColChk (the collision check's "to": the position it moves in front of walls, then what it hit)
constexpr u32 COLCHK_SIZE = 0x28;
constexpr u32 kScratch = 0x40;  // the collision check's arguments: this far below the caller's stack, 16-byte aligned
constexpr u32 kTo = 0x10;       // ("from", a Vec3f, then "to")

constexpr float kDeadzone = 0.15f;
constexpr float kBinangPerRad = 32768 / float(M_PI);  // angles are s16, a full turn 0x10000
constexpr float kYawRate = 0x8000;                    // at full tilt, per second (half a turn)
constexpr float kPitchRate = 0x5000;
// the game's pitch is up from the camera to what it looks at: looking down from above (67.5 degrees) or up from
// below (22.5: Link's head is not far above the floor, which stops the camera)
constexpr float kPitchDown = -0x3000, kPitchUp = 0x1000;
constexpr float kMinDist = 50;                        // (a game camera inside Link's head as the stick moves)
constexpr float kFollow = 0.3f;                       // share of the way to where it is going, per 30 fps frame
constexpr float kGameFps = 30;
constexpr double kVBlankHz = 60;

std::mutex m;
float stick_x, stick_y;  // past the dead zone and curved (-1..1, up positive)
bool applies;            // the profile's condition and the setting
bool hooked;

// the game thread's
bool engaged;            // the stick moved since the game's camera last took over
struct V3 { float x, y, z; };
float yaw, pitch, dist;  // where the stick has turned the camera (binangs), how far it is (as the game had it)
V3 eye;                  // where the free camera is
u64 last_frame;

V3 get_v3(u32 a) {
    V3 v;
    memcpy(&v, gptr(a), sizeof v);
    return v;
}
void put_v3(u32 a, V3 v) { memcpy(gptr(a), &v, sizeof v); }
V3 lerp(V3 a, V3 b, float t) { return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t}; }
s16 binang(float rad) { return (s16)(s32)std::lround(rad * kBinangPerRad); }

bool game_camera_needed(u32 cam) {
    const u32 gctx = r32(cam + CAM_GLOBAL_CTX), player = r32(cam + CAM_PLAYER);
    if (!gctx || !player || cam != gctx + GCTX_MAIN_CAMERA) return true;
    if (r32(player + PLAYER_STATE1) & STATE1_OWN_CAMERA || r32(player + PLAYER_STATE2) & STATE2_OWN_CAMERA) return true;
    if ((s16)r16(cam + CAM_STATUS) != CAM_STAT_ACTIVE || (s16)r16(gctx + GCTX_SCENE) == SCENE_HEDGE_MAZE) return true;
    const s16 setting = (s16)r16(cam + CAM_SETTING);
    return std::find(std::begin(kOwnSettings), std::end(kOwnSettings), setting) != std::end(kOwnSettings);
}

// as the game calls Camera_Update(Vec3s *direction out, Camera *camera)
void update_hook(CPU *c) {
    const profile::Camera &p = profile::camera();
    const u32 out = c->r[0], cam = c->r[1];
    const u32 ret = rt_guest_call(c, p.update, out, cam);
    float sx, sy;
    bool on;
    {
        std::lock_guard<std::mutex> g(m);
        sx = stick_x, sy = stick_y, on = applies;
    }
    const float dt = std::clamp(float((g_frame - last_frame) / kVBlankHz), 1 / 60.0f, 1 / 10.0f);
    last_frame = g_frame;
    if (!on || game_camera_needed(cam)) {
        engaged = false;
    } else {
        const V3 at = get_v3(cam + CAM_AT), game_eye = get_v3(cam + CAM_EYE);
        if (!engaged && (sx != 0 || sy != 0)) {  // from where the game's camera is
            engaged = true;
            const float dx = at.x - game_eye.x, dy = at.y - game_eye.y, dz = at.z - game_eye.z;
            yaw = std::atan2(dx, dz) * kBinangPerRad;
            pitch = std::clamp(std::atan2(dy, std::hypot(dx, dz)) * kBinangPerRad, kPitchDown, kPitchUp);
            dist = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), kMinDist);
            eye = game_eye;
        }
        if (engaged) {
            const u32 save = p.save;
            // (screen right is the other way round in Master Quest's mirrored world)
            const float turn = r8(save + SAVE_MASTER_QUEST) ? -sx : sx;
            yaw -= turn * kYawRate * dt;
            yaw = std::remainder(yaw, 65536.0f);
            pitch = std::clamp(pitch + (g_settings.camera_invert_y ? -sy : sy) * kPitchRate * dt, kPitchDown, kPitchUp);
            const float follow = 1 - std::pow(1 - kFollow, dt * kGameFps);
            const float pr = pitch / kBinangPerRad, yr = yaw / kBinangPerRad;
            // where the camera would be (the game's pitch is up from the camera to what it looks at), moved in front
            // of walls by the game's collision check, on the stack below the caller's
            const V3 want = {at.x - dist * std::cos(pr) * std::sin(yr), at.y - dist * std::sin(pr),
                             at.z - dist * std::cos(pr) * std::cos(yr)};
            const u32 sp = c->r[13];
            const u32 from = (sp - kScratch) & ~0xfu, to = from + kTo;
            put_v3(from, at);
            memset(gptr(to), 0, COLCHK_SIZE);
            put_v3(to, want);
            c->r[13] = from;  // (the check's own frame goes below)
            rt_guest_call(c, p.collide, cam, from, to);
            c->r[13] = sp;
            eye = lerp(eye, get_v3(to), follow);
            // the view: the camera plus what the game added to it this frame (a quake's shaking)
            const u32 gctx = r32(cam + CAM_GLOBAL_CTX);
            const V3 view = get_v3(gctx + GCTX_VIEW_EYE);
            put_v3(gctx + GCTX_VIEW_EYE, {eye.x + view.x - game_eye.x, eye.y + view.y - game_eye.y,
                                          eye.z + view.z - game_eye.z});
            put_v3(cam + CAM_EYE, eye);
            put_v3(cam + CAM_EYE_NEXT, eye);
            // the direction it looks in, as it is (Link's movement follows it)
            const float dx = at.x - eye.x, dy = at.y - eye.y, dz = at.z - eye.z;
            const s16 dir[3] = {binang(std::atan2(dy, std::hypot(dx, dz))), binang(std::atan2(dx, dz)), 0};
            for (u32 a : {out, cam + CAM_INPUT_DIR, cam + CAM_DIR}) memcpy(gptr(a), dir, sizeof dir);
        }
    }
    c->r[0] = ret;
    c->pc = c->r[14];  // (the update ran: the call returns)
}

}  // namespace

bool configured() { return profile::camera().update && profile::camera().cond != -2; }

bool update(float rsx, float rsy) {
    if (!configured()) return false;
    const profile::Camera &p = profile::camera();
    if (!hooked) {
        hooked = true;
        rt_add_dispatch_hook(p.update, update_hook);
    }
    const bool now = g_settings.free_camera && profile::cond(p.cond);
    auto curve = [](float v) {
        if (std::fabs(v) < kDeadzone) return 0.0f;
        const float t = (std::fabs(v) - kDeadzone) / (1 - kDeadzone);
        return std::copysign(t * t, v);  // (fine turns near the centre, full speed at the rim)
    };
    std::lock_guard<std::mutex> g(m);
    stick_x = now ? curve(rsx) : 0, stick_y = now ? curve(rsy) : 0;
    applies = now;
    return now;
}

}  // namespace freecam
