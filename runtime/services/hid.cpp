// hid:USER / hid:SPVR: pad, circle pad, touch screen, accelerometer and gyroscope through shared memory.
#include "../service.h"
#include "services.h"

namespace kernel {

namespace {
struct HidState {
    std::shared_ptr<SharedMem> shm;
    std::shared_ptr<Event> ev_pad0 = make_event(ONESHOT, "HID pad0");
    std::shared_ptr<Event> ev_pad1 = make_event(ONESHOT, "HID pad1");
    std::shared_ptr<Event> ev_accel = make_event(ONESHOT, "HID accel");
    std::shared_ptr<Event> ev_gyro = make_event(ONESHOT, "HID gyro");
    std::shared_ptr<Event> ev_debug = make_event(ONESHOT, "HID debugpad");
    u32 last_buttons = 0;
    int accel_on = 0, gyro_on = 0;
    u32 tick = 0;
};
HidState hid;
}  // namespace

static void ensure_shm() {
    if (hid.shm) return;
    hid.shm = std::make_shared<SharedMem>();
    hid.shm->name = "HID";
    hid.shm->size = 0x2B0;
    hid.shm->backing = mem::hle_alloc(0x1000);
}

// RECOMP_LIVE=<file>: input for a running test, one command per line appended to the file while the game runs
// ("<buttons> <frames>", buttons as in RECOMP_INPUT: A, B, A+B, CUP, CDOWN+B, T@x/y...; "WAIT <frames>"). Commands
// run one after the other for that many VBlanks, each followed by a few released frames; the log says when each
// starts. A line "DUMP" writes the state of every thread to the log at once.
namespace {
struct LiveCmd {
    u32 buttons = 0;
    int cx = 0, cy = 0, tx = -1, ty = -1, frames = 0;
};
struct Live {
    FILE *f = nullptr;
    std::string partial;
    std::vector<LiveCmd> queue;
    size_t next = 0;
    u64 until = 0;  // (VBlanks)
    bool gap = false;
    LiveCmd cur;
    bool active = false;
} live;

LiveCmd parse_live(const std::string &line) {
    LiveCmd c;
    char names[128] = {};
    int n = 0;
    if (sscanf(line.c_str(), "%127s %d", names, &n) < 1) return c;
    c.frames = n > 0 ? n : 1;
    std::string s = names;
    std::vector<std::string> parts;
    for (size_t p = 0; p <= s.size();) {
        size_t e = s.find('+', p);
        if (e == std::string::npos) e = s.size();
        parts.push_back(s.substr(p, e - p));
        p = e + 1;
    }
    static const std::pair<const char *, u32> btns[] = {
        {"A", 1 << 0}, {"B", 1 << 1}, {"SELECT", 1 << 2}, {"START", 1 << 3}, {"DRIGHT", 1 << 4}, {"DLEFT", 1 << 5},
        {"DUP", 1 << 6}, {"DDOWN", 1 << 7}, {"R", 1 << 8}, {"L", 1 << 9}, {"X", 1 << 10}, {"Y", 1 << 11},
        {"ZL", 1 << 14}, {"ZR", 1 << 15}};
    for (auto &p : parts) {
        for (auto &[name, bit] : btns)
            if (p == name) c.buttons |= bit;
        if (p == "CUP") c.cy = 155;
        if (p == "CDOWN") c.cy = -155;
        if (p == "CLEFT") c.cx = -155;
        if (p == "CRIGHT") c.cx = 155;
        if (p.rfind("T@", 0) == 0) sscanf(p.c_str() + 2, "%d/%d", &c.tx, &c.ty);
    }
    return c;
}

void live_input(u32 &buttons, s16 &cx, s16 &cy, bool &touch, u16 &tx, u16 &ty) {
    static const char *path = getenv("RECOMP_LIVE");
    if (!path) return;
    if (!live.f) live.f = fopen(path, "r");
    if (!live.f) return;
    char buf[512];
    clearerr(live.f);
    while (fgets(buf, sizeof buf, live.f)) {
        live.partial += buf;
        if (live.partial.back() != '\n') break;
        std::string line = live.partial.substr(0, live.partial.size() - 1);
        live.partial.clear();
        if (line == "DUMP") dump_threads();  // right away: the state of every thread in the log
        else if (!line.empty()) live.queue.push_back(parse_live(line));
    }
    if (g_frame >= live.until) {
        if (live.active) {
            live.active = false;
            live.until = g_frame + 4;  // released for a few frames
            return;
        }
        if (live.next >= live.queue.size()) return;
        live.cur = live.queue[live.next++];
        live.until = g_frame + live.cur.frames;
        live.active = true;
        LOG("[live] command %zu starts (%d frames)\n", live.next, live.cur.frames);
    }
    if (!live.active) return;
    buttons |= live.cur.buttons;
    if (live.cur.cx || live.cur.cy) {
        cx = (s16)live.cur.cx;
        cy = (s16)live.cur.cy;
    }
    if (live.cur.tx >= 0) {
        touch = true;
        tx = (u16)live.cur.tx;
        ty = (u16)live.cur.ty;
    }
}
}  // namespace

void hid_update(u32 buttons, s16 cx, s16 cy, bool touch, u16 tx, u16 ty) {
    live_input(buttons, cx, cy, touch, tx, ty);
    Lock lk(kmutex);
    if (!hid.shm) return;
    u32 b = hid.shm->backing;
    u64 now = system_tick();
    // circle pad as directional buttons too
    const int th = 41;
    if (cx >= th) buttons |= 1u << 28;
    if (cx <= -th) buttons |= 1u << 29;
    if (cy >= th) buttons |= 1u << 30;
    if (cy <= -th) buttons |= 1u << 31;
    if (touch) buttons |= 1u << 20;
    // pad
    u32 idx = (r32(b + 0x10) + 1) % 8;
    if (idx == 0) {
        w64(b + 0x08, r64(b + 0x00));
        w64(b + 0x00, now);
    }
    w32(b + 0x10, idx);
    w32(b + 0x1C, buttons);
    u32 e = b + 0x28 + idx * 0x10;
    w32(e + 0, buttons);
    w32(e + 4, buttons & ~hid.last_buttons);
    w32(e + 8, hid.last_buttons & ~buttons);
    w16(e + 12, (u16)cx);
    w16(e + 14, (u16)cy);
    hid.last_buttons = buttons;
    // touch
    u32 tidx = (r32(b + 0xB8) + 1) % 8;
    if (tidx == 0) {
        w64(b + 0xB0, r64(b + 0xA8));
        w64(b + 0xA8, now);
    }
    w32(b + 0xB8, tidx);
    u32 te = b + 0xC8 + tidx * 8;
    w16(te, touch ? tx : 0);
    w16(te + 2, touch ? ty : 0);
    w32(te + 4, touch ? 1 : 0);
    w16(b + 0xC0, touch ? tx : 0);
    w16(b + 0xC2, touch ? ty : 0);
    w32(b + 0xC4, touch ? 1 : 0);
    hid.ev_pad0->signal();
    hid.ev_pad1->signal();
    // accelerometer: at rest, face up (y = -1g in the 3DS's frame)
    if (hid.accel_on && !getenv("RECOMP_NO_ACCEL")) {
        u32 aidx = (r32(b + 0x118) + 1) % 8;
        if (aidx == 0) {
            w64(b + 0x110, r64(b + 0x108));
            w64(b + 0x108, now);
        }
        w32(b + 0x118, aidx);
        u32 ae = b + 0x128 + aidx * 6;
        w16(ae, 0);
        w16(ae + 2, (u16)(s16)-512);
        w16(ae + 4, 0);
        hid.ev_accel->signal();
    }
    if (hid.gyro_on) {
        u32 gidx = (r32(b + 0x168) + 1) % 32;
        if (gidx == 0) {
            w64(b + 0x160, r64(b + 0x158));
            w64(b + 0x158, now);
        }
        w32(b + 0x168, gidx);
        u32 ge = b + 0x178 + gidx * 6;
        w16(ge, 0);
        w16(ge + 2, 0);
        w16(ge + 4, 0);
        hid.ev_gyro->signal();
    }
    schedule_async();
}

struct Hid : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x000A:  // GetIPCHandles
            ensure_shm();
            ipc.ok(1, 7);
            ipc.set(2, 5u << 26);
            ipc.set(3, add_handle(hid.shm));
            ipc.set(4, add_handle(hid.ev_pad0));
            ipc.set(5, add_handle(hid.ev_pad1));
            ipc.set(6, add_handle(hid.ev_accel));
            ipc.set(7, add_handle(hid.ev_gyro));
            ipc.set(8, add_handle(hid.ev_debug));
            break;
        case 0x0011:
            hid.accel_on++;
            ipc.ok();
            break;
        case 0x0012:
            if (hid.accel_on) hid.accel_on--;
            ipc.ok();
            break;
        case 0x0013:
            hid.gyro_on++;
            ipc.ok();
            break;
        case 0x0014:
            if (hid.gyro_on) hid.gyro_on--;
            ipc.ok();
            break;
        case 0x0015: {  // GetGyroscopeLowRawToDpsCoefficient
            float f = 14.375f;
            ipc.ok(2);
            u32 u;
            memcpy(&u, &f, 4);
            ipc.set(2, u);
            break;
        }
        case 0x0016: {  // GetGyroscopeLowCalibrateParam
            ipc.ok(6);
            const s16 p[9] = {0, 6700, -6700, 0, 6700, -6700, 0, 6700, -6700};
            memcpy(gptr(ipc.buf + 8), p, 18);
            break;
        }
        case 0x0017:  // GetSoundVolume
            ipc.ok(2);
            ipc.set(2, 0x3F);
            break;
        default:
            unimplemented(ipc);
        }
    }
};

void register_hid() {
    register_service(std::make_shared<Hid>("hid:USER"));
    register_service(std::make_shared<Hid>("hid:SPVR"));
}

}  // namespace kernel
