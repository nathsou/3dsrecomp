// hid:USER / hid:SPVR: pad, circle pad, touch screen, accelerometer and gyroscope through shared memory.
#include <algorithm>
#include <atomic>
#include "../service.h"
#include "services.h"

extern std::atomic<int> g_winshot_request;  // (frontend.cpp)
extern std::atomic<u32> g_live_host;         // (frontend.cpp): host inputs for one-screen play (onescreen.h)
void frontend_live_text(const std::string &s);  // (frontend.cpp): typed as on the host's keyboard
void frontend_live_key(const std::string &name);  // (frontend.cpp): a host key press (SDL key name)
void frontend_live_mouse(float dx, float dy);       // (frontend.cpp): captured mouse motion (aiming)
void watch_arm(u32 guest_addr, u32 bytes);  // (watch.cpp)

namespace onescreen {
u32 host_mask(const char *names);
}

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
// starts. A line "DUMP" writes the state of every thread to the log at once, "MEMDUMP <tag>" snapshots guest memory
// (mem_<tag>_<address>.bin, as RECOMP_MEMDUMP), "POKE8 <hex address> <hex value>" writes a byte at once,
// "WATCH <hex address> [<bytes>]" logs every write to that memory from then on (as RECOMP_WATCH), "POKE32 <hex
// address> <hex value>" / "POKEF <hex address> <float>" write a word at once, "MOUSEREL <dx> <dy>" moves
// the captured mouse (aiming), "MEMDUMPNOW <tag>" and "WINSHOTNOW" snapshot memory or the window at once (while a
// queued command holds its input),
// "DRAG <x>/<y> <x1>/<y1> <frames>" touches and moves (bottom-screen pixels), "PAD <host inputs> <frames>" plays host inputs by position (one-screen play: "PAD zl 6", "PAD rs-up 30"),
// "KEY <SDL key name>" presses a host key, "TEXT <text>" types on the game's own keyboard (the profile's keys), "WINSHOT" writes the whole window at the next frame (window_<frame>.ppm, as RECOMP_WINSHOT).
namespace {
struct LiveCmd {
    u32 buttons = 0;
    int cx = 0, cy = 0, tx = -1, ty = -1, frames = 0;
    std::string memdump;  // "MEMDUMP <tag>": guest memory as mem_<tag>_<address>.bin when the command is reached
    u32 host = 0;         // "PAD <inputs> <frames>": host inputs by position (zl, back, dup, rs-up, ...)
    int x1 = -1, y1 = -1;  // "DRAG <x>/<y> <x1>/<y1> <frames> [<move>]": a touch moving from (tx, ty) to here
    int move = 0;          // (in that many frames, then held there; 0: over the whole command)
    u64 start = 0;
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
    if (line.rfind("DRAG ", 0) == 0) {
        int n = 0;
        sscanf(line.c_str() + 5, "%d/%d %d/%d %d %d", &c.tx, &c.ty, &c.x1, &c.y1, &n, &c.move);
        c.frames = n > 0 ? n : 1;
        return c;
    }
    if (line.rfind("PAD ", 0) == 0) {
        char names[128] = {};
        int n = 0;
        sscanf(line.c_str() + 4, "%127s %d", names, &n);
        c.host = onescreen::host_mask(names);
        c.frames = n > 0 ? n : 1;
        return c;
    }
    if (line.rfind("KEY ", 0) == 0) {  // a host key press (Tab, F2, Escape, ...)
        c.memdump = "@key:" + line.substr(4);
        c.frames = 1;
        return c;
    }
    if (line.rfind("TEXT ", 0) == 0) {  // typed on the game's own keyboard, as on the host's
        c.memdump = "@text:" + line.substr(5);
        c.frames = 1;
        return c;
    }
    if (line == "WINSHOT") {  // the whole window at the next frame (window_<frame>.ppm)
        c.memdump = "@window";
        c.frames = 1;
        return c;
    }
    if (line.rfind("MEMDUMP", 0) == 0) {
        c.memdump = line.size() > 8 ? line.substr(8) : "live";
        c.frames = 1;
        return c;
    }
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
        else if (line.rfind("MEMDUMPNOW ", 0) == 0) dump_memory(("mem_" + line.substr(11)).c_str());  // right away
        else if (line == "WINSHOTNOW") g_winshot_request = 1;  // right away: the window at the next frame
        else if (line.rfind("MOUSEREL ", 0) == 0) {  // right away: mouse motion while aiming (window points)
            float dx = 0, dy = 0;
            if (sscanf(line.c_str() + 9, "%f %f", &dx, &dy) == 2) frontend_live_mouse(dx, dy);
        }
        else if (line.rfind("POKE32 ", 0) == 0 || line.rfind("POKEF ", 0) == 0) {  // right away: a word or a float
            unsigned a = 0;
            char v[64] = {};
            if (sscanf(line.c_str() + line.find(' ') + 1, "%x %63s", &a, v) == 2) {
                u32 w = (u32)strtoul(v, nullptr, 16);
                if (line[4] == 'F') {
                    const float f = strtof(v, nullptr);
                    memcpy(&w, &f, 4);
                }
                w32(a, w);
            }
        }
        else if (line.rfind("WATCH ", 0) == 0) {  // right away: "WATCH <address> [<bytes>]" logs every write (RECOMP_WATCH)
            unsigned a = 0, n = 4;
            if (sscanf(line.c_str() + 6, "%x %u", &a, &n) >= 1) {
                ::watch_arm(a, n);
            }
        }
        else if (line.rfind("POKE8 ", 0) == 0) {  // right away: "POKE8 <address> <value>" writes a byte of guest memory
            unsigned a = 0, v = 0;
            if (sscanf(line.c_str() + 6, "%x %x", &a, &v) == 2) w8(a, (u8)v);
        }
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
        live.cur.start = g_frame;
        live.active = true;
        LOG("[live] command %zu starts (%d frames)\n", live.next, live.cur.frames);
        if (live.cur.memdump == "@window") {
            g_winshot_request = 1;
        } else if (live.cur.memdump.rfind("@key:", 0) == 0) {
            frontend_live_key(live.cur.memdump.substr(5));
        } else if (live.cur.memdump.rfind("@text:", 0) == 0) {
            frontend_live_text(live.cur.memdump.substr(6));
        } else if (!live.cur.memdump.empty()) {
            dump_memory(("mem_" + live.cur.memdump).c_str());
        }
    }
    g_live_host = live.active ? live.cur.host : 0;
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
        if (live.cur.x1 >= 0) {  // a drag: there linearly, the last frames held at the end
            const int span = live.cur.move > 0 ? live.cur.move : live.cur.frames - 6;
            const float t = std::min(1.0f, (float)(g_frame - live.cur.start) / std::max(1, span));
            tx = (u16)(live.cur.tx + (live.cur.x1 - live.cur.tx) * t);
            ty = (u16)(live.cur.ty + (live.cur.y1 - live.cur.ty) * t);
        }
    }
}
}  // namespace

void hid_update(u32 buttons, s16 cx, s16 cy, bool touch, u16 tx, u16 ty, s16 sx, s16 sy) {
    live_input(buttons, cx, cy, touch, tx, ty);
    Lock lk(kmutex);
    ir_poll(buttons, sx, sy);
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
    static const bool no_accel = getenv("RECOMP_NO_ACCEL") != nullptr;
    if (hid.accel_on && !no_accel) {
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
