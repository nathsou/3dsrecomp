// Small services: ptm, ndm, ac, frd, boss, cecd, news, ir, mic, cam, y2r, ldr:ro, soc, ssl, http, ...
#include <algorithm>
#include <cstring>
#include <vector>

#include "../loader.h"
#include "../service.h"
#include "services.h"

namespace kernel {

struct Ptm : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0005:  // GetAdapterState
        case 0x0006:  // GetShellState (open)
        case 0x0008:  // GetBatteryChargeState
            ipc.ok(2);
            ipc.set(2, 1);
            break;
        case 0x0007:  // GetBatteryLevel
            ipc.ok(2);
            ipc.set(2, 5);
            break;
        case 0x0009:  // GetPedometerState
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x000B: {  // GetStepHistory(hours, start time u64, mapped buffer)
            u32 n = ipc[1];
            memset(gptr(ipc[5]), 0, n * 2);
            ipc.ok(1, 2);
            ipc.set(2, ((n * 2) << 4) | 0xC);
            ipc.set(3, ipc[5]);
            break;
        }
        case 0x000C:  // GetTotalStepCount
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x0818:  // ptm:sysm ConfigureNew3DSCPU (libctru osSetSpeedupEnable): an old 3DS ignores it
            ipc.ok();
            break;
        case 0x0001:
        case 0x0002:
        case 0x0003:
        case 0x0004:
        default:
            unimplemented(ipc);
        }
    }
};

struct Ac : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: {  // CreateDefaultConfig -> static buffer 0x200
            u32 dst = ipc.static_buffer(0);
            memset(gptr(dst), 0, 0x200);
            ipc.ok(1, 2);
            ipc.static_desc(2, 0, 0x200);
            break;
        }
        case 0x0004:  // ConnectAsync
            ipc.ok();
            if (auto ev = get<Event>(ipc[5])) ev->signal();
            break;
        case 0x0005:  // GetConnectResult
            ipc.result(0xE0A09D2E);  // not connected
            break;
        case 0x000D:  // GetStatus
        case 0x000F:  // GetWifiStatus
        case 0x003E:  // IsConnected
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x000E:  // GetLastErrorCode
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        default:
            unimplemented(ipc);
        }
    }
};

struct Frd : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // HasLoggedIn
        case 0x0002:  // IsOnline
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x0005: {  // GetMyPresence -> static buffer
            u32 dst = ipc.static_buffer(0);
            memset(gptr(dst), 0, 0x12C);
            ipc.ok(1, 2);
            ipc.static_desc(2, 0, 0x12C);
            break;
        }
        case 0x0006:  // GetMyFriendKey
            ipc.ok(5);
            ipc.set(2, 0);
            ipc.set(3, 0);
            ipc.set64(4, 0);
            break;
        case 0x0009:  // GetMyPreference
            ipc.ok(4);
            ipc.set(2, 0);
            ipc.set(3, 0);
            ipc.set(4, 0);
            break;
        case 0x000A:  // GetMyProfile
            ipc.ok(3);
            ipc.set(2, 0);
            ipc.set(3, 0);
            break;
        case 0x000B: {  // GetMyScreenName
            ipc.ok(7);
            memset(gptr(ipc.buf + 8), 0, 24);
            const char *n = "Mario";
            for (int i = 0; n[i]; i++) w16(ipc.buf + 8 + 2 * i, (u8)n[i]);
            break;
        }
        case 0x0011:  // GetFriendKeyList
            ipc.ok(2, 2);
            ipc.set(2, 0);
            ipc.static_desc(3, 0, 0);
            break;
        default:
            unimplemented(ipc);
        }
    }
};

struct Ndm : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x000D:  // GetCurrentState
        case 0x000E:  // QueryStatus
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        default:
            ipc.ok();
        }
    }
};

struct Cecd : Service {
    using Service::Service;
    std::shared_ptr<Event> info = make_event(ONESHOT, "cecd info");
    std::shared_ptr<Event> state = make_event(ONESHOT, "cecd state");
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x000F:  // GetCecInfoEventHandle
            ipc.ok(1, 2);
            ipc.handles(2, {info});
            break;
        case 0x0010:  // GetChangeStateEventHandle
            ipc.ok(1, 2);
            ipc.handles(2, {state});
            break;
        case 0x000E:  // GetCecStateAbbreviated
            ipc.ok(2);
            ipc.set(2, 1);  // idle
            break;
        case 0x0001:  // Open -> fails: no message box
        case 0x0002:
            ipc.result(0xC8A11BFA, 2);
            ipc.set(2, 0);
            break;
        default:
            unimplemented(ipc);
        }
    }
};

struct Boss : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // InitializeSession
        case 0x0002:  // SetStorageInfo
        case 0x0003:  // UnregisterStorage
            ipc.ok();
            break;
        case 0x0004:  // GetStorageInfo
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x0006:  // GetOptoutFlag
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        case 0x000D: {  // GetTaskIdList
            ipc.ok();
            break;
        }
        default:
            unimplemented(ipc);
        }
    }
};

struct Mic : Service {
    using Service::Service;
    std::shared_ptr<Event> buf_full = make_event(ONESHOT, "mic");
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0007:  // GetBufferFullEvent
            ipc.ok(1, 2);
            ipc.handles(2, {buf_full});
            break;
        case 0x0004:  // IsSampling
        case 0x000B:  // GetPower
        case 0x0010:  // GetClamp
        case 0x0012:  // GetAllowShellClosed
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        default:
            ipc.ok();
        }
    }
};

// ir:USER with a Circle Pad Pro always attached (as a New 3DS emulates one for older games): the second circle pad
// (the host's right stick), ZL and ZR. The game talks to it through packets in a shared memory block: a header,
// then the receive buffer (its info, then a ring of packet infos and data); what it sends arrives with SendIrNop.
// Requests: 1 ConfigureHIDPolling (period in ms), 2 ReadCalibrationData (offset, size); the replies are 0x11
// (calibration data) and 0x10 (the pad's state, every period).
namespace {
enum : u32 {
    IR_HDR_STATUS = 0x08, IR_HDR_ROLE = 0x0A, IR_HDR_CONNECTED = 0x0C, IR_HDR_NETWORK_ID = 0x0D,
    IR_HDR_INITIALIZED = 0x0E,
    IR_RECV_INFO = 0x10,  // begin index, end index, packet count, unknown
    IR_RECV_RING = 0x20,  // packet infos (offset, size), then the data
};
enum : u8 { CPP_DEVICE_ID = 1, CPP_REQ_POLLING = 1, CPP_REQ_CALIBRATION = 2, CPP_RSP_PAD = 0x10, CPP_RSP_CALIBRATION = 0x11 };
constexpr u8 IR_PACKET_MAGIC = 0xA5;
constexpr u32 IR_STATUS_CONNECTED = 2, IR_ROLE_CONNECTED = 2;  // (values a New 3DS shows)
constexpr int CSTICK_CENTER = 0x800, CSTICK_RADIUS = 0x7FF;    // 12-bit axes
constexpr u8 CPP_BATTERY_FULL = 0x1F;
// A Circle Pad Pro's calibration block (centres and scales of the second circle pad)
const u8 cpp_calibration[0x40] = {
    0x00, 0x00, 0x08, 0x80, 0x85, 0xEB, 0x11, 0x3F, 0x85, 0xEB, 0x11, 0x3F, 0xFF, 0xFF, 0xFF, 0xF5,
    0xFF, 0x00, 0x08, 0x80, 0x85, 0xEB, 0x11, 0x3F, 0x85, 0xEB, 0x11, 0x3F, 0xFF, 0xFF, 0xFF, 0x65,
    0xFF, 0x00, 0x08, 0x80, 0x85, 0xEB, 0x11, 0x3F, 0x85, 0xEB, 0x11, 0x3F, 0xFF, 0xFF, 0xFF, 0x65,
    0xFF, 0x00, 0x08, 0x80, 0x85, 0xEB, 0x11, 0x3F, 0x85, 0xEB, 0x11, 0x3F, 0xFF, 0xFF, 0xFF, 0x65,
};

struct IrState {
    std::shared_ptr<Event> conn = make_event(ONESHOT, "ir conn");
    std::shared_ptr<Event> recv = make_event(ONESHOT, "ir recv");
    std::shared_ptr<Event> send = make_event(ONESHOT, "ir send");
    std::shared_ptr<SharedMem> shm;
    u32 packets = 0, ring_size = 0;  // receive buffer: packet slots, bytes (infos and data)
    bool connected = false;
    u32 period_ms = 0;  // the pad's polling period (0: not asked yet)
    u64 next_poll = 0;  // (system ticks)
    // receive buffer state
    u32 begin = 0, end = 0, count = 0;
};
IrState ir;

u32 ir_data_size() { return ir.ring_size - 8 * ir.packets; }
void ir_store_info() {
    const u32 b = ir.shm->backing + IR_RECV_INFO;
    w32(b, ir.begin);
    w32(b + 4, ir.end);
    w32(b + 8, ir.count);
}
u8 crc8(const std::vector<u8> &d) {
    u8 c = 0;
    for (u8 v : d) {
        c ^= v;
        for (int i = 0; i < 8; i++) {
            c = (c & 0x80) ? (u8)((c << 1) ^ 0x07) : (u8)(c << 1);
        }
    }
    return c;
}
// One packet from the device into the receive buffer: magic, network id, size (1 byte, or 2 with bit 6 set), the
// payload and a CRC-8 of all that
void ir_put(const std::vector<u8> &payload) {
    if (!ir.shm || !ir.packets) {
        return;
    }
    std::vector<u8> p = {IR_PACKET_MAGIC, r8(ir.shm->backing + IR_HDR_NETWORK_ID)};
    if (payload.size() < 0x40) {
        p.push_back((u8)payload.size());
    } else {
        p.push_back((u8)(0x40 | payload.size() >> 8));
        p.push_back((u8)payload.size());
    }
    p.insert(p.end(), payload.begin(), payload.end());
    p.push_back(crc8(p));
    const u32 infos = ir.shm->backing + IR_RECV_RING, data = infos + 8 * ir.packets, max = ir_data_size();
    if (ir.count == ir.packets) {
        return;  // full: the game has not released what it read
    }
    u32 off = 0;
    if (ir.count) {
        const u32 last = (ir.end + ir.packets - 1) % ir.packets;
        off = (r32(infos + 8 * last) + r32(infos + 8 * last + 4)) % max;
        const u32 free = (r32(infos + 8 * ir.begin) + max - off) % max;
        if (p.size() > free) {
            return;
        }
    } else if (p.size() > max) {
        return;
    }
    w32(infos + 8 * ir.end, off);
    w32(infos + 8 * ir.end + 4, (u32)p.size());
    for (size_t i = 0; i < p.size(); i++) {
        w8(data + (off + i) % max, p[i]);
    }
    ir.end = (ir.end + 1) % ir.packets;
    ir.count++;
    ir_store_info();
    ir.recv->signal();
}
void ir_disconnect() {
    ir.connected = false;
    ir.period_ms = 0;
    if (ir.shm) {
        w8(ir.shm->backing + IR_HDR_STATUS, 0);
        w8(ir.shm->backing + IR_HDR_ROLE, 0);
        w8(ir.shm->backing + IR_HDR_CONNECTED, 0);
    }
    ir.conn->signal();
}
// A request the game sent to the Circle Pad Pro
void cpp_request(const u8 *d, u32 n) {
    if (n == 3 && d[0] == CPP_REQ_POLLING) {
        ir.period_ms = d[1];
        ir.next_poll = 0;
    } else if (n == 6 && d[0] == CPP_REQ_CALIBRATION) {
        const u32 off = (d[2] | d[3] << 8) & ~15u, size = (d[4] | d[5] << 8) & ~15u;
        if (off + size > sizeof cpp_calibration) {
            LOG("[ir] calibration data read out of range (%u+%u)\n", off, size);
            return;
        }
        std::vector<u8> r = {CPP_RSP_CALIBRATION, d[2], d[3], d[4], d[5]};
        r.insert(r.end(), cpp_calibration + off, cpp_calibration + off + size);
        ir_put(r);
    } else {
        LOG("[ir] unknown Circle Pad Pro request %02x (%u bytes)\n", n ? d[0] : 0, n);
    }
}
}  // namespace

void ir_poll(u32 buttons, s16 sx, s16 sy) {
    if (!ir.connected || !ir.period_ms) {
        return;
    }
    const u64 now = system_tick();
    if (now < ir.next_poll) {
        return;
    }
    constexpr u64 TICKS_PER_MS = 268111856 / 1000;
    ir.next_poll = now + ir.period_ms * TICKS_PER_MS;
    constexpr u32 ZL = 1 << 14, ZR = 1 << 15;
    // the second circle pad: x in bits 8-19, y in 20-31 (sx, sy in -0x7FF..0x7FF)
    const u32 w = CPP_RSP_PAD | (u32)(CSTICK_CENTER + std::clamp<int>(sx, -CSTICK_RADIUS, CSTICK_RADIUS)) << 8 |
                  (u32)(CSTICK_CENTER + std::clamp<int>(sy, -CSTICK_RADIUS, CSTICK_RADIUS)) << 20;
    // battery level, then ZL, ZR and the Circle Pad Pro's R not held (active low)
    const u8 btn = CPP_BATTERY_FULL | (buttons & ZL ? 0 : 1 << 5) | (buttons & ZR ? 0 : 1 << 6) | 1 << 7;
    ir_put({(u8)w, (u8)(w >> 8), (u8)(w >> 16), (u8)(w >> 24), btn, 0});
}

struct Ir : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0002:  // FinalizeIrNop
            ir_disconnect();
            ir.shm.reset();
            ir.packets = 0;
            ipc.ok();
            break;
        case 0x0006:  // RequireConnection(device id)
            if (ir.shm && (u8)ipc[1] == CPP_DEVICE_ID) {
                w8(ir.shm->backing + IR_HDR_STATUS, IR_STATUS_CONNECTED);
                w8(ir.shm->backing + IR_HDR_ROLE, IR_ROLE_CONNECTED);
                w8(ir.shm->backing + IR_HDR_CONNECTED, 1);
                static bool told;
                if (!told) {
                    LOG("[ir] Circle Pad Pro connected\n");
                    told = true;
                }
                ir.connected = true;
                ir.conn->signal();
            }
            ipc.ok();
            break;
        case 0x0009:  // Disconnect
            ir_disconnect();
            ipc.ok();
            break;
        case 0x000A:  // GetReceiveEvent
            ipc.ok(1, 2);
            ipc.handles(2, {ir.recv});
            break;
        case 0x000B:  // GetSendEvent
            ipc.ok(1, 2);
            ipc.handles(2, {ir.send});
            break;
        case 0x000C:  // GetConnectionStatusEvent
            ipc.ok(1, 2);
            ipc.handles(2, {ir.conn});
            break;
        case 0x000D:  // SendIrNop(size, static buffer)
            if (!ir.connected) {
                ipc.result(0xC8A10414);  // not connected
                break;
            }
            cpp_request(gptr(ipc[3]), ipc[1]);
            ir.send->signal();
            ipc.ok();
            break;
        case 0x0018:  // InitializeIrNopShared(shared size, recv size, recv packets, send size, send packets, baud, shm)
            if (auto m = get<SharedMem>(ipc[8])) {
                ir.shm = m;
                ir.ring_size = ipc[2];
                ir.packets = ipc[3];
                ir.begin = ir.end = ir.count = 0;
                memset(gptr(m->backing), 0, IR_RECV_RING);
                w8(m->backing + IR_HDR_INITIALIZED, 1);
            }
            ipc.ok();
            break;
        case 0x0019:  // ReleaseReceivedData(count)
            if (ir.shm && ipc[1] <= ir.count) {
                ir.count -= ipc[1];
                ir.begin = (ir.begin + ipc[1]) % ir.packets;
                ir_store_info();
            }
            ipc.ok();
            break;
        default:
            ipc.ok();
        }
    }
};

struct Y2r : Service {
    using Service::Service;
    std::shared_ptr<Event> done = make_event(ONESHOT, "y2r");
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x000F:  // GetTransferEndEvent
            ipc.ok(1, 2);
            ipc.handles(2, {done});
            break;
        case 0x0026:  // StartConversion: we don't convert
            done->signal();
            ipc.ok();
            break;
        case 0x0027:  // IsBusyConversion
        case 0x0014:  // IsFinishedSendingYuv
        case 0x0015:
        case 0x0016:
        case 0x0017:
        case 0x0018:
            ipc.ok(2);
            ipc.set(2, ipc.cmd() == 0x0027 ? 0 : 1);
            break;
        case 0x0028:  // PingProcess
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        default:
            ipc.ok();
        }
    }
};

struct Generic : Service {
    using Service::Service;
    void handle(IPC &ipc) override { unimplemented(ipc); }
};

void register_misc() {
    register_service(std::make_shared<Ptm>("ptm:u"));
    register_service(std::make_shared<Ptm>("ptm:sysm"));
    register_service(std::make_shared<Ac>("ac:u"));
    register_service(std::make_shared<Ac>("ac:i"));
    register_service(std::make_shared<Frd>("frd:u"));
    register_service(std::make_shared<Frd>("frd:a"));
    register_service(std::make_shared<Ndm>("ndm:u"));
    register_service(std::make_shared<Cecd>("cecd:u"));
    register_service(std::make_shared<Boss>("boss:U"));
    register_service(std::make_shared<Mic>("mic:u"));
    register_service(std::make_shared<Ir>("ir:USER"));
    register_service(std::make_shared<Ir>("ir:u"));
    register_service(std::make_shared<Y2r>("y2r:u"));
    for (const char *n : {"news:u", "cam:u", "soc:U", "ssl:C", "http:C", "nwm::UDS", "dlp:FKCL", "dlp:SRVR",
                          "pxi:dev", "act:u", "am:app", "nim:aoc", "qtm:u", "csnd:SND"})
        register_service(std::make_shared<Generic>(n));
}

void register_all_services() {
    register_apt();
    register_fs();
    register_cfg();
    register_gsp();
    register_hid();
    register_dsp();
    register_misc();
    register_ro();
}

// ---- shared system font (APT GetSharedFont): the BCFNT from system archive 0004009B00014002, LZ11-decompressed
// after a 0x80-byte header (status 2 = loaded, region, size), with its section pointers relocated to the
// address where the game sees the block. The block is in the linear heap: the GPU samples the glyph sheets.
static std::vector<u8> lz11(const std::vector<u8> &d) {
    if (d.size() < 4 || d[0] != 0x11) return {};
    size_t size = d[1] | d[2] << 8 | d[3] << 16, p = 4;
    if (!size && d.size() >= 8) size = d[4] | d[5] << 8 | d[6] << 16 | (size_t)d[7] << 24, p = 8;
    std::vector<u8> out;
    out.reserve(size);
    while (out.size() < size && p < d.size()) {
        u8 flags = d[p++];
        for (int b = 0; b < 8 && out.size() < size && p < d.size(); b++) {
            if (!(flags & (0x80 >> b))) {
                out.push_back(d[p++]);
                continue;
            }
            size_t len, disp;
            u8 x = d[p];
            if (x >> 4 == 0) {
                len = ((x & 15) << 4 | d[p + 1] >> 4) + 0x11;
                disp = ((d[p + 1] & 15) << 8 | d[p + 2]) + 1;
                p += 3;
            } else if (x >> 4 == 1) {
                len = ((x & 15) << 12 | d[p + 1] << 4 | d[p + 2] >> 4) + 0x111;
                disp = ((d[p + 2] & 15) << 8 | d[p + 3]) + 1;
                p += 4;
            } else {
                len = (x >> 4) + 1;
                disp = ((x & 15) << 8 | d[p + 1]) + 1;
                p += 2;
            }
            if (disp > out.size()) return {};
            for (size_t i = 0; i < len && out.size() < size; i++) out.push_back(out[out.size() - disp]);
        }
    }
    return out;
}

// Rebases the section pointers of the font at addr + 0x80 so that they point inside the block at addr, and marks
// it resolved ("CFNU", as the system's copy is: the font library would otherwise add its address again).
static void relocate_shared_font(u32 addr) {
    u32 cfnt = addr + 0x80;
    if (memcmp(gptr(cfnt), "CFNT", 4) && memcmp(gptr(cfnt), "CFNU", 4)) return;
    memcpy(gptr(cfnt), "CFNU", 4);
    u32 nblocks = r32(cfnt + 0x10), first = cfnt + r16(cfnt + 6);
    u32 cmap = 0, cmap_ptr = 0, at = first;
    for (u32 i = 0; i < nblocks; i++, at += r32(at + 4)) {
        if (!cmap && !memcmp(gptr(at), "CMAP", 4)) cmap = at;
        if (!memcmp(gptr(at), "FINF", 4)) cmap_ptr = r32(at + 24);
    }
    if (!cmap || !cmap_ptr) return;
    u32 delta = (cmap + 8) - cmap_ptr;  // where the pointers assume the font is vs where it is
    if (!delta) return;
    at = first;
    for (u32 i = 0; i < nblocks; i++, at += r32(at + 4)) {
        auto fix = [&](u32 a) { if (r32(a)) w32(a, r32(a) + delta); };
        if (!memcmp(gptr(at), "FINF", 4)) { fix(at + 16); fix(at + 20); fix(at + 24); }
        else if (!memcmp(gptr(at), "TGLP", 4)) fix(at + 28);
        else if (!memcmp(gptr(at), "CWDH", 4)) fix(at + 12);
        else if (!memcmp(gptr(at), "CMAP", 4)) fix(at + 16);
    }
}

std::shared_ptr<SharedMem> load_shared_font() {
    std::vector<u8> font = lz11(system_archive_file(0x0004009B00014002ull, "/cbf_std.bcfnt.lz"));
    const u32 size = 0x332000;
    if (font.empty() || font.size() > size - 0x80) {
        LOG("[apt] shared font: system archive 0004009b00014002 missing or unreadable (%zu bytes)\n", font.size());
        return nullptr;
    }
    auto sm = std::make_shared<SharedMem>();
    sm->name = "shared font";
    sm->size = size;
    sm->backing = alloc_linear(size);
    if (!sm->backing) {
        LOG("[apt] shared font: no linear memory left\n");
        return nullptr;
    }
    memset(gptr(sm->backing), 0, size);
    memcpy(gptr(sm->backing + 0x80), font.data(), font.size());
    w32(sm->backing, 2);  // loaded
    w32(sm->backing + 4, 1);
    w32(sm->backing + 8, (u32)font.size());
    relocate_shared_font(sm->backing);
    sm->on_move = relocate_shared_font;
    LOG("[apt] shared font: %zu bytes at %08x\n", font.size(), sm->backing);
    return sm;
}

}  // namespace kernel
