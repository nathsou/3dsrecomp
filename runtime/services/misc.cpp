// Small services: ptm, ndm, ac, frd, boss, cecd, news, ir, mic, cam, y2r, ldr:ro, soc, ssl, http, ...
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

struct Ir : Service {
    using Service::Service;
    std::shared_ptr<Event> conn = make_event(ONESHOT, "ir conn");
    std::shared_ptr<Event> recv = make_event(ONESHOT, "ir recv");
    std::shared_ptr<Event> send = make_event(ONESHOT, "ir send");
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x000C:  // GetConnectionStatusEvent
            ipc.ok(1, 2);
            ipc.handles(2, {conn});
            break;
        case 0x000D:  // GetReceiveEvent
            ipc.ok(1, 2);
            ipc.handles(2, {recv});
            break;
        case 0x000E:  // GetSendEvent
            ipc.ok(1, 2);
            ipc.handles(2, {send});
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
