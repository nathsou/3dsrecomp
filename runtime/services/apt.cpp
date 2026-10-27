// APT: applet manager. Only the application slot exists; the home menu and library applets are absent.
#include <deque>

#include "../service.h"
#include "services.h"

namespace kernel {

namespace {
struct Parameter {
    u32 sender = 0, dest = 0x300, signal = 0;
    ObjPtr object;
    std::vector<u8> buffer;
};

struct AptState {
    std::shared_ptr<Mutex> lock = std::make_shared<Mutex>();
    std::shared_ptr<Event> notif = make_event(ONESHOT, "APT notification");
    std::shared_ptr<Event> param_ev = make_event(ONESHOT, "APT parameter");
    std::deque<Parameter> params;
    std::deque<u32> notifications;
    bool initialized = false;
    u32 cpu_limit = 0;
    u32 screencap = 0;
    std::shared_ptr<SharedMem> font;
    u32 lib_applet = 0;
};
AptState apt;
}  // namespace

static void send_parameter(Parameter p) {
    apt.params.push_back(std::move(p));
    apt.param_ev->signal();
}

// Library applets run as HLE: they answer the capture-buffer request and finish at once.
static void applet_receive(u32 id, u32 signal, u32 ptr, u32 size) {
    if (signal == 2) {  // Request: wants a shared memory block for the screen capture
        u32 cap = size >= 4 ? r32(ptr) : 0x1000;
        if (!cap) cap = 0x1000;
        static std::shared_ptr<SharedMem> capture;  // one block, reused by every launch (nothing reads it)
        if (!capture || capture->size < cap) {
            capture = std::make_shared<SharedMem>();
            capture->name = "applet capture";
            capture->size = cap;
            capture->backing = mem::hle_alloc(cap);
        }
        send_parameter({id, 0x300, 3, capture, {}});  // Response
    }
}

// The Mii selector returns a standard Mii: a plain male Mii named "Mario" (made-up console/Mii IDs and a
// placeholder MAC; games mostly check the flags and the checksum)
static std::vector<u8> mii_selector_result() {
    std::vector<u8> r(0x84, 0);
    auto be32 = [&](size_t o, u32 v) { r[o] = v >> 24; r[o + 1] = v >> 16; r[o + 2] = v >> 8; r[o + 3] = v; };
    auto be16 = [&](size_t o, u16 v) { r[o] = v >> 8; r[o + 1] = (u8)v; };
    be32(0, 0);           // return code: a Mii was selected
    be32(4, 0);           // not a guest Mii
    be32(8, 0xFFFFFFFF);  // guest index
    size_t m = 0x0C;
    r[m + 0] = 0x03;
    r[m + 1] = 0x01;  // copying allowed (games that make look-alikes, e.g. Tomodachi Life, refuse it otherwise)
    r[m + 2] = 0x10;
    r[m + 3] = 0x30;
    u64 sys = 0x3D5000000000C0DEull;  // console id of the "author"
    for (int i = 0; i < 8; i++) r[m + 4 + i] = (u8)(sys >> (56 - 8 * i));
    be32(m + 0x0C, 0x98000001);  // Mii id: normal Mii flags, creation time 1
    const u8 mac[6] = {0x02, 0x00, 0x00, 0x00, 0x00, 0x01};  // locally administered placeholder
    memcpy(&r[m + 0x10], mac, 6);
    be16(m + 0x18, 0xA600);
    const char *name = "Mario";
    for (int i = 0; name[i]; i++) r[m + 0x1A + 2 * i] = (u8)name[i];
    r[m + 0x2E] = 0x40;
    r[m + 0x2F] = 0x40;
    r[m + 0x32] = 0x21;
    r[m + 0x33] = 0x01;
    be32(m + 0x34, 0x02684418);
    be32(m + 0x38, 0x26344614);
    be16(m + 0x3C, 0x8112);
    be16(m + 0x3E, 0x1768);
    be16(m + 0x40, 0x0D00);
    be16(m + 0x42, 0x0029);
    be16(m + 0x44, 0x0052);
    be16(m + 0x46, 0x4850);
    const char *author = "Player";
    for (int i = 0; author[i]; i++) r[m + 0x48 + 2 * i] = (u8)author[i];
    // CRC-16/XMODEM over the Mii data and padding
    u16 crc = 0;
    for (size_t i = 0; i < 0x5E; i++) {
        crc ^= (u16)r[m + i] << 8;
        for (int b = 0; b < 8; b++) crc = (crc & 0x8000) ? (u16)((crc << 1) ^ 0x1021) : (u16)(crc << 1);
    }
    be16(m + 0x5E, crc);
    return r;
}

// Software keyboard (SwkbdState, 0x400 bytes): the user "types" RECOMP_SWKBD_TEXT (else the initial text, else
// "Player"; digits for the numpad) and presses the confirm button. The text goes into the shared memory block
// the application passed, as UTF-16 at offset 0.
static void swkbd_run(std::vector<u8> &cfg, ObjPtr mem) {
    if (cfg.size() < 0x14C) return;
    auto rd32 = [&](size_t o) { u32 v; memcpy(&v, &cfg[o], 4); return v; };
    auto wr32 = [&](size_t o, u32 v) { memcpy(&cfg[o], &v, 4); };
    auto sm = std::dynamic_pointer_cast<SharedMem>(mem);
    u32 type = rd32(0x00), buttons = rd32(0x04) + 1, max_len = cfg[0x20] | cfg[0x21] << 8;
    std::u16string text;
    if (const char *t = getenv("RECOMP_SWKBD_TEXT")) {
        for (const char *c = t; *c; c++) text += (char16_t)(u8)*c;
    } else if (sm && (s32)rd32(0x120) >= 0 && rd32(0x120) + 2 <= sm->size) {
        for (u32 a = sm->backing + rd32(0x120); a + 2 <= sm->backing + sm->size && r16(a); a += 2) text += (char16_t)r16(a);
    }
    if (text.empty()) text = type == 2 ? u"1" : u"Player";
    if (max_len && text.size() > max_len) text.resize(max_len);
    if (sm && (text.size() + 1) * 2 <= sm->size) {
        for (size_t i = 0; i < text.size(); i++) w16(sm->backing + 2 * (u32)i, text[i]);
        w16(sm->backing + 2 * (u32)text.size(), 0);
    }
    wr32(0x138, buttons >= 3 ? 5 : buttons == 2 ? 2 : 0);  // D2_CLICK2 / D1_CLICK1 / D0_CLICK: the confirm button
    wr32(0x144, 0);                                         // text_offset
    cfg[0x148] = (u8)text.size();
    cfg[0x149] = (u8)(text.size() >> 8);
    wr32(0x14C, 0);  // callback_result: OK
    std::string ascii(text.begin(), text.end());
    LOG("[apt] software keyboard: typed \"%s\"\n", ascii.c_str());
}

static void applet_run(u32 id, std::vector<u8> buf, ObjPtr mem = nullptr) {
    u32 kind = id & 0xFF;
    if (kind == 0x02) {
        LOG("[apt] Mii selector: returning the standard Mii\n");
        send_parameter({id, 0x300, 10, nullptr, mii_selector_result()});
        return;
    }
    if (kind == 0x06 && buf.size() >= 8) {
        // error/EULA applet: return code 0 (success). ErrEulaConfig.return_code is at offset 0xF0C? keep the
        // buffer and clear the result fields the caller checks
        LOG("[apt] error applet: type %u code %08x\n", buf[0], *(u32 *)&buf[4]);
    } else if (kind == 0x01) {
        swkbd_run(buf, mem);
    } else {
        LOG("[apt] library applet %x: finishing immediately\n", id);
    }
    send_parameter({id, 0x300, 10, nullptr, std::move(buf)});  // WakeupByExit
}

void apt_notify(u32 signal) {
    apt.notifications.push_back(signal);
    apt.notif->signal();
}

struct AptService : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // GetLockHandle(attributes)
            ipc.ok(3, 2);
            ipc.set(2, ipc[1]);
            ipc.set(3, 0);
            ipc.handles(4, {apt.lock});
            break;
        case 0x0002:  // Initialize(app_id, attributes)
            ipc.ok(1, 3);
            set_handles(ipc);
            if (!apt.initialized) {
                apt.initialized = true;
                send_parameter({0, 0x300, 1, nullptr, {}});  // Wakeup
            }
            break;
        case 0x0003:  // Enable
        case 0x0004:  // Finalize
            ipc.ok();
            break;
        case 0x0005:  // GetAppletManInfo(pos)
            ipc.ok(5);
            ipc.set(2, 0);       // active applet pos: app
            ipc.set(3, 0);       // requested applet id
            ipc.set(4, 0x101);   // home menu
            ipc.set(5, 0x300);   // active: application
            break;
        case 0x0006: {  // GetAppletInfo(app_id)
            ipc.ok(7);
            ipc.set64(2, 0x0004003000008F02ull);
            ipc.set(4, 0);  // media type
            ipc.set(5, 1);  // registered
            ipc.set(6, 1);  // loaded
            ipc.set(7, 0);  // attributes
            break;
        }
        case 0x0009:  // IsRegistered(app_id)
            ipc.ok(2);
            ipc.set(2, ipc[1] == 0x300 || ipc[1] == 0x101 || (apt.lib_applet && ipc[1] == apt.lib_applet) ? 1 : 0);
            break;
        case 0x000B:  // InquireNotification
            ipc.ok(2);
            ipc.set(2, apt.notifications.empty() ? 0 : apt.notifications.front());
            if (!apt.notifications.empty()) apt.notifications.pop_front();
            break;
        case 0x000C: {  // SendParameter(src, dst, signal, size, handle, buffer)
            u32 dst = ipc[2], signal = ipc[3], size = ipc[4];
            LOG("[apt] SendParameter src=%x dst=%x signal=%x size=%x\n", ipc[1], dst, signal, size);
            if ((dst & 0xF00) == 0x200 || (dst & 0xF00) == 0x400) applet_receive(dst, signal, ipc[8], size);
            ipc.ok();
            break;
        }
        case 0x0016:  // PreloadLibraryApplet(id)
        case 0x0018:  // PrepareToStartLibraryApplet(id)
            LOG("[apt] %s library applet %x\n", ipc.cmd() == 0x16 ? "preload" : "prepare", ipc[1]);
            apt.lib_applet = ipc[1];
            ipc.ok();
            break;
        case 0x0017:  // FinishPreloadingLibraryApplet
        case 0x0022:  // PrepareToCloseApplication(cancel preload)
        case 0x0027:  // CloseApplication(size, handle, buffer): ExitProcess follows
            ipc.ok();
            break;
        case 0x001E: {  // StartLibraryApplet(id, size, handle, buffer)
            u32 id = ipc[1], size = ipc[2], src = ipc[6];
            LOG("[apt] StartLibraryApplet %x size %x\n", id, size);
            std::vector<u8> buf(gptr(src), gptr(src) + size);
            applet_run(id, buf, get_object(ipc[4]));
            ipc.ok();
            break;
        }
        case 0x000D:    // ReceiveParameter(app_id, size)
        case 0x000E: {  // GlanceParameter
            if (apt.params.empty()) {
                ipc.result(0xC8A0CFEF);  // no data
                break;
            }
            Parameter p = apt.params.front();
            if (ipc.cmd() == 0x000D) apt.params.pop_front();
            u32 max = ipc[2];
            u32 n = std::min<u32>(max, (u32)p.buffer.size());
            ipc.ok(4, 4);
            ipc.set(2, p.sender);
            ipc.set(3, p.signal);
            ipc.set(4, n);
            ipc.handles(5, {p.object}, true);
            u32 dst = ipc.static_buffer(0);
            if (n) memcpy(gptr(dst), p.buffer.data(), n);
            ipc.static_desc(7, 0, n);
            break;
        }
        case 0x000F:  // CancelParameter
            ipc.ok(2);
            ipc.set(2, 1);
            break;
        case 0x003E:  // ReplySleepQuery (APT:U numbering)
        case 0x0040:  // SendCaptureBufferInfo
        case 0x0043:  // NotifyToWait
        case 0x003B:  // ReplySleepQuery
        case 0x003C:  // ReplySleepNotificationComplete
        case 0x0055:  // SetScreenCapPostPermission
            if (ipc.cmd() == 0x0055) apt.screencap = ipc[1];
            ipc.ok();
            break;
        case 0x0056:
            ipc.ok(2);
            ipc.set(2, apt.screencap);
            break;
        case 0x0044:  // GetSharedFont
            get_shared_font(ipc);
            break;
        case 0x004B: {  // AppletUtility(cmd, in_size, out_size, in static buffer)
            u32 out_size = ipc[3];
            u32 dst = ipc.static_buffer(0);
            if (out_size) memset(gptr(dst), 0, out_size);
            if (ipc[1] == 6 && out_size) w8(dst, 1);  // TryLockTransition: allowed
            ipc.ok(2, 2);
            ipc.set(2, 0);
            ipc.static_desc(3, 0, out_size);
            break;
        }
        case 0x004F:  // SetAppCpuTimeLimit
            apt.cpu_limit = ipc[2];
            ipc.ok();
            break;
        case 0x0050:
            ipc.ok(2);
            ipc.set(2, apt.cpu_limit);
            break;
        case 0x004D: {  // GetStartupArgument(size, type)
            u32 size = ipc[1];
            u32 dst = ipc.static_buffer(0);
            if (size) memset(gptr(dst), 0, std::min<u32>(size, 0x1000));
            ipc.ok(2, 2);
            ipc.set(2, 0);
            ipc.static_desc(3, 0, std::min<u32>(size, 0x1000));
            break;
        }
        case 0x0101:  // CheckNew3DSApp
        case 0x0102:  // CheckNew3DS
            ipc.ok(2);
            ipc.set(2, 0);
            break;
        default:
            unimplemented(ipc);
        }
    }
    void set_handles(IPC &ipc) {
        ipc.set(2, 1 << 26);  // copy 2 handles
        ipc.set(3, add_handle(apt.notif));
        ipc.set(4, add_handle(apt.param_ev));
    }
    void get_shared_font(IPC &ipc) {
        if (!apt.font) {
            apt.font = load_shared_font();
        }
        if (!apt.font) {
            LOG("[apt] GetSharedFont: no shared font available\n");
            ipc.result(0xC8A0CFEF);
            return;
        }
        ipc.ok(2, 2);
        ipc.set(2, apt.font->backing);
        ipc.handles(3, {apt.font});
    }
};

void register_apt() {
    register_service(std::make_shared<AptService>("APT:U"));
    register_service(std::make_shared<AptService>("APT:A"));
    register_service(std::make_shared<AptService>("APT:S"));
}

}  // namespace kernel
