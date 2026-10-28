// gsp::Gpu: GPU access rights, the interrupt relay queue, GX commands and framebuffer swaps.
// GX commands run asynchronously on the GPU (host main) thread, which signals their interrupts.
#include <condition_variable>
#include <deque>
#include <sys/stat.h>

#include "../gpu_iface.h"
#include "../service.h"
#include "services.h"

namespace kernel {

enum Irq { PSC0 = 0, PSC1 = 1, PDC0 = 2, PDC1 = 3, PPF = 4, P3D = 5, DMA = 6 };

namespace {
struct GspState {
    std::shared_ptr<SharedMem> shm;
    std::shared_ptr<Event> irq_event[4];
    bool registered[4] = {};
    u32 rights_thread = 0;
    u32 next_thread = 0;
    // LCD registers: what is on screen
    FramebufferInfo fb[2] = {};
    // GX queue (guarded by its own mutex: the GPU thread consumes it without the kernel lock)
    std::mutex qm;
    std::condition_variable qcv;
    std::deque<std::array<u32, 8>> queue;
    int busy = 0;
};
GspState gsp;
std::mutex fb_mutex;
}  // namespace

static u32 shm_base() { return gsp.shm ? gsp.shm->backing : 0; }

// (kernel lock held)
static void signal_irq_now(Irq id);
static void signal_irq(Irq id) {
    run_when_quiescent([id] { signal_irq_now(id); });
}

static void signal_irq_now(Irq id) {
    u32 base = shm_base();
    if (!base) return;
    bool pdc = id == PDC0 || id == PDC1;
    for (u32 t = 0; t < 4; t++) {
        if (!gsp.registered[t] || !gsp.irq_event[t]) continue;
        if (!pdc && t != gsp.rights_thread) continue;
        u32 q = base + t * 0x40;
        u8 index = r8(q), n = r8(q + 1), flags = r8(q + 3);
        if (pdc && (flags & 1)) continue;
        if (pdc && n >= 0x20) {
            w32(q + 4 + (id == PDC0 ? 0 : 4), r32(q + 4 + (id == PDC0 ? 0 : 4)) + 1);
            continue;
        }
        if (n >= 0x34) {
            w8(q + 2, 1);
            continue;
        }
        u8 next = (u8)((index + n) % 0x34);
        w8(q + 0xC + next, (u8)id);
        w8(q + 1, n + 1);
        gsp.irq_event[t]->signal();
    }
}

// framebuffer update from the shared memory (at VBlank), like the GSP module does
static void update_framebuffer(int screen) {
    u32 base = shm_base();
    if (!base) return;
    u32 info = base + 0x200 + gsp.rights_thread * 0x80 + screen * 0x40;
    u8 idx = r8(info), dirty = r8(info + 1);
    if (!dirty) return;
    w8(info + 1, 0);
    u32 fbi = info + 4 + (idx & 1) * 0x1C;
    std::lock_guard<std::mutex> lk(fb_mutex);
    FramebufferInfo &f = gsp.fb[screen];
    if (f.format != r32(fbi + 16) || f.stride != r32(fbi + 12))
        VLOG("[gsp] screen %d framebuffer %08x stride %x format %x\n", screen, r32(fbi + 4), r32(fbi + 12), r32(fbi + 16));
    f.addr_left = r32(fbi + 4);
    f.addr_right = r32(fbi + 8);
    f.stride = r32(fbi + 12);
    f.format = r32(fbi + 16);
    f.valid = f.addr_left != 0;
}

FramebufferInfo gsp_framebuffer(int screen) {
    std::lock_guard<std::mutex> lk(fb_mutex);
    return gsp.fb[screen];
}

void gsp_frame_boundary();
void gsp_vblank() {
    gsp_frame_boundary();
    Lock lk(kmutex);
    update_framebuffer(0);
    update_framebuffer(1);
    signal_irq(PDC0);
    signal_irq(PDC1);
    schedule_async();
}


bool gsp_gx_pending() {
    std::lock_guard<std::mutex> lk(gsp.qm);
    return !gsp.queue.empty();
}

static u32 va2pa(u32 va) { return mem::vaddr_to_paddr(va); }

GxStats gx_stats;

// ---- GX dumps for the GPU replay tool: RECOMP_DUMP=<frame> dumps the GX work of that frame
static u64 dump_frame = ~0ull, cur_frame;
static int dump_state, dump_frames_left;  // 0 idle, 1 armed (waiting for first op), 2 recording
static FILE *dump_ops;
static std::string dump_dir;

static void dump_write(const std::string &path, u32 va, u32 size) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(gptr(va), 1, size, f);
    fclose(f);
}

static void dump_begin() {
    dump_dir = "dump_" + std::to_string(cur_frame);
    mkdir(dump_dir.c_str(), 0755);
    dump_write(dump_dir + "/fcram.bin", mem::LINEAR_BASE, mem::LINEAR_SIZE);
    dump_write(dump_dir + "/vram.bin", mem::VRAM_BASE, mem::VRAM_SIZE);
    gpu::save_state((dump_dir + "/gpustate.bin").c_str());
    dump_ops = fopen((dump_dir + "/ops.txt").c_str(), "w");
    dump_state = 2;
    dump_frames_left = 3;  // this frame and the next two, so the frame's own display transfer is included
    LOG("[gsp] dumping frame %llu to %s\n", (unsigned long long)cur_frame, dump_dir.c_str());
}

void gsp_frame_boundary() {
    cur_frame++;
    if (dump_frame == ~0ull) {
        const char *d = getenv("RECOMP_DUMP");
        dump_frame = d ? strtoull(d, nullptr, 10) : ~1ull;
    }
    if (dump_state == 2 && --dump_frames_left == 0) {
        for (int s = 0; s < 2; s++) {
            FramebufferInfo f = gsp_framebuffer(s);
            fprintf(dump_ops, "present %x %d %d %x %x\n", mem::vaddr_to_paddr(f.addr_left), 240, s ? 320 : 400, f.stride,
                    f.format);
        }
        fclose(dump_ops);
        dump_state = 0;
        LOG("[gsp] dump done\n");
    }
    if (cur_frame == dump_frame) dump_state = 1;
}

// (GPU thread) executes queued GX commands
void gsp_process_gx() {
    for (;;) {
        std::array<u32, 8> c;
        {
            std::lock_guard<std::mutex> lk(gsp.qm);
            if (gsp.queue.empty()) return;
            c = gsp.queue.front();
            gsp.queue.pop_front();
        }
        std::vector<Irq> irqs;
        if (dump_state == 1) dump_begin();
        if (dump_state == 2) {
            switch (c[0] & 0xFF) {
            case 1: fprintf(dump_ops, "cmdlist %x %x\n", va2pa(c[1]), c[2]); break;
            case 2:
                if (c[1]) fprintf(dump_ops, "fill %x %x %x %x\n", va2pa(c[1]), va2pa(c[1]) + (c[3] - c[1]), c[2], c[7] & 0xFFFF);
                if (c[4]) fprintf(dump_ops, "fill %x %x %x %x\n", va2pa(c[4]), va2pa(c[4]) + (c[6] - c[4]), c[5], c[7] >> 16);
                break;
            case 3: fprintf(dump_ops, "transfer %x %x %x %x %x\n", va2pa(c[1]), va2pa(c[2]), c[3], c[4], c[5]); break;
            case 4: fprintf(dump_ops, "copy %x %x %x %x %x %x\n", va2pa(c[1]), va2pa(c[2]), c[3], c[4], c[5], c[6]); break;
            }
        }
        switch (c[0] & 0xFF) {
        case 0x00:  // RequestDma(src, dst, size)
            memmove(gptr(c[2]), gptr(c[1]), c[3]);
            if (u32 pa = va2pa(c[2])) gpu::invalidate_memory(pa, c[3]);
            irqs.push_back(DMA);
            break;
        case 0x01: {  // SubmitGpuCommandList(addr, size, flags)
            u64 t0 = real_ns();
            gpu::process_command_list(va2pa(c[1]), c[2]);
            gx_stats.cmdlists++;
            gx_stats.gpu_ns += real_ns() - t0;
            irqs.push_back(P3D);
            break;
        }
        case 0x02: {  // MemoryFill
            u32 s1 = c[1], v1 = c[2], e1 = c[3], s2 = c[4], v2 = c[5], e2 = c[6];
            u32 ctl1 = c[7] & 0xFFFF, ctl2 = c[7] >> 16;
            if (s1) gpu::memory_fill(va2pa(s1), va2pa(s1) + (e1 - s1), v1, ctl1);
            if (s2) gpu::memory_fill(va2pa(s2), va2pa(s2) + (e2 - s2), v2, ctl2);
            if (s1 && s2) irqs.push_back(PSC0);
            else if (s1) irqs.push_back(PSC0);
            else if (s2) irqs.push_back(PSC1);
            break;
        }
        case 0x03:  // DisplayTransfer
            gx_stats.transfers++;
            gpu::display_transfer(va2pa(c[1]), va2pa(c[2]), c[3], c[4], c[5]);
            irqs.push_back(PPF);
            break;
        case 0x04:  // TextureCopy
            gpu::texture_copy(va2pa(c[1]), va2pa(c[2]), c[3], c[4], c[5], c[6]);
            irqs.push_back(PPF);
            break;
        case 0x05:  // FlushCacheRegions
            break;
        case 0xFF:  // (ours) the CPU wrote this range
            if (u32 pa = va2pa(c[1])) gpu::invalidate_memory(pa, c[2]);
            break;
        default:
            LOG("[gsp] unknown GX command %08x\n", c[0]);
        }
        Lock lk(kmutex);
        if ((c[0] & 0xFF) == 0x04) {
            // Texture copies complete late: Super Mario 3D Land hangs after the title screen otherwise
            // (the same workaround as Citra's delay_texture_copy_completion).
            add_timeout(now_ns() + 1000000, [] { signal_irq(PPF); });
            irqs.clear();
        }
        for (Irq i : irqs) signal_irq(i);
        {
            std::lock_guard<std::mutex> ql(gsp.qm);
            gsp.busy--;
        }
        schedule_async();
    }
}

void frontend_wake();

struct GspGpu : Service {
    using Service::Service;
    void write_hw_regs(u32 reg, u32 size, u32 src) {
        for (u32 i = 0; i < size; i += 4) {
            u32 r = reg + i, v = r32(src + i);
            for (int s = 0; s < 2; s++) {
                u32 b = 0x400 + s * 0x100;
                std::lock_guard<std::mutex> lk(fb_mutex);
                FramebufferInfo &f = gsp.fb[s];
                if (r == b + 0x68) { f.addr_left = mem::paddr_to_vaddr(v); f.valid = true; }
                if (r == b + 0x94) f.addr_right = mem::paddr_to_vaddr(v);
                if (r == b + 0x70) f.format = v;
                if (r == b + 0x90) f.stride = v;
            }
        }
    }
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // WriteHWRegs(reg, size, static buffer)
            write_hw_regs(ipc[1], ipc[2], ipc[4]);
            ipc.ok();
            break;
        case 0x0002:  // WriteHWRegsWithMask
            write_hw_regs(ipc[1], ipc[2], ipc[4]);
            ipc.ok();
            break;
        case 0x0004: {  // ReadHWRegs(reg, size) -> static buffer 0
            u32 size = ipc[2];
            u32 dst = ipc.static_buffer(0);
            memset(gptr(dst), 0, size);
            ipc.ok(1, 2);
            ipc.static_desc(2, 0, size);
            break;
        }
        case 0x0005: {  // SetBufferSwap(screen, FrameBufferInfo)
            u32 screen = ipc[1];
            std::lock_guard<std::mutex> lk(fb_mutex);
            FramebufferInfo &f = gsp.fb[screen & 1];
            f.addr_left = ipc[3];
            f.addr_right = ipc[4];
            f.stride = ipc[5];
            f.format = ipc[6];
            f.valid = true;
            ipc.ok();
            break;
        }
        case 0x0008:    // FlushDataCache(addr, size, process)
        case 0x0009: {  // InvalidateDataCache
            // queued so the GPU thread sees it in order with the GX commands
            std::array<u32, 8> c{0xFF, ipc[1], ipc[2]};
            {
                std::lock_guard<std::mutex> lk(gsp.qm);
                gsp.queue.push_back(c);
                gsp.busy++;
            }
            ipc.ok();
            break;
        }
        case 0x000B:  // SetLcdForceBlack
        case 0x0010:  // SetAxiConfigQoSMode
        case 0x0016:  // AcquireRight
        case 0x0017:  // ReleaseRight
        case 0x0019:  // SaveVramSysArea
        case 0x001A:  // RestoreVramSysArea
        case 0x001E:  // SetInternalPriorities
        case 0x001F:  // StoreDataCache
            ipc.ok();
            break;
        case 0x000C:  // TriggerCmdReqQueue
            trigger(ipc);
            ipc.ok();
            break;
        case 0x0013: {  // RegisterInterruptRelayQueue(flags, event handle)
            auto ev = get<Event>(ipc[3]);
            u32 tid = gsp.next_thread++ & 3;
            gsp.irq_event[tid] = ev;
            gsp.registered[tid] = true;
            if (!gsp.shm) {
                gsp.shm = std::make_shared<SharedMem>();
                gsp.shm->name = "GSP";
                gsp.shm->size = 0x1000;
                gsp.shm->backing = mem::hle_alloc(0x1000);
            }
            if (tid == 0) gsp.rights_thread = 0;
            ipc.result(tid == 0 ? 0x2A07 : 0, 2, 2);
            ipc.set(2, tid);
            ipc.handles(3, {gsp.shm});
            break;
        }
        case 0x0014:  // UnregisterInterruptRelayQueue
            ipc.ok();
            break;
        case 0x0018: {  // ImportDisplayCaptureInfo
            ipc.ok(9);
            for (int s = 0; s < 2; s++) {
                FramebufferInfo f = gsp_framebuffer(s);
                ipc.set(2 + s * 4, f.addr_left);
                ipc.set(3 + s * 4, f.addr_right);
                ipc.set(4 + s * 4, f.format);
                ipc.set(5 + s * 4, f.stride);
            }
            break;
        }
        default:
            unimplemented(ipc);
        }
    }
    void trigger(IPC &ipc) {
        u32 base = shm_base();
        if (!base) return;
        u32 cb = base + 0x800 + gsp.rights_thread * 0x200;
        for (;;) {
            u32 hdr = r32(cb);
            u32 index = hdr & 0xFF, n = (hdr >> 8) & 0xFF;
            if (!n) break;
            if ((hdr >> 24) & 0xFF) {  // should stop
                w32(cb, (hdr & 0xFF00FFFF) | (1 << 16));
                break;
            }
            std::array<u32, 8> c;
            for (int i = 0; i < 8; i++) c[i] = r32(cb + 0x20 + index * 0x20 + 4 * i);
            n--;
            index = (index + 1) % 0xF;
            w32(cb, (hdr & 0xFFFF0000) | (n << 8) | index);
            {
                std::lock_guard<std::mutex> lk(gsp.qm);
                gsp.queue.push_back(c);
                gsp.busy++;
            }
            if (g_verbose >= 2) LOG("[gsp] GX %08x %08x %08x %08x\n", c[0], c[1], c[2], c[3]);
        }
        frontend_wake();
    }
};

void register_gsp() {
    register_service(std::make_shared<GspGpu>("gsp::Gpu"));
}

}  // namespace kernel
