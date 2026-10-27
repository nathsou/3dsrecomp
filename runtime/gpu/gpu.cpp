// PICA200 command processing, register side effects, thread pool and shared helpers.
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "gpu/pica.h"

namespace gpu {

GpuState g;
bool g_stats = false, g_trace = false;
double g_t_shade = 0, g_t_setup = 0, g_t_flush = 0;
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- helpers
float float_from_raw(u32 hex, u32 M, u32 E) {
    const s32 bias = 127 - ((1 << (E - 1)) - 1);
    const u32 mmask = (1u << M) - 1, emask = (1u << E) - 1;
    s32 exponent = (hex >> M) & emask;
    u32 mantissa = hex & mmask;
    const u32 sign = ((hex >> (E + M)) & 1) << 31;
    u32 out;
    if (exponent == 0 && mantissa == 0) {
        out = sign;
    } else if (exponent == 0) {
        exponent = bias + 1;
        while ((mantissa & (1u << M)) == 0) {
            exponent--;
            mantissa <<= 1;
        }
        mantissa &= mmask;
        out = sign | ((u32)exponent << 23) | (mantissa << (23 - M));
    } else if ((u32)exponent == emask) {
        out = sign | (0xFFu << 23) | (mantissa << (23 - M));
    } else {
        out = sign | ((u32)(bias + exponent) << 23) | (mantissa << (23 - M));
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

u64 hash_bytes(const u8 *p, size_t n) {
    const u64 k1 = 0x9E3779B97F4A7C15ull, k2 = 0xC2B2AE3D27D4EB4Full;
    u64 h0 = n * k1, h1 = k2, h2 = 0x165667B19E3779F9ull, h3 = 0x27D4EB2F165667C5ull;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        u64 a, b, c, d;
        memcpy(&a, p + i, 8);
        memcpy(&b, p + i + 8, 8);
        memcpy(&c, p + i + 16, 8);
        memcpy(&d, p + i + 24, 8);
        h0 = (h0 ^ a) * k1;
        h1 = (h1 ^ b) * k2;
        h2 = (h2 ^ c) * k1;
        h3 = (h3 ^ d) * k2;
        h0 = (h0 << 31) | (h0 >> 33);
        h1 = (h1 << 29) | (h1 >> 35);
        h2 = (h2 << 27) | (h2 >> 37);
        h3 = (h3 << 33) | (h3 >> 31);
    }
    for (; i < n; i++) h0 = (h0 ^ p[i]) * k1;
    u64 h = h0 ^ (h1 * 3) ^ (h2 * 5) ^ (h3 * 7);
    h ^= h >> 33;
    h *= k2;
    h ^= h >> 29;
    return h;
}

static inline u32 c5to8(u32 v) { return (v << 3) | (v >> 2); }
static inline u32 c6to8(u32 v) { return (v << 2) | (v >> 4); }
static inline u32 c4to8(u32 v) { return (v << 4) | v; }

u32 decode_pixel(u32 fmt, const u8 *p) {
    switch (fmt) {
    case 0: return rgba(p[3], p[2], p[1], p[0]);
    case 1: return rgba(p[2], p[1], p[0], 255);
    case 2: {
        u32 v = p[0] | (p[1] << 8);
        return rgba(c5to8(v >> 11), c6to8((v >> 5) & 63), c5to8(v & 31), 255);
    }
    case 3: {
        u32 v = p[0] | (p[1] << 8);
        return rgba(c5to8(v >> 11), c5to8((v >> 6) & 31), c5to8((v >> 1) & 31), (v & 1) * 255);
    }
    case 4: {
        u32 v = p[0] | (p[1] << 8);
        return rgba(c4to8(v >> 12), c4to8((v >> 8) & 15), c4to8((v >> 4) & 15), c4to8(v & 15));
    }
    default: return 0;
    }
}

void encode_pixel(u32 fmt, u32 c, u8 *p) {
    u32 r = c & 255, gg = (c >> 8) & 255, b = (c >> 16) & 255, a = c >> 24;
    switch (fmt) {
    case 0: p[0] = a; p[1] = b; p[2] = gg; p[3] = r; break;
    case 1: p[0] = b; p[1] = gg; p[2] = r; break;
    case 2: {
        u32 v = ((r >> 3) << 11) | ((gg >> 2) << 5) | (b >> 3);
        p[0] = v; p[1] = v >> 8;
        break;
    }
    case 3: {
        u32 v = ((r >> 3) << 11) | ((gg >> 3) << 6) | ((b >> 3) << 1) | (a >> 7);
        p[0] = v; p[1] = v >> 8;
        break;
    }
    case 4: {
        u32 v = ((r >> 4) << 12) | ((gg >> 4) << 8) | ((b >> 4) << 4) | (a >> 4);
        p[0] = v; p[1] = v >> 8;
        break;
    }
    default: break;
    }
}

// ---------------------------------------------------------------- thread pool
// Work items are claimed from one atomic word holding (generation, item count, next item), so a
// worker that wakes up late can never run items of a newer job, and parallel_for() returns as
// soon as all items are done - it never waits for descheduled idle workers.
namespace {
std::vector<std::thread> workers;
// never destroyed: detached workers may still be waiting on them during process exit
std::mutex &pool_mtx = *new std::mutex;
std::condition_variable &pool_cv = *new std::condition_variable;
std::atomic<u64> pool_work{0};  // gen << 48 | n << 24 | next
std::atomic<u32> pool_done{0};
std::atomic<u32> pool_sleepers{0};
const std::function<void(u32, u32)> *pool_jobs[2] = {nullptr, nullptr};
u32 pool_gen = 0;

inline void cpu_relax() {
#if defined(__aarch64__)
    asm volatile("yield");
#elif defined(__x86_64__)
    asm volatile("pause");
#endif
}

// claim and run items of generation `gen` until none are left
inline void run_items(u32 gen, u32 tid) {
    for (;;) {
        u64 p = pool_work.load(std::memory_order_acquire);
        if ((u32)(p >> 48) != gen) return;
        const u32 idx = (u32)(p & 0xffffff), n = (u32)((p >> 24) & 0xffffff);
        if (idx >= n) return;
        if (!pool_work.compare_exchange_weak(p, p + 1, std::memory_order_acq_rel)) continue;
        (*pool_jobs[gen & 1])(idx, tid);
        pool_done.fetch_add(1, std::memory_order_release);
    }
}

void worker_main(u32 tid) {
    u32 seen = 0;
    for (;;) {
        u32 gen;
        u32 spins = 0;
        while ((gen = (u32)(pool_work.load(std::memory_order_acquire) >> 48)) == seen) {
            if (++spins < 20000) {
                cpu_relax();
            } else {
                std::unique_lock<std::mutex> lk(pool_mtx);
                pool_sleepers.fetch_add(1);
                pool_cv.wait(lk, [&] { return (u32)(pool_work.load() >> 48) != seen; });
                pool_sleepers.fetch_sub(1);
                spins = 0;
            }
        }
        seen = gen;
        run_items(gen, tid);
    }
}
}  // namespace

void pool_init() {
    if (!workers.empty()) return;
    u32 hw = std::thread::hardware_concurrency();
    u32 n = hw > 2 ? hw - 2 : 0;
    if (n > 8) n = 8;
    if (const char *e = getenv("GPU_THREADS")) n = (u32)atoi(e) > 0 ? (u32)atoi(e) - 1 : 0;
    for (u32 i = 0; i < n; i++) {
        workers.emplace_back(worker_main, i + 1);
        workers.back().detach();
    }
}

u32 pool_threads() { return (u32)workers.size() + 1; }

void parallel_for(u32 n, const std::function<void(u32, u32)> &fn) {
    if (n == 0) return;
    if (n == 1 || workers.empty()) {
        for (u32 i = 0; i < n; i++) fn(i, 0);
        return;
    }
    for (u32 base = 0; base < n; base += 0xffffff) {
        const u32 cnt = std::min<u32>(n - base, 0xffffff);
        std::function<void(u32, u32)> shifted;
        const std::function<void(u32, u32)> *job = &fn;
        if (base) {
            shifted = [&fn, base](u32 i, u32 t) { fn(i + base, t); };
            job = &shifted;
        }
        pool_gen = (pool_gen + 1) & 0xffff;
        if (pool_gen == 0) pool_gen = 1;
        pool_jobs[pool_gen & 1] = job;
        pool_done.store(0, std::memory_order_relaxed);
        pool_work.store((u64)pool_gen << 48 | (u64)cnt << 24, std::memory_order_release);
        if (pool_sleepers.load()) {
            std::lock_guard<std::mutex> lk(pool_mtx);
            pool_cv.notify_all();
        }
        run_items(pool_gen, 0);
        while (pool_done.load(std::memory_order_acquire) != cnt) cpu_relax();
    }
}

// ---------------------------------------------------------------- uniform / program upload
static V4 unpack_f24(const u32 *b) {
    V4 r;
    r.v[0] = f24(b[2] & 0xFFFFFF);
    r.v[1] = f24(((b[1] & 0xFFFF) << 8) | ((b[2] >> 24) & 0xFF));
    r.v[2] = f24(((b[0] & 0xFF) << 16) | ((b[1] >> 16) & 0xFFFF));
    r.v[3] = f24(b[0] >> 8);
    return r;
}

// returns uniform index written, or -1
static int upload_float(ShaderSetup &s, u32 base, u32 value) {
    u32 &cfg = g.regs[base + R::SH_FLOAT_IDX];
    const bool is_f32 = cfg >> 31;
    s.queue[s.queue_n++] = value;
    if (!((is_f32 && s.queue_n >= 4) || (!is_f32 && s.queue_n >= 3))) return -1;
    s.queue_n = 0;
    V4 u;
    if (is_f32) {
        for (int i = 0; i < 4; i++) memcpy(&u.v[3 - i], &s.queue[i], 4);
    } else {
        u = unpack_f24(s.queue);
    }
    u32 idx = cfg & 0x7f;
    if (idx >= 96) return -1;
    s.f[idx] = u;
    cfg = (cfg & ~0x7fu) | ((idx + 1) & 0x7f);
    return (int)idx;
}

static void upload_code(ShaderSetup &s, u32 offset, u32 value) {
    if (s.code[offset] != value) {
        s.code[offset] = value;
        s.dirty = true;
        if (offset + 1 > s.code_size) s.code_size = offset + 1;
    }
}
static void upload_swizzle(ShaderSetup &s, u32 offset, u32 value) {
    if (s.swizzle[offset] != value) {
        s.swizzle[offset] = value;
        s.dirty = true;
        if (offset + 1 > s.swizzle_size) s.swizzle_size = offset + 1;
    }
}

static inline bool mirror_vs_to_gs() {
    return !(g.regs[R::GS_EXCLUSIVE] & 1) && (g.regs[R::GS_MISC] & 3) == 0;
}

static void submit_immediate(u32 value) {
    g.imm_queue[g.imm_n++] = value;
    if (g.imm_n < 3) return;
    g.imm_n = 0;
    u32 &idx = g.regs[R::FIXATTR_INDEX];
    if (idx > 15) return;
    V4 a = unpack_f24(g.imm_queue);
    if (idx < 15) {
        g.default_attr[idx] = a;
        idx++;
        return;
    }
    g.imm_vertex[g.imm_attr] = a;
    if (g.imm_attr < (g.regs[R::VS_NUM_ATTR] & 0xf)) {
        g.imm_attr++;
        return;
    }
    vtx_immediate_vertex();
    g.imm_attr = 0;
}

// ---------------------------------------------------------------- command lists
namespace {
struct CmdList {
    const u32 *head = nullptr;
    u32 len = 0, idx = 0;
} cl;
bool stop_requested;

void set_list(u32 paddr, u32 size) {
    cl.head = (const u32 *)pptr(paddr);
    cl.len = cl.head ? size / 4 : 0;
    cl.idx = 0;
}

constexpr u32 expand_mask[16] = {
    0x00000000, 0x000000ff, 0x0000ff00, 0x0000ffff, 0x00ff0000, 0x00ff00ff, 0x00ffff00, 0x00ffffff,
    0xff000000, 0xff0000ff, 0xff00ff00, 0xff00ffff, 0xffff0000, 0xffff00ff, 0xffffff00, 0xffffffff,
};

// registers needing side effects
bool special[0x300];
void init_special() {
    const u32 list[] = {R::IRQ_REQ,      R::TOPOLOGY,      R::RESTART_PRIM, R::FIXATTR_INDEX,
                        R::DRAW,         R::DRAW_INDEXED,  R::GS + R::SH_BOOL,
                        R::VS + R::SH_BOOL, R::VS + R::SH_OUTMASK, R::CMDBUF_JUMP0,
                        R::CMDBUF_JUMP1};
    for (u32 r : list) special[r] = true;
    for (u32 i = 0; i < 3; i++) special[R::FIXATTR_DATA + i] = true;
    for (u32 i = 0; i < 4; i++) {
        special[R::GS + R::SH_INT + i] = true;
        special[R::VS + R::SH_INT + i] = true;
    }
    for (u32 i = 0; i < 8; i++) {
        special[R::GS + R::SH_FLOAT_DATA + i] = true;
        special[R::VS + R::SH_FLOAT_DATA + i] = true;
        special[R::GS + R::SH_CODE_DATA + i] = true;
        special[R::VS + R::SH_CODE_DATA + i] = true;
        special[R::GS + R::SH_SWZ_DATA + i] = true;
        special[R::VS + R::SH_SWZ_DATA + i] = true;
        special[R::LUT_DATA + i] = true;
        special[R::FOG_LUT_DATA + i] = true;
        special[R::PT_LUT_DATA + i] = true;
    }
}

void handle_special(u32 id, u32 value) {
    u32 *regs = g.regs;
    if (id >= R::VS + R::SH_FLOAT_DATA && id < R::VS + R::SH_FLOAT_DATA + 8) {
        int idx = upload_float(g.vs, R::VS, value);
        if (idx >= 0 && mirror_vs_to_gs()) g.gs.f[idx] = g.vs.f[idx];
        return;
    }
    if (id >= R::VS + R::SH_CODE_DATA && id < R::VS + R::SH_CODE_DATA + 8) {
        u32 &off = regs[R::VS + R::SH_CODE_IDX];
        if (off < 512) {
            upload_code(g.vs, off, value);
            if (mirror_vs_to_gs()) upload_code(g.gs, off, value);
            off++;
        }
        return;
    }
    if (id >= R::VS + R::SH_SWZ_DATA && id < R::VS + R::SH_SWZ_DATA + 8) {
        u32 &off = regs[R::VS + R::SH_SWZ_IDX];
        if (off < 4096) {
            upload_swizzle(g.vs, off, value);
            if (mirror_vs_to_gs()) upload_swizzle(g.gs, off, value);
            off++;
        }
        return;
    }
    if (id >= R::GS + R::SH_FLOAT_DATA && id < R::GS + R::SH_FLOAT_DATA + 8) {
        upload_float(g.gs, R::GS, value);
        return;
    }
    if (id >= R::GS + R::SH_CODE_DATA && id < R::GS + R::SH_CODE_DATA + 8) {
        u32 &off = regs[R::GS + R::SH_CODE_IDX];
        if (off < 4096) {
            upload_code(g.gs, off, value);
            off++;
        }
        return;
    }
    if (id >= R::GS + R::SH_SWZ_DATA && id < R::GS + R::SH_SWZ_DATA + 8) {
        u32 &off = regs[R::GS + R::SH_SWZ_IDX];
        if (off < 4096) {
            upload_swizzle(g.gs, off, value);
            off++;
        }
        return;
    }
    if (id >= R::LUT_DATA && id < R::LUT_DATA + 8) {
        u32 &cfg = regs[R::LUT_INDEX];
        u32 index = cfg & 0xff, type = (cfg >> 8) & 0x1f;
        if (type < 24 && g.light_lut[type][index] != value) {
            g.light_lut[type][index] = value;
            g.light_lut_dirty = true;
        }
        cfg = (cfg & ~0xffu) | ((index + 1) & 0xff);
        return;
    }
    if (id >= R::FOG_LUT_DATA && id < R::FOG_LUT_DATA + 8) {
        u32 &off = regs[R::FOG_LUT_OFFSET];
        u32 o = off & 0xffff;
        if (g.fog_lut[o % 128] != value) {
            g.fog_lut[o % 128] = value;
            g.fog_lut_dirty = true;
        }
        off = (off & ~0xffffu) | ((o + 1) & 0xffff);
        return;
    }
    if (id >= R::PT_LUT_DATA && id < R::PT_LUT_DATA + 8) {
        u32 &cfg = regs[R::PT_LUT_CONFIG];
        u32 index = cfg & 0xff, table = (cfg >> 8) & 0xf;
        u32 *t = nullptr;
        u32 size = 0;
        switch (table) {
        case 0: t = g.pt_noise; size = 128; break;
        case 2: t = g.pt_cmap; size = 128; break;
        case 3: t = g.pt_amap; size = 128; break;
        case 4: t = g.pt_color; size = 256; break;
        case 5: t = g.pt_cdiff; size = 256; break;
        default: break;
        }
        if (t && t[index % size] != value) {
            t[index % size] = value;
            g.pt_dirty = true;
        }
        cfg = (cfg & ~0xffu) | ((index + 1) & 0xff);
        return;
    }
    switch (id) {
    case R::IRQ_REQ: {
        u32 a = regs[R::IRQ_REQ], b = regs[R::IRQ_CMP];
        bool match = false;
        for (int i = 0; i < 4; i++)
            if (((a >> (8 * i)) & 0xff) == ((b >> (8 * i)) & 0xff)) match = true;
        if (match && regs[R::IRQ_AUTOSTOP]) stop_requested = true;
        if (g_trace) LOG("gpu trace: irq_req %08x (cmp %08x) stop=%d at word %u/%u\n", a, b, (int)stop_requested, cl.idx, cl.len);
        break;
    }
    case R::TOPOLOGY: pa_reconfigure((regs[R::TOPOLOGY] >> 8) & 3); break;
    case R::RESTART_PRIM: pa_reset(); break;
    case R::FIXATTR_INDEX: vtx_reset_immediate(); break;
    case R::FIXATTR_DATA:
    case R::FIXATTR_DATA + 1:
    case R::FIXATTR_DATA + 2: submit_immediate(value); break;
    case R::CMDBUF_JUMP0:
    case R::CMDBUF_JUMP1: {
        u32 i = id - R::CMDBUF_JUMP0;
        u32 addr = (regs[R::CMDBUF_ADDR0 + i] & 0x0fffffff) * 8;
        u32 size = (regs[R::CMDBUF_SIZE0 + i] & 0xfffff) * 8;
        if (g_trace) LOG("gpu trace: cmdbuf jump%u -> %08x size %x\n", i, addr, size);
        set_list(addr, size);
        break;
    }
    case R::DRAW:
    case R::DRAW_INDEXED:
        if (g_trace) LOG("gpu trace: draw%s n=%u\n", id == R::DRAW_INDEXED ? " indexed" : "", regs[R::NUM_VERTICES]);
        vtx_draw(id == R::DRAW_INDEXED);
        break;
    case R::GS + R::SH_BOOL:
        for (int i = 0; i < 16; i++) g.gs.b[i] = (regs[id] >> i) & 1;
        break;
    case R::VS + R::SH_BOOL:
        for (int i = 0; i < 16; i++) g.vs.b[i] = (regs[id] >> i) & 1;
        if (mirror_vs_to_gs())
            for (int i = 0; i < 16; i++) g.gs.b[i] = g.vs.b[i];
        break;
    case R::GS + R::SH_INT:
    case R::GS + R::SH_INT + 1:
    case R::GS + R::SH_INT + 2:
    case R::GS + R::SH_INT + 3: {
        u32 i = id - (R::GS + R::SH_INT);
        for (int c = 0; c < 4; c++) g.gs.i[i][c] = regs[id] >> (8 * c);
        break;
    }
    case R::VS + R::SH_INT:
    case R::VS + R::SH_INT + 1:
    case R::VS + R::SH_INT + 2:
    case R::VS + R::SH_INT + 3: {
        u32 i = id - (R::VS + R::SH_INT);
        for (int c = 0; c < 4; c++) g.vs.i[i][c] = regs[id] >> (8 * c);
        if (mirror_vs_to_gs()) memcpy(g.gs.i[i], g.vs.i[i], 4);
        break;
    }
    case R::VS + R::SH_OUTMASK:
        if (mirror_vs_to_gs()) regs[R::GS + R::SH_OUTMASK] = value;
        break;
    default: break;
    }
}

inline void write_reg(u32 id, u32 value, u32 mask) {
    if (id >= 0x300) return;
    const u32 m = expand_mask[mask];
    g.regs[id] = (g.regs[id] & ~m) | (value & m);
    if (special[id]) {
        // data streams don't affect the draw state
        if (!(id >= R::FIXATTR_DATA && id < R::FIXATTR_DATA + 3)) g.regs_version++;
        handle_special(id, value);
    } else {
        g.regs_version++;
    }
}
}  // namespace

void cmd_run(u32 paddr, u32 size) {
    set_list(paddr, size);
    if (g_trace) LOG("gpu trace: cmdlist %08x size %x (ptr %p)\n", paddr, size, (const void *)cl.head);
    stop_requested = false;
    while (cl.idx < cl.len && !stop_requested) {
        if (cl.idx & 1) cl.idx++;
        if (cl.idx + 1 >= cl.len) break;
        const u32 value = cl.head[cl.idx];
        const u32 header = cl.head[cl.idx + 1];
        cl.idx += 2;
        const u32 id = header & 0xffff;
        const u32 mask = (header >> 16) & 0xf;
        u32 count = (header >> 20) & 0xff;
        const bool group = header >> 31;
        // capture the extra parameters before handling: a command-buffer jump may replace cl
        const u32 *extra = cl.head + cl.idx;
        if (cl.idx + count > cl.len) count = cl.len - cl.idx;
        cl.idx += count;
        write_reg(id, value, mask);
        for (u32 i = 0; i < count && !stop_requested; i++)
            write_reg(group ? id + 1 + i : id, extra[i], mask);
    }
}

// ---------------------------------------------------------------- public API
static bool inited = false;

void init() {
    if (inited) return;
    inited = true;
    // g has static storage: everything starts zeroed
    memset(g.regs, 0, sizeof(g.regs));
    g.regs[R::IRQ_AUTOSTOP] = 1;
    g.regs[R::IRQ_MASK] = 0xFFFFFFF0;
    g.regs[R::IRQ_CMP] = 0x12345678;
    g.regs[R::GS + R::SH_INPUT_CFG] = 1 | (0xA0u << 24);
    g.light_lut_dirty = g.fog_lut_dirty = g.pt_dirty = true;
    g.imm_reset_gs = true;
    g.regs_version = 1;
    init_special();
    g_stats = getenv("GPU_STATS") != nullptr;
    g_trace = getenv("GPU_TRACE") != nullptr;
    pool_init();
    raster_init();
}

void ensure_init() {
    if (!inited) init();
}

void process_command_list(u32 paddr, u32 size_bytes) {
    ensure_init();
    metal_poll();
    texcache_new_epoch();
    g.ds_version = 0;
    static const double hitch_ms = getenv("GPU_HITCH_MS") ? atof(getenv("GPU_HITCH_MS")) : 0;
    const double t0 = g_stats || hitch_ms ? now_ms() : 0;
    cmd_run(paddr, size_bytes);
    raster_flush();
    if (hitch_ms) {
        // command lists that took long enough to delay a frame (GPU_HITCH_MS=<threshold>)
        static double worst = 0;
        const double dt = now_ms() - t0;
        worst = std::max(worst, dt);
        if (dt >= hitch_ms) LOG("gpu: slow command list %08x: %.1f ms (worst %.1f ms)\n", paddr, dt, worst);
    }
    if (g_stats) {
        extern std::atomic<u64> g_sh_instr, g_sh_runs;
        LOG("gpu stats: shader runs %llu, instructions %llu (%.1f/run)\n",
            (unsigned long long)g_sh_runs.load(), (unsigned long long)g_sh_instr.load(),
            g_sh_runs ? (double)g_sh_instr / g_sh_runs : 0.0);
        g_sh_instr = 0;
        g_sh_runs = 0;
        LOG("gpu stats: cmdlist %08x: total %.2f ms (vertex shading %.2f, assembly+setup %.2f, "
            "tiles %.2f)\n", paddr, now_ms() - t0, g_t_shade, g_t_setup, g_t_flush);
        g_t_shade = g_t_setup = g_t_flush = 0;
    }
}

// ---------------------------------------------------------------- state save / restore
namespace {
constexpr u32 STATE_MAGIC = 0x53504750;  // "PGPS"
constexpr u32 STATE_VERSION = 1;
template <class F>
void state_fields(F &&f) {
    f(g.regs, sizeof(g.regs));
    for (ShaderSetup *s : {&g.vs, &g.gs}) {
        f(s->code, sizeof(s->code));
        f(s->swizzle, sizeof(s->swizzle));
        f(s->f, sizeof(s->f));
        f(s->b, sizeof(s->b));
        f(s->i, sizeof(s->i));
        f(s->queue, sizeof(s->queue));
        f(&s->queue_n, sizeof(s->queue_n));
        f(&s->code_size, sizeof(s->code_size));
        f(&s->swizzle_size, sizeof(s->swizzle_size));
    }
    f(g.default_attr, sizeof(g.default_attr));
    f(g.imm_queue, sizeof(g.imm_queue));
    f(&g.imm_n, sizeof(g.imm_n));
    f(g.imm_vertex, sizeof(g.imm_vertex));
    f(&g.imm_attr, sizeof(g.imm_attr));
    f(&g.imm_reset_gs, sizeof(g.imm_reset_gs));
    f(g.light_lut, sizeof(g.light_lut));
    f(g.fog_lut, sizeof(g.fog_lut));
    f(g.pt_noise, sizeof(g.pt_noise));
    f(g.pt_cmap, sizeof(g.pt_cmap));
    f(g.pt_amap, sizeof(g.pt_amap));
    f(g.pt_color, sizeof(g.pt_color));
    f(g.pt_cdiff, sizeof(g.pt_cdiff));
    f(g.gs_unit.r, sizeof(g.gs_unit.r));
    f(g.gs_unit.addr, sizeof(g.gs_unit.addr));
    f(g.gs_unit.cond, sizeof(g.gs_unit.cond));
    f(&g.gs_unit.out_bank, sizeof(g.gs_unit.out_bank));
    f(&g.pa, sizeof(g.pa));
}
}  // namespace

bool save_state(const char *path) {
    ensure_init();
    raster_flush();
    metal_writeback_all();
    FILE *fp = fopen(path, "wb");
    if (!fp) return false;
    const u32 hdr[2] = {STATE_MAGIC, STATE_VERSION};
    fwrite(hdr, 4, 2, fp);
    state_fields([&](void *p, size_t n) { fwrite(p, 1, n, fp); });
    fclose(fp);
    return true;
}

bool load_state(const char *path) {
    ensure_init();
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    u32 hdr[2] = {};
    bool ok = fread(hdr, 4, 2, fp) == 2 && hdr[0] == STATE_MAGIC && hdr[1] == STATE_VERSION;
    if (ok) {
        raster_flush();
        state_fields([&](void *p, size_t n) { ok &= fread(p, 1, n, fp) == n; });
        g.vs.dirty = g.gs.dirty = true;
        g.vs.prog = g.gs.prog = nullptr;
        g.light_lut_dirty = g.fog_lut_dirty = g.pt_dirty = true;
        g.regs_version++;
        g.ds_version = 0;
    }
    fclose(fp);
    if (!ok) LOG("gpu: bad state file %s\n", path);
    return ok;
}

void flush() {
    ensure_init();
    raster_flush();
    metal_writeback_all();
}

}  // namespace gpu
