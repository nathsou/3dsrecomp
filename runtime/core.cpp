// Logging, guest memory reservation, and the address -> recompiled function dispatcher.
#include <sys/mman.h>
#include <unistd.h>

#include <dlfcn.h>

#include <algorithm>
#include <map>
#include <mutex>

#include "kernel.h"
#include "loader.h"
#include "rt.h"

u8 *g_mem;
int g_verbose;
static std::mutex log_mutex;
static FILE *log_file;
void (*g_fatal_hook)(const char *msg);

u64 g_frame;

void rt_log_open(const std::string &path) {
    std::lock_guard<std::mutex> lk(log_mutex);
    rename(path.c_str(), (path + ".prev").c_str());
    log_file = fopen(path.c_str(), "w");
}

void rt_log(const char *fmt, ...) {
    std::lock_guard<std::mutex> lk(log_mutex);
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    fprintf(stderr, "%6llu %s", (unsigned long long)g_frame, msg);
    fflush(stderr);
    if (log_file) {
        fprintf(log_file, "%6llu %s", (unsigned long long)g_frame, msg);
        fflush(log_file);
    }
}

void rt_fatal(const char *fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    {
        std::lock_guard<std::mutex> lk(log_mutex);
        fprintf(stderr, "FATAL: %s", msg);
        fflush(stderr);
        if (log_file) {
            fprintf(log_file, "%6llu FATAL: %s", (unsigned long long)g_frame, msg);
            fflush(log_file);
        }
    }
    if (g_fatal_hook) g_fatal_hook(msg);
    _exit(1);
}

std::string read_cstr(u32 a, size_t max) {
    std::string s;
    for (size_t i = 0; i < max; i++) {
        char ch = (char)g_mem[a + i];
        if (!ch) break;
        s += ch;
    }
    return s;
}

namespace mem {
static u32 hle_next = HLE_BASE;

void init() {
    // 4 GiB for the guest address space plus a guard so unaligned accesses at the top stay mapped
    size_t size = (1ull << 32) + 0x10000;
    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) FATAL("cannot reserve guest memory\n");
    g_mem = (u8 *)p;
}

u32 paddr_to_vaddr(u32 pa) {
    if (pa >= FCRAM_PADDR && pa < FCRAM_PADDR + LINEAR_SIZE) return pa - FCRAM_PADDR + LINEAR_BASE;
    if (pa >= VRAM_PADDR && pa < VRAM_PADDR + VRAM_SIZE) return pa - VRAM_PADDR + VRAM_BASE;
    if (pa >= DSP_PADDR && pa < DSP_PADDR + DSP_SIZE) return pa - DSP_PADDR + DSP_BASE;
    return 0;
}

u8 *paddr_ptr(u32 pa) {
    u32 va = paddr_to_vaddr(pa);
    return va ? g_mem + va : nullptr;
}

u32 vaddr_to_paddr(u32 va) {
    if (va >= LINEAR_BASE && va < LINEAR_BASE + LINEAR_SIZE) return va - LINEAR_BASE + FCRAM_PADDR;
    if (va >= VRAM_BASE && va < VRAM_BASE + VRAM_SIZE) return va - VRAM_BASE + VRAM_PADDR;
    if (va >= DSP_BASE && va < DSP_BASE + DSP_SIZE) return va - DSP_BASE + DSP_PADDR;
    return 0;
}

u32 hle_alloc(u32 size, u32 align) {
    u32 a = (hle_next + align - 1) & ~(align - 1);
    hle_next = a + ((size + 0xFFF) & ~0xFFFu);
    if (hle_next > HLE_BASE + HLE_SIZE) FATAL("HLE memory exhausted\n");
    memset(g_mem + a, 0, size);
    return a;
}
}  // namespace mem

// ---------------------------------------------------------------- dispatch
extern "C" const GuestEntry guest_entries[];
extern "C" const u32 guest_entry_count;

const GuestEntry *dispatch_lookup(u32 addr) {
    u32 lo = 0, hi = guest_entry_count;
    while (lo < hi) {
        u32 mid = (lo + hi) / 2;
        if (guest_entries[mid].addr < addr) lo = mid + 1;
        else hi = mid;
    }
    if (lo < guest_entry_count && guest_entries[lo].addr == addr) return &guest_entries[lo];
    return nullptr;
}

#define DCACHE 8192
static thread_local struct { u32 addr; const GuestEntry *e; } dcache[DCACHE];

static const GuestEntry *lazy_lookup(u32 addr);

static const GuestEntry *lookup_cached(u32 addr) {
    auto &slot = dcache[(addr >> 2) & (DCACHE - 1)];
    if (slot.e && slot.addr == addr) return slot.e;
    static u32 pretend_missing = getenv("RECOMP_TEST_LAZY") ? (u32)strtoul(getenv("RECOMP_TEST_LAZY"), nullptr, 16) : 0;
    const GuestEntry *e = addr == pretend_missing ? nullptr : dispatch_lookup(addr);
    if (!e && !(addr & 1) && addr != pretend_missing) e = dispatch_lookup(addr | 1);  // a Thumb target reached without bit 0
    if (!e) e = lazy_lookup(addr);
    slot.addr = addr;
    slot.e = e;
    return e;
}

// ---- lazy recompilation: code the static analysis missed is recompiled into a dylib on first use
static std::mutex lazy_m;
static std::map<u32, GuestEntry> lazy_map;

static const GuestEntry *lazy_lookup(u32 addr) {
    std::lock_guard<std::mutex> lk(lazy_m);
    auto it = lazy_map.find(addr);
    return it == lazy_map.end() ? nullptr : &it->second;
}

static bool lazy_compile(u32 addr) {
#ifdef RECOMP_SOURCE_DIR
    std::string root = RECOMP_SOURCE_DIR;
    std::string game = RECOMP_GAME;
    char gamedir[4096];
    if (!realpath(g_game.dir.c_str(), gamedir)) return false;
    char base[256];
    snprintf(base, sizeof base, "%s/build/lazy/lazy_%08x", root.c_str(), addr);
    char hexaddr[16];
    snprintf(hexaddr, sizeof hexaddr, "%x", addr);
    std::string cmd = "mkdir -p '" + root + "/build/lazy' && cd '" + root + "' && $( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 ) -m tools.recomp.gen '" +
                      gamedir + "' generated/" + game + " --lazy " + hexaddr + " --lazy-out " + base +
                      ".c > /dev/null && clang -O1 -w -ffp-contract=off -fno-strict-aliasing -shared -fPIC "
                      "-undefined dynamic_lookup -Iruntime -Igenerated/" + game + " " + base + ".c -o " + base + ".dylib";
    LOG("[lazy] recompiling missing code at %08x\n", addr);
    if (system(cmd.c_str()) != 0) return false;
    void *h = dlopen((std::string(base) + ".dylib").c_str(), RTLD_NOW);
    if (!h) {
        LOG("[lazy] dlopen: %s\n", dlerror());
        return false;
    }
    auto ents = (const GuestEntry *)dlsym(h, "guest_lazy_entries");
    auto count = (const u32 *)dlsym(h, "guest_lazy_count");
    if (!ents || !count) return false;
    std::lock_guard<std::mutex> lk(lazy_m);
    for (u32 i = 0; i < *count; i++) lazy_map[ents[i].addr] = ents[i];
    return true;
#else
    (void)addr;
    return false;
#endif
}

static const GuestEntry *dispatch_miss(CPU *c) {
    u32 addr = c->pc;
    if (auto e = lazy_lookup(addr)) return e;
    FILE *f = fopen((g_save_base + "/missing_functions.txt").c_str(), "a");
    if (f) {
        fprintf(f, "0x%08x\n", addr);
        fclose(f);
    }
    LOG("[lazy] no code at %08x (lr=%08x)\n", addr, c->r[14]);
    if (lazy_compile(addr) || ((addr & 1) && lazy_compile(addr & ~1u))) {
        if (auto e = lazy_lookup(addr)) return e;
    }
    FATAL("no recompiled code at 0x%08x (lr=%08x sp=%08x r0=%08x); added to %s/missing_functions.txt\n", addr,
          c->r[14], c->r[13], c->r[0], g_save_base.c_str());
}

static int trace_left = -1;
static void trace(CPU *c, const char *why) {
    if (trace_left < 0) {
        const char *t = getenv("RECOMP_TRACE");
        trace_left = t ? atoi(t) : 0;
    }
    if (trace_left > 0) {
        trace_left--;
        LOG("[trace] %s %08x r0=%08x r1=%08x r4=%08x r5=%08x sp=%08x lr=%08x\n", why, c->pc, c->r[0], c->r[1], c->r[4],
            c->r[5], c->r[13], c->r[14]);
    }
}

extern "C" void rt_dispatch(CPU *c, u32) {
    trace(c, "dispatch");
    const GuestEntry *e = lookup_cached(c->pc);
    if (__builtin_expect(!e, 0)) {
        if (c->pc == EXIT_MAGIC) return;
        e = dispatch_miss(c);
    }
    e->fn(c, e->entry);
}

void run_guest(CPU *c) {
    while (c->pc != EXIT_MAGIC) {
        trace(c, "top");
        const GuestEntry *e = lookup_cached(c->pc);
        if (!e) e = dispatch_miss(c);
        e->fn(c, e->entry);
    }
}

extern "C" void rt_undefined(CPU *c, u32 addr, u32 word) {
    FATAL("undefined instruction %08x at %08x (lr=%08x)\n", word, addr, c->r[14]);
}
