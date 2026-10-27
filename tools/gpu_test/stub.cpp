// Minimal runtime stand-ins so the GPU can run outside the game executable: a guest memory
// reservation, physical address translation and logging.
#include <sys/mman.h>

#include <cstdarg>

#include "loader.h"
#include "rt.h"

u8 *g_mem = nullptr;
GameInfo g_game;  // replays: no game, no pipeline lists unless the dump says otherwise
std::string g_save_base = "save";
int g_verbose = 0;

void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    abort();
}

namespace mem {
void init() {
    size_t size = (1ull << 32) + 0x10000;
    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
                   -1, 0);
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
}  // namespace mem
