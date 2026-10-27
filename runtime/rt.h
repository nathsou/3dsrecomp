// Shared runtime declarations.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "cpu.h"

void rt_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
[[noreturn]] void rt_fatal(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#define LOG(...) rt_log(__VA_ARGS__)
#define FATAL(...) rt_fatal(__VA_ARGS__)
// Also write the log to a file (truncated; the previous run's log is kept as <path>.prev).
void rt_log_open(const std::string &path);
// Called with the message before a fatal error exits (the frontend shows it in a dialog).
extern void (*g_fatal_hook)(const char *msg);
extern std::string g_save_base;  // save/ (or Application Support/3dsrecomp in an app bundle)
extern int g_verbose;
extern u64 g_frame;  // VBlanks since boot (for logs)
#define VLOG(...) do { if (g_verbose) rt_log(__VA_ARGS__); } while (0)

// ---- guest memory
inline u8 *gptr(u32 a) { return g_mem + a; }
inline u8 r8(u32 a) { return g_mem[a]; }
inline u16 r16(u32 a) { u16 v; memcpy(&v, g_mem + a, 2); return v; }
inline u32 r32(u32 a) { u32 v; memcpy(&v, g_mem + a, 4); return v; }
inline u64 r64(u32 a) { u64 v; memcpy(&v, g_mem + a, 8); return v; }
inline void w8(u32 a, u8 v) { g_mem[a] = v; }
inline void w16(u32 a, u16 v) { memcpy(g_mem + a, &v, 2); }
inline void w32(u32 a, u32 v) { memcpy(g_mem + a, &v, 4); }
inline void w64(u32 a, u64 v) { memcpy(g_mem + a, &v, 8); }
std::string read_cstr(u32 a, size_t max = 256);

// Guest virtual memory layout (old 3DS application, kernel < 2.44)
namespace mem {
constexpr u32 CODE_BASE = 0x00100000;
constexpr u32 HEAP_BASE = 0x08000000;
constexpr u32 STACK_TOP = 0x10000000;
constexpr u32 LINEAR_BASE = 0x14000000;
constexpr u32 LINEAR_SIZE = 0x08000000;
constexpr u32 VRAM_BASE = 0x1F000000;
constexpr u32 VRAM_SIZE = 0x00600000;
constexpr u32 DSP_BASE = 0x1FF00000;
constexpr u32 DSP_SIZE = 0x00080000;
constexpr u32 CONFIG_MEM = 0x1FF80000;
constexpr u32 SHARED_PAGE = 0x1FF81000;
constexpr u32 TLS_BASE = 0x1FF82000;
constexpr u32 TLS_SLOT = 0x200;
// physical
constexpr u32 FCRAM_PADDR = 0x20000000;
constexpr u32 VRAM_PADDR = 0x18000000;
constexpr u32 DSP_PADDR = 0x1FF00000;

// HLE allocations (service shared memory etc.), outside anything the game maps
constexpr u32 HLE_BASE = 0x1E000000;
constexpr u32 HLE_SIZE = 0x00800000;

void init();
u32 paddr_to_vaddr(u32 paddr);  // 0 if not mapped
u8 *paddr_ptr(u32 paddr);
u32 vaddr_to_paddr(u32 vaddr);
u32 hle_alloc(u32 size, u32 align = 0x1000);
}  // namespace mem

// ---- dispatch
const GuestEntry *dispatch_lookup(u32 addr);
void run_guest(CPU *c);  // runs until the thread returns to EXIT_MAGIC
constexpr u32 EXIT_MAGIC = 0xFFFF0000u;
