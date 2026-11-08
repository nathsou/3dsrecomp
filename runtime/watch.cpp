// Debugging: who writes some guest memory.
//
// RECOMP_WATCH=<guest addr>[+<bytes>] (default 4 bytes), or a live script's "WATCH <addr> [<bytes>]" for an address
// known only once the game runs. Where the host can single-step (Linux x86-64), every write to those bytes is logged
// as it happens: the host function that made it (f_<address>: the recompiled guest function, or the runtime's own
// code: a service, a GPU write-back, memcpy...) with a short backtrace, the guest thread and the value before and
// after. The host page holding them is made read-only; a write to it faults, the page is opened for that one
// instruction (single-stepped with the trap flag), and closed again. Writes to the rest of the page go through the
// same way, unlogged. A write another host thread makes while the page is open for one is missed.
//
// Elsewhere (macOS), the page is protected entirely and the fault addresses are sampled: watch_report() prints
// which host functions touched the 16 KiB page, as a histogram, at the end of the run or before a fatal error.
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "kernel.h"
#include "platform.h"
#include "rt.h"

namespace {
constexpr uintptr_t kPage = 0x4000;  // (16 KiB: a page on every host, a whole number of pages on x86-64)
u8 *watch_page;                      // host page holding the watched bytes
u8 *watch_lo, *watch_hi;             // the watched bytes
u32 watch_guest;                     // their guest address
bool precise;                        // single-stepping writes (else: sampling the page's faults)

// records of writes, filled in by the signal handlers and printed by watch_poll() (outside them)
struct Record {
    u64 pc;
    void *bt[6];
    int nbt;
    u32 before, after;
    u32 thread;
    u64 frame;
};
constexpr int kRecords = 1024;
Record records[kRecords];
std::atomic<int> written{0};
int printed;
thread_local int stepping = -1;  // the record this thread's single step completes (-2: an unlogged write)

std::mutex wm;
std::map<u64, u64> hits;  // sampled mode: fault PCs

u32 read_watched() {
    u32 v = 0;
    memcpy(&v, watch_lo, std::min<size_t>(4, watch_hi - watch_lo));
    return v;
}

void segv(int sig, siginfo_t *si, void *ctx) {
    u8 *a = (u8 *)si->si_addr;
    if (!watch_page || a < watch_page || a >= watch_page + kPage) {
        signal(sig, SIG_DFL);  // (not ours: the fault goes on as it would have)
        return;
    }
    if (!precise) {
        hits[platform::fault_pc(ctx)]++;  // (racy but fine for a debug histogram)
        mprotect(watch_page, kPage, PROT_READ | PROT_WRITE);
        return;
    }
    if (a >= watch_lo && a < watch_hi) {
        const int i = written.load(std::memory_order_relaxed);
        if (i < kRecords) {
            Record &r = records[i];
            r.pc = platform::fault_pc(ctx);
            r.nbt = backtrace(r.bt, 6);
            r.before = read_watched();
            const kernel::Thread *t = kernel::current_thread();
            r.thread = t ? t->id : 0;
            r.frame = g_frame;
            stepping = i;
        } else {
            stepping = -2;
        }
    } else {
        stepping = -2;
    }
    mprotect(watch_page, kPage, PROT_READ | PROT_WRITE);
    platform::set_single_step(ctx, true);
}

void trap(int sig, siginfo_t *, void *ctx) {
    if (stepping == -1) {
        signal(sig, SIG_DFL);
        raise(sig);
        return;
    }
    platform::set_single_step(ctx, false);
    if (stepping >= 0) {
        records[stepping].after = read_watched();
        written.store(stepping + 1, std::memory_order_release);
    }
    stepping = -1;
    mprotect(watch_page, kPage, PROT_READ);
}

std::string symbol_of(void *pc) {
    Dl_info di;
    if (dladdr(pc, &di) && di.dli_sname) return di.dli_sname;
    char b[32];
    snprintf(b, sizeof b, "%p", pc);
    return b;
}
}  // namespace

void watch_arm(u32 ga, u32 bytes) {
    std::lock_guard<std::mutex> lk(wm);
    if (watch_page) {
        LOG("[watch] already watching %08x; %08x not watched\n", watch_guest, ga);
        return;
    }
    watch_guest = ga;
    watch_lo = g_mem + ga;
    watch_hi = watch_lo + std::max<u32>(bytes, 1);
    watch_page = (u8 *)((uintptr_t)watch_lo & ~(kPage - 1));
    if ((uintptr_t)(watch_hi - 1) >= (uintptr_t)watch_page + kPage) watch_hi = watch_page + kPage;  // (one page)
    void *warm[2];
    backtrace(warm, 2);  // (loads what backtrace() needs before a signal handler calls it)
    precise = platform::can_single_step();
    struct sigaction sa {};
    sa.sa_sigaction = segv;
    sa.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    if (precise) {
        struct sigaction st {};
        st.sa_sigaction = trap;
        st.sa_flags = SA_SIGINFO;
        sigaction(SIGTRAP, &st, nullptr);
        mprotect(watch_page, kPage, PROT_READ);
        LOG("[watch] logging every write to %08x-%08x (now %08x)\n", ga, ga + (u32)(watch_hi - watch_lo) - 1, read_watched());
    } else {
        mprotect(watch_page, kPage, PROT_NONE);
        std::thread([] {
            for (;;) {
                usleep(3000);
                mprotect(watch_page, kPage, PROT_NONE);
            }
        }).detach();
        LOG("[watch] sampling accesses to the host page of %08x (no single-stepping here)\n", ga);
    }
}

void watch_init() {
    const char *w = getenv("RECOMP_WATCH");
    if (!w) return;
    char *end;
    const u32 ga = (u32)strtoul(w, &end, 16);
    const u32 n = *end == '+' ? (u32)strtoul(end + 1, nullptr, 0) : 4;
    watch_arm(ga, n);
}

// (each frame) the writes logged since the last call
void watch_poll() {
    if (!precise) return;
    const int n = written.load(std::memory_order_acquire);
    for (; printed < n; printed++) {
        const Record &r = records[printed];
        std::string bt;
        for (int k = 1; k < r.nbt; k++) {
            Dl_info di;
            if (!dladdr(r.bt[k], &di) || !di.dli_sname) continue;  // (the signal trampoline, ...)
            bt += (bt.empty() ? "" : " < ") + std::string(di.dli_sname);
        }
        LOG("[watch] %08x: %08x -> %08x by %s (guest thread %u, frame %llu)%s%s\n", watch_guest, r.before, r.after,
            symbol_of((void *)r.pc).c_str(), r.thread, (unsigned long long)r.frame, bt.empty() ? "" : "; ", bt.c_str());
    }
    if (n >= kRecords && printed == kRecords) {
        LOG("[watch] %d writes logged: no more are\n", kRecords);
        printed++;
    }
}

void watch_report() {
    if (!watch_page) return;
    if (precise) {
        watch_poll();
        return;
    }
    mprotect(watch_page, kPage, PROT_READ | PROT_WRITE);
    std::map<std::string, u64> by_fn;
    for (auto &kv : hits) by_fn[symbol_of((void *)kv.first)] += kv.second;
    for (auto &kv : by_fn) LOG("[watch] %8llu %s\n", (unsigned long long)kv.second, kv.first.c_str());
}
