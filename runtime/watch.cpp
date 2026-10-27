// Debugging: RECOMP_WATCH=<guest addr> reports which host code (recompiled guest functions) touches the 16 KiB
// host page holding that guest address. The page is protected; a fault records the PC and unprotects it, and a
// thread re-protects it every few milliseconds. The histogram is printed by watch_report().
#include <dlfcn.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <map>
#include <mutex>
#include <thread>

#include "rt.h"

static u8 *watch_page;
static std::mutex wm;
static std::map<u64, u64> hits;

static void handler(int sig, siginfo_t *si, void *ctx) {
    u8 *a = (u8 *)si->si_addr;
    if (watch_page && a >= watch_page && a < watch_page + 0x4000) {
        auto *uc = (ucontext_t *)ctx;
        u64 pc = uc->uc_mcontext->__ss.__pc;
        hits[pc]++;  // (racy but fine for a debug histogram)
        mprotect(watch_page, 0x4000, PROT_READ | PROT_WRITE);
        return;
    }
    signal(sig, SIG_DFL);
}

void watch_init() {
    const char *w = getenv("RECOMP_WATCH");
    if (!w) return;
    u32 ga = (u32)strtoul(w, nullptr, 16);
    watch_page = (u8 *)((uintptr_t)(g_mem + ga) & ~(uintptr_t)0x3FFF);
    struct sigaction sa {};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    std::thread([] {
        for (;;) {
            usleep(3000);
            mprotect(watch_page, 0x4000, PROT_NONE);
        }
    }).detach();
    LOG("[watch] watching host page of %08x\n", ga);
}

void watch_report() {
    if (!watch_page) return;
    mprotect(watch_page, 0x4000, PROT_READ | PROT_WRITE);
    std::map<std::string, u64> by_fn;
    for (auto &kv : hits) {
        Dl_info di;
        std::string name = "?";
        if (dladdr((void *)kv.first, &di) && di.dli_sname) name = di.dli_sname;
        by_fn[name] += kv.second;
    }
    for (auto &kv : by_fn) LOG("[watch] %8llu %s\n", (unsigned long long)kv.second, kv.first.c_str());
}
