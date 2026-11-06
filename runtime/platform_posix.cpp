// runtime/platform.h on Linux (and other POSIX systems without Apple's frameworks).
#include "platform.h"

#include <SDL3/SDL.h>
#include <dirent.h>
#include <execinfo.h>
#include <semaphore.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "uiview.h"

namespace platform {

std::string exe_path() {
    char buf[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    return n > 0 ? std::string(buf, (size_t)n) : "";
}

std::string app_data_dir() {
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/3dsrecomp";
    const char *home = getenv("HOME");
    return std::string(home ? home : ".") + "/.local/share/3dsrecomp";
}

const char *module_suffix() { return ".so"; }

void open_folder(const std::string &path) { SDL_OpenURL(("file://" + path).c_str()); }

std::string device_name() {
    char name[HOST_NAME_MAX + 1] = {};
    if (gethostname(name, sizeof name - 1) != 0 || !name[0]) return "Linux";
    return name;
}

void set_thread_interactive() { SDL_SetCurrentThreadPriority(SDL_THREAD_PRIORITY_HIGH); }

void set_thread_name(const char *name) {
    char n[16];
    snprintf(n, sizeof n, "%s", name);
    pthread_setname_np(pthread_self(), n);
}

// each thread's time on a CPU (ns) from /proc/self/task/<tid>/schedstat, its name from comm
std::string thread_cpu_report(double span_s) {
#ifdef __linux__
    static std::map<long, uint64_t> last;
    std::map<long, uint64_t> now;
    std::map<std::string, double> by_name;
    if (DIR *d = opendir("/proc/self/task")) {
        while (dirent *e = readdir(d)) {
            const long tid = atol(e->d_name);
            if (tid <= 0) continue;
            char path[64], name[32] = {};
            unsigned long long ns = 0;
            snprintf(path, sizeof path, "/proc/self/task/%ld/schedstat", tid);
            if (FILE *f = fopen(path, "r")) {
                if (fscanf(f, "%llu", &ns) != 1) ns = 0;
                fclose(f);
            }
            snprintf(path, sizeof path, "/proc/self/task/%ld/comm", tid);
            if (FILE *f = fopen(path, "r")) {
                if (!fgets(name, sizeof name, f)) name[0] = 0;
                fclose(f);
            }
            name[strcspn(name, "\n")] = 0;
            now[tid] = ns;
            const auto it = last.find(tid);
            if (it != last.end() && ns > it->second) by_name[name] += (ns - it->second) / 1e6 / span_s;
        }
        closedir(d);
    }
    last = std::move(now);
    std::vector<std::pair<double, std::string>> v;
    for (auto &[n, ms] : by_name)
        if (ms >= 5) v.push_back({ms, n});
    std::sort(v.rbegin(), v.rend());
    std::string out;
    for (auto &[ms, n] : v) out += (out.empty() ? "" : ", ") + n + " " + std::to_string((int)(ms + 0.5));
    return out;
#else
    (void)span_s;
    return "";
#endif
}

// ---------------------------------------------------------------- thread sampling
// The sampled thread is sent a signal; its handler walks its own stack with the unwinder (x86-64 code keeps no
// frame pointers, but every function has unwind tables) and hands the addresses back.
namespace {
std::mutex sample_mx;  // one sample at a time
sem_t sample_done;
uint64_t sample_buf[96];  // (not the caller's: an answer may come after the caller gave up)
std::atomic<int> sample_n;
int sample_sig() { return SIGRTMAX - 2; }

void sample_handler(int, siginfo_t *, void *ctx) {
    const int saved = errno;
    void *frames[95];
    int n = 0;
    sample_buf[n++] = fault_pc(ctx);
    const int k = backtrace(frames, 95);
    for (int i = 0; i < k; i++) sample_buf[n++] = (uint64_t)frames[i];
    sample_n = n;
    sem_post(&sample_done);
    errno = saved;
}

void sample_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        void *warm[2];
        backtrace(warm, 2);  // (the first call loads the unwinder: not in a signal handler)
        sem_init(&sample_done, 0, 0);
        struct sigaction sa {};
        sa.sa_sigaction = sample_handler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(sample_sig(), &sa, nullptr);
    });
}
}  // namespace

uint64_t thread_handle(pthread_t t) {
    sample_init();
    return (uint64_t)t;
}

bool is_current_thread(uint64_t handle) { return handle && pthread_equal((pthread_t)handle, pthread_self()); }

int sample_thread_pcs(uint64_t handle, uint64_t *pcs, int max) {
    if (!handle || max <= 0 || is_current_thread(handle)) return 0;
    std::lock_guard<std::mutex> lk(sample_mx);
    while (sem_trywait(&sample_done) == 0) {}  // (a late answer to an earlier sample)
    sample_n = 0;
    if (pthread_kill((pthread_t)handle, sample_sig()) != 0) return 0;
    timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_nsec += 200000000;
    if (until.tv_nsec >= 1000000000) until.tv_sec++, until.tv_nsec -= 1000000000;
    while (sem_timedwait(&sample_done, &until) != 0)
        if (errno != EINTR) return 0;  // (it did not answer: blocked signals)
    const int n = std::min(max, sample_n.load());
    std::copy(sample_buf, sample_buf + n, pcs);
    return n;
}

uint64_t fault_pc(void *ucontext) {
    const auto *uc = (const ucontext_t *)ucontext;
#if defined(__x86_64__)
    return (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
    return (uint64_t)uc->uc_mcontext.pc;
#else
    (void)uc;
    return 0;
#endif
}

}  // namespace platform

// (iOS only: runtime/uiview.mm)
namespace uiview {
bool keep_filling(SDL_Renderer *) { return false; }
}  // namespace uiview
