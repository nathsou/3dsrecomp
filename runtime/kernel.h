// A small Horizon (3DS) kernel: handles, threads on host pthreads with a single-runner scheduler,
// synchronisation objects, shared memory and sessions to HLE services.
#pragma once
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rt.h"

using Handle = u32;

namespace kernel {
std::string save_root();  // <save base>/<title id>: saves, log.txt
u32 alloc_linear(u32 size);  // (lock held) linear heap memory for HLE objects the GPU may read (0 if none)

constexpr u32 RES_OK = 0;
constexpr u32 RES_TIMEOUT = 0x09401BFE;
constexpr u32 RES_INVALID_HANDLE = 0xD8E007F7;
constexpr u32 RES_INVALID_ADDRESS = 0xE0E01BF5;
constexpr u32 RES_OUT_OF_MEMORY = 0xD86007F3;
constexpr u32 RES_NOT_FOUND = 0xD88007FA;
constexpr u32 RES_INVALID_COMBINATION = 0xE0E01BEE;
constexpr u32 RES_NOT_AUTHORIZED = 0xD9001BEA;
constexpr u32 RES_SESSION_CLOSED = 0xC920181A;

constexpr Handle CURRENT_THREAD = 0xFFFF8000;
constexpr Handle CURRENT_PROCESS = 0xFFFF8001;

enum class Type { Event, Mutex, Semaphore, Timer, Thread, Process, Arbiter, SharedMem, Port, Session, ResLimit };
enum ResetType { ONESHOT = 0, STICKY = 1, PULSE = 2 };

struct Thread;
struct Service;

struct Object : std::enable_shared_from_this<Object> {
    virtual ~Object() = default;
    virtual Type type() const = 0;
    std::string name;
};
using ObjPtr = std::shared_ptr<Object>;

struct WaitObject : Object {
    std::vector<Thread *> waiters;
    virtual bool should_wait(Thread *t) = 0;
    virtual void acquire(Thread *t) = 0;
    virtual void after_wakeups() {}
    void add_waiter(Thread *t);
    void remove_waiter(Thread *t);
    void wakeup_waiters();  // (kernel lock held) after this object's state changed
};

struct Event : WaitObject {
    ResetType reset = ONESHOT;
    bool signaled = false;
    Type type() const override { return Type::Event; }
    bool should_wait(Thread *) override { return !signaled; }
    void acquire(Thread *) override {
        if (reset == ONESHOT) signaled = false;
    }
    void after_wakeups() override {
        if (reset == PULSE) signaled = false;
    }
    void signal();
    void clear() { signaled = false; }
};

struct Mutex : WaitObject {
    Thread *owner = nullptr;
    u32 lock_count = 0;
    Type type() const override { return Type::Mutex; }
    bool should_wait(Thread *t) override { return owner && owner != t; }
    void acquire(Thread *t) override;
    u32 release(Thread *t);
};

struct Semaphore : WaitObject {
    s32 count = 0, max = 0;
    Type type() const override { return Type::Semaphore; }
    bool should_wait(Thread *) override { return count <= 0; }
    void acquire(Thread *) override { count--; }
};

struct Timer : WaitObject {
    ResetType reset = ONESHOT;
    bool signaled = false;
    s64 initial = 0, interval = 0;
    u64 generation = 0;
    Type type() const override { return Type::Timer; }
    bool should_wait(Thread *) override { return !signaled; }
    void acquire(Thread *) override {
        if (reset == ONESHOT) signaled = false;
    }
    void after_wakeups() override {
        if (reset == PULSE) signaled = false;
    }
    void fire();
};

enum class TStatus { Ready, Running, WaitSync, WaitSleep, WaitArb, Dormant, Dead };

struct Thread : WaitObject {
    u32 id = 0;
    s32 priority = 0x30;
    TStatus status = TStatus::Dormant;
    CPU cpu{};
    u32 tls = 0;
    u32 entry = 0;
    std::condition_variable cv;
    std::thread host;
    unsigned mach = 0;  // host thread's Mach port (thread dumps sample its stack for the guest call chain)
    u64 ready_seq = 0;
    // waiting
    std::vector<std::shared_ptr<WaitObject>> wait_objs;
    bool wait_all = false;
    bool wait_out_index = false;
    u32 wait_result = 0;
    s32 wait_index = 0;
    u64 wait_gen = 0;
    u32 arb_addr = 0;
    Object *arb = nullptr;
    Type type() const override { return Type::Thread; }
    bool should_wait(Thread *) override { return status != TStatus::Dead; }
    void acquire(Thread *) override {}
};

struct Arbiter : Object {
    Type type() const override { return Type::Arbiter; }
};

struct SharedMem : Object {
    u32 backing = 0;  // guest address of the memory
    u32 size = 0;
    u32 my_perm = 0, other_perm = 0;
    void (*on_move)(u32 addr) = nullptr;  // fix up contents after MapMemoryBlock moved the block
    Type type() const override { return Type::SharedMem; }
};

struct Session : Object {
    std::shared_ptr<Service> service;
    Type type() const override { return Type::Session; }
};

struct ResLimit : Object {
    Type type() const override { return Type::ResLimit; }
};

struct Process : Object {
    Type type() const override { return Type::Process; }
};

// ---- the kernel
extern std::mutex kmutex;  // guards all kernel and HLE service state
using Lock = std::unique_lock<std::mutex>;

Handle add_handle(ObjPtr o);
ObjPtr get_object(Handle h);
template <typename T>
std::shared_ptr<T> get(Handle h) {
    auto o = get_object(h);
    return std::dynamic_pointer_cast<T>(o);
}
bool close_handle(Handle h);

Thread *current_thread();
std::shared_ptr<Thread> create_thread(u32 entry, u32 arg, u32 stack_top, s32 prio, bool start);
void resume_thread(Thread *t);         // (lock held) make a waiting thread ready
void schedule_async();                 // (lock held) from a host thread: run someone if idle
void run_when_quiescent(std::function<void()> fn);  // (lock held) run fn when no guest code is executing
void reschedule(Lock &lk);             // (lock held, from a guest thread) yield to a higher priority thread
void block_current(Lock &lk);          // (lock held) the current thread waits until resumed
void sleep_current(u64 ns, Lock &lk);  // (lock held) the current thread sleeps for ns of guest time
u64 now_ns();   // the guest clock: real time scaled by the game speed
u64 real_ns();
void set_clock_speed(u32 percent);
u64 system_tick();
void add_timeout(u64 deadline_ns, std::function<void()> fn);  // (lock held) run fn (lock held) at deadline

std::shared_ptr<Event> make_event(ResetType rt, const char *name);
void svc_dispatch(CPU *c, u32 imm);
void dump_threads();
void dump_memory(const char *prefix);  // RECOMP_MEMDUMP
bool guest_readable(u32 addr, u32 size);
bool guest_busy();  // a guest thread is running (not all waiting)  // mapped and readable (host-side reads of game state)

struct BootInfo {
    u32 entry;
    u32 stack_size;
    s32 priority;
    u64 program_id;
};
[[noreturn]] void boot(const BootInfo &bi);  // starts the main guest thread; returns on the host main thread
void start(const BootInfo &bi);

extern u32 heap_size, linear_used;
}  // namespace kernel
