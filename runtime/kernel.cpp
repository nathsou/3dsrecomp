// Kernel objects, scheduler and SVCs.
#include "kernel.h"

#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <set>

#include "service.h"

namespace kernel {

std::mutex kmutex;
static std::map<Handle, ObjPtr> handles;
static Handle next_handle = 0x00010001;
static std::vector<std::shared_ptr<Thread>> threads;
static Thread *running = nullptr;
static u64 ready_counter = 1;
static u32 next_thread_id = 1;
static std::set<u32> tls_used;
static u64 boot_ns;
static std::shared_ptr<Process> self_process = std::make_shared<Process>();
static std::shared_ptr<ResLimit> self_reslimit = std::make_shared<ResLimit>();
u32 heap_size, linear_used;
static thread_local Thread *tls_thread;
static void flush_deferred();

// ---------------------------------------------------------------- handles
Handle add_handle(ObjPtr o) {
    Handle h = next_handle++;
    handles[h] = std::move(o);
    return h;
}

ObjPtr get_object(Handle h) {
    if (h == CURRENT_THREAD) {
        Thread *t = current_thread();
        return t ? t->shared_from_this() : nullptr;
    }
    if (h == CURRENT_PROCESS) return self_process;
    auto it = handles.find(h);
    return it == handles.end() ? nullptr : it->second;
}

bool close_handle(Handle h) { return handles.erase(h) > 0; }

Thread *current_thread() { return tls_thread; }

std::shared_ptr<Event> make_event(ResetType rt, const char *name) {
    auto e = std::make_shared<Event>();
    e->reset = rt;
    e->name = name;
    return e;
}

// ---------------------------------------------------------------- time
u64 now_ns() {
    return (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

u64 system_tick() {
    u64 ns = now_ns() - boot_ns;
    // 268111856 Hz
    return (u64)((__uint128_t)ns * 268111856u / 1000000000u);
}

// timeouts run on a host thread
struct TimeoutQueue {
    std::multimap<u64, std::function<void()>> q;
    std::condition_variable cv;
    std::thread th;
};
static TimeoutQueue tq;

void add_timeout(u64 deadline, std::function<void()> fn) {
    tq.q.emplace(deadline, std::move(fn));
    tq.cv.notify_one();
}

static void timeout_thread() {
    Lock lk(kmutex);
    for (;;) {
        if (tq.q.empty()) {
            tq.cv.wait(lk);
            continue;
        }
        u64 t = tq.q.begin()->first;
        u64 n = now_ns();
        if (t > n) {
            tq.cv.wait_for(lk, std::chrono::nanoseconds(t - n));
            continue;
        }
        auto fn = std::move(tq.q.begin()->second);
        tq.q.erase(tq.q.begin());
        fn();
        schedule_async();
    }
}

// ---------------------------------------------------------------- scheduling
static Thread *best_ready() {
    Thread *best = nullptr;
    for (auto &t : threads) {
        if (t->status != TStatus::Ready) continue;
        if (!best || t->priority < best->priority ||
            (t->priority == best->priority && t->ready_seq < best->ready_seq))
            best = t.get();
    }
    return best;
}

static void make_ready(Thread *t, bool front = false) {
    t->status = TStatus::Ready;
    t->ready_seq = front ? 0 : ready_counter++;
}

// Gives the CPU to the best ready thread and, if that is not `cur`, waits until `cur` runs again.
static void switch_from(Thread *cur, Lock &lk) {
    flush_deferred();
    Thread *next = best_ready();
    running = next;
    if (next) {
        next->status = TStatus::Running;
        if (next != cur) next->cv.notify_one();
    }
    if (next == cur) return;
    cur->cv.wait(lk, [&] { return running == cur; });
}

void resume_thread(Thread *t) {
    if (t->status == TStatus::Dead || t->status == TStatus::Ready || t->status == TStatus::Running) return;
    for (auto &o : t->wait_objs) o->remove_waiter(t);
    t->wait_objs.clear();
    t->wait_gen++;
    make_ready(t);
}

// Host-side updates of guest shared memory that guest code modifies with read-modify-write sequences (e.g. the
// GSP interrupt queue) must not interleave with guest code. Only the `running` thread executes guest code, and
// while the kernel lock is held it can only be doing so outside an SVC: such updates are deferred until it
// enters its next SVC (or blocks).
static std::vector<std::function<void()>> deferred_work;

void run_when_quiescent(std::function<void()> fn) {
    if (!running) fn();
    else deferred_work.push_back(std::move(fn));
}

static void flush_deferred() {
    while (!deferred_work.empty()) {
        auto d = std::move(deferred_work);
        deferred_work.clear();
        for (auto &f : d) f();
    }
}

void schedule_async() {
    if (running) return;
    flush_deferred();
    Thread *next = best_ready();
    if (!next) return;
    running = next;
    next->status = TStatus::Running;
    next->cv.notify_one();
}

void reschedule(Lock &lk) {
    Thread *cur = current_thread();
    Thread *best = best_ready();
    if (!best || best->priority >= cur->priority) return;
    make_ready(cur, true);
    switch_from(cur, lk);
}

void block_current(Lock &lk) {
    Thread *cur = current_thread();
    switch_from(cur, lk);
}

static void yield_current(Lock &lk) {
    Thread *cur = current_thread();
    make_ready(cur);
    switch_from(cur, lk);
}

// ---------------------------------------------------------------- wait objects
void WaitObject::add_waiter(Thread *t) {
    if (std::find(waiters.begin(), waiters.end(), t) == waiters.end()) waiters.push_back(t);
}

void WaitObject::remove_waiter(Thread *t) { waiters.erase(std::remove(waiters.begin(), waiters.end(), t), waiters.end()); }

void WaitObject::wakeup_waiters() {
    for (;;) {
        Thread *cand = nullptr;
        for (Thread *t : waiters) {
            if (t->status != TStatus::WaitSync) continue;
            bool ok;
            if (t->wait_all) {
                ok = true;
                for (auto &o : t->wait_objs)
                    if (o->should_wait(t)) ok = false;
            } else {
                ok = !should_wait(t);
            }
            if (ok && (!cand || t->priority < cand->priority)) cand = t;
        }
        if (!cand) break;
        if (cand->wait_all) {
            for (auto &o : cand->wait_objs) o->acquire(cand);
            cand->wait_index = 0;
        } else {
            acquire(cand);
            for (size_t i = 0; i < cand->wait_objs.size(); i++)
                if (cand->wait_objs[i].get() == this) cand->wait_index = (s32)i;
        }
        cand->wait_result = RES_OK;
        resume_thread(cand);
    }
    after_wakeups();
}

void Event::signal() {
    signaled = true;
    wakeup_waiters();
}

void Mutex::acquire(Thread *t) {
    owner = t;
    lock_count++;
}

u32 Mutex::release(Thread *t) {
    if (owner != t) return 0xD8E0041F;  // not owner
    if (--lock_count == 0) {
        owner = nullptr;
        wakeup_waiters();
    }
    return RES_OK;
}

static void timer_schedule(std::shared_ptr<Timer> tm, s64 delay) {
    u64 gen = tm->generation;
    add_timeout(now_ns() + (u64)std::max<s64>(delay, 0), [tm, gen] {
        if (tm->generation != gen) return;
        tm->fire();
        if (tm->interval > 0) timer_schedule(tm, tm->interval);
    });
}

void Timer::fire() {
    signaled = true;
    wakeup_waiters();
}

// ---------------------------------------------------------------- threads
static u32 alloc_tls() {
    for (u32 i = 0;; i++)
        if (!tls_used.count(i)) {
            tls_used.insert(i);
            u32 a = mem::TLS_BASE + i * mem::TLS_SLOT;
            memset(gptr(a), 0, mem::TLS_SLOT);
            return a;
        }
}

static void thread_exit_locked(Thread *t, Lock &lk);

static void *thread_main(void *arg) {
    Thread *t = (Thread *)arg;
    tls_thread = t;
    {
        Lock lk(kmutex);
        t->cv.wait(lk, [&] { return running == t; });
    }
    run_guest(&t->cpu);
    Lock lk(kmutex);
    thread_exit_locked(t, lk);
    return nullptr;
}

std::shared_ptr<Thread> create_thread(u32 entry, u32 arg, u32 stack_top, s32 prio, bool start) {
    auto t = std::make_shared<Thread>();
    t->id = next_thread_id++;
    t->priority = prio;
    t->entry = entry;
    t->tls = alloc_tls();
    t->cpu.r[0] = arg;
    t->cpu.r[13] = stack_top & ~7u;
    t->cpu.r[14] = EXIT_MAGIC;
    t->cpu.pc = entry;
    t->cpu.tls = t->tls;
    t->cpu.fpscr = 0x03C00000;  // RunFast: default NaN, flush to zero (what the SDK sets)
    t->cpu.thread = t.get();
    threads.push_back(t);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 64 << 20);
    pthread_t pt;
    if (pthread_create(&pt, &attr, thread_main, t.get())) FATAL("pthread_create failed\n");
    pthread_detach(pt);
    if (start) make_ready(t.get());
    return t;
}

static void thread_exit_locked(Thread *t, Lock &lk) {
    t->status = TStatus::Dead;
    tls_used.erase((t->tls - mem::TLS_BASE) / mem::TLS_SLOT);
    t->wakeup_waiters();
    // release mutexes it holds
    for (auto &kv : handles)
        if (auto m = std::dynamic_pointer_cast<Mutex>(kv.second))
            if (m->owner == t) {
                m->lock_count = 1;
                m->release(t);
            }
    Thread *next = best_ready();
    running = next;
    if (next) {
        next->status = TStatus::Running;
        next->cv.notify_one();
    }
    lk.unlock();
    pthread_exit(nullptr);
}

// ---------------------------------------------------------------- memory bookkeeping
struct Region {
    u32 base, size, perm, state;
};
static std::map<u32, Region> regions;  // by base

static void add_region(u32 base, u32 size, u32 perm, u32 state) { regions[base] = {base, size, perm, state}; }

static void remove_region(u32 base, u32 size) {
    // (only exact or splitting removal at region granularity)
    for (auto it = regions.begin(); it != regions.end();) {
        Region r = it->second;
        u32 rend = r.base + r.size, end = base + size;
        if (rend <= base || r.base >= end) {
            ++it;
            continue;
        }
        it = regions.erase(it);
        if (r.base < base) regions[r.base] = {r.base, base - r.base, r.perm, r.state};
        if (rend > end) regions[end] = {end, rend - end, r.perm, r.state};
    }
}

static const Region *find_region(u32 a) {
    auto it = regions.upper_bound(a);
    if (it == regions.begin()) return nullptr;
    --it;
    if (a < it->second.base + it->second.size) return &it->second;
    return nullptr;
}

// linear heap allocator: first fit over free ranges
static std::map<u32, u32> linear_free;  // base -> size

static u32 linear_alloc(u32 size) {
    for (auto it = linear_free.begin(); it != linear_free.end(); ++it) {
        if (it->second >= size) {
            u32 a = it->first, s = it->second;
            linear_free.erase(it);
            if (s > size) linear_free[a + size] = s - size;
            linear_used += size;
            return a;
        }
    }
    return 0;
}

u32 alloc_linear(u32 size) { return linear_alloc((size + 0xFFF) & ~0xFFFu); }

static void linear_free_range(u32 a, u32 size) {
    linear_free[a] = size;
    linear_used -= size;
    // merge neighbours
    auto it = linear_free.find(a);
    if (it != linear_free.begin()) {
        auto p = std::prev(it);
        if (p->first + p->second == it->first) {
            p->second += it->second;
            linear_free.erase(it);
            it = p;
        }
    }
    auto n = std::next(it);
    if (n != linear_free.end() && it->first + it->second == n->first) {
        it->second += n->second;
        linear_free.erase(n);
    }
}

// ---------------------------------------------------------------- SVCs
static void wait_on(Thread *t, std::vector<std::shared_ptr<WaitObject>> objs, bool all, s64 timeout, Lock &lk,
                    CPU *c, bool out_index) {
    t->wait_objs = objs;
    t->wait_all = all;
    t->status = TStatus::WaitSync;
    t->wait_result = RES_TIMEOUT;
    t->wait_index = -1;
    for (auto &o : objs) o->add_waiter(t);
    if (timeout >= 0) {
        u64 gen = t->wait_gen;
        add_timeout(now_ns() + (u64)timeout, [t, gen] {
            if (t->wait_gen != gen || t->status != TStatus::WaitSync) return;
            t->wait_result = RES_TIMEOUT;
            t->wait_index = -1;
            resume_thread(t);
        });
    }
    block_current(lk);
    c->r[0] = t->wait_result;
    if (out_index) c->r[1] = (u32)t->wait_index;
}

static u32 svc_control_memory(CPU *c) {
    u32 op = c->r[0], addr0 = c->r[1], addr1 = c->r[2], size = c->r[3], perm = c->r[4];
    u32 kind = op & 0xFF;
    bool linear = op & 0x10000;
    size = (size + 0xFFF) & ~0xFFFu;
    VLOG("[svc] ControlMemory op=%x addr0=%08x addr1=%08x size=%x perm=%x\n", op, addr0, addr1, size, perm);
    switch (kind) {
    case 3: {  // COMMIT
        u32 a;
        if (linear) {
            a = linear_alloc(size);
            if (!a) return RES_OUT_OF_MEMORY;
        } else {
            a = addr0;
            if (a < mem::HEAP_BASE || a + size > mem::STACK_TOP - 0x04000000) {
                LOG("[svc] ControlMemory: bad heap address %08x\n", a);
                return RES_INVALID_ADDRESS;
            }
            heap_size += size;
        }
        memset(gptr(a), 0, size);
        add_region(a, size, perm, linear ? 5 : 5);
        c->r[1] = a;
        return RES_OK;
    }
    case 1: {  // FREE
        if (addr0 >= mem::LINEAR_BASE && addr0 < mem::LINEAR_BASE + mem::LINEAR_SIZE) linear_free_range(addr0, size);
        else heap_size -= std::min(heap_size, size);
        remove_region(addr0, size);
        c->r[1] = addr0;
        return RES_OK;
    }
    case 4: {  // MAP addr1 -> addr0 (alias)
        LOG("[svc] ControlMemory MAP %08x -> %08x size %x (copying, no alias)\n", addr1, addr0, size);
        memmove(gptr(addr0), gptr(addr1), size);
        add_region(addr0, size, perm, 6);
        c->r[1] = addr0;
        return RES_OK;
    }
    case 5:
        remove_region(addr0, size);
        return RES_OK;
    case 6:
        return RES_OK;
    }
    LOG("[svc] ControlMemory: unknown op %x\n", op);
    return 0xE0E01BEE;
}

static u32 svc_query_memory(CPU *c) {
    u32 a = c->r[2];
    const Region *r = find_region(a);
    if (r) {
        c->r[1] = r->base;
        c->r[2] = r->size;
        c->r[3] = r->perm;
        c->r[4] = r->state;
    } else {
        // free gap: up to the next region
        auto it = regions.upper_bound(a);
        u32 base = a & ~0xFFFu;
        u32 end = it == regions.end() ? 0x40000000u : it->first;
        c->r[1] = base;
        c->r[2] = end - base;
        c->r[3] = 0;
        c->r[4] = 0;
    }
    c->r[5] = 0;
    return RES_OK;
}

static u32 svc_arbitrate(CPU *c, Lock &lk) {
    Handle h = c->r[0];
    u32 addr = c->r[1], type = c->r[2];
    s32 value = (s32)c->r[3];
    s64 ns = (s64)((u64)c->r[4] | ((u64)c->r[5] << 32));
    auto arb = get<Arbiter>(h);
    if (!arb) return RES_INVALID_HANDLE;
    Thread *t = current_thread();
    switch (type) {
    case 0: {  // signal
        std::vector<Thread *> ws;
        for (auto &th : threads)
            if (th->status == TStatus::WaitArb && th->arb == arb.get() && th->arb_addr == addr) ws.push_back(th.get());
        std::sort(ws.begin(), ws.end(), [](Thread *a, Thread *b) {
            return a->priority != b->priority ? a->priority < b->priority : a->ready_seq < b->ready_seq;
        });
        s32 n = value < 0 ? (s32)ws.size() : std::min<s32>(value, (s32)ws.size());
        if (g_verbose >= 3) LOG("[arb] signal %08x value %d: %zu waiting, waking %d (lr %08x)\n", addr, value, ws.size(), n, c->r[14]);
        for (s32 i = 0; i < n; i++) {
            ws[i]->wait_result = RES_OK;
            resume_thread(ws[i]);
        }
        reschedule(lk);
        return RES_OK;
    }
    case 1:
    case 2:
    case 3:
    case 4: {
        s32 mv = (s32)r32(addr);
        if (g_verbose >= 3) LOG("[arb] thread %u type %u addr %08x mem %d value %d (lr %08x)\n", t->id, type, addr, mv, value, c->r[14]);
        if (!(mv < value)) return RES_OK;
        if (type == 2 || type == 4) w32(addr, (u32)(mv - 1));
        t->status = TStatus::WaitArb;
        t->arb = arb.get();
        t->arb_addr = addr;
        t->wait_result = RES_OK;
        if (type >= 3 && ns >= 0) {
            u64 gen = t->wait_gen;
            t->wait_result = RES_TIMEOUT;
            add_timeout(now_ns() + (u64)ns, [t, gen] {
                if (t->wait_gen != gen || t->status != TStatus::WaitArb) return;
                t->wait_result = RES_TIMEOUT;
                resume_thread(t);
            });
        }
        block_current(lk);
        return t->wait_result == RES_TIMEOUT ? RES_TIMEOUT : RES_OK;
    }
    }
    return 0xD8E093ED;
}

static std::shared_ptr<WaitObject> waitable(Handle h) {
    return std::dynamic_pointer_cast<WaitObject>(get_object(h));
}

static void svc_locked(CPU *c, u32 imm, Lock &lk) {
    Thread *t = current_thread();
    switch (imm) {
    case 0x01: c->r[0] = svc_control_memory(c); break;
    case 0x02: c->r[0] = svc_query_memory(c); break;
    case 0x03:
        LOG("[svc] ExitProcess\n");
        _exit(0);
    case 0x08: {  // CreateThread(prio=r0, entry=r1, arg=r2, stack=r3, core=r4)
        s32 prio = (s32)c->r[0];
        auto nt = create_thread(c->r[1], c->r[2], c->r[3], prio, true);
        VLOG("[svc] CreateThread entry=%08x arg=%08x stack=%08x prio=%d core=%d -> id %u\n", c->r[1], c->r[2],
             c->r[3], prio, (s32)c->r[4], nt->id);
        c->r[1] = add_handle(nt);
        c->r[0] = RES_OK;
        reschedule(lk);
        break;
    }
    case 0x09:
        VLOG("[svc] ExitThread %u\n", t->id);
        thread_exit_locked(t, lk);
        break;
    case 0x0A: {  // SleepThread
        s64 ns = (s64)((u64)c->r[0] | ((u64)c->r[1] << 32));
        if (ns <= 0) {
            yield_current(lk);
        } else {
            t->status = TStatus::WaitSleep;
            u64 gen = t->wait_gen;
            add_timeout(now_ns() + (u64)ns, [t, gen] {
                if (t->wait_gen == gen && t->status == TStatus::WaitSleep) resume_thread(t);
            });
            block_current(lk);
        }
        break;
    }
    case 0x0B: {
        auto th = get<Thread>(c->r[1]);
        if (!th) { c->r[0] = RES_INVALID_HANDLE; break; }
        c->r[1] = (u32)th->priority;
        c->r[0] = RES_OK;
        break;
    }
    case 0x0C: {
        auto th = get<Thread>(c->r[0]);
        if (!th) { c->r[0] = RES_INVALID_HANDLE; break; }
        th->priority = (s32)c->r[1];
        c->r[0] = RES_OK;
        reschedule(lk);
        break;
    }
    case 0x13: {  // CreateMutex(initial=r1)
        auto m = std::make_shared<Mutex>();
        m->name = "mutex@" + std::to_string(c->r[14]);
        if (c->r[1]) m->acquire(t);
        c->r[1] = add_handle(m);
        c->r[0] = RES_OK;
        break;
    }
    case 0x14: {
        auto m = get<Mutex>(c->r[0]);
        if (!m) { c->r[0] = RES_INVALID_HANDLE; break; }
        c->r[0] = m->release(t);
        reschedule(lk);
        break;
    }
    case 0x15: {  // CreateSemaphore(initial=r1, max=r2)
        auto s = std::make_shared<Semaphore>();
        s->name = "sema@" + std::to_string(c->r[14]);
        s->count = (s32)c->r[1];
        s->max = (s32)c->r[2];
        c->r[1] = add_handle(s);
        c->r[0] = RES_OK;
        break;
    }
    case 0x16: {  // ReleaseSemaphore(handle=r1, count=r2) -> r1 old count
        auto s = get<Semaphore>(c->r[1]);
        if (!s) { c->r[0] = RES_INVALID_HANDLE; break; }
        s32 old = s->count;
        if (s->count + (s32)c->r[2] > s->max) { c->r[0] = 0xD8E007FD; break; }
        s->count += (s32)c->r[2];
        s->wakeup_waiters();
        c->r[1] = (u32)old;
        c->r[0] = RES_OK;
        reschedule(lk);
        break;
    }
    case 0x17: {  // CreateEvent(reset=r1)
        auto e = make_event((ResetType)c->r[1], "guest");
        char nm[32];
        snprintf(nm, sizeof nm, "event@%08x", c->r[14]);
        e->name = nm;
        c->r[1] = add_handle(e);
        c->r[0] = RES_OK;
        break;
    }
    case 0x18: {
        auto e = get<Event>(c->r[0]);
        if (!e) {
            LOG("[svc] SignalEvent: bad handle %08x (lr=%08x)\n", c->r[0], c->r[14]);
            c->r[0] = RES_INVALID_HANDLE;
            break;
        }
        e->signal();
        c->r[0] = RES_OK;
        reschedule(lk);
        break;
    }
    case 0x19: {
        auto e = get<Event>(c->r[0]);
        if (!e) { c->r[0] = RES_INVALID_HANDLE; break; }
        e->clear();
        c->r[0] = RES_OK;
        break;
    }
    case 0x1A: {  // CreateTimer(reset=r1)
        auto tm = std::make_shared<Timer>();
        tm->name = "timer@" + std::to_string(c->r[14]);
        tm->reset = (ResetType)c->r[1];
        c->r[1] = add_handle(tm);
        c->r[0] = RES_OK;
        break;
    }
    case 0x1B: {  // SetTimer(handle=r0, interval lo=r1, initial=r2:r3, interval hi=r4)
        auto tm = get<Timer>(c->r[0]);
        if (!tm) { c->r[0] = RES_INVALID_HANDLE; break; }
        tm->initial = (s64)((u64)c->r[2] | ((u64)c->r[3] << 32));
        tm->interval = (s64)((u64)c->r[1] | ((u64)c->r[4] << 32));
        tm->generation++;
        timer_schedule(tm, tm->initial);
        c->r[0] = RES_OK;
        break;
    }
    case 0x1C: {  // CancelTimer
        auto tm = get<Timer>(c->r[0]);
        if (tm) tm->generation++;
        c->r[0] = tm ? RES_OK : RES_INVALID_HANDLE;
        break;
    }
    case 0x1D: {  // ClearTimer
        auto tm = get<Timer>(c->r[0]);
        if (tm) tm->signaled = false;
        c->r[0] = tm ? RES_OK : RES_INVALID_HANDLE;
        break;
    }
    case 0x1E: {  // CreateMemoryBlock(other_perm=r0, addr=r1, size=r2, my_perm=r3)
        auto sm = std::make_shared<SharedMem>();
        sm->backing = c->r[1];
        sm->size = c->r[2];
        sm->my_perm = c->r[3];
        sm->other_perm = c->r[0];
        if (!sm->backing) {
            sm->backing = linear_alloc((sm->size + 0xFFF) & ~0xFFFu);
            memset(gptr(sm->backing), 0, sm->size);
        }
        c->r[1] = add_handle(sm);
        c->r[0] = RES_OK;
        break;
    }
    case 0x1F: {  // MapMemoryBlock(handle, addr, perm, other)
        auto sm = get<SharedMem>(c->r[0]);
        if (!sm) { c->r[0] = RES_INVALID_HANDLE; break; }
        u32 addr = c->r[1];
        VLOG("[svc] MapMemoryBlock %s at %08x (backing %08x size %x)\n", sm->name.c_str(), addr, sm->backing, sm->size);
        if (addr && sm->backing && addr != sm->backing) {
            // No aliasing: the block moves to where the game maps it.
            memmove(gptr(addr), gptr(sm->backing), sm->size);
            sm->backing = addr;
            if (sm->on_move) sm->on_move(addr);
        } else if (addr) {
            sm->backing = addr;
        }
        add_region(sm->backing, (sm->size + 0xFFF) & ~0xFFFu, c->r[2], 6);
        c->r[0] = RES_OK;
        break;
    }
    case 0x20:
        c->r[0] = RES_OK;
        break;
    case 0x21: {
        c->r[1] = add_handle(std::make_shared<Arbiter>());
        c->r[0] = RES_OK;
        break;
    }
    case 0x22: c->r[0] = svc_arbitrate(c, lk); break;
    case 0x23:
        c->r[0] = close_handle(c->r[0]) ? RES_OK : RES_INVALID_HANDLE;
        break;
    case 0x24: {  // WaitSynchronization1(handle=r0, ns=r2:r3)
        auto o = waitable(c->r[0]);
        s64 ns = (s64)((u64)c->r[2] | ((u64)c->r[3] << 32));
        if (!o) { c->r[0] = RES_INVALID_HANDLE; break; }
        if (!o->should_wait(t)) {
            o->acquire(t);
            o->after_wakeups();
            c->r[0] = RES_OK;
            break;
        }
        if (ns == 0) { c->r[0] = RES_TIMEOUT; break; }
        if (ns > 100000000) VLOG("[svc] thread %u waits on %s for %lld ms (lr %08x)\n", t->id, o->name.c_str(), (long long)(ns / 1000000), c->r[14]);
        wait_on(t, {o}, false, ns, lk, c, false);
        break;
    }
    case 0x25: {  // WaitSynchronizationN(ns lo=r0, handles=r1, count=r2, all=r3, ns hi=r4) -> r1 index
        s64 ns = (s64)((u64)c->r[0] | ((u64)c->r[4] << 32));
        u32 hp = c->r[1];
        s32 n = (s32)c->r[2];
        bool all = c->r[3] != 0;
        std::vector<std::shared_ptr<WaitObject>> objs;
        for (s32 i = 0; i < n; i++) {
            auto o = waitable(r32(hp + 4 * i));
            if (!o) {
                c->r[0] = RES_INVALID_HANDLE;
                return;
            }
            objs.push_back(o);
        }
        if (all) {
            bool ready = true;
            for (auto &o : objs)
                if (o->should_wait(t)) ready = false;
            if (ready) {
                for (auto &o : objs) o->acquire(t);
                c->r[0] = RES_OK;
                c->r[1] = 0;
                break;
            }
        } else {
            bool done = false;
            for (s32 i = 0; i < n; i++)
                if (!objs[i]->should_wait(t)) {
                    objs[i]->acquire(t);
                    objs[i]->after_wakeups();
                    c->r[0] = RES_OK;
                    c->r[1] = (u32)i;
                    done = true;
                    break;
                }
            if (done) break;
        }
        if (ns == 0 || n == 0) {
            c->r[0] = RES_TIMEOUT;
            if (n == 0 && ns != 0) {
                // waits for nothing: just sleep
                t->status = TStatus::WaitSleep;
                u64 gen = t->wait_gen;
                add_timeout(now_ns() + (u64)std::max<s64>(ns, 0), [t, gen] {
                    if (t->wait_gen == gen && t->status == TStatus::WaitSleep) resume_thread(t);
                });
                block_current(lk);
            }
            break;
        }
        if (ns > 100000000) VLOG("[svc] thread %u waits on %d objects (%s) for %lld ms (lr %08x)\n", t->id, n, objs[0]->name.c_str(), (long long)(ns / 1000000), c->r[14]);
        wait_on(t, objs, all, ns, lk, c, true);
        break;
    }
    case 0x27: {  // DuplicateHandle(r1)
        auto o = get_object(c->r[1]);
        if (!o) { c->r[0] = RES_INVALID_HANDLE; break; }
        c->r[1] = add_handle(o);
        c->r[0] = RES_OK;
        break;
    }
    case 0x28: {
        u64 tk = system_tick();
        c->r[0] = (u32)tk;
        c->r[1] = (u32)(tk >> 32);
        break;
    }
    case 0x29:  // GetHandleInfo
        c->r[1] = 0;
        c->r[2] = 0;
        c->r[0] = RES_OK;
        break;
    case 0x2A: {  // GetSystemInfo(type=r1, param=r2) -> r1:r2
        u32 type = c->r[1], param = c->r[2];
        u64 v = 0;
        if (type == 0) {  // memory usage by region
            if (param == 1) v = 0x04000000;            // application
            else if (param == 2) v = 0x00A00000;       // system
            else if (param == 3) v = 0x00200000;       // base
            else v = 0x04000000 + 0x00A00000 + 0x00200000;
        } else if (type == 2) {
            v = 0;
        } else if (type == 26) {
            v = 5;  // number of processes created by the kernel at boot
        }
        c->r[1] = (u32)v;
        c->r[2] = (u32)(v >> 32);
        c->r[0] = RES_OK;
        break;
    }
    case 0x2B: {  // GetProcessInfo(handle=r1, type=r2)
        u32 type = c->r[2];
        u64 v = 0;
        if (type == 0 || type == 2) v = heap_size + linear_used + 0x00400000;
        else if (type == 20) v = mem::FCRAM_PADDR - mem::LINEAR_BASE;  // linear address offset
        c->r[1] = (u32)v;
        c->r[2] = (u32)(v >> 32);
        c->r[0] = RES_OK;
        break;
    }
    case 0x2D: {  // ConnectToPort(name=r1)
        std::string name = read_cstr(c->r[1], 12);
        auto svc = find_port(name);
        if (!svc) {
            LOG("[svc] ConnectToPort(%s): no such port\n", name.c_str());
            c->r[0] = 0xD88007FA;
            break;
        }
        auto s = std::make_shared<Session>();
        s->service = svc;
        s->name = name;
        c->r[1] = add_handle(s);
        c->r[0] = RES_OK;
        break;
    }
    case 0x32: {  // SendSyncRequest(handle=r0)
        auto s = get<Session>(c->r[0]);
        if (!s) {
            LOG("[svc] SendSyncRequest: bad handle %08x\n", c->r[0]);
            c->r[0] = RES_INVALID_HANDLE;
            break;
        }
        c->r[0] = service_request(s, t, lk);
        reschedule(lk);
        break;
    }
    case 0x35:  // GetProcessId
        c->r[1] = 0x28;
        c->r[0] = RES_OK;
        break;
    case 0x37: {  // GetThreadId(r1)
        auto th = get<Thread>(c->r[1]);
        c->r[1] = th ? th->id : 0;
        c->r[0] = th ? RES_OK : RES_INVALID_HANDLE;
        break;
    }
    case 0x38:  // GetResourceLimit
        c->r[1] = add_handle(self_reslimit);
        c->r[0] = RES_OK;
        break;
    case 0x39:
    case 0x3A: {  // GetResourceLimit{Limit,Current}Values(values=r0, handle=r1, names=r2, count=r3)
        u32 vals = c->r[0], names = c->r[2], n = c->r[3];
        for (u32 i = 0; i < n; i++) {
            u32 name = r32(names + 4 * i);
            u64 v = 0;
            bool limit = imm == 0x39;
            switch (name) {
            case 0: v = limit ? 0x18 : 0x18; break;                                   // priority
            case 1: v = limit ? 0x04000000 : heap_size + linear_used + 0x00400000; break;  // commit
            case 2: v = limit ? 32 : threads.size(); break;
            case 3: v = limit ? 32 : 8; break;
            case 4: v = limit ? 32 : 8; break;
            case 5: v = limit ? 32 : 8; break;
            case 6: v = limit ? 8 : 2; break;
            case 7: v = limit ? 16 : 4; break;
            case 8: v = limit ? 2 : 0; break;
            case 9: v = limit ? 32 : 2; break;
            case 10: v = limit ? 1000 : 0; break;
            }
            w64(vals + 8 * i, v);
        }
        c->r[0] = RES_OK;
        break;
    }
    case 0x3C: {  // Break: the game's panic (failed assertion, pure virtual call...)
        for (int i = 0; i < 16; i += 4)
            LOG("[svc] r%d-r%d: %08x %08x %08x %08x\n", i, i + 3, c->r[i], c->r[i + 1], c->r[i + 2], c->r[i + 3]);
        for (u32 a = c->r[13]; a < c->r[13] + 0x80; a += 16)
            LOG("[svc] stack %08x: %08x %08x %08x %08x\n", a, r32(a), r32(a + 4), r32(a + 8), r32(a + 12));
        FATAL("[svc] Break(%u) from thread %u, lr=%08x\n", c->r[0], t->id, c->r[14]);
    }
    case 0x3D: {
        std::string s((const char *)gptr(c->r[0]), (size_t)(s32)c->r[1]);
        LOG("[guest] %s\n", s.c_str());
        c->r[0] = RES_OK;
        break;
    }
    case 0x52:
    case 0x53:
    case 0x54:
        c->r[0] = RES_OK;
        break;
    default:
        LOG("[svc] unimplemented SVC 0x%02x (lr=%08x)\n", imm, c->r[14]);
        c->r[0] = 0xF8C007F4;
        break;
    }
}

void svc_dispatch(CPU *c, u32 imm) {
    Lock lk(kmutex);
    flush_deferred();
    static int svclog = getenv("RECOMP_SVCLOG") ? atoi(getenv("RECOMP_SVCLOG")) : -1;
    if (svclog >= 0 && (svclog == 0 || current_thread()->id == (u32)svclog))
        LOG("[svc %u] %02x lr=%08x r0=%08x r1=%08x r2=%08x r3=%08x\n", current_thread()->id, imm, c->r[14], c->r[0],
            c->r[1], c->r[2], c->r[3]);
    svc_locked(c, imm, lk);
}

void dump_threads() {
    Lock lk(kmutex);
    static const char *st[] = {"ready", "running", "wait-sync", "sleep", "wait-arb", "dormant", "dead"};
    for (auto &t : threads) {
        std::string w;
        for (auto &o : t->wait_objs) w += " " + (o->name.empty() ? std::string("?") : o->name);
        LOG("[thread %u] prio %d %s pc=%08x lr=%08x sp=%08x arb=%08x%s\n", t->id, t->priority, st[(int)t->status],
            t->cpu.pc, t->cpu.r[14], t->cpu.r[13], t->arb_addr, w.c_str());
    }
}

// ---------------------------------------------------------------- boot
void start(const BootInfo &bi) {
    boot_ns = now_ns();
    linear_free[mem::LINEAR_BASE] = mem::LINEAR_SIZE - 0x00200000;  // (a little kept for the system)
    tq.th = std::thread(timeout_thread);
    tq.th.detach();
    u32 stack_top = mem::STACK_TOP;
    add_region(stack_top - bi.stack_size, bi.stack_size, 3, 7);
    Lock lk(kmutex);
    auto t = create_thread(bi.entry, 0, stack_top, bi.priority, true);
    t->name = "main";
    schedule_async();
}

}  // namespace kernel

extern "C" void rt_svc(CPU *c, u32 imm) { kernel::svc_dispatch(c, imm); }
