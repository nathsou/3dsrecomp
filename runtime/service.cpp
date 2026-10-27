// Service registry, IPC helpers and srv: (the service manager).
#include "service.h"

#include <set>

namespace kernel {

static std::map<std::string, std::shared_ptr<Service>> services;
static std::map<std::string, std::shared_ptr<Service>> ports;

int IPC::handles(int pos, std::initializer_list<ObjPtr> objs, bool move) {
    set(pos++, (((u32)objs.size() - 1) << 26) | (move ? 0x10 : 0));
    for (auto &o : objs) set(pos++, o ? add_handle(o) : 0);
    return pos;
}

u32 IPC::static_buffer(int index, u32 *size) const {
    u32 d = r32(t->tls + 0x180 + 8 * index);
    if (size) *size = d >> 14;
    return r32(t->tls + 0x180 + 8 * index + 4);
}

void IPC::static_desc(int pos, int index, u32 size) {
    set(pos, (size << 14) | ((u32)index << 10) | 2);
    set(pos + 1, static_buffer(index));
}

void Service::unimplemented(IPC &ipc, bool succeed) {
    static std::set<std::pair<std::string, u32>> seen;
    if (seen.insert({name, ipc.word(0)}).second)
        LOG("[%s] unimplemented command 0x%08x (%s)\n", name.c_str(), ipc.word(0), succeed ? "returning success" : "failing");
    if (succeed) {
        ipc.ok(1);
    } else {
        ipc.result(0xD900182F);
    }
}

void register_service(std::shared_ptr<Service> s) { services[s->name] = s; }

std::shared_ptr<Service> find_service(const std::string &name) {
    auto it = services.find(name);
    return it == services.end() ? nullptr : it->second;
}

std::shared_ptr<Service> find_port(const std::string &name) {
    auto it = ports.find(name);
    return it == ports.end() ? nullptr : it->second;
}

Handle new_session(std::shared_ptr<Service> s) {
    auto ses = std::make_shared<Session>();
    ses->service = s;
    ses->name = s->name;
    return add_handle(ses);
}

u32 service_request(std::shared_ptr<Session> s, Thread *t, Lock &lk) {
    IPC ipc(t, t->tls + 0x80, &lk);
    if (g_verbose >= 2) LOG("[ipc] %s cmd %08x\n", s->service->name.c_str(), ipc.word(0));
    u32 cmd = ipc.word(0);
    s->service->handle(ipc);
    if (g_verbose && (ipc.word(1) & 0x80000000))
        LOG("[ipc] %s cmd %08x -> error %08x\n", s->service->name.c_str(), cmd, ipc.word(1));
    return RES_OK;
}

// ---------------------------------------------------------------- srv:
struct Srv : Service {
    std::shared_ptr<Semaphore> notif = std::make_shared<Semaphore>();
    std::vector<u32> pending;
    Srv() : Service("srv:") { notif->max = 64; }
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001:  // RegisterClient
            ipc.ok();
            break;
        case 0x0002:  // EnableNotification
            ipc.ok(1, 2);
            ipc.handles(2, {notif});
            break;
        case 0x0005: {  // GetServiceHandle(name[8], len, flags)
            char nm[9] = {};
            memcpy(nm, gptr(ipc.buf + 4), 8);
            u32 len = std::min<u32>(ipc[3], 8);
            std::string name(nm, len);
            auto svc = find_service(name);
            if (!svc) {
                LOG("[srv] GetServiceHandle(%s): not implemented, giving a stub\n", name.c_str());
                svc = make_stub(name);
            }
            VLOG("[srv] GetServiceHandle(%s)\n", name.c_str());
            auto ses = std::make_shared<Session>();
            ses->service = svc;
            ses->name = name;
            ipc.ok(1, 2);
            ipc.handles(2, {ses}, true);
            break;
        }
        case 0x0009:  // Subscribe
        case 0x000A:  // Unsubscribe
            ipc.ok();
            break;
        case 0x000B:  // ReceiveNotification
            ipc.ok(2);
            ipc.set(2, pending.empty() ? 0 : pending.front());
            if (!pending.empty()) pending.erase(pending.begin());
            break;
        case 0x000C:  // PublishToSubscriber
            ipc.ok();
            break;
        default:
            unimplemented(ipc);
        }
    }
    std::shared_ptr<Service> make_stub(const std::string &name);
};

struct StubService : Service {
    explicit StubService(std::string n) : Service(std::move(n)) {}
    void handle(IPC &ipc) override { unimplemented(ipc); }
};

std::shared_ptr<Service> Srv::make_stub(const std::string &name) {
    auto s = std::make_shared<StubService>(name);
    register_service(s);
    return s;
}

struct ErrF : Service {
    ErrF() : Service("err:f") {}
    void handle(IPC &ipc) override {
        if (ipc.cmd() == 1) {  // ThrowFatalError
            u32 type = ipc[1] & 0xFF;
            u32 code = ipc[2];
            u32 pc = ipc[3];
            FATAL("[err:f] fatal error type %u result %08x pc %08x lr %08x\n", type, code, pc, ipc[0x14 / 4]);
        }
        ipc.ok();
    }
};

void register_all_services();

void init_services() {
    ports["srv:"] = std::make_shared<Srv>();
    ports["err:f"] = std::make_shared<ErrF>();
    register_all_services();
}

}  // namespace kernel
