// HLE services: IPC command buffers and the service registry.
#pragma once
#include <initializer_list>
#include <map>
#include <memory>
#include <string>

#include "kernel.h"

namespace kernel {

struct IPC {
    Thread *t;
    u32 buf;  // guest address of the command buffer (TLS + 0x80)
    Lock *lk;
    u32 req[64];  // the request as received (replies overwrite the buffer in place)
    IPC(Thread *t_, u32 buf_, Lock *lk_) : t(t_), buf(buf_), lk(lk_) { memcpy(req, gptr(buf), sizeof req); }
    u32 word(int i) const { return req[i]; }
    u16 cmd() const { return (u16)(req[0] >> 16); }
    u32 operator[](int i) const { return req[i]; }
    void set(int i, u32 v) { w32(buf + 4 * i, v); }
    void set64(int i, u64 v) {
        set(i, (u32)v);
        set(i + 1, (u32)(v >> 32));
    }
    void header(u32 normal, u32 translate) { set(0, ((u32)cmd() << 16) | (normal << 6) | translate); }
    // header + result word
    void result(u32 res, u32 normal = 1, u32 translate = 0) {
        header(normal, translate);
        set(1, res);
    }
    void ok(u32 normal = 1, u32 translate = 0) { result(0, normal, translate); }
    // Writes a handle descriptor at word `pos` followed by the handles; returns the next word index.
    int handles(int pos, std::initializer_list<ObjPtr> objs, bool move = false);
    // The client's receive buffer for static buffer `index` (its TLS + 0x180)
    u32 static_buffer(int index, u32 *size = nullptr) const;
    // Writes a static buffer descriptor (pointing at the client's receive buffer) at word `pos`
    void static_desc(int pos, int index, u32 size);
};

struct Service : std::enable_shared_from_this<Service> {
    std::string name;
    explicit Service(std::string n) : name(std::move(n)) {}
    virtual ~Service() = default;
    virtual void handle(IPC &ipc) = 0;
    void unimplemented(IPC &ipc, bool succeed = true);
};

void register_service(std::shared_ptr<Service> s);
std::shared_ptr<Service> find_service(const std::string &name);
std::shared_ptr<Service> find_port(const std::string &name);
u32 service_request(std::shared_ptr<Session> s, Thread *t, Lock &lk);
void init_services();

// Session handle to a host service object (for services that hand out sub-sessions, e.g. fs files)
Handle new_session(std::shared_ptr<Service> s);

}  // namespace kernel
