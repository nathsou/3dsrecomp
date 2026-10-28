// ldr:ro: loading CRO modules (relocatable code the game reads from its RomFS, described by static.crs for the main
// program). The module format and the loader's behaviour (rebasing the header tables, relocations, linking imports
// and exports by name/index/segment offset, the auto-link list, fixing) follow the RO system module as documented
// on 3dbrew and implemented by Citra.
//
// One difference for the recompiler: a module's code and read-only data are moved to a fixed slot (generated
// guest_cros[], see tools/recomp/cro.py) and relocated against it, since the recompiled code was generated for
// those addresses. The game only ever sees code addresses through relocations, exports and the segment table, so
// it calls the slot. The header and tables stay in the game's buffer; .data/.bss stay where the game put them.
#include <cstring>
#include <map>
#include <string>

#include "../service.h"
#include "services.h"

extern "C" {
extern const GuestCro guest_cros[];
extern const u32 guest_cro_count;
__attribute__((weak)) const GuestCro guest_cros[1] = {};
__attribute__((weak)) const u32 guest_cro_count = 0;
}

namespace kernel {
namespace {

// header fields (offsets from the module start)
enum : u32 {
    H_MAGIC = 0x80,
    H_NAME = 0x84,
    H_NEXT = 0x88,
    H_PREV = 0x8C,
    H_FILE_SIZE = 0x90,
    H_FIXED_SIZE = 0x98,
    H_ON_UNRESOLVED = 0xAC,
    H_CODE = 0xB0,
    H_DATA = 0xB8,
    H_MODULE_NAME = 0xC0,
    H_MODULE_NAME_SIZE = 0xC4,
    H_SEGMENTS = 0xC8,
    H_SEGMENT_NUM = 0xCC,
    H_EXPORT_NAMED = 0xD0,
    H_EXPORT_NAMED_NUM = 0xD4,
    H_EXPORT_INDEXED = 0xD8,
    H_EXPORT_INDEXED_NUM = 0xDC,
    H_EXPORT_STRINGS = 0xE0,
    H_EXPORT_STRINGS_SIZE = 0xE4,
    H_IMPORT_MODULES = 0xF0,
    H_IMPORT_MODULE_NUM = 0xF4,
    H_EXT_RELOCS = 0xF8,
    H_EXT_RELOC_NUM = 0xFC,
    H_IMPORT_NAMED = 0x100,
    H_IMPORT_NAMED_NUM = 0x104,
    H_IMPORT_INDEXED = 0x108,
    H_IMPORT_INDEXED_NUM = 0x10C,
    H_IMPORT_ANON = 0x110,
    H_IMPORT_ANON_NUM = 0x114,
    H_IMPORT_STRINGS = 0x118,
    H_IMPORT_STRINGS_SIZE = 0x11C,
    H_STATIC_ANON = 0x120,
    H_STATIC_ANON_NUM = 0x124,
    H_INT_RELOCS = 0x128,
    H_INT_RELOC_NUM = 0x12C,
    H_STATIC_RELOCS = 0x130,
    H_STATIC_RELOC_NUM = 0x134,
    H_END = 0x138,
    // fix levels drop the tables from these fields on
    FIX1 = H_STATIC_ANON,
    FIX2 = H_IMPORT_MODULES,
    FIX3 = H_EXPORT_NAMED,
};
constexpr u32 MAGIC_CRO0 = 0x304F5243, MAGIC_FIXD = 0x44584946;
enum { SEG_TEXT = 0, SEG_RODATA = 1, SEG_DATA = 2, SEG_BSS = 3 };

u32 g_crs;  // the static module (0 before Initialize)
// while a module is being loaded: where its .data was in the buffer and where it goes (relocations of .data are
// written to both, whichever copy the game keeps)
u32 g_old_data, g_new_data, g_data_size;

struct Cro {
    u32 m;
    explicit Cro(u32 a) : m(a) {}
    u32 f(u32 off) const { return r32(m + off); }
    void set(u32 off, u32 v) const { w32(m + off, v); }
    u32 next() const { return f(H_NEXT); }
    u32 prev() const { return f(H_PREV); }
    std::string name() const { return read_cstr(f(H_MODULE_NAME), f(H_MODULE_NAME_SIZE) + 1); }
    bool fixed() const { return f(H_MAGIC) == MAGIC_FIXD; }

    u32 seg_entry(u32 i) const { return f(H_SEGMENTS) + 12 * i; }
    // a segment tag (offset << 4 | segment) to an address, 0 if invalid
    u32 tag_addr(u32 tag) const {
        u32 idx = tag & 15, off = tag >> 4;
        if (idx >= f(H_SEGMENT_NUM)) return 0;
        u32 e = seg_entry(idx);
        if (off >= r32(e + 4)) return 0;
        return r32(e) + off;
    }

    static void apply(u32 target, u8 type, u32 addend, u32 sym, u32 future) {
        u32 v;
        switch (type) {
        case 0: return;
        case 2:
        case 38: v = sym + addend; break;
        case 3: v = sym + addend - future; break;
        default:
            LOG("[ldr:ro] unsupported relocation type %u at %08x\n", type, target);
            return;
        }
        w32(target, v);
        if (g_data_size && target - g_new_data < g_data_size) w32(g_old_data + (target - g_new_data), v);
    }

    // an external relocation batch (12-byte entries: tag, type, batch end, resolved, padding, addend)
    void apply_batch(u32 batch, u32 sym, bool reset = false) const {
        for (u32 e = batch;; e += 12) {
            u32 t = tag_addr(r32(e));
            if (t) apply(t, g8(e + 4), r32(e + 8), sym, t);
            if (g8(e + 5)) break;
        }
        w8(batch + 6, reset ? 0 : 1);
    }
    static u8 g8(u32 a) { return *gptr(a); }

    u32 unresolved() const { return tag_addr(f(H_ON_UNRESOLVED)); }

    u32 find_export(const std::string &sym) const {
        u32 n = f(H_EXPORT_NAMED_NUM), tab = f(H_EXPORT_NAMED);
        for (u32 i = 0; i < n; i++)
            if (read_cstr(r32(tab + 8 * i), 256) == sym) return tag_addr(r32(tab + 8 * i + 4));
        return 0;
    }

    // ---- rebasing (code_base: where the text and rodata segments are)
    void rebase_header(int dir) const {
        auto fix = [&](u32 off) {
            u32 v = f(off);
            if (v) set(off, v + dir * m);
        };
        fix(H_NAME);
        for (u32 off = H_CODE; off < H_END; off += 8) fix(off);
    }
    u32 rebase_segments(u32 code_base, u32 cro_size, u32 data, u32 data_size, u32 bss, u32 bss_size) const {
        u32 prev_data = 0;
        for (u32 i = 0; i < f(H_SEGMENT_NUM); i++) {
            u32 e = seg_entry(i), off = r32(e), size = r32(e + 4), type = r32(e + 8);
            if (type == SEG_DATA) {
                if (size) {
                    if (size > data_size) LOG("[ldr:ro] %s: .data %x > buffer %x\n", name().c_str(), size, data_size);
                    prev_data = off;
                    off = data;
                }
            } else if (type == SEG_BSS) {
                if (size) {
                    if (size > bss_size) LOG("[ldr:ro] %s: .bss %x > buffer %x\n", name().c_str(), size, bss_size);
                    off = bss;
                }
            } else if (off) {
                off += code_base;
            }
            w32(e, off);
        }
        (void)cro_size;
        return prev_data + m;
    }
    void unrebase_segments(u32 code_base) const {
        for (u32 i = 0; i < f(H_SEGMENT_NUM); i++) {
            u32 e = seg_entry(i), off = r32(e), type = r32(e + 8);
            if (type == SEG_BSS) off = 0;
            else if (type == SEG_DATA) off = f(H_DATA) - m;
            else if (off) off -= code_base;
            w32(e, off);
        }
    }
    void rebase_tables(int dir) const {
        auto fix = [&](u32 a) {
            u32 v = r32(a);
            if (v) w32(a, v + dir * m);
        };
        for (u32 i = 0; i < f(H_EXPORT_NAMED_NUM); i++) fix(f(H_EXPORT_NAMED) + 8 * i);
        for (u32 i = 0; i < f(H_IMPORT_MODULE_NUM); i++) {
            u32 e = f(H_IMPORT_MODULES) + 20 * i;
            fix(e);
            fix(e + 4);
            fix(e + 12);
        }
        for (u32 i = 0; i < f(H_IMPORT_NAMED_NUM); i++) {
            fix(f(H_IMPORT_NAMED) + 8 * i);
            fix(f(H_IMPORT_NAMED) + 8 * i + 4);
        }
        for (u32 i = 0; i < f(H_IMPORT_INDEXED_NUM); i++) fix(f(H_IMPORT_INDEXED) + 8 * i + 4);
        for (u32 i = 0; i < f(H_IMPORT_ANON_NUM); i++) fix(f(H_IMPORT_ANON) + 8 * i + 4);
    }

    // every external relocation to the module's own "unresolved symbol" handler
    void reset_external_relocations() const {
        u32 sym = unresolved(), n = f(H_EXT_RELOC_NUM), tab = f(H_EXT_RELOCS);
        bool begin = true;
        for (u32 i = 0; i < n; i++) {
            u32 e = tab + 12 * i;
            u32 t = tag_addr(r32(e));
            if (t) apply(t, g8(e + 4), r32(e + 8), sym, t);
            if (begin) w8(e + 6, 0);
            begin = g8(e + 5) != 0;
        }
    }
    void internal_relocations(u32 old_data) const {
        u32 n = f(H_INT_RELOC_NUM), tab = f(H_INT_RELOCS), nseg = f(H_SEGMENT_NUM);
        for (u32 i = 0; i < n; i++) {
            u32 e = tab + 12 * i, tag = r32(e);
            u32 future = tag_addr(tag);
            u8 symseg = g8(e + 5);
            if (!future || symseg >= nseg) continue;
            u32 target = future;
            if (r32(seg_entry(tag & 15) + 8) == SEG_DATA) target = old_data + (tag >> 4);
            apply(target, g8(e + 4), r32(e + 8), r32(seg_entry(symseg)), future);
        }
    }
    void clear_relocations() const {
        // back to the file's state (zeros) so the buffer could be loaded again
        auto clear = [](u32 t, u8 type) {
            if (t && (type == 2 || type == 3 || type == 38)) w32(t, 0);
        };
        for (u32 i = 0; i < f(H_EXT_RELOC_NUM); i++) {
            u32 e = f(H_EXT_RELOCS) + 12 * i;
            clear(tag_addr(r32(e)), g8(e + 4));
        }
        for (u32 i = 0; i < f(H_INT_RELOC_NUM); i++) {
            u32 e = f(H_INT_RELOCS) + 12 * i;
            clear(tag_addr(r32(e)), g8(e + 4));
        }
    }

    // ---- linking
    template <class F> static void for_each_auto_link(F fn) {
        for (u32 a = g_crs; a; a = Cro(a).next())
            if (!fn(Cro(a))) break;
    }
    // imports this module's symbols from the loaded ones
    void import_from(const Cro &src) const {
        std::string sname = src.name();
        for (u32 i = 0; i < f(H_IMPORT_NAMED_NUM); i++) {
            u32 e = f(H_IMPORT_NAMED) + 8 * i, batch = r32(e + 4);
            if (g8(batch + 6)) continue;
            std::string sym = read_cstr(r32(e), 256);
            u32 a = src.find_export(sym == "__aeabi_atexit" ? "nnroAeabiAtexit_" : sym);
            if (a) apply_batch(batch, a);
        }
        for (u32 i = 0; i < f(H_IMPORT_MODULE_NUM); i++) {
            u32 e = f(H_IMPORT_MODULES) + 20 * i;
            if (read_cstr(r32(e), 256) != sname) continue;
            for (u32 j = 0; j < r32(e + 8); j++) {
                u32 ie = r32(e + 4) + 8 * j, idx = r32(ie);
                if (idx >= src.f(H_EXPORT_INDEXED_NUM)) continue;
                u32 a = src.tag_addr(r32(src.f(H_EXPORT_INDEXED) + 4 * idx));
                apply_batch(r32(ie + 4), a);
            }
            for (u32 j = 0; j < r32(e + 16); j++) {
                u32 ie = r32(e + 12) + 8 * j;
                apply_batch(r32(ie + 4), src.tag_addr(r32(ie)));
            }
        }
    }
    // resets the imports of target that resolve to this module
    void reset_exports_of(const Cro &target) const {
        std::string me = name();
        u32 sym = target.unresolved();
        for (u32 i = 0; i < target.f(H_IMPORT_NAMED_NUM); i++) {
            u32 e = target.f(H_IMPORT_NAMED) + 8 * i, batch = r32(e + 4);
            if (!g8(batch + 6)) continue;
            std::string s = read_cstr(r32(e), 256);
            if (find_export(s == "__aeabi_atexit" ? "nnroAeabiAtexit_" : s)) target.apply_batch(batch, sym, true);
        }
        for (u32 i = 0; i < target.f(H_IMPORT_MODULE_NUM); i++) {
            u32 e = target.f(H_IMPORT_MODULES) + 20 * i;
            if (read_cstr(r32(e), 256) != me) continue;
            for (u32 j = 0; j < r32(e + 8); j++) target.apply_batch(r32(r32(e + 4) + 8 * j + 4), sym, true);
            for (u32 j = 0; j < r32(e + 16); j++) target.apply_batch(r32(r32(e + 12) + 8 * j + 4), sym, true);
        }
    }
    void link() const {
        for_each_auto_link([&](const Cro &src) {
            if (src.m != m) import_from(src);
            return true;
        });
        for_each_auto_link([&](const Cro &target) {
            if (target.m != m) target.import_from(*this);
            return true;
        });
    }
    void unlink() const {
        u32 sym = unresolved();
        for (u32 i = 0; i < f(H_IMPORT_NAMED_NUM); i++) apply_batch(r32(f(H_IMPORT_NAMED) + 8 * i + 4), sym, true);
        for (u32 i = 0; i < f(H_IMPORT_INDEXED_NUM); i++) apply_batch(r32(f(H_IMPORT_INDEXED) + 8 * i + 4), sym, true);
        for (u32 i = 0; i < f(H_IMPORT_ANON_NUM); i++) apply_batch(r32(f(H_IMPORT_ANON) + 8 * i + 4), sym, true);
        for_each_auto_link([&](const Cro &target) {
            if (target.m != m) reset_exports_of(target);
            return true;
        });
    }

    // ---- the module lists: the static module's next = auto-link list, previous = the others; a list's head
    // has previous = its tail
    void register_in(bool auto_link) const {
        Cro crs(g_crs);
        u32 head = auto_link ? crs.next() : crs.prev();
        if (head) {
            Cro h(head), tail(Cro(head).prev());
            set(H_PREV, tail.m);
            tail.set(H_NEXT, m);
            h.set(H_PREV, m);
        } else {
            set(H_PREV, m);
            crs.set(auto_link ? H_NEXT : H_PREV, m);
        }
        set(H_NEXT, 0);
    }
    void unregister() const {
        Cro crs(g_crs);
        u32 next_head = crs.next(), prev_head = crs.prev();
        Cro nx(next()), pv(prev());
        if (m == next_head || m == prev_head) {
            if (nx.m) nx.set(H_PREV, pv.m);
            crs.set(m == prev_head ? H_PREV : H_NEXT, nx.m);
        } else if (nx.m) {
            pv.set(H_NEXT, nx.m);
            nx.set(H_PREV, pv.m);
        } else {
            pv.set(H_NEXT, 0);
            if (next_head && Cro(next_head).prev() == m) Cro(next_head).set(H_PREV, pv.m);
            else if (prev_head && Cro(prev_head).prev() == m) Cro(prev_head).set(H_PREV, pv.m);
        }
        set(H_NEXT, 0);
        set(H_PREV, 0);
    }

    u32 fix(u32 level, u32 cro_size) const {
        static const u32 barrier[] = {H_END, FIX1, FIX2, FIX3};
        if (level == 0 || level > 3) return cro_size;
        u32 end = f(barrier[level]);
        if (!end) return cro_size;
        for (u32 off = barrier[level]; off < H_END; off += 8) {
            set(off, end);
            set(off + 4, 0);
        }
        u32 size = (end - m + 0xFFF) & ~0xFFFu;
        set(H_MAGIC, MAGIC_FIXD);
        set(H_FIXED_SIZE, size);
        return std::min(size, cro_size);
    }
};

const GuestCro *slot_of(const std::string &name) {
    for (u32 i = 0; i < guest_cro_count; i++)
        if (name == guest_cros[i].name) return &guest_cros[i];
    return nullptr;
}

// loaded modules by address: where the game's buffer is (the real loader maps the buffer at the module address,
// so both show the same bytes: here the address gets a copy, written back after every change), how much of it
// the module keeps after fixing, and where its code is
struct Loaded {
    u32 buf, size, code_base;
};
std::map<u32, Loaded> g_mods;
void sync_buffers() {
    for (auto &[addr, l] : g_mods)
        if (addr != l.buf) memmove(gptr(l.buf), gptr(addr), l.size);
}

struct Ro : Service {
    using Service::Service;
    void handle(IPC &ipc) override {
        switch (ipc.cmd()) {
        case 0x0001: {  // Initialize(crs buffer, size, address, process)
            u32 buf = ipc[1], size = ipc[2], addr = ipc[3];
            if (addr != buf) memmove(gptr(addr), gptr(buf), size);
            g_crs = addr;
            Cro crs(addr);
            crs.set(H_NEXT, 0);
            crs.set(H_PREV, 0);
            crs.rebase_header(1);
            crs.rebase_tables(1);
            crs.reset_external_relocations();
            g_mods[addr] = {buf, size, 0};
            sync_buffers();
            LOG("[ldr:ro] Initialize: static module at %08x (%x bytes), %u imports from modules\n", addr, size,
                crs.f(H_IMPORT_MODULE_NUM));
            ipc.ok();
            break;
        }
        case 0x0002:  // LoadCRR (hashes of the modules: not checked)
        case 0x0003:  // UnloadCRR
            ipc.ok();
            break;
        case 0x0004:  // LoadCRO
        case 0x0009:  // LoadCRO (new)
            load(ipc);
            break;
        case 0x0005: {  // UnloadCRO(address, zero, buffer, process)
            u32 addr = ipc[1], buf = ipc[3];
            Cro c(addr);
            std::string name = c.name();
            u32 code_base = g_mods.count(addr) ? g_mods[addr].code_base : addr;
            c.unregister();
            c.unlink();
            if (!c.fixed()) c.clear_relocations();
            c.unrebase_segments(code_base);
            c.rebase_tables(-1);
            c.rebase_header(-1);
            sync_buffers();
            if (addr != buf) memmove(gptr(buf), gptr(addr), g_mods.count(addr) ? g_mods[addr].size : 0);
            g_mods.erase(addr);
            LOG("[ldr:ro] unloaded %s from %08x\n", name.c_str(), addr);
            ipc.ok();
            break;
        }
        case 0x0006: {  // LinkCRO(address)
            Cro(ipc[1]).link();
            sync_buffers();
            ipc.ok();
            break;
        }
        case 0x0007: {  // UnlinkCRO(address)
            Cro(ipc[1]).unlink();
            sync_buffers();
            ipc.ok();
            break;
        }
        case 0x0008:  // Shutdown
            g_crs = 0;
            g_mods.clear();
            ipc.ok();
            break;
        default:
            unimplemented(ipc);
        }
    }

    void load(IPC &ipc) {
        u32 buf = ipc[1], addr = ipc[2], size = ipc[3], data = ipc[4], data_size = ipc[6], bss = ipc[7],
            bss_size = ipc[8], auto_link = ipc[9] & 0xFF, level = ipc[10];
        if (addr != buf) memmove(gptr(addr), gptr(buf), size);
        Cro c(addr);
        std::string name = read_cstr(addr + c.f(H_MODULE_NAME), c.f(H_MODULE_NAME_SIZE) + 1);
        if (!g_crs || c.f(H_MAGIC) != MAGIC_CRO0 || c.next() || c.prev()) {
            LOG("[ldr:ro] LoadCRO %s at %08x: not a fresh module (crs %08x)\n", name.c_str(), addr, g_crs);
            ipc.result(0xD9012C19, 2);
            ipc.set(2, 0);
            return;
        }
        u32 code_base = addr;
        const GuestCro *slot = slot_of(name);
        if (slot && size <= slot->size) {
            code_base = slot->base;
            memcpy(gptr(code_base), gptr(addr), size);
        } else {
            LOG("[ldr:ro] %s has no recompiled code slot: its code will be missing\n", name.c_str());
        }
        c.rebase_header(1);
        u32 old_data = c.rebase_segments(code_base, size, data, data_size, bss, bss_size);
        g_old_data = old_data;
        g_new_data = data;
        g_data_size = data_size;
        c.rebase_tables(1);
        c.reset_external_relocations();
        c.internal_relocations(old_data);
        c.link();
        c.register_in(auto_link != 0);
        g_data_size = 0;
        u32 fixed = c.fix(level, size);
        // the game copies the relocated .data from its buffer into the .data block itself, then frees what the
        // module no longer needs
        if (addr != buf) memmove(gptr(buf), gptr(addr), size);
        g_mods[addr] = {buf, fixed, code_base};
        sync_buffers();
        LOG("[ldr:ro] loaded %s at %08x (buffer %08x, %x bytes, code at %08x, data %08x+%x, bss %08x+%x, auto link %u, "
            "fix level %u -> %x)\n",
            name.c_str(), addr, buf, size, code_base, data, data_size, bss, bss_size, auto_link, level, fixed);
        ipc.ok(2);
        ipc.set(2, fixed);
    }
};

}  // namespace

void register_ro() { register_service(std::make_shared<Ro>("ldr:ro")); }

}  // namespace kernel
