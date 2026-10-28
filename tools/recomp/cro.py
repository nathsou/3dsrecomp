"""CRO modules (dynamically loaded code: `ldr:ro`, the static module's exports in static.crs).

A game that uses CROs reads them from its RomFS and asks ldr:ro to load them at an address of its heap, which
changes from run to run. Recompiled code needs fixed addresses, so every CRO gets a fixed *slot* in an otherwise
unused part of the address space (CRO_BASE and up, in name order): the runtime's ldr:ro copies the module there
and relocates the code and read-only data segments against the slot instead of the buffer, so every pointer to
module code (relocations, exports, return addresses) is a slot address the recompiled code knows. The writable
data and bss segments stay where the game put them.

For the analysis, each module is linked here the same way (its .data at a scratch address after the slot), so
that function pointers in its tables are found. Words that the loader patches at run time (relocation targets)
are reported so the recompiler never folds them as constants: that includes words of the main program patched
with addresses of module symbols (static.crs imports).
"""
import struct

CRO_BASE = 0x40000000
SLOT_ALIGN = 0x10000


def u32(b, o):
    return struct.unpack_from('<I', b, o)[0]


def romfs_files(path, want):
    """{path: bytes} of the RomFS image files whose name satisfies want(name)."""
    out = {}
    with open(path, 'rb') as f:
        hdr = f.read(0x28)
        dir_meta_off, dir_meta_size = u32(hdr, 0x0C), u32(hdr, 0x10)
        file_meta_off, file_meta_size = u32(hdr, 0x1C), u32(hdr, 0x20)
        data_off = u32(hdr, 0x24)
        f.seek(dir_meta_off)
        dmeta = f.read(dir_meta_size)
        f.seek(file_meta_off)
        fmeta = f.read(file_meta_size)

        def walk(doff, prefix):
            child, file = u32(dmeta, doff + 8), u32(dmeta, doff + 12)
            while file != 0xFFFFFFFF:
                fdata, fsize = struct.unpack_from('<QQ', fmeta, file + 8)
                nlen = u32(fmeta, file + 28)
                name = fmeta[file + 32:file + 32 + nlen].decode('utf-16-le')
                if want(name):
                    f.seek(data_off + fdata)
                    out[prefix + name] = f.read(fsize)
                file = u32(fmeta, file + 4)
            while child != 0xFFFFFFFF:
                nlen = u32(dmeta, child + 20)
                name = dmeta[child + 24:child + 24 + nlen].decode('utf-16-le')
                walk(child, prefix + name + '/')
                child = u32(dmeta, child + 4)
        walk(0, '/')
    return out


class Module:
    def __init__(self, path, data):
        self.path = path
        self.d = bytearray(data)
        h = lambda o: u32(data, o)
        self.h = h
        if data[0x80:0x84] != b'CRO0':
            raise ValueError('not a CRO: ' + path)
        self.name = self.cstr(h(0xC0))
        self.segs = [struct.unpack_from('<III', data, h(0xC8) + 12 * k) for k in range(h(0xCC))]
        self.base = None    # slot address of file offset 0
        self.seg_addr = []  # analysis address of each segment

    def cstr(self, off):
        e = self.d.index(b'\0', off)
        return self.d[off:e].decode('latin-1')

    def seg_size(self, typ):
        return sum(s for o, s, t in self.segs if t == typ)

    def slot_size(self):
        data, bss = self.seg_size(2), self.seg_size(3)
        return (len(self.d) + data + bss + 0x2000 + SLOT_ALIGN - 1) & ~(SLOT_ALIGN - 1)

    def tag_addr(self, tag):
        idx, off = tag & 15, tag >> 4
        if idx >= len(self.seg_addr):
            return None
        return self.seg_addr[idx] + off

    def named_exports(self):
        h = self.h
        out = {}
        for k in range(h(0xD4)):
            no, tag = struct.unpack_from('<II', self.d, h(0xD0) + 8 * k)
            out[self.cstr(no)] = tag
        return out


class CroSet:
    """The static module (main program exports/imports) and the CRO modules of a game, linked at their slots."""

    def __init__(self, files):
        self.crs = None
        self.mods = []
        for p in sorted(files):
            if p.endswith('.crs'):
                self.crs = Module(p, files[p])
            elif p.endswith('.cro'):
                self.mods.append(Module(p, files[p]))
        self.mods.sort(key=lambda m: m.name)
        a = CRO_BASE
        for m in self.mods:
            m.base = a
            a += m.slot_size()
        self.end = a
        if self.crs:
            self.crs.seg_addr = [o for o, s, t in self.crs.segs]  # the static module's segments are absolute
        for m in self.mods:
            data_at = (m.base + len(m.d) + 0xFFF) & ~0xFFF
            bss_at = (data_at + m.seg_size(2) + 0xFFF) & ~0xFFF
            m.data_at = data_at
            m.seg_addr = [data_at if t == 2 else bss_at if t == 3 else m.base + o for o, s, t in m.segs]
        self.by_name = {m.name: m for m in self.mods}
        if self.crs:
            self.by_name[self.crs.name] = self.crs
        self.volatile = set()   # words patched by the loader (absolute addresses)
        self.patches = {}       # address -> value as linked here (main program words included)

    def link(self):
        """Applies every module's relocations as if all modules were loaded. Returns patches for the main
        program's words (address -> value)."""
        for m in self.mods:
            h = m.h
            for k in range(h(0x12C)):
                tag, typ, symseg, _, _, add = struct.unpack_from('<IBBBBI', m.d, h(0x128) + 12 * k)
                t = m.tag_addr(tag)
                if t is None or symseg >= len(m.seg_addr):
                    continue
                v = (m.seg_addr[symseg] + add) & 0xFFFFFFFF
                self.patch(m, t, typ, v, deterministic=m.segs[symseg][2] in (0, 1))
        for m in self.mods + ([self.crs] if self.crs else []):
            self.link_imports(m)
        return self.patches

    def patch(self, m, addr, typ, value, deterministic):
        if typ == 3:
            value = (value - addr) & 0xFFFFFFFF
        elif typ not in (2, 38):
            return
        self.patches[addr] = value
        if not deterministic:
            self.volatile.add(addr)

    def batch(self, m, off, value, deterministic=False):
        """Applies the external relocation batch at file offset off of module m."""
        while True:
            tag, typ, end, res, _, add = struct.unpack_from('<IBBBBI', m.d, off)
            t = m.tag_addr(tag)
            if t is not None:
                self.patch(m, t, typ, (value + add) & 0xFFFFFFFF, deterministic)
            if end:
                break
            off += 12

    def find_named(self, name):
        for src in ([self.crs] if self.crs else []) + self.mods:
            ex = src.named_exports()
            if name in ex:
                return src.tag_addr(ex[name])
        return None

    def link_imports(self, m):
        h = m.h
        for k in range(h(0x104)):
            no, bo = struct.unpack_from('<II', m.d, h(0x100) + 8 * k)
            name = m.cstr(no)
            v = self.find_named('nnroAeabiAtexit_' if name == '__aeabi_atexit' else name)
            if v is None:
                v = self.find_named(name)
            if v is not None:
                self.batch(m, bo, v)
        for k in range(h(0xF4)):
            no, io, inum, ao, anum = struct.unpack_from('<IIIII', m.d, h(0xF0) + 20 * k)
            src = self.by_name.get(m.cstr(no))
            if src is None:
                continue
            for j in range(inum):
                idx, bo = struct.unpack_from('<II', m.d, io + 8 * j)
                if idx < src.h(0xDC):
                    v = src.tag_addr(u32(src.d, src.h(0xD8) + 4 * idx))
                    if v is not None:
                        self.batch(m, bo, v, src is self.crs)
            for j in range(anum):
                tag, bo = struct.unpack_from('<II', m.d, ao + 8 * j)
                v = src.tag_addr(tag)
                if v is not None:
                    # symbols of the main program are always there; another module's may be unloaded
                    self.batch(m, bo, v, src is self.crs)

    def entry_points(self):
        """Code addresses named by the module headers (control function, OnLoad, OnExit, OnUnresolved)."""
        out = []
        for m in self.mods:
            for o in (0xA0, 0xA4, 0xA8, 0xAC):
                tag = m.h(o)
                if tag != 0xFFFFFFFF:
                    a = m.tag_addr(tag)
                    if a is not None:
                        out.append(a)
        return out


def load(gamedir):
    """The game's CroSet, or None if it has no static.crs."""
    import os
    path = os.path.join(gamedir, 'romfs.bin')
    if not os.path.exists(path):
        return None
    files = romfs_files(path, lambda n: n.endswith('.cro') or n.endswith('.crs'))
    if not any(p.endswith('.crs') for p in files):
        return None
    s = CroSet(files)
    return s if s.mods else None
