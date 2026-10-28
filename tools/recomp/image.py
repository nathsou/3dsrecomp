"""The guest program image: code.bin laid out by the exheader's code sets."""
import bisect
import json
import os
import struct

from . import cro


class Segment:
    def __init__(self, name, addr, data, size, writable):
        self.name = name
        self.addr = addr
        self.data = data
        self.size = size
        self.writable = writable

    def contains(self, a):
        return self.addr <= a < self.addr + self.size


class Image:
    def __init__(self, gamedir):
        with open(os.path.join(gamedir, 'info.json')) as f:
            info = json.load(f)
        self.info = info
        with open(os.path.join(gamedir, 'code.bin'), 'rb') as f:
            code = f.read()
        t, r, d = info['text'], info['rodata'], info['data']
        base = t['addr']
        def seg(name, cs, writable):
            off = cs['addr'] - base
            return Segment(name, cs['addr'], bytearray(code[off:off + cs['size']]), cs['size'], writable)
        self.text = seg('text', t, False)
        self.rodata = seg('rodata', r, False)
        self.data = seg('data', d, True)
        self.segments = [self.text, self.rodata, self.data]
        self.texts = [self.text]
        self.entry = base
        self.volatile = set()  # words the loader patches at run time: never folded as constants
        self.extra_entries = []
        self.index()
        self.cros = cro.load(gamedir)
        if self.cros:
            self.add_cros(self.cros)
        for s in self.segments:
            s.data = bytes(s.data)

    def index(self):
        self.segments.sort(key=lambda s: s.addr)
        self._starts = [s.addr for s in self.segments]

    def add_cros(self, cs):
        """CRO modules at their slots (see cro.py), linked; the static module's patches applied to the main
        program's words."""
        patches = cs.link()
        for m in cs.mods:
            for k, (o, sz, t) in enumerate(m.segs):
                if not sz:
                    continue
                a = m.seg_addr[k]
                src = bytes(m.d[o:o + sz]) if t != 3 else bytes(sz)
                s = Segment('%s.%d' % (m.name, k), a, bytearray(src), sz, t >= 2)
                self.segments.append(s)
                if t == 0:
                    self.texts.append(s)
        self.index()
        for a, v in patches.items():
            s = self.seg_of(a)
            if s is not None and a + 4 <= s.addr + len(s.data):
                struct.pack_into('<I', s.data, a - s.addr, v)
        self.volatile = set(cs.volatile)
        self.extra_entries = cs.entry_points()

    def seg_of(self, a):
        k = bisect.bisect_right(self._starts, a) - 1
        if k >= 0 and self.segments[k].contains(a):
            return self.segments[k]
        return None

    def read32(self, a):
        s = self.seg_of(a)
        if s is None or a + 4 > s.addr + len(s.data):
            return None
        return struct.unpack_from('<I', s.data, a - s.addr)[0]

    def read16(self, a):
        s = self.seg_of(a)
        if s is None or a + 2 > s.addr + len(s.data):
            return None
        return struct.unpack_from('<H', s.data, a - s.addr)[0]

    def ro_read(self, a):
        """A word from read-only memory (text, rodata), or None."""
        if a & 3:
            return None
        s = self.seg_of(a)
        if s is None or s.writable or a in self.volatile:
            return None
        return self.read32(a)

    def in_text(self, a):
        s = self.seg_of(a)
        return s is not None and s in self.texts
