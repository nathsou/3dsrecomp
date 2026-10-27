"""The guest program image: code.bin laid out by the exheader's code sets."""
import json
import os
import struct


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
            return Segment(name, cs['addr'], code[off:off + cs['size']], cs['size'], writable)
        self.text = seg('text', t, False)
        self.rodata = seg('rodata', r, False)
        self.data = seg('data', d, True)
        self.segments = [self.text, self.rodata, self.data]
        self.entry = base

    def seg_of(self, a):
        for s in self.segments:
            if s.contains(a):
                return s
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
        if s is None or s.writable:
            return None
        return self.read32(a)

    def in_text(self, a):
        return self.text.contains(a)
