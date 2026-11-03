"""C++ classes from the game's RTTI, for profiles that name classes instead of addresses (runtime/profile.cpp).

A class's typeinfo object is {vptr of its typeinfo class, name, ...}, its name the mangled one ("9BsMenuMap"), and
its vtable {offset to top, typeinfo, functions...}: the address objects hold (the "vptr") is just after the
typeinfo word. A class's primary vtable has offset to top 0; secondary ones (multiple inheritance) are negative.
Classes in CRO modules are found at their slots (tools/recomp/cro.py), where the runtime's loader puts them.

Only classes outside namespaces with exactly one primary vtable are listed (by their plain name).
"""
import re
import struct

_NAME = re.compile(rb'(\d+)([A-Za-z_]\w*)\0')


def classes(img):
    """{class name: vptr} for the program and its CRO modules"""
    segs = img.segments
    names = {}  # string address -> class name
    for s in segs:
        d = bytes(s.data)
        for m in _NAME.finditer(d):
            if int(m.group(1)) == len(m.group(2)) and (m.start() == 0 or d[m.start() - 1] == 0):
                names[s.addr + m.start()] = m.group(2).decode()
    typeinfo = {}  # typeinfo address -> class name
    words = []
    for s in segs:
        d = bytes(s.data)
        n = len(d) // 4
        ws = struct.unpack_from('<%dI' % n, d)
        words.append((s.addr, ws))
        for i, w in enumerate(ws):
            if w in names and i:
                typeinfo[s.addr + 4 * i - 4] = names[w]
    vptrs = {}  # class name -> [vptr]
    for base, ws in words:
        for i, w in enumerate(ws):
            if w in typeinfo and i and ws[i - 1] == 0:
                vptrs.setdefault(typeinfo[w], []).append(base + 4 * i + 4)
    return {n: v[0] for n, v in vptrs.items() if len(v) == 1}


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from tools.recomp.image import Image
    cs = classes(Image(sys.argv[1]))
    for n in sys.argv[2:] or sorted(cs):
        print('%-32s %s' % (n, '%08x' % cs[n] if n in cs else 'not found'))
