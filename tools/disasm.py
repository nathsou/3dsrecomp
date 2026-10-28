#!/usr/bin/env python3
"""Disassemble guest code: disasm.py <gamedir> <addr> [count] [thumb]

Addresses in CRO module slots (0x40000000 and up, see tools/recomp/cro.py) show the module's code as linked
for the analysis."""
import os
import sys

import capstone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tools.recomp.image import Image  # noqa: E402

d = sys.argv[1]; a = int(sys.argv[2], 16); n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
img = Image(d)
s = img.seg_of(a)
if s is None:
    sys.exit('%08x is not in the program or a CRO module' % a)
md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB if len(sys.argv) > 4 else capstone.CS_MODE_ARM)
md.skipdata = True
print('[%s]' % s.name)
for i in md.disasm(s.data[a - s.addr:a - s.addr + n * 4], a):
    print('%08x  %-10s %s' % (i.address, i.mnemonic, i.op_str))
