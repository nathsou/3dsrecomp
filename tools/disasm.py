#!/usr/bin/env python3
"""Disassemble guest code: dis.py <gamedir> <addr> [count] [thumb]"""
import sys, capstone, json
d = sys.argv[1]; a = int(sys.argv[2], 16); n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
info = json.load(open(d + '/info.json'))
code = open(d + '/code.bin', 'rb').read()
base = info['text']['addr']
md = capstone.Cs(capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB if len(sys.argv) > 4 else capstone.CS_MODE_ARM)
md.skipdata = True
for i in md.disasm(code[a - base:a - base + n * 4], a):
    print('%08x  %-10s %s' % (i.address, i.mnemonic, i.op_str))
