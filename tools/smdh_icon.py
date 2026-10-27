#!/usr/bin/env python3
"""Writes the 48x48 icon of an SMDH (exefs/icon.bin) as a PPM: smdh_icon.py icon.bin out.ppm"""
import struct
import sys

data = open(sys.argv[1], 'rb').read()
px = data[0x24C0:0x24C0 + 48 * 48 * 2]
img = [[(0, 0, 0)] * 48 for _ in range(48)]


def morton(x, y):
    r = 0
    for b in range(3):
        r |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1)
    return r


i = 0
for ty in range(0, 48, 8):
    for tx in range(0, 48, 8):
        for k in range(64):
            # inverse morton within the tile
            x = y = 0
            for b in range(3):
                x |= ((k >> (2 * b)) & 1) << b
                y |= ((k >> (2 * b + 1)) & 1) << b
            v = struct.unpack_from('<H', px, i * 2)[0]
            i += 1
            r, g, b_ = (v >> 11) & 31, (v >> 5) & 63, v & 31
            img[ty + y][tx + x] = (r * 255 // 31, g * 255 // 63, b_ * 255 // 31)
with open(sys.argv[2], 'wb') as f:
    f.write(b'P6\n48 48\n255\n')
    for row in img:
        for p in row:
            f.write(bytes(p))
