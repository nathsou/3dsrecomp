#!/usr/bin/env python3
"""Compare two PNGs written by gpu_replay (RGBA8, filter 0): print differing pixel count,
max channel delta and the bounding box of differences."""
import struct
import sys
import zlib


def load(path):
    data = open(path, 'rb').read()
    pos = 8
    w = h = 0
    idat = b''
    while pos < len(data):
        n, = struct.unpack('>I', data[pos:pos + 4])
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if typ == b'IHDR':
            w, h = struct.unpack('>II', body[:8])
        elif typ == b'IDAT':
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[y * (w * 4 + 1) + 1:(y + 1) * (w * 4 + 1)] for y in range(h)]
    return w, h, rows


def save(path, w, h, rows):
    raw = b''.join(b'\0' + bytes(r) for r in rows)
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    open(path, 'wb').write(png)


def main():
    """pngdiff.py a.png b.png [diff.png [threshold]]"""
    w1, h1, a = load(sys.argv[1])
    w2, h2, b = load(sys.argv[2])
    if (w1, h1) != (w2, h2):
        print('size differs')
        return
    thr = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n = 0
    big = 0
    mx = 0
    box = [w1, h1, -1, -1]
    vis = []
    for y in range(h1):
        ra, rb = a[y], b[y]
        row = bytearray(ra)
        for i in range(0, len(row), 4):  # dimmed copy of a
            row[i], row[i + 1], row[i + 2] = row[i] // 4, row[i + 1] // 4, row[i + 2] // 4
        if ra != rb:
            for x in range(w1):
                pa, pb = ra[4 * x:4 * x + 4], rb[4 * x:4 * x + 4]
                if pa != pb:
                    n += 1
                    d = max(abs(pa[i] - pb[i]) for i in range(4))
                    mx = max(mx, d)
                    if d > thr:
                        big += 1
                        row[4 * x:4 * x + 4] = bytes((255, max(0, 255 - d * 2), 0, 255))
                    box = [min(box[0], x), min(box[1], y), max(box[2], x), max(box[3], y)]
        vis.append(row)
    print(f'{n} pixels differ ({big} by more than {thr}), max delta {mx}, box {box}')
    if len(sys.argv) > 3:
        save(sys.argv[3], w1, h1, vis)


main()
