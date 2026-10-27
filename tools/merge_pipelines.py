#!/usr/bin/env python3
"""Merge pipeline lists (the runtime's shadercache/<title id>.bin files, games/<title id>/pipelines.bin) or
vertex program lists (shadercache/<title id>.vs.bin, games/<title id>/vsprograms.bin):
tools/merge_pipelines.py <out> <in...>. Keys are deduplicated, first occurrence order kept; <out> may be
one of the inputs. All inputs must be lists of the same kind."""
import struct
import sys

MAGIC = None

KINDS = {0x59454B50: (1, 'pipelines'), 0x52505356: (1, 'vertex programs')}  # magic: version, name


def read(path):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 8:
        return []
    magic, version = struct.unpack_from('<II', data)
    if KINDS.get(magic, (None,))[0] != version:
        sys.exit(f'{path}: not a known list (magic {magic:08x} version {version})')
    global MAGIC
    if MAGIC is not None and magic != MAGIC:
        sys.exit(f'{path}: another kind of list than the previous inputs')
    MAGIC = magic
    keys, p = [], 8
    while p + 4 <= len(data):
        n, = struct.unpack_from('<I', data, p)
        if n == 0 or p + 4 + n > len(data):
            break
        keys.append(data[p + 4:p + 4 + n])
        p += 4 + n
    return keys


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    keys, seen = [], set()
    for path in ins:
        for k in read(path):
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(out, 'wb') as f:
        f.write(struct.pack('<II', MAGIC, KINDS[MAGIC][0]))
        for k in keys:
            f.write(struct.pack('<I', len(k)) + k)
    print(f'{out}: {len(keys)} {KINDS[MAGIC][1]}')


if __name__ == '__main__':
    main()
