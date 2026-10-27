#!/usr/bin/env python3
"""Extract a decrypted 3DS game (.cci/.3ds NCSD or .cxi/.app NCCH) or a homebrew .3dsx into a folder:

  exheader.bin   the NCCH extended header (memory layout, services, kernel caps)
  code.bin       the decompressed ExeFS .code (text + rodata + data, page aligned)
  romfs.bin      the RomFS level-3 image the game reads through fs (IVFC stripped)
  romfs/         the RomFS files, unpacked (for inspection only; --no-unpack skips it to save disk space)
  info.json      the parsed layout

Only decrypted dumps are supported (NCCH flag NoCrypto). A .3dsx is relocated to 0x00100000 and gets a
synthetic exheader, as if it were an application started by the system (see extract_3dsx).
Usage: extract.py <dump> <outdir> [--no-unpack] [--argv0 sdmc:/3ds/<name>.3dsx]
"""
import json
import os
import struct
import sys
import zlib

MU = 0x200


def u16(b, o): return struct.unpack_from('<H', b, o)[0]
def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]


def lzss_decompress(comp):
    """The ExeFS .code backward LZ77 ("BLZ")."""
    top_bottom = u32(comp, len(comp) - 8)
    extra = u32(comp, len(comp) - 4)
    out_size = len(comp) + extra
    out = bytearray(out_size)
    out[:len(comp)] = comp
    index = len(comp) - ((top_bottom >> 24) & 0xFF)
    stop = len(comp) - (top_bottom & 0xFFFFFF)
    o = out_size
    while index > stop:
        index -= 1
        control = comp[index]
        for _ in range(8):
            if index <= stop or o <= 0:
                break
            if control & 0x80:
                index -= 2
                seg = comp[index] | (comp[index + 1] << 8)
                size = ((seg >> 12) & 15) + 3
                seg = (seg & 0xFFF) + 2
                for _ in range(size):
                    b = out[o + seg]
                    o -= 1
                    out[o] = b
            else:
                index -= 1
                o -= 1
                out[o] = comp[index]
            control = (control << 1) & 0xFF
    return bytes(out)


def parse_codeset(b, o):
    return {'addr': u32(b, o), 'pages': u32(b, o + 4), 'size': u32(b, o + 8)}


def parse_exheader(ex):
    info = {
        'title': ex[0:8].rstrip(b'\0').decode('ascii', 'replace'),
        'flags': ex[0xD],
        'compressed': bool(ex[0xD] & 1),
        'text': parse_codeset(ex, 0x10),
        'stack_size': u32(ex, 0x1C),
        'rodata': parse_codeset(ex, 0x20),
        'data': parse_codeset(ex, 0x30),
        'bss_size': u32(ex, 0x3C),
        'dependencies': ['%016x' % u64(ex, 0x40 + 8 * i) for i in range(48) if u64(ex, 0x40 + 8 * i)],
        'save_data_size': u64(ex, 0x1C0),
        'jump_id': '%016x' % u64(ex, 0x1C8),
        'program_id': '%016x' % u64(ex, 0x200),
        'core_version': u32(ex, 0x208),
        'arm11_flags': list(ex[0x20C:0x210]),
        'priority': ex[0x20F],
        'services': [ex[0x250 + 8 * i:0x258 + 8 * i].rstrip(b'\0').decode('ascii', 'replace')
                     for i in range(34) if ex[0x250 + 8 * i]],
        'kernel_caps': ['%08x' % u32(ex, 0x370 + 4 * i) for i in range(28) if u32(ex, 0x370 + 4 * i) != 0xFFFFFFFF],
    }
    # system mode / memory type from the ARM11 local caps flags
    f2 = ex[0x20E]
    info['memory_type'] = (f2 >> 4) & 0xF if False else None
    return info


UNPACK = True  # write romfs/ (the --no-unpack option only counts the files)


def extract_romfs_files(img, outdir):
    hdr_len = u32(img, 0)
    dir_meta_off = u32(img, 0x0C)
    file_meta_off = u32(img, 0x1C)
    data_off = u32(img, 0x24)
    count = 0

    def name_of(off, ln):
        return img[off:off + ln].decode('utf-16-le')

    def walk_dir(doff, path):
        nonlocal count
        d = dir_meta_off + doff
        child = u32(img, d + 8)
        file = u32(img, d + 12)
        while file != 0xFFFFFFFF:
            f = file_meta_off + file
            fdata = u64(img, f + 8)
            fsize = u64(img, f + 16)
            nlen = u32(img, f + 28)
            name = name_of(f + 32, nlen)
            p = os.path.join(path, name)
            if UNPACK:
                with open(p, 'wb') as fh:
                    fh.write(img[data_off + fdata:data_off + fdata + fsize])
            count += 1
            file = u32(img, f + 4)
        while child != 0xFFFFFFFF:
            c = dir_meta_off + child
            nlen = u32(img, c + 20)
            name = name_of(c + 24, nlen)
            p = os.path.join(path, name)
            if UNPACK:
                os.makedirs(p, exist_ok=True)
            walk_dir(child, p)
            child = u32(img, c + 4)

    if UNPACK:
        os.makedirs(outdir, exist_ok=True)
    walk_dir(0, outdir)
    return count


# ---- homebrew (.3dsx)
# A .3dsx is linked at 0x00100000 with page-aligned segments; its relocation tables say which words hold
# absolute or PC-relative addresses. Loading it at its link address makes them the identity, but we apply
# them like the Homebrew Launcher's loader does. It then runs like an ordinary application: libctru's _prm
# block (code + 4) keeps __service_ptr = 0, so libctru uses srv:, APT and the heap sizes from _prm (or all
# free memory). __system_arglist gets argv[0], so libctru's cwd is the .3dsx folder as under the launcher.
HB_BASE = 0x00100000
HB_STACK = 0x100000  # the launcher's process stack is large; libctru's __stacksize__ only sizes threads


def page(n): return (n + 0xFFF) & ~0xFFF


def extract_3dsx(d, outdir, name, argv0):
    hsize, rsize, ver, flags, csize, rosize, dsize, bss = struct.unpack_from('<HHIIIIII', d, 4)
    smdh_off = smdh_size = fs_off = 0
    if hsize > 32:
        smdh_off, smdh_size, fs_off = struct.unpack_from('<III', d, 32)
    relocs = [struct.unpack_from('<II', d, hsize + rsize * i) for i in range(3)]
    o = hsize + 3 * rsize
    segs = []
    for sz in (csize, rosize, dsize - bss):
        segs.append(bytearray(d[o:o + sz]))
        o += sz
    addrs = [HB_BASE, HB_BASE + page(csize), HB_BASE + page(csize) + page(rosize)]
    lim = [page(csize), page(csize) + page(rosize)]

    def translate(off):
        if off < lim[0]: return addrs[0] + off
        if off < lim[1]: return addrs[1] + off - lim[0]
        return addrs[2] + off - lim[1]

    for i, seg in enumerate(segs):
        for kind, count in enumerate(relocs[i]):
            pos = 0  # each table walks the segment from its start
            for _ in range(count):
                skip, patch = struct.unpack_from('<HH', d, o)
                o += 4
                pos += skip
                for _ in range(patch):
                    if pos * 4 + 4 <= len(seg):
                        v = u32(seg, pos * 4)
                        sub, a = v >> 28, translate(v & 0x0FFFFFFF)
                        if kind == 0:
                            nv = a
                        else:
                            nv = (a - (addrs[i] + pos * 4)) & 0xFFFFFFFF
                            if sub == 1: nv &= 0x7FFFFFFF  # prel31 (exception index tables)
                        struct.pack_into('<I', seg, pos * 4, nv)
                    pos += 1
    code, ro, data = segs
    title, author = 'homebrew', ''
    smdh = d[smdh_off:smdh_off + smdh_size] if smdh_size else b''
    if smdh[:4] == b'SMDH':
        title = smdh[8:8 + 0x80].decode('utf-16-le').split('\0')[0] or title
        author = smdh[8 + 0x180:8 + 0x200].decode('utf-16-le').split('\0')[0]
        os.makedirs(os.path.join(outdir, 'exefs'), exist_ok=True)
        with open(os.path.join(outdir, 'exefs', 'icon.bin'), 'wb') as f:
            f.write(smdh)
    # argv for libctru (u32 argc, then NUL-terminated strings), in the free tail of the rodata page
    arg = struct.pack('<I', 1) + argv0.encode() + b'\0'
    argaddr = addrs[1] + ((rosize + 3) & ~3)
    if argaddr + len(arg) > addrs[1] + page(rosize):
        sys.exit('3dsx: no room for the argument list after rodata')
    ro += bytes(argaddr - addrs[1] - len(ro)) + arg
    if code[4:8] == b'_prm':
        struct.pack_into('<I', code, 0x18, argaddr)
    # synthetic title id in the homebrew unique-id range 0xF0000-0xFFFFF
    uid = 0xF0000 | (zlib.crc32((title + '/' + author).encode()) & 0xFFFF)
    tid = 0x0004000000000000 | (uid << 8)
    rosize = len(ro)
    body = bytes(code) + bytes(page(csize) - csize) + bytes(ro) + bytes(page(rosize) - rosize) + bytes(data)
    with open(os.path.join(outdir, 'code.bin'), 'wb') as f:
        f.write(body)
    ex = bytearray(0x800)
    ex[0:8] = title.encode('ascii', 'replace')[:8]
    for off, a, sz in ((0x10, addrs[0], csize), (0x20, addrs[1], rosize), (0x30, addrs[2], dsize - bss)):
        struct.pack_into('<III', ex, off, a, page(sz) >> 12, sz)
    struct.pack_into('<I', ex, 0x1C, HB_STACK)
    struct.pack_into('<I', ex, 0x3C, bss)
    struct.pack_into('<Q', ex, 0x1C8, tid)
    struct.pack_into('<Q', ex, 0x200, tid)
    ex[0x20F] = 0x30
    ex[0x370:0x3F0] = b'\xff' * 0x80  # no kernel capabilities
    with open(os.path.join(outdir, 'exheader.bin'), 'wb') as f:
        f.write(ex)
    info = parse_exheader(bytes(ex))
    info.update({'title': title, 'author': author, 'format': '3dsx', 'argv0': argv0, 'code_size': len(body),
                 'product_code': 'HB-' + name})
    if fs_off and fs_off < len(d):
        img = d[fs_off:]  # the embedded RomFS is a bare level-3 image
        with open(os.path.join(outdir, 'romfs.bin'), 'wb') as f:
            f.write(img)
        info['romfs_size'] = len(img)
        info['romfs_files'] = extract_romfs_files(img, os.path.join(outdir, 'romfs'))
    return info


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, outdir = sys.argv[1], sys.argv[2]
    global UNPACK
    UNPACK = '--no-unpack' not in sys.argv
    os.makedirs(outdir, exist_ok=True)
    with open(src, 'rb') as fh:
        if fh.read(4) == b'3DSX':
            name = os.path.basename(src)
            argv0 = sys.argv[sys.argv.index('--argv0') + 1] if '--argv0' in sys.argv else 'sdmc:/3ds/' + name
            fh.seek(0)
            info = extract_3dsx(fh.read(), outdir, os.path.splitext(name)[0], argv0)
            with open(os.path.join(outdir, 'info.json'), 'w') as o:
                json.dump(info, o, indent=2)
            print(json.dumps(info, indent=2))
            return
        fh.seek(0)
        head = fh.read(0x200)
        ncch_off = 0
        if head[0x100:0x104] == b'NCSD':
            ncch_off = u32(head, 0x120) * MU  # partition 0: the executable content
        fh.seek(ncch_off)
        nh = fh.read(0x200)
        if nh[0x100:0x104] != b'NCCH':
            sys.exit('not an NCCH/NCSD image')
        flags = nh[0x188:0x190]
        if not (flags[7] & 4):
            sys.exit('encrypted NCCH: decrypt the dump first')
        exh_size = u32(nh, 0x180)
        exefs_off, exefs_size = u32(nh, 0x1A0) * MU, u32(nh, 0x1A4) * MU
        romfs_off, romfs_size = u32(nh, 0x1B0) * MU, u32(nh, 0x1B4) * MU
        fh.seek(ncch_off + 0x200)
        exheader = fh.read(0x800)
        info = parse_exheader(exheader)
        info['ncch_program_id'] = '%016x' % u64(nh, 0x118)
        info['product_code'] = nh[0x150:0x160].rstrip(b'\0').decode('ascii', 'replace')
        info['exheader_size'] = exh_size
        with open(os.path.join(outdir, 'exheader.bin'), 'wb') as o:
            o.write(exheader)

        fh.seek(ncch_off + exefs_off)
        exefs = fh.read(exefs_size)
        files = {}
        for i in range(10):
            name = exefs[i * 16:i * 16 + 8].rstrip(b'\0').decode('ascii')
            if not name:
                continue
            off, size = u32(exefs, i * 16 + 8), u32(exefs, i * 16 + 12)
            files[name] = exefs[0x200 + off:0x200 + off + size]
        os.makedirs(os.path.join(outdir, 'exefs'), exist_ok=True)
        for name, data in files.items():
            with open(os.path.join(outdir, 'exefs', name.lstrip('.') + '.bin'), 'wb') as o:
                o.write(data)
        code = files['.code']
        if info['compressed']:
            code = lzss_decompress(code)
        with open(os.path.join(outdir, 'code.bin'), 'wb') as o:
            o.write(code)
        info['code_size'] = len(code)

        if romfs_size:
            fh.seek(ncch_off + romfs_off)
            ivfc = fh.read(0x60)
            assert ivfc[0:4] == b'IVFC', 'bad IVFC'
            master_hash = u32(ivfc, 8)
            l3_bs = 1 << u32(ivfc, 0x4C)  # level 3 block size
            l3_size = u64(ivfc, 0x44)
            l3_off = (0x60 + master_hash + l3_bs - 1) & ~(l3_bs - 1)
            fh.seek(ncch_off + romfs_off + l3_off)
            img = fh.read(l3_size)
            with open(os.path.join(outdir, 'romfs.bin'), 'wb') as o:
                o.write(img)
            info['romfs_level3_offset'] = l3_off
            info['romfs_size'] = l3_size
            info['romfs_files'] = extract_romfs_files(img, os.path.join(outdir, 'romfs'))

    with open(os.path.join(outdir, 'info.json'), 'w') as o:
        json.dump(info, o, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
