"""Decryption of 3DS content with your own console's keys, for tools/extract.py: the CIA layer (AES-CBC with the
ticket's title key) and the NCCH layer (AES-CTR with keyslot keys), read on the fly through a file-like object
so the decrypted image never has to be written out.

No keys are in this repository. They come from files you dump from your own console:
  --boot9 boot9.bin     KeyX 0x2C (NCCH) and 0x3D (title keys), from the boot ROM (boot9strap: hold
                        Start + Select + X at power-on; or GodMode9's [M:] drive)
  --keys aes_keys.txt   the Citra/Azahar key file: common0..common5 (KeyY 0x3D of the ticket's common key),
                        slot0x25KeyX / slot0x18KeyX / slot0x1BKeyX (NCCH made for firmware 7.x and the New 3DS,
                        set by NATIVE_FIRM, not in boot9), and slot0x2CKeyX / slot0x3DKeyX in place of boot9.
  --seeddb seeddb.bin   per-title seeds, for titles that use seed crypto (from firmware 9.6 on): GodMode9's
                        "Build support files" writes it (gm9/out/seeddb.bin) from the seeds of the titles installed
                        on the console; Azahar's sysdata/seeddb.bin has the same format
  --seed <32 hex digits> one title's seed instead
AES comes from the system: macOS's CommonCrypto, elsewhere OpenSSL's libcrypto.
"""
import ctypes
import ctypes.util
import hashlib
import os
import struct
import sys

M128 = (1 << 128) - 1
SCRAMBLER = 0x1FF9E9AAC5FE0408024591DC5D52768A

MODE_CBC, MODE_CTR = 2, 4
if sys.platform == 'darwin':
    _cc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    _cc.CCCryptorCreateWithMode.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.c_int, ctypes.c_uint32,
                                            ctypes.POINTER(ctypes.c_void_p)]
    _cc.CCCryptorUpdate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p,
                                    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
    _cc.CCCryptorRelease.argtypes = [ctypes.c_void_p]
    DECRYPT, AES, CTR_BE = 1, 0, 2

    def _aes(mode, key, iv, data):
        ref = ctypes.c_void_p()
        st = _cc.CCCryptorCreateWithMode(DECRYPT, mode, AES, 0, iv, key, 16, None, 0, 0,
                                         CTR_BE if mode == MODE_CTR else 0, ctypes.byref(ref))
        assert st == 0, 'CCCryptorCreateWithMode: %d' % st
        out = ctypes.create_string_buffer(len(data))
        moved = ctypes.c_size_t()
        st = _cc.CCCryptorUpdate(ref, data, len(data), out, len(data), ctypes.byref(moved))
        _cc.CCCryptorRelease(ref)
        assert st == 0 and moved.value == len(data), 'CCCryptorUpdate: %d' % st
        return out.raw
else:
    _lib = ctypes.util.find_library('crypto')
    assert _lib, 'AES needs OpenSSL (libcrypto)'
    _ssl = ctypes.CDLL(_lib)
    _ssl.EVP_CIPHER_CTX_new.restype = ctypes.c_void_p
    _ssl.EVP_CIPHER_CTX_free.argtypes = [ctypes.c_void_p]
    _ssl.EVP_aes_128_cbc.restype = ctypes.c_void_p
    _ssl.EVP_aes_128_ctr.restype = ctypes.c_void_p
    _ssl.EVP_DecryptInit_ex.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
                                        ctypes.c_char_p]
    _ssl.EVP_CIPHER_CTX_set_padding.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _ssl.EVP_DecryptUpdate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_char_p, ctypes.c_int]

    def _aes(mode, key, iv, data):
        ctx = _ssl.EVP_CIPHER_CTX_new()
        assert ctx, 'EVP_CIPHER_CTX_new'
        try:
            cipher = _ssl.EVP_aes_128_cbc() if mode == MODE_CBC else _ssl.EVP_aes_128_ctr()  # (CTR: big-endian)
            assert _ssl.EVP_DecryptInit_ex(ctx, cipher, None, key, iv) == 1, 'EVP_DecryptInit_ex'
            _ssl.EVP_CIPHER_CTX_set_padding(ctx, 0)
            out = ctypes.create_string_buffer(len(data))
            moved = ctypes.c_int()
            assert _ssl.EVP_DecryptUpdate(ctx, out, ctypes.byref(moved), data, len(data)) == 1, 'EVP_DecryptUpdate'
            assert moved.value == len(data), 'EVP_DecryptUpdate: %d of %d bytes' % (moved.value, len(data))
            return out.raw
        finally:
            _ssl.EVP_CIPHER_CTX_free(ctx)


def cbc_decrypt(key, iv, data): return _aes(MODE_CBC, key, iv, data)
def ctr_crypt(key, ctr, data): return _aes(MODE_CTR, key, ctr.to_bytes(16, 'big'), data)


def normal_key(key_x, key_y):
    """The AES engine's key scrambler: (((KeyX <<< 2) ^ KeyY) + C) <<< 87."""
    def rol(v, r): return ((v << r) | (v >> (128 - r))) & M128
    x, y = int.from_bytes(key_x, 'big'), int.from_bytes(key_y, 'big')
    return rol((rol(x, 2) ^ y) + SCRAMBLER & M128, 87).to_bytes(16, 'big')


def find_key_files(*dirs):
    """boot9.bin, aes_keys.txt and seeddb.bin found in the given folders, the current one, or an emulator's sysdata
    folder (Azahar's and Citra's, which keep them under these names): {name: path}."""
    home = os.path.expanduser('~')
    where = list(dirs) + [os.getcwd()]
    for app in ('azahar-emu', 'citra-emu', 'lime3ds-emu'):  # Linux
        where.append(os.path.join(os.environ.get('XDG_DATA_HOME', os.path.join(home, '.local', 'share')), app, 'sysdata'))
    for app in ('Azahar', 'Citra', 'Lime3DS'):  # macOS
        where.append(os.path.join(home, 'Library', 'Application Support', app, 'sysdata'))
    found = {}
    for d in where:
        for name in ('boot9.bin', 'aes_keys.txt', 'seeddb.bin'):
            p = os.path.join(d, name)
            if name not in found and os.path.isfile(p):
                found[name] = p
    return found


SEEDDB_HEADER, SEEDDB_ENTRY = 0x10, 0x20  # u32 count + padding; u64 title id, 16-byte seed, padding


class Keys:
    def __init__(self, boot9=None, keyfile=None, seeddb=None, seed=None):
        self.x, self.common, self.seeds = {}, {}, {}
        self.seed = bytes.fromhex(seed) if seed else None
        if self.seed is not None and len(self.seed) != 16:
            sys.exit('--seed: expected 16 bytes (32 hex digits)')
        if seeddb:
            with open(seeddb, 'rb') as f:
                db = f.read()
            count = struct.unpack_from('<I', db, 0)[0]
            if len(db) < SEEDDB_HEADER + count * SEEDDB_ENTRY:
                sys.exit('%s: not a seeddb.bin (%d entries announced, file too short)' % (seeddb, count))
            for i in range(count):
                o = SEEDDB_HEADER + i * SEEDDB_ENTRY
                self.seeds[struct.unpack_from('<Q', db, o)[0]] = db[o + 8:o + 24]
        if boot9:
            with open(boot9, 'rb') as f:
                b9 = f.read()
            if len(b9) not in (0x10000, 0x8000):
                sys.exit('%s: not a boot9.bin (expected 64 KiB, or its 32 KiB protected half)' % boot9)
            o = 0x59D0 if len(b9) == 0x8000 else 0xD9D0  # the KeyX table the boot ROM loads
            self.x[0x2C] = b9[o:o + 16]
            self.x[0x3D] = b9[o + 0x50:o + 0x60]
        if keyfile:
            for line in open(keyfile):
                k, _, v = line.strip().partition('=')
                if not v or k.startswith('#'):
                    continue
                v = bytes.fromhex(v)
                if k.startswith('common') and k[6:].isdigit():
                    self.common[int(k[6:])] = v
                elif k.startswith('slot0x') and k.endswith('KeyX'):
                    self.x[int(k[6:8], 16)] = v

    def key_x(self, slot, what):
        if slot not in self.x:
            src = '--boot9 boot9.bin' if slot in (0x2C, 0x3D) else '--keys aes_keys.txt (slot0x%02XKeyX)' % slot
            sys.exit('%s needs KeyX 0x%02X: pass %s' % (what, slot, src))
        return self.x[slot]

    def title_seed(self, program_id, check):
        """The seed of a title, checked against the NCCH header's hash of it (SHA-256 of seed + program id)."""
        seed = self.seed if self.seed is not None else self.seeds.get(program_id)
        if seed is None:
            sys.exit('NCCH %016X uses seed crypto: pass --seeddb seeddb.bin (or --seed) with its seed'
                     % program_id)
        if hashlib.sha256(seed + struct.pack('<Q', program_id)).digest()[:4] != check:
            sys.exit('the seed given for %016X does not match the one its NCCH was made with' % program_id)
        return seed


class Reader:
    """A file-like view of a source with pread(pos, n)."""
    def __init__(self, size): self.size, self.pos = size, 0
    def seek(self, pos, whence=0): self.pos = pos if whence == 0 else self.pos + pos if whence == 1 else self.size + pos
    def tell(self): return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else max(0, min(n, self.size - self.pos))
        data = self.pread(self.pos, n)
        self.pos += len(data)
        return data


class FileSlice(Reader):
    def __init__(self, f, off, size):
        super().__init__(size)
        self.f, self.off = f, off

    def pread(self, pos, n):
        self.f.seek(self.off + pos)
        return self.f.read(n)


class CbcContent(Reader):
    """A CIA content encrypted with the title key (AES-CBC, IV = the content index)."""
    def __init__(self, src, key, index):
        super().__init__(src.size)
        self.src, self.key, self.iv = src, key, index.to_bytes(2, 'big') + bytes(14)

    def pread(self, pos, n):
        a, b = pos & ~15, min(self.size, (pos + n + 15) & ~15)
        if a >= b:
            return b''
        ct = self.src.pread(a - 16, b - a + 16) if a else self.iv + self.src.pread(0, b)
        return cbc_decrypt(self.key, ct[:16], ct[16:])[pos - a:pos - a + n]


def _sig_size(blob):
    return {0x10000: 0x240, 0x10001: 0x140, 0x10002: 0x80,
            0x10003: 0x240, 0x10004: 0x140, 0x10005: 0x80}[struct.unpack_from('>I', blob)[0]]


def open_cia(f, keys):
    """The CIA's main content (index 0, the program's NCCH), with the CIA layer removed."""
    f.seek(0)
    hdr = f.read(0x20)
    hdr_size, _, _, cert_size, tik_size, tmd_size = struct.unpack_from('<IHHIII', hdr)

    def al(n): return (n + 63) & ~63
    tik_off = al(hdr_size) + al(cert_size)
    tmd_off = tik_off + al(tik_size)
    content_off = tmd_off + al(tmd_size)
    f.seek(tik_off)
    tik = f.read(tik_size)
    tik = tik[_sig_size(tik):]
    f.seek(tmd_off)
    tmd = f.read(tmd_size)
    tmd = tmd[_sig_size(tmd):]
    count = struct.unpack_from('>H', tmd, 0x9E)[0]
    chunks = [struct.unpack_from('>IHHQ', tmd, 0xC4 + 64 * 36 + 0x30 * i) for i in range(count)]
    _, index, ctype, size = chunks[0]  # contents are stored in TMD order: the first one is the program
    if index != 0:
        sys.exit('CIA: the first content is not the program (index %d)' % index)
    src = FileSlice(f, content_off, size)
    if not ctype & 1:
        return src
    title_id = tik[0x9C:0xA4]
    common = tik[0xB1]
    if common not in keys.common:
        sys.exit('the CIA\'s ticket uses common key %d: pass --keys aes_keys.txt (common%d=<KeyY>)' % (common, common))
    kek = normal_key(keys.key_x(0x3D, 'the CIA\'s title key'), keys.common[common])
    title_key = cbc_decrypt(kek, title_id + bytes(8), tik[0x7F:0x8F])
    return CbcContent(src, title_key, index)


class NcchContent(Reader):
    """An encrypted NCCH with its keyslot layer removed; its header says NoCrypto."""
    def __init__(self, src, keys):
        super().__init__(src.size)
        self.src = src
        nh = bytearray(src.pread(0, 0x200))
        flags = nh[0x188:0x190]
        if flags[7] & 1:
            key0 = key1 = bytes(16)  # fixed key: zero for applications
        else:
            key_y = bytes(nh[0:16])  # the first 16 bytes of the header's signature
            key0 = normal_key(keys.key_x(0x2C, 'NCCH'), key_y)
            slot = {0x00: 0x2C, 0x01: 0x25, 0x0A: 0x18, 0x0B: 0x1B}.get(flags[3])
            if slot is None:
                sys.exit('NCCH crypto method 0x%02X is unknown' % flags[3])
            if flags[7] & 0x20:  # seed crypto: the secondary key's KeyY also hashes in the title's seed
                seed = keys.title_seed(struct.unpack_from('<Q', nh, 0x118)[0], bytes(nh[0x114:0x118]))
                key_y = hashlib.sha256(key_y + seed).digest()[:16]
            key1 = normal_key(keys.key_x(slot, 'NCCH crypto method 0x%02X' % flags[3]), key_y)
        version = struct.unpack_from('<H', nh, 0x112)[0]
        exefs_off, exefs_size = struct.unpack_from('<II', nh, 0x1A0)
        romfs_off, romfs_size = struct.unpack_from('<II', nh, 0x1B0)
        exefs_off, exefs_size, romfs_off, romfs_size = (v * 0x200 for v in (exefs_off, exefs_size, romfs_off, romfs_size))

        def ctr(kind, off):
            if version == 1:  # partition id, then the region's byte offset
                return int.from_bytes(bytes(nh[0x108:0x110]) + bytes(4) + off.to_bytes(4, 'big'), 'big')
            return int.from_bytes(bytes(nh[0x108:0x110])[::-1] + bytes([kind]) + bytes(7), 'big')

        # (start, end, key, counter at the region's base, base); later entries win where they overlap
        self.regions = []
        if struct.unpack_from('<I', nh, 0x180)[0]:
            self.regions.append((0x200, 0xA00, key0, ctr(1, 0x200), 0x200))
        if exefs_size:
            base = ctr(2, exefs_off)
            self.regions.append((exefs_off, exefs_off + exefs_size, key0, base, exefs_off))
            exefs_hdr = self.pread(exefs_off, 0x200)
            for i in range(10):
                name = exefs_hdr[i * 16:i * 16 + 8].rstrip(b'\0')
                off, size = struct.unpack_from('<II', exefs_hdr, i * 16 + 8)
                if name and name not in (b'icon', b'banner') and key1 != key0:
                    start = exefs_off + 0x200 + off
                    self.regions.append((start, start + ((size + 15) & ~15), key1, base, exefs_off))
        if romfs_size:
            self.regions.append((romfs_off, romfs_off + romfs_size, key1, ctr(3, romfs_off), romfs_off))
        nh[0x18B] = 0                       # crypto method
        nh[0x18F] = (nh[0x18F] & ~0x21) | 4  # NoCrypto, no fixed key or seed
        self.header = bytes(nh)

    def pread(self, pos, n):
        n = max(0, min(n, self.size - pos))
        out = bytearray(self.src.pread(pos, n))
        end = pos + n
        for start, stop, key, ctr, base in self.regions:
            a, b = max(pos, start), min(end, stop)
            if a >= b:
                continue
            a16 = base + ((a - base) & ~15)
            data = ctr_crypt(key, ctr + (a16 - base) // 16, self.src.pread(a16, b - a16))
            out[a - pos:b - pos] = data[a - a16:]
        if pos < 0x200 and hasattr(self, 'header'):
            out[:min(end, 0x200) - pos] = self.header[pos:min(end, 0x200)]
        return bytes(out)
