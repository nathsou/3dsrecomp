"""Function discovery and C emission for a 3DS ARM11 program.

Each guest function becomes `void f_XXXXXXXX(CPU *c, u32 entry)`. entry 0 starts at the function's address;
entry k > 0 resumes at the k-th return site (the instruction after a call) so the dispatcher can continue a guest
call chain whose host frames were unwound. Guest registers live in C locals inside a function and in the CPU struct
across calls. Direct calls are C calls followed by a check that the callee returned where expected; tail branches
and indirect jumps are `musttail` calls.
"""
import os
import re
import struct
import sys
from collections import defaultdict

from . import arm
from . import thumb
from .ins import B, CALL, ICALL, JMP, SVC, UDF, OP, COND_C
from . import flags


def fname(key):
    return ('t_%08x' if key & 1 else 'f_%08x') % (key & ~1)


class Analyzer:
    def __init__(self, image, log=print):
        self.img = image
        self.ctx = arm.Ctx(image.ro_read)
        self.cache = {}
        self.log = log
        self.funcs = {}          # key (addr | thumb) -> True
        self.rejected = set()
        self.notes = defaultdict(int)
        self.rel_tables = set()
        self.pc_rel_ptrs = set()
        self.extra = set()

    def decode(self, addr, th):
        k = addr | th
        i = self.cache.get(k)
        if i is None:
            if not self.img.in_text(addr):
                return None
            if th:
                hw = self.img.read16(addr)
                hw2 = self.img.read16(addr + 2)
                if hw is None:
                    return None
                i = thumb.decode(addr, hw, hw2 if hw2 is not None else 0, self.ctx)
                self.fixup_thumb(i, hw)
            else:
                if addr & 3:
                    return None
                w = self.img.read32(addr)
                if w is None:
                    return None
                i = arm.decode(addr, w, self.ctx)
                self.fixup_arm(i)
            self.cache[k] = i
        return i

    def fixup_thumb(self, i, hw):
        # `pop {..., rX}` then `bx rX`: Thumb's return through the popped lr (Thumb-1 code cannot pop pc and switch to
        # ARM). Returning is right for any `bx` (a caller whose return site does not match returns too, until the
        # dispatcher continues at pc); here it is the common case, and much cheaper than dispatching.
        if (hw & 0xFF87) == 0x4700 and i.kind == JMP and not i.ret:
            rm = (hw >> 3) & 15
            a = i.addr - 2
            prev = self.img.read16(a)
            while prev is not None and (prev & 0xFF80) == 0xB000:  # add sp, #n (a varargs function's pushed arguments)
                a -= 2
                prev = self.img.read16(a)
            if rm < 8 and prev is not None and (prev & 0xFF00) == 0xBC00 and (prev >> rm) & 1:
                i.ret = True
        # `ldr rX, [pc, #n]` then `add rX, pc`: a pc-relative pointer (a function, or the base of a table of
        # self-relative offsets such as an init array when the two are adjacent)
        if (hw & 0xFF78) == 0x4478 and not hw & 0x80:
            rx = hw & 7
            for back in range(1, 7):
                pa = i.addr - 2 * back
                p = self.img.read16(pa)
                if p is None:
                    break
                if (p & 0xFF00) == (0x4800 | (rx << 8)):
                    lit = self.img.ro_read(((pa + 4) & ~3) + (p & 0xFF) * 4)
                    if lit is not None:
                        v = (i.addr + 4 + lit) & 0xFFFFFFFF
                        self.pc_rel_ptrs.add(v)
                        if back == 1:
                            self.rel_tables.add(v)
                    break
                if (p & 7) == rx and (p >> 11) not in (0x1D, 0x1E, 0x1F) and (p & 0xF800) != 0x6000 and (p & 0xF800) != 0x9000:
                    break  # (rough) another 16-bit instruction writing rX: give up

    def fixup_arm(self, i):
        img = self.img
        if i.kind in (CALL, B) and not img.in_text(i.target & ~1):
            i.kind = UDF
            i.note = 'branch outside text'
            return
        if i.kind == CALL and not i.target & 1:
            # RVCT helpers taking an inline argument word after the BL: `ldr rX, [lr], #4` on entry
            f = img.read32(i.target)
            if f is not None and (f & 0x0FFF0FFF) == 0x049E0004 and (f >> 28) == 14:
                i.link = i.addr + 8
                i.size = 8
                i.note += ' (inline arg)'
        if i.kind == JMP and not i.ret:
            # `mov lr, pc` / `add lr, pc, #n` then a write to pc: an indirect call
            link = None
            for back in (1, 2):
                pa = i.addr - 4 * back
                p = img.read32(pa)
                if p is None:
                    break
                if p == 0xE1A0E00F:
                    link = pa + 8
                    break
                if (p & 0xFFFFF000) == 0xE28FE000:
                    link = pa + 8 + arm.ror(p & 0xFF, ((p >> 8) & 15) * 2)
                    break
                if (p >> 12) & 15 in (14, 15) or (p >> 25) & 7 in (4, 5):
                    break  # writes lr/pc or is a branch / ldm: stop looking
            if link is not None and link > i.addr:
                i.kind = ICALL
                i.link = link
                i.size = link - i.addr
                i.note += ' (call via lr)'
        if i.kind == OP and (i.word & 0x0FEF0000) == 0x008F0000 and (i.word & 0xFF0) == 0:
            # `add rX, pc, rY` with rY loaded from a literal shortly before: a pc-relative code/data pointer
            ry = i.word & 15
            for back in (1, 2, 3, 4):
                p = img.read32(i.addr - 4 * back)
                if p is None:
                    break
                if (p & 0x0FFFF000) == (0x059F0000 | (ry << 12)):
                    lit = img.ro_read(i.addr - 4 * back + 8 + (p & 0xFFF))
                    if lit is not None:
                        self.pc_rel_ptrs.add((i.addr + 8 + lit) & 0xFFFFFFFF)
                    break
                if (p >> 12) & 15 == ry:
                    break
        if i.kind == OP and (i.word & 0x0FFF0FF0) == 0x008F0000 and (i.word >> 28) == 14:
            # `add rX, pc, rX` after `ldr rX, [pc, #n]`: a pc-relative table base (RVCT init arrays)
            rx = (i.word >> 12) & 15
            if (i.word >> 16) & 15 == 15 and i.word & 15 == rx:
                p = img.read32(i.addr - 4)
                if p is not None and (p & 0x0FFFF000) == (0x059F0000 | (rx << 12)):
                    lit = img.ro_read(i.addr - 4 + 8 + (p & 0xFFF))
                    if lit is not None:
                        self.rel_tables.add((i.addr + 8 + lit) & 0xFFFFFFFF)

    # ------------------------------------------------------------ jump tables
    def jump_table(self, i, th):
        """Targets of a recognised switch (list of keys), or None."""
        if th:
            return self.thumb_table(i)
        w = i.word
        a = i.addr
        kind = None
        if (w & 0x0FFFFFF0) == 0x079FF100:
            kind = 'ldr'
        elif (w & 0x0FFFFFF0) == 0x008FF100:
            kind = 'add'
        if kind is None:
            return None
        rm = w & 15
        if kind == 'add':
            t = self.byte_table(a, rm)
            if t is not None:
                return t
        n = None
        for back in range(1, 6):
            p = self.img.read32(a - 4 * back)
            if p is None:
                break
            if (p & 0x0FFFF000) == (0x03500000 | (rm << 16)):
                imm = arm.ror(p & 0xFF, ((p >> 8) & 15) * 2)
                if i.cond == 3:
                    n = imm
                elif i.cond == 9:
                    n = imm + 1
                break
        targets = []
        lim = n if n is not None else 1024
        for k in range(lim):
            if kind == 'ldr':
                t = self.img.read32(a + 8 + 4 * k)
                if t is None or not self.img.in_text(t & ~1):
                    if n is None:
                        break
                    return None
                targets.append(t if t & 1 else t & ~3)
            else:
                t = a + 8 + 4 * k
                targets.append(t)
                if n is None:
                    # unbounded: a run of unconditional branches, the last case may be inline code
                    e = self.img.read32(t)
                    if e is None or (e >> 24) != 0xEA:
                        break
        return targets if targets else None

    def thumb_table(self, i):
        """RVCT's Thumb switch on a table of byte (or halfword) offsets right after the jump:
            cmp rI, #n ; bhs default ; movs rX, rI ; add rX, pc ; ldrb rX, [rX, #k] ; adds rX, rX, rX ; add pc, rX
        (the halfword form indexes with `lsls rX, rI, #1` and loads with ldrh)."""
        img = self.img
        a = i.addr
        hw = img.read16(a)
        if hw is None or (hw & 0xFF87) != 0x4487:
            return None
        rx = (hw >> 3) & 15
        if rx > 7:
            return None
        h = [img.read16(a - 2 * k) for k in range(1, 5)]
        if None in h:
            return None
        if h[0] not in (0x1800 | rx << 6 | rx << 3 | rx, 0x0040 | rx << 3 | rx):  # adds rX, rX, rX / lsls rX, rX, #1
            return None
        if (h[1] & 0xF83F) == (0x7800 | rx << 3 | rx):
            size, k = 1, (h[1] >> 6) & 31
        elif (h[1] & 0xF83F) == (0x8800 | rx << 3 | rx):
            size, k = 2, ((h[1] >> 6) & 31) * 2
        else:
            return None
        if h[2] != (0x4478 | rx):  # add rX, pc
            return None
        if size == 1 and (h[3] & 0xFFC7) == rx:  # movs rX, rI
            ri = (h[3] >> 3) & 7
        elif size == 2 and (h[3] & 0xFFC7) == (0x0040 | rx):  # lsls rX, rI, #1
            ri = (h[3] >> 3) & 7
        else:
            return None
        n = None
        for back in range(5, 14):
            p = img.read16(a - 2 * back)
            if p is None:
                break
            if (p & 0xFF00) == (0x2800 | ri << 8):  # cmp rI, #n
                q = img.read16(a - 2 * back + 2)
                # the bhs/bcs to the default case (possibly a few instructions later)
                for f in range(a - 2 * back + 2, a - 8, 2):
                    q = img.read16(f)
                    if q is not None and (q & 0xFF00) == 0xD200:
                        n = p & 0xFF
                        break
                break
        if not n:
            return None
        tbase = a - 2 + k  # rX = idx + (address of `add rX, pc`) + 4
        targets = []
        for idx in range(n):
            if size == 1:
                s = img.seg_of(tbase + idx)
                if s is None:
                    return None
                v = s.data[tbase + idx - s.addr]
            else:
                v = img.read16(tbase + 2 * idx)
                if v is None:
                    return None
            t = a + 4 + 2 * v
            if not img.in_text(t):
                return None
            targets.append(t | 1)
        return targets

    @staticmethod
    def is_trap(i):
        """The permanently undefined instruction compilers emit for __builtin_trap (GCC: `udf #0`, 0xE7F000F0):
        a deliberate stop inside real code, not a sign that the bytes are data."""
        if i.thumb:
            return (i.word & 0xFF00) == 0xDE00
        return (i.word & 0xFFF000F0) == 0xE7F000F0

    def writes(self, a, r):
        """Does the ARM instruction at a (possibly) write register r?"""
        i = self.decode(a, 0)
        if i is None or i.kind not in (OP,):
            return i is None or (i.kind in (CALL, ICALL, SVC) and r in (0, 1, 2, 3, 12))
        pat = re.compile(r'\br%d\s*=(?!=)' % r)
        return any(pat.search(line) for line in i.code)

    def byte_table(self, a, rm):
        """GCC's compressed switch: `ldr[s]b rm, [rb, rx]` then `add pc, pc, rm, lsl #2`, the byte table at a
        literal-pool address in rb, its size from an earlier `cmp rx, #n` + `bhi`/`bcs` to the default case."""
        img = self.img
        p = img.read32(a - 4)
        if p is None or p >> 28 != 14 or (p >> 12) & 15 != rm:
            return None
        if (p & 0x0FF00FF0) == 0x07D00000:
            signed = False
        elif (p & 0x0FF00FF0) == 0x019000D0:
            signed = True
        else:
            return None
        rb, rx = (p >> 16) & 15, p & 15
        n = base = None
        for back in range(2, 40):
            q = a - 4 * back
            w = img.read32(q)
            if w is None:
                return None
            if n is None and (w & 0x0FFFF000) == (0x03500000 | (rx << 16)) and w >> 28 == 14:
                imm = arm.ror(w & 0xFF, ((w >> 8) & 15) * 2)
                for f in range(q + 4, a - 4, 4):
                    b = self.decode(f, 0)
                    if b is not None and b.kind == B and b.cond in (2, 8):
                        n = imm + 1 if b.cond == 8 else imm
                        break
                if n is None:
                    return None
            elif n is None and self.writes(q, rx):
                return None
            if base is None and self.writes(q, rb):
                if (w & 0x0FFFF000) != (0x059F0000 | (rb << 12)) or w >> 28 != 14:
                    return None
                base = img.ro_read(q + 8 + (w & 0xFFF))
                if base is None:
                    return None
            if n is not None and base is not None:
                break
        if n is None or base is None or not 0 < n <= 1024:
            return None
        targets = []
        for k in range(n):
            s = img.seg_of(base + k)
            if s is None:
                return None
            v = s.data[base + k - s.addr]
            if signed and v >= 0x80:
                v -= 0x100
            t = a + 8 + 4 * v
            if not img.in_text(t):
                return None
            targets.append(t)
        return targets

    # ------------------------------------------------------------ exploration
    def explore(self, entry, limit=None):
        """Instructions reachable from entry without crossing calls. Returns (insts dict, ok)."""
        th = entry & 1
        start = entry & ~1
        insts = {}
        work = [start]
        ok = True
        while work:
            a = work.pop()
            if a in insts:
                continue
            i = self.decode(a, th)
            if i is None:
                ok = False
                continue
            insts[a] = i
            if limit and len(insts) > limit:
                return insts, ok
            if i.kind == UDF and not self.is_trap(i):
                ok = False
            if i.kind == B:
                tk = i.target
                if (tk | th) not in self.funcs or tk == start:
                    work.append(tk)
            if i.kind == JMP and i.jt is None:
                i.jt = self.jump_table(i, th)
            if i.kind == JMP and i.jt:
                for t in i.jt:
                    if (t & 1) == th and t not in insts:
                        work.append(t & ~1)
            if i.falls_through:
                nxt = a + i.size
                if nxt != start and (nxt | th) in self.funcs:
                    continue  # tail call into the next function
                work.append(nxt)
        return insts, ok

    def gap_candidates(self):
        """Code addresses suggested by the parts of .text no function covers: pointers into .text stored there
        (tables), and function prologues right after a function's last instruction."""
        img = self.img
        cov = set()
        lits = set()
        for k in self.funcs:
            insts, _ = self.explore(k)
            if k & 1:
                continue
            cov |= set(insts)
            for i in insts.values():
                if i.reads_pc_lit is not None:
                    lits.add(i.reads_pc_lit & ~3)
        out = set()
        for text in img.texts:
            self.gap_scan(text, cov, lits, out)
        return out

    def gap_scan(self, text, cov, lits, out):
        img = self.img
        prev_end = False
        for off in range(0, text.size, 4):
            a = text.addr + off
            if a in cov:
                i = self.cache.get(a)
                prev_end = i is not None and not i.falls_through and i.kind != 'b'
                continue
            w = img.read32(a)
            if a in lits:
                prev_end = True
                continue
            if w is not None and img.in_text(w) and not w & 3:
                out.add(w)
            # push {..., lr} / stmdb sp!, {..., lr} after the end of a function
            if prev_end and w is not None and (w & 0xFFFF4000) == 0xE92D4000:
                out.add(a)
            prev_end = False

    def saves_lr(self, key):
        """Whether the code at key starts by saving lr, after at most a few register moves: push {..., lr} (ARM stmdb
        sp!, {..., lr} or str lr, [sp, #-4]!, Thumb push {..., lr}). Compilers save lr only at a function's entry."""
        th = key & 1
        a = key & ~1
        for _ in range(4):
            i = self.decode(a, th)
            if i is None or i.cond != 14:
                return False
            if th:
                return (i.word & 0xFF00) == 0xB500
            w = i.word
            if (w & 0xFFFF4000) == 0xE92D4000 or w == 0xE52DE004:
                return True
            # mov rd, rm (no shift, no flags) with rd not sp, lr or pc
            if (w & 0xFFF00FF0) != 0xE1A00000 or ((w >> 12) & 15) >= 13:
                return False
            a += 4
        return False

    def plausible_entry(self, key):
        th = key & 1
        a = key & ~1
        if not self.img.in_text(a) or (not th and a & 3):
            return False
        i = self.decode(a, th)
        if i is None or i.kind == UDF or i.cond != 14:
            return False
        insts, ok = self.explore(key, limit=20000)
        return ok

    def discover(self):
        img = self.img
        queue = [img.entry]
        self.funcs[img.entry] = True
        for e in img.extra_entries:  # CRO OnLoad / OnExit / OnUnresolved
            if self.decode(e & ~1, e & 1) is not None:
                self.funcs[e] = True
                queue.append(e)
        candidates = set()
        # pointers in data/rodata into text
        for s in img.segments:
            if s in img.texts:
                continue
            d = s.data
            for off in range(0, len(d) - 3, 4):
                v = struct.unpack_from('<I', d, off)[0]
                if img.in_text(v & ~1):
                    candidates.add(v)
        done = set()
        gap_done = False
        lit_candidates = set()
        rounds = 0
        while True:
            rounds += 1
            while queue:
                f = queue.pop()
                if f in done:
                    continue
                done.add(f)
                insts, _ = self.explore(f)
                for i in insts.values():
                    if i.kind == B:
                        # a branch to code that saves lr is a tail call to a function of its own (not code of this
                        # one): otherwise each function branching there would get a copy of it
                        t = i.target | (f & 1)
                        if t not in self.funcs and i.target != f & ~1 and self.saves_lr(t):
                            self.funcs[t] = True
                            queue.append(t)
                    if i.kind == CALL:
                        t = i.target
                        if t not in self.funcs:
                            if self.decode(t & ~1, t & 1) is None:
                                continue
                            self.funcs[t] = True
                            queue.append(t)
                    if i.reads_pc_lit is not None:
                        v = img.read32(i.reads_pc_lit)
                        if v is not None and img.in_text(v & ~1):
                            lit_candidates.add(v)
                    if i.kind == OP and not i.thumb and (i.word & 0x0FFF0000) == 0x028F0000:  # ADR (add rX, pc, #imm)
                        v = (i.addr + 8 + arm.ror(i.word & 0xFF, ((i.word >> 8) & 15) * 2)) & 0xFFFFFFFF
                        if img.in_text(v & ~1):
                            lit_candidates.add(v)
            for v in self.pc_rel_ptrs:
                if img.in_text(v & ~1):
                    lit_candidates.add(v)
            for t in sorted(self.rel_tables):
                k = 0
                while True:
                    w = img.read32(t + 4 * k)
                    if w is None:
                        break
                    f = (t + 4 * k + w) & 0xFFFFFFFF
                    if not img.in_text(f & ~1) or (f & 3 and not f & 1):
                        break
                    lit_candidates.add(f)
                    k += 1
            added = 0
            for v in sorted(self.extra):
                if v not in self.funcs:
                    self.funcs[v] = True
                    queue.append(v)
                    added += 1
            for v in sorted(candidates | lit_candidates):
                if v in self.funcs or v in self.rejected:
                    continue
                if self.plausible_entry(v):
                    self.funcs[v] = True
                    queue.append(v)
                    added += 1
                else:
                    self.rejected.add(v)
            self.log('discovery round %d: %d functions (+%d from pointers)' % (rounds, len(self.funcs), added))
            if not queue and not gap_done:
                gap_done = True
                g = self.gap_candidates()
                for v in sorted(g):
                    if v in self.funcs or v in self.rejected:
                        continue
                    if self.plausible_entry(v):
                        self.funcs[v] = True
                        queue.append(v)
                    else:
                        self.rejected.add(v)
                self.log('gap filling: +%d functions' % len(queue))
            if not queue:
                break
        # functions found later may cut earlier explorations: re-explore everything with final boundaries
        return sorted(self.funcs)


class Emitter:
    def __init__(self, an, outdir, per_file=400):
        self.an = an
        self.outdir = outdir
        self.per_file = per_file
        self.entries = []  # (key, fname, entry index)
        self.stats = defaultdict(int)
        self.undefined = []
        self.flag_opt = True
        self.flag_returning = set()
        self.flag_reading_entries = set()
        self.override = {}

    def emit_function(self, key):
        an = self.an
        th = key & 1
        start = key & ~1
        insts, _ = an.explore(key)
        addrs = sorted(insts)
        self.override = {}
        if self.flag_opt:
            lo = flags.liveness(insts, th, start, an.funcs, key in self.flag_returning, self.flag_reading_entries)
            for a, i in insts.items():
                c = flags.strip_dead(i, lo[a])
                if c is not None:
                    self.override[a] = c
                    self.stats['flag_strips'] += 1
        labels = set([start])
        resume = []
        for a in addrs:
            i = insts[a]
            if i.kind == B and ((i.target | th) not in an.funcs or i.target == start):
                labels.add(i.target)
            if i.kind == JMP and i.jt:
                for t in i.jt:
                    if (t & 1) == th:
                        labels.add(t & ~1)
            if i.kind in (CALL, ICALL, SVC) and i.falls_through:
                nxt = a + i.size
                if nxt in insts:
                    if i.kind != SVC:
                        resume.append(nxt)
            if i.falls_through:
                labels.add(a + i.size) if False else None
        resume = sorted(set(resume))
        labels |= set(resume)
        regs = set()
        vregs = set()
        uses_flags = False
        uses_q = uses_ge = uses_fpsr = False
        for i in insts.values():
            regs |= i.regs()
            vregs |= i.vregs()
            txt = ' '.join(i.code)
            if i.cond != 14 or any(f in txt for f in ('nf', 'zf', 'cf', 'vf')):
                uses_flags = True
            if 'qf' in txt:
                uses_q = True
            if 'ge' in txt.replace('ge(', ''):
                uses_ge = True
            if 'fpsr' in txt:
                uses_fpsr = True
        regs.discard(15)
        regs = sorted(regs)
        vregs = sorted(vregs)
        fn = fname(key)
        out = []
        so = ['c->r[%d] = r%d;' % (r, r) for r in regs]
        si = ['r%d = c->r[%d];' % (r, r) for r in regs]
        if uses_flags:
            so.append('c->nf = nf; c->zf = zf; c->cf = cf; c->vf = vf;')
            si.append('nf = c->nf; zf = c->zf; cf = c->cf; vf = c->vf;')
        if uses_q:
            so.append('c->qf = qf;')
            si.append('qf = c->qf;')
        if uses_ge:
            so.append('c->ge = ge;')
            si.append('ge = c->ge;')
        if uses_fpsr:
            so.append('c->fpscr = fpsr;')
            si.append('fpsr = c->fpscr;')
        for v in vregs:
            so.append('c->s[%d] = S%d;' % (v, v))
            si.append('S%d = c->s[%d];' % (v, v))
        out.append('void %s(CPU *c, u32 entry) {' % fn)
        decl = []
        if regs:
            decl.append('u32 ' + ', '.join('r%d = c->r[%d]' % (r, r) for r in regs) + ';')
        if uses_flags:
            decl.append('u32 nf = c->nf, zf = c->zf, cf = c->cf, vf = c->vf;')
        if uses_q:
            decl.append('u32 qf = c->qf;')
        if uses_ge:
            decl.append('u32 ge = c->ge;')
        if uses_fpsr:
            decl.append('u32 fpsr = c->fpscr;')
        if vregs:
            decl.append('float ' + ', '.join('S%d = c->s[%d]' % (v, v) for v in vregs) + ';')
        decl.append('u32 tpc = 0; u8 *const M = MEMBASE; (void)M; (void)tpc;')
        out += ['  ' + d for d in decl]
        out.append('#define SO() do { %s } while (0)' % ' '.join(so))
        out.append('#define SI() do { %s } while (0)' % ' '.join(si))
        if resume:
            out.append('  switch (entry) {')
            for k, a in enumerate(resume, 1):
                out.append('  case %d: goto L_%x;' % (k, a))
                self.entries.append(((a | th), fn, k))
            out.append('  default: break;\n  }')
        self.entries.append((key, fn, 0))
        if addrs and addrs[0] != start:
            out.append('  goto L_%x;' % start)
        prev_falls = None
        for idx, a in enumerate(addrs):
            i = insts[a]
            if prev_falls is not None and prev_falls != a:
                out.append('  goto L_%x;' % prev_falls)
            if a in labels:
                out.append('L_%x:' % a)
            out.append(self.emit_ins(i, key, insts))
            prev_falls = None
            if i.falls_through:
                nxt = a + i.size
                if nxt in insts:
                    prev_falls = nxt
                else:
                    # a tail call into the following function
                    out.append('  SO(); __attribute__((musttail)) return %s(c, 0);' % fname(nxt | th))
                    self.stats['fallthrough_tail'] += 1
        if prev_falls is not None:
            out.append('  goto L_%x;' % prev_falls)
        out.append('#undef SO\n#undef SI\n}')
        self.stats['functions'] += 1
        self.stats['instructions'] += len(insts)
        return '\n'.join(out)

    def emit_ins(self, i, key, insts):
        th = key & 1
        an = self.an
        body = []
        body += self.override.get(i.addr, i.code)
        if i.kind == OP:
            pass
        elif i.kind == B:
            t = i.target
            if (t | th) in an.funcs and t != (key & ~1):
                body.append('SO(); __attribute__((musttail)) return %s(c, 0);' % fname(t | th))
            elif t <= i.addr:
                # a loop: give way if an interrupt readied a higher-priority thread (a busy wait for that
                # thread would otherwise never end; the 3DS kernel preempts)
                body.append('if (__builtin_expect(rt_preempt_flag, 0)) { SO(); c->pc = 0x%xu; rt_preempt(c); SI(); } '
                            'goto L_%x;' % (t | th, t))
                self.stats['loop_checks'] += 1
            else:
                body.append('goto L_%x;' % t)
        elif i.kind == CALL and i.target not in an.funcs:
            # a call to something discovery rejected (outside the code, or code reached only from data that
            # was decoded as instructions): go through the dispatcher, which reports it if it is ever reached
            body.append('SO(); c->pc = 0x%xu; rt_dispatch(c, 0); SI(); if (__builtin_expect(c->pc != 0x%xu, 0)) return;' % (
                i.target, i.link | th))
        elif i.kind == CALL:
            body.append('SO(); %s(c, 0); SI(); if (__builtin_expect(c->pc != 0x%xu, 0)) return;' % (
                fname(i.target), i.link | th))
        elif i.kind == ICALL:
            body.append('SO(); c->pc = tpc; rt_dispatch(c, 0); SI(); if (__builtin_expect(c->pc != 0x%xu, 0)) return;' % (
                i.link | th))
        elif i.kind == JMP:
            if i.jt:
                cases = ' '.join('case 0x%xu: goto L_%x;' % (t, t & ~1) for t in sorted(set(i.jt)) if (t & 1) == th)
                body.append('switch (tpc) { %s default: break; }' % cases)
                self.stats['jump_tables'] += 1
            if i.ret:
                body.append('c->pc = tpc; SO(); return;')
            else:
                body.append('c->pc = tpc; SO(); __attribute__((musttail)) return rt_dispatch(c, 0);')
                self.stats['indirect_jumps'] += 1
        elif i.kind == SVC:
            body.append('SO(); c->pc = 0x%xu; rt_svc(c, 0x%x); SI();' % (i.addr + i.size, i.imm))
        elif i.kind == UDF:
            body.append('SO(); rt_undefined(c, 0x%xu, 0x%xu); SI();' % (i.addr | th, i.word))
            self.undefined.append((i.addr, i.word, i.note))
        s = ' '.join(body)
        cc = i.cond_c
        note = i.note.replace('*/', '')
        if cc:
            return '  if (%s) { %s } /* %x: %s */' % (cc, s, i.addr, note)
        return '  %s /* %x: %s */' % (s, i.addr, note)

    def find_flag_returning(self, keys):
        an = self.an
        fr = set()
        tails = {}
        for k in keys:
            th = k & 1
            insts, _ = an.explore(k)
            for i in insts.values():
                if i.kind == CALL:
                    seq = []
                    a = i.link
                    for _ in range(4):
                        n = an.decode(a, th)
                        seq.append(n)
                        if n is None:
                            break
                        a += n.size
                    if flags.reads_flags_first(seq):
                        fr.add(i.target)
                if i.kind == B and (i.target | th) in an.funcs and i.target != (k & ~1):
                    tails.setdefault(k, set()).add(i.target | th)
                if i.falls_through and (i.addr + i.size) not in insts:
                    tails.setdefault(k, set()).add((i.addr + i.size) | th)
        changed = True
        while changed:
            changed = False
            for f in list(fr):
                for t in tails.get(f, ()):
                    if t not in fr:
                        fr.add(t)
                        changed = True
        self.flag_returning = fr
        # entries that read flags first are mid-function code split off as a function: jumps into them keep flags
        fre = set()
        for k in an.funcs:
            seq, a = [], k & ~1
            for _ in range(4):
                n = an.decode(a, k & 1)
                seq.append(n)
                if n is None:
                    break
                a += n.size
            if flags.reads_flags_first(seq):
                fre.add(k)
        self.flag_reading_entries = fre
        print('flag-returning functions: %d, entries reading flags: %d' % (len(fr), len(fre)))

    def run(self, keys):
        if self.flag_opt:
            self.find_flag_returning(keys)
        os.makedirs(self.outdir, exist_ok=True)
        for f in os.listdir(self.outdir):
            if f.startswith('code_') and f.endswith('.c'):
                os.remove(os.path.join(self.outdir, f))
        files = []
        chunk = []
        n = 0
        def flush():
            nonlocal chunk, n
            if not chunk:
                return
            path = os.path.join(self.outdir, 'code_%03d.c' % n)
            with open(path, 'w') as f:
                f.write('#include "guest.h"\n\n')
                f.write('\n\n'.join(chunk))
                f.write('\n')
            files.append(path)
            n += 1
            chunk = []
        size = 0
        for k in keys:
            src = self.emit_function(k)
            chunk.append(src)
            size += len(src)
            if size > 1_500_000:
                flush()
                size = 0
        flush()
        with open(os.path.join(self.outdir, 'guest.h'), 'w') as f:
            f.write('#pragma once\n#include "cpu.h"\n\n')
            for k in keys:
                f.write('void %s(CPU *c, u32 entry);\n' % fname(k))
        ents = sorted(set(self.entries))  # a resume site shared by several functions: the same pick every run
        seen = set()
        with open(os.path.join(self.outdir, 'table.c'), 'w') as f:
            f.write('#include "guest.h"\n\nconst GuestEntry guest_entries[] = {\n')
            cnt = 0
            for addr, fn, k in ents:
                if addr in seen:
                    continue
                seen.add(addr)
                f.write('  {0x%xu, %s, %d},\n' % (addr, fn, k))
                cnt += 1
            f.write('};\nconst u32 guest_entry_count = %d;\n' % cnt)
            cros = self.an.img.cros
            if cros:
                # the slots the runtime's ldr:ro moves each CRO module's code to (tools/recomp/cro.py)
                f.write('\nconst GuestCro guest_cros[] = {\n')
                for m in cros.mods:
                    f.write('  {"%s", 0x%xu, 0x%xu},\n' % (m.name, m.base, m.slot_size()))
                f.write('};\nconst u32 guest_cro_count = %d;\n' % len(cros.mods))
        files.append(os.path.join(self.outdir, 'table.c'))
        return files


def main():
    import argparse
    from .image import Image
    ap = argparse.ArgumentParser()
    ap.add_argument('gamedir')
    ap.add_argument('outdir')
    ap.add_argument('--extra', action='append', default=[], help='file listing extra function addresses')
    ap.add_argument('--lazy', help='only emit the function at this address (and new functions it calls)')
    ap.add_argument('--lazy-out', help='C file for --lazy')
    args = ap.parse_args()
    if args.lazy:
        return lazy(args)
    img = Image(args.gamedir)
    an = Analyzer(img)
    for path in args.extra:
        if os.path.exists(path):
            for line in open(path):
                line = line.split('#')[0].strip()
                if line:
                    an.extra.add(int(line, 16))
    keys = an.discover()
    em = Emitter(an, args.outdir)
    files = em.run(keys)
    with open(os.path.join(args.outdir, 'funcs.txt'), 'w') as f:
        for k in keys:
            f.write('%x\n' % k)
    print('functions: %d, entries: %d, files: %d' % (len(keys), len(em.entries), len(files)))
    for k, v in sorted(em.stats.items()):
        print('  %s: %d' % (k, v))
    if em.undefined:
        from collections import Counter
        cnt = Counter(n for _, _, n in em.undefined)
        print('undefined instructions in reachable code: %d' % len(em.undefined))
        for k, v in cnt.most_common(30):
            print('   %5d %s' % (v, k))


def lazy(args):
    """Emits one C file with a function the full build lacks, for the runtime to load as a dylib."""
    from .image import Image
    img = Image(args.gamedir)
    an = Analyzer(img, log=lambda *a: None)
    known = set()
    for line in open(os.path.join(args.outdir, 'funcs.txt')):
        known.add(int(line, 16))
    for k in known:
        an.funcs[k] = True
    root = int(args.lazy, 16)
    new = []
    queue = [root]
    an.funcs[root] = True
    while queue:
        f = queue.pop()
        new.append(f)
        insts, _ = an.explore(f)
        for i in insts.values():
            if i.kind == CALL and i.target not in an.funcs and an.decode(i.target & ~1, i.target & 1):
                an.funcs[i.target] = True
                queue.append(i.target)
    em = Emitter(an, args.outdir)
    em.flag_opt = False
    srcs = [em.emit_function(k) for k in new]
    with open(args.lazy_out, 'w') as f:
        f.write('#include "guest.h"\n\n')
        for k in new:
            f.write('void %s(CPU *c, u32 entry);\n' % fname(k))
        f.write('\n'.join(srcs))
        ents = sorted(set(em.entries), key=lambda e: e[0])
        f.write('\nconst GuestEntry guest_lazy_entries[] = {\n')
        for addr, fn, k in ents:
            f.write('  {0x%xu, %s, %d},\n' % (addr, fn, k))
        f.write('};\nconst u32 guest_lazy_count = %d;\n' % len(ents))
    print('lazy: %d functions from %x' % (len(new), root))


if __name__ == '__main__':
    main()
