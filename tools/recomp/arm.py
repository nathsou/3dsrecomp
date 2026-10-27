"""ARMv6K (ARM11 MPCore) ARM-state decoder that lowers each instruction to C.

Guest registers are C locals r0..r14, flags nf/zf/cf/vf (0/1), qf (sticky saturation), ge (4 GE bits),
VFP single registers s0..s31 (float locals), FPSCR in fpsr. A write to r15 sets tpc and makes the
instruction a jump. Memory goes through RD8/16/32/64 and WR8/16/32/64 (runtime/cpu.h).
"""
from .ins import Ins, OP, B, CALL, ICALL, JMP, SVC, UDF


def ror(v, n):
    n &= 31
    return ((v >> n) | (v << (32 - n))) & 0xFFFFFFFF if n else v


def sx(v, bits):
    m = 1 << (bits - 1)
    return (v ^ m) - m


def hx(v):
    return '0x%xu' % (v & 0xFFFFFFFF)


class Ctx:
    """What the decoder may know about memory: read-only words (literal pools)."""

    def __init__(self, ro_read=None):
        self.ro_read = ro_read  # addr -> u32 or None if not read-only


def R(i, n, pc_extra=0):
    if n == 15:
        return hx((i.addr & ~3) + (4 if i.thumb else 8) + pc_extra)
    return 'r%d' % n


def W(i, n, expr):
    if n == 15:
        i.kind = JMP
        return 'tpc = %s;' % expr
    return 'r%d = %s;' % (n, expr)


def shift_imm(x, t, n):
    """Returns (value expr, carry expr or None for 'unchanged')."""
    if t == 0:
        if n == 0:
            return x, None
        return '(%s << %d)' % (x, n), '((%s >> %d) & 1)' % (x, 32 - n)
    if t == 1:
        if n == 0:
            return '0u', '(%s >> 31)' % x
        return '(%s >> %d)' % (x, n), '((%s >> %d) & 1)' % (x, n - 1)
    if t == 2:
        if n == 0:
            return '(u32)((s32)%s >> 31)' % x, '(%s >> 31)' % x
        return '(u32)((s32)%s >> %d)' % (x, n), '((%s >> %d) & 1)' % (x, n - 1)
    if n == 0:
        return '((cf << 31) | (%s >> 1))' % x, '(%s & 1)' % x
    return 'ror32(%s, %d)' % (x, n), '((%s >> %d) & 1)' % (x, n - 1)


SHIFT_NAMES = ['lsl', 'lsr', 'asr', 'ror']

DP_NAMES = ['and', 'eor', 'sub', 'rsb', 'add', 'adc', 'sbc', 'rsc', 'tst', 'teq', 'cmp', 'cmn', 'orr', 'mov', 'bic',
            'mvn']


def dp_core(op, S, a, b, carry_line):
    """C statements computing res (and flags when S) from a, b already in temps. Returns (lines, writes_result)."""
    L = []
    logical = op in (0, 1, 8, 9, 12, 13, 14, 15)
    if op in (0, 8):
        L.append('u32 res = a & b;')
    elif op in (1, 9):
        L.append('u32 res = a ^ b;')
    elif op in (2, 10):
        L.append('u32 res = a - b;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = a >= b; vf = ((a ^ b) & (a ^ res)) >> 31;')
    elif op == 3:
        L.append('u32 res = b - a;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = b >= a; vf = ((b ^ a) & (b ^ res)) >> 31;')
    elif op in (4, 11):
        L.append('u32 res = a + b;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = res < a; vf = ((a ^ res) & (b ^ res)) >> 31;')
    elif op == 5:
        L.append('u64 t = (u64)a + b + cf; u32 res = (u32)t;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = (u32)(t >> 32); vf = ((a ^ res) & (b ^ res)) >> 31;')
    elif op == 6:
        L.append('u64 t = (u64)a + (u32)~b + cf; u32 res = (u32)t;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = (u32)(t >> 32); vf = ((a ^ b) & (a ^ res)) >> 31;')
    elif op == 7:
        L.append('u64 t = (u64)b + (u32)~a + cf; u32 res = (u32)t;')
        if S:
            L.append('nf = res >> 31; zf = res == 0; cf = (u32)(t >> 32); vf = ((b ^ a) & (b ^ res)) >> 31;')
    elif op == 12:
        L.append('u32 res = a | b;')
    elif op == 13:
        L.append('u32 res = b;')
    elif op == 14:
        L.append('u32 res = a & ~b;')
    elif op == 15:
        L.append('u32 res = ~b;')
    if S and logical:
        L.append('nf = res >> 31; zf = res == 0;')
        if carry_line:
            L.append(carry_line)
    return L, op not in (8, 9, 10, 11)


def decode_dp(i, w):
    op = (w >> 21) & 15
    S = (w >> 20) & 1
    rn = (w >> 16) & 15
    rd = (w >> 12) & 15
    logical = op in (0, 1, 8, 9, 12, 13, 14, 15)
    if rd == 15 and op not in (8, 9, 10, 11):
        S = 0  # (MOVS pc, lr: an exception return; user code doesn't do this)
    need_c = S and logical
    pre = []
    carry_line = None
    if (w >> 25) & 1:
        imm8 = w & 0xFF
        rot = ((w >> 8) & 15) * 2
        val = ror(imm8, rot)
        pre.append('u32 b = %s;' % hx(val))
        if need_c and rot:
            carry_line = 'cf = %d;' % (val >> 31)
        opdesc = '#%s' % hex(val)
    else:
        rm = w & 15
        t = (w >> 5) & 3
        if (w >> 4) & 1 == 0:
            n = (w >> 7) & 31
            v, c = shift_imm(R(i, rm), t, n)
            pre.append('u32 b = %s;' % v)
            if need_c and c:
                pre.append('u32 sco = %s;' % c)
                carry_line = 'cf = sco;'
            opdesc = 'r%d %s #%d' % (rm, SHIFT_NAMES[t], n)
        else:
            rs = (w >> 8) & 15
            x = R(i, rm, 4)
            s = R(i, rs, 4)
            if need_c:
                pre.append('u32 sco = cf; u32 b = %s_rc(%s, %s & 0xff, &sco);' % (SHIFT_NAMES[t], x, s))
                carry_line = 'cf = sco;'
            else:
                pre.append('u32 b = %s_r(%s, %s & 0xff);' % (SHIFT_NAMES[t], x, s))
            opdesc = 'r%d %s r%d' % (rm, SHIFT_NAMES[t], rs)
    if op not in (13, 15):
        pre.insert(0, 'u32 a = %s;' % R(i, rn, 0 if ((w >> 25) & 1 or not (w >> 4) & 1) else 4))
    else:
        pre.insert(0, 'u32 a = 0; (void)a;')
    body, writes = dp_core(op, S, 'a', 'b', carry_line)
    lines = pre + body
    if writes:
        lines.append(W(i, rd, 'res'))
    else:
        lines.append('(void)res;')
    i.code.append('{ ' + ' '.join(lines) + ' }')
    i.setflags = bool(S)
    i.readflags = op in (5, 6, 7) or 'cf' in ''.join(pre)
    i.note = '%s%s r%d, r%d, %s' % (DP_NAMES[op], 's' if S else '', rd, rn, opdesc)
    if i.kind == JMP and op == 13 and not (w >> 25) & 1 and (w & 0xFF0) == 0 and (w & 15) == 14:
        i.ret = True  # mov pc, lr


def ldst_addr(i, w, rn, P, U, Wb, off):
    """Returns (pre lines computing ea, writeback lines). off is a C expression."""
    base = R(i, rn)
    sgn = '+' if U else '-'
    if P:
        pre = ['u32 ea = %s %s %s;' % (base, sgn, off)]
        wb = [W(i, rn, 'ea')] if Wb else []
    else:
        pre = ['u32 ea = %s;' % base]
        wb = [W(i, rn, '%s %s %s' % (base, sgn, off))]
    return pre, wb


def decode_ldst(i, w, ctx):
    I = (w >> 25) & 1
    P = (w >> 24) & 1
    U = (w >> 23) & 1
    Bb = (w >> 22) & 1
    Wb = (w >> 21) & 1
    L = (w >> 20) & 1
    rn = (w >> 16) & 15
    rd = (w >> 12) & 15
    if I:
        rm = w & 15
        v, _ = shift_imm(R(i, rm), (w >> 5) & 3, (w >> 7) & 31)
        off = v
    else:
        imm = w & 0xFFF
        off = hx(imm)
        if rn == 15 and P and not Wb and L:
            ea = ((i.addr + 8) + (imm if U else -imm)) & 0xFFFFFFFF
            i.reads_pc_lit = ea
            if ctx.ro_read and not Bb:
                val = ctx.ro_read(ea)
                if val is not None:
                    i.code.append(W(i, rd, hx(val)))
                    i.note = 'ldr r%d, =%s' % (rd, hex(val))
                    if rd == 15:
                        i.ret = False
                    return
    pre, wb = ldst_addr(i, w, rn, P, U, Wb, off)
    sz = 8 if Bb else 32
    if L:
        lines = pre + ['u32 v = RD%d(ea);' % sz] + wb + [W(i, rd, 'v')]
        if rd == 15 and rn == 13 and not P:
            i.ret = True  # ldr pc, [sp], #n
    else:
        val = R(i, rd)
        lines = pre + ['WR%d(ea, %s);' % (sz, val)] + wb
    i.code.append('{ ' + ' '.join(lines) + ' }')
    i.note = '%s%s r%d, [r%d]' % ('ldr' if L else 'str', 'b' if Bb else '', rd, rn)


def decode_extra_ldst(i, w):
    P = (w >> 24) & 1
    U = (w >> 23) & 1
    Iimm = (w >> 22) & 1
    Wb = (w >> 21) & 1
    L = (w >> 20) & 1
    rn = (w >> 16) & 15
    rd = (w >> 12) & 15
    sh = (w >> 5) & 3
    if Iimm:
        off = hx(((w >> 4) & 0xF0) | (w & 15))
    else:
        off = R(i, w & 15)
    pre, wb = ldst_addr(i, w, rn, P, U, Wb, off)
    if sh == 1:
        if L:
            lines = pre + ['u32 v = RD16(ea);'] + wb + [W(i, rd, 'v')]
        else:
            lines = pre + ['WR16(ea, %s);' % R(i, rd)] + wb
        i.note = 'ldrh' if L else 'strh'
    elif sh == 2 and L:
        lines = pre + ['u32 v = (u32)(s32)(s8)RD8(ea);'] + wb + [W(i, rd, 'v')]
        i.note = 'ldrsb'
    elif sh == 3 and L:
        lines = pre + ['u32 v = (u32)(s32)(s16)RD16(ea);'] + wb + [W(i, rd, 'v')]
        i.note = 'ldrsh'
    elif sh == 2:  # LDRD
        lines = pre + ['u32 v0 = RD32(ea); u32 v1 = RD32(ea + 4);'] + wb + [W(i, rd, 'v0'), W(i, rd + 1, 'v1')]
        i.note = 'ldrd'
    else:  # STRD
        lines = pre + ['WR32(ea, %s); WR32(ea + 4, %s);' % (R(i, rd), R(i, rd + 1))] + wb
        i.note = 'strd'
    i.code.append('{ ' + ' '.join(lines) + ' }')


def decode_ldm(i, w):
    P = (w >> 24) & 1
    U = (w >> 23) & 1
    Wb = (w >> 21) & 1
    L = (w >> 20) & 1
    rn = (w >> 16) & 15
    regs = [r for r in range(16) if (w >> r) & 1]
    n = len(regs)
    if n == 0:
        i.kind = UDF
        i.note = 'ldm/stm with empty list'
        return
    base = R(i, rn)
    if U:
        start = '%s + %d' % (base, 4 if P else 0)
        new = '%s + %d' % (base, 4 * n)
    else:
        start = '%s - %d' % (base, 4 * n - (0 if P else 4))
        new = '%s - %d' % (base, 4 * n)
    lines = ['u32 ea = %s;' % start]
    if L:
        if Wb:
            lines.append('u32 nb = %s;' % new)
        loads = []
        for k, r in enumerate(regs):
            loads.append(W(i, r, 'RD32(ea + %d)' % (4 * k)))
        if Wb and rn not in regs:
            lines.append(W(i, rn, 'nb'))
        lines += loads
        if 15 in regs and rn == 13:
            i.ret = True
        i.note = 'ldm r%d%s, {%s}' % (rn, '!' if Wb else '', ','.join('r%d' % r for r in regs))
    else:
        for k, r in enumerate(regs):
            lines.append('WR32(ea + %d, %s);' % (4 * k, R(i, r)))
        if Wb:
            lines.append(W(i, rn, new))
        i.note = 'stm r%d%s, {%s}' % (rn, '!' if Wb else '', ','.join('r%d' % r for r in regs))
    i.code.append('{ ' + ' '.join(lines) + ' }')


def decode_mul(i, w):
    op = (w >> 21) & 7
    S = (w >> 20) & 1
    rd = (w >> 16) & 15
    rn = (w >> 12) & 15
    rs = (w >> 8) & 15
    rm = w & 15
    m, s_, n = R(i, rm), R(i, rs), R(i, rn)
    i.setflags = bool(S)
    if op == 0:
        L = ['u32 res = %s * %s;' % (m, s_)]
        if S:
            L.append('nf = res >> 31; zf = res == 0;')
        L.append(W(i, rd, 'res'))
        i.note = 'mul'
    elif op == 1:
        L = ['u32 res = %s * %s + %s;' % (m, s_, n)]
        if S:
            L.append('nf = res >> 31; zf = res == 0;')
        L.append(W(i, rd, 'res'))
        i.note = 'mla'
    elif op == 2:  # UMAAL
        L = ['u64 res = (u64)%s * %s + %s + %s;' % (m, s_, n, R(i, rd)), W(i, rn, '(u32)res'),
             W(i, rd, '(u32)(res >> 32)')]
        i.note = 'umaal'
    elif op in (4, 5, 6, 7):
        lo, hi = rn, rd
        if op & 2:
            prod = '(u64)((s64)(s32)%s * (s32)%s)' % (m, s_)
        else:
            prod = '((u64)%s * %s)' % (m, s_)
        if op & 1:
            prod = '(%s + (((u64)%s << 32) | %s))' % (prod, R(i, hi), R(i, lo))
        L = ['u64 res = %s;' % prod]
        if S:
            L.append('nf = (u32)(res >> 63); zf = res == 0;')
        L += [W(i, lo, '(u32)res'), W(i, hi, '(u32)(res >> 32)')]
        i.note = ['umull', 'umlal', 'smull', 'smlal'][op - 4]
    else:
        i.kind = UDF
        i.note = 'mul op %d' % op
        return
    i.code.append('{ ' + ' '.join(L) + ' }')


def half(x, top):
    return '(s32)(s16)(%s >> 16)' % x if top else '(s32)(s16)%s' % x


def decode_misc(i, w):
    """bits 27-23 = 00010, bit 20 = 0 (MRS/MSR/BX/BLX/CLZ/QADD/BKPT/SMLAxy)."""
    op = (w >> 21) & 3
    op2 = (w >> 4) & 15
    rd = (w >> 12) & 15
    rn = (w >> 16) & 15
    rm = w & 15
    if op2 == 0:
        if op & 1 == 0:  # MRS
            if (w >> 22) & 1:
                i.code.append(W(i, rd, '0x10u'))  # SPSR: not in user mode
            else:
                i.code.append(W(i, rd, '(nf << 31) | (zf << 30) | (cf << 29) | (vf << 28) | (qf << 27) | (ge << 16) | 0x10u'))
            i.readflags = True
            i.note = 'mrs'
        else:  # MSR reg
            emit_msr(i, w, R(i, rm))
        return
    if op2 == 1 and op == 1:
        i.kind = JMP
        i.code.append('tpc = %s;' % R(i, rm))
        i.ret = rm == 14
        i.note = 'bx r%d' % rm
        return
    if op2 == 1 and op == 3:
        i.code.append(W(i, rd, '%s ? (u32)__builtin_clz(%s) : 32u' % (R(i, rm), R(i, rm))))
        i.note = 'clz'
        return
    if op2 == 2 and op == 1:  # BXJ: behaves as BX
        i.kind = JMP
        i.code.append('tpc = %s;' % R(i, rm))
        i.note = 'bxj'
        return
    if op2 == 3 and op == 1:
        i.kind = ICALL
        i.code.append('tpc = %s; r14 = %s;' % (R(i, rm), hx(i.addr + 4)))
        i.link = i.addr + 4
        i.note = 'blx r%d' % rm
        return
    if op2 == 5:
        fn = ['qadd', 'qsub', 'qdadd', 'qdsub'][op]
        i.code.append(W(i, rd, '%s(%s, %s, &qf)' % (fn, R(i, rm), R(i, rn))))
        i.note = fn
        return
    if op2 == 7 and op == 1:
        i.kind = UDF
        i.note = 'bkpt'
        return
    if op2 & 9 == 8:  # SMLAxy family: bit7=1 bit4=0
        x = (w >> 5) & 1
        y = (w >> 6) & 1
        rd_ = rn  # bits 19-16
        ra = rd   # bits 15-12
        rs = (w >> 8) & 15
        if op == 0:
            i.code.append(W(i, rd_, 'smlaxy(%s, %s, %s, &qf)' % (half(R(i, rm), x), half(R(i, rs), y), R(i, ra))))
            i.note = 'smlaxy'
        elif op == 1:
            prod = '(u32)(((s64)(s32)%s * %s) >> 16)' % (R(i, rm), half(R(i, rs), y))
            if x == 0:
                i.code.append(W(i, rd_, 'smlaxy_add(%s, %s, &qf)' % (prod, R(i, ra))))
                i.note = 'smlawy'
            else:
                i.code.append(W(i, rd_, prod))
                i.note = 'smulwy'
        elif op == 2:
            lo, hi = ra, rd_
            i.code.append('{ u64 acc = ((u64)%s << 32) | %s; acc += (u64)(s64)(%s * %s); %s %s }' % (
                R(i, hi), R(i, lo), half(R(i, rm), x), half(R(i, rs), y), W(i, lo, '(u32)acc'),
                W(i, hi, '(u32)(acc >> 32)')))
            i.note = 'smlalxy'
        else:
            i.code.append(W(i, rd_, '(u32)(%s * %s)' % (half(R(i, rm), x), half(R(i, rs), y))))
            i.note = 'smulxy'
        return
    i.kind = UDF
    i.note = 'misc %08x' % w


def emit_msr(i, w, val):
    mask = (w >> 16) & 15
    if (w >> 22) & 1:
        i.note = 'msr spsr (ignored)'
        return
    L = ['u32 v = %s;' % val]
    if mask & 8:
        L.append('nf = v >> 31; zf = (v >> 30) & 1; cf = (v >> 29) & 1; vf = (v >> 28) & 1; qf = (v >> 27) & 1;')
    if mask & 4:
        L.append('ge = (v >> 16) & 15;')
    L.append('(void)v;')
    i.code.append('{ ' + ' '.join(L) + ' }')
    i.setflags = True
    i.note = 'msr cpsr'


def decode_sync(i, w):
    """SWP/SWPB, LDREX/STREX family (bits 27-24 = 0001, bits 7-4 = 1001)."""
    op = (w >> 20) & 0x1F
    rn = (w >> 16) & 15
    rd = (w >> 12) & 15
    rm = w & 15
    if op in (0x10, 0x14):  # SWP / SWPB
        sz = 8 if op == 0x14 else 32
        i.code.append('{ u32 ea = %s; u32 v = RD%d(ea); WR%d(ea, %s); %s }' % (
            R(i, rn), sz, sz, R(i, rm), W(i, rd, 'v')))
        i.note = 'swp'
        return
    kinds = {0x18: ('strex', 32), 0x19: ('ldrex', 32), 0x1A: ('strexd', 64), 0x1B: ('ldrexd', 64),
             0x1C: ('strexb', 8), 0x1D: ('ldrexb', 8), 0x1E: ('strexh', 16), 0x1F: ('ldrexh', 16)}
    if op not in kinds:
        i.kind = UDF
        i.note = 'sync %x' % op
        return
    name, sz = kinds[op]
    i.note = name
    # A single guest thread runs at a time and switches only at SVCs, so exclusives always succeed.
    if name.startswith('ldrex'):
        if sz == 64:
            i.code.append('{ u32 ea = %s; u32 v0 = RD32(ea), v1 = RD32(ea + 4); %s %s }' % (
                R(i, rn), W(i, rd, 'v0'), W(i, rd + 1, 'v1')))
        else:
            i.code.append('{ u32 v = RD%d(%s); %s }' % (sz, R(i, rn), W(i, rd, 'v')))
    else:
        if sz == 64:
            i.code.append('{ u32 ea = %s; WR32(ea, %s); WR32(ea + 4, %s); %s }' % (
                R(i, rn), R(i, rm), R(i, rm + 1), W(i, rd, '0')))
        else:
            i.code.append('{ WR%d(%s, %s); %s }' % (sz, R(i, rn), R(i, rm), W(i, rd, '0')))


EXT_FN = {0x68: 'sxtb16', 0x6A: 'sxtb', 0x6B: 'sxth', 0x6C: 'uxtb16', 0x6E: 'uxtb', 0x6F: 'uxth'}
PAS_PRE = {1: 's', 2: 'q', 3: 'sh', 5: 'u', 6: 'uq', 7: 'uh'}
PAS_OP = {0: 'add16', 1: 'asx', 2: 'sax', 3: 'sub16', 4: 'add8', 7: 'sub8'}


def decode_media(i, w):
    op1 = (w >> 20) & 0xFF
    op2 = (w >> 5) & 7
    rn = (w >> 16) & 15
    rd = (w >> 12) & 15
    rs = (w >> 8) & 15
    rm = w & 15
    if 0x61 <= op1 <= 0x67 and (op1 & 7) in PAS_PRE and op2 in PAS_OP:
        fn = 'pas_%s%s' % (PAS_PRE[op1 & 7], PAS_OP[op2])
        i.code.append(W(i, rd, '%s(%s, %s, &ge, &qf)' % (fn, R(i, rn), R(i, rm))))
        i.note = fn
        return
    if op1 == 0x68 and op2 & 1 == 0:  # PKH
        sh = (w >> 7) & 31
        if (w >> 6) & 1:  # PKHTB: top of Rn, bottom of Rm ASR
            v, _ = shift_imm(R(i, rm), 2, sh)
            i.code.append(W(i, rd, '(%s & 0xffff0000u) | (%s & 0xffffu)' % (R(i, rn), v)))
            i.note = 'pkhtb'
        else:
            v, _ = shift_imm(R(i, rm), 0, sh)
            i.code.append(W(i, rd, '(%s & 0xffffu) | (%s & 0xffff0000u)' % (R(i, rn), v)))
            i.note = 'pkhbt'
        return
    if op1 in EXT_FN and op2 == 3:
        rot = ((w >> 10) & 3) * 8
        v = 'ror32(%s, %d)' % (R(i, rm), rot) if rot else R(i, rm)
        fn = EXT_FN[op1]
        if rn == 15:
            i.code.append(W(i, rd, '%s(%s)' % (fn, v)))
            i.note = fn
        else:
            if fn.endswith('16'):
                i.code.append(W(i, rd, 'add16x(%s, %s(%s))' % (R(i, rn), fn, v)))
            else:
                i.code.append(W(i, rd, '%s + %s(%s)' % (R(i, rn), fn, v)))
            i.note = fn[0] + 'xta' + fn[3:]
        return
    if op1 == 0x68 and op2 == 5:
        i.code.append(W(i, rd, 'sel(%s, %s, ge)' % (R(i, rn), R(i, rm))))
        i.note = 'sel'
        return
    if op1 in (0x6A, 0x6B, 0x6E, 0x6F) and op2 & 1 == 0:  # SSAT/USAT
        sat = (w >> 16) & 31
        sh = (w >> 7) & 31
        v, _ = shift_imm(R(i, rm), 2 if (w >> 6) & 1 else 0, sh)
        if (w >> 6) & 1 and sh == 0:
            v = '(u32)((s32)%s >> 31)' % R(i, rm)
        if op1 & 4:
            i.code.append(W(i, rd, 'usat(%s, %d, &qf)' % (v, sat)))
            i.note = 'usat'
        else:
            i.code.append(W(i, rd, 'ssat(%s, %d, &qf)' % (v, sat + 1)))
            i.note = 'ssat'
        return
    if op1 in (0x6A, 0x6E) and op2 == 1:
        sat = (w >> 16) & 15
        if op1 == 0x6E:
            i.code.append(W(i, rd, 'usat16(%s, %d, &qf)' % (R(i, rm), sat)))
        else:
            i.code.append(W(i, rd, 'ssat16(%s, %d, &qf)' % (R(i, rm), sat + 1)))
        i.note = 'sat16'
        return
    if op1 == 0x6B and op2 == 1:
        i.code.append(W(i, rd, '__builtin_bswap32(%s)' % R(i, rm)))
        i.note = 'rev'
        return
    if op1 == 0x6B and op2 == 5:
        i.code.append(W(i, rd, 'rev16(%s)' % R(i, rm)))
        i.note = 'rev16'
        return
    if op1 == 0x6F and op2 == 5:
        i.code.append(W(i, rd, '(u32)(s32)(s16)__builtin_bswap16((u16)%s)' % R(i, rm)))
        i.note = 'revsh'
        return
    # signed multiplies: Rd = 19-16, Ra = 15-12, Rm = 11-8, Rn = 3-0
    rd_, ra, rm_, rn_ = rn, rd, rs, rm
    if op1 == 0x70 and op2 in (0, 1, 2, 3):
        X = op2 & 1
        sub = op2 & 2
        m = 'ror32(%s, 16)' % R(i, rm_) if X else R(i, rm_)
        p1 = '(s64)((s32)(s16)%s * (s32)(s16)%s)' % (R(i, rn_), m)
        p2 = '(s64)((s32)(s16)(%s >> 16) * (s32)(s16)(%s >> 16))' % (R(i, rn_), m)
        acc = '0' if ra == 15 else '(s64)(s32)%s' % R(i, ra)
        i.code.append('{ u32 mm = %s; (void)mm; s64 res = %s %s %s + %s; if (res != (s64)(s32)res) qf = 1; %s }' % (
            m, p1, '-' if sub else '+', p2, acc, W(i, rd_, '(u32)res')))
        i.note = 'smlad/smlsd'
        return
    if op1 == 0x74 and op2 in (0, 1, 2, 3):
        X = op2 & 1
        sub = op2 & 2
        m = 'ror32(%s, 16)' % R(i, rm_) if X else R(i, rm_)
        p1 = '(s64)((s32)(s16)%s * (s32)(s16)%s)' % (R(i, rn_), m)
        p2 = '(s64)((s32)(s16)(%s >> 16) * (s32)(s16)(%s >> 16))' % (R(i, rn_), m)
        lo, hi = ra, rd_
        i.code.append('{ s64 acc = (s64)(((u64)%s << 32) | %s); acc += %s %s %s; %s %s }' % (
            R(i, hi), R(i, lo), p1, '-' if sub else '+', p2, W(i, lo, '(u32)acc'), W(i, hi, '(u32)((u64)acc >> 32)')))
        i.note = 'smlald'
        return
    if op1 == 0x75 and op2 in (0, 1, 6, 7):
        rnd = '0x80000000ll' if op2 & 1 else '0'
        prod = '(s64)(s32)%s * (s32)%s' % (R(i, rn_), R(i, rm_))
        if op2 >= 6:
            e = '(u32)((((s64)%s << 32) - (%s) + %s) >> 32)' % (R(i, ra), prod, rnd)
        elif ra == 15:
            e = '(u32)(((%s) + %s) >> 32)' % (prod, rnd)
        else:
            e = '(u32)((((s64)%s << 32) + (%s) + %s) >> 32)' % (R(i, ra), prod, rnd)
        i.code.append(W(i, rd_, e))
        i.note = 'smmul'
        return
    if op1 == 0x78 and op2 == 0:
        e = 'usad8(%s, %s)' % (R(i, rn_), R(i, rm_))
        if ra != 15:
            e = '%s + %s' % (e, R(i, ra))
        i.code.append(W(i, rd_, e))
        i.note = 'usad8'
        return
    i.kind = UDF
    i.note = 'media %08x' % w


# ---------------------------------------------------------------- VFP

def sreg(v, b):
    return (v << 1) | b


def S(n):
    return 'S%d' % n


def Dget(n):
    return 'D2F(S%d, S%d)' % (2 * n, 2 * n + 1)


def Dset(n, e):
    return 'DSET(S%d, S%d, %s);' % (2 * n, 2 * n + 1, e)


def decode_vfp_dp(i, w):
    dbl = (w >> 8) & 1
    opc1 = (w >> 20) & 0xB
    op = (w >> 6) & 1
    Vd, Vn, Vm = (w >> 12) & 15, (w >> 16) & 15, w & 15
    Db, Nb, Mb = (w >> 22) & 1, (w >> 7) & 1, (w >> 5) & 1
    if dbl:
        # in conversions with a single-precision operand, D (destination) or M (source) is part of an S number
        cvt = opc1 == 0xB and ((w >> 6) & 1)
        s_dst = cvt and Vn in (7, 12, 13)  # f64->f32, f64->int
        s_src = cvt and Vn == 8  # int->f64
        if (Db and not s_dst) or (Mb and not s_src) or (Nb and opc1 != 0xB):
            i.kind = UDF
            i.note = 'vfp d16+'
            return
        d, n, m = Vd, Vn, Vm
        get = Dget
        put = Dset
        ty = 'double'
        sfx = ''
    else:
        d, n, m = sreg(Vd, Db), sreg(Vn, Nb), sreg(Vm, Mb)
        get = S
        put = lambda r, e: 'S%d = %s;' % (r, e)
        ty = 'float'
        sfx = 'f'
    tag = '.f64' if dbl else '.f32'
    if opc1 != 0xB:
        a, b, dd = get(n), get(m), get(d)
        if opc1 == 0:
            e = '%s %s (%s)(%s * %s)' % (dd, '-' if op else '+', ty, a, b)
            nm = 'vmls' if op else 'vmla'
        elif opc1 == 1:
            # VNMLS: d = -d + n*m ; VNMLA: d = -d - n*m
            e = '-%s %s (%s)(%s * %s)' % (dd, '-' if op else '+', ty, a, b)
            nm = 'vnmla' if op else 'vnmls'
        elif opc1 == 2:
            e = '-(%s)(%s * %s)' % (ty, a, b) if op else '%s * %s' % (a, b)
            nm = 'vnmul' if op else 'vmul'
        elif opc1 == 3:
            e = '%s %s %s' % (a, '-' if op else '+', b)
            nm = 'vsub' if op else 'vadd'
        elif opc1 == 8 and not op:
            e = '%s / %s' % (a, b)
            nm = 'vdiv'
        else:
            i.kind = UDF
            i.note = 'vfp opc1 %x' % opc1
            return
        i.code.append('{ %s t = %s; %s }' % (ty, e, put(d, 't')))
        i.note = nm + tag
        return
    opc2 = Vn
    opc3 = (w >> 6) & 3
    if opc3 & 1 == 0:
        i.kind = UDF
        i.note = 'vmov imm (VFPv3)'
        return
    if opc2 == 0:
        if opc3 == 1:
            i.code.append(put(d, get(m)))
            i.note = 'vmov' + tag
        else:
            i.code.append(put(d, 'fabs%s(%s)' % ('' if dbl else 'f', get(m))))
            i.note = 'vabs' + tag
        return
    if opc2 == 1:
        if opc3 == 1:
            i.code.append(put(d, '-%s' % get(m)))
            i.note = 'vneg' + tag
        else:
            i.code.append(put(d, 'sqrt%s(%s)' % ('' if dbl else 'f', get(m))))
            i.note = 'vsqrt' + tag
        return
    if opc2 in (4, 5):
        b = '0.0' if opc2 == 5 else get(m)
        i.code.append('fpsr = (fpsr & 0x0fffffffu) | vfp_cmp%s(%s, %s);' % ('d' if dbl else 'f', get(d), b))
        i.note = 'vcmp' + tag
        return
    if opc2 == 7 and opc3 == 3:
        if dbl:  # double -> single: Sd = (float)Dm
            i.code.append('S%d = (float)%s;' % (sreg(Vd, Db), Dget(Vm)))
            i.note = 'vcvt.f32.f64'
        else:
            i.code.append(Dset(Vd, '(double)%s' % S(sreg(Vm, Mb))))
            i.note = 'vcvt.f64.f32'
        return
    if opc2 == 8:  # int -> float; source is always a single register
        src = 'S%d' % sreg(Vm, Mb)
        signed = (w >> 7) & 1
        cv = '(%s)(%s)F2U(%s)' % (ty, 's32' if signed else 'u32', src)
        if dbl:
            i.code.append(Dset(Vd, cv))
        else:
            i.code.append('S%d = %s;' % (sreg(Vd, Db), cv))
        i.note = 'vcvt.from_int' + tag
        return
    if opc2 in (12, 13):
        signed = opc2 & 1
        rz = (w >> 7) & 1
        src = Dget(Vm) if dbl else S(sreg(Vm, Mb))
        fn = 'vcvt_%s%s_%s' % ('s' if signed else 'u', 'z' if rz else 'r', 'd' if dbl else 'f')
        i.code.append('S%d = U2F(%s(%s%s));' % (sreg(Vd, Db), fn, src, '' if rz else ', fpsr'))
        i.note = 'vcvt.to_int' + tag
        return
    i.kind = UDF
    i.note = 'vfp other opc2=%d' % opc2


def decode_vfp_xfer(i, w):
    """MCR/MRC with cp10/cp11."""
    L = (w >> 20) & 1
    opc1 = (w >> 21) & 7
    rt = (w >> 12) & 15
    cp = (w >> 8) & 15
    Vn = (w >> 16) & 15
    N = (w >> 7) & 1
    if cp == 10 and opc1 == 0:
        sn = sreg(Vn, N)
        if L:
            i.code.append(W(i, rt, 'F2U(S%d)' % sn) if rt != 15 else '')
        else:
            i.code.append('S%d = U2F(%s);' % (sn, R(i, rt)))
        i.note = 'vmov s<->r'
        return
    if cp == 10 and opc1 == 7:
        reg = Vn
        if L:
            if reg == 1:
                if rt == 15:
                    i.code.append('nf = fpsr >> 31; zf = (fpsr >> 30) & 1; cf = (fpsr >> 29) & 1; vf = (fpsr >> 28) & 1;')
                    i.setflags = True
                    i.note = 'vmrs apsr_nzcv, fpscr'
                else:
                    i.code.append(W(i, rt, 'fpsr'))
                    i.note = 'vmrs fpscr'
            elif reg == 0:
                i.code.append(W(i, rt, '0x410120b4u'))  # FPSID of the ARM11 VFP
                i.note = 'vmrs fpsid'
            elif reg == 8:
                i.code.append(W(i, rt, '0x40000000u'))  # FPEXC.EN
                i.note = 'vmrs fpexc'
            else:
                i.code.append(W(i, rt, '0u'))
                i.note = 'vmrs other'
        else:
            if reg == 1:
                i.code.append('fpsr = %s;' % R(i, rt))
                i.note = 'vmsr fpscr'
            else:
                i.note = 'vmsr other (ignored)'
        return
    if cp == 11 and opc1 in (0, 1):
        dn = Vn
        lohi = opc1 & 1
        sn = 2 * dn + lohi
        if L:
            i.code.append(W(i, rt, 'F2U(S%d)' % sn))
        else:
            i.code.append('S%d = U2F(%s);' % (sn, R(i, rt)))
        i.note = 'vmov d[x]<->r'
        return
    i.kind = UDF
    i.note = 'vfp xfer %08x' % w


def decode_vfp_ldst(i, w):
    P = (w >> 24) & 1
    U = (w >> 23) & 1
    Db = (w >> 22) & 1
    Wb = (w >> 21) & 1
    L = (w >> 20) & 1
    rn = (w >> 16) & 15
    Vd = (w >> 12) & 15
    dbl = (w >> 8) & 1
    imm8 = w & 0xFF
    first = Vd if dbl else sreg(Vd, Db)
    if P and not Wb:  # VLDR / VSTR
        base = R(i, rn) if rn != 15 else hx((i.addr & ~3) + 8)
        ea = '%s %s %s' % (base, '+' if U else '-', hx(imm8 * 4))
        if dbl:
            if L:
                i.code.append('{ u32 ea = %s; S%d = U2F(RD32(ea)); S%d = U2F(RD32(ea + 4)); }' % (ea, 2 * first, 2 * first + 1))
            else:
                i.code.append('{ u32 ea = %s; WR32(ea, F2U(S%d)); WR32(ea + 4, F2U(S%d)); }' % (ea, 2 * first, 2 * first + 1))
        else:
            if L:
                i.code.append('S%d = U2F(RD32(%s));' % (first, ea))
            else:
                i.code.append('WR32(%s, F2U(S%d));' % (ea, first))
        i.note = ('vldr' if L else 'vstr') + ('.64' if dbl else '.32')
        return
    # VLDM / VSTM
    if dbl:
        nregs = imm8 // 2
        sregs = []
        for k in range(nregs):
            sregs += [2 * (first + k), 2 * (first + k) + 1]
    else:
        sregs = [first + k for k in range(imm8)]
    nbytes = imm8 * 4
    if any(s > 31 for s in sregs) or not sregs:
        i.kind = UDF
        i.note = 'vldm range'
        return
    base = R(i, rn)
    if P and not U:   # DB, writeback required
        start = '%s - %s' % (base, hx(nbytes))
        new = start
    elif not P and U:  # IA
        start = base
        new = '%s + %s' % (base, hx(nbytes))
    else:
        i.kind = UDF
        i.note = 'vldm mode'
        return
    lines = ['u32 ea = %s;' % start]
    for k, s in enumerate(sregs):
        if L:
            lines.append('S%d = U2F(RD32(ea + %d));' % (s, 4 * k))
        else:
            lines.append('WR32(ea + %d, F2U(S%d));' % (4 * k, s))
    if Wb:
        lines.append(W(i, rn, new))
    i.code.append('{ ' + ' '.join(lines) + ' }')
    i.note = ('vldm' if L else 'vstm') + (' sp!' if rn == 13 and Wb else '')


def decode_vfp_xfer64(i, w):
    """MCRR/MRRC with cp10 / cp11."""
    L = (w >> 20) & 1
    rt2 = (w >> 16) & 15
    rt = (w >> 12) & 15
    cp = (w >> 8) & 15
    M = (w >> 5) & 1
    Vm = w & 15
    if cp == 11:
        a, b = 2 * Vm, 2 * Vm + 1
    else:
        a = sreg(Vm, M)
        b = a + 1
    if L:
        i.code.append('{ u32 t0 = F2U(S%d), t1 = F2U(S%d); %s %s }' % (a, b, W(i, rt, 't0'), W(i, rt2, 't1')))
    else:
        i.code.append('S%d = U2F(%s); S%d = U2F(%s);' % (a, R(i, rt), b, R(i, rt2)))
    i.note = 'vmov 64'


def decode_cp15(i, w):
    L = (w >> 20) & 1
    opc1 = (w >> 21) & 7
    crn = (w >> 16) & 15
    rt = (w >> 12) & 15
    opc2 = (w >> 5) & 7
    crm = w & 15
    if L and crn == 13 and crm == 0 and opc2 == 3 and opc1 == 0:
        i.code.append(W(i, rt, 'c->tls'))
        i.note = 'mrc tls'
        return
    if not L and crn == 7:
        i.note = 'mcr p15 c7 (cache/barrier)'
        return
    if L and crn == 0:
        i.code.append(W(i, rt, '0x410fb025u'))  # MIDR / CPUID: core 0
        i.note = 'mrc cpuid'
        return
    i.kind = UDF
    i.note = 'cp15 %08x' % w


# ---------------------------------------------------------------- top level

def decode(addr, w, ctx):
    i = Ins(addr, 4, False, w)
    cond = w >> 28
    i.cond = cond
    if cond == 15:
        decode_uncond(i, w)
        return i
    t = (w >> 25) & 7
    if t == 0:
        if (w & 0x90) == 0x90:
            if (w >> 5) & 3 == 0:
                if (w >> 24) & 1:
                    decode_sync(i, w)
                else:
                    decode_mul(i, w)
            else:
                decode_extra_ldst(i, w)
        elif (w >> 23) & 3 == 2 and not (w >> 20) & 1:
            decode_misc(i, w)
        else:
            decode_dp(i, w)
    elif t == 1:
        if (w >> 23) & 3 == 2 and not (w >> 20) & 1:
            if (w >> 21) & 1:  # MSR imm / hints
                if (w >> 16) & 15 == 0 and not (w >> 22) & 1:
                    i.note = 'nop/hint %x' % (w & 0xFF)
                else:
                    emit_msr(i, w, hx(ror(w & 0xFF, ((w >> 8) & 15) * 2)))
            else:
                i.kind = UDF
                i.note = 'movw/movt (not ARMv6)'
        else:
            decode_dp(i, w)
    elif t == 2:
        decode_ldst(i, w, ctx)
    elif t == 3:
        if (w >> 4) & 1:
            decode_media(i, w)
        else:
            decode_ldst(i, w, ctx)
    elif t == 4:
        decode_ldm(i, w)
    elif t == 5:
        off = sx(w & 0xFFFFFF, 24) << 2
        i.target = (addr + 8 + off) & 0xFFFFFFFF
        if (w >> 24) & 1:
            i.kind = CALL
            i.link = addr + 4
            i.code.append('r14 = %s;' % hx(addr + 4))
            i.note = 'bl %x' % i.target
        else:
            i.kind = B
            i.note = 'b %x' % i.target
    elif t == 6:
        cp = (w >> 8) & 15
        if cp in (10, 11):
            if (w >> 21) & 0x7F == 0x62:
                decode_vfp_xfer64(i, w)
            else:
                decode_vfp_ldst(i, w)
        else:
            i.kind = UDF
            i.note = 'ldc/stc p%d' % cp
    else:
        if (w >> 24) & 1:
            i.kind = SVC
            i.imm = w & 0xFFFFFF
            i.note = 'svc %x' % i.imm
        else:
            cp = (w >> 8) & 15
            if cp in (10, 11):
                if (w >> 4) & 1:
                    decode_vfp_xfer(i, w)
                else:
                    decode_vfp_dp(i, w)
            elif cp == 15 and (w >> 4) & 1:
                decode_cp15(i, w)
            else:
                i.kind = UDF
                i.note = 'cdp/mcr p%d' % cp
    i.code = [c for c in i.code if c]
    return i


def decode_uncond(i, w):
    i.cond = 14
    if (w >> 25) & 7 == 5:  # BLX imm
        off = (sx(w & 0xFFFFFF, 24) << 2) | (((w >> 24) & 1) << 1)
        i.target = ((i.addr + 8 + off) & 0xFFFFFFFF) | 1
        i.kind = CALL
        i.link = i.addr + 4
        i.code.append('r14 = %s;' % hx(i.addr + 4))
        i.note = 'blx %x' % i.target
        return
    if (w & 0x0D700000) == 0x05500000:  # PLD
        i.note = 'pld'
        return
    if w == 0xF57FF01F:
        i.note = 'clrex'
        return
    if (w & 0xFFFFFFF0) in (0xF57FF040, 0xF57FF050, 0xF57FF060):
        i.note = 'barrier'
        return
    if (w & 0xFFF10020) == 0xF1000000:
        i.note = 'cps (ignored)'
        return
    if (w & 0xFFFFFDFF) == 0xF1010000:
        i.note = 'setend (ignored)'
        return
    i.kind = UDF
    i.note = 'uncond %08x' % w
