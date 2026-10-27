"""ARMv6 Thumb (Thumb-1 plus BLX) decoder lowering to the same C as arm.py."""
from .ins import Ins, OP, B, CALL, ICALL, JMP, SVC, UDF
from .arm import hx, sx, shift_imm, dp_core, W


def r(n):
    return 'r%d' % n


def pcval(i):
    return hx(i.addr + 4)


def flags_nz():
    return 'nf = res >> 31; zf = res == 0;'


def decode(addr, hw, hw2, ctx):
    i = Ins(addr, 2, True, hw)
    try:
        _decode(i, hw, hw2, ctx)
    except IndexError:
        i.kind = UDF
    i.code = [c for c in i.code if c]
    return i


def _decode(i, hw, hw2, ctx):
    top = hw >> 13
    if top == 0:
        op = (hw >> 11) & 3
        rd, rm = hw & 7, (hw >> 3) & 7
        if op < 3:
            n = (hw >> 6) & 31
            v, c = shift_imm('b0', op, n)
            L = ['u32 b0 = %s;' % r(rm), 'u32 b = %s;' % v]
            if c:
                L.append('cf = %s;' % c)
            L.append('u32 res = b; nf = res >> 31; zf = res == 0; %s' % W(i, rd, 'res'))
            i.code.append('{ ' + ' '.join(L) + ' }')
            i.setflags = True
            i.note = ['lsls', 'lsrs', 'asrs'][op]
            return
        rn = (hw >> 6) & 7
        a = 'u32 a = %s;' % r(rm)
        b = 'u32 b = %s;' % (hx(rn) if (hw >> 10) & 1 else r(rn))
        body, _ = dp_core(2 if (hw >> 9) & 1 else 4, 1, 'a', 'b', None)
        i.code.append('{ %s %s %s %s }' % (a, b, ' '.join(body), W(i, rd, 'res')))
        i.setflags = True
        i.note = 'adds/subs'
        return
    if top == 1:
        op = (hw >> 11) & 3
        rd = (hw >> 8) & 7
        imm = hw & 0xFF
        if op == 0:
            i.code.append('{ u32 res = %s; nf = 0; zf = res == 0; %s }' % (hx(imm), W(i, rd, 'res')))
            i.note = 'movs'
        else:
            dop = {1: 10, 2: 4, 3: 2}[op]
            body, wr = dp_core(dop, 1, 'a', 'b', None)
            i.code.append('{ u32 a = %s; u32 b = %s; %s %s }' % (r(rd), hx(imm), ' '.join(body),
                                                                W(i, rd, 'res') if wr else '(void)res;'))
            i.note = ['', 'cmp', 'adds', 'subs'][op]
        i.setflags = True
        return
    if hw >> 10 == 0x10:
        op = (hw >> 6) & 15
        rm, rd = (hw >> 3) & 7, hw & 7
        a, b = r(rd), r(rm)
        i.setflags = True
        if op in (2, 3, 4, 7):
            sh = {2: 'lsl', 3: 'lsr', 4: 'asr', 7: 'ror'}[op]
            i.code.append('{ u32 sco = cf; u32 res = %s_rc(%s, %s & 0xff, &sco); cf = sco; %s %s }' % (
                sh, a, b, flags_nz(), W(i, rd, 'res')))
            i.note = sh + 's'
            return
        if op == 9:  # NEG = RSBS rd, rm, #0
            body, _ = dp_core(3, 1, 'a', 'b', None)
            i.code.append('{ u32 a = %s; u32 b = 0; %s %s }' % (b, ' '.join(body), W(i, rd, 'res')))
            i.note = 'negs'
            return
        if op == 13:
            i.code.append('{ u32 res = %s * %s; %s %s }' % (a, b, flags_nz(), W(i, rd, 'res')))
            i.note = 'muls'
            return
        dop = {0: 0, 1: 1, 5: 5, 6: 6, 8: 8, 10: 10, 11: 11, 12: 12, 14: 14, 15: 15}[op]
        body, wr = dp_core(dop, 1, 'a', 'b', None)
        i.code.append('{ u32 a = %s; u32 b = %s; %s %s }' % (a, b, ' '.join(body), W(i, rd, 'res') if wr else '(void)res;'))
        i.note = 'alu%d' % op
        return
    if hw >> 10 == 0x11:
        op = (hw >> 8) & 3
        rd = ((hw >> 4) & 8) | (hw & 7)
        rm = (hw >> 3) & 15
        src = pcval(i) if rm == 15 else r(rm)
        if op == 0:
            if rd == 15:
                i.kind = JMP
                i.code.append('tpc = (%s + %s) | 1;' % (pcval(i), src))
            else:
                i.code.append('r%d = r%d + %s;' % (rd, rd, src))
            i.note = 'add hi'
        elif op == 1:
            body, _ = dp_core(10, 1, 'a', 'b', None)
            i.code.append('{ u32 a = %s; u32 b = %s; %s (void)res; }' % (pcval(i) if rd == 15 else r(rd), src, ' '.join(body)))
            i.setflags = True
            i.note = 'cmp hi'
        elif op == 2:
            if rd == 15:
                i.kind = JMP
                i.code.append('tpc = %s | 1;' % src)
                i.ret = rm == 14
            else:
                i.code.append('r%d = %s;' % (rd, src))
            i.note = 'mov hi'
        else:
            if (hw >> 7) & 1:
                i.kind = ICALL
                i.code.append('tpc = %s; r14 = %s;' % (src, hx((i.addr + 2) | 1)))
                i.link = i.addr + 2
                i.note = 'blx r%d' % rm
            else:
                i.kind = JMP
                i.code.append('tpc = %s;' % src)
                i.ret = rm == 14
                i.note = 'bx r%d' % rm
        return
    if hw >> 11 == 9:
        rd = (hw >> 8) & 7
        ea = ((i.addr + 4) & ~3) + (hw & 0xFF) * 4
        i.reads_pc_lit = ea
        v = ctx.ro_read(ea) if ctx.ro_read else None
        if v is not None:
            i.code.append('r%d = %s;' % (rd, hx(v)))
        else:
            i.code.append('r%d = RD32(%s);' % (rd, hx(ea)))
        i.note = 'ldr literal'
        return
    if hw >> 12 == 5:
        op = (hw >> 9) & 7
        rm, rn, rd = (hw >> 6) & 7, (hw >> 3) & 7, hw & 7
        ea = 'u32 ea = r%d + r%d;' % (rn, rm)
        e = ['WR32(ea, r%d);', 'WR16(ea, r%d);', 'WR8(ea, r%d);', 'r%d = (u32)(s32)(s8)RD8(ea);',
             'r%d = RD32(ea);', 'r%d = RD16(ea);', 'r%d = RD8(ea);', 'r%d = (u32)(s32)(s16)RD16(ea);'][op] % rd
        i.code.append('{ %s %s }' % (ea, e))
        i.note = 'ldst reg'
        return
    if top == 3:
        Bb = (hw >> 12) & 1
        L = (hw >> 11) & 1
        imm = (hw >> 6) & 31
        rn, rd = (hw >> 3) & 7, hw & 7
        off = imm if Bb else imm * 4
        sz = 8 if Bb else 32
        if L:
            i.code.append('r%d = RD%d(r%d + %s);' % (rd, sz, rn, hx(off)))
        else:
            i.code.append('WR%d(r%d + %s, r%d);' % (sz, rn, hx(off), rd))
        i.note = 'ldst imm'
        return
    if hw >> 12 == 8:
        L = (hw >> 11) & 1
        off = ((hw >> 6) & 31) * 2
        rn, rd = (hw >> 3) & 7, hw & 7
        if L:
            i.code.append('r%d = RD16(r%d + %s);' % (rd, rn, hx(off)))
        else:
            i.code.append('WR16(r%d + %s, r%d);' % (rn, hx(off), rd))
        i.note = 'ldst h'
        return
    if hw >> 12 == 9:
        L = (hw >> 11) & 1
        rd = (hw >> 8) & 7
        off = (hw & 0xFF) * 4
        if L:
            i.code.append('r%d = RD32(r13 + %s);' % (rd, hx(off)))
        else:
            i.code.append('WR32(r13 + %s, r%d);' % (hx(off), rd))
        i.note = 'ldst sp'
        return
    if hw >> 12 == 10:
        rd = (hw >> 8) & 7
        off = (hw & 0xFF) * 4
        if (hw >> 11) & 1:
            i.code.append('r%d = r13 + %s;' % (rd, hx(off)))
        else:
            i.code.append('r%d = %s;' % (rd, hx(((i.addr + 4) & ~3) + off)))
        i.note = 'add pc/sp'
        return
    if hw >> 12 == 11:
        sub = (hw >> 8) & 15
        if sub == 0:
            off = (hw & 0x7F) * 4
            i.code.append('r13 = r13 %s %s;' % ('-' if (hw >> 7) & 1 else '+', hx(off)))
            i.note = 'add sp'
            return
        if sub == 2:
            rm, rd = (hw >> 3) & 7, hw & 7
            fn = ['sxth', 'sxtb', 'uxth', 'uxtb'][(hw >> 6) & 3]
            i.code.append('r%d = %s(r%d);' % (rd, fn, rm))
            i.note = fn
            return
        if sub == 0xA:
            rm, rd = (hw >> 3) & 7, hw & 7
            k = (hw >> 6) & 3
            if k == 0:
                i.code.append('r%d = __builtin_bswap32(r%d);' % (rd, rm))
            elif k == 1:
                i.code.append('r%d = rev16(r%d);' % (rd, rm))
            elif k == 3:
                i.code.append('r%d = (u32)(s32)(s16)__builtin_bswap16((u16)r%d);' % (rd, rm))
            else:
                i.kind = UDF
            i.note = 'rev'
            return
        if (hw >> 9) & 3 == 2:
            regs = [k for k in range(8) if (hw >> k) & 1]
            if (hw >> 11) & 1:  # POP
                if (hw >> 8) & 1:
                    regs.append(15)
                L = ['u32 ea = r13;']
                for k, rg in enumerate(regs):
                    L.append(W(i, rg, 'RD32(ea + %d)' % (4 * k)))
                L.append('r13 = ea + %d;' % (4 * len(regs)))
                if 15 in regs:
                    i.ret = True
                i.note = 'pop'
            else:
                if (hw >> 8) & 1:
                    regs.append(14)
                n = len(regs)
                L = ['u32 ea = r13 - %d;' % (4 * n)]
                for k, rg in enumerate(regs):
                    L.append('WR32(ea + %d, r%d);' % (4 * k, rg))
                L.append('r13 = ea;')
                i.note = 'push'
            i.code.append('{ ' + ' '.join(L) + ' }')
            return
        if sub == 0xE:
            i.kind = UDF
            i.note = 'bkpt'
            return
        if (hw & 0xFFE8) == 0xB660 or (hw & 0xFFF7) == 0xB650:
            i.note = 'cps/setend (ignored)'
            return
        i.kind = UDF
        i.note = 'thumb misc %04x' % hw
        return
    if hw >> 12 == 12:
        L = (hw >> 11) & 1
        rn = (hw >> 8) & 7
        regs = [k for k in range(8) if (hw >> k) & 1]
        if not regs:
            i.kind = UDF
            return
        lines = ['u32 ea = r%d;' % rn]
        if L:
            for k, rg in enumerate(regs):
                lines.append('r%d = RD32(ea + %d);' % (rg, 4 * k))
            if rn not in regs:
                lines.append('r%d = ea + %d;' % (rn, 4 * len(regs)))
        else:
            for k, rg in enumerate(regs):
                lines.append('WR32(ea + %d, r%d);' % (4 * k, rg))
            lines.append('r%d = ea + %d;' % (rn, 4 * len(regs)))
        i.code.append('{ ' + ' '.join(lines) + ' }')
        i.note = 'ldmia/stmia'
        return
    if hw >> 12 == 13:
        cond = (hw >> 8) & 15
        if cond == 15:
            i.kind = SVC
            i.imm = hw & 0xFF
            i.note = 'svc'
            return
        if cond == 14:
            i.kind = UDF
            i.note = 'udf'
            return
        i.cond = cond
        i.kind = B
        i.target = (i.addr + 4 + (sx(hw & 0xFF, 8) << 1)) & 0xFFFFFFFF
        i.note = 'b<c> %x' % i.target
        return
    if hw >> 11 == 28:
        i.kind = B
        i.target = (i.addr + 4 + (sx(hw & 0x7FF, 11) << 1)) & 0xFFFFFFFF
        i.note = 'b %x' % i.target
        return
    if hw >> 11 == 30:
        hi = sx(hw & 0x7FF, 11) << 12
        if hw2 >> 11 == 31:
            i.size = 4
            i.pair = True
            i.kind = CALL
            i.target = ((i.addr + 4 + hi + ((hw2 & 0x7FF) << 1)) & 0xFFFFFFFF) | 1
            i.link = i.addr + 4
            i.code.append('r14 = %s;' % hx((i.addr + 4) | 1))
            i.note = 'bl %x' % (i.target & ~1)
            return
        if hw2 >> 11 == 29:
            i.size = 4
            i.pair = True
            i.kind = CALL
            i.target = (i.addr + 4 + hi + ((hw2 & 0x7FF) << 1)) & 0xFFFFFFFC
            i.link = i.addr + 4
            i.code.append('r14 = %s;' % hx((i.addr + 4) | 1))
            i.note = 'blx %x' % i.target
            return
    i.kind = UDF
    i.note = 'thumb %04x' % hw
