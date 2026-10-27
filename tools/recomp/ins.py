"""Decoded instruction: the C it lowers to, plus what the analyser needs (control flow kind, targets)."""
import re

COND_C = ['zf', '!zf', 'cf', '!cf', 'nf', '!nf', 'vf', '!vf',
          '(cf&&!zf)', '(!cf||zf)', '(nf==vf)', '(nf!=vf)',
          '(!zf&&nf==vf)', '(zf||nf!=vf)', None, None]

# control-flow kinds
OP = 'op'          # straight-line code
B = 'b'            # direct jump to .target
CALL = 'call'      # direct call to .target (bit 0 = Thumb), returns to .link
ICALL = 'icall'    # indirect call: .code leaves the target in tpc, returns to .link
JMP = 'jmp'        # indirect jump: .code leaves the target in tpc (.ret: it is a function return)
SVC = 'svc'        # supervisor call .imm
UDF = 'udf'        # undefined / unsupported instruction

_REG_RE = re.compile(r'\br(\d+)\b')
_VREG_RE = re.compile(r'\bS(\d+)\b')


class Ins:
    __slots__ = ('addr', 'size', 'thumb', 'word', 'cond', 'kind', 'code', 'target', 'link', 'ret',
                 'imm', 'jt', 'reads_pc_lit', 'note', 'setflags', 'readflags', 'pair')

    def __init__(self, addr, size, thumb, word):
        self.addr = addr
        self.size = size
        self.thumb = thumb
        self.word = word
        self.cond = 14
        self.kind = OP
        self.code = []
        self.target = None
        self.link = None
        self.ret = False
        self.imm = None
        self.jt = None          # jump table: list of target addresses (filled by the analyser)
        self.reads_pc_lit = None  # (addr) of a literal this instruction loads (LDR rX, [pc, #imm])
        self.note = ''
        self.setflags = False
        self.readflags = False
        self.pair = False       # Thumb BL/BLX pair: the instruction is 4 bytes

    @property
    def cond_c(self):
        return COND_C[self.cond]

    @property
    def falls_through(self):
        """Can execution continue at addr+size?"""
        if self.cond != 14:
            return True
        return self.kind in (OP, CALL, ICALL, SVC)

    def regs(self):
        s = set()
        for line in self.code:
            for m in _REG_RE.findall(line):
                s.add(int(m))
        return s

    def vregs(self):
        s = set()
        for line in self.code:
            for m in _VREG_RE.findall(line):
                s.add(int(m))
        return s

    def __repr__(self):
        return '<%08x %s %s %s>' % (self.addr, self.kind, self.note, ' '.join(self.code))
