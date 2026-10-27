"""Flag liveness: removes NZCV computations whose result is never read.

Assumptions (AAPCS): flags are not live into a call, and a call defines them (the callee's flags are
reloaded after it). They are live out of a function only for "flag-returning" helpers, i.e. functions
whose callers test flags right after the call (RVCT's __aeabi_cfcmple and friends).
"""
import re

from .ins import B, CALL, ICALL, JMP, SVC, UDF

FLAGS = ('nf', 'zf', 'cf', 'vf')
ALL = frozenset(FLAGS)
COND_READS = [
    {'zf'}, {'zf'}, {'cf'}, {'cf'}, {'nf'}, {'nf'}, {'vf'}, {'vf'},
    {'cf', 'zf'}, {'cf', 'zf'}, {'nf', 'vf'}, {'nf', 'vf'},
    {'zf', 'nf', 'vf'}, {'zf', 'nf', 'vf'}, set(), set(),
]
_TOK = re.compile(r'\b(nf|zf|cf|vf)\b')
_ASSIGN = re.compile(r'\b(nf|zf|cf|vf) = ')
_ASSIGN_STMT = {f: re.compile(r'\b%s = [^;]*; ?' % f) for f in FLAGS}

_cache = {}


def rw(i):
    """(reads, writes) of an instruction's code (cached per instruction object)."""
    r = _cache.get(id(i))
    if r is not None and r[0] is i:
        return r[1], r[2]
    txt = ' '.join(i.code)
    writes = set(_ASSIGN.findall(txt))
    reads = set()
    # a token is a read unless it is the left side of an assignment
    for m in _TOK.finditer(txt):
        tail = txt[m.end():m.end() + 3]
        if tail == ' = ':
            continue
        reads.add(m.group(1))
    reads |= COND_READS[i.cond]
    _cache[id(i)] = (i, reads, writes)
    return reads, writes


def reads_flags_first(insts_seq):
    """Does this straight sequence read a flag before writing it?"""
    written = set()
    for i in insts_seq:
        if i is None:
            return False
        r, w = rw(i)
        if r - written:
            return True
        if i.cond == 14:
            written |= w
        if written >= ALL or i.kind != 'op':
            return False
    return False


def liveness(insts, th, start, funcs, live_at_exit, entry_reads=frozenset()):
    """Returns {addr: live-out flag set} for every instruction of a function. entry_reads: functions that read
    flags before setting them (mid-function code taken for an entry): flags are live at a jump into one."""
    succ = {}
    for a, i in insts.items():
        s = []
        exit_ = False
        if i.kind == B:
            if (i.target | th) in funcs and i.target != start:
                exit_ = 'all' if (i.target | th) in entry_reads else True
            else:
                s.append(i.target)
        elif i.kind == JMP:
            if i.jt:
                s += [t & ~1 for t in i.jt if (t & 1) == th and (t & ~1) in insts]
            exit_ = True
        elif i.kind == UDF:
            exit_ = True
        if i.falls_through:
            n = a + i.size
            if n in insts:
                s.append(n)
            elif (n | th) in entry_reads:
                exit_ = 'all'
            elif not exit_:
                exit_ = True
        succ[a] = (s, exit_)
    live_in = {a: set() for a in insts}
    live_out = {a: set() for a in insts}
    exit_live = ALL if live_at_exit else frozenset()
    order = sorted(insts, reverse=True)
    changed = True
    while changed:
        changed = False
        for a in order:
            i = insts[a]
            s, exit_ = succ[a]
            out = set(ALL) if exit_ == 'all' else set(exit_live) if exit_ else set()
            for b in s:
                out |= live_in[b]
            r, w = rw(i)
            if i.kind in (CALL, ICALL):
                # the call defines all flags; nothing is live into it except what the condition reads
                inn = set(COND_READS[i.cond]) if i.cond != 14 else set()
                if i.cond != 14:
                    inn |= out
            else:
                kill = w if i.cond == 14 else set()
                inn = (out - kill) | r
            if out != live_out[a] or inn != live_in[a]:
                live_out[a] = out
                live_in[a] = inn
                changed = True
    return live_out


def strip_dead(i, live_out):
    """The instruction's code with dead flag assignments removed (or None if unchanged)."""
    r, w = rw(i)
    dead = [f for f in w if f not in live_out]
    if not dead or i.kind in (CALL, ICALL):
        return None
    code = []
    for line in i.code:
        for f in dead:
            line = _ASSIGN_STMT[f].sub('', line)
        code.append(line)
    return code
