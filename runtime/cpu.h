// Guest CPU state and the helpers the generated code uses.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef struct CPU {
    u32 r[16];
    u32 nf, zf, cf, vf, qf, ge;
    u32 fpscr;
    u32 pc;     // where a returning / jumping function went
    float s[32];
    u32 tls;    // CP15 thread id register (TLS pointer)
    u32 pad;
    void *thread;
} CPU;

typedef void (*GuestFn)(CPU *, u32);
typedef struct {
    u32 addr;  // bit 0: Thumb
    GuestFn fn;
    u32 entry;
} GuestEntry;

// A CRO module's fixed slot: where ldr:ro puts its code so the recompiled code finds it (tools/recomp/cro.py)
typedef struct {
    const char *name;
    u32 base, size;
} GuestCro;

extern u8 *g_mem;  // 4 GiB host reservation mirroring the guest address space
#define MEMBASE g_mem

typedef u16 __attribute__((aligned(1), may_alias)) u16_u;
typedef u32 __attribute__((aligned(1), may_alias)) u32_u;
typedef u64 __attribute__((aligned(1), may_alias)) u64_u;

#define RD8(a) (*(u8 *)(M + (u32)(a)))
#define RD16(a) (*(u16_u *)(M + (u32)(a)))
#define RD32(a) (*(u32_u *)(M + (u32)(a)))
#define WR8(a, v) (*(u8 *)(M + (u32)(a)) = (u8)(v))
#define WR16(a, v) (*(u16_u *)(M + (u32)(a)) = (u16)(v))
#define WR32(a, v) (*(u32_u *)(M + (u32)(a)) = (u32)(v))

void rt_dispatch(CPU *c, u32 unused);
void rt_svc(CPU *c, u32 imm);
// Set by the host when a thread with a higher priority than the running one became ready (an interrupt woke it):
// generated code checks it on backward branches and calls rt_preempt, the way the 3DS kernel preempts a busy loop.
extern volatile u32 rt_preempt_flag;
void rt_preempt(CPU *c);
void rt_undefined(CPU *c, u32 addr, u32 word);

static inline u32 F2U(float f) { u32 u; memcpy(&u, &f, 4); return u; }
static inline float U2F(u32 u) { float f; memcpy(&f, &u, 4); return f; }
static inline double D2F(float lo, float hi) {
    u64 b = (u64)F2U(lo) | ((u64)F2U(hi) << 32);
    double d;
    memcpy(&d, &b, 8);
    return d;
}
#define DSET(lo, hi, val) do { double _d = (val); u64 _b; memcpy(&_b, &_d, 8); lo = U2F((u32)_b); hi = U2F((u32)(_b >> 32)); } while (0)

static inline u32 ror32(u32 x, u32 n) { n &= 31; return n ? (x >> n) | (x << (32 - n)) : x; }

// register-specified shifts, amount = Rs[7:0]
static inline u32 lsl_r(u32 x, u32 a) { return a >= 32 ? 0 : x << a; }
static inline u32 lsr_r(u32 x, u32 a) { return a >= 32 ? 0 : x >> a; }
static inline u32 asr_r(u32 x, u32 a) { return a >= 32 ? (u32)((s32)x >> 31) : (u32)((s32)x >> a); }
static inline u32 ror_r(u32 x, u32 a) { return ror32(x, a & 31); }
static inline u32 lsl_rc(u32 x, u32 a, u32 *c) {
    if (a == 0) return x;
    if (a < 32) { *c = (x >> (32 - a)) & 1; return x << a; }
    *c = a == 32 ? (x & 1) : 0;
    return 0;
}
static inline u32 lsr_rc(u32 x, u32 a, u32 *c) {
    if (a == 0) return x;
    if (a < 32) { *c = (x >> (a - 1)) & 1; return x >> a; }
    *c = a == 32 ? (x >> 31) : 0;
    return 0;
}
static inline u32 asr_rc(u32 x, u32 a, u32 *c) {
    if (a == 0) return x;
    if (a < 32) { *c = (x >> (a - 1)) & 1; return (u32)((s32)x >> a); }
    *c = x >> 31;
    return (u32)((s32)x >> 31);
}
static inline u32 ror_rc(u32 x, u32 a, u32 *c) {
    if (a == 0) return x;
    a &= 31;
    if (a == 0) { *c = x >> 31; return x; }
    *c = (x >> (a - 1)) & 1;
    return ror32(x, a);
}

// saturating arithmetic
static inline u32 sat32(s64 v, u32 *q) {
    if (v > 0x7fffffffLL) { *q = 1; return 0x7fffffffu; }
    if (v < -0x80000000LL) { *q = 1; return 0x80000000u; }
    return (u32)(s32)v;
}
static inline u32 qadd(u32 m, u32 n, u32 *q) { return sat32((s64)(s32)m + (s32)n, q); }
static inline u32 qsub(u32 m, u32 n, u32 *q) { return sat32((s64)(s32)m - (s32)n, q); }
static inline u32 qdadd(u32 m, u32 n, u32 *q) { u32 d = sat32((s64)(s32)n * 2, q); return sat32((s64)(s32)m + (s32)d, q); }
static inline u32 qdsub(u32 m, u32 n, u32 *q) { u32 d = sat32((s64)(s32)n * 2, q); return sat32((s64)(s32)m - (s32)d, q); }
static inline u32 smlaxy(s32 a, s32 b, u32 acc, u32 *q) {
    s64 r = (s64)(a * b) + (s32)acc;
    if (r != (s32)r) *q = 1;
    return (u32)r;
}
static inline u32 smlaxy_add(u32 p, u32 acc, u32 *q) {
    s64 r = (s64)(s32)p + (s32)acc;
    if (r != (s32)r) *q = 1;
    return (u32)r;
}
static inline u32 ssat(u32 v, int bits, u32 *q) {
    s32 x = (s32)v, hi = (1 << (bits - 1)) - 1, lo = -(1 << (bits - 1));
    if (bits == 32) return v;
    if (x > hi) { *q = 1; return (u32)hi; }
    if (x < lo) { *q = 1; return (u32)lo; }
    return v;
}
static inline u32 usat(u32 v, int bits, u32 *q) {
    s32 x = (s32)v;
    s64 hi = ((s64)1 << bits) - 1;
    if (x < 0) { *q = 1; return 0; }
    if (x > hi) { *q = 1; return (u32)hi; }
    return v;
}
static inline u32 ssat16(u32 v, int bits, u32 *q) {
    u32 lo = ssat((u32)(s32)(s16)v, bits, q) & 0xffff, hi = ssat((u32)(s32)(s16)(v >> 16), bits, q) & 0xffff;
    return lo | (hi << 16);
}
static inline u32 usat16(u32 v, int bits, u32 *q) {
    u32 lo = usat((u32)(s32)(s16)v, bits, q) & 0xffff, hi = usat((u32)(s32)(s16)(v >> 16), bits, q) & 0xffff;
    return lo | (hi << 16);
}

// extends
static inline u32 sxtb(u32 v) { return (u32)(s32)(s8)v; }
static inline u32 sxth(u32 v) { return (u32)(s32)(s16)v; }
static inline u32 uxtb(u32 v) { return v & 0xff; }
static inline u32 uxth(u32 v) { return v & 0xffff; }
static inline u32 sxtb16(u32 v) { return ((u32)(s32)(s8)v & 0xffff) | ((u32)(s32)(s8)(v >> 16) << 16); }
static inline u32 uxtb16(u32 v) { return v & 0x00ff00ffu; }
static inline u32 add16x(u32 a, u32 b) { return ((a + b) & 0xffff) | (((a >> 16) + (b >> 16)) << 16); }
static inline u32 rev16(u32 v) { return ((v & 0x00ff00ffu) << 8) | ((v >> 8) & 0x00ff00ffu); }
static inline u32 sel(u32 n, u32 m, u32 ge) {
    u32 r = 0;
    for (int i = 0; i < 4; i++) r |= ((ge >> i) & 1 ? n : m) & (0xffu << (8 * i));
    return r;
}
static inline u32 usad8(u32 a, u32 b) {
    u32 s = 0;
    for (int i = 0; i < 4; i++) {
        int x = (a >> (8 * i)) & 0xff, y = (b >> (8 * i)) & 0xff;
        s += x > y ? x - y : y - x;
    }
    return s;
}

// parallel add/subtract: kind 0 = signed (sets GE), 1 = signed saturating, 2 = signed halving,
// 3 = unsigned (sets GE), 4 = unsigned saturating, 5 = unsigned halving
static inline s32 pas_lane(int kind, s32 r, int bits, int lane_ge_add, u32 *ge, int lane, int gebits) {
    s32 lo, hi;
    switch (kind) {
    case 0:
        if (r >= 0) *ge |= ((1u << gebits) - 1) << (lane * gebits);
        return r;
    case 1: hi = (1 << (bits - 1)) - 1; lo = -(1 << (bits - 1)); return r > hi ? hi : r < lo ? lo : r;
    case 2: return r >> 1;
    case 3:
        if (lane_ge_add ? (r >= (1 << bits)) : (r >= 0)) *ge |= ((1u << gebits) - 1) << (lane * gebits);
        return r;
    case 4: hi = (1 << bits) - 1; return r > hi ? hi : r < 0 ? 0 : r;
    default: return r >> 1;
    }
}
static inline u32 pas16(int kind, int op, u32 a, u32 b, u32 *ge) {
    // op: 0 add16, 1 asx, 2 sax, 3 sub16
    int sgn = kind < 3;
    s32 a0 = sgn ? (s16)a : (u16)a, a1 = sgn ? (s16)(a >> 16) : (u16)(a >> 16);
    s32 b0 = sgn ? (s16)b : (u16)b, b1 = sgn ? (s16)(b >> 16) : (u16)(b >> 16);
    s32 r0, r1;
    int add0, add1;
    switch (op) {
    case 0: r0 = a0 + b0; r1 = a1 + b1; add0 = add1 = 1; break;
    case 1: r0 = a0 - b1; r1 = a1 + b0; add0 = 0; add1 = 1; break;
    case 2: r0 = a0 + b1; r1 = a1 - b0; add0 = 1; add1 = 0; break;
    default: r0 = a0 - b0; r1 = a1 - b1; add0 = add1 = 0; break;
    }
    if (kind == 0 || kind == 3) *ge = 0;
    r0 = pas_lane(kind, r0, 16, add0, ge, 0, 2);
    r1 = pas_lane(kind, r1, 16, add1, ge, 1, 2);
    return ((u32)r0 & 0xffff) | ((u32)r1 << 16);
}
static inline u32 pas8(int kind, int sub, u32 a, u32 b, u32 *ge) {
    int sgn = kind < 3;
    u32 out = 0;
    if (kind == 0 || kind == 3) *ge = 0;
    for (int i = 0; i < 4; i++) {
        s32 x = sgn ? (s8)(a >> (8 * i)) : (u8)(a >> (8 * i));
        s32 y = sgn ? (s8)(b >> (8 * i)) : (u8)(b >> (8 * i));
        s32 r = sub ? x - y : x + y;
        r = pas_lane(kind, r, 8, !sub, ge, i, 1);
        out |= ((u32)r & 0xff) << (8 * i);
    }
    return out;
}
#define PAS_DEF(pre, kind) \
    static inline u32 pas_##pre##add16(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas16(kind, 0, a, b, ge); } \
    static inline u32 pas_##pre##asx(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas16(kind, 1, a, b, ge); } \
    static inline u32 pas_##pre##sax(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas16(kind, 2, a, b, ge); } \
    static inline u32 pas_##pre##sub16(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas16(kind, 3, a, b, ge); } \
    static inline u32 pas_##pre##add8(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas8(kind, 0, a, b, ge); } \
    static inline u32 pas_##pre##sub8(u32 a, u32 b, u32 *ge, u32 *q) { (void)q; return pas8(kind, 1, a, b, ge); }
PAS_DEF(s, 0)
PAS_DEF(q, 1)
PAS_DEF(sh, 2)
PAS_DEF(u, 3)
PAS_DEF(uq, 4)
PAS_DEF(uh, 5)

// VFP
static inline u32 vfp_cmpf(float a, float b) {
    if (a == b) return 0x60000000u;
    if (a < b) return 0x80000000u;
    if (a > b) return 0x20000000u;
    return 0x30000000u;
}
static inline u32 vfp_cmpd(double a, double b) {
    if (a == b) return 0x60000000u;
    if (a < b) return 0x80000000u;
    if (a > b) return 0x20000000u;
    return 0x30000000u;
}
static inline u32 cvt_s32(double f) {
    if (f != f) return 0;
    if (f >= 2147483648.0) return 0x7fffffffu;
    if (f <= -2147483649.0) return 0x80000000u;
    return (u32)(s32)f;
}
static inline u32 cvt_u32(double f) {
    if (f != f) return 0;
    if (f >= 4294967296.0) return 0xffffffffu;
    if (f <= -1.0) return 0;
    return (u32)f;
}
static inline double vfp_round(double f, u32 fpscr) {
    switch ((fpscr >> 22) & 3) {
    case 0: return rint(f);
    case 1: return ceil(f);
    case 2: return floor(f);
    default: return trunc(f);
    }
}
static inline u32 vcvt_sz_f(float f) { return cvt_s32(f); }
static inline u32 vcvt_uz_f(float f) { return cvt_u32(f); }
static inline u32 vcvt_sz_d(double f) { return cvt_s32(f); }
static inline u32 vcvt_uz_d(double f) { return cvt_u32(f); }
static inline u32 vcvt_sr_f(float f, u32 fpscr) { return cvt_s32(vfp_round(f, fpscr)); }
static inline u32 vcvt_ur_f(float f, u32 fpscr) { return cvt_u32(vfp_round(f, fpscr)); }
static inline u32 vcvt_sr_d(double f, u32 fpscr) { return cvt_s32(vfp_round(f, fpscr)); }
static inline u32 vcvt_ur_d(double f, u32 fpscr) { return cvt_u32(vfp_round(f, fpscr)); }

#ifdef __cplusplus
}
#endif
