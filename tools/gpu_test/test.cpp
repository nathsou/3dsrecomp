// Unit tests for the software PICA200.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gpu/pica.h"

using namespace gpu;

static int failures = 0, checks = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        checks++;                                                           \
        if (!(cond)) {                                                      \
            failures++;                                                     \
            printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);          \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
        }                                                                   \
    } while (0)

static bool feq(float a, float b, float eps = 1e-5f) { return fabsf(a - b) <= eps * (1 + fabsf(b)); }

// ---------------------------------------------------------------- float24 helpers
static u32 to_f24(float f) {
    if (f == 0) return std::signbit(f) ? 0x800000 : 0;
    u32 u;
    memcpy(&u, &f, 4);
    u32 sign = u >> 31, exp = (u >> 23) & 0xff, mant = u & 0x7fffff;
    return (sign << 23) | ((exp - 127 + 63) << 16) | (mant >> 7);
}
static void pack_f24(const float v[4], u32 out[3]) {
    u32 x = to_f24(v[0]), y = to_f24(v[1]), z = to_f24(v[2]), w = to_f24(v[3]);
    out[0] = (w << 8) | (z >> 16);
    out[1] = ((z & 0xffff) << 16) | (y >> 8);
    out[2] = ((y & 0xff) << 24) | x;
}

static void test_float_formats() {
    CHECK(f24(to_f24(1.0f)) == 1.0f, "f24 1.0");
    CHECK(f24(to_f24(-2.5f)) == -2.5f, "f24 -2.5");
    CHECK(f24(to_f24(0.125f)) == 0.125f, "f24 0.125");
    CHECK(f16(0x3c00) == 1.0f, "f16 1.0");
    CHECK(f16(0xc000) == -2.0f, "f16 -2");
    CHECK(f20(0x3f000) == 1.0f, "f20 1.0 got %f", f20(0x3f000));
}

// ---------------------------------------------------------------- morton & textures
static u32 ref_morton(u32 x, u32 y) {
    u32 i = 0;
    for (int b = 0; b < 3; b++) i |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1);
    return i;
}

static void test_morton_and_decode() {
    for (u32 y = 0; y < 8; y++)
        for (u32 x = 0; x < 8; x++) CHECK(morton8(x, y) == ref_morton(x, y), "morton %u %u", x, y);
    CHECK(tiled_offset(9, 0, 16, 4) == 64 * 4 + 4, "tiled offset");
    CHECK(tiled_offset(0, 8, 16, 1) == 128, "tiled offset row");

    // build a 16x16 RGBA8 tiled texture from a known image (memory rows top-down)
    const u32 W = 16, H = 16;
    std::vector<u8> tex(W * H * 4);
    auto color_at = [](u32 x, u32 r) { return rgba(x * 13, r * 7, (x ^ r) * 5, 200 + x); };
    for (u32 r = 0; r < H; r++)
        for (u32 x = 0; x < W; x++) {
            u32 c = color_at(x, r);
            u8 *p = tex.data() + ((r / 8) * (W / 8) + x / 8) * 256 + ref_morton(x % 8, r % 8) * 4;
            p[0] = c >> 24;
            p[1] = c >> 16;
            p[2] = c >> 8;
            p[3] = c;
        }
    std::vector<u32> out(W * H);
    texture_decode(tex.data(), W, H, 0, out.data());
    bool ok = true;
    for (u32 r = 0; r < H; r++)
        for (u32 x = 0; x < W; x++)
            if (out[(H - 1 - r) * W + x] != color_at(x, r)) ok = false;
    CHECK(ok, "RGBA8 decode / morton / bottom-up rows");

    // small formats on one 8x8 tile
    u8 t8[64];
    for (u32 i = 0; i < 64; i++) t8[i] = (u8)(i * 4 + 1);
    u32 o8[64];
    texture_decode(t8, 8, 8, 7, o8);  // I8
    CHECK(o8[7 * 8 + 0] == rgba(1, 1, 1, 255), "I8 texel (0,top)");
    CHECK(o8[7 * 8 + 1] == rgba(5, 5, 5, 255), "I8 texel (1,top)");
    texture_decode(t8, 8, 8, 8, o8);  // A8
    CHECK(o8[7 * 8 + 2] == rgba(0, 0, 0, 17), "A8 texel (2,top) = morton 4");
    texture_decode(t8, 8, 8, 9, o8);  // IA4: byte 1 = 0x01 -> i=0, a=0x11
    CHECK(o8[7 * 8 + 0] == rgba(0, 0, 0, 0x11), "IA4");
    u8 t4[32];
    for (u32 i = 0; i < 32; i++) t4[i] = (u8)(0x10 * (i & 15) + 0x3);
    texture_decode(t4, 8, 8, 10, o8);  // I4: texel 0 = low nibble of byte 0, texel 1 = high nibble
    CHECK(o8[7 * 8 + 0] == rgba(0x33, 0x33, 0x33, 255), "I4 low nibble");
    CHECK(o8[7 * 8 + 1] == rgba(0, 0, 0, 255), "I4 high nibble");
    u8 t16[128];
    for (u32 i = 0; i < 64; i++) {
        t16[2 * i] = 0x34;
        t16[2 * i + 1] = 0x12;
    }
    texture_decode(t16, 8, 8, 4, o8);  // RGBA4 0x1234
    CHECK(o8[0] == rgba(0x11, 0x22, 0x33, 0x44), "RGBA4");
    texture_decode(t16, 8, 8, 5, o8);  // IA8: i = byte1, a = byte0
    CHECK(o8[0] == rgba(0x12, 0x12, 0x12, 0x34), "IA8");
    texture_decode(t16, 8, 8, 6, o8);  // RG8: r = byte1, g = byte0
    CHECK(o8[0] == rgba(0x12, 0x34, 0, 255), "RG8");
    texture_decode(t16, 8, 8, 3, o8);  // RGB565 0x1234: r=2, g=0x11, b=0x14
    CHECK(o8[0] == rgba((2 << 3) | 0, (0x11 << 2) | (0x11 >> 4), (0x14 << 3) | (0x14 >> 2), 255),
          "RGB565 %08x", o8[0]);
}

static void test_etc1() {
    // individual mode, no flip: left half base (15,0,0), right half (0,15,0); table1 = 0, table2 = 7
    u64 blk = 0;
    blk |= (u64)0xF << 60;  // r1
    blk |= (u64)0x0 << 56;  // r2
    blk |= (u64)0x0 << 52;  // g1
    blk |= (u64)0xF << 48;  // g2
    blk |= (u64)0 << 37;    // table1
    blk |= (u64)7 << 34;    // table2
    // texel index = 4*x + y. texel (1,2)=6: negative; texel (3,0)=12: large modifier
    blk |= (u64)1 << (16 + 6);
    blk |= (u64)1 << 12;
    CHECK(decode_etc1_texel(blk, 0, 0) == rgba(255, 2, 2, 255), "etc1 left %08x",
          decode_etc1_texel(blk, 0, 0));
    CHECK(decode_etc1_texel(blk, 1, 2) == rgba(253, 0, 0, 255), "etc1 neg");
    CHECK(decode_etc1_texel(blk, 2, 1) == rgba(47, 255, 47, 255), "etc1 right");
    CHECK(decode_etc1_texel(blk, 3, 0) == rgba(183, 255, 183, 255), "etc1 large");
    // differential mode: base (16,8,4) + delta (-1,+3,0) for the right half
    u64 d = (u64)1 << 33;
    d |= (u64)16 << 59 | (u64)7 << 56;  // r, dr=-1
    d |= (u64)8 << 51 | (u64)3 << 48;   // g, dg=+3
    d |= (u64)4 << 43;                  // b
    u32 l = decode_etc1_texel(d, 0, 0), r = decode_etc1_texel(d, 3, 3);
    CHECK(l == rgba(132 + 2, 66 + 2, 33 + 2, 255), "etc1 diff left %08x", l);
    CHECK(r == rgba(123 + 2, 90 + 2, 33 + 2, 255), "etc1 diff right %08x", r);
    // ETC1A4 tile: alpha nibble for texel (x,y) at 4*(x*4+y)
    u8 tile[64] = {};
    u64 alpha = (u64)0xA << (4 * (1 * 4 + 0));  // texel x=1,y=0 of subtile 0
    memcpy(tile, &alpha, 8);
    memcpy(tile + 8, &blk, 8);
    u32 out[64];
    texture_decode(tile, 8, 8, 13, out);
    // memory row 0 -> output row 7
    CHECK(out[7 * 8 + 1] == ((decode_etc1_texel(blk, 1, 0) & 0xffffff) | 0xAA000000u), "etc1a4");
    CHECK((out[7 * 8 + 0] >> 24) == 0, "etc1a4 alpha 0");
}

// ---------------------------------------------------------------- shader assembler
static u32 swz(const char *mask, const char *s1 = "xyzw", bool n1 = false, const char *s2 = "xyzw",
               bool n2 = false, const char *s3 = "xyzw", bool n3 = false) {
    auto sel = [](const char *s, int base) {
        u32 v = 0;
        for (int i = 0; i < 4; i++) {
            u32 c = s[i] == 'x' ? 0 : s[i] == 'y' ? 1 : s[i] == 'z' ? 2 : 3;
            v |= c << (base - 2 * i);
        }
        return v;
    };
    u32 m = 0;
    for (const char *p = mask; *p; p++) m |= *p == 'x' ? 8 : *p == 'y' ? 4 : *p == 'z' ? 2 : 1;
    return m | (n1 << 4) | sel(s1, 11) | (n2 << 13) | sel(s2, 20) | (n3 << 22) | sel(s3, 29);
}
// registers: v(n)=n, r(n)=0x10+n, c(n)=0x20+n ; dst o(n)=n, r(n)=0x10+n
static u32 V(u32 n) { return n; }
static u32 Rg(u32 n) { return 0x10 + n; }
static u32 C(u32 n) { return 0x20 + n; }
static u32 O(u32 n) { return n; }
static u32 arith(u32 op, u32 dst, u32 s1, u32 s2, u32 desc, u32 idx = 0) {
    return op << 26 | dst << 21 | idx << 19 | s1 << 12 | s2 << 7 | desc;
}
static u32 arith_i(u32 op, u32 dst, u32 s1, u32 s2, u32 desc, u32 idx = 0) {
    return op << 26 | dst << 21 | idx << 19 | s1 << 14 | s2 << 7 | desc;
}
static u32 mad(u32 dst, u32 s1, u32 s2, u32 s3, u32 desc, u32 idx = 0) {
    return 0x38u << 26 | dst << 24 | idx << 22 | s1 << 17 | s2 << 10 | s3 << 5 | desc;
}
static u32 cmp(u32 s1, u32 s2, u32 cx, u32 cy, u32 desc) {
    return 0x17u << 27 | cx << 24 | cy << 21 | s1 << 12 | s2 << 7 | desc;
}
static u32 flow(u32 op, u32 dst, u32 num, u32 condop = 0, u32 refx = 0, u32 refy = 0) {
    return op << 26 | refx << 25 | refy << 24 | condop << 22 | dst << 10 | num;
}
static u32 flowu(u32 op, u32 dst, u32 num, u32 id) { return op << 26 | id << 22 | dst << 10 | num; }
static const u32 ENDI = 0x88000000u;
enum { ADD = 0, DP3 = 1, DP4 = 2, DPH = 3, EX2 = 5, LG2 = 6, MUL = 8, SGE = 9, SLT = 0xa, FLR = 0xb,
       MAX = 0xc, MIN = 0xd, RCP = 0xe, RSQ = 0xf, MOVA = 0x12, MOV = 0x13, SGEI = 0x1a,
       BREAK = 0x20, NOP = 0x21, END = 0x22, BREAKC = 0x23, CALL = 0x24, CALLC = 0x25, CALLU = 0x26,
       IFU = 0x27, IFC = 0x28, LOOP = 0x29, JMPC = 0x2c, JMPU = 0x2d };

struct ShTest {
    ShaderSetup s;
    ShaderUnit u;
    std::vector<u32> code;
    std::vector<u32> sw;
    ShTest() { memset(&s, 0, sizeof(s)); }
    u32 desc(u32 pat) {
        sw.push_back(pat);
        return (u32)sw.size() - 1;
    }
    void run(const V4 *in, u32 nin) {
        std::vector<u32> swz_full(4096, 0);
        for (size_t i = 0; i < sw.size(); i++) swz_full[i] = sw[i];
        ShaderProgram *p = shader_decode(code.data(), (u32)code.size(), swz_full.data());
        shader_unit_init(u, s);
        for (u32 i = 0; i < nin; i++) u.r[REG_IN + i] = in[i];
        shader_run(p, u, s, 0);
        shader_free(p);
    }
    const float *o(u32 n) { return u.r[REG_OUT + n].v; }
};

static void test_shader() {
    {
        ShTest t;
        u32 d0 = t.desc(swz("xz", "wzyx", true));
        u32 dall = t.desc(swz("xyzw"));
        t.code = {arith(MOV, O(0), V(0), 0, dall), arith(MOV, O(0), V(1), 0, d0), ENDI};
        V4 in[2] = {{{9, 9, 9, 9}}, {{1, 2, 3, 4}}};
        t.run(in, 2);
        CHECK(t.o(0)[0] == -4 && t.o(0)[1] == 9 && t.o(0)[2] == -2 && t.o(0)[3] == 9,
              "swizzle/negate/mask %f %f %f %f", t.o(0)[0], t.o(0)[1], t.o(0)[2], t.o(0)[3]);
    }
    {
        ShTest t;
        u32 d = t.desc(swz("xyzw"));
        t.s.f[0] = V4{{1, 2, 3, 4}};
        t.s.f[1] = V4{{0.5f, -1, 2, 0}};
        t.code = {arith(ADD, O(0), C(0), V(0), d),  arith(MUL, O(1), C(0), V(0), d),
                  arith(DP3, O(2), C(0), V(0), d),  arith(DP4, O(3), C(0), V(0), d),
                  arith(DPH, O(4), C(0), V(0), d),  mad(O(5), V(0), C(0), V(1), d),
                  arith(MAX, O(6), C(1), V(0), d),  arith(MIN, O(7), C(1), V(0), d),
                  arith(SGE, O(8), C(1), V(0), d),  arith(SLT, O(9), C(1), V(0), d),
                  arith(RCP, O(10), C(1), V(0), d), arith(RSQ, O(11), C(0), V(0), d),
                  arith(EX2, O(12), C(0), V(0), d), arith(LG2, O(13), C(0), V(0), d),
                  arith(FLR, O(14), C(1), V(0), d), arith(MUL, O(15), V(2), V(3), d),
                  ENDI};
        float inf = INFINITY;
        V4 in[4] = {{{2, 2, 2, 2}}, {{1, 1, 1, 1}}, {{0, inf, 2, 0}}, {{inf, 0, 3, 5}}};
        t.run(in, 4);
        CHECK(t.o(0)[0] == 3 && t.o(0)[3] == 6, "ADD");
        CHECK(t.o(1)[1] == 4 && t.o(1)[3] == 8, "MUL");
        CHECK(t.o(2)[0] == 12 && t.o(2)[3] == 12, "DP3");
        CHECK(t.o(3)[0] == 20, "DP4");
        CHECK(t.o(4)[0] == 14, "DPH %f", t.o(4)[0]);  // 2+4+6+1*2
        CHECK(t.o(5)[0] == 3 && t.o(5)[3] == 9, "MAD");
        CHECK(t.o(6)[0] == 2 && t.o(6)[2] == 2 && t.o(7)[0] == 0.5f && t.o(7)[1] == -1, "MAX/MIN");
        CHECK(t.o(8)[0] == 0 && t.o(8)[2] == 1 && t.o(9)[0] == 1 && t.o(9)[2] == 0, "SGE/SLT");
        CHECK(t.o(10)[0] == 2 && t.o(10)[3] == 2, "RCP");
        CHECK(t.o(11)[0] == 1, "RSQ");
        CHECK(t.o(12)[0] == 2, "EX2");
        CHECK(t.o(13)[0] == 0, "LG2");
        CHECK(t.o(14)[1] == -1 && t.o(14)[0] == 0, "FLR");
        CHECK(t.o(15)[0] == 0 && t.o(15)[1] == 0 && t.o(15)[2] == 6, "0*inf=0");
    }
    {
        // relative addressing via MOVA, inverted SGEI
        ShTest t;
        u32 d = t.desc(swz("xyzw"));
        u32 dx = t.desc(swz("x"));
        for (int i = 0; i < 8; i++) t.s.f[i] = V4{{(float)i, 0, 0, 0}};
        t.code = {arith(MOVA, 0, V(0), 0, dx), arith(MOV, O(0), C(1), 0, d, 1),
                  arith_i(SGEI, O(1), V(0), C(3), d), ENDI};
        V4 in[1] = {{{3, 0, 0, 0}}};
        t.run(in, 1);
        CHECK(t.o(0)[0] == 4, "c[1+a0.x] = %f", t.o(0)[0]);
        CHECK(t.o(1)[0] == 1, "SGEI");
    }
    {
        // LOOP summing c[aL], int uniform: count 2 (3 iterations), start 1, step 2 -> c1+c3+c5
        ShTest t;
        u32 d = t.desc(swz("xyzw"));
        for (int i = 0; i < 8; i++) t.s.f[i] = V4{{(float)(1 << i), 0, 0, 0}};
        t.s.i[1][0] = 2;
        t.s.i[1][1] = 1;
        t.s.i[1][2] = 2;
        t.code = {arith(MOV, Rg(0), V(0), 0, d), flowu(LOOP, 2, 0, 1),
                  arith(ADD, Rg(0), Rg(0), 0, d) | 0,  // placeholder replaced below
                  arith(MOV, O(0), Rg(0), 0, d), ENDI};
        // ADD r0, c0[aL], r0 (src1 relative)
        t.code[2] = arith(ADD, Rg(0), C(0), Rg(0), d, 3);
        V4 in[1] = {{{0, 0, 0, 0}}};
        t.run(in, 1);
        CHECK(t.o(0)[0] == 2 + 8 + 32, "LOOP sum %f", t.o(0)[0]);
    }
    {
        // IFU with else, CALL, CALLU, JMPU
        ShTest t;
        u32 d = t.desc(swz("xyzw"));
        t.s.f[0] = V4{{1, 0, 0, 0}};
        t.s.f[1] = V4{{2, 0, 0, 0}};
        t.s.b[0] = true;
        t.s.b[1] = false;
        t.code = {
            flowu(IFU, 3, 2, 0),             // 0: if b0 { 1..2 } else { 3..4 }
            arith(MOV, O(0), C(0), 0, d),    // 1
            arith(MOV, O(1), C(0), 0, d),    // 2
            arith(MOV, O(0), C(1), 0, d),    // 3 (else)
            arith(MOV, O(1), C(1), 0, d),    // 4
            flowu(IFU, 7, 1, 1),             // 5: if b1 { 6 } else { 7 }
            arith(MOV, O(2), C(0), 0, d),    // 6
            arith(MOV, O(2), C(1), 0, d),    // 7
            flow(CALL, 12, 1),               // 8: call 12
            flowu(CALLU, 13, 1, 1),          // 9: callu b1 -> not taken
            flowu(JMPU, 14, 0, 0),           // 10: jmpu b0 -> 14
            arith(MOV, O(5), C(0), 0, d),    // 11 (skipped)
            arith(MOV, O(3), C(1), 0, d),    // 12: sub
            arith(MOV, O(4), C(1), 0, d),    // 13: sub2
            ENDI,                       // 14
        };
        t.u.r[REG_OUT + 5].v[0] = 0;
        t.run(nullptr, 0);
        CHECK(t.o(0)[0] == 1 && t.o(1)[0] == 1, "IFU true branch %f %f", t.o(0)[0], t.o(1)[0]);
        CHECK(t.o(2)[0] == 2, "IFU false -> else %f", t.o(2)[0]);
        CHECK(t.o(3)[0] == 2, "CALL");
        CHECK(t.o(4)[0] == 0, "CALLU not taken / JMPU %f", t.o(4)[0]);
        CHECK(t.o(5)[0] == 0, "JMPU skipped");
    }
    {
        // CMP + IFC + BREAKC in a loop: count iterations until r0.x >= 5
        ShTest t;
        u32 d = t.desc(swz("xyzw"));
        t.s.f[0] = V4{{1, 1, 1, 1}};
        t.s.f[1] = V4{{5, 5, 5, 5}};
        t.s.i[0][0] = 100;
        t.code = {
            arith(MOV, Rg(0), V(0), 0, d),      // 0 r0 = 0
            flowu(LOOP, 4, 0, 0),               // 1 loop to 4
            arith(ADD, Rg(0), C(0), Rg(0), d),  // 2 r0 += 1
            cmp(C(1), Rg(0), 3, 3, d),          // 3 cond = 5 <= r0
            flow(BREAKC, 0, 0, 2, 1, 0),        // 4 breakc cc.x
            arith(MOV, O(0), Rg(0), 0, d),      // 5
            cmp(C(1), Rg(0), 0, 1, d),          // 6 x: ==, y: !=
            flow(IFC, 9, 1, 1, 1, 0),           // 7 if (cc.x && !cc.y)
            arith(MOV, O(1), C(1), 0, d),       // 8
            flow(JMPC, 11, 0, 3, 0, 0),         // 9 (else) jmpc !cc.y
            arith(MOV, O(2), C(1), 0, d),       // 10
            ENDI,                          // 11
        };
        V4 in[1] = {{{0, 0, 0, 0}}};
        t.run(in, 1);
        CHECK(t.o(0)[0] == 5, "BREAKC loop %f", t.o(0)[0]);
        CHECK(t.o(1)[0] == 5, "IFC");
        CHECK(t.o(2)[0] == 5, "else block (JMPC) skipped, code after IF runs");
    }
}

// ---------------------------------------------------------------- command list builder
struct Cmd {
    std::vector<u32> w;
    void reg(u32 id, u32 v, u32 mask = 0xf) {
        w.push_back(v);
        w.push_back(id | mask << 16);
    }
    void seq(u32 id, const std::vector<u32> &vals, bool group) {
        w.push_back(vals[0]);
        w.push_back(id | 0xfu << 16 | (u32)(vals.size() - 1) << 20 | (group ? 1u << 31 : 0));
        for (size_t i = 1; i < vals.size(); i++) w.push_back(vals[i]);
        if (w.size() & 1) w.push_back(0);
    }
    void run(u32 paddr = 0x20100000) {
        if (w.size() & 1) w.push_back(0);
        memcpy(mem::paddr_ptr(paddr), w.data(), w.size() * 4);
        process_command_list(paddr, (u32)w.size() * 4);
        w.clear();
    }
};

static const u32 FB = 0x18000000, FBW = 32, FBH = 32;

static u32 fb_pixel(u32 x, u32 y) {  // y up (PICA screen coords)
    const u8 *p = mem::paddr_ptr(FB) + tiled_offset(x, FBH - 1 - y, FBW, 4);
    return rgba(p[3], p[2], p[1], p[0]);
}

// pipeline: VS passes v0 -> position, v1 -> color; TEV outputs a constant; additive blending
static void setup_counting_pipeline(Cmd &c) {
    const u32 vs[3] = {0x13u << 26 | 0 << 21 | 0 << 12 | 0, 0x13u << 26 | 1 << 21 | 1 << 12 | 0,
                       0x88000000u};
    c.reg(R::VS + R::SH_CODE_IDX, 0);
    c.seq(R::VS + R::SH_CODE_DATA, {vs[0], vs[1], vs[2]}, false);
    c.reg(R::VS + R::SH_SWZ_IDX, 0);
    c.reg(R::VS + R::SH_SWZ_DATA, swz("xyzw"));
    c.reg(R::VS + R::SH_ENTRY, 0);
    c.reg(R::VS + R::SH_INPUT_CFG, 1 | 0xA0u << 24);
    c.reg(R::VS + R::SH_PERM_LO, 0x10);
    c.reg(R::VS + R::SH_OUTMASK, 3);
    c.reg(R::VS_NUM_ATTR, 1);
    c.reg(R::VS_OUT_TOTAL, 2);
    c.reg(R::VS_OUT_MAP, 0x03020100);
    c.reg(R::VS_OUT_MAP + 1, 0x0b0a0908);
    c.reg(R::CULL, 0);
    c.reg(R::VP_W, to_f24(FBW / 2.0f));
    c.reg(R::VP_H, to_f24(FBH / 2.0f));
    c.reg(R::VP_CORNER, 0);
    c.reg(R::DEPTHMAP_EN, 1);
    c.reg(R::TEX_CONFIG, 0);
    c.reg(R::LIGHTING_DISABLE, 1);
    // TEV stage 0: replace with constant; others pass previous
    c.reg(R::TEV0, 0x000e000e);
    c.reg(R::TEV0 + 1, 0);
    c.reg(R::TEV0 + 2, 0);
    c.reg(R::TEV0 + 3, 0x01010101);
    c.reg(R::TEV0 + 4, 0);
    const u32 others[5] = {R::TEV1, R::TEV2, R::TEV3, R::TEV4, R::TEV5};
    for (u32 b : others) {
        c.reg(b, 0x000f000f);
        c.reg(b + 1, 0);
        c.reg(b + 2, 0);
        c.reg(b + 4, 0);
    }
    c.reg(R::OM_MODE, 0x100);
    c.reg(R::BLEND, 0x11110000);
    c.reg(R::ALPHA_TEST, 0);
    c.reg(R::STENCIL_TEST, 0);
    c.reg(R::DEPTH_COLOR_MASK, 0xf00);
    c.reg(R::COLOR_WRITE, 0xf);
    c.reg(R::DEPTH_WRITE, 0);
    c.reg(R::COLOR_FMT, 0);
    c.reg(R::COLOR_ADDR, FB / 8);
    c.reg(R::DEPTH_ADDR, 0);
    c.reg(R::FB_DIM, FBW | (FBH - 1) << 12);
    c.reg(R::SCISSOR_MODE, 0);
    c.reg(R::GS_MISC, 0);
    c.reg(R::TOPOLOGY, 0);
}

// submit a vertex in immediate mode: screen-space position (pixels) -> clip coords with w=1
static void imm_vertex(Cmd &c, float sx, float sy, float z = -0.5f, u32 color = 0xffffffff) {
    float pos[4] = {sx / (FBW / 2.0f) - 1, sy / (FBH / 2.0f) - 1, z, 1};
    float col[4] = {(color & 255) / 255.0f, ((color >> 8) & 255) / 255.0f,
                    ((color >> 16) & 255) / 255.0f, (color >> 24) / 255.0f};
    u32 p[3], q[3];
    pack_f24(pos, p);
    pack_f24(col, q);
    for (u32 i = 0; i < 3; i++) c.reg(R::FIXATTR_DATA + i, p[i]);
    for (u32 i = 0; i < 3; i++) c.reg(R::FIXATTR_DATA + i, q[i]);
}

static void test_raster_coverage() {
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    Cmd c;
    setup_counting_pipeline(c);
    c.reg(R::FIXATTR_INDEX, 0xF);
    // a 4x4 grid of quads over [4,28]^2 with jittered interior vertices, two triangles per quad
    const int N = 4;
    float vx[N + 1][N + 1], vy[N + 1][N + 1];
    u32 seed = 12345;
    auto rnd = [&]() { seed = seed * 1103515245 + 12345; return ((seed >> 16) & 0x7fff) / 32768.0f; };
    for (int j = 0; j <= N; j++)
        for (int i = 0; i <= N; i++) {
            vx[j][i] = 4 + i * 6.0f;
            vy[j][i] = 4 + j * 6.0f;
            if (i > 0 && i < N) vx[j][i] += (rnd() - 0.5f) * 3.3f;
            if (j > 0 && j < N) vy[j][i] += (rnd() - 0.5f) * 3.3f;
        }
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            // alternate winding to exercise the KeepAll reordering
            imm_vertex(c, vx[j][i], vy[j][i]);
            imm_vertex(c, vx[j][i + 1], vy[j][i + 1]);
            imm_vertex(c, vx[j + 1][i + 1], vy[j + 1][i + 1]);
            imm_vertex(c, vx[j][i], vy[j][i]);
            imm_vertex(c, vx[j + 1][i], vy[j + 1][i]);
            imm_vertex(c, vx[j + 1][i + 1], vy[j + 1][i + 1]);
        }
    c.run();
    int doubles = 0, holes = 0, outside = 0;
    for (u32 y = 0; y < FBH; y++)
        for (u32 x = 0; x < FBW; x++) {
            u32 n = fb_pixel(x, y) & 255;
            bool inside = x >= 4 && x < 28 && y >= 4 && y < 28;  // centers strictly inside
            if (n > 1) doubles++;
            if (inside && n == 0) holes++;
            if (!inside && n) outside++;
        }
    CHECK(doubles == 0 && holes == 0 && outside == 0, "coverage: doubles %d holes %d outside %d",
          doubles, holes, outside);
}

static void test_cull_and_depth() {
    // culling: CCW-only mode must reject a clockwise triangle
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    Cmd c;
    setup_counting_pipeline(c);
    c.reg(R::CULL, 2);  // keep counter-clockwise
    c.reg(R::FIXATTR_INDEX, 0xF);
    imm_vertex(c, 2, 2);
    imm_vertex(c, 2, 20);
    imm_vertex(c, 20, 2);  // clockwise in y-up coords
    imm_vertex(c, 2, 2);
    imm_vertex(c, 20, 2);
    imm_vertex(c, 2, 20);  // counter-clockwise
    c.run();
    CHECK((fb_pixel(5, 5) & 255) == 1, "CCW kept, CW culled: %u", fb_pixel(5, 5) & 255);
}

// ---------------------------------------------------------------- vertex arrays & draw
static void test_vertex_arrays() {
    // indexed triangle strip from a float/byte vertex buffer
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    const u32 VB = 0x20200000;
    struct Vtx {
        float x, y, z;
        u8 r, g, b, a;
    } verts[4] = {{-1, -1, -0.5f, 255, 0, 0, 255},
                  {1, -1, -0.5f, 0, 255, 0, 255},
                  {-1, 1, -0.5f, 0, 0, 255, 255},
                  {1, 1, -0.5f, 255, 255, 255, 255}};
    memcpy(mem::paddr_ptr(VB), verts, sizeof(verts));
    u16 idx[4] = {0, 1, 2, 3};
    memcpy(mem::paddr_ptr(VB + 0x100), idx, sizeof(idx));
    Cmd c;
    setup_counting_pipeline(c);
    c.reg(R::OM_MODE, 0);       // no blending
    c.reg(R::LOGIC_OP, 3);      // copy
    c.reg(R::TEV0, 0x00000000);  // primary color
    c.reg(R::ATTR_BASE, VB / 16 << 1);
    // attr0: float x3 (fmt 3, size 2), attr1: ubyte x4 (fmt 1, size 3)
    c.reg(R::ATTR_FMT_LO, (3 | 2 << 2) | (1 | 3 << 2) << 4);
    c.reg(R::ATTR_FMT_HI, 1u << 28);
    c.reg(R::ATTR_LOADER, 0);
    c.reg(R::ATTR_LOADER + 1, 0x10);
    c.reg(R::ATTR_LOADER + 2, 16 << 16 | 2u << 28);
    c.reg(R::INDEX_ARRAY, 0x100 | 1u << 31);
    c.reg(R::NUM_VERTICES, 4);
    c.reg(R::TOPOLOGY, 1 << 8);  // strip
    c.reg(R::RESTART_PRIM, 1);
    c.reg(R::DRAW_INDEXED, 1);
    c.run();
    u32 bl = fb_pixel(0, 0), br = fb_pixel(31, 0), tl = fb_pixel(0, 31), tr = fb_pixel(31, 31);
    CHECK((bl & 0xffffff) == 0x0000f8 || (bl & 0xff) > 0xe0, "strip corner bl %08x", bl);
    CHECK(((br >> 8) & 0xff) > 0xe0 && (br & 0xff) < 0x20, "strip corner br %08x", br);
    CHECK(((tl >> 16) & 0xff) > 0xe0, "strip corner tl %08x", tl);
    CHECK((tr & 0xffffff) > 0xe0e0e0, "strip corner tr %08x", tr);
    u32 mid = fb_pixel(16, 16);
    CHECK((mid >> 24) == 255, "strip interior written %08x", mid);
}

// ---------------------------------------------------------------- TEV
static void test_tev() {
    bool div_ok = true;
    for (u32 x = 0; x <= 65025; x++)
        if (div255(x) != x / 255) div_ok = false;
    CHECK(div_ok, "div255 exact");
    bool mul_ok = true;  // multiply-shift division used by the vectorized blender
    for (u32 x = 0; x <= 2 * 65025; x++)
        if ((u32)(((u64)x * 8421505u) >> 31) != x / 255) mul_ok = false;
    CHECK(mul_ok, "x*8421505>>31 == x/255 for x <= 130050");
    CHECK(raster_blend_selftest(), "vectorized blending == scalar blending (all factors/equations)");
    DrawState ds{};
    for (int s = 0; s < 6; s++) {
        TevStageState &t = ds.tev[s];
        t = TevStageState{};
        t.csrc[0] = t.csrc[1] = t.csrc[2] = 15;
        t.asrc[0] = t.asrc[1] = t.asrc[2] = 15;
        t.cmul = t.amul = 1;
    }
    u32 tex[4] = {rgba(200, 100, 50, 128), rgba(10, 20, 30, 40), 0, 0};
    // stage 0: modulate primary * tex0
    ds.tev[0].csrc[0] = 0;
    ds.tev[0].csrc[1] = 3;
    ds.tev[0].asrc[0] = 0;
    ds.tev[0].asrc[1] = 3;
    ds.tev[0].cop = ds.tev[0].aop = 1;
    auto run = [&](u32 p, u32 fp, u32 fs, const u32 *tx) { tev_prepare(ds); return tev_combine(ds, p, fp, fs, tx); };
    u32 r = run( rgba(255, 128, 0, 255), 0, 0, tex);
    CHECK(r == rgba(200, 50, 0, 128), "modulate %08x", r);
    // stage 1: add-signed with tex1, scale 2
    ds.tev[1].csrc[0] = 15;
    ds.tev[1].csrc[1] = 4;
    ds.tev[1].cop = 3;
    ds.tev[1].cmul = 2;
    r = run( rgba(255, 128, 0, 255), 0, 0, tex);
    CHECK(r == rgba(164, 0, 0, 128), "addsigned*2 %08x", r);  // (200+10-128)*2=164, 50+20-128<0
    // lerp, one-minus modifiers, dot3
    ds.tev[1] = ds.tev[2];
    ds.tev[0].cop = 4;  // lerp(primary, tex0, const.a)
    ds.tev[0].csrc[2] = 14;
    ds.tev[0].cmod[2] = 2;  // source alpha
    ds.tev[0].konst = rgba(0, 0, 0, 51);
    r = run( rgba(255, 0, 0, 255), 0, 0, tex);
    CHECK((r & 0xffffff) == rgba((255 * 51 + 200 * 204) / 255, (100 * 204) / 255, (50 * 204) / 255, 0),
          "lerp %08x", r);
    ds.tev[0].cop = 7;  // dot3_rgba
    r = run( rgba(255, 128, 128, 255), 0, 0, tex);
    // ((255*2-255)*(200*2-255)+128)/256 + ((1)*(-55)+128)/256 + ((1)*(-155)+128)/256 = 144+0+0
    CHECK(r == rgba(144, 144, 144, 144), "dot3_rgba %08x", r);
    // combiner buffer: stage0 writes buffer, stage1 reads PreviousBuffer
    ds.tev[0].cop = 0;
    ds.tev[0].csrc[0] = 3;
    ds.tev[0].upd_rgb = true;
    ds.tev[1].csrc[0] = 13;
    ds.tev[1].cop = 0;
    ds.tev_buf_color = rgba(1, 2, 3, 4);
    r = run( 0, 0, 0, tex);
    CHECK((r & 0xffffff) == rgba(1, 2, 3, 0), "stage1 sees initial buffer %08x", r);
    ds.tev[2].csrc[0] = 13;
    ds.tev[2].cop = 0;
    r = run( 0, 0, 0, tex);
    CHECK((r & 0xffffff) == (tex[0] & 0xffffff), "stage2 sees stage0 output %08x", r);
}

// ---------------------------------------------------------------- transfers
static void test_transfers() {
    // tiled RGBA8 16x16 -> linear RGB565, and box-filtered XY downscale
    const u32 IN = 0x18100000, OUT = 0x20300000;
    for (u32 y = 0; y < 16; y++)
        for (u32 x = 0; x < 16; x++) {
            u8 *p = mem::paddr_ptr(IN) + tiled_offset(x, y, 16, 4);
            u32 c = rgba(x * 16, y * 16, 128, 255);
            p[0] = 255;
            p[1] = 128;
            p[2] = c >> 8;
            p[3] = c;
        }
    display_transfer(IN, OUT, 16 | 16 << 16, 16 | 16 << 16, 2 << 12);
    const u8 *o = mem::paddr_ptr(OUT);
    bool ok = true;
    for (u32 y = 0; y < 16; y++)
        for (u32 x = 0; x < 16; x++) {
            u16 v;
            memcpy(&v, o + (y * 16 + x) * 2, 2);
            u16 e = ((x * 16) >> 3) << 11 | ((y * 16) >> 2) << 5 | (128 >> 3);
            if (v != e) ok = false;
        }
    CHECK(ok, "tiled->linear RGBA8->RGB565");
    display_transfer(IN, OUT, 16 | 16 << 16, 16 | 16 << 16, 1 << 12 | 2 << 24 | 1);
    // output 8x8 RGB8 flipped vertically; pixel (0, 7-0) = avg of input (0..1, 0..1)
    const u8 *q = mem::paddr_ptr(OUT) + (7 * 8 + 0) * 3;
    CHECK(q[2] == 8 && q[1] == 8 && q[0] == 128, "scale XY + flip %u %u %u", q[2], q[1], q[0]);
    // linear -> tiled round trip
    display_transfer(OUT, IN + 0x10000, 8 | 8 << 16, 8 | 8 << 16, 1 << 8 | 1 << 12 | 2);
    const u8 *tq = mem::paddr_ptr(IN + 0x10000) + tiled_offset(0, 7, 8, 3);
    CHECK(tq[2] == 8 && tq[0] == 128, "linear->tiled");

    // memory fills
    const u32 F = 0x20400000;
    memory_fill(F, F + 12, 0x112233, 1 | 0x100);
    const u8 *f = mem::paddr_ptr(F);
    CHECK(f[0] == 0x33 && f[1] == 0x22 && f[2] == 0x11 && f[9] == 0x33, "fill24");
    memory_fill(F, F + 8, 0xdeadbeef, 1 | 0x200);
    u32 v32;
    memcpy(&v32, f + 4, 4);
    CHECK(v32 == 0xdeadbeef, "fill32");
    memory_fill(F, F + 8, 0xabcd, 1);
    u16 v16;
    memcpy(&v16, f + 6, 2);
    CHECK(v16 == 0xabcd, "fill16");

    // texture copy with gaps: copy 2 rows of 32 bytes, input gap 16, output gap 0
    const u32 S = 0x20500000, D = 0x20510000;
    for (u32 i = 0; i < 128; i++) mem::paddr_ptr(S)[i] = (u8)i;
    memset(mem::paddr_ptr(D), 0, 128);
    texture_copy(S, D, 64, 2 | 1 << 16, 0, 0);
    const u8 *dd = mem::paddr_ptr(D);
    CHECK(dd[0] == 0 && dd[31] == 31 && dd[32] == 48 && dd[63] == 79, "texture copy gaps");
}

// ---------------------------------------------------------------- command list details
static void test_cmdlist() {
    Cmd c;
    // masked write
    c.reg(R::BLEND_COLOR, 0x11223344);
    c.reg(R::BLEND_COLOR, 0xaabbccdd, 0x5);  // bytes 0 and 2
    // consecutive writes (group) and repeated register writes (fog lut stream)
    c.seq(R::TEV0, {1, 2, 3, 4, 5}, true);
    c.reg(R::FOG_LUT_OFFSET, 0);
    c.seq(R::FOG_LUT_DATA, {10, 11, 12}, false);
    c.run();
    CHECK(g.regs[R::BLEND_COLOR] == 0x11bb33dd, "mask write %08x", g.regs[R::BLEND_COLOR]);
    CHECK(g.regs[R::TEV0] == 1 && g.regs[R::TEV0 + 4] == 5, "group write");
    CHECK(g.fog_lut[0] == 10 && g.fog_lut[2] == 12 && g.regs[R::FOG_LUT_OFFSET] == 3, "lut stream");
    // command buffer jump: main list jumps to a second list which sets a register
    Cmd sub;
    sub.reg(R::BLEND_COLOR, 0x12345678);
    memcpy(mem::paddr_ptr(0x20110000), sub.w.data(), sub.w.size() * 4);
    c.reg(R::CMDBUF_SIZE0, (u32)sub.w.size() * 4 / 8);
    c.reg(R::CMDBUF_ADDR0, 0x20110000 / 8);
    c.reg(R::CMDBUF_JUMP0, 1);
    c.reg(R::BLEND_COLOR, 0);  // must not execute (jump replaced the list)
    c.run();
    CHECK(g.regs[R::BLEND_COLOR] == 0x12345678, "cmdbuf jump %08x", g.regs[R::BLEND_COLOR]);
    // float uniform upload in f24 and f32 modes
    float v[4] = {1, 2, 3, 4};
    u32 p[3];
    pack_f24(v, p);
    c.reg(R::VS + R::SH_FLOAT_IDX, 5);
    c.seq(R::VS + R::SH_FLOAT_DATA, {p[0], p[1], p[2]}, false);
    u32 f32w[4];
    float vv[4] = {8, 7, 6, 5};  // sent w first
    memcpy(&f32w[0], &vv[0], 4);
    memcpy(&f32w[1], &vv[1], 4);
    memcpy(&f32w[2], &vv[2], 4);
    memcpy(&f32w[3], &vv[3], 4);
    c.reg(R::VS + R::SH_FLOAT_IDX, 7 | 1u << 31);
    c.seq(R::VS + R::SH_FLOAT_DATA, {f32w[0], f32w[1], f32w[2], f32w[3]}, false);
    c.run();
    CHECK(g.vs.f[5].v[0] == 1 && g.vs.f[5].v[3] == 4, "f24 uniform %f %f", g.vs.f[5].v[0],
          g.vs.f[5].v[3]);
    CHECK(g.vs.f[7].v[0] == 5 && g.vs.f[7].v[3] == 8, "f32 uniform %f %f", g.vs.f[7].v[0],
          g.vs.f[7].v[3]);
}

// ---------------------------------------------------------------- fragment lighting
static u32 light_col(u32 r, u32 gg, u32 b) { return r << 20 | gg << 10 | b; }

// draw a full-screen quad whose vertices carry the given quaternion and view vector
static void lit_quad(Cmd &c, const float q[4], const float view[3]) {
    const float xy[4][2] = {{-1, -1}, {1, -1}, {-1, 1}, {1, 1}};
    const int order[6] = {0, 1, 2, 2, 1, 3};
    for (int k : order) {
        float pos[4] = {xy[k][0], xy[k][1], -0.5f, 1};
        float col[4] = {1, 1, 1, 1};
        float vw[4] = {view[0], view[1], view[2], 0};
        float qq[4] = {q[0], q[1], q[2], q[3]};
        const float *a[4] = {pos, col, qq, vw};
        for (int i = 0; i < 4; i++) {
            u32 p[3];
            pack_f24(a[i], p);
            for (u32 j = 0; j < 3; j++) c.reg(R::FIXATTR_DATA + j, p[j]);
        }
    }
}

static void setup_lighting(Cmd &c, u32 tev_src) {
    setup_counting_pipeline(c);
    // VS: 4 passthrough outputs
    std::vector<u32> vs;
    for (u32 i = 0; i < 4; i++) vs.push_back(0x13u << 26 | i << 21 | i << 12);
    vs.push_back(0x88000000u);
    c.reg(R::VS + R::SH_CODE_IDX, 0);
    c.seq(R::VS + R::SH_CODE_DATA, vs, false);
    c.reg(R::VS + R::SH_INPUT_CFG, 3 | 0xA0u << 24);
    c.reg(R::VS + R::SH_PERM_LO, 0x3210);
    c.reg(R::VS + R::SH_OUTMASK, 0xf);
    c.reg(R::VS_NUM_ATTR, 3);
    c.reg(R::VS_OUT_TOTAL, 4);
    c.reg(R::VS_OUT_MAP + 2, 0x07060504);
    c.reg(R::VS_OUT_MAP + 3, 0x1f141312);
    c.reg(R::OM_MODE, 0);
    c.reg(R::LOGIC_OP, 3);
    c.reg(R::TEV0, tev_src | tev_src << 16);
    c.reg(R::LIGHTING_DISABLE, 0);
    c.reg(R::LIGHT_NUM, 0);
    c.reg(R::LIGHT_PERMUTATION, 0);
    c.reg(R::LIGHT0 + 0, light_col(255, 255, 255));  // specular 0
    c.reg(R::LIGHT0 + 1, 0);
    c.reg(R::LIGHT0 + 2, light_col(200, 100, 50));  // diffuse
    c.reg(R::LIGHT0 + 3, 0);
    c.reg(R::LIGHT0 + 4, 0);                        // x = y = 0
    c.reg(R::LIGHT0 + 5, 0x3c00);                   // z = 1.0
    c.reg(R::LIGHT0 + 9, 1);                        // directional
    c.reg(R::LIGHT_CONFIG0, 0);
    c.reg(R::LIGHT_CONFIG1, 1u << 8 | 1u << 24 | 1u << 17 | 1u << 19 | 1u << 20 | 1u << 21 | 1u << 22);
    c.reg(R::LUT_ABS, 0);
    c.reg(R::LUT_SELECT, 0);
    c.reg(R::LUT_SCALE, 0);
    c.reg(R::LIGHT_AMBIENT, light_col(10, 10, 10));
    c.reg(R::LUT_INDEX, 0);  // D0
    std::vector<u32> lut(256, 0x800);
    c.seq(R::LUT_DATA, lut, false);
    c.reg(R::FIXATTR_INDEX, 0xF);
}

static void test_lighting() {
    const float qid[4] = {0, 0, 0, 1}, vz[3] = {0, 0, 1};
    Cmd c;
    setup_lighting(c, 1);  // primary fragment color
    lit_quad(c, qid, vz);
    c.run();
    u32 p = fb_pixel(16, 16);
    CHECK(abs((int)(p & 255) - 210) <= 1 && abs((int)((p >> 8) & 255) - 110) <= 1 &&
              abs((int)((p >> 16) & 255) - 60) <= 1,
          "diffuse + ambient %08x", p);
    setup_lighting(c, 2);  // secondary (specular): D0 LUT = 0.5 * spec0
    lit_quad(c, qid, vz);
    c.run();
    p = fb_pixel(16, 16);
    CHECK(abs((int)(p & 255) - 127) <= 1 && abs((int)((p >> 16) & 255) - 127) <= 1,
          "specular D0 %08x", p);
    // rotated normal: 90 degrees about X maps +Z to -Y; a light along -Y lights it fully
    setup_lighting(c, 1);
    c.reg(R::LIGHT0 + 4, 0xbc00u << 16);  // y = -1
    c.reg(R::LIGHT0 + 5, 0);
    const float qx[4] = {0.70710678f, 0, 0, 0.70710678f};
    lit_quad(c, qx, vz);
    c.run();
    p = fb_pixel(16, 16);
    CHECK(abs((int)(p & 255) - 210) <= 1, "quaternion-rotated normal %08x", p);
    // light behind the surface -> ambient only
    setup_lighting(c, 1);
    c.reg(R::LIGHT0 + 5, 0xbc00);  // z = -1
    lit_quad(c, qid, vz);
    c.run();
    p = fb_pixel(16, 16);
    CHECK((p & 0xffffff) == rgba(10, 10, 10, 0), "back light -> ambient %08x", p);
}

// ---------------------------------------------------------------- geometry shader
static void test_geometry_shader() {
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    Cmd c;
    setup_counting_pipeline(c);
    c.reg(R::OM_MODE, 0);
    c.reg(R::LOGIC_OP, 3);
    c.reg(R::TEV0, 0);  // primary color
    c.reg(R::GS_MISC, 2);
    c.reg(R::GS_EXCLUSIVE, 1);
    c.reg(R::GS_CONFIG, 0);
    c.reg(R::VS_OUTMAP_TOTAL1, 1);
    c.reg(R::VS_OUTMAP_TOTAL2, 1);
    c.reg(R::GS + R::SH_INPUT_CFG, 1 | 0x08u << 24);
    c.reg(R::GS + R::SH_PERM_LO, 0x10);
    c.reg(R::GS + R::SH_OUTMASK, 3);
    c.reg(R::GS + R::SH_ENTRY, 0);
    auto setemit = [](u32 vid, u32 prim, u32 wind) { return 0x2bu << 26 | vid << 24 | prim << 23 | wind << 22; };
    const u32 mov1 = 0x13u << 26 | 1 << 21 | 1 << 12;
    std::vector<u32> gs = {
        setemit(0, 0, 0), arith(ADD, O(0), C(0), V(0), 0), mov1, 0x2au << 26,
        setemit(1, 0, 0), arith(ADD, O(0), C(1), V(0), 0), mov1, 0x2au << 26,
        setemit(2, 1, 0), arith(ADD, O(0), C(2), V(0), 0), mov1, 0x2au << 26,
        0x88000000u};
    c.reg(R::GS + R::SH_CODE_IDX, 0);
    c.seq(R::GS + R::SH_CODE_DATA, gs, false);
    c.reg(R::GS + R::SH_SWZ_IDX, 0);
    c.reg(R::GS + R::SH_SWZ_DATA, swz("xyzw"));
    // offsets of the emitted triangle around the input point (clip space)
    const float off[3][4] = {{-0.5f, -0.5f, 0, 0}, {0.5f, -0.5f, 0, 0}, {0, 0.5f, 0, 0}};
    c.reg(R::GS + R::SH_FLOAT_IDX, 0);
    for (auto &o : off) {
        u32 p[3];
        pack_f24(o, p);
        c.seq(R::GS + R::SH_FLOAT_DATA, {p[0], p[1], p[2]}, false);
    }
    c.reg(R::TOPOLOGY, 3 << 8);
    c.reg(R::FIXATTR_INDEX, 0xF);
    imm_vertex(c, 16, 16);  // center of the screen, white
    c.run();
    // triangle spans x in [8,24] at the bottom (y=8), apex at (16,24)
    CHECK((fb_pixel(16, 12) & 0xffffff) == 0xffffff, "GS triangle interior %08x", fb_pixel(16, 12));
    CHECK((fb_pixel(16, 4) & 0xffffff) == 0, "GS outside below");
    CHECK((fb_pixel(4, 12) & 0xffffff) == 0, "GS outside left");
    // restore VS-only mode for later tests
    c.reg(R::GS_MISC, 0);
    c.reg(R::GS_EXCLUSIVE, 0);
    c.reg(R::TOPOLOGY, 0);
    c.run();
}

// ---------------------------------------------------------------- depth / stencil
static void quad(Cmd &c, float z, u32 color, float x0 = 0, float y0 = 0, float x1 = 32, float y1 = 32) {
    imm_vertex(c, x0, y0, z, color);
    imm_vertex(c, x1, y0, z, color);
    imm_vertex(c, x0, y1, z, color);
    imm_vertex(c, x0, y1, z, color);
    imm_vertex(c, x1, y0, z, color);
    imm_vertex(c, x1, y1, z, color);
}

static void test_depth_stencil() {
    const u32 DB = 0x18100000;
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    memory_fill(DB, DB + FBW * FBH * 4, 0x00ffffff, 1 | 0x200);  // depth far, stencil 0
    Cmd c;
    setup_counting_pipeline(c);
    c.reg(R::OM_MODE, 0);
    c.reg(R::LOGIC_OP, 3);
    c.reg(R::TEV0, 0);
    c.reg(R::DEPTH_SCALE, to_f24(-1.0f));
    c.reg(R::DEPTH_OFFSET, 0);
    c.reg(R::DEPTH_ADDR, DB / 8);
    c.reg(R::DEPTH_FMT, 3);
    c.reg(R::DEPTH_WRITE, 3);
    c.reg(R::DEPTH_COLOR_MASK, 1 | 4 << 4 | 0xf00 | 1 << 12);  // depth test LESS, write
    c.reg(R::FIXATTR_INDEX, 0xF);
    quad(c, -0.75f, 0xff0000ff);  // red at depth 0.75
    quad(c, -0.25f, 0xff00ff00);  // green at 0.25
    quad(c, -0.5f, 0xffff0000);   // blue at 0.5: hidden
    c.run();
    u32 p = fb_pixel(10, 10);
    CHECK((p & 0xffffff) == 0x00ff00, "depth LESS keeps nearest %08x", p);
    const u8 *d = mem::paddr_ptr(DB) + tiled_offset(10, FBH - 1 - 10, FBW, 4);
    u32 dz = d[0] | d[1] << 8 | d[2] << 16;
    CHECK(abs((int)dz - (int)(0.25 * 16777215)) < 4, "depth value %06x", dz);
    // stencil: write 1 in the left half (always pass, replace), then draw where stencil == 0
    c.reg(R::DEPTH_COLOR_MASK, 0xf00);  // no depth test / write
    c.reg(R::STENCIL_TEST, 1 | 1 << 4 | 0xff << 8 | 1 << 16 | 0xff << 24);
    c.reg(R::STENCIL_OP, 2 << 8 | 2 << 4 | 2);
    c.reg(R::FIXATTR_INDEX, 0xF);
    quad(c, -0.5f, 0xff808080, 0, 0, 16, 32);
    c.reg(R::STENCIL_TEST, 1 | 2 << 4 | 0xff << 8 | 0 << 16 | 0xff << 24);  // EQUAL 0
    c.reg(R::STENCIL_OP, 0);
    c.reg(R::FIXATTR_INDEX, 0xF);
    quad(c, -0.5f, 0xffffffff);
    c.run();
    CHECK((fb_pixel(5, 5) & 0xffffff) == 0x808080, "stencil masked %08x", fb_pixel(5, 5));
    CHECK((fb_pixel(25, 5) & 0xffffff) == 0xffffff, "stencil pass %08x", fb_pixel(25, 5));
    c.reg(R::STENCIL_TEST, 0);
    c.reg(R::DEPTH_ADDR, 0);
    c.run();
}

// ---------------------------------------------------------------- texturing through the pipeline
static void test_texturing() {
    const u32 TEX = 0x18200000;
    // 8x8 RGBA8: memory row 0 red, rest blue
    u8 *t = mem::paddr_ptr(TEX);
    for (u32 y = 0; y < 8; y++)
        for (u32 x = 0; x < 8; x++) {
            u8 *p = t + ref_morton(x, y) * 4;
            u32 c = y == 0 ? rgba(255, 0, 0, 255) : rgba(0, 0, 255, 255);
            p[0] = c >> 24; p[1] = c >> 16; p[2] = c >> 8; p[3] = c;
        }
    memset(mem::paddr_ptr(FB), 0, FBW * FBH * 4);
    Cmd c;
    setup_counting_pipeline(c);
    std::vector<u32> vs;
    for (u32 i = 0; i < 3; i++) vs.push_back(0x13u << 26 | i << 21 | i << 12);
    vs.push_back(0x88000000u);
    c.reg(R::VS + R::SH_CODE_IDX, 0);
    c.seq(R::VS + R::SH_CODE_DATA, vs, false);
    c.reg(R::VS + R::SH_INPUT_CFG, 2 | 0xA0u << 24);
    c.reg(R::VS + R::SH_PERM_LO, 0x210);
    c.reg(R::VS + R::SH_OUTMASK, 7);
    c.reg(R::VS_NUM_ATTR, 2);
    c.reg(R::VS_OUT_TOTAL, 3);
    c.reg(R::VS_OUT_MAP + 2, 0x1f1f0d0c);
    c.reg(R::OM_MODE, 0);
    c.reg(R::LOGIC_OP, 3);
    c.reg(R::TEV0, 3 | 3 << 16);  // texture 0
    c.reg(R::TEX_CONFIG, 1);
    c.reg(R::TEX0 + 1, 8 | 8 << 16);
    c.reg(R::TEX0 + 2, 2 << 8 | 2 << 12);  // nearest, repeat
    c.reg(R::TEX0 + 3, 0);
    c.reg(R::TEX0 + 4, TEX / 8);
    c.reg(R::TEX0_TYPE, 0);
    c.reg(R::FIXATTR_INDEX, 0xF);
    const float xy[4][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    for (int k : {0, 1, 2, 2, 1, 3}) {
        float pos[4] = {xy[k][0] * 2 - 1, xy[k][1] * 2 - 1, -0.5f, 1};
        float col[4] = {1, 1, 1, 1};
        float uv[4] = {xy[k][0], xy[k][1], 0, 1};
        const float *a[3] = {pos, col, uv};
        for (auto *v : a) {
            u32 p[3];
            pack_f24(v, p);
            for (u32 j = 0; j < 3; j++) c.reg(R::FIXATTR_DATA + j, p[j]);
        }
    }
    c.run();
    CHECK((fb_pixel(16, 30) & 0xffffff) == 0x0000ff, "top of screen = texture memory row 0 %08x",
          fb_pixel(16, 30));
    CHECK((fb_pixel(16, 10) & 0xffffff) == 0xff0000, "rest blue %08x", fb_pixel(16, 10));

    // mipmaps: 16x16 level 0 red, 8x8 level 1 green
    const u32 MT = 0x18210000;
    u8 *m = mem::paddr_ptr(MT);
    for (u32 i = 0; i < 16 * 16; i++) {
        m[4 * i] = 255; m[4 * i + 1] = 0; m[4 * i + 2] = 0; m[4 * i + 3] = 255;
    }
    for (u32 i = 0; i < 8 * 8; i++) {
        u8 *p = m + 1024 + 4 * i;
        p[0] = 255; p[1] = 0; p[2] = 255; p[3] = 0;
    }
    auto draw_tex_quad = [&](float size) {
        Cmd d;
        d.reg(R::TEX_CONFIG, 1);
        d.reg(R::TEX0 + 1, 16 | 16 << 16);
        d.reg(R::TEX0 + 2, 2 << 8 | 2 << 12);  // nearest, nearest mip
        d.reg(R::TEX0 + 3, 1 << 16);            // max level 1
        d.reg(R::TEX0 + 4, MT / 8);
        d.reg(R::FIXATTR_INDEX, 0xF);
        for (int k : {0, 1, 2, 2, 1, 3}) {
            float pos[4] = {(xy[k][0] * size) / 16.0f - 1, (xy[k][1] * size) / 16.0f - 1, -0.5f, 1};
            float col[4] = {1, 1, 1, 1};
            float uv[4] = {xy[k][0], xy[k][1], 0, 1};
            const float *a[3] = {pos, col, uv};
            for (auto *v : a) {
                u32 p[3];
                pack_f24(v, p);
                for (u32 j = 0; j < 3; j++) d.reg(R::FIXATTR_DATA + j, p[j]);
            }
        }
        d.run();
    };
    draw_tex_quad(16);  // 16x16 texels on 16x16 pixels: level 0
    CHECK((fb_pixel(8, 8) & 0xffffff) == 0x0000ff, "mip level 0 %08x", fb_pixel(8, 8));
    draw_tex_quad(8);  // minified 2x: level 1
    CHECK((fb_pixel(4, 4) & 0xffffff) == 0x00ff00, "mip level 1 %08x", fb_pixel(4, 4));
    c.reg(R::TEX_CONFIG, 0);
    c.run();
}

// ---------------------------------------------------------------- synthetic benchmark
// A lit, textured, depth-tested mesh of ~40k small triangles on a 240x400 target.
static int bench(int frames) {
    const u32 W = 240, H = 400, CB = 0x18000000, DB = 0x18100000, TEX = 0x18200000;
    const u32 VB = 0x20200000, IB = 0x20400000, CL = 0x20600000;
    // texture 128x128 RGBA8 noise-ish pattern
    for (u32 i = 0; i < 128 * 128 * 4; i++) mem::paddr_ptr(TEX)[i] = (u8)(i * 2654435761u >> 13);
    // vertices: grid NX x NY, position (x,y,z), quaternion (4 floats), uv (2 floats)
    const int NX = 129, NY = 81;
    struct Vtx { float p[3], q[4], uv[2]; };
    std::vector<Vtx> vtx;
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++) {
            Vtx v;
            v.p[0] = i / (float)(NX - 1) * 2 - 1;
            v.p[1] = j / (float)(NY - 1) * 2 - 1;
            v.p[2] = 0.3f * sinf(i * 0.2f) * cosf(j * 0.3f);
            float a = 0.4f * sinf(i * 0.1f + j * 0.05f);
            v.q[0] = sinf(a); v.q[1] = 0; v.q[2] = 0; v.q[3] = cosf(a);
            v.uv[0] = i / 16.0f; v.uv[1] = j / 16.0f;
            vtx.push_back(v);
        }
    memcpy(mem::paddr_ptr(VB), vtx.data(), vtx.size() * sizeof(Vtx));
    std::vector<u16> idx;
    for (int j = 0; j < NY - 1; j++)
        for (int i = 0; i < NX - 1; i++) {
            u16 a = j * NX + i, b = a + 1, c2 = a + NX, d = c2 + 1;
            idx.insert(idx.end(), {a, b, c2, c2, b, d});
        }
    memcpy(mem::paddr_ptr(IB), idx.data(), idx.size() * 2);

    Cmd c;
    // vertex shader: o0 = MVP * v0, o1 = -(MV * v0) (view), o2 = quat, o3 = uv, o4 = c7 color
    u32 d = 0;
    std::vector<u32> sw = {swz("xyzw"), swz("x"), swz("y"), swz("z"), swz("w"), swz("xyz", "xyzw", true)};
    std::vector<u32> vs = {
        arith(DP4, O(0), C(0), V(0), 4 - 3), arith(DP4, O(0), C(1), V(0), 2), arith(DP4, O(0), C(2), V(0), 3),
        arith(DP4, O(0), C(3), V(0), 4), arith(DP4, Rg(0), C(4), V(0), 1), arith(DP4, Rg(0), C(5), V(0), 2),
        arith(DP4, Rg(0), C(6), V(0), 3), arith(MOV, O(1), Rg(0), 0, 5), arith(MOV, O(2), V(1), 0, d),
        arith(MOV, O(3), V(2), 0, d), arith(MOV, O(4), C(7), 0, d), ENDI};
    c.reg(R::VS + R::SH_CODE_IDX, 0);
    c.seq(R::VS + R::SH_CODE_DATA, vs, false);
    c.reg(R::VS + R::SH_SWZ_IDX, 0);
    c.seq(R::VS + R::SH_SWZ_DATA, sw, false);
    c.reg(R::VS + R::SH_ENTRY, 0);
    c.reg(R::VS + R::SH_INPUT_CFG, 2 | 0xA0u << 24);
    c.reg(R::VS + R::SH_PERM_LO, 0x210);
    c.reg(R::VS + R::SH_OUTMASK, 0x1f);
    c.reg(R::VS_NUM_ATTR, 2);
    c.reg(R::VS_OUT_TOTAL, 5);
    c.reg(R::VS_OUT_MAP + 0, 0x03020100);
    c.reg(R::VS_OUT_MAP + 1, 0x1f141312);
    c.reg(R::VS_OUT_MAP + 2, 0x07060504);
    c.reg(R::VS_OUT_MAP + 3, 0x1f1f0d0c);
    c.reg(R::VS_OUT_MAP + 4, 0x0b0a0908);
    // uniforms: simple perspective-ish matrices
    const float mvp[4][4] = {{0.9f, 0, 0, 0}, {0, 0.9f, 0, 0}, {0, 0, -0.5f, -0.6f}, {0, 0, -0.2f, 1.2f}};
    const float mv[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, -3}};
    c.reg(R::VS + R::SH_FLOAT_IDX, 0 | 1u << 31);
    auto up = [&](const float *v) {
        u32 w[4];
        for (int k = 0; k < 4; k++) memcpy(&w[k], &v[3 - k], 4);
        c.seq(R::VS + R::SH_FLOAT_DATA, {w[0], w[1], w[2], w[3]}, false);
    };
    for (auto &r : mvp) up(r);
    for (auto &r : mv) up(r);
    const float col[4] = {1, 1, 1, 1};
    up(col);
    // vertex arrays: attr0 float3, attr1 float4, attr2 float2 in one loader, stride 36
    c.reg(R::ATTR_BASE, VB / 16 << 1);
    c.reg(R::ATTR_FMT_LO, (3 | 2 << 2) | (3 | 3 << 2) << 4 | (3 | 1 << 2) << 8);
    c.reg(R::ATTR_FMT_HI, 2u << 28);
    c.reg(R::ATTR_LOADER, 0);
    c.reg(R::ATTR_LOADER + 1, 0x210);
    c.reg(R::ATTR_LOADER + 2, 36 << 16 | 3u << 28);
    // raster / framebuffer
    c.reg(R::CULL, 0);
    c.reg(R::VP_W, to_f24(W / 2.0f));
    c.reg(R::VP_H, to_f24(H / 2.0f));
    c.reg(R::VP_CORNER, 0);
    c.reg(R::DEPTHMAP_EN, 1);
    c.reg(R::DEPTH_SCALE, to_f24(-1.0f));
    c.reg(R::DEPTH_OFFSET, 0);
    c.reg(R::COLOR_FMT, 0);
    c.reg(R::DEPTH_FMT, 3);
    c.reg(R::COLOR_ADDR, CB / 8);
    c.reg(R::DEPTH_ADDR, DB / 8);
    c.reg(R::FB_DIM, W | (H - 1) << 12);
    c.reg(R::COLOR_WRITE, 0xf);
    c.reg(R::DEPTH_WRITE, 3);
    c.reg(R::DEPTH_COLOR_MASK, 1 | 5 << 4 | 0xf00 | 1 << 12);
    c.reg(R::OM_MODE, 0);
    c.reg(R::LOGIC_OP, 3);
    c.reg(R::ALPHA_TEST, 0);
    c.reg(R::STENCIL_TEST, 0);
    c.reg(R::SCISSOR_MODE, 0);
    c.reg(R::GS_MISC, 0);
    // texture 0: 128x128 RGBA8, bilinear, repeat
    c.reg(R::TEX_CONFIG, 1);
    c.reg(R::TEX0 + 1, 128 | 128 << 16);
    c.reg(R::TEX0 + 2, 1 << 1 | 1 << 2 | 2 << 8 | 2 << 12);
    c.reg(R::TEX0 + 3, 0);
    c.reg(R::TEX0 + 4, TEX / 8);
    c.reg(R::TEX0_TYPE, 0);
    // TEV: tex0 * frag_primary + frag_secondary
    c.reg(R::TEV0, 3 | 1 << 4 | 2 << 8 | (3 | 1 << 4) << 16);
    c.reg(R::TEV0 + 1, 0);
    c.reg(R::TEV0 + 2, 8 | 1 << 16);
    c.reg(R::TEV0 + 3, 0);
    c.reg(R::TEV0 + 4, 0);
    for (u32 b : {R::TEV1, R::TEV2, R::TEV3, R::TEV4, R::TEV5}) {
        c.reg(b, 0x000f000f);
        c.reg(b + 1, 0);
        c.reg(b + 2, 0);
        c.reg(b + 4, 0);
    }
    // two lights, D0 LUT on NH
    c.reg(R::LIGHTING_DISABLE, 0);
    c.reg(R::LIGHT_NUM, 1);
    c.reg(R::LIGHT_PERMUTATION, 0x10);
    for (u32 l = 0; l < 2; l++) {
        const u32 b = R::LIGHT0 + 16 * l;
        c.reg(b + 0, light_col(200, 200, 200));
        c.reg(b + 1, 0);
        c.reg(b + 2, light_col(180, 160, 140));
        c.reg(b + 3, light_col(20, 20, 20));
        c.reg(b + 4, l ? 0x3c00bc00u : 0x3800);
        c.reg(b + 5, 0x3c00);
        c.reg(b + 9, l ? 0 : 1);
    }
    c.reg(R::LIGHT_CONFIG0, 0 << 4);
    c.reg(R::LIGHT_CONFIG1, 0xffu << 8 | 0xffu << 24 | 1u << 17 | 1u << 19 | 1u << 20 | 1u << 21 | 1u << 22);
    c.reg(R::LUT_ABS, 0);
    c.reg(R::LUT_SELECT, 0);
    c.reg(R::LUT_SCALE, 0);
    c.reg(R::LIGHT_AMBIENT, light_col(16, 16, 16));
    c.reg(R::LUT_INDEX, 0);
    std::vector<u32> lut;
    for (int i = 0; i < 256; i++) lut.push_back((u32)(powf(i / 255.0f, 16) * 4095));
    c.seq(R::LUT_DATA, lut, false);
    // two draws of the mesh (second one slightly behind -> depth rejected)
    c.reg(R::INDEX_ARRAY, (IB - VB) | 1u << 31);
    c.reg(R::NUM_VERTICES, (u32)idx.size());
    c.reg(R::TOPOLOGY, 0);
    c.reg(R::RESTART_PRIM, 1);
    c.reg(R::DRAW_INDEXED, 1);
    float mvp2[4] = {0, 0, -0.5f, -0.4f};
    c.reg(R::VS + R::SH_FLOAT_IDX, 2 | 1u << 31);
    up(mvp2);
    c.reg(R::RESTART_PRIM, 1);
    c.reg(R::DRAW_INDEXED, 1);
    c.w.push_back(0x12345678);
    c.w.push_back(R::IRQ_REQ | 0xfu << 16);
    memcpy(mem::paddr_ptr(CL), c.w.data(), c.w.size() * 4);
    const u32 clsize = (u32)c.w.size() * 4;
    double total = 0, best = 1e9;
    for (int f = 0; f < frames; f++) {
        memory_fill(CB, CB + W * H * 4, 0x404040ff, 1 | 0x200);
        memory_fill(DB, DB + W * H * 4, 0x00ffffff, 1 | 0x200);
        auto t0 = std::chrono::steady_clock::now();
        process_command_list(CL, clsize);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (f > 0) total += ms;
        if (f > 0 && ms < best) best = ms;
        if (getenv("BENCH_VERBOSE")) printf("frame %d: %.2f ms\n", f, ms);
    }
    printf("average %.2f ms, best %.2f ms (%zu triangles/draw)\n", total / (frames - 1), best, idx.size() / 3);
    // write the output for inspection
    std::vector<u32> img(W * H);
    for (u32 y = 0; y < H; y++)
        for (u32 x = 0; x < W; x++) img[(H - 1 - y) * W + x] = decode_pixel(0, mem::paddr_ptr(CB) + tiled_offset(x, H - 1 - y, W, 4)) | 0xff000000u;
    extern bool write_png(const char *, uint32_t, uint32_t, const uint32_t *);
    write_png("bench.png", W, H, img.data());
    return 0;
}

int main(int argc, char **argv) {
    setenv("GPU_BACKEND", "sw", 0);  // the unit tests check the software renderer bit-exactly
    mem::init();
    init();
    if (argc > 1 && std::string(argv[1]) == "bench") return bench(argc > 2 ? atoi(argv[2]) : 10);
    test_float_formats();
    test_morton_and_decode();
    test_etc1();
    test_shader();
    test_cmdlist();
    test_tev();
    test_transfers();
    test_raster_coverage();
    test_cull_and_depth();
    test_vertex_arrays();
    test_lighting();
    test_geometry_shader();
    test_depth_stencil();
    test_texturing();
    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
