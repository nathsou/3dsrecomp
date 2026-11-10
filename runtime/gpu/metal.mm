// Metal backend: rasterization and the whole fragment pipeline (textures, fragment lighting, TEV,
// fog, alpha / stencil / depth tests, blending, logic ops) run on the GPU at an internal resolution
// scale (changeable while running), optionally multisampled. Vertex programs run as compute kernels;
// clipping, culling and snapping in the shared vertex shader. Guest memory stays authoritative: render targets are synchronised with it
// (hash-checked, once per command list) before drawing, and written back (downsampled to 1x, PICA tiled format)
// lazily, when a GPU operation reads them. Texture copies and tiled-to-tiled transfers of render targets, and
// draws sampling the results, stay on the GPU. A command list's passes share one command buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach-o/dyld.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <thread>
#include <cstdarg>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

#include "gpu/fsgen.h"
#include "gpu/vsblob.h"
#include "gpu/pica.h"
#include "gpu/texfilter.h"
#include "loader.h"

extern std::string g_save_base;

namespace gpu {

// the backend's hooks (pica.h's HwBackend; the table is at the end of this file)
bool metal_init();
void metal_flush(const RTConfig &rt, const MBatch &b);
const void *metal_vs_kernel(const ShaderProgram *p, u32 entry, const ShaderSetup &s);
void metal_guest_read(u32 paddr, u32 size);
void metal_guest_written(u32 paddr, u32 size, bool definite = true);
void metal_guest_prewrite(u32 paddr, u32 size);
void metal_writeback_all();
void metal_poll();
bool metal_texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width, u32 in_gap, u32 out_width, u32 out_gap);
bool metal_gpu_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
bool metal_surface_for(u32 addr, u32 w, u32 h, u32 fmt);
void metal_end_list();
bool metal_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size);
void metal_display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
bool metal_gpu_made(u32 paddr, u32 size);
void metal_note_gpu_output(u32 paddr, u32 size);
void metal_set_texture_filter(int mode);

namespace {

// shaders compiled at run time without fast math, as the precompiled ones (MTLMathModeSafe; before macOS 15 and
// iOS 18, which iPhones and iPads may still run, the older switch)
MTLCompileOptions *safe_math_options() {
    MTLCompileOptions *opt = [MTLCompileOptions new];
    if (@available(macOS 15.0, iOS 18.0, *)) {
        opt.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    return opt;
}

// ---------------------------------------------------------------- shader sources
// the vertex shader shared by all pipelines: vertices in the OutVertex layout (shaded by a compute
// kernel or the CPU), PICA clipping (z planes, w > 0, user plane) as clip distances, x/y clipped by
// Metal, snapping to the PICA's 1/16 pixel grid, quaternions flipped toward the triangle's first vertex
const char *kVertexMSL = R"(
struct VsDraw {
    float4 rt;    // w, h, depth scale, depth offset
    float4 vp;    // viewport half width, half height, x, y
    float4 clip;  // user clip plane
    uint4 flags;  // x: user clip plane enabled
};
struct VOutV {
    float4 pos [[position]];
    float4 color;
    float4 tc01;
    float4 tc2w;
    float4 quat;
    float4 view;
    float2 zw;
    float clip [[clip_distance]] [4];
};
static inline device const float *vs_vertex(device const float *gv, device const float *cv, uint i) {
    return (i & 0x80000000u) != 0u ? cv + (i & 0x7fffffffu) * 24u : gv + i * 24u;
}
vertex VOutV vs_main(uint vid [[vertex_id]], device const float *gv [[buffer(0)]], device const float *cv [[buffer(1)]],
                     device const uint *idx [[buffer(2)]], constant VsDraw &D [[buffer(3)]]) {
    device const float *v = vs_vertex(gv, cv, idx[vid]);
    VOutV o;
    const float x = v[0], y = v[1], z = v[2], w = v[3];
    float px, py;
    if (w > 0.0) {
        const float iw = 1.0 / w;
        px = (x * iw + 1.0) * D.vp.x + D.vp.z;
        py = (y * iw + 1.0) * D.vp.y + D.vp.w;
        if (abs(px) < 8192.0 && abs(py) < 8192.0) {
            px = round(px * 16.0) / 16.0;
            py = round(py * 16.0) / 16.0;
        }
        o.pos.x = (px / D.rt.x * 2.0 - 1.0) * w;
        o.pos.y = (py / D.rt.y * 2.0 - 1.0) * w;
    } else {
        o.pos.x = ((x + w) * D.vp.x + D.vp.z * w) / D.rt.x * 2.0 - w;
        o.pos.y = ((y + w) * D.vp.y + D.vp.w * w) / D.rt.y * 2.0 - w;
    }
    o.pos.z = z * D.rt.z + D.rt.w * w;  // depth z/w * scale + offset, clamped by the depth clip mode
    o.pos.w = w;
    // inside up to raster.cpp's EPS_Z, as in the software renderer: values below the PICA's f24 range are 0
    // there (Picross 3D: Round 2 draws its 2D menus at z = 1e-37, which a strict near plane would clip)
    const float eps = 0.00000001;
    o.clip[0] = eps - z;
    o.clip[1] = z + w + eps;
    o.clip[2] = w - 0.00001 + eps;
    o.clip[3] = D.flags.x != 0u ? x * D.clip.x + y * D.clip.y + z * D.clip.z + w * D.clip.w + eps : 1.0;
    float4 q = float4(v[4], v[5], v[6], v[7]);
    const uint first = vid - vid % 3u;
    if (vid != first) {
        device const float *v0 = vs_vertex(gv, cv, idx[first]);
        const float4 q0 = float4(v0[4], v0[5], v0[6], v0[7]);
        if (q.x * q0.x + q.y * q0.y + q.z * q0.z + q.w * q0.w < 0.0) q = -q;
    }
    o.quat = q;
    o.color = float4(v[8], v[9], v[10], v[11]);
    o.tc01 = float4(v[12], v[13], v[14], v[15]);
    o.tc2w = float4(v[22], v[23], v[16], 0.0);
    o.view = float4(v[18], v[19], v[20], 0.0);
    o.zw = float2(z, w);
    return o;
}
)";

// helpers for the compute kernels translated from vertex programs (shader_msl() in shader.cpp)
const char *kVsCommonMSL = R"(
#include <metal_stdlib>
using namespace metal;
struct VsAttr { int offset; uint stride, fmt, elems; };
struct VsParams {
    float4 f[96];
    float4 defaults[16];
    VsAttr attr[16];
    int reg_attr[16];
    int slot_src[24];
    uint4 iu[4];
    uint b, nverts, out_base, first;
};
struct VsFlow {
    uint ie[8], iend[8], nif;
    uint ce[4], cr[4], ncall;
    uint le[4], lend[4], lc[4], li[4], lp[4], nloop;
    int aL;
};
static inline float vs_rdf(device const uchar *d, int a) {
    return as_type<float>(uint(d[a]) | (uint(d[a + 1]) << 8) | (uint(d[a + 2]) << 16) | (uint(d[a + 3]) << 24));
}
static inline float4 vs_input(device const uchar *d, constant VsParams &P, uint r, uint vid) {
    const int a = P.reg_attr[r];
    if (a < 0) return float4(0.0);
    const VsAttr A = P.attr[a];
    if (A.elems == 5u) return P.defaults[a];
    float4 v = float4(0.0, 0.0, 0.0, 1.0);
    if (A.elems == 0u) return v;
    const int base = A.offset + int(A.stride * vid);
    for (uint c = 0; c < A.elems; c++) {
        switch (A.fmt) {
        case 0: v[c] = float(as_type<char>(d[base + int(c)])); break;
        case 1: v[c] = float(d[base + int(c)]); break;
        case 2: v[c] = float(as_type<short>(ushort(ushort(d[base + int(2 * c)]) | (ushort(d[base + int(2 * c) + 1]) << 8)))); break;
        default: v[c] = vs_rdf(d, base + int(4 * c)); break;
        }
    }
    return v;
}
// PICA multiplication: 0 * inf = 0
static inline float4 vs_mul(float4 a, float4 b) {
    const float4 r = a * b;
    return select(r, float4(0.0), isnan(r) && !isnan(a) && !isnan(b));
}
static inline float4 vs_dp3(float4 a, float4 b) { const float4 p = vs_mul(a, b); float d = 0.0; d += p.x; d += p.y; d += p.z; return float4(d); }
static inline float4 vs_dp4(float4 a, float4 b) { const float4 p = vs_mul(a, b); float d = 0.0; d += p.x; d += p.y; d += p.z; d += p.w; return float4(d); }
static inline float4 vs_dph(float4 a, float4 b) { a.w = 1.0; return vs_dp4(a, b); }
static inline float4 vs_dst(float4 a, float4 b) { return float4(1.0, vs_mul(a, b).y, a.z, b.w); }
static inline float4 vs_max(float4 a, float4 b) { return select(b, a, a > b); }
static inline float4 vs_min(float4 a, float4 b) { return select(b, a, a < b); }
static inline int vs_s32(float f) {
    if (isnan(f)) return 0;
    if (f >= 2147483647.0) return 0x7fffffff;
    if (f <= -2147483648.0) return int(0x80000000u);
    return int(f);
}
static inline float4 vs_rel(constant VsParams &P, int base, int off) {
    if (off < -128 || off > 127) off = 0;
    const uint u = uint(base + off) & 0x7fu;
    return u >= 96u ? float4(1.0) : P.f[u];
}
static inline void vs_push_if(thread VsFlow &S, uint e, uint end) {
    if (S.nif == 8u) { for (int i = 0; i < 7; i++) { S.ie[i] = S.ie[i + 1]; S.iend[i] = S.iend[i + 1]; } S.nif = 7u; }
    S.ie[S.nif] = e; S.iend[S.nif] = end; S.nif++;
}
static inline void vs_push_call(thread VsFlow &S, uint end, uint ret) {
    if (S.ncall == 4u) { for (int i = 0; i < 3; i++) { S.ce[i] = S.ce[i + 1]; S.cr[i] = S.cr[i + 1]; } S.ncall = 3u; }
    S.ce[S.ncall] = end; S.cr[S.ncall] = ret; S.ncall++;
}
static inline void vs_push_loop(thread VsFlow &S, uint entry, uint end, uint cnt, uint inc, uint prev) {
    if (S.nloop == 4u) {
        for (int i = 0; i < 3; i++) { S.le[i] = S.le[i + 1]; S.lend[i] = S.lend[i + 1]; S.lc[i] = S.lc[i + 1]; S.li[i] = S.li[i + 1]; S.lp[i] = S.lp[i + 1]; }
        S.nloop = 3u;
    }
    S.le[S.nloop] = entry; S.lend[S.nloop] = end; S.lc[S.nloop] = cnt & 255u; S.li[S.nloop] = inc & 255u; S.lp[S.nloop] = prev & 255u;
    S.nloop++;
}
// shader_run()'s bookkeeping after the instruction at `old` (pc: the next instruction)
static inline uint vs_flow(thread VsFlow &S, uint old, uint pc, bool brk) {
    if ((S.ncall | S.nif | S.nloop) == 0u) return pc;
    uint next = old + 1u;
    for (uint i = 0; i < 4u; i++) {
        if (S.ncall == 0u || S.ce[S.ncall - 1u] != next) break;
        if (i < 3u) { pc = S.cr[S.ncall - 1u]; next = pc; }
        S.ncall--;
    }
    if (S.nif != 0u && S.ie[S.nif - 1u] == old + 1u) { pc = S.iend[S.nif - 1u]; S.nif--; }
    if (S.nloop != 0u && (S.lend[S.nloop - 1u] == old + 1u || brk)) {
        const uint k = S.nloop - 1u;
        S.aL += int(S.li[k]);
        const uint c = S.lc[k];
        S.lc[k] = (c - 1u) & 255u;
        if (!brk && c != 0u) {
            pc = S.le[k];
        } else {
            pc = S.lend[k];
            if (S.nloop > 1u) S.aL = int(S.lp[k]);
            S.nloop--;
        }
    }
    return pc;
}
static inline void vs_output(constant VsParams &P, thread float4 *o, device float *outv, uint tid) {
    device float *d = outv + (P.out_base + tid) * 24u;
    for (int i = 0; i < 24; i++) {
        const int s = P.slot_src[i];
        float v = s >= 0 ? o[s >> 2][s & 3] : 1.0;
        if (i >= 8 && i < 12) v = min(abs(v), 1.0);
        d[i] = v;
    }
}
)";


const char *kKernelsMSL = R"(
#include <metal_stdlib>
using namespace metal;
// MSAA: fill a multisampled target from its resolved (single-sample) copy, every sample alike
struct RVOut { float4 pos [[position]]; };
vertex RVOut k_full(uint vid [[vertex_id]]) {
    RVOut o;
    const float2 p = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
fragment float4 k_rcolor(RVOut in [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) {
    return t.read(uint2(in.pos.xy));
}
struct RDOut { float d [[depth(any)]]; uint s [[stencil]]; };
fragment RDOut k_rdepth(RVOut in [[stage_in]], depth2d<float, access::read> d [[texture(0)]],
                        texture2d<uint, access::read> s [[texture(1)]]) {
    RDOut o;
    o.d = d.read(uint2(in.pos.xy));
    o.s = s.read(uint2(in.pos.xy)).r;
    return o;
}
// upright display image: p = texture width, rows used, flip (bit 0) and black keyed out (bit 1), first row
kernel void k_rot(texture2d<float, access::read> src [[texture(0)]], device uint *dst [[buffer(0)]],
                  constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.y || gid.y >= p.x) return;
    const uint row = ((p.z & 1) ? p.y - 1 - gid.x : gid.x) + p.w;
    const uint col = p.x - 1 - gid.y;
    const uint4 b = uint4(round(saturate(src.read(uint2(col, row))) * 255.0));
    uint a = b.a;
    if (p.z & 2) {  // black keyed out; what was drawn translucent over a background (now hidden) more opaque
        a = min(255u, a * 17u / 10u);
        a = min(a, uint(clamp((int(max(b.r, max(b.g, b.b))) - 3) * 64, 0, 255)));
    }
    dst[gid.y * p.y + gid.x] = b.r | (b.g << 8) | (b.b << 16) | (a << 24);
}
// Guest memory formats, as gpu.cpp and transfer.cpp handle them on the CPU (bit for bit): pixel formats 0 RGBA8,
// 1 RGB8, 2 RGB565, 3 RGB5A1, 4 RGBA4; depth formats 0 D16, 2 D24, 3 D24S8; tiled images in 8x8 tiles, Morton order.
inline uint g_bpp(uint f) { return f == 0 ? 4u : f == 1 ? 3u : 2u; }
inline uint g_dbpp(uint f) { return f == 0 ? 2u : f == 2 ? 3u : 4u; }
inline uint g_tiled(uint x, uint y, uint w, uint bpp) {
    const uint m = (x & 1) | ((y & 1) << 1) | ((x & 2) << 1) | ((y & 2) << 2) | ((x & 4) << 2) | ((y & 4) << 3);
    return ((y & ~7u) * w + (x & ~7u) * 8 + m) * bpp;
}
inline uint c5to8(uint v) { return (v << 3) | (v >> 2); }
inline uint c6to8(uint v) { return (v << 2) | (v >> 4); }
inline uint c4to8(uint v) { return (v << 4) | v; }
inline uint4 g_decode(uint f, device const uchar *p) {
    const uint v = p[0] | (uint(p[1]) << 8);
    switch (f) {
    case 0: return uint4(p[3], p[2], p[1], p[0]);
    case 1: return uint4(p[2], p[1], p[0], 255);
    case 2: return uint4(c5to8(v >> 11), c6to8((v >> 5) & 63), c5to8(v & 31), 255);
    case 3: return uint4(c5to8(v >> 11), c5to8((v >> 6) & 31), c5to8((v >> 1) & 31), (v & 1) * 255);
    default: return uint4(c4to8(v >> 12), c4to8((v >> 8) & 15), c4to8((v >> 4) & 15), c4to8(v & 15));
    }
}
inline void g_encode(uint f, uint4 c, device uchar *p) {
    uint v;
    switch (f) {
    case 0: p[0] = c.a; p[1] = c.b; p[2] = c.g; p[3] = c.r; return;
    case 1: p[0] = c.b; p[1] = c.g; p[2] = c.r; return;
    case 2: v = ((c.r >> 3) << 11) | ((c.g >> 2) << 5) | (c.b >> 3); break;
    case 3: v = ((c.r >> 3) << 11) | ((c.g >> 3) << 6) | ((c.b >> 3) << 1) | (c.a >> 7); break;
    default: v = ((c.r >> 4) << 12) | ((c.g >> 4) << 8) | ((c.b >> 4) << 4) | (c.a >> 4); break;
    }
    p[0] = v;
    p[1] = v >> 8;
}
// a colour target's GPU copy as guest memory (box-filtered to 1x): p = w, h (1x), scale, pixel format
kernel void k_enc(texture2d<float, access::read> src [[texture(0)]], device uchar *dst [[buffer(0)]],
                  constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint s = p.z;
    uint4 acc = uint4(0);
    for (uint j = 0; j < s; j++)
        for (uint i = 0; i < s; i++) acc += uint4(round(src.read(uint2(gid.x * s + i, gid.y * s + j)) * 255.0));
    const uint n = s * s;
    g_encode(p.w, (acc + n / 2) / n, dst + g_tiled(gid.x, gid.y, p.x, g_bpp(p.w)));
}
// guest memory into a colour target's GPU copy (each pixel scale x scale times): p = w, h (1x), scale, pixel format
kernel void k_dec(device const uchar *src [[buffer(0)]], texture2d<float, access::write> dst [[texture(0)]],
                  constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x * p.z || gid.y >= p.y * p.z) return;
    const uint4 c = g_decode(p.w, src + g_tiled(gid.x / p.z, gid.y / p.z, p.x, g_bpp(p.w)));
    dst.write(float4(c) / 255.0, gid);
}
// guest depth / stencil memory as scaled rows of depth and stencil values: p = w, h (1x), scale, depth format
kernel void k_dec_depth(device const uchar *src [[buffer(0)]], device float *dz [[buffer(1)]], device uchar *dsn [[buffer(2)]],
                        constant uint4 &p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint W = p.x * p.z;
    if (gid.x >= W || gid.y >= p.y * p.z) return;
    device const uchar *q = src + g_tiled(gid.x / p.z, gid.y / p.z, p.x, g_dbpp(p.w));
    const uint z = p.w == 0 ? (q[0] | (uint(q[1]) << 8)) : (q[0] | (uint(q[1]) << 8) | (uint(q[2]) << 16));
    dz[gid.y * W + gid.x] = float(z) * (p.w == 0 ? 1.0 / 65536.0 : 1.0 / 16777216.0);
    dsn[gid.y * W + gid.x] = p.w == 3 ? q[3] : 0;
}
// scaled rows of depth and stencil values (the top-left sample of each pixel) as guest memory: p as k_dec_depth
kernel void k_enc_depth(device const float *dz [[buffer(0)]], device const uchar *dsn [[buffer(1)]], device uchar *dst [[buffer(2)]],
                        constant uint4 &p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint i = gid.y * p.z * (p.x * p.z) + gid.x * p.z;
    const uint z = uint(dz[i] * (p.w == 0 ? 65536.0 : 16777216.0));
    device uchar *q = dst + g_tiled(gid.x, gid.y, p.x, g_dbpp(p.w));
    q[0] = z;
    q[1] = z >> 8;
    if (p.w != 0) q[2] = z >> 16;
    if (p.w == 3) q[3] = dsn[i];
}
// a display transfer's conversion (transfer_convert in transfer.cpp) from a tiled image in guest format: p = input
// width, output width and rows (after downscaling), flags; q.x = first input byte
kernel void k_tconv(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint4 &p [[buffer(2)]],
                    constant uint4 &q [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint out_w = p.y, out_h = p.z, flags = p.w;
    if (gid.x >= out_w || gid.y >= out_h) return;
    const bool flip = flags & 1, in_linear = (flags >> 1) & 1, dont_swizzle = (flags >> 5) & 1;
    const uint in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7, scaling = (flags >> 24) & 3;
    const uint ibpp = g_bpp(in_fmt), obpp = g_bpp(out_fmt);
    const uint hs = scaling != 0 ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const uint x = gid.x, y = gid.y, ix = x << hs, iy = y << vs, oy = flip ? out_h - 1 - y : y;
    uint so, doff;
    if (in_linear) {
        so = (ix + iy * p.x) * ibpp;
        doff = dont_swizzle ? (x + oy * out_w) * obpp : g_tiled(x, oy, out_w, obpp);
    } else {
        so = g_tiled(ix, iy, p.x, ibpp);
        doff = dont_swizzle ? g_tiled(x, oy, out_w, obpp) : (x + oy * out_w) * obpp;
    }
    device const uchar *sp = src + q.x + so;
    uint4 c = g_decode(in_fmt, sp);
    if (scaling == 1) c = (c + g_decode(in_fmt, sp + ibpp)) / 2;
    else if (scaling == 2)
        c = (c + g_decode(in_fmt, sp + ibpp) + g_decode(in_fmt, sp + 2 * ibpp) + g_decode(in_fmt, sp + 3 * ibpp)) / 4;
    g_encode(out_fmt, c, dst + doff);
}
// tiled-to-tiled display transfer between GPU copies of guest images: p = output width, rows (scaled), first source
// row (scaled); q = horizontal and vertical downscale shifts, flip, source without alpha
kernel void k_xfer(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                   constant uint4 &p [[buffer(0)]], constant uint4 &q [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint y = q.z ? p.y - 1 - gid.y : gid.y;
    const uint fx = 1u << q.x, fy = 1u << q.y;
    float4 acc = float4(0.0);
    for (uint j = 0; j < fy; j++)
        for (uint i = 0; i < fx; i++) acc += src.read(uint2(gid.x * fx + i, p.z + y * fy + j));
    acc /= float(fx * fy);
    if (q.w) acc.a = 1.0;
    dst.write(acc, gid);
}
// a GPU copy of a guest image as a texture: rows upside down (the order of decoded textures), alpha opaque for formats
// without one; p = width, rows (scaled), opaque
kernel void k_snap(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                   constant uint4 &p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x || gid.y >= p.y) return;
    float4 c = src.read(uint2(gid.x, p.y - 1 - gid.y));
    if (p.z) c.a = 1.0;
    dst.write(c, gid);
}
)";

// C++ mirror of the MSL uniform block
struct alignas(16) LightU {
    float spec0[4], spec1[4], diffuse[4], ambient[4], pos[4], spot[4], att[4];
};
struct alignas(16) Uni {
    float rt[4];
    float depth[4];
    s32 scissor[4];
    u32 konst[8];
    u32 tevc[4];
    u32 misc[4];
    u32 border[4];
    float bias[4];
    float lscale[8];
    float ambient[4];
    LightU lights[8];
};
static_assert(sizeof(Uni) == 16 * 12 + 8 * 112, "uniform layout");

// ---------------------------------------------------------------- device state
id<MTLDevice> dev;
id<MTLCommandQueue> queue;
id<MTLComputePipelineState> k_rot, k_xfer, k_snap, k_enc, k_dec, k_dec_depth, k_enc_depth, k_tconv;
u32 S = 1;                     // internal resolution scale
u32 MS = 1;                    // MSAA samples per pixel of the scaled render targets (GPU_MSAA)
// The multisampled targets live in tile memory only (memoryless): each pass resolves them into the single-sample
// targets and the next one restores its samples from those, instead of loading and storing every sample through
// memory between passes (most of the GPU's time at high scales). Earlier passes' edges stay antialiased (resolved);
// only depth loses its subsamples between passes.
bool ms_memoryless = false;
u32 aniso = 1;                 // max anisotropy of linearly filtered textures (GPU_ANISO)
bool scale_fixed = false;      // GPU_SCALE given (else the frontend sizes it to the window)
constexpr u32 kDefaultMSAA = 4, kDefaultAniso = 16;
std::atomic<u32> want_scale{0};  // scale change requested by set_scale(), applied by the GX thread
id<MTLRenderPipelineState> p_rcolor, p_rdepth;  // restore a multisampled target from its resolved copy
id<MTLDepthStencilState> dss_off, dss_rdepth;
bool active = false;
bool tried = false;

// Time the GPU spends on our command buffers: the frontend's automatic resolution backs off when the GPU is
// saturated (gpu::busy), and RECOMP_STATS logs it every 2 s from metal_poll.
bool log_timing = false;
std::atomic<float> busy_frac{0};
std::atomic<u64> vs_dispatches{0}, vs_verts{0}, vs_passes{0}, flushes{0};  // vertex kernel work, for the stats line
std::atomic<u64> cbs_made{0};                                                // command buffers, for the stats line
std::mutex busy_mtx;
double gpu_busy = 0, busy_end = 0;  // union of the command buffers' GPU execution intervals
id<MTLCommandBuffer> new_cb_raw() {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    cbs_made.fetch_add(1, std::memory_order_relaxed);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        std::lock_guard<std::mutex> lk(busy_mtx);
        const double b = std::max(c.GPUStartTime, busy_end);
        if (c.GPUEndTime > b) gpu_busy += c.GPUEndTime - b;
        busy_end = std::max(busy_end, c.GPUEndTime);
    }];
    return cb;
}

id<MTLBuffer> make_buffer(size_t n) {
    return [dev newBufferWithLength:std::max<size_t>(n, 16) options:MTLResourceStorageModeShared];
}

// The passes of a command list (and the work that prepares them) share one command buffer, committed at the end of
// the list (metal_end_list) or as soon as other GPU work is made (new_cb), which must run after them. What the CPU
// writes for them (vertices, indices, uniforms, uploads) is carved out of large shared chunks (arena), which go back
// to chunk_free when the command buffer completes.
id<MTLCommandBuffer> cur_cb;
std::vector<id<MTLBuffer>> cur_chunks;
size_t chunk_off = 0;
size_t chunk_last = SIZE_MAX;  // offset of the last slice taken from the current chunk (the only one arena_trim may shrink)
std::mutex chunk_mtx;
std::vector<id<MTLBuffer>> chunk_free;
constexpr size_t kChunkBytes = 4u << 20, kMaxFreeChunks = 16;

id<MTLCommandBuffer> list_cb() {
    if (!cur_cb) cur_cb = new_cb_raw();
    return cur_cb;
}
void commit_list_cb() {
    if (!cur_cb) return;
    std::vector<id<MTLBuffer>> chunks;
    chunks.swap(cur_chunks);
    chunk_off = 0;
    [cur_cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        std::lock_guard<std::mutex> lk(chunk_mtx);
        for (id<MTLBuffer> b : chunks)
            if (chunk_free.size() < kMaxFreeChunks) chunk_free.push_back(b);
    }];
    [cur_cb commit];
    cur_cb = nil;
}
// a command buffer for work that must come after everything encoded so far
id<MTLCommandBuffer> new_cb() {
    commit_list_cb();
    return new_cb_raw();
}

struct Slice {
    id<MTLBuffer> buf;
    size_t off;
    u8 *p;
};
// n bytes for the current list's command buffer (256-byte aligned)
Slice arena(size_t n) {
    n = (std::max<size_t>(n, 16) + 255) & ~(size_t)255;
    list_cb();
    if (cur_chunks.empty() || chunk_off + n > cur_chunks.back().length) {
        id<MTLBuffer> b = nil;
        {
            std::lock_guard<std::mutex> lk(chunk_mtx);
            for (size_t i = 0; i < chunk_free.size(); i++)
                if (chunk_free[i].length >= n) {
                    b = chunk_free[i];
                    chunk_free.erase(chunk_free.begin() + i);
                    break;
                }
        }
        if (!b) b = make_buffer(std::max(kChunkBytes, n));
        cur_chunks.push_back(b);
        chunk_off = 0;
    }
    Slice s{cur_chunks.back(), chunk_off, (u8 *)cur_chunks.back().contents + chunk_off};
    chunk_last = chunk_off;
    chunk_off += n;
    return s;
}
// gives back the end of the last slice taken (only `used` bytes of it were needed); a slice with others taken after
// it keeps its size (they lie beyond its end)
void arena_trim(const Slice &s, size_t used) {
    if (!cur_chunks.empty() && s.buf == cur_chunks.back() && s.off == chunk_last)
        chunk_off = std::min(chunk_off, s.off + ((used + 255) & ~(size_t)255));
}

// Shared buffers for data written by the CPU while earlier command buffers may still be in
// flight: taken from a pool and returned when the command buffer using them completes.
std::mutex pool_mtx;
std::vector<id<MTLBuffer>> pool_free;
id<MTLBuffer> pool_get(size_t n, id<MTLCommandBuffer> cb) {
    id<MTLBuffer> b = nil;
    {
        std::lock_guard<std::mutex> lk(pool_mtx);
        for (size_t i = 0; i < pool_free.size(); i++)
            if (pool_free[i].length >= n) {
                b = pool_free[i];
                pool_free.erase(pool_free.begin() + i);
                break;
            }
    }
    if (!b) b = make_buffer(n + n / 4 + 4096);
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        std::lock_guard<std::mutex> lk(pool_mtx);
        if (pool_free.size() < 32) pool_free.push_back(b);
    }];
    return b;
}
void ensure_buffer(id<MTLBuffer> __strong &b, size_t n) {
    if (!b || b.length < n) b = make_buffer(n + n / 2 + 4096);
}

// ---------------------------------------------------------------- render targets
struct GRT {
    u32 addr = 0, w = 0, h = 0, fmt = 0;
    bool depth = false;
    id<MTLTexture> tex;
    id<MTLTexture> stencil_view;  // tex's stencil as a texture (MSAA restore)
    id<MTLTexture> ms;       // with MSAA: the multisampled target, resolved into tex at the end of each pass
    bool ms_valid = false;   // ms matches tex (else it is restored from tex before drawing; never with ms_memoryless)
    bool synced = false;     // GPU copy corresponds to guest memory as hashed below
    u64 hash = 0;
    u64 checked = 0;         // g_frame when guest memory last matched the hash
    u64 id = 0;              // unique among all targets ever made
    u64 version = 0;         // changes whenever the GPU copy does
    bool gpu_newer = false;  // depth/stencil changes not yet written to guest memory
    u32 skipped = 0;         // draws left out since its last display transfer (pipelines still compiling)
    bool clear_next = false; // set_clear_after_top_transfer: cleared to transparent before it is drawn into again,
    bool own = false;        // ... and its GPU copy kept (guest memory ignored) until its own next display transfer
};
std::atomic<bool> clear_after_top{false};
std::atomic<bool> key_black_bottom{false};  // set_key_black_bottom

std::vector<std::unique_ptr<GRT>> rts;

u32 rt_bytes(const GRT &r) {
    if (r.depth) return r.w * r.h * (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4);
    return r.w * r.h * pixel_bytes(cbuf_to_pixfmt(r.fmt));
}

// (set_clear_after_top_transfer) the colour targets in the same memory as r: cleared before their next draw
void clear_overlapping(GRT &r) {
    for (auto &o : rts)
        if (!o->depth && o->addr < r.addr + rt_bytes(r) && r.addr < o->addr + rt_bytes(*o)) o->clear_next = o->own = true;
}

GRT &get_rt(u32 addr, u32 w, u32 h, u32 fmt, bool depth) {
    for (auto &r : rts)
        if (r->addr == addr && r->w == w && r->h == h && r->fmt == fmt && r->depth == depth) return *r;
    static u64 next_id;
    auto r = std::make_unique<GRT>();
    r->id = ++next_id;
    r->addr = addr;
    r->w = w;
    r->h = h;
    r->fmt = fmt;
    r->depth = depth;
    MTLTextureDescriptor *d = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA8Unorm
                                     width:w * S
                                    height:h * S
                                 mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
              (depth ? MTLTextureUsagePixelFormatView : MTLTextureUsageShaderWrite);
    r->tex = [dev newTextureWithDescriptor:d];
    if (MS > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = MS;
        d.usage = MTLTextureUsageRenderTarget;
        if (ms_memoryless) d.storageMode = MTLStorageModeMemoryless;
        r->ms = [dev newTextureWithDescriptor:d];
    }
    rts.push_back(std::move(r));
    return *rts.back();
}

// scratch depth target for render targets without a guest depth buffer
std::unordered_map<u64, id<MTLTexture>> scratch_depth;
id<MTLTexture> get_scratch_depth(u32 w, u32 h) {
    const u64 key = (u64)w << 32 | h;
    auto it = scratch_depth.find(key);
    if (it != scratch_depth.end()) return it->second;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                                                 width:w * S
                                                                                height:h * S
                                                                             mipmapped:NO];
    d.storageMode = MTLStorageModeMemoryless;  // cleared at the start of each pass, never stored
    d.usage = MTLTextureUsageRenderTarget;
    if (MS > 1) {
        d.textureType = MTLTextureType2DMultisample;
        d.sampleCount = MS;
    }
    id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    scratch_depth[key] = t;
    return t;
}

id<MTLBuffer> stage_color, stage_depth, stage_stencil, stage_dbytes;

// GPU_LABELS=1: encoders are labelled (what they do and to which guest memory), for Instruments and GPU captures
bool labels_on = getenv("GPU_LABELS") != nullptr;
template <class E>
void label(E e, const char *fmt, ...) {
    if (!labels_on) return;
    char b[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    e.label = [NSString stringWithUTF8String:b];
}

// a dispatch of one thread per pixel of a w x h grid
void dispatch2d(id<MTLComputeCommandEncoder> ce, u32 w, u32 h) {
    [ce dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}
// encodes the colour target's GPU copy as guest memory (its format, tiled, 1x) into dst
void encode_guest_color(id<MTLComputeCommandEncoder> ce, const GRT &r, id<MTLBuffer> dst) {
    [ce setComputePipelineState:k_enc];
    [ce setTexture:r.tex atIndex:0];
    [ce setBuffer:dst offset:0 atIndex:0];
    const u32 prm[4] = {r.w, r.h, S, cbuf_to_pixfmt(r.fmt)};
    [ce setBytes:prm length:16 atIndex:1];
    dispatch2d(ce, r.w, r.h);
}

struct LoadAction {
    bool clear = false;
    u32 color = 0;       // packed RGBA
    float depth = 0;
    u32 stencil = 0;
};

inline float depth_quantum(u32 fmt) { return fmt == 0 ? 1.0f / 65536.0f : 1.0f / 16777216.0f; }

// make the GPU copy of a render target match guest memory (if the guest changed since the last
// sync); uniform contents become a clear of the render pass
LoadAction sync_rt(GRT &r, id<MTLCommandBuffer> cb) {
    LoadAction la;
    const u8 *p = pptr(r.addr);
    if (!p) return la;
    // Checked during this frame already: writes reported since (GPU operations, cache flushes, DMA) clear synced, and a
    // CPU write the game does not flush out of its data cache would not reach a 3DS GPU before the frame's end either.
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (r.synced && r.checked == frame) return la;
    const u64 hv = hash_mem(p, rt_bytes(r));
    r.checked = frame;
    if (r.synced && hv == r.hash) return la;
    if (r.own) {  // (cleared after a transfer to the top screen: the top screen's write-back is not its picture)
        r.hash = hv;
        r.synced = true;
        return la;
    }
    r.synced = true;
    r.hash = hv;
    r.gpu_newer = false;
    // a target filled with one value (typically a memory fill clearing it): bytes repeat every pixel
    const u32 rbpp = r.depth ? (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4) : pixel_bytes(cbuf_to_pixfmt(r.fmt));
    if (rt_bytes(r) > rbpp && memcmp(p, p + rbpp, rt_bytes(r) - rbpp) == 0) {
        la.clear = true;
        if (!r.depth) {
            la.color = decode_pixel(cbuf_to_pixfmt(r.fmt), p);
        } else {
            const u32 z = r.fmt == 0 ? (u32)(p[0] | p[1] << 8) : (u32)(p[0] | p[1] << 8 | p[2] << 16);
            la.depth = (float)z * depth_quantum(r.fmt);
            la.stencil = r.fmt == 3 ? p[3] : 0;
        }
        return la;
    }
    // otherwise the guest bytes as they are, decoded by the GPU
    const Slice raw = arena(rt_bytes(r));
    memcpy(raw.p, p, rt_bytes(r));
    r.ms_valid = false;
    r.version++;
    if (!r.depth) {
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        label(ce, "upload %08x %ux%u", r.addr, r.w, r.h);
        [ce setComputePipelineState:k_dec];
        [ce setBuffer:raw.buf offset:raw.off atIndex:0];
        [ce setTexture:r.tex atIndex:0];
        const u32 prm[4] = {r.w, r.h, S, cbuf_to_pixfmt(r.fmt)};
        [ce setBytes:prm length:16 atIndex:1];
        dispatch2d(ce, r.w * S, r.h * S);
        [ce endEncoding];
        return la;
    }
    // depth / stencil: as scaled rows of values, copied into the target
    const u32 W = r.w * S, H = r.h * S;
    const Slice upd = arena((size_t)W * H * 4), ups = arena((size_t)W * H);
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    label(ce, "upload depth %08x %ux%u", r.addr, r.w, r.h);
    [ce setComputePipelineState:k_dec_depth];
    [ce setBuffer:raw.buf offset:raw.off atIndex:0];
    [ce setBuffer:upd.buf offset:upd.off atIndex:1];
    [ce setBuffer:ups.buf offset:ups.off atIndex:2];
    const u32 prm[4] = {r.w, r.h, S, r.fmt};
    [ce setBytes:prm length:16 atIndex:3];
    dispatch2d(ce, W, H);
    [ce endEncoding];
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:upd.buf sourceOffset:upd.off sourceBytesPerRow:W * 4 sourceBytesPerImage:(NSUInteger)W * H * 4
            sourceSize:MTLSizeMake(W, H, 1) toTexture:r.tex destinationSlice:0 destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0) options:MTLBlitOptionDepthFromDepthStencil];
    [be copyFromBuffer:ups.buf sourceOffset:ups.off sourceBytesPerRow:W sourceBytesPerImage:(NSUInteger)W * H
            sourceSize:MTLSizeMake(W, H, 1) toTexture:r.tex destinationSlice:0 destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0) options:MTLBlitOptionStencilFromDepthStencil];
    [be endEncoding];
    return la;
}

// the GX thread waits for the GPU (GPU_STATS: logged when it actually had to wait)
void wait_cb(id<MTLCommandBuffer> cb, const char *why, u32 addr) {
    if (!g_stats) {
        [cb waitUntilCompleted];
        return;
    }
    const double t0 = now_ms();
    [cb waitUntilCompleted];
    const double t = now_ms() - t0;
    if (t > 0.05) LOG("gpu metal: waited %.2f ms for the GPU (%s %08x)\n", t, why, addr);
}

// Before writing a target back: the guest memory must still hold what it held when the GPU copy took over. A
// game that freed a target and reused its memory (heap objects, other buffers) has written it since, and on the
// 3DS those later CPU writes land on top of what the GPU drew: the guest's bytes win, the GPU copy is dropped.
bool guest_kept(GRT &r) {
    if (hash_mem(pptr(r.addr), rt_bytes(r)) == r.hash) return true;
    static int logged;
    if (logged < 8 || g_stats) {
        logged++;
        LOG("gpu metal: %s target %08x (%ux%u) was rewritten by the game since it was drawn; not written back\n",
            r.depth ? "depth" : "colour", r.addr, r.w, r.h);
    }
    r.gpu_newer = false;
    r.synced = false;
    return false;
}

// write a colour target's GPU contents back to guest memory (box-filtered to 1x)
void write_back_color(GRT &r) {
    if (!guest_kept(r)) return;
    ensure_buffer(stage_color, rt_bytes(r));
    id<MTLCommandBuffer> cb = new_cb();
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    label(ce, "write-back %08x %ux%u", r.addr, r.w, r.h);
    encode_guest_color(ce, r, stage_color);
    [ce endEncoding];
    [cb commit];
    wait_cb(cb, "colour write-back", r.addr);  // also waits for all earlier work on the queue
    r.hash = copy_hash(pptr(r.addr), (const u8 *)stage_color.contents, rt_bytes(r));
    r.checked = g_frame.load(std::memory_order_relaxed);
    r.synced = true;
    r.gpu_newer = false;
}

// write a depth target's GPU contents back to guest memory (top-left sample of each pixel)
void write_back_depth(GRT &r) {
    if (!guest_kept(r)) return;
    const u32 W = r.w * S, H = r.h * S;
    ensure_buffer(stage_depth, (size_t)W * H * 4);
    ensure_buffer(stage_stencil, (size_t)W * H);
    id<MTLCommandBuffer> cb = new_cb();
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromTexture:r.tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(W, H, 1) toBuffer:stage_depth destinationOffset:0
    destinationBytesPerRow:W * 4 destinationBytesPerImage:(NSUInteger)W * H * 4
                options:MTLBlitOptionDepthFromDepthStencil];
    [be copyFromTexture:r.tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(W, H, 1) toBuffer:stage_stencil destinationOffset:0
    destinationBytesPerRow:W destinationBytesPerImage:(NSUInteger)W * H
                options:MTLBlitOptionStencilFromDepthStencil];
    [be endEncoding];
    // into guest memory's format (unchanged where a format has no stencil or fewer depth bytes)
    ensure_buffer(stage_dbytes, rt_bytes(r));
    u8 *p = pptr(r.addr);
    memcpy(stage_dbytes.contents, p, rt_bytes(r));
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    label(ce, "write-back depth %08x %ux%u", r.addr, r.w, r.h);
    [ce setComputePipelineState:k_enc_depth];
    [ce setBuffer:stage_depth offset:0 atIndex:0];
    [ce setBuffer:stage_stencil offset:0 atIndex:1];
    [ce setBuffer:stage_dbytes offset:0 atIndex:2];
    const u32 prm[4] = {r.w, r.h, S, r.fmt};
    [ce setBytes:prm length:16 atIndex:3];
    dispatch2d(ce, r.w, r.h);
    [ce endEncoding];
    [cb commit];
    wait_cb(cb, "depth write-back", r.addr);
    r.hash = copy_hash(p, (const u8 *)stage_dbytes.contents, rt_bytes(r));
    r.gpu_newer = false;
    r.checked = g_frame.load(std::memory_order_relaxed);
    r.synced = true;
}

// ---------------------------------------------------------------- textures and samplers
// Decoded textures as Metal textures, by guest address and layout. With a texture filter (set_texture_filter,
// GPU_TEX_FILTER) the 2D textures the game samples are upscaled once per decoded content by compute kernels
// (gpu/texfilter.mm): level 0 is the upscaled image, the next levels down to the original size are reduced from
// it, and the game's own levels follow (its level l is level l + ul), so get_sampler shifts the LOD clamps.
struct MTex {
    u64 hash = 0;
    id<MTLTexture> tex;
    int req = 0;               // texture filter requested when it was made (0: none)
    u32 ul = 0;                // log2 of the upscale factor (0: not upscaled)
    u32 wraps = 0;             // wrap modes of the neighbourhood it was upscaled with
    size_t bytes = 0;          // GPU memory of an upscaled texture (with src and mid)
    u64 last_use = 0;          // flush_no
    u32 changes = 0;           // content changes seen
    u64 win_start = 0;         // complete re-upscales since win_start (flush_no): textures the game keeps
    u32 win_changes = 0;       // rewriting completely (video) are no longer upscaled
    // A texture the game has rewritten (animated tiles in an atlas) keeps what its upscaled image was made
    // from, so that the next updates only redo the 8x8 tiles that changed.
    id<MTLTexture> src, mid;   // level 0 as decoded; the sharp filter's 2x image (at 4x)
    std::vector<u32> prev;     // level 0 as decoded (CPU copy, compared with the new contents)
};
std::unordered_map<u64, MTex> tex_cache;
// (sizes are below 2048, formats and level counts below 16)
u64 tex_key(const TexEntry &e) { return (u64)e.addr << 32 | e.w << 20 | e.h << 8 | e.fmt << 4 | e.levels; }
std::atomic<int> want_filter{0};  // set_texture_filter (any thread), applied by the GX thread at the next flush
int tf_mode = 0;
u64 flush_no = 0;
size_t tf_bytes = 0;                      // upscaled textures' memory
constexpr size_t kTfBudget = 512u << 20;  // beyond: least recently used ones evicted, else no more upscaling
id<MTLCommandBuffer> tf_cb;               // this flush's upscaling work, committed before its draws
u32 st_uploads = 0, st_upscaled = 0, st_partial = 0, st_tiles = 0;  // RECOMP_STATS
double st_upload_ms = 0, st_upscale_ms = 0, st_upscale_gpu = 0;      // st_upscale_gpu: under busy_mtx

// Guest memory holding GPU output (render targets, or transfer / copy outputs made from them, as long as the
// bytes are unchanged): render-to-texture, not upscaled. Only tracked while a texture filter is on.
struct GpuOut {
    u32 addr, size;
    u64 hash;
};
std::deque<GpuOut> gpu_outs;
bool gpu_made(u32 addr, u32 size) {
    auto ov = [](u32 a, u32 an, u32 b, u32 bn) { return a < b + bn && b < a + an; };
    for (auto &r : rts)
        if (ov(r->addr, rt_bytes(*r), addr, size) && (r->gpu_newer || hash_mem(pptr(r->addr), rt_bytes(*r)) == r->hash))
            return true;
    for (const GpuOut &g : gpu_outs)
        if (ov(g.addr, g.size, addr, size) && hash_mem(pptr(g.addr), g.size) == g.hash) return true;
    return false;
}
void note_gpu_out(u32 addr, u32 size) {
    if (!pptr(addr) || !pptr(addr + size - 1)) return;
    for (auto it = gpu_outs.begin(); it != gpu_outs.end(); ++it)
        if (it->addr == addr) {
            gpu_outs.erase(it);
            break;
        }
    gpu_outs.push_back({addr, size, hash_mem(pptr(addr), size)});
    if (gpu_outs.size() > 64) gpu_outs.pop_front();
}

// upscale factor of a texture (0: not upscaled): 4x at internal scales above 2, else 2x, up to 1024x1024;
// never lookup / gradient tables (thin or tiny) or HILO8 (vectors, not colours)
u32 tf_factor(const TexEntry *e) {
    const u32 w = e->w, h = e->h, n = w * h;
    if (e->fmt == 6 || n < 256 || w >= 8 * h || h >= 8 * w || n > 1024 * 1024) return 0;
    return S > 2 ? 4 : 2;
}
u32 tf_wrap(u32 wrap) { return wrap == 2 || wrap == 6 || wrap == 7 ? 1 : wrap == 3 ? 2 : 0; }

void tf_drop_all() {
    tex_cache.clear();
    tf_bytes = 0;
}

void tf_cb_begin() {
    if (tf_cb) return;
    tf_cb = new_cb_raw();  // (while the list's command buffer is being encoded; committed before it)
    if (log_timing)
        [tf_cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
            std::lock_guard<std::mutex> lk(busy_mtx);
            st_upscale_gpu += c.GPUEndTime - c.GPUStartTime;
        }];
}

// The 8x8 tiles of an image (w x h texels) that differ between prev and cur, grown by one tile (the filters
// read up to 3 texels around: tiles next to a changed one change too; across the edges when the texture
// repeats), one rectangle per tile in rc (the kernels dispatch a grid of the largest one per rectangle);
// copies them from cur to prev. Returns the number of tiles.
u32 changed_tiles(std::vector<u32> &prev, const std::vector<u32> &cur, u32 w, u32 h, u32 wraps,
                  std::vector<TfRect> &rc) {
    const u32 tw = w / 8, th = h / 8;
    std::vector<u8> d((size_t)tw * th), g((size_t)tw * th);
    bool any = false;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw; tx++)
            for (u32 y = 0; y < 8; y++) {
                const size_t o = (size_t)(ty * 8 + y) * w + tx * 8;
                if (memcmp(&prev[o], &cur[o], 32)) {
                    d[ty * tw + tx] = 1;
                    any = true;
                    break;
                }
            }
    if (!any) return 0;
    const bool rs = (wraps & 3) == 1, rt = (wraps >> 2) == 1;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw; tx++) {
            if (!d[ty * tw + tx]) continue;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int x = (int)tx + dx, y = (int)ty + dy;
                    if (x < 0 || x >= (int)tw) {
                        if (!rs) continue;
                        x = (x + tw) % tw;
                    }
                    if (y < 0 || y >= (int)th) {
                        if (!rt) continue;
                        y = (y + th) % th;
                    }
                    g[y * tw + x] = 1;
                }
        }
    u32 n = 0;
    for (u32 ty = 0; ty < th; ty++)
        for (u32 tx = 0; tx < tw;) {
            if (!g[ty * tw + tx]) {
                tx++;
                continue;
            }
            u32 e = tx;
            while (e < tw && g[ty * tw + e]) e++;
            for (u32 x = tx; x < e; x++) rc.push_back({x * 8, ty * 8, 8, 8});
            for (u32 y = 0; y < 8; y++) {
                const size_t o = (size_t)(ty * 8 + y) * w + tx * 8;
                memcpy(&prev[o], &cur[o], (e - tx) * 32);
            }
            n += e - tx;
            tx = e;
        }
    return n;
}

// up: the unit sampling it, if it may be upscaled (2D / projected textures); *ul: log2 of the upscale factor
id<MTLTexture> get_texture(const TexEntry *e, const TexUnitState *up = nullptr, u32 *ul = nullptr) {
    if (ul) *ul = 0;
    if (!e || e->lv.empty() || e->lv[0].data.empty()) return nil;
    MTex &m = tex_cache[tex_key(*e)];
    const int want = up ? tf_mode : 0;
    if (m.tex && m.hash == e->hash && m.req == want) {
        m.last_use = flush_no;
        if (ul) *ul = m.ul;
        return m.tex;
    }
    const double t0 = log_timing ? now_ms() : 0;
    const bool changed = m.tex && m.hash != e->hash;
    if (changed) m.changes++;
    m.last_use = flush_no;
    u32 nl = 0;
    while (nl < e->lv.size() && !e->lv[nl].data.empty() && e->lv[nl].w >= 1) nl++;
    u32 f = want ? tf_factor(e) : 0;
    const u32 u = f == 4 ? 2 : f == 2 ? 1 : 0;
    const u32 wraps = up ? tf_wrap(up->wrap_s) | tf_wrap(up->wrap_t) << 2 : 0;
    const u32 w = e->w, h = e->h;

    // an upscaled texture of which only some tiles changed: redo those
    if (changed && f && m.req == want && m.ul == u && m.wraps == wraps && m.src && nl == 1 && m.prev.size() == (size_t)w * h) {
        std::vector<TfRect> rc;
        const u32 nt = changed_tiles(m.prev, e->lv[0].data, w, h, wraps, rc);
        if (nt * 2 <= (w / 8) * (h / 8)) {
            if (nt) {
                tf_cb_begin();
                size_t total = 0;
                for (const TfRect &r : rc) total += (size_t)r.w * r.h * 4;
                id<MTLBuffer> sb = pool_get(total, tf_cb);
                id<MTLBlitCommandEncoder> be = [tf_cb blitCommandEncoder];
                size_t off = 0;
                for (const TfRect &r : rc) {
                    u8 *dp = (u8 *)sb.contents + off;
                    for (u32 y = 0; y < r.h; y++)
                        memcpy(dp + (size_t)y * r.w * 4, &e->lv[0].data[(size_t)(r.y + y) * w + r.x], r.w * 4);
                    [be copyFromBuffer:sb sourceOffset:off sourceBytesPerRow:r.w * 4 sourceBytesPerImage:(size_t)r.w * r.h * 4
                            sourceSize:MTLSizeMake(r.w, r.h, 1) toTexture:m.src destinationSlice:0 destinationLevel:0
                     destinationOrigin:MTLOriginMake(r.x, r.y, 0)];
                    off += (size_t)r.w * r.h * 4;
                }
                [be endEncoding];
                texfilter_encode(dev, tf_cb, tf_mode, m.src, m.tex, m.mid, f, wraps & 3, wraps >> 2, rc.data(), (u32)rc.size());
                be = [tf_cb blitCommandEncoder];
                for (const TfRect &r : rc)  // the game's level 0
                    [be copyFromTexture:m.src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(r.x, r.y, 0)
                             sourceSize:MTLSizeMake(r.w, r.h, 1) toTexture:m.tex destinationSlice:0 destinationLevel:u
                      destinationOrigin:MTLOriginMake(r.x, r.y, 0)];
                [be endEncoding];
            }
            m.hash = e->hash;
            if (ul) *ul = u;
            if (log_timing) {
                st_partial++;
                st_tiles += nt;
                st_upscale_ms += now_ms() - t0;
            }
            return m.tex;
        }
    }

    if (changed) {
        if (flush_no - m.win_start > 600) {
            m.win_start = flush_no;
            m.win_changes = 0;
        }
        m.win_changes++;
    }
    tf_bytes -= m.bytes;
    m.bytes = 0;
    m.ul = 0;
    m.req = want;
    m.wraps = wraps;
    m.hash = e->hash;
    m.src = m.mid = nil;
    m.prev.clear();
    m.prev.shrink_to_fit();
    if (f && (m.win_changes > 6 || gpu_made(e->addr, e->size))) f = 0;
    // rewritten before: keep what partial updates need
    const bool keep = f && m.changes > 0 && nl == 1;
    size_t bytes = 0;
    for (u32 l = 0; l < nl + u; l++) bytes += (size_t)std::max(1u, (w * f) >> l) * std::max(1u, (h * f) >> l) * 4;
    if (keep) bytes += (size_t)w * h * 4 * (tf_mode == kTexFilterSharp && f == 4 ? 5 : 1);
    if (f && tf_bytes + bytes > kTfBudget) {
        for (auto it = tex_cache.begin(); it != tex_cache.end();)
            if (it->second.bytes && it->second.last_use + 120 < flush_no) {
                tf_bytes -= it->second.bytes;
                it = tex_cache.erase(it);
            } else {
                ++it;
            }
        if (tf_bytes + bytes > kTfBudget) f = 0;
    }
    if (f) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:w * f
                                                                                    height:h * f
                                                                                 mipmapped:YES];
        d.mipmapLevelCount = nl + u;
        d.storageMode = MTLStorageModePrivate;  // GPU layout: these are large, sampled a lot
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> t = [dev newTextureWithDescriptor:d];
        // the decoded texture, upscaled from and copied below the upscaled levels
        MTLTextureDescriptor *sd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:w
                                                                                     height:h
                                                                                  mipmapped:nl > 1];
        if (nl > 1) sd.mipmapLevelCount = nl;
        sd.storageMode = MTLStorageModeShared;
        sd.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> src = [dev newTextureWithDescriptor:sd];
        for (u32 l = 0; l < nl; l++) {
            const TexImage &im = e->lv[l];
            if (!im.w || !im.h) continue;
            [src replaceRegion:MTLRegionMake2D(0, 0, im.w, im.h) mipmapLevel:l withBytes:im.data.data()
                   bytesPerRow:im.w * 4];
        }
        id<MTLTexture> mid = keep && tf_mode == kTexFilterSharp && f == 4 ? texfilter_new_mid(dev, w, h) : nil;
        tf_cb_begin();
        if (t && src && texfilter_encode(dev, tf_cb, tf_mode, src, t, mid, f, wraps & 3, wraps >> 2)) {
            id<MTLBlitCommandEncoder> be = [tf_cb blitCommandEncoder];
            for (u32 l = 0; l < nl; l++) {
                const TexImage &im = e->lv[l];
                if (!im.w || !im.h) continue;
                [be copyFromTexture:src sourceSlice:0 sourceLevel:l sourceOrigin:MTLOriginMake(0, 0, 0)
                         sourceSize:MTLSizeMake(im.w, im.h, 1) toTexture:t destinationSlice:0 destinationLevel:l + u
                  destinationOrigin:MTLOriginMake(0, 0, 0)];
            }
            [be endEncoding];
            m.tex = t;
            m.ul = u;
            m.bytes = bytes;
            tf_bytes += bytes;
            if (keep) {
                m.src = src;
                m.mid = mid;
                m.prev = e->lv[0].data;
            }
            if (ul) *ul = u;
            if (log_timing) {
                st_upscaled++;
                st_upscale_ms += now_ms() - t0;
            }
            return m.tex;
        }
        want_filter = tf_mode = 0;  // no kernels: filter off
        m.req = 0;
        m.bytes = 0;
    }
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:w
                                                                                height:h
                                                                             mipmapped:nl > 1];
    if (nl > 1) d.mipmapLevelCount = nl;
    d.storageMode = MTLStorageModeShared;
    d.usage = MTLTextureUsageShaderRead;
    m.tex = [dev newTextureWithDescriptor:d];
    for (u32 l = 0; l < nl; l++) {
        const TexImage &im = e->lv[l];
        [m.tex replaceRegion:MTLRegionMake2D(0, 0, im.w, im.h) mipmapLevel:l withBytes:im.data.data()
                 bytesPerRow:im.w * 4];
    }
    if (log_timing) {
        st_uploads++;
        st_upload_ms += now_ms() - t0;
    }
    return m.tex;
}

std::unordered_map<u64, id<MTLSamplerState>> sampler_cache;
MTLSamplerAddressMode addr_mode(u32 wrap) {
    switch (wrap) {
    case 2:
    case 6:
    case 7: return MTLSamplerAddressModeRepeat;
    case 3: return MTLSamplerAddressModeMirrorRepeat;
    default: return MTLSamplerAddressModeClampToEdge;  // border modes are emulated in the shader
    }
}
// ul: log2 of the texture's upscale factor (its first ul levels come before the game's level 0)
id<MTLSamplerState> get_sampler(const TexUnitState &tu, u32 levels, u32 ul = 0) {
    const u64 key = (u64)ul << 28 | (u64)aniso << 22 | (u64)tu.mag_linear | (u64)tu.min_linear << 1 | (u64)tu.mip_linear << 2 | (u64)tu.wrap_s << 3 |
                    (u64)tu.wrap_t << 6 | (u64)levels << 9 | (u64)(tu.min_level & 15) << 14 |
                    (u64)(tu.max_level & 15) << 18;
    auto it = sampler_cache.find(key);
    if (it != sampler_cache.end()) return it->second;
    const u32 glevels = levels - ul;  // the game's
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.magFilter = tu.mag_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.minFilter = tu.min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = glevels > 1 ? (tu.mip_linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest)
                              : MTLSamplerMipFilterNotMipmapped;
    d.lodMinClamp = (float)tu.min_level;
    d.lodMaxClamp = glevels > 1 ? (float)std::max(tu.min_level, tu.max_level) : 0.0f;
    if (ul) {
        // upscaled: what the game magnifies may now be minified (LOD 0..ul), so the minification filter follows
        // the magnification one; the game's levels moved down by ul
        if (tu.mag_linear) d.minFilter = MTLSamplerMinMagFilterLinear;
        d.mipFilter = glevels > 1 && !tu.mip_linear ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
        if (!tu.mag_linear && !tu.min_linear) d.mipFilter = MTLSamplerMipFilterNearest;
        d.lodMinClamp = tu.min_level > 0 ? (float)(tu.min_level + ul) : 0.0f;
        d.lodMaxClamp += (float)ul;
    }
    d.sAddressMode = addr_mode(tu.wrap_s);
    d.tAddressMode = addr_mode(tu.wrap_t);
    // anisotropic filtering where the game asked for linear minification of a mipmapped texture
    if (aniso > 1 && tu.min_linear && glevels > 1) d.maxAnisotropy = aniso;
    id<MTLSamplerState> s = [dev newSamplerStateWithDescriptor:d];
    sampler_cache[key] = s;
    return s;
}

// ---------------------------------------------------------------- pipeline cache
// A pipeline is fully described by its fs_key(). Every key a game uses is recorded in a per-user cache
// file, and a list can be shipped with the game (games/<title id>/pipelines.bin). The build compiles the
// shipped list's fragment shaders ahead of time into shaders.metallib (functions named after a hash of
// their MSL source, so a stale library is never used for a changed generator), and all known pipelines are
// created on background threads at startup. A pipeline never seen before is compiled in the background
// and its draws are skipped until it is ready (GPU_SYNC_SHADERS=1 compiles it on the spot instead); the screen
// keeps the previous image while a frame lacks draws.
struct Pipe {
    id<MTLRenderPipelineState> ps = nil;
    int state = 0;       // 0 queued, 1 compiling, 2 done (ps is nil if it failed)
    bool known = false;  // from a pipeline list: wait for it rather than skipping its draws
};
std::mutex pipe_mtx;
std::condition_variable pipe_cv;
std::unordered_map<std::string, std::unique_ptr<Pipe>> pipe_cache;
std::deque<std::string> pipe_queue;
int pipe_workers = 0;
id<MTLLibrary> pre_lib;  // shaders.metallib
id<MTLFunction> pre_vs;
FILE *key_log;  // the user's pipeline list, appended to
bool dump_shaders = false, sync_shaders = false, verify_keys = false;
struct {
    u32 lib = 0, src = 0, fresh = 0, waited = 0, skipped = 0;
    double lib_ms = 0, src_ms = 0, max_ms = 0, wait_ms = 0;
} pstats;

id<MTLRenderPipelineState> build_pipeline(const DrawState &ds) {
    @autoreleasepool {
        const double t0 = now_ms();
        const std::string src = gen_fs(ds);
        if (dump_shaders) LOG("gpu metal: new fragment shader:\n%s\n", src.c_str());
        NSError *err = nil;
        id<MTLFunction> vf = nil, ff = nil;
        if (pre_lib) {
            ff = [pre_lib newFunctionWithName:[NSString stringWithUTF8String:fs_name(src).c_str()]];
            vf = pre_vs;
        }
        const bool from_lib = ff != nil;
        if (!ff) {
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:(src + kVertexMSL).c_str()]
                                                   options:safe_math_options()
                                                     error:&err];
            if (!lib) {
                LOG("gpu metal: shader compile failed: %s\n%s\n", err.localizedDescription.UTF8String, src.c_str());
                return nil;
            }
            vf = [lib newFunctionWithName:@"vs_main"];
            ff = [lib newFunctionWithName:@"fs_main"];
        }
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = vf;
        pd.fragmentFunction = ff;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        MTLColorWriteMask wm = MTLColorWriteMaskNone;
        if (ds.color_write && ds.need_shade) {
            if (!hw_blend(ds)) {
                wm = MTLColorWriteMaskAll;  // masking done in the shader
            } else {
                if (ds.color_mask & 0xff) wm |= MTLColorWriteMaskRed;
                if (ds.color_mask & 0xff00) wm |= MTLColorWriteMaskGreen;
                if (ds.color_mask & 0xff0000) wm |= MTLColorWriteMaskBlue;
                if (ds.color_mask & 0xff000000u) wm |= MTLColorWriteMaskAlpha;
            }
        }
        pd.colorAttachments[0].writeMask = wm;
        if (ds.need_shade && ds.blend && hw_blend(ds)) {
            static const MTLBlendFactor fac[15] = {
                MTLBlendFactorZero, MTLBlendFactorOne, MTLBlendFactorSourceColor, MTLBlendFactorOneMinusSourceColor,
                MTLBlendFactorDestinationColor, MTLBlendFactorOneMinusDestinationColor, MTLBlendFactorSourceAlpha,
                MTLBlendFactorOneMinusSourceAlpha, MTLBlendFactorDestinationAlpha, MTLBlendFactorOneMinusDestinationAlpha,
                MTLBlendFactorBlendColor, MTLBlendFactorOneMinusBlendColor, MTLBlendFactorBlendAlpha,
                MTLBlendFactorOneMinusBlendAlpha, MTLBlendFactorSourceAlphaSaturated};
            static const MTLBlendOperation ops[5] = {MTLBlendOperationAdd, MTLBlendOperationSubtract,
                                                     MTLBlendOperationReverseSubtract, MTLBlendOperationMin,
                                                     MTLBlendOperationMax};
            auto F = [](u32 f) { return fac[f < 15 ? f : 2]; };
            MTLRenderPipelineColorAttachmentDescriptor *ca = pd.colorAttachments[0];
            ca.blendingEnabled = YES;
            ca.rgbBlendOperation = ops[ds.eq_rgb];
            ca.alphaBlendOperation = ops[ds.eq_a];
            ca.sourceRGBBlendFactor = F(ds.sf_rgb);
            ca.destinationRGBBlendFactor = F(ds.df_rgb);
            ca.sourceAlphaBlendFactor = F(ds.sf_a);
            ca.destinationAlphaBlendFactor = F(ds.df_a);
        }
        pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
        pd.rasterSampleCount = MS;
        id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
        if (!ps) LOG("gpu metal: pipeline creation failed: %s\n", err.localizedDescription.UTF8String);
        const double dt = now_ms() - t0;
        std::lock_guard<std::mutex> lk(pipe_mtx);
        (from_lib ? pstats.lib : pstats.src)++;
        (from_lib ? pstats.lib_ms : pstats.src_ms) += dt;
        pstats.max_ms = std::max(pstats.max_ms, dt);
        return ps;
    }
}

void pipe_worker() {
    std::unique_lock<std::mutex> lk(pipe_mtx);
    for (;;) {
        pipe_cv.wait(lk, [] { return !pipe_queue.empty(); });
        const std::string key = std::move(pipe_queue.front());
        pipe_queue.pop_front();
        Pipe &p = *pipe_cache[key];
        if (p.state != 0) continue;
        p.state = 1;
        lk.unlock();
        id<MTLRenderPipelineState> ps = build_pipeline(decode_key(key));
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
        static bool reported = false;
        if (!reported && pipe_queue.empty() && pstats.lib + pstats.src == pipe_cache.size() && (reported = true))
            LOG("gpu metal: %zu pipelines ready (%u precompiled: %.1f ms avg, %u from source: %.1f ms avg, "
                "slowest %.1f ms; draws waited %u times, %.1f ms in all)\n",
                pipe_cache.size(), pstats.lib, pstats.lib ? pstats.lib_ms / pstats.lib : 0.0, pstats.src,
                pstats.src ? pstats.src_ms / pstats.src : 0.0, pstats.max_ms, pstats.waited, pstats.wait_ms);
    }
}
// with pipe_mtx held
void start_workers() {
    if (pipe_workers) return;
    pipe_workers = (int)std::clamp(std::thread::hardware_concurrency() / 2, 2u, 6u);
    for (int i = 0; i < pipe_workers; i++) std::thread(pipe_worker).detach();
}

// with pipe_mtx held
void record_key(const std::string &key) {
    if (!key_log) return;
    const u32 n = (u32)key.size();
    fwrite(&n, 4, 1, key_log);
    fwrite(key.data(), 1, n, key_log);
    fflush(key_log);
}

void verify_key(const std::string &key, const DrawState &ds) {
    const DrawState d = decode_key(key);
    if (fs_key(d) != key) LOG("gpu metal: KEY MISMATCH: fs_key(decode_key(k)) != k\n");
    else if (gen_fs(d) != gen_fs(ds)) LOG("gpu metal: KEY MISMATCH: decoded state generates another shader\n");
}

std::string exe_dir() {
    char exe[4096];
    uint32_t len = sizeof exe;
    if (_NSGetExecutablePath(exe, &len) != 0) return ".";
    return std::filesystem::path(exe).parent_path().string();
}

void vs_kernels_init(const std::string &tid);
// the shipped and user pipeline lists, the precompiled library, and the background workers
void pipelines_init() {
    sync_shaders = getenv("GPU_SYNC_SHADERS") != nullptr;
    verify_keys = getenv("GPU_VERIFY_KEYS") != nullptr;
    char tid[20];
    snprintf(tid, sizeof tid, "%016llx", (unsigned long long)g_game.program_id);
    const double t0 = now_ms();
    if (getenv("GPU_NO_PRECOMPILE")) {
        LOG("gpu metal: GPU_NO_PRECOMPILE: no precompiled shaders or known pipelines\n");
        return;
    }
    // precompiled fragment shaders
    if (!getenv("GPU_NO_METALLIB"))
        for (const auto &p : {g_game.dir + "/shaders.metallib", exe_dir() + "/shaders.metallib",
                              exe_dir() + "/../Resources/shaders.metallib"}) {  // (the game's own: in a multi-game app)
            if (!std::filesystem::exists(p)) continue;
            NSError *err = nil;
            pre_lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]] error:&err];
            if (pre_lib) pre_vs = [pre_lib newFunctionWithName:@"vs_main"];
            if (!pre_vs) pre_lib = nil;
            if (pre_lib)
                LOG("gpu metal: precompiled shaders from %s (%lu functions)\n", p.c_str(),
                    (unsigned long)pre_lib.functionNames.count);
            break;
        }
    {
        std::error_code ec;
        std::filesystem::create_directories(g_save_base + "/shadercache", ec);
    }
    vs_kernels_init(tid);
    // known pipelines: shipped with the game, then the user's own
    std::vector<std::string> keys;
    for (const auto &p : {g_game.dir + "/pipelines.bin", std::string(RECOMP_SOURCE_DIR) + "/games/" + tid + "/pipelines.bin"}) {
        if (!std::filesystem::exists(p)) continue;
        keys = read_keys(p);
        LOG("gpu metal: %zu pipelines listed in %s\n", keys.size(), p.c_str());
        break;
    }
    const size_t shipped = keys.size();
    const std::string dir = g_save_base + "/shadercache", user = dir + "/" + tid + ".bin";
    std::vector<std::string> mine = read_keys(user);
    keys.insert(keys.end(), mine.begin(), mine.end());
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (mine.empty()) {
        // new or outdated: start over
        key_log = fopen(user.c_str(), "wb");
        const u32 hdr[2] = {kKeyMagic, kKeyVersion};
        if (key_log) fwrite(hdr, 4, 2, key_log);
    } else {
        key_log = fopen(user.c_str(), "ab");
    }
    std::lock_guard<std::mutex> lk(pipe_mtx);
    for (auto &k : keys) {
        auto &p = pipe_cache[k];
        if (p) continue;
        p = std::make_unique<Pipe>();
        p->known = true;
        pipe_queue.push_back(k);
    }
    if (!pipe_queue.empty()) start_workers();
    pipe_cv.notify_all();
    LOG("gpu metal: %zu known pipelines (%zu shipped, %zu in %s), building in the background%s (%.1f ms)\n",
        pipe_cache.size(), shipped, mine.size(), user.c_str(), sync_shaders ? "; new ones synchronously" : "",
        now_ms() - t0);
}


// ---------------------------------------------------------------- vertex program kernels
// Vertex programs run as compute kernels (shader_msl()). A program seen for the first time is compiled in
// the background while the CPU shades its draws; programs are recorded like pipelines (blob: entry, code
// size, swizzle size, code words, swizzle words) and the shipped ones are in shaders.metallib.
// A blob may end with a specialization (SPEC, bool uniforms, int uniforms): the kernel is then
// shader_msl_spec()'s for those uniforms, much faster on the GPU; while it compiles, draws use the
// program's generic kernel if that one is ready.
struct VsKern {
    id<MTLComputePipelineState> ps = nil;
    bool done = false;
};
std::recursive_mutex vk_mtx;  // recursive: synchronous builds (GPU_SYNC_SHADERS) run under it
std::unordered_map<u64, std::unique_ptr<VsKern>> vk_cache;  // by program hash and entry
FILE *vk_log;
struct {
    u32 lib = 0, src = 0, fail = 0, cpu_draws = 0;
    double ms = 0;
} vkstats;

// the kernel's source (named vk_<hash of the source>) or "" if the program cannot be translated
std::string vs_source(const VsBlob &v, std::string &name) {
    ShaderProgram *p = shader_decode(v.code.data(), v.code_size, v.swz.data());
    std::string src = v.spec ? shader_msl_spec(p, v.entry, "vk_main", v.sp.b, v.sp.iu) : std::string();
    if (src.empty()) src = shader_msl(p, v.entry, "vk_main");
    shader_free(p);
    if (src.empty()) return src;
    char b[24];
    snprintf(b, sizeof b, "vk_%016llx", (unsigned long long)fnv64(src));
    name = b;
    src.replace(src.find("vk_main("), 7, name);
    return src;
}

void build_vs_kernel(VsKern *k, const std::string &blob) {
    @autoreleasepool {
        const double t0 = now_ms();
        VsBlob v;
        std::string name;
        const std::string src = parse_vs_blob(blob, v) ? vs_source(v, name) : std::string();
        id<MTLComputePipelineState> ps = nil;
        bool from_lib = false;
        if (!src.empty()) {
            id<MTLFunction> fn = pre_lib ? [pre_lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]] : nil;
            from_lib = fn != nil;
            NSError *err = nil;
            if (!fn) {
                id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:(std::string(kVsCommonMSL) + src).c_str()]
                                                       options:safe_math_options()
                                                         error:&err];
                if (!lib) LOG("gpu metal: vertex kernel compile failed: %s\n%s\n", err.localizedDescription.UTF8String, src.c_str());
                fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
            }
            if (fn) ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (fn && !ps) LOG("gpu metal: vertex kernel pipeline failed: %s\n", err.localizedDescription.UTF8String);
            if (dump_shaders) LOG("gpu metal: vertex kernel %s:\n%s\n", name.c_str(), src.c_str());
        }
        std::lock_guard<std::recursive_mutex> lk(vk_mtx);
        k->ps = ps;
        k->done = true;
        if (!ps) vkstats.fail++;
        else (from_lib ? vkstats.lib : vkstats.src)++;
        vkstats.ms += now_ms() - t0;
    }
}

void vs_compile_async(VsKern *k, std::string blob) {
    if (sync_shaders) {
        build_vs_kernel(k, blob);
        return;
    }
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        build_vs_kernel(k, blob);
    });
}

// with vk_mtx held
VsKern *vs_add(const std::string &blob) {
    VsBlob v;
    if (!parse_vs_blob(blob, v)) return nullptr;
    const u64 key = vk_key(vk_hash(v.code.data(), v.code_size, v.swz.data(), v.swz_size), v.entry);
    auto &k = vk_cache[v.spec ? spec_key(key, v.sp) : key];
    if (k) return nullptr;
    k = std::make_unique<VsKern>();
    vs_compile_async(k.get(), blob);
    return k.get();
}

void vs_kernels_init(const std::string &tid) {
    std::vector<std::string> blobs;
    for (const auto &p : {g_game.dir + "/vsprograms.bin", std::string(RECOMP_SOURCE_DIR) + "/games/" + tid + "/vsprograms.bin"}) {
        if (!std::filesystem::exists(p)) continue;
        blobs = read_blobs(p, kVsMagic, kVsVersion);
        break;
    }
    const size_t shipped = blobs.size();
    const std::string user = g_save_base + "/shadercache/" + tid + ".vs.bin";
    std::vector<std::string> mine = read_blobs(user, kVsMagic, kVsVersion);
    blobs.insert(blobs.end(), mine.begin(), mine.end());
    if (mine.empty()) {
        vk_log = fopen(user.c_str(), "wb");
        const u32 hdr[2] = {kVsMagic, kVsVersion};
        if (vk_log) fwrite(hdr, 4, 2, vk_log);
    } else {
        vk_log = fopen(user.c_str(), "ab");
    }
    std::lock_guard<std::recursive_mutex> lk(vk_mtx);
    for (const auto &b : blobs) vs_add(b);
    LOG("gpu metal: %zu known vertex programs (%zu shipped, %zu in %s), compiling in the background\n", vk_cache.size(),
        shipped, mine.size(), user.c_str());
}
}  // namespace

const void *metal_vs_kernel(const ShaderProgram *p, u32 entry, const ShaderSetup &s) {
    if (!active) return nullptr;
    std::lock_guard<std::recursive_mutex> lk(vk_mtx);
    // the uniforms the program's flow control reads (masks cached per program)
    static std::unordered_map<u64, std::pair<u32, u32>> masks;
    const u64 ph = shader_program_hash(p);
    auto mi = masks.find(ph);
    if (mi == masks.end()) {
        u32 lm;
        const u32 bm = shader_static_uniforms(p, &lm);
        mi = masks.emplace(ph, std::make_pair(bm, lm)).first;
    }
    VsSpec sp;
    for (u32 i = 0; i < 16; i++)
        if (((mi->second.first >> i) & 1) && s.b[i]) sp.b |= 1u << i;
    for (u32 i = 0; i < 4; i++)
        if ((mi->second.second >> i) & 1) memcpy(sp.iu[i], s.i[i], 3);
    const u64 key = vk_key(ph, entry);
    auto it = vk_cache.find(spec_key(key, sp));
    if (it == vk_cache.end()) {
        const std::string blob = vs_blob(s, entry, &sp);
        VsKern *k = vs_add(blob);
        if (k && vk_log) {
            const u32 n = (u32)blob.size();
            fwrite(&n, 4, 1, vk_log);
            fwrite(blob.data(), 1, n, vk_log);
            fflush(vk_log);
        }
        LOG("gpu metal: new vertex program %016llx entry %u (bools %04x, loops %02x%02x%02x %02x%02x%02x %02x%02x%02x "
            "%02x%02x%02x)%s\n", (unsigned long long)ph, entry, sp.b, sp.iu[0][0], sp.iu[0][1], sp.iu[0][2], sp.iu[1][0],
            sp.iu[1][1], sp.iu[1][2], sp.iu[2][0], sp.iu[2][1], sp.iu[2][2], sp.iu[3][0], sp.iu[3][1], sp.iu[3][2],
            sync_shaders ? "" : ", compiling in the background");
        if (k && k->done && k->ps) return k;
    } else if (it->second->done && it->second->ps) {
        return it->second.get();
    }
    // not ready: the program's generic kernel if there is one, else the CPU
    auto g = vk_cache.find(key);
    if (g != vk_cache.end() && g->second->done && g->second->ps) return g->second.get();
    vkstats.cpu_draws++;
    return nullptr;
}

namespace {
id<MTLRenderPipelineState> get_pipeline(const DrawState &ds) {
    const std::string key = fs_key(ds);
    std::unique_lock<std::mutex> lk(pipe_mtx);
    auto it = pipe_cache.find(key);
    if (it == pipe_cache.end()) {
        if (verify_keys) verify_key(key, ds);
        record_key(key);
        Pipe &p = *(pipe_cache[key] = std::make_unique<Pipe>());
        pstats.fresh++;
        LOG("gpu metal: new pipeline %016llx (%u this session)%s\n", (unsigned long long)fnv64(key), pstats.fresh,
            sync_shaders ? "" : ", compiling in the background");
        if (!sync_shaders) {
            pipe_queue.push_front(key);
            start_workers();
            pipe_cv.notify_all();
            pstats.skipped++;
            return nil;
        }
        p.state = 1;
        lk.unlock();
        id<MTLRenderPipelineState> ps = build_pipeline(ds);
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
        return ps;
    }
    Pipe &p = *it->second;
    if (p.state == 2) return p.ps;
    if (!p.known && !sync_shaders) {
        pstats.skipped++;
        return nil;  // still compiling
    }
    // a known pipeline needed before the background workers got to it
    const double t0 = now_ms();
    if (p.state == 0) {
        p.state = 1;
        lk.unlock();
        id<MTLRenderPipelineState> ps = build_pipeline(ds);
        lk.lock();
        p.ps = ps;
        p.state = 2;
        pipe_cv.notify_all();
    } else {
        pipe_cv.wait(lk, [&] { return p.state == 2; });
    }
    pstats.waited++;
    pstats.wait_ms += now_ms() - t0;
    if (g_stats) LOG("gpu metal: waited %.1f ms for a known pipeline\n", now_ms() - t0);
    return p.ps;
}

std::unordered_map<u64, id<MTLDepthStencilState>> dss_cache;
MTLCompareFunction cmp_fn(u32 f) {
    static const MTLCompareFunction m[8] = {MTLCompareFunctionNever,    MTLCompareFunctionAlways,
                                            MTLCompareFunctionEqual,    MTLCompareFunctionNotEqual,
                                            MTLCompareFunctionLess,     MTLCompareFunctionLessEqual,
                                            MTLCompareFunctionGreater,  MTLCompareFunctionGreaterEqual};
    return m[f & 7];
}
MTLStencilOperation st_op(u32 o) {
    static const MTLStencilOperation m[8] = {
        MTLStencilOperationKeep,           MTLStencilOperationZero,           MTLStencilOperationReplace,
        MTLStencilOperationIncrementClamp, MTLStencilOperationDecrementClamp, MTLStencilOperationInvert,
        MTLStencilOperationIncrementWrap,  MTLStencilOperationDecrementWrap};
    return m[o & 7];
}
id<MTLDepthStencilState> get_dss(const DrawState &ds) {
    u64 key = (u64)ds.depth_test | (u64)ds.depth_func << 1 | (u64)ds.depth_write << 4 | (u64)ds.stencil << 5;
    if (ds.stencil)
        key |= (u64)ds.st_func << 6 | (u64)ds.st_fail << 9 | (u64)ds.st_zfail << 12 | (u64)ds.st_zpass << 15 |
               (u64)ds.st_mask_in << 18 | (u64)ds.st_mask_w << 26;
    auto it = dss_cache.find(key);
    if (it != dss_cache.end()) return it->second;
    MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = ds.depth_test ? cmp_fn(ds.depth_func) : MTLCompareFunctionAlways;
    d.depthWriteEnabled = ds.depth_write;
    if (ds.stencil) {
        MTLStencilDescriptor *s = [MTLStencilDescriptor new];
        s.stencilCompareFunction = cmp_fn(ds.st_func);
        s.stencilFailureOperation = st_op(ds.st_fail);
        s.depthFailureOperation = st_op(ds.st_zfail);
        s.depthStencilPassOperation = st_op(ds.st_zpass);
        s.readMask = ds.st_mask_in;
        s.writeMask = ds.st_mask_w;
        d.frontFaceStencil = s;
        d.backFaceStencil = s;
    }
    id<MTLDepthStencilState> st = [dev newDepthStencilStateWithDescriptor:d];
    dss_cache[key] = st;
    return st;
}

// ---------------------------------------------------------------- display images
struct DisplayImage {
    u32 out_addr = 0, out_size = 0;
    u32 out_w = 0, out_bpp = 0, hs = 0, vs = 0;  // output rows' pixels and bytes per pixel; downscaling shifts
    u32 scale = 1;                               // internal scale the image was made at
    bool incomplete = false;                     // some draws were left out (the screen keeps the last image)
    u64 out_hash = 0;  // guest framebuffer bytes the image corresponds to
    int screen = -1;
    u32 w = 0, h = 0;
    u64 version = 0;          // new for every new image
    id<MTLBuffer> buf;        // the image (RGBA8 rows)
    id<MTLCommandBuffer> cb;  // while the image is still being made: its command buffer
};
std::mutex disp_mtx;
std::unordered_map<u32, DisplayImage> displays;  // by framebuffer physical address
u32 last_screen_addr[2] = {0, 0};
u64 disp_version = 0;

bool eager_writeback = false;
bool warned_shadow = false, warned_proctex = false;

bool overlaps(u32 a, u32 an, u32 b, u32 bn) { return a < b + bn && b < a + an; }

// ---------------------------------------------------------------- deferred display transfers
// A display transfer from a target with GPU work in flight is not waited for: one command buffer encodes the target
// as guest memory (k_enc), converts that as the transfer does (k_tconv) and makes the display image (k_rot); the CPU
// part (copying the target and the transfer output to guest memory) runs once it completed, when polled at a later
// GPU call or when something touches those memory ranges.
struct PendingXfer {
    id<MTLCommandBuffer> cb;
    id<MTLBuffer> gbytes;  // the target as guest memory
    id<MTLBuffer> obytes;  // the transfer's output
    id<MTLBuffer> disp;    // display image
    GRT *rt = nullptr;
    bool rt_writeback = true;  // the target's memory is still to be written (not overwritten since)
    u32 in_paddr = 0, out_paddr = 0, in_dims = 0, out_dims = 0, flags = 0, out_size = 0;
};
std::deque<std::unique_ptr<PendingXfer>> pending;
std::vector<id<MTLBuffer>> xbuf_free;  // GX thread only: gbytes / obytes buffers
std::vector<id<MTLBuffer>> disp_free;  // under disp_mtx
std::vector<id<MTLBuffer>> disp_reading;  // under disp_mtx: images the UI reads outside the lock (not reused meanwhile)
// (disp_mtx held) a display image buffer no longer needed may go back to disp_free
bool recyclable(id<MTLBuffer> b) {
    for (id<MTLBuffer> r : disp_reading)
        if (r == b) return false;
    return disp_free.size() < 6;
}

id<MTLBuffer> take_buffer(std::vector<id<MTLBuffer>> &fl, size_t n) {
    for (size_t i = 0; i < fl.size(); i++)
        if (fl[i].length >= n) {
            id<MTLBuffer> b = fl[i];
            fl.erase(fl.begin() + i);
            return b;
        }
    return make_buffer(n);
}

// guest memory written by a GPU operation: stop trusting overlapping render targets' GPU copies
void rts_overwritten(u32 paddr, u32 size) {
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), paddr, size)) r->synced = false;
}

void finalize(PendingXfer &op) {
    wait_cb(op.cb, "display transfer", op.out_paddr);
    GRT &r = *op.rt;
    const u32 nb = rt_bytes(r);
    if (op.rt_writeback && guest_kept(r)) {
        r.hash = copy_hash(pptr(r.addr), (const u8 *)op.gbytes.contents, nb);
        r.checked = g_frame.load(std::memory_order_relaxed);
    }
    memcpy(pptr(op.out_paddr), op.obytes.contents, op.out_size);
    texcache_invalidate(op.out_paddr, op.out_size);
    rts_overwritten(op.out_paddr, op.out_size);
    if (tf_mode) note_gpu_out(op.out_paddr, op.out_size);
    xbuf_free.push_back(op.gbytes);
    xbuf_free.push_back(op.obytes);
    std::lock_guard<std::mutex> lk(disp_mtx);
    auto it = displays.find(op.out_paddr);
    if (it != displays.end() && it->second.cb == op.cb) {
        DisplayImage &d = it->second;
        d.out_hash = hash_mem(pptr(d.out_addr), d.out_size);
        d.cb = nil;  // the image stays in op.disp
    } else if (recyclable(op.disp)) {
        disp_free.push_back(op.disp);  // replaced meanwhile
    }
}

// finish pending transfers up to and including index i (in order)
void drain(size_t i) {
    for (size_t k = 0; k <= i; k++) {
        @autoreleasepool {
            finalize(*pending.front());
        }
        pending.pop_front();
    }
}
void drain_all() {
    if (!pending.empty()) drain(pending.size() - 1);
}
bool op_touches(const PendingXfer &op, u32 paddr, u32 size) {
    return overlaps(op.out_paddr, op.out_size, paddr, size) ||
           (op.rt_writeback && overlaps(op.rt->addr, rt_bytes(*op.rt), paddr, size));
}
void drain_touching(u32 paddr, u32 size) {
    for (size_t i = pending.size(); i-- > 0;)
        if (op_touches(*pending[i], paddr, size)) {
            drain(i);
            return;
        }
}

// ---------------------------------------------------------------- GPU copies of guest images
// Texture copies and tiled-to-tiled display transfers from a render target the GPU holds newer contents of (and
// from their own results) run on the GPU: their destinations become GPU copies too (GRTs, like render targets,
// written back to guest memory only when something reads it there), and draws sample them directly (a snapshot,
// surface_snapshot). A game's post-processing chain (copy a frame, shrink it a few times, blend it back) then
// never waits for the GPU nor goes through the CPU.
inline u32 pix_to_cbuf(u32 pf) {
    static constexpr u8 m[8] = {0, 1, 3, 2, 4, 0, 0, 0};
    return m[pf & 7];
}
inline u32 grt_bpp(const GRT &r) { return pixel_bytes(cbuf_to_pixfmt(r.fmt)); }

// r's GPU copy holds its current contents: newer than guest memory, or with a write-back of them queued
bool gpu_valid(const GRT &r) {
    if (r.depth || !r.synced) return false;
    if (r.gpu_newer) return true;
    for (const auto &op : pending)
        if (op->rt == &r && op->rt_writeback) return true;
    return false;
}

// the GPU copy of an image at addr with this width and format (the tallest; null: none); any: also one that no
// longer matches guest memory (synced again from it before it is drawn into)
GRT *find_view(u32 addr, u32 w, u32 fmt, bool depth, bool any = false) {
    GRT *best = nullptr;
    for (auto &r : rts)
        if (r->depth == depth && (r->synced || any) && r->addr == addr && r->w == w && r->fmt == fmt && (!best || r->h > best->h))
            best = r.get();
    return best;
}
GRT *find_color(u32 addr, u32 w, u32 fmt) { return find_view(addr, w, fmt, false); }

// r's GPU copy made guest memory's contents (unless it is already)
void fill_from_guest(GRT &r, id<MTLCommandBuffer> cb) {
    const LoadAction la = sync_rt(r, cb);
    if (!la.clear) return;
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    if (r.depth) {
        rp.depthAttachment.texture = rp.stencilAttachment.texture = r.tex;
        rp.depthAttachment.loadAction = rp.stencilAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.clearDepth = la.depth;
        rp.stencilAttachment.clearStencil = la.stencil;
        rp.depthAttachment.storeAction = rp.stencilAttachment.storeAction = MTLStoreActionStore;
    } else {
        rp.colorAttachments[0].texture = r.tex;
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        const u32 c = la.color;
        rp.colorAttachments[0].clearColor =
            MTLClearColorMake((c & 255) / 255.0, ((c >> 8) & 255) / 255.0, ((c >> 16) & 255) / 255.0, (c >> 24) / 255.0);
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> ce = [cb renderCommandEncoderWithDescriptor:rp];
    label(ce, "clear %08x %ux%u", r.addr, r.w, r.h);
    [ce endEncoding];
    r.ms_valid = false;
    r.version++;
}

// a taller GPU copy of old's image (h rows): old's rows, then guest memory's; null if a queued transfer uses old
GRT *grow(GRT &old, u32 h, id<MTLCommandBuffer> cb) {
    for (const auto &op : pending)
        if (op->rt == &old) return nullptr;
    GRT &n = get_rt(old.addr, old.w, h, old.fmt, old.depth);
    n.synced = false;
    fill_from_guest(n, cb);
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromTexture:old.tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(old.w * S, old.h * S, 1) toTexture:n.tex destinationSlice:0 destinationLevel:0
      destinationOrigin:MTLOriginMake(0, 0, 0)];
    [be endEncoding];
    n.gpu_newer = old.gpu_newer;
    n.ms_valid = false;
    n.version++;
    old.synced = false;
    old.gpu_newer = false;
    return &n;
}

// The GPU copy to draw a view of h rows (of an image at addr with this width and format) into, h tall: the tallest
// copy of that image (a shorter view draws into its first rows: one copy of the memory instead of two kept in step
// through guest memory), grown from a shorter one when this view is the tallest yet.
GRT &target_for(u32 addr, u32 w, u32 h, u32 fmt, bool depth) {
    GRT *t = find_view(addr, w, fmt, depth, true);
    if (t && t->h == h) return *t;
    if (t && t->h < h && t->gpu_newer)
        if (GRT *n = grow(*t, h, list_cb())) return *n;
    return get_rt(addr, w, h, fmt, depth);
}

void drop_displays(u32 paddr, u32 size, bool definite);

// Before the GPU writes [paddr, paddr + size) into dst (reading src): queued CPU parts writing there go first; other
// GPU copies of that memory go back to it (or are dropped where the write covers them) and are no longer trusted.
// True if guest memory there changed meanwhile.
bool claim(GRT *src, GRT *dst, u32 paddr, u32 size) {
    drain_touching(paddr, size);
    bool changed = false;
    for (auto &r : rts) {
        if (r.get() == src || r.get() == dst || !overlaps(r->addr, rt_bytes(*r), paddr, size)) continue;
        if (r->gpu_newer) {
            if (r->addr >= paddr && r->addr + rt_bytes(*r) <= paddr + size) {
                r->gpu_newer = false;
            } else {
                if (r->depth)
                    write_back_depth(*r);
                else
                    write_back_color(*r);
                changed = true;
            }
        }
        r->synced = false;
    }
    return changed;
}

// The GPU copy that an operation reading src writes (all or the first rows of) the w x h image at addr in (bytes: the
// guest memory it writes); null if it can't be done on the GPU.
GRT *dest_surface(u32 addr, u32 w, u32 h, u32 fmt, GRT *src, u32 bytes) {
    if (overlaps(addr, bytes, src->addr, rt_bytes(*src))) return nullptr;
    GRT *d = find_color(addr, w, fmt);
    if (d && overlaps(d->addr, rt_bytes(*d), src->addr, rt_bytes(*src))) return nullptr;
    if (claim(src, d, addr, bytes)) return nullptr;  // (rare: other GPU copies of it were newer; the CPU does it)
    id<MTLCommandBuffer> cb = list_cb();  // (after claim's write-backs, which commit it)
    if (!d) {
        d = &get_rt(addr, w, h, fmt, false);
        fill_from_guest(*d, cb);
    } else if (d->h < h) {
        d = grow(*d, h, cb);
    }
    return d;
}

// the GPU wrote dst's copy of [paddr, paddr + size)
void gpu_wrote(GRT &d, u32 paddr, u32 size) {
    d.gpu_newer = true;
    d.ms_valid = false;
    d.version++;
    texcache_invalidate(paddr, size);
    drop_displays(paddr, size, true);
}

// A texture (w x h, format fmt) at addr as its GPU copy holds it, rows in the decoded textures' order (made once per
// change of the copy); nil without a copy.
struct Snap {
    id<MTLTexture> tex;
    u64 id = 0, version = 0;
};
std::unordered_map<u64, Snap> snaps;
// (ce: the compute encoder of the flush's snapshots, made on cb by the first one needed)
id<MTLTexture> surface_snapshot(u32 addr, u32 w, u32 h, u32 fmt, id<MTLCommandBuffer> cb,
                                id<MTLComputeCommandEncoder> __strong &ce) {
    GRT *r = find_color(addr, w, fmt);
    if (!r || r->h < h) return nil;
    Snap &sn = snaps[(u64)addr << 32 ^ (u64)w << 20 ^ (u64)h << 4 ^ fmt];
    if (!sn.tex || sn.tex.width != w * S || sn.tex.height != h * S) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:w * S
                                                                                    height:h * S
                                                                                 mipmapped:NO];
        d.storageMode = MTLStorageModePrivate;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        sn.tex = [dev newTextureWithDescriptor:d];
        sn.id = 0;
    } else if (sn.id == r->id && sn.version == r->version) {
        return sn.tex;
    }
    if (!ce) {
        ce = [cb computeCommandEncoder];
        label(ce, "snapshots");
    }
    [ce setComputePipelineState:k_snap];
    [ce setTexture:r->tex atIndex:0];
    [ce setTexture:sn.tex atIndex:1];
    const u32 p[4] = {w * S, h * S, fmt == 1 || fmt == 3 ? 1u : 0u, 0};  // RGB8, RGB565: no alpha
    [ce setBytes:p length:16 atIndex:0];
    [ce dispatchThreads:MTLSizeMake(w * S, h * S, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    sn.id = r->id;
    sn.version = r->version;
    return sn.tex;
}
}  // namespace

// ---------------------------------------------------------------- public backend interface
bool metal_init() {
    if (tried) return active;
    tried = true;
    const char *be = getenv("GPU_BACKEND");
    if (be && strcmp(be, "sw") == 0) return false;  // default: Metal when available
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            LOG("gpu metal: no Metal device, using the software renderer\n");
            return false;
        }
        queue = [dev newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kKernelsMSL] options:nil error:&err];
        if (!lib) {
            LOG("gpu metal: kernel compile failed: %s\n", err.localizedDescription.UTF8String);
            return false;
        }
        k_rot = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_rot"] error:&err];
        k_xfer = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_xfer"] error:&err];
        k_snap = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_snap"] error:&err];
        k_enc = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_enc"] error:&err];
        k_dec = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_dec"] error:&err];
        k_dec_depth = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_dec_depth"] error:&err];
        k_enc_depth = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_enc_depth"] error:&err];
        k_tconv = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tconv"] error:&err];
        if (!k_rot || !k_xfer || !k_snap || !k_enc || !k_dec || !k_dec_depth || !k_enc_depth || !k_tconv) {
            LOG("gpu metal: kernel pipelines failed: %s\n", err.localizedDescription.UTF8String);
            return false;
        }
        // GPU_SCALE: 1-4, or auto (default): the frontend picks it from the window size (set_scale)
        const char *sc = getenv("GPU_SCALE");
        scale_fixed = sc && atoi(sc) > 0;
        S = scale_fixed ? (u32)std::clamp(atoi(sc), 1, 4) : 2;
        const char *ms = getenv("GPU_MSAA");
        MS = ms ? (u32)std::clamp(atoi(ms), 1, 8) : kDefaultMSAA;
        while (MS > 1 && ((MS & (MS - 1)) || ![dev supportsTextureSampleCount:MS])) MS--;
        ms_memoryless = [dev supportsFamily:MTLGPUFamilyApple1] && !getenv("GPU_MSAA_STORED");
        if (MS > 1) {
            MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
            pd.vertexFunction = [lib newFunctionWithName:@"k_full"];
            pd.fragmentFunction = [lib newFunctionWithName:@"k_rcolor"];
            pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
            pd.rasterSampleCount = MS;
            p_rcolor = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
            pd.fragmentFunction = [lib newFunctionWithName:@"k_rdepth"];
            pd.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
            p_rdepth = [dev newRenderPipelineStateWithDescriptor:pd error:&err];
            MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
            dss_off = [dev newDepthStencilStateWithDescriptor:d];
            d.depthWriteEnabled = YES;
            MTLStencilDescriptor *st = [MTLStencilDescriptor new];
            st.depthStencilPassOperation = MTLStencilOperationReplace;  // with the shader's stencil value
            d.frontFaceStencil = st;
            d.backFaceStencil = st;
            dss_rdepth = [dev newDepthStencilStateWithDescriptor:d];
            if (!p_rcolor || !p_rdepth) {
                LOG("gpu metal: MSAA unavailable (%s)\n", err.localizedDescription.UTF8String);
                MS = 1;
            }
        }
        const char *an = getenv("GPU_ANISO");
        aniso = an ? (u32)std::clamp(atoi(an), 1, 16) : kDefaultAniso;
    }
    dump_shaders = getenv("GPU_DUMP_SHADERS") != nullptr;
    eager_writeback = getenv("GPU_METAL_EAGER") != nullptr;
    log_timing = getenv("RECOMP_STATS") != nullptr;
    if (const char *tf = getenv("GPU_TEX_FILTER")) metal_set_texture_filter(atoi(tf));
    active = true;
    LOG("gpu: Metal backend on %s, internal scale %s%u, MSAA %ux, anisotropic filtering %ux\n", dev.name.UTF8String,
        scale_fixed ? "" : "auto, now ", S, MS, aniso);
    @autoreleasepool {
        pipelines_init();
    }
    return true;
}

// <outdir> <pipeline lists...>: one .metal file per fragment shader (fs_<hash of its source>) plus vs.metal,
// for building shaders.metallib ahead of time. Unchanged files are left alone and stale ones removed, so
// only new shaders get recompiled.
int metal_emit_shaders(int argc, char **argv) {
    if (argc < 1) {
        fprintf(stderr, "usage: --emit-shaders <outdir> <pipelines.bin...>\n");
        return 1;
    }
    const std::filesystem::path out = argv[0];
    std::filesystem::create_directories(out);
    std::unordered_map<std::string, std::string> files;  // file name -> contents
    files["vs.metal"] = std::string(kCommonMSL) + kVertexMSL;
    size_t nkeys = 0;
    for (int i = 1; i < argc; i++) {
        const std::vector<std::string> progs = read_blobs(argv[i], kVsMagic, kVsVersion);
        for (const auto &b : progs) {
            VsBlob v;
            std::string name;
            const std::string src = parse_vs_blob(b, v) ? vs_source(v, name) : std::string();
            if (!src.empty()) files[name + ".metal"] = std::string(kVsCommonMSL) + src;
        }
        if (!progs.empty()) continue;
        const std::vector<std::string> keys = read_keys(argv[i]);
        if (keys.empty()) fprintf(stderr, "emit-shaders: no pipelines in %s\n", argv[i]);
        for (const auto &k : keys) {
            const std::string src = gen_fs(decode_key(k)), name = fs_name(src);
            std::string s = src;
            s.replace(s.find("fs_main("), 7, name);
            files[name + ".metal"] = std::move(s);
            nkeys++;
        }
    }
    size_t written = 0, removed = 0;
    for (auto &[name, s] : files) {
        const auto p = out / name;
        std::string old;
        if (FILE *f = fopen(p.c_str(), "rb")) {
            char b[65536];
            size_t n;
            while ((n = fread(b, 1, sizeof b, f)) > 0) old.append(b, n);
            fclose(f);
        }
        if (old == s) continue;
        FILE *f = fopen(p.c_str(), "wb");
        if (!f) return 1;
        fwrite(s.data(), 1, s.size(), f);
        fclose(f);
        written++;
    }
    for (const auto &e : std::filesystem::directory_iterator(out)) {
        const auto ext = e.path().extension();
        if (ext != ".metal" && ext != ".air") continue;
        if (files.count(e.path().stem().string() + ".metal")) continue;
        std::filesystem::remove(e.path());
        removed++;
    }
    printf("emit-shaders: %zu pipelines, %zu shaders (%zu written, %zu removed)\n", nkeys, files.size(), written,
           removed);
    return 0;
}

namespace {
// change the internal resolution: render targets go back to guest memory and are recreated at the
// new scale (from guest memory) when next drawn to
void apply_scale(u32 ns) {
    drain_all();
    for (auto &r : rts)
        if (r->gpu_newer) {
            if (r->depth) write_back_depth(*r);
            else write_back_color(*r);
        }
    rts.clear();
    scratch_depth.clear();
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        disp_free.clear();  // display image buffers of the old size
    }
    LOG("gpu metal: internal scale %u -> %u\n", S, ns);
    if (tf_mode && (S > 2) != (ns > 2)) tf_drop_all();  // upscale factors follow the scale (tf_factor)
    S = ns;
}
}  // namespace

void metal_set_scale(int s) { want_scale = (u32)std::clamp(s, 1, 4); }
int metal_scale() { return active ? (int)(want_scale.load() ? want_scale.load() : S) : 0; }
bool metal_scale_is_auto() { return active && !scale_fixed; }
float metal_busy() { return active ? busy_frac.load() : 0; }

void metal_set_texture_filter(int mode) { want_filter = std::clamp(mode, 0, (int)kTexFilterCount - 1); }
int metal_texture_filter_count() { return kTexFilterCount; }
const char *metal_texture_filter_name(int mode) {
    static const char *const names[kTexFilterCount] = {"Off", "Sharp", "Smooth"};
    return mode >= 0 && mode < kTexFilterCount ? names[mode] : "?";
}
bool metal_gpu_made(u32 paddr, u32 size) { return active && tf_mode && size && gpu_made(paddr, size); }
void metal_note_gpu_output(u32 paddr, u32 size) {
    if (active && tf_mode && size) note_gpu_out(paddr, size);
}

void metal_flush(const RTConfig &rt, const MBatch &mb) {
    if (!active || mb.empty() || !rt.color_addr || !pptr(rt.color_addr)) return;
    if (rt.color_fmt & 0x80) {
        if (!warned_shadow) LOG("gpu metal: shadow-map rendering is not supported, draws skipped\n");
        warned_shadow = true;
        return;
    }
    const double t0 = g_stats ? now_ms() : 0;
    flushes++;
    if (const u32 ws = want_scale.load(); ws && ws != S) apply_scale(ws);
    if (const int wf = want_filter.load(); wf != tf_mode) {
        LOG("gpu metal: texture filter %s -> %s\n", metal_texture_filter_name(tf_mode), metal_texture_filter_name(wf));
        tf_mode = wf;
        tf_drop_all();
        gpu_outs.clear();
    }
    flush_no++;
    @autoreleasepool {
        // a view shorter than an image the game drew taller before (the bottom screen's pass in the first rows of the
        // top screen's target) draws into that taller copy, if its depth buffer has one as tall too (the attachments
        // of a pass are as tall as each other)
        // a batch whose draws neither test nor write depth or stencil (full-screen post-processing) leaves the depth
        // buffer out of the pass: no restore, load or resolve of it
        bool uses_depth = false;
        for (const MSeg &seg : mb.segs) uses_depth |= seg.ds->depth_test || seg.ds->depth_write || seg.ds->stencil;
        const bool has_dep = uses_depth && rt.depth_addr && pptr(rt.depth_addr);
        const GRT *tall_c = find_view(rt.color_addr, rt.w, rt.color_fmt, false, true);
        const GRT *tall_d = has_dep ? find_view(rt.depth_addr, rt.w, rt.depth_fmt, true, true) : nullptr;
        const u32 hc = std::max(rt.h, tall_c ? tall_c->h : 0u), hd = has_dep ? std::max(rt.h, tall_d ? tall_d->h : 0u) : hc;
        const u32 H = hc == hd ? hc : rt.h;
        GRT &col = target_for(rt.color_addr, rt.w, H, rt.color_fmt, false);
        GRT *dep = has_dep ? &target_for(rt.depth_addr, rt.w, H, rt.depth_fmt, true) : nullptr;
        // pending transfers still to write these targets' memory (sync_rt below hashes it)
        drain_touching(col.addr, rt_bytes(col));
        if (dep) drain_touching(dep->addr, rt_bytes(*dep));
        // other targets sharing memory with these (another size or format at the same address):
        // flush their pending changes to guest memory first, and stop trusting their GPU copies
        for (auto &o : rts) {
            if (o.get() == &col || o.get() == dep) continue;
            const bool ov_c = o->addr < col.addr + rt_bytes(col) && col.addr < o->addr + rt_bytes(*o);
            const bool ov_d = dep && o->addr < dep->addr + rt_bytes(*dep) && dep->addr < o->addr + rt_bytes(*o);
            if (!ov_c && !ov_d) continue;
            if (o->gpu_newer) {
                if (o->depth) write_back_depth(*o);
                else write_back_color(*o);
            }
            o->synced = false;
        }
        // textures sampled from GPU copies (TexUnitState::surf): a copy gone meanwhile (scale change) is written back
        // and decoded instead
        std::unordered_map<u64, const TexEntry *> surf_decoded;
        auto surf_key = [](const TexUnitState &t) { return (u64)t.surf_addr << 32 ^ (u64)t.w << 20 ^ (u64)t.h << 4 ^ t.surf_fmt; };
        for (const MSeg &seg : mb.segs)
            for (const TexUnitState &t : seg.ds->tex) {
                if (!t.surf || surf_decoded.count(surf_key(t))) continue;
                GRT *r = find_color(t.surf_addr, t.w, t.surf_fmt);
                if (r && r->h >= t.h) continue;
                metal_guest_read(t.surf_addr, t.w * t.h * pixel_bytes(cbuf_to_pixfmt(t.surf_fmt)));
                surf_decoded[surf_key(t)] = texcache_get(t.surf_addr, t.w, t.h, t.surf_fmt, 1);
            }
        id<MTLCommandBuffer> cb = list_cb();  // (after the write-backs above, which commit it)
        LoadAction cla = sync_rt(col, cb);
        if (col.clear_next) {  // (set_clear_after_top_transfer)
            col.clear_next = false;
            cla.clear = true;
            cla.color = 0;
        }
        LoadAction dla;
        if (dep) dla = sync_rt(*dep, cb);

        // vertices: compute kernels shade the draws' vertex arrays into gbuf; the CPU's are in cbuf
        auto upload = [&](const void *p, size_t n) {
            const Slice sl = arena(n);
            if (n) memcpy(sl.p, p, n);
            return sl;
        };
        const Slice gbuf = arena((size_t)mb.gpu_verts * sizeof(OutVertex));
        const Slice cbuf = upload(mb.cverts.data(), mb.cverts.size() * sizeof(OutVertex));
        const Slice ibuf = upload(mb.idx.data(), mb.idx.size() * 4);
        if (!mb.jobs.empty()) {
            const Slice dbuf = upload(mb.vdata.data(), mb.vdata.size());
            // one dispatch per kernel: its draws' vertices one after the other, with their parameter blocks
            // in a buffer and each thread's draw in tj (draws are often only a few vertices, and a dispatch
            // per draw costs far more GPU time than the shading). The draws write disjoint ranges of gbuf.
            // Each group's parameter blocks, vertex ids and draw-of-vertex table are written straight into the
            // arena: counted first, then filled.
            struct Group {
                id<MTLComputePipelineState> ps;
                u32 njobs = 0, nverts = 0;  // counted, then how many are written
                Slice vb, pb, tb;
            };
            std::vector<Group> groups;
            auto group_of = [&](const GpuVsJob &j) -> Group & {
                id<MTLComputePipelineState> ps = ((const VsKern *)j.kernel)->ps;
                for (Group &x : groups)
                    if (x.ps == ps) return x;
                return groups.emplace_back(Group{ps});
            };
            for (const GpuVsJob &j : mb.jobs) {
                Group &g = group_of(j);
                g.njobs++;
                g.nverts += j.p.nverts;
            }
            for (Group &g : groups) {
                g.vb = arena((size_t)g.nverts * 4);
                g.pb = arena((size_t)g.njobs * sizeof(VsParamsG));
                g.tb = arena((size_t)g.nverts * 4);
                g.njobs = g.nverts = 0;
            }
            for (const GpuVsJob &j : mb.jobs) {
                Group &g = group_of(j);
                VsParamsG &P = ((VsParamsG *)g.pb.p)[g.njobs];
                memcpy(&P, &j.p, sizeof P);
                P.first = g.nverts;
                if (j.p.nverts) memcpy(g.vb.p + (size_t)g.nverts * 4, mb.vids.data() + j.vids_off, (size_t)j.p.nverts * 4);
                std::fill_n((u32 *)g.tb.p + g.nverts, j.p.nverts, g.njobs);
                g.nverts += j.p.nverts;
                g.njobs++;
            }
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            label(ce, "vertex kernels %08x", rt.color_addr);
            [ce setBuffer:dbuf.buf offset:dbuf.off atIndex:0];
            [ce setBuffer:gbuf.buf offset:gbuf.off atIndex:3];
            vs_dispatches += groups.size();
            vs_passes++;
            vs_verts += mb.vids.size();
            for (const Group &g : groups) {
                if (!g.nverts) continue;
                [ce setComputePipelineState:g.ps];
                [ce setBuffer:g.vb.buf offset:g.vb.off atIndex:1];
                [ce setBuffer:g.pb.buf offset:g.pb.off atIndex:2];
                [ce setBuffer:g.tb.buf offset:g.tb.off atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(64, g.ps.maxTotalThreadsPerThreadgroup);
                [ce dispatchThreads:MTLSizeMake(g.nverts, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            }
            [ce endEncoding];
        }

        std::unordered_map<u64, id<MTLTexture>> surf_tex;
        id<MTLComputeCommandEncoder> snap_ce = nil;
        for (const MSeg &seg : mb.segs)
            for (const TexUnitState &t : seg.ds->tex)
                if (t.surf && !surf_decoded.count(surf_key(t)) && !surf_tex.count(surf_key(t)))
                    surf_tex[surf_key(t)] = surface_snapshot(t.surf_addr, t.w, t.h, t.surf_fmt, cb, snap_ce);
        if (snap_ce) [snap_ce endEncoding];

        // with MSAA the pass draws into the multisampled targets and resolves them into the single-sample
        // ones (which everything else reads); a multisampled target older than its resolved copy (guest
        // memory uploaded since) is restored from it by a full-screen draw at the start of the pass
        const bool msaa = MS > 1;
        const bool restore_c = msaa && !cla.clear && !col.ms_valid;
        const bool restore_d = msaa && dep && !dla.clear && !dep->ms_valid;
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = msaa ? col.ms : col.tex;
        rp.colorAttachments[0].loadAction = cla.clear ? MTLLoadActionClear
                                            : restore_c ? MTLLoadActionDontCare
                                                        : MTLLoadActionLoad;
        rp.colorAttachments[0].clearColor =
            MTLClearColorMake((cla.color & 255) / 255.0, ((cla.color >> 8) & 255) / 255.0,
                              ((cla.color >> 16) & 255) / 255.0, (cla.color >> 24) / 255.0);
        const MTLStoreAction ms_store = ms_memoryless ? MTLStoreActionMultisampleResolve : MTLStoreActionStoreAndMultisampleResolve;
        rp.colorAttachments[0].storeAction = msaa ? ms_store : MTLStoreActionStore;
        if (msaa) rp.colorAttachments[0].resolveTexture = col.tex;
        id<MTLTexture> dt = dep ? (msaa ? dep->ms : dep->tex) : get_scratch_depth(rt.w, H);
        rp.depthAttachment.texture = dt;
        rp.stencilAttachment.texture = dt;
        if (dep) {
            const MTLLoadAction la = dla.clear ? MTLLoadActionClear : restore_d ? MTLLoadActionDontCare : MTLLoadActionLoad;
            rp.depthAttachment.loadAction = la;
            rp.stencilAttachment.loadAction = la;
            rp.depthAttachment.clearDepth = dla.depth;
            rp.stencilAttachment.clearStencil = dla.stencil;
            if (msaa) {
                rp.depthAttachment.storeAction = ms_store;
                rp.stencilAttachment.storeAction = ms_store;
                rp.depthAttachment.resolveTexture = dep->tex;
                rp.stencilAttachment.resolveTexture = dep->tex;
                rp.depthAttachment.depthResolveFilter = MTLMultisampleDepthResolveFilterSample0;
                rp.stencilAttachment.stencilResolveFilter = MTLMultisampleStencilResolveFilterSample0;
            } else {
                rp.depthAttachment.storeAction = MTLStoreActionStore;
                rp.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        } else {
            rp.depthAttachment.loadAction = MTLLoadActionClear;
            rp.stencilAttachment.loadAction = MTLLoadActionClear;
            rp.depthAttachment.storeAction = MTLStoreActionDontCare;
            rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
        }
        id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
        label(re, "draw %08x %ux%u (%zu runs of draws)", rt.color_addr, rt.w, rt.h, mb.segs.size());
        [re setViewport:(MTLViewport){0, 0, (double)rt.w * S, (double)H * S, 0, 1}];  // (the restores: all of it)
        [re setDepthClipMode:MTLDepthClipModeClamp];
        [re setFrontFacingWinding:MTLWindingCounterClockwise];
        [re setVertexBuffer:gbuf.buf offset:gbuf.off atIndex:0];
        [re setVertexBuffer:cbuf.buf offset:cbuf.off atIndex:1];
        [re setVertexBuffer:ibuf.buf offset:ibuf.off atIndex:2];
        if (restore_c) {
            [re setRenderPipelineState:p_rcolor];
            [re setDepthStencilState:dss_off];
            [re setFragmentTexture:col.tex atIndex:0];
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        if (restore_d) {
            [re setRenderPipelineState:p_rdepth];
            [re setDepthStencilState:dss_rdepth];
            [re setFragmentTexture:dep->tex atIndex:0];
            if (!dep->stencil_view) dep->stencil_view = [dep->tex newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
            [re setFragmentTexture:dep->stencil_view atIndex:1];
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        col.ms_valid = !ms_memoryless;
        if (dep) dep->ms_valid = !ms_memoryless;
        [re setViewport:(MTLViewport){0, 0, (double)rt.w * S, (double)rt.h * S, 0, 1}];

        // per-draw uniforms and LUTs go into one shared buffer
        size_t uoff = 0;
        auto ualloc = [&](size_t n) {
            const size_t o = (uoff + 255) & ~(size_t)255;
            uoff = o + n;
            return o;
        };
        // first pass: size (a uniform block for each draw, each distinct LUT snapshot once)
        std::unordered_map<const void *, size_t> lut_map;  // a snapshot's place in the buffer
        size_t need = mb.segs.size() * (256 + sizeof(Uni));
        for (const MSeg &seg : mb.segs) {
            const DrawState &ds = *seg.ds;
            if (ds.lighting && lut_map.emplace(ds.light.luts.get(), 0).second) need += 256 + sizeof(LightLuts);
            if (ds.fog && lut_map.emplace(ds.fog_lut.get(), 0).second) need += 256 + sizeof(FogLut);
        }
        lut_map.clear();
        const Slice us = arena(need);
        id<MTLBuffer> ubuf = us.buf;
        u8 *ub = us.p;
        bool wrote_color = false, wrote_depth = false;
        for (const MSeg &seg : mb.segs) {
            const DrawState &ds = *seg.ds;
            if (ds.skip || ds.shadow_mode || ds.gas) continue;
            id<MTLRenderPipelineState> ps = get_pipeline(ds);
            if (!ps) {
                col.skipped++;
                continue;
            }
            if (ds.need_tex[3] && !warned_proctex) {
                LOG("gpu metal: procedural textures are not supported (texture 3 reads as 0)\n");
                warned_proctex = true;
            }
            [re setRenderPipelineState:ps];
            [re setDepthStencilState:get_dss(ds)];
            {
                struct {
                    float rt[4], vp[4], clip[4];
                    u32 flags[4];
                } vd = {{(float)rt.w, (float)rt.h, ds.depth_scale, ds.depth_offset},
                        {ds.vp_hw, ds.vp_hh, ds.vp_x, ds.vp_y},
                        {ds.clip[0], ds.clip[1], ds.clip[2], ds.clip[3]},
                        {(u32)ds.clip_enable, 0, 0, 0}};
                [re setVertexBytes:&vd length:sizeof vd atIndex:3];
                // PICA culling (by the winding in PICA window coordinates, y up)
                [re setCullMode:ds.cull == 1 ? MTLCullModeFront : ds.cull == 2 ? MTLCullModeBack : MTLCullModeNone];
                const u32 bc = ds.blend_color;
                [re setBlendColorRed:(bc & 255) / 255.0f green:((bc >> 8) & 255) / 255.0f
                                blue:((bc >> 16) & 255) / 255.0f alpha:(bc >> 24) / 255.0f];
            }
            if (ds.stencil) [re setStencilReferenceValue:ds.st_ref];
            // scissor
            MTLScissorRect sr = {0, 0, rt.w * S, rt.h * S};
            if (ds.scissor_mode == 3) {
                const s32 x1 = std::clamp<s32>(ds.sc_x1, 0, rt.w), x2 = std::clamp<s32>(ds.sc_x2, 0, rt.w);
                const s32 y1 = std::clamp<s32>(ds.sc_y1, 0, rt.h), y2 = std::clamp<s32>(ds.sc_y2, 0, rt.h);
                if (x2 <= x1 || y2 <= y1) continue;
                sr = {(NSUInteger)x1 * S, (NSUInteger)(rt.h - y2) * S, (NSUInteger)(x2 - x1) * S, (NSUInteger)(y2 - y1) * S};
            }
            [re setScissorRect:sr];
            // uniforms
            const size_t uo = ualloc(sizeof(Uni));
            Uni &u = *(Uni *)(ub + uo);
            memset(&u, 0, sizeof u);
            u.rt[0] = (float)rt.w;
            u.rt[1] = (float)rt.h;
            u.rt[2] = (float)S;
            u.rt[3] = (float)((1u << ds.depth_bits) - 1);
            u.depth[0] = ds.depth_scale;
            u.depth[1] = ds.depth_offset;
            u.depth[2] = ds.depth_bits == 16 ? 1.0f / 65536.0f : 1.0f / 16777216.0f;
            u.scissor[0] = ds.sc_x1;
            u.scissor[1] = ds.sc_y1;
            u.scissor[2] = ds.sc_x2;
            u.scissor[3] = ds.sc_y2;
            for (int k = 0; k < 6; k++) u.konst[k] = ds.tev[k].konst;
            u.tevc[0] = ds.tev_init_out;
            u.tevc[1] = ds.tev_init_buf;
            u.tevc[2] = ds.tev_init_next;
            u.misc[0] = ds.blend_color;
            u.misc[1] = ds.fog_color;
            u.misc[2] = ds.alpha_ref;
            u.misc[3] = ds.tex[0].shadow_bias;
            for (int k = 0; k < 3; k++) {
                u.border[k] = ds.tex[k].border;
                u.bias[k] = ds.tex[k].bias;
            }
            if (ds.lighting) {
                const LightingState &l = ds.light;
                const LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
                for (int k = 0; k < 7; k++) u.lscale[k] = sels[k]->scale;
                for (int k = 0; k < 3; k++) u.ambient[k] = l.global_ambient[k];
                for (u32 k = 0; k < l.num_lights; k++) {
                    const LightState &s = l.lights[k];
                    LightU &d = u.lights[k];
                    for (int c = 0; c < 3; c++) {
                        d.spec0[c] = s.spec0[c];
                        d.spec1[c] = s.spec1[c];
                        d.diffuse[c] = s.diffuse[c];
                        d.ambient[c] = s.ambient[c];
                        d.pos[c] = s.pos[c];
                        d.spot[c] = s.spot[c];
                    }
                    d.att[0] = s.da_scale;
                    d.att[1] = s.da_bias;
                }
                // LUTs (once per distinct snapshot)
                const void *lk = l.luts.get();
                auto it = lut_map.find(lk);
                size_t lo;
                if (it == lut_map.end()) {
                    lo = ualloc(sizeof(LightLuts));
                    memcpy(ub + lo, l.luts->vd, sizeof(LightLuts));
                    lut_map[lk] = lo;
                } else {
                    lo = it->second;
                }
                [re setFragmentBuffer:ubuf offset:us.off + lo atIndex:1];
            }
            if (ds.fog) {
                const void *fk = ds.fog_lut.get();
                auto it = lut_map.find(fk);
                size_t fo;
                if (it == lut_map.end()) {
                    fo = ualloc(sizeof(FogLut));
                    memcpy(ub + fo, ds.fog_lut->vd, sizeof(FogLut));
                    lut_map[fk] = fo;
                } else {
                    fo = it->second;
                }
                [re setFragmentBuffer:ubuf offset:us.off + fo atIndex:2];
            }
            [re setFragmentBuffer:ubuf offset:us.off + uo atIndex:0];
            // textures
            if (ds.need_shade) {
                for (int k = 0; k < 3; k++) {
                    const TexUnitState &t = ds.tex[k];
                    if (!ds.need_tex[k] || t.null_tex) continue;
                    // upscaling (texture filter): 2D and projected textures only (not cube or shadow maps)
                    const bool up = tf_mode && (k != 0 || t.type == 0 || t.type == 3);
                    u32 ul = 0;
                    id<MTLTexture> tex = nil;
                    if (t.surf) {
                        auto si = surf_tex.find(surf_key(t));
                        if (si != surf_tex.end()) tex = si->second;
                        auto di = surf_decoded.find(surf_key(t));
                        if (di != surf_decoded.end()) tex = get_texture(di->second);
                    } else {
                        tex = get_texture(t.face[0], up ? &t : nullptr, &ul);
                    }
                    if (!tex) {
                        // missing texture data: bind a 1x1 black texture
                        static id<MTLTexture> black;
                        if (!black) {
                            MTLTextureDescriptor *d = [MTLTextureDescriptor
                                texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
                            black = [dev newTextureWithDescriptor:d];
                            const u32 z = 0;
                            [black replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:&z bytesPerRow:4];
                        }
                        tex = black;
                    }
                    [re setFragmentTexture:tex atIndex:k];
                    [re setFragmentSamplerState:get_sampler(t, (u32)tex.mipmapLevelCount, ul) atIndex:k];
                    // the LOD bias of a texture without mipmaps does nothing on the PICA (games leave any value);
                    // upscaled, it has levels: no bias then, or a positive one would pick the original level
                    if (ul && tex.mipmapLevelCount == ul + 1) u.bias[k] = 0;
                    if (k == 0 && (t.type == 1 || t.type == 4))
                        for (int f = 1; f < 6; f++) {
                            id<MTLTexture> ft = get_texture(t.face[f]);
                            [re setFragmentTexture:ft ? ft : tex atIndex:2 + f];
                        }
                }
            }
            [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:seg.first vertexCount:seg.count];
            if (ds.color_write) wrote_color = true;
            if (ds.depth_write || (ds.stencil && ds.st_mask_w)) wrote_depth = true;
        }
        [re endEncoding];
        arena_trim(us, uoff);
        const double t1 = g_stats ? now_ms() : 0;
        if (tf_cb) {  // texture upscaling, before the draws that sample the results
            [tf_cb commit];
            tf_cb = nil;
        }
        // no wait: write-back happens when something reads the target. With a texture filter, the passes go now:
        // a later flush's upscaling (committed before the list's command buffer) may update textures they sample
        if (tf_mode) commit_list_cb();
        if (wrote_color) col.gpu_newer = true;
        col.version++;
        if (dep && wrote_depth) dep->gpu_newer = true;
        if (eager_writeback) {
            if (col.gpu_newer) write_back_color(col);
            if (dep && dep->gpu_newer) write_back_depth(*dep);
        }
        if (g_stats)
            LOG("gpu metal: flush %zu tris (%zu kernel draws): prepare+encode %.2f ms%s%s%s\n", mb.idx.size() / 3, mb.jobs.size(),
                t1 - t0, eager_writeback ? " (eager write-back)" : "", restore_c ? ", colour restored" : "",
                restore_d ? ", depth restored" : "");
    }
}

void metal_poll() {
    if (!active) return;
    {
        static double last = now_ms();
        const double t = now_ms();
        if (t - last >= 2000) {
            double b;
            {
                std::lock_guard<std::mutex> lk(busy_mtx);
                b = gpu_busy;
                gpu_busy = 0;
            }
            busy_frac = (float)(b * 1000 / (t - last));
            if (log_timing)
                LOG("[gpu] GPU busy %.1f ms/s (scale %u, MSAA %ux), %llu command buffers, %llu passes (%llu with vertex "
                    "kernels: %llu dispatches, %llu vertices)\n",
                    b * 1000 / ((t - last) / 1000), S, MS, (unsigned long long)cbs_made.exchange(0),
                    (unsigned long long)flushes.exchange(0),
                    (unsigned long long)vs_passes.exchange(0), (unsigned long long)vs_dispatches.exchange(0),
                    (unsigned long long)vs_verts.exchange(0));
            if (log_timing) {
                double ug;
                {
                    std::lock_guard<std::mutex> lk(busy_mtx);
                    ug = st_upscale_gpu;
                    st_upscale_gpu = 0;
                }
                LOG("[gpu] textures: %u decoded in %.2f ms, %u uploaded in %.2f ms; filter %s: %u upscaled + %u updated "
                    "(%u tiles) in %.2f ms (GPU %.2f ms), %zu MiB\n",
                    g_tex_decodes, g_tex_decode_ms, st_uploads, st_upload_ms, metal_texture_filter_name(tf_mode), st_upscaled,
                    st_partial, st_tiles, st_upscale_ms, ug * 1000, tf_bytes >> 20);
                g_tex_decodes = st_uploads = st_upscaled = st_partial = st_tiles = 0;
                g_tex_decode_ms = st_upload_ms = st_upscale_ms = 0;
            }
            last = t;
        }
    }
    while (!pending.empty()) {
        const MTLCommandBufferStatus st = pending.front()->cb.status;
        if (st != MTLCommandBufferStatusCompleted && st != MTLCommandBufferStatusError) break;
        drain(0);
    }
}

void metal_end_list() {
    if (!active) return;
    @autoreleasepool {
        commit_list_cb();
    }
}

void metal_guest_read(u32 paddr, u32 size) {
    if (!active) return;
    drain_touching(paddr, size);
    for (auto &r : rts)
        if (r->gpu_newer && overlaps(r->addr, rt_bytes(*r), paddr, size)) {
            @autoreleasepool {
                if (r->depth)
                    write_back_depth(*r);
                else
                    write_back_color(*r);
            }
        }
}

void metal_guest_prewrite(u32 paddr, u32 size) {
    if (!active) return;
    // targets the write replaces completely need no write-back of what the GPU holds
    for (auto &op : pending)
        if (op->rt_writeback && op->rt->addr >= paddr && op->rt->addr + rt_bytes(*op->rt) <= paddr + size)
            op->rt_writeback = false;
    drain_touching(paddr, size);
    for (auto &r : rts) {
        if (!r->gpu_newer || !overlaps(r->addr, rt_bytes(*r), paddr, size)) continue;
        if (r->addr >= paddr && r->addr + rt_bytes(*r) <= paddr + size) {
            r->gpu_newer = false;
        } else {
            @autoreleasepool {
                if (r->depth)
                    write_back_depth(*r);
                else
                    write_back_color(*r);
            }
        }
    }
}

void metal_writeback_all() {
    if (!active) return;
    drain_all();
    metal_guest_read(0, 0xffffffffu);
}

void metal_guest_written(u32 paddr, u32 size, bool definite) {
    if (!active) return;
    // A definite write (fill, transfer, copy) makes the guest copy authoritative again (a hash can't
    // tell when e.g. a clear writes the same bytes as last time while the GPU holds newer depth).
    // A possible write (cache flush / invalidate reported by the OS layer) only counts if the
    // guest bytes really changed.
    for (auto &r : rts)
        if (overlaps(r->addr, rt_bytes(*r), paddr, size)) {
            if (definite || !r->synced || hash_mem(pptr(r->addr), rt_bytes(*r)) != r->hash) r->synced = false;
        }
    drop_displays(paddr, size, definite);
}

namespace {
// display images made from guest memory in [paddr, paddr + size): dropped if it was written (definite: certainly)
void drop_displays(u32 paddr, u32 size, bool definite) {
    std::lock_guard<std::mutex> lk(disp_mtx);
    for (auto it = displays.begin(); it != displays.end();) {
        const DisplayImage &d = it->second;
        if (overlaps(d.out_addr, d.out_size, paddr, size) && (definite || (!d.cb && hash_mem(pptr(d.out_addr), d.out_size) != d.out_hash))) {
            if (g_stats)
                LOG("gpu metal: display image %08x dropped (%s write %08x+%x)\n", d.out_addr, definite ? "GPU" : "CPU",
                    paddr, size);
            if (!d.cb && recyclable(d.buf)) disp_free.push_back(d.buf);
            it = displays.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace

namespace {
// the colour target a display transfer reads, starting at one of its tile rows (row0)
GRT *transfer_source(u32 in_paddr, u32 in_dims, u32 out_dims, u32 &row0) {
    const u32 in_w = in_dims & 0xffff, out_h0 = out_dims >> 16;
    GRT *src = nullptr;
    for (auto &r : rts) {
        if (r->depth || !r->synced || r->w != in_w || in_paddr < r->addr) continue;
        const u32 line = r->w * pixel_bytes(cbuf_to_pixfmt(r->fmt));
        const u32 off = in_paddr - r->addr;
        if (off % (line * 8) || off / line + out_h0 > r->h) continue;
        src = r.get();
        row0 = off / line;
    }
    if (src && (out_dims & 0xffff) > src->w) return nullptr;  // (narrower: the transfer keeps the first columns)
    return src;
}

// TW: the target's first columns (scaled) the transfer keeps (all, or fewer)
void encode_rot(id<MTLComputeCommandEncoder> ce, GRT &src, id<MTLBuffer> dst, u32 rows, u32 TW, u32 flags, u32 row0) {
    [ce setComputePipelineState:k_rot];
    [ce setTexture:src.tex atIndex:0];
    [ce setBuffer:dst offset:0 atIndex:0];
    const u32 key = key_black_bottom && rows == 320 * S ? 2 : 0;  // (a bottom screen: 320 rows)
    const u32 prm[4] = {TW, rows, (flags & 1) | key, row0 * S};
    [ce setBytes:prm length:16 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(rows, TW, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

// new display image entry for out_paddr (lock held), replacing any image the output overlaps; the
// buffers of finished images go back to disp_free (those of unfinished ones when they finish)
DisplayImage &new_display(u32 out_paddr, u32 out_size, u32 out_dims, u32 flags, u32 w, u32 h) {
    const u32 out_h0 = out_dims >> 16;
    for (auto it = displays.begin(); it != displays.end();) {
        if (overlaps(it->second.out_addr, it->second.out_size, out_paddr, out_size)) {
            if (!it->second.cb && recyclable(it->second.buf)) disp_free.push_back(it->second.buf);
            it = displays.erase(it);
        } else {
            ++it;
        }
    }
    DisplayImage &d = displays[out_paddr];
    d.out_addr = out_paddr;
    d.out_size = out_size;
    d.w = w;
    d.h = h;
    d.hs = ((flags >> 24) & 3) ? 1 : 0;
    d.vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
    d.out_w = (out_dims & 0xffff) >> d.hs;
    d.out_bpp = pixel_bytes((flags >> 12) & 7);
    d.scale = S;
    d.version = ++disp_version;
    d.screen = out_h0 == 400 ? 0 : out_h0 == 320 ? 1 : -1;
    if (d.screen >= 0) last_screen_addr[d.screen] = out_paddr;
    return d;
}
}  // namespace

// (set_clear_after_top_transfer) a target transferred again with no draw since its clear was asked for: cleared
// now, so the transfer shows nothing (the bottom screen's pass drew nothing over the top screen's picture)
void clear_now(GRT &r, id<MTLCommandBuffer> cb) {
    r.clear_next = false;
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = r.tex;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    r.ms_valid = false;
}

bool metal_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size) {
    if (!active || eager_writeback || (flags & 2)) return false;
    u32 row0 = 0;
    GRT *src = transfer_source(in_paddr, in_dims, out_dims, row0);
    if (!src || !src->gpu_newer || overlaps(src->addr, rt_bytes(*src), out_paddr, out_size)) return false;
    for (auto &op : pending)
        if (op->rt_writeback && op->rt == src) return false;
    metal_guest_prewrite(out_paddr, out_size);
    auto op = std::make_unique<PendingXfer>();
    const u32 TW = (out_dims & 0xffff) * S, rows = (out_dims >> 16) * S;
    @autoreleasepool {
        op->cb = new_cb();
        if (src->clear_next) clear_now(*src, op->cb);
        op->gbytes = take_buffer(xbuf_free, rt_bytes(*src));
        op->obytes = take_buffer(xbuf_free, out_size);
        {
            std::lock_guard<std::mutex> lk(disp_mtx);
            op->disp = take_buffer(disp_free, (size_t)TW * rows * 4);
        }
        id<MTLComputeCommandEncoder> ce = [op->cb computeCommandEncoder];
        label(ce, "display transfer %08x -> %08x", in_paddr, out_paddr);
        encode_guest_color(ce, *src, op->gbytes);
        [ce setComputePipelineState:k_tconv];
        [ce setBuffer:op->gbytes offset:0 atIndex:0];
        [ce setBuffer:op->obytes offset:0 atIndex:1];
        const u32 scaling = (flags >> 24) & 3;
        const u32 out_w = (out_dims & 0xffff) >> (scaling ? 1 : 0), out_h = (out_dims >> 16) >> (scaling == 2 ? 1 : 0);
        const u32 tp[4] = {in_dims & 0xffff, out_w, out_h, flags}, tq[4] = {in_paddr - src->addr, 0, 0, 0};
        [ce setBytes:tp length:16 atIndex:2];
        [ce setBytes:tq length:16 atIndex:3];
        dispatch2d(ce, out_w, out_h);
        encode_rot(ce, *src, op->disp, rows, TW, flags, row0);
        [ce endEncoding];
        [op->cb commit];
    }
    src->gpu_newer = false;  // the pending transfer writes it back
    src->own = false;
    if (clear_after_top) clear_overlapping(*src);
    op->rt = src;
    op->in_paddr = in_paddr;
    op->out_paddr = out_paddr;
    op->in_dims = in_dims;
    op->out_dims = out_dims;
    op->flags = flags;
    op->out_size = out_size;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        DisplayImage &d = new_display(out_paddr, out_size, out_dims, flags, rows, TW);
        d.cb = op->cb;
        d.buf = op->disp;
        d.incomplete = src->skipped > 0;
        src->skipped = 0;
    }
    pending.push_back(std::move(op));
    return true;
}

bool metal_texture_copy(u32 in_paddr, u32 out_paddr, u32 size, u32 in_width, u32 in_gap, u32 out_width, u32 out_gap) {
    if (!active || !size) return false;
    // the source: a GPU copy the copy starts in at one of its tile rows
    GRT *src = nullptr;
    u32 row = 0, bpp = 0;
    for (auto &r : rts) {
        if (!gpu_valid(*r) || in_paddr < r->addr || in_paddr >= r->addr + rt_bytes(*r)) continue;
        const u32 b = grt_bpp(*r), rw = r->w * 8 * b;
        if ((in_paddr - r->addr) % rw) continue;
        src = r.get();
        row = rw;
        bpp = b;
        break;
    }
    if (!src || src->clear_next) return false;
    // whole tiles of the first columns of whole tile rows (a contiguous copy: whole tile rows)
    if (in_gap && in_width + in_gap != row) return false;
    const u32 line = in_gap ? in_width : row;
    if (line % (64 * bpp) || size % line) return false;
    const u32 cols = line / (8 * bpp), rows = size / line * 8, sy0 = (in_paddr - src->addr) / row * 8;
    if (sy0 + rows > src->h) return false;
    // into the first columns of an image as wide as the output's rows with their gaps (contiguous: packed)
    if (out_gap && (out_width != line || (out_width + out_gap) % (64 * bpp))) return false;
    const u32 wd = out_gap ? (out_width + out_gap) / (8 * bpp) : cols;
    const u32 bytes = (rows / 8 - 1) * wd * 8 * bpp + line;
    GRT *d = dest_surface(out_paddr, wd, rows, src->fmt, src, bytes);
    if (!d) return false;
    @autoreleasepool {
        id<MTLBlitCommandEncoder> be = [list_cb() blitCommandEncoder];
        label(be, "copy %08x -> %08x", in_paddr, out_paddr);
        [be copyFromTexture:src->tex sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, sy0 * S, 0)
                 sourceSize:MTLSizeMake(cols * S, rows * S, 1) toTexture:d->tex destinationSlice:0 destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
        [be endEncoding];
    }
    gpu_wrote(*d, out_paddr, bytes);
    return true;
}

bool metal_gpu_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    if (!active || (flags & 2) || !(flags & 0x20) || (flags & 4)) return false;  // tiled to tiled, not cropped
    const u32 in_w = in_dims & 0xffff, in_h = in_dims >> 16, out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const u32 in_fmt = (flags >> 8) & 7, out_fmt = (flags >> 12) & 7, scaling = (flags >> 24) & 3;
    if (in_w != out_w0 || in_h != out_h0 || in_fmt > 4 || out_fmt > 4 || scaling > 2 || in_w % 8 || in_h % 8) return false;
    const u32 hs = scaling ? 1 : 0, vs = scaling == 2 ? 1 : 0;
    const u32 out_w = out_w0 >> hs, out_h = out_h0 >> vs;
    if (!out_w || !out_h || out_w % 8 || out_h % 8) return false;
    const u32 sfmt = pix_to_cbuf(in_fmt), rw = in_w * 8 * pixel_bytes(in_fmt);
    GRT *src = nullptr;
    u32 sy0 = 0;
    for (auto &r : rts) {
        if (r->w != in_w || r->fmt != sfmt || !gpu_valid(*r) || in_paddr < r->addr || in_paddr >= r->addr + rt_bytes(*r) ||
            (in_paddr - r->addr) % rw)
            continue;
        src = r.get();
        sy0 = (in_paddr - r->addr) / rw * 8;
        break;
    }
    if (!src || src->clear_next) return false;
    if (sy0 + in_h > src->h) {  // (an image read taller than what was copied there)
        if (sy0) return false;
        @autoreleasepool {
            src = grow(*src, in_h, list_cb());
        }
        if (!src) return false;
    }
    const u32 bytes = out_w * out_h * pixel_bytes(out_fmt);
    GRT *d = dest_surface(out_paddr, out_w, out_h, pix_to_cbuf(out_fmt), src, bytes);
    if (!d) return false;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> ce = [list_cb() computeCommandEncoder];
        label(ce, "transfer %08x -> %08x", in_paddr, out_paddr);
        [ce setComputePipelineState:k_xfer];
        [ce setTexture:src->tex atIndex:0];
        [ce setTexture:d->tex atIndex:1];
        const u32 p[4] = {out_w * S, out_h * S, sy0 * S, 0};
        const u32 q[4] = {hs, vs, flags & 1, in_fmt == 1 || in_fmt == 2 ? 1u : 0u};  // RGB8, RGB565: no alpha
        [ce setBytes:p length:16 atIndex:0];
        [ce setBytes:q length:16 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(out_w * S, out_h * S, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [ce endEncoding];
    }
    gpu_wrote(*d, out_paddr, bytes);
    return true;
}

bool metal_surface_for(u32 addr, u32 w, u32 h, u32 fmt) {
    if (!active || fmt > 4) return false;
    GRT *r = find_color(addr, w, fmt);
    if (!r || !gpu_valid(*r) || r->clear_next) return false;
    if (r->h < h) {
        @autoreleasepool {
            r = grow(*r, h, list_cb());
        }
    }
    return r != nullptr;
}

void metal_display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags) {
    if (!active) return;
    if (flags & 2) return;  // linear input: not a render target
    u32 row0 = 0;
    GRT *src = transfer_source(in_paddr, in_dims, out_dims, row0);
    if (!src) {
        if (g_stats) LOG("gpu metal: transfer %08x -> %08x: no matching render target, no display image\n", in_paddr, out_paddr);
        return;
    }
    if (hash_mem(pptr(src->addr), rt_bytes(*src)) != src->hash) {  // guest changed since
        if (g_stats) LOG("gpu metal: transfer %08x -> %08x: target changed in guest memory, no display image\n", in_paddr, out_paddr);
        return;
    }
    const u32 out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const u32 TW = out_w0 * S, rows = out_h0 * S;
    id<MTLBuffer> buf;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        buf = take_buffer(disp_free, (size_t)TW * rows * 4);
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = new_cb();
        if (src->clear_next) clear_now(*src, cb);
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        encode_rot(ce, *src, buf, rows, TW, flags, row0);
        [ce endEncoding];
        [cb commit];
        wait_cb(cb, "display image", out_paddr);
    }
    const u32 hs = ((flags >> 24) & 3) ? 1 : 0, vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
    const u32 out_size = (out_w0 >> hs) * (out_h0 >> vs) * pixel_bytes((flags >> 12) & 7);
    std::lock_guard<std::mutex> lk(disp_mtx);
    DisplayImage &d = new_display(out_paddr, out_size, out_dims, flags, rows, TW);
    d.out_hash = hash_mem(pptr(out_paddr), out_size);
    d.cb = nil;
    d.buf = buf;
    d.incomplete = src->skipped > 0;
    src->skipped = 0;
    src->own = false;
    if (clear_after_top) clear_overlapping(*src);
}

// The bottom screen's pass of some games (Animal Crossing: New Leaf) reuses the render target the top screen's
// picture came from and covers it with a background; with that background hidden (a game profile's "set"), the
// target is cleared to transparent after each display transfer (before the next draw into it, so a picture
// transferred again unchanged stays), so the bottom screen's picture keeps only what its pass draws (and its
// alpha says where that is). The top screen's pass draws all of its picture anyway.
void metal_set_clear_after_top_transfer(bool on) { clear_after_top = on; }
// One-screen play shows the bottom screen over the top one: black there is made transparent as the display image is
// made (a screen left black shows nothing; a dialog's choices on it float), and what the game drew translucent, to
// blend with a background now hidden, more opaque.
void metal_set_key_black_bottom(bool on) { key_black_bottom = on; }

namespace {
// The part of a display image the screen shows: the framebuffer (`lines` rows of 240 pixels, `stride` bytes
// apart) may start inside a display transfer's output, e.g. NSMB2 transfers 256x416 and shows rows 16 to 415.
// The image's x runs along the output rows, its y backwards along each row's pixels. (disp_mtx held)
struct DisplayView {
    const DisplayImage *d = nullptr;
    u32 x0 = 0, y0 = 0, w = 0, h = 0;
    u64 key = 0;  // changes with the image or the part shown
};
bool find_view(u32 fb, u32 stride, u32 lines, DisplayView &v) {
    for (const auto &[addr, d] : displays) {
        if (fb < d.out_addr || fb >= d.out_addr + d.out_size) continue;
        const u32 row_bytes = d.out_w * d.out_bpp;
        if (!row_bytes || (stride && stride != row_bytes)) return false;
        const u32 off = fb - d.out_addr, r0 = off / row_bytes, c0 = off % row_bytes / d.out_bpp;
        const u32 fx = d.scale << d.vs, fy = d.scale << d.hs;  // image pixels per output row / pixel
        if (c0 + 240 > d.out_w || (r0 + lines) * fx > d.w || (c0 + 240) * fy > d.h) return false;
        v.d = &d;
        v.x0 = r0 * fx;
        v.w = lines * fx;
        v.y0 = d.h - (c0 + 240) * fy;
        v.h = 240 * fy;
        v.key = d.version << 20 ^ off;
        return true;
    }
    return false;
}
// The image for the framebuffer finished on the GPU, waited for without disp_mtx held (the GX thread takes it to
// finish transfers and to replace images meanwhile); false: no image there.
bool wait_for_view(u32 fb, u32 stride, u32 lines) {
    id<MTLCommandBuffer> cb;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        DisplayView v;
        if (!find_view(fb, stride, lines, v)) return false;
        cb = v.d->cb;
    }
    if (cb) [cb waitUntilCompleted];
    return true;
}
// calls fn with the view's pixels, in place (px: the image's pixels, rows of dw)
void view_pixels(const u32 *px, u32 dw, u32 dh, const DisplayView &v, const std::function<void(const ScreenImage &)> &fn) {
    fn(ScreenImage{px + (size_t)v.y0 * dw + v.x0, (int)v.w, (int)v.h, (int)dw, {}});
}
void with_view(const DisplayView &v, const std::function<void(const ScreenImage &)> &fn) {
    if (v.d->cb) [v.d->cb waitUntilCompleted];
    view_pixels((const u32 *)v.d->buf.contents, v.d->w, v.d->h, v, fn);
}
}  // namespace

bool metal_get_display_image_for(u32 fb_paddr, u32 stride, u32 lines, const u32 **rgba, int *w, int *h) {
    thread_local std::vector<u32> out;
    if (!wait_for_view(fb_paddr, stride, lines)) return false;
    std::lock_guard<std::mutex> lk(disp_mtx);
    DisplayView v;
    if (!find_view(fb_paddr, stride, lines, v)) return false;
    with_view(v, [&](const ScreenImage &si) {
        out.resize((size_t)si.w * si.h);
        for (int y = 0; y < si.h; y++) memcpy(&out[(size_t)y * si.w], si.px + (size_t)y * si.pitch, (size_t)si.w * 4);
        *w = si.w;
        *h = si.h;
    });
    *rgba = out.data();
    return true;
}

u64 metal_use_display_image(u32 fb_paddr, u32 stride, u32 lines, u64 have, const std::function<void(const ScreenImage &)> &fn) {
    if (!wait_for_view(fb_paddr, stride, lines)) return 0;
    // fn (an upload to the UI's texture: slow for a large image) runs without disp_mtx held, the buffer kept from
    // being reused meanwhile
    DisplayView v;
    id<MTLBuffer> buf;
    u32 dw, dh;
    {
        std::unique_lock<std::mutex> lk(disp_mtx);
        if (!find_view(fb_paddr, stride, lines, v)) return 0;
        if (v.key == have) return have;
        // a frame drawn while new pipelines were compiling lacks those draws (often most of the scene: a flash of
        // the clear colour); keep showing the last image instead, for a few frames at most
        static int held[2];
        int &h = held[lines == 400 ? 0 : 1];
        if (v.d->incomplete && have && h < 6) {
            h++;
            return have;
        }
        h = 0;
        if (v.d->cb) [v.d->cb waitUntilCompleted];  // (an image made since wait_for_view: rare)
        buf = v.d->buf;
        dw = v.d->w;
        dh = v.d->h;
        disp_reading.push_back(buf);
    }
    view_pixels((const u32 *)buf.contents, dw, dh, v, fn);
    std::lock_guard<std::mutex> lk(disp_mtx);
    for (size_t i = 0; i < disp_reading.size(); i++)
        if (disp_reading[i] == buf) {
            disp_reading.erase(disp_reading.begin() + i);
            break;
        }
    return v.key;
}

bool metal_get_display_image(int screen, const u32 **rgba, int *w, int *h) {
    u32 addr;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        if (screen < 0 || screen > 1 || !last_screen_addr[screen]) return false;
        addr = last_screen_addr[screen];
    }
    return metal_get_display_image_for(addr, 0, screen ? 320 : 400, rgba, w, h);
}

const HwBackend *metal_backend() {
    static const HwBackend b = {
        .name = "metal",
        .init = metal_init,
        .flush = metal_flush,
        .vs_kernel = metal_vs_kernel,
        .guest_read = metal_guest_read,
        .guest_written = metal_guest_written,
        .guest_prewrite = metal_guest_prewrite,
        .writeback_all = metal_writeback_all,
        .poll = metal_poll,
        .texture_copy = metal_texture_copy,
        .gpu_transfer = metal_gpu_transfer,
        .surface_for = metal_surface_for,
        .end_list = metal_end_list,
        .defer_transfer = metal_defer_transfer,
        .display_transfer = metal_display_transfer,
        .gpu_made = metal_gpu_made,
        .note_gpu_output = metal_note_gpu_output,
        .get_display_image_for = metal_get_display_image_for,
        .use_display_image = metal_use_display_image,
        .get_display_image = metal_get_display_image,
        .set_scale = metal_set_scale,
        .scale = metal_scale,
        .scale_is_auto = metal_scale_is_auto,
        .busy = metal_busy,
        .set_texture_filter = metal_set_texture_filter,
        .texture_filter_count = metal_texture_filter_count,
        .texture_filter_name = metal_texture_filter_name,
        .set_clear_after_top_transfer = metal_set_clear_after_top_transfer,
        .set_key_black_bottom = metal_set_key_black_bottom,
        .emit_shaders = metal_emit_shaders,
    };
    return &b;
}

}  // namespace gpu
