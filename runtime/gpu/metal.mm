// Metal backend: rasterization and the whole fragment pipeline (textures, fragment lighting, TEV,
// fog, alpha / stencil / depth tests, blending, logic ops) run on the GPU at an internal resolution
// scale (changeable while running), optionally multisampled. Vertex programs run as compute kernels;
// clipping, culling and snapping in the shared vertex shader. Guest memory stays authoritative: render targets are synchronised with it
// (hash-checked) before drawing and colour is written back (downsampled to 1x, PICA tiled format)
// after every batch; depth / stencil are written back lazily when a GPU operation reads them.
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

#include "gpu/pica.h"
#include "loader.h"

extern std::string g_save_base;

namespace gpu {

namespace {
// ---------------------------------------------------------------- shader sources
const char *kCommonMSL = R"(
#include <metal_stdlib>
using namespace metal;
struct VOut {
    float4 pos [[position]];
    float4 color;
    float4 tc01;
    float4 tc2w;
    float4 quat;
    float4 view;
    float2 zw;  // clip-space z, w
};
struct Light { float4 spec0, spec1, diffuse, ambient, pos, spot, att; };
struct Uni {
    float4 rt;       // w, h (1x), scale, zmax
    float4 depth;    // scale, offset, quantum, 0
    int4 scissor;    // exclude rect x1, y1, x2, y2 (PICA coordinates, y up)
    uint4 konst0, konst1;
    uint4 tevc;      // initial out, buffer, next buffer
    uint4 misc;      // blend color, fog color, alpha ref, shadow bias
    uint4 border;
    float4 bias;
    float4 lscale0;  // d0, d1, sp, fr
    float4 lscale1;  // rb, rg, rr
    float4 ambient;
    Light lights[8];
};
)";
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
    o.clip[0] = -z;
    o.clip[1] = z + w;
    o.clip[2] = w - 0.00001;
    o.clip[3] = D.flags.x != 0u ? x * D.clip.x + y * D.clip.y + z * D.clip.z + w * D.clip.w : 1.0;
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
// p: w, h (1x), scale
kernel void k_down(texture2d<float, access::read> src [[texture(0)]], device uint *dst [[buffer(0)]],
                   constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint s = p.z;
    uint4 acc = uint4(0);
    for (uint j = 0; j < s; j++)
        for (uint i = 0; i < s; i++) acc += uint4(round(src.read(uint2(gid.x * s + i, gid.y * s + j)) * 255.0));
    const uint n = s * s;
    acc = (acc + n / 2) / n;
    dst[gid.y * p.x + gid.x] = acc.r | (acc.g << 8) | (acc.b << 16) | (acc.a << 24);
}
kernel void k_up(device const uint *src [[buffer(0)]], texture2d<float, access::write> dst [[texture(0)]],
                 constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.x * p.z || gid.y >= p.y * p.z) return;
    const uint c = src[(gid.y / p.z) * p.x + gid.x / p.z];
    dst.write(float4(c & 255u, (c >> 8) & 255u, (c >> 16) & 255u, c >> 24) / 255.0, gid);
}
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
// upright display image: p = texture width, rows used, flip, first row
kernel void k_rot(texture2d<float, access::read> src [[texture(0)]], device uint *dst [[buffer(0)]],
                  constant uint4 &p [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.y || gid.y >= p.x) return;
    const uint row = (p.z ? p.y - 1 - gid.x : gid.x) + p.w;
    const uint col = p.x - 1 - gid.y;
    const uint4 b = uint4(round(saturate(src.read(uint2(col, row))) * 255.0));
    dst[gid.y * p.y + gid.x] = b.r | (b.g << 8) | (b.b << 16) | (255u << 24);
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
id<MTLComputePipelineState> k_down, k_up, k_rot;
u32 S = 1;                     // internal resolution scale
u32 MS = 1;                    // MSAA samples per pixel of the scaled render targets (GPU_MSAA)
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
bool gpu_timing = true, log_timing = false;
std::atomic<float> busy_frac{0};
std::atomic<u64> vs_dispatches{0}, vs_verts{0}, vs_passes{0}, flushes{0};  // vertex kernel work, for the stats line
std::mutex busy_mtx;
double gpu_busy = 0, busy_end = 0;  // union of the command buffers' GPU execution intervals
id<MTLCommandBuffer> new_cb() {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    if (gpu_timing)
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
    bool ms_valid = false;   // ms matches tex (else it is restored from tex before drawing)
    bool synced = false;     // GPU copy corresponds to guest memory as hashed below
    u64 hash = 0;
    bool gpu_newer = false;  // depth/stencil changes not yet written to guest memory
};
std::vector<std::unique_ptr<GRT>> rts;

u32 rt_bytes(const GRT &r) {
    if (r.depth) return r.w * r.h * (r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4);
    return r.w * r.h * pixel_bytes(cbuf_to_pixfmt(r.fmt));
}

GRT &get_rt(u32 addr, u32 w, u32 h, u32 fmt, bool depth) {
    for (auto &r : rts)
        if (r->addr == addr && r->w == w && r->h == h && r->fmt == fmt && r->depth == depth) return *r;
    auto r = std::make_unique<GRT>();
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

id<MTLBuffer> stage_color, stage_depth, stage_stencil;

// decode a guest color target into linear 1x rows (memory order) of packed RGBA
void guest_color_to_linear(const GRT &r, u32 *out) {
    const u8 *p = pptr(r.addr);
    const u32 pf = cbuf_to_pixfmt(r.fmt), bpp = pixel_bytes(pf);
    for (u32 y = 0; y < r.h; y++)
        for (u32 x = 0; x < r.w; x++) out[y * r.w + x] = decode_pixel(pf, p + tiled_offset(x, y, r.w, bpp));
}

void linear_to_guest_color(const GRT &r, const u32 *in, u8 *p = nullptr) {
    if (!p) p = pptr(r.addr);
    const u32 pf = cbuf_to_pixfmt(r.fmt), bpp = pixel_bytes(pf);
    if (pf == 0) {
        for (u32 y = 0; y < r.h; y++)
            for (u32 x = 0; x < r.w; x++) {
                const u32 v = __builtin_bswap32(in[y * r.w + x]);
                memcpy(p + tiled_offset(x, y, r.w, 4), &v, 4);
            }
        return;
    }
    for (u32 y = 0; y < r.h; y++)
        for (u32 x = 0; x < r.w; x++) encode_pixel(pf, in[y * r.w + x], p + tiled_offset(x, y, r.w, bpp));
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
    const u64 hv = hash_bytes(p, rt_bytes(r));
    if (r.synced && hv == r.hash) return la;
    r.synced = true;
    r.hash = hv;
    r.gpu_newer = false;
    const u32 n = r.w * r.h;
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
    if (!r.depth) {
        id<MTLBuffer> up = pool_get((size_t)n * 4, cb);
        u32 *lin = (u32 *)up.contents;
        guest_color_to_linear(r, lin);
        bool uniform = true;
        for (u32 i = 1; i < n && uniform; i++) uniform = lin[i] == lin[0];
        if (uniform) {
            la.clear = true;
            la.color = lin[0];
            return la;
        }
        r.ms_valid = false;
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:k_up];
        [ce setBuffer:up offset:0 atIndex:0];
        [ce setTexture:r.tex atIndex:0];
        const u32 prm[4] = {r.w, r.h, S, 0};
        [ce setBytes:prm length:16 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(r.w * S, r.h * S, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [ce endEncoding];
        return la;
    }
    // depth / stencil
    const u32 bpp = r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4;
    auto zat = [&](u32 x, u32 y, u32 &z, u32 &st) {
        const u8 *q = p + tiled_offset(x, y, r.w, bpp);
        z = r.fmt == 0 ? (u32)(q[0] | q[1] << 8) : (u32)(q[0] | q[1] << 8 | q[2] << 16);
        st = r.fmt == 3 ? q[3] : 0;
    };
    u32 z0, s0;
    zat(0, 0, z0, s0);
    bool uniform = true;
    for (u32 y = 0; y < r.h && uniform; y++)
        for (u32 x = 0; x < r.w && uniform; x++) {
            u32 z, st;
            zat(x, y, z, st);
            uniform = z == z0 && st == s0;
        }
    if (uniform) {
        la.clear = true;
        la.depth = (float)z0 * depth_quantum(r.fmt);
        la.stencil = s0;
        return la;
    }
    r.ms_valid = false;
    const u32 W = r.w * S, H = r.h * S;
    id<MTLBuffer> upd = pool_get((size_t)W * H * 4, cb), ups = pool_get((size_t)W * H, cb);
    float *dz = (float *)upd.contents;
    u8 *dsn = (u8 *)ups.contents;
    const float qn = depth_quantum(r.fmt);
    for (u32 y = 0; y < r.h; y++)
        for (u32 x = 0; x < r.w; x++) {
            u32 z, st;
            zat(x, y, z, st);
            for (u32 j = 0; j < S; j++)
                for (u32 i = 0; i < S; i++) {
                    dz[(size_t)(y * S + j) * W + x * S + i] = (float)z * qn;
                    dsn[(size_t)(y * S + j) * W + x * S + i] = (u8)st;
                }
        }
    id<MTLBlitCommandEncoder> be = [cb blitCommandEncoder];
    [be copyFromBuffer:upd sourceOffset:0 sourceBytesPerRow:W * 4 sourceBytesPerImage:(NSUInteger)W * H * 4
            sourceSize:MTLSizeMake(W, H, 1) toTexture:r.tex destinationSlice:0 destinationLevel:0
     destinationOrigin:MTLOriginMake(0, 0, 0) options:MTLBlitOptionDepthFromDepthStencil];
    [be copyFromBuffer:ups sourceOffset:0 sourceBytesPerRow:W sourceBytesPerImage:(NSUInteger)W * H
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

// write a colour target's GPU contents back to guest memory (box-filtered to 1x)
void write_back_color(GRT &r) {
    ensure_buffer(stage_color, (size_t)r.w * r.h * 4);
    id<MTLCommandBuffer> cb = new_cb();
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    [ce setComputePipelineState:k_down];
    [ce setTexture:r.tex atIndex:0];
    [ce setBuffer:stage_color offset:0 atIndex:0];
    const u32 prm[4] = {r.w, r.h, S, 0};
    [ce setBytes:prm length:16 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(r.w, r.h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [ce endEncoding];
    [cb commit];
    wait_cb(cb, "colour write-back", r.addr);  // also waits for all earlier work on the queue
    linear_to_guest_color(r, (const u32 *)stage_color.contents);
    r.hash = hash_bytes(pptr(r.addr), rt_bytes(r));
    r.synced = true;
    r.gpu_newer = false;
}

// write a depth target's GPU contents back to guest memory (top-left sample of each pixel)
void write_back_depth(GRT &r) {
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
    [cb commit];
    wait_cb(cb, "depth write-back", r.addr);
    const float *dz = (const float *)stage_depth.contents;
    const u8 *dsn = (const u8 *)stage_stencil.contents;
    u8 *p = pptr(r.addr);
    const u32 bpp = r.fmt == 0 ? 2 : r.fmt == 2 ? 3 : 4;
    const float inv = r.fmt == 0 ? 65536.0f : 16777216.0f;
    for (u32 y = 0; y < r.h; y++)
        for (u32 x = 0; x < r.w; x++) {
            const size_t i = (size_t)(y * S) * W + x * S;
            const u32 z = (u32)(dz[i] * inv);
            u8 *q = p + tiled_offset(x, y, r.w, bpp);
            q[0] = z;
            q[1] = z >> 8;
            if (r.fmt != 0) q[2] = z >> 16;
            if (r.fmt == 3) q[3] = dsn[i];
        }
    r.gpu_newer = false;
    r.hash = hash_bytes(p, rt_bytes(r));
    r.synced = true;
}

// ---------------------------------------------------------------- textures and samplers
struct MTex {
    u64 hash = 0;
    id<MTLTexture> tex;
};
std::unordered_map<std::string, MTex> tex_cache;

id<MTLTexture> get_texture(const TexEntry *e) {
    if (!e || e->lv.empty() || e->lv[0].data.empty()) return nil;
    char key[64];
    snprintf(key, sizeof key, "%08x:%u:%u:%u:%u", e->addr, e->w, e->h, e->fmt, e->levels);
    MTex &m = tex_cache[key];
    if (m.tex && m.hash == e->hash) return m.tex;
    u32 nl = 0;
    while (nl < e->lv.size() && !e->lv[nl].data.empty() && e->lv[nl].w >= 1) nl++;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:e->w
                                                                                height:e->h
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
    m.hash = e->hash;
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
id<MTLSamplerState> get_sampler(const TexUnitState &tu, u32 levels) {
    const u64 key = (u64)aniso << 22 | (u64)tu.mag_linear | (u64)tu.min_linear << 1 | (u64)tu.mip_linear << 2 | (u64)tu.wrap_s << 3 |
                    (u64)tu.wrap_t << 6 | (u64)levels << 9 | (u64)(tu.min_level & 15) << 14 |
                    (u64)(tu.max_level & 15) << 18;
    auto it = sampler_cache.find(key);
    if (it != sampler_cache.end()) return it->second;
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.magFilter = tu.mag_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.minFilter = tu.min_linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = levels > 1 ? (tu.mip_linear ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest)
                             : MTLSamplerMipFilterNotMipmapped;
    d.sAddressMode = addr_mode(tu.wrap_s);
    d.tAddressMode = addr_mode(tu.wrap_t);
    d.lodMinClamp = (float)tu.min_level;
    d.lodMaxClamp = levels > 1 ? (float)std::max(tu.min_level, tu.max_level) : 0.0f;
    // anisotropic filtering where the game asked for linear minification of a mipmapped texture
    if (aniso > 1 && tu.min_linear && levels > 1) d.maxAnisotropy = aniso;
    id<MTLSamplerState> s = [dev newSamplerStateWithDescriptor:d];
    sampler_cache[key] = s;
    return s;
}

// ---------------------------------------------------------------- fragment shader generation
struct Gen {
    std::string s;
    void f(const char *fmt, ...) __attribute__((format(printf, 2, 3))) {
        va_list ap, ap2;
        va_start(ap, fmt);
        va_copy(ap2, ap);
        const int n = vsnprintf(nullptr, 0, fmt, ap);
        va_end(ap);
        const size_t old = s.size();
        s.resize(old + n + 1);
        vsnprintf(&s[old], n + 1, fmt, ap2);
        va_end(ap2);
        s.resize(old + n);
    }
};

const char *src_expr(u32 i) {
    switch (i) {
    case 0: return "prim";
    case 1: return "fpc";
    case 2: return "fsc";
    case 3: return "tx0";
    case 4: return "tx1";
    case 5: return "tx2";
    case 6: return "tx3";
    case 13: return "buf";
    case 14: return "K";
    case 15: return "outc";
    default: return "uint4(0)";
    }
}

std::string color_mod(u32 m, const char *v) {
    char b[160];
    switch (m) {
    case 0: snprintf(b, sizeof b, "int3(%s.rgb)", v); break;
    case 1: snprintf(b, sizeof b, "(255 - int3(%s.rgb))", v); break;
    case 2: snprintf(b, sizeof b, "int3(int(%s.a))", v); break;
    case 3: snprintf(b, sizeof b, "int3(255 - int(%s.a))", v); break;
    case 4: snprintf(b, sizeof b, "int3(int(%s.r))", v); break;
    case 5: snprintf(b, sizeof b, "int3(255 - int(%s.r))", v); break;
    case 8: snprintf(b, sizeof b, "int3(int(%s.g))", v); break;
    case 9: snprintf(b, sizeof b, "int3(255 - int(%s.g))", v); break;
    case 12: snprintf(b, sizeof b, "int3(int(%s.b))", v); break;
    case 13: snprintf(b, sizeof b, "int3(255 - int(%s.b))", v); break;
    default: snprintf(b, sizeof b, "int3(%s.rgb)", v); break;
    }
    return b;
}
std::string alpha_mod(u32 m, const char *v) {
    static const char *ch[8] = {"a", "a", "r", "r", "g", "g", "b", "b"};
    char b[96];
    snprintf(b, sizeof b, (m & 1) ? "(255 - int(%s.%s))" : "int(%s.%s)", v, ch[m & 7]);
    return b;
}
// combiner op on int3 (T = "int3") or int; a, b, c are expression names
std::string combine(u32 op, const char *a, const char *b, const char *c, bool vec) {
    char o[256];
    const char *k255 = vec ? "int3(255)" : "255";
    switch (op) {
    case 0: snprintf(o, sizeof o, "%s", a); break;
    case 1: snprintf(o, sizeof o, "(%s * %s / 255)", a, b); break;
    case 2: snprintf(o, sizeof o, "min(%s + %s, %s)", a, b, k255); break;
    case 3: snprintf(o, sizeof o, "clamp(%s + %s - 128, 0, 255)", a, b); break;
    case 4: snprintf(o, sizeof o, "((%s * %s + %s * (255 - %s)) / 255)", a, c, b, c); break;
    case 5: snprintf(o, sizeof o, "max(%s - %s, 0)", a, b); break;
    case 8: snprintf(o, sizeof o, "min((%s * %s + 255 * %s) / 255, %s)", a, b, c, k255); break;
    case 9: snprintf(o, sizeof o, "(min(%s + %s, %s) * %s / 255)", a, b, k255, c); break;
    default: snprintf(o, sizeof o, "%s", a); break;
    }
    return o;
}

const char *cmp_expr(u32 func) {
    switch (func) {
    case 0: return "false";
    case 1: return "true";
    case 2: return "A == B";
    case 3: return "A != B";
    case 4: return "A < B";
    case 5: return "A <= B";
    case 6: return "A > B";
    default: return "A >= B";
    }
}

std::string blend_factor(u32 f, int ch) {
    static const char *c = "rgba";
    char b[64];
    switch (f) {
    case 0: return "0";
    case 1: return "255";
    case 2: snprintf(b, sizeof b, "S.%c", c[ch]); break;
    case 3: snprintf(b, sizeof b, "(255 - S.%c)", c[ch]); break;
    case 4: snprintf(b, sizeof b, "D.%c", c[ch]); break;
    case 5: snprintf(b, sizeof b, "(255 - D.%c)", c[ch]); break;
    case 6: return "S.a";
    case 7: return "(255 - S.a)";
    case 8: return "D.a";
    case 9: return "(255 - D.a)";
    case 10: snprintf(b, sizeof b, "Kb.%c", c[ch]); break;
    case 11: snprintf(b, sizeof b, "(255 - Kb.%c)", c[ch]); break;
    case 12: return "Kb.a";
    case 13: return "(255 - Kb.a)";
    case 14: return ch == 3 ? "255" : "sat";
    default: snprintf(b, sizeof b, "S.%c", c[ch]); break;
    }
    return b;
}
std::string blend_eq(u32 eq, const char *s, const char *d) {
    char b[128];
    switch (eq) {
    case 1: snprintf(b, sizeof b, "clamp((%s - %s) / 255, 0, 255)", s, d); break;
    case 2: snprintf(b, sizeof b, "clamp((%s - %s) / 255, 0, 255)", d, s); break;
    case 3: snprintf(b, sizeof b, "clamp(min(%s, %s) / 255, 0, 255)", s, d); break;
    case 4: snprintf(b, sizeof b, "clamp(max(%s, %s) / 255, 0, 255)", s, d); break;
    default: snprintf(b, sizeof b, "clamp((%s + %s) / 255, 0, 255)", s, d); break;
    }
    return b;
}
const char *logic_expr(u32 op) {
    switch (op) {
    case 0: return "uint4(0)";
    case 1: return "s & d";
    case 2: return "s & ~d";
    case 3: return "s";
    case 4: return "uint4(255)";
    case 5: return "~s";
    case 6: return "d";
    case 7: return "~d";
    case 8: return "~(s & d)";
    case 9: return "s | d";
    case 10: return "~(s | d)";
    case 11: return "s ^ d";
    case 12: return "~(s ^ d)";
    case 13: return "~s & d";
    case 14: return "s | ~d";
    default: return "~s | d";
    }
}

// can the PICA blend / logic op be done by Metal's fixed-function blender?
bool hw_blend(const DrawState &ds) {
    if (!ds.blend) return ds.logic_op == 3;  // copy
    auto ok = [](u32 eq, u32 sf, u32 df) { return eq <= 2 || (eq <= 4 && sf == 1 && df == 1); };
    return ok(ds.eq_rgb, ds.sf_rgb, ds.df_rgb) && ok(ds.eq_a, ds.sf_a, ds.df_a);
}
// depth computed per fragment only for W-buffering (Z-buffer depth is affine in screen space)
bool frag_depth(const DrawState &ds) { return ds.wbuffer; }

// key describing everything the generated fragment shader depends on
std::string fs_key(const DrawState &ds) {
    std::string k;
    auto put = [&](u32 v) { k.append((const char *)&v, 4); };
    put(ds.need_shade | ds.wbuffer << 1 | (ds.scissor_mode == 1) << 2 | ds.depth_bits << 3 | ds.color_write << 9 |
        hw_blend(ds) << 10);
    if (!ds.need_shade) return k;
    for (int i = 0; i < 3; i++) {
        const TexUnitState &t = ds.tex[i];
        put(ds.need_tex[i] | t.null_tex << 1 | t.type << 2 | t.coord << 6 | t.wrap_s << 8 | t.wrap_t << 11 |
            t.shadow_ortho << 14);
    }
    put(ds.tev_count);
    for (u32 i = 0; i < ds.tev_count; i++) {
        const u32 si = ds.tev_list[i];
        const TevStageState &t = ds.tev[si];
        put(si);
        for (int j = 0; j < 3; j++) put(t.csrc[j] | t.asrc[j] << 4 | t.cmod[j] << 8 | t.amod[j] << 12);
        put(t.cop | t.aop << 4 | t.cmul << 8 | t.amul << 12 | t.upd_rgb << 16 | t.upd_a << 17);
    }
    put(ds.lighting);
    if (ds.lighting) {
        const LightingState &l = ds.light;
        put(l.num_lights | l.config << 4 | l.bump_mode << 8 | l.bump_sel << 10 | l.shadow_sel << 12 |
            l.bump_renorm << 14 | l.shadow_enable << 15 | l.shadow_invert << 16 | l.shadow_primary << 17 |
            l.shadow_secondary << 18 | l.shadow_alpha << 19 | l.primary_alpha << 20 | l.secondary_alpha << 21 |
            l.clamp_highlights << 22);
        const LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
        for (auto *s : sels) put(s->enabled | s->abs << 1 | s->input << 2);
        for (u32 i = 0; i < l.num_lights; i++) {
            const LightState &lt = l.lights[i];
            put(lt.num | lt.directional << 3 | lt.two_sided << 4 | lt.geo0 << 5 | lt.geo1 << 6 | lt.dist_atten << 7 |
                lt.spot_atten << 8 | lt.shadow << 9);
        }
    }
    put(ds.fog | ds.fog_flip << 1 | ds.alpha_test << 2 | ds.alpha_func << 3);
    put(ds.blend | ds.eq_rgb << 1 | ds.eq_a << 4 | ds.sf_rgb << 8 | ds.df_rgb << 12 | ds.sf_a << 16 |
        ds.df_a << 20 | ds.logic_op << 24);
    put(ds.color_mask);
    return k;
}

void gen_lighting(Gen &g, const LightingState &l) {
    g.f("    float4 shadow = float4(1.0);\n");
    if (l.shadow_enable) {
        g.f("    shadow = float4(tx%u) / 255.0;\n", l.shadow_sel & 3);
        if (l.shadow_invert) g.f("    shadow = 1.0 - shadow;\n");
    }
    g.f("    float3 sn = float3(0, 0, 1), st = float3(1, 0, 0);\n");
    if (l.bump_mode == 1 || l.bump_mode == 2) {
        g.f("    { float3 p = float3(tx%u.rgb) / 127.5 - 1.0;\n", l.bump_sel & 3);
        if (l.bump_mode == 1) {
            if (l.bump_renorm) g.f("      p.z = sqrt(max(1.0 - (p.x * p.x + p.y * p.y), 0.0));\n");
            g.f("      sn = p; }\n");
        } else {
            g.f("      st = p; }\n");
        }
    }
    g.f("    const float4 q = in.quat;\n"
        "    const float ql = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);\n"
        "    const float qi = ql > 0.0 ? 1.0 / ql : 0.0;\n"
        "    const float3 qv = q.xyz * qi; const float qw = q.w * qi;\n"
        "    const float3 normal = sn + cross(qv, cross(qv, sn) + sn * qw) * 2.0;\n"
        "    const float3 tangent = st + cross(qv, cross(qv, st) + st * qw) * 2.0;\n"
        "    const float3 view = in.view.xyz;\n"
        "    const float vl = length(view); const float3 nv = vl > 0.0 ? view * (1.0 / vl) : view;\n"
        "    float4 dsum = float4(0, 0, 0, 1), ssum = float4(0, 0, 0, 1);\n");
    const char *sc[7] = {"u.lscale0.x", "u.lscale0.y", "u.lscale0.z", "u.lscale0.w",
                         "u.lscale1.x", "u.lscale1.y", "u.lscale1.z"};
    const LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
    for (u32 li = 0; li < l.num_lights; li++) {
        const LightState &lt = l.lights[li];
        g.f("    { // light %u (#%u)\n      constant Light &L = u.lights[%u];\n", li, lt.num, li);
        g.f("      float3 lv = L.pos.xyz%s;\n", lt.directional ? "" : " + view");
        g.f("      const float len = length(lv); if (len > 0.0) lv = lv * (1.0 / len);\n"
            "      const float3 hv = nv + lv;\n"
            "      float dist = 1.0;\n");
        if (lt.dist_atten)
            g.f("      { const float loc = clamp(L.att.x * len + L.att.y, 0.0, 1.0); const float fl = floor(loc * 256.0);\n"
                "        const uint idx = uint(clamp(fl, 0.0, 255.0)); const float2 e = lut[%u * 256 + idx];\n"
                "        dist = e.x + e.y * (loc * 256.0 - float(idx)); }\n",
                16 + lt.num);
        g.f("      const float hl = length(hv); const float3 nh = hl > 0.0 ? hv * (1.0 / hl) : hv;\n"
            "      float lin[6];\n"
            "      lin[0] = dot(normal, nh); lin[1] = dot(nv, nh); lin[2] = dot(normal, nv);\n"
            "      lin[3] = dot(lv, normal); lin[4] = dot(lv, L.spot.xyz);\n");
        if (l.config == 8)
            g.f("      lin[5] = dot(nh - normal * lin[0], tangent);\n");
        else
            g.f("      lin[5] = 0.0;\n");
        auto lv = [&](int si, u32 sampler) {
            const LutSel &s = *sels[si];
            char b[200];
            snprintf(b, sizeof b, "(%s * lutv(lut, %u, %s, %s, %s))", sc[si], sampler,
                     s.input < 6 ? (std::string("lin[") + std::to_string(s.input) + "]").c_str() : "0.0",
                     s.abs ? "true" : "false", lt.two_sided ? "true" : "false");
            return std::string(b);
        };
        g.f("      const float spot = %s;\n", lt.spot_atten && l.sp.enabled ? lv(2, 8 + lt.num).c_str() : "1.0");
        g.f("      const float d0 = %s;\n", l.d0.enabled ? lv(0, 0).c_str() : "1.0");
        g.f("      float3 refl; refl.x = %s;\n", l.rr.enabled ? lv(6, 6).c_str() : "1.0");
        g.f("      refl.y = %s;\n", l.rg.enabled ? lv(5, 5).c_str() : "refl.x");
        g.f("      refl.z = %s;\n", l.rb.enabled ? lv(4, 4).c_str() : "refl.x");
        g.f("      const float d1 = %s;\n", l.d1.enabled ? lv(1, 1).c_str() : "1.0");
        if (li == l.num_lights - 1 && l.fr.enabled) {
            g.f("      { const float fr = %s;\n", lv(3, 3).c_str());
            if (l.primary_alpha) g.f("        dsum.a = fr;\n");
            if (l.secondary_alpha) g.f("        ssum.a = fr;\n");
            g.f("      }\n");
        }
        g.f("      float dp = lin[3]; dp = %s; if (isnan(dp)) dp = 0.0;\n",
            lt.two_sided ? "abs(dp)" : "(dp > 0.0 ? dp : 0.0)");
        g.f("      const float clamp_h = %s;\n", l.clamp_highlights ? "(dp == 0.0 ? 0.0 : 1.0)" : "1.0");
        if (lt.geo0 || lt.geo1)
            g.f("      const float h2 = dot(hv, hv); const float geo = h2 == 0.0 ? 0.0 : min(dp / h2, 1.0);\n");
        g.f("      const float g0 = %s, g1 = %s;\n", lt.geo0 ? "geo" : "1.0", lt.geo1 ? "geo" : "1.0");
        const bool shp = l.shadow_primary && lt.shadow, shs = l.shadow_secondary && lt.shadow;
        g.f("      const float att = dist * spot;\n"
            "      const float3 s0 = d0 * L.spec0.rgb * g0;\n"
            "      const float3 s1 = d1 * refl * L.spec1.rgb * g1;\n"
            "      dsum.rgb += (L.diffuse.rgb * dp%s + L.ambient.rgb) * att;\n"
            "      ssum.rgb += (s0 + s1) * clamp_h * att%s;\n"
            "    }\n",
            shp ? " * shadow.rgb" : "", shs ? " * shadow.rgb" : "");
    }
    if (l.shadow_alpha) {
        if (l.primary_alpha) g.f("    dsum.a *= shadow.a;\n");
        if (l.secondary_alpha) g.f("    ssum.a *= shadow.a;\n");
    }
    g.f("    dsum.rgb += u.ambient.rgb;\n"
        "    fpc = packl(dsum); fsc = packl(ssum);\n");
}

std::string gen_fs(const DrawState &ds) {
    Gen g;
    g.s = kCommonMSL;
    const bool hwb = hw_blend(ds), fdep = frag_depth(ds);
    g.f(fdep ? "struct FOut { float4 c [[color(0)]]; float d [[depth(any)]]; };\n"
             : "struct FOut { float4 c [[color(0)]]; };\n");
    g.f("static inline uint rc(float f) { const float v = round(f * 255.0); return v > 0.0 ? (v >= 255.0 ? 255u : uint(v)) : 0u; }\n"
        "static inline uint4 tou(float4 c) { return uint4(round(saturate(c) * 255.0)); }\n"
        "static inline uint4 unpack(uint c) { return uint4(c & 255u, (c >> 8) & 255u, (c >> 16) & 255u, c >> 24); }\n"
        "static inline uint4 packl(float4 v) { uint4 o; for (int i = 0; i < 4; i++) { float x = isnan(v[i]) ? 0.0 : clamp(v[i], 0.0, 1.0); o[i] = uint(x * 255.0); } return o; }\n"
        "static inline float lutv(device const float2 *lut, uint s, float r, bool absm, bool two) {\n"
        "    if (isnan(r)) r = 0.0; uint idx; float delta;\n"
        "    if (absm) { r = two ? abs(r) : max(r, 0.0); const float fl = floor(r * 256.0); idx = uint(clamp(fl, 0.0, 255.0)); delta = r * 256.0 - float(idx); }\n"
        "    else { const float fl = floor(r * 128.0); const int si = int(clamp(fl, -128.0, 127.0)); delta = r * 128.0 - float(si); idx = uint(si) & 255u; }\n"
        "    const float2 e = lut[s * 256 + idx]; return e.x + e.y * delta; }\n");
    // parameters
    g.f("fragment FOut fs_main(VOut in [[stage_in]], constant Uni &u [[buffer(0)]]");
    if (ds.need_shade && ds.lighting) g.f(", device const float2 *lut [[buffer(1)]]");
    if (ds.need_shade && ds.fog) g.f(", device const float2 *fogl [[buffer(2)]]");
    bool cube = false;
    for (int i = 0; i < 3; i++)
        if (ds.need_shade && ds.need_tex[i] && !ds.tex[i].null_tex) {
            g.f(", texture2d<float> t%d [[texture(%d)]], sampler s%d [[sampler(%d)]]", i, i, i, i);
            if (i == 0 && (ds.tex[0].type == 1 || ds.tex[0].type == 4)) cube = true;
        }
    if (cube)
        for (int f = 1; f < 6; f++) g.f(", texture2d<float> f%d [[texture(%d)]]", f, 2 + f);
    g.f("%s) {\n    FOut o;\n", hwb || !ds.need_shade ? "" : ", float4 dstc [[color(0)]]");
    // depth
    g.f("    float zd = in.zw.x / in.zw.y * u.depth.x + u.depth.y;\n");
    if (ds.wbuffer) g.f("    zd *= in.zw.y;\n");
    g.f("    zd = zd > 0.0 ? min(zd, 1.0) : 0.0;\n");
    if (fdep) g.f("    o.d = float(uint(zd * u.rt.w)) * u.depth.z;\n");
    if (ds.scissor_mode == 1)
        g.f("    { const float2 pp = floor(in.pos.xy / u.rt.z); const int px = int(pp.x), py = int(u.rt.y) - 1 - int(pp.y);\n"
            "      if (px >= u.scissor.x && px < u.scissor.z && py >= u.scissor.y && py < u.scissor.w) discard_fragment(); }\n");
    if (!ds.need_shade) {
        g.f("    o.c = float4(0.0);\n    return o;\n}\n");  // colour writes are masked off
        return g.s;
    }
    g.f("    const float4 col = in.color;\n"
        "    const uint4 prim = uint4(rc(col.x), rc(col.y), rc(col.z), rc(col.w));\n"
        "    const float2 c0 = in.tc01.xy, c1 = in.tc01.zw, c2 = in.tc2w.xy; const float tw = in.tc2w.z;\n"
        "    uint4 tx0 = uint4(0), tx1 = uint4(0), tx2 = uint4(0), tx3 = uint4(0);\n"
        "    uint4 fpc = uint4(0), fsc = uint4(0);\n");
    for (int i = 0; i < 3; i++) {
        if (!ds.need_tex[i]) continue;
        const TexUnitState &t = ds.tex[i];
        if (t.null_tex) {
            g.f("    tx%d = uint4(0, 0, 0, 255);\n", i);
            continue;
        }
        const u32 type = i == 0 ? t.type : 0;
        if (type == 1 || type == 4) {
            // cube map: face selection as in the software renderer
            g.f("    { const float cu = c0.x, cv = c0.y, cw = tw; const float au = abs(cu), av = abs(cv), aw = abs(cw);\n"
                "      float x, y, z; int face;\n"
                "      if (au > av && au > aw) { face = cu > 0 ? 0 : 1; y = cu > 0 ? -cv : cv; x = -cw; z = cu; }\n"
                "      else if (av > aw) { face = cv > 0 ? 2 : 3; x = cv > 0 ? cu : -cu; y = cw; z = cv; }\n"
                "      else { face = cw > 0 ? 4 : 5; y = cw > 0 ? -cv : cv; x = cu; z = cw; }\n"
                "      const float2 uv = float2(x / z * 0.5 + 0.5, y / z * 0.5 + 0.5);\n"
                "      float4 sv;\n"
                "      switch (face) { case 0: sv = t0.sample(s0, uv, level(0)); break; case 1: sv = f1.sample(s0, uv, level(0)); break;\n"
                "        case 2: sv = f2.sample(s0, uv, level(0)); break; case 3: sv = f3.sample(s0, uv, level(0)); break;\n"
                "        case 4: sv = f4.sample(s0, uv, level(0)); break; default: sv = f5.sample(s0, uv, level(0)); break; }\n"
                "      tx0 = tou(sv); }\n");
            continue;
        }
        if (type == 2) {
            g.f("    { const float2 uvs = %s; const uint2 sz = uint2(t0.get_width(), t0.get_height());\n"
                "      const int2 p = clamp(int2(floor(uvs * float2(sz))), int2(0), int2(sz) - 1);\n"
                "      const uint4 b = tou(t0.read(uint2(p)));\n"
                "      const float zz = min(abs(tw), 1.0); const int zi = int(zz * 16777215.0) - int(u.misc.w << 1);\n"
                "      const int zref = int((b.a << 16) | (b.b << 8) | b.g);\n"
                "      tx0 = uint4(zref >= zi ? b.r : 0u); }\n",
                t.shadow_ortho ? "c0" : "c0 / tw");
            continue;
        }
        const char *coord = t.coord == 0 ? "c0" : t.coord == 1 ? "c1" : "c2";
        g.f("    { const float2 uv = %s%s;\n", coord, type == 3 ? " / tw" : "");
        g.f("      const float4 sv = t%d.sample(s%d, uv, bias(u.bias[%d]));\n", i, i, i);
        std::string border;
        auto bcond = [&](u32 wrap, const char *c) {
            if (wrap == 1) border += std::string(border.empty() ? "" : " || ") + "uv." + c + " < 0.0 || uv." + c + " >= 1.0";
            if (wrap == 5) border += std::string(border.empty() ? "" : " || ") + "uv." + c + " >= 1.0";
        };
        bcond(t.wrap_s, "x");
        bcond(t.wrap_t, "y");
        if (!border.empty())
            g.f("      tx%d = (%s) ? unpack(u.border[%d]) : tou(sv); }\n", i, border.c_str(), i);
        else
            g.f("      tx%d = tou(sv); }\n", i);
    }
    if (ds.lighting) gen_lighting(g, ds.light);
    // texture combiners
    g.f("    uint4 outc = unpack(u.tevc.x), buf = unpack(u.tevc.y), nxt = unpack(u.tevc.z);\n");
    for (u32 li = 0; li < ds.tev_count; li++) {
        const u32 si = ds.tev_list[li];
        const TevStageState &t = ds.tev[si];
        g.f("    { // stage %u\n      const uint4 K = unpack(%s[%u]);\n", si, si < 4 ? "u.konst0" : "u.konst1", si & 3);
        for (int k = 0; k < 3; k++) {
            g.f("      const int3 c%d = %s;\n", k, color_mod(t.cmod[k], src_expr(t.csrc[k])).c_str());
            g.f("      const int a%d = %s;\n", k, alpha_mod(t.amod[k], src_expr(t.asrc[k])).c_str());
        }
        if (t.cop == 6 || t.cop == 7)
            g.f("      const int dd = ((c0.r * 2 - 255) * (c1.r * 2 - 255) + 128) / 256 + ((c0.g * 2 - 255) * (c1.g * 2 - 255) + 128) / 256 + ((c0.b * 2 - 255) * (c1.b * 2 - 255) + 128) / 256;\n"
                "      int3 r = int3(clamp(dd, 0, 255));\n");
        else
            g.f("      int3 r = %s;\n", combine(t.cop, "c0", "c1", "c2", true).c_str());
        if (t.cop == 7)
            g.f("      int a = r.x;\n");
        else if (t.aop == 6 || t.aop == 7)
            g.f("      int a = 0;\n");
        else
            g.f("      int a = %s;\n", combine(t.aop, "a0", "a1", "a2", false).c_str());
        if (t.cmul != 1) g.f("      r = min(r * %u, int3(255));\n", t.cmul);
        if (t.amul != 1) g.f("      a = min(a * %u, 255);\n", t.amul);
        g.f("      outc = uint4(uint3(r), uint(a));\n      buf = nxt;\n");
        if (t.upd_rgb) g.f("      nxt.rgb = outc.rgb;\n");
        if (t.upd_a) g.f("      nxt.a = outc.a;\n");
        g.f("    }\n");
    }
    if (ds.alpha_test) g.f("    { const uint A = outc.a, B = u.misc.z; if (!(%s)) discard_fragment(); }\n", cmp_expr(ds.alpha_func));
    if (ds.fog)
        g.f("    { const float fi = (%s) * 128.0; const float fl = clamp(floor(fi), 0.0, 127.0); const uint ii = uint(fl);\n"
            "      const float2 fe = fogl[ii]; const float ff = clamp(fe.x + fe.y * (fi - fl), 0.0, 1.0);\n"
            "      const float3 fc = float3(unpack(u.misc.y).rgb);\n"
            "      outc.rgb = uint3(ff * float3(outc.rgb) + (1.0 - ff) * fc); }\n",
            ds.fog_flip ? "1.0 - zd" : "zd");
    if (hwb) {
        // fixed-function blending / plain copy; channel masks are in the pipeline
        g.f("    o.c = float4(outc) / 255.0;\n    return o;\n}\n");
        return g.s;
    }
    // blending / logic op with the destination (programmable blending)
    g.f("    const uint4 dst = uint4(round(dstc * 255.0));\n    uint4 res;\n");
    if (ds.blend) {
        g.f("    { const int4 S = int4(outc), D = int4(dst), Kb = int4(unpack(u.misc.x)); const int sat = min(S.a, 255 - D.a);\n");
        g.f("      const int4 sf = int4(%s, %s, %s, %s);\n", blend_factor(ds.sf_rgb, 0).c_str(),
            blend_factor(ds.sf_rgb, 1).c_str(), blend_factor(ds.sf_rgb, 2).c_str(), blend_factor(ds.sf_a, 3).c_str());
        g.f("      const int4 df = int4(%s, %s, %s, %s);\n", blend_factor(ds.df_rgb, 0).c_str(),
            blend_factor(ds.df_rgb, 1).c_str(), blend_factor(ds.df_rgb, 2).c_str(), blend_factor(ds.df_a, 3).c_str());
        g.f("      const int4 sm = S * sf, dm = D * df;\n");
        g.f("      res.rgb = uint3(%s);\n", blend_eq(ds.eq_rgb, "sm.rgb", "dm.rgb").c_str());
        g.f("      res.a = uint(%s); }\n", blend_eq(ds.eq_a, "sm.a", "dm.a").c_str());
    } else {
        g.f("    { const uint4 s = outc, d = dst; res = (%s) & uint4(255); }\n", logic_expr(ds.logic_op));
    }
    const u32 m = ds.color_mask;
    g.f("    res = select(dst, res, bool4(%s, %s, %s, %s));\n", (m & 0xff) ? "true" : "false",
        (m & 0xff00) ? "true" : "false", (m & 0xff0000) ? "true" : "false", (m & 0xff000000u) ? "true" : "false");
    g.f("    o.c = float4(res) / 255.0;\n    return o;\n}\n");
    return g.s;
}

// ---------------------------------------------------------------- pipeline cache
// A pipeline is fully described by its fs_key(). Every key a game uses is recorded in a per-user cache
// file, and a list can be shipped with the game (games/<title id>/pipelines.bin). The build compiles the
// shipped list's fragment shaders ahead of time into shaders.metallib (functions named after a hash of
// their MSL source, so a stale library is never used for a changed generator), and all known pipelines are
// created on background threads at startup. A pipeline never seen before is compiled in the background
// and its draws are skipped until it is ready (GPU_SYNC_SHADERS=1 compiles it on the spot instead).
constexpr u32 kKeyMagic = 0x59454B50;  // "PKEY"
constexpr u32 kKeyVersion = 1;         // bump when fs_key() changes

// the DrawState fields gen_fs() and build_pipeline() depend on, from a key (inverse of fs_key)
DrawState decode_key(const std::string &k) {
    DrawState ds{};
    size_t p = 0;
    auto get = [&]() -> u32 {
        u32 v = 0;
        if (p + 4 <= k.size()) memcpy(&v, k.data() + p, 4);
        p += 4;
        return v;
    };
    u32 w = get();
    ds.need_shade = w & 1;
    ds.wbuffer = w >> 1 & 1;
    ds.scissor_mode = (w >> 2 & 1) ? 1 : 0;
    ds.depth_bits = w >> 3 & 63;
    ds.color_write = w >> 9 & 1;
    if (!ds.need_shade) {
        ds.logic_op = (w >> 10 & 1) ? 3 : 0;  // reproduces hw_blend()
        return ds;
    }
    for (int i = 0; i < 3; i++) {
        w = get();
        TexUnitState &t = ds.tex[i];
        ds.need_tex[i] = w & 1;
        t.null_tex = w >> 1 & 1;
        t.type = w >> 2 & 15;
        t.coord = w >> 6 & 3;
        t.wrap_s = w >> 8 & 7;
        t.wrap_t = w >> 11 & 7;
        t.shadow_ortho = w >> 14 & 1;
    }
    ds.tev_count = std::min(get(), 6u);
    for (u32 i = 0; i < ds.tev_count; i++) {
        const u32 si = get() % 6;
        ds.tev_list[i] = (u8)si;
        TevStageState &t = ds.tev[si];
        for (int j = 0; j < 3; j++) {
            w = get();
            t.csrc[j] = w & 15;
            t.asrc[j] = w >> 4 & 15;
            t.cmod[j] = w >> 8 & 15;
            t.amod[j] = w >> 12 & 15;
        }
        w = get();
        t.cop = w & 15;
        t.aop = w >> 4 & 15;
        t.cmul = w >> 8 & 15;
        t.amul = w >> 12 & 15;
        t.upd_rgb = w >> 16 & 1;
        t.upd_a = w >> 17 & 1;
    }
    ds.lighting = get();
    if (ds.lighting) {
        LightingState &l = ds.light;
        w = get();
        l.num_lights = std::min(w & 15, 8u);
        l.config = w >> 4 & 15;
        l.bump_mode = w >> 8 & 3;
        l.bump_sel = w >> 10 & 3;
        l.shadow_sel = w >> 12 & 3;
        l.bump_renorm = w >> 14 & 1;
        l.shadow_enable = w >> 15 & 1;
        l.shadow_invert = w >> 16 & 1;
        l.shadow_primary = w >> 17 & 1;
        l.shadow_secondary = w >> 18 & 1;
        l.shadow_alpha = w >> 19 & 1;
        l.primary_alpha = w >> 20 & 1;
        l.secondary_alpha = w >> 21 & 1;
        l.clamp_highlights = w >> 22 & 1;
        LutSel *sels[7] = {&l.d0, &l.d1, &l.sp, &l.fr, &l.rb, &l.rg, &l.rr};
        for (auto *s : sels) {
            w = get();
            s->enabled = w & 1;
            s->abs = w >> 1 & 1;
            s->input = w >> 2 & 7;
        }
        for (u32 i = 0; i < l.num_lights; i++) {
            w = get();
            LightState &lt = l.lights[i];
            lt.num = w & 7;
            lt.directional = w >> 3 & 1;
            lt.two_sided = w >> 4 & 1;
            lt.geo0 = w >> 5 & 1;
            lt.geo1 = w >> 6 & 1;
            lt.dist_atten = w >> 7 & 1;
            lt.spot_atten = w >> 8 & 1;
            lt.shadow = w >> 9 & 1;
        }
    }
    w = get();
    ds.fog = w & 1;
    ds.fog_flip = w >> 1 & 1;
    ds.alpha_test = w >> 2 & 1;
    ds.alpha_func = w >> 3 & 7;
    w = get();
    ds.blend = w & 1;
    ds.eq_rgb = w >> 1 & 7;
    ds.eq_a = w >> 4 & 15;
    ds.sf_rgb = w >> 8 & 15;
    ds.df_rgb = w >> 12 & 15;
    ds.sf_a = w >> 16 & 15;
    ds.df_a = w >> 20 & 15;
    ds.logic_op = w >> 24 & 15;
    ds.color_mask = get();
    return ds;
}

u64 fnv64(const std::string &s) {
    u64 h = 0xcbf29ce484222325ull;
    for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
    return h;
}
std::string fs_name(const std::string &src) {
    char b[24];
    snprintf(b, sizeof b, "fs_%016llx", (unsigned long long)fnv64(src));
    return b;
}

std::vector<std::string> read_keys(const std::string &path) {
    std::vector<std::string> keys;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return keys;
    u32 hdr[2] = {};
    if (fread(hdr, 4, 2, f) == 2 && hdr[0] == kKeyMagic && hdr[1] == kKeyVersion) {
        u32 n;
        while (fread(&n, 4, 1, f) == 1 && n > 0 && n < 4096) {
            std::string k(n, '\0');
            if (fread(&k[0], 1, n, f) != n) break;
            keys.push_back(std::move(k));
        }
    }
    fclose(f);
    return keys;
}

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
            MTLCompileOptions *opt = [MTLCompileOptions new];
            opt.mathMode = MTLMathModeSafe;
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:(src + kVertexMSL).c_str()]
                                                   options:opt
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
        for (const auto &p : {exe_dir() + "/shaders.metallib", exe_dir() + "/../Resources/shaders.metallib"}) {
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
constexpr u32 kVsMagic = 0x52505356;  // "VSPR"
constexpr u32 kVsVersion = 1;
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

u64 vk_hash(const u32 *code, u32 code_size, const u32 *swz, u32 swz_size) {
    return hash_bytes((const u8 *)code, code_size * 4) * 31 + hash_bytes((const u8 *)swz, swz_size * 4) + code_size;
}
u64 vk_key(u64 prog_hash, u32 entry) { return prog_hash * 0x9E3779B97F4A7C15ull + entry; }

constexpr u32 kVsSpec = 0x43455053;  // "SPEC"
struct VsSpec {
    u32 b = 0;
    u8 iu[4][4] = {};
};
u64 spec_key(u64 key, const VsSpec &sp) { return (key ^ hash_bytes((const u8 *)&sp, sizeof sp)) * 0x100000001B3ull + 1; }
struct VsBlob {
    u32 entry = 0, code_size = 0, swz_size = 0;
    std::vector<u32> code, swz;
    bool spec = false;
    VsSpec sp;
};
bool parse_vs_blob(const std::string &b, VsBlob &v) {
    if (b.size() < 12) return false;
    memcpy(&v.entry, b.data(), 4);
    memcpy(&v.code_size, b.data() + 4, 4);
    memcpy(&v.swz_size, b.data() + 8, 4);
    if (v.code_size > 4096 || v.swz_size > 4096) return false;
    const size_t base = 12 + 4 * (size_t)(v.code_size + v.swz_size);
    if (b.size() == base + 4 + sizeof(VsSpec)) {
        u32 m;
        memcpy(&m, b.data() + base, 4);
        if (m != kVsSpec) return false;
        v.spec = true;
        memcpy(&v.sp, b.data() + base + 4, sizeof(VsSpec));
    } else if (b.size() != base) {
        return false;
    }
    v.code.resize(4096);
    v.swz.resize(4096);
    memcpy(v.code.data(), b.data() + 12, 4 * v.code_size);
    memcpy(v.swz.data(), b.data() + 12 + 4 * v.code_size, 4 * v.swz_size);
    return true;
}
std::string vs_blob(const ShaderSetup &s, u32 entry, const VsSpec *sp) {
    const size_t base = 12 + 4 * (size_t)(s.code_size + s.swizzle_size);
    std::string b(base + (sp ? 4 + sizeof(VsSpec) : 0), '\0');
    const u32 hdr[3] = {entry, s.code_size, s.swizzle_size};
    memcpy(&b[0], hdr, 12);
    memcpy(&b[12], s.code, 4 * s.code_size);
    memcpy(&b[12 + 4 * s.code_size], s.swizzle, 4 * s.swizzle_size);
    if (sp) {
        memcpy(&b[base], &kVsSpec, 4);
        memcpy(&b[base + 4], sp, sizeof(VsSpec));
    }
    return b;
}
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
std::vector<std::string> read_blobs(const std::string &path, u32 magic, u32 version) {
    std::vector<std::string> out;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return out;
    u32 hdr[2] = {};
    if (fread(hdr, 4, 2, f) == 2 && hdr[0] == magic && hdr[1] == version) {
        u32 n;
        while (fread(&n, 4, 1, f) == 1 && n > 0 && n < (1u << 16)) {
            std::string k(n, '\0');
            if (fread(&k[0], 1, n, f) != n) break;
            out.push_back(std::move(k));
        }
    }
    fclose(f);
    return out;
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
                MTLCompileOptions *opt = [MTLCompileOptions new];
                opt.mathMode = MTLMathModeSafe;
                id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:(std::string(kVsCommonMSL) + src).c_str()]
                                                       options:opt
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
// A display transfer from a target with GPU work in flight is not waited for: one command buffer
// downsamples the target (k_down) and makes the display image (k_rot); the CPU part (writing the
// target and the transfer output to guest memory) runs once it completed, when polled at a later
// GPU call or when something touches those memory ranges.
struct PendingXfer {
    id<MTLCommandBuffer> cb;
    id<MTLBuffer> lin;   // target contents at 1x, linear rows
    id<MTLBuffer> disp;  // display image
    GRT *rt = nullptr;
    bool rt_writeback = true;  // the target's memory is still to be written (not overwritten since)
    u32 in_paddr = 0, out_paddr = 0, in_dims = 0, out_dims = 0, flags = 0, out_size = 0;
};
std::deque<std::unique_ptr<PendingXfer>> pending;
std::vector<id<MTLBuffer>> lin_free;   // GX thread only
std::vector<id<MTLBuffer>> disp_free;  // under disp_mtx
std::vector<u8> xfer_scratch;

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
    xfer_scratch.resize(nb);
    linear_to_guest_color(r, (const u32 *)op.lin.contents, xfer_scratch.data());
    if (op.rt_writeback) {
        memcpy(pptr(r.addr), xfer_scratch.data(), nb);
        r.hash = hash_bytes(xfer_scratch.data(), nb);
    }
    transfer_convert(xfer_scratch.data() + (op.in_paddr - r.addr), pptr(op.out_paddr), op.in_dims, op.out_dims,
                     op.flags);
    texcache_invalidate(op.out_paddr, op.out_size);
    rts_overwritten(op.out_paddr, op.out_size);
    lin_free.push_back(op.lin);
    std::lock_guard<std::mutex> lk(disp_mtx);
    auto it = displays.find(op.out_paddr);
    if (it != displays.end() && it->second.cb == op.cb) {
        DisplayImage &d = it->second;
        d.out_hash = hash_bytes(pptr(d.out_addr), d.out_size);
        d.cb = nil;  // the image stays in op.disp
    } else if (disp_free.size() < 6) {
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
        k_down = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_down"] error:&err];
        k_up = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_up"] error:&err];
        k_rot = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_rot"] error:&err];
        if (!k_down || !k_up || !k_rot) return false;
        // GPU_SCALE: 1-4, or auto (default): the frontend picks it from the window size (set_scale)
        const char *sc = getenv("GPU_SCALE");
        scale_fixed = sc && atoi(sc) > 0;
        S = scale_fixed ? (u32)std::clamp(atoi(sc), 1, 4) : 2;
        const char *ms = getenv("GPU_MSAA");
        MS = ms ? (u32)std::clamp(atoi(ms), 1, 8) : kDefaultMSAA;
        while (MS > 1 && ((MS & (MS - 1)) || ![dev supportsTextureSampleCount:MS])) MS--;
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
    active = true;
    LOG("gpu: Metal backend on %s, internal scale %s%u, MSAA %ux, anisotropic filtering %ux\n", dev.name.UTF8String,
        scale_fixed ? "" : "auto, now ", S, MS, aniso);
    @autoreleasepool {
        pipelines_init();
    }
    return true;
}

bool metal_active() { return active; }

// <outdir> <pipeline lists...>: one .metal file per fragment shader (fs_<hash of its source>) plus vs.metal,
// for building shaders.metallib ahead of time. Unchanged files are left alone and stale ones removed, so
// only new shaders get recompiled.
int emit_shaders(int argc, char **argv) {
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
    S = ns;
}
}  // namespace

void set_scale(int s) { want_scale = (u32)std::clamp(s, 1, 4); }
int scale() { return active ? (int)(want_scale.load() ? want_scale.load() : S) : 0; }
bool scale_is_auto() { return active && !scale_fixed; }
float busy() { return active ? busy_frac.load() : 0; }

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
    @autoreleasepool {
        id<MTLCommandBuffer> cb = new_cb();
        GRT &col = get_rt(rt.color_addr, rt.w, rt.h, rt.color_fmt, false);
        GRT *dep = rt.depth_addr && pptr(rt.depth_addr) ? &get_rt(rt.depth_addr, rt.w, rt.h, rt.depth_fmt, true) : nullptr;
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
        const LoadAction cla = sync_rt(col, cb);
        LoadAction dla;
        if (dep) dla = sync_rt(*dep, cb);

        // vertices: compute kernels shade the draws' vertex arrays into gbuf; the CPU's are in cbuf
        auto upload = [&](const void *p, size_t n) {
            id<MTLBuffer> b = pool_get(n, cb);
            if (n) memcpy(b.contents, p, n);
            return b;
        };
        id<MTLBuffer> gbuf = pool_get((size_t)mb.gpu_verts * sizeof(OutVertex), cb);
        id<MTLBuffer> cbuf = upload(mb.cverts.data(), mb.cverts.size() * sizeof(OutVertex));
        id<MTLBuffer> ibuf = upload(mb.idx.data(), mb.idx.size() * 4);
        if (!mb.jobs.empty()) {
            id<MTLBuffer> dbuf = upload(mb.vdata.data(), mb.vdata.size());
            // one dispatch per kernel: its draws' vertices one after the other, with their parameter blocks
            // in a buffer and each thread's draw in tj (draws are often only a few vertices, and a dispatch
            // per draw costs far more GPU time than the shading). The draws write disjoint ranges of gbuf.
            struct Group {
                id<MTLComputePipelineState> ps;
                std::vector<VsParamsG> params;
                std::vector<u32> vids, tj;
            };
            std::vector<Group> groups;
            for (const GpuVsJob &j : mb.jobs) {
                id<MTLComputePipelineState> ps = ((const VsKern *)j.kernel)->ps;
                Group *g = nullptr;
                for (Group &x : groups)
                    if (x.ps == ps) g = &x;
                if (!g) {
                    groups.push_back({ps, {}, {}, {}});
                    g = &groups.back();
                }
                VsParamsG &P = g->params.emplace_back(j.p);
                P.first = (u32)g->vids.size();
                g->vids.insert(g->vids.end(), mb.vids.begin() + j.vids_off, mb.vids.begin() + j.vids_off + j.p.nverts);
                g->tj.insert(g->tj.end(), j.p.nverts, (u32)g->params.size() - 1);
            }
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [ce setBuffer:dbuf offset:0 atIndex:0];
            [ce setBuffer:gbuf offset:0 atIndex:3];
            vs_dispatches += groups.size();
            vs_passes++;
            vs_verts += mb.vids.size();
            for (const Group &g : groups) {
                if (g.vids.empty()) continue;
                [ce setComputePipelineState:g.ps];
                [ce setBuffer:upload(g.vids.data(), g.vids.size() * 4) offset:0 atIndex:1];
                [ce setBuffer:upload(g.params.data(), g.params.size() * sizeof(VsParamsG)) offset:0 atIndex:2];
                [ce setBuffer:upload(g.tj.data(), g.tj.size() * 4) offset:0 atIndex:4];
                const NSUInteger tg = std::min<NSUInteger>(64, g.ps.maxTotalThreadsPerThreadgroup);
                [ce dispatchThreads:MTLSizeMake(g.vids.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            }
            [ce endEncoding];
        }

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
        rp.colorAttachments[0].storeAction = msaa ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionStore;
        if (msaa) rp.colorAttachments[0].resolveTexture = col.tex;
        id<MTLTexture> dt = dep ? (msaa ? dep->ms : dep->tex) : get_scratch_depth(rt.w, rt.h);
        rp.depthAttachment.texture = dt;
        rp.stencilAttachment.texture = dt;
        if (dep) {
            const MTLLoadAction la = dla.clear ? MTLLoadActionClear : restore_d ? MTLLoadActionDontCare : MTLLoadActionLoad;
            rp.depthAttachment.loadAction = la;
            rp.stencilAttachment.loadAction = la;
            rp.depthAttachment.clearDepth = dla.depth;
            rp.stencilAttachment.clearStencil = dla.stencil;
            if (msaa) {
                rp.depthAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
                rp.stencilAttachment.storeAction = MTLStoreActionStoreAndMultisampleResolve;
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
        [re setViewport:(MTLViewport){0, 0, (double)rt.w * S, (double)rt.h * S, 0, 1}];
        [re setDepthClipMode:MTLDepthClipModeClamp];
        [re setFrontFacingWinding:MTLWindingCounterClockwise];
        [re setVertexBuffer:gbuf offset:0 atIndex:0];
        [re setVertexBuffer:cbuf offset:0 atIndex:1];
        [re setVertexBuffer:ibuf offset:0 atIndex:2];
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
        col.ms_valid = true;
        if (dep) dep->ms_valid = true;

        // per-draw uniforms and LUTs go into one shared buffer
        size_t uoff = 0;
        std::vector<std::pair<const void *, size_t>> lut_offs;
        auto ualloc = [&](size_t n) {
            const size_t o = (uoff + 255) & ~(size_t)255;
            uoff = o + n;
            return o;
        };
        // first pass: size
        const size_t need = mb.segs.size() * (256 + sizeof(Uni) + 256 + 24 * 256 * 8 + 256 + 128 * 8);
        id<MTLBuffer> ubuf = pool_get(need, cb);
        u8 *ub = (u8 *)ubuf.contents;
        std::unordered_map<const void *, size_t> lut_map;
        bool wrote_color = false, wrote_depth = false;
        for (const MSeg &seg : mb.segs) {
            const DrawState &ds = *seg.ds;
            if (ds.skip || ds.shadow_mode || ds.gas) continue;
            id<MTLRenderPipelineState> ps = get_pipeline(ds);
            if (!ps) continue;
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
                    lo = ualloc(24 * 256 * 8);
                    float *dst = (float *)(ub + lo);
                    for (int t = 0; t < 24; t++)
                        for (int e = 0; e < 256; e++) {
                            dst[(t * 256 + e) * 2] = l.luts->val[t][e];
                            dst[(t * 256 + e) * 2 + 1] = l.luts->diff[t][e];
                        }
                    lut_map[lk] = lo;
                } else {
                    lo = it->second;
                }
                [re setFragmentBuffer:ubuf offset:lo atIndex:1];
            }
            if (ds.fog) {
                const void *fk = ds.fog_lut.get();
                auto it = lut_map.find(fk);
                size_t fo;
                if (it == lut_map.end()) {
                    fo = ualloc(128 * 8);
                    float *dst = (float *)(ub + fo);
                    for (int e = 0; e < 128; e++) {
                        dst[e * 2] = ds.fog_lut->val[e];
                        dst[e * 2 + 1] = ds.fog_lut->diff[e];
                    }
                    lut_map[fk] = fo;
                } else {
                    fo = it->second;
                }
                [re setFragmentBuffer:ubuf offset:fo atIndex:2];
            }
            [re setFragmentBuffer:ubuf offset:uo atIndex:0];
            // textures
            if (ds.need_shade) {
                for (int k = 0; k < 3; k++) {
                    const TexUnitState &t = ds.tex[k];
                    if (!ds.need_tex[k] || t.null_tex) continue;
                    id<MTLTexture> tex = get_texture(t.face[0]);
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
                    [re setFragmentSamplerState:get_sampler(t, (u32)tex.mipmapLevelCount) atIndex:k];
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
        const double t1 = g_stats ? now_ms() : 0;
        [cb commit];  // no wait: write-back happens when something reads the target
        if (wrote_color) col.gpu_newer = true;
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
    if (gpu_timing) {
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
                LOG("[gpu] GPU busy %.1f ms/s (scale %u, MSAA %ux), %llu passes (%llu with vertex kernels: %llu "
                    "dispatches, %llu vertices)\n",
                    b * 1000 / ((t - last) / 1000), S, MS, (unsigned long long)flushes.exchange(0),
                    (unsigned long long)vs_passes.exchange(0), (unsigned long long)vs_dispatches.exchange(0),
                    (unsigned long long)vs_verts.exchange(0));
            last = t;
        }
    }
    while (!pending.empty()) {
        const MTLCommandBufferStatus st = pending.front()->cb.status;
        if (st != MTLCommandBufferStatusCompleted && st != MTLCommandBufferStatusError) break;
        drain(0);
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
            if (definite || !r->synced || hash_bytes(pptr(r->addr), rt_bytes(*r)) != r->hash) r->synced = false;
        }
    std::lock_guard<std::mutex> lk(disp_mtx);
    for (auto it = displays.begin(); it != displays.end();) {
        const DisplayImage &d = it->second;
        if (overlaps(d.out_addr, d.out_size, paddr, size) && (definite || (!d.cb && hash_bytes(pptr(d.out_addr), d.out_size) != d.out_hash))) {
            if (g_stats)
                LOG("gpu metal: display image %08x dropped (%s write %08x+%x)\n", d.out_addr, definite ? "GPU" : "CPU",
                    paddr, size);
            if (!d.cb && disp_free.size() < 6) disp_free.push_back(d.buf);
            it = displays.erase(it);
        } else {
            ++it;
        }
    }
}

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
    if (src && (out_dims & 0xffff) != src->w) return nullptr;
    return src;
}

void encode_rot(id<MTLComputeCommandEncoder> ce, GRT &src, id<MTLBuffer> dst, u32 rows, u32 flags, u32 row0) {
    const u32 TW = src.w * S;
    [ce setComputePipelineState:k_rot];
    [ce setTexture:src.tex atIndex:0];
    [ce setBuffer:dst offset:0 atIndex:0];
    const u32 prm[4] = {TW, rows, flags & 1, row0 * S};
    [ce setBytes:prm length:16 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(rows, TW, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}

// new display image entry for out_paddr (lock held), replacing any image the output overlaps; the
// buffers of finished images go back to disp_free (those of unfinished ones when they finish)
DisplayImage &new_display(u32 out_paddr, u32 out_size, u32 out_h0, u32 w, u32 h) {
    for (auto it = displays.begin(); it != displays.end();) {
        if (overlaps(it->second.out_addr, it->second.out_size, out_paddr, out_size)) {
            if (!it->second.cb && disp_free.size() < 6) disp_free.push_back(it->second.buf);
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
    d.version = ++disp_version;
    d.screen = out_h0 == 400 ? 0 : out_h0 == 320 ? 1 : -1;
    if (d.screen >= 0) last_screen_addr[d.screen] = out_paddr;
    return d;
}
}  // namespace

bool metal_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size) {
    if (!active || eager_writeback || (flags & 2)) return false;
    u32 row0 = 0;
    GRT *src = transfer_source(in_paddr, in_dims, out_dims, row0);
    if (!src || !src->gpu_newer || overlaps(src->addr, rt_bytes(*src), out_paddr, out_size)) return false;
    for (auto &op : pending)
        if (op->rt_writeback && op->rt == src) return false;
    metal_guest_prewrite(out_paddr, out_size);
    auto op = std::make_unique<PendingXfer>();
    const u32 TW = src->w * S, rows = (out_dims >> 16) * S;
    @autoreleasepool {
        op->cb = new_cb();
        op->lin = take_buffer(lin_free, (size_t)src->w * src->h * 4);
        {
            std::lock_guard<std::mutex> lk(disp_mtx);
            op->disp = take_buffer(disp_free, (size_t)TW * rows * 4);
        }
        id<MTLComputeCommandEncoder> ce = [op->cb computeCommandEncoder];
        [ce setComputePipelineState:k_down];
        [ce setTexture:src->tex atIndex:0];
        [ce setBuffer:op->lin offset:0 atIndex:0];
        const u32 prm[4] = {src->w, src->h, S, 0};
        [ce setBytes:prm length:16 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(src->w, src->h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        encode_rot(ce, *src, op->disp, rows, flags, row0);
        [ce endEncoding];
        [op->cb commit];
    }
    src->gpu_newer = false;  // the pending transfer writes it back
    op->rt = src;
    op->in_paddr = in_paddr;
    op->out_paddr = out_paddr;
    op->in_dims = in_dims;
    op->out_dims = out_dims;
    op->flags = flags;
    op->out_size = out_size;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        DisplayImage &d = new_display(out_paddr, out_size, out_dims >> 16, rows, TW);
        d.cb = op->cb;
        d.buf = op->disp;
    }
    pending.push_back(std::move(op));
    return true;
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
    if (hash_bytes(pptr(src->addr), rt_bytes(*src)) != src->hash) {  // guest changed since
        if (g_stats) LOG("gpu metal: transfer %08x -> %08x: target changed in guest memory, no display image\n", in_paddr, out_paddr);
        return;
    }
    const u32 out_w0 = out_dims & 0xffff, out_h0 = out_dims >> 16;
    const u32 TW = src->w * S, rows = out_h0 * S;
    id<MTLBuffer> buf;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        buf = take_buffer(disp_free, (size_t)TW * rows * 4);
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cb = new_cb();
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        encode_rot(ce, *src, buf, rows, flags, row0);
        [ce endEncoding];
        [cb commit];
        wait_cb(cb, "display image", out_paddr);
    }
    const u32 hs = ((flags >> 24) & 3) ? 1 : 0, vs = ((flags >> 24) & 3) == 2 ? 1 : 0;
    const u32 out_size = (out_w0 >> hs) * (out_h0 >> vs) * pixel_bytes((flags >> 12) & 7);
    std::lock_guard<std::mutex> lk(disp_mtx);
    DisplayImage &d = new_display(out_paddr, out_size, out_h0, rows, TW);
    d.out_hash = hash_bytes(pptr(out_paddr), out_size);
    d.cb = nil;
    d.buf = buf;
}

bool get_display_image_for(u32 fb_paddr, const u32 **rgba, int *w, int *h) {
    thread_local std::vector<u32> out;
    std::lock_guard<std::mutex> lk(disp_mtx);
    auto it = displays.find(fb_paddr);
    if (it == displays.end()) return false;
    const DisplayImage &d = it->second;
    if (d.cb) [d.cb waitUntilCompleted];
    const u32 *px = (const u32 *)d.buf.contents;
    out.assign(px, px + (size_t)d.w * d.h);
    *rgba = out.data();
    *w = (int)d.w;
    *h = (int)d.h;
    return true;
}

u64 use_display_image(u32 fb_paddr, u64 have, const std::function<void(const u32 *, int, int)> &fn) {
    std::lock_guard<std::mutex> lk(disp_mtx);
    auto it = displays.find(fb_paddr);
    if (it == displays.end()) return 0;
    const DisplayImage &d = it->second;
    if (d.version == have) return have;
    if (d.cb) [d.cb waitUntilCompleted];
    fn((const u32 *)d.buf.contents, (int)d.w, (int)d.h);
    return d.version;
}

bool get_display_image(int screen, const u32 **rgba, int *w, int *h) {
    u32 addr;
    {
        std::lock_guard<std::mutex> lk(disp_mtx);
        if (screen < 0 || screen > 1 || !last_screen_addr[screen]) return false;
        addr = last_screen_addr[screen];
    }
    return get_display_image_for(addr, rgba, w, h);
}

}  // namespace gpu
