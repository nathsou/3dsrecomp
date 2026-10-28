// Internal declarations shared by the software PICA200 implementation.
#pragma once
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "gpu/gpu.h"
#include "rt.h"

namespace gpu {

// ---------------------------------------------------------------- register indices
namespace R {
constexpr u32 IRQ_REQ = 0x010, IRQ_CMP = 0x020, IRQ_MASK = 0x030, IRQ_AUTOSTOP = 0x034;
constexpr u32 CULL = 0x040, VP_W = 0x041, VP_H = 0x043;
constexpr u32 CLIP_EN = 0x047, CLIP_COEF = 0x048;
constexpr u32 DEPTH_SCALE = 0x04d, DEPTH_OFFSET = 0x04e, VS_OUT_TOTAL = 0x04f, VS_OUT_MAP = 0x050;
constexpr u32 SCISSOR_MODE = 0x065, SCISSOR_XY1 = 0x066, SCISSOR_XY2 = 0x067, VP_CORNER = 0x068;
constexpr u32 DEPTHMAP_EN = 0x06d;
constexpr u32 TEX_CONFIG = 0x080;
constexpr u32 TEX0 = 0x081, TEX1 = 0x091, TEX2 = 0x099;  // border, dim, param, lod, addr
constexpr u32 TEX0_CUBE = 0x086, TEX0_SHADOW = 0x08b, TEX0_TYPE = 0x08e, TEX1_TYPE = 0x096,
              TEX2_TYPE = 0x09e;
constexpr u32 PROCTEX = 0x0a8, PT_NOISE_U = 0x0a9, PT_NOISE_V = 0x0aa, PT_NOISE_FREQ = 0x0ab,
              PT_LUT = 0x0ac, PT_LUT_OFFSET = 0x0ad, PT_LUT_CONFIG = 0x0af, PT_LUT_DATA = 0x0b0;
constexpr u32 TEV0 = 0x0c0, TEV1 = 0x0c8, TEV2 = 0x0d0, TEV3 = 0x0d8, TEV4 = 0x0f0, TEV5 = 0x0f8;
constexpr u32 FOG_MODE = 0x0e0, FOG_COLOR = 0x0e1, FOG_LUT_OFFSET = 0x0e6, FOG_LUT_DATA = 0x0e8;
constexpr u32 TEV_BUF_COLOR = 0x0fd;
constexpr u32 OM_MODE = 0x100, BLEND = 0x101, LOGIC_OP = 0x102, BLEND_COLOR = 0x103,
              ALPHA_TEST = 0x104, STENCIL_TEST = 0x105, STENCIL_OP = 0x106,
              DEPTH_COLOR_MASK = 0x107;
constexpr u32 COLOR_READ = 0x112, COLOR_WRITE = 0x113, DEPTH_READ = 0x114, DEPTH_WRITE = 0x115,
              DEPTH_FMT = 0x116, COLOR_FMT = 0x117, DEPTH_ADDR = 0x11c, COLOR_ADDR = 0x11d,
              FB_DIM = 0x11e;
constexpr u32 SHADOW = 0x130;
constexpr u32 LIGHT0 = 0x140, LIGHT_AMBIENT = 0x1c0, LIGHT_NUM = 0x1c2, LIGHT_CONFIG0 = 0x1c3,
              LIGHT_CONFIG1 = 0x1c4, LUT_INDEX = 0x1c5, LIGHTING_DISABLE = 0x1c6,
              LUT_DATA = 0x1c8, LUT_ABS = 0x1d0, LUT_SELECT = 0x1d1, LUT_SCALE = 0x1d2,
              LIGHT_PERMUTATION = 0x1d9;
constexpr u32 ATTR_BASE = 0x200, ATTR_FMT_LO = 0x201, ATTR_FMT_HI = 0x202, ATTR_LOADER = 0x203,
              INDEX_ARRAY = 0x227, NUM_VERTICES = 0x228, GS_MISC = 0x229, VERTEX_OFFSET = 0x22a,
              DRAW = 0x22e, DRAW_INDEXED = 0x22f, FIXATTR_INDEX = 0x232, FIXATTR_DATA = 0x233,
              CMDBUF_SIZE0 = 0x238, CMDBUF_ADDR0 = 0x23a, CMDBUF_JUMP0 = 0x23c,
              CMDBUF_JUMP1 = 0x23d, VS_NUM_ATTR = 0x242, GS_EXCLUSIVE = 0x244,
              VS_OUTMAP_TOTAL1 = 0x24a, VS_OUTMAP_TOTAL2 = 0x251, GS_CONFIG = 0x252,
              GS_VAR_MAIN_NUM = 0x254, TOPOLOGY = 0x25e, RESTART_PRIM = 0x25f;
constexpr u32 GS = 0x280, VS = 0x2b0;
// offsets inside a shader unit block
constexpr u32 SH_BOOL = 0x0, SH_INT = 0x1, SH_INPUT_CFG = 0x9, SH_ENTRY = 0xa, SH_PERM_LO = 0xb,
              SH_PERM_HI = 0xc, SH_OUTMASK = 0xd, SH_FLOAT_IDX = 0x10, SH_FLOAT_DATA = 0x11,
              SH_CODE_IDX = 0x1b, SH_CODE_DATA = 0x1c, SH_SWZ_IDX = 0x25, SH_SWZ_DATA = 0x26;
}  // namespace R

inline u32 bits(u32 v, u32 lo, u32 n) { return (v >> lo) & ((1u << n) - 1); }
inline s32 sbits(u32 v, u32 lo, u32 n) { return (s32)(v << (32 - lo - n)) >> (32 - n); }

// Generic small float format (M mantissa bits, E exponent bits, 1 sign bit) to float32.
float float_from_raw(u32 hex, u32 M, u32 E);
inline float f24(u32 v) { return float_from_raw(v, 16, 7); }
inline float f16(u32 v) { return float_from_raw(v, 10, 5); }
inline float f20(u32 v) { return float_from_raw(v, 12, 7); }

struct alignas(16) V4 {
    float v[4];
};

// ---------------------------------------------------------------- shader state
struct ShaderProgram;

struct ShaderSetup {
    u32 code[4096];
    u32 swizzle[4096];
    V4 f[96];
    bool b[16];
    u8 i[4][4];
    u32 queue[4];
    u32 queue_n;
    u32 code_size, swizzle_size;  // highest written word + 1
    bool dirty;                   // code or swizzle changed since last decode
    const ShaderProgram *prog;
};

// Emitter used by the geometry shader EMIT/SETEMIT instructions.
struct Emitter {
    u32 vertex_id;
    bool prim_emit, winding;
    u32 output_mask;
    V4 buf[3][16];
    void (*emit_prim)(Emitter *);
};

// Register file of one shader unit. Unified indexing used by the decoded program:
//   0..15 inputs, 16..31 temporaries, 32..127 float uniforms (copied in), 128..143 outputs bank 0,
//   144..159 outputs bank 1, 160 = zeros, 161 = ones.
constexpr u32 REG_IN = 0, REG_TMP = 16, REG_UNI = 32, REG_OUT = 128, REG_ZERO = 160,
              REG_ONE = 161, REG_COUNT = 162;
struct ShaderUnit {
    V4 r[REG_COUNT];
    s32 addr[3];
    bool cond[2];
    u32 out_bank;
    Emitter *emitter;
};

const ShaderProgram *shader_get_program(ShaderSetup &s);
void shader_unit_init(ShaderUnit &u, const ShaderSetup &s);  // zero regs, copy uniforms
void shader_copy_uniforms(ShaderUnit &u, const ShaderSetup &s);
void shader_run(const ShaderProgram *p, ShaderUnit &u, const ShaderSetup &s, u32 entry);
void shader_run(const ShaderProgram *p, ShaderUnit &u, const bool *b, const u8 (*i)[4], u32 entry);
// 4-wide SoA execution of four vertices (vertex shaders only). in/out are indexed
// [lane][register]; only registers in inmask / outmask are transferred. Returns false if the
// lanes diverged on a condition-code branch (or the program is unsupported): use shader_run().
struct ShaderUnit4;
ShaderUnit4 *shader_unit4_new();
void shader_unit4_reset(ShaderUnit4 *u);
bool shader_program_simd(const ShaderProgram *p);
u64 shader_program_hash(const ShaderProgram *p);  // of the code and swizzle words it was decoded from
// the program as a Metal compute kernel named `name` ("" if it cannot be translated)
std::string shader_msl(const ShaderProgram *p, u32 entry, const char *name);
// the same for one set of bool/int uniforms (flow resolved statically), or "" if it cannot be done exactly
std::string shader_msl_spec(const ShaderProgram *p, u32 entry, const char *name, u32 b, const u8 (*iu)[4]);
// the bool uniforms (bit mask) and int uniforms (*loops, bit mask) the program's flow control reads
u32 shader_static_uniforms(const ShaderProgram *p, u32 *loops);
bool shader_run4(const ShaderProgram *p, ShaderUnit4 &u, const V4 *uniforms, const bool *b,
                 const u8 (*i)[4], u32 entry, const V4 (*in)[16], u32 inmask, V4 (*out)[16],
                 u32 outmask);
// Decode a program into its compact form (exposed for tests).
ShaderProgram *shader_decode(const u32 *code, u32 code_size, const u32 *swizzle);
void shader_free(const ShaderProgram *p);

// ---------------------------------------------------------------- vertices
// Output vertex: semantic slots as in GPUREG_SH_OUTATTR (0..23).
struct alignas(16) OutVertex {
    float pos[4];    // 0
    float quat[4];   // 4
    float color[4];  // 8
    float tc0[2];    // 12
    float tc1[2];    // 14
    float tc0w;      // 16
    float pad0;      // 17
    float view[3];   // 18
    float pad1;      // 21
    float tc2[2];    // 22
};
static_assert(sizeof(OutVertex) == 24 * 4);

// ---------------------------------------------------------------- textures
struct TexImage {
    u32 w = 0, h = 0;
    std::vector<u32> data;  // RGBA8 packed r | g<<8 | b<<16 | a<<24; row 0 = t=0 (bottom)
};
struct TexEntry {
    u32 addr = 0, fmt = 0, w = 0, h = 0, levels = 0, size = 0;
    u64 hash = 0;
    u64 epoch = 0;
    u64 last_use = 0;
    bool stale = true;
    std::vector<TexImage> lv;
};

// decode a whole texture level from guest memory (exposed for tests)
void texture_decode(const u8 *src, u32 w, u32 h, u32 fmt, u32 *out_rgba_rows_bottom_up);
u32 texture_level_bytes(u32 w, u32 h, u32 fmt);
u32 decode_etc1_texel(u64 block, u32 x, u32 y);  // x,y within 4x4, y=0 top row in memory
const TexEntry *texcache_get(u32 addr, u32 w, u32 h, u32 fmt, u32 levels);
void texcache_invalidate(u32 paddr, u32 size);
void texcache_new_epoch();
void texcache_trim();

// ---------------------------------------------------------------- per-draw state
struct TexUnitState {
    bool enabled = false;
    bool null_tex = false;  // address 0: returns (0,0,0,255)
    u8 type = 0;            // 0 2D, 1 cube, 2 shadow2D, 3 projection, 4 shadow cube, 5 disabled
    u8 wrap_s = 0, wrap_t = 0;
    bool mag_linear = false, min_linear = false, mip_linear = false;
    bool need_lod = false;
    u8 coord = 0;
    s32 min_level = 0, max_level = 0;
    float bias = 0;
    u32 border = 0;
    u32 w = 0, h = 0;
    const TexEntry *face[6] = {};
    u32 shadow_bias = 0;
    bool shadow_ortho = false;
    bool fast_wrap = false;  // both wrap modes are clamp-to-edge or repeat, power-of-two size
    bool simple2d = false;   // 2D, no LOD needed, fast wrap: sample_simple2d() applies
    float mag_rho2 = 0;      // rho^2 below this certainly gives lod <= 0 (magnification)
};
u32 sample_simple2d(const TexUnitState &tu, float u, float v);

struct TevStageState {
    u8 csrc[3], asrc[3], cmod[3], amod[3];
    u8 cop, aop, cmul, amul;
    u32 konst;
    bool upd_rgb, upd_a;
    // precomputed for the vectorized combiner (tev_prepare)
    alignas(16) u8 idx[16];  // byte indices into the 64-byte source table: operand k lanes 4k..4k+3
    alignas(16) u8 inv[16];  // 0xff where the modifier inverts (1 - x)
    // 4-pixel variant (tev_combine4): per operand, shuffles of a 4-pixel RGBA vector
    alignas(16) u8 q_idxc[3][16];  // color lanes from the color source (alpha lanes -> 0)
    alignas(16) u8 q_idxa[3][16];  // alpha lanes from the alpha source (color lanes -> 0)
    alignas(16) u8 q_idx[3][16];   // both, when color and alpha sources are the same
    alignas(16) u8 q_inv[3][16];
    bool q_same[3];
};
struct DrawState;
void tev_prepare(DrawState &ds);  // fills idx/inv and the number of active stages
inline u32 div255(u32 x) { return (x + 1 + (x >> 8)) >> 8; }  // exact for x <= 65025

struct LightLuts {
    float val[24][256];
    float diff[24][256];
};
struct FogLut {
    float val[128], diff[128];
};
struct ProcTexLuts {
    float noise_v[128], noise_d[128];
    float cmap_v[128], cmap_d[128];
    float amap_v[128], amap_d[128];
    u32 color[256];
    s32 cdiff[256][4];
};

struct LutSel {
    bool enabled = false, abs = false;
    u8 input = 0;
    float scale = 1;
};
struct LightState {
    float spec0[3], spec1[3], diffuse[3], ambient[3], pos[3], spot[3];
    bool directional, two_sided, geo0, geo1, dist_atten, spot_atten, shadow;
    float da_scale, da_bias;
    u32 num;
};
struct LightingState {
    u32 num_lights = 0;
    LightState lights[8];
    LutSel d0, d1, sp, fr, rr, rg, rb;
    float global_ambient[3];
    u8 bump_mode = 0, bump_sel = 0, shadow_sel = 0, config = 0;
    bool bump_renorm = false, shadow_enable = false, shadow_invert = false, shadow_primary = false,
         shadow_secondary = false, shadow_alpha = false, primary_alpha = false,
         secondary_alpha = false, clamp_highlights = false;
    std::shared_ptr<const LightLuts> luts;
};

struct ProcTexState {
    bool enabled = false;
    u8 coord = 0;
    u8 u_clamp, v_clamp, color_comb, alpha_comb, u_shift, v_shift, filter;
    bool separate_alpha, noise;
    float noise_amp_u, noise_amp_v, noise_phase_u, noise_phase_v, noise_freq_u, noise_freq_v;
    u32 lut_offset, lut_width;
    std::shared_ptr<const ProcTexLuts> luts;
};

// attribute groups interpolated per pixel
enum : u32 {
    AG_COLOR = 1,  // color rgba
    AG_TC01 = 2,   // tc0 uv, tc1 uv
    AG_TC2W = 4,   // tc2 uv, tc0w
    AG_QUAT = 8,
    AG_VIEW = 16,
};
constexpr u32 NUM_ATTR = 20;

struct RTConfig {
    u32 color_addr = 0, depth_addr = 0;
    u32 color_fmt = 0, depth_fmt = 0;
    u32 w = 0, h = 0;
    bool operator==(const RTConfig &o) const {
        return color_addr == o.color_addr && depth_addr == o.depth_addr &&
               color_fmt == o.color_fmt && depth_fmt == o.depth_fmt && w == o.w && h == o.h;
    }
};

struct DrawState {
    // rasterization
    float vp_hw, vp_hh, vp_x, vp_y;
    float depth_scale, depth_offset;
    bool wbuffer;
    u32 cull;
    u32 scissor_mode;
    s32 sc_x1, sc_y1, sc_x2, sc_y2;  // inclusive-exclusive pixel rect (x2,y2 exclusive)
    bool clip_enable;
    float clip[4];
    u32 attr_groups;
    RTConfig rt;
    bool skip;  // nothing can be drawn (alpha test never, gas mode, ...)

    // texturing / combiners
    TexUnitState tex[3];
    ProcTexState proctex;
    bool need_tex[4];
    TevStageState tev[6];
    u32 tev_n;  // stages after this are pass-through
    // per-pixel stage program after constant folding / pass-through removal (tev_prepare)
    u8 tev_list[6];
    u32 tev_count;
    u32 tev_init_out, tev_init_buf, tev_init_next;
    u32 tev_buf_color;
    bool lighting;
    LightingState light;
    bool fog;
    bool fog_flip;
    u32 fog_color;
    std::shared_ptr<const FogLut> fog_lut;
    bool gas;

    // output merger
    bool shadow_mode;
    float shadow_const, shadow_linear;
    bool blend;
    u8 eq_rgb, eq_a, sf_rgb, df_rgb, sf_a, df_a;
    // blend factor byte selectors into {src rgba, dst rgba, const rgba, 0, 255, saturate, 0}
    u8 bf_idx[8], bf_inv[8];  // [0..3] source factor per channel, [4..7] destination factor
    u8 bf_w[8], bf_sh[8];     // same selection as (word, bit shift), see blend()
    alignas(8) u8 bf_q[8];    // same selection as byte indices into {src,dst,const,special}
    alignas(8) u8 bf_qinv[8];
    u32 blend_color;
    u8 logic_op;
    bool alpha_test;
    u8 alpha_func, alpha_ref;
    bool stencil;
    u8 st_func, st_ref, st_mask_in, st_mask_w, st_fail, st_zfail, st_zpass;
    bool depth_test;
    u8 depth_func;
    bool depth_write;
    u32 color_mask;  // per byte of packed RGBA: 0xff per enabled channel
    bool color_write;
    bool early_z;      // depth/stencil test can run before shading
    bool need_shade;   // fragment color is needed at all
    u32 depth_bits;
};

// ---------------------------------------------------------------- global GPU state
struct PrimAssembler {
    u32 topology = 0;
    u32 index = 0;
    bool strip_ready = false, winding = false;
    OutVertex buf[2];
};

struct GpuState {
    u32 regs[0x300];
    ShaderSetup vs, gs;
    V4 default_attr[16];
    // immediate mode
    u32 imm_queue[3];
    u32 imm_n;
    V4 imm_vertex[16];
    u32 imm_attr;
    bool imm_reset_gs;
    // LUTs
    u32 light_lut[24][256];
    u32 fog_lut[128];
    u32 pt_noise[128], pt_cmap[128], pt_amap[128], pt_color[256], pt_cdiff[256];
    bool light_lut_dirty, fog_lut_dirty, pt_dirty;
    std::shared_ptr<const LightLuts> light_luts;
    std::shared_ptr<const FogLut> fog_luts;
    std::shared_ptr<const ProcTexLuts> pt_luts;
    // geometry shader persistent unit
    ShaderUnit gs_unit;
    Emitter gs_emitter;
    PrimAssembler pa;
    // draw state cache
    u64 regs_version;
    u64 ds_version;
    DrawState *ds;
};
extern GpuState g;

// ---------------------------------------------------------------- module interfaces
void ensure_init();
void cmd_run(u32 paddr, u32 size);
void vtx_draw(bool indexed);
void vtx_immediate_vertex();
void vtx_reset_immediate();
void pa_reconfigure(u32 topology);
void pa_reset();

DrawState *draw_state();  // current draw state (builds it and handles batch breaks)
void raster_triangle(const DrawState *ds, const OutVertex &v0, const OutVertex &v1,
                     const OutVertex &v2);
// screen-space vertex (after perspective divide); attributes premultiplied by 1/w
struct SV {
    float sx, sy, sz, iw;
    float attr[NUM_ATTR];
};
// a set-up triangle ready for tile rasterization
struct Tri {
    const DrawState *ds;
    s32 x0, y0, x1, y1;   // pixel bbox [x0,x1) x [y0,y1)
    s64 e0[3];            // edge values (incl. fill-rule bias) at pixel (x0, y0)
    s64 dex[3], dey[3];   // per pixel steps
    double z0, zx, zy;    // z/w plane
    float q0, qx, qy;     // 1/w plane
    float a0[NUM_ATTR], ax[NUM_ATTR], ay[NUM_ATTR];  // attr/w planes
};
// triangle setup output of one worker task (software rasterizer)
struct TriSink {
    std::vector<Tri> tris;
    void clear() { tris.clear(); }
};

// ---------------------------------------------------------------- Metal batches
// Vertices reach the Metal backend in the OutVertex layout, either shaded by the CPU or by a compute
// kernel translated from the vertex program; clipping, culling and snapping happen in Metal.
struct alignas(16) VsAttrG {
    s32 offset;  // of vertex 0 in MBatch::vdata (may be negative)
    u32 stride, fmt;
    u32 elems;   // 0: (0, 0, 0, 1), 5: the fixed attribute value
};
struct alignas(16) VsParamsG {  // VsParams in metal.mm
    float f[96][4];
    float defaults[16][4];
    VsAttrG attr[16];
    s32 reg_attr[16];  // input register -> attribute (-1: zero)
    s32 slot_src[24];  // OutVertex slot -> output register * 4 + component (-1: 1.0)
    u32 iu[4][4];
    u32 b, nverts, out_base, first;  // first: the first thread of this draw in its kernel dispatch
};
struct GpuVsJob {
    const void *kernel;  // metal_vs_kernel()
    u32 vids_off;        // first vertex id in MBatch::vids
    VsParamsG p;
};
constexpr u32 MIDX_CPU = 0x80000000u;  // index flag: MBatch::cverts, else the compute kernels' output
struct MSeg {
    const DrawState *ds;
    u32 first, count;  // in MBatch::idx (three per triangle)
};
struct MBatch {
    std::vector<u8> vdata;  // copied vertex arrays
    std::vector<u32> vids;
    std::vector<GpuVsJob> jobs;
    u32 gpu_verts = 0;
    std::vector<OutVertex> cverts;
    std::vector<u32> idx;
    std::vector<MSeg> segs;
    bool empty() const { return segs.empty(); }
    void clear() {
        vdata.clear();
        vids.clear();
        jobs.clear();
        gpu_verts = 0;
        cverts.clear();
        idx.clear();
        segs.clear();
    }
    // the triangles just appended to idx (from `first`) are drawn with ds
    void add_seg(const DrawState *ds, u32 first) {
        const u32 count = (u32)idx.size() - first;
        if (!count) return;
        if (!segs.empty() && segs.back().ds == ds && segs.back().first + segs.back().count == first)
            segs.back().count += count;
        else
            segs.push_back({ds, first, count});
    }
};
MBatch &raster_mbatch();  // the current batch's Metal data
// Redirect this thread's triangle setup output into `sink` (null: straight into the batch).
void raster_set_sink(TriSink *sink);
// Add triangles (in order) to the current batch. Main thread only.
void raster_append(const TriSink &tris);

// ---------------------------------------------------------------- Metal backend (metal.mm)
bool metal_active();  // Metal device available and GPU_BACKEND != sw
bool metal_init();
// draw the batch's triangles into the render target and write the color back to guest memory
void metal_flush(const RTConfig &rt, const MBatch &b);
// the compute kernel for a vertex program (null while it compiles: shade on the CPU meanwhile)
const void *metal_vs_kernel(const ShaderProgram *p, u32 entry, const ShaderSetup &s);
// guest memory in [paddr, paddr+size) is about to be read by a non-render GPU operation (texture
// decode, transfer source): write back render-target data held only on the GPU
void metal_guest_read(u32 paddr, u32 size);
// guest memory was overwritten by a non-render operation (fill, transfer, copy, CPU)
// (definite: the bytes were certainly rewritten; else only if their hash changed)
void metal_guest_written(u32 paddr, u32 size, bool definite = true);
// guest memory in [paddr, paddr+size) is about to be overwritten by a fill, transfer or copy
void metal_guest_prewrite(u32 paddr, u32 size);
// write every GPU-only render-target change back to guest memory
void metal_writeback_all();
// finish deferred work whose GPU part completed (called at the start of GPU operations)
void metal_poll();
// try to run a display transfer from a render target asynchronously (true: taken care of)
bool metal_defer_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags, u32 out_size);
// the display transfer's pixel conversion (src/dst: first input / output byte)
void transfer_convert(const u8 *src, u8 *dst, u32 in_dims, u32 out_dims, u32 flags);
// a display transfer completed: remember high-resolution sources of framebuffers
void metal_display_transfer(u32 in_paddr, u32 out_paddr, u32 in_dims, u32 out_dims, u32 flags);
// texture filter (set_texture_filter): textures in guest memory holding GPU output (render-to-texture) are
// not upscaled. Both do nothing while no filter is on. metal_gpu_made: [paddr, paddr+size) holds GPU output
// (call after metal_guest_read); metal_note_gpu_output: a transfer or copy of GPU output wrote it.
bool metal_gpu_made(u32 paddr, u32 size);
void metal_note_gpu_output(u32 paddr, u32 size);
// deferred draws (vertex.cpp): draws are recorded and processed in parallel at batch flush
bool vtx_jobs_pending();
void vtx_finish_jobs();
// per-vertex precomputation for the fast (unclipped) triangle path
void raster_prepare_vertex(const DrawState &ds, const OutVertex &v, SV &sv, u32 &outcode);
void raster_triangle_sv(const DrawState *ds, const SV &a, const SV &b, const SV &c);
void raster_flush();
void raster_init();
bool raster_blend_selftest();  // vectorized vs scalar blending (unit tests)
void raster_begin_draw(const DrawState *ds);

void build_draw_state(DrawState &ds);

// fragment pipeline helpers (fragment.cpp)
struct FragIn {
    float attr[NUM_ATTR];
    float lod_inputs[3][4];  // per texcoord set: du/dx, dv/dx, du/dy, dv/dy (unscaled)
};
u32 sample_texture(const TexUnitState &tu, float u, float v, float w, float lod);
u32 sample_proctex(const ProcTexState &pt, float u, float v);
void compute_lighting(const LightingState &ls, const float quat[4], const float view[3],
                      const u32 tex[4], u32 &primary, u32 &secondary);
// four fragments: quat[component][lane], view[component][lane], tex[unit][lane]
void compute_lighting4(const LightingState &ls, const float (*quat)[4], const float (*view)[4],
                       const u32 (*tex)[4], u32 primary[4], u32 secondary[4]);
u32 tev_combine(const DrawState &ds, u32 primary, u32 frag_pri, u32 frag_sec, const u32 tex[4]);
// four fragments at once; tex is indexed [unit][pixel]; bit-identical to tev_combine
void tev_combine4(const DrawState &ds, const u32 primary[4], const u32 frag_pri[4],
                  const u32 frag_sec[4], const u32 tex[4][4], u32 out[4]);

// ---------------------------------------------------------------- thread pool
void pool_init();
void parallel_for(u32 n, const std::function<void(u32 index, u32 thread)> &fn);
u32 pool_threads();  // number of threads incl. caller

// ---------------------------------------------------------------- misc helpers
inline u8 *pptr(u32 paddr) { return mem::paddr_ptr(paddr); }
// 8x8 morton interleave
inline u32 morton8(u32 x, u32 y) {
    static constexpr u8 xl[8] = {0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15};
    static constexpr u8 yl[8] = {0x00, 0x02, 0x08, 0x0a, 0x20, 0x22, 0x28, 0x2a};
    return xl[x & 7] + yl[y & 7];
}
// byte offset of pixel (x, y) (y in memory order) in a tiled image of the given width
inline u32 tiled_offset(u32 x, u32 y, u32 width, u32 bpp) {
    return ((y & ~7u) * width + (x & ~7u) * 8 + morton8(x, y)) * bpp;
}
inline u32 rgba(u32 r, u32 g, u32 b, u32 a) { return r | (g << 8) | (b << 16) | (a << 24); }

// pixel format codecs shared by framebuffer and transfer engine (format = GPU PixelFormat)
u32 decode_pixel(u32 fmt, const u8 *p);       // display formats 0..4 (RGBA8 RGB8 565 5551 4444)
void encode_pixel(u32 fmt, u32 c, u8 *p);
inline u32 pixel_bytes(u32 fmt) { return fmt == 0 ? 4 : fmt == 1 ? 3 : 2; }
// color buffer formats (FramebufferRegs::ColorFormat) map: 0 RGBA8, 1 RGB8, 2 RGB5A1, 3 RGB565,
// 4 RGBA4 -> display format codes
inline u32 cbuf_to_pixfmt(u32 cf) {
    static constexpr u8 m[8] = {0, 1, 3, 2, 4, 0, 0, 0};
    return m[cf & 7];
}

u64 hash_bytes(const u8 *p, size_t n);

// timing statistics (GPU_STATS=1)
extern bool g_stats, g_trace;
extern double g_t_shade, g_t_setup, g_t_flush;
// texture decoding (texture.cpp), for RECOMP_STATS (the Metal backend logs and resets them)
extern double g_tex_decode_ms;
extern u32 g_tex_decodes;
double now_ms();

}  // namespace gpu
