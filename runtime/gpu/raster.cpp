// Draw state construction, clipping, triangle setup, screen-tile binning and the parallel tile
// rasterizer with the full per-fragment pipeline (textures, lighting, TEV, fog, alpha / stencil /
// depth tests, blending, logic ops) operating on PICA tiled framebuffers in guest memory.
#include <algorithm>

#include "gpu/pica.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace gpu {

namespace {
constexpr u32 TILE_SHIFT = 5, TILE = 1u << TILE_SHIFT;

// where triangle setup output goes on this thread (null: straight into the batch, binned)
thread_local TriSink *tl_sink = nullptr;

struct Batch {
    RTConfig rt;
    bool has_rt = false;
    std::vector<Tri> tris;
    MBatch mb;  // Metal backend
    std::vector<std::vector<u32>> bins;
    u32 tiles_x = 0, tiles_y = 0;
    std::vector<std::unique_ptr<DrawState>> states;
    std::vector<u32> active_tiles;
} batch;

struct TileBuf {
    u32 color[TILE * TILE];
    u32 depth[TILE * TILE];
    u8 stencil[TILE * TILE];
};
std::vector<std::unique_ptr<TileBuf>> tilebufs;

// optional statistics (GPU_STATS=1)
bool stats_on = false;
std::atomic<u64> st_frags{0}, st_shaded{0};
u64 st_tris = 0, st_flushes = 0, st_draws = 0;

u32 color_bytes(u32 cf) { return cf == 0 ? 4 : cf == 1 ? 3 : 2; }
u32 depth_bytes(u32 df) { return df == 0 ? 2 : df == 2 ? 3 : 4; }

bool batch_nonempty() { return !batch.tris.empty() || !batch.mb.empty() || vtx_jobs_pending(); }

// add the last triangle of batch.tris to the tile bins (drops it if it covers no tile)
void bin_last_tri() {
    const Tri &t = batch.tris.back();
    const u32 idx = (u32)batch.tris.size() - 1;
    bool any = false;
    const s32 tx0 = t.x0 >> TILE_SHIFT, tx1 = (t.x1 - 1) >> TILE_SHIFT;
    const s32 ty0 = t.y0 >> TILE_SHIFT, ty1 = (t.y1 - 1) >> TILE_SHIFT;
    const bool single = tx0 == tx1 && ty0 == ty1;
    for (s32 ty = ty0; ty <= ty1; ty++) {
        for (s32 tx = tx0; tx <= tx1; tx++) {
            if (!single) {
                // reject tiles fully outside an edge (test the tile corner maximizing each edge)
                const s64 cx0 = std::max<s64>(tx * TILE, t.x0) - t.x0;
                const s64 cy0 = std::max<s64>(ty * TILE, t.y0) - t.y0;
                const s64 cx1 = std::min<s64>(tx * TILE + TILE, t.x1) - 1 - t.x0;
                const s64 cy1 = std::min<s64>(ty * TILE + TILE, t.y1) - 1 - t.y0;
                bool out = false;
                for (int i = 0; i < 3 && !out; i++) {
                    s64 e = t.e0[i] + t.dex[i] * (t.dex[i] > 0 ? cx1 : cx0) +
                            t.dey[i] * (t.dey[i] > 0 ? cy1 : cy0);
                    if (e < 0) out = true;
                }
                if (out) continue;
            }
            batch.bins[ty * batch.tiles_x + tx].push_back(idx);
            any = true;
        }
    }
    if (!any) batch.tris.pop_back();
}
}  // namespace

void raster_set_sink(TriSink *sink) { tl_sink = sink; }

void raster_append(const TriSink &s) {
    for (const Tri &t : s.tris) {
        batch.tris.push_back(t);
        bin_last_tri();
    }
}

static bool use_metal = false;

MBatch &raster_mbatch() { return batch.mb; }

void raster_init() {
    use_metal = metal_init();
    stats_on = getenv("GPU_STATS") != nullptr;
    tilebufs.clear();
    for (u32 i = 0; i < pool_threads(); i++) tilebufs.push_back(std::make_unique<TileBuf>());
}

// ---------------------------------------------------------------- LUT snapshots
static void update_luts() {
    if (g.light_lut_dirty || !g.light_luts) {
        auto L = std::make_shared<LightLuts>();
        for (u32 t = 0; t < 24; t++)
            for (u32 i = 0; i < 256; i++) {
                u32 raw = g.light_lut[t][i];
                L->val[t][i] = (raw & 0xfff) / 4095.0f;
                float d = ((raw >> 12) & 0x7ff) / 2047.0f;
                L->diff[t][i] = (raw >> 23) & 1 ? -d : d;
            }
        g.light_luts = L;
        g.light_lut_dirty = false;
    }
    if (g.fog_lut_dirty || !g.fog_luts) {
        auto F = std::make_shared<FogLut>();
        for (u32 i = 0; i < 128; i++) {
            u32 raw = g.fog_lut[i];
            F->val[i] = ((raw >> 13) & 0x7ff) / 2047.0f;
            F->diff[i] = sbits(raw, 0, 13) / 2047.0f;
        }
        g.fog_luts = F;
        g.fog_lut_dirty = false;
    }
    if (g.pt_dirty || !g.pt_luts) {
        auto P = std::make_shared<ProcTexLuts>();
        for (u32 i = 0; i < 128; i++) {
            P->noise_v[i] = (g.pt_noise[i] & 0xfff) / 4095.0f;
            P->noise_d[i] = sbits(g.pt_noise[i], 12, 12) / 4095.0f;
            P->cmap_v[i] = (g.pt_cmap[i] & 0xfff) / 4095.0f;
            P->cmap_d[i] = sbits(g.pt_cmap[i], 12, 12) / 4095.0f;
            P->amap_v[i] = (g.pt_amap[i] & 0xfff) / 4095.0f;
            P->amap_d[i] = sbits(g.pt_amap[i], 12, 12) / 4095.0f;
        }
        for (u32 i = 0; i < 256; i++) {
            P->color[i] = g.pt_color[i];
            for (u32 k = 0; k < 4; k++) P->cdiff[i][k] = sbits(g.pt_cdiff[i], 8 * k, 8) * 2;
        }
        g.pt_luts = P;
        g.pt_dirty = false;
    }
}

// ---------------------------------------------------------------- draw state
static bool sampler_supported(u32 config, u32 sampler) {
    switch (sampler) {
    case 0: return config != 1;                              // D0
    case 1: return config != 0 && config != 1 && config != 5;  // D1
    case 8: return config != 2 && config != 3;               // SP
    case 3: return config != 0 && config != 2 && config != 4;  // FR
    case 6: return config != 3;                              // RR
    case 5:
    case 4: return config == 4 || config == 5 || config == 8;  // RG, RB
    default: return false;
    }
}

static float lut_scale(u32 v) {
    switch (v) {
    case 0: return 1;
    case 1: return 2;
    case 2: return 4;
    case 3: return 8;
    case 6: return 0.25f;
    case 7: return 0.5f;
    default: return 0;
    }
}

static u32 mip_levels(u32 w, u32 h, u32 max_level) {
    u32 levels = 1;
    while (w > 8 && h > 8) {
        levels++;
        w >>= 1;
        h >>= 1;
    }
    return std::min(levels, max_level + 1);
}

static void rt_from_regs(RTConfig &rt) {
    const u32 *regs = g.regs;
    rt.color_addr = (regs[R::COLOR_ADDR] & 0x0fffffff) * 8;
    rt.depth_addr = (regs[R::DEPTH_ADDR] & 0x0fffffff) * 8;
    rt.color_fmt = (regs[R::COLOR_FMT] >> 16) & 7;
    rt.depth_fmt = regs[R::DEPTH_FMT] & 3;
    rt.w = regs[R::FB_DIM] & 0x7ff;
    rt.h = ((regs[R::FB_DIM] >> 12) & 0x3ff) + 1;
    if ((regs[R::OM_MODE] & 3) == 3) rt.color_fmt |= 0x80;  // shadow rendering: raw color access
}

static bool range_overlap(u32 a, u32 asz, u32 b, u32 bsz) { return a < b + bsz && b < a + asz; }

static bool overlaps_rt(u32 addr, u32 size) {
    if (!batch.has_rt || !batch_nonempty()) return false;
    const RTConfig &rt = batch.rt;
    u32 cs = rt.w * rt.h * color_bytes(rt.color_fmt & 7);
    u32 dsz = rt.w * rt.h * depth_bytes(rt.depth_fmt);
    return (rt.color_addr && range_overlap(addr, size, rt.color_addr, cs)) ||
           (rt.depth_addr && range_overlap(addr, size, rt.depth_addr, dsz));
}

static void setup_texunit(DrawState &ds, u32 i) {
    const u32 *regs = g.regs;
    static const u32 base[3] = {R::TEX0, R::TEX1, R::TEX2};
    static const u32 typereg[3] = {R::TEX0_TYPE, R::TEX1_TYPE, R::TEX2_TYPE};
    TexUnitState &tu = ds.tex[i];
    const u32 b = base[i];
    tu = TexUnitState{};
    tu.enabled = (regs[R::TEX_CONFIG] >> i) & 1;
    if (!tu.enabled) return;
    tu.border = regs[b];
    tu.h = regs[b + 1] & 0x7ff;
    tu.w = (regs[b + 1] >> 16) & 0x7ff;
    const u32 p = regs[b + 2];
    tu.mag_linear = (p >> 1) & 1;
    tu.min_linear = (p >> 2) & 1;
    tu.wrap_t = (p >> 8) & 7;
    tu.wrap_s = (p >> 12) & 7;
    tu.mip_linear = (p >> 24) & 1;
    tu.type = i == 0 ? (p >> 28) & 7 : 0;
    const u32 lod = regs[b + 3];
    tu.bias = sbits(lod, 0, 13) / 256.0f;
    tu.max_level = (lod >> 16) & 15;
    tu.min_level = (lod >> 24) & 15;
    const u32 addr_reg = regs[b + 4] & 0x0fffffff;
    const u32 addr = addr_reg * 8;
    const u32 fmt = regs[typereg[i]] & 0xf;
    tu.coord = i == 2 && ((regs[R::TEX_CONFIG] >> 13) & 1) ? 1 : (u8)i;
    if (tu.type == 5) {
        tu.enabled = false;
        return;
    }
    if (addr_reg == 0) {
        tu.null_tex = true;
        return;
    }
    tu.shadow_ortho = regs[R::TEX0_SHADOW] & 1;
    tu.shadow_bias = (regs[R::TEX0_SHADOW] >> 1) & 0x7fffff;
    const bool shadow = tu.type == 2 || tu.type == 4;
    const u32 levels = shadow ? 1 : mip_levels(tu.w, tu.h, tu.max_level);
    tu.need_lod = (levels > 1 || tu.min_linear != tu.mag_linear) && tu.type != 1 && tu.type != 4 &&
                  !shadow;
    u32 faddr[6];
    u32 nfaces = 1;
    faddr[0] = addr;
    if (tu.type == 1 || tu.type == 4) {
        nfaces = 6;
        for (u32 f = 1; f < 6; f++)
            faddr[f] = ((addr_reg & ~0x3fffffu) | (regs[R::TEX0_CUBE + f - 1] & 0x3fffff)) * 8;
    }
    u32 total = 0;
    for (u32 l = 0; l < levels; l++) total += texture_level_bytes(tu.w >> l, tu.h >> l, fmt);
    for (u32 f = 0; f < nfaces; f++)
        if (overlaps_rt(faddr[f], total)) raster_flush();
    for (u32 f = 0; f < nfaces; f++) {
        metal_guest_read(faddr[f], total);  // render-to-texture: GPU-only data to guest first
        tu.face[f] = texcache_get(faddr[f], tu.w, tu.h, fmt, levels);
    }
    auto simple = [](u32 m) { return m == 0 || m == 2 || m == 6 || m == 7; };
    const bool pow2 = (tu.w & (tu.w - 1)) == 0 && (tu.h & (tu.h - 1)) == 0;
    tu.fast_wrap = simple(tu.wrap_s) && simple(tu.wrap_t) && pow2;
    tu.simple2d = tu.type == 0 && tu.fast_wrap && !tu.need_lod && tu.face[0] && !tu.face[0]->lv.empty();
    tu.mag_rho2 = exp2f(-2.0f * tu.bias) * 0.999f;
}

// selection tables for blend factor slot k (0..3 source, 4..7 destination; channel = k & 3)
static void blend_factor_tables(DrawState &ds, u32 k, u32 f) {
    const u32 ch = k & 3;
    u8 idx = 12, inv = 0;
    switch (f) {
    case 0: idx = 12; break;
    case 1: idx = 13; break;
    case 2: case 3: idx = (u8)ch; break;
    case 4: case 5: idx = (u8)(4 + ch); break;
    case 6: case 7: idx = 3; break;
    case 8: case 9: idx = 7; break;
    case 10: case 11: idx = (u8)(8 + ch); break;
    case 12: case 13: idx = 11; break;
    case 14: idx = ch == 3 ? 13 : 14; break;
    default: idx = (u8)ch; break;
    }
    if (f >= 2 && f <= 13 && (f & 1)) inv = 0xff;
    ds.bf_idx[k] = idx;
    ds.bf_w[k] = idx < 12 ? idx >> 2 : 3;
    ds.bf_sh[k] = idx < 12 ? (idx & 3) * 8 : idx == 12 ? 16 : idx == 13 ? 8 : 0;
    ds.bf_q[k] = (u8)(ds.bf_w[k] * 4 + ds.bf_sh[k] / 8);
    ds.bf_qinv[k] = inv;
    ds.bf_inv[k] = inv;
}

void build_draw_state(DrawState &ds) {
    const u32 *regs = g.regs;
    update_luts();
    ds.vp_hw = f24(regs[R::VP_W] & 0xffffff);
    ds.vp_hh = f24(regs[R::VP_H] & 0xffffff);
    ds.vp_x = (float)sbits(regs[R::VP_CORNER], 0, 10);
    ds.vp_y = (float)sbits(regs[R::VP_CORNER], 16, 10);
    ds.depth_scale = f24(regs[R::DEPTH_SCALE] & 0xffffff);
    ds.depth_offset = f24(regs[R::DEPTH_OFFSET] & 0xffffff);
    ds.wbuffer = (regs[R::DEPTHMAP_EN] & 1) == 0;
    ds.cull = regs[R::CULL] & 3;
    ds.scissor_mode = regs[R::SCISSOR_MODE] & 3;
    ds.sc_x1 = regs[R::SCISSOR_XY1] & 0x3ff;
    ds.sc_y1 = (regs[R::SCISSOR_XY1] >> 16) & 0x3ff;
    ds.sc_x2 = (regs[R::SCISSOR_XY2] & 0x3ff) + 1;
    ds.sc_y2 = ((regs[R::SCISSOR_XY2] >> 16) & 0x3ff) + 1;
    ds.clip_enable = regs[R::CLIP_EN] & 1;
    for (int i = 0; i < 4; i++) ds.clip[i] = f24(regs[R::CLIP_COEF + i] & 0xffffff);
    rt_from_regs(ds.rt);

    // output merger
    const u32 om = regs[R::OM_MODE];
    ds.shadow_mode = (om & 3) == 3;
    ds.gas = (om & 3) == 1;
    ds.shadow_const = f16(regs[R::SHADOW] & 0xffff);
    ds.shadow_linear = f16(regs[R::SHADOW] >> 16);
    ds.blend = (om >> 8) & 1;
    const u32 bl = regs[R::BLEND];
    ds.eq_rgb = bl & 7;
    ds.eq_a = (bl >> 8) & 7;
    ds.sf_rgb = (bl >> 16) & 15;
    ds.df_rgb = (bl >> 20) & 15;
    ds.sf_a = (bl >> 24) & 15;
    ds.df_a = (bl >> 28) & 15;
    ds.logic_op = regs[R::LOGIC_OP] & 15;
    for (u32 k = 0; k < 8; k++) {
        const u32 ch = k & 3;
        const u32 f = k < 4 ? (ch == 3 ? ds.sf_a : ds.sf_rgb) : (ch == 3 ? ds.df_a : ds.df_rgb);
        blend_factor_tables(ds, k, f);
    }
    ds.blend_color = regs[R::BLEND_COLOR];
    const u32 at = regs[R::ALPHA_TEST];
    ds.alpha_test = at & 1;
    ds.alpha_func = (at >> 4) & 7;
    ds.alpha_ref = (at >> 8) & 255;
    if (ds.alpha_test && ds.alpha_func == 1) ds.alpha_test = false;
    const u32 st = regs[R::STENCIL_TEST], so = regs[R::STENCIL_OP];
    ds.stencil = (st & 1) && ds.rt.depth_fmt == 3;
    ds.st_func = (st >> 4) & 7;
    ds.st_mask_w = (st >> 8) & 255;
    ds.st_ref = (st >> 16) & 255;
    ds.st_mask_in = (st >> 24) & 255;
    ds.st_fail = so & 7;
    ds.st_zfail = (so >> 4) & 7;
    ds.st_zpass = (so >> 8) & 7;
    const u32 dcm = regs[R::DEPTH_COLOR_MASK];
    const bool ds_write = (regs[R::DEPTH_WRITE] & 3) != 0;
    ds.depth_test = dcm & 1;
    ds.depth_func = (dcm >> 4) & 7;
    ds.depth_write = ds_write && ((dcm >> 12) & 1);
    if (!ds_write) ds.st_mask_w = 0;
    ds.color_mask = (((dcm >> 8) & 1) ? 0xffu : 0) | (((dcm >> 9) & 1) ? 0xff00u : 0) |
                    (((dcm >> 10) & 1) ? 0xff0000u : 0) | (((dcm >> 11) & 1) ? 0xff000000u : 0);
    ds.color_write = (regs[R::COLOR_WRITE] & 0xf) != 0 && ds.color_mask != 0;
    ds.depth_bits = ds.rt.depth_fmt == 0 ? 16 : 24;
    if (ds.rt.depth_addr == 0) {
        ds.depth_test = false;
        ds.depth_write = false;
        ds.stencil = false;
    }
    ds.early_z = !ds.alpha_test && !ds.shadow_mode;
    ds.need_shade = ds.shadow_mode || ds.color_write || ds.alpha_test;
    ds.skip = ds.gas || (ds.alpha_test && ds.alpha_func == 0) ||
              (!ds.need_shade && !ds.depth_write && !ds.stencil) || ds.rt.w == 0;

    // combiners
    static const u32 tevbase[6] = {R::TEV0, R::TEV1, R::TEV2, R::TEV3, R::TEV4, R::TEV5};
    const u32 bufin = regs[R::FOG_MODE];
    bool uses_tex[4] = {false, false, false, false};
    for (u32 s = 0; s < 6; s++) {
        TevStageState &t = ds.tev[s];
        const u32 b = tevbase[s];
        const u32 src = regs[b], mod = regs[b + 1], ops = regs[b + 2], sc = regs[b + 4];
        for (u32 k = 0; k < 3; k++) {
            t.csrc[k] = (src >> (4 * k)) & 15;
            t.asrc[k] = (src >> (16 + 4 * k)) & 15;
            t.cmod[k] = (mod >> (4 * k)) & 15;
            t.amod[k] = (mod >> (12 + 4 * k)) & 7;
        }
        if (s == 0) {
            if (t.csrc[0] == 15) t.csrc[0] = t.csrc[2];
            if (t.csrc[1] == 15) t.csrc[1] = t.csrc[2];
        }
        t.cop = ops & 15;
        t.aop = (ops >> 16) & 15;
        t.konst = regs[b + 3];
        u32 cs = sc & 3, as = (sc >> 16) & 3;
        t.cmul = cs < 3 ? 1 << cs : 1;
        t.amul = as < 3 ? 1 << as : 1;
        t.upd_rgb = s < 4 && ((bufin >> (8 + s)) & 1);
        t.upd_a = s < 4 && ((bufin >> (12 + s)) & 1);
        for (u32 k = 0; k < 3; k++) {
            if (t.csrc[k] >= 3 && t.csrc[k] <= 6) uses_tex[t.csrc[k] - 3] = true;
            if (t.asrc[k] >= 3 && t.asrc[k] <= 6) uses_tex[t.asrc[k] - 3] = true;
        }
    }
    ds.tev_buf_color = regs[R::TEV_BUF_COLOR];
    tev_prepare(ds);
    if (stats_on && getenv("GPU_TEV")) {
        LOG("draw: tev_n %u lighting %d tex %d%d%d%d fog %d blend %d at %d\n", ds.tev_n, (int)!(regs[R::LIGHTING_DISABLE] & 1),
            (int)((regs[R::TEX_CONFIG]) & 1), (int)((regs[R::TEX_CONFIG] >> 1) & 1), (int)((regs[R::TEX_CONFIG] >> 2) & 1), (int)((regs[R::TEX_CONFIG] >> 10) & 1),
            (int)ds.fog, (int)ds.blend, (int)ds.alpha_test);
        for (u32 i = 0; i < ds.tev_n; i++) {
            const TevStageState &t = ds.tev[i];
            LOG("  stage %u: c %x%x%x m %x%x%x op %u | a %x%x%x m %x%x%x op %u | k %08x x%u x%u buf %d%d\n", i, t.csrc[0], t.csrc[1], t.csrc[2], t.cmod[0], t.cmod[1], t.cmod[2], t.cop,
                t.asrc[0], t.asrc[1], t.asrc[2], t.amod[0], t.amod[1], t.amod[2], t.aop, t.konst, t.cmul, t.amul, t.upd_rgb, t.upd_a);
        }
    }
    const u32 fm = bufin & 7;
    ds.fog = fm == 5;
    if (fm == 7) ds.gas = ds.skip = true;
    ds.fog_flip = (bufin >> 16) & 1;
    ds.fog_color = regs[R::FOG_COLOR] & 0xffffff;
    ds.fog_lut = g.fog_luts;

    // lighting
    ds.lighting = !(regs[R::LIGHTING_DISABLE] & 1);
    LightingState &ls = ds.light;
    if (ds.lighting) {
        const u32 c0 = regs[R::LIGHT_CONFIG0], c1 = regs[R::LIGHT_CONFIG1];
        ls.num_lights = (regs[R::LIGHT_NUM] & 7) + 1;
        ls.config = (c0 >> 4) & 15;
        ls.shadow_enable = c0 & 1;
        ls.primary_alpha = (c0 >> 2) & 1;
        ls.secondary_alpha = (c0 >> 3) & 1;
        ls.shadow_primary = (c0 >> 16) & 1;
        ls.shadow_secondary = (c0 >> 17) & 1;
        ls.shadow_invert = (c0 >> 18) & 1;
        ls.shadow_alpha = (c0 >> 19) & 1;
        ls.bump_sel = (c0 >> 22) & 3;
        ls.shadow_sel = (c0 >> 24) & 3;
        ls.clamp_highlights = (c0 >> 27) & 1;
        ls.bump_mode = (c0 >> 28) & 3;
        ls.bump_renorm = !((c0 >> 30) & 1);
        const u32 absr = regs[R::LUT_ABS], inr = regs[R::LUT_SELECT], scr = regs[R::LUT_SCALE];
        auto sel = [&](LutSel &s, u32 shift, u32 absbit, bool disabled, u32 sampler) {
            s.enabled = !disabled && sampler_supported(ls.config, sampler);
            s.abs = !((absr >> absbit) & 1);
            s.input = (inr >> shift) & 7;
            s.scale = lut_scale((scr >> shift) & 7);
        };
        sel(ls.d0, 0, 1, (c1 >> 16) & 1, 0);
        sel(ls.d1, 4, 5, (c1 >> 17) & 1, 1);
        sel(ls.sp, 8, 9, false, 8);
        sel(ls.fr, 12, 13, (c1 >> 19) & 1, 3);
        sel(ls.rb, 16, 17, (c1 >> 22) & 1, 4);
        sel(ls.rg, 20, 21, (c1 >> 21) & 1, 5);
        sel(ls.rr, 24, 25, (c1 >> 20) & 1, 6);
        const u32 amb = regs[R::LIGHT_AMBIENT];
        ls.global_ambient[0] = ((amb >> 20) & 0x3ff) / 255.0f;
        ls.global_ambient[1] = ((amb >> 10) & 0x3ff) / 255.0f;
        ls.global_ambient[2] = (amb & 0x3ff) / 255.0f;
        const u32 perm = regs[R::LIGHT_PERMUTATION];
        for (u32 i = 0; i < ls.num_lights; i++) {
            LightState &lt = ls.lights[i];
            const u32 num = (perm >> (4 * i)) & 7;
            const u32 *L = regs + R::LIGHT0 + 16 * num;
            auto col = [](u32 v, float *o) {
                o[0] = ((v >> 20) & 0x3ff) / 255.0f;
                o[1] = ((v >> 10) & 0x3ff) / 255.0f;
                o[2] = (v & 0x3ff) / 255.0f;
            };
            col(L[0], lt.spec0);
            col(L[1], lt.spec1);
            col(L[2], lt.diffuse);
            col(L[3], lt.ambient);
            lt.pos[0] = f16(L[4] & 0xffff);
            lt.pos[1] = f16(L[4] >> 16);
            lt.pos[2] = f16(L[5] & 0xffff);
            lt.spot[0] = sbits(L[6], 0, 13) / 2047.0f;
            lt.spot[1] = sbits(L[6], 16, 13) / 2047.0f;
            lt.spot[2] = sbits(L[7], 0, 13) / 2047.0f;
            lt.directional = L[9] & 1;
            lt.two_sided = (L[9] >> 1) & 1;
            lt.geo0 = (L[9] >> 2) & 1;
            lt.geo1 = (L[9] >> 3) & 1;
            lt.da_bias = f20(L[10] & 0xfffff);
            lt.da_scale = f20(L[11] & 0xfffff);
            lt.dist_atten = !((c1 >> (24 + num)) & 1);
            lt.spot_atten = !((c1 >> (8 + num)) & 1);
            lt.shadow = !((c1 >> num) & 1);
            lt.num = num;
        }
        ls.luts = g.light_luts;
        if (ls.bump_mode == 1 || ls.bump_mode == 2) uses_tex[ls.bump_sel & 3] = true;
        if (ls.shadow_enable) uses_tex[ls.shadow_sel & 3] = true;
    }

    // textures
    for (u32 i = 0; i < 3; i++) {
        if (uses_tex[i] && ds.need_shade && !ds.skip) {
            setup_texunit(ds, i);
        } else {
            ds.tex[i] = TexUnitState{};
        }
        ds.need_tex[i] = ds.tex[i].enabled;
    }
    ProcTexState &pt = ds.proctex;
    pt.enabled = ((regs[R::TEX_CONFIG] >> 10) & 1) && uses_tex[3];
    ds.need_tex[3] = pt.enabled;
    if (pt.enabled) {
        const u32 p = regs[R::PROCTEX];
        pt.coord = (regs[R::TEX_CONFIG] >> 8) & 3;
        if (pt.coord > 2) pt.coord = 2;
        pt.u_clamp = p & 7;
        pt.v_clamp = (p >> 3) & 7;
        pt.color_comb = (p >> 6) & 15;
        pt.alpha_comb = (p >> 10) & 15;
        pt.separate_alpha = (p >> 14) & 1;
        pt.noise = (p >> 15) & 1;
        pt.u_shift = (p >> 16) & 3;
        pt.v_shift = (p >> 18) & 3;
        pt.noise_amp_u = sbits(regs[R::PT_NOISE_U], 0, 16) / 4095.0f;
        pt.noise_amp_v = sbits(regs[R::PT_NOISE_V], 0, 16) / 4095.0f;
        pt.noise_phase_u = f16(regs[R::PT_NOISE_U] >> 16);
        pt.noise_phase_v = f16(regs[R::PT_NOISE_V] >> 16);
        pt.noise_freq_u = f16(regs[R::PT_NOISE_FREQ] & 0xffff);
        pt.noise_freq_v = f16(regs[R::PT_NOISE_FREQ] >> 16);
        pt.filter = regs[R::PT_LUT] & 7;
        pt.lut_width = (regs[R::PT_LUT] >> 11) & 0xff;
        pt.lut_offset = regs[R::PT_LUT_OFFSET] & 0xff;
        pt.luts = g.pt_luts;
    }

    // attributes to interpolate
    u32 ag = 0;
    if (ds.need_shade) {
        ag |= AG_COLOR;
        auto need_coord = [&](u32 c) { ag |= c == 2 ? AG_TC2W : AG_TC01; };
        for (u32 i = 0; i < 3; i++)
            if (ds.need_tex[i]) need_coord(ds.tex[i].coord);
        if (ds.need_tex[0] && ds.tex[0].type != 0) ag |= AG_TC2W;  // tc0.w
        if (ds.need_tex[3]) need_coord(pt.coord);
        if (ds.lighting) ag |= AG_QUAT | AG_VIEW;
    }
    ds.attr_groups = ag;
}

DrawState *draw_state() {
    if (g.ds && g.ds_version == g.regs_version) return g.ds;
    RTConfig rt;
    rt_from_regs(rt);
    if (batch.has_rt && !(rt == batch.rt) && batch_nonempty()) raster_flush();
    if (!batch.has_rt || !(rt == batch.rt)) {
        batch.rt = rt;
        batch.has_rt = true;
        batch.tiles_x = (rt.w + TILE - 1) >> TILE_SHIFT;
        batch.tiles_y = (rt.h + TILE - 1) >> TILE_SHIFT;
        batch.bins.resize(batch.tiles_x * batch.tiles_y);
        for (auto &b : batch.bins) b.clear();
    }
    auto ds = std::make_unique<DrawState>();
    build_draw_state(*ds);  // may flush (render-to-texture); states vector survives as needed
    // the texture setup may have flushed the batch: make sure our RT is still registered
    if (!batch.has_rt) {
        batch.rt = rt;
        batch.has_rt = true;
    }
    g.ds = ds.get();
    g.ds_version = g.regs_version;
    batch.states.push_back(std::move(ds));
    return g.ds;
}

void raster_begin_draw(const DrawState *) {}

// ---------------------------------------------------------------- clipping & setup
namespace {
struct CV {
    float a[24];  // OutVertex layout
};

inline float clip_dist(const CV &v, int plane, const float *uc) {
    const float x = v.a[0], y = v.a[1], z = v.a[2], w = v.a[3];
    switch (plane) {
    case 0: return w - x;
    case 1: return w + x;
    case 2: return w - y;
    case 3: return w + y;
    case 4: return -z;
    case 5: return z + w;
    case 6: return w - 0.00001f;  // keep w strictly positive
    default: return x * uc[0] + y * uc[1] + z * uc[2] + w * uc[3];
    }
}
constexpr float EPS_Z = 0.00000001f;

void setup_triangle(const DrawState &ds, const CV &v0in, const CV &v1in, const CV &v2in);
void setup_triangle_sv(const DrawState &ds, const SV &s0, const SV &s1, const SV &s2);

}  // namespace

void raster_prepare_vertex(const DrawState &ds, const OutVertex &v, SV &sv, u32 &outcode) {
    CV cv;
    memcpy(cv.a, &v, sizeof(CV));
    const int nplanes = ds.clip_enable ? 8 : 7;
    u32 oc = 0;
    for (int p = 0; p < nplanes; p++)
        if (!(clip_dist(cv, p, ds.clip) >= -EPS_Z)) oc |= 1u << p;
    outcode = oc;
    extern void raster_to_screen(const DrawState &ds, const float *a, SV &s);
    if (!oc) raster_to_screen(ds, cv.a, sv);
}

void raster_triangle_sv(const DrawState *ds, const SV &a, const SV &b, const SV &c) {
    if (!tl_sink) vtx_finish_jobs();  // keep primitive order with deferred draws
    if (ds->attr_groups & AG_QUAT) {
        // flip quaternions opposite to vertex 0's (sign is preserved by the 1/w scaling)
        auto dq = [&](const SV &v) {
            float d = 0;
            for (int i = 12; i < 16; i++) d += v.attr[i] * a.attr[i];
            return d;
        };
        if (dq(b) < 0 || dq(c) < 0) {
            SV b2 = b, c2 = c;
            if (dq(b) < 0)
                for (int i = 12; i < 16; i++) b2.attr[i] = -b2.attr[i];
            if (dq(c) < 0)
                for (int i = 12; i < 16; i++) c2.attr[i] = -c2.attr[i];
            setup_triangle_sv(*ds, a, b2, c2);
            return;
        }
    }
    setup_triangle_sv(*ds, a, b, c);
}

void raster_triangle(const DrawState *ds, const OutVertex &a, const OutVertex &b,
                     const OutVertex &c) {
    if (!tl_sink) vtx_finish_jobs();  // keep primitive order with deferred draws
    if (use_metal) {
        // clipping, culling and snapping happen on the GPU
        MBatch &mb = batch.mb;
        const u32 first = (u32)mb.idx.size(), v = (u32)mb.cverts.size();
        mb.cverts.push_back(a);
        mb.cverts.push_back(b);
        mb.cverts.push_back(c);
        mb.idx.insert(mb.idx.end(), {v | MIDX_CPU, (v + 1) | MIDX_CPU, (v + 2) | MIDX_CPU});
        mb.add_seg(ds, first);
        return;
    }
    CV buf_a[12], buf_b[12];
    memcpy(buf_a[0].a, &a, sizeof(CV));
    memcpy(buf_a[1].a, &b, sizeof(CV));
    memcpy(buf_a[2].a, &c, sizeof(CV));
    // flip quaternions that are opposite to vertex 0's
    for (int k = 1; k < 3; k++) {
        float d = 0;
        for (int i = 4; i < 8; i++) d += buf_a[k].a[i] * buf_a[0].a[i];
        if (d < 0)
            for (int i = 4; i < 8; i++) buf_a[k].a[i] = -buf_a[k].a[i];
    }
    const int nplanes = ds->clip_enable ? 8 : 7;
    u32 outside = 0;
    for (int k = 0; k < 3; k++)
        for (int p = 0; p < nplanes; p++)
            if (!(clip_dist(buf_a[k], p, ds->clip) >= -EPS_Z)) outside |= 1u << p;
    CV *in = buf_a, *out = buf_b;
    int n = 3;
    if (outside) {
        for (int p = 0; p < nplanes; p++) {
            if (!(outside & (1u << p))) continue;
            int m = 0;
            const CV *ref = &in[n - 1];
            float dref = clip_dist(*ref, p, ds->clip);
            for (int i = 0; i < n; i++) {
                const CV &v = in[i];
                float dv = clip_dist(v, p, ds->clip);
                bool vin = dv >= -EPS_Z, rin = dref >= -EPS_Z;
                if (vin != rin && m < 12) {
                    // intersection: factor = dp_prev / (dp_prev - dp); v*factor + ref*(1-factor)
                    float f = dref / (dref - dv);
                    CV &o = out[m++];
                    for (int j = 0; j < 24; j++) o.a[j] = v.a[j] * f + ref->a[j] * (1.0f - f);
                }
                if (vin && m < 12) out[m++] = v;
                ref = &v;
                dref = dv;
            }
            std::swap(in, out);
            n = m;
            if (n < 3) return;
        }
    }
    for (int i = 0; i + 2 < n; i++) setup_triangle(*ds, in[0], in[i + 1], in[i + 2]);
}

namespace {
}  // namespace
void raster_to_screen(const DrawState &ds, const float *a, SV &s);
namespace {
void to_screen(const DrawState &ds, const CV &v, SV &s) { raster_to_screen(ds, v.a, s); }
}  // namespace
void raster_to_screen(const DrawState &ds, const float *va, SV &s) {
    struct { const float *a; } v{va};
    const float iw = 1.0f / v.a[3];
    s.iw = iw;
    s.sx = (v.a[0] * iw + 1.0f) * ds.vp_hw + ds.vp_x;
    s.sy = (v.a[1] * iw + 1.0f) * ds.vp_hh + ds.vp_y;
    s.sz = v.a[2] * iw;
    // attribute layout: color 0-3, tc0 4-5, tc1 6-7, tc2 8-9, tc0w 10, -, quat 12-15, view 16-18
    static const s8 src[NUM_ATTR] = {8, 9, 10, 11, 12, 13, 14, 15, 22, 23,
                                     16, -1, 4, 5, 6, 7, 18, 19, 20, -1};
    for (u32 i = 0; i < NUM_ATTR; i++) s.attr[i] = src[i] >= 0 ? v.a[src[i]] * iw : 0.0f;
}
namespace {

inline s32 to_fix(float f) {
    float r = roundf(f * 16.0f);
    if (!(r > -1e8f)) return -100000000;
    if (r > 1e8f) return 100000000;
    return (s32)r;
}

inline s64 signed_area(s64 ax, s64 ay, s64 bx, s64 by, s64 cx, s64 cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

bool right_or_flat_bottom(s64 vx, s64 vy, s64 l1x, s64 l1y, s64 l2x, s64 l2y) {
    if (l1y == l2y) return vy < l1y;
    return vx < l1x + (l2x - l1x) * (vy - l1y) / (l2y - l1y);
}

void setup_triangle(const DrawState &ds, const CV &c0, const CV &c1, const CV &c2) {
    SV sv[3];
    to_screen(ds, c0, sv[0]);
    to_screen(ds, c1, sv[1]);
    to_screen(ds, c2, sv[2]);
    setup_triangle_sv(ds, sv[0], sv[1], sv[2]);
}

void setup_triangle_sv(const DrawState &ds, const SV &sv0, const SV &sv1, const SV &sv2) {
    const SV *sv[3] = {&sv0, &sv1, &sv2};
    s64 X[3], Y[3];
    for (int i = 0; i < 3; i++) {
        X[i] = to_fix(sv[i]->sx);
        Y[i] = to_fix(sv[i]->sy);
    }
    s64 area = signed_area(X[0], Y[0], X[1], Y[1], X[2], Y[2]);
    int o1 = 1, o2 = 2;
    if (ds.cull == 0 || ds.cull == 3) {
        if (area < 0) {
            std::swap(o1, o2);
            area = -area;
        }
    } else if (ds.cull == 1) {
        std::swap(o1, o2);
        area = -area;
    }
    if (area <= 0) return;
    const SV *v[3] = {sv[0], sv[o1], sv[o2]};
    const s64 x[3] = {X[0], X[o1], X[o2]}, y[3] = {Y[0], Y[o1], Y[o2]};

    // bounding box in 12.4, clipped by include-scissor, then to pixels
    s64 minx = std::min({x[0], x[1], x[2]}), maxx = std::max({x[0], x[1], x[2]});
    s64 miny = std::min({y[0], y[1], y[2]}), maxy = std::max({y[0], y[1], y[2]});
    if (ds.scissor_mode == 3) {
        minx = std::max<s64>(minx, (s64)ds.sc_x1 << 4);
        miny = std::max<s64>(miny, (s64)ds.sc_y1 << 4);
        maxx = std::min<s64>(maxx, (s64)ds.sc_x2 << 4);
        maxy = std::min<s64>(maxy, (s64)ds.sc_y2 << 4);
    }
    minx &= ~(s64)15;
    miny &= ~(s64)15;
    maxx = (maxx + 15) & ~(s64)15;
    maxy = (maxy + 15) & ~(s64)15;
    s64 px0 = minx >> 4, py0 = miny >> 4, px1 = maxx >> 4, py1 = maxy >> 4;
    px0 = std::max<s64>(px0, 0);
    py0 = std::max<s64>(py0, 0);
    px1 = std::min<s64>(px1, ds.rt.w);
    py1 = std::min<s64>(py1, ds.rt.h);
    if (px0 >= px1 || py0 >= py1) return;

    std::vector<Tri> &dst = tl_sink ? tl_sink->tris : batch.tris;
    dst.emplace_back();
    Tri &t = dst.back();
    t.ds = &ds;
    t.x0 = (s32)px0;
    t.y0 = (s32)py0;
    t.x1 = (s32)px1;
    t.y1 = (s32)py1;
    const s64 bias[3] = {right_or_flat_bottom(x[0], y[0], x[1], y[1], x[2], y[2]) ? -1 : 0,
                         right_or_flat_bottom(x[1], y[1], x[2], y[2], x[0], y[0]) ? -1 : 0,
                         right_or_flat_bottom(x[2], y[2], x[0], y[0], x[1], y[1]) ? -1 : 0};
    const s64 Px = px0 * 16 + 8, Py = py0 * 16 + 8;
    s64 eraw[3];
    for (int i = 0; i < 3; i++) {
        const int a = (i + 1) % 3, b = (i + 2) % 3;
        eraw[i] = signed_area(x[a], y[a], x[b], y[b], Px, Py);
        t.e0[i] = eraw[i] + bias[i];
        t.dex[i] = -(y[b] - y[a]) * 16;
        t.dey[i] = (x[b] - x[a]) * 16;
    }
    const double inv_area = 1.0 / (double)area;
    double b0[3], bx[3], by[3];
    for (int i = 0; i < 3; i++) {
        b0[i] = eraw[i] * inv_area;
        bx[i] = t.dex[i] * inv_area;
        by[i] = t.dey[i] * inv_area;
    }
    auto plane = [&](float q0, float q1, float q2, float &p0, float &px, float &py) {
        p0 = (float)(q0 * b0[0] + q1 * b0[1] + q2 * b0[2]);
        px = (float)(q0 * bx[0] + q1 * bx[1] + q2 * bx[2]);
        py = (float)(q0 * by[0] + q1 * by[1] + q2 * by[2]);
    };
    t.z0 = v[0]->sz * b0[0] + v[1]->sz * b0[1] + v[2]->sz * b0[2];
    t.zx = v[0]->sz * bx[0] + v[1]->sz * bx[1] + v[2]->sz * bx[2];
    t.zy = v[0]->sz * by[0] + v[1]->sz * by[1] + v[2]->sz * by[2];
    plane(v[0]->iw, v[1]->iw, v[2]->iw, t.q0, t.qx, t.qy);
    const u32 ag = ds.attr_groups;
    for (u32 grp = 0; grp < 5; grp++) {
        if (!(ag & (1u << grp))) continue;
        for (u32 k = grp * 4; k < grp * 4 + 4; k++)
            plane(v[0]->attr[k], v[1]->attr[k], v[2]->attr[k], t.a0[k], t.ax[k], t.ay[k]);
    }

    if (!tl_sink) bin_last_tri();
}

// ---------------------------------------------------------------- framebuffer tile I/O
struct TileRect {
    s32 x0, y0, x1, y1;
};

void load_tile(const RTConfig &rt, const TileRect &r, TileBuf &tb) {
    u8 *cbuf = rt.color_addr ? pptr(rt.color_addr) : nullptr;
    u8 *dbuf = rt.depth_addr ? pptr(rt.depth_addr) : nullptr;
    const u32 cf = rt.color_fmt;
    const u32 cbpp = color_bytes(cf & 7), dbpp = depth_bytes(rt.depth_fmt);
    const u32 pf = cbuf_to_pixfmt(cf & 7);
    for (s32 y = r.y0; y < r.y1; y++) {
        const u32 my = rt.h - 1 - y;
        u32 *crow = tb.color + (y - r.y0) * TILE;
        u32 *drow = tb.depth + (y - r.y0) * TILE;
        u8 *srow = tb.stencil + (y - r.y0) * TILE;
        for (s32 x = r.x0; x < r.x1; x++) {
            if (cbuf) {
                const u8 *p = cbuf + tiled_offset(x, my, rt.w, cbpp);
                if (cf & 0x80) {
                    u32 v;
                    memcpy(&v, p, 4);
                    crow[x - r.x0] = v;
                } else {
                    crow[x - r.x0] = decode_pixel(pf, p);
                }
            }
            if (dbuf) {
                const u8 *p = dbuf + tiled_offset(x, my, rt.w, dbpp);
                if (rt.depth_fmt == 0) {
                    drow[x - r.x0] = p[0] | (p[1] << 8);
                    srow[x - r.x0] = 0;
                } else {
                    drow[x - r.x0] = p[0] | (p[1] << 8) | (p[2] << 16);
                    srow[x - r.x0] = rt.depth_fmt == 3 ? p[3] : 0;
                }
            }
        }
    }
}

void store_tile(const RTConfig &rt, const TileRect &r, const TileBuf &tb, bool color,
                bool depth) {
    u8 *cbuf = rt.color_addr ? pptr(rt.color_addr) : nullptr;
    u8 *dbuf = rt.depth_addr ? pptr(rt.depth_addr) : nullptr;
    if (!cbuf) color = false;
    if (!dbuf) depth = false;
    const u32 cf = rt.color_fmt;
    const u32 cbpp = color_bytes(cf & 7), dbpp = depth_bytes(rt.depth_fmt);
    const u32 pf = cbuf_to_pixfmt(cf & 7);
    for (s32 y = r.y0; y < r.y1; y++) {
        const u32 my = rt.h - 1 - y;
        const u32 *crow = tb.color + (y - r.y0) * TILE;
        const u32 *drow = tb.depth + (y - r.y0) * TILE;
        const u8 *srow = tb.stencil + (y - r.y0) * TILE;
        for (s32 x = r.x0; x < r.x1; x++) {
            if (color) {
                u8 *p = cbuf + tiled_offset(x, my, rt.w, cbpp);
                if (cf & 0x80)
                    memcpy(p, &crow[x - r.x0], 4);
                else
                    encode_pixel(pf, crow[x - r.x0], p);
            }
            if (depth) {
                u8 *p = dbuf + tiled_offset(x, my, rt.w, dbpp);
                const u32 d = drow[x - r.x0];
                p[0] = d;
                p[1] = d >> 8;
                if (rt.depth_fmt != 0) p[2] = d >> 16;
                if (rt.depth_fmt == 3) p[3] = srow[x - r.x0];
            }
        }
    }
}

// ---------------------------------------------------------------- per-fragment operations
inline bool compare(u32 func, u32 a, u32 b) {
    switch (func) {
    case 0: return false;
    case 1: return true;
    case 2: return a == b;
    case 3: return a != b;
    case 4: return a < b;
    case 5: return a <= b;
    case 6: return a > b;
    default: return a >= b;
    }
}

inline u8 stencil_op(u32 op, u8 old, u8 ref) {
    switch (op) {
    case 0: return old;
    case 1: return 0;
    case 2: return ref;
    case 3: return old == 255 ? 255 : old + 1;
    case 4: return old == 0 ? 0 : old - 1;
    case 5: return ~old;
    case 6: return old + 1;
    default: return old - 1;
    }
}

struct TileState {
    TileBuf *tb;
    bool color_dirty, depth_dirty;
};

// returns false if the fragment is discarded
__attribute__((always_inline)) inline bool depth_stencil(const DrawState &ds, TileState &ts, u32 i, u32 z) {
    TileBuf &tb = *ts.tb;
    u8 old_st = 0;
    auto upd = [&](u32 op) {
        if (!ds.st_mask_w) return;
        u8 ns = stencil_op(op, old_st, ds.st_ref);
        u8 v = (ns & ds.st_mask_w) | (old_st & ~ds.st_mask_w);
        if (v != tb.stencil[i]) {
            tb.stencil[i] = v;
            ts.depth_dirty = true;
        }
    };
    if (ds.stencil) {
        old_st = tb.stencil[i];
        const u8 dest = old_st & ds.st_mask_in, ref = ds.st_ref & ds.st_mask_in;
        if (!compare(ds.st_func, ref, dest)) {
            upd(ds.st_fail);
            return false;
        }
    }
    if (ds.depth_test && !compare(ds.depth_func, z, tb.depth[i])) {
        if (ds.stencil) upd(ds.st_zfail);
        return false;
    }
    if (ds.depth_write) {
        tb.depth[i] = z;
        ts.depth_dirty = true;
    }
    if (ds.stencil) upd(ds.st_zpass);
    return true;
}

inline s32 blend_eq(u32 eq, s32 s, s32 d) {
    s32 r;
    switch (eq) {
    case 1: r = (s - d) / 255; break;
    case 2: r = (d - s) / 255; break;
    case 3: r = std::min(s, d) / 255; break;
    case 4: r = std::max(s, d) / 255; break;
    default: r = (s + d) / 255; break;
    }
    return r < 0 ? 0 : r > 255 ? 255 : r;
}

#if defined(__aarch64__)
// x / 255, exact for x < 11M ((x * 8421505) >> 31, 64-bit products; verified by the unit tests)
inline uint32x4_t q_div255_u32(uint32x4_t x) {
    const uint32x2_t m = vdup_n_u32(8421505);
    const uint64x2_t lo = vmull_u32(vget_low_u32(x), m), hi = vmull_u32(vget_high_u32(x), m);
    return vcombine_u32(vshrn_n_u64(lo, 31), vshrn_n_u64(hi, 31));
}

inline uint32x4_t blend_eq4(u32 eq, uint16x4_t s, uint16x4_t d) {
    switch (eq) {
    case 1: return q_div255_u32(vmovl_u16(vqsub_u16(s, d)));  // negative -> 0 like the clamp
    case 2: return q_div255_u32(vmovl_u16(vqsub_u16(d, s)));
    case 3: return q_div255_u32(vmovl_u16(vmin_u16(s, d)));
    case 4: return q_div255_u32(vmovl_u16(vmax_u16(s, d)));
    default: return q_div255_u32(vaddl_u16(s, d));
    }
}
#endif

// factors are bytes of {src, dst, const, special}; special = saturate | 255 << 8 (0 at byte 2)
inline u32 blend_scalar(const DrawState &ds, u32 src, u32 dst) {
    const u32 sa = src >> 24, ida = 255 - (dst >> 24);
    const u32 W[4] = {src, dst, ds.blend_color, (sa < ida ? sa : ida) | 0xff00u};
    u32 out = 0;
    for (u32 ch = 0; ch < 4; ch++) {
        const u32 sf = ((W[ds.bf_w[ch]] >> ds.bf_sh[ch]) & 255) ^ ds.bf_inv[ch];
        const u32 df = ((W[ds.bf_w[4 + ch]] >> ds.bf_sh[4 + ch]) & 255) ^ ds.bf_inv[4 + ch];
        const s32 s = (s32)(((src >> (8 * ch)) & 255) * sf);
        const s32 d = (s32)(((dst >> (8 * ch)) & 255) * df);
        out |= (u32)blend_eq(ch == 3 ? ds.eq_a : ds.eq_rgb, s, d) << (8 * ch);
    }
    return out;
}

inline u32 blend(const DrawState &ds, u32 src, u32 dst) {
#if defined(__aarch64__)
    const u32 sa = src >> 24, ida = 255 - (dst >> 24);
    const uint32x4_t W = {src, dst, ds.blend_color, (sa < ida ? sa : ida) | 0xff00u};
    const uint8x8_t f = veor_u8(vqtbl1_u8(vreinterpretq_u8_u32(W), vld1_u8(ds.bf_q)), vld1_u8(ds.bf_qinv));
    const uint8x8_t sd = vreinterpret_u8_u64(vcreate_u64((u64)src | (u64)dst << 32));
    const uint16x8_t p = vmull_u8(sd, f);
    const uint16x4_t s = vget_low_u16(p), d = vget_high_u16(p);
    uint32x4_t r = blend_eq4(ds.eq_rgb, s, d);
    if (ds.eq_a != ds.eq_rgb) r = vsetq_lane_u32(vgetq_lane_u32(blend_eq4(ds.eq_a, s, d), 3), r, 3);
    const uint8x8_t o = vqmovn_u16(vcombine_u16(vqmovn_u32(r), vdup_n_u16(0)));
    return vget_lane_u32(vreinterpret_u32_u8(o), 0);
#else
    return blend_scalar(ds, src, dst);
#endif
}
}  // namespace

// self test: vectorized vs scalar blending for every factor / equation combination
bool raster_blend_selftest() {
    DrawState ds{};
    u32 seed = 1;
    auto rnd = [&]() { return seed = seed * 1664525u + 1013904223u; };
    for (u32 eq = 0; eq < 25; eq++)
        for (u32 fs = 0; fs < 15; fs++)
            for (u32 fd = 0; fd < 15; fd++) {
                ds.eq_rgb = eq % 5;
                ds.eq_a = eq / 5;
                ds.sf_rgb = (u8)fs;
                ds.df_rgb = (u8)fd;
                ds.sf_a = (u8)((fs + 3) % 15);
                ds.df_a = (u8)((fd + 7) % 15);
                for (u32 k = 0; k < 8; k++) {
                    const u32 ch = k & 3;
                    const u32 f = k < 4 ? (ch == 3 ? ds.sf_a : ds.sf_rgb) : (ch == 3 ? ds.df_a : ds.df_rgb);
                    blend_factor_tables(ds, k, f);
                }
                for (int i = 0; i < 64; i++) {
                    ds.blend_color = rnd();
                    const u32 s = i < 4 ? (i & 1 ? 0xffffffffu : 0) : rnd(), d = i < 4 ? (i & 2 ? 0xffffffffu : 0) : rnd();
                    if (blend(ds, s, d) != blend_scalar(ds, s, d)) return false;
                }
            }
    return true;
}
namespace {

inline u32 logic_op(u32 op, u32 s, u32 d) {
    switch (op) {
    case 0: return 0;
    case 1: return s & d;
    case 2: return s & ~d;
    case 3: return s;
    case 4: return 0xffffffffu;
    case 5: return ~s;
    case 6: return d;
    case 7: return ~d;
    case 8: return ~(s & d);
    case 9: return s | d;
    case 10: return ~(s | d);
    case 11: return s ^ d;
    case 12: return ~(s ^ d);
    case 13: return ~s & d;
    case 14: return s | ~d;
    default: return ~s | d;
    }
}

inline u32 round_color(float f) {
    float v = roundf(f * 255.0f);
    return !(v > 0) ? 0 : v >= 255 ? 255 : (u32)v;
}

// compute the combiner output for a fragment
inline u32 shade(const DrawState &ds, const Tri &t, float dx, float dy, float W) {
    float a[NUM_ATTR];
    const u32 ag = ds.attr_groups;
    for (u32 grp = 0; grp < 5; grp++) {
        if (!(ag & (1u << grp))) continue;
        for (u32 k = grp * 4; k < grp * 4 + 4; k++)
            a[k] = (t.a0[k] + t.ax[k] * dx + t.ay[k] * dy) * W;
    }
    const u32 primary = rgba(round_color(a[0]), round_color(a[1]), round_color(a[2]),
                             round_color(a[3]));
    u32 tex[4] = {0, 0, 0, 0};
    static const u8 uvidx[3] = {4, 6, 8};
    for (u32 i = 0; i < 3; i++) {
        if (!ds.need_tex[i]) continue;
        const TexUnitState &tu = ds.tex[i];
        const u32 k = uvidx[tu.coord];
        const float u = a[k], v = a[k + 1];
        float lod = 0;
        if (tu.need_lod) {
            const float Wq = W;
            const float dudx = (t.ax[k] - u * t.qx) * Wq, dvdx = (t.ax[k + 1] - v * t.qx) * Wq;
            const float dudy = (t.ay[k] - u * t.qy) * Wq, dvdy = (t.ay[k + 1] - v * t.qy) * Wq;
            const float fw = (float)tu.w, fh = (float)tu.h;
            const float rx = dudx * dudx * fw * fw + dvdx * dvdx * fh * fh;
            const float ry = dudy * dudy * fw * fw + dvdy * dvdy * fh * fh;
            const float rho2 = rx > ry ? rx : ry;
            lod = rho2 > 0 ? 0.5f * log2f(rho2) : -100.0f;
            lod += tu.bias;
        }
        tex[i] = sample_texture(tu, u, v, a[10], lod);
    }
    if (ds.need_tex[3]) {
        const u32 k = uvidx[ds.proctex.coord];
        tex[3] = sample_proctex(ds.proctex, a[k], a[k + 1]);
    }
    u32 fp = 0, fs = 0;
    if (ds.lighting) compute_lighting(ds.light, &a[12], &a[16], tex, fp, fs);
    return tev_combine(ds, primary, fp, fs, tex);
}

typedef float f4v __attribute__((ext_vector_type(4)));

// combiner output for up to 4 fragments of one triangle (same arithmetic as shade())
void shade4(const DrawState &ds, const Tri &t, u32 n, const float *dxs, const float *dys,
            const float *Ws, u32 *out) {
    const f4v vdx = {dxs[0], dxs[1], dxs[2], dxs[3]};
    const f4v vdy = {dys[0], dys[1], dys[2], dys[3]};
    const f4v vW = {Ws[0], Ws[1], Ws[2], Ws[3]};
    alignas(16) float a[NUM_ATTR][4];
    const u32 ag = ds.attr_groups;
    for (u32 grp = 0; grp < 5; grp++) {
        if (!(ag & (1u << grp))) continue;
        for (u32 k = grp * 4; k < grp * 4 + 4; k++) {
            const f4v v = (t.a0[k] + t.ax[k] * vdx + t.ay[k] * vdy) * vW;
            *(f4v *)a[k] = v;
        }
    }
    alignas(16) u32 prim[4], fp[4] = {0, 0, 0, 0}, fs[4] = {0, 0, 0, 0};
    alignas(16) u32 tex[4][4] = {};
#if defined(__aarch64__)
    {
        // same as round_color(): round half away from zero, NaN/<=0 -> 0, >=255 -> 255
        auto rc = [](const float *v) {
            float32x4_t x = vrndaq_f32(vmulq_f32(vld1q_f32(v), vdupq_n_f32(255.0f)));
            x = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vcgtq_f32(x, vdupq_n_f32(0))));
            return vcvtq_u32_f32(vminq_f32(x, vdupq_n_f32(255.0f)));
        };
        uint32x4_t c = vorrq_u32(vorrq_u32(rc(a[0]), vshlq_n_u32(rc(a[1]), 8)),
                                 vorrq_u32(vshlq_n_u32(rc(a[2]), 16), vshlq_n_u32(rc(a[3]), 24)));
        vst1q_u32(prim, c);
    }
#else
    for (u32 p = 0; p < 4; p++)
        prim[p] = rgba(round_color(a[0][p]), round_color(a[1][p]), round_color(a[2][p]),
                       round_color(a[3][p]));
#endif
    static const u8 uvidx[3] = {4, 6, 8};
    static const bool dbg_notex = getenv("GPU_NOTEX"), dbg_nolight = getenv("GPU_NOLIGHT");
    for (u32 i = 0; i < 3; i++) {
        if (!ds.need_tex[i] || dbg_notex) continue;
        const TexUnitState &tu = ds.tex[i];
        const u32 k = uvidx[tu.coord];
        if (tu.simple2d) {
            for (u32 p = 0; p < n; p++) tex[i][p] = sample_simple2d(tu, a[k][p], a[k + 1][p]);
            continue;
        }
        for (u32 p = 0; p < n; p++) {
            const float u = a[k][p], v = a[k + 1][p];
            float lod = 0;
            if (tu.need_lod) {
                const float Wq = Ws[p];
                const float dudx = (t.ax[k] - u * t.qx) * Wq, dvdx = (t.ax[k + 1] - v * t.qx) * Wq;
                const float dudy = (t.ay[k] - u * t.qy) * Wq, dvdy = (t.ay[k + 1] - v * t.qy) * Wq;
                const float fw = (float)tu.w, fh = (float)tu.h;
                const float rx = dudx * dudx * fw * fw + dvdx * dvdx * fh * fh;
                const float ry = dudy * dudy * fw * fw + dvdy * dvdy * fh * fh;
                const float rho2 = rx > ry ? rx : ry;
                if (rho2 < tu.mag_rho2) {
                    lod = -1.0f;  // certainly magnification: only the sign of lod matters
                } else {
                    lod = rho2 > 0 ? 0.5f * log2f(rho2) : -100.0f;
                    lod += tu.bias;
                }
            }
            tex[i][p] = sample_texture(tu, u, v, a[10][p], lod);
        }
    }
    if (ds.need_tex[3]) {
        const u32 k = uvidx[ds.proctex.coord];
        for (u32 p = 0; p < n; p++) tex[3][p] = sample_proctex(ds.proctex, a[k][p], a[k + 1][p]);
    }
    if (ds.lighting && !dbg_nolight) {
        if (n == 1) {
            const float q[4] = {a[12][0], a[13][0], a[14][0], a[15][0]};
            const float vw[3] = {a[16][0], a[17][0], a[18][0]};
            const u32 tx[4] = {tex[0][0], tex[1][0], tex[2][0], tex[3][0]};
            compute_lighting(ds.light, q, vw, tx, fp[0], fs[0]);
        } else {
            compute_lighting4(ds.light, &a[12], &a[16], tex, fp, fs);
        }
    }
    tev_combine4(ds, prim, fp, fs, tex, out);
}

struct Pending {
    u32 *cpx;
    u32 i;
    float dx, dy, W, depth;
    u32 z;
};

// alpha test, fog, late depth/stencil, blending and write for one shaded fragment
inline void finish_fragment(const DrawState &ds, TileState &ts, const Pending &f, u32 c) {
    if (ds.alpha_test && !compare(ds.alpha_func, c >> 24, ds.alpha_ref)) return;
    if (ds.fog) {
        const FogLut &F = *ds.fog_lut;
        const float fi = (ds.fog_flip ? 1.0f - f.depth : f.depth) * 128.0f;
        float fl = floorf(fi);
        fl = fl < 0 ? 0 : fl > 127 ? 127 : fl;
        const u32 ii = (u32)fl;
        float ff = F.val[ii] + F.diff[ii] * (fi - fl);
        ff = ff < 0 ? 0 : ff > 1 ? 1 : ff;
        u32 o = c & 0xff000000u;
        for (u32 ch = 0; ch < 3; ch++) {
            const float cc = (float)((c >> (8 * ch)) & 255);
            const float fc = (float)((ds.fog_color >> (8 * ch)) & 255);
            o |= (u32)(u8)(ff * cc + (1.0f - ff) * fc) << (8 * ch);
        }
        c = o;
    }
    if (!ds.early_z && !depth_stencil(ds, ts, f.i, f.z)) return;
    if (!ds.color_write) return;
    const u32 dst = *f.cpx;
    u32 res = ds.blend ? blend(ds, c, dst) : logic_op(ds.logic_op, c, dst);
    res = (res & ds.color_mask) | (dst & ~ds.color_mask);
    if (res != dst) {
        *f.cpx = res;
        ts.color_dirty = true;
    }
}

void raster_tri_tile(const Tri &t, const TileRect &r, TileState &ts) {
    const DrawState &ds = *t.ds;
    const s32 rx0 = std::max(t.x0, r.x0), rx1 = std::min(t.x1, r.x1);
    const s32 ry0 = std::max(t.y0, r.y0), ry1 = std::min(t.y1, r.y1);
    if (rx0 >= rx1 || ry0 >= ry1) return;
    TileBuf &tb = *ts.tb;
    u64 nfrag = 0, nshade = 0;
    const float zmax = (float)((1u << ds.depth_bits) - 1);
    Pending pend[4];
    u32 np = 0;
    auto flush_pending = [&]() {
        alignas(16) float dxs[4], dys[4], Ws[4];
        for (u32 p = 0; p < 4; p++) {
            const Pending &q = pend[p < np ? p : 0];
            dxs[p] = q.dx;
            dys[p] = q.dy;
            Ws[p] = q.W;
        }
        if (!ds.wbuffer) {
            // perspective divide deferred to here (same expression as the per-pixel one)
            const f4v vdx = *(const f4v *)dxs, vdy = *(const f4v *)dys;
            const f4v vq = t.q0 + t.qx * vdx + t.qy * vdy;
            *(f4v *)Ws = 1.0f / vq;
        }
        u32 c4[4];
        shade4(ds, t, np, dxs, dys, Ws, c4);
        nshade += np;
        for (u32 p = 0; p < np; p++) finish_fragment(ds, ts, pend[p], c4[p]);
        np = 0;
    };
    for (s32 py = ry0; py < ry1; py++) {
        const s64 oy = py - t.y0, ox = rx0 - t.x0;
        s64 e0 = t.e0[0] + t.dex[0] * ox + t.dey[0] * oy;
        s64 e1 = t.e0[1] + t.dex[1] * ox + t.dey[1] * oy;
        s64 e2 = t.e0[2] + t.dex[2] * ox + t.dey[2] * oy;
        u32 *crow = tb.color + (py - r.y0) * TILE - r.x0;
        const u32 ibase = (py - r.y0) * TILE - r.x0;
        // covered span of this row: all k in [kmin, kmax] with e_i + dex_i * k >= 0 for all i
        s64 kmin = 0, kmax = rx1 - rx0 - 1;
        const s64 ev[3] = {e0, e1, e2};
        for (int ei = 0; ei < 3 && kmin <= kmax; ei++) {
            const s64 e = ev[ei], d = t.dex[ei];
            if (d > 0) {
                if (e < 0) kmin = std::max(kmin, (-e + d - 1) / d);
            } else if (d < 0) {
                if (e < 0) kmax = -1;
                else kmax = std::min(kmax, e / (-d));
            } else if (e < 0) {
                kmax = -1;
            }
        }
        if (kmin > kmax) continue;
        for (s32 px = rx0 + (s32)kmin; px <= rx0 + (s32)kmax; px++) {
            if (ds.scissor_mode == 1 && px >= ds.sc_x1 && px < ds.sc_x2 && py >= ds.sc_y1 &&
                py < ds.sc_y2)
                continue;
            const float dx = (float)(px - t.x0), dy = (float)(py - t.y0);
            const double zw = t.z0 + t.zx * dx + t.zy * dy;
            float W = 0;  // only needed here for W-buffering / shadow maps; else computed later
            if (ds.wbuffer || ds.shadow_mode) {
                const float q = t.q0 + t.qx * dx + t.qy * dy;
                W = 1.0f / q;
            }
            float depth = (float)zw * ds.depth_scale + ds.depth_offset;
            if (ds.wbuffer) depth *= W;
            depth = depth < 0 ? 0 : depth > 1 ? 1 : depth;
            if (!(depth == depth)) depth = 0;
            const u32 i = ibase + px;
            nfrag++;
            if (ds.shadow_mode) {
                const u32 c = shade(ds, t, dx, dy, W);
                const u32 dz = (u32)(depth * 16777215.0f);
                const u32 stv = (c >> 8) & 255;
                u32 raw = crow[px];
                const u32 ref_z = ((raw & 255) << 16) | (((raw >> 8) & 255) << 8) | ((raw >> 16) & 255);
                const u32 ref_s = raw >> 24;
                if (dz >= ref_z) continue;
                if (stv == 0) {
                    raw = (raw & 0xff000000u) | ((dz >> 16) & 255) | (((dz >> 8) & 255) << 8) |
                          ((dz & 255) << 16);
                } else {
                    float x = ref_z ? (float)dz / (float)ref_z : 0.0f;
                    float ns = (float)stv / (ds.shadow_const + ds.shadow_linear * x);
                    ns = ns < 0 ? 0 : ns > 255 ? 255 : ns;
                    if (!(ns == ns)) ns = 0;
                    const u32 s8v = (u32)ns;
                    if (s8v < ref_s) raw = (raw & 0xffffff) | (s8v << 24);
                }
                crow[px] = raw;
                ts.color_dirty = true;
                continue;
            }
            const u32 z = (u32)(depth * zmax);
            if (ds.early_z) {
                if (!depth_stencil(ds, ts, i, z)) continue;
                if (!ds.need_shade) continue;
            } else if (ds.depth_test && !ds.stencil && !compare(ds.depth_func, z, tb.depth[i])) {
                continue;  // would fail the depth test anyway (no stencil side effects)
            }
            // queue the fragment; fragments of one triangle never overlap, so deferring is safe
            pend[np++] = Pending{&crow[px], i, dx, dy, W, depth, z};
            if (np == 4) flush_pending();
        }
    }
    if (np) flush_pending();
    if (stats_on) {
        st_frags += nfrag;
        st_shaded += nshade;
    }
}

// Block-wise tile I/O: when the tile maps to whole 8x8 memory blocks, walk each block in memory
// order with a morton table instead of computing tiled offsets per pixel.
struct MortonTable {
    u8 x[64], y[64];
    MortonTable() {
        for (u32 yy = 0; yy < 8; yy++)
            for (u32 xx = 0; xx < 8; xx++) {
                x[morton8(xx, yy)] = (u8)xx;
                y[morton8(xx, yy)] = (u8)yy;
            }
    }
};
const MortonTable mtab;

inline bool tile_blocks_aligned(const RTConfig &rt, const TileRect &r) {
    return (r.x1 - r.x0) % 8 == 0 && (r.y1 - r.y0) % 8 == 0 && ((s32)rt.h - r.y1) % 8 == 0;
}

// visit every pixel of the tile: f(local index, pointer to the pixel in memory)
template <class F>
inline void for_tile_blocks(const RTConfig &rt, const TileRect &r, u8 *base, u32 bpp, F &&f) {
    for (s32 by = r.y0; by < r.y1; by += 8) {
        const u32 bmy = rt.h - (u32)by - 8;  // memory row of the block's first line
        for (s32 bx = r.x0; bx < r.x1; bx += 8) {
            u8 *blk = base + ((size_t)bmy * rt.w + (size_t)bx * 8) * bpp;
            for (u32 i = 0; i < 64; i++) {
                const u32 ly = (u32)(by + 7 - mtab.y[i] - r.y0), lx = (u32)(bx + mtab.x[i] - r.x0);
                f(ly * TILE + lx, blk + i * bpp);
            }
        }
    }
}

void load_tile_fast(const RTConfig &rt, const TileRect &r, TileBuf &tb, bool want_depth) {
    u8 *cbuf = rt.color_addr ? pptr(rt.color_addr) : nullptr;
    u8 *dbuf = rt.depth_addr && want_depth ? pptr(rt.depth_addr) : nullptr;
    const u32 cf = rt.color_fmt;
    if (cbuf) {
        if (cf == 0) {
            for_tile_blocks(rt, r, cbuf, 4, [&](u32 i, const u8 *p) {
                u32 v;
                memcpy(&v, p, 4);
                tb.color[i] = __builtin_bswap32(v);
            });
        } else if (cf & 0x80) {
            for_tile_blocks(rt, r, cbuf, 4, [&](u32 i, const u8 *p) { memcpy(&tb.color[i], p, 4); });
        } else {
            const u32 pf = cbuf_to_pixfmt(cf & 7);
            for_tile_blocks(rt, r, cbuf, color_bytes(cf & 7),
                            [&](u32 i, const u8 *p) { tb.color[i] = decode_pixel(pf, p); });
        }
    }
    if (dbuf) {
        if (rt.depth_fmt == 3) {
            for_tile_blocks(rt, r, dbuf, 4, [&](u32 i, const u8 *p) {
                u32 v;
                memcpy(&v, p, 4);
                tb.depth[i] = v & 0xffffff;
                tb.stencil[i] = (u8)(v >> 24);
            });
        } else if (rt.depth_fmt == 0) {
            for_tile_blocks(rt, r, dbuf, 2, [&](u32 i, const u8 *p) {
                tb.depth[i] = p[0] | (p[1] << 8);
                tb.stencil[i] = 0;
            });
        } else {
            for_tile_blocks(rt, r, dbuf, 3, [&](u32 i, const u8 *p) {
                tb.depth[i] = p[0] | (p[1] << 8) | (p[2] << 16);
                tb.stencil[i] = 0;
            });
        }
    }
}

void store_tile_fast(const RTConfig &rt, const TileRect &r, const TileBuf &tb, bool color, bool depth) {
    u8 *cbuf = rt.color_addr && color ? pptr(rt.color_addr) : nullptr;
    u8 *dbuf = rt.depth_addr && depth ? pptr(rt.depth_addr) : nullptr;
    const u32 cf = rt.color_fmt;
    if (cbuf) {
        if (cf == 0) {
            for_tile_blocks(rt, r, cbuf, 4, [&](u32 i, u8 *p) {
                const u32 v = __builtin_bswap32(tb.color[i]);
                memcpy(p, &v, 4);
            });
        } else if (cf & 0x80) {
            for_tile_blocks(rt, r, cbuf, 4, [&](u32 i, u8 *p) { memcpy(p, &tb.color[i], 4); });
        } else {
            const u32 pf = cbuf_to_pixfmt(cf & 7);
            for_tile_blocks(rt, r, cbuf, color_bytes(cf & 7),
                            [&](u32 i, u8 *p) { encode_pixel(pf, tb.color[i], p); });
        }
    }
    if (dbuf) {
        const u32 df = rt.depth_fmt;
        for_tile_blocks(rt, r, dbuf, depth_bytes(df), [&](u32 i, u8 *p) {
            const u32 d = tb.depth[i];
            if (df == 3) {
                const u32 v = (d & 0xffffff) | (u32)tb.stencil[i] << 24;
                memcpy(p, &v, 4);
            } else {
                p[0] = d;
                p[1] = d >> 8;
                if (df != 0) p[2] = d >> 16;
            }
        });
    }
}

bool batch_uses_depth = true;

// work item = bin * 4 + quadrant: each 32x32 bin is rasterized as four 16x16 pieces so that a
// heavy bin doesn't serialize the whole flush
void process_tile(u32 item, u32 tid) {
    const RTConfig &rt = batch.rt;
    const u32 tile = item >> 2, quad = item & 3;
    const u32 tx = tile % batch.tiles_x, ty = tile / batch.tiles_x;
    constexpr s32 HALF = TILE / 2;
    TileRect r;
    r.x0 = tx * TILE + (quad & 1) * HALF;
    r.y0 = ty * TILE + (quad >> 1) * HALF;
    r.x1 = std::min<s32>(r.x0 + HALF, rt.w);
    r.y1 = std::min<s32>(r.y0 + HALF, rt.h);
    if (r.x0 >= r.x1 || r.y0 >= r.y1) return;
    TileState ts;
    ts.tb = tilebufs[tid].get();
    ts.color_dirty = ts.depth_dirty = false;
    const bool fast = tile_blocks_aligned(rt, r);
    if (fast)
        load_tile_fast(rt, r, *ts.tb, batch_uses_depth);
    else
        load_tile(rt, r, *ts.tb);
    for (u32 ti : batch.bins[tile]) raster_tri_tile(batch.tris[ti], r, ts);
    if (fast)
        store_tile_fast(rt, r, *ts.tb, ts.color_dirty, ts.depth_dirty);
    else
        store_tile(rt, r, *ts.tb, ts.color_dirty, ts.depth_dirty);
}
}  // namespace

void raster_flush() {
    vtx_finish_jobs();
    if (!batch.mb.empty()) {
        const double tf = g_stats ? now_ms() : 0;
        metal_flush(batch.rt, batch.mb);
        batch.mb.clear();
        const RTConfig &rt = batch.rt;
        if (rt.color_addr) texcache_invalidate(rt.color_addr, rt.w * rt.h * color_bytes(rt.color_fmt & 7));
        if (rt.depth_addr) texcache_invalidate(rt.depth_addr, rt.w * rt.h * depth_bytes(rt.depth_fmt));
        if (g_stats) g_t_flush += now_ms() - tf;
    }
    if (!batch.tris.empty()) {
        const double tf = g_stats ? now_ms() : 0;
        batch.active_tiles.clear();
        batch_uses_depth = false;
        for (const auto &s : batch.states)
            if (s->depth_test || s->depth_write || s->stencil) batch_uses_depth = true;
        for (u32 i = 0; i < batch.bins.size(); i++)
            if (!batch.bins[i].empty())
                for (u32 q = 0; q < 4; q++) batch.active_tiles.push_back(i * 4 + q);
        // costliest bins first for better load balance
        std::stable_sort(batch.active_tiles.begin(), batch.active_tiles.end(), [](u32 a, u32 b) {
            return batch.bins[a >> 2].size() > batch.bins[b >> 2].size();
        });
        const std::vector<u32> &tiles = batch.active_tiles;
        if (g_trace && tiles.size() > 16) {
            std::vector<double> tt(tiles.size());
            parallel_for((u32)tiles.size(), [&](u32 i, u32 tid) {
                const double t0 = now_ms();
                process_tile(tiles[i], tid);
                tt[i] = now_ms() - t0;
            });
            double sum = 0, mx = 0;
            for (double v : tt) {
                sum += v;
                mx = std::max(mx, v);
            }
            LOG("gpu trace: %zu tiles, sum %.2f ms, max tile %.2f ms, ideal/9 %.2f ms\n", tiles.size(), sum, mx,
                sum / 9);
        } else {
            parallel_for((u32)tiles.size(), [&](u32 i, u32 tid) { process_tile(tiles[i], tid); });
        }
        const RTConfig &rt = batch.rt;
        if (rt.color_addr) texcache_invalidate(rt.color_addr, rt.w * rt.h * color_bytes(rt.color_fmt & 7));
        if (rt.depth_addr) texcache_invalidate(rt.depth_addr, rt.w * rt.h * depth_bytes(rt.depth_fmt));
        if (g_stats) g_t_flush += now_ms() - tf;
        if (stats_on) {
            st_flushes++;
            st_tris += batch.tris.size();
            LOG("gpu stats: flush %llu: %zu tris, %zu tiles, %llu frags, %llu shaded, rt %08x %ux%u\n",
                (unsigned long long)st_flushes, batch.tris.size(), tiles.size(),
                (unsigned long long)st_frags.load(), (unsigned long long)st_shaded.load(),
                rt.color_addr, rt.w, rt.h);
            st_frags = 0;
            st_shaded = 0;
        }
        batch.tris.clear();
        for (auto &b : batch.bins) b.clear();
    }
    batch.states.clear();
    g.ds = nullptr;
    g.ds_version = 0;
}

}  // namespace gpu
