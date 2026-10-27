// Texture decoding (all PICA formats incl. ETC1/ETC1A4, morton tiling), decoded-texture cache and
// texel sampling (wrap modes, border, nearest/bilinear, mipmaps, cube maps, shadow textures).
#include <unordered_map>

#include "gpu/pica.h"

namespace gpu {

static const u32 fmt_bits[16] = {32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8, 0, 0};

u32 texture_level_bytes(u32 w, u32 h, u32 fmt) { return w * h * fmt_bits[fmt & 15] / 8; }

static inline u32 c4(u32 v) { return (v << 4) | v; }
static inline u32 c5(u32 v) { return (v << 3) | (v >> 2); }
static inline u32 c6(u32 v) { return (v << 2) | (v >> 4); }

u32 decode_etc1_texel(u64 blk, u32 x, u32 y) {
    static const int mod[8][2] = {{2, 8},   {5, 17},  {9, 29},   {13, 42},
                                  {18, 60}, {24, 80}, {33, 106}, {47, 183}};
    const u32 texel = 4 * x + y;
    const bool flip = (blk >> 32) & 1, diff = (blk >> 33) & 1;
    if (flip) std::swap(x, y);
    int r, gg, b;
    if (diff) {
        r = (int)((blk >> 59) & 31);
        gg = (int)((blk >> 51) & 31);
        b = (int)((blk >> 43) & 31);
        if (x >= 2) {
            r += sbits((u32)(blk >> 32), 56 - 32, 3);
            gg += sbits((u32)(blk >> 32), 48 - 32, 3);
            b += sbits((u32)(blk >> 32), 40 - 32, 3);
        }
        r = c5(r & 31);
        gg = c5(gg & 31);
        b = c5(b & 31);
    } else if (x < 2) {
        r = c4((blk >> 60) & 15);
        gg = c4((blk >> 52) & 15);
        b = c4((blk >> 44) & 15);
    } else {
        r = c4((blk >> 56) & 15);
        gg = c4((blk >> 48) & 15);
        b = c4((blk >> 40) & 15);
    }
    const u32 table = x < 2 ? (blk >> 37) & 7 : (blk >> 34) & 7;
    int m = mod[table][(blk >> texel) & 1];
    if ((blk >> (16 + texel)) & 1) m = -m;
    auto cl = [](int v) { return (u32)(v < 0 ? 0 : v > 255 ? 255 : v); };
    return rgba(cl(r + m), cl(gg + m), cl(b + m), 255);
}

static inline u32 decode_texel(const u8 *tile, u32 x, u32 y, u32 fmt) {
    const u32 m = morton8(x, y);
    switch (fmt) {
    case 0: { const u8 *p = tile + m * 4; return rgba(p[3], p[2], p[1], p[0]); }
    case 1: { const u8 *p = tile + m * 3; return rgba(p[2], p[1], p[0], 255); }
    case 2: {
        u32 v = tile[m * 2] | (tile[m * 2 + 1] << 8);
        return rgba(c5(v >> 11), c5((v >> 6) & 31), c5((v >> 1) & 31), (v & 1) * 255);
    }
    case 3: {
        u32 v = tile[m * 2] | (tile[m * 2 + 1] << 8);
        return rgba(c5(v >> 11), c6((v >> 5) & 63), c5(v & 31), 255);
    }
    case 4: {
        u32 v = tile[m * 2] | (tile[m * 2 + 1] << 8);
        return rgba(c4(v >> 12), c4((v >> 8) & 15), c4((v >> 4) & 15), c4(v & 15));
    }
    case 5: { const u8 *p = tile + m * 2; return rgba(p[1], p[1], p[1], p[0]); }
    case 6: { const u8 *p = tile + m * 2; return rgba(p[1], p[0], 0, 255); }
    case 7: { u32 i = tile[m]; return rgba(i, i, i, 255); }
    case 8: return rgba(0, 0, 0, tile[m]);
    case 9: { u32 i = c4(tile[m] >> 4), a = c4(tile[m] & 15); return rgba(i, i, i, a); }
    case 10: {
        u32 b = tile[m / 2];
        u32 i = c4((m & 1) ? b >> 4 : b & 15);
        return rgba(i, i, i, 255);
    }
    case 11: {
        u32 b = tile[m / 2];
        return rgba(0, 0, 0, c4((m & 1) ? b >> 4 : b & 15));
    }
    case 12:
    case 13: {
        const bool alpha = fmt == 13;
        const u32 sub = (x / 4) + 2 * (y / 4);
        const u8 *sp = tile + sub * (alpha ? 16 : 8);
        x &= 3;
        y &= 3;
        u32 a = 255;
        if (alpha) {
            u64 pa;
            memcpy(&pa, sp, 8);
            a = c4((pa >> (4 * (x * 4 + y))) & 15);
            sp += 8;
        }
        u64 blk;
        memcpy(&blk, sp, 8);
        return (decode_etc1_texel(blk, x, y) & 0xffffff) | (a << 24);
    }
    default: return 0;
    }
}

void texture_decode(const u8 *src, u32 w, u32 h, u32 fmt, u32 *out) {
    const u32 tile_bytes = 64 * fmt_bits[fmt & 15] / 8;
    const u32 tw = w / 8, th = h / 8;
    for (u32 ty = 0; ty < th; ty++) {
        for (u32 tx = 0; tx < tw; tx++) {
            const u8 *tile = src + (ty * tw + tx) * tile_bytes;
            for (u32 y = 0; y < 8; y++) {
                u32 *row = out + (h - 1 - (ty * 8 + y)) * w + tx * 8;
                for (u32 x = 0; x < 8; x++) row[x] = decode_texel(tile, x, y, fmt);
            }
        }
    }
}

// ---------------------------------------------------------------- cache
namespace {
std::unordered_map<u64, std::unique_ptr<TexEntry>> cache;
u64 epoch = 1;
size_t total_bytes = 0;
}  // namespace

static bool trust_invalidations = false;
static u64 use_counter = 1;

void set_trust_invalidations(bool trust) { trust_invalidations = trust; }

void invalidate_memory(u32 paddr, u32 size) {
    ensure_init();
    raster_flush();
    metal_poll();
    texcache_invalidate(paddr, size);
    metal_guest_written(paddr, size, false);  // CPU cache maintenance: maybe written
}

void texcache_new_epoch() {
    if (g_trace) LOG("gpu trace: texture cache %zu entries, %zu KiB decoded\n", cache.size(), total_bytes >> 10);
    use_counter++;
    if (!trust_invalidations) epoch++;
    texcache_trim();
}

void texcache_trim() {
    if (total_bytes < (256u << 20)) return;
    for (auto it = cache.begin(); it != cache.end();) {
        if (it->second->last_use + 2 < use_counter) {
            for (auto &l : it->second->lv) total_bytes -= l.data.size() * 4;
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void texcache_invalidate(u32 paddr, u32 size) {
    for (auto &kv : cache) {
        TexEntry &e = *kv.second;
        if (e.addr < paddr + size && paddr < e.addr + e.size) e.stale = true;
    }
}

const TexEntry *texcache_get(u32 addr, u32 w, u32 h, u32 fmt, u32 levels) {
    if (w < 8 || h < 8 || fmt > 13) return nullptr;
    u64 key = ((u64)addr << 32) ^ ((u64)w << 20) ^ ((u64)h << 8) ^ (fmt << 4) ^ levels;
    auto &slot = cache[key];
    if (!slot) {
        slot = std::make_unique<TexEntry>();
        slot->addr = addr;
        slot->fmt = fmt;
        slot->w = w;
        slot->h = h;
        slot->levels = levels;
        u32 size = 0;
        for (u32 l = 0; l < levels; l++) size += texture_level_bytes(w >> l, h >> l, fmt);
        slot->size = size;
    }
    TexEntry &e = *slot;
    e.last_use = use_counter;
    if (e.epoch == epoch && !e.stale) return &e;
    const u8 *p = pptr(addr);
    if (!p || !pptr(addr + e.size - 1)) return nullptr;
    u64 hv = hash_bytes(p, e.size);
    if (hv != e.hash || e.lv.empty()) {
        for (auto &l : e.lv) total_bytes -= l.data.size() * 4;
        e.lv.resize(levels);
        u32 off = 0;
        for (u32 l = 0; l < levels; l++) {
            TexImage &im = e.lv[l];
            im.w = w >> l;
            im.h = h >> l;
            im.data.resize(im.w * im.h);
            if (im.w >= 8 && im.h >= 8) texture_decode(p + off, im.w, im.h, fmt, im.data.data());
            off += texture_level_bytes(im.w, im.h, fmt);
            total_bytes += im.data.size() * 4;
        }
        e.hash = hv;
    }
    e.epoch = epoch;
    e.stale = false;
    return &e;
}

// ---------------------------------------------------------------- sampling
static inline s32 wrap(u32 mode, s32 v, s32 size) {
    switch (mode) {
    case 4:
        if (v < 0) return (s32)((u32)v % (u32)size);
        [[fallthrough]];
    case 0:
    case 1:
        return v < 0 ? 0 : v >= size ? size - 1 : v;
    case 3: {
        u32 c = (u32)v % (2 * (u32)size);
        if (c >= (u32)size) c = 2 * size - 1 - c;
        return (s32)c;
    }
    default:  // 2, 5 (negative side), 6, 7
        return (s32)((u32)v % (u32)size);
    }
}
static inline bool is_border(u32 mode, s32 v, s32 size) {
    if (mode == 1) return v < 0 || v >= size;
    if (mode == 5) return v >= size;
    return false;
}

static inline u32 fetch(const TexUnitState &tu, const TexImage &im, s32 s, s32 t) {
    if (is_border(tu.wrap_s, s, im.w) || is_border(tu.wrap_t, t, im.h)) return tu.border;
    s = wrap(tu.wrap_s, s, im.w);
    t = wrap(tu.wrap_t, t, im.h);
    return im.data[t * im.w + s];
}

static inline u32 lerp4(u32 c00, u32 c10, u32 c01, u32 c11, u32 fx, u32 fy) {
    const u32 w00 = (256 - fx) * (256 - fy), w10 = fx * (256 - fy), w01 = (256 - fx) * fy,
              w11 = fx * fy;
    u32 out = 0;
    for (u32 sh = 0; sh < 32; sh += 8) {
        u32 v = ((c00 >> sh) & 255) * w00 + ((c10 >> sh) & 255) * w10 + ((c01 >> sh) & 255) * w01 +
                ((c11 >> sh) & 255) * w11;
        out |= ((v + 32768) >> 16) << sh;
    }
    return out;
}

static inline u32 lerp2(u32 a, u32 b, u32 f) {
    const u32 nf = 256 - f;
    const u32 rb = (((a & 0xff00ff) * nf + (b & 0xff00ff) * f) >> 8) & 0xff00ff;
    const u32 ag = (((a >> 8) & 0xff00ff) * nf + ((b >> 8) & 0xff00ff) * f) & 0xff00ff00u;
    return rb | ag;
}

static inline s32 to_int_clamped(float x) {
    x = x < -1e9f ? -1e9f : x > 1e9f ? 1e9f : x;
    return x == x ? (s32)x : 0;
}

static inline u32 sample_level_fast(const TexUnitState &tu, const TexImage &im, float u, float v,
                                    bool linear) {
    const s32 W = (s32)im.w, H = (s32)im.h;
    const bool rs = tu.wrap_s != 0, rt = tu.wrap_t != 0;
    auto ws = [&](s32 s) { return rs ? (s & (W - 1)) : (s < 0 ? 0 : s >= W ? W - 1 : s); };
    auto wt = [&](s32 t) { return rt ? (t & (H - 1)) : (t < 0 ? 0 : t >= H ? H - 1 : t); };
    const u32 *d = im.data.data();
    if (!linear) {
        const s32 s = ws(to_int_clamped(floorf(u * W))), t = wt(to_int_clamped(floorf(v * H)));
        return d[t * W + s];
    }
    const float x = u * W - 0.5f, y = v * H - 0.5f;
    const float x0 = floorf(x), y0 = floorf(y);
    const u32 fx = (u32)((x - x0) * 256.0f), fy = (u32)((y - y0) * 256.0f);
    const s32 xi = to_int_clamped(x0), yi = to_int_clamped(y0);
    const s32 s0 = ws(xi), s1 = ws(xi + 1), t0 = wt(yi), t1 = wt(yi + 1);
    const u32 top = lerp2(d[t0 * W + s0], d[t0 * W + s1], fx > 256 ? 256 : fx);
    const u32 bot = lerp2(d[t1 * W + s0], d[t1 * W + s1], fx > 256 ? 256 : fx);
    return lerp2(top, bot, fy > 256 ? 256 : fy);
}

static inline u32 sample_level(const TexUnitState &tu, const TexImage &im, float u, float v,
                               bool linear) {
    if (im.data.empty()) return 0;
    if (tu.fast_wrap) return sample_level_fast(tu, im, u, v, linear);
    if (!linear) {
        float fs = floorf(u * im.w), ft = floorf(v * im.h);
        s32 s = fs < -1e9f ? -1000000000 : fs > 1e9f ? 1000000000 : (s32)fs;
        s32 t = ft < -1e9f ? -1000000000 : ft > 1e9f ? 1000000000 : (s32)ft;
        return fetch(tu, im, s, t);
    }
    float x = u * im.w - 0.5f, y = v * im.h - 0.5f;
    if (!(x > -1e9f && x < 1e9f)) x = 0;
    if (!(y > -1e9f && y < 1e9f)) y = 0;
    float x0 = floorf(x), y0 = floorf(y);
    u32 fx = (u32)((x - x0) * 256.0f), fy = (u32)((y - y0) * 256.0f);
    if (fx > 255) fx = 255;
    if (fy > 255) fy = 255;
    s32 s = (s32)x0, t = (s32)y0;
    return lerp4(fetch(tu, im, s, t), fetch(tu, im, s + 1, t), fetch(tu, im, s, t + 1),
                 fetch(tu, im, s + 1, t + 1), fx, fy);
}

static u32 sample_2d(const TexUnitState &tu, const TexEntry *e, float u, float v, float lod) {
    const s32 nlv = (s32)e->lv.size();
    if (!tu.need_lod || lod <= 0.0f) {
        s32 l = tu.min_level < nlv ? tu.min_level : nlv - 1;
        if (l < 0) l = 0;
        return sample_level(tu, e->lv[l], u, v, tu.need_lod ? tu.mag_linear : tu.mag_linear);
    }
    float lmin = (float)tu.min_level, lmax = (float)std::min(tu.max_level, nlv - 1);
    if (lmax < lmin) lmax = lmin;
    float lc = lod < lmin ? lmin : lod > lmax ? lmax : lod;
    if (!tu.mip_linear || nlv == 1) {
        s32 l = (s32)floorf(lc + 0.5f);
        if (l >= nlv) l = nlv - 1;
        return sample_level(tu, e->lv[l], u, v, tu.min_linear);
    }
    s32 l0 = (s32)floorf(lc);
    if (l0 >= nlv) l0 = nlv - 1;
    s32 l1 = l0 + 1 < nlv && (float)(l0 + 1) <= lmax ? l0 + 1 : l0;
    u32 c0 = sample_level(tu, e->lv[l0], u, v, tu.min_linear);
    if (l1 == l0) return c0;
    u32 c1 = sample_level(tu, e->lv[l1], u, v, tu.min_linear);
    u32 f = (u32)((lc - l0) * 256.0f);
    if (f > 256) f = 256;
    u32 out = 0;
    for (u32 sh = 0; sh < 32; sh += 8)
        out |= ((((c0 >> sh) & 255) * (256 - f) + ((c1 >> sh) & 255) * f + 128) >> 8) << sh;
    return out;
}

static inline u32 shadow_compare(const TexUnitState &tu, u32 c, float z) {
    if (z > 1.0f) z = 1.0f;
    s32 zi = (s32)(z * 16777215.0f) - (s32)(tu.shadow_bias << 1);
    s32 zref = (s32)(((c >> 24) << 16) | (((c >> 16) & 255) << 8) | ((c >> 8) & 255));
    u32 d = zref >= zi ? (c & 255) : 0;
    return d * 0x01010101u;
}

// equivalent to sample_texture() for units with simple2d set
u32 sample_simple2d(const TexUnitState &tu, float u, float v) {
    const TexEntry *e = tu.face[0];
    const s32 nlv = (s32)e->lv.size();
    s32 l = tu.min_level < nlv ? tu.min_level : nlv - 1;
    if (l < 0) l = 0;
    const TexImage &im = e->lv[l];
    if (im.data.empty()) return 0;
    return sample_level_fast(tu, im, u, v, tu.mag_linear);
}

u32 sample_texture(const TexUnitState &tu, float u, float v, float w, float lod) {
    if (tu.null_tex) return rgba(0, 0, 0, 255);
    switch (tu.type) {
    case 1:
    case 4: {
        // cube map
        float au = fabsf(u), av = fabsf(v), aw = fabsf(w);
        float x, y, z;
        int face;
        if (au > av && au > aw) {
            face = u > 0 ? 0 : 1;
            y = u > 0 ? -v : v;
            x = -w;
            z = u;
        } else if (av > aw) {
            face = v > 0 ? 2 : 3;
            x = v > 0 ? u : -u;
            y = w;
            z = v;
        } else {
            face = w > 0 ? 4 : 5;
            y = w > 0 ? -v : v;
            x = u;
            z = w;
        }
        const TexEntry *e = tu.face[face];
        if (!e) return 0;
        float cu = x / z * 0.5f + 0.5f, cv = y / z * 0.5f + 0.5f;
        if (tu.type == 4) {
            u32 c = sample_level(tu, e->lv[0], cu, cv, false);
            return shadow_compare(tu, c, fabsf(z));
        }
        return sample_2d(tu, e, cu, cv, lod);
    }
    case 2: {
        const TexEntry *e = tu.face[0];
        if (!e) return 0;
        if (!tu.shadow_ortho) {
            u /= w;
            v /= w;
        }
        u32 c = sample_level(tu, e->lv[0], u, v, false);
        return shadow_compare(tu, c, fabsf(w));
    }
    case 3: {
        const TexEntry *e = tu.face[0];
        if (!e) return 0;
        return sample_2d(tu, e, u / w, v / w, lod);
    }
    default: {
        const TexEntry *e = tu.face[0];
        if (!e) return 0;
        return sample_2d(tu, e, u, v, lod);
    }
    }
}

}  // namespace gpu
