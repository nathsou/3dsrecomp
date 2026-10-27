// Per-fragment shading helpers: fragment lighting, texture combiners (TEV), procedural texture.
#include "gpu/pica.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace gpu {

// ---------------------------------------------------------------- fragment lighting
// The lighting math is written once as a template over the value type: `float` for one fragment
// and a 4-lane vector for four fragments. Both instantiations evaluate the same expressions, so
// the vector path is bit-identical to the scalar one. LUT lookups are done per lane.
namespace {
typedef float lf4 __attribute__((ext_vector_type(4)));
typedef s32 li4 __attribute__((ext_vector_type(4)));

template <class T>
struct V3T {
    T x, y, z;
};
template <class T>
inline V3T<T> operator+(V3T<T> a, V3T<T> b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
template <class T>
inline V3T<T> operator-(V3T<T> a, V3T<T> b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
template <class T>
inline V3T<T> operator*(V3T<T> a, T s) { return {a.x * s, a.y * s, a.z * s}; }
template <class T>
inline T dot(V3T<T> a, V3T<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
template <class T>
inline V3T<T> cross(V3T<T> a, V3T<T> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

// per-type helpers
inline float tsqrt(float v) { return sqrtf(v); }
#if defined(__aarch64__)
inline lf4 tsqrt(lf4 v) { return (lf4)vsqrtq_f32((float32x4_t)v); }  // IEEE, same as sqrtf
#else
inline lf4 tsqrt(lf4 v) { return (lf4){sqrtf(v[0]), sqrtf(v[1]), sqrtf(v[2]), sqrtf(v[3])}; }
#endif
inline float tsel(bool c, float a, float b) { return c ? a : b; }
inline lf4 tsel(li4 c, lf4 a, lf4 b) { return (lf4)((c & (li4)a) | (~c & (li4)b)); }
template <class T>
inline T K(float v) { return (T)v; }
template <class T>
constexpr int lanes() { return sizeof(T) / sizeof(float); }
template <class T>
inline float &lane(T &v, int i) { return ((float *)&v)[i]; }
template <class T>
inline float lane(const T &v, int i) { return ((const float *)&v)[i]; }

template <class T>
inline V3T<T> normalized(V3T<T> a) {
    T l = tsqrt(dot(a, a));
    const V3T<T> n = a * (K<T>(1.0f) / l);
    const auto pos = l > 0;
    return {tsel(pos, n.x, a.x), tsel(pos, n.y, a.y), tsel(pos, n.z, a.z)};
}
inline float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

inline float lut_lookup(const LightLuts &L, u32 sampler, u32 idx, float delta) {
    return L.val[sampler][idx] + L.diff[sampler][idx] * delta;
}

// texel(unit, lane) -> packed RGBA
template <class T, class TexFn>
void lighting_impl(const LightingState &ls, const T quat_in[4], const T view_in[3], TexFn texel,
                   u32 *primary, u32 *secondary) {
    constexpr int NL = lanes<T>();
    const LightLuts &L = *ls.luts;
    T shadow[4] = {K<T>(1), K<T>(1), K<T>(1), K<T>(1)};
    if (ls.shadow_enable) {
        for (int i = 0; i < 4; i++) {
            for (int l = 0; l < NL; l++) {
                const u32 c = texel(ls.shadow_sel & 3, l);
                float s = ((c >> (8 * i)) & 255) / 255.0f;
                if (ls.shadow_invert) s = 1.0f - s;
                lane(shadow[i], l) = s;
            }
        }
    }
    V3T<T> sn = {K<T>(0), K<T>(0), K<T>(1)}, st = {K<T>(1), K<T>(0), K<T>(0)};
    if (ls.bump_mode == 1 || ls.bump_mode == 2) {
        V3T<T> p;
        for (int l = 0; l < NL; l++) {
            const u32 c = texel(ls.bump_sel & 3, l);
            lane(p.x, l) = (c & 255) / 127.5f - 1.0f;
            lane(p.y, l) = ((c >> 8) & 255) / 127.5f - 1.0f;
            lane(p.z, l) = ((c >> 16) & 255) / 127.5f - 1.0f;
        }
        if (ls.bump_mode == 1) {
            if (ls.bump_renorm) {
                T zz = 1.0f - (p.x * p.x + p.y * p.y);
                p.z = tsqrt(tsel(zz > 0, zz, K<T>(0)));
            }
            sn = p;
        } else {
            st = p;
        }
    }
    // normalize quaternion and rotate the surface-local frame into eye space
    T ql = tsqrt(quat_in[0] * quat_in[0] + quat_in[1] * quat_in[1] + quat_in[2] * quat_in[2] +
                 quat_in[3] * quat_in[3]);
    T qi = tsel(ql > 0, K<T>(1.0f) / ql, K<T>(0.0f));
    V3T<T> qv = {quat_in[0] * qi, quat_in[1] * qi, quat_in[2] * qi};
    T qw = quat_in[3] * qi;
    auto qrot = [&](V3T<T> v) { return v + cross(qv, cross(qv, v) + v * qw) * K<T>(2.0f); };
    const V3T<T> normal = qrot(sn);
    const V3T<T> tangent = qrot(st);
    const V3T<T> view = {view_in[0], view_in[1], view_in[2]};
    const V3T<T> norm_view = normalized(view);

    T dsum[4] = {K<T>(0), K<T>(0), K<T>(0), K<T>(1)}, ssum[4] = {K<T>(0), K<T>(0), K<T>(0), K<T>(1)};
    for (u32 li = 0; li < ls.num_lights; li++) {
        const LightState &lt = ls.lights[li];
        V3T<T> lv = {K<T>(lt.pos[0]), K<T>(lt.pos[1]), K<T>(lt.pos[2])};
        if (!lt.directional) lv = lv + view;
        const T length = tsqrt(dot(lv, lv));
        {
            const V3T<T> n = lv * (K<T>(1.0f) / length);
            const auto pos = length > 0;
            lv = {tsel(pos, n.x, lv.x), tsel(pos, n.y, lv.y), tsel(pos, n.z, lv.z)};
        }
        const V3T<T> half = norm_view + lv;

        T dist_atten = K<T>(1.0f);
        if (lt.dist_atten) {
            const T locv = lt.da_scale * length + lt.da_bias;
            for (int l = 0; l < NL; l++) {
                float loc = clamp01(lane(locv, l));
                float fl = floorf(loc * 256.0f);
                u32 idx = fl < 0 ? 0 : fl > 255 ? 255 : (u32)fl;
                lane(dist_atten, l) = lut_lookup(L, 16 + lt.num, idx, loc * 256.0f - idx);
            }
        }
        // LUT inputs, each computed once per light
        T in[6];
        {
            const V3T<T> nh = normalized(half);
            in[0] = dot(normal, nh);
            in[1] = dot(norm_view, nh);
            in[2] = dot(normal, norm_view);
            in[3] = dot(lv, normal);
            in[4] = dot(lv, V3T<T>{K<T>(lt.spot[0]), K<T>(lt.spot[1]), K<T>(lt.spot[2])});
            in[5] = K<T>(0);
            if (ls.config == 8) {
                const V3T<T> proj = nh - normal * in[0];
                in[5] = dot(proj, tangent);
            }
        }
        auto lut_value = [&](const LutSel &sel, u32 sampler) {
            T out;
            for (int l = 0; l < NL; l++) {
                float r = sel.input < 6 ? lane(in[sel.input], l) : 0.0f;
                u32 idx;
                float delta;
                if (!(r == r)) r = 0;
                if (sel.abs) {
                    r = lt.two_sided ? fabsf(r) : (r > 0 ? r : 0);
                    float fl = floorf(r * 256.0f);
                    idx = fl < 0 ? 0 : fl > 255 ? 255 : (u32)fl;
                    delta = r * 256.0f - idx;
                } else {
                    float fl = floorf(r * 128.0f);
                    s32 si = fl < -128 ? -128 : fl > 127 ? 127 : (s32)fl;
                    delta = r * 128.0f - si;
                    idx = (u32)(u8)(s8)si;
                }
                lane(out, l) = sel.scale * lut_lookup(L, sampler, idx, delta);
            }
            return out;
        };

        T spot_atten = K<T>(1.0f);
        if (lt.spot_atten && ls.sp.enabled) spot_atten = lut_value(ls.sp, 8 + lt.num);
        T d0 = ls.d0.enabled ? lut_value(ls.d0, 0) : K<T>(1.0f);
        T refl[3];
        refl[0] = ls.rr.enabled ? lut_value(ls.rr, 6) : K<T>(1.0f);
        refl[1] = ls.rg.enabled ? lut_value(ls.rg, 5) : refl[0];
        refl[2] = ls.rb.enabled ? lut_value(ls.rb, 4) : refl[0];
        T d1 = ls.d1.enabled ? lut_value(ls.d1, 1) : K<T>(1.0f);
        if (li == ls.num_lights - 1 && ls.fr.enabled) {
            T f = lut_value(ls.fr, 3);
            if (ls.primary_alpha) dsum[3] = f;
            if (ls.secondary_alpha) ssum[3] = f;
        }
        T dp = in[3];
        for (int l = 0; l < NL; l++) {
            float v = lane(dp, l);
            v = lt.two_sided ? fabsf(v) : (v > 0 ? v : 0);
            if (!(v == v)) v = 0;
            lane(dp, l) = v;
        }
        T clamp_h = K<T>(1.0f);
        if (ls.clamp_highlights) clamp_h = tsel(dp == 0.0f, K<T>(0.0f), K<T>(1.0f));
        T geo = K<T>(1.0f);
        if (lt.geo0 || lt.geo1) {
            T h2 = dot(half, half);
            const T q = dp / h2;
            geo = tsel(h2 == 0.0f, K<T>(0.0f), tsel(K<T>(1.0f) < q, K<T>(1.0f), q));
        }
        const T g0 = lt.geo0 ? geo : K<T>(1.0f), g1 = lt.geo1 ? geo : K<T>(1.0f);
        const bool shp = ls.shadow_primary && lt.shadow;
        const bool shs = ls.shadow_secondary && lt.shadow;
        const T att = dist_atten * spot_atten;
        for (int c = 0; c < 3; c++) {
            T s0 = d0 * lt.spec0[c] * g0;
            T s1 = d1 * refl[c] * lt.spec1[c] * g1;
            T diff = (lt.diffuse[c] * dp * (shp ? shadow[c] : K<T>(1.0f)) + lt.ambient[c]) * att;
            T spec = (s0 + s1) * clamp_h * att * (shs ? shadow[c] : K<T>(1.0f));
            dsum[c] += diff;
            ssum[c] += spec;
        }
    }
    if (ls.shadow_alpha) {
        if (ls.primary_alpha) dsum[3] *= shadow[3];
        if (ls.secondary_alpha) ssum[3] *= shadow[3];
    }
    for (int c = 0; c < 3; c++) dsum[c] += ls.global_ambient[c];
    auto pack = [](const T *v, int l) {
        u32 o = 0;
        for (int i = 0; i < 4; i++) {
            float x = lane(v[i], l);
            if (!(x == x)) x = 0;
            o |= (u32)(clamp01(x) * 255.0f) << (8 * i);
        }
        return o;
    };
    for (int l = 0; l < NL; l++) {
        primary[l] = pack(dsum, l);
        secondary[l] = pack(ssum, l);
    }
}
}  // namespace

void compute_lighting(const LightingState &ls, const float quat_in[4], const float view_in[3],
                      const u32 tex[4], u32 &primary, u32 &secondary) {
    lighting_impl<float>(ls, quat_in, view_in, [&](u32 unit, int) { return tex[unit]; }, &primary,
                         &secondary);
}

void compute_lighting4(const LightingState &ls, const float (*quat)[4], const float (*view)[4],
                       const u32 (*tex)[4], u32 primary[4], u32 secondary[4]) {
    lf4 q[4], v[3];
    for (int i = 0; i < 4; i++) q[i] = *(const lf4 *)quat[i];
    for (int i = 0; i < 3; i++) v[i] = *(const lf4 *)view[i];
    lighting_impl<lf4>(ls, q, v, [&](u32 unit, int l) { return tex[unit][l]; }, primary, secondary);
}

// ---------------------------------------------------------------- texture combiners
#if !defined(__aarch64__)
namespace {
struct C3 {
    s32 r, g, b;
};
inline u8 ch(u32 c, u32 i) { return (c >> (8 * i)) & 255; }

inline C3 color_mod(u32 m, u32 v) {
    s32 r = ch(v, 0), gg = ch(v, 1), b = ch(v, 2), a = ch(v, 3);
    switch (m) {
    case 0: return {r, gg, b};
    case 1: return {255 - r, 255 - gg, 255 - b};
    case 2: return {a, a, a};
    case 3: return {255 - a, 255 - a, 255 - a};
    case 4: return {r, r, r};
    case 5: return {255 - r, 255 - r, 255 - r};
    case 8: return {gg, gg, gg};
    case 9: return {255 - gg, 255 - gg, 255 - gg};
    case 12: return {b, b, b};
    case 13: return {255 - b, 255 - b, 255 - b};
    default: return {r, gg, b};
    }
}
inline s32 alpha_mod(u32 m, u32 v) {
    switch (m) {
    case 0: return ch(v, 3);
    case 1: return 255 - ch(v, 3);
    case 2: return ch(v, 0);
    case 3: return 255 - ch(v, 0);
    case 4: return ch(v, 1);
    case 5: return 255 - ch(v, 1);
    case 6: return ch(v, 2);
    default: return 255 - ch(v, 2);
    }
}
inline s32 cl255(s32 v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

inline s32 combine1(u32 op, s32 a, s32 b, s32 c) {
    switch (op) {
    case 0: return a;
    case 1: return a * b / 255;
    case 2: return std::min(255, a + b);
    case 3: return cl255(a + b - 128);
    case 4: return (a * c + b * (255 - c)) / 255;
    case 5: return std::max(0, a - b);
    case 8: return std::min(255, (a * b + 255 * c) / 255);
    case 9: return std::min(255, a + b) * c / 255;
    default: return a;
    }
}
}  // namespace
#endif

void tev_finish_prepare(DrawState &ds);

void tev_prepare(DrawState &ds) {
    ds.tev_n = 0;
    for (u32 si = 0; si < 6; si++) {
        TevStageState &t = ds.tev[si];
        const bool identity = t.cop == 0 && t.aop == 0 && t.csrc[0] == 15 && t.cmod[0] == 0 &&
                              t.asrc[0] == 15 && t.amod[0] == 0 && t.cmul == 1 && t.amul == 1;
        if (!identity) ds.tev_n = si + 1;
        memset(t.idx, 0, sizeof(t.idx));
        memset(t.inv, 0, sizeof(t.inv));
        for (u32 k = 0; k < 3; k++) {
            u32 sel;
            switch (t.cmod[k] & ~1u) {
            case 2: sel = 0x333; break;
            case 4: sel = 0x000; break;
            case 8: sel = 0x111; break;
            case 12: sel = 0x222; break;
            default: sel = 0x210; break;
            }
            static const u8 asel[4] = {3, 0, 1, 2};
            for (u32 c = 0; c < 3; c++) t.idx[4 * k + c] = (u8)(t.csrc[k] * 4 + ((sel >> (4 * c)) & 15));
            t.idx[4 * k + 3] = (u8)(t.asrc[k] * 4 + asel[(t.amod[k] >> 1) & 3]);
            const bool cinv = (t.cmod[k] & 1) && (t.cmod[k] <= 5 || (t.cmod[k] & ~1u) == 8 ||
                                                 (t.cmod[k] & ~1u) == 12);
            for (u32 c = 0; c < 3; c++) t.inv[4 * k + c] = cinv ? 0xff : 0;
            t.inv[4 * k + 3] = (t.amod[k] & 1) ? 0xff : 0;
        }
        for (u32 i = 12; i < 16; i++) t.idx[i] = 0xff;  // out of range -> 0
        // 4-pixel tables: byte (p*4 + c) of the source vector; t.idx holds src*4 + byte
        for (u32 k = 0; k < 3; k++) {
            t.q_same[k] = t.csrc[k] == t.asrc[k];
            for (u32 p = 0; p < 4; p++) {
                for (u32 c = 0; c < 4; c++) {
                    const u8 sel = (u8)(p * 4 + (t.idx[4 * k + c] & 3));
                    t.q_idxc[k][p * 4 + c] = c < 3 ? sel : 0xff;
                    t.q_idxa[k][p * 4 + c] = c == 3 ? sel : 0xff;
                    t.q_idx[k][p * 4 + c] = sel;
                    t.q_inv[k][p * 4 + c] = t.inv[4 * k + c];
                }
            }
        }
    }
    tev_finish_prepare(ds);
}

#if defined(__aarch64__)
static inline uint16x8_t vdiv255(uint16x8_t x) {
    return vshrq_n_u16(vaddq_u16(vaddq_u16(x, vdupq_n_u16(1)), vshrq_n_u16(x, 8)), 8);
}

// one combiner op on 4 lanes (u8x8, lanes 0..3 used)
static inline uint8x8_t tev_op(u32 op, uint8x8_t a, uint8x8_t b, uint8x8_t c) {
    switch (op) {
    case 0: return a;
    case 1: return vmovn_u16(vdiv255(vmull_u8(a, b)));
    case 2: return vqadd_u8(a, b);
    case 3: {
        int16x8_t s = vreinterpretq_s16_u16(vaddl_u8(a, b));
        s = vsubq_s16(s, vdupq_n_s16(128));
        return vqmovun_s16(s);
    }
    case 4: {
        uint16x8_t x = vmull_u8(a, c);
        x = vmlal_u8(x, b, vmvn_u8(c));
        return vmovn_u16(vdiv255(x));
    }
    case 5: return vqsub_u8(a, b);
    case 8: return vqadd_u8(vmovn_u16(vdiv255(vmull_u8(a, b))), c);
    case 9: return vmovn_u16(vdiv255(vmull_u8(vqadd_u8(a, b), c)));
    default: return a;
    }
}

struct TevRun {
    uint8x16x4_t T;
    uint32x4_t t3;
    u32 out, buf, next_buf;
};

static inline void tev_stage(const TevStageState &t, TevRun &st) {
    st.t3 = vsetq_lane_u32(st.buf, st.t3, 1);
    st.t3 = vsetq_lane_u32(t.konst, st.t3, 2);
    st.t3 = vsetq_lane_u32(st.out, st.t3, 3);
    st.T.val[3] = vreinterpretq_u8_u32(st.t3);
    const uint8x16_t ops = veorq_u8(vqtbl4q_u8(st.T, vld1q_u8(t.idx)), vld1q_u8(t.inv));
    const uint8x8_t a = vget_low_u8(ops);
    const uint8x8_t b = vext_u8(vget_low_u8(ops), vget_low_u8(ops), 4);
    const uint8x8_t c = vget_high_u8(ops);
    uint8x8_t r;
    if (t.cop == 6 || t.cop == 7) {
        u32 av = vget_lane_u32(vreinterpret_u32_u8(a), 0);
        u32 bv = vget_lane_u32(vreinterpret_u32_u8(b), 0);
        s32 d = 0;
        for (int k = 0; k < 3; k++) {
            s32 x = (s32)((av >> (8 * k)) & 255), y = (s32)((bv >> (8 * k)) & 255);
            d += ((x * 2 - 255) * (y * 2 - 255) + 128) / 256;
        }
        d = d < 0 ? 0 : d > 255 ? 255 : d;
        u32 alpha;
        if (t.cop == 7)
            alpha = (u32)d;
        else
            alpha = (t.aop == 6 || t.aop == 7) ? 0 : vget_lane_u8(tev_op(t.aop, a, b, c), 3);
        r = vreinterpret_u8_u32(vdup_n_u32((u32)d * 0x010101u | alpha << 24));
    } else {
        r = tev_op(t.cop, a, b, c);
        if (t.aop != t.cop) {
            u8 al = (t.aop == 6 || t.aop == 7) ? 0 : vget_lane_u8(tev_op(t.aop, a, b, c), 3);
            r = vset_lane_u8(al, r, 3);
        }
    }
    if (t.cmul != 1 || t.amul != 1) {
        const u16 m[4] = {t.cmul, t.cmul, t.cmul, t.amul};
        uint16x4_t w = vmul_u16(vget_low_u16(vmovl_u8(r)), vld1_u16(m));
        r = vqmovn_u16(vcombine_u16(w, w));
    }
    st.out = vget_lane_u32(vreinterpret_u32_u8(r), 0);
    st.buf = st.next_buf;
    if (t.upd_rgb) st.next_buf = (st.next_buf & 0xff000000u) | (st.out & 0xffffff);
    if (t.upd_a) st.next_buf = (st.next_buf & 0xffffff) | (st.out & 0xff000000u);
}

static inline void tev_load(TevRun &st, const u32 *src7) {
    alignas(16) u32 src[8] = {src7[0], src7[1], src7[2], src7[3], src7[4], src7[5], src7[6], 0};
    st.T.val[0] = vld1q_u8((const u8 *)&src[0]);
    st.T.val[1] = vld1q_u8((const u8 *)&src[4]);
    st.T.val[2] = vdupq_n_u8(0);
    st.t3 = vdupq_n_u32(0);
}

static void tev_run_range(const DrawState &ds, u32 begin, u32 end, const u32 *src7, u32 &out,
                          u32 &buf, u32 &next_buf) {
    TevRun st;
    tev_load(st, src7);
    st.out = out;
    st.buf = buf;
    st.next_buf = next_buf;
    for (u32 si = begin; si < end; si++) tev_stage(ds.tev[si], st);
    out = st.out;
    buf = st.buf;
    next_buf = st.next_buf;
}

// ---- 4 fragments per 16-byte vector
static inline uint8x16_t q_div255_narrow(uint16x8_t lo, uint16x8_t hi) {
    return vcombine_u8(vmovn_u16(vdiv255(lo)), vmovn_u16(vdiv255(hi)));
}
static inline uint8x16_t q_mul255(uint8x16_t a, uint8x16_t b) {
    return q_div255_narrow(vmull_u8(vget_low_u8(a), vget_low_u8(b)), vmull_high_u8(a, b));
}
static inline uint8x16_t tev_op4(u32 op, uint8x16_t a, uint8x16_t b, uint8x16_t c) {
    switch (op) {
    case 0: return a;
    case 1: return q_mul255(a, b);
    case 2: return vqaddq_u8(a, b);
    case 3: {
        const int16x8_t k = vdupq_n_s16(128);
        int16x8_t lo = vsubq_s16(vreinterpretq_s16_u16(vaddl_u8(vget_low_u8(a), vget_low_u8(b))), k);
        int16x8_t hi = vsubq_s16(vreinterpretq_s16_u16(vaddl_high_u8(a, b)), k);
        return vcombine_u8(vqmovun_s16(lo), vqmovun_s16(hi));
    }
    case 4: {
        const uint8x16_t nc = vmvnq_u8(c);
        uint16x8_t lo = vmlal_u8(vmull_u8(vget_low_u8(a), vget_low_u8(c)), vget_low_u8(b), vget_low_u8(nc));
        uint16x8_t hi = vmlal_high_u8(vmull_high_u8(a, c), b, nc);
        return q_div255_narrow(lo, hi);
    }
    case 5: return vqsubq_u8(a, b);
    case 8: return vqaddq_u8(q_mul255(a, b), c);
    case 9: return q_mul255(vqaddq_u8(a, b), c);
    default: return a;
    }
}

void tev_combine4(const DrawState &ds, const u32 primary[4], const u32 frag_pri[4],
                  const u32 frag_sec[4], const u32 tex[4][4], u32 out4[4]) {
    uint8x16_t S[16];
    S[0] = vld1q_u8((const u8 *)primary);
    S[1] = vld1q_u8((const u8 *)frag_pri);
    S[2] = vld1q_u8((const u8 *)frag_sec);
    for (int i = 0; i < 4; i++) S[3 + i] = vld1q_u8((const u8 *)tex[i]);
    for (int i = 7; i < 13; i++) S[i] = vdupq_n_u8(0);
    uint8x16_t out = vreinterpretq_u8_u32(vdupq_n_u32(ds.tev_init_out));
    uint8x16_t buf = vreinterpretq_u8_u32(vdupq_n_u32(ds.tev_init_buf));
    uint8x16_t next = vreinterpretq_u8_u32(vdupq_n_u32(ds.tev_init_next));
    const uint8x16_t amask = vreinterpretq_u8_u32(vdupq_n_u32(0xff000000u));
    for (u32 li = 0; li < ds.tev_count; li++) {
        const TevStageState &t = ds.tev[ds.tev_list[li]];
        S[13] = buf;
        S[14] = vreinterpretq_u8_u32(vdupq_n_u32(t.konst));
        S[15] = out;
        uint8x16_t o[3];
        const u32 nops = std::max(t.cop == 0 ? 1u : (t.cop == 4 || t.cop == 8 || t.cop == 9) ? 3u : 2u,
                                  t.aop == 0 ? 1u : (t.aop == 4 || t.aop == 8 || t.aop == 9) ? 3u : 2u);
        for (u32 k = 0; k < nops; k++) {
            uint8x16_t v;
            if (t.q_same[k])
                v = vqtbl1q_u8(S[t.csrc[k]], vld1q_u8(t.q_idx[k]));
            else
                v = vorrq_u8(vqtbl1q_u8(S[t.csrc[k]], vld1q_u8(t.q_idxc[k])),
                             vqtbl1q_u8(S[t.asrc[k]], vld1q_u8(t.q_idxa[k])));
            o[k] = veorq_u8(v, vld1q_u8(t.q_inv[k]));
        }
        for (u32 k = nops; k < 3; k++) o[k] = vdupq_n_u8(0);
        uint8x16_t r;
        if (t.cop == 6 || t.cop == 7 || t.aop == 6 || t.aop == 7) {
            // dot3 or unusual alpha ops: per pixel, same math as tev_stage
            alignas(16) u8 A[16], B[16], Cc[16], R[16];
            vst1q_u8(A, o[0]);
            vst1q_u8(B, o[1]);
            vst1q_u8(Cc, o[2]);
            for (int p = 0; p < 4; p++) {
                const uint8x8_t a = vreinterpret_u8_u32(vdup_n_u32(*(const u32 *)&A[4 * p]));
                const uint8x8_t b = vreinterpret_u8_u32(vdup_n_u32(*(const u32 *)&B[4 * p]));
                const uint8x8_t c = vreinterpret_u8_u32(vdup_n_u32(*(const u32 *)&Cc[4 * p]));
                uint8x8_t rr;
                if (t.cop == 6 || t.cop == 7) {
                    s32 d = 0;
                    for (int kk = 0; kk < 3; kk++) {
                        s32 x = A[4 * p + kk], y = B[4 * p + kk];
                        d += ((x * 2 - 255) * (y * 2 - 255) + 128) / 256;
                    }
                    d = d < 0 ? 0 : d > 255 ? 255 : d;
                    u32 alpha;
                    if (t.cop == 7)
                        alpha = (u32)d;
                    else
                        alpha = (t.aop == 6 || t.aop == 7) ? 0 : vget_lane_u8(tev_op(t.aop, a, b, c), 3);
                    rr = vreinterpret_u8_u32(vdup_n_u32((u32)d * 0x010101u | alpha << 24));
                } else {
                    rr = vset_lane_u8(0, tev_op(t.cop, a, b, c), 3);
                }
                vst1_lane_u32((u32 *)&R[4 * p], vreinterpret_u32_u8(rr), 0);
            }
            r = vld1q_u8(R);
        } else {
            r = tev_op4(t.cop, o[0], o[1], o[2]);
            if (t.aop != t.cop) r = vbslq_u8(amask, tev_op4(t.aop, o[0], o[1], o[2]), r);
        }
        if (t.cmul != 1 || t.amul != 1) {
            const uint16x8_t m = vreinterpretq_u16_u64(vdupq_n_u64(
                (u64)t.cmul | (u64)t.cmul << 16 | (u64)t.cmul << 32 | (u64)t.amul << 48));
            r = vcombine_u8(vqmovn_u16(vmulq_u16(vmovl_u8(vget_low_u8(r)), m)),
                            vqmovn_u16(vmulq_u16(vmovl_high_u8(r), m)));
        }
        out = r;
        buf = next;
        if (t.upd_rgb) next = vbslq_u8(amask, next, out);
        if (t.upd_a) next = vbslq_u8(amask, out, next);
    }
    vst1q_u8((u8 *)out4, out);
}

u32 tev_combine(const DrawState &ds, u32 primary, u32 frag_pri, u32 frag_sec, const u32 tex[4]) {
    const u32 src7[7] = {primary, frag_pri, frag_sec, tex[0], tex[1], tex[2], tex[3]};
    TevRun st;
    tev_load(st, src7);
    st.out = ds.tev_init_out;
    st.buf = ds.tev_init_buf;
    st.next_buf = ds.tev_init_next;
    for (u32 i = 0; i < ds.tev_count; i++) tev_stage(ds.tev[ds.tev_list[i]], st);
    return st.out;
}
#else
static void tev_run_list(const DrawState &ds, const u8 *list, u32 count, const u32 *src7, u32 &out,
                         u32 &buf, u32 &next_buf) {
    u32 src[16] = {};
    for (int i = 0; i < 7; i++) src[i] = src7[i];
    for (u32 li = 0; li < count; li++) {
        const TevStageState &t = ds.tev[list[li]];
        src[0xd] = buf;
        src[0xe] = t.konst;
        src[0xf] = out;
        C3 c0 = color_mod(t.cmod[0], src[t.csrc[0]]);
        C3 c1 = color_mod(t.cmod[1], src[t.csrc[1]]);
        C3 c2 = color_mod(t.cmod[2], src[t.csrc[2]]);
        s32 r, gg, b, a;
        if (t.cop == 6 || t.cop == 7) {
            s32 d = ((c0.r * 2 - 255) * (c1.r * 2 - 255) + 128) / 256 +
                    ((c0.g * 2 - 255) * (c1.g * 2 - 255) + 128) / 256 +
                    ((c0.b * 2 - 255) * (c1.b * 2 - 255) + 128) / 256;
            r = gg = b = cl255(d);
        } else {
            r = combine1(t.cop, c0.r, c1.r, c2.r);
            gg = combine1(t.cop, c0.g, c1.g, c2.g);
            b = combine1(t.cop, c0.b, c1.b, c2.b);
        }
        if (t.cop == 7) {
            a = r;
        } else {
            s32 a0 = alpha_mod(t.amod[0], src[t.asrc[0]]);
            s32 a1 = alpha_mod(t.amod[1], src[t.asrc[1]]);
            s32 a2 = alpha_mod(t.amod[2], src[t.asrc[2]]);
            a = t.aop == 6 || t.aop == 7 ? 0 : combine1(t.aop, a0, a1, a2);
        }
        r = std::min(255, r * t.cmul);
        gg = std::min(255, gg * t.cmul);
        b = std::min(255, b * t.cmul);
        a = std::min(255, a * t.amul);
        out = rgba(r, gg, b, a);
        buf = next_buf;
        if (t.upd_rgb) next_buf = (next_buf & 0xff000000u) | (out & 0xffffff);
        if (t.upd_a) next_buf = (next_buf & 0xffffff) | (out & 0xff000000u);
    }
}
static void tev_run_range(const DrawState &ds, u32 begin, u32 end, const u32 *src7, u32 &out,
                          u32 &buf, u32 &next_buf) {
    u8 list[6];
    for (u32 i = begin; i < end; i++) list[i - begin] = (u8)i;
    tev_run_list(ds, list, end - begin, src7, out, buf, next_buf);
}
u32 tev_combine(const DrawState &ds, u32 primary, u32 frag_pri, u32 frag_sec, const u32 tex[4]) {
    const u32 src7[7] = {primary, frag_pri, frag_sec, tex[0], tex[1], tex[2], tex[3]};
    u32 out = ds.tev_init_out, buf = ds.tev_init_buf, next_buf = ds.tev_init_next;
    tev_run_list(ds, ds.tev_list, ds.tev_count, src7, out, buf, next_buf);
    return out;
}
void tev_combine4(const DrawState &ds, const u32 primary[4], const u32 frag_pri[4],
                  const u32 frag_sec[4], const u32 tex[4][4], u32 out[4]) {
    for (int p = 0; p < 4; p++) {
        const u32 tx[4] = {tex[0][p], tex[1][p], tex[2][p], tex[3][p]};
        out[p] = tev_combine(ds, primary[p], frag_pri[p], frag_sec[p], tx);
    }
}
#endif

// operands read by an op: Replace 1, Lerp / MultiplyThenAdd / AddThenMultiply 3, others 2
static inline u32 op_operands(u32 op) { return op == 0 ? 1 : (op == 4 || op == 8 || op == 9) ? 3 : 2; }

void tev_finish_prepare(DrawState &ds) {
    // fold the leading stages that only depend on constants
    u32 out = 0, buf = 0, next_buf = ds.tev_buf_color;
    bool out_const = true, buf_const = true, next_const = true;
    u32 first = 0;
    const u32 zero7[7] = {0, 0, 0, 0, 0, 0, 0};
    for (; first < ds.tev_n; first++) {
        const TevStageState &t = ds.tev[first];
        auto is_const = [&](u32 src) {
            return src == 14 || (src == 15 && out_const) || (src == 13 && buf_const) ||
                   (src >= 7 && src <= 12);
        };
        bool c = true;
        const u32 nc = op_operands(t.cop) , na = op_operands(t.aop);
        for (u32 k = 0; k < nc; k++) c &= is_const(t.csrc[k]);
        if (t.cop != 7)
            for (u32 k = 0; k < na; k++) c &= is_const(t.asrc[k]);
        if (!c) break;
        tev_run_range(ds, first, first + 1, zero7, out, buf, next_buf);
        buf_const = next_const;
        (void)out_const;
    }
    ds.tev_init_out = out;
    ds.tev_init_buf = buf;
    ds.tev_init_next = next_buf;
    // remaining stages; pass-through ones can be dropped when nothing reads the combiner buffer
    bool reads_buf = false;
    for (u32 si = first; si < ds.tev_n; si++)
        for (u32 k = 0; k < 3; k++)
            if (ds.tev[si].csrc[k] == 13 || ds.tev[si].asrc[k] == 13) reads_buf = true;
    ds.tev_count = 0;
    for (u32 si = first; si < ds.tev_n; si++) {
        const TevStageState &t = ds.tev[si];
        const bool identity = t.cop == 0 && t.aop == 0 && t.csrc[0] == 15 && t.cmod[0] == 0 &&
                              t.asrc[0] == 15 && t.amod[0] == 0 && t.cmul == 1 && t.amul == 1;
        if (identity && !reads_buf) continue;
        ds.tev_list[ds.tev_count++] = (u8)si;
    }
}

// ---------------------------------------------------------------- procedural texture
namespace {
inline float pt_lut(const float *v, const float *d, float coord) {
    coord *= 128.0f;
    int i = coord < 0 ? 0 : (int)coord;
    if (i > 127) i = 127;
    return v[i] + (coord - i) * d[i];
}
u32 noise_rand1d(u32 v) {
    static const u32 table[16] = {0, 4, 10, 8, 4, 9, 7, 12, 5, 15, 13, 14, 11, 15, 2, 11};
    return ((v % 9 + 2) * 3 & 0xF) ^ table[(v / 9) & 0xF];
}
float noise_rand2d(u32 x, u32 y) {
    static const u32 table[16] = {10, 2, 15, 8, 0, 7, 4, 5, 5, 13, 2, 6, 13, 9, 3, 14};
    u32 u2 = noise_rand1d(x);
    u32 v2 = noise_rand1d(y);
    v2 += ((u2 & 3) == 1) ? 4 : 0;
    v2 ^= (u2 & 1) * 6;
    v2 += 10 + u2;
    v2 &= 0xF;
    v2 ^= table[u2];
    return -1.0f + v2 * 2.0f / 15.0f;
}
float noise_coef(const ProcTexState &pt, float u, float v) {
    const ProcTexLuts &L = *pt.luts;
    float x = 9 * pt.noise_freq_u * fabsf(u + pt.noise_phase_u);
    float y = 9 * pt.noise_freq_v * fabsf(v + pt.noise_phase_v);
    int xi = (int)x, yi = (int)y;
    float xf = x - xi, yf = y - yi;
    float g0 = noise_rand2d(xi, yi) * (xf + yf);
    float g1 = noise_rand2d(xi + 1, yi) * (xf + yf - 1);
    float g2 = noise_rand2d(xi, yi + 1) * (xf + yf - 1);
    float g3 = noise_rand2d(xi + 1, yi + 1) * (xf + yf - 2);
    float xn = pt_lut(L.noise_v, L.noise_d, xf);
    float yn = pt_lut(L.noise_v, L.noise_d, yf);
    float a = g0 * (1 - xn) + g1 * xn;
    float b = g2 * (1 - xn) + g3 * xn;
    return a * (1 - yn) + b * yn;
}
float shift_offset(float v, u32 mode, u32 clamp_mode) {
    float off = clamp_mode == 3 ? 1.0f : 0.5f;
    if (mode == 1) return off * (((int)v / 2) % 2);
    if (mode == 2) return off * ((((int)v + 1) / 2) % 2);
    return 0;
}
void clamp_coord(float &c, u32 mode) {
    switch (mode) {
    case 0: if (c > 1.0f) c = 0.0f; break;
    case 1: c = std::min(c, 1.0f); break;
    case 2: c = c - floorf(c); break;
    case 3: {
        int i = (int)c;
        float f = c - i;
        c = (i % 2) == 0 ? f : 1.0f - f;
        break;
    }
    case 4: c = c <= 0.5f ? 0.0f : 1.0f; break;
    default: c = std::min(c, 1.0f); break;
    }
}
float combine_map(float u, float v, u32 comb, const float *mv, const float *md) {
    float f;
    switch (comb) {
    case 0: f = u; break;
    case 1: f = u * u; break;
    case 2: f = v; break;
    case 3: f = v * v; break;
    case 4: f = (u + v) * 0.5f; break;
    case 5: f = (u * u + v * v) * 0.5f; break;
    case 6: f = std::min(sqrtf(u * u + v * v), 1.0f); break;
    case 7: f = std::min(u, v); break;
    case 8: f = std::max(u, v); break;
    case 9: f = std::min(((u + v) * 0.5f + sqrtf(u * u + v * v)) * 0.5f, 1.0f); break;
    default: f = 0; break;
    }
    return pt_lut(mv, md, f);
}
}  // namespace

u32 sample_proctex(const ProcTexState &pt, float u, float v) {
    const ProcTexLuts &L = *pt.luts;
    u = fabsf(u);
    v = fabsf(v);
    const float us = shift_offset(v, pt.u_shift, pt.u_clamp);
    const float vs = shift_offset(u, pt.v_shift, pt.v_clamp);
    if (pt.noise) {
        float n = noise_coef(pt, u, v);
        u = fabsf(u + n * pt.noise_amp_u);
        v = fabsf(v + n * pt.noise_amp_v);
    }
    u += us;
    v += vs;
    clamp_coord(u, pt.u_clamp);
    clamp_coord(v, pt.v_clamp);
    const float lc = combine_map(u, v, pt.color_comb, L.cmap_v, L.cmap_d);
    float index = pt.lut_offset + lc * (float)((s32)pt.lut_width - 1);
    if (!(index >= 0)) index = 0;
    if (index > 255) index = 255;
    u32 color;
    if (pt.filter == 1 || pt.filter == 3 || pt.filter == 5) {
        int ii = (int)index;
        float f = index - ii;
        u32 c = L.color[ii];
        color = 0;
        for (int k = 0; k < 4; k++) {
            float x = ((c >> (8 * k)) & 255) + f * L.cdiff[ii][k];
            s32 xi = (s32)x;
            color |= (u32)(xi & 255) << (8 * k);
        }
    } else {
        int ii = (int)roundf(index);
        if (ii > 255) ii = 255;
        color = L.color[ii];
    }
    if (pt.separate_alpha) {
        float a = combine_map(u, v, pt.alpha_comb, L.amap_v, L.amap_d);
        color = (color & 0xffffff) | ((u32)(u8)(s32)(a * 255) << 24);
    }
    return color;
}

}  // namespace gpu
