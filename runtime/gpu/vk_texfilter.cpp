// Texture upscaling filters for the Vulkan backend: GLSL ports of texfilter.mm's Metal kernels (see texfilter.mm
// for what Sharp and Smooth do; the logic here is the same, line for line). vulkan.cpp dispatches them.
#include "gpu/vk_texfilter.h"

namespace gpu {
namespace {

const char *const kTfCommon = R"(#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) uniform texture2D src;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D dst;
layout(std430, set = 0, binding = 2) readonly buffer RC { uvec4 rc[]; };
layout(push_constant) uniform TFP { uint sw, sh, dw, dh, wrap_s, wrap_t, factor, pad; } P;

int tf_wrap(int v, int n, uint m) {
    if (m == 1u) { v %= n; return v < 0 ? v + n : v; }
    if (m == 2u) { const int p = 2 * n; v %= p; if (v < 0) v += p; return v >= n ? p - 1 - v : v; }
    return clamp(v, 0, n - 1);
}
vec4 tf_fetch(ivec2 p) {
    return texelFetch(src, ivec2(tf_wrap(p.x, int(P.sw), P.wrap_s), tf_wrap(p.y, int(P.sh), P.wrap_t)), 0);
}
// perceptual difference of two texels (luma + chroma), counting colour less where both are transparent
float tf_dist(vec4 a, vec4 b) {
    const vec3 d = a.rgb - b.rgb;
    const float y = dot(d, vec3(0.299, 0.587, 0.114));
    const float cb = (d.b - y) * 0.564, cr = (d.r - y) * 0.713;
    const float vis = 0.25 + 0.75 * max(a.a, b.a);
    return (abs(y) + 0.5 * (abs(cb) + abs(cr))) * vis + abs(a.a - b.a);
}
float tf_step(float d) { return smoothstep(0.15, 0.45, d); }
float tf_val(vec4 c) { return c.a * (0.5 + 0.5 * dot(c.rgb, vec3(0.299, 0.587, 0.114))); }
float tf_ramp(float a, float b, float c, float d) {
    const float s = c - b, m = abs(s) + 1e-4;
    const float l = s * (b - a) > 0.0 ? abs(b - a) / m : 0.0, r = s * (d - c) > 0.0 ? abs(d - c) / m : 0.0;
    return smoothstep(0.15, 0.4, max(l, r));
}
const float kEps = 1.0 / 64.0;  // colour weight of a fully transparent texel relative to an opaque one
vec4 tf_mix(vec4 a, vec4 b, float t) {
    const float wa = (1.0 - t) * (a.a + kEps), wb = t * (b.a + kEps);
    return vec4((a.rgb * wa + b.rgb * wb) / (wa + wb), mix(a.a, b.a, t));
}
// the output texel gid of this invocation: grid z = rectangle rc[z], x / y within it
#define TF_RECT(W, H) \
    const uvec3 tid = gl_GlobalInvocationID; \
    const uvec4 r = rc[tid.z]; \
    if (tid.x >= r.z || tid.y >= r.w) return; \
    const uvec2 gid = r.xy + tid.xy; \
    if (gid.x >= (W) || gid.y >= (H)) return
)";

const char *const kTfSharp = R"(
vec4 n[5][5];  // the 5x5 neighbourhood of the centre texel E = n[2][2]
struct TfCorner {
    vec4 px;
    float a, shallow, steep;
};
TfCorner tf_corner(ivec2 s) {
#define R(dx, dy) n[2 + (dy) * s.y][2 + (dx) * s.x]
    const vec4 E = R(0, 0), F = R(1, 0), H = R(0, 1), I = R(1, 1);
    const vec4 B = R(0, -1), D = R(-1, 0), C = R(1, -1), G = R(-1, 1);
    const vec4 F4 = R(2, 0), H5 = R(0, 2), I4 = R(2, 1), I5 = R(1, 2);
#undef R
    const float dEF = tf_dist(E, F), dEH = tf_dist(E, H), dEC = tf_dist(E, C), dEG = tf_dist(E, G);
    const float dFG = tf_dist(F, G), dHC = tf_dist(H, C);
    const float wa = dEC + dEG + tf_dist(I, F4) + tf_dist(I, H5) + 4.0 * tf_dist(F, H);
    const float wd = tf_dist(H, D) + tf_dist(H, I5) + tf_dist(F, I4) + tf_dist(F, B) + 4.0 * tf_dist(E, I);
    const float edge = clamp(2.0 * (wd - wa) / (wd + 1e-3), 0.0, 1.0);
    const bool f_side = dEF <= dEH;
    const float dEP = f_side ? dEF : dEH;
    const float cont = 1.0 - smoothstep(0.3, 0.6, min(dEC, dEG) / (dEP + 1e-3));
    TfCorner k;
    k.px = f_side ? F : H;
    k.a = edge * cont * tf_step(dEP);
    k.shallow = 2.0 * dFG <= dHC && dEG > 0.1 && tf_dist(D, G) > 0.1 ? 1.0 : 0.0;
    k.steep = 2.0 * dHC <= dFG && dEC > 0.1 && tf_dist(B, C) > 0.1 ? 1.0 : 0.0;
    return k;
}
// one 2x pass (P.sw x P.sh -> P.dw x P.dh), an invocation per source texel (rectangles in source texels)
void main() {
    TF_RECT(P.sw, P.sh);
    const ivec2 e = ivec2(gid);
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++)
            n[y][x] = (x == 0 || x == 4) && (y == 0 || y == 4) ? vec4(0.0) : tf_fetch(e + ivec2(x - 2, y - 2));
    TfCorner k[2][2];
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) k[j][i] = tf_corner(ivec2(i != 0 ? 1 : -1, j != 0 ? 1 : -1));
    const vec4 E = n[2][2];
    const float vE = tf_val(E);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            const ivec2 s = ivec2(i != 0 ? 1 : -1, j != 0 ? 1 : -1);
            const vec4 F = n[2][2 + s.x], H = n[2 + s.y][2], I = n[2 + s.y][2 + s.x];
            const float vF = tf_val(F), vH = tf_val(H);
            const float rx = tf_ramp(tf_val(n[2][2 - s.x]), vE, vF, tf_val(n[2][2 + 2 * s.x]));
            const float ry = tf_ramp(tf_val(n[2 - s.y][2]), vE, vH, tf_val(n[2 + 2 * s.y][2]));
            const float wx = 0.25 * (1.0 - tf_step(tf_dist(E, F)) * (1.0 - rx));
            const float wy = 0.25 * (1.0 - tf_step(tf_dist(E, H)) * (1.0 - ry));
            vec4 o = tf_mix(tf_mix(E, F, wx), tf_mix(H, I, wx * (1.0 - tf_step(tf_dist(E, I)))), wy);
            const TfCorner kc = k[j][i], kh = k[j][1 - i], kv = k[1 - j][i];
            o = tf_mix(o, kc.px, kc.a * (0.5 + 0.25 * max(kc.shallow, kc.steep)));
            o = tf_mix(o, kh.px, 0.25 * kh.a * kh.shallow);
            o = tf_mix(o, kv.px, 0.25 * kv.a * kv.steep);
            const uvec2 g = gid * 2u + uvec2(i, j);
            if (g.x < P.dw && g.y < P.dh) imageStore(dst, ivec2(g), o);
        }
}
)";

const char *const kTfSmooth = R"(
float tf_luma(vec4 c) { return dot(c.rgb, vec3(0.299, 0.587, 0.114)) * (0.3 + 0.7 * c.a) * 0.7 + c.a * 0.3; }
float tf_lanczos2(float r2) {
    if (r2 >= 4.0) return 0.0;
    if (r2 < 1e-5) return 1.0;
    const float x = sqrt(r2) * 3.14159265358979;
    return 2.0 * sin(x) * sin(x * 0.5) / (x * x);
}
// one pass, any integer factor
void main() {
    TF_RECT(P.dw, P.dh);
    const vec2 p = (vec2(gid) + 0.5) / float(P.factor) - 0.5;
    const vec2 fl = floor(p), f = p - fl;
    const ivec2 b = ivec2(fl);
    vec4 c[4][4];
    float L[4][4];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            c[j][i] = tf_fetch(b + ivec2(i - 1, j - 1));
            L[j][i] = tf_luma(c[j][i]);
        }
    float jxx = 0.0, jxy = 0.0, jyy = 0.0;
    for (int j = 1; j <= 2; j++)
        for (int i = 1; i <= 2; i++) {
            const float w = (i == 1 ? 1.0 - f.x : f.x) * (j == 1 ? 1.0 - f.y : f.y);
            const float gx = 0.5 * (L[j][i + 1] - L[j][i - 1]), gy = 0.5 * (L[j + 1][i] - L[j - 1][i]);
            jxx += w * gx * gx;
            jxy += w * gx * gy;
            jyy += w * gy * gy;
        }
    const float tr = jxx + jyy, dd = sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy);
    const float l1 = 0.5 * (tr + dd), l2 = 0.5 * (tr - dd);
    vec2 n = vec2(jxy, l1 - jxx);
    const vec2 n2 = vec2(l1 - jyy, jxy);
    if (dot(n2, n2) > dot(n, n)) n = n2;
    n = dot(n, n) > 1e-12 ? normalize(n) : vec2(1.0, 0.0);
    const vec2 t = vec2(-n.y, n.x);
    const float coh = (l1 - l2) / (l1 + l2 + 1e-6);
    const float a = coh * smoothstep(0.0005, 0.01, l1);
    const float sx = 1.0 + 0.5 * a, sy = 1.0 / (1.0 + 2.0 * a);
    vec4 acc = vec4(0.0);
    float wsum = 0.0, csum = 0.0;
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            const vec2 v = vec2(i - 1, j - 1) - f;
            const float va = dot(v, n) * sx, vt = dot(v, t) * sy;
            const float w = tf_lanczos2(va * va + vt * vt);
            const float wc = w * (c[j][i].a + kEps);
            acc.rgb += c[j][i].rgb * wc;
            acc.a += c[j][i].a * w;
            wsum += w;
            csum += wc;
        }
    vec4 o = vec4(abs(csum) > 1e-4 ? acc.rgb / csum : c[1][1].rgb, wsum > 1e-4 ? acc.a / wsum : c[1][1].a);
    vec4 lo = vec4(1.0), hi = vec4(0.0);
    vec3 clo = vec3(1.0), chi = vec3(0.0);
    bool any = false;
    for (int j = 1; j <= 2; j++)
        for (int i = 1; i <= 2; i++) {
            lo = min(lo, c[j][i]);
            hi = max(hi, c[j][i]);
            if (c[j][i].a > 0.02) {
                clo = min(clo, c[j][i].rgb);
                chi = max(chi, c[j][i].rgb);
                any = true;
            }
        }
    if (!any) {
        clo = lo.rgb;
        chi = hi.rgb;
    }
    o = vec4(clamp(o.rgb, clo, chi), clamp(o.a, lo.a, hi.a));
    imageStore(dst, ivec2(gid), o);
}
)";

// mip level: alpha-weighted 2x2 average (src: the level above)
const char *const kTfDown = R"(
void main() {
    const ivec2 ds = imageSize(dst);
    TF_RECT(uint(ds.x), uint(ds.y));
    vec3 rgb = vec3(0.0);
    float w = 0.0, al = 0.0;
    for (uint j = 0u; j < 2u; j++)
        for (uint i = 0u; i < 2u; i++) {
            const vec4 c = texelFetch(src, ivec2(gid * 2u + uvec2(i, j)), 0);
            const float cw = c.a + kEps;
            rgb += c.rgb * cw;
            w += cw;
            al += c.a;
        }
    imageStore(dst, ivec2(gid), vec4(rgb / w, al * 0.25));
}
)";

}  // namespace

const char *vk_texfilter_glsl(VkTfKernel k) {
    static std::string src[kVkTfKernCount];
    const char *body[kVkTfKernCount] = {kTfSharp, kTfSmooth, kTfDown};
    if (src[k].empty()) src[k] = std::string(kTfCommon) + body[k];
    return src[k].c_str();
}

}  // namespace gpu
