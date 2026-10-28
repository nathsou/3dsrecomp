// Texture upscaling filters (Metal compute), used by the Metal backend's texture cache (metal.mm) when
// gpu::set_texture_filter() / GPU_TEX_FILTER selects one. Each decoded texture is upscaled once, when it is
// uploaded; the result replaces it (normalized texture coordinates need no change) and keeps the game's own
// mip levels below the upscaled ones.
//
// Both filters work on straight (not premultiplied) RGBA and weight colours by alpha when mixing texels, so
// that the colour of fully transparent texels (often black) does not bleed into cut-out edges, while texels
// that are all transparent (or alpha used as a mask) keep their own colours.
//
// - Sharp: an edge-directed 2x pass in the spirit of xBR (applied twice for 4x). For each output quarter
//   of a source texel E it compares, over a 5x5 neighbourhood, how much the colours vary along the two
//   diagonals around the corner; when the diagonal towards the corner is crossed by an edge (the other
//   diagonal being the more uniform one), the corner takes the colour of the side it belongs to, which
//   turns staircases into straight diagonal edges. A corner is only cut when the edge continues along the
//   diagonal (one of the two texels diagonally next to E is close to E), which keeps square corners square.
//   Half the corner quarter is cut for a 45 degree edge; for a shallower or steeper one (the far side also
//   continues beside or above E) three quarters of it and a quarter of the next quarter. The base is a
//   contrast-adaptive bilinear 2x: it interpolates between similar texels (gradients, shading, compression
//   noise) and across ramps (anti-aliased edges), and keeps hard steps, so that only real edges become crisp
//   and smooth areas do not turn into flat blocks.
// - Smooth: an edge-adaptive Lanczos-2 resampler (in the spirit of FSR1's EASU). The local structure
//   tensor of the 2x2 texels around the sample gives the edge direction and how directional it is; the
//   16-tap kernel is stretched along edges and narrowed across them, and the result is clamped to the
//   range of the 4 nearest texels (no ringing halos).
//
// Written for this project from the published descriptions of these algorithms (no code taken from them).
#include "gpu/texfilter.h"

#include "gpu/pica.h"

namespace gpu {
namespace {

const char *kTexFilterMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct TFP { uint sw, sh, dw, dh, wrap_s, wrap_t, factor, pad; };

inline int tf_wrap(int v, int n, uint m) {
    if (m == 1) { v %= n; return v < 0 ? v + n : v; }
    if (m == 2) { const int p = 2 * n; v %= p; if (v < 0) v += p; return v >= n ? p - 1 - v : v; }
    return clamp(v, 0, n - 1);
}
inline float4 tf_fetch(texture2d<float, access::read> t, int2 p, constant TFP &P) {
    return t.read(uint2(tf_wrap(p.x, int(P.sw), P.wrap_s), tf_wrap(p.y, int(P.sh), P.wrap_t)));
}
// perceptual difference of two texels (luma + chroma), counting colour less where both are transparent
inline float tf_dist(float4 a, float4 b) {
    const float3 d = a.rgb - b.rgb;
    const float y = dot(d, float3(0.299, 0.587, 0.114));
    const float cb = (d.b - y) * 0.564, cr = (d.r - y) * 0.713;
    const float vis = 0.25 + 0.75 * max(a.a, b.a);
    return (abs(y) + 0.5 * (abs(cb) + abs(cr))) * vis + abs(a.a - b.a);
}
// how surely two texels are on different sides of an edge
inline float tf_step(float d) { return smoothstep(0.15, 0.45, d); }
// a texel as one value (coverage and brightness), to tell ramps from steps
inline float tf_val(float4 c) { return c.a * (0.5 + 0.5 * dot(c.rgb, float3(0.299, 0.587, 0.114))); }
// how much the change b -> c is part of a ramp (anti-aliasing, gradient): the change continues in the same
// direction on a side (a -> b or c -> d) by a fair part of it
inline float tf_ramp(float a, float b, float c, float d) {
    const float s = c - b, m = abs(s) + 1e-4;
    const float l = s * (b - a) > 0.0 ? abs(b - a) / m : 0.0, r = s * (d - c) > 0.0 ? abs(d - c) / m : 0.0;
    return smoothstep(0.15, 0.4, max(l, r));
}
constant float kEps = 1.0 / 64.0;  // colour weight of a fully transparent texel relative to an opaque one
// a * (1 - t) + b * t, the colours weighted by alpha
inline float4 tf_mix(float4 a, float4 b, float t) {
    const float wa = (1.0 - t) * (a.a + kEps), wb = t * (b.a + kEps);
    return float4((a.rgb * wa + b.rgb * wb) / (wa + wb), mix(a.a, b.a, t));
}
// the output texel gid of thread tid: grid z = rectangle rc[z] (x, y, w, h), x / y within it
#define TF_RECT(W, H) \
    const uint4 r = rc[tid.z]; \
    if (tid.x >= r.z || tid.y >= r.w) return; \
    const uint2 gid = r.xy + tid.xy; \
    if (gid.x >= (W) || gid.y >= (H)) return

// Sharp, the corner of the centre texel E of the 5x5 neighbourhood n (n[y][x], E = n[2][2]) towards s (each
// component +-1). The neighbourhood is read mirrored so that this corner is always the bottom-right one:
//        B  C
//     D  E  F  F4
//     G  H  I  I4
//        H5 I5
// px: the colour of the other side of a diagonal edge cutting the corner; a: how surely there is one;
// shallow / steep: the edge is closer to horizontal / vertical (it also cuts the next quarter of E).
struct TfCorner {
    float4 px;
    float a, shallow, steep;
};
inline TfCorner tf_corner(thread const float4 (&n)[5][5], int2 s) {
#define R(dx, dy) n[2 + (dy) * s.y][2 + (dx) * s.x]
    const float4 E = R(0, 0), F = R(1, 0), H = R(0, 1), I = R(1, 1);
    const float4 B = R(0, -1), D = R(-1, 0), C = R(1, -1), G = R(-1, 1);
    const float4 F4 = R(2, 0), H5 = R(0, 2), I4 = R(2, 1), I5 = R(1, 2);
#undef R
    const float dEF = tf_dist(E, F), dEH = tf_dist(E, H), dEC = tf_dist(E, C), dEG = tf_dist(E, G);
    const float dFG = tf_dist(F, G), dHC = tf_dist(H, C);
    // the variation along the anti-diagonal (parallel to F-H) against the variation along the diagonal
    // (parallel to E-I): an edge parallel to F-H separates E from I, E's corner belongs to the F/H side ...
    const float wa = dEC + dEG + tf_dist(I, F4) + tf_dist(I, H5) + 4.0 * tf_dist(F, H);
    const float wd = tf_dist(H, D) + tf_dist(H, I5) + tf_dist(F, I4) + tf_dist(F, B) + 4.0 * tf_dist(E, I);
    const float edge = saturate(2.0 * (wd - wa) / (wd + 1e-3));
    // ... if the edge runs on along the diagonal (else E is the corner of a square block: keep it)
    const bool f_side = dEF <= dEH;
    const float dEP = f_side ? dEF : dEH;
    const float cont = 1.0 - smoothstep(0.3, 0.6, min(dEC, dEG) / (dEP + 1e-3));
    TfCorner k;
    k.px = f_side ? F : H;
    k.a = edge * cont * tf_step(dEP);
    // shallow: the far side continues under D (G like F, unlike E and D); steep: likewise above, at C
    k.shallow = 2.0 * dFG <= dHC && dEG > 0.1 && tf_dist(D, G) > 0.1 ? 1.0 : 0.0;
    k.steep = 2.0 * dHC <= dFG && dEC > 0.1 && tf_dist(B, C) > 0.1 ? 1.0 : 0.0;
    return k;
}

// Sharp: one 2x pass (P.sw x P.sh -> P.dw x P.dh), a thread per source texel (rectangles in source texels)
kernel void k_tf_sharp(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                       constant TFP &P [[buffer(0)]], device const uint4 *rc [[buffer(1)]],
                       uint3 tid [[thread_position_in_grid]]) {
    TF_RECT(P.sw, P.sh);
    const int2 e = int2(gid);
    float4 n[5][5];
    for (int y = 0; y < 5; y++)
        for (int x = 0; x < 5; x++)
            n[y][x] = (x == 0 || x == 4) && (y == 0 || y == 4) ? float4(0) : tf_fetch(src, e + int2(x - 2, y - 2), P);
    TfCorner k[2][2];
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) k[j][i] = tf_corner(n, int2(i ? 1 : -1, j ? 1 : -1));
    const float4 E = n[2][2];
    const float vE = tf_val(E);
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 2; i++) {
            // the quarter of E towards s
            const int2 s = int2(i ? 1 : -1, j ? 1 : -1);
            const float4 F = n[2][2 + s.x], H = n[2 + s.y][2], I = n[2 + s.y][2 + s.x];
            // contrast-adaptive bilinear: interpolates between texels of the same surface (gradients, shading,
            // compression noise) and across anti-aliased edges (ramps over several texels), keeps hard steps
            const float vF = tf_val(F), vH = tf_val(H);
            const float rx = tf_ramp(tf_val(n[2][2 - s.x]), vE, vF, tf_val(n[2][2 + 2 * s.x]));
            const float ry = tf_ramp(tf_val(n[2 - s.y][2]), vE, vH, tf_val(n[2 + 2 * s.y][2]));
            const float wx = 0.25 * (1.0 - tf_step(tf_dist(E, F)) * (1.0 - rx));
            const float wy = 0.25 * (1.0 - tf_step(tf_dist(E, H)) * (1.0 - ry));
            float4 o = tf_mix(tf_mix(E, F, wx), tf_mix(H, I, wx * (1.0 - tf_step(tf_dist(E, I)))), wy);
            // diagonal edges: this quarter's corner (half of it beyond a 45 degree edge, three quarters beyond
            // a shallower or steeper one), and the shallow / steep edges of the corners next to it
            const TfCorner kc = k[j][i], kh = k[j][1 - i], kv = k[1 - j][i];
            o = tf_mix(o, kc.px, kc.a * (0.5 + 0.25 * max(kc.shallow, kc.steep)));
            o = tf_mix(o, kh.px, 0.25 * kh.a * kh.shallow);
            o = tf_mix(o, kv.px, 0.25 * kv.a * kv.steep);
            const uint2 g = gid * 2 + uint2(i, j);
            if (g.x < P.dw && g.y < P.dh) dst.write(o, g);
        }
}

inline float tf_luma(float4 c) { return dot(c.rgb, float3(0.299, 0.587, 0.114)) * (0.3 + 0.7 * c.a) * 0.7 + c.a * 0.3; }
inline float tf_lanczos2(float r2) {
    if (r2 >= 4.0) return 0.0;
    if (r2 < 1e-5) return 1.0;
    const float x = sqrt(r2) * M_PI_F;
    return 2.0 * sin(x) * sin(x * 0.5) / (x * x);
}

// Smooth: one pass, any integer factor
kernel void k_tf_smooth(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                        constant TFP &P [[buffer(0)]], device const uint4 *rc [[buffer(1)]],
                       uint3 tid [[thread_position_in_grid]]) {
    TF_RECT(P.dw, P.dh);
    const float2 p = (float2(gid) + 0.5) / float(P.factor) - 0.5;
    const float2 fl = floor(p), f = p - fl;
    const int2 b = int2(fl);
    float4 c[4][4];
    float L[4][4];
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            c[j][i] = tf_fetch(src, b + int2(i - 1, j - 1), P);
            L[j][i] = tf_luma(c[j][i]);
        }
    // structure tensor of the inner 2x2 (bilinear weights)
    float jxx = 0, jxy = 0, jyy = 0;
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
    float2 n = float2(jxy, l1 - jxx);
    const float2 n2 = float2(l1 - jyy, jxy);
    if (dot(n2, n2) > dot(n, n)) n = n2;
    n = dot(n, n) > 1e-12 ? normalize(n) : float2(1, 0);  // across the edge
    const float2 t = float2(-n.y, n.x);                  // along it
    const float coh = (l1 - l2) / (l1 + l2 + 1e-6);
    const float a = coh * smoothstep(0.0005, 0.01, l1);
    const float sx = 1.0 + 0.5 * a, sy = 1.0 / (1.0 + 2.0 * a);
    float4 acc = 0;
    float wsum = 0, csum = 0;
    for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++) {
            const float2 v = float2(i - 1, j - 1) - f;
            const float va = dot(v, n) * sx, vt = dot(v, t) * sy;
            const float w = tf_lanczos2(va * va + vt * vt);
            const float wc = w * (c[j][i].a + kEps);
            acc.rgb += c[j][i].rgb * wc;
            acc.a += c[j][i].a * w;
            wsum += w;
            csum += wc;
        }
    float4 o = float4(abs(csum) > 1e-4 ? acc.rgb / csum : c[1][1].rgb, wsum > 1e-4 ? acc.a / wsum : c[1][1].a);
    // clamp to the 4 nearest texels (colour: of those that are not transparent, if any)
    float4 lo = 1, hi = 0;
    float3 clo = 1, chi = 0;
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
    o = float4(clamp(o.rgb, clo, chi), clamp(o.a, lo.a, hi.a));
    dst.write(o, gid);
}

// mip level: alpha-weighted 2x2 average
kernel void k_tf_down(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                      device const uint4 *rc [[buffer(1)]], uint3 tid [[thread_position_in_grid]]) {
    TF_RECT(dst.get_width(), dst.get_height());
    float3 rgb = 0;
    float w = 0, al = 0;
    for (uint j = 0; j < 2; j++)
        for (uint i = 0; i < 2; i++) {
            const float4 c = src.read(gid * 2 + uint2(i, j));
            const float cw = c.a + kEps;
            rgb += c.rgb * cw;
            w += cw;
            al += c.a;
        }
    dst.write(float4(rgb / w, al * 0.25), gid);
}
)";

id<MTLComputePipelineState> k_sharp, k_smooth, k_down;
bool tried = false;

bool tf_init(id<MTLDevice> dev) {
    if (tried) return k_sharp && k_smooth && k_down;
    tried = true;
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kTexFilterMSL] options:nil error:&err];
    if (!lib) {
        LOG("gpu metal: texture filter kernels failed to compile: %s\n", err.localizedDescription.UTF8String);
        return false;
    }
    k_sharp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tf_sharp"] error:&err];
    k_smooth = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tf_smooth"] error:&err];
    k_down = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"k_tf_down"] error:&err];
    if (!k_sharp || !k_smooth || !k_down) LOG("gpu metal: texture filter kernels unavailable\n");
    return k_sharp && k_smooth && k_down;
}

struct TFP {
    u32 sw, sh, dw, dh, wrap_s, wrap_t, factor, pad;
};

id<MTLTexture> level_view(id<MTLTexture> t, u32 l) {
    return [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(l, 1)
                                     slices:NSMakeRange(0, 1)];
}

// one dispatch over the rectangles rc (source rectangles scaled by num / den, in output texels)
void dispatch(id<MTLDevice> dev, id<MTLComputeCommandEncoder> ce, const TfRect *rc, u32 n, u32 num, u32 den) {
    std::vector<u32> v(n * 4);
    u32 mw = 1, mh = 1;
    for (u32 i = 0; i < n; i++) {
        v[i * 4] = rc[i].x * num / den;
        v[i * 4 + 1] = rc[i].y * num / den;
        v[i * 4 + 2] = std::max(1u, rc[i].w * num / den);
        v[i * 4 + 3] = std::max(1u, rc[i].h * num / den);
        mw = std::max(mw, v[i * 4 + 2]);
        mh = std::max(mh, v[i * 4 + 3]);
    }
    id<MTLBuffer> b = [dev newBufferWithBytes:v.data() length:v.size() * 4 options:MTLResourceStorageModeShared];
    [ce setBuffer:b offset:0 atIndex:1];
    [ce dispatchThreads:MTLSizeMake(mw, mh, n) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
}
}  // namespace

bool texfilter_encode(id<MTLDevice> dev, id<MTLCommandBuffer> cb, int mode, id<MTLTexture> src, id<MTLTexture> dst,
                      id<MTLTexture> mid, u32 factor, u32 wrap_s, u32 wrap_t, const TfRect *rects, u32 nrects) {
    if (!tf_init(dev) || (factor != 2 && factor != 4)) return false;
    const u32 w = (u32)src.width, h = (u32)src.height;
    const TfRect all = {0, 0, w, h};
    if (!rects || !nrects) {
        rects = &all;
        nrects = 1;
    }
    id<MTLTexture> top = dst.mipmapLevelCount > 1 ? level_view(dst, 0) : dst;
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
    if (mode == kTexFilterSmooth) {
        const TFP p = {w, h, w * factor, h * factor, wrap_s, wrap_t, factor, 0};
        [ce setComputePipelineState:k_smooth];
        [ce setTexture:src atIndex:0];
        [ce setTexture:top atIndex:1];
        [ce setBytes:&p length:sizeof p atIndex:0];
        dispatch(dev, ce, rects, nrects, factor, 1);
    } else {
        // 2x, or 2x twice through the intermediate texture mid
        [ce setComputePipelineState:k_sharp];
        if (factor == 4 && !mid) mid = texfilter_new_mid(dev, w, h);
        id<MTLTexture> in = src;
        u32 iw = w, ih = h, scale = 1;
        for (u32 f = factor; f > 1; f /= 2) {
            id<MTLTexture> out = f > 2 ? mid : top;
            const TFP q = {iw, ih, iw * 2, ih * 2, wrap_s, wrap_t, 2, 0};
            [ce setTexture:in atIndex:0];
            [ce setTexture:out atIndex:1];
            [ce setBytes:&q length:sizeof q atIndex:0];
            dispatch(dev, ce, rects, nrects, scale, 1);  // a thread per input texel
            scale *= 2;
            if (f > 2) [ce memoryBarrierWithScope:MTLBarrierScopeTextures];
            in = out;
            iw *= 2;
            ih *= 2;
        }
    }
    // the levels between the upscaled one and the original size
    u32 ul = 0;
    while ((1u << ul) < factor) ul++;
    if (ul > 1) {
        [ce setComputePipelineState:k_down];
        id<MTLTexture> prev = top;
        for (u32 l = 1; l < ul; l++) {
            [ce memoryBarrierWithScope:MTLBarrierScopeTextures];
            id<MTLTexture> v = level_view(dst, l);
            [ce setTexture:prev atIndex:0];
            [ce setTexture:v atIndex:1];
            dispatch(dev, ce, rects, nrects, factor, 1u << l);
            prev = v;
        }
    }
    [ce endEncoding];
    return true;
}

id<MTLTexture> texfilter_new_mid(id<MTLDevice> dev, u32 w, u32 h) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:w * 2
                                                                                height:h * 2
                                                                             mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    return [dev newTextureWithDescriptor:d];
}

}  // namespace gpu
