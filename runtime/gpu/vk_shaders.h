// GLSL (Vulkan) versions of metal.mm's fixed shaders, for runtime/gpu/vulkan.cpp: the vertex shader every pipeline
// shares, the MSAA restores and the compute kernels. They do what the MSL ones do, bit for bit where integers are
// involved; Metal's round() (halves away from zero) is round_away here, since GLSL leaves halves to the driver.
#pragma once

namespace gpu {

// helpers every shader below starts with (after its #version and extensions)
#define VK_GLSL_ROUND                                                                                                  \
    "float round_away(float x) { const float a = abs(x); float r = floor(a); r += a - r >= 0.5 ? 1.0 : 0.0; "          \
    "return x < 0.0 ? -r : r; }\n"                                                                                     \
    "vec4 round_away(vec4 x) { const vec4 a = abs(x); vec4 r = floor(a); r += vec4(greaterThanEqual(a - r, "           \
    "vec4(0.5))); return mix(r, -r, lessThan(x, vec4(0.0))); }\n"

// The vertex shader of all pipelines (kVertexMSL): vertices in the OutVertex layout (24 floats: shaded by the CPU, or
// by a compute kernel), pulled through the index buffer; PICA clipping as clip distances, snapping to 1/16 pixel,
// quaternions flipped toward the triangle's first vertex. Bindings 15-17 of the pipelines' set, VsDraw as push
// constants. Outputs at the locations fsgen.h's fragment shaders read.
static const char *const kVkVertexGLSL = "#version 450\n" VK_GLSL_ROUND R"(
layout(std430, set = 0, binding = 15) readonly buffer GV { float gv[]; };
layout(std430, set = 0, binding = 16) readonly buffer CV { float cv[]; };
layout(std430, set = 0, binding = 17) readonly buffer IX { uint idx[]; };
layout(push_constant) uniform VsDraw {
    vec4 rt;    // w, h, depth scale, depth offset
    vec4 vp;    // viewport half width, half height, x, y
    vec4 clip;  // user clip plane
    uvec4 flags;  // x: user clip plane enabled
} D;
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec4 v_tc01;
layout(location = 2) out vec4 v_tc2w;
layout(location = 3) out vec4 v_quat;
layout(location = 4) out vec4 v_view;
layout(location = 5) out vec2 v_zw;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };
float V(uint i, uint k) { return (i & 0x80000000u) != 0u ? cv[(i & 0x7fffffffu) * 24u + k] : gv[i * 24u + k]; }
void main() {
    const uint vid = uint(gl_VertexIndex);
    const uint i = idx[vid];
    const float x = V(i, 0u), y = V(i, 1u), z = V(i, 2u), w = V(i, 3u);
    vec4 pos;
    float px, py;
    if (w > 0.0) {
        const float iw = 1.0 / w;
        px = (x * iw + 1.0) * D.vp.x + D.vp.z;
        py = (y * iw + 1.0) * D.vp.y + D.vp.w;
        if (abs(px) < 8192.0 && abs(py) < 8192.0) {
            px = round_away(px * 16.0) / 16.0;
            py = round_away(py * 16.0) / 16.0;
        }
        pos.x = (px / D.rt.x * 2.0 - 1.0) * w;
        pos.y = (py / D.rt.y * 2.0 - 1.0) * w;
    } else {
        pos.x = ((x + w) * D.vp.x + D.vp.z * w) / D.rt.x * 2.0 - w;
        pos.y = ((y + w) * D.vp.y + D.vp.w * w) / D.rt.y * 2.0 - w;
    }
    pos.z = z * D.rt.z + D.rt.w * w;
    pos.w = w;
    gl_Position = pos;
    const float eps = 0.00000001;
    gl_ClipDistance[0] = eps - z;
    gl_ClipDistance[1] = z + w + eps;
    gl_ClipDistance[2] = w - 0.00001 + eps;
    gl_ClipDistance[3] = D.flags.x != 0u ? x * D.clip.x + y * D.clip.y + z * D.clip.z + w * D.clip.w + eps : 1.0;
    vec4 q = vec4(V(i, 4u), V(i, 5u), V(i, 6u), V(i, 7u));
    const uint first = vid - vid % 3u;
    if (vid != first) {
        const uint i0 = idx[first];
        const vec4 q0 = vec4(V(i0, 4u), V(i0, 5u), V(i0, 6u), V(i0, 7u));
        if (q.x * q0.x + q.y * q0.y + q.z * q0.z + q.w * q0.w < 0.0) q = -q;
    }
    v_quat = q;
    v_color = vec4(V(i, 8u), V(i, 9u), V(i, 10u), V(i, 11u));
    v_tc01 = vec4(V(i, 12u), V(i, 13u), V(i, 14u), V(i, 15u));
    v_tc2w = vec4(V(i, 22u), V(i, 23u), V(i, 16u), 0.0);
    v_view = vec4(V(i, 18u), V(i, 19u), V(i, 20u), 0.0);
    v_zw = vec2(z, w);
}
)";

// MSAA: a full-screen triangle filling a multisampled target from its resolved copy, every sample alike
static const char *const kVkFullGLSL = R"(#version 450
void main() {
    const vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
static const char *const kVkRestoreColorGLSL = R"(#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set = 0, binding = 3) uniform texture2D t;
layout(location = 0) out vec4 o;
void main() { o = texelFetch(t, ivec2(gl_FragCoord.xy), 0); }
)";
static const char *const kVkRestoreDepthGLSL = R"(#version 450
#extension GL_EXT_samplerless_texture_functions : require
#extension GL_ARB_shader_stencil_export : require
layout(set = 0, binding = 3) uniform texture2D d;
layout(set = 0, binding = 4) uniform utexture2D s;
void main() {
    gl_FragDepth = texelFetch(d, ivec2(gl_FragCoord.xy), 0).r;
    gl_FragStencilRefARB = int(texelFetch(s, ivec2(gl_FragCoord.xy), 0).r);
}
)";

// Without stencil export (NVIDIA): depth alone (stencil zeroed by the pipeline's state), then the stencil one bit
// at a time, each pass keeping the pixels whose bit is set (the bit index a push constant)
static const char *const kVkRestoreDepthOnlyGLSL = R"(#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set = 0, binding = 3) uniform texture2D d;
void main() { gl_FragDepth = texelFetch(d, ivec2(gl_FragCoord.xy), 0).r; }
)";
static const char *const kVkRestoreStencilBitGLSL = R"(#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(set = 0, binding = 4) uniform utexture2D s;
layout(push_constant) uniform PC { layout(offset = 64) uint bit; };
void main() {
    if (((texelFetch(s, ivec2(gl_FragCoord.xy), 0).r >> bit) & 1u) == 0u) discard;
}
)";

// The compute kernels (kKernelsMSL), each kVkKernelHead + its own source: storage buffers at bindings 0-3 (bytes where Metal had `device uchar *`),
// storage images at 4 and 5, parameters p and q as push constants; 16x16 work groups, each kernel checking its own
// bounds. Guest memory formats as gpu.cpp and transfer.cpp handle them: pixel formats 0 RGBA8, 1 RGB8, 2 RGB565,
// 3 RGB5A1, 4 RGBA4; depth formats 0 D16, 2 D24, 3 D24S8; tiled images in 8x8 tiles, Morton order.
static const char *const kVkKernelHead =
    "#version 450\n#extension GL_EXT_shader_8bit_storage : require\n"
    "layout(local_size_x = 16, local_size_y = 16) in;\n"
    "layout(push_constant) uniform PC { uvec4 p; uvec4 q; };\n" VK_GLSL_ROUND R"(
uint g_bpp(uint f) { return f == 0u ? 4u : f == 1u ? 3u : 2u; }
uint g_dbpp(uint f) { return f == 0u ? 2u : f == 2u ? 3u : 4u; }
uint g_tiled(uint x, uint y, uint w, uint bpp) {
    const uint m = (x & 1u) | ((y & 1u) << 1) | ((x & 2u) << 1) | ((y & 2u) << 2) | ((x & 4u) << 2) | ((y & 4u) << 3);
    return ((y & ~7u) * w + (x & ~7u) * 8u + m) * bpp;
}
uint c5to8(uint v) { return (v << 3) | (v >> 2); }
uint c6to8(uint v) { return (v << 2) | (v >> 4); }
uint c4to8(uint v) { return (v << 4) | v; }
uvec4 g_decode_bytes(uint f, uint p0, uint p1, uint p2, uint p3) {
    const uint v = p0 | (p1 << 8);
    switch (f) {
    case 0u: return uvec4(p3, p2, p1, p0);
    case 1u: return uvec4(p2, p1, p0, 255u);
    case 2u: return uvec4(c5to8(v >> 11), c6to8((v >> 5) & 63u), c5to8(v & 31u), 255u);
    case 3u: return uvec4(c5to8(v >> 11), c5to8((v >> 6) & 31u), c5to8((v >> 1) & 31u), (v & 1u) * 255u);
    default: return uvec4(c4to8(v >> 12), c4to8((v >> 8) & 15u), c4to8((v >> 4) & 15u), c4to8(v & 15u));
    }
}
// decode from / encode into a byte buffer (B: its array) at byte offset o
#define G_DECODE(B, f, o) g_decode_bytes(f, uint(B[o]), uint(B[(o) + 1u]), \
    g_bpp(f) > 2u ? uint(B[(o) + 2u]) : 0u, g_bpp(f) > 3u ? uint(B[(o) + 3u]) : 0u)
#define G_ENCODE(B, f, c, o) { const uint ff = f; const uvec4 cc = c; const uint oo = o; \
    if (ff == 0u) { B[oo] = uint8_t(cc.a); B[oo + 1u] = uint8_t(cc.b); B[oo + 2u] = uint8_t(cc.g); B[oo + 3u] = uint8_t(cc.r); } \
    else if (ff == 1u) { B[oo] = uint8_t(cc.b); B[oo + 1u] = uint8_t(cc.g); B[oo + 2u] = uint8_t(cc.r); } \
    else { uint v; \
        if (ff == 2u) v = ((cc.r >> 3) << 11) | ((cc.g >> 2) << 5) | (cc.b >> 3); \
        else if (ff == 3u) v = ((cc.r >> 3) << 11) | ((cc.g >> 3) << 6) | ((cc.b >> 3) << 1) | (cc.a >> 7); \
        else v = ((cc.r >> 4) << 12) | ((cc.g >> 4) << 8) | ((cc.b >> 4) << 4) | (cc.a >> 4); \
        B[oo] = uint8_t(v & 255u); B[oo + 1u] = uint8_t((v >> 8) & 255u); } }
)";

// upright display image: p = texture width, rows used, flip (bit 0) and black keyed out (bit 1), first row
static const char *const kVkRotGLSL = R"(
layout(set = 0, binding = 4, rgba8) uniform readonly image2D src;
layout(std430, set = 0, binding = 0) writeonly buffer B0 { uint dst[]; };
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.y || gid.y >= p.x) return;
    const uint row = ((p.z & 1u) != 0u ? p.y - 1u - gid.x : gid.x) + p.w;
    const uint col = p.x - 1u - gid.y;
    const uvec4 b = uvec4(round_away(clamp(imageLoad(src, ivec2(col, row)), 0.0, 1.0) * 255.0));
    uint a = b.a;
    if ((p.z & 2u) != 0u) {
        a = min(255u, a * 17u / 10u);
        a = min(a, uint(clamp((int(max(b.r, max(b.g, b.b))) - 3) * 64, 0, 255)));
    }
    dst[gid.y * p.y + gid.x] = b.r | (b.g << 8) | (b.b << 16) | (a << 24);
}
)";

// a colour target's GPU copy as guest memory (box-filtered to 1x): p = w, h (1x), scale, pixel format
static const char *const kVkEncGLSL = R"(
layout(set = 0, binding = 4, rgba8) uniform readonly image2D src;
layout(std430, set = 0, binding = 0) buffer B0 { uint8_t dst[]; };
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint s = p.z;
    uvec4 acc = uvec4(0u);
    for (uint j = 0u; j < s; j++)
        for (uint i = 0u; i < s; i++) acc += uvec4(round_away(imageLoad(src, ivec2(gid.x * s + i, gid.y * s + j)) * 255.0));
    const uint n = s * s;
    G_ENCODE(dst, p.w, (acc + n / 2u) / n, g_tiled(gid.x, gid.y, p.x, g_bpp(p.w)));
}
)";

// guest memory into a colour target's GPU copy (each pixel scale x scale times): p = w, h (1x), scale, pixel format
static const char *const kVkDecGLSL = R"(
layout(std430, set = 0, binding = 0) readonly buffer B0 { uint8_t src[]; };
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D dst;
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.x * p.z || gid.y >= p.y * p.z) return;
    const uvec4 c = G_DECODE(src, p.w, g_tiled(gid.x / p.z, gid.y / p.z, p.x, g_bpp(p.w)));
    imageStore(dst, ivec2(gid), vec4(c) / 255.0);
}
)";

// guest depth / stencil memory as scaled rows of depth and stencil values: p = w, h (1x), scale, depth format
static const char *const kVkDecDepthGLSL = R"(
layout(std430, set = 0, binding = 0) readonly buffer B0 { uint8_t src[]; };
layout(std430, set = 0, binding = 1) writeonly buffer B1 { float dz[]; };
layout(std430, set = 0, binding = 2) writeonly buffer B2 { uint8_t dsn[]; };
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    const uint W = p.x * p.z;
    if (gid.x >= W || gid.y >= p.y * p.z) return;
    const uint o = g_tiled(gid.x / p.z, gid.y / p.z, p.x, g_dbpp(p.w));
    const uint z = p.w == 0u ? (uint(src[o]) | (uint(src[o + 1u]) << 8))
                             : (uint(src[o]) | (uint(src[o + 1u]) << 8) | (uint(src[o + 2u]) << 16));
    dz[gid.y * W + gid.x] = float(z) * (p.w == 0u ? 1.0 / 65536.0 : 1.0 / 16777216.0);
    dsn[gid.y * W + gid.x] = p.w == 3u ? src[o + 3u] : uint8_t(0);
}
)";

// scaled rows of depth and stencil values (the top-left sample of each pixel) as guest memory: p as k_dec_depth
static const char *const kVkEncDepthGLSL = R"(
layout(std430, set = 0, binding = 0) readonly buffer B0 { float dz[]; };
layout(std430, set = 0, binding = 1) readonly buffer B1 { uint8_t dsn[]; };
layout(std430, set = 0, binding = 2) buffer B2 { uint8_t dst[]; };
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint i = gid.y * p.z * (p.x * p.z) + gid.x * p.z;
    const uint z = uint(dz[i] * (p.w == 0u ? 65536.0 : 16777216.0));
    const uint o = g_tiled(gid.x, gid.y, p.x, g_dbpp(p.w));
    dst[o] = uint8_t(z & 255u);
    dst[o + 1u] = uint8_t((z >> 8) & 255u);
    if (p.w != 0u) dst[o + 2u] = uint8_t((z >> 16) & 255u);
    if (p.w == 3u) dst[o + 3u] = dsn[i];
}
)";

// a display transfer's conversion (transfer_convert in transfer.cpp) from a tiled image in guest format: p = input
// width, output width and rows (after downscaling), flags; q.x = first input byte
static const char *const kVkTconvGLSL = R"(
layout(std430, set = 0, binding = 0) readonly buffer B0 { uint8_t src[]; };
layout(std430, set = 0, binding = 1) buffer B1 { uint8_t dst[]; };
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    const uint out_w = p.y, out_h = p.z, flags = p.w;
    if (gid.x >= out_w || gid.y >= out_h) return;
    const bool flip = (flags & 1u) != 0u, in_linear = ((flags >> 1) & 1u) != 0u, dont_swizzle = ((flags >> 5) & 1u) != 0u;
    const uint in_fmt = (flags >> 8) & 7u, out_fmt = (flags >> 12) & 7u, scaling = (flags >> 24) & 3u;
    const uint ibpp = g_bpp(in_fmt), obpp = g_bpp(out_fmt);
    const uint hs = scaling != 0u ? 1u : 0u, vs = scaling == 2u ? 1u : 0u;
    const uint x = gid.x, y = gid.y, ix = x << hs, iy = y << vs, oy = flip ? out_h - 1u - y : y;
    uint so, doff;
    if (in_linear) {
        so = (ix + iy * p.x) * ibpp;
        doff = dont_swizzle ? (x + oy * out_w) * obpp : g_tiled(x, oy, out_w, obpp);
    } else {
        so = g_tiled(ix, iy, p.x, ibpp);
        doff = dont_swizzle ? g_tiled(x, oy, out_w, obpp) : (x + oy * out_w) * obpp;
    }
    const uint sp = q.x + so;
    uvec4 c = G_DECODE(src, in_fmt, sp);
    if (scaling == 1u) c = (c + G_DECODE(src, in_fmt, sp + ibpp)) / 2u;
    else if (scaling == 2u)
        c = (c + G_DECODE(src, in_fmt, sp + ibpp) + G_DECODE(src, in_fmt, sp + 2u * ibpp) + G_DECODE(src, in_fmt, sp + 3u * ibpp)) / 4u;
    G_ENCODE(dst, out_fmt, c, doff);
}
)";

// tiled-to-tiled display transfer between GPU copies of guest images: p = output width, rows (scaled), first source
// row (scaled); q = horizontal and vertical downscale shifts, flip, source without alpha
static const char *const kVkXferGLSL = R"(
layout(set = 0, binding = 4, rgba8) uniform readonly image2D src;
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D dst;
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.x || gid.y >= p.y) return;
    const uint y = q.z != 0u ? p.y - 1u - gid.y : gid.y;
    const uint fx = 1u << q.x, fy = 1u << q.y;
    vec4 acc = vec4(0.0);
    for (uint j = 0u; j < fy; j++)
        for (uint i = 0u; i < fx; i++) acc += imageLoad(src, ivec2(gid.x * fx + i, p.z + y * fy + j));
    acc /= float(fx * fy);
    if (q.w != 0u) acc.a = 1.0;
    imageStore(dst, ivec2(gid), acc);
}
)";

// a GPU copy of a guest image as a texture: rows upside down (the order of decoded textures), alpha opaque for formats
// without one; p = width, rows (scaled), opaque
static const char *const kVkSnapGLSL = R"(
layout(set = 0, binding = 4, rgba8) uniform readonly image2D src;
layout(set = 0, binding = 5, rgba8) uniform writeonly image2D dst;
void main() {
    const uvec2 gid = gl_GlobalInvocationID.xy;
    if (gid.x >= p.x || gid.y >= p.y) return;
    vec4 c = imageLoad(src, ivec2(gid.x, p.y - 1u - gid.y));
    if (p.z != 0u) c.a = 1.0;
    imageStore(dst, ivec2(gid), c);
}
)";

}  // namespace gpu
