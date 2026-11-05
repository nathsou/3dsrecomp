// The fragment shaders of the GPU backends, generated from the PICA's fragment pipeline state (TEV combiners,
// fragment lighting, fog, alpha test, blending and logic ops): its key (fs_key, everything a shader depends on, the
// unit of the pipeline lists games/<title id>/pipelines.bin), the state back from a key (decode_key) and the shader's
// source (gen_fs). Shared by the Metal (metal.mm) and Vulkan (vulkan.cpp) backends.
#include "gpu/fsgen.h"

#include <cstdarg>
#include <cstdio>

namespace gpu {

// the declarations every fragment shader starts with
const char *const kCommonMSL = R"(
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

// GLSL (Vulkan): the same declarations, the MSL type names the generator writes as macros, and the vertex
// shader's outputs gathered into `vin` (VOut) by main() (the generator's `in.` becomes `vin.`)
const char *const kCommonGLSL = R"(#version 450
#define float2 vec2
#define float3 vec3
#define float4 vec4
#define int2 ivec2
#define int3 ivec3
#define int4 ivec4
#define uint2 uvec2
#define uint3 uvec3
#define uint4 uvec4
#define bool4 bvec4
#define saturate(x) clamp(x, 0.0, 1.0)
#define discard_fragment() discard
#define select(a, b, c) mix(a, b, c)
struct VOut {
    vec4 pos;
    vec4 color;
    vec4 tc01;
    vec4 tc2w;
    vec4 quat;
    vec4 view;
    vec2 zw;
};
layout(location = 0) in vec4 v_color;
layout(location = 1) in vec4 v_tc01;
layout(location = 2) in vec4 v_tc2w;
layout(location = 3) in vec4 v_quat;
layout(location = 4) in vec4 v_view;
layout(location = 5) in vec2 v_zw;
layout(location = 0) out vec4 o_color;
VOut vin;
struct Light { vec4 spec0, spec1, diffuse, ambient, pos, spot, att; };
layout(std140, set = 0, binding = 0) uniform Uni {
    vec4 rt;
    vec4 depth;
    ivec4 scissor;
    uvec4 konst0, konst1;
    uvec4 tevc;
    uvec4 misc;
    uvec4 border;
    vec4 bias;
    vec4 lscale0;
    vec4 lscale1;
    vec4 ambient;
    Light lights[8];
} u;
// Metal's round(): halves away from zero (GLSL's may round them to even); the values rounded are not negative
float round_hu(float x) { const float r = floor(x); return x - r >= 0.5 ? r + 1.0 : r; }
vec4 round_hu(vec4 x) { const vec4 r = floor(x); return r + vec4(greaterThanEqual(x - r, vec4(0.5))); }
#define round(x) round_hu(x)
)";

namespace {
struct Gen {
    std::string s;
    bool glsl = false;
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
        g.f(g.glsl ? "    { // light %u (#%u)\n      const Light L = u.lights[%u];\n"
                   : "    { // light %u (#%u)\n      constant Light &L = u.lights[%u];\n",
            li, lt.num, li);
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

}  // namespace

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

namespace {
// GLSL: the shader's resources (Metal: fs_main's parameters), as vulkan.cpp binds them (set 0)
void glsl_resources(Gen &g, const DrawState &ds, const FsOptions &o, bool fetch) {
    if (ds.need_shade && ds.lighting) g.f("layout(std430, set = 0, binding = 1) readonly buffer Lut { vec2 lut[]; };\n");
    if (ds.need_shade && ds.fog) g.f("layout(std430, set = 0, binding = 2) readonly buffer FogLut { vec2 fogl[]; };\n");
    bool cube = false;
    for (int i = 0; i < 3; i++)
        if (ds.need_shade && ds.need_tex[i] && !ds.tex[i].null_tex) {
            g.f("layout(set = 0, binding = %d) uniform texture2D t%d;\nlayout(set = 0, binding = %d) uniform sampler s%d;\n",
                3 + i, i, 6 + i, i);
            if (i == 0 && (ds.tex[0].type == 1 || ds.tex[0].type == 4)) cube = true;
        }
    if (cube)
        for (int f = 1; f < 6; f++) g.f("layout(set = 0, binding = %d) uniform texture2D f%d;\n", 8 + f, f);
    if (fetch)
        g.f(o.samples > 1 ? "layout(input_attachment_index = 0, set = 0, binding = 14) uniform subpassInputMS dst_in;\n"
                          : "layout(input_attachment_index = 0, set = 0, binding = 14) uniform subpassInput dst_in;\n");
}

// GLSL: main() runs the generated fs_main with the inputs in `vin` (and the destination colour in dstc)
void glsl_main(Gen &g, bool fdep, bool fetch, const FsOptions &o) {
    g.f("void main() {\n"
        "    vin.pos = gl_FragCoord; vin.color = v_color; vin.tc01 = v_tc01; vin.tc2w = v_tc2w;\n"
        "    vin.quat = v_quat; vin.view = v_view; vin.zw = v_zw;\n");
    if (fetch) g.f(o.samples > 1 ? "    dstc = subpassLoad(dst_in, gl_SampleID);\n" : "    dstc = subpassLoad(dst_in);\n");
    g.f("    const FOut o = fs_main();\n    o_color = o.c;\n");
    if (fdep) g.f("    gl_FragDepth = o.d;\n");
    g.f("}\n");
}

// the generator writes the stage input as Metal's `in.`, which GLSL reserves: `vin.` there
std::string glsl_inputs(const std::string &src) {
    std::string out;
    out.reserve(src.size() + 256);
    for (size_t i = 0; i < src.size(); i++) {
        const bool word_start = i == 0 || !(isalnum((unsigned char)src[i - 1]) || src[i - 1] == '_');
        if (word_start && src.compare(i, 3, "in.") == 0) out += 'v';
        out += src[i];
    }
    return out;
}

std::string gen_fs_body(const DrawState &ds, const FsOptions &opt, bool &hwb_out, bool &fetch_out);
}  // namespace

bool fs_hw_blend(const DrawState &ds, const FsOptions &o) { return hw_blend(ds) || (o.hw_logic_op && !ds.blend); }

std::string gen_fs(const DrawState &ds, const FsOptions &opt) {
    bool hwb, fetch;
    std::string body = gen_fs_body(ds, opt, hwb, fetch);
    if (opt.lang == ShaderLang::MSL) return body;
    Gen g;
    g.glsl = true;
    g.s = std::move(body);
    glsl_main(g, frag_depth(ds), fetch, opt);
    return glsl_inputs(g.s);
}

namespace {
std::string gen_fs_body(const DrawState &ds, const FsOptions &opt, bool &hwb_out, bool &fetch_out) {
    Gen g;
    g.glsl = opt.lang == ShaderLang::GLSL;
    const bool hwb = fs_hw_blend(ds, opt), fdep = frag_depth(ds);
    const bool fetch = !hwb && ds.need_shade;  // programmable blending: reads the destination colour
    hwb_out = hwb;
    fetch_out = fetch;
    if (g.glsl) {
        g.s = kCommonGLSL;
        glsl_resources(g, ds, opt, fetch);
        if (fetch) g.f("vec4 dstc;\n");
        g.f("struct FOut { vec4 c; float d; };\n");
        g.f("uint rc(float f) { const float v = round(f * 255.0); return v > 0.0 ? (v >= 255.0 ? 255u : uint(v)) : 0u; }\n"
            "uvec4 tou(vec4 c) { return uvec4(round(saturate(c) * 255.0)); }\n"
            "uvec4 unpack(uint c) { return uvec4(c & 255u, (c >> 8) & 255u, (c >> 16) & 255u, c >> 24); }\n"
            "uvec4 packl(vec4 v) { uvec4 o; for (int i = 0; i < 4; i++) { float x = isnan(v[i]) ? 0.0 : clamp(v[i], 0.0, 1.0); o[i] = uint(x * 255.0); } return o; }\n");
        if (ds.need_shade && ds.lighting)
            g.f("float lutv_(uint s, float r, bool absm, bool two) {\n"
                "    if (isnan(r)) r = 0.0; uint idx; float delta;\n"
                "    if (absm) { r = two ? abs(r) : max(r, 0.0); const float fl = floor(r * 256.0); idx = uint(clamp(fl, 0.0, 255.0)); delta = r * 256.0 - float(idx); }\n"
                "    else { const float fl = floor(r * 128.0); const int si = int(clamp(fl, -128.0, 127.0)); delta = r * 128.0 - float(si); idx = uint(si) & 255u; }\n"
                "    const vec2 e = lut[s * 256u + idx]; return e.x + e.y * delta; }\n"
                "#define lutv(L, s, r, a, t) lutv_(s, r, a, t)\n");
        g.f("FOut fs_main() {\n    FOut o;\n");
    } else {
    g.s = kCommonMSL;
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
    }
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
                "      float4 sv;\n");
            if (g.glsl)
                g.f("      switch (face) { case 0: sv = textureLod(sampler2D(t0, s0), uv, 0.0); break; case 1: sv = textureLod(sampler2D(f1, s0), uv, 0.0); break;\n"
                    "        case 2: sv = textureLod(sampler2D(f2, s0), uv, 0.0); break; case 3: sv = textureLod(sampler2D(f3, s0), uv, 0.0); break;\n"
                    "        case 4: sv = textureLod(sampler2D(f4, s0), uv, 0.0); break; default: sv = textureLod(sampler2D(f5, s0), uv, 0.0); break; }\n");
            else
                g.f("      switch (face) { case 0: sv = t0.sample(s0, uv, level(0)); break; case 1: sv = f1.sample(s0, uv, level(0)); break;\n"
                    "        case 2: sv = f2.sample(s0, uv, level(0)); break; case 3: sv = f3.sample(s0, uv, level(0)); break;\n"
                    "        case 4: sv = f4.sample(s0, uv, level(0)); break; default: sv = f5.sample(s0, uv, level(0)); break; }\n");
            g.f("      tx0 = tou(sv); }\n");
            continue;
        }
        if (type == 2) {
            if (g.glsl)
                g.f("    { const float2 uvs = %s; const uint2 sz = uint2(textureSize(sampler2D(t0, s0), 0));\n"
                    "      const int2 p = clamp(int2(floor(uvs * float2(sz))), int2(0), int2(sz) - 1);\n"
                    "      const uint4 b = tou(texelFetch(sampler2D(t0, s0), p, 0));\n"
                    "      const float zz = min(abs(tw), 1.0); const int zi = int(zz * 16777215.0) - int(u.misc.w << 1);\n"
                    "      const int zref = int((b.a << 16) | (b.b << 8) | b.g);\n"
                    "      tx0 = uint4(zref >= zi ? b.r : 0u); }\n",
                    t.shadow_ortho ? "c0" : "c0 / tw");
            else
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
        g.f(g.glsl ? "      const float4 sv = texture(sampler2D(t%d, s%d), uv, u.bias[%d]);\n"
                   : "      const float4 sv = t%d.sample(s%d, uv, bias(u.bias[%d]));\n",
            i, i, i);
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

}  // namespace

// the DrawState fields gen_fs() and the backends' pipelines depend on, from a key (inverse of fs_key)
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

}  // namespace gpu
